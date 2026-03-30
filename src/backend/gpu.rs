use ndarray::Array1;
use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
};
use cudarc::cublas::sys as blas;
use cudarc::cusparse::sys as sp;

use super::OplsBackend;

pub struct GpuBackend {
    stream: Arc<CudaStream>,
    cusparse_handle: sp::cusparseHandle_t,
    cublas_handle: blas::cublasHandle_t,
    mat_descr: sp::cusparseSpMatDescr_t,
    _d_values: CudaSlice<f64>,
    _d_col_indices: CudaSlice<i32>,
    _d_row_offsets: CudaSlice<i32>,
    n: usize,
    p: usize,
    d_mu: Option<CudaSlice<f64>>,
    d_t_defl: Vec<CudaSlice<f64>>,
    d_p_defl: Vec<CudaSlice<f64>>,
    d_ones_n: CudaSlice<f64>,
    /// Device constant [-1.0] for negating scalars via dscal
    d_neg_one: CudaSlice<f64>,
    scratch: RefCell<ScratchBuffers>,
}

struct ScratchBuffers {
    d_x: CudaSlice<f64>,
    d_y: CudaSlice<f64>,
    d_w: CudaSlice<f64>,
    /// Device scalar buffer for dot product results (1 element)
    d_scalar: CudaSlice<f64>,
    workspace_fwd: CudaSlice<u8>,
    workspace_trans: CudaSlice<u8>,
}

fn raw_ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> cudarc::driver::sys::CUdeviceptr {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}

fn raw_ptr_mut<T>(slice: &mut CudaSlice<T>, stream: &CudaStream) -> cudarc::driver::sys::CUdeviceptr {
    let (ptr, _guard) = slice.device_ptr_mut(stream);
    ptr
}

impl GpuBackend {
    pub fn try_new(x_sp: &sprs::CsMat<f64>) -> Result<Self, Box<dyn std::error::Error>> {
        let n = x_sp.rows();
        let p = x_sp.cols();
        let nnz = x_sp.nnz();

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let indptr_raw = x_sp.indptr();
        let indptr = indptr_raw.as_slice().ok_or("non-contiguous indptr")?;
        let row_offsets_i32: Vec<i32> = indptr.iter().map(|&x| x as i32).collect();
        let col_indices_i32: Vec<i32> = x_sp.indices().iter().map(|&x| x as i32).collect();
        let values = x_sp.data();

        let d_row_offsets = stream.clone_htod(&row_offsets_i32)?;
        let d_col_indices = stream.clone_htod(&col_indices_i32)?;
        let d_values = stream.clone_htod(values)?;

        // cuSPARSE
        let cusparse_handle = cudarc::cusparse::result::create()
            .map_err(|e| format!("cusparseCreate: {:?}", e))?;

        let mat_descr = unsafe {
            let mut descr = std::mem::MaybeUninit::uninit();
            sp::cusparseCreateCsr(
                descr.as_mut_ptr(),
                n as i64, p as i64, nnz as i64,
                raw_ptr(&d_row_offsets, &stream) as *mut _,
                raw_ptr(&d_col_indices, &stream) as *mut _,
                raw_ptr(&d_values, &stream) as *mut _,
                sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                sp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                sp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                sp::cudaDataType::CUDA_R_64F,
            ).result().map_err(|e| format!("cusparseCreateCsr: {:?}", e))?;
            descr.assume_init()
        };

        // cuBLAS — DEVICE pointer mode: scalars stay on GPU
        let cublas_handle = cudarc::cublas::result::create_handle()
            .map_err(|e| format!("cublasCreate: {:?}", e))?;
        unsafe {
            blas::cublasSetPointerMode_v2(
                cublas_handle,
                blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            ).result().map_err(|e| format!("cublasSetPointerMode: {:?}", e))?;
            blas::cublasSetStream_v2(
                cublas_handle,
                stream.cu_stream() as _,
            ).result().map_err(|e| format!("cublasSetStream: {:?}", e))?;
        }

        // Allocate scratch
        let max_dim = n.max(p);
        let mut d_x: CudaSlice<f64> = stream.alloc_zeros(max_dim)?;
        let mut d_y: CudaSlice<f64> = stream.alloc_zeros(max_dim)?;
        let d_w: CudaSlice<f64> = stream.alloc_zeros(p)?;
        let d_scalar: CudaSlice<f64> = stream.alloc_zeros(1)?;

        // Device constants
        let d_ones_n = stream.clone_htod(&vec![1.0_f64; n])?;
        let d_neg_one = stream.clone_htod(&[-1.0_f64])?;

        // Workspace sizes (cuSPARSE takes host pointers for alpha/beta regardless)
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;

        let workspace_fwd_size = unsafe {
            Self::workspace_size(
                cusparse_handle, mat_descr,
                sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, &beta,
                raw_ptr_mut(&mut d_x, &stream),
                raw_ptr_mut(&mut d_y, &stream),
                p, n,
            )?
        };
        let workspace_trans_size = unsafe {
            Self::workspace_size(
                cusparse_handle, mat_descr,
                sp::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
                &alpha, &beta,
                raw_ptr_mut(&mut d_x, &stream),
                raw_ptr_mut(&mut d_y, &stream),
                n, p,
            )?
        };

        let workspace_fwd: CudaSlice<u8> = stream.alloc_zeros(workspace_fwd_size.max(1))?;
        let workspace_trans: CudaSlice<u8> = stream.alloc_zeros(workspace_trans_size.max(1))?;

        Ok(GpuBackend {
            stream, cusparse_handle, cublas_handle, mat_descr,
            _d_values: d_values, _d_col_indices: d_col_indices, _d_row_offsets: d_row_offsets,
            n, p,
            d_mu: None, d_t_defl: Vec::new(), d_p_defl: Vec::new(),
            d_ones_n, d_neg_one,
            scratch: RefCell::new(ScratchBuffers {
                d_x, d_y, d_w, d_scalar, workspace_fwd, workspace_trans,
            }),
        })
    }

    unsafe fn workspace_size(
        handle: sp::cusparseHandle_t,
        mat_descr: sp::cusparseSpMatDescr_t,
        op: sp::cusparseOperation_t,
        alpha: &f64, beta: &f64,
        x_ptr: cudarc::driver::sys::CUdeviceptr,
        y_ptr: cudarc::driver::sys::CUdeviceptr,
        input_len: usize, output_len: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        unsafe {
            let mut vec_x = std::mem::MaybeUninit::uninit();
            sp::cusparseCreateDnVec(vec_x.as_mut_ptr(), input_len as i64, x_ptr as *mut _, sp::cudaDataType::CUDA_R_64F)
                .result().map_err(|e| format!("cusparseCreateDnVec: {:?}", e))?;
            let vec_x = vec_x.assume_init();

            let mut vec_y = std::mem::MaybeUninit::uninit();
            sp::cusparseCreateDnVec(vec_y.as_mut_ptr(), output_len as i64, y_ptr as *mut _, sp::cudaDataType::CUDA_R_64F)
                .result().map_err(|e| format!("cusparseCreateDnVec: {:?}", e))?;
            let vec_y = vec_y.assume_init();

            let mut buf_size: usize = 0;
            sp::cusparseSpMV_bufferSize(
                handle, op, alpha as *const f64 as *const _, mat_descr, vec_x,
                beta as *const f64 as *const _, vec_y, sp::cudaDataType::CUDA_R_64F,
                sp::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT, &mut buf_size as *mut usize,
            ).result().map_err(|e| format!("cusparseSpMV_bufferSize: {:?}", e))?;

            sp::cusparseDestroyDnVec(vec_x).result().ok();
            sp::cusparseDestroyDnVec(vec_y).result().ok();
            Ok(buf_size)
        }
    }

    /// Run cuSPARSE SpMV. Note: cuSPARSE always uses host pointers for alpha/beta
    /// regardless of cuBLAS pointer mode.
    fn run_spmv(
        &self,
        op: sp::cusparseOperation_t,
        d_in_ptr: u64, d_out_ptr: u64,
        input_len: usize, output_len: usize,
        ws_ptr: u64,
    ) {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            let mut vec_x = std::mem::MaybeUninit::uninit();
            sp::cusparseCreateDnVec(vec_x.as_mut_ptr(), input_len as i64, d_in_ptr as *mut _, sp::cudaDataType::CUDA_R_64F)
                .result().expect("cusparseCreateDnVec");
            let vec_x = vec_x.assume_init();

            let mut vec_y = std::mem::MaybeUninit::uninit();
            sp::cusparseCreateDnVec(vec_y.as_mut_ptr(), output_len as i64, d_out_ptr as *mut _, sp::cudaDataType::CUDA_R_64F)
                .result().expect("cusparseCreateDnVec");
            let vec_y = vec_y.assume_init();

            sp::cusparseSpMV(
                self.cusparse_handle, op,
                &alpha as *const f64 as *const _, self.mat_descr, vec_x,
                &beta as *const f64 as *const _, vec_y,
                sp::cudaDataType::CUDA_R_64F,
                sp::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                ws_ptr as *mut _,
            ).result().expect("cusparseSpMV");

            sp::cusparseDestroyDnVec(vec_x).result().ok();
            sp::cusparseDestroyDnVec(vec_y).result().ok();
        }
    }

    // ── cuBLAS helpers (DEVICE pointer mode) ───────────────────────────────

    /// ddot → d_scalar (device). No host sync.
    fn ddot_to_device(&self, x_ptr: u64, y_ptr: u64, len: usize, scalar_ptr: u64) {
        unsafe {
            blas::cublasDdot_v2(
                self.cublas_handle, len as i32,
                x_ptr as *const f64, 1,
                y_ptr as *const f64, 1,
                scalar_ptr as *mut f64,
            ).result().expect("cublasDdot");
        }
    }

    /// Negate d_scalar in-place: d_scalar *= -1. Uses d_neg_one device constant.
    fn negate_scalar(&self, scalar_ptr: u64) {
        let neg_one_ptr = raw_ptr(&self.d_neg_one, &self.stream);
        unsafe {
            blas::cublasDscal_v2(
                self.cublas_handle, 1,
                neg_one_ptr as *const f64,
                scalar_ptr as *mut f64, 1,
            ).result().expect("cublasDscal negate");
        }
    }

    /// daxpy with alpha from device: y += d_scalar * x. No host sync.
    fn daxpy_device(&self, scalar_ptr: u64, x_ptr: u64, y_ptr: u64, len: usize) {
        unsafe {
            blas::cublasDaxpy_v2(
                self.cublas_handle, len as i32,
                scalar_ptr as *const f64,
                x_ptr as *const f64, 1,
                y_ptr as *mut f64, 1,
            ).result().expect("cublasDaxpy");
        }
    }

    /// dnrm2 → d_scalar (device). No host sync.
    fn dnrm2_to_device(&self, x_ptr: u64, len: usize, scalar_ptr: u64) {
        unsafe {
            blas::cublasDnrm2_v2(
                self.cublas_handle, len as i32,
                x_ptr as *const f64, 1,
                scalar_ptr as *mut f64,
            ).result().expect("cublasDnrm2");
        }
    }

    /// Download a single f64 from device scalar buffer to host.
    fn download_scalar(&self, d_scalar: &CudaSlice<f64>) -> f64 {
        let mut val = [0.0_f64];
        self.stream.memcpy_dtoh(d_scalar, &mut val).expect("dtoh scalar");
        self.stream.synchronize().expect("sync");
        val[0]
    }

    /// dscal with alpha from host (temporarily switch to host pointer mode).
    fn dscal_host(&self, alpha: f64, x_ptr: u64, len: usize) {
        unsafe {
            blas::cublasSetPointerMode_v2(
                self.cublas_handle,
                blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            ).result().expect("set host mode");
            blas::cublasDscal_v2(
                self.cublas_handle, len as i32,
                &alpha as *const f64,
                x_ptr as *mut f64, 1,
            ).result().expect("cublasDscal");
            blas::cublasSetPointerMode_v2(
                self.cublas_handle,
                blas::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            ).result().expect("restore device mode");
        }
    }

    // ── Correction helpers ─────────────────────────────────────────────────

    /// ddot → negate → daxpy, all on device. No sync.
    fn dot_negate_axpy(
        &self, dot_x: u64, dot_y: u64, dot_len: usize,
        axpy_src: u64, axpy_dst: u64, axpy_len: usize,
        scalar_ptr: u64,
    ) {
        self.ddot_to_device(dot_x, dot_y, dot_len, scalar_ptr);
        self.negate_scalar(scalar_ptr);
        self.daxpy_device(scalar_ptr, axpy_src, axpy_dst, axpy_len);
    }

    /// Apply forward corrections: result -= (mu·v)*1 - Σ(p_k·v)*t_k
    fn apply_forward_corrections_gpu(&self, result_ptr: u64, v_ptr: u64, scalar_ptr: u64) {
        if let Some(d_mu) = &self.d_mu {
            let mu_ptr = raw_ptr(d_mu, &self.stream);
            let ones_ptr = raw_ptr(&self.d_ones_n, &self.stream);
            self.dot_negate_axpy(mu_ptr, v_ptr, self.p, ones_ptr, result_ptr, self.n, scalar_ptr);
        }
        for (d_t, d_p) in self.d_t_defl.iter().zip(self.d_p_defl.iter()) {
            let p_ptr = raw_ptr(d_p, &self.stream);
            let t_ptr = raw_ptr(d_t, &self.stream);
            self.dot_negate_axpy(p_ptr, v_ptr, self.p, t_ptr, result_ptr, self.n, scalar_ptr);
        }
    }

    /// Apply transpose corrections: result -= sum(v)*mu - Σ(t_k·v)*p_k
    fn apply_transpose_corrections_gpu(&self, result_ptr: u64, v_ptr: u64, scalar_ptr: u64) {
        if let Some(d_mu) = &self.d_mu {
            let ones_ptr = raw_ptr(&self.d_ones_n, &self.stream);
            let mu_ptr = raw_ptr(d_mu, &self.stream);
            self.dot_negate_axpy(ones_ptr, v_ptr, self.n, mu_ptr, result_ptr, self.p, scalar_ptr);
        }
        for (d_t, d_p) in self.d_t_defl.iter().zip(self.d_p_defl.iter()) {
            let t_ptr = raw_ptr(d_t, &self.stream);
            let p_ptr = raw_ptr(d_p, &self.stream);
            self.dot_negate_axpy(t_ptr, v_ptr, self.n, p_ptr, result_ptr, self.p, scalar_ptr);
        }
    }

    fn download(&self, d_buf: &CudaSlice<f64>, len: usize) -> Array1<f64> {
        let result = self.stream.clone_dtoh(d_buf).expect("dtoh copy");
        Array1::from_vec(result[..len].to_vec())
    }
}

impl OplsBackend for GpuBackend {
    fn set_mean(&mut self, mu: Array1<f64>) {
        self.d_mu = Some(self.stream.clone_htod(mu.as_slice().unwrap()).expect("htod mu"));
    }

    fn push_deflation(&mut self, t: Array1<f64>, p: Array1<f64>) {
        self.d_t_defl.push(self.stream.clone_htod(t.as_slice().unwrap()).expect("htod t"));
        self.d_p_defl.push(self.stream.clone_htod(p.as_slice().unwrap()).expect("htod p"));
    }

    fn truncate_deflation(&mut self, keep: usize) {
        self.d_t_defl.truncate(keep);
        self.d_p_defl.truncate(keep);
    }

    fn num_deflation(&self) -> usize {
        self.d_t_defl.len()
    }

    fn xeff_forward(&self, v: &Array1<f64>, n: usize) -> Array1<f64> {
        let mut scratch = self.scratch.borrow_mut();
        self.stream.memcpy_htod(v.as_slice().unwrap(), &mut scratch.d_x).expect("htod v");
        self.stream.memset_zeros(&mut scratch.d_y).expect("memset");

        let x_ptr = raw_ptr(&scratch.d_x, &self.stream);
        let y_ptr = raw_ptr_mut(&mut scratch.d_y, &self.stream);
        let ws_ptr = raw_ptr_mut(&mut scratch.workspace_fwd, &self.stream);
        let sc_ptr = raw_ptr_mut(&mut scratch.d_scalar, &self.stream);

        self.run_spmv(sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, x_ptr, y_ptr, self.p, n, ws_ptr);
        self.apply_forward_corrections_gpu(y_ptr, x_ptr, sc_ptr);
        self.download(&scratch.d_y, n)
    }

    fn xeff_transpose(&self, v: &Array1<f64>, p: usize) -> Array1<f64> {
        let mut scratch = self.scratch.borrow_mut();
        self.stream.memcpy_htod(v.as_slice().unwrap(), &mut scratch.d_x).expect("htod v");
        self.stream.memset_zeros(&mut scratch.d_y).expect("memset");

        let x_ptr = raw_ptr(&scratch.d_x, &self.stream);
        let y_ptr = raw_ptr_mut(&mut scratch.d_y, &self.stream);
        let ws_ptr = raw_ptr_mut(&mut scratch.workspace_trans, &self.stream);
        let sc_ptr = raw_ptr_mut(&mut scratch.d_scalar, &self.stream);

        self.run_spmv(sp::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE, x_ptr, y_ptr, self.n, p, ws_ptr);
        self.apply_transpose_corrections_gpu(y_ptr, x_ptr, sc_ptr);
        self.download(&scratch.d_y, p)
    }

    fn nipals_wt(&self, u: &Array1<f64>, n: usize, p: usize) -> (Array1<f64>, Array1<f64>, f64) {
        let mut scratch = self.scratch.borrow_mut();

        // Step 1: w = normalize(X_eff^T * u)
        self.stream.memcpy_htod(u.as_slice().unwrap(), &mut scratch.d_x).expect("htod u");
        self.stream.memset_zeros(&mut scratch.d_w).expect("memset");

        let u_ptr = raw_ptr(&scratch.d_x, &self.stream);
        let w_ptr = raw_ptr_mut(&mut scratch.d_w, &self.stream);
        let ws_ptr = raw_ptr_mut(&mut scratch.workspace_trans, &self.stream);
        let sc_ptr = raw_ptr_mut(&mut scratch.d_scalar, &self.stream);

        self.run_spmv(sp::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE, u_ptr, w_ptr, self.n, p, ws_ptr);
        self.apply_transpose_corrections_gpu(w_ptr, u_ptr, sc_ptr);

        // Normalize w: nrm2 → device scalar → download 1 f64 → dscal with host alpha
        self.dnrm2_to_device(w_ptr, p, sc_ptr);
        let nrm = self.download_scalar(&scratch.d_scalar); // 1 sync
        if nrm > 0.0 {
            self.dscal_host(1.0 / nrm, w_ptr, p);
        }

        // Step 2: t = X_eff * w (w already on device — no upload)
        self.stream.memset_zeros(&mut scratch.d_y).expect("memset");
        let w_ptr = raw_ptr(&scratch.d_w, &self.stream);
        let t_ptr = raw_ptr_mut(&mut scratch.d_y, &self.stream);
        let ws_ptr = raw_ptr_mut(&mut scratch.workspace_fwd, &self.stream);
        let sc_ptr = raw_ptr_mut(&mut scratch.d_scalar, &self.stream);

        self.run_spmv(sp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE, w_ptr, t_ptr, self.p, n, ws_ptr);
        self.apply_forward_corrections_gpu(t_ptr, w_ptr, sc_ptr);

        // t·t → device scalar → download 1 f64
        self.ddot_to_device(t_ptr, t_ptr, n, sc_ptr);
        let tt = self.download_scalar(&scratch.d_scalar); // 1 sync

        // Download w and t
        let w = self.download(&scratch.d_w, p);
        let t = self.download(&scratch.d_y, n);
        (w, t, tt)
    }
}

impl Drop for GpuBackend {
    fn drop(&mut self) {
        unsafe {
            sp::cusparseDestroySpMat(self.mat_descr).result().ok();
            cudarc::cusparse::result::destroy(self.cusparse_handle).ok();
            blas::cublasDestroy_v2(self.cublas_handle).result().ok();
        }
    }
}
