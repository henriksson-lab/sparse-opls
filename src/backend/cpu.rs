use ndarray::Array1;
use super::OplsBackend;

pub struct CpuBackend {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    n: usize,
    mu: Option<Array1<f64>>,
    t_defl: Vec<Array1<f64>>,
    p_defl: Vec<Array1<f64>>,
}

impl CpuBackend {
    pub fn new(x_sp: &sprs::CsMat<f64>) -> Self {
        let indptr_raw = x_sp.indptr();
        let indptr = indptr_raw.as_slice().expect("non-contiguous indptr").to_vec();
        let indices = x_sp.indices().to_vec();
        let data = x_sp.data().to_vec();
        let n = x_sp.rows();
        CpuBackend {
            indptr, indices, data, n,
            mu: None,
            t_defl: Vec::new(),
            p_defl: Vec::new(),
        }
    }

    fn spmv_forward(&self, v: &[f64], n: usize) -> Vec<f64> {
        let mut result = vec![0.0_f64; n];
        for row in 0..n {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.data[idx] * v[self.indices[idx]];
            }
            result[row] = sum;
        }
        result
    }

    fn spmv_transpose(&self, v: &[f64], p: usize) -> Vec<f64> {
        let mut result = vec![0.0_f64; p];
        for row in 0..self.n {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let vi = v[row];
            for idx in start..end {
                result[self.indices[idx]] += self.data[idx] * vi;
            }
        }
        result
    }

    /// Apply dense corrections for forward: result -= (mu·v)*1 - Σ(p_k·v)*t_k
    fn apply_forward_corrections(&self, result: &mut Array1<f64>, v: &Array1<f64>) {
        if let Some(mu) = &self.mu {
            let mu_dot_v = mu.dot(v);
            *result -= mu_dot_v;
        }
        for (t_k, p_k) in self.t_defl.iter().zip(self.p_defl.iter()) {
            let coeff = p_k.dot(v);
            result.scaled_add(-coeff, t_k);
        }
    }

    /// Apply dense corrections for transpose: result -= sum(v)*mu - Σ(t_k·v)*p_k
    fn apply_transpose_corrections(&self, result: &mut Array1<f64>, v: &Array1<f64>) {
        if let Some(mu) = &self.mu {
            let sum_v = v.sum();
            result.scaled_add(-sum_v, mu);
        }
        for (t_k, p_k) in self.t_defl.iter().zip(self.p_defl.iter()) {
            let coeff = t_k.dot(v);
            result.scaled_add(-coeff, p_k);
        }
    }
}

fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

impl OplsBackend for CpuBackend {
    fn set_mean(&mut self, mu: Array1<f64>) {
        self.mu = Some(mu);
    }

    fn push_deflation(&mut self, t: Array1<f64>, p: Array1<f64>) {
        self.t_defl.push(t);
        self.p_defl.push(p);
    }

    fn truncate_deflation(&mut self, keep: usize) {
        self.t_defl.truncate(keep);
        self.p_defl.truncate(keep);
    }

    fn num_deflation(&self) -> usize {
        self.t_defl.len()
    }

    fn xeff_forward(&self, v: &Array1<f64>, n: usize) -> Array1<f64> {
        let v_slice = v.as_slice().expect("non-contiguous v");
        let mut result = Array1::from_vec(self.spmv_forward(v_slice, n));
        self.apply_forward_corrections(&mut result, v);
        result
    }

    fn xeff_transpose(&self, v: &Array1<f64>, p: usize) -> Array1<f64> {
        let v_slice = v.as_slice().expect("non-contiguous v");
        let mut result = Array1::from_vec(self.spmv_transpose(v_slice, p));
        self.apply_transpose_corrections(&mut result, v);
        result
    }

    fn nipals_wt(&self, u: &Array1<f64>, n: usize, p: usize) -> (Array1<f64>, Array1<f64>, f64) {
        let mut w = self.xeff_transpose(u, p);
        let n_w = norm(&w);
        if n_w > 0.0 {
            w /= n_w;
        }
        let t = self.xeff_forward(&w, n);
        let tt = t.dot(&t);
        (w, t, tt)
    }
}
