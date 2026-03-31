use ndarray::Array1;

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;

/// Backend trait that encapsulates sparse mat-vec and dense corrections.
/// The GPU backend keeps vectors on-device; the CPU backend uses ndarray.
pub trait OplsBackend {
    /// Set the column mean vector (uploaded to device for GPU backend).
    fn set_mean(&mut self, mu: Array1<f64>);

    /// Add a (t, p) deflation pair.
    fn push_deflation(&mut self, t: Array1<f64>, p: Array1<f64>);

    /// Truncate deflation vectors to `keep` pairs.
    fn truncate_deflation(&mut self, keep: usize);

    /// Number of stored deflation pairs.
    fn num_deflation(&self) -> usize;

    /// Compute X_eff * v = X_sp*v - (mu·v)*1 - Σ(p_k·v)*t_k
    fn xeff_forward(&self, v: &Array1<f64>, n: usize) -> Array1<f64>;

    /// Compute X_eff^T * v = X_sp^T*v - sum(v)*mu - Σ(t_k·v)*p_k
    fn xeff_transpose(&self, v: &Array1<f64>, p: usize) -> Array1<f64>;

    /// Fused: w = normalize(X_eff^T * u), t = X_eff * w. Returns (w, t, t·t).
    /// GPU backend keeps w on device between the two SpMVs.
    fn nipals_wt(&self, u: &Array1<f64>, n: usize, p: usize) -> (Array1<f64>, Array1<f64>, f64);
}

/// Backend selection for fit/predict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Automatically select GPU if available, otherwise CPU.
    Auto,
    /// Force CPU backend.
    Cpu,
}

/// Create the best available backend for the given sparse matrix.
pub fn create_backend(x_sp: &sprs::CsMat<f64>) -> Box<dyn OplsBackend> {
    create_backend_with(x_sp, Backend::Auto)
}

/// Create a backend with explicit selection.
pub fn create_backend_with(x_sp: &sprs::CsMat<f64>, choice: Backend) -> Box<dyn OplsBackend> {
    #[cfg(feature = "cuda")]
    if choice != Backend::Cpu {
        match gpu::GpuBackend::try_new(x_sp) {
            Ok(b) => return Box::new(b),
            Err(e) => {
                eprintln!("sparse_olps: GPU init failed ({e}), falling back to CPU");
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = choice;
    Box::new(cpu::CpuBackend::new(x_sp))
}
