use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sparse_olps::{Backend, OplsModel};
use std::time::Instant;

fn generate_sparse_matrix(
    rng: &mut ChaCha8Rng,
    n: usize,
    p: usize,
    density: f64,
) -> sprs::CsMat<f64> {
    let mut tri = sprs::TriMat::new((n, p));
    for i in 0..n {
        for j in 0..p {
            if rng.random::<f64>() < density {
                tri.add_triplet(i, j, rng.random_range(1.0..100.0));
            }
        }
    }
    tri.to_csr()
}

struct BenchResult {
    best_fit: f64,
    best_predict: f64,
}

impl BenchResult {
    fn total(&self) -> f64 {
        self.best_fit + self.best_predict
    }
}

fn bench_backend(
    x: &sprs::CsMat<f64>,
    y: &Array2<f64>,
    n_pred: usize,
    n_orth: usize,
    backend: Backend,
    label: &str,
) -> BenchResult {
    let mut best_fit = f64::MAX;
    let mut best_predict = f64::MAX;
    for run in 0..3 {
        eprint!("  {label} run {}/3: fit... ", run + 1);
        let start = Instant::now();
        let model = OplsModel::fit_with(x, y, n_pred, n_orth, backend);
        let fit_time = start.elapsed().as_secs_f64();
        best_fit = best_fit.min(fit_time);
        eprint!("{fit_time:.1}s, predict... ");

        let start = Instant::now();
        let _y_hat = model.predict_with(x, backend);
        let predict_time = start.elapsed().as_secs_f64();
        best_predict = best_predict.min(predict_time);
        eprintln!("{predict_time:.1}s");

        if run == 0 {
            let d = &model.dispersions;
            let d_slice = d.as_slice().unwrap();
            let mean_d: f64 = d_slice.iter().sum::<f64>() / d_slice.len() as f64;
            println!("  dispersions: mean={mean_d:.4}");
        }
    }
    println!("  {label} best: fit={best_fit:.3}s  predict={best_predict:.3}s  total={:.3}s",
        best_fit + best_predict);
    BenchResult { best_fit, best_predict }
}

fn bench_config(n: usize, p: usize, m: usize, density: f64, n_pred: usize, n_orth: usize, run_cpu: bool) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    eprint!("  Generating matrix... ");
    let x = generate_sparse_matrix(&mut rng, n, p, density);
    eprintln!("done ({:.1}M nnz)", x.nnz() as f64 / 1e6);
    println!("  Matrix: {n} x {p}, m={m}, density={density}, nnz={:.1}M", x.nnz() as f64 / 1e6);

    let y_data: Vec<f64> = (0..n * m).map(|_| rng.random::<f64>() * 10.0).collect();
    let y = Array2::from_shape_vec((n, m), y_data).unwrap();

    let cpu_result = if run_cpu {
        Some(bench_backend(&x, &y, n_pred, n_orth, Backend::Cpu, "CPU"))
    } else {
        None
    };

    let gpu_result = bench_backend(&x, &y, n_pred, n_orth, Backend::Auto, "GPU");

    if let Some(cpu) = cpu_result {
        let speedup = cpu.total() / gpu_result.total();
        println!("  Speedup: {speedup:.2}x");
    }
}

fn main() {
    println!("Benchmark: sparse OPLS with DESeq2-style VST (m=5)");

    println!();
    println!("--- 10k samples x 20k features, 1% density ---");
    bench_config(10_000, 20_000, 5, 0.01, 2, 2, true);

    println!();
    println!("--- 10k samples x 20k features, 5% density ---");
    bench_config(10_000, 20_000, 5, 0.05, 2, 2, true);

    println!();
    println!("--- 50k samples x 20k features, 1% density ---");
    bench_config(50_000, 20_000, 5, 0.01, 2, 2, true);

    println!();
    println!("--- 500k samples x 20k features, 5% density ---");
    bench_config(500_000, 20_000, 5, 0.05, 2, 2, false);
}
