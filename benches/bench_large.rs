use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sparse_olps::OplsModel;
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

fn bench_config(n: usize, p: usize, density: f64, n_pred: usize, n_orth: usize) {
    let m = 1;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    println!("  Matrix: {n} x {p}, density={density}");
    let x = generate_sparse_matrix(&mut rng, n, p, density);
    println!("  nnz = {} ({:.1}M)", x.nnz(), x.nnz() as f64 / 1e6);

    let y_data: Vec<f64> = (0..n * m).map(|_| rng.random::<f64>() * 10.0).collect();
    let y = Array2::from_shape_vec((n, m), y_data).unwrap();

    // Run 3 times, take best
    let mut best_fit = f64::MAX;
    let mut best_predict = f64::MAX;
    for run in 0..3 {
        let start = Instant::now();
        let model = OplsModel::fit(&x, &y, n_pred, n_orth);
        let fit_time = start.elapsed().as_secs_f64();
        best_fit = best_fit.min(fit_time);

        let start = Instant::now();
        let _y_hat = model.predict(&x);
        let predict_time = start.elapsed().as_secs_f64();
        best_predict = best_predict.min(predict_time);

        if run == 0 {
            // Print dispersion stats on first run
            let d = &model.dispersions;
            let d_slice = d.as_slice().unwrap();
            let mean_d: f64 = d_slice.iter().sum::<f64>() / d_slice.len() as f64;
            let min_d = d_slice.iter().cloned().fold(f64::MAX, f64::min);
            let max_d = d_slice.iter().cloned().fold(f64::MIN, f64::max);
            println!("  dispersions: mean={mean_d:.4}, min={min_d:.6}, max={max_d:.4}");
        }
    }
    println!("  fit:     {best_fit:.3}s");
    println!("  predict: {best_predict:.3}s");
}

fn main() {
    println!("Benchmark: sparse OPLS with DESeq2-style VST");

    println!();
    println!("--- 10k samples x 20k features, 1% density ---");
    bench_config(10_000, 20_000, 0.01, 2, 2);

    println!();
    println!("--- 10k samples x 20k features, 5% density ---");
    bench_config(10_000, 20_000, 0.05, 2, 2);

    println!();
    println!("--- 50k samples x 20k features, 1% density ---");
    bench_config(50_000, 20_000, 0.01, 2, 2);
}
