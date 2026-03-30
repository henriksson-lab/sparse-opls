use ndarray::{Array1, Array2, Axis, s};
use sprs::CsMatI;

pub mod backend;
use backend::OplsBackend;

/// Fitted OPLS model containing all components needed for prediction.
pub struct OplsModel {
    pub dispersions: Array1<f64>,
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub weights: Array2<f64>,
    pub loadings_x: Array2<f64>,
    pub loadings_y: Array2<f64>,
    pub weights_orth: Array2<f64>,
    pub loadings_orth: Array2<f64>,
    pub coefficients: Array2<f64>,
}

// ── Preprocessing ──────────────────────────────────────────────────────────

const MIN_DISPERSION: f64 = 1e-8;
const MEAN_THRESHOLD: f64 = 1e-8;

fn sparse_row_normalize(x: &CsMatI<f64, usize>) -> sprs::CsMat<f64> {
    let n = x.rows();
    let p = x.cols();
    let mut row_sums = vec![0.0_f64; n];
    for (&val, (r, _)) in x.iter() {
        row_sums[r] += val;
    }
    let mut tri = sprs::TriMat::new((n, p));
    for (&val, (r, c)) in x.iter() {
        let rs = row_sums[r];
        if rs > 0.0 {
            let v = val / rs;
            if v != 0.0 {
                tri.add_triplet(r, c, v);
            }
        }
    }
    tri.to_csr()
}

fn estimate_dispersions(x_norm: &sprs::CsMat<f64>, n: usize) -> Array1<f64> {
    let p = x_norm.cols();
    let nf = n as f64;
    let mut col_sum = vec![0.0_f64; p];
    let mut col_sum_sq = vec![0.0_f64; p];
    for (&val, (_, c)) in x_norm.iter() {
        col_sum[c] += val;
        col_sum_sq[c] += val * val;
    }
    let mut means = vec![0.0_f64; p];
    let mut raw_alpha = vec![0.0_f64; p];
    let mut valid = vec![false; p];
    for j in 0..p {
        let mu = col_sum[j] / nf;
        means[j] = mu;
        if mu < MEAN_THRESHOLD { continue; }
        let var = (col_sum_sq[j] - nf * mu * mu) / (nf - 1.0);
        let alpha = (var - mu) / (mu * mu);
        if alpha > 0.0 && alpha.is_finite() {
            raw_alpha[j] = alpha;
            valid[j] = true;
        }
    }
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut count = 0.0_f64;
    for j in 0..p {
        if valid[j] {
            let x = 1.0 / means[j];
            let y = raw_alpha[j];
            if x.is_finite() && y < 1e6 {
                sum_x += x; sum_y += y;
                sum_xx += x * x; sum_xy += x * y;
                count += 1.0;
            }
        }
    }
    let (a1, a0) = if count >= 2.0 {
        let denom = count * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-30 {
            let a1 = (count * sum_xy - sum_x * sum_y) / denom;
            let a0 = (sum_y - a1 * sum_x) / count;
            (a1.max(0.0), a0.max(MIN_DISPERSION))
        } else {
            (0.0, (sum_y / count).max(MIN_DISPERSION))
        }
    } else {
        (0.0, 0.01)
    };
    let mut dispersions = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mu = means[j];
        dispersions[j] = if mu > MEAN_THRESHOLD {
            (a1 / mu + a0).max(MIN_DISPERSION)
        } else {
            a0.max(MIN_DISPERSION)
        };
    }
    dispersions
}

fn sparse_apply_vst(x_norm: &sprs::CsMat<f64>, dispersions: &Array1<f64>) -> sprs::CsMat<f64> {
    let n = x_norm.rows();
    let p = x_norm.cols();
    let mut tri = sprs::TriMat::new((n, p));
    for (&val, (r, c)) in x_norm.iter() {
        let alpha = dispersions[c];
        let v = 2.0 / alpha.sqrt() * (alpha * val).sqrt().asinh();
        if v != 0.0 { tri.add_triplet(r, c, v); }
    }
    tri.to_csr()
}

fn preprocess_fit(x: &CsMatI<f64, usize>, n: usize) -> (sprs::CsMat<f64>, Array1<f64>) {
    let x_norm = sparse_row_normalize(x);
    let dispersions = estimate_dispersions(&x_norm, n);
    let x_vst = sparse_apply_vst(&x_norm, &dispersions);
    (x_vst, dispersions)
}

fn preprocess_predict(x: &CsMatI<f64, usize>, dispersions: &Array1<f64>) -> sprs::CsMat<f64> {
    let x_norm = sparse_row_normalize(x);
    sparse_apply_vst(&x_norm, dispersions)
}

fn sparse_col_means(x: &sprs::CsMat<f64>, n: usize) -> Array1<f64> {
    let p = x.cols();
    let mut sums = Array1::<f64>::zeros(p);
    for (&val, (_, c)) in x.iter() { sums[c] += val; }
    sums / n as f64
}

// ── Linear algebra helpers ─────────────────────────────────────────────────

fn norm(v: &Array1<f64>) -> f64 { v.dot(v).sqrt() }

fn normalize(v: &mut Array1<f64>) {
    let n = norm(v);
    if n > 0.0 { *v /= n; }
}

fn solve_coefficients(w: &Array2<f64>, p: &Array2<f64>, c: &Array2<f64>) -> Array2<f64> {
    let ptw = p.t().dot(w);
    let a = ptw.nrows();
    let mut aug = Array2::<f64>::zeros((a, 2 * a));
    aug.slice_mut(s![.., ..a]).assign(&ptw);
    for i in 0..a { aug[[i, a + i]] = 1.0; }
    for col in 0..a {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..a {
            let v = aug[[row, col]].abs();
            if v > max_val { max_val = v; max_row = row; }
        }
        if max_row != col {
            for j in 0..(2 * a) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        assert!(pivot.abs() > 1e-15, "Singular matrix in coefficient solve");
        for j in 0..(2 * a) { aug[[col, j]] /= pivot; }
        for row in 0..a {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * a) { aug[[row, j]] -= factor * aug[[col, j]]; }
            }
        }
    }
    let inv_ptw = aug.slice(s![.., a..]).to_owned();
    w.dot(&inv_ptw).dot(&c.t())
}

fn center_columns(mat: &mut Array2<f64>) -> Array1<f64> {
    let means = mat.mean_axis(Axis(0)).unwrap();
    for mut row in mat.rows_mut() { row -= &means; }
    means
}

// ── NIPALS PLS weight computation ──────────────────────────────────────────

fn pls_weight_single(be: &dyn OplsBackend, y: &Array1<f64>, p: usize) -> Array1<f64> {
    let mut w = be.xeff_transpose(y, p);
    normalize(&mut w);
    w
}

fn pls_weight_multi(
    be: &dyn OplsBackend,
    y: &Array2<f64>,
    n: usize,
    p: usize,
    max_iter: usize,
    tol: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut u = y.column(0).to_owned();
    for _ in 0..max_iter {
        let (w, t, tt) = be.nipals_wt(&u, n, p);
        let c = y.t().dot(&t) / tt;
        let u_new = y.dot(&c) / c.dot(&c);
        let diff = norm(&(&u_new - &u));
        u = u_new;
        if diff < tol {
            // Final computation with converged u
            let (w, t, tt) = be.nipals_wt(&u, n, p);
            let c = y.t().dot(&t) / tt;
            return (w, c);
        }
        // Keep w from last iteration for potential use
        if false { let _ = w; } // suppress unused warning during loop
    }
    let (_, t, tt) = be.nipals_wt(&u, n, p);
    let c = y.t().dot(&t) / tt;
    let mut w = be.xeff_transpose(&u, p);
    normalize(&mut w);
    (w, c)
}

// ── OPLS fitting ───────────────────────────────────────────────────────────

impl OplsModel {
    pub fn fit(
        x: &CsMatI<f64, usize>,
        y: &Array2<f64>,
        n_predictive: usize,
        n_orthogonal: usize,
    ) -> Self {
        let n = y.nrows();
        let p = x.cols();
        let m = y.ncols();

        let (x_sp, dispersions) = preprocess_fit(x, n);
        let x_mean = sparse_col_means(&x_sp, n);
        let mut be = backend::create_backend(&x_sp);
        be.set_mean(x_mean.clone());

        let mut yd = y.clone();
        let y_mean = center_columns(&mut yd);

        let mut weights_orth = Array2::<f64>::zeros((p, n_orthogonal));
        let mut loadings_orth = Array2::<f64>::zeros((p, n_orthogonal));

        if n_orthogonal > 0 {
            let w = if m == 1 {
                pls_weight_single(be.as_ref(), &yd.column(0).to_owned(), p)
            } else {
                pls_weight_multi(be.as_ref(), &yd, n, p, 500, 1e-10).0
            };

            let mut t = be.xeff_forward(&w, n);
            let mut p_loading = be.xeff_transpose(&t, p) / t.dot(&t);

            for a in 0..n_orthogonal {
                let mut w_orth = &p_loading - &(&w * (w.dot(&p_loading) / w.dot(&w)));
                normalize(&mut w_orth);

                let t_orth = be.xeff_forward(&w_orth, n);
                let tt_orth = t_orth.dot(&t_orth);
                let p_orth = be.xeff_transpose(&t_orth, p) / tt_orth;

                be.push_deflation(t_orth, p_orth.clone());

                weights_orth.column_mut(a).assign(&w_orth);
                loadings_orth.column_mut(a).assign(&p_orth);

                t = be.xeff_forward(&w, n);
                p_loading = be.xeff_transpose(&t, p) / t.dot(&t);
            }
        }

        let n_orth_defl = be.num_deflation();

        let mut weights = Array2::<f64>::zeros((p, n_predictive));
        let mut loadings_x = Array2::<f64>::zeros((p, n_predictive));
        let mut loadings_y = Array2::<f64>::zeros((m, n_predictive));

        for a in 0..n_predictive {
            let (w, c) = if m == 1 {
                let w = pls_weight_single(be.as_ref(), &yd.column(0).to_owned(), p);
                let t = be.xeff_forward(&w, n);
                let tt = t.dot(&t);
                let c_val = yd.column(0).dot(&t) / tt;
                let mut c = Array1::<f64>::zeros(1);
                c[0] = c_val;
                (w, c)
            } else {
                pls_weight_multi(be.as_ref(), &yd, n, p, 500, 1e-10)
            };

            let t = be.xeff_forward(&w, n);
            let tt = t.dot(&t);
            let p_loading = be.xeff_transpose(&t, p) / tt;

            weights.column_mut(a).assign(&w);
            loadings_x.column_mut(a).assign(&p_loading);
            loadings_y.column_mut(a).assign(&c);

            be.push_deflation(t.clone(), p_loading);

            for i in 0..n {
                for j in 0..m {
                    yd[[i, j]] -= t[i] * c[j];
                }
            }
        }

        be.truncate_deflation(n_orth_defl);

        let coefficients = solve_coefficients(&weights, &loadings_x, &loadings_y);

        OplsModel {
            dispersions, x_mean, y_mean,
            weights, loadings_x, loadings_y,
            weights_orth, loadings_orth, coefficients,
        }
    }

    pub fn predict(&self, x: &CsMatI<f64, usize>) -> Array2<f64> {
        let n = x.rows();
        let x_sp = preprocess_predict(x, &self.dispersions);
        let mut be = backend::create_backend(&x_sp);
        be.set_mean(self.x_mean.clone());

        let n_orth = self.weights_orth.ncols();
        for a in 0..n_orth {
            let w_orth = self.weights_orth.column(a).to_owned();
            let t_orth = be.xeff_forward(&w_orth, n);
            let p_orth = self.loadings_orth.column(a).to_owned();
            be.push_deflation(t_orth, p_orth);
        }

        let m = self.coefficients.ncols();
        let mut y_hat = Array2::<f64>::zeros((n, m));
        for j in 0..m {
            let b_col = self.coefficients.column(j).to_owned();
            let col = be.xeff_forward(&b_col, n);
            y_hat.column_mut(j).assign(&col);
        }
        for mut row in y_hat.rows_mut() {
            row += &self.y_mean;
        }
        y_hat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::CsMat;

    fn sparse_from_dense(data: &[&[f64]], rows: usize, cols: usize) -> CsMat<f64> {
        let mut tri = sprs::TriMat::new((rows, cols));
        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 { tri.add_triplet(i, j, val); }
            }
        }
        tri.to_csr()
    }

    #[test]
    fn test_vst_preserves_sparsity() {
        let x = sparse_from_dense(&[&[2.0, 0.0, 8.0, 0.0, 0.0]], 1, 5);
        let x_norm = sparse_row_normalize(&x);
        let disp = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let x_vst = sparse_apply_vst(&x_norm, &disp);
        assert_eq!(x_vst.nnz(), 2);
        let dense = x_vst.to_dense();
        assert_eq!(dense[[0, 1]], 0.0);
        assert!(dense[[0, 0]] > 0.0);
        assert!(dense[[0, 2]] > 0.0);
    }

    #[test]
    fn test_vst_formula() {
        let alpha = 0.5_f64;
        let expected = 2.0 / alpha.sqrt() * (alpha * 0.3).sqrt().asinh();
        let x = sparse_from_dense(&[&[3.0, 0.0, 7.0]], 1, 3);
        let x_norm = sparse_row_normalize(&x);
        let disp = Array1::from_vec(vec![alpha, alpha, alpha]);
        let x_vst = sparse_apply_vst(&x_norm, &disp);
        let dense = x_vst.to_dense();
        assert!((dense[[0, 0]] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_dispersion_estimation() {
        let x = sparse_from_dense(
            &[
                &[100.0, 1.0, 50.0, 2.0],
                &[90.0, 2.0, 55.0, 1.0],
                &[110.0, 1.0, 45.0, 3.0],
                &[95.0, 3.0, 60.0, 1.0],
                &[105.0, 2.0, 40.0, 2.0],
                &[100.0, 1.0, 50.0, 1.0],
            ],
            6, 4,
        );
        let x_norm = sparse_row_normalize(&x);
        let disp = estimate_dispersions(&x_norm, 6);
        for j in 0..4 {
            assert!(disp[j] > 0.0);
        }
    }

    #[test]
    fn test_implicit_xeff_matches_explicit() {
        let x = sparse_from_dense(
            &[&[10.0, 0.0, 5.0], &[0.0, 8.0, 2.0], &[3.0, 3.0, 4.0]],
            3, 3,
        );
        let (x_sp, _) = preprocess_fit(&x, 3);
        let mu = sparse_col_means(&x_sp, 3);
        let mut be = backend::create_backend(&x_sp);
        be.set_mean(mu.clone());

        let t_k = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let p_k = Array1::from_vec(vec![0.2, 0.8, -0.1]);
        be.push_deflation(t_k.clone(), p_k.clone());

        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result_impl = be.xeff_forward(&v, 3);

        // Explicit dense computation
        let dense = x_sp.to_dense();
        let mut xd = dense.mapv(|x| x);
        for mut row in xd.rows_mut() { row -= &mu; }
        for i in 0..3 { for j in 0..3 { xd[[i, j]] -= t_k[i] * p_k[j]; } }
        let result_expl = xd.dot(&v);

        for i in 0..3 {
            assert!((result_impl[i] - result_expl[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fit_and_predict_single_y() {
        let x = sparse_from_dense(
            &[
                &[10.0, 0.0, 5.0, 1.0], &[8.0, 2.0, 3.0, 1.0],
                &[0.0, 12.0, 1.0, 3.0], &[1.0, 10.0, 0.0, 5.0],
                &[5.0, 5.0, 4.0, 2.0],  &[3.0, 7.0, 2.0, 4.0],
            ],
            6, 4,
        );
        let y = Array2::from_shape_vec((6, 1), vec![1.0, 1.2, 3.0, 3.5, 2.0, 2.5]).unwrap();
        let model = OplsModel::fit(&x, &y, 1, 1);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 6);
        for i in 0..6 {
            assert!(y_hat[[i, 0]] > 0.0 && y_hat[[i, 0]] < 5.0,
                "prediction {} out of range: {}", i, y_hat[[i, 0]]);
        }
    }

    #[test]
    fn test_fit_and_predict_multi_y() {
        let x = sparse_from_dense(
            &[
                &[10.0, 0.0, 5.0, 1.0], &[8.0, 2.0, 3.0, 1.0],
                &[0.0, 12.0, 1.0, 3.0], &[1.0, 10.0, 0.0, 5.0],
                &[5.0, 5.0, 4.0, 2.0],  &[3.0, 7.0, 2.0, 4.0],
            ],
            6, 4,
        );
        let y = Array2::from_shape_vec(
            (6, 2), vec![1.0, 5.0, 1.2, 4.8, 3.0, 2.0, 3.5, 1.5, 2.0, 3.0, 2.5, 2.5],
        ).unwrap();
        let model = OplsModel::fit(&x, &y, 2, 1);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.ncols(), 2);
    }

    #[test]
    fn test_zero_orthogonal_components() {
        let x = sparse_from_dense(
            &[&[10.0, 0.0, 5.0], &[0.0, 10.0, 1.0], &[5.0, 5.0, 3.0], &[8.0, 1.0, 4.0]],
            4, 3,
        );
        let y = Array2::from_shape_vec((4, 1), vec![1.0, 3.0, 2.0, 1.5]).unwrap();
        let model = OplsModel::fit(&x, &y, 1, 0);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 4);
    }
}
