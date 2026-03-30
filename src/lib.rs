use ndarray::{Array1, Array2, Axis, s};
use sprs::CsMatI;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fitted OPLS model containing all components needed for prediction.
pub struct OplsModel {
    /// Column means of X after row-normalization and pseudolog (p,)
    pub x_mean: Array1<f64>,
    /// Column means of Y (m,)
    pub y_mean: Array1<f64>,
    /// Predictive weight vectors (p x A_pred)
    pub weights: Array2<f64>,
    /// Predictive X-loadings (p x A_pred)
    pub loadings_x: Array2<f64>,
    /// Y-loadings (m x A_pred)
    pub loadings_y: Array2<f64>,
    /// Orthogonal weight vectors (p x A_orth)
    pub weights_orth: Array2<f64>,
    /// Orthogonal X-loadings (p x A_orth)
    pub loadings_orth: Array2<f64>,
    /// Regression coefficients (p x m)
    pub coefficients: Array2<f64>,
}

// ── Implicit sparse X representation ───────────────────────────────────────
//
// X_eff = X_sparse_pl - 1·μ^T - T_defl · P_defl^T
//
// All matrix-vector products go through this representation so the n×p dense
// matrix is never materialized.

/// Row-normalize sparse X and apply pseudolog. Sparsity preserved since log(0+1)=0.
fn sparse_row_normalize_pseudolog(x: &CsMatI<f64, usize>) -> sprs::CsMat<f64> {
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
            let v = (val / rs + 1.0).ln();
            if v != 0.0 {
                tri.add_triplet(r, c, v);
            }
        }
    }
    tri.to_csr()
}

fn sparse_col_means(x: &sprs::CsMat<f64>, n: usize) -> Array1<f64> {
    let p = x.cols();
    let mut sums = Array1::<f64>::zeros(p);
    for (&val, (_, c)) in x.iter() {
        sums[c] += val;
    }
    sums / n as f64
}

// ── Sparse mat-vec: serial implementations ─────────────────────────────────

/// X_sp * v (CSR forward multiply, serial): result[row] = Σ_j X[row,j] * v[j]
#[cfg(not(feature = "parallel"))]
fn spmv_forward_serial(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    v: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut result = vec![0.0_f64; n];
    for row in 0..n {
        let start = indptr[row];
        let end = indptr[row + 1];
        let mut sum = 0.0;
        for idx in start..end {
            sum += data[idx] * v[indices[idx]];
        }
        result[row] = sum;
    }
    result
}

/// X_sp^T * v (CSR transpose multiply, serial): result[col] = Σ_i X[i,col] * v[i]
fn spmv_transpose_serial(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    v: &[f64],
    p: usize,
) -> Vec<f64> {
    let n = indptr.len() - 1;
    let mut result = vec![0.0_f64; p];
    for row in 0..n {
        let start = indptr[row];
        let end = indptr[row + 1];
        let vi = v[row];
        for idx in start..end {
            result[indices[idx]] += data[idx] * vi;
        }
    }
    result
}

// ── Sparse mat-vec: parallel implementations ───────────────────────────────

#[cfg(feature = "parallel")]
fn spmv_forward_parallel(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    v: &[f64],
    n: usize,
) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .map(|row| {
            let start = indptr[row];
            let end = indptr[row + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += data[idx] * v[indices[idx]];
            }
            sum
        })
        .collect()
}

// ── Unified dispatch ───────────────────────────────────────────────────────

fn spmv_forward(x_sp: &sprs::CsMat<f64>, v: &Array1<f64>, n: usize) -> Array1<f64> {
    let indptr_raw = x_sp.indptr();
    let indptr = indptr_raw.as_slice().expect("non-contiguous indptr");
    let indices = x_sp.indices();
    let data = x_sp.data();
    let v_slice = v.as_slice().expect("non-contiguous v");

    #[cfg(feature = "parallel")]
    let result = spmv_forward_parallel(indptr, indices, data, v_slice, n);
    #[cfg(not(feature = "parallel"))]
    let result = spmv_forward_serial(indptr, indices, data, v_slice, n);

    Array1::from_vec(result)
}

fn spmv_transpose(x_sp: &sprs::CsMat<f64>, v: &Array1<f64>, p: usize) -> Array1<f64> {
    // Transpose multiply is scatter-based (accumulate into result[col]).
    // Parallelizing with per-chunk buffers adds allocation overhead that
    // outweighs the gain for typical sizes. Keep serial.
    let indptr_raw = x_sp.indptr();
    let indptr = indptr_raw.as_slice().expect("non-contiguous indptr");
    let indices = x_sp.indices();
    let data = x_sp.data();
    let v_slice = v.as_slice().expect("non-contiguous v");
    let result = spmv_transpose_serial(indptr, indices, data, v_slice, p);
    Array1::from_vec(result)
}

// ── Implicit X_eff operations ──────────────────────────────────────────────

/// X_eff * v = X_sp * v - (μ·v)·1_n - Σ (p_k·v) * t_k
fn xeff_mul_vec(
    x_sp: &sprs::CsMat<f64>,
    mu: &Array1<f64>,
    t_defl: &[Array1<f64>],
    p_defl: &[Array1<f64>],
    v: &Array1<f64>,
    n: usize,
) -> Array1<f64> {
    let mut result = spmv_forward(x_sp, v, n);
    let mu_dot_v = mu.dot(v);
    result -= mu_dot_v;
    for (t_k, p_k) in t_defl.iter().zip(p_defl.iter()) {
        let coeff = p_k.dot(v);
        result.scaled_add(-coeff, t_k);
    }
    result
}

/// X_eff^T * v = X_sp^T * v - μ*sum(v) - Σ (t_k·v) * p_k
fn xeff_tmul_vec(
    x_sp: &sprs::CsMat<f64>,
    mu: &Array1<f64>,
    t_defl: &[Array1<f64>],
    p_defl: &[Array1<f64>],
    v: &Array1<f64>,
    p: usize,
) -> Array1<f64> {
    let mut result = spmv_transpose(x_sp, v, p);
    let sum_v = v.sum();
    result.scaled_add(-sum_v, mu);
    for (t_k, p_k) in t_defl.iter().zip(p_defl.iter()) {
        let coeff = t_k.dot(v);
        result.scaled_add(-coeff, p_k);
    }
    result
}

// ── Linear algebra helpers ─────────────────────────────────────────────────

fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn normalize(v: &mut Array1<f64>) {
    let n = norm(v);
    if n > 0.0 {
        *v /= n;
    }
}

fn solve_coefficients(
    w: &Array2<f64>,
    p: &Array2<f64>,
    c: &Array2<f64>,
) -> Array2<f64> {
    let ptw = p.t().dot(w);
    let a = ptw.nrows();
    let mut aug = Array2::<f64>::zeros((a, 2 * a));
    aug.slice_mut(s![.., ..a]).assign(&ptw);
    for i in 0..a {
        aug[[i, a + i]] = 1.0;
    }
    for col in 0..a {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..a {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
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
        for j in 0..(2 * a) {
            aug[[col, j]] /= pivot;
        }
        for row in 0..a {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * a) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }
    let inv_ptw = aug.slice(s![.., a..]).to_owned();
    w.dot(&inv_ptw).dot(&c.t())
}

fn center_columns(mat: &mut Array2<f64>) -> Array1<f64> {
    let means = mat.mean_axis(Axis(0)).unwrap();
    for mut row in mat.rows_mut() {
        row -= &means;
    }
    means
}

// ── NIPALS PLS weight computation (using implicit X) ───────────────────────

fn pls_weight_single(
    x_sp: &sprs::CsMat<f64>,
    mu: &Array1<f64>,
    t_defl: &[Array1<f64>],
    p_defl: &[Array1<f64>],
    y: &Array1<f64>,
    p: usize,
) -> Array1<f64> {
    let mut w = xeff_tmul_vec(x_sp, mu, t_defl, p_defl, y, p);
    normalize(&mut w);
    w
}

fn pls_weight_multi(
    x_sp: &sprs::CsMat<f64>,
    mu: &Array1<f64>,
    t_defl: &[Array1<f64>],
    p_defl: &[Array1<f64>],
    y: &Array2<f64>,
    n: usize,
    p: usize,
    max_iter: usize,
    tol: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut u = y.column(0).to_owned();
    for _ in 0..max_iter {
        let mut w = xeff_tmul_vec(x_sp, mu, t_defl, p_defl, &u, p);
        normalize(&mut w);
        let t = xeff_mul_vec(x_sp, mu, t_defl, p_defl, &w, n);
        let tt = t.dot(&t);
        let c = y.t().dot(&t) / tt;
        let u_new = y.dot(&c) / c.dot(&c);
        let diff = norm(&(&u_new - &u));
        u = u_new;
        if diff < tol {
            let mut w = xeff_tmul_vec(x_sp, mu, t_defl, p_defl, &u, p);
            normalize(&mut w);
            let t = xeff_mul_vec(x_sp, mu, t_defl, p_defl, &w, n);
            let tt2 = t.dot(&t);
            let c = y.t().dot(&t) / tt2;
            return (w, c);
        }
    }
    let mut w = xeff_tmul_vec(x_sp, mu, t_defl, p_defl, &u, p);
    normalize(&mut w);
    let t = xeff_mul_vec(x_sp, mu, t_defl, p_defl, &w, n);
    let tt = t.dot(&t);
    let c = y.t().dot(&t) / tt;
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

        let x_sp = sparse_row_normalize_pseudolog(x);
        let x_mean = sparse_col_means(&x_sp, n);

        let mut yd = y.clone();
        let y_mean = center_columns(&mut yd);

        let mut t_defl: Vec<Array1<f64>> = Vec::new();
        let mut p_defl: Vec<Array1<f64>> = Vec::new();

        let mut weights_orth = Array2::<f64>::zeros((p, n_orthogonal));
        let mut loadings_orth = Array2::<f64>::zeros((p, n_orthogonal));

        if n_orthogonal > 0 {
            let w = if m == 1 {
                pls_weight_single(&x_sp, &x_mean, &t_defl, &p_defl, &yd.column(0).to_owned(), p)
            } else {
                pls_weight_multi(&x_sp, &x_mean, &t_defl, &p_defl, &yd, n, p, 500, 1e-10).0
            };

            let mut t = xeff_mul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &w, n);
            let mut p_loading = xeff_tmul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &t, p) / t.dot(&t);

            for a in 0..n_orthogonal {
                let mut w_orth = &p_loading - &(&w * (w.dot(&p_loading) / w.dot(&w)));
                normalize(&mut w_orth);

                let t_orth = xeff_mul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &w_orth, n);
                let tt_orth = t_orth.dot(&t_orth);
                let p_orth = xeff_tmul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &t_orth, p) / tt_orth;

                t_defl.push(t_orth);
                p_defl.push(p_orth.clone());

                weights_orth.column_mut(a).assign(&w_orth);
                loadings_orth.column_mut(a).assign(&p_orth);

                t = xeff_mul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &w, n);
                p_loading = xeff_tmul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &t, p) / t.dot(&t);
            }
        }

        let n_orth_defl = t_defl.len();

        let mut weights = Array2::<f64>::zeros((p, n_predictive));
        let mut loadings_x = Array2::<f64>::zeros((p, n_predictive));
        let mut loadings_y = Array2::<f64>::zeros((m, n_predictive));

        for a in 0..n_predictive {
            let (w, c) = if m == 1 {
                let w = pls_weight_single(&x_sp, &x_mean, &t_defl, &p_defl, &yd.column(0).to_owned(), p);
                let t = xeff_mul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &w, n);
                let tt = t.dot(&t);
                let c_val = yd.column(0).dot(&t) / tt;
                let mut c = Array1::<f64>::zeros(1);
                c[0] = c_val;
                (w, c)
            } else {
                pls_weight_multi(&x_sp, &x_mean, &t_defl, &p_defl, &yd, n, p, 500, 1e-10)
            };

            let t = xeff_mul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &w, n);
            let tt = t.dot(&t);
            let p_loading = xeff_tmul_vec(&x_sp, &x_mean, &t_defl, &p_defl, &t, p) / tt;

            weights.column_mut(a).assign(&w);
            loadings_x.column_mut(a).assign(&p_loading);
            loadings_y.column_mut(a).assign(&c);

            t_defl.push(t.clone());
            p_defl.push(p_loading);

            for i in 0..n {
                for j in 0..m {
                    yd[[i, j]] -= t[i] * c[j];
                }
            }
        }

        t_defl.truncate(n_orth_defl);
        p_defl.truncate(n_orth_defl);

        let coefficients = solve_coefficients(&weights, &loadings_x, &loadings_y);

        OplsModel {
            x_mean,
            y_mean,
            weights,
            loadings_x,
            loadings_y,
            weights_orth,
            loadings_orth,
            coefficients,
        }
    }

    pub fn predict(&self, x: &CsMatI<f64, usize>) -> Array2<f64> {
        let n = x.rows();
        let x_sp = sparse_row_normalize_pseudolog(x);

        let n_orth = self.weights_orth.ncols();
        let mut t_defl: Vec<Array1<f64>> = Vec::new();
        let mut p_defl: Vec<Array1<f64>> = Vec::new();

        for a in 0..n_orth {
            let w_orth = self.weights_orth.column(a).to_owned();
            let t_orth = xeff_mul_vec(&x_sp, &self.x_mean, &t_defl, &p_defl, &w_orth, n);
            let p_orth = self.loadings_orth.column(a).to_owned();
            t_defl.push(t_orth);
            p_defl.push(p_orth);
        }

        let m = self.coefficients.ncols();
        let mut y_hat = Array2::<f64>::zeros((n, m));
        for j in 0..m {
            let b_col = self.coefficients.column(j).to_owned();
            let col = xeff_mul_vec(&x_sp, &self.x_mean, &t_defl, &p_defl, &b_col, n);
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
                if val != 0.0 {
                    tri.add_triplet(i, j, val);
                }
            }
        }
        tri.to_csr()
    }

    #[test]
    fn test_preprocess_preserves_sparsity() {
        let x = sparse_from_dense(&[&[2.0, 0.0, 8.0, 0.0, 0.0]], 1, 5);
        let x_pl = sparse_row_normalize_pseudolog(&x);
        assert_eq!(x_pl.nnz(), 2);
        let expected_0 = (0.2_f64 + 1.0).ln();
        let expected_2 = (0.8_f64 + 1.0).ln();
        let dense = x_pl.to_dense();
        assert!((dense[[0, 0]] - expected_0).abs() < 1e-12);
        assert_eq!(dense[[0, 1]], 0.0);
        assert!((dense[[0, 2]] - expected_2).abs() < 1e-12);
    }

    #[test]
    fn test_implicit_xeff_matches_explicit() {
        let x = sparse_from_dense(
            &[
                &[10.0, 0.0, 5.0],
                &[0.0, 8.0, 2.0],
                &[3.0, 3.0, 4.0],
            ],
            3, 3,
        );
        let x_sp = sparse_row_normalize_pseudolog(&x);
        let mu = sparse_col_means(&x_sp, 3);

        let t_k = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let p_k = Array1::from_vec(vec![0.2, 0.8, -0.1]);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result_impl = xeff_mul_vec(&x_sp, &mu, &[t_k.clone()], &[p_k.clone()], &v, 3);

        let dense = x_sp.to_dense();
        let mut xd = dense.mapv(|x| x);
        for mut row in xd.rows_mut() {
            row -= &mu;
        }
        for i in 0..3 {
            for j in 0..3 {
                xd[[i, j]] -= t_k[i] * p_k[j];
            }
        }
        let result_expl = xd.dot(&v);

        for i in 0..3 {
            assert!((result_impl[i] - result_expl[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fit_and_predict_single_y() {
        let x = sparse_from_dense(
            &[
                &[10.0, 0.0, 5.0, 1.0],
                &[8.0, 2.0, 3.0, 1.0],
                &[0.0, 12.0, 1.0, 3.0],
                &[1.0, 10.0, 0.0, 5.0],
                &[5.0, 5.0, 4.0, 2.0],
                &[3.0, 7.0, 2.0, 4.0],
            ],
            6, 4,
        );
        let y = Array2::from_shape_vec((6, 1), vec![1.0, 1.2, 3.0, 3.5, 2.0, 2.5]).unwrap();

        let model = OplsModel::fit(&x, &y, 1, 1);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 6);
        assert_eq!(y_hat.ncols(), 1);

        for i in 0..6 {
            assert!(
                y_hat[[i, 0]] > 0.0 && y_hat[[i, 0]] < 5.0,
                "prediction {} out of range: {}", i, y_hat[[i, 0]]
            );
        }
    }

    #[test]
    fn test_fit_and_predict_multi_y() {
        let x = sparse_from_dense(
            &[
                &[10.0, 0.0, 5.0, 1.0],
                &[8.0, 2.0, 3.0, 1.0],
                &[0.0, 12.0, 1.0, 3.0],
                &[1.0, 10.0, 0.0, 5.0],
                &[5.0, 5.0, 4.0, 2.0],
                &[3.0, 7.0, 2.0, 4.0],
            ],
            6, 4,
        );
        let y = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 5.0, 1.2, 4.8, 3.0, 2.0, 3.5, 1.5, 2.0, 3.0, 2.5, 2.5],
        ).unwrap();

        let model = OplsModel::fit(&x, &y, 2, 1);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 6);
        assert_eq!(y_hat.ncols(), 2);
    }

    #[test]
    fn test_zero_orthogonal_components() {
        let x = sparse_from_dense(
            &[
                &[10.0, 0.0, 5.0],
                &[0.0, 10.0, 1.0],
                &[5.0, 5.0, 3.0],
                &[8.0, 1.0, 4.0],
            ],
            4, 3,
        );
        let y = Array2::from_shape_vec((4, 1), vec![1.0, 3.0, 2.0, 1.5]).unwrap();

        let model = OplsModel::fit(&x, &y, 1, 0);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 4);
    }
}
