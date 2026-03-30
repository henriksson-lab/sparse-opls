use ndarray::{Array1, Array2, Axis, s};
use sprs::CsMatI;

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

// ── Preprocessing ──────────────────────────────────────────────────────────

/// Row-normalize sparse X (divide each row by its sum), apply pseudolog, return dense.
fn preprocess_x(x: &CsMatI<f64, usize>, n: usize, p: usize) -> Array2<f64> {
    let mut dense = Array2::<f64>::zeros((n, p));
    // Compute row sums
    let mut row_sums = vec![0.0_f64; n];
    for (&val, (r, _c)) in x.iter() {
        row_sums[r] += val;
    }
    // Normalize and pseudolog
    for (&val, (r, c)) in x.iter() {
        let rs = row_sums[r];
        if rs > 0.0 {
            dense[[r, c]] = (val / rs + 1.0).ln();
        }
    }
    dense
}

/// Center columns in-place, return the column means.
fn center_columns(mat: &mut Array2<f64>) -> Array1<f64> {
    let means = mat.mean_axis(Axis(0)).unwrap();
    for mut row in mat.rows_mut() {
        row -= &means;
    }
    means
}

// ── Linear algebra helpers ─────────────────────────────────────────────────

fn dot_mv(a: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    a.dot(v)
}

fn dot_mtv(a: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    a.t().dot(v)
}

fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn normalize(v: &mut Array1<f64>) {
    let n = norm(v);
    if n > 0.0 {
        *v /= n;
    }
}

/// Solve B = W (P^T W)^{-1} C^T  for small square (P^T W).
fn solve_coefficients(
    w: &Array2<f64>,
    p: &Array2<f64>,
    c: &Array2<f64>,
) -> Array2<f64> {
    let ptw = p.t().dot(w); // (A x A)
    let a = ptw.nrows();
    // Invert P^T W via Gauss-Jordan
    let mut aug = Array2::<f64>::zeros((a, 2 * a));
    aug.slice_mut(s![.., ..a]).assign(&ptw);
    for i in 0..a {
        aug[[i, a + i]] = 1.0;
    }
    for col in 0..a {
        // Partial pivot
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

// ── NIPALS PLS weight computation ──────────────────────────────────────────

/// Compute PLS weight vector w for single-Y case.
fn pls_weight_single(x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut w = dot_mtv(x, y);
    normalize(&mut w);
    w
}

/// Compute PLS weight vector w and Y-loading c for multi-Y case via NIPALS.
fn pls_weight_multi(x: &Array2<f64>, y: &Array2<f64>, max_iter: usize, tol: f64) -> (Array1<f64>, Array1<f64>) {
    let mut u = y.column(0).to_owned();
    let mut w;
    let mut t;
    let mut c;
    for _ in 0..max_iter {
        w = dot_mtv(x, &u);
        normalize(&mut w);
        t = dot_mv(x, &w);
        let tt = t.dot(&t);
        c = y.t().dot(&t) / tt;
        let u_new = y.dot(&c) / c.dot(&c);
        let diff = norm(&(&u_new - &u));
        u = u_new;
        if diff < tol {
            w = dot_mtv(x, &u);
            normalize(&mut w);
            t = dot_mv(x, &w);
            let tt2 = t.dot(&t);
            c = y.t().dot(&t) / tt2;
            return (w, c);
        }
    }
    w = dot_mtv(x, &u);
    normalize(&mut w);
    t = dot_mv(x, &w);
    let tt = t.dot(&t);
    c = y.t().dot(&t) / tt;
    (w, c)
}

// ── OPLS fitting ───────────────────────────────────────────────────────────

impl OplsModel {
    /// Fit an OPLS model.
    ///
    /// - `x`: sparse CSR matrix (n x p) of raw counts
    /// - `y`: dense matrix (n x m) of responses
    /// - `n_predictive`: number of predictive components
    /// - `n_orthogonal`: number of orthogonal components to remove
    pub fn fit(
        x: &CsMatI<f64, usize>,
        y: &Array2<f64>,
        n_predictive: usize,
        n_orthogonal: usize,
    ) -> Self {
        let n = y.nrows();
        let p = x.cols();
        let m = y.ncols();

        // Preprocess X: row normalize, pseudolog, center
        let mut xd = preprocess_x(x, n, p);
        let x_mean = center_columns(&mut xd);

        // Center Y
        let mut yd = y.clone();
        let y_mean = center_columns(&mut yd);

        // --- Orthogonal component extraction ---
        let mut weights_orth = Array2::<f64>::zeros((p, n_orthogonal));
        let mut loadings_orth = Array2::<f64>::zeros((p, n_orthogonal));

        if n_orthogonal > 0 {
            // Compute initial PLS weight
            let w = if m == 1 {
                pls_weight_single(&xd, &yd.column(0).to_owned())
            } else {
                pls_weight_multi(&xd, &yd, 500, 1e-10).0
            };

            let mut t = dot_mv(&xd, &w);
            let mut p_loading = dot_mtv(&xd, &t) / t.dot(&t);

            for a in 0..n_orthogonal {
                // Orthogonal weight: component of p orthogonal to w
                let mut w_orth = &p_loading - &(&w * (w.dot(&p_loading) / w.dot(&w)));
                normalize(&mut w_orth);

                // Orthogonal score and loading
                let t_orth = dot_mv(&xd, &w_orth);
                let tt_orth = t_orth.dot(&t_orth);
                let p_orth = dot_mtv(&xd, &t_orth) / tt_orth;

                // Deflate X
                for i in 0..n {
                    for j in 0..p {
                        xd[[i, j]] -= t_orth[i] * p_orth[j];
                    }
                }

                // Store
                weights_orth.column_mut(a).assign(&w_orth);
                loadings_orth.column_mut(a).assign(&p_orth);

                // Recompute predictive score and loading on deflated X
                t = dot_mv(&xd, &w);
                p_loading = dot_mtv(&xd, &t) / t.dot(&t);
            }
        }

        // --- Predictive PLS on filtered X ---
        let mut weights = Array2::<f64>::zeros((p, n_predictive));
        let mut loadings_x = Array2::<f64>::zeros((p, n_predictive));
        let mut loadings_y = Array2::<f64>::zeros((m, n_predictive));

        for a in 0..n_predictive {
            let (w, c) = if m == 1 {
                let w = pls_weight_single(&xd, &yd.column(0).to_owned());
                let t = dot_mv(&xd, &w);
                let tt = t.dot(&t);
                let c_val = yd.column(0).dot(&t) / tt;
                let mut c = Array1::<f64>::zeros(1);
                c[0] = c_val;
                (w, c)
            } else {
                pls_weight_multi(&xd, &yd, 500, 1e-10)
            };

            let t = dot_mv(&xd, &w);
            let tt = t.dot(&t);
            let p_loading = dot_mtv(&xd, &t) / tt;

            // Store
            weights.column_mut(a).assign(&w);
            loadings_x.column_mut(a).assign(&p_loading);
            loadings_y.column_mut(a).assign(&c);

            // Deflate X and Y
            for i in 0..n {
                for j in 0..p {
                    xd[[i, j]] -= t[i] * p_loading[j];
                }
                for j in 0..m {
                    yd[[i, j]] -= t[i] * c[j];
                }
            }
        }

        // Compute regression coefficients: B = W (P^T W)^{-1} C^T
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

    /// Predict Y for new sparse X data.
    pub fn predict(&self, x: &CsMatI<f64, usize>) -> Array2<f64> {
        let n = x.rows();
        let p = x.cols();

        // Preprocess: row normalize, pseudolog
        let mut xd = preprocess_x(x, n, p);

        // Center using training means
        for mut row in xd.rows_mut() {
            row -= &self.x_mean;
        }

        // Remove orthogonal components
        let n_orth = self.weights_orth.ncols();
        for a in 0..n_orth {
            let w_orth = self.weights_orth.column(a);
            let p_orth = self.loadings_orth.column(a);
            let t_orth = xd.dot(&w_orth);
            for i in 0..n {
                for j in 0..p {
                    xd[[i, j]] -= t_orth[i] * p_orth[j];
                }
            }
        }

        // Y_hat = X_filtered B + y_mean
        let mut y_hat = xd.dot(&self.coefficients);
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
    fn test_preprocess_row_normalization() {
        let x = sparse_from_dense(&[&[2.0, 0.0, 8.0]], 1, 3);
        let dense = preprocess_x(&x, 1, 3);
        let expected = [
            (0.2_f64 + 1.0).ln(),
            0.0,
            (0.8_f64 + 1.0).ln(),
        ];
        for j in 0..3 {
            assert!(
                (dense[[0, j]] - expected[j]).abs() < 1e-12,
                "column {j}: got {} expected {}",
                dense[[0, j]],
                expected[j]
            );
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
            6,
            4,
        );
        let y = Array2::from_shape_vec((6, 1), vec![1.0, 1.2, 3.0, 3.5, 2.0, 2.5]).unwrap();

        let model = OplsModel::fit(&x, &y, 1, 1);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 6);
        assert_eq!(y_hat.ncols(), 1);

        for i in 0..6 {
            assert!(
                y_hat[[i, 0]] > 0.0 && y_hat[[i, 0]] < 5.0,
                "prediction {} out of range: {}",
                i,
                y_hat[[i, 0]]
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
            6,
            4,
        );
        let y = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 5.0, 1.2, 4.8, 3.0, 2.0, 3.5, 1.5, 2.0, 3.0, 2.5, 2.5],
        )
        .unwrap();

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
            4,
            3,
        );
        let y = Array2::from_shape_vec((4, 1), vec![1.0, 3.0, 2.0, 1.5]).unwrap();

        let model = OplsModel::fit(&x, &y, 1, 0);
        let y_hat = model.predict(&x);
        assert_eq!(y_hat.nrows(), 4);
    }
}
