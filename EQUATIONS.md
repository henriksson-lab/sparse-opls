# OPLS: Orthogonal Projections to Latent Structures

Reference: Trygg & Wold (2002), "Orthogonal projections to latent structures (O-PLS)",
Journal of Chemometrics 16:119-128.

## Input

- **X**: n x p sparse matrix (observations x variables)
- **Y**: n x m dense matrix (observations x responses)

## Preprocessing: Normalization of X

Each row of X is divided by its total count (sum across all variables for that sample):

```
r_i = sum_j X_ij          (total count for sample i)
X_ij <- X_ij / r_i        for all j
```

This converts raw counts to relative abundances (compositional normalization).

After normalization, apply a variance-stabilizing transform (VST) to X.

### Dispersion estimation (DESeq2-style)

For negative-binomial count data, Var(K) = μ + α·μ², where α is the dispersion.

1. Compute per-feature mean and variance from normalized X:
   ```
   μ_j = (1/n) Σ_i X_ij
   σ²_j = (1/(n-1)) Σ_i (X_ij - μ_j)²
   ```

2. Estimate raw dispersions by method of moments:
   ```
   α_j^raw = (σ²_j - μ_j) / μ_j²
   ```

3. Fit a mean-dispersion trend (parametric):
   ```
   α(μ) = a_1 / μ + a_0
   ```
   Fit by linear regression of α_j^raw on 1/μ_j, using features with
   α_j^raw > 0 and μ_j > threshold.

4. Final dispersion: use the trend value for each feature:
   ```
   α_j = a_1 / μ_j + a_0
   ```
   Clamp to a minimum of 1e-8 to avoid degenerate transforms.

### Variance-stabilizing transform

The NB variance-stabilizing transform integrates 1/√Var(μ):

```
∫ dμ / √(μ + α·μ²) = (2/√α) · asinh(√(α·μ))
```

Apply per feature using its dispersion:

```
X_ij <- (2/√α_j) · asinh(√(α_j · X_ij))
```

Note: f(0) = 0, so sparsity is preserved.

Then center both matrices by subtracting column means:

```
x_mean_j = (1/n) sum_i X_ij
y_mean_k = (1/n) sum_i Y_ik

X_ij <- X_ij - x_mean_j
Y_ik <- Y_ik - y_mean_k
```

## OPLS Algorithm

The OPLS algorithm first identifies and removes structured variation in X that is
orthogonal to Y, then fits a PLS model on the filtered X.

### Step 1: Compute PLS weight vector

```
w = X^T Y / ||X^T Y||
```

For a single-column Y (m=1), this simplifies to:

```
w = X^T y / ||X^T y||
```

For multi-column Y (m>1), use the first PLS component via NIPALS:

1. Initialize u to first column of Y
2. Iterate until convergence:
   a. w = X^T u / ||X^T u||        (X-weight, normalized)
   b. t = X w                       (X-score)
   c. c = Y^T t / (t^T t)          (Y-loading)
   d. u = Y c / (c^T c)            (Y-score, update)

### Step 2: Compute predictive scores and loadings

```
t = X w                             (predictive score vector)
p = X^T t / (t^T t)                 (X-loading for predictive component)
```

### Step 3: Extract orthogonal components

Repeat for each desired orthogonal component (a = 1, ..., A_orth):

1. Compute orthogonal weight:
   ```
   w_orth = p - (w^T p / w^T w) w
   w_orth = w_orth / ||w_orth||
   ```
   This is the part of p that is orthogonal to w (Gram-Schmidt).

2. Compute orthogonal score and loading:
   ```
   t_orth = X w_orth
   p_orth = X^T t_orth / (t_orth^T t_orth)
   ```

3. Deflate X (remove orthogonal component):
   ```
   X = X - t_orth p_orth^T
   ```

4. Recompute predictive score and loading on deflated X:
   ```
   t = X w
   p = X^T t / (t^T t)
   ```

### Step 4: Fit PLS on filtered X

After removing all orthogonal components, run standard PLS on the filtered X and
original Y. For additional predictive components beyond the first, repeat standard
PLS deflation:

For each predictive component (a = 1, ..., A_pred):

1. Compute weight, score, loadings (NIPALS as in Step 1)
2. Deflate:
   ```
   X = X - t p^T
   Y = Y - t c^T
   ```

### Step 5: Regression coefficients

Collect all predictive weights W = [w_1, ..., w_A], loadings P = [p_1, ..., p_A],
and Y-loadings C = [c_1, ..., c_A]:

```
B = W (P^T W)^{-1} C^T
```

Prediction for new data X_new (after normalization, centering, and orthogonal filtering):

```
Y_hat = X_filtered B + y_mean
```

## Summary of stored model components

| Symbol    | Dimension         | Description                          |
|-----------|-------------------|--------------------------------------|
| α         | 1 x p             | Per-feature NB dispersions           |
| x_mean    | 1 x p             | Column means of VST-transformed X    |
| y_mean    | 1 x m             | Column means of Y                    |
| W         | p x A_pred        | Predictive weight vectors            |
| P         | p x A_pred        | Predictive X-loadings                |
| C         | m x A_pred        | Y-loadings                           |
| W_orth    | p x A_orth        | Orthogonal weight vectors            |
| P_orth    | p x A_orth        | Orthogonal X-loadings                |
| T_orth    | n x A_orth        | Orthogonal scores (for training data)|
| B         | p x m             | Final regression coefficients        |
