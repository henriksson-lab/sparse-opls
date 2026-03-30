# sparse_olps

OPLS (Orthogonal Projections to Latent Structures) solver in Rust, designed for
sparse X matrices and dense Y matrices.

## Overview

This library implements the OPLS algorithm (Trygg & Wold, 2002) for multivariate
regression. It separates variation in X into:

- **Predictive components**: correlated with Y
- **Orthogonal components**: structured variation in X uncorrelated with Y

This separation improves model interpretability without changing predictive performance
compared to standard PLS.

## Preprocessing pipeline

1. **Row normalization**: each row of X is divided by its total count (compositional normalization)
2. **DESeq2-style VST**: variance-stabilizing transform for negative-binomial count data — `f(q) = (2/√α) · asinh(√(α·q))` with per-feature dispersion α estimated from a mean-dispersion trend
3. **Mean centering**: column means are subtracted from both X and Y

## Features

- Sparse matrix input for X (efficient for high-dimensional, sparse count data)
- Dense matrix Y (single or multiple response variables)
- Configurable number of orthogonal and predictive components
- Optional GPU acceleration via CUDA (cuSPARSE + cuBLAS)

## Algorithm

See [EQUATIONS.md](EQUATIONS.md) for the full mathematical specification.

## Building

```
cargo build --release
```

With GPU support (requires CUDA toolkit):

```
cargo build --release --features cuda,cuda-version-from-build-system
```

## Usage

```rust
use sparse_olps::OplsModel;

let model = OplsModel::fit(&x_sparse, &y_dense, n_predictive, n_orthogonal);
let y_hat = model.predict(&x_new);
```

## Benchmarks

All benchmarks: 2 predictive + 2 orthogonal components, best of 3 runs.

### Single-Y (m=1)

| Config | nnz | CPU fit | GPU fit | Speedup |
|---|---|---|---|---|
| 10k x 20k, 1% | 2M | 0.81s | 0.85s | 0.94x |
| 10k x 20k, 5% | 10M | 2.62s | 2.09s | **1.25x** |
| 50k x 20k, 1% | 10M | 2.72s | 2.57s | **1.06x** |

### Multi-Y (m=5, exercises NIPALS inner loop)

| Config | nnz | CPU fit | GPU fit | Speedup |
|---|---|---|---|---|
| 10k x 20k, 1% | 2M | 18.0s | 1.66s | **10.8x** |
| 10k x 20k, 5% | 10M | 79.2s | 4.00s | **19.8x** |
| 50k x 20k, 1% | 10M | 72.8s | 5.33s | **13.7x** |

GPU acceleration is most effective with multiple response variables (m > 1), where the
NIPALS iterative loop runs entirely on GPU with device-pointer mode cuBLAS, avoiding
per-iteration host synchronization.
