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
2. **Pseudolog transform**: `x <- log(x + 1)`
3. **Mean centering**: column means are subtracted from both X and Y

## Features

- Sparse matrix input for X (efficient for high-dimensional, sparse count data)
- Dense matrix Y (single or multiple response variables)
- Configurable number of orthogonal and predictive components

## Algorithm

See [EQUATIONS.md](EQUATIONS.md) for the full mathematical specification.

## Building

```
cargo build --release
```

## Usage

```rust
use sparse_olps::Opls;

let model = Opls::fit(&x_sparse, &y_dense, n_predictive, n_orthogonal);
let y_hat = model.predict(&x_new);
```
