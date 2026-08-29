// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Snake1d POC for issue #18.
//!
//! The proc-macro has since landed in `nn-macros`, re-exported as `#[nn::kernel]`
//! by the `nn` facade under the `dsl` feature. This example still calls the
//! library helpers directly because `nn-macros` depends on `nn-dsl`, so `nn-dsl`'s
//! own examples cannot use the macro. Its shape:
//! ```ignore
//! #[nn::kernel]
//! fn snake(x: f32, alpha: f32) -> f32 {
//!     x + (1.0 / alpha) * (alpha * x).sin().powi(2)
//! }
//! ```

use nn_dsl::{snake_ref_f32, snake_scalar_bounds};

fn main() {
    let x = vec![0.1f32, 0.2, 0.3, 0.4];
    let alpha = vec![1.0f32];
    let out = snake_ref_f32(&x, &alpha, 1, 4).expect("valid Snake1d layout");
    let bounds = snake_scalar_bounds(-10.0, 10.0, 0.01, 100.0).expect("finite bounds");

    println!("snake1d output: {out:?}");
    println!("conservative bounds: [{}, {}]", bounds.0, bounds.1);
}
