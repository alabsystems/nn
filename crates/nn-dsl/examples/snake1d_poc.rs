// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Snake1d POC for issue #18.
//!
//! Intended final shape once the proc-macro crate lands:
//! ```ignore
//! // NOTE: ignore — #[nn::kernel] proc-macro not yet implemented
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
