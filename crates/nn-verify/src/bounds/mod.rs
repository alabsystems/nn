// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical output bounds for kernel verification.
//!
//! Extracted from the `ay` module (#859) so that analytical bounds functions
//! and their tests are always available — even without the `ay-smt` feature
//! flag. These functions are pure Rust math (no ay-bindings dependency).
//!
//! The `ay` module imports from here when it needs bounds for SMT queries.

// Submodules are consumed by ay (feature-gated) and by #[cfg(test)] modules.
// Without ay-smt, the functions appear as dead code in non-test builds.
#[allow(dead_code)]
mod activation;
#[allow(dead_code)]
mod binary;
#[allow(dead_code)]
mod conv1d_k1;
#[allow(dead_code)]
mod dispatch;
#[allow(dead_code)]
mod norm;
#[allow(dead_code)]
mod rope;
mod snake; // snake has inline tests, no dead_code suppression needed
           // harmonic_source is pub (not pub(crate)) because integration tests
           // and consumers use HarmonicSourceBounds for analytical bypass validation.
#[allow(dead_code)]
pub(crate) mod harmonic_source;

// Re-exports used by the ay module and tests.
// These items are consumed by ay (feature-gated) and by #[cfg(test)] modules,
// so they appear unused during non-test, non-ay compilation.
#[allow(unused_imports)]
pub(crate) use activation::{
    exp_output_bounds, gelu_output_bounds, leaky_relu_output_bounds, relu_output_bounds,
    sigmoid_output_bounds, silu_mul_output_bounds, softplus_output_bounds, tanh_output_bounds,
};
#[allow(unused_imports)]
pub(crate) use binary::binary_add_output_bounds;
#[allow(unused_imports)]
pub(crate) use conv1d_k1::conv1d_k1_scalar_output_bounds;
#[allow(unused_imports)]
pub(crate) use dispatch::compute_output_bounds_heuristic;
#[allow(unused_imports)]
pub(crate) use harmonic_source::HarmonicSourceBounds;
#[allow(unused_imports)]
pub(crate) use norm::{
    adain_output_bounds, instance_norm_output_bounds, norm_affine_output_bounds,
    rms_norm_scalar_output_bounds,
};
#[allow(unused_imports)]
pub(crate) use rope::rope_output_bounds;
#[allow(unused_imports)]
pub(crate) use snake::snake_output_bounds;

// Re-export dispatch-level bounds functions used by tests.
#[allow(unused_imports)]
pub(crate) use dispatch::{
    bounds_ada_layer_norm, bounds_adain, bounds_adain_leaky_relu, bounds_adain_snake,
    bounds_binary_add, bounds_binary_mul, bounds_conv1d_k1_scalar, bounds_exp, bounds_gelu,
    bounds_instance_norm, bounds_leaky_relu, bounds_norm_affine, bounds_relu,
    bounds_rms_norm_scalar, bounds_rope_cos, bounds_rope_sin, bounds_sigmoid, bounds_silu_mul,
    bounds_snake, bounds_softplus, bounds_tanh_act,
};

/// Shared test helper: assert two f64 values are within `tol` of each other.
/// Delegates to [`nn_core::test_utils::assert_close_scalar_f64`].
#[cfg(test)]
pub(crate) fn assert_close_f64(actual: f64, expected: f64, tol: f64, msg: &str) {
    nn_core::test_utils::assert_close_scalar_f64(actual, expected, tol, msg);
}
