// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Re-export of bounds dispatch from the always-available `crate::bounds` module.
//!
//! The analytical bounds code was extracted to `crate::bounds` (#859) so tests
//! run without the `ay-smt` feature flag. This module preserves the existing
//! import paths used by `prove.rs` and ay test files.

// Re-export all public items from the bounds module that ay code needs.
// Visibility must be `pub(in crate::ay)` so prove.rs can re-export within ay.
pub(in crate::ay) use crate::bounds::compute_output_bounds_heuristic;

// Re-export individual bounds functions used by ay test files.
// These are consumed by `#[cfg(test)]` modules in prove.rs, so they appear
// unused during non-test compilation — allow that.
#[allow(unused_imports)]
pub(in crate::ay) use crate::bounds::{
    adain_output_bounds, bounds_adain, bounds_adain_snake, bounds_exp, bounds_gelu,
    bounds_instance_norm, bounds_leaky_relu, bounds_norm_affine, bounds_relu,
    bounds_rms_norm_scalar, bounds_rope_cos, bounds_rope_sin, bounds_sigmoid, bounds_silu_mul,
    bounds_snake, bounds_softplus, bounds_tanh_act, exp_output_bounds, gelu_output_bounds,
    instance_norm_output_bounds, leaky_relu_output_bounds, norm_affine_output_bounds,
    relu_output_bounds, rms_norm_scalar_output_bounds, rope_output_bounds, sigmoid_output_bounds,
    silu_mul_output_bounds, softplus_output_bounds, tanh_output_bounds,
};
