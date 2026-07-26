// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for prove_bounds.rs: input finiteness guards (#394) and
//! numerical correctness (#425). Updated for #448/#459 variable-first
//! convention where param 0 (x) is always the symbolic variable.
//!
//! Split into three kernel-family modules (#453):
//! - silu_rope: silu_mul + rope bounds tests
//! - rms_norm: rms_norm_scalar + norm_affine bounds tests
//! - norm: instance_norm + adain bounds tests

use super::prove_dispatch::{
    adain_output_bounds, bounds_adain_snake, gelu_output_bounds, instance_norm_output_bounds,
    norm_affine_output_bounds, relu_output_bounds, rms_norm_scalar_output_bounds,
    rope_output_bounds, sigmoid_output_bounds, silu_mul_output_bounds, tanh_output_bounds,
};

/// Assert that an expression returns `Err` whose message contains the expected substring.
macro_rules! assert_bounds_error {
    ($expr:expr, $expected:expr) => {{
        let err = $expr.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains($expected),
            "expected error containing {:?}, got: {msg}",
            $expected
        );
    }};
    ($expr:expr, $expected:expr, $index:expr) => {{
        let err = $expr.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains($expected),
            "expected error containing {:?}, got: {msg}",
            $expected
        );
        let idx_str = format!("index {}", $index);
        assert!(
            msg.contains(&idx_str),
            "expected error citing {idx_str}, got: {msg}"
        );
    }};
}

#[path = "prove_bounds_tests_silu_rope.rs"]
mod silu_rope;

#[path = "prove_bounds_tests_rms_norm.rs"]
mod rms_norm;

#[path = "prove_bounds_tests_norm.rs"]
mod norm;

#[path = "prove_bounds_tests_gelu.rs"]
mod gelu;

#[path = "prove_bounds_tests_sigmoid.rs"]
mod sigmoid;

#[path = "prove_bounds_tests_adain_snake.rs"]
mod adain_snake;

#[path = "prove_relu_bounds_tests.rs"]
mod relu;

#[path = "prove_tanh_bounds_tests.rs"]
mod tanh;
