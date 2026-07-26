// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for tensor_dispatch internals.

use super::helpers::{checked_product_of_shape, next_power_of_2};
use super::*;
use nn_dsl::{build_dispatch_plan, emit_tensor_msl_with_contract};

#[test]
fn test_checked_product_of_shape_normal() {
    assert_eq!(checked_product_of_shape(&[2, 3, 4]).unwrap(), 24);
    assert_eq!(checked_product_of_shape(&[1]).unwrap(), 1);
    assert_eq!(checked_product_of_shape(&[]).unwrap(), 1);
}

#[test]
fn test_checked_product_of_shape_overflow() {
    let huge = &[usize::MAX, 2];
    let err = checked_product_of_shape(huge).unwrap_err();
    assert!(
        matches!(err, TensorDispatchError::ShapeOverflow { .. }),
        "expected ShapeOverflow, got {err:?}"
    );
}

#[test]
fn test_checked_product_of_shape_zero_dim() {
    assert_eq!(checked_product_of_shape(&[4, 0, 8]).unwrap(), 0);
}

// --- next_power_of_2 tests ---

#[test]
fn test_next_power_of_2_zero_and_one() {
    assert_eq!(next_power_of_2(0), 1);
    assert_eq!(next_power_of_2(1), 1);
}

#[test]
fn test_next_power_of_2_exact_powers() {
    assert_eq!(next_power_of_2(2), 2);
    assert_eq!(next_power_of_2(4), 4);
    assert_eq!(next_power_of_2(256), 256);
    assert_eq!(next_power_of_2(1024), 1024);
    assert_eq!(next_power_of_2(1 << 31), 1 << 31);
}

#[test]
fn test_next_power_of_2_non_powers() {
    assert_eq!(next_power_of_2(3), 4);
    assert_eq!(next_power_of_2(5), 8);
    assert_eq!(next_power_of_2(255), 256);
    assert_eq!(next_power_of_2(257), 512);
}

/// Regression test: values > 2^31 caused `1u32 << 32` overflow panic
/// before the fix in P1-56.
#[test]
fn test_next_power_of_2_overflow_no_panic() {
    // Values > 2^31 have no exact power-of-2 fit in u32; returns 2^31.
    let result = next_power_of_2(u32::MAX);
    assert_eq!(result, 1 << 31);

    let result = next_power_of_2((1 << 31) + 1);
    assert_eq!(result, 1 << 31);
}

#[test]
fn test_next_power_of_2_boundary() {
    // 2^31 - 1 = 2147483647, next power of 2 is 2^31
    assert_eq!(next_power_of_2((1 << 31) - 1), 1 << 31);
    // 2^31 exactly
    assert_eq!(next_power_of_2(1 << 31), 1 << 31);
}

/// K6 RoPE: tensor MSL emits distinct kernel entry points for rope_cos (node 6)
/// and rope_sin (node 7) even though both elementwise ops share the same inputs
/// `[3, 4, 5]`. Regression test for #321 AC2/AC3: matching by output node ID
/// (not inputs) is required to avoid duplicating rope_cos MSL for both steps.
#[test]
fn test_k6_rope_tensor_msl_emits_distinct_cos_sin() {
    use nn_dsl::{build_rope_rotate_kernel, ir::ScalarType};

    let kernel = build_rope_rotate_kernel(2, 4, 8).expect("build K6 RoPE");
    let dtype = ScalarType::F32;
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, dtype);
    let (_plan, _effective_output) = build_dispatch_plan(&kernel, dtype).expect("dispatch plan");
    let combined = emit_tensor_msl_with_contract(&kernel, dtype, contract).expect("tensor MSL");

    // Both rope_cos and rope_sin kernel entry points must appear.
    let has_cos_kernel = combined.contains("rope_rotate_rope_cos_n6_kernel");
    let has_sin_kernel = combined.contains("rope_rotate_rope_sin_n7_kernel");
    assert!(
        has_cos_kernel,
        "combined MSL must contain rope_cos kernel entry point\nMSL:\n{combined}"
    );
    assert!(
        has_sin_kernel,
        "combined MSL must contain rope_sin kernel entry point\nMSL:\n{combined}"
    );

    // The scalar helper functions must differ: rope_cos computes
    // `x0*cos(f) - x1*sin(f)` while rope_sin computes `x0*sin(f) + x1*cos(f)`.
    // If the bug existed (matching by inputs), both would be rope_cos.
    // Verify the helper functions contain distinctive arithmetic operators.
    // MSL structure: helper `_nn_rope_rotate_rope_cos_n6(...)` appears before
    // the `[[kernel]] rope_rotate_rope_cos_n6_kernel(...)` entry point, so we
    // search for the helper function bodies directly.
    let cos_helper = "_nn_rope_rotate_rope_cos_n6";
    let sin_helper = "_nn_rope_rotate_rope_sin_n7";
    let cos_pos = combined.find(cos_helper).expect("cos helper function");
    let sin_pos = combined.find(sin_helper).expect("sin helper function");

    // Extract the helper function body (from its definition to the next
    // top-level declaration, approximated by the next `[[kernel]]` or
    // `#include` or helper function).
    let cos_end = combined[cos_pos + cos_helper.len()..]
        .find("[[kernel]]")
        .map_or(combined.len(), |p| p + cos_pos + cos_helper.len());
    let cos_body = &combined[cos_pos..cos_end];

    let sin_end = combined[sin_pos + sin_helper.len()..]
        .find("[[kernel]]")
        .map_or(combined.len(), |p| p + sin_pos + sin_helper.len());
    let sin_body = &combined[sin_pos..sin_end];

    // rope_cos helper must contain subtraction (x0*cos - x1*sin).
    assert!(
        cos_body.contains(" - "),
        "rope_cos helper must contain subtraction operator\nBody:\n{cos_body}"
    );
    // rope_sin helper must contain addition (x0*sin + x1*cos).
    assert!(
        sin_body.contains(" + "),
        "rope_sin helper must contain addition operator\nBody:\n{sin_body}"
    );
}
