// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `validate_fusion_params` error branches (#275).
//! Extracted from fusion_equivalence.rs (#356).
//! Updated to use `FusionSpec::new()` after `#[non_exhaustive]` (#457).

use nn_dsl::test_kernels::{identity_kernel, parse_kernel};
use nn_verify::FusionSpec;

/// Helper: `fn add(x: f32, y: f32) -> f32 { x + y }` — 2-param kernel.
fn add_kernel() -> nn_dsl::ir::KernelDef {
    parse_kernel("fn add(x: f32, y: f32) -> f32 { x + y }")
}

#[test]
fn test_validate_first_param_indices_length_mismatch() {
    // first has 2 params (add), but first_param_indices has length 1.
    let fused = add_kernel();
    let first = add_kernel();
    let second = identity_kernel();

    let err = FusionSpec::new(
        &fused,
        &first,
        &second,
        2,
        &[0], // length 1, but first has 2 params
        &[0],
        0,
    )
    .expect_err("should reject mismatched first indices");
    let msg = err.to_string();
    assert!(
        msg.contains("first_param_indices length"),
        "error should mention first_param_indices, got: {msg}"
    );
}

#[test]
fn test_validate_second_param_indices_length_mismatch() {
    // second has 1 param (identity), but second_param_indices has length 2.
    let fused = add_kernel();
    let first = identity_kernel();
    let second = identity_kernel();

    let err = FusionSpec::new(
        &fused,
        &first,
        &second,
        2,
        &[0],
        &[0, 1], // length 2, but second has 1 param
        0,
    )
    .expect_err("should reject mismatched second indices");
    let msg = err.to_string();
    assert!(
        msg.contains("second_param_indices length"),
        "error should mention second_param_indices, got: {msg}"
    );
}

#[test]
fn test_validate_second_input_from_first_out_of_range() {
    // second has 1 param (identity), but second_input_from_first = 5 (out of range).
    let fused = add_kernel();
    let first = identity_kernel();
    let second = identity_kernel();

    let err = FusionSpec::new(
        &fused,
        &first,
        &second,
        2,
        &[0],
        &[0],
        5, // >= second.params.len() (1)
    )
    .expect_err("should reject out-of-range second_input_from_first");
    let msg = err.to_string();
    assert!(
        msg.contains("second_input_from_first"),
        "error should mention second_input_from_first, got: {msg}"
    );
}

#[test]
fn test_validate_fused_param_count_vs_shared_inputs() {
    // fused has 2 params (add), but num_shared_inputs = 3.
    let fused = add_kernel();
    let first = identity_kernel();
    let second = identity_kernel();

    let err = FusionSpec::new(
        &fused,
        &first,
        &second,
        3, // != fused.params.len() (2)
        &[0],
        &[0],
        0,
    )
    .expect_err("should reject fused/shared mismatch");
    let msg = err.to_string();
    assert!(
        msg.contains("fused param count") && msg.contains("num_shared_inputs"),
        "error should mention fused param count and num_shared_inputs, got: {msg}"
    );
}

#[test]
fn test_validate_first_param_index_out_of_range() {
    // first_param_indices[0] = 99, but num_shared_inputs = 2.
    let fused = add_kernel();
    let first = identity_kernel();
    let second = identity_kernel();

    let err = FusionSpec::new(
        &fused,
        &first,
        &second,
        2,
        &[99], // 99 >= num_shared_inputs (2)
        &[0],
        0,
    )
    .expect_err("should reject out-of-range first param index");
    let msg = err.to_string();
    assert!(
        msg.contains("first_param_indices[0]") && msg.contains("99"),
        "error should mention first_param_indices[0] = 99, got: {msg}"
    );
}

#[test]
fn test_validate_second_param_index_out_of_range() {
    // Build a fusion where second has 2 params: one from first's output,
    // one from shared inputs. Make the non-first-output index out of range.
    let fused = add_kernel();
    let first = identity_kernel();
    let second = add_kernel(); // 2 params: param 0 = from first, param 1 = shared

    let err = FusionSpec::new(
        &fused,
        &first,
        &second,
        2,
        &[0],
        &[0, 99], // index 0 is from first (ignored), index 1 = 99 (out of range)
        0,
    )
    .expect_err("should reject out-of-range second param index");
    let msg = err.to_string();
    assert!(
        msg.contains("second_param_indices[1]") && msg.contains("99"),
        "error should mention second_param_indices[1] = 99, got: {msg}"
    );
}
