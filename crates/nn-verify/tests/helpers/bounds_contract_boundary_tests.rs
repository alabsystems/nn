// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounds contract boundary tests: non-finite rejection at the verify
//! boundary.
//!
//! Split from `bounds_contract_tests.rs` to keep each file under 500 lines.
//! IBP arithmetic overflow/shape-mismatch tests removed in #2005 —
//! arithmetic is provided by `ny_tensor::BoundedTensor`.
//!
//! Part of #1942.

// ===========================================================================
// D4: Boundary tests — non-finite rejection at the verify boundary
// ===========================================================================

#[test]
fn boundary_scalar_input_bounds_rejects_non_finite() {
    use nn_verify::scalar_input_bounds;

    // NaN bounds
    assert!(
        scalar_input_bounds(f32::NAN, 1.0).is_err(),
        "scalar_input_bounds should reject NaN lower"
    );
    assert!(
        scalar_input_bounds(0.0, f32::NAN).is_err(),
        "scalar_input_bounds should reject NaN upper"
    );

    // Infinite bounds
    assert!(
        scalar_input_bounds(f32::NEG_INFINITY, 1.0).is_err(),
        "scalar_input_bounds should reject -Inf lower"
    );
    assert!(
        scalar_input_bounds(0.0, f32::INFINITY).is_err(),
        "scalar_input_bounds should reject +Inf upper"
    );

    // Inverted bounds
    assert!(
        scalar_input_bounds(5.0, 1.0).is_err(),
        "scalar_input_bounds should reject inverted bounds"
    );

    // Valid bounds succeed
    assert!(
        scalar_input_bounds(-10.0, 10.0).is_ok(),
        "scalar_input_bounds should accept valid finite ordered bounds"
    );
}

#[test]
fn boundary_multi_scalar_input_bounds_rejects_non_finite() {
    use nn_verify::multi_scalar_input_bounds;

    // One non-finite bound among several variables
    assert!(
        multi_scalar_input_bounds(&[(-1.0, 1.0), (f32::NAN, 1.0)]).is_err(),
        "multi_scalar_input_bounds should reject NaN in any variable"
    );
    assert!(
        multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, f32::INFINITY)]).is_err(),
        "multi_scalar_input_bounds should reject Inf in any variable"
    );

    // All valid
    assert!(
        multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, 10.0)]).is_ok(),
        "multi_scalar_input_bounds should accept all-finite ordered bounds"
    );
}

#[test]
fn boundary_constant_binding_rejects_non_finite() {
    use nn_verify::kernel_to_graph;

    // Parse a simple kernel.
    let func: syn::ItemFn =
        syn::parse_str("fn add_const(x: f32, c: f32) -> f32 { x + c }").expect("valid Rust");
    let kernel = nn_dsl::lower::Lowerer::lower_fn(&func).expect("valid kernel");

    // Non-finite constant should be rejected.
    let result = kernel_to_graph(&kernel, &[f32::NAN]);
    assert!(
        result.is_err(),
        "kernel_to_graph should reject NaN constant"
    );

    let result = kernel_to_graph(&kernel, &[f32::INFINITY]);
    assert!(
        result.is_err(),
        "kernel_to_graph should reject +Inf constant"
    );

    let result = kernel_to_graph(&kernel, &[f32::NEG_INFINITY]);
    assert!(
        result.is_err(),
        "kernel_to_graph should reject -Inf constant"
    );

    // Finite constant should succeed.
    let result = kernel_to_graph(&kernel, &[1.0]);
    assert!(
        result.is_ok(),
        "kernel_to_graph should accept finite constant"
    );
}
