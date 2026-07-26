// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests: GPU output within NY verified bounds.
//!
//! Kernels covered: relu, clamped, bounds tightening.
//! Extended kernels (silu_mul, gelu, inv_norm, rope) in `contract_extended.rs` (#542).
//! Split from `end_to_end_pipeline.rs` to stay under 500 lines (#533).
//! Refactored to use shared harness (#700).

use super::*;

// ============================================================================
// Kernel definitions for cross-backend contract testing (#506)
// ============================================================================

#[nn_macros::kernel]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[nn_macros::kernel]
fn clamped(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

fn relu_kernel_def() -> nn_dsl::ir::KernelDef {
    let src = "fn relu(x: f32) -> f32 { x.max(0.0) }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse relu source");
    nn_dsl::Lowerer::lower_fn(&func).expect("lower relu to KernelDef")
}

fn clamped_kernel_def() -> nn_dsl::ir::KernelDef {
    let src = "fn clamped(x: f32) -> f32 { x.clamp(-1.0, 1.0) }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse clamped source");
    nn_dsl::Lowerer::lower_fn(&func).expect("lower clamped to KernelDef")
}

/// Cross-backend contract test for relu: GPU output within verified bounds.
///
/// relu(x) = max(x, 0) for x ∈ [-10, 10] → output ∈ [0, 10].
/// Part of #506.
#[test]
fn test_relu_gpu_output_within_verified_bounds() {
    contract_harness::assert_single_var_contract(
        &relu_kernel_def(),
        &RELU_DESCRIPTOR,
        (-10.0, 10.0),
        &[-10.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0],
        "relu",
    );
}

/// Cross-backend contract test for clamped: GPU output within verified bounds.
///
/// clamped(x) = clamp(x, -1, 1) for x ∈ [-100, 100] → output ∈ [-1, 1].
/// Part of #506.
#[test]
fn test_clamped_gpu_output_within_verified_bounds() {
    contract_harness::assert_single_var_contract(
        &clamped_kernel_def(),
        &CLAMPED_DESCRIPTOR,
        (-100.0, 100.0),
        &[
            -100.0, -10.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.0, 100.0,
        ],
        "clamped",
    );
}

// ============================================================================
// Critical-value and denormal contract tests (#728)
// ============================================================================

/// ReLU denormal-boundary test: near f32 denormal range.
///
/// ReLU uses max(x, 0) which is not FTZ-sensitive. Tests values near the
/// denormal boundary where max() must correctly distinguish positive
/// denormals from zero.
/// Part of #728.
#[test]
fn test_relu_denormal_boundary_gpu_within_verified_bounds() {
    contract_harness::assert_single_var_contract(
        &relu_kernel_def(),
        &RELU_DESCRIPTOR,
        (-10.0, 10.0),
        &[
            -1e-38, // negative denormal → relu output = 0
            -1e-30, 1e-38, // positive denormal → relu output = 1e-38
            1e-30, 1e-10, -1e-10,
        ],
        "relu_denormal",
    );
}

/// NY bounds tighten with narrower input ranges.
#[test]
fn test_verification_tightens_with_narrow_range() {
    let kernel_def = snake_kernel_def();
    let bindings = vec![
        nn_verify::ParamBinding::Variable,
        nn_verify::ParamBinding::Constant(1.0),
    ];

    let wide = nn_verify::VerifyRequest::new(&kernel_def)
        .bindings(&bindings)
        .variable_bounds(&[(-10.0, 10.0)])
        .verify_bounds()
        .expect("wide verification");
    let narrow = nn_verify::VerifyRequest::new(&kernel_def)
        .bindings(&bindings)
        .variable_bounds(&[(-1.0, 1.0)])
        .verify_bounds()
        .expect("narrow verification");

    assert!(wide.is_finite, "wide bounds must be finite");
    assert!(narrow.is_finite, "narrow bounds must be finite");
    assert!(
        narrow.output_width <= wide.output_width,
        "narrow range [{:.1}] should produce tighter bounds than wide [{:.1}]",
        narrow.output_width,
        wide.output_width,
    );
}
