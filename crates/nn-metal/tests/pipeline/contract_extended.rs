// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended cross-backend contract tests: silu_mul, gelu, inv_norm,
//! rope_cos, rope_sin, sigmoid — GPU output within NY verified bounds.
//!
//! Extracted from `contract.rs` to stay under the 500-line file limit (#542).
//! Refactored to use shared harness (#700).

use super::contract_harness;

// ============================================================================
// Extended kernel definitions for cross-backend contract testing (#506)
// ============================================================================

#[nn_macros::kernel]
fn silu_mul(x: f32, up: f32) -> f32 {
    (x / (1.0 + (-x).exp())) * up
}

#[nn_macros::kernel]
fn gelu(x: f32) -> f32 {
    let k: f32 = 0.797_884_6;
    let inner = k * (x + 0.044715 * x * x * x);
    let e2 = (2.0 * inner).exp();
    0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
}

#[nn_macros::kernel]
fn inv_norm(x: f32) -> f32 {
    (x * x + 1.0).sqrt().recip()
}

#[nn_macros::kernel(
    precision = "relaxed",
    bounds(x0 = "-1000..1000", x1 = "-1000..1000", freq = "-100..100")
)]
fn rope_cos_k(x0: f32, x1: f32, freq: f32) -> f32 {
    x0 * freq.cos() - x1 * freq.sin()
}

#[nn_macros::kernel(
    precision = "relaxed",
    bounds(x0 = "-1000..1000", x1 = "-1000..1000", freq = "-100..100")
)]
fn rope_sin_k(x0: f32, x1: f32, freq: f32) -> f32 {
    x0 * freq.sin() + x1 * freq.cos()
}

#[nn_macros::kernel]
fn sigmoid_k(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Contract tests
// ============================================================================

const TEST_X_11: [f32; 11] = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

/// silu_mul(x, up) = (x / (1 + exp(-x))) * up for x ∈ [-10, 10], up=2.0.
/// Part of #506.
#[test]
fn test_silu_mul_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_silu_mul_kernel().expect("build silu_mul KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &SILU_MUL_DESCRIPTOR,
        &[2.0], // up
        (-10.0, 10.0),
        &TEST_X_11,
        "silu_mul",
    );
}

/// gelu(x) for x ∈ [-10, 10]. Formula corrected in #671.
/// Part of #506.
#[test]
fn test_gelu_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_gelu_kernel().expect("build gelu KernelDef");

    // Numerical spot-check: gelu(1.0) ≈ 0.8412 (#671 AC4).
    let gelu_1 = nn_dsl::gelu_scalar(1.0).expect("gelu_scalar(1.0)");
    assert!(
        (gelu_1 - 0.8412).abs() < 0.001,
        "gelu(1.0) should be ~0.8412, got {gelu_1}",
    );

    contract_harness::assert_single_var_contract(
        &kernel_def,
        &GELU_DESCRIPTOR,
        (-10.0, 10.0),
        &TEST_X_11,
        "gelu",
    );
}

/// inv_norm(x) = 1 / sqrt(x^2 + 1) for x ∈ [-10, 10].
/// Part of #506.
#[test]
fn test_inv_norm_gpu_output_within_verified_bounds() {
    let src = "fn inv_norm(x: f32) -> f32 { (x * x + 1.0).sqrt().recip() }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse inv_norm source");
    let kernel_def = nn_dsl::Lowerer::lower_fn(&func).expect("lower inv_norm");
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &INV_NORM_DESCRIPTOR,
        (-10.0, 10.0),
        &TEST_X_11,
        "inv_norm",
    );
}

/// rope_cos(x0, x1, freq) = x0 * cos(freq) - x1 * sin(freq)
/// for x0 ∈ [-10, 10], x1=1.0, freq=0.5.
/// Part of #506.
#[test]
fn test_rope_cos_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_rope_cos_kernel().expect("build rope_cos KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &ROPE_COS_K_DESCRIPTOR,
        &[1.0, 0.5], // x1, freq
        (-10.0, 10.0),
        &TEST_X_11,
        "rope_cos",
    );
}

/// rope_sin(x0, x1, freq) = x0 * sin(freq) + x1 * cos(freq)
/// for x0 ∈ [-10, 10], x1=1.0, freq=0.5.
/// Part of #506.
#[test]
fn test_rope_sin_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_rope_sin_kernel().expect("build rope_sin KernelDef");
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &ROPE_SIN_K_DESCRIPTOR,
        &[1.0, 0.5], // x1, freq
        (-10.0, 10.0),
        &TEST_X_11,
        "rope_sin",
    );
}

/// sigmoid(x) = 1 / (1 + exp(-x)) for x ∈ [-10, 10].
/// Part of #678.
#[test]
fn test_sigmoid_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_sigmoid_kernel().expect("build sigmoid KernelDef");

    // Numerical spot-check: sigmoid(0.0) = 0.5 exactly.
    let sig_0 = nn_dsl::sigmoid_scalar(0.0).expect("sigmoid_scalar(0.0)");
    assert!(
        (sig_0 - 0.5).abs() < 1e-6,
        "sigmoid(0.0) should be 0.5, got {sig_0}",
    );

    contract_harness::assert_single_var_contract(
        &kernel_def,
        &SIGMOID_K_DESCRIPTOR,
        (-10.0, 10.0),
        &TEST_X_11,
        "sigmoid",
    );
}

// ============================================================================
// Critical-value, denormal, and large-magnitude contract tests (#728)
//
// The tests above use 9-11 evenly-spaced points in [-10, 10]. These tests
// add kernel-specific critical values (inflection points, minima), denormal
// boundary values (for kernels where FTZ does not cause divergence at the
// denormal boundary), and large magnitudes (for overflow-sensitive kernels).
// ============================================================================

/// GELU critical-value test: inflection points and global minimum.
///
/// GELU has a global minimum at x ≈ -0.752 and inflection points near
/// x ≈ -1.13 and x ≈ 0. Also tests the transition region at ±3.
/// Part of #728.
#[test]
fn test_gelu_critical_values_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_gelu_kernel().expect("build gelu KernelDef");
    let critical_x: &[f32] = &[
        -3.0,   // deep negative (output ≈ -0.004)
        -1.13,  // near second inflection point
        -0.752, // global minimum of GELU
        -0.1,   // near origin, steep region
        0.1,    // near origin, steep region
        3.0,    // transition region complete (output ≈ 2.996)
    ];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &GELU_DESCRIPTOR,
        (-10.0, 10.0),
        critical_x,
        "gelu_critical",
    );
}

/// Sigmoid critical-value test: steep gradient zone and near-saturation.
///
/// Sigmoid's gradient is steepest at x = 0 (σ'(0) = 0.25) and approaches
/// saturation exponentially. Tests ±3 (steep zone) and ±6 (near-saturation
/// where σ ≈ 0.9975 or σ ≈ 0.0025).
/// Part of #728.
#[test]
fn test_sigmoid_critical_values_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_sigmoid_kernel().expect("build sigmoid KernelDef");
    let critical_x: &[f32] = &[
        -6.0, // near-saturation: σ(-6) ≈ 0.0025
        -3.0, // steep zone: σ(-3) ≈ 0.047
        -0.5, // moderate gradient
        0.5,  // moderate gradient
        3.0,  // steep zone: σ(3) ≈ 0.953
        6.0,  // near-saturation: σ(6) ≈ 0.9975
    ];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &SIGMOID_K_DESCRIPTOR,
        (-10.0, 10.0),
        critical_x,
        "sigmoid_critical",
    );
}

/// SiLU_mul critical-value test: derivative crossings and minimum.
///
/// SiLU(x) = x * σ(x). Its derivative crosses 1.0 at x ≈ 1.278 and the
/// function has a minimum at x ≈ -1.278 (where SiLU ≈ -0.279).
/// Part of #728.
#[test]
fn test_silu_mul_critical_values_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_silu_mul_kernel().expect("build silu_mul KernelDef");
    let critical_x: &[f32] = &[
        -3.0,   // deep negative: silu(-3) ≈ -0.143
        -1.278, // near SiLU minimum (silu ≈ -0.279)
        -0.278, // near the second zero of silu'(x) - 1
        0.278, 1.278, // where silu'(x) ≈ 1.0
        3.0,   // transition complete: silu(3) ≈ 2.858
    ];
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &SILU_MUL_DESCRIPTOR,
        &[2.0], // up
        (-10.0, 10.0),
        critical_x,
        "silu_mul_critical",
    );
}

/// GELU denormal-boundary test: near f32 denormal range.
///
/// GELU contains a division (`2.0 / (e2 + 1.0)`) so `has_ftz_sensitive_op()`
/// returns true. However, for denormal x values the divisor is `e2 + 1 ≈ 2.0`
/// (well away from zero), so FTZ does not cause divergence here.
/// Tests that codegen handles near-zero inputs correctly.
/// Part of #728.
#[test]
fn test_gelu_denormal_boundary_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_gelu_kernel().expect("build gelu KernelDef");
    let denormal_x: &[f32] = &[
        1e-38, // near f32::MIN_POSITIVE (1.17e-38)
        -1e-38, 1e-30, // well above denormal, below typical test range
        -1e-30, 1e-10, // small but normal
        -1e-10,
    ];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &GELU_DESCRIPTOR,
        (-10.0, 10.0),
        denormal_x,
        "gelu_denormal",
    );
}

/// Sigmoid denormal-boundary test: near f32 denormal range.
///
/// Sigmoid uses `1 / (1 + exp(-x))` so `has_ftz_sensitive_op()` returns true.
/// However, the divisor `(1 + exp(-x))` is always ≥ 1.0, so FTZ does not
/// cause divergence at the denormal boundary.
/// For x ≈ 0, σ(x) ≈ 0.5 — verifies codegen correctness near zero.
/// Part of #728.
#[test]
fn test_sigmoid_denormal_boundary_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_sigmoid_kernel().expect("build sigmoid KernelDef");
    let denormal_x: &[f32] = &[1e-38, -1e-38, 1e-30, -1e-30, 1e-10, -1e-10];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &SIGMOID_K_DESCRIPTOR,
        (-10.0, 10.0),
        denormal_x,
        "sigmoid_denormal",
    );
}

/// GELU large-magnitude test: overflow/underflow boundary.
///
/// GELU uses exp(-0.5 * x^2) which underflows to 0 for |x| > ~15, making
/// GELU(x) ≈ x for large positive and ≈ 0 for large negative.
/// Also tests exp(2 * inner) overflow for large |x|.
/// Uses wider verification range (-100, 100).
/// Part of #728.
#[test]
fn test_gelu_large_magnitude_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_gelu_kernel().expect("build gelu KernelDef");
    let large_x: &[f32] = &[
        -50.0, // deep negative: gelu ≈ 0
        -15.0, // exp underflow boundary
        -10.0, 10.0, 15.0, // exp underflow boundary
        50.0, // large positive: gelu ≈ x
    ];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &GELU_DESCRIPTOR,
        (-100.0, 100.0),
        large_x,
        "gelu_large",
    );
}

/// SiLU_mul large-magnitude test: exp overflow boundary.
///
/// SiLU uses exp(-x) which overflows for x < ~-88.7. For large positive x,
/// σ(x) ≈ 1 so silu(x) ≈ x. Multiplied by up=2.0 at x=50 gives ≈100.
/// Uses wider verification range (-100, 100).
/// Part of #728.
#[test]
fn test_silu_mul_large_magnitude_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_silu_mul_kernel().expect("build silu_mul KernelDef");
    let large_x: &[f32] = &[
        -50.0, // deep negative: silu ≈ 0
        -20.0, -10.0, 10.0, 20.0, 50.0, // large positive: silu ≈ x, silu_mul ≈ 2x = 100
    ];
    contract_harness::assert_scalar_contract(
        &kernel_def,
        &SILU_MUL_DESCRIPTOR,
        &[2.0], // up
        (-100.0, 100.0),
        large_x,
        "silu_mul_large",
    );
}
