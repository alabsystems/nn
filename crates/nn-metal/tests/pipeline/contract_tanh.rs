// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tanh standalone GPU contract test: Metal output within NY verified bounds.
//!
//! Tanh is used in LSTM gate decomposition (#761):
//! - g-candidate: `g = tanh(W_ig @ x + b_ig + W_hg @ h + b_hg)`
//! - cell output: `h_new = o * tanh(c_new)`
//!
//! Part of #776.

use super::contract_harness;

// ============================================================================
// Kernel definition for descriptor generation
// ============================================================================

#[nn_macros::kernel]
fn tanh_act(x: f32) -> f32 {
    x.tanh()
}

// ============================================================================
// Contract tests
// ============================================================================

const TEST_X_11: [f32; 11] = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

/// tanh(x) for x in [-10, 10].
/// Tanh is monotonically increasing with output in (-1, 1).
/// Part of #776 AC1+AC2.
#[test]
fn test_tanh_gpu_output_within_verified_bounds() {
    let kernel_def = nn_dsl::build_tanh_kernel().expect("build tanh KernelDef");

    // Numerical spot-check: tanh(0.0) = 0.0 exactly.
    let tanh_0 = nn_dsl::tanh_scalar(0.0).expect("tanh_scalar(0.0)");
    assert!(tanh_0.abs() < 1e-6, "tanh(0.0) should be 0.0, got {tanh_0}",);

    contract_harness::assert_single_var_contract(
        &kernel_def,
        &TANH_ACT_DESCRIPTOR,
        (-10.0, 10.0),
        &TEST_X_11,
        "tanh",
    );
}

/// Tanh near-zero edge case: steep gradient zone.
///
/// tanh'(0) = 1.0 (steepest gradient). Near zero, tanh(x) ~ x.
/// Tests the linear approximation region and transition to saturation.
/// Part of #776 AC3.
#[test]
fn test_tanh_near_zero_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_tanh_kernel().expect("build tanh KernelDef");
    let near_zero_x: &[f32] = &[-0.1, -0.01, -0.001, -1e-6, 0.0, 1e-6, 0.001, 0.01, 0.1];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &TANH_ACT_DESCRIPTOR,
        (-10.0, 10.0),
        near_zero_x,
        "tanh_near_zero",
    );
}

/// Tanh large magnitude edge case: saturation region.
///
/// For |x| > ~4, tanh(x) approaches +/-1 exponentially.
/// tanh(10) ~ 1.0 - 4.1e-9, tanh(50) = 1.0 in f32.
/// Tests that Metal codegen handles the saturation correctly.
/// Part of #776 AC3.
#[test]
fn test_tanh_large_magnitude_gpu_within_verified_bounds() {
    let kernel_def = nn_dsl::build_tanh_kernel().expect("build tanh KernelDef");
    let large_x: &[f32] = &[
        -50.0, // deep negative: tanh ~ -1.0
        -10.0, // near saturation: tanh ~ -1.0 + 4.1e-9
        -4.0,  // transition: tanh(-4) ~ -0.9993
        4.0,   // transition: tanh(4) ~ 0.9993
        10.0,  // near saturation: tanh ~ 1.0 - 4.1e-9
        50.0,  // deep positive: tanh = 1.0 in f32
    ];
    contract_harness::assert_single_var_contract(
        &kernel_def,
        &TANH_ACT_DESCRIPTOR,
        (-100.0, 100.0),
        large_x,
        "tanh_large",
    );
}
