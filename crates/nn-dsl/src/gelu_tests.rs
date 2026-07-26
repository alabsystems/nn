// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GELU kernel.

use super::*;
use crate::kernel_error::KernelError;

// --- KernelDef builder ---

#[test]
fn test_gelu_kernel_builds_and_validates() {
    let kernel = build_gelu_kernel().expect("build must succeed");
    kernel.validate().expect("IR must validate");
    assert_eq!(kernel.name, "gelu");
    assert_eq!(kernel.params.len(), 1);
    assert_eq!(kernel.params[0].name, "x");
}

#[test]
fn test_gelu_kernel_msl_codegen() {
    let kernel = build_gelu_kernel().expect("build must succeed");
    let msl = crate::emit_msl(&kernel).expect("MSL codegen");
    assert!(msl.contains("gelu"), "MSL should contain kernel name");
    assert!(msl.contains("exp("), "MSL should use exp for tanh");
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should have kernel attribute"
    );
}

#[test]
fn test_gelu_kernel_kani_harness_codegen() {
    let kernel = build_gelu_kernel().expect("build must succeed");
    let harness = crate::emit_kani_harness(&kernel).expect("kani codegen");
    assert!(harness.contains("#[kani::proof]"));
    assert!(harness.contains("kani::any"));
}

// --- Scalar reference ---

#[test]
fn test_gelu_scalar_at_zero() {
    // gelu(0) = 0.5 * 0 * (1 + tanh(0)) = 0
    let result = gelu_scalar(0.0).expect("must succeed");
    assert!(result.abs() < 1e-7, "gelu(0) = 0, got {result}");
}

#[test]
fn test_gelu_scalar_positive() {
    // gelu(1) ≈ 0.8412
    let result = gelu_scalar(1.0).expect("must succeed");
    assert!(
        (result - 0.8412).abs() < 1e-3,
        "gelu(1) ≈ 0.8412, got {result}"
    );
}

#[test]
fn test_gelu_scalar_negative() {
    // gelu(-1) ≈ -0.1588
    let result = gelu_scalar(-1.0).expect("must succeed");
    assert!(
        (result - (-0.1588)).abs() < 1e-3,
        "gelu(-1) ≈ -0.1588, got {result}"
    );
}

#[test]
fn test_gelu_scalar_large_positive() {
    // For large x, gelu(x) ≈ x (tanh saturates to 1)
    let result = gelu_scalar(50.0).expect("must succeed");
    assert!((result - 50.0).abs() < 1e-4, "gelu(50) ≈ 50, got {result}");
}

#[test]
fn test_gelu_scalar_large_negative() {
    // For large negative x, gelu(x) ≈ 0 (tanh saturates to -1)
    let result = gelu_scalar(-50.0).expect("must succeed");
    assert!(result.abs() < 1e-4, "gelu(-50) ≈ 0, got {result}");
}

#[test]
fn test_gelu_scalar_at_minimum() {
    // GELU has a global minimum at x ≈ -0.752 where gelu ≈ -0.170
    let result = gelu_scalar(GELU_ARGMIN).expect("must succeed");
    assert!(
        result < -0.16 && result > -0.18,
        "gelu(GELU_ARGMIN) ≈ -0.170, got {result}"
    );
}

// --- Scalar error paths ---

#[test]
fn test_gelu_scalar_nan_returns_err() {
    let err = gelu_scalar(f32::NAN).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
}

#[test]
fn test_gelu_scalar_inf_returns_err() {
    let err = gelu_scalar(f32::INFINITY).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
}

#[test]
fn test_gelu_scalar_neg_inf_returns_err() {
    let err = gelu_scalar(f32::NEG_INFINITY).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
}

// --- Bounds ---

#[test]
fn test_gelu_bounds_positive_range() {
    let (lo, hi) = gelu_scalar_bounds(1.0, 3.0).expect("finite inputs");
    // gelu(1) ≈ 0.841, gelu(3) ≈ 2.996 — monotonically increasing in this range
    assert!(lo <= gelu_scalar(1.0).unwrap() + 1e-5);
    assert!(hi >= gelu_scalar(3.0).unwrap() - 1e-5);
}

#[test]
fn test_gelu_bounds_contain_sample() {
    let (lo, hi) = gelu_scalar_bounds(-3.0, 3.0).expect("finite inputs");
    for &x in &[-3.0, -2.0, -1.0, -0.752, 0.0, 1.0, 2.0, 3.0] {
        let val = gelu_scalar(x).unwrap();
        assert!(
            val >= lo - 1e-5 && val <= hi + 1e-5,
            "gelu({x}) = {val} outside bounds [{lo}, {hi}]"
        );
    }
}

#[test]
fn test_gelu_bounds_spanning_minimum() {
    // Range spans the GELU minimum — bounds must include the minimum value
    let (lo, _hi) = gelu_scalar_bounds(-2.0, 0.0).expect("finite inputs");
    let min_val = gelu_scalar(GELU_ARGMIN).unwrap();
    assert!(
        lo <= min_val + 1e-5,
        "lower bound {lo} must be <= gelu(GELU_ARGMIN)={min_val}"
    );
}

#[test]
fn test_gelu_bounds_not_spanning_minimum() {
    // Range entirely above the minimum — should be tighter
    let (lo, hi) = gelu_scalar_bounds(0.0, 2.0).expect("finite inputs");
    let g0 = gelu_scalar(0.0).unwrap();
    let g2 = gelu_scalar(2.0).unwrap();
    assert!(
        (lo - g0).abs() < 1e-5,
        "lower bound should be gelu(0)={g0}, got {lo}"
    );
    assert!(
        (hi - g2).abs() < 1e-5,
        "upper bound should be gelu(2)={g2}, got {hi}"
    );
}

// --- Bounds error paths ---

#[test]
fn test_gelu_bounds_nan_returns_err() {
    let err = gelu_scalar_bounds(f32::NAN, 1.0).expect_err("NaN should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

#[test]
fn test_gelu_bounds_inf_returns_err() {
    let err = gelu_scalar_bounds(0.0, f32::INFINITY).expect_err("Inf should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with Inf, got {err:?}"
    );
}

#[test]
fn test_gelu_bounds_rejects_inverted() {
    let err = gelu_scalar_bounds(5.0, -5.0).unwrap_err();
    assert!(
        matches!(err, KernelError::InvertedBounds { lower, upper } if lower == 5.0 && upper == -5.0),
        "inverted bounds should be rejected, got: {err}"
    );
}

// --- Bounds soundness grid ---

#[test]
fn test_gelu_bounds_soundness_grid() {
    let x_vals: &[f32] = &[-10.0, -5.0, -2.0, -0.752, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0];
    for &x_lo in x_vals {
        for &x_hi in x_vals {
            if x_lo > x_hi {
                continue;
            }
            let (lo, hi) = gelu_scalar_bounds(x_lo, x_hi).expect("finite inputs");
            for xi in 0..=10 {
                let x = x_lo + (x_hi - x_lo) * (xi as f32) / 10.0;
                let val = gelu_scalar(x).unwrap();
                assert!(
                    val >= lo - 1e-4 && val <= hi + 1e-4,
                    "gelu({x}) = {val} outside [{lo}, {hi}] for box [{x_lo},{x_hi}]"
                );
            }
        }
    }
}

// --- 1d array ---

#[test]
fn test_gelu_ref_basic() {
    let x = [0.0, 1.0, -1.0, 2.0];
    let result = gelu_ref(&x).expect("must succeed");
    assert_eq!(result.len(), 4);
    // Verify all 4 outputs against known GELU(tanh approx) values (#650 AC2).
    assert!(result[0].abs() < 1e-6, "gelu(0) ≈ 0, got {}", result[0]);
    assert!(
        (result[1] - 0.8412).abs() < 1e-3,
        "gelu(1) ≈ 0.8412, got {}",
        result[1]
    );
    assert!(
        (result[2] - (-0.1588)).abs() < 1e-3,
        "gelu(-1) ≈ -0.1588, got {}",
        result[2]
    );
    assert!(
        (result[3] - 1.9545).abs() < 1e-3,
        "gelu(2) ≈ 1.9545, got {}",
        result[3]
    );
}

#[test]
fn test_gelu_ref_empty_returns_err() {
    let result = gelu_ref(&[]);
    assert!(result.is_err());
}

// --- MSL diff test codegen ---

#[test]
fn test_gelu_differential_test_codegen() {
    use crate::precision::PrecisionTier;
    let kernel = build_gelu_kernel().expect("build");
    let test_code = crate::codegen_difftest::emit_differential_test(&kernel, PrecisionTier::Normal)
        .expect("difftest codegen");
    assert!(test_code.contains("GELU_DESCRIPTOR"));
    assert!(test_code.contains("#[test]"));
}
