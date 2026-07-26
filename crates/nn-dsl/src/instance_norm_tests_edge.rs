// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edge-case coverage for InstanceNorm K2: numerical stability,
//! error handling, precision boundary, and non-finite input tests.

use super::*;

// --- Edge-case coverage for numerical stability (proof_coverage gap) ---

/// Very small epsilon: variance + eps ≈ 0 for constant channels.
///
/// When input is constant (var=0), `inv_std = 1/sqrt(eps)` can be very large
/// for small eps. Output should still be approximately zero (since x-mean=0).
#[test]
fn test_instance_norm_ref_small_eps_constant_input() {
    let x = vec![42.0f32; 8];
    let eps = 1e-12;
    let out = instance_norm_ref(&x, 1, 1, 8, eps).expect("ref must succeed");
    for &v in &out {
        assert!(
            v.abs() < 1e-3,
            "constant input with small eps should still normalize to ~0, got {v}"
        );
    }
}

/// Very small epsilon with near-constant input: inv_std becomes very large
/// when variance is near zero (nearly constant channel).
///
/// Uses `base * f32::EPSILON` as the perturbation, NOT `f32::EPSILON` alone.
/// `f32::EPSILON` is the ULP for 1.0; for values around 100.0, the ULP is
/// `~7.6e-6`, so `100.0 + f32::EPSILON == 100.0` in f32 (rounds down).
/// We need `base * f32::EPSILON` (≈1.19e-5) to produce a representable delta.
#[test]
fn test_instance_norm_ref_small_eps_near_constant() {
    let base = 100.0_f32;
    // ULP-scale perturbation: base * EPSILON ≈ 1.19e-5, just above the ULP for base.
    let delta = base * f32::EPSILON;
    debug_assert_ne!(base, base + delta, "delta must be representable in f32");
    let x = vec![base, base + delta, base, base + delta];
    let eps = 1e-12;
    let out = instance_norm_ref(&x, 1, 1, 4, eps).expect("ref must succeed");
    for &v in &out {
        assert!(
            v.is_finite(),
            "near-constant input with tiny eps must produce finite output, got {v}"
        );
    }
}

/// Regression test for catastrophic cancellation (fixed by #102).
///
/// The old one-pass formula `mean(x²) - mean(x)²` produced negative variance
/// for large inputs, causing NaN. The two-pass centered formula
/// `mean((x - mean)²)` avoids this.
#[test]
fn test_instance_norm_ref_large_values_no_cancellation() {
    let base = 1e6_f32;
    let x = vec![base, base + 1.0, base + 2.0, base + 3.0];
    let eps = 1e-5;
    let out = instance_norm_ref(&x, 1, 1, 4, eps).expect("ref must succeed");

    for &v in &out {
        assert!(
            v.is_finite(),
            "two-pass variance must produce finite output for large inputs, got {v}"
        );
    }
    // Verify output is approximately normalized (mean ≈ 0)
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-3, "output mean should be ~0, got {mean}");
}

/// Higher-magnitude regression test: base=1e7 to verify the fix holds.
///
/// At 1e7, the f32 ULP is 1.0 so `base + 1.0` etc. are still representable.
/// The centered differences are [-1.5, -0.5, 0.5, 1.5], well within f32 precision.
/// However, the f32 mean computation itself accumulates rounding error at this
/// magnitude (sum of 4 values near 1e7 loses low bits), so the normalized output
/// mean drifts to ~0.4 rather than ~0. This is expected f32 behavior — the key
/// property is finiteness (no NaN from the old cancellation bug).
#[test]
fn test_instance_norm_ref_large_values_1e7_no_cancellation() {
    let base = 1e7_f32;
    let x = vec![base, base + 1.0, base + 2.0, base + 3.0];
    let eps = 1e-5;
    let out = instance_norm_ref(&x, 1, 1, 4, eps).expect("ref must succeed");

    for &v in &out {
        assert!(
            v.is_finite(),
            "two-pass variance must produce finite output at 1e7 magnitude, got {v}"
        );
    }
    // Output mean drifts from 0 due to f32 rounding in mean computation
    // at 1e7 magnitude. Assert bounded, not exact.
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(
        mean.abs() < 1.0,
        "output mean should be bounded, got {mean}"
    );
}

/// Large constant input: variance is exactly zero (no cancellation for
/// truly identical values). Verifies that the constant-input path remains
/// correct even at large magnitudes where cancellation could occur for
/// non-constant inputs.
///
/// Contrast with `test_instance_norm_ref_large_values_no_cancellation` which
/// uses NON-constant large input (fixed by #102 two-pass variance).
#[test]
fn test_instance_norm_ref_large_constant_input_zero_variance() {
    let base = 1e7_f32;
    let x = vec![base; 4];
    let eps = 1e-5;
    let out = instance_norm_ref(&x, 1, 1, 4, eps).expect("ref must succeed");

    // Constant input: var = mean((x - mean)²) = mean(0²) = 0 exactly.
    // inv_std = 1/sqrt(0 + eps) = 1/sqrt(eps). Output = (x - mean) * inv_std = 0.
    for &v in &out {
        assert!(
            v.is_finite(),
            "constant large input must produce finite output, got {v}"
        );
        assert!(v.abs() < 1e-3, "expected ~0, got {v}");
    }
}

/// eps = 0 is now rejected (#162). Previously this produced finite output
/// when variance > 0 but NaN when variance = 0.
#[test]
fn test_instance_norm_ref_zero_eps_returns_err() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = instance_norm_ref(&x, 1, 1, 4, 0.0);
    assert!(result.is_err(), "eps=0 must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("eps"),
        "error should mention eps, got: {err}"
    );
}

/// Negative eps is rejected (#162).
#[test]
fn test_instance_norm_ref_negative_eps_returns_err() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = instance_norm_ref(&x, 1, 1, 4, -1.0);
    assert!(result.is_err(), "negative eps must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("eps"),
        "error should mention eps, got: {err}"
    );
}

/// Dimension overflow is caught (#162).
#[test]
fn test_instance_norm_ref_dimension_overflow_returns_err() {
    let result = instance_norm_ref(&[1.0; 4], usize::MAX, usize::MAX, 4, 1e-5);
    assert!(result.is_err(), "dimension overflow must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "error should mention overflow, got: {err}"
    );
}

/// NaN eps is rejected by the `!eps.is_finite()` guard.
/// Without this guard, `f32::NAN <= 0.0` is false (IEEE 754 #66),
/// so NaN eps would bypass the `eps <= 0.0` check.
#[test]
fn test_instance_norm_ref_nan_eps_returns_err() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = instance_norm_ref(&x, 1, 1, 4, f32::NAN);
    assert!(result.is_err(), "NaN eps must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("eps"),
        "error should mention eps, got: {err}"
    );
}

/// +Infinity eps is rejected by the `!eps.is_finite()` guard.
/// Without this, `var + Inf = Inf` and `1/sqrt(Inf) = 0`, silently
/// zeroing out all output.
#[test]
fn test_instance_norm_ref_inf_eps_returns_err() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = instance_norm_ref(&x, 1, 1, 4, f32::INFINITY);
    assert!(result.is_err(), "+Inf eps must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("eps"),
        "error should mention eps, got: {err}"
    );
}

/// -Infinity eps is rejected (caught by both `!is_finite()` and `<= 0.0`).
#[test]
fn test_instance_norm_ref_neg_inf_eps_returns_err() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = instance_norm_ref(&x, 1, 1, 4, f32::NEG_INFINITY);
    assert!(result.is_err(), "-Inf eps must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("eps"),
        "error should mention eps, got: {err}"
    );
}

/// Regression: t > 2^24 causes silent precision loss in `t as f32`.
/// The guard must reject this before computing mean/variance.
#[test]
fn test_instance_norm_ref_t_exceeds_f32_precision_returns_err() {
    let t = (1 << 24) + 1; // 16_777_217 — first integer not representable in f32
    let result = instance_norm_ref(&[], 1, 1, t, 1e-5);
    assert!(result.is_err(), "t > 2^24 must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("f32 precision"),
        "error should mention f32 precision, got: {err}"
    );
}

/// t == 2^24 is the boundary — should be accepted (lossless).
#[test]
fn test_instance_norm_ref_t_at_f32_precision_boundary_ok() {
    let t = 1 << 24;
    let result = instance_norm_ref(&[], 1, 1, t, 1e-5);
    assert!(result.is_err(), "should fail at shape check, not precision");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("shape mismatch") || err.to_string().contains("ShapeMismatch"),
        "t=2^24 should pass precision check but fail shape check, got: {err}"
    );
}

#[test]
fn test_instance_norm_ref_nan_input_rejected() {
    let x = &[1.0, f32::NAN, 3.0, 4.0];
    let err = instance_norm_ref(x, 1, 1, 4, 1e-5).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "x",
                index: 1,
                ..
            }
        ),
        "NaN at x[1] should produce NonFiniteSliceElement, got: {err}"
    );
}
