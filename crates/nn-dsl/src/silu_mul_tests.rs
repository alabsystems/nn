// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SiLU-Mul (K8) kernel.

use super::*;
use crate::kernel_error::KernelError;

// --- KernelDef builder ---

#[test]
fn test_silu_mul_kernel_builds_and_validates() {
    let kernel = build_silu_mul_kernel().expect("build must succeed");
    kernel.validate().expect("IR must validate");
    assert_eq!(kernel.name, "silu_mul");
    assert_eq!(kernel.params.len(), 2);
    assert_eq!(kernel.params[0].name, "x");
    assert_eq!(kernel.params[1].name, "up");
}

#[test]
fn test_silu_mul_kernel_msl_codegen() {
    let kernel = build_silu_mul_kernel().expect("build must succeed");
    let msl = crate::emit_msl(&kernel).expect("MSL codegen");
    assert!(msl.contains("silu_mul"), "MSL should contain kernel name");
    assert!(msl.contains("exp("), "MSL should use exp for sigmoid");
    assert!(
        msl.contains("[[kernel]]"),
        "MSL should have kernel attribute"
    );
}

#[test]
fn test_silu_mul_kernel_kani_harness_codegen() {
    let kernel = build_silu_mul_kernel().expect("build must succeed");
    let harness = crate::emit_kani_harness(&kernel).expect("kani codegen");
    assert!(harness.contains("#[kani::proof]"));
    assert!(harness.contains("kani::any"));
}

// --- Scalar reference ---

#[test]
fn test_silu_mul_scalar_at_zero() {
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let result = silu_mul_scalar(0.0, 1.0).expect("must succeed");
    assert!((result - 0.0).abs() < 1e-6, "silu(0)*1 = 0, got {result}");
}

#[test]
fn test_silu_mul_scalar_positive() {
    // silu(1) = 1 / (1 + exp(-1)) ≈ 0.7311
    let result = silu_mul_scalar(1.0, 2.0).expect("must succeed");
    let expected = 1.0 / (1.0 + (-1.0_f32).exp()) * 2.0;
    assert!(
        (result - expected).abs() < 1e-6,
        "silu(1)*2 = {expected}, got {result}"
    );
}

#[test]
fn test_silu_mul_scalar_negative() {
    // silu(-3) = -3 / (1 + exp(3)) ≈ -0.1429
    let result = silu_mul_scalar(-3.0, 1.0).expect("must succeed");
    let expected = -3.0 / (1.0 + 3.0_f32.exp());
    assert!(
        (result - expected).abs() < 1e-5,
        "silu(-3)*1 = {expected}, got {result}"
    );
}

#[test]
fn test_silu_mul_scalar_large_positive() {
    // For large x, sigmoid(x) ≈ 1, so silu(x) ≈ x
    let result = silu_mul_scalar(50.0, 1.0).expect("must succeed");
    assert!((result - 50.0).abs() < 1e-4, "silu(50) ≈ 50, got {result}");
}

#[test]
fn test_silu_mul_scalar_large_negative() {
    // For large negative x, sigmoid(x) ≈ 0, so silu(x) ≈ 0
    let result = silu_mul_scalar(-50.0, 1.0).expect("must succeed");
    assert!(result.abs() < 1e-4, "silu(-50) ≈ 0, got {result}");
}

#[test]
fn test_silu_mul_scalar_up_zero() {
    let result = silu_mul_scalar(5.0, 0.0).expect("must succeed");
    assert!(result.abs() < 1e-10, "silu(x) * 0 = 0, got {result}");
}

// --- silu_mul_scalar error paths ---

#[test]
fn test_silu_mul_scalar_nan_x_returns_err() {
    let err = silu_mul_scalar(f32::NAN, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
}

#[test]
fn test_silu_mul_scalar_nan_up_returns_err() {
    let err = silu_mul_scalar(1.0, f32::NAN).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "up", .. }),
        "expected NonFiniteInput for up, got {err:?}"
    );
}

#[test]
fn test_silu_mul_scalar_inf_x_returns_err() {
    let err = silu_mul_scalar(f32::INFINITY, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
}

#[test]
fn test_silu_mul_scalar_overflow_returns_err() {
    // x * sigmoid(x) * up can overflow for extreme finite values
    let err = silu_mul_scalar(f32::MAX, f32::MAX).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteOutput { name: "output", .. }),
        "expected NonFiniteOutput for output, got {err:?}"
    );
}

// --- Bounds ---

#[test]
fn test_silu_mul_bounds_positive_range() {
    let (lo, hi) = silu_mul_scalar_bounds(1.0, 3.0, 1.0, 2.0).expect("finite inputs");
    // silu(1) ≈ 0.731, silu(3) ≈ 2.858
    // Min: silu(1)*1 ≈ 0.731, Max: silu(3)*2 ≈ 5.716
    assert!(lo <= silu_mul_scalar(1.0, 1.0).unwrap() + 1e-5);
    assert!(hi >= silu_mul_scalar(3.0, 2.0).unwrap() - 1e-5);
}

#[test]
fn test_silu_mul_bounds_contain_sample() {
    // Check that random points within the input box produce outputs within bounds.
    let (lo, hi) = silu_mul_scalar_bounds(-2.0, 2.0, -1.0, 3.0).expect("finite inputs");
    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        for &up in &[-1.0, 0.0, 1.5, 3.0] {
            let val = silu_mul_scalar(x, up).unwrap();
            assert!(
                val >= lo - 1e-5 && val <= hi + 1e-5,
                "silu_mul({x}, {up}) = {val} outside bounds [{lo}, {hi}]"
            );
        }
    }
}

// --- 1d array ---

#[test]
fn test_silu_mul_ref_basic() {
    let x = [0.0, 1.0, -1.0, 2.0];
    let up = [1.0, 1.0, 1.0, 1.0];
    let result = silu_mul_ref(&x, &up).expect("must succeed");
    assert_eq!(result.len(), 4);
    assert!((result[0] - 0.0).abs() < 1e-6, "silu(0)*1 = 0");
}

#[test]
fn test_silu_mul_ref_empty_returns_err() {
    let result = silu_mul_ref(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_silu_mul_ref_length_mismatch_returns_err() {
    let result = silu_mul_ref(&[1.0, 2.0], &[1.0]);
    assert!(result.is_err());
}

// --- Non-finite input guards ---

#[test]
fn test_silu_mul_bounds_nan_x_lo_returns_err() {
    let err = silu_mul_scalar_bounds(f32::NAN, 1.0, 0.0, 1.0).expect_err("NaN x_lo should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_nan()),
        "expected NonFiniteBound with NaN, got {err:?}"
    );
}

#[test]
fn test_silu_mul_bounds_inf_up_hi_returns_err() {
    let err =
        silu_mul_scalar_bounds(0.0, 1.0, 0.0, f32::INFINITY).expect_err("Inf up_hi should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with Inf, got {err:?}"
    );
}

#[test]
fn test_silu_mul_bounds_neg_inf_x_hi_returns_err() {
    let err = silu_mul_scalar_bounds(0.0, f32::NEG_INFINITY, 0.0, 1.0)
        .expect_err("-Inf x_hi should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with -Inf, got {err:?}"
    );
}

// --- Output finiteness guard ---

#[test]
fn test_silu_mul_bounds_overflow_to_inf_returns_err() {
    // Large-magnitude inputs where silu(x) ≈ x, so silu(x)*up can overflow to Inf.
    // silu(1e20) ≈ 1e20, 1e20 * 1e20 = 1e40 > f32::MAX ≈ 3.4e38 → Inf.
    let err =
        silu_mul_scalar_bounds(0.0, 1e20, 0.0, 1e20).expect_err("overflow to Inf should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with Inf, got {err:?}"
    );
}

#[test]
fn test_silu_mul_bounds_negative_overflow_returns_err() {
    // silu(1e20) ≈ 1e20, 1e20 * (-1e20) = -1e40 → -Inf.
    let err =
        silu_mul_scalar_bounds(0.0, 1e20, -1e20, 0.0).expect_err("negative overflow should fail");
    assert!(
        matches!(err, KernelError::NonFiniteBound { value } if value.is_infinite()),
        "expected NonFiniteBound with -Inf, got {err:?}"
    );
}

// --- SiLU non-monotonicity evidence ---

/// SiLU is NOT monotonically increasing: it has a global minimum at x ≈ -1.278.
///
/// Regression test preserving the mathematical fact that motivated the #268 fix.
#[test]
fn test_silu_is_not_monotone() {
    let s_10 = silu_scalar(-10.0);
    let s_5 = silu_scalar(-5.0);
    assert!(
        s_5 < s_10,
        "silu(-5)={s_5} must be < silu(-10)={s_10} (both negative, silu(-5) more negative)"
    );
    let s_min = silu_scalar(-1.278);
    assert!(
        s_min < -0.27 && s_min > -0.29,
        "silu global minimum ≈ -0.278, got {s_min}"
    );
}

/// Regression test for #268: bounds at the SiLU global minimum are now sound.
///
/// Before the fix, silu_mul_scalar_bounds(-10, 0, -10, -10) only evaluated
/// silu at endpoints (≈0 and 0), giving upper ≈ 0.005. But silu(-1.278)*-10 ≈ 2.78
/// was far outside. After the fix, the global minimum is included and bounds
/// correctly contain all outputs.
#[test]
fn test_silu_mul_bounds_sound_at_global_minimum() {
    let (lo, hi) = silu_mul_scalar_bounds(-10.0, 0.0, -10.0, -10.0).expect("finite inputs");
    let actual = silu_mul_scalar(-1.278, -10.0).unwrap();
    assert!(
        actual >= lo - 1e-4 && actual <= hi + 1e-4,
        "silu_mul(-1.278, -10) = {actual} must be within [{lo}, {hi}]"
    );
}

/// Grid soundness test covering the full input range including negative x.
///
/// Sweeps x values spanning the SiLU global minimum (x ≈ -1.278) to verify
/// that bounds are sound everywhere, not just in the positive domain.
#[test]
fn test_silu_mul_bounds_soundness_grid() {
    let x_vals: &[f32] = &[-10.0, -5.0, -2.0, -1.278, -1.0, 0.0, 1.0, 3.0, 10.0];
    for &x_lo in x_vals {
        for &x_hi in x_vals {
            if x_lo > x_hi {
                continue;
            }
            for &up_lo in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
                for &up_hi in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
                    if up_lo > up_hi {
                        continue;
                    }
                    let (lo, hi) =
                        silu_mul_scalar_bounds(x_lo, x_hi, up_lo, up_hi).expect("finite inputs");
                    for xi in 0..5 {
                        let x = x_lo + (x_hi - x_lo) * (xi as f32) / 4.0;
                        for ui in 0..5 {
                            let up = up_lo + (up_hi - up_lo) * (ui as f32) / 4.0;
                            let val = silu_mul_scalar(x, up).unwrap();
                            assert!(
                                val >= lo - 1e-4 && val <= hi + 1e-4,
                                "silu_mul({x}, {up}) = {val} outside [{lo}, {hi}] \
                                 for box x:[{x_lo},{x_hi}] up:[{up_lo},{up_hi}]"
                            );
                        }
                    }
                }
            }
        }
    }
}

// --- InvertedBounds rejection (#271) ---

#[test]
fn test_silu_mul_bounds_rejects_inverted_x() {
    let err = silu_mul_scalar_bounds(5.0, -5.0, -1.0, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::InvertedBounds { lower, upper } if lower == 5.0 && upper == -5.0),
        "inverted x bounds should be rejected, got: {err}"
    );
}

#[test]
fn test_silu_mul_bounds_rejects_inverted_up() {
    let err = silu_mul_scalar_bounds(-1.0, 1.0, 10.0, -10.0).unwrap_err();
    assert!(
        matches!(err, KernelError::InvertedBounds { lower, upper } if lower == 10.0 && upper == -10.0),
        "inverted up bounds should be rejected, got: {err}"
    );
}

// --- MSL diff test codegen ---

#[test]
fn test_silu_mul_differential_test_codegen() {
    use crate::precision::PrecisionTier;
    let kernel = build_silu_mul_kernel().expect("build");
    let test_code = crate::codegen_difftest::emit_differential_test(&kernel, PrecisionTier::Normal)
        .expect("difftest codegen");
    assert!(test_code.contains("SILU_MUL_DESCRIPTOR"));
    assert!(test_code.contains("#[test]"));
}

// --- Tensor-level builder tests (#733) ---

#[test]
fn test_silu_mul_k8_tensor_builds_and_validates() {
    let k8 = build_silu_mul_tensor(4, 128).expect("build must succeed");
    k8.validate().expect("SiLU-Mul K8 tensor IR must validate");
}

#[test]
fn test_silu_mul_k8_tensor_node_count() {
    let k8 = build_silu_mul_tensor(4, 128).expect("build");
    // 3 nodes: x input, up input, elementwise
    assert_eq!(k8.nodes.len(), 3, "SiLU-Mul K8 should have 3 nodes");
}

#[test]
fn test_silu_mul_k8_tensor_output_shape() {
    let k8 = build_silu_mul_tensor(4, 128).expect("build");
    let output = &k8.nodes[k8.output.index()];
    assert_eq!(output.shape, vec![4, 128], "output shape must be [N, dim]");
}

#[test]
fn test_silu_mul_k8_tensor_zero_dim_returns_err() {
    assert!(build_silu_mul_tensor(0, 128).is_err(), "zero n");
    assert!(build_silu_mul_tensor(4, 0).is_err(), "zero dim");
}

// --- Fused vs sequential equivalence (#3537) ---

/// Fused path: single expression `gate * sigmoid(gate) * up` — matches the
/// Metal kernel in `generate_silu_mul_msl`.
fn silu_mul_fused(gate: f32, up: f32) -> f32 {
    let sigmoid = 1.0_f32 / (1.0 + (-gate).exp());
    gate * sigmoid * up
}

/// Sequential path: `silu(gate)` then `* up` — matches the DynTensor bridge
/// in `execute_native_silu_mul`.
fn silu_mul_sequential(gate: f32, up: f32) -> f32 {
    let sigmoid = 1.0_f32 / (1.0 + (-gate).exp());
    let silu_val = gate * sigmoid;
    silu_val * up
}

#[test]
fn test_silu_mul_fused_equals_sequential_grid() {
    // Test that fused and sequential paths produce identical results across
    // a wide range of inputs including edge cases.
    let gate_vals: &[f32] = &[
        -100.0, -50.0, -10.0, -5.0, -2.0, -1.278, -1.0, -0.5, -0.001, 0.0, 0.001, 0.5, 1.0, 2.0,
        5.0, 10.0, 50.0, 100.0,
    ];
    let up_vals: &[f32] = &[-1e4, -100.0, -1.0, -0.001, 0.0, 0.001, 1.0, 100.0, 1e4];
    for &gate in gate_vals {
        for &up in up_vals {
            let fused = silu_mul_fused(gate, up);
            let sequential = silu_mul_sequential(gate, up);
            assert_eq!(
                fused.to_bits(),
                sequential.to_bits(),
                "fused and sequential must be bit-identical for gate={gate}, up={up} \
                 (fused={fused}, sequential={sequential})"
            );
        }
    }
}

#[test]
fn test_silu_mul_fused_matches_scalar_ref() {
    // Verify that the fused formula matches `silu_mul_scalar` reference.
    let test_cases: &[(f32, f32)] = &[
        (0.0, 1.0),
        (1.0, 2.0),
        (-3.0, 1.0),
        (5.0, -0.5),
        (-1.278, -10.0),
        (50.0, 0.001),
    ];
    for &(gate, up) in test_cases {
        let fused = silu_mul_fused(gate, up);
        let reference = silu_mul_scalar(gate, up).expect("finite inputs");
        assert_eq!(
            fused.to_bits(),
            reference.to_bits(),
            "fused must match silu_mul_scalar for gate={gate}, up={up}"
        );
    }
}
