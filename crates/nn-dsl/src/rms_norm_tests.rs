// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for RMSNorm (K5) kernel.

use super::*;
use crate::ir::ScalarType;

// --- TensorKernelDef builder ---

#[test]
fn test_rms_norm_decomposed_validates() {
    let k5 = build_rms_norm_decomposed(4, 8).expect("build must succeed");
    k5.validate().expect("K5 RMSNorm IR must validate");
}

#[test]
fn test_rms_norm_decomposed_zero_dim_returns_err() {
    let result = build_rms_norm_decomposed(0, 8);
    assert!(result.is_err(), "zero n must return Err");
    let result2 = build_rms_norm_decomposed(4, 0);
    assert!(result2.is_err(), "zero hidden must return Err");
}

#[test]
fn test_rms_norm_decomposed_node_count() {
    let k5 = build_rms_norm_decomposed(4, 8).expect("build must succeed");
    assert_eq!(
        k5.nodes.len(),
        12,
        "3 inputs + 1 reduce + 3 broadcast + 5 elementwise = 12"
    );
}

#[test]
fn test_rms_norm_decomposed_output_shape() {
    let k5 = build_rms_norm_decomposed(4, 8).expect("build must succeed");
    let output_shape = &k5.nodes[k5.output.index()].shape;
    assert_eq!(output_shape, &[4, 8]);
}

#[test]
fn test_rms_norm_decomposed_pretty_print() {
    let k5 = build_rms_norm_decomposed(2, 4).expect("build must succeed");
    let ir = crate::tensor_ir::tensor_ir_pretty_print(&k5);
    assert!(ir.contains("tensor_kernel rms_norm"));
    assert!(ir.contains("reduce_mean"));
    assert!(ir.contains("broadcast"));
    assert!(ir.contains("elementwise(rsqrt"));
    assert!(ir.contains("return %11"));
}

#[test]
fn test_rms_norm_decomposed_dispatch_plan() {
    use crate::codegen_msl_tensor::{build_dispatch_plan, DispatchStep};

    let k5 = build_rms_norm_decomposed(4, 8).expect("build must succeed");
    let (plan, _) = build_dispatch_plan(&k5, ScalarType::F32).expect("dispatch plan must succeed");

    let reduce_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    let ew_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Elementwise { .. }))
        .count();
    let bc_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Broadcast { .. }))
        .count();

    assert_eq!(reduce_count, 1, "1 reduction: mean(x²)");
    assert_eq!(
        ew_count, 5,
        "5 element-wise: square, add, rsqrt, mul(x*rsqrt), mul(*weight)"
    );
    assert_eq!(bc_count, 3, "3 broadcasts: mean(x²), eps, weight");
}

#[test]
fn test_rms_norm_decomposed_msl_codegen() {
    let k5 = build_rms_norm_decomposed(4, 8).expect("build must succeed");
    let msl = crate::codegen_msl_tensor_emit::emit_tensor_msl(&k5, ScalarType::F32)
        .expect("MSL codegen must succeed");
    assert!(msl.contains("reduce_dim"));
    assert!(msl.contains("threadgroup_barrier"));
}

// --- Reference implementation ---

#[test]
fn test_rms_norm_ref_known_values() {
    // x = [1, 2, 3, 4], weight = [1, 1, 1, 1], eps = 1e-5
    // mean(x²) = (1 + 4 + 9 + 16) / 4 = 7.5
    // rms_inv = 1/sqrt(7.5 + 1e-5) ≈ 0.3651
    // output = x * rms_inv * 1
    let x = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0; 4];
    let eps = 1e-5;
    let out = rms_norm_ref(&x, &weight, 1, 4, eps).expect("ref must succeed");

    let mean_sq = 7.5;
    let rms_inv = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| v * rms_inv).collect();

    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_rms_norm_ref_with_weight() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let weight = [0.5, 1.0, 2.0, 0.1];
    let eps = 1e-5;
    let out = rms_norm_ref(&x, &weight, 1, 4, eps).expect("ref must succeed");

    let mean_sq = 7.5;
    let rms_inv = 1.0 / (mean_sq + eps).sqrt();
    let expected: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * rms_inv * wi)
        .collect();

    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_rms_norm_ref_multi_row() {
    let x = [1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0]; // 2 rows × 4 hidden
    let weight = [1.0; 4];
    let eps = 1e-5;
    let out = rms_norm_ref(&x, &weight, 2, 4, eps).expect("ref must succeed");
    assert_eq!(out.len(), 8);

    // Row 0: mean(x²) = (1+0+0+1)/4 = 0.5
    let rms_inv_0 = 1.0 / (0.5 + eps).sqrt();
    assert!((out[0] - 1.0 * rms_inv_0).abs() < 1e-5);
    assert!((out[1] - 0.0).abs() < 1e-5);

    // Row 1: mean(x²) = (0+4+4+0)/4 = 2.0
    let rms_inv_1 = 1.0 / (2.0 + eps).sqrt();
    assert!((out[5] - 2.0 * rms_inv_1).abs() < 1e-5);
}

#[test]
fn test_rms_norm_ref_constant_input() {
    // For constant input c: mean(c²) = c², rms_inv = 1/sqrt(c²+eps) ≈ 1/|c|
    // output ≈ c/|c| * weight = sign(c) * weight
    let c = 5.0;
    let x = vec![c; 8];
    let weight = vec![1.0; 8];
    let out = rms_norm_ref(&x, &weight, 1, 8, 1e-5).expect("ref must succeed");
    for &v in &out {
        assert!(
            (v - 1.0).abs() < 0.01,
            "constant positive input with weight=1 should normalize to ~1, got {v}"
        );
    }
}

#[test]
fn test_rms_norm_ref_zero_input() {
    let x = vec![0.0f32; 4];
    let weight = vec![1.0; 4];
    let out = rms_norm_ref(&x, &weight, 1, 4, 1e-5).expect("ref must succeed");
    for &v in &out {
        assert!(v.abs() < 1e-3, "zero input should normalize to ~0, got {v}");
    }
}

// --- Error cases ---

#[test]
fn test_rms_norm_ref_zero_eps_returns_err() {
    let result = rms_norm_ref(&[1.0; 4], &[1.0; 4], 1, 4, 0.0);
    assert!(result.is_err(), "eps=0 must return Err");
}

#[test]
fn test_rms_norm_ref_nan_eps_returns_err() {
    let result = rms_norm_ref(&[1.0; 4], &[1.0; 4], 1, 4, f32::NAN);
    assert!(result.is_err(), "NaN eps must return Err");
}

#[test]
fn test_rms_norm_ref_wrong_x_length_returns_err() {
    let result = rms_norm_ref(&[1.0; 3], &[1.0; 4], 1, 4, 1e-5);
    assert!(result.is_err(), "wrong x length must return Err");
}

#[test]
fn test_rms_norm_ref_wrong_weight_length_returns_err() {
    let result = rms_norm_ref(&[1.0; 4], &[1.0; 3], 1, 4, 1e-5);
    assert!(result.is_err(), "wrong weight length must return Err");
}

#[test]
fn test_rms_norm_ref_large_values_no_overflow() {
    let x = vec![1e3_f32; 4];
    let weight = vec![1.0; 4];
    let out = rms_norm_ref(&x, &weight, 1, 4, 1e-5).expect("ref must succeed");
    for &v in &out {
        assert!(
            v.is_finite(),
            "large input must produce finite output, got {v}"
        );
        assert!(
            (v - 1.0).abs() < 0.01,
            "large constant input should normalize to ~1, got {v}"
        );
    }
}

// --- rms_norm_scalar unit tests ---

#[test]
fn test_rms_norm_scalar_known_values() {
    // x=2, rms_inv=0.5, weight=1 → 2 * 0.5 * 1 = 1.0
    let y = rms_norm_scalar(2.0, 0.5, 1.0).expect("must succeed");
    assert!((y - 1.0).abs() < 1e-6, "expected 1.0, got {y}");
}

#[test]
fn test_rms_norm_scalar_with_weight() {
    // x=3, rms_inv=0.25, weight=2 → 3 * 0.25 * 2 = 1.5
    let y = rms_norm_scalar(3.0, 0.25, 2.0).expect("must succeed");
    assert!((y - 1.5).abs() < 1e-6, "expected 1.5, got {y}");
}

#[test]
fn test_rms_norm_scalar_zero_input() {
    let y = rms_norm_scalar(0.0, 0.5, 1.0).expect("must succeed");
    assert!(y == 0.0, "rms_norm_scalar(0, _, _) should be 0, got {y}");
}

#[test]
fn test_rms_norm_scalar_negative_input() {
    // RMSNorm preserves sign: x<0 → output<0 (when rms_inv>0, weight>0)
    let y = rms_norm_scalar(-4.0, 0.5, 1.0).expect("must succeed");
    assert!((y - (-2.0)).abs() < 1e-6, "expected -2.0, got {y}");
}

#[test]
fn test_rms_norm_scalar_matches_ref() {
    // Verify rms_norm_scalar matches rms_norm_ref for a single-element row.
    let x = [3.0f32];
    let weight = [2.0f32];
    let eps = 1e-5f32;
    let ref_out = rms_norm_ref(&x, &weight, 1, 1, eps).expect("ref must succeed");

    // For single-element: mean(x²)=x²=9, rms_inv=1/sqrt(9+eps)≈1/3
    let rms_inv = 1.0 / (9.0f32 + eps).sqrt();
    let scalar_out = rms_norm_scalar(3.0, rms_inv, 2.0).expect("must succeed");

    assert!(
        (ref_out[0] - scalar_out).abs() < 1e-5,
        "scalar ({scalar_out}) should match ref ({})",
        ref_out[0]
    );
}

// --- rms_norm_scalar error path tests ---

#[test]
fn test_rms_norm_scalar_nan_input_returns_err() {
    use crate::kernel_error::KernelError;
    let err = rms_norm_scalar(f32::NAN, 0.5, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
    let err = rms_norm_scalar(1.0, f32::NAN, 1.0).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteInput {
                name: "rms_inv",
                ..
            }
        ),
        "expected NonFiniteInput for rms_inv, got {err:?}"
    );
    let err = rms_norm_scalar(1.0, 0.5, f32::NAN).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "weight", .. }),
        "expected NonFiniteInput for weight, got {err:?}"
    );
}

#[test]
fn test_rms_norm_scalar_inf_input_returns_err() {
    use crate::kernel_error::KernelError;
    let err = rms_norm_scalar(f32::INFINITY, 0.5, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x", .. }),
        "expected NonFiniteInput for x, got {err:?}"
    );
    let err = rms_norm_scalar(1.0, f32::NEG_INFINITY, 1.0).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteInput {
                name: "rms_inv",
                ..
            }
        ),
        "expected NonFiniteInput for rms_inv, got {err:?}"
    );
}

#[test]
fn test_rms_norm_scalar_overflow_returns_err() {
    use crate::kernel_error::KernelError;
    // x * rms_inv * weight overflows to Inf for extreme values
    let err = rms_norm_scalar(f32::MAX, f32::MAX, 2.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteOutput { name: "output", .. }),
        "expected NonFiniteOutput for output, got {err:?}"
    );
}

/// Regression: hidden > 2^24 causes silent precision loss in `hidden as f32`.
/// The guard must reject this before computing mean(x²).
#[test]
fn test_rms_norm_ref_hidden_exceeds_f32_precision_returns_err() {
    let hidden = (1 << 24) + 1; // 16_777_217 — first integer not representable in f32
    let result = rms_norm_ref(&[], &[], 1, hidden, 1e-5);
    assert!(result.is_err(), "hidden > 2^24 must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("f32 precision"),
        "error should mention f32 precision, got: {err}"
    );
}

/// hidden == 2^24 is the boundary — should be accepted (lossless).
#[test]
fn test_rms_norm_ref_hidden_at_f32_precision_boundary_ok() {
    let hidden = 1 << 24;
    let result = rms_norm_ref(&[], &[], 1, hidden, 1e-5);
    assert!(result.is_err(), "should fail at shape check, not precision");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("shape mismatch") || err.to_string().contains("ShapeMismatch"),
        "hidden=2^24 should pass precision check but fail shape check, got: {err}"
    );
}

#[test]
fn test_rms_norm_ref_nan_x_rejected() {
    let x = &[1.0, f32::NAN, 3.0, 4.0];
    let weight = &[1.0, 1.0, 1.0, 1.0];
    let err = rms_norm_ref(x, weight, 1, 4, 1e-5).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "x",
                index: 1,
                ..
            }
        ),
        "NaN at x[1] should be caught, got: {err}"
    );
}

#[test]
fn test_rms_norm_ref_inf_weight_rejected() {
    let x = &[1.0, 2.0, 3.0, 4.0];
    let weight = &[1.0, 1.0, f32::INFINITY, 1.0];
    let err = rms_norm_ref(x, weight, 1, 4, 1e-5).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "weight",
                index: 2,
                ..
            }
        ),
        "Inf at weight[2] should be caught, got: {err}"
    );
}
