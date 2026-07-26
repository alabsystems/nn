// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for LayerNorm (K7) kernel.

use super::*;
use crate::ir::ScalarType;

// --- TensorKernelDef builder ---

#[test]
fn test_layer_norm_decomposed_validates() {
    let k7 = build_layer_norm_decomposed(4, 8).expect("build must succeed");
    k7.validate().expect("K7 LayerNorm IR must validate");
}

#[test]
fn test_layer_norm_decomposed_zero_dim_returns_err() {
    assert!(build_layer_norm_decomposed(0, 8).is_err());
    assert!(build_layer_norm_decomposed(4, 0).is_err());
}

#[test]
fn test_layer_norm_decomposed_node_count() {
    let k7 = build_layer_norm_decomposed(4, 8).expect("build must succeed");
    assert_eq!(
        k7.nodes.len(),
        18,
        "4 inputs + 2 reduce + 5 broadcast + 7 elementwise = 18"
    );
}

#[test]
fn test_layer_norm_decomposed_output_shape() {
    let k7 = build_layer_norm_decomposed(4, 8).expect("build must succeed");
    let output_shape = &k7.nodes[k7.output.index()].shape;
    assert_eq!(output_shape, &[4, 8]);
}

#[test]
fn test_layer_norm_decomposed_pretty_print() {
    let k7 = build_layer_norm_decomposed(2, 4).expect("build must succeed");
    let ir = crate::tensor_ir::tensor_ir_pretty_print(&k7);
    assert!(ir.contains("tensor_kernel layer_norm"));
    assert!(ir.contains("reduce_mean"));
    assert!(ir.contains("broadcast"));
    assert!(ir.contains("elementwise(rsqrt"));
    assert!(ir.contains("return %17"));
}

#[test]
fn test_layer_norm_decomposed_dispatch_plan() {
    use crate::codegen_msl_tensor::{build_dispatch_plan, DispatchStep};

    let k7 = build_layer_norm_decomposed(4, 8).expect("build must succeed");
    let (plan, _) = build_dispatch_plan(&k7, ScalarType::F32).expect("dispatch plan must succeed");

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

    assert_eq!(reduce_count, 2, "2 reductions: mean(x), mean((x-mean)²)");
    assert_eq!(
        ew_count, 7,
        "7 element-wise: sub, square, add(var+eps), rsqrt, mul(norm), mul(gamma), add(beta)"
    );
    assert_eq!(bc_count, 5, "5 broadcasts: mean, var, eps, gamma, beta");
}

#[test]
fn test_layer_norm_decomposed_msl_codegen() {
    let k7 = build_layer_norm_decomposed(4, 8).expect("build must succeed");
    let msl = crate::codegen_msl_tensor_emit::emit_tensor_msl(&k7, ScalarType::F32)
        .expect("MSL codegen must succeed");
    assert!(msl.contains("reduce_dim"));
    assert!(msl.contains("threadgroup_barrier"));
}

// --- Reference implementation ---

#[test]
fn test_layer_norm_ref_known_values() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let gamma = [1.0; 4];
    let beta = [0.0; 4];
    let eps = 1e-5;
    let out = layer_norm_ref(&x, &gamma, &beta, 1, 4, eps).expect("ref must succeed");

    // mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25 + eps)
    let mean = 2.5;
    let var = 1.25;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| (v - mean) * inv_std).collect();

    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_layer_norm_ref_with_affine() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let gamma = [2.0, 0.5, 1.0, 3.0];
    let beta = [0.1, -0.1, 0.0, 0.5];
    let eps = 1e-5;
    let out = layer_norm_ref(&x, &gamma, &beta, 1, 4, eps).expect("ref must succeed");

    let mean = 2.5;
    let var = 1.25;
    let inv_std = 1.0 / (var + eps).sqrt();

    for (i, &xi) in x.iter().enumerate() {
        let expected = (xi - mean) * inv_std * gamma[i] + beta[i];
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "mismatch at index {i}: got {}, expected {expected}",
            out[i]
        );
    }
}

#[test]
fn test_layer_norm_ref_identity_affine_matches_instance_norm() {
    // With gamma=1, beta=0, LayerNorm on [1, hidden] == InstanceNorm on [1, 1, hidden]
    let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let eps = 1e-5;

    let ln_out = layer_norm_ref(&x, &gamma, &beta, 1, 8, eps).expect("LN ref");
    let in_out = crate::instance_norm_ref(&x, 1, 1, 8, eps).expect("IN ref");

    for (i, (&ln, &in_val)) in ln_out.iter().zip(in_out.iter()).enumerate() {
        assert!(
            (ln - in_val).abs() < 1e-5,
            "LN vs IN mismatch at {i}: {ln} vs {in_val}"
        );
    }
}

#[test]
fn test_layer_norm_ref_output_has_zero_mean_unit_var() {
    let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 1.5).collect();
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let eps = 1e-5;
    let out = layer_norm_ref(&x, &gamma, &beta, 2, 8, eps).expect("ref must succeed");

    for row in 0..2 {
        let slice = &out[row * 8..(row + 1) * 8];
        let mean: f32 = slice.iter().sum::<f32>() / 8.0;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-5, "row {row} mean should be ~0, got {mean}");
        assert!(
            (var - 1.0).abs() < 0.01,
            "row {row} variance should be ~1, got {var}"
        );
    }
}

#[test]
fn test_layer_norm_ref_constant_input() {
    let x = vec![42.0f32; 8];
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let out = layer_norm_ref(&x, &gamma, &beta, 1, 8, 1e-5).expect("ref must succeed");
    for &v in &out {
        assert!(v.abs() < 1e-3, "constant input normalizes to ~0, got {v}");
    }
}

#[test]
fn test_layer_norm_ref_large_values_no_cancellation() {
    let base = 1e6_f32;
    let x = vec![base, base + 1.0, base + 2.0, base + 3.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let out = layer_norm_ref(&x, &gamma, &beta, 1, 4, 1e-5).expect("ref must succeed");
    for &v in &out {
        assert!(
            v.is_finite(),
            "two-pass variance must produce finite output for large inputs, got {v}"
        );
    }
}

// --- Per-element scalar function ---

#[test]
fn test_layer_norm_scalar_known_values() {
    // mean = 2.5, var = 1.25, eps = 1e-5, gamma = 1, beta = 0
    let mean = 2.5f32;
    let var = 1.25f32;
    let eps = 1e-5f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let x_vals = [1.0f32, 2.0, 3.0, 4.0];
    for &x in &x_vals {
        let y = layer_norm_scalar(x, mean, var, eps, 1.0, 0.0).expect("finite inputs");
        let expected = (x - mean) * inv_std;
        assert!(
            (y - expected).abs() < 1e-5,
            "element({x}): got {y}, expected {expected}"
        );
    }
}

#[test]
fn test_layer_norm_scalar_with_affine() {
    let y = layer_norm_scalar(3.0, 2.5, 1.25, 1e-5, 2.0, 0.5).expect("finite inputs");
    let inv_std = 1.0 / (1.25f32 + 1e-5).sqrt();
    let expected = (3.0 - 2.5) * inv_std * 2.0 + 0.5;
    assert!(
        (y - expected).abs() < 1e-5,
        "affine element: got {y}, expected {expected}"
    );
}

#[test]
fn test_layer_norm_scalar_zero_variance() {
    // All inputs same → var = 0, mean = x → output = beta
    let y = layer_norm_scalar(5.0, 5.0, 0.0, 1e-5, 1.0, 0.0).expect("finite inputs");
    assert!(
        y.abs() < 1e-3,
        "constant input should normalize to ~0, got {y}"
    );

    let y_beta = layer_norm_scalar(5.0, 5.0, 0.0, 1e-5, 1.0, 3.0).expect("finite inputs");
    assert!(
        (y_beta - 3.0).abs() < 1e-3,
        "constant input with beta=3 should give ~3, got {y_beta}"
    );
}

// --- Extreme parameter ranges (P1]71 audit coverage) ---

#[test]
fn test_layer_norm_scalar_large_gamma_amplifies() {
    // Large gamma amplifies the normalized value. With var near eps,
    // inv_std ~ 1/sqrt(eps), so the product can be large.
    let x = 10.0f32;
    let mean = 0.0;
    let var = 0.0; // zero variance → inv_std = 1/sqrt(eps)
    let eps = 1e-5;
    let gamma = 1e4;
    let beta = 0.0;
    let y = layer_norm_scalar(x, mean, var, eps, gamma, beta).expect("finite inputs");
    // (10 - 0) * 1/sqrt(1e-5) * 1e4 = 10 * 316.23 * 1e4 = ~3.16e7
    assert!(y.is_finite(), "large gamma with zero var: got {y}");
    assert!(y > 1e6, "expected large output, got {y}");
}

#[test]
fn test_layer_norm_scalar_extreme_spread_stays_finite() {
    // x at Kani domain edge, gamma at Kani edge, tiny eps
    let x = 1e3_f32;
    let mean = -1e3;
    let var = 0.0;
    let eps = 1e-8;
    let gamma = 10.0;
    let beta = 10.0;
    let y = layer_norm_scalar(x, mean, var, eps, gamma, beta).expect("finite inputs");
    // (1e3 - (-1e3)) * 1/sqrt(1e-8) * 10 + 10 = 2e3 * 1e4 * 10 + 10 = 2e8
    assert!(y.is_finite(), "extreme spread: got {y}");
}

#[test]
fn test_layer_norm_scalar_large_but_finite_beyond_kani_domain() {
    // Beyond Kani-proved domain but still within f32 range — returns Ok.
    let x = 1e10_f32;
    let mean = 0.0;
    let var = 0.0;
    let eps = 1e-8;
    let gamma = 1e10;
    let beta = 0.0;
    let y = layer_norm_scalar(x, mean, var, eps, gamma, beta).expect("output is finite");
    // (1e10) * 1e4 * 1e10 = 1e24 → still finite in f32 (f32::MAX ~ 3.4e38)
    assert!(y > 0.0, "expected positive output, got {y}");
}

#[test]
fn test_layer_norm_scalar_overflow_returns_err() {
    // Inputs that cause overflow: huge x, tiny eps, huge gamma
    // (1e20) * 1/sqrt(1e-8) * 1e20 = 1e20 * 1e4 * 1e20 = 1e44 > f32::MAX
    let result = layer_norm_scalar(1e20, 0.0, 0.0, 1e-8, 1e20, 0.0);
    assert!(
        result.is_err(),
        "overflow beyond f32::MAX should return Err, got {result:?}"
    );
}

#[test]
fn test_layer_norm_scalar_nan_input_returns_err() {
    assert!(layer_norm_scalar(f32::NAN, 0.0, 0.0, 1e-5, 1.0, 0.0).is_err());
    assert!(layer_norm_scalar(0.0, f32::NAN, 0.0, 1e-5, 1.0, 0.0).is_err());
    assert!(layer_norm_scalar(0.0, 0.0, f32::NAN, 1e-5, 1.0, 0.0).is_err());
    assert!(layer_norm_scalar(0.0, 0.0, 0.0, f32::NAN, 1.0, 0.0).is_err());
    assert!(layer_norm_scalar(0.0, 0.0, 0.0, 1e-5, f32::NAN, 0.0).is_err());
    assert!(layer_norm_scalar(0.0, 0.0, 0.0, 1e-5, 1.0, f32::NAN).is_err());
}

#[test]
fn test_layer_norm_scalar_inf_input_returns_err() {
    assert!(layer_norm_scalar(f32::INFINITY, 0.0, 0.0, 1e-5, 1.0, 0.0).is_err());
    assert!(layer_norm_scalar(0.0, 0.0, 0.0, 1e-5, f32::NEG_INFINITY, 0.0).is_err());
}

#[test]
fn test_layer_norm_scalar_negative_gamma_flips_sign() {
    let y = layer_norm_scalar(3.0, 2.5, 1.25, 1e-5, -2.0, 0.0).expect("finite inputs");
    let inv_std = 1.0 / (1.25f32 + 1e-5).sqrt();
    let expected = (3.0 - 2.5) * inv_std * -2.0;
    assert!(
        (y - expected).abs() < 1e-5,
        "negative gamma: got {y}, expected {expected}"
    );
}

// --- Error cases ---

#[test]
fn test_layer_norm_ref_zero_eps_returns_err() {
    assert!(layer_norm_ref(&[1.0; 4], &[1.0; 4], &[0.0; 4], 1, 4, 0.0).is_err());
}

#[test]
fn test_layer_norm_ref_nan_eps_returns_err() {
    assert!(layer_norm_ref(&[1.0; 4], &[1.0; 4], &[0.0; 4], 1, 4, f32::NAN).is_err());
}

#[test]
fn test_layer_norm_ref_wrong_x_length_returns_err() {
    assert!(layer_norm_ref(&[1.0; 3], &[1.0; 4], &[0.0; 4], 1, 4, 1e-5).is_err());
}

#[test]
fn test_layer_norm_ref_wrong_gamma_length_returns_err() {
    assert!(layer_norm_ref(&[1.0; 4], &[1.0; 3], &[0.0; 4], 1, 4, 1e-5).is_err());
}

#[test]
fn test_layer_norm_ref_wrong_beta_length_returns_err() {
    assert!(layer_norm_ref(&[1.0; 4], &[1.0; 4], &[0.0; 3], 1, 4, 1e-5).is_err());
}

/// Regression: hidden > 2^24 causes silent precision loss in `hidden as f32`.
/// The guard must reject this before computing mean/variance.
#[test]
fn test_layer_norm_ref_hidden_exceeds_f32_precision_returns_err() {
    let hidden = (1 << 24) + 1; // 16_777_217 — first integer not representable in f32
    let result = layer_norm_ref(&[], &[], &[], 1, hidden, 1e-5);
    assert!(result.is_err(), "hidden > 2^24 must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("f32 precision"),
        "error should mention f32 precision, got: {err}"
    );
}

/// hidden == 2^24 is the boundary — should be accepted (lossless).
#[test]
fn test_layer_norm_ref_hidden_at_f32_precision_boundary_ok() {
    let hidden = 1 << 24;
    let result = layer_norm_ref(&[], &[], &[], 1, hidden, 1e-5);
    assert!(result.is_err(), "should fail at shape check, not precision");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("shape mismatch") || err.to_string().contains("ShapeMismatch"),
        "hidden=2^24 should pass precision check but fail shape check, got: {err}"
    );
}

#[test]
fn test_layer_norm_ref_nan_x_rejected() {
    let x = &[1.0, f32::NAN, 3.0, 4.0];
    let gamma = &[1.0, 1.0, 1.0, 1.0];
    let beta = &[0.0, 0.0, 0.0, 0.0];
    let err = layer_norm_ref(x, gamma, beta, 1, 4, 1e-5).unwrap_err();
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
fn test_layer_norm_ref_inf_gamma_rejected() {
    let x = &[1.0, 2.0, 3.0, 4.0];
    let gamma = &[1.0, f32::INFINITY, 1.0, 1.0];
    let beta = &[0.0, 0.0, 0.0, 0.0];
    let err = layer_norm_ref(x, gamma, beta, 1, 4, 1e-5).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "gamma",
                index: 1,
                ..
            }
        ),
        "Inf at gamma[1] should be caught, got: {err}"
    );
}
