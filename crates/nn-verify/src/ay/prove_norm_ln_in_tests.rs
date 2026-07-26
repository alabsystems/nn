// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K7 LayerNorm + K2 InstanceNorm + K2 InstanceNorm+Affine ay SMT verification tests (#19).
//! Extracted from prove_norm_tests.rs (#356).

use super::prove_dispatch::{instance_norm_output_bounds, norm_affine_output_bounds};
use super::*;
use crate::test_helpers::bounds;

// --- K7 LayerNorm scalar ay SMT tests ---

/// Helper: build LayerNorm (K7) scalar element kernel.
fn layer_norm_kernel() -> KernelDef {
    nn_dsl::build_layer_norm_scalar_kernel().expect("layer_norm_scalar kernel must build")
}

#[test]
fn test_verify_layer_norm_scalar_translates() {
    // layer_norm_scalar(x, mean, var_val, eps, gamma, beta)
    //   = (x - mean) * rsqrt(var_val + eps) * gamma + beta
    // Variable-first (#448): param 0 (x) is symbolic, params 1..5 = Constant.
    // constant_params = [mean=1.0, var_val=0.0, eps=1.0, gamma=0.001, beta=1.0].
    // rsqrt(var_val + eps) operates on constants → ground-folded (#376),
    // yielding Exact encoding and ay direct execution.
    let kernel = layer_norm_kernel();
    let result =
        verify_kernel_smt(&kernel, &[1.0, 0.0, 1.0, 0.001, 1.0], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    // Ground-folding (#376): rsqrt(1.0) = 1.0, identity elimination removes *1.0.
    // But gamma=0.001 still produces real_mul with fractional coefficient.
    // ay#5605 fixed: real_mul with fractional coefficients now works.
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: ground-folded layer_norm must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_layer_norm_scalar_smt2_output() {
    // With ground-folding (#376), rsqrt(constant) is folded to a Real literal.
    // The SMT-LIB2 should NOT contain rsqrt_approx UF.
    let kernel = layer_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.0, 1.0, 0.001, 1.0], bounds(-10.0, 10.0)).unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded layer_norm should NOT contain rsqrt_approx UF"
    );
}

#[test]
fn test_layer_norm_scalar_smt2_declares_variable_no_uf() {
    // layer_norm_scalar has 6 params. With 5 constant_params → 1 declare-const (x).
    // Ground-folding (#376) eliminates rsqrt UF → 0 declare-fun.
    let kernel = layer_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.0, 1.0, 0.001, 1.0], bounds(-10.0, 10.0)).unwrap();
    let declare_const_count = smt2.matches("declare-const").count();
    assert_eq!(
        declare_const_count, 1,
        "expected 1 declare-const (symbolic x), got {declare_const_count}"
    );
    let declare_fun_count = smt2.matches("declare-fun").count();
    assert_eq!(
        declare_fun_count, 0,
        "expected 0 declare-fun (rsqrt ground-folded), got {declare_fun_count}"
    );
}

#[test]
fn test_layer_norm_scalar_nan_constant_params_rejected() {
    let kernel = layer_norm_kernel();
    let err = verify_kernel_smt(
        &kernel,
        &[1.0, f32::NAN, 1.0, 0.001, 1.0],
        bounds(-10.0, 10.0),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_layer_norm_scalar_with_explicit_bounds() {
    // Ground-folded (#376): rsqrt(1.0) eliminated via identity, but gamma=0.001
    // still produces real_mul. ay#5605 fixed: real_mul now works.
    let kernel = layer_norm_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[1.0, 0.0, 1.0, 0.001, 1.0],
        bounds(-10.0, 10.0),
        Some((-20.0, 20.0)),
    )
    .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: layer_norm with explicit bounds must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_layer_norm_scalar_smt2_encodes_subtraction_and_folded_rsqrt() {
    // layer_norm_scalar(x, mean, var_val, eps, gamma, beta)
    //   = (x - mean) * rsqrt(var_val + eps) * gamma + beta
    // Variable-first convention (#448): param 0 (x) is symbolic, params 1..5 = Constant.
    // constant_params = [mean=1.0, var_val=1.0, eps=0.001, gamma=2.0, beta=0.5].
    // x symbolic in [-10, 10].
    // rsqrt(1.0 + 0.001) = rsqrt(1.001) ≈ 0.9995 → ground-folded to Real literal.
    let kernel = layer_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 1.0, 0.001, 2.0, 0.5], bounds(-10.0, 10.0)).unwrap();

    // rsqrt is ground-folded (#376) — no UF declaration.
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded rsqrt should NOT appear as UF. SMT2:\n{smt2}"
    );

    // x is symbolic; mean=1.0 is constant. SMT2 should encode (- x 1.0).
    assert!(
        smt2.contains("(- x 1.0)"),
        "layer_norm SMT2 should contain (- x 1.0) for x-mean subtraction. SMT2:\n{smt2}"
    );

    // gamma=2.0 must appear in the encoding (multiplication with gamma).
    assert!(
        smt2.contains(" 2.0)"),
        "layer_norm SMT2 should encode constant gamma=2.0. SMT2:\n{smt2}"
    );

    // Count assertions: input bounds (2) + output violation (1) + div guard(s).
    let assert_count = smt2.matches("(assert").count();
    assert!(
        assert_count >= 3,
        "expected >= 3 assertions (input bounds + violation), got {assert_count}. SMT2:\n{smt2}"
    );
}

// --- K2 InstanceNorm scalar ay SMT tests ---

/// Helper: build InstanceNorm (K2) scalar element kernel.
fn instance_norm_kernel() -> KernelDef {
    nn_dsl::build_instance_norm_scalar_kernel().expect("instance_norm_scalar kernel must build")
}

#[test]
fn test_verify_instance_norm_scalar_translates() {
    // instance_norm_scalar(x, mean, var_val, eps)
    //   = (x - mean) * rsqrt(var_val + eps)
    // Variable-first (#448): param 0 (x) is symbolic, params 1..3 = Constant.
    // constant_params = [mean=1.0, var_val=0.0, eps=1.0].
    // rsqrt(var_val + eps) = rsqrt(0.0 + 1.0) = 1.0 → ground-folded (#376).
    // Identity elimination: (x-1.0)*1.0 → (x-1.0), no real_mul emitted.
    // Yields Exact encoding → ay direct execution attempted.
    let kernel = instance_norm_kernel();
    let result = verify_kernel_smt(&kernel, &[1.0, 0.0, 1.0], bounds(0.001, 0.1)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    // Ground-folding (#376) + identity elimination (*1.0) → pure subtraction.
    // ay#5357 fixed: pure QF_LRA reaches Proven.
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "instance_norm with *1.0 eliminated must be Proven (pure subtraction), got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_instance_norm_scalar_smt2_output() {
    // With ground-folding (#376), rsqrt(constant) is folded to Real literal.
    let kernel = instance_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.0, 1.0], bounds(0.001, 0.1)).unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded instance_norm should NOT contain rsqrt_approx UF"
    );
}

#[test]
fn test_instance_norm_scalar_smt2_declares_variable_no_uf() {
    // instance_norm_scalar has 4 params. With 3 constant_params → 1 declare-const (x).
    // Ground-folding (#376) eliminates rsqrt UF → 0 declare-fun.
    let kernel = instance_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.0, 1.0], bounds(0.001, 0.1)).unwrap();
    let declare_const_count = smt2.matches("declare-const").count();
    assert_eq!(
        declare_const_count, 1,
        "expected 1 declare-const (symbolic x), got {declare_const_count}"
    );
    let declare_fun_count = smt2.matches("declare-fun").count();
    assert_eq!(
        declare_fun_count, 0,
        "expected 0 declare-fun (rsqrt ground-folded), got {declare_fun_count}"
    );
}

#[test]
fn test_instance_norm_scalar_nan_constant_params_rejected() {
    let kernel = instance_norm_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::NAN, 0.0, 1.0], bounds(0.001, 0.1)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_instance_norm_scalar_with_explicit_bounds() {
    // Ground-folded (#376): rsqrt(1.0) = 1.0 → identity elimination (*1.0 removed).
    // Resulting formula is pure subtraction: (x - 1.0).
    let kernel = instance_norm_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[1.0, 0.0, 1.0],
        bounds(0.001, 0.1),
        Some((-5.0, 5.0)),
    )
    .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    // After identity elimination + ay#5357: pure subtraction → Proven.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "instance_norm *1.0 eliminated → pure subtraction must be Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_instance_norm_scalar_smt2_encodes_subtraction_folded_rsqrt() {
    // instance_norm_scalar(x, mean, var_val, eps)
    //   = (x - mean) * rsqrt(var_val + eps)
    // Variable-first convention (#448): param 0 (x) is symbolic, params 1..3 = Constant.
    // constant_params = [mean=1.0, var_val=1.0, eps=0.001].
    // rsqrt(1.0 + 0.001) ≈ 0.9995 → ground-folded to Real literal.
    let kernel = instance_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 1.0, 0.001], bounds(0.001, 0.1)).unwrap();

    // rsqrt is ground-folded (#376) — no UF declaration.
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded rsqrt should NOT appear as UF. SMT2:\n{smt2}"
    );

    // x is symbolic; mean=1.0 is constant. SMT2 should encode (- x 1.0).
    assert!(
        smt2.contains("(- x 1.0)"),
        "instance_norm SMT2 should contain (- x 1.0) for x-mean. SMT2:\n{smt2}"
    );
}

#[test]
fn test_instance_norm_scalar_zero_variance_translates() {
    // Zero variance (var_val=0.0): formula becomes (x - mean) * rsqrt(0 + eps).
    // Variable-first convention (#448): param 0 (x) is symbolic, params 1..3 = Constant.
    // constant_params = [mean=2.0, var_val=0.0, eps=0.001].
    // rsqrt(0.0 + 0.001) = rsqrt(0.001) ≈ 31.62 → ground-folded (#376).
    //
    // Analytical bounds are widened by SMT_QUANTIZATION_MARGIN (#539) to account
    // for real_from_f64 encoding error in ground-folded rsqrt(0.001) constant.
    // ay#5605 fixed: real_mul with fractional coefficients now works.
    let kernel = instance_norm_kernel();
    let result = verify_kernel_smt(&kernel, &[2.0, 0.0, 0.001], bounds(0.001, 0.1)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: zero-variance instance_norm must reach Proven, got: {:?}",
        result.outcome,
    );
    // x is symbolic; mean=2.0 is constant. SMT2 should encode (- x 2.0).
    let smt2 = kernel_to_smt2(&kernel, &[2.0, 0.0, 0.001], bounds(0.001, 0.1)).unwrap();
    assert!(
        smt2.contains("(- x 2.0)"),
        "zero-variance SMT2 should contain (- x 2.0) for x-mean. SMT2:\n{smt2}"
    );
}

#[test]
fn test_instance_norm_scalar_inf_constant_param_rejected() {
    let kernel = instance_norm_kernel();
    // Positive infinity at x position.
    let err =
        verify_kernel_smt(&kernel, &[f32::INFINITY, 0.0, 1.0], bounds(0.001, 0.1)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "+INF constant param should be rejected, got: {msg}"
    );
    // Negative infinity at var_val position.
    let err =
        verify_kernel_smt(&kernel, &[1.0, 0.0, f32::NEG_INFINITY], bounds(0.001, 0.1)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "-INF constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_instance_norm_scalar_nan_mean_and_var_rejected() {
    let kernel = instance_norm_kernel();
    // NaN at constant_params index 1 (mean).
    let err = verify_kernel_smt(&kernel, &[1.0, f32::NAN, 1.0], bounds(0.001, 0.1)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN at mean param should be rejected, got: {msg}"
    );
    // NaN at constant_params index 2 (var_val).
    let err = verify_kernel_smt(&kernel, &[1.0, 0.0, f32::NAN], bounds(0.001, 0.1)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN at var_val param should be rejected, got: {msg}"
    );
}

// --- K2 InstanceNorm+Affine scalar ay SMT tests ---

/// Helper: build InstanceNorm+Affine (K2) scalar element kernel.
fn instance_norm_affine_kernel() -> KernelDef {
    nn_dsl::build_instance_norm_affine_scalar_kernel()
        .expect("instance_norm_affine_scalar kernel must build")
}

#[test]
fn test_verify_instance_norm_affine_scalar_translates() {
    // instance_norm_affine_scalar(x, mean, var_val, eps, gamma, beta)
    //   = (x - mean) * rsqrt(var_val + eps) * gamma + beta
    // Variable-first (#448): param 0 (x) is symbolic, params 1..5 = Constant.
    // constant_params = [mean=0.0, var=1.0, eps=0.001, gamma=1.0, beta=0.0].
    // rsqrt(1.0 + 0.001) on constants → ground-folded (#376).
    // Yields Exact encoding → ay-direct execution.
    let kernel = instance_norm_affine_kernel();
    let result =
        verify_kernel_smt(&kernel, &[0.0, 1.0, 0.001, 1.0, 0.0], bounds(-5.0, 5.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    // Ground-folding (#376) + linearity detection ensures ay-direct.
    // ay#5605 fixed: real_mul with fractional rsqrt(1.001) ≈ 0.9995 now works.
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: instance_norm_affine must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_instance_norm_affine_scalar_with_explicit_bounds() {
    // instance_norm_affine_scalar(x, mean=0.0, var=1.0, eps=0.001, gamma=2.0, beta=1.0)
    // rsqrt(1.001) ≈ 0.9995 → ground-folded. kernel ≈ 2*0.9995*x + 1 ≈ 2*x + 1.
    // x ∈ [-5, 5] → output ≈ [-9, 11]. Use wide bounds [-15, 15].
    let kernel = instance_norm_affine_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[0.0, 1.0, 0.001, 2.0, 1.0],
        bounds(-5.0, 5.0),
        Some((-15.0, 15.0)),
    )
    .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    // ay#5605 fixed: real_mul with fractional rsqrt(1.001), gamma=2.0 now works.
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: instance_norm_affine with explicit bounds must reach Proven, got: {:?}",
        result.outcome,
    );
}

// --- #381: IEEE 754 NaN comparison bypass guards ---

#[test]
fn test_instance_norm_output_bounds_negative_var_eps_returns_error() {
    // #459: instance_norm_output_bounds(mean=0.0, var=-1.0, eps=-2.0, x_lo=-1.0, x_hi=1.0)
    // var + eps = -1.0 + (-2.0) = -3.0 ≤ 0 → sqrt undefined.
    // Pre-guard must catch this and return error, not NaN bounds.
    let result = instance_norm_output_bounds(0.0, -1.0, -2.0, -1.0, 1.0);
    assert!(
        result.is_err(),
        "negative var+eps should return error, got: {result:?}"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("non-finite output bounds"),
        "error should mention non-finite bounds, got: {msg}"
    );
}

#[test]
fn test_norm_affine_output_bounds_negative_var_eps_returns_error() {
    // #459: norm_affine_output_bounds(mean=0.0, var=-1.0, eps=-2.0, gamma=1.0, beta=0.0, x_lo=-1.0, x_hi=1.0)
    // var + eps = -1.0 + (-2.0) = -3.0 ≤ 0 → sqrt undefined.
    // Pre-guard must catch this and return error.
    let result = norm_affine_output_bounds(0.0, -1.0, -2.0, 1.0, 0.0, -1.0, 1.0);
    assert!(
        result.is_err(),
        "negative var+eps should return error, got: {result:?}"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("non-finite output bounds"),
        "error should mention non-finite bounds, got: {msg}"
    );
}
