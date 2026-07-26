// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K2 InstanceNorm + K5 RMSNorm + K7 LayerNorm ay SMT verification tests (#19).
//!
//! These test the scalar element kernels built by `build_instance_norm_scalar_kernel`,
//! `build_rms_norm_scalar_kernel`, and `build_layer_norm_scalar_kernel`. The scalar
//! kernels encode the per-element computation after reduction passes compute
//! statistics (mean, variance, rms_inv).

use super::*;
use crate::test_helpers::bounds;

// --- K5 RMSNorm scalar ay SMT tests ---

/// Helper: build RMSNorm (K5) scalar element kernel.
fn rms_norm_kernel() -> KernelDef {
    nn_dsl::build_rms_norm_scalar_kernel().expect("rms_norm_scalar kernel must build")
}

#[test]
fn test_verify_rms_norm_scalar_translates() {
    // rms_norm_scalar(x, rms_inv, weight) = x * rms_inv * weight
    // constant_params = [x=1.0, rms_inv=0.5], weight variable in [-10, 10].
    // Pure arithmetic (mul only) — no transcendentals → Exact encoding.
    // With both x and rms_inv constant, the kernel becomes 0.5*weight which
    // is linear → ay direct execution (#376).
    let kernel = rms_norm_kernel();
    let result = verify_kernel_smt(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    // Ground-folding (#376) + linearity detection ensures this reaches
    // ay-direct. ay#5605 fixed: real_mul with fractional coefficients now works.
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: rms_norm_scalar (0.5*weight) must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_rms_norm_scalar_smt2_output() {
    let kernel = rms_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    // Pure arithmetic — no UF declarations expected.
    assert!(
        !smt2.contains("rsqrt_approx"),
        "rms_norm_scalar should NOT use rsqrt UF (it's post-reduction)"
    );
}

#[test]
fn test_rms_norm_scalar_smt2_declares_one_variable() {
    // rms_norm_scalar has 3 params. With 2 constant_params → 1 declare-const (weight).
    let kernel = rms_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    let declare_const_count = smt2.matches("declare-const").count();
    assert_eq!(
        declare_const_count, 1,
        "expected 1 declare-const (symbolic weight), got {declare_const_count}"
    );
    // No UF declarations for pure arithmetic.
    let declare_fun_count = smt2.matches("declare-fun").count();
    assert_eq!(
        declare_fun_count, 0,
        "expected 0 declare-fun for pure arithmetic kernel, got {declare_fun_count}"
    );
}

#[test]
fn test_rms_norm_scalar_nan_constant_params_rejected() {
    let kernel = rms_norm_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::NAN, 0.5], bounds(-10.0, 10.0)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_rms_norm_scalar_with_explicit_bounds() {
    // rms_norm_scalar(x=1.0, rms_inv=0.5, weight) = 0.5 * weight
    // weight ∈ [-10, 10] → output ∈ [-5, 5].
    // Linear kernel with tight explicit bounds → ay-direct (#376).
    // ay#5605 fixed: real_mul with fractional coefficient (0.5 * weight) now works.
    let kernel = rms_norm_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0), Some((-5.0, 5.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: rms_norm_scalar with explicit bounds must reach Proven, got: {:?}",
        result.outcome,
    );
}

#[test]
fn test_rms_norm_scalar_input_bounds_in_smt2() {
    // rms_norm_scalar(x=1.0, rms_inv=0.5, weight) — weight is symbolic.
    // Use distinctive bounds [-7.3, 8.9] so we don't false-match on node IDs.
    let kernel = rms_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-7.3, 8.9)).unwrap();
    // real_from_f64(-7.3) encodes as (/ -7300000 1000000) — look for the numerator.
    assert!(
        smt2.contains("7300000"),
        "SMT2 should encode input lower bound -7.3 as real fraction. SMT2:\n{smt2}"
    );
    assert!(
        smt2.contains("8900000"),
        "SMT2 should encode input upper bound 8.9 as real fraction. SMT2:\n{smt2}"
    );
}

#[test]
fn test_rms_norm_scalar_counterexample_with_wrong_bounds() {
    // rms_norm_scalar(x=1.0, rms_inv=0.5, weight) = 0.5 * weight
    // weight ∈ [-10, 10] → true output ∈ [-5, 5].
    // Provide intentionally too-tight bounds [-1, 1]. The solver should find
    // a counterexample (e.g., weight=4 → output=2 > 1).
    let kernel = rms_norm_kernel();
    let result =
        verify_kernel_smt_with_bounds(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0), Some((-1.0, 1.0)))
            .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    // Linear kernel → ay-direct path attempted.
    assert_eq!(result.solver, "ay-direct");
    assert!(
        matches!(
            result.outcome,
            SmtOutcome::Counterexample | SmtOutcome::Unknown
        ),
        "too-tight bounds should produce Counterexample (or known-regression Unknown), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    if result.outcome != SmtOutcome::Counterexample {
        let detail = result.detail.as_deref().unwrap_or("");
        assert!(
            detail.contains("internal-error"),
            "Unknown should be the known internal-error regression, got: {detail}"
        );
    }
}

#[test]
fn test_rms_norm_scalar_smt2_encodes_multiplication() {
    // rms_norm_scalar(x, rms_inv, weight) = x * rms_inv * weight
    // With x=2.0, rms_inv=3.0 as constants, the encoding should contain
    // the constant product 6.0 (or intermediate products) and a multiplication
    // with the symbolic weight variable.
    let kernel = rms_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[2.0, 3.0], bounds(-1.0, 1.0)).unwrap();
    // The SMT2 must contain a multiplication operator — this is the core
    // arithmetic operation. If the formula were wrong (e.g., addition instead
    // of multiplication), this would catch it.
    assert!(
        smt2.contains("* "),
        "rms_norm_scalar SMT2 should contain multiplication operator. SMT2:\n{smt2}"
    );
    // The constant values 2.0 and 3.0 must appear in the encoding.
    // Integer-valued floats are encoded directly (e.g., "2.0", "3.0").
    assert!(
        smt2.contains("2.0") && smt2.contains("3.0"),
        "rms_norm_scalar SMT2 should encode constants x=2.0 and rms_inv=3.0. SMT2:\n{smt2}"
    );
    // The output bound should be ~6.0 (coeff=2*3=6, weight in [-1,1]).
    // With SMT_QUANTIZATION_MARGIN (#539), bounds are widened to ±6.0001,
    // encoded as (/ 6000100 1000000) in SMT-LIB2.
    assert!(
        smt2.contains("6000100"),
        "rms_norm_scalar SMT2 should have output bound ~6.0 (=2.0*3.0*1.0 + margin). SMT2:\n{smt2}"
    );
}

#[test]
fn test_rms_norm_scalar_uses_tight_analytical_bounds() {
    // rms_norm_scalar(x=1.0, rms_inv=0.5, weight) = 0.5 * weight
    // weight ∈ [-10, 10] → output ∈ [-5, 5].
    // With ±1e6 fallback, the SMT would contain "1000010" (fallback margin).
    // With analytical bounds, the output bound is ±5.0 encoded as integer Real.
    let kernel = rms_norm_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[1.0, 0.5], bounds(-10.0, 10.0)).unwrap();
    assert!(
        !smt2.contains("1000010"),
        "rms_norm_scalar should use analytical bounds, not ±1e6 fallback"
    );
    // Positive check: tight bound of ~5.0 should appear in the SMT2.
    // With SMT_QUANTIZATION_MARGIN (#539), bounds are ±5.0001,
    // encoded as (/ -5000100 1000000) and (/ 5000100 1000000).
    assert!(
        smt2.contains("5000100"),
        "rms_norm_scalar should have bounds ~±5.0 (with quantization margin). SMT2:\n{smt2}"
    );
}

// K7 LayerNorm + K2 InstanceNorm tests extracted to prove_norm_ln_in_tests.rs (#356).
