// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K3 AdaIN + K4 AdaIN+Snake fused ay SMT verification tests (#19).

use super::*;
use crate::test_helpers::bounds;

/// Helper: build AdaIN (K3) scalar kernel.
fn adain_kernel() -> KernelDef {
    nn_dsl::build_adain_scalar_kernel().expect("adain kernel must build")
}

/// Helper: build AdaIN+Snake fused (K4) scalar kernel.
fn adain_snake_kernel() -> KernelDef {
    nn_dsl::build_adain_snake_fused_kernel().expect("adain_snake kernel must build")
}

// --- K3 AdaIN ay SMT tests ---

#[test]
fn test_verify_adain_translates() {
    // adain(x, mu, var_val, gamma, beta, eps) = gamma*(x-mu)*rsqrt(var+eps)+beta
    // Variable-first (#448): param 0 (x) is symbolic variable in [-1.0, 1.0].
    // constant_params = [mu=0.0, var_val=1.0, gamma=1.0, beta=0.0, eps=0.001].
    // rsqrt(var+eps) operates on constants → ground-folded (#376).
    // Yields Exact encoding → ay direct execution attempted.
    let kernel = adain_kernel();
    let result =
        verify_kernel_smt(&kernel, &[0.0, 1.0, 1.0, 0.0, 0.001], bounds(-1.0, 1.0)).unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.property, "output_bounded");
    // Ground-folding (#376) + identity elimination (*1.0 gamma, -0.0 mu, +0.0 beta).
    // ay#5605 fixed: real_mul with fractional rsqrt(1.001) ≈ 0.9995 now works.
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: adain must reach Proven, got: {:?}",
        result.outcome,
    );
    // #383: adain has analytical bounds → BoundsSource::Analytical.
    assert_eq!(
        result.bounds_source,
        BoundsSource::Analytical,
        "verify_kernel_smt (no explicit bounds) should use analytical bounds for adain"
    );
    assert!(
        result.expected_bounds.is_some(),
        "expected_bounds should be populated from analytical computation"
    );
}

#[test]
fn test_adain_smt2_output() {
    // With ground-folding (#376), rsqrt(constant) is folded to Real literal.
    let kernel = adain_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[0.0, 1.0, 1.0, 0.0, 0.001], bounds(-1.0, 1.0)).unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded adain should NOT contain rsqrt_approx UF"
    );
}

#[test]
fn test_adain_smt2_declares_variable_no_uf() {
    // adain has 6 params. With 5 constant_params, param 0 (x) is symbolic → 1 declare-const.
    // Ground-folding (#376) eliminates rsqrt UF → 0 declare-fun.
    let kernel = adain_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[0.0, 1.0, 1.0, 0.0, 0.001], bounds(-1.0, 1.0)).unwrap();
    let declare_const_count = smt2.matches("declare-const").count();
    assert_eq!(
        declare_const_count, 1,
        "expected 1 declare-const (symbolic x), got {declare_const_count}. SMT2:\n{smt2}"
    );
    let declare_fun_count = smt2.matches("declare-fun").count();
    assert_eq!(
        declare_fun_count, 0,
        "expected 0 declare-fun (rsqrt ground-folded), got {declare_fun_count}"
    );
}

#[test]
fn test_adain_nan_constant_params_rejected() {
    let kernel = adain_kernel();
    let err = verify_kernel_smt(&kernel, &[f32::NAN, 0.0, 1.0, 1.0, 0.0], bounds(0.001, 1.0))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_adain_with_explicit_bounds() {
    // Ground-folded (#376): rsqrt on constants → Exact encoding.
    // ay#5605 fixed: real_mul with fractional rsqrt(1.001) now works.
    let kernel = adain_kernel();
    let result = verify_kernel_smt_with_bounds(
        &kernel,
        &[0.0, 1.0, 1.0, 0.0, 0.001],
        bounds(-1.0, 1.0),
        Some((-5.0, 5.0)),
    )
    .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::Exact);
    assert_eq!(result.solver, "ay-direct");
    assert_eq!(
        result.outcome,
        SmtOutcome::Proven,
        "ay#5605 fixed: adain with explicit bounds must reach Proven, got: {:?}",
        result.outcome,
    );
    // #383: explicit bounds path → BoundsSource::CallerProvided.
    assert_eq!(
        result.bounds_source,
        BoundsSource::CallerProvided,
        "verify_kernel_smt_with_bounds(Some(...)) should use CallerProvided"
    );
    assert_eq!(
        result.expected_bounds,
        Some((-5.0, 5.0)),
        "expected_bounds should match the caller-provided values"
    );
}

#[test]
fn test_adain_smt2_encodes_subtraction_folded_rsqrt_mul_add() {
    // adain(x, mu, var_val, gamma, beta, eps) = gamma*(x-mu)*rsqrt(var+eps)+beta
    // Variable-first convention (#448): param 0 (x) is symbolic, params 1..5 = Constant.
    // constant_params = [mu=2.0, var_val=0.25, gamma=3.0, beta=0.7, eps=0.001].
    // rsqrt(0.25 + 0.001) ≈ 1.998 → ground-folded to Real literal (#376).
    let kernel = adain_kernel();
    let smt2 = kernel_to_smt2(&kernel, &[2.0, 0.25, 3.0, 0.7, 0.001], bounds(0.001, 1.0)).unwrap();

    // rsqrt is ground-folded — no UF.
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded rsqrt should NOT appear as UF. SMT2:\n{smt2}"
    );

    // mu=2.0, gamma=3.0 are integer-valued → encoded directly.
    assert!(
        smt2.contains("2.0"),
        "adain SMT2 should encode constant mu=2.0. SMT2:\n{smt2}"
    );
    assert!(
        smt2.contains("3.0"),
        "adain SMT2 should encode constant gamma=3.0. SMT2:\n{smt2}"
    );
    assert!(
        smt2.contains("700000"),
        "adain SMT2 should encode constant beta=0.7. SMT2:\n{smt2}"
    );

    // x is symbolic; mu=2.0 is constant. SMT2 should encode (- x 2.0) for (x - mu).
    assert!(
        smt2.contains("(- x 2.0)"),
        "adain SMT2 should contain (- x 2.0) for x-mu subtraction. SMT2:\n{smt2}"
    );

    // Must contain addition with beta.
    assert!(
        smt2.contains("(+ "),
        "adain SMT2 should contain addition operator (for + beta). SMT2:\n{smt2}"
    );

    // Must contain multiplication operator (gamma * ...).
    assert!(
        smt2.contains("(* "),
        "adain SMT2 should contain multiplication operator. SMT2:\n{smt2}"
    );

    // Must have input bounds + output violation + div guard assertions.
    let assert_count = smt2.matches("(assert").count();
    assert!(
        assert_count >= 3,
        "expected >= 3 assertions, got {assert_count}. SMT2:\n{smt2}"
    );
}

// --- K4 AdaIN+Snake fused ay SMT tests ---

#[test]
fn test_verify_adain_snake_translates() {
    // adain_snake(x, mu, var_val, gamma, beta, alpha, eps)
    // Variable-first (#448): param 0 (x) is symbolic in [-1.0, 1.0].
    // constant_params = [mu=0.0, var=1.0, gamma=1.0, beta=0.0, alpha=1.0, eps=0.001].
    // Ground-folding (#376): rsqrt(var+eps) on constants → folded to Real literal.
    // sin(alpha * adain_output) still symbolic → sin_approx UF → UfApprox → Unexecuted.
    let kernel = adain_snake_kernel();
    let result = verify_kernel_smt(
        &kernel,
        &[0.0, 1.0, 1.0, 0.0, 1.0, 0.001],
        bounds(-1.0, 1.0),
    )
    .unwrap();
    assert_eq!(result.encoding, SmtEncodingKind::UfApprox);
    assert_eq!(result.property, "output_bounded");
    // #2640: adain_snake (nonlinear + UF) now routes to ay NRA solver.
    assert_ne!(
        result.outcome,
        SmtOutcome::Unexecuted,
        "adain_snake should reach NRA solver (#2640), got: {:?} (detail: {:?})",
        result.outcome,
        result.detail,
    );
    // #383: adain_snake has analytical bounds → BoundsSource::Analytical.
    assert_eq!(
        result.bounds_source,
        BoundsSource::Analytical,
        "verify_kernel_smt (no explicit bounds) should use analytical bounds for adain_snake"
    );
}

#[test]
fn test_adain_snake_smt2_output() {
    let kernel = adain_snake_kernel();
    let smt2 = kernel_to_smt2(
        &kernel,
        &[0.0, 1.0, 1.0, 0.0, 1.0, 0.001],
        bounds(-1.0, 1.0),
    )
    .unwrap();
    assert!(smt2.contains("set-logic"));
    assert!(smt2.contains("check-sat"));
    // Ground-folding (#376): rsqrt on constants → folded, no UF.
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded rsqrt should NOT appear as UF in adain_snake"
    );
    // sin operates on symbolic arg → still UF.
    assert!(
        smt2.contains("sin_approx"),
        "adain_snake SMT should declare sin_approx UF"
    );
}

#[test]
fn test_adain_snake_smt2_declares_variable_and_sin_uf() {
    // adain_snake has 7 params. With 6 constant_params → 1 declare-const (x).
    // Ground-folding (#376): rsqrt on constants → folded. Only sin_approx UF remains.
    let kernel = adain_snake_kernel();
    let smt2 = kernel_to_smt2(
        &kernel,
        &[0.0, 1.0, 1.0, 0.0, 1.0, 0.001],
        bounds(-1.0, 1.0),
    )
    .unwrap();
    let declare_const_count = smt2.matches("declare-const").count();
    assert_eq!(
        declare_const_count, 1,
        "expected 1 declare-const (symbolic x), got {declare_const_count}"
    );
    // UF declarations: only sin_approx (rsqrt ground-folded).
    let declare_fun_count = smt2.matches("declare-fun").count();
    assert_eq!(
        declare_fun_count, 1,
        "expected 1 declare-fun (sin_approx only, rsqrt ground-folded), got {declare_fun_count}"
    );
}

#[test]
fn test_adain_snake_nan_constant_params_rejected() {
    let kernel = adain_snake_kernel();
    let err = verify_kernel_smt(
        &kernel,
        &[1.0, 0.0, f32::NAN, 1.0, 0.0, 1.0],
        bounds(0.001, 1.0),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite constant parameter"),
        "NaN constant param should be rejected, got: {msg}"
    );
}

#[test]
fn test_adain_snake_smt2_has_sin_invocations_rsqrt_folded() {
    // adain_snake fuses AdaIN + Snake: the formula applies snake activation
    // to the AdaIN output. Ground-folding (#376): rsqrt(var+eps) on constants
    // is folded to Real literal. sin(alpha*adain_output) operates on symbolic
    // arg → sin_approx UF is still present.
    // A regression that drops the Snake fusion would lose sin_approx.
    let kernel = adain_snake_kernel();
    let smt2 =
        kernel_to_smt2(&kernel, &[4.0, 1.0, 0.5, 2.0, 0.3, 1.5], bounds(0.001, 1.0)).unwrap();

    // rsqrt is ground-folded — should NOT appear as UF.
    assert!(
        !smt2.contains("rsqrt_approx"),
        "ground-folded rsqrt should NOT appear as UF. SMT2:\n{smt2}"
    );

    // sin_approx must be *invoked*, not just declared. Count applications:
    let sin_invocations = smt2.matches("sin_approx").count();
    assert!(
        sin_invocations >= 2,
        "expected >= 2 sin_approx occurrences (declare + invoke), got {sin_invocations}. SMT2:\n{smt2}"
    );

    // Snake uses `powi(sin(alpha*y), 2)` — the SMT encoding should contain
    // an exponentiation or self-multiplication representing sin²(alpha*y).
    // This is encoded as multiplication: (* (sin_approx ...) (sin_approx ...))
    // or via a let-binding.
    assert!(
        smt2.contains("(* "),
        "adain_snake SMT2 should contain multiplication (for gamma*..., sin²). SMT2:\n{smt2}"
    );

    // alpha=0.3 must appear in the encoding (used in snake: sin(alpha * y)).
    // 0.3 * 1_000_000 = 300_000 → encoded as 300000.
    assert!(
        smt2.contains("300000"),
        "adain_snake SMT2 should encode constant alpha=0.3. SMT2:\n{smt2}"
    );
}
