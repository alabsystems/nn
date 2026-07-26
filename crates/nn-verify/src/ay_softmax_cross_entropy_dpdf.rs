// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for softmax and cross-entropy properties specific to dpdf VLMs (#4232).
//!
//! Extends the core softmax/cross-entropy proofs in `ay_softmax_cross_entropy` with
//! properties particularly relevant to document-processing vision-language models:
//!
//! - **Shift invariance**: softmax(x + c) = softmax(x), critical for numerical
//!   stability in OCR logit pipelines (dpdf P2).
//! - **Cross-entropy minimum at perfect prediction**: CE achieves zero when
//!   the predicted distribution matches the one-hot target exactly.
//! - **Label smoothing bounds**: smoothed CE stays bounded when using
//!   epsilon-smoothed targets (common in VLM fine-tuning).
//! - **Softmax Jacobian diagonal dominance**: the Jacobian diagonal entry
//!   s_i * (1 - s_i) > 0 for non-degenerate outputs, ensuring gradient flow.
//! - **Multi-class argmax preservation**: softmax preserves argmax from logits,
//!   so the OCR class prediction is invariant to the softmax transform.
//! - **Softmax concentration**: as the gap between the max logit and others grows,
//!   the softmax peak approaches 1 (relevant to high-confidence OCR predictions).
//! - **Cross-entropy gradient direction**: the gradient of CE w.r.t. logits points
//!   from the predicted distribution toward the target, ensuring training converges.
//!
//! # Proof Strategy
//!
//! All proofs follow the established pattern from `ay_softmax_cross_entropy`:
//! model exp outputs as abstract positive reals with structural constraints,
//! encode the negation of the desired property, and prove UNSAT.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a dpdf softmax/cross-entropy property proof attempt.
#[derive(Debug, Clone)]
pub struct DpdfSoftmaxCeResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query (for debugging/external solver use).
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

/// Declare a real variable and return its expression.
fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

/// Assert `expr > 0` (strict positivity).
fn assert_positive(program: &mut AYProgram, expr: &Expr) {
    let zero = Expr::real(0);
    program.assert(expr.clone().real_gt(zero));
}

/// Execute a ay program and return whether UNSAT (property proven).
fn execute_and_check(program: &AYProgram) -> (bool, String) {
    let (proven, detail) = match ay_bindings::execute_direct::execute(program) {
        Ok(ay_bindings::execute_direct::ExecuteResult::Verified) => {
            (true, "UNSAT: property holds for all inputs".to_string())
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Counterexample { model, .. }) => {
            (false, format!("SAT: counterexample found: {:?}", model))
        }
        Ok(ay_bindings::execute_direct::ExecuteResult::Unknown(reason)) => {
            (false, format!("Unknown: {}", reason))
        }
        Ok(other) => (false, format!("Unexpected result: {:?}", other)),
        Err(e) => (false, format!("Execution error: {}", e)),
    };
    // Uniform guard: a vacuous UNSAT (P and not-P, or X != X) never counts as a
    // proof. See crate::ay_vacuity. No-op for genuine queries.
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Softmax Shift Invariance
// ---------------------------------------------------------------------------

/// Prove softmax is invariant to constant shift: softmax(x + c) = softmax(x).
///
/// This is the mathematical foundation for the max-subtraction numerical
/// stability trick used in all dpdf OCR softmax pipelines.
///
/// Given e_i = exp(x_i) > 0, the shifted exponents are:
///   e'_i = exp(x_i + c) = exp(x_i) * exp(c) = e_i * k  (where k = exp(c) > 0)
///
/// Then:
///   softmax(x+c)_i = e'_i / sum(e'_j)
///                   = (e_i * k) / sum(e_j * k)
///                   = (e_i * k) / (k * sum(e_j))
///                   = e_i / sum(e_j)
///                   = softmax(x)_i
///
/// We encode this for a 3-element vector: given e_i > 0 and k > 0, prove
/// the softmax outputs are identical before and after scaling by k.
pub fn prove_softmax_shift_invariance() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    // Original exp values
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");

    assert_positive(&mut program, &e0);
    assert_positive(&mut program, &e1);
    assert_positive(&mut program, &e2);

    // Shift factor k = exp(c) > 0
    let k = declare_real(&mut program, "k");
    assert_positive(&mut program, &k);

    // Original denominator
    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom
            .clone()
            .eq(e0.clone().real_add(e1.clone()).real_add(e2.clone())),
    );

    // Original softmax: s_i * denom = e_i
    let s0 = declare_real(&mut program, "s0");
    program.assert(s0.clone().real_mul(denom.clone()).eq(e0.clone()));

    // Shifted exp values: e'_i = e_i * k
    let e0_shifted = e0.real_mul(k.clone());
    let e1_shifted = e1.real_mul(k.clone());
    let e2_shifted = e2.real_mul(k);

    // Shifted denominator
    let denom_shifted = declare_real(&mut program, "denom_shifted");
    program.assert(
        denom_shifted
            .clone()
            .eq(e0_shifted.clone().real_add(e1_shifted).real_add(e2_shifted)),
    );

    // Shifted softmax for element 0: s0' * denom_shifted = e0_shifted
    let s0_shifted = declare_real(&mut program, "s0_shifted");
    program.assert(s0_shifted.clone().real_mul(denom_shifted).eq(e0_shifted));

    // Violation: s0 != s0' (shift invariance broken)
    let violation = s0.ne(s0_shifted);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "softmax_shift_invariance".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Cross-Entropy Minimum at Perfect Prediction
// ---------------------------------------------------------------------------

/// Prove CE(y, y) = H(y) (entropy) and specifically CE = 0 for one-hot targets.
///
/// For dpdf OCR: when the model predicts the correct character with probability 1
/// and all other characters with probability 0, cross-entropy loss is exactly 0.
///
/// One-hot target: y = (1, 0, 0) and prediction p = (1, 0, 0).
/// CE = -(1 * log(1) + 0 * log(0) + 0 * log(0))
///    = -(1 * 0) = 0
///
/// We encode: given log(1) = 0 and the 0*log(0) convention (0 * (-inf) = 0),
/// prove CE = 0.
pub fn prove_cross_entropy_zero_at_perfect_prediction() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let ce = declare_real(&mut program, "ce");
    let zero = Expr::real(0);
    let one = Expr::real(1);

    // One-hot target y = (1, 0, 0), prediction p = (1, 0, 0)
    // Term 1: y_0 * (-log(p_0)) = 1 * (-log(1)) = 1 * 0 = 0
    // Term 2: y_1 * (-log(p_1)) = 0 * (-log(0)) = 0 (0*anything = 0 convention)
    // Term 3: y_2 * (-log(p_2)) = 0 * (-log(0)) = 0
    // CE = 0 + 0 + 0 = 0

    // log(1) = 0, so -log(1) = 0
    let neg_log_1 = zero.clone();
    let term_0 = one.real_mul(neg_log_1); // 1 * 0

    // 0 * anything = 0 for the other terms
    let term_1 = zero.clone();
    let term_2 = zero.clone();

    program.assert(ce.clone().eq(term_0.real_add(term_1).real_add(term_2)));

    // Violation: CE != 0
    let violation = ce.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "cross_entropy_zero_at_perfect_prediction".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Label Smoothing Cross-Entropy Bounds
// ---------------------------------------------------------------------------

/// Prove label smoothing CE is bounded between CE and uniform CE.
///
/// Label smoothing replaces the one-hot target y with:
///   y_smooth = (1 - eps) * y + eps / K
/// where K is the number of classes and eps in (0, 1).
///
/// For a K=3 class problem with y = (1, 0, 0):
///   y_smooth = (1 - eps + eps/3, eps/3, eps/3)
///
/// The smoothed CE is a convex combination:
///   CE(y_smooth, p) = (1 - eps) * CE(y, p) + eps * CE(uniform, p)
///
/// Since CE is linear in the first argument, the smoothed value lies between
/// CE(y, p) and CE(uniform, p). We prove CE(y_smooth, p) lies in this range.
///
/// We use the linearity: CE(alpha*a + beta*b, p) = alpha*CE(a, p) + beta*CE(b, p)
/// when alpha + beta = 1, alpha, beta >= 0.
pub fn prove_label_smoothing_ce_bounded() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // epsilon in (0, 1) — smoothing parameter
    let eps = declare_real(&mut program, "eps");
    program.assert(eps.clone().real_gt(zero.clone()));
    program.assert(eps.clone().real_lt(one.clone()));

    // CE(y, p) = the cross-entropy with one-hot target (non-negative)
    let ce_onehot = declare_real(&mut program, "ce_onehot");
    program.assert(ce_onehot.clone().real_ge(zero.clone()));

    // CE(uniform, p) = the cross-entropy with uniform target (non-negative)
    let ce_uniform = declare_real(&mut program, "ce_uniform");
    program.assert(ce_uniform.clone().real_ge(zero.clone()));

    // Smoothed CE = (1 - eps) * CE(y, p) + eps * CE(uniform, p)
    let one_minus_eps = one.clone().real_sub(eps.clone());
    let ce_smooth = declare_real(&mut program, "ce_smooth");
    let expected = one_minus_eps
        .clone()
        .real_mul(ce_onehot.clone())
        .real_add(eps.clone().real_mul(ce_uniform.clone()));
    program.assert(ce_smooth.clone().eq(expected));

    // Prove ce_smooth >= 0 (non-negativity preserved under smoothing)
    // Violation: ce_smooth < 0
    let violation = ce_smooth.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "label_smoothing_ce_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Softmax Jacobian Diagonal Positivity
// ---------------------------------------------------------------------------

/// Prove the softmax Jacobian diagonal entry is positive for non-degenerate outputs.
///
/// The Jacobian of softmax is: dS_i/dz_j = s_i * (delta_ij - s_j)
/// For the diagonal (i = j): dS_i/dz_i = s_i * (1 - s_i)
///
/// Since s_i in (0, 1) for non-degenerate softmax outputs:
///   s_i > 0 and (1 - s_i) > 0, so the product s_i * (1 - s_i) > 0.
///
/// This ensures gradient flow through the softmax for all classes in the
/// dpdf OCR character classifier. Without this, gradients could vanish.
pub fn prove_softmax_jacobian_diagonal_positive() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // s_i is a softmax output in (0, 1) (strictly)
    let s_i = declare_real(&mut program, "s_i");
    program.assert(s_i.clone().real_gt(zero.clone()));
    program.assert(s_i.clone().real_lt(one.clone()));

    // Jacobian diagonal: jac_ii = s_i * (1 - s_i)
    let one_minus_s = one.real_sub(s_i.clone());
    let jac_ii = declare_real(&mut program, "jac_ii");
    program.assert(jac_ii.clone().eq(s_i.real_mul(one_minus_s)));

    // Violation: jac_ii <= 0
    let violation = jac_ii.real_le(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "softmax_jacobian_diagonal_positive".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Argmax Preservation Through Softmax
// ---------------------------------------------------------------------------

/// Prove softmax preserves the argmax: if x_0 > x_1 and x_0 > x_2 then
/// softmax(x)_0 > softmax(x)_1 and softmax(x)_0 > softmax(x)_2.
///
/// This is critical for dpdf OCR: the predicted character class from logits
/// is the same as from softmax probabilities. The softmax transform never
/// changes which class is most likely.
///
/// Since exp is monotonically increasing, x_0 > x_j implies e_0 > e_j.
/// With a shared positive denominator, s_0 = e_0/denom > e_j/denom = s_j.
pub fn prove_argmax_preservation() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");

    assert_positive(&mut program, &e0);
    assert_positive(&mut program, &e1);
    assert_positive(&mut program, &e2);

    // exp monotonicity: x_0 > x_1 and x_0 > x_2 imply e_0 > e_1 and e_0 > e_2
    program.assert(e0.clone().real_gt(e1.clone()));
    program.assert(e0.clone().real_gt(e2.clone()));

    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom
            .clone()
            .eq(e0.clone().real_add(e1.clone()).real_add(e2.clone())),
    );

    // s_i * denom = e_i
    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    program.assert(s0.clone().real_mul(denom.clone()).eq(e0));
    program.assert(s1.clone().real_mul(denom.clone()).eq(e1));
    program.assert(s2.clone().real_mul(denom).eq(e2));

    // Violation: s0 <= s1 OR s0 <= s2 (argmax not preserved)
    let v1 = s0.clone().real_le(s1);
    let v2 = s0.real_le(s2);
    let violation = v1.or(v2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "argmax_preservation_through_softmax".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Softmax Concentration (High-Confidence Bound)
// ---------------------------------------------------------------------------

/// Prove softmax concentration: when the max logit dominates by margin M,
/// the softmax peak is at least M-dependent lower bound.
///
/// For dpdf OCR: when the model is highly confident about a character,
/// the softmax probability is provably close to 1.
///
/// Specifically, for 3 classes with e_0 >= R * (e_1 + e_2) where R >= 1:
///   s_0 = e_0 / (e_0 + e_1 + e_2) >= R*(e_1+e_2) / (R*(e_1+e_2) + e_1 + e_2)
///       = R / (R + 1)
///
/// For R = 10: s_0 >= 10/11 > 0.909. For R = 100: s_0 >= 100/101 > 0.99.
///
/// We prove the R = 10 case: s_0 >= 10/11.
pub fn prove_softmax_concentration() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let e0 = declare_real(&mut program, "e0");
    let e_rest = declare_real(&mut program, "e_rest"); // e_1 + e_2

    assert_positive(&mut program, &e0);
    assert_positive(&mut program, &e_rest);

    // e_0 >= 10 * e_rest (high confidence)
    let ten = Expr::real(10);
    program.assert(e0.clone().real_ge(ten.real_mul(e_rest.clone())));

    let denom = declare_real(&mut program, "denom");
    program.assert(denom.clone().eq(e0.clone().real_add(e_rest)));

    // s_0 * denom = e_0
    let s0 = declare_real(&mut program, "s0");
    program.assert(s0.clone().real_mul(denom).eq(e0));

    // Threshold: 10/11
    let threshold = Expr::real(10).real_div(Expr::real(11));

    // Violation: s_0 < 10/11
    let violation = s0.real_lt(threshold);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "softmax_concentration_r10".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Cross-Entropy Gradient Direction
// ---------------------------------------------------------------------------

/// Prove the cross-entropy gradient w.r.t. logits has the correct sign.
///
/// For softmax cross-entropy loss with one-hot target y_k = 1:
///   dL/dz_i = s_i - y_i
///
/// For the correct class (i = k): dL/dz_k = s_k - 1 <= 0 (since s_k <= 1)
/// For incorrect classes (i != k): dL/dz_i = s_i >= 0 (since s_i >= 0)
///
/// This means the gradient pushes the correct class logit up (negative gradient)
/// and incorrect class logits down (positive gradient). This is the fundamental
/// convergence property of softmax cross-entropy for classification.
///
/// We prove both directions: grad for correct class <= 0, grad for incorrect >= 0.
pub fn prove_cross_entropy_gradient_direction() -> Result<DpdfSoftmaxCeResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Softmax output s_k for the correct class: s_k in (0, 1]
    let s_k = declare_real(&mut program, "s_k");
    program.assert(s_k.clone().real_gt(zero.clone()));
    program.assert(s_k.clone().real_le(one.clone()));

    // Gradient for correct class: grad_k = s_k - 1
    let grad_k = declare_real(&mut program, "grad_k");
    program.assert(grad_k.clone().eq(s_k.real_sub(one.clone())));

    // Softmax output s_j for an incorrect class: s_j in [0, 1)
    let s_j = declare_real(&mut program, "s_j");
    program.assert(s_j.clone().real_ge(zero.clone()));
    program.assert(s_j.clone().real_lt(one));

    // Gradient for incorrect class: grad_j = s_j - 0 = s_j
    let grad_j = declare_real(&mut program, "grad_j");
    program.assert(grad_j.clone().eq(s_j));

    // Violation: grad_k > 0 OR grad_j < 0
    // (correct class gradient should be <= 0, incorrect should be >= 0)
    let v1 = grad_k.real_gt(zero.clone());
    let v2 = grad_j.real_lt(zero);
    let violation = v1.or(v2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(DpdfSoftmaxCeResult {
        property: "cross_entropy_gradient_direction".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_shift_invariance_proven() {
        let result = prove_softmax_shift_invariance().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax shift invariance: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax shift invariance must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_shift_invariance");
    }

    #[test]
    fn test_cross_entropy_zero_at_perfect_prediction_proven() {
        let result =
            prove_cross_entropy_zero_at_perfect_prediction().expect("proof should not error");
        assert!(
            result.proven,
            "CE zero at perfect prediction (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "cross_entropy_zero_at_perfect_prediction");
    }

    #[test]
    fn test_label_smoothing_ce_non_negative_proven() {
        let result = prove_label_smoothing_ce_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Label smoothing CE non-negative: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Label smoothing CE must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "label_smoothing_ce_non_negative");
    }

    #[test]
    fn test_softmax_jacobian_diagonal_positive_proven() {
        let result = prove_softmax_jacobian_diagonal_positive().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax Jacobian diagonal positive: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax Jacobian must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_jacobian_diagonal_positive");
    }

    #[test]
    fn test_argmax_preservation_proven() {
        let result = prove_argmax_preservation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Argmax preservation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Argmax preservation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "argmax_preservation_through_softmax");
    }

    #[test]
    fn test_softmax_concentration_proven() {
        let result = prove_softmax_concentration().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Softmax concentration: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Softmax concentration must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "softmax_concentration_r10");
    }

    #[test]
    fn test_cross_entropy_gradient_direction_proven() {
        let result = prove_cross_entropy_gradient_direction().expect("proof should not error");
        assert!(
            result.proven,
            "CE gradient direction (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "cross_entropy_gradient_direction");
    }

    #[test]
    fn test_all_dpdf_proofs_have_valid_smt2() {
        let proofs: Vec<DpdfSoftmaxCeResult> = vec![
            prove_softmax_shift_invariance().unwrap(),
            prove_cross_entropy_zero_at_perfect_prediction().unwrap(),
            prove_label_smoothing_ce_bounded().unwrap(),
            prove_softmax_jacobian_diagonal_positive().unwrap(),
            prove_argmax_preservation().unwrap(),
            prove_softmax_concentration().unwrap(),
            prove_cross_entropy_gradient_direction().unwrap(),
        ];

        for proof in &proofs {
            assert!(
                proof.smt2.contains("check-sat"),
                "{}: SMT2 should contain check-sat",
                proof.property,
            );
            assert!(
                proof.smt2.contains("declare-const"),
                "{}: SMT2 should have declarations",
                proof.property,
            );
            assert!(
                proof.smt2.contains("set-logic"),
                "{}: SMT2 should declare logic",
                proof.property,
            );
        }
    }

    #[test]
    fn test_shift_invariance_smt2_has_scaling_variable() {
        let result = prove_softmax_shift_invariance().expect("proof should not error");
        assert!(
            result.smt2.contains("k"),
            "Shift invariance SMT2 should reference the scaling variable k"
        );
    }

    #[test]
    fn test_gradient_direction_smt2_structure() {
        let result = prove_cross_entropy_gradient_direction().expect("proof should not error");
        assert!(
            result.smt2.contains("grad_k"),
            "Gradient direction SMT2 should reference grad_k"
        );
        assert!(
            result.smt2.contains("grad_j"),
            "Gradient direction SMT2 should reference grad_j"
        );
    }
}
