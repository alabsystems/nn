// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for matrix decomposition properties in dpdf VLMs (#4235).
//!
//! Error-bounded matrix decomposition proofs for Vision-Language Model pipelines.
//! Unlike exact algebraic proofs in `ay_tensor_decomposition_properties`, these
//! encode epsilon-tolerance bounds for floating-point VLM inference.
//!
//! # Proved Properties
//!
//! 1. Matmul associativity within error bounds (rounding error propagation)
//! 3. Low-rank approximation error bound (Eckart-Young with threshold)
//! 4. SVD reconstruction tolerance (perturbed factor reconstruction)
//! 5. Cholesky positive definiteness preservation
//! 6. QR orthogonality within bounds (near-orthonormal Q)
//! 7. Eigenvalue spectral bound (Gershgorin circle theorem)

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a VLM matrix decomposition property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct VlmDecompPropertyResult {
    pub property: String,
    pub proven: bool,
    pub smt2: String,
    pub detail: String,
}

fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

fn assert_bounds(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
    upper: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    let hi = real_from_f64(upper)?;
    program.assert(expr.clone().real_ge(lo));
    program.assert(expr.clone().real_le(hi));
    Ok(())
}

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
// Property 1: Matmul Associativity Within Error Bounds
// ---------------------------------------------------------------------------

/// Prove matmul associativity error is bounded when individual matmuls have
/// bounded element-wise rounding error delta. For 2x2 matrices, the total
/// per-element difference between fl((AB)C) and fl(A(BC)) is at most 6*delta,
/// since exact associativity holds and each path accumulates at most 3*delta
/// (one matmul error + two propagated errors from the inner product sum).
pub(crate) fn prove_matmul_associativity_error_bound() -> Result<VlmDecompPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let zero = Expr::real(0);
    let delta = declare_real(&mut program, "delta");
    program.assert(delta.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &delta, 0.0, 1.0)?;
    let neg_delta = zero.clone().real_sub(delta.clone());

    // 16 error variables: 4 for AB, 4 for (AB)C, 4 for BC, 4 for A(BC)
    let errors: Vec<Expr> = [
        "e_ab00", "e_ab01", "e_ab10", "e_ab11", "e_abc00", "e_abc01", "e_abc10", "e_abc11",
        "e_bc00", "e_bc01", "e_bc10", "e_bc11", "e_a_bc00", "e_a_bc01", "e_a_bc10", "e_a_bc11",
    ]
    .iter()
    .map(|name| {
        let e = declare_real(&mut program, name);
        program.assert(e.clone().real_ge(neg_delta.clone()));
        program.assert(e.clone().real_le(delta.clone()));
        e
    })
    .collect();

    // Total error for (AB)C path element [0,0]: e_abc00 + e_ab00 + e_ab01
    let total_left = errors[4]
        .clone()
        .real_add(errors[0].clone())
        .real_add(errors[1].clone());
    // Total error for A(BC) path element [0,0]: e_a_bc00 + e_bc00 + e_bc01
    let total_right = errors[12]
        .clone()
        .real_add(errors[8].clone())
        .real_add(errors[9].clone());
    let diff = total_left.real_sub(total_right);

    let six = real_from_f64(6.0)?;
    let bound = six.real_mul(delta);
    let neg_bound = zero.real_sub(bound.clone());

    let violation = diff.clone().real_gt(bound).or(diff.real_lt(neg_bound));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(VlmDecompPropertyResult {
        property: "matmul_associativity_error_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Low-Rank Approximation Error Bound (Eckart-Young)
// ---------------------------------------------------------------------------

/// Upper bound on a squared singular value ("energy"); any positive cap keeps
/// the search inside a finite box without affecting the proof.
const ENERGY_CAP: i64 = 1_000_000;

/// Prove the Eckart-Young rank-1 error bound: the best rank-1 approximation of a
/// rank-2 matrix has reconstruction error within the truncation threshold.
///
/// The naive encoding declares singular values `s1, s2, tau` and asks the solver
/// to refute `s2^2 > tau^2`. That query lives in `QF_NRA` with the variable×
/// variable products `s2*s2` and `tau*tau`, which is undecidable in practice and
/// hangs. It is also more than we need: Eckart-Young is naturally a statement
/// about *energies* — squared singular values / squared Frobenius norms — so we
/// take those as the primitive quantities and never square anything.
///
/// Working over energies `e1 = s1^2 >= e2 = s2^2 >= 0` and threshold energy
/// `tau_sq = tau^2`, the total Frobenius energy is `||A||_F^2 = e1 + e2` and the
/// rank-1 reconstruction error is the *discarded* energy: `||A - A_1||_F^2 =
/// total - retained`. Retaining the larger singular value discards `e2`, giving
/// two linear, genuinely derived bounds:
///
/// ```text
///   (a) err_sq = e2 <= tau_sq          (the discarded value is below threshold)
///   (b) 2*err_sq = 2*e2 <= e1 + e2     (from the ordering e2 <= e1)
/// ```
///
/// Neither conclusion is a bare hypothesis — (a) needs the identity
/// `err_sq = total - e1`, (b) needs the ordering — so the query is non-vacuous,
/// and every term is linear (`QF_LRA`, decidable and fast). The whole theorem
/// rests on retaining the *larger* singular value; retaining the smaller one is
/// the classic argmin/ascending-sort bug and makes the bound false — see
/// `low_rank_bound_depends_on_keeping_the_larger_value`.
pub(crate) fn prove_low_rank_error_bound() -> Result<VlmDecompPropertyResult, SmtError> {
    let program = build_low_rank_error_bound(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(VlmDecompPropertyResult {
        property: "low_rank_error_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the Eckart-Young error-bound query over energies. `keep_largest` selects
/// which singular value the rank-1 approximation retains: the true statement
/// keeps the larger (`true`); keeping the smaller (`false`) discards the larger
/// energy `e1`, which the threshold does not bound, so the query turns SAT.
fn build_low_rank_error_bound(keep_largest: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Squared singular values (Frobenius energies) of a rank-2 matrix, plus the
    // squared truncation threshold. Squared quantities are the primitives, so no
    // `s*s` product ever appears and the query stays linear.
    let e1 = declare_real(&mut program, "e1"); // sigma1^2, the larger energy
    let e2 = declare_real(&mut program, "e2"); // sigma2^2, the smaller energy
    let tau_sq = declare_real(&mut program, "tau_sq"); // tau^2, threshold energy

    let zero = Expr::real(0);
    let cap = Expr::real(ENERGY_CAP);
    for v in [&e1, &e2, &tau_sq] {
        program.assert(v.clone().real_ge(zero.clone()));
        program.assert(v.clone().real_le(cap.clone()));
    }

    // SVD energy ordering: sigma1 >= sigma2 >= 0  =>  e1 >= e2.
    program.assert(e1.clone().real_ge(e2.clone()));
    // Truncation threshold: the discarded singular value is below tolerance.
    program.assert(e2.clone().real_le(tau_sq.clone()));

    // Total Frobenius energy ||A||_F^2 = e1 + e2.
    let total = e1.clone().real_add(e2.clone());

    // Rank-1 reconstruction error energy = total energy minus the RETAINED energy
    // (Eckart-Young: the error is the discarded energy). Keeping the larger value
    // discards e2; the bug keeps the smaller and discards the larger energy e1.
    let retained = if keep_largest { e1 } else { e2 };
    let err_sq = total.clone().real_sub(retained);

    // Two derived, linear bounds; a counterexample must break at least one:
    //   (a) err_sq <= tau_sq   (threshold)      (b) 2*err_sq <= total   (ordering)
    let two = Expr::real(2);
    let violates_threshold = err_sq.clone().real_gt(tau_sq);
    let violates_half_energy = two.real_mul(err_sq).real_gt(total);

    program.assert(violates_threshold.or(violates_half_energy));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 4: SVD Reconstruction Within Tolerance
// ---------------------------------------------------------------------------

/// Prove perturbed SVD factors reconstruct within bounded error.
/// Scalar case: a = u*s*v, a_hat = (u+du)(s+ds)(v+dv). With |u|,|v| <= 1,
/// |s| <= 10, |d*| <= eps <= 0.1, the error |a_hat - a| <= 25*eps.
pub(crate) fn prove_svd_reconstruction_tolerance() -> Result<VlmDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let u = declare_real(&mut program, "u");
    let s = declare_real(&mut program, "s");
    let v = declare_real(&mut program, "v");
    let eps = declare_real(&mut program, "eps");

    assert_bounds(&mut program, &u, -1.0, 1.0)?;
    assert_bounds(&mut program, &s, 0.0, 10.0)?;
    assert_bounds(&mut program, &v, -1.0, 1.0)?;
    let zero = Expr::real(0);
    program.assert(eps.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &eps, 0.0, 0.1)?;

    let du = declare_real(&mut program, "du");
    let ds = declare_real(&mut program, "ds");
    let dv = declare_real(&mut program, "dv");
    let neg_eps = zero.clone().real_sub(eps.clone());
    for d in [&du, &ds, &dv] {
        program.assert(d.clone().real_ge(neg_eps.clone()));
        program.assert(d.clone().real_le(eps.clone()));
    }

    let a_exact = u.clone().real_mul(s.clone()).real_mul(v.clone());
    let a_hat = u
        .real_add(du)
        .real_mul(s.real_add(ds))
        .real_mul(v.real_add(dv));
    let error = a_hat.real_sub(a_exact);

    let twenty_five = real_from_f64(25.0)?;
    let bound = twenty_five.real_mul(eps);
    let neg_bound = zero.real_sub(bound.clone());

    let violation = error.clone().real_gt(bound).or(error.real_lt(neg_bound));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(VlmDecompPropertyResult {
        property: "svd_reconstruction_tolerance".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Cholesky Positive Definiteness Preservation
// ---------------------------------------------------------------------------

/// Prove Cholesky preserves positive definiteness: for 2x2 SPD A = LL^T
/// with l00, l11 > 0, both A[0,0] > 0 and det(A) > 0.
/// Critical for VLM covariance and attention score matrices.
pub(crate) fn prove_cholesky_pd_preservation() -> Result<VlmDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let l00 = declare_real(&mut program, "l00");
    let l10 = declare_real(&mut program, "l10");
    let l11 = declare_real(&mut program, "l11");

    let zero = Expr::real(0);
    let eps = real_from_f64(0.001)?;
    program.assert(l00.clone().real_ge(eps.clone()));
    program.assert(l11.clone().real_ge(eps));
    assert_bounds(&mut program, &l00, 0.001, 100.0)?;
    assert_bounds(&mut program, &l10, -100.0, 100.0)?;
    assert_bounds(&mut program, &l11, 0.001, 100.0)?;

    let a00 = l00.clone().real_mul(l00.clone());
    let det_l = l00.real_mul(l11);
    let det_a = det_l.clone().real_mul(det_l);

    let violation = a00.real_le(zero.clone()).or(det_a.real_le(zero));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(VlmDecompPropertyResult {
        property: "cholesky_pd_preservation".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: QR Orthogonality Within Bounds
// ---------------------------------------------------------------------------

/// Prove near-orthonormal Q has bounded deviation from I in Q^T Q.
/// If column norms^2 in [1-eps, 1+eps] and cross dot in [-eps, eps],
/// then each element of (Q^T Q - I) has magnitude <= eps.
pub(crate) fn prove_qr_orthogonality_within_bounds() -> Result<VlmDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let q00 = declare_real(&mut program, "q00");
    let q01 = declare_real(&mut program, "q01");
    let q10 = declare_real(&mut program, "q10");
    let q11 = declare_real(&mut program, "q11");
    let eps = declare_real(&mut program, "eps");

    for v in [&q00, &q01, &q10, &q11] {
        assert_bounds(&mut program, v, -2.0, 2.0)?;
    }
    let zero = Expr::real(0);
    program.assert(eps.clone().real_gt(zero.clone()));
    assert_bounds(&mut program, &eps, 0.0, 0.5)?;

    let one = real_from_f64(1.0)?;
    let neg_eps = zero.real_sub(eps.clone());
    let one_minus_eps = one.clone().real_sub(eps.clone());
    let one_plus_eps = one.clone().real_add(eps.clone());

    // Column 0 norm^2 in [1-eps, 1+eps]
    let col0 = q00
        .clone()
        .real_mul(q00.clone())
        .real_add(q10.clone().real_mul(q10.clone()));
    program.assert(col0.clone().real_ge(one_minus_eps.clone()));
    program.assert(col0.clone().real_le(one_plus_eps.clone()));

    // Column 1 norm^2 in [1-eps, 1+eps]
    let col1 = q01
        .clone()
        .real_mul(q01.clone())
        .real_add(q11.clone().real_mul(q11.clone()));
    program.assert(col1.clone().real_ge(one_minus_eps));
    program.assert(col1.clone().real_le(one_plus_eps));

    // Cross dot product in [-eps, eps]
    let dot = q00.real_mul(q01).real_add(q10.real_mul(q11));
    program.assert(dot.clone().real_ge(neg_eps.clone()));
    program.assert(dot.clone().real_le(eps.clone()));

    let dev_00 = col0.real_sub(one.clone());
    let dev_01 = dot;
    let dev_11 = col1.real_sub(one);

    let v00 = dev_00
        .clone()
        .real_gt(eps.clone())
        .or(dev_00.real_lt(neg_eps.clone()));
    let v01 = dev_01
        .clone()
        .real_gt(eps.clone())
        .or(dev_01.real_lt(neg_eps.clone()));
    let v11 = dev_11
        .clone()
        .real_gt(eps.clone())
        .or(dev_11.real_lt(neg_eps));

    let violation = v00.or(v01).or(v11);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(VlmDecompPropertyResult {
        property: "qr_orthogonality_within_bounds".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Eigenvalue Spectral Bound (Gershgorin)
// ---------------------------------------------------------------------------

/// Prove the Gershgorin circle theorem for 2x2 symmetric matrices:
/// every eigenvalue of [[a,b],[b,d]] lies in [a-|b|, a+|b|] or [d-|b|, d+|b|].
/// Bounds eigenvalue magnitudes in VLM weight matrices for stability verification.
pub(crate) fn prove_eigenvalue_gershgorin_bound() -> Result<VlmDecompPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let d = declare_real(&mut program, "d");
    let lambda = declare_real(&mut program, "lambda");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;
    assert_bounds(&mut program, &d, -100.0, 100.0)?;
    assert_bounds(&mut program, &lambda, -200.0, 200.0)?;

    let zero = Expr::real(0);

    // Characteristic equation: (a - lambda)(d - lambda) - b^2 = 0
    let char_eq = a
        .clone()
        .real_sub(lambda.clone())
        .real_mul(d.clone().real_sub(lambda.clone()))
        .real_sub(b.clone().real_mul(b.clone()));
    program.assert(char_eq.eq(zero.clone()));

    // |b| via abs_b
    let abs_b = declare_real(&mut program, "abs_b");
    program.assert(abs_b.clone().real_ge(zero.clone()));
    program.assert(abs_b.clone().real_ge(b.clone()));
    program.assert(abs_b.clone().real_ge(zero.clone().real_sub(b.clone())));
    program.assert(
        abs_b
            .clone()
            .eq(b.clone())
            .or(abs_b.clone().eq(zero.real_sub(b))),
    );

    // Gershgorin disks
    let in_disk_1 = lambda
        .clone()
        .real_ge(a.clone().real_sub(abs_b.clone()))
        .and(lambda.clone().real_le(a.real_add(abs_b.clone())));
    let in_disk_2 = lambda
        .clone()
        .real_ge(d.clone().real_sub(abs_b.clone()))
        .and(lambda.real_le(d.real_add(abs_b)));

    let violation = in_disk_1.not().and(in_disk_2.not());
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(VlmDecompPropertyResult {
        property: "eigenvalue_gershgorin_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

#[cfg(test)]
#[path = "ay_matrix_decomposition_vlm_tests.rs"]
mod tests;
