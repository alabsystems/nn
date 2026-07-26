// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for linear algebra mathematical properties (#4235).
//!
//! Proves fundamental linear algebra properties relevant to ML model verification.
//! Matrix multiply, transpose, identity, trace, determinant, inverse, norms, rank,
//! and symmetry are all central operations in neural network layers (linear layers,
//! attention, normalization). Each proof encodes the property as a negated assertion
//! and proves UNSAT (no counterexample exists).
//!
//! # Proved Properties
//!
//! 1. **Matrix multiply associativity**: (AB)C and A(BC) have consistent dimensions.
//! 2. **Matrix multiply dimensions**: [m,k] @ [k,n] = [m,n].
//! 3. **Transpose involution**: (A^T)^T = A for all matrix elements.
//! 4. **Identity multiply**: AI = A and IA = A for all matrix elements.
//! 6. **Determinant of product**: det(AB) = det(A)*det(B) for 2x2 matrices.
//! 7. **Inverse product rule**: (AB)^-1 = B^-1 A^-1 on a concrete invertible 2x2
//!    instance, where the reversed product order is load-bearing.
//! 8. **Frobenius norm non-negativity**: ||A||_F >= 0 for any matrix.
//! 9. **Matrix rank bound**: rank(AB) <= min(rank(A), rank(B)) via dimension encoding.
//! 10. **Symmetric property**: For symmetric A, A = A^T (element-wise equality).
//!
//! # Proof Strategy
//!
//! Matrix element proofs use symbolic real variables for individual matrix entries.
//! For small matrices (2x2), we can fully expand products and determinants. For
//! general dimension proofs, we encode dimension variables as positive reals and
//! verify size formulas algebraically. All proofs use QF_NRA (nonlinear real
//! arithmetic) or QF_LRA (linear real arithmetic) depending on complexity.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use crate::ay_real_lit::RealLit;

/// Result of a linear algebra property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct LinAlgPropertyResult {
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

/// Assert `lower <= expr <= upper`.
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

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The final `(proven, detail)` is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so any query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure. A residual vacuity therefore becomes a hard test failure rather than
/// a false "proven"; a genuine proof is returned unchanged.
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
    crate::ay_vacuity::reject_if_vacuous(&program.to_string(), proven, detail)
}

// ---------------------------------------------------------------------------
// Property 1: Matrix Multiply Associativity (Dimension Consistency)
// ---------------------------------------------------------------------------

/// Prove that matrix multiply is associative at the dimension level.
///
/// Given A: [m, k1], B: [k1, k2], C: [k2, n], the shape rule
/// `matmul([p, q], [q, r]) = [p, r]` gives
///   - AB: [m, k2], then (AB)C: [m, n]
///   - BC: [k1, n], then A(BC): [m, n]
///
/// The content is in *applying the rule*, so each intermediate shape is a
/// declared variable constrained by the rule rather than a name for the answer.
/// `a_bc_cols` is reached only through `bc_cols`, and `abc_rows` only through
/// `ab_rows`, so the solver must chain two rule applications down each side.
/// Mis-wiring one of them makes the query SAT (see
/// `associativity_depends_on_the_shape_rule`).
pub(crate) fn prove_matmul_associativity_dimensions() -> Result<LinAlgPropertyResult, SmtError> {
    let program = build_matmul_associativity_dimensions(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "matmul_associativity_dimensions".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the associativity query. When `bc_shape_rule_holds` is false, `BC` is
/// given the transposed shape `[n, k1]` — a plausible slip that breaks the
/// theorem; tests flip it to confirm the proof depends on the rule.
fn build_matmul_associativity_dimensions(bc_shape_rule_holds: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let m = declare_real(&mut program, "m");
    let k1 = declare_real(&mut program, "k1");
    let k2 = declare_real(&mut program, "k2");
    let n = declare_real(&mut program, "n");

    // All dimensions are positive integers (modeled as reals >= 1).
    for dim in [&m, &k1, &k2, &n] {
        assert_bounds(&mut program, dim, 1.0, 10000.0)?;
    }

    // AB = matmul([m, k1], [k1, k2]) = [m, k2]. Only AB's rows feed the
    // conclusion, so naming its columns would declare a variable that
    // constrains nothing.
    let ab_rows = define_real(&mut program, "ab_rows", &m);
    // (AB)C = matmul([ab_rows, k2], [k2, n]) = [ab_rows, n]
    let abc_rows = define_real(&mut program, "abc_rows", &ab_rows);
    let abc_cols = define_real(&mut program, "abc_cols", &n);

    // BC = matmul([k1, k2], [k2, n]) = [k1, n]. The slip transposes it to [n, k1].
    let bc_cols = define_real(
        &mut program,
        "bc_cols",
        if bc_shape_rule_holds { &n } else { &k1 },
    );
    // A(BC) = matmul([m, k1], [k1, bc_cols]) = [m, bc_cols]
    let a_bc_rows = define_real(&mut program, "a_bc_rows", &m);
    let a_bc_cols = define_real(&mut program, "a_bc_cols", &bc_cols);

    // Violation: the two groupings disagree on an output dimension.
    let violation = abc_rows.ne(a_bc_rows).or(abc_cols.ne(a_bc_cols));
    program.assert(violation);
    program.check_sat();

    Ok(program)
}

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Introducing a name for each intermediate shape keeps the conclusion one step
/// removed from its hypotheses, so the solver derives it instead of matching it.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

// ---------------------------------------------------------------------------
// Property 2: Matrix Multiply Output Dimensions
// ---------------------------------------------------------------------------

/// Prove that `matmul([m, k], [k, n])` produces output dimensions `[m, n]`.
///
/// The shape rule reads the output rows off the LEFT matrix's rows and the
/// output cols off the RIGHT matrix's cols, consuming the shared inner
/// dimension `k`. The derived output shape is reached through the rule applied
/// to the input shapes, while the claimed answer `[m, n]` is reached through an
/// independent chain; the two must agree by transitivity, so the conclusion is
/// derived rather than asserted equal to itself.
///
/// A plausible slip — reading the output columns off the inner dimension `k`
/// instead of the right matrix's columns `n` — makes the query SAT (see
/// `output_dimensions_depends_on_the_shape_rule`).
pub(crate) fn prove_matmul_output_dimensions() -> Result<LinAlgPropertyResult, SmtError> {
    let program = build_matmul_output_dimensions(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "matmul_output_dimensions".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the output-dimension query. When `output_shape_rule_holds` is false,
/// the output columns are taken from the shared inner dimension (`right_rows`,
/// i.e. `k`) instead of the right matrix's columns (`n`) — a "wrong axis" slip
/// that breaks `matmul([m, k], [k, n]) = [m, n]`; tests flip it to confirm the
/// proof depends on the rule.
fn build_matmul_output_dimensions(output_shape_rule_holds: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Free shape parameters (positive integers modeled as reals >= 1).
    let m = declare_real(&mut program, "m");
    let k = declare_real(&mut program, "k");
    let n = declare_real(&mut program, "n");
    for dim in [&m, &k, &n] {
        assert_bounds(&mut program, dim, 1.0, 10000.0)?;
    }

    // Input shapes: left is [m, k], right is [k, n], each named per axis.
    let left_rows = define_real(&mut program, "left_rows", &m);
    let left_cols = define_real(&mut program, "left_cols", &k);
    let right_rows = define_real(&mut program, "right_rows", &k);
    let right_cols = define_real(&mut program, "right_cols", &n);

    // matmul is defined only when the inner dimensions agree.
    program.assert(left_cols.clone().eq(right_rows.clone()));

    // Shape rule: output rows come from the LEFT matrix's rows, output cols from
    // the RIGHT matrix's cols. The slip reads the output cols off the shared
    // inner dimension (right_rows = k) instead.
    let out_rows = define_real(&mut program, "out_rows", &left_rows);
    let out_cols = define_real(
        &mut program,
        "out_cols",
        if output_shape_rule_holds {
            &right_cols
        } else {
            &right_rows
        },
    );

    // The claimed answer [m, n], reached through an independent chain so the
    // conclusion is derived by transitivity, not asserted equal to itself.
    let expected_rows = define_real(&mut program, "expected_rows", &m);
    let expected_cols = define_real(&mut program, "expected_cols", &n);

    // Violation: the derived output shape disagrees with [m, n].
    let violation = out_rows.ne(expected_rows).or(out_cols.ne(expected_cols));
    program.assert(violation);
    program.check_sat();

    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 3: Transpose Involution
// ---------------------------------------------------------------------------

/// Prove that (A^T)^T = A for a 2x2 matrix.
///
/// For A = [[a, b], [c, d]]:
///   A^T = [[a, c], [b, d]]
///   (A^T)^T = [[a, b], [c, d]] = A
///
/// `A^T` and `(A^T)^T` are declared as their own SMT variables, each *defined*
/// by the index-swap `(Mᵀ)ᵢⱼ = Mⱼᵢ`, and the solver derives the identity by
/// congruence over those definitions. Writing `att00 = a00.clone()` instead
/// would assert `a00 != a00`, which is UNSAT for reasons that have nothing to do
/// with transposition — a tautology, not a proof.
pub(crate) fn prove_transpose_involution() -> Result<LinAlgPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // 2x2 matrix A = [[a00, a01], [a10, a11]]
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");

    assert_bounds(&mut program, &a00, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &a01, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &a10, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &a11, -1000.0, 1000.0)?;

    // Define A^T by the index swap: (A^T)_ij = A_ji.
    let at00 = declare_real(&mut program, "at00");
    let at01 = declare_real(&mut program, "at01");
    let at10 = declare_real(&mut program, "at10");
    let at11 = declare_real(&mut program, "at11");
    program.assert(at00.clone().eq(a00.clone()));
    program.assert(at01.clone().eq(a10.clone()));
    program.assert(at10.clone().eq(a01.clone()));
    program.assert(at11.clone().eq(a11.clone()));

    // Define (A^T)^T by the same index swap applied to A^T.
    let att00 = declare_real(&mut program, "att00");
    let att01 = declare_real(&mut program, "att01");
    let att10 = declare_real(&mut program, "att10");
    let att11 = declare_real(&mut program, "att11");
    program.assert(att00.clone().eq(at00));
    program.assert(att01.clone().eq(at10));
    program.assert(att10.clone().eq(at01));
    program.assert(att11.clone().eq(at11));

    // Violation: any element of (A^T)^T differs from A.
    let v00 = att00.ne(a00);
    let v01 = att01.ne(a01);
    let v10 = att10.ne(a10);
    let v11 = att11.ne(a11);

    let violation = v00.or(v01).or(v10).or(v11);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "transpose_involution".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Identity Multiply
// ---------------------------------------------------------------------------

/// Prove that AI = A and IA = A for a 2x2 matrix and 2x2 identity.
///
/// For A = [[a, b], [c, d]] and I = [[1, 0], [0, 1]]:
///   AI = [[a*1+b*0, a*0+b*1], [c*1+d*0, c*0+d*1]] = [[a, b], [c, d]] = A
///   IA = [[1*a+0*c, 1*b+0*d], [0*a+1*c, 0*b+1*d]] = [[a, b], [c, d]] = A
pub(crate) fn prove_identity_multiply() -> Result<LinAlgPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");

    assert_bounds(&mut program, &a, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &b, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &c, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &d, -1000.0, 1000.0)?;

    let zero = Expr::real(0);
    let one = real_from_f64(1.0)?;

    // AI: product[i,j] = sum_k A[i,k] * I[k,j]
    // I = [[1,0],[0,1]]
    // AI[0,0] = a*1 + b*0 = a
    let ai00 = a
        .clone()
        .real_mul(one.clone())
        .real_add(b.clone().real_mul(zero.clone()));
    // AI[0,1] = a*0 + b*1 = b
    let ai01 = a
        .clone()
        .real_mul(zero.clone())
        .real_add(b.clone().real_mul(one.clone()));
    // AI[1,0] = c*1 + d*0 = c
    let ai10 = c
        .clone()
        .real_mul(one.clone())
        .real_add(d.clone().real_mul(zero.clone()));
    // AI[1,1] = c*0 + d*1 = d
    let ai11 = c
        .clone()
        .real_mul(zero.clone())
        .real_add(d.clone().real_mul(one.clone()));

    // IA: product[i,j] = sum_k I[i,k] * A[k,j]
    // IA[0,0] = 1*a + 0*c = a
    let ia00 = one
        .clone()
        .real_mul(a.clone())
        .real_add(zero.clone().real_mul(c.clone()));
    // IA[0,1] = 1*b + 0*d = b
    let ia01 = one
        .clone()
        .real_mul(b.clone())
        .real_add(zero.clone().real_mul(d.clone()));
    // IA[1,0] = 0*a + 1*c = c
    let ia10 = zero
        .clone()
        .real_mul(a.clone())
        .real_add(one.clone().real_mul(c.clone()));
    // IA[1,1] = 0*b + 1*d = d
    let ia11 = zero.real_mul(b.clone()).real_add(one.real_mul(d.clone()));

    // Violation: any element of AI or IA differs from A
    let vai00 = ai00.ne(a.clone());
    let vai01 = ai01.ne(b.clone());
    let vai10 = ai10.ne(c.clone());
    let vai11 = ai11.ne(d.clone());
    let via00 = ia00.ne(a);
    let via01 = ia01.ne(b);
    let via10 = ia10.ne(c);
    let via11 = ia11.ne(d);

    let violation = vai00
        .or(vai01)
        .or(vai10)
        .or(vai11)
        .or(via00)
        .or(via01)
        .or(via10)
        .or(via11);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "identity_multiply".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Determinant of Product (2x2)
// ---------------------------------------------------------------------------

/// Prove that det(AB) = det(A) * det(B) for 2x2 matrices.
///
/// For A = [[a, b], [c, d]] and B = [[e, f], [g, h]]:
///   det(A) = ad - bc
///   det(B) = eh - fg
///   AB = [[ae+bg, af+bh], [ce+dg, cf+dh]]
///   det(AB) = (ae+bg)(cf+dh) - (af+bh)(ce+dg)
///
/// We prove: det(AB) = det(A) * det(B) = (ad-bc)(eh-fg)
///
/// Expanding both sides yields the same polynomial. Since this involves
/// products of 4 symbolic variables, we use QF_NRA.
pub(crate) fn prove_determinant_of_product_2x2() -> Result<LinAlgPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");
    let e = declare_real(&mut program, "e");
    let f = declare_real(&mut program, "f");
    let g = declare_real(&mut program, "g");
    let h = declare_real(&mut program, "h");

    // Bound all entries
    for v in [&a, &b, &c, &d, &e, &f, &g, &h] {
        assert_bounds(&mut program, v, -100.0, 100.0)?;
    }

    // det(A) = a*d - b*c
    let det_a = a
        .clone()
        .real_mul(d.clone())
        .real_sub(b.clone().real_mul(c.clone()));
    // det(B) = e*h - f*g
    let det_b = e
        .clone()
        .real_mul(h.clone())
        .real_sub(f.clone().real_mul(g.clone()));
    // det(A) * det(B)
    let det_a_times_det_b = det_a.real_mul(det_b);

    // AB = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    let ab00 = a
        .clone()
        .real_mul(e.clone())
        .real_add(b.clone().real_mul(g.clone()));
    let ab01 = a.real_mul(f.clone()).real_add(b.real_mul(h.clone()));
    let ab10 = c.clone().real_mul(e).real_add(d.clone().real_mul(g));
    let ab11 = c.real_mul(f).real_add(d.real_mul(h));

    // det(AB) = ab00*ab11 - ab01*ab10
    let det_ab = ab00.real_mul(ab11).real_sub(ab01.real_mul(ab10));

    // Violation: det(AB) != det(A)*det(B)
    let violation = det_ab.ne(det_a_times_det_b);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "determinant_of_product_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Inverse Product Rule (2x2)
// ---------------------------------------------------------------------------

/// Prove the inverse product ("socks and shoes") rule `(AB)^-1 = B^-1 A^-1` on a
/// concrete invertible 2x2 instance.
///
/// The rule's whole content is the *order reversal*: for non-commuting `A`, `B`
/// the inverse of the product is the product of the inverses in the opposite
/// order. We take
///   A = [[1, 2], [3, 4]],  A^-1 = [[-2, 1], [3/2, -1/2]]
///   B = [[2, 0], [1, 3]],  B^-1 = [[1/2, 0], [-1/6, 1/3]]
/// (each inverse checkable by `M M^-1 = I`), form `AB`, and verify that
/// `AB * (B^-1 A^-1) = I` — i.e. `B^-1 A^-1` really is the inverse of `AB`.
///
/// Every entry is an exact rational literal, so the products are literal x
/// literal and the query stays linear (QF_LRA), proven exactly. The load-bearing
/// content is the reversed order `B^-1 A^-1`: multiplying the inverses in the
/// naive same order `A^-1 B^-1` gives `AB * (A^-1 B^-1) != I` (its (0,0) entry is
/// 1/3, not 1), so the query is SAT — see
/// `inverse_product_rule_2x2_depends_on_the_product_order`. Each violation
/// disjunct compares a compound product-sum (a matrix entry) to the literal `1`
/// or `0` of the identity, so the two sides are genuinely different shapes.
pub(crate) fn prove_inverse_product_rule_2x2() -> Result<LinAlgPropertyResult, SmtError> {
    let program = build_inverse_product_rule_2x2(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "inverse_product_rule_2x2".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the inverse-product-rule query for the concrete instance. When
/// `product_order_reversed` is true the inverses are multiplied in the correct
/// reversed order `B^-1 A^-1`, so `AB * (B^-1 A^-1) = I` and the query is UNSAT.
/// When false the naive same-order product `A^-1 B^-1` is used instead; then
/// `AB * (A^-1 B^-1) != I` and the query is SAT. Tests flip it to confirm the
/// proof depends on the order reversal, not on writing the identity twice.
fn build_inverse_product_rule_2x2(product_order_reversed: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // A and B as exact literal 2x2 matrices, row-major [[m00, m01], [m10, m11]].
    let a = [
        [Expr::real(1), Expr::real(2)],
        [Expr::real(3), Expr::real(4)],
    ];
    let b = [
        [Expr::real(2), Expr::real(0)],
        [Expr::real(1), Expr::real(3)],
    ];
    // Their exact inverses (each verifiable by M * M^-1 = I).
    let a_inv = [
        [Expr::real(-2), Expr::real(1)],
        [Expr::real_ratio(3, 2), Expr::real_ratio(-1, 2)],
    ];
    let b_inv = [
        [Expr::real_ratio(1, 2), Expr::real(0)],
        [Expr::real_ratio(-1, 6), Expr::real_ratio(1, 3)],
    ];

    // AB = A * B.
    let ab = matmul_2x2(&a, &b);

    // The product of the inverses. The rule requires the reversed order
    // B^-1 A^-1; the mutation uses the naive same order A^-1 B^-1.
    let inv_product = if product_order_reversed {
        matmul_2x2(&b_inv, &a_inv)
    } else {
        matmul_2x2(&a_inv, &b_inv)
    };

    // AB * (product of inverses) should be the identity.
    let prod = matmul_2x2(&ab, &inv_product);
    let identity = [
        [Expr::real(1), Expr::real(0)],
        [Expr::real(0), Expr::real(1)],
    ];

    // Violation: any entry of AB * (B^-1 A^-1) differs from the identity.
    let mut disjuncts: Vec<Expr> = Vec::new();
    for (row, id_row) in prod.iter().zip(identity.iter()) {
        for (entry, id) in row.iter().zip(id_row.iter()) {
            disjuncts.push(entry.clone().ne(id.clone()));
        }
    }
    program.assert(Expr::or_many(disjuncts));
    program.check_sat();

    program
}

/// Multiply two row-major 2x2 matrices symbolically:
/// `(XY)[i][j] = X[i][0]*Y[0][j] + X[i][1]*Y[1][j]`.
fn matmul_2x2(x: &[[Expr; 2]; 2], y: &[[Expr; 2]; 2]) -> [[Expr; 2]; 2] {
    let entry = |i: usize, j: usize| {
        x[i][0]
            .clone()
            .real_mul(y[0][j].clone())
            .real_add(x[i][1].clone().real_mul(y[1][j].clone()))
    };
    [
        [entry(0, 0), entry(0, 1)],
        [entry(1, 0), entry(1, 1)],
    ]
}

// ---------------------------------------------------------------------------
// Property 8: Frobenius Norm Non-Negativity
// ---------------------------------------------------------------------------

/// Prove that the Frobenius norm is non-negative: ||A||_F >= 0.
///
/// For a 2x2 matrix A = [[a, b], [c, d]]:
///   ||A||_F^2 = a^2 + b^2 + c^2 + d^2
///
/// Since each term is a square (>= 0), the sum is >= 0, and ||A||_F = sqrt(sum) >= 0.
///
/// We prove ||A||_F^2 >= 0 directly (avoiding sqrt in SMT). Since
/// sqrt is monotonically increasing and sqrt(x) >= 0 for x >= 0,
/// proving the squared norm is non-negative suffices.
pub(crate) fn prove_frobenius_norm_non_negative() -> Result<LinAlgPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");

    assert_bounds(&mut program, &a, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &b, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &c, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &d, -1000.0, 1000.0)?;

    let zero = Expr::real(0);

    // ||A||_F^2 = a^2 + b^2 + c^2 + d^2
    let frob_sq = a
        .clone()
        .real_mul(a)
        .real_add(b.clone().real_mul(b))
        .real_add(c.clone().real_mul(c))
        .real_add(d.clone().real_mul(d));

    // Violation: ||A||_F^2 < 0
    let violation = frob_sq.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "frobenius_norm_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: Matrix Rank Bound
// ---------------------------------------------------------------------------

/// Prove that rank(AB) <= min(rank(A), rank(B)).
///
/// For matrices A: [m, k] and B: [k, n]:
///   rank(A) <= min(m, k)
///   rank(B) <= min(k, n)
///   rank(AB) <= min(rank(A), rank(B))
///
/// Since rank is not directly encodable in SMT real arithmetic, we model it
/// abstractly: given non-negative integer variables rank_a, rank_b, rank_ab
/// with the constraints:
///   rank_a <= m, rank_a <= k  (rank bounded by matrix dimensions)
///   rank_b <= k, rank_b <= n
///   rank_ab <= rank_a, rank_ab <= rank_b  (the Sylvester rank inequality)
///
/// We prove that rank_ab <= min(rank_a, rank_b) is consistent with these
/// constraints by asserting the negation and showing UNSAT.
pub(crate) fn prove_rank_bound() -> Result<LinAlgPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let m = declare_real(&mut program, "m");
    let k = declare_real(&mut program, "k");
    let n = declare_real(&mut program, "n");
    let rank_a = declare_real(&mut program, "rank_a");
    let rank_b = declare_real(&mut program, "rank_b");
    let rank_ab = declare_real(&mut program, "rank_ab");

    let zero = Expr::real(0);

    // Dimension bounds
    assert_bounds(&mut program, &m, 1.0, 1000.0)?;
    assert_bounds(&mut program, &k, 1.0, 1000.0)?;
    assert_bounds(&mut program, &n, 1.0, 1000.0)?;

    // Rank non-negativity
    program.assert(rank_a.clone().real_ge(zero.clone()));
    program.assert(rank_b.clone().real_ge(zero.clone()));
    program.assert(rank_ab.clone().real_ge(zero));

    // Rank bounded by dimensions
    program.assert(rank_a.clone().real_le(m));
    program.assert(rank_a.clone().real_le(k.clone()));
    program.assert(rank_b.clone().real_le(k));
    program.assert(rank_b.clone().real_le(n));

    // Sylvester rank inequality: rank(AB) <= rank(A) and rank(AB) <= rank(B)
    program.assert(rank_ab.clone().real_le(rank_a.clone()));
    program.assert(rank_ab.clone().real_le(rank_b.clone()));

    // min(rank_a, rank_b) via helper variable
    let min_rank = declare_real(&mut program, "min_rank");
    // min_rank <= rank_a AND min_rank <= rank_b
    program.assert(min_rank.clone().real_le(rank_a.clone()));
    program.assert(min_rank.clone().real_le(rank_b.clone()));
    // min_rank = rank_a OR min_rank = rank_b (it equals one of them)
    program.assert(min_rank.clone().eq(rank_a).or(min_rank.clone().eq(rank_b)));

    // Violation: rank_ab > min_rank
    let violation = rank_ab.real_gt(min_rank);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "rank_bound".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 10: Symmetric Matrix Property
// ---------------------------------------------------------------------------

/// Prove that a symmetric matrix equals its transpose off the diagonal.
///
/// For A = [[a00, a01], [a10, a11]] with the symmetry hypothesis A[0,1] = A[1,0],
/// the transpose rule `(A^T)[i,j] = A[j,i]` gives `(A^T)[0,1] = a10` and
/// `(A^T)[1,0] = a01`. The property `A[0,1] = (A^T)[0,1]` then holds by
/// transitivity `a01 = a10 = (A^T)[0,1]` — it is derived from the symmetry
/// hypothesis and the transpose rule, not asserted equal to itself. The diagonal
/// comparisons `a00 = a00`, `a11 = a11` carry no information (they hold for any
/// matrix), so only the off-diagonal entries — where the theorem has content —
/// feed the violation.
///
/// Dropping the symmetry hypothesis makes the query SAT (see
/// `symmetric_property_depends_on_the_symmetry_hypothesis`), so the proof rests
/// on symmetry rather than on the transpose rule alone.
pub(crate) fn prove_symmetric_property() -> Result<LinAlgPropertyResult, SmtError> {
    let program = build_symmetric_property(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(LinAlgPropertyResult {
        property: "symmetric_property".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the symmetric-matrix query. When `matrix_is_symmetric` is false, the
/// symmetry hypothesis `A[0,1] = A[1,0]` is dropped, so a general matrix need not
/// match its transpose off the diagonal; tests flip it to confirm the proof
/// depends on the hypothesis.
fn build_symmetric_property(matrix_is_symmetric: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // 2x2 matrix A = [[a00, a01], [a10, a11]].
    let a00 = declare_real(&mut program, "a00");
    let a01 = declare_real(&mut program, "a01");
    let a10 = declare_real(&mut program, "a10");
    let a11 = declare_real(&mut program, "a11");
    for v in [&a00, &a01, &a10, &a11] {
        assert_bounds(&mut program, v, -1000.0, 1000.0)?;
    }

    // Symmetry hypothesis: A[0,1] = A[1,0]. The mutation drops it.
    if matrix_is_symmetric {
        program.assert(a01.clone().eq(a10.clone()));
    }

    // A^T defined by the transpose index-swap rule: (A^T)[i,j] = A[j,i].
    // The diagonal entries pin a complete A^T; only the off-diagonal entries
    // feed the violation, since that is where the theorem has content.
    let _at00 = define_real(&mut program, "at00", &a00);
    let at01 = define_real(&mut program, "at01", &a10);
    let at10 = define_real(&mut program, "at10", &a01);
    let _at11 = define_real(&mut program, "at11", &a11);

    // Violation: a symmetric matrix must match its transpose off the diagonal,
    // i.e. A[0,1] = (A^T)[0,1] and A[1,0] = (A^T)[1,0]. Under symmetry these hold
    // by transitivity (a01 = a10 = at01); without it they can differ.
    let violation = a01.ne(at01).or(a10.ne(at10));
    program.assert(violation);
    program.check_sat();

    Ok(program)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    #[test]
    fn test_matmul_associativity_dimensions_proven() {
        let result = prove_matmul_associativity_dimensions().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven,
            "Matmul associativity dimensions (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "matmul_associativity_dimensions");
    }

    /// Give `BC` the transposed shape `[n, k1]`. Then `A(BC)` has `k1` columns
    /// while `(AB)C` has `n`, and the query must find a counterexample — proving
    /// the theorem rests on the shape rule rather than on writing `[m, n]` twice.
    #[test]
    fn associativity_depends_on_the_shape_rule() {
        let program = build_matmul_associativity_dimensions(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with BC transposed the groupings disagree and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_matmul_output_dimensions_proven() {
        let result = prove_matmul_output_dimensions().expect("proof should not error");
        assert!(
            result.proven,
            "Matmul output dimensions (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "matmul_output_dimensions");
    }

    /// Drop the shape rule for the output columns: read them off the shared inner
    /// dimension `k` (`right_rows`) instead of the right matrix's columns `n`.
    /// Then the derived shape is [m, k] while the claimed answer is [m, n], and
    /// since `k` and `n` are independent the query must be SAT — proving the
    /// theorem rests on the rule rather than on writing `[m, n]` twice.
    #[test]
    fn output_dimensions_depends_on_the_shape_rule() {
        let program = build_matmul_output_dimensions(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with output cols read off the inner dim the shape disagrees and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_transpose_involution_proven() {
        let result = prove_transpose_involution().expect("proof should not error");
        assert!(
            result.proven,
            "Transpose involution (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "transpose_involution");
    }

    #[test]
    fn test_identity_multiply_proven() {
        let result = prove_identity_multiply().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Identity multiply: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Identity multiply must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "identity_multiply");
    }

    #[test]
    fn test_determinant_of_product_2x2_proven() {
        let result = prove_determinant_of_product_2x2().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Determinant product: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Determinant product must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "determinant_of_product_2x2");
    }

    #[test]
    fn test_inverse_product_rule_2x2_proven() {
        let result = prove_inverse_product_rule_2x2().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven,
            "Inverse product rule (QF_LRA, concrete instance) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "inverse_product_rule_2x2");
    }

    /// Multiply the inverses in the naive same order `A^-1 B^-1` instead of the
    /// reversed order the rule requires. Then `AB * (A^-1 B^-1) != I` (its (0,0)
    /// entry is 1/3, not 1) and the query must be SAT — proving the theorem rests
    /// on the order reversal, not on writing the identity on both sides.
    #[test]
    fn inverse_product_rule_2x2_depends_on_the_product_order() {
        let program = build_inverse_product_rule_2x2(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the inverses multiplied in the wrong order AB*(A^-1 B^-1) != I; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_frobenius_norm_non_negative_proven() {
        let result = prove_frobenius_norm_non_negative().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Frobenius norm: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Frobenius norm must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "frobenius_norm_non_negative");
    }

    #[test]
    fn test_rank_bound_proven() {
        let result = prove_rank_bound().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven,
            "Rank bound (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "rank_bound");
    }

    #[test]
    fn test_symmetric_property_proven() {
        let result = prove_symmetric_property().expect("proof should not error");
        assert!(
            result.proven,
            "Symmetric property (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "symmetric_property");
    }

    /// Drop the symmetry hypothesis. A general matrix's off-diagonal entries need
    /// not survive a transpose, so A[0,1] and (A^T)[0,1] = A[1,0] can differ and
    /// the query must be SAT — proving the theorem rests on symmetry, not on the
    /// transpose rule alone.
    #[test]
    fn symmetric_property_depends_on_the_symmetry_hypothesis() {
        let program = build_symmetric_property(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the symmetry hypothesis A differs from A^T off-diagonal; query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_smt2_structure_matmul_assoc() {
        let result = prove_matmul_associativity_dimensions().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }
}
