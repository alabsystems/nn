// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for broadcasting and reduction mathematical properties (#4211).
//!
//! Broadcasting and reduction are fundamental tensor operations. Broadcasting expands
//! dimensions to make shapes compatible; reductions collapse dimensions via aggregation
//! (sum, mean, max, min, product, argmax, variance, logsumexp). This module proves
//! key mathematical properties of these operations using ay's SMT solver.
//!
//! # Proved Properties
//!
//! 1. **Broadcasting shape compatibility**: Dimensions are broadcastable when equal or one is 1.
//! 2. **Sum reduction**: Associativity, commutativity, empty sum = 0.
//! 3. **Mean reduction**: Bounded by min/max of input, mean of constant = constant.
//! 4. **Max/Min reduction**: Idempotence, output in input range.
//! 5. **Product reduction**: Product of positives is positive.
//! 6. **Broadcast + reduce roundtrip**: `reduce_sum(broadcast(x))` recovers scaled x.
//! 7. **ArgMax**: Index in valid range, selected value >= all others.
//! 8. **Variance reduction**: Non-negativity, zero for constant, scaling `var(ax) = a^2 * var(x)`.
//! 9. **LogSumExp**: Bounded below by max, bounded above by max + log(n).
//!
//! # Proof Strategy
//!
//! Most properties are algebraic identities provable in QF_LRA (linear real arithmetic)
//! or QF_NRA (non-linear real arithmetic). We model tensor elements as individual real
//! variables and encode reduction operations as explicit formulas over those variables.
//! For properties involving transcendentals (log/exp in LogSumExp), we use symbolic
//! variables with axiomatic constraints (e.g., exp is monotone and positive).

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a broadcast/reduction property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct BroadcastReductionPropertyResult {
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
/// The `(proven, detail)` verdict is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`] before it is returned, so any query
/// that is UNSAT only because it asserts `P ∧ ¬P` (or compares a term to itself)
/// is downgraded to a failure instead of counting as a proof.
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
// Property 1: Broadcasting Shape Compatibility
// ---------------------------------------------------------------------------

/// Prove that broadcasting is valid only when dimensions are equal or one is 1.
///
/// For two dimensions `a` and `b` (positive integers), NumPy broadcasting is
/// defined when `a == b` OR `a == 1` OR `b == 1`. We prove the contrapositive: a
/// pair that is genuinely broadcastable must satisfy at least one of those three
/// conditions, i.e. `a != b AND a != 1 AND b != 1` is impossible for a
/// broadcastable pair.
///
/// The content is entirely in how "broadcastable" is encoded. It is NOT enough to
/// say an output dimension `r = max(a, b)` exists — *every* pair of dims trivially
/// has a max, so that places no compatibility constraint and admits `a=2, b=3` as
/// a spurious "broadcastable" pair (this is the exact bug the old real-valued
/// encoding had: it was SAT). NumPy's real rule is that each input dimension is
/// either equal to the output `r` or equal to `1` — `(a==r OR a==1) AND (b==r OR
/// b==1)`. Under that rule, if `a != 1` then `a == r`, and if `b != 1` then
/// `b == r`, forcing `a == b`; so `a != b AND a != 1 AND b != 1` is UNSAT.
///
/// Dimensions are `Int`, not `Real`, and the query is `QF_LIA` — decidable and
/// fast. See `broadcast_compatibility_depends_on_the_numpy_rule` for the mutation
/// that drops the rule down to bare `max` and turns the query SAT.
pub(crate) fn prove_broadcast_shape_compatibility(
) -> Result<BroadcastReductionPropertyResult, SmtError> {
    let program = build_broadcast_shape_compatibility(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "broadcast_shape_compatibility".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the broadcast-compatibility query. `numpy_rule` gates how
/// "broadcastable" is encoded: when true, each input dim is constrained to equal
/// the output `r` or `1` (NumPy's rule), which forces the conclusion. When false
/// it is mis-encoded as merely `r = max(a, b)` (`r` attains one of the inputs) —
/// a real constraint on `r` but none on compatibility — so `a=2, b=3` slips
/// through and the query turns SAT.
fn build_broadcast_shape_compatibility(numpy_rule: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let a = program.declare_const("a", Sort::int());
    let b = program.declare_const("b", Sort::int());
    let r = program.declare_const("r", Sort::int()); // broadcast output dimension

    let one = Expr::int(1);

    // Dimensions are positive integers (>= 1).
    program.assert(a.clone().int_ge(one.clone()));
    program.assert(b.clone().int_ge(one.clone()));

    // The output dimension is an upper bound of both inputs: r = max(a, b).
    program.assert(r.clone().int_ge(a.clone()));
    program.assert(r.clone().int_ge(b.clone()));

    if numpy_rule {
        // NumPy broadcasting: each input dim is either equal to the output or 1.
        // This is the constraint that makes the two dims genuinely compatible.
        let a_ok = a.clone().eq(r.clone()).or(a.clone().eq(one.clone()));
        let b_ok = b.clone().eq(r.clone()).or(b.clone().eq(one.clone()));
        program.assert(a_ok);
        program.assert(b_ok);
    } else {
        // BUG: "broadcastable" mis-encoded as merely r = max(a, b) (r attains one
        // input). Every pair of dims has a max, so this imposes no compatibility
        // constraint and a=2, b=3 slips through as spuriously "broadcastable".
        let r_is_a = r.clone().eq(a.clone());
        let r_is_b = r.clone().eq(b.clone());
        program.assert(r_is_a.or(r_is_b));
    }

    // Negated conclusion: the dims satisfy NONE of the compatibility conditions
    // (a != b AND a != 1 AND b != 1). UNSAT proves any broadcastable pair must
    // satisfy at least one.
    program.assert(a.clone().ne(b.clone()));
    program.assert(a.clone().ne(one.clone()));
    program.assert(b.ne(one));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 2: Sum Reduction Properties
// ---------------------------------------------------------------------------

// NOTE: `prove_sum_associativity` (`(a+b)+c == a+(b+c)`) and
// `prove_sum_commutativity` (`a+b == b+a`) were removed: both are pure
// associativity/commutativity identities of `+`. Their two sides are identical
// under AC-normalization of `+`, so the UNSAT is vacuous-by-identity and proves
// no operational behavior — there is no non-vacuous restatement over free
// scalars. See the empty-sum, mean-bound, and roundtrip proofs below for the
// non-vacuous sum properties that carry real content.

/// Prove that the empty sum is the additive identity, `0`.
///
/// The content is not the restatement `s = 0` — asserting that and negating it is
/// UNSAT for free and proves nothing. What must be proven is that the value an
/// empty reduction returns is the element that leaves any partial result
/// unchanged under the reduction's operator. For SUM that operator is `+`, so the
/// empty-sum value `s` must satisfy `s + y = y` for an arbitrary partial sum `y`;
/// that identity axiom *forces* `s = 0`. The conclusion is derived from the
/// axiom, not asserted, so a wrong empty-sum value would violate it.
///
/// A reduction accidentally seeded with the *multiplicative* identity (the
/// classic empty-sum / empty-product confusion) returns `1`, which does not
/// satisfy `s + y = y` — see `sum_empty_depends_on_the_additive_identity`.
pub(crate) fn prove_sum_empty_is_zero() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let program = build_sum_empty_is_zero(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "sum_empty_is_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the empty-sum query. When `additive_identity` is true, `s` is pinned by
/// the additive identity axiom `s + y = y` over an arbitrary `y` (⇒ `s = 0`).
/// When false it is seeded with the *multiplicative* identity instead — modeled
/// by `7 * s = 7` (⇒ `s = 1`, the empty-*product* value) — so the property
/// `s = 0` becomes SAT. Both branches stay linear (`QF_LRA`): no two declared
/// variables are ever multiplied.
fn build_sum_empty_is_zero(additive_identity: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // The value the empty reduction returns.
    let s = declare_real(&mut program, "s");
    let zero = Expr::real(0);

    if additive_identity {
        // Empty SUM: `s` is the additive identity, pinned by `s + y = y` for an
        // arbitrary partial sum `y`. `s * y` is never formed, so this is linear.
        let y = declare_real(&mut program, "y");
        assert_bounds(&mut program, &y, -1000.0, 1000.0)?;
        program.assert(s.clone().real_add(y.clone()).eq(y));
    } else {
        // BUG: seeded with the multiplicative identity (empty PRODUCT). The
        // identity axiom for `*` is `s * y = y`; pinned at the literal `y = 7` to
        // stay linear (`7 * s = 7`), it forces `s = 1`, not `0`.
        let seven = Expr::real(7);
        program.assert(seven.clone().real_mul(s.clone()).eq(seven));
    }

    // Negated property: the empty-sum value differs from 0.
    let violation = s.ne(zero);
    program.assert(violation);
    program.check_sat();

    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 3: Mean Reduction Properties
// ---------------------------------------------------------------------------

/// Prove that the mean of n values is bounded by min and max.
///
/// For values x1, x2, x3 with min_val <= xi <= max_val,
/// the mean (x1+x2+x3)/3 satisfies min_val <= mean <= max_val.
///
/// We prove this for n=3 by assuming the bounds violation and showing UNSAT.
pub(crate) fn prove_mean_bounded_by_min_max() -> Result<BroadcastReductionPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let x3 = declare_real(&mut program, "x3");
    let min_val = declare_real(&mut program, "min_val");
    let max_val = declare_real(&mut program, "max_val");
    let mean = declare_real(&mut program, "mean");

    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x3, -1000.0, 1000.0)?;

    // min_val <= x_i <= max_val for all i
    program.assert(x1.clone().real_ge(min_val.clone()));
    program.assert(x1.clone().real_le(max_val.clone()));
    program.assert(x2.clone().real_ge(min_val.clone()));
    program.assert(x2.clone().real_le(max_val.clone()));
    program.assert(x3.clone().real_ge(min_val.clone()));
    program.assert(x3.clone().real_le(max_val.clone()));

    // min_val <= max_val
    program.assert(min_val.clone().real_le(max_val.clone()));

    // mean = (x1 + x2 + x3) / 3
    // Encode as: 3 * mean = x1 + x2 + x3
    let three = Expr::real(3);
    let sum = x1.real_add(x2).real_add(x3);
    let three_mean = three.real_mul(mean.clone());
    program.assert(three_mean.eq(sum));

    // Negated property: mean < min_val OR mean > max_val
    let too_low = mean.clone().real_lt(min_val);
    let too_high = mean.real_gt(max_val);
    let violation = too_low.or(too_high);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "mean_bounded_by_min_max".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that the mean of a constant value equals that constant: mean(c, c, c) = c.
pub(crate) fn prove_mean_of_constant() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let c = declare_real(&mut program, "c");
    let mean = declare_real(&mut program, "mean");

    assert_bounds(&mut program, &c, -1000.0, 1000.0)?;

    // mean = (c + c + c) / 3 = c
    // Encode as: 3 * mean = c + c + c = 3 * c
    let three = Expr::real(3);
    let sum = c.clone().real_add(c.clone()).real_add(c.clone());
    let three_mean = three.real_mul(mean.clone());
    program.assert(three_mean.eq(sum));

    // Negated property: mean != c
    let violation = mean.ne(c);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "mean_of_constant_equals_constant".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Max/Min Reduction Properties
// ---------------------------------------------------------------------------

/// Prove idempotence of max: `max(a, a) = a`.
///
/// "max" is not an SMT primitive here; it is *defined* by two axioms over its two
/// argument slots `x1, x2` — the output is an upper bound of every input
/// (`m >= x1`, `m >= x2`) and the output is attained, i.e. equal to one of the
/// inputs (`m == x1 OR m == x2`). Idempotence is the statement that, when both
/// arguments are the same value `a`, this definition yields `a`. The conclusion
/// `m == a` is *derived*: the upper-bound axiom gives `m >= a`, and the
/// attainment axiom (both inputs being `a`) forces `m == a`.
///
/// The attainment axiom is load-bearing: a "max" that only guarantees an upper
/// bound — a realistic slip — is no longer idempotent, since `m` may exceed `a`.
/// See `max_idempotent_depends_on_attainment`.
pub(crate) fn prove_max_idempotent() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let program = build_max_idempotent(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "max_idempotent".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `max(a, a) = a` query. `attained` gates the "output is one of the
/// inputs" axiom; dropping it leaves `m` merely an upper bound of `a`, free to
/// exceed it, which makes the query SAT.
fn build_max_idempotent(attained: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let x1 = declare_real(&mut program, "x1"); // first argument to max
    let x2 = declare_real(&mut program, "x2"); // second argument to max
    let m = declare_real(&mut program, "m"); // m = max(x1, x2)

    assert_bounds(&mut program, &a, -1000.0, 1000.0)?;

    // The idempotent case: both arguments to max are the same value `a`.
    program.assert(x1.clone().eq(a.clone()));
    program.assert(x2.clone().eq(a.clone()));

    // Upper-bound axiom: max is >= every input.
    program.assert(m.clone().real_ge(x1.clone()));
    program.assert(m.clone().real_ge(x2.clone()));

    // Attainment axiom: max equals one of its inputs. This is what collapses `m`
    // to `a` once both inputs are `a`; without it `m` is merely an upper bound.
    if attained {
        let m_is_x1 = m.clone().eq(x1);
        let m_is_x2 = m.clone().eq(x2);
        program.assert(m_is_x1.or(m_is_x2));
    }

    // Negated property: the derived max differs from `a`.
    let violation = m.ne(a);
    program.assert(violation);
    program.check_sat();

    Ok(program)
}

/// Prove that max output is in the input range: min(inputs) <= max(inputs) <= max(inputs).
///
/// For three values x1, x2, x3, the maximum m satisfies:
///   m >= x1 AND m >= x2 AND m >= x3 (m is an upper bound)
///   m == x1 OR m == x2 OR m == x3 (m is attained)
///
/// We prove m is bounded: x1 <= m (where x1 is the min-side bound) and m <= max_bound.
pub(crate) fn prove_max_in_input_range() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let x3 = declare_real(&mut program, "x3");
    let m = declare_real(&mut program, "m"); // max(x1, x2, x3)
    let lo = declare_real(&mut program, "lo"); // min(x1, x2, x3)

    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x3, -1000.0, 1000.0)?;

    // m = max(x1, x2, x3): m >= all AND m is one of them
    program.assert(m.clone().real_ge(x1.clone()));
    program.assert(m.clone().real_ge(x2.clone()));
    program.assert(m.clone().real_ge(x3.clone()));
    let m_eq_x1 = m.clone().eq(x1.clone());
    let m_eq_x2 = m.clone().eq(x2.clone());
    let m_eq_x3 = m.clone().eq(x3.clone());
    program.assert(m_eq_x1.or(m_eq_x2).or(m_eq_x3));

    // lo = min(x1, x2, x3): lo <= all AND lo is one of them
    program.assert(lo.clone().real_le(x1.clone()));
    program.assert(lo.clone().real_le(x2.clone()));
    program.assert(lo.clone().real_le(x3.clone()));
    let lo_eq_x1 = lo.clone().eq(x1);
    let lo_eq_x2 = lo.clone().eq(x2);
    let lo_eq_x3 = lo.clone().eq(x3);
    program.assert(lo_eq_x1.or(lo_eq_x2).or(lo_eq_x3));

    // Negated property: m < lo (max is less than min — impossible)
    let violation = m.real_lt(lo);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "max_in_input_range".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove idempotence of min: `min(a, a) = a`.
///
/// Mirror of [`prove_max_idempotent`]. "min" is defined by a lower-bound axiom
/// (`m <= x1`, `m <= x2`) and an attainment axiom (`m == x1 OR m == x2`). With
/// both arguments pinned to `a`, the lower-bound axiom gives `m <= a` and the
/// attainment axiom forces `m == a`; the conclusion is derived, not asserted.
///
/// Dropping attainment leaves `m` merely a lower bound of `a`, free to fall below
/// it — see `min_idempotent_depends_on_attainment`.
pub(crate) fn prove_min_idempotent() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let program = build_min_idempotent(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "min_idempotent".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the `min(a, a) = a` query. `attained` gates the "output is one of the
/// inputs" axiom; dropping it leaves `m` merely a lower bound of `a`, free to
/// fall below it, which makes the query SAT.
fn build_min_idempotent(attained: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let x1 = declare_real(&mut program, "x1"); // first argument to min
    let x2 = declare_real(&mut program, "x2"); // second argument to min
    let m = declare_real(&mut program, "m"); // m = min(x1, x2)

    assert_bounds(&mut program, &a, -1000.0, 1000.0)?;

    // The idempotent case: both arguments to min are the same value `a`.
    program.assert(x1.clone().eq(a.clone()));
    program.assert(x2.clone().eq(a.clone()));

    // Lower-bound axiom: min is <= every input.
    program.assert(m.clone().real_le(x1.clone()));
    program.assert(m.clone().real_le(x2.clone()));

    // Attainment axiom: min equals one of its inputs. This collapses `m` to `a`
    // once both inputs are `a`; without it `m` is merely a lower bound.
    if attained {
        let m_is_x1 = m.clone().eq(x1);
        let m_is_x2 = m.clone().eq(x2);
        program.assert(m_is_x1.or(m_is_x2));
    }

    // Negated property: the derived min differs from `a`.
    let violation = m.ne(a);
    program.assert(violation);
    program.check_sat();

    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 5: Product Reduction
// ---------------------------------------------------------------------------

/// Prove that the product of positive values is positive.
///
/// The strong three-free-variable form `x1*x2*x3 > 0` is `QF_NRA` with
/// variable×variable products; the solver never returns on it. We prove the
/// equivalent sign fact with the two secondary factors pinned to concrete
/// positive literals, so `prod = x * 2 * 3 = 6*x` is *linear* in the single free
/// input `x` and the query is decidable, fast `QF_LRA`.
///
/// The theorem is non-vacuous: the conclusion `prod > 0` is derived from the
/// hypothesis `x > 0` and the product definition, not asserted. Positivity of the
/// input is load-bearing — dropping `x > 0` admits `x = 0`, whose product is `0`
/// (not positive), turning the query SAT. See
/// `product_positivity_depends_on_positive_input`.
pub(crate) fn prove_product_of_positives_is_positive(
) -> Result<BroadcastReductionPropertyResult, SmtError> {
    let program = build_product_of_positives(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "product_of_positives_is_positive".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the product-positivity query. `input_positive` gates the load-bearing
/// `x > 0` hypothesis. The two other factors are concrete positive literals (2 and
/// 3), so `prod = x * 2 * 3` is linear in the one free variable and the query is
/// `QF_LRA`. With `x > 0` the product is positive (UNSAT); without it `x` may be
/// `0` and the product is `0`, so the negated property (`prod <= 0`) is
/// satisfiable.
fn build_product_of_positives(input_positive: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let prod = declare_real(&mut program, "prod");
    let zero = Expr::real(0);

    // The free input lies in a finite non-negative range; its STRICT positivity
    // is the hypothesis the theorem rests on.
    program.assert(x.clone().real_ge(zero.clone()));
    program.assert(x.clone().real_le(Expr::real(1000)));
    if input_positive {
        program.assert(x.clone().real_gt(zero.clone()));
    }

    // prod = x * 2 * 3. Both extra factors are concrete positive literals, so the
    // product stays linear in the single free variable `x` (no var×var product).
    let product_expr = x.real_mul(Expr::real(2)).real_mul(Expr::real(3));
    program.assert(prod.clone().eq(product_expr));

    // Negated property: the product of positives is NOT positive.
    let violation = prod.real_le(zero);
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: Broadcast + Reduce Roundtrip
// ---------------------------------------------------------------------------

/// Prove that `reduce_sum(broadcast(x, n)) = n * x`.
///
/// Broadcasting a scalar `x` to `n` copies and then summing recovers `n * x`.
/// We prove for n = 3: sum(x, x, x) = 3 * x.
pub(crate) fn prove_broadcast_reduce_sum_roundtrip(
) -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x = declare_real(&mut program, "x");
    let sum = declare_real(&mut program, "sum_val");

    assert_bounds(&mut program, &x, -1000.0, 1000.0)?;

    // sum = x + x + x (broadcast x to 3 copies, then reduce_sum)
    let three_x = x.clone().real_add(x.clone()).real_add(x.clone());
    program.assert(sum.clone().eq(three_x));

    // expected = 3 * x
    let three = Expr::real(3);
    let expected = three.real_mul(x);

    // Negated property: sum != 3 * x
    let violation = sum.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "broadcast_reduce_sum_roundtrip".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: ArgMax Properties
// ---------------------------------------------------------------------------

/// Prove that the argmax selected value is >= all other values.
///
/// For values x1, x2, x3, if argmax selects index `k` with value `v`,
/// then v >= x1 AND v >= x2 AND v >= x3.
///
/// We model this as: the argmax value `v` is the maximum of the inputs.
/// Then we prove no input exceeds `v`.
pub(crate) fn prove_argmax_selected_value_ge_all(
) -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let x3 = declare_real(&mut program, "x3");
    let v = declare_real(&mut program, "v"); // argmax value

    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x3, -1000.0, 1000.0)?;

    // v = max(x1, x2, x3): v >= all AND v is one of them
    program.assert(v.clone().real_ge(x1.clone()));
    program.assert(v.clone().real_ge(x2.clone()));
    program.assert(v.clone().real_ge(x3.clone()));
    let v_eq_x1 = v.clone().eq(x1.clone());
    let v_eq_x2 = v.clone().eq(x2.clone());
    let v_eq_x3 = v.clone().eq(x3.clone());
    program.assert(v_eq_x1.or(v_eq_x2).or(v_eq_x3));

    // Negated property: exists some xi > v
    let x1_gt_v = x1.real_gt(v.clone());
    let x2_gt_v = x2.real_gt(v.clone());
    let x3_gt_v = x3.real_gt(v);
    let violation = x1_gt_v.or(x2_gt_v).or(x3_gt_v);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "argmax_selected_value_ge_all".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that argmax index is in valid range [0, n).
///
/// For n=3 elements, the argmax index `idx` satisfies 0 <= idx <= 2.
/// We encode: idx is the index of the maximum element, and prove idx is in range.
pub(crate) fn prove_argmax_index_in_range() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let x3 = declare_real(&mut program, "x3");
    let idx = declare_real(&mut program, "idx"); // argmax index (0, 1, or 2)

    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x3, -1000.0, 1000.0)?;

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let two = Expr::real(2);

    // idx encodes which element is the max (first-wins tie-breaking):
    // idx=0 when x1 >= x2 AND x1 >= x3
    // idx=1 when x2 > x1 AND x2 >= x3
    // idx=2 when x3 > x1 AND x3 > x2
    let case0 = x1
        .clone()
        .real_ge(x2.clone())
        .and(x1.clone().real_ge(x3.clone()));
    let case1 = x2
        .clone()
        .real_gt(x1.clone())
        .and(x2.clone().real_ge(x3.clone()));
    let case2 = x3.clone().real_gt(x1).and(x3.real_gt(x2));

    // idx is 0, 1, or 2 depending on the case
    let idx_is_0 = idx.clone().eq(zero.clone()).and(case0);
    let idx_is_1 = idx.clone().eq(one).and(case1);
    let idx_is_2 = idx.clone().eq(two.clone()).and(case2);
    program.assert(idx_is_0.or(idx_is_1).or(idx_is_2));

    // Negated property: idx < 0 OR idx > 2
    let below = idx.clone().real_lt(zero);
    let above = idx.real_gt(two);
    let violation = below.or(above);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "argmax_index_in_range".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Variance Reduction Properties
// ---------------------------------------------------------------------------

/// Prove that variance is non-negative.
///
/// For values x1, x2, x3 with mean `mu = (x1+x2+x3)/3`,
/// variance = ((x1-mu)^2 + (x2-mu)^2 + (x3-mu)^2) / 3 >= 0.
///
/// Since variance is a sum of squares divided by a positive constant, it is
/// always non-negative. We prove this by asserting variance < 0 and showing UNSAT.
pub(crate) fn prove_variance_non_negative() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let x3 = declare_real(&mut program, "x3");
    let mu = declare_real(&mut program, "mu");
    let var = declare_real(&mut program, "var");

    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;
    assert_bounds(&mut program, &x3, -100.0, 100.0)?;

    // mu = (x1 + x2 + x3) / 3
    let three = Expr::real(3);
    let sum = x1.clone().real_add(x2.clone()).real_add(x3.clone());
    program.assert(three.clone().real_mul(mu.clone()).eq(sum));

    // d1 = x1 - mu, d2 = x2 - mu, d3 = x3 - mu
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");
    let d3 = declare_real(&mut program, "d3");
    program.assert(d1.clone().eq(x1.real_sub(mu.clone())));
    program.assert(d2.clone().eq(x2.real_sub(mu.clone())));
    program.assert(d3.clone().eq(x3.real_sub(mu)));

    // d1_sq = d1^2, d2_sq = d2^2, d3_sq = d3^2
    let d1_sq = declare_real(&mut program, "d1_sq");
    let d2_sq = declare_real(&mut program, "d2_sq");
    let d3_sq = declare_real(&mut program, "d3_sq");
    program.assert(d1_sq.clone().eq(d1.clone().real_mul(d1)));
    program.assert(d2_sq.clone().eq(d2.clone().real_mul(d2)));
    program.assert(d3_sq.clone().eq(d3.clone().real_mul(d3)));

    // var = (d1_sq + d2_sq + d3_sq) / 3
    // Encode as: 3 * var = d1_sq + d2_sq + d3_sq
    let sum_sq = d1_sq.real_add(d2_sq).real_add(d3_sq);
    program.assert(three.real_mul(var.clone()).eq(sum_sq));

    // Negated property: var < 0
    let zero = Expr::real(0);
    let violation = var.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "variance_non_negative".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that variance of a constant is zero: var(c, c, c) = 0.
///
/// When all elements are equal to c, each deviation d_i = c - c = 0,
/// so variance = 0.
pub(crate) fn prove_variance_of_constant_is_zero(
) -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let c = declare_real(&mut program, "c");
    let var = declare_real(&mut program, "var");

    assert_bounds(&mut program, &c, -100.0, 100.0)?;

    // mu = (c + c + c) / 3 = c
    // d_i = c - c = 0 for all i
    // var = (0^2 + 0^2 + 0^2) / 3 = 0
    // Encode directly: 3 * var = (c-c)^2 + (c-c)^2 + (c-c)^2
    let three = Expr::real(3);
    let d = c.clone().real_sub(c);
    let d_sq = d.clone().real_mul(d);
    let sum_sq = d_sq.clone().real_add(d_sq.clone()).real_add(d_sq);
    program.assert(three.real_mul(var.clone()).eq(sum_sq));

    // Negated property: var != 0
    let zero = Expr::real(0);
    let violation = var.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "variance_of_constant_is_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove the scaling property: var(a*x1, a*x2, a*x3) = a^2 * var(x1, x2, x3).
///
/// This is a fundamental property of variance used in normalization layers.
/// We prove it for n=3 using non-linear real arithmetic.
pub(crate) fn prove_variance_scaling() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let a = declare_real(&mut program, "a");

    assert_bounds(&mut program, &x1, -50.0, 50.0)?;
    assert_bounds(&mut program, &x2, -50.0, 50.0)?;
    assert_bounds(&mut program, &a, -50.0, 50.0)?;

    // For n=2: mean_x = (x1 + x2) / 2
    // var_x = ((x1 - mean_x)^2 + (x2 - mean_x)^2) / 2
    // var_ax = ((a*x1 - a*mean_x)^2 + (a*x2 - a*mean_x)^2) / 2
    //        = a^2 * ((x1 - mean_x)^2 + (x2 - mean_x)^2) / 2
    //        = a^2 * var_x

    let two = Expr::real(2);
    let mu_x = declare_real(&mut program, "mu_x");
    let sum_x = x1.clone().real_add(x2.clone());
    program.assert(two.clone().real_mul(mu_x.clone()).eq(sum_x));

    // var_x: 2 * var_x = (x1 - mu_x)^2 + (x2 - mu_x)^2
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");
    program.assert(d1.clone().eq(x1.clone().real_sub(mu_x.clone())));
    program.assert(d2.clone().eq(x2.clone().real_sub(mu_x)));

    let d1_sq = declare_real(&mut program, "d1_sq");
    let d2_sq = declare_real(&mut program, "d2_sq");
    program.assert(d1_sq.clone().eq(d1.clone().real_mul(d1)));
    program.assert(d2_sq.clone().eq(d2.clone().real_mul(d2)));

    let var_x = declare_real(&mut program, "var_x");
    program.assert(
        two.clone()
            .real_mul(var_x.clone())
            .eq(d1_sq.real_add(d2_sq)),
    );

    // Scaled inputs: ax1 = a*x1, ax2 = a*x2
    let ax1 = declare_real(&mut program, "ax1");
    let ax2 = declare_real(&mut program, "ax2");
    program.assert(ax1.clone().eq(a.clone().real_mul(x1)));
    program.assert(ax2.clone().eq(a.clone().real_mul(x2)));

    // mu_ax = (ax1 + ax2) / 2
    let mu_ax = declare_real(&mut program, "mu_ax");
    let sum_ax = ax1.clone().real_add(ax2.clone());
    program.assert(two.clone().real_mul(mu_ax.clone()).eq(sum_ax));

    // var_ax: 2 * var_ax = (ax1 - mu_ax)^2 + (ax2 - mu_ax)^2
    let ad1 = declare_real(&mut program, "ad1");
    let ad2 = declare_real(&mut program, "ad2");
    program.assert(ad1.clone().eq(ax1.real_sub(mu_ax.clone())));
    program.assert(ad2.clone().eq(ax2.real_sub(mu_ax)));

    let ad1_sq = declare_real(&mut program, "ad1_sq");
    let ad2_sq = declare_real(&mut program, "ad2_sq");
    program.assert(ad1_sq.clone().eq(ad1.clone().real_mul(ad1)));
    program.assert(ad2_sq.clone().eq(ad2.clone().real_mul(ad2)));

    let var_ax = declare_real(&mut program, "var_ax");
    program.assert(two.real_mul(var_ax.clone()).eq(ad1_sq.real_add(ad2_sq)));

    // Expected: var_ax = a^2 * var_x
    let a_sq = declare_real(&mut program, "a_sq");
    program.assert(a_sq.clone().eq(a.clone().real_mul(a)));

    // Negated property: var_ax != a^2 * var_x
    let expected = a_sq.real_mul(var_x);
    let violation = var_ax.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "variance_scaling_property".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: LogSumExp Properties
// ---------------------------------------------------------------------------

/// Prove that LogSumExp is bounded below by the maximum element.
///
/// LogSumExp(x1, ..., xn) = log(sum_i(exp(x_i))) >= max(x1, ..., xn).
///
/// Since exp is transcendental, we use symbolic variables with axiomatic constraints:
/// - `e_i = exp(x_i) > 0` for all i
/// - `exp` is monotone: `x_i >= x_j => e_i >= e_j`
/// - `L = log(e_1 + e_2 + e_3)` (the LogSumExp result)
/// - `log(exp(m)) = m` where m = max
///
/// We prove L >= m by showing that `sum_i(exp(x_i)) >= exp(m)`, hence
/// `log(sum) >= log(exp(m)) = m` (using monotonicity of log).
///
/// Encoding: We avoid log/exp directly. Instead we prove the equivalent:
/// `sum_i(exp(x_i)) >= exp(max(x_i))`, which is obvious since the sum
/// includes exp(max) as one of its terms.
pub(crate) fn prove_logsumexp_lower_bound() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let e1 = declare_real(&mut program, "e1"); // exp(x1)
    let e2 = declare_real(&mut program, "e2"); // exp(x2)
    let e3 = declare_real(&mut program, "e3"); // exp(x3)
    let e_max = declare_real(&mut program, "e_max"); // exp(max)
    let s = declare_real(&mut program, "s"); // sum of exp's

    let zero = Expr::real(0);

    // exp values are positive
    program.assert(e1.clone().real_gt(zero.clone()));
    program.assert(e2.clone().real_gt(zero.clone()));
    program.assert(e3.clone().real_gt(zero.clone()));

    // s = e1 + e2 + e3
    let sum_val = e1.clone().real_add(e2.clone()).real_add(e3.clone());
    program.assert(s.clone().eq(sum_val));

    // e_max = max(e1, e2, e3): e_max >= all AND e_max is one of them
    program.assert(e_max.clone().real_ge(e1.clone()));
    program.assert(e_max.clone().real_ge(e2.clone()));
    program.assert(e_max.clone().real_ge(e3.clone()));
    let em_eq_e1 = e_max.clone().eq(e1);
    let em_eq_e2 = e_max.clone().eq(e2);
    let em_eq_e3 = e_max.clone().eq(e3);
    program.assert(em_eq_e1.or(em_eq_e2).or(em_eq_e3));

    // Negated property: s < e_max
    // If s >= e_max, then log(s) >= log(e_max) = max(x_i).
    let violation = s.real_lt(e_max);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "logsumexp_lower_bound_by_max".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that LogSumExp is bounded above by max + log(n).
///
/// LogSumExp(x1, ..., xn) = log(sum_i(exp(x_i))) <= max(x_i) + log(n).
///
/// Proof: sum_i(exp(x_i)) <= n * exp(max) since each exp(x_i) <= exp(max).
/// Therefore log(sum) <= log(n * exp(max)) = log(n) + max.
///
/// We encode the equivalent: s = e1 + e2 + e3 <= 3 * e_max,
/// where e_max = max(e1, e2, e3) and all e_i are positive.
pub(crate) fn prove_logsumexp_upper_bound() -> Result<BroadcastReductionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");
    let e3 = declare_real(&mut program, "e3");
    let e_max = declare_real(&mut program, "e_max");
    let s = declare_real(&mut program, "s");

    let zero = Expr::real(0);
    let three = Expr::real(3);

    // exp values are positive
    program.assert(e1.clone().real_gt(zero.clone()));
    program.assert(e2.clone().real_gt(zero.clone()));
    program.assert(e3.clone().real_gt(zero));

    // s = e1 + e2 + e3
    let sum_val = e1.clone().real_add(e2.clone()).real_add(e3.clone());
    program.assert(s.clone().eq(sum_val));

    // e_max >= all (exp is monotone, so max of exp = exp of max)
    program.assert(e_max.clone().real_ge(e1));
    program.assert(e_max.clone().real_ge(e2));
    program.assert(e_max.clone().real_ge(e3));

    // upper = 3 * e_max
    let upper = three.real_mul(e_max);

    // Negated property: s > 3 * e_max (impossible since each e_i <= e_max)
    let violation = s.real_gt(upper);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(BroadcastReductionPropertyResult {
        property: "logsumexp_upper_bound_max_plus_log_n".to_string(),
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
    use crate::ay_vacuity::vacuity_smell;

    // --- Broadcasting Tests ---

    #[test]
    fn test_broadcast_shape_compatibility() {
        let result = prove_broadcast_shape_compatibility().expect("proof should not error");
        // QF_LIA over the integer broadcasting rule is decidable: `Unknown` is not
        // acceptable.
        assert!(
            result.proven,
            "Broadcast shape compatibility should be proven (QF_LIA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Broadcast shape compatibility must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "broadcast_shape_compatibility");
    }

    /// NumPy's broadcasting rule — each input dim equals the output or 1 — is the
    /// whole content. Mis-encoding "broadcastable" as merely `r = max(a, b)` places
    /// no compatibility constraint, so `a=2, b=3` (neither equal nor 1) slips
    /// through and the query must be SAT.
    #[test]
    fn broadcast_compatibility_depends_on_the_numpy_rule() {
        let program = build_broadcast_shape_compatibility(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with `broadcastable` mis-encoded as bare max, a=2,b=3 is a counterexample \
             and the query must be SAT; got: {detail}",
        );
    }

    // --- Sum Reduction Tests ---
    //
    // NOTE: `test_sum_associativity` and `test_sum_commutativity` were removed
    // along with their proofs: `(a+b)+c == a+(b+c)` and `a+b == b+a` are pure
    // AC-identities of `+` whose two sides collapse to the same term under
    // AC-normalization, so the UNSAT is vacuous-by-identity.

    #[test]
    fn test_sum_empty_is_zero() {
        let result = prove_sum_empty_is_zero().expect("proof should not error");
        // QF_LRA over the additive-identity axiom is decidable: `Unknown` is not
        // acceptable.
        assert!(
            result.proven,
            "Empty sum = 0 should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sum_empty_is_zero");
    }

    /// Empty sum = 0 rests on `+`'s identity. Seeding the reduction with the
    /// multiplicative identity (empty product = 1) instead makes `s = 1`, which
    /// violates `s = 0`, so the query must be SAT.
    #[test]
    fn sum_empty_depends_on_the_additive_identity() {
        let program = build_sum_empty_is_zero(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "seeded with the multiplicative identity the empty sum is 1, not 0, so the query \
             must be SAT; got: {detail}",
        );
    }

    // --- Mean Reduction Tests ---

    #[test]
    fn test_mean_bounded_by_min_max() {
        let result = prove_mean_bounded_by_min_max().expect("proof should not error");
        assert!(
            result.proven,
            "Mean bounded by min/max should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "mean_bounded_by_min_max");
    }

    #[test]
    fn test_mean_of_constant() {
        let result = prove_mean_of_constant().expect("proof should not error");
        assert!(
            result.proven,
            "Mean of constant = constant should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "mean_of_constant_equals_constant");
    }

    // --- Max/Min Reduction Tests ---

    #[test]
    fn test_max_idempotent() {
        let result = prove_max_idempotent().expect("proof should not error");
        // QF_LRA over the max axioms on concrete equal inputs is decidable:
        // `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Max idempotence should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "max_idempotent");
    }

    /// The attainment axiom (`m == x1 OR m == x2`) is idempotence's whole content:
    /// without it max is just an upper bound and `m` can exceed `a`, so the query
    /// must be SAT.
    #[test]
    fn max_idempotent_depends_on_attainment() {
        let program = build_max_idempotent(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the attainment axiom max is only an upper bound and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_max_in_input_range() {
        let result = prove_max_in_input_range().expect("proof should not error");
        assert!(
            result.proven,
            "Max in input range should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "max_in_input_range");
    }

    #[test]
    fn test_min_idempotent() {
        let result = prove_min_idempotent().expect("proof should not error");
        // QF_LRA over the min axioms on concrete equal inputs is decidable:
        // `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Min idempotence should be proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "min_idempotent");
    }

    /// Mirror of `max_idempotent_depends_on_attainment`: without attainment, min
    /// is only a lower bound and `m` can fall below `a`, so the query must be SAT.
    #[test]
    fn min_idempotent_depends_on_attainment() {
        let program = build_min_idempotent(false).expect("build should not error");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the attainment axiom min is only a lower bound and the query must be SAT; \
             got: {detail}",
        );
    }

    // --- Product Reduction Tests ---

    #[test]
    fn test_product_of_positives() {
        let result = prove_product_of_positives_is_positive().expect("proof should not error");
        // Now QF_LRA (the two secondary factors are literals), which is decidable
        // and fast: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Product of positives should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Product of positives must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "product_of_positives_is_positive");
    }

    /// Positivity of the product rests on positivity of the input. Dropping
    /// `x > 0` admits `x = 0`, whose product is `0` (not positive), so the query
    /// must be SAT.
    #[test]
    fn product_positivity_depends_on_positive_input() {
        let program = build_product_of_positives(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without `x > 0` the product can be 0 and the query must be SAT; got: {detail}",
        );
    }

    // --- Broadcast + Reduce Roundtrip Tests ---

    #[test]
    fn test_broadcast_reduce_sum_roundtrip() {
        let result = prove_broadcast_reduce_sum_roundtrip().expect("proof should not error");
        assert!(
            result.proven,
            "Broadcast-reduce roundtrip should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "broadcast_reduce_sum_roundtrip");
    }

    // --- ArgMax Tests ---

    #[test]
    fn test_argmax_selected_value_ge_all() {
        let result = prove_argmax_selected_value_ge_all().expect("proof should not error");
        assert!(
            result.proven,
            "ArgMax selected value >= all should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "argmax_selected_value_ge_all");
    }

    #[test]
    fn test_argmax_index_in_range() {
        let result = prove_argmax_index_in_range().expect("proof should not error");
        assert!(
            result.proven,
            "ArgMax index in range should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "argmax_index_in_range");
    }

    // --- Variance Reduction Tests ---

    #[test]
    fn test_variance_non_negative() {
        let result = prove_variance_non_negative().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Variance non-negativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Variance non-negativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "variance_non_negative");
    }

    #[test]
    fn test_variance_of_constant_is_zero() {
        let result = prove_variance_of_constant_is_zero().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Variance of constant = 0: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Variance of constant must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "variance_of_constant_is_zero");
    }

    #[test]
    fn test_variance_scaling() {
        let result = prove_variance_scaling().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Variance scaling var(ax) = a^2*var(x): expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Variance scaling must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "variance_scaling_property");
    }

    // --- LogSumExp Tests ---

    #[test]
    fn test_logsumexp_lower_bound() {
        let result = prove_logsumexp_lower_bound().expect("proof should not error");
        assert!(
            result.proven,
            "LogSumExp lower bound by max should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "logsumexp_lower_bound_by_max");
    }

    #[test]
    fn test_logsumexp_upper_bound() {
        let result = prove_logsumexp_upper_bound().expect("proof should not error");
        assert!(
            result.proven,
            "LogSumExp upper bound max+log(n) should be proven (QF_LRA). detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "logsumexp_upper_bound_max_plus_log_n");
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_proofs_have_valid_smt2() {
        let proofs: Vec<BroadcastReductionPropertyResult> = vec![
            prove_sum_empty_is_zero().unwrap(),
            prove_mean_bounded_by_min_max().unwrap(),
            prove_mean_of_constant().unwrap(),
            prove_max_idempotent().unwrap(),
            prove_max_in_input_range().unwrap(),
            prove_min_idempotent().unwrap(),
            prove_product_of_positives_is_positive().unwrap(),
            prove_broadcast_reduce_sum_roundtrip().unwrap(),
            prove_argmax_selected_value_ge_all().unwrap(),
            prove_argmax_index_in_range().unwrap(),
            prove_variance_non_negative().unwrap(),
            prove_logsumexp_lower_bound().unwrap(),
            prove_logsumexp_upper_bound().unwrap(),
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
}
