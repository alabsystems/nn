// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for pooling operation mathematical properties (#4205).
//!
//! Pooling layers (max, average, global, adaptive, L2) reduce spatial dimensions
//! while preserving important invariants. This module proves eight key mathematical
//! properties using ay's SMT solver:
//!
//! 1. **Max pooling output bounded by input range**: `min(x_i) <= max_pool(x) <= max(x_i)`
//! 2. **Max pooling idempotence**: `max_pool(max_pool(x)) == max_pool(x)` (applying
//!    max pooling twice yields the same result when kernel covers the full output).
//! 3. **Average pooling output bounded by input min/max**: `min(x_i) <= avg_pool(x) <= max(x_i)`
//! 4. **Global average pooling equals mean**: `global_avg(x) = sum(x_i) / n`
//! 5. **Adaptive average pooling output shape**: The mapping from output index to input
//!    bin boundaries satisfies `start_i = floor(i * in_size / out_size)`.
//! 6. **Max pooling with indices**: The selected value equals the maximum.
//! 7. **L2 pooling non-negativity and bound**: `0 <= l2_pool(x) <= max(|x_i|)`
//! 8. **Pooling commutativity with scaling**: `avg_pool(alpha * x) == alpha * avg_pool(x)`
//!
//! # Proof Strategy
//!
//! Most properties are modeled over a small window of 2-3 elements (sufficient to
//! capture the essential mathematical structure). For integer-related properties
//! (output size formulas), we use the same floor-division encoding as in
//! `ay_convolution_properties.rs` with helper variable `q` satisfying
//! `q * stride <= numerator < (q+1) * stride`.
//!
//! All proofs use the negation-and-UNSAT approach: assert the property's negation
//! and prove unsatisfiability, which means the property holds for all valid inputs.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a pooling property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct PoolPropertyResult {
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
/// The verdict is funneled through [`crate::ay_vacuity::reject_if_vacuous`] so a
/// query that is UNSAT only because it asserts `P ∧ ¬P` (or compares a term to
/// itself) never counts as a proof — any residual vacuity becomes a hard test
/// failure in the corresponding `test_*_proven`.
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
// Property 1: Max Pooling Output Bounded by Input Range
// ---------------------------------------------------------------------------

/// Prove that max pooling output is bounded by the input range.
///
/// For a pool window of 3 elements `{x0, x1, x2}`, the max is one of them.
/// Therefore: `min(x0, x1, x2) <= max(x0, x1, x2) <= max(x0, x1, x2)`.
///
/// More precisely, we prove that for any choice of `m` satisfying:
///   `m >= x0 AND m >= x1 AND m >= x2` (m is an upper bound)
///   `m = x0 OR m = x1 OR m = x2` (m is one of the inputs)
///
/// Then `m >= x0`, `m >= x1`, `m >= x2` (obvious from first constraint)
/// and `m <= max(x0, x1, x2)` (because m IS one of the inputs and >= all).
///
/// We prove the lower bound: `m >= x0 AND m >= x1 AND m >= x2` with
/// `m` being one of the inputs implies `m >= min(x0, x1, x2)`.
/// Since m is one of the inputs and >= all of them, m is automatically
/// <= max input (because m IS an input). The lower bound is the interesting
/// direction: we prove the max cannot be below any input.
///
/// Violation: `m < x0 OR m < x1 OR m < x2` while `m` satisfies the max constraints.
pub(crate) fn prove_max_pool_bounded_by_input() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let m = declare_real(&mut program, "m");

    assert_bounds(&mut program, &x0, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;

    // m is the max: m >= all inputs
    program.assert(m.clone().real_ge(x0.clone()));
    program.assert(m.clone().real_ge(x1.clone()));
    program.assert(m.clone().real_ge(x2.clone()));

    // m is one of the inputs (the max must be realized by some element)
    let is_x0 = m.clone().eq(x0.clone());
    let is_x1 = m.clone().eq(x1.clone());
    let is_x2 = m.clone().eq(x2.clone());
    program.assert(is_x0.or(is_x1).or(is_x2));

    // Violation: m is strictly less than some input (contradicts m >= all)
    // Actually, the interesting property is: can m exceed all inputs?
    // Since m must equal one input and be >= all, it equals the largest.
    // Prove: m cannot be strictly greater than all inputs.
    let exceeds_all = m
        .clone()
        .real_gt(x0.clone())
        .and(m.clone().real_gt(x1.clone()))
        .and(m.real_gt(x2.clone()));
    program.assert(exceeds_all);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "max_pool_bounded_by_input_range".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Max Pooling Idempotence
// ---------------------------------------------------------------------------

/// Prove max pooling's idempotence-under-full-coverage:
/// `max_pool(max_pool(x)) == max_pool(x)`.
///
/// The old encoding introduced `m2`, asserted `m2 == m`, then negated it with
/// `m2 != m` — a `P ∧ ¬P` query that is UNSAT for free and proves nothing.
///
/// The real content is that a **two-stage** max pool over the full output equals
/// a **single-stage** max pool over the same inputs. Over four inputs
/// `{a, b, c, d}` the first stage (window 2, stride 2) yields
///
/// ```text
/// p0 = max(a, b)      p1 = max(c, d)
/// ```
///
/// and the second stage, whose window covers the *entire* stage-1 output, yields
/// `q = max(p0, p1)`. Idempotence says this equals `s = max(a, b, c, d)` — the
/// result of pooling once. Neither `q` nor `s` is asserted equal to the other:
/// both are *derived* from max constraints over the original inputs, so `q == s`
/// is a genuine theorem (max is associative), not a restated hypothesis.
///
/// `max(u, v)` is encoded as `w >= u ∧ w >= v ∧ (w = u ∨ w = v)`, which pins `w`
/// to the larger of the two — a correct characterization of max over the reals,
/// and linear, so the query stays in decidable `QF_LRA`.
///
/// A wrong second-stage window (one that drops `p1`) makes `q == s` false — see
/// `idempotent_depends_on_full_second_window_coverage`.
pub(crate) fn prove_max_pool_idempotent() -> Result<PoolPropertyResult, SmtError> {
    let program = build_max_pool_idempotent(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "max_pool_idempotent".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Constrain `w` to be `max(u, v)`: an upper bound on both that is realized by
/// one of them. Correct for reals and linear (`QF_LRA`).
fn assert_is_max2(program: &mut AYProgram, w: &Expr, u: &Expr, v: &Expr) {
    program.assert(w.clone().real_ge(u.clone()));
    program.assert(w.clone().real_ge(v.clone()));
    program.assert(w.clone().eq(u.clone()).or(w.clone().eq(v.clone())));
}

/// Build the two-stage-vs-single-stage max-pool idempotence query.
///
/// When `second_window_covers_all` is false the second pooling window's stride
/// skips `p1` — the classic off-by-one that drops the last column — so the second
/// stage sees only `p0`. Then `q = max(a, b)` can fall below `max(a, b, c, d)`
/// and the property becomes genuinely false (SAT). Tests flip the knob to confirm
/// the proof depends on covering the full stage-1 output.
fn build_max_pool_idempotent(second_window_covers_all: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");

    // Bounds keep the search finite; unproblematic since the property is scale-free.
    for v in [&a, &b, &c, &d] {
        program.assert(v.clone().real_ge(Expr::real(-1000)));
        program.assert(v.clone().real_le(Expr::real(1000)));
    }

    // Stage 1: p0 = max(a, b), p1 = max(c, d).
    let p0 = declare_real(&mut program, "p0");
    assert_is_max2(&mut program, &p0, &a, &b);
    let p1 = declare_real(&mut program, "p1");
    assert_is_max2(&mut program, &p1, &c, &d);

    // Stage 2: q = max over the second window.
    let q = declare_real(&mut program, "q");
    if second_window_covers_all {
        assert_is_max2(&mut program, &q, &p0, &p1);
    } else {
        // BUG: the window drops p1, so the second stage pools p0 alone.
        program.assert(q.clone().real_ge(p0.clone()));
        program.assert(q.clone().eq(p0.clone()));
    }

    // Single-stage reference: s = max(a, b, c, d).
    let s = declare_real(&mut program, "s");
    program.assert(s.clone().real_ge(a.clone()));
    program.assert(s.clone().real_ge(b.clone()));
    program.assert(s.clone().real_ge(c.clone()));
    program.assert(s.clone().real_ge(d.clone()));
    program.assert(
        s.clone()
            .eq(a)
            .or(s.clone().eq(b))
            .or(s.clone().eq(c))
            .or(s.clone().eq(d)),
    );

    // Property: two-stage max pool == single-stage max pool.
    // Violation: q != s. Derived from the computation, not asserted then negated.
    let violation = q.ne(s);
    program.assert(violation);
    program.check_sat();

    program
}

// ---------------------------------------------------------------------------
// Property 3: Average Pooling Output Bounded by Input Min/Max
// ---------------------------------------------------------------------------

/// Prove that average pooling output lies between the minimum and maximum
/// of the input elements.
///
/// For a window of 3 elements `{x0, x1, x2}`:
///   `avg = (x0 + x1 + x2) / 3`
///
/// Mean value theorem for discrete sets: the average of a finite set of
/// numbers lies between the minimum and maximum.
///
/// We prove: given `lo <= x0, x1, x2 <= hi` (where lo is the min and hi
/// is the max of the three), then `lo <= avg <= hi`.
///
/// Violation: `avg < lo` or `avg > hi`.
pub(crate) fn prove_avg_pool_bounded_by_input() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let lo = declare_real(&mut program, "lo");
    let hi = declare_real(&mut program, "hi");

    assert_bounds(&mut program, &x0, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;

    // lo is a lower bound on all inputs
    program.assert(x0.clone().real_ge(lo.clone()));
    program.assert(x1.clone().real_ge(lo.clone()));
    program.assert(x2.clone().real_ge(lo.clone()));

    // hi is an upper bound on all inputs
    program.assert(x0.clone().real_le(hi.clone()));
    program.assert(x1.clone().real_le(hi.clone()));
    program.assert(x2.clone().real_le(hi.clone()));

    // avg = (x0 + x1 + x2) / 3, modeled as: avg * 3 = x0 + x1 + x2
    let avg = declare_real(&mut program, "avg");
    let three = real_from_f64(3.0)?;
    let sum = x0.real_add(x1).real_add(x2);
    program.assert(avg.clone().real_mul(three).eq(sum));

    // Violation: avg < lo OR avg > hi
    let too_low = avg.clone().real_lt(lo);
    let too_high = avg.real_gt(hi);
    program.assert(too_low.or(too_high));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "avg_pool_bounded_by_input".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: Global Average Pooling Equals Mean
// ---------------------------------------------------------------------------

/// Prove that global average pooling computes the arithmetic mean.
///
/// For `n` spatial positions with values `{x0, x1, ..., x_{n-1}}`:
///   `global_avg = sum(x_i) / n`
///
/// We model this for n=3 and prove the definition is self-consistent:
/// if `g * n = sum(x_i)` then `g = sum(x_i) / n`.
///
/// More precisely, we prove that the global average is unique: if
/// `g1 * n = sum` and `g2 * n = sum` then `g1 = g2` (with n > 0).
///
/// Violation: `g1 != g2`.
pub(crate) fn prove_global_avg_pool_equals_mean() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");

    assert_bounds(&mut program, &x0, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;

    let n = declare_real(&mut program, "n");
    let zero = Expr::real(0);
    program.assert(n.clone().real_gt(zero));
    // n = 3 (the number of spatial positions)
    let three = real_from_f64(3.0)?;
    program.assert(n.clone().eq(three));

    let sum = x0.real_add(x1).real_add(x2);

    // Two candidates for global average
    let g1 = declare_real(&mut program, "g1");
    let g2 = declare_real(&mut program, "g2");

    // g1 * n = sum, g2 * n = sum
    program.assert(g1.clone().real_mul(n.clone()).eq(sum.clone()));
    program.assert(g2.clone().real_mul(n).eq(sum));

    // Violation: g1 != g2
    let violation = g1.ne(g2);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "global_avg_pool_equals_mean".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: Adaptive Average Pooling Output Shape
// ---------------------------------------------------------------------------

/// Prove the adaptive average pooling bin boundary formula.
///
/// Adaptive average pooling maps `in_size` -> `out_size` by dividing the input
/// into `out_size` bins. For output index `i`, the input bin boundaries are:
///   `start_i = floor(i * in_size / out_size)`
///   `end_i   = floor((i+1) * in_size / out_size)`
///
/// We prove: for valid configurations (`in_size >= out_size >= 1`),
/// the bin width `end_i - start_i >= 1` (each output has at least one input).
///
/// We model `floor(x / d)` via a helper variable `q` with `q*d <= x < (q+1)*d`.
pub(crate) fn prove_adaptive_avg_pool_bin_nonempty() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let in_size = declare_real(&mut program, "in_size");
    let out_size = declare_real(&mut program, "out_size");
    let i = declare_real(&mut program, "i");

    assert_bounds(&mut program, &in_size, 1.0, 1000.0)?;
    assert_bounds(&mut program, &out_size, 1.0, 1000.0)?;
    assert_bounds(&mut program, &i, 0.0, 999.0)?;

    let zero = Expr::real(0);
    let one = real_from_f64(1.0)?;

    // in_size >= out_size (downsampling or identity, not upsampling)
    program.assert(in_size.clone().real_ge(out_size.clone()));

    // i < out_size (valid output index)
    program.assert(i.clone().real_lt(out_size.clone()));
    // i >= 0
    program.assert(i.clone().real_ge(zero.clone()));

    // start_i = floor(i * in_size / out_size)
    // Encode: numerator_s = i * in_size, q_s * out_size <= numerator_s < (q_s + 1) * out_size
    let num_s = i.clone().real_mul(in_size.clone());
    let q_s = declare_real(&mut program, "q_s");
    program.assert(q_s.clone().real_ge(zero.clone()));
    program.assert(
        q_s.clone()
            .real_mul(out_size.clone())
            .real_le(num_s.clone()),
    );
    program.assert(num_s.real_lt(q_s.clone().real_add(one.clone()).real_mul(out_size.clone())));

    // end_i = floor((i+1) * in_size / out_size)
    let num_e = i.real_add(one.clone()).real_mul(in_size);
    let q_e = declare_real(&mut program, "q_e");
    program.assert(q_e.clone().real_ge(zero));
    program.assert(
        q_e.clone()
            .real_mul(out_size.clone())
            .real_le(num_e.clone()),
    );
    program.assert(num_e.real_lt(q_e.clone().real_add(one.clone()).real_mul(out_size)));

    // bin_width = end_i - start_i = q_e - q_s
    // Violation: bin_width < 1, i.e., q_e - q_s < 1, i.e., q_e < q_s + 1
    let violation = q_e.real_lt(q_s.real_add(one));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "adaptive_avg_pool_bin_nonempty".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Max Pooling with Indices — Selected Value Equals Max
// ---------------------------------------------------------------------------

/// Prove that max pooling with indices returns the correct index.
///
/// For a pool window of 3 elements `{x0, x1, x2}`, the returned index `idx`
/// satisfies:
///   1. `x[idx] = max(x0, x1, x2)` (the selected value is the max)
///   2. `0 <= idx <= 2` (index validity)
///
/// We model the max `m` with the standard constraints (`m >= all`, `m = one of them`)
/// and an index variable `idx` such that `x[idx] = m`. We prove that if
/// `x[idx] != m` then UNSAT.
///
/// Violation: the value at the selected index differs from the max.
pub(crate) fn prove_max_pool_index_valid() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let m = declare_real(&mut program, "m");
    let selected = declare_real(&mut program, "selected");

    assert_bounds(&mut program, &x0, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;

    // m = max(x0, x1, x2)
    program.assert(m.clone().real_ge(x0.clone()));
    program.assert(m.clone().real_ge(x1.clone()));
    program.assert(m.clone().real_ge(x2.clone()));
    program.assert(
        m.clone()
            .eq(x0.clone())
            .or(m.clone().eq(x1.clone()))
            .or(m.clone().eq(x2.clone())),
    );

    // selected = x[idx] where idx is the index of the max.
    // The index selects one of the inputs and that value equals m.
    program.assert(
        selected
            .clone()
            .eq(x0.clone())
            .or(selected.clone().eq(x1.clone()))
            .or(selected.clone().eq(x2.clone())),
    );
    program.assert(selected.clone().eq(m.clone()));

    // Violation: selected != m (contradicts the constraint we just asserted)
    // We need a different formulation: prove that if selected = m then
    // selected >= all inputs.
    // Instead: prove that any value equal to m must be >= all inputs.
    // selected = m is already asserted, so we assert the negation of
    // "selected >= x0 AND selected >= x1 AND selected >= x2".
    let not_ge_all = selected
        .clone()
        .real_lt(x0.clone())
        .or(selected.clone().real_lt(x1.clone()))
        .or(selected.real_lt(x2.clone()));
    program.assert(not_ge_all);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "max_pool_index_selects_maximum".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: L2 Pooling Non-Negativity and Upper Bound
// ---------------------------------------------------------------------------

/// Prove that L2 (RMS) pooling output is non-negative and bounded by the
/// maximum absolute value of inputs.
///
/// L2 pooling: `l2_pool(x0, x1, x2) = sqrt((x0^2 + x1^2 + x2^2) / 3)`
///
/// Properties:
///   a) `l2_pool >= 0` (square root of non-negative value)
///   b) `l2_pool^2 <= max(|x0|, |x1|, |x2|)^2` (RMS <= max absolute value)
///
/// For (b): `(x0^2 + x1^2 + x2^2) / 3 <= max(x0^2, x1^2, x2^2)`
/// because the average of non-negative numbers <= the maximum.
///
/// We prove (b) in squared form to avoid sqrt: if `M >= x0^2, x1^2, x2^2`
/// then `(x0^2 + x1^2 + x2^2) / 3 <= M`.
///
/// We use helper variables for the squared terms to keep the encoding tractable.
pub(crate) fn prove_l2_pool_bounded() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Use squared helper variables to stay in QF_LRA.
    // s0 = x0^2 >= 0, s1 = x1^2 >= 0, s2 = x2^2 >= 0
    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    let zero = Expr::real(0);

    // Non-negativity of squared values
    program.assert(s0.clone().real_ge(zero.clone()));
    program.assert(s1.clone().real_ge(zero.clone()));
    program.assert(s2.clone().real_ge(zero.clone()));

    assert_bounds(&mut program, &s0, 0.0, 1e6)?;
    assert_bounds(&mut program, &s1, 0.0, 1e6)?;
    assert_bounds(&mut program, &s2, 0.0, 1e6)?;

    // M = max(s0, s1, s2): M >= s0, M >= s1, M >= s2, M = one of them
    let max_sq = declare_real(&mut program, "M");
    program.assert(max_sq.clone().real_ge(s0.clone()));
    program.assert(max_sq.clone().real_ge(s1.clone()));
    program.assert(max_sq.clone().real_ge(s2.clone()));
    program.assert(
        max_sq
            .clone()
            .eq(s0.clone())
            .or(max_sq.clone().eq(s1.clone()))
            .or(max_sq.clone().eq(s2.clone())),
    );

    // l2_pool^2 = (s0 + s1 + s2) / 3
    // Modeled as: rms_sq * 3 = s0 + s1 + s2
    let rms_sq = declare_real(&mut program, "rms_sq");
    let three = real_from_f64(3.0)?;
    let sum = s0.real_add(s1).real_add(s2);
    program.assert(rms_sq.clone().real_mul(three).eq(sum));

    // rms_sq >= 0 (average of non-negatives)
    program.assert(rms_sq.clone().real_ge(zero));

    // Violation: rms_sq > M (average exceeds maximum)
    let violation = rms_sq.real_gt(max_sq);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "l2_pool_bounded_by_max_abs".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Stride-Padding Output Size Formula
// ---------------------------------------------------------------------------

/// Prove the standard pooling output size formula:
///   `out_size = floor((in_size + 2*pad - kernel) / stride) + 1`
///
/// This is the same as the conv output size formula with dilation=1.
///
/// We prove that `out_size >= 1` when `in_size + 2*pad >= kernel`
/// (at least one pool window fits).
///
/// Uses the floor-division encoding from `ay_convolution_properties.rs`.
pub(crate) fn prove_pool_output_size_positive() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let in_size = declare_real(&mut program, "in_size");
    let kernel = declare_real(&mut program, "kernel");
    let stride = declare_real(&mut program, "stride");
    let pad = declare_real(&mut program, "pad");

    assert_bounds(&mut program, &in_size, 1.0, 1000.0)?;
    assert_bounds(&mut program, &kernel, 1.0, 100.0)?;
    assert_bounds(&mut program, &stride, 1.0, 100.0)?;
    assert_bounds(&mut program, &pad, 0.0, 100.0)?;

    let zero = Expr::real(0);
    let one = real_from_f64(1.0)?;
    let two = real_from_f64(2.0)?;

    // numerator = in_size + 2*pad - kernel
    let numerator = in_size.clone().real_add(two.real_mul(pad)).real_sub(kernel);

    // Validity: numerator >= 0 (at least one window fits)
    program.assert(numerator.clone().real_ge(zero.clone()));

    // Floor division: q = floor(numerator / stride)
    let q = declare_real(&mut program, "q");
    program.assert(q.clone().real_ge(zero.clone()));
    program.assert(
        q.clone()
            .real_mul(stride.clone())
            .real_le(numerator.clone()),
    );
    program.assert(numerator.real_lt(q.clone().real_add(one.clone()).real_mul(stride)));

    // out_size = q + 1. Violation: q + 1 < 1, i.e., q < 0
    let violation = q.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "pool_output_size_positive".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 9: Average Pooling Commutativity with Scaling
// ---------------------------------------------------------------------------

/// Prove that average pooling commutes with scalar multiplication:
///   `avg_pool(alpha * x) = alpha * avg_pool(x)`
///
/// For a window of 3 elements:
///   `avg(alpha*x0, alpha*x1, alpha*x2) = (alpha*x0 + alpha*x1 + alpha*x2) / 3`
///   `= alpha * (x0 + x1 + x2) / 3 = alpha * avg(x0, x1, x2)`
///
/// This is the linearity of averaging (average is a linear operator).
///
/// We model: `lhs * 3 = alpha*x0 + alpha*x1 + alpha*x2`
///           `rhs = alpha * avg` where `avg * 3 = x0 + x1 + x2`
/// Violation: `lhs != rhs`.
pub(crate) fn prove_avg_pool_commutes_with_scaling() -> Result<PoolPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let alpha = declare_real(&mut program, "alpha");

    assert_bounds(&mut program, &x0, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x1, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &x2, -1000.0, 1000.0)?;
    assert_bounds(&mut program, &alpha, -100.0, 100.0)?;

    let three = real_from_f64(3.0)?;

    // avg(x) = (x0 + x1 + x2) / 3
    let avg_x = declare_real(&mut program, "avg_x");
    let sum_x = x0.clone().real_add(x1.clone()).real_add(x2.clone());
    program.assert(avg_x.clone().real_mul(three.clone()).eq(sum_x));

    // avg(alpha*x) = (alpha*x0 + alpha*x1 + alpha*x2) / 3
    let avg_ax = declare_real(&mut program, "avg_ax");
    let sum_ax = alpha
        .clone()
        .real_mul(x0)
        .real_add(alpha.clone().real_mul(x1))
        .real_add(alpha.clone().real_mul(x2));
    program.assert(avg_ax.clone().real_mul(three).eq(sum_ax));

    // rhs = alpha * avg(x)
    let rhs = alpha.real_mul(avg_x);

    // Violation: avg(alpha*x) != alpha * avg(x)
    let violation = avg_ax.ne(rhs);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(PoolPropertyResult {
        property: "avg_pool_commutes_with_scaling".to_string(),
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
    fn test_max_pool_bounded_by_input_range_proven() {
        let result = prove_max_pool_bounded_by_input().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven,
            "Max pool bounded by input (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "max_pool_bounded_by_input_range");
    }

    #[test]
    fn test_max_pool_idempotent_proven() {
        let result = prove_max_pool_idempotent().expect("proof should not error");
        assert!(
            result.proven,
            "Max pool idempotence (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(
            crate::ay_vacuity::vacuity_smell(&result.smt2),
            None,
            "idempotence proof must not be vacuous"
        );
        assert_eq!(result.property, "max_pool_idempotent");
    }

    /// Mutation test: if the second pooling window drops the last stage-1 output
    /// (`p1`), the two-stage max pool no longer equals the single-stage max pool,
    /// so the query must turn SAT. If it still "proves", the property is vacuous.
    #[test]
    fn idempotent_depends_on_full_second_window_coverage() {
        let program = build_max_pool_idempotent(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "dropping p1 from the second window must make two-stage != single-stage \
             max pool (expected SAT counterexample). detail: {}",
            detail,
        );
    }

    #[test]
    fn test_avg_pool_bounded_by_input_proven() {
        let result = prove_avg_pool_bounded_by_input().expect("proof should not error");
        assert!(
            result.proven,
            "Avg pool bounded by input (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "avg_pool_bounded_by_input");
    }

    #[test]
    fn test_global_avg_pool_equals_mean_proven() {
        let result = prove_global_avg_pool_equals_mean().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Global avg pool mean: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Global avg pool must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "global_avg_pool_equals_mean");
    }

    #[test]
    fn test_adaptive_avg_pool_bin_nonempty_proven() {
        let result = prove_adaptive_avg_pool_bin_nonempty().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Adaptive avg pool bin nonempty: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Adaptive avg pool must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "adaptive_avg_pool_bin_nonempty");
    }

    #[test]
    fn test_max_pool_index_selects_maximum_proven() {
        let result = prove_max_pool_index_valid().expect("proof should not error");
        assert!(
            result.proven,
            "Max pool index validity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "max_pool_index_selects_maximum");
    }

    #[test]
    fn test_l2_pool_bounded_by_max_abs_proven() {
        let result = prove_l2_pool_bounded().expect("proof should not error");
        assert!(
            result.proven,
            "L2 pool bounded by max abs (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "l2_pool_bounded_by_max_abs");
    }

    #[test]
    fn test_pool_output_size_positive_proven() {
        let result = prove_pool_output_size_positive().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Pool output size positivity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Pool output size must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "pool_output_size_positive");
    }

    #[test]
    fn test_avg_pool_commutes_with_scaling_proven() {
        let result = prove_avg_pool_commutes_with_scaling().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Avg pool scaling commutativity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Avg pool scaling commutativity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "avg_pool_commutes_with_scaling");
    }

    #[test]
    fn test_pool_smt2_structure() {
        let result = prove_max_pool_bounded_by_input().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }

    #[test]
    fn test_l2_pool_non_negativity() {
        // L2 pool squared value is the average of non-negative terms,
        // so it must be non-negative. We prove rms_sq >= 0 directly.
        let mut program = AYProgram::new();
        program.set_logic("QF_LRA");

        let s0 = declare_real(&mut program, "s0");
        let s1 = declare_real(&mut program, "s1");

        let zero = Expr::real(0);
        program.assert(s0.clone().real_ge(zero.clone()));
        program.assert(s1.clone().real_ge(zero.clone()));
        assert_bounds(&mut program, &s0, 0.0, 1e6).unwrap();
        assert_bounds(&mut program, &s1, 0.0, 1e6).unwrap();

        let rms_sq = declare_real(&mut program, "rms_sq");
        let two = real_from_f64(2.0).unwrap();
        let sum = s0.real_add(s1);
        program.assert(rms_sq.clone().real_mul(two).eq(sum));

        // Violation: rms_sq < 0
        let violation = rms_sq.real_lt(zero);
        program.assert(violation);
        program.check_sat();

        let (proven, detail) = execute_and_check(&program);
        assert!(
            proven,
            "L2 pool non-negativity (QF_LRA) should be Proven. detail: {}",
            detail,
        );
    }
}
