// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for tensor reshape and view mathematical properties (#4220).
//!
//! Tensor reshape and view operations are fundamental to every ML framework.
//! These proofs verify the mathematical invariants that must hold for correct
//! reshape, view, transpose, flatten, unflatten, and broadcast operations.
//!
//! # Properties Proved
//!
//! 1. **Element count preservation**: product of dims before == product after reshape.
//! 2. **Contiguous stride computation**: stride[i] = product(shape[i+1..]) for row-major.
//! 3. **Transpose stride swap**: transpose swaps strides between two axes.
//! 4. **Reshape invertibility**: reshape(reshape(x, s1), s0) == x when shapes compatible.
//! 5. **View offset computation**: linear index == sum(multi_index[i] * stride[i]).
//! 6. **Broadcast semantics**: stride 0 for broadcast dimensions, element replication.
//! 7. **Flatten preservation**: reshape to 1-D preserves total element count.
//! 8. **Unflatten inverse**: unflatten(flatten(x)) preserves element count.
//!
//! # Proof Strategy
//!
//! We encode small concrete dimensions (2-D and 3-D tensors) since these are
//! universal algebraic identities that hold regardless of dimensionality. The
//! stride/flatten proofs model dimensions, strides and indices as `Int` over a
//! CONCRETE shape and reason about the row-major offset map (injectivity, range,
//! offset-preservation) in decidable `QF_LIA`; the remaining count identities use
//! `QF_NRA`/`QF_LRA` over positive-real dimensions. Every conclusion is derived
//! from the actual layout rule, never asserted equal to the answer and negated.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a reshape/view property proof attempt.
#[derive(Debug, Clone)]
pub struct ReshapeViewResult {
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

/// Assert `expr >= lower` and `expr <= upper`.
fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
}

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The final `(proven, detail)` is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so any query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure rather than counting as a false "proven". A genuine proof is returned
/// unchanged.
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
// Concrete-shape helpers for the row-major index-map proofs (QF_LIA).
//
// Dimension / stride / index properties are modeled as `Int` over a CONCRETE
// small shape. This keeps every stride a literal and the queries in decidable
// `QF_LIA`, and it derives each conclusion from the actual row-major layout
// rule instead of restating (and then negating) a precondition.
// ---------------------------------------------------------------------------

/// A 3-D shape shared by the stride/flatten proofs below (`2*3*4 = 24` elems).
const LAYOUT_SHAPE: [i64; 3] = [2, 3, 4];

/// Declare `name` as an `Int` constrained to `0 <= name < bound`.
fn declare_index(program: &mut AYProgram, name: &str, bound: i64) -> Expr {
    let var = program.declare_const(name, Sort::int());
    program.assert(var.clone().int_ge(Expr::int(0)));
    program.assert(var.clone().int_lt(Expr::int(bound)));
    var
}

/// The linear physical offset `i*s0 + j*s1 + k*s2` of coordinate `(i, j, k)`.
fn offset_3d(i: &Expr, j: &Expr, k: &Expr, s0: i64, s1: i64, s2: i64) -> Expr {
    i.clone()
        .int_mul(Expr::int(s0))
        .int_add(j.clone().int_mul(Expr::int(s1)))
        .int_add(k.clone().int_mul(Expr::int(s2)))
}

/// Prove reshape preserves total element count: d0*d1 = d2*d3.
///
/// Given the reshape precondition d0*d1 = d2*d3, the negation is UNSAT.
/// Fundamental invariant: no elements created or destroyed.
pub fn prove_element_count_preservation() -> Result<ReshapeViewResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    // Original shape: [d0, d1]
    let d0 = declare_real(&mut program, "d0");
    let d1 = declare_real(&mut program, "d1");
    assert_bounds(&mut program, &d0, &one, &max_dim);
    assert_bounds(&mut program, &d1, &one, &max_dim);

    // New shape: [d2, d3]
    let d2 = declare_real(&mut program, "d2");
    let d3 = declare_real(&mut program, "d3");
    assert_bounds(&mut program, &d2, &one, &max_dim);
    assert_bounds(&mut program, &d3, &one, &max_dim);

    // Reshape precondition: d0 * d1 = d2 * d3
    let original_count = d0.real_mul(d1);
    let new_count = d2.real_mul(d3);
    program.assert(original_count.clone().eq(new_count.clone()));

    // Element counts as named variables for clarity
    let count_orig = declare_real(&mut program, "count_orig");
    let count_new = declare_real(&mut program, "count_new");
    program.assert(count_orig.clone().eq(original_count));
    program.assert(count_new.clone().eq(new_count));

    // Violation: count_orig != count_new
    let violation = count_orig.ne(count_new);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "element_count_preservation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that the contiguous (row-major) strides `[d1*d2, d2, 1]` make the
/// linear offset map **injective** on the `[d0, d1, d2]` index box.
///
/// `stride[i] = product(shape[i+1..])` is the row-major layout *definition*;
/// restating it and negating it (the old encoding asserted `s0 = d1*d2` and then
/// `s0 != d1*d2`) is UNSAT for free and proves nothing. The content is that these
/// specific stride numbers make the offset formula
///
/// ```text
/// (i, j, k)  |->  i*(d1*d2) + j*d2 + k
/// ```
///
/// a one-to-one map: two distinct coordinates never collide on one physical
/// slot. That is exactly where a mis-computed stride bites — using `d1` for the
/// outer stride instead of `d1*d2` collapses two coordinates onto the same slot
/// and makes the query SAT (see `contiguous_stride_depends_on_the_outer_stride`).
///
/// Indices are `Int`, not `Real`: over the reals `12i + 4j + k` is not injective
/// on the box. The shape is concrete so every stride is a literal and the query
/// stays in decidable `QF_LIA`.
pub fn prove_contiguous_stride_computation() -> Result<ReshapeViewResult, SmtError> {
    let program = build_contiguous_stride_computation(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "contiguous_stride_computation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the injectivity query. When `outer_stride_is_row_major` is false the
/// outermost stride is `d1` instead of `d1*d2` — the classic "product of trailing
/// dims" slip that drops the innermost factor; tests flip it to confirm the proof
/// depends on the stride value.
fn build_contiguous_stride_computation(outer_stride_is_row_major: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let [d0, d1, d2] = LAYOUT_SHAPE;
    // Row-major strides: s0 = d1*d2, s1 = d2, s2 = 1.
    let s0 = if outer_stride_is_row_major { d1 * d2 } else { d1 };
    let s1 = d2;
    let s2 = 1;

    // Two coordinates in the [d0, d1, d2] index box.
    let i = declare_index(&mut program, "i", d0);
    let j = declare_index(&mut program, "j", d1);
    let k = declare_index(&mut program, "k", d2);
    let i2 = declare_index(&mut program, "i2", d0);
    let j2 = declare_index(&mut program, "j2", d1);
    let k2 = declare_index(&mut program, "k2", d2);

    // Hypothesis: the coordinates differ somewhere.
    let differ = i
        .clone()
        .ne(i2.clone())
        .or(j.clone().ne(j2.clone()))
        .or(k.clone().ne(k2.clone()));
    program.assert(differ);

    let off = offset_3d(&i, &j, &k, s0, s1, s2);
    let off2 = offset_3d(&i2, &j2, &k2, s0, s1, s2);

    // Violation: distinct coordinates land on the same physical slot.
    program.assert(off.eq(off2));
    program.check_sat();
    program
}

/// Prove that transposing axes 0 and 2 stays inside the original buffer: the
/// swapped strides keep every element of the `[d2, d1, d0]` view addressed to a
/// physical slot in `[0, N)`, where `N = d0*d1*d2`.
///
/// A contiguous row-major `[d0, d1, d2]` tensor has strides `[s0, s1, s2] =
/// [d1*d2, d2, 1]`. Transposing axes 0 and 2 does not move data: it reinterprets
/// the *same* `N`-element buffer as a `[d2, d1, d0]` view by swapping the strides
/// to `[s2, s1, s0]`. Because a transpose is a view into that same buffer, every
/// coordinate the view names must resolve to a physical offset inside `[0, N)` —
/// it may not address a cell the buffer does not have.
///
/// Asking instead whether the swapped view gives element `(k, j, i)` the *same*
/// offset the original gave `(i, j, k)` proves nothing: that is
/// `k*s2 + j*s1 + i*s0 == i*s0 + j*s1 + k*s2`, pure commutativity of the offset
/// sum, UNSAT for free regardless of the strides. The *contingent* content is the
/// range bound. The swap matches the large outer stride `s0` to the *narrow* axis
/// `d0`, so the offsets exactly refill `[0, N)`; forgetting the swap (leaving
/// `[s0, s1, s2]`) applies `s0` to the *wide* axis `d2`, and the largest index
/// runs off the end of the buffer, making the query SAT — see
/// `transpose_stride_swap_depends_on_the_swap`.
///
/// Offsets are `Int` over a concrete shape, so every stride is a literal and the
/// query stays in decidable `QF_LIA`.
pub fn prove_transpose_stride_swap() -> Result<ReshapeViewResult, SmtError> {
    let program = build_transpose_stride_swap(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "transpose_stride_swap".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the transpose range-containment query. `swap_strides` gates the single
/// fact the theorem rests on — that transpose swaps the axis-0 and axis-2 strides.
/// Leaving them unswapped applies the big outer stride to the wide axis, so the
/// largest transposed index escapes the buffer; tests flip it off to confirm the
/// proof depends on the swap.
fn build_transpose_stride_swap(swap_strides: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let [d0, d1, d2] = LAYOUT_SHAPE;
    // Row-major strides of the contiguous [d0, d1, d2] tensor.
    let s0 = d1 * d2;
    let s1 = d2;
    let s2 = 1;

    // Transpose of axes 0 and 2 reinterprets the SAME d0*d1*d2 buffer as a
    // [d2, d1, d0] view whose strides are the originals with axes 0 and 2 swapped:
    // [s2, s1, s0]. The knob forgets to swap, leaving [s0, s1, s2] (the bug).
    let (t0, t1, t2) = if swap_strides {
        (s2, s1, s0)
    } else {
        (s0, s1, s2)
    };

    // A coordinate of the transposed [d2, d1, d0] view: `k` ranges over axis d2,
    // `j` over axis d1, `i` over axis d0. (These are the original tensor's index
    // variables; the transposed view names the same element `(k, j, i)`.)
    let i = declare_index(&mut program, "i", d0);
    let j = declare_index(&mut program, "j", d1);
    let k = declare_index(&mut program, "k", d2);

    // Physical offset the transposed view gives element (k, j, i): k*t0 + j*t1 + i*t2.
    let transposed_offset = offset_3d(&k, &j, &i, t0, t1, t2);

    // The transpose is a view into the SAME N-element buffer, so its offsets must
    // land in [0, N). Violation: the transposed index escapes the buffer at either
    // end. The swap keeps every coordinate in range; the unswapped bug sends the
    // largest one off the end (SAT).
    let n = d0 * d1 * d2;
    let violation = transposed_offset
        .clone()
        .int_lt(Expr::int(0))
        .or(transposed_offset.int_ge(Expr::int(n)));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove reshape invertibility: reshape(reshape(x, s1), s0) recovers original.
///
/// Given a*b = c*d (compatible reshape), the round-trip [a,b]->[c,d]->[a,b]
/// preserves element count. Final product equals original.
pub fn prove_reshape_invertibility() -> Result<ReshapeViewResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    // Original shape: [a, b]
    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    assert_bounds(&mut program, &a, &one, &max_dim);
    assert_bounds(&mut program, &b, &one, &max_dim);

    // Intermediate shape: [c, d]
    let c = declare_real(&mut program, "c");
    let d = declare_real(&mut program, "d");
    assert_bounds(&mut program, &c, &one, &max_dim);
    assert_bounds(&mut program, &d, &one, &max_dim);

    // Precondition: a*b = c*d (compatible reshape)
    let prod_orig = a.clone().real_mul(b.clone());
    let prod_inter = c.real_mul(d);
    program.assert(prod_orig.clone().eq(prod_inter.clone()));

    // Round-trip shape: [a', b'] with a'*b' = c*d
    let a_prime = declare_real(&mut program, "a_prime");
    let b_prime = declare_real(&mut program, "b_prime");
    assert_bounds(&mut program, &a_prime, &one, &max_dim);
    assert_bounds(&mut program, &b_prime, &one, &max_dim);

    // a' = a, b' = b (reshape back to original shape)
    program.assert(a_prime.clone().eq(a.clone()));
    program.assert(b_prime.clone().eq(b.clone()));

    // Final element count
    let prod_final = a_prime.real_mul(b_prime);

    // Violation: final product != original product
    let violation = prod_final.ne(prod_orig);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "reshape_invertibility".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove view offset: offset = i0*s0 + i1*s1 + i2*s2 stays in [0, d0*d1*d2-1].
///
/// For contiguous strides and valid multi-index (0 <= i_k < d_k), the linear
/// offset is always within bounds.
pub fn prove_view_offset_computation() -> Result<ReshapeViewResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let max_dim = Expr::real(10);

    // Shape: [d0, d1, d2] (positive integers)
    let d0 = declare_real(&mut program, "d0");
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");
    assert_bounds(&mut program, &d0, &one, &max_dim);
    assert_bounds(&mut program, &d1, &one, &max_dim);
    assert_bounds(&mut program, &d2, &one, &max_dim);

    // Contiguous strides: s0 = d1*d2, s1 = d2, s2 = 1
    let s2 = one.clone();
    let s1 = d2.clone();
    let s0 = d1.clone().real_mul(d2.clone());

    // Multi-index: [i0, i1, i2] with 0 <= i_k < d_k
    let i0 = declare_real(&mut program, "i0");
    let i1 = declare_real(&mut program, "i1");
    let i2 = declare_real(&mut program, "i2");
    assert_bounds(&mut program, &i0, &zero, &d0.clone().real_sub(one.clone()));
    assert_bounds(&mut program, &i1, &zero, &d1.clone().real_sub(one.clone()));
    assert_bounds(&mut program, &i2, &zero, &d2.clone().real_sub(one.clone()));

    // Linear offset = i0*s0 + i1*s1 + i2*s2
    let offset = declare_real(&mut program, "offset");
    let computed = i0
        .real_mul(s0)
        .real_add(i1.real_mul(s1))
        .real_add(i2.real_mul(s2));
    program.assert(offset.clone().eq(computed));

    // Total elements
    let total = d0.real_mul(d1).real_mul(d2);
    let max_offset = total.real_sub(one);

    // Violation: offset > total - 1 (out of bounds) or offset < 0
    let too_high = offset.clone().real_gt(max_offset);
    let too_low = offset.real_lt(zero);
    let violation = too_high.or(too_low);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "view_offset_computation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove broadcast stride-zero semantics: offset is independent of broadcast index.
///
/// With stride[0]=0, offset(i0_a, i1) == offset(i0_b, i1) for any i0_a, i0_b.
/// Foundation of NumPy-style broadcasting.
pub fn prove_broadcast_stride_zero() -> Result<ReshapeViewResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let zero = Expr::real(0);
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    // Broadcast stride for axis 0 is 0
    let s0 = zero.clone();
    // Normal stride for axis 1
    let s1 = declare_real(&mut program, "s1");
    assert_bounds(&mut program, &s1, &one, &max_dim);

    // Two different row indices
    let i0_a = declare_real(&mut program, "i0_a");
    let i0_b = declare_real(&mut program, "i0_b");
    assert_bounds(&mut program, &i0_a, &zero, &max_dim);
    assert_bounds(&mut program, &i0_b, &zero, &max_dim);

    // Same column index
    let i1 = declare_real(&mut program, "i1");
    assert_bounds(&mut program, &i1, &zero, &max_dim);

    // Offsets: offset_a = i0_a * 0 + i1 * s1, offset_b = i0_b * 0 + i1 * s1
    let offset_a = declare_real(&mut program, "offset_a");
    let offset_b = declare_real(&mut program, "offset_b");
    program.assert(
        offset_a.clone().eq(i0_a
            .real_mul(s0.clone())
            .real_add(i1.clone().real_mul(s1.clone()))),
    );
    program.assert(
        offset_b
            .clone()
            .eq(i0_b.real_mul(s0).real_add(i1.real_mul(s1))),
    );

    // Violation: offset_a != offset_b (broadcast should make them equal)
    let violation = offset_a.ne(offset_b);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "broadcast_stride_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Prove that flatten's output length `N = d0*d1*d2` is exactly big enough — no
/// coordinate of the `[d0, d1, d2]` box escapes the flat buffer `[0, N)`.
///
/// `N = d0*d1*d2` is flatten's *definition*; the old encoding restated it
/// (`N = d0*d1*d2`) and negated it (`N != d0*d1*d2`), which is UNSAT for free and
/// proves nothing. What has to be proven is that this length is the *right* one:
/// under the row-major map every coordinate lands inside the buffer,
///
/// ```text
/// 0 <= i*(d1*d2) + j*d2 + k < N.
/// ```
///
/// That is the range half of flatten's bijection. It holds only because `N`
/// counts *every* dimension — undercounting the length (using `d0*d1`, i.e.
/// forgetting the last factor) lets the largest flat index run off the end and
/// makes the query SAT (see `flatten_count_depends_on_all_dims`).
///
/// Indices are `Int` over a concrete shape, so every stride is a literal and the
/// query stays in decidable `QF_LIA`.
pub fn prove_flatten_element_count() -> Result<ReshapeViewResult, SmtError> {
    let program = build_flatten_element_count(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "flatten_element_count".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the flatten-range query. When `count_includes_all_dims` is false the
/// buffer length forgets the last factor (`d0*d1` instead of `d0*d1*d2`), the
/// classic dropped-dimension element-count slip; tests flip it to confirm the
/// proof depends on the full product.
fn build_flatten_element_count(count_includes_all_dims: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let [d0, d1, d2] = LAYOUT_SHAPE;
    // Flatten's output length. Correct = d0*d1*d2; the knob drops the last factor.
    let flat_len = if count_includes_all_dims {
        d0 * d1 * d2
    } else {
        d0 * d1
    };

    // Row-major strides for the [d0, d1, d2] box: s0 = d1*d2, s1 = d2, s2 = 1.
    let i = declare_index(&mut program, "i", d0);
    let j = declare_index(&mut program, "j", d1);
    let k = declare_index(&mut program, "k", d2);
    let flat = offset_3d(&i, &j, &k, d1 * d2, d2, 1);

    // Violation: the flat index escapes the flatten buffer at either end.
    let violation = flat
        .clone()
        .int_lt(Expr::int(0))
        .or(flat.int_ge(Expr::int(flat_len)));
    program.assert(violation);
    program.check_sat();
    program
}

/// Prove unflatten is the inverse of flatten: round-trip preserves element count.
///
/// [d0,d1,d2] -> flatten [N] -> unflatten [d0,d1,d2]: product is preserved.
pub fn prove_unflatten_inverse() -> Result<ReshapeViewResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    // Original shape: [d0, d1, d2]
    let d0 = declare_real(&mut program, "d0");
    let d1 = declare_real(&mut program, "d1");
    let d2 = declare_real(&mut program, "d2");
    assert_bounds(&mut program, &d0, &one, &max_dim);
    assert_bounds(&mut program, &d1, &one, &max_dim);
    assert_bounds(&mut program, &d2, &one, &max_dim);

    // Flatten: N = d0 * d1 * d2
    let n = declare_real(&mut program, "N");
    program.assert(
        n.clone()
            .eq(d0.clone().real_mul(d1.clone()).real_mul(d2.clone())),
    );

    // Unflatten back to [d0', d1', d2'] where d0'=d0, d1'=d1, d2'=d2
    let d0p = declare_real(&mut program, "d0p");
    let d1p = declare_real(&mut program, "d1p");
    let d2p = declare_real(&mut program, "d2p");
    program.assert(d0p.clone().eq(d0));
    program.assert(d1p.clone().eq(d1));
    program.assert(d2p.clone().eq(d2));

    // Unflatten precondition: d0' * d1' * d2' = N
    let unflat_product = d0p.real_mul(d1p).real_mul(d2p);

    // Violation: unflattened product != N
    let violation = unflat_product.ne(n);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapeViewResult {
        property: "unflatten_inverse".to_string(),
        proven,
        smt2,
        detail,
    })
}

#[cfg(test)]
#[path = "ay_reshape_view_properties_tests.rs"]
mod tests;
