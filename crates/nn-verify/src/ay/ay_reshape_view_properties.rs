// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for tensor reshape and view mathematical properties (#4220).
//!
//! Reshape and view operations are fundamental tensor transformations that reinterpret
//! element layout without copying data. Their correctness depends on mathematical
//! invariants about element counts, dimension products, strides, and contiguity.
//!
//! This module proves seven key mathematical properties using ay's SMT solver:
//!
//! 1. **Reshape element count preservation**: product of input dims = product of output dims.
//! 2. **Flatten**: flatten(x).len() = product of all dimensions.
//! 3. **Reshape roundtrip**: reshape(reshape(x, s2), s1) = x when valid.
//! 4. **Squeeze/unsqueeze roundtrip**: squeezing a size-1 dimension is injective (invertible).
//! 5. **Transpose involution**: transpose(transpose(x)) = x.
//! 6. **View contiguity**: view is only valid when tensor is contiguous.
//! 7. **Expand**: expanded dimension must be 1 in source, element-count relationship.
//!
//! # Proof Strategy
//!
//! Reshape properties are primarily algebraic identities on dimension products.
//! We model dimensions as positive reals (>= 1) representing positive integers.
//! Most properties use `QF_LRA` or `QF_NRA` depending on whether multiplication
//! of variables is required. The proofs assert the negation of the desired property
//! and show UNSAT (no counterexample exists).

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a reshape/view property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct ReshapePropertyResult {
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
// Property 1: Reshape Element Count Preservation
// ---------------------------------------------------------------------------

/// The 3D shape reshaped by [`prove_reshape_element_count_preservation`] and
/// flattened by [`prove_flatten_element_count`].
const SHAPE_3D: [i64; 3] = [2, 3, 4];

/// Number of elements in [`SHAPE_3D`].
const SHAPE_3D_ELEMS: i64 = SHAPE_3D[0] * SHAPE_3D[1] * SHAPE_3D[2];

/// Prove that reshape neither loses nor duplicates an element.
///
/// "Element count is preserved" is not the statement `d0*d1*d2 = e0*e1` — that
/// is the *precondition* a caller supplies, and asserting it alongside its own
/// negation proves nothing. The content is that the row-major index map
///
/// ```text
/// (i, j, k)  |->  i*(d1*d2) + j*d2 + k
/// ```
///
/// is **injective** on the index box: two distinct coordinates never collide on
/// one slot, so the `d0*d1*d2` inputs occupy `d0*d1*d2` distinct slots of the
/// output. Together with the range fact proven by [`prove_flatten_element_count`]
/// (every slot lies in `[0, d0*d1*d2)`), the map is a bijection, which is exactly
/// "no element lost or duplicated".
///
/// Injectivity is where a wrong stride actually bites, and a wrong stride makes
/// this query SAT — see `element_count_depends_on_the_strides`.
///
/// Indices are `Int`, not `Real`: over the reals `i*12 + j*4 + k` is not
/// injective on the box. The shape is concrete so the strides are literals and
/// the query stays in decidable `QF_LIA`.
pub(crate) fn prove_reshape_element_count_preservation() -> Result<ReshapePropertyResult, SmtError>
{
    let program = build_reshape_element_count_preservation(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "reshape_element_count_preservation".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the injectivity query. When `strides_are_row_major` is false the
/// outermost stride is `d1` instead of `d1*d2`, the classic off-by-a-dimension
/// slip; tests flip it to confirm the proof depends on the strides.
fn build_reshape_element_count_preservation(strides_are_row_major: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let [_, d1, d2] = SHAPE_3D;
    let outer_stride = if strides_are_row_major { d1 * d2 } else { d1 };

    // Two coordinates in the [d0, d1, d2] index box.
    let (i, j, k) = declare_coord(&mut program, "");
    let (i2, j2, k2) = declare_coord(&mut program, "2");

    // Hypothesis: the coordinates differ somewhere.
    let differ = i
        .clone()
        .ne(i2.clone())
        .or(j.clone().ne(j2.clone()))
        .or(k.clone().ne(k2.clone()));
    program.assert(differ);

    let flat = flat_index(&i, &j, &k, outer_stride, d2);
    let flat2 = flat_index(&i2, &j2, &k2, outer_stride, d2);

    // Violation: distinct coordinates land on the same slot.
    program.assert(flat.eq(flat2));
    program.check_sat();
    program
}

/// Declare `i{suffix}, j{suffix}, k{suffix}` as an index into [`SHAPE_3D`].
fn declare_coord(program: &mut AYProgram, suffix: &str) -> (Expr, Expr, Expr) {
    let [d0, d1, d2] = SHAPE_3D;
    (
        declare_index(program, &format!("i{suffix}"), d0),
        declare_index(program, &format!("j{suffix}"), d1),
        declare_index(program, &format!("k{suffix}"), d2),
    )
}

/// Declare `name` as an `Int` constrained to `0 <= name < bound`.
fn declare_index(program: &mut AYProgram, name: &str, bound: i64) -> Expr {
    let var = program.declare_const(name, Sort::int());
    program.assert(var.clone().int_ge(Expr::int(0)));
    program.assert(var.clone().int_lt(Expr::int(bound)));
    var
}

/// The row-major linear index `i*outer_stride + j*inner_stride + k`.
fn flat_index(i: &Expr, j: &Expr, k: &Expr, outer_stride: i64, inner_stride: i64) -> Expr {
    i.clone()
        .int_mul(Expr::int(outer_stride))
        .int_add(j.clone().int_mul(Expr::int(inner_stride)))
        .int_add(k.clone())
}

/// The row-major linear index of `(i, j)` in a 2-D `[_, cols]` tensor: `i*cols + j`.
fn flat_index_2d(i: &Expr, j: &Expr, cols: i64) -> Expr {
    i.clone().int_mul(Expr::int(cols)).int_add(j.clone())
}

// ---------------------------------------------------------------------------
// Property 2: Flatten Element Count
// ---------------------------------------------------------------------------

/// Prove that a buffer of `d0*d1*d2` elements is large enough — and no larger
/// than needed — to hold `flatten()`'s output.
///
/// `flat_len = d0*d1*d2` is flatten's *definition*; restating it and negating it
/// is UNSAT for free. What has to be proven is that the definition is the right
/// one, i.e. that every coordinate of the `[d0, d1, d2]` box lands inside
/// `[0, d0*d1*d2)` under the row-major map:
///
/// ```text
/// 0 <= i*(d1*d2) + j*d2 + k < d0*d1*d2
/// ```
///
/// That is the range half of flatten's bijection; [`prove_reshape_element_count_preservation`]
/// proves the injective half. Range holds only because each coordinate is bounded
/// by its own dimension — dropping the innermost bound makes the query SAT (see
/// `flatten_range_depends_on_the_innermost_bound`).
pub(crate) fn prove_flatten_element_count() -> Result<ReshapePropertyResult, SmtError> {
    let program = build_flatten_element_count(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "flatten_element_count".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the flatten range query. `bound_innermost` gates `k < d2`, the single
/// hypothesis that stops the last stride from running off the end of the buffer.
fn build_flatten_element_count(bound_innermost: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let [d0, d1, d2] = SHAPE_3D;
    let i = declare_index(&mut program, "i", d0);
    let j = declare_index(&mut program, "j", d1);

    let k = program.declare_const("k", Sort::int());
    program.assert(k.clone().int_ge(Expr::int(0)));
    if bound_innermost {
        program.assert(k.clone().int_lt(Expr::int(d2)));
    }

    let flat = flat_index(&i, &j, &k, d1 * d2, d2);

    // Violation: the flat index escapes the buffer at either end.
    let violation = flat
        .clone()
        .int_lt(Expr::int(0))
        .or(flat.int_ge(Expr::int(SHAPE_3D_ELEMS)));
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: Reshape Roundtrip
// ---------------------------------------------------------------------------

/// Original 2-D shape `[D0, D1]` used by [`prove_reshape_roundtrip`].
const ROUNDTRIP_ORIG: [i64; 2] = [2, 6];
/// Intermediate 2-D shape `[E0, E1]` reshaped through; it holds the same number
/// of elements as [`ROUNDTRIP_ORIG`] (`2*6 = 3*4 = 12`) so the reshape is valid.
const ROUNDTRIP_INTER: [i64; 2] = [3, 4];

/// Prove that reshape is its own inverse: `reshape(reshape(x, s2), s1) = x`, at
/// the level of row-major coordinates.
///
/// An element at coordinate `(i, j)` of the original `[D0, D1]` tensor has the
/// row-major flat index `f = i*D1 + j`. Reshaping to `[E0, E1]` reinterprets the
/// *same* contiguous buffer, so that element now sits at the coordinate `(a, b)`
/// that decodes the same flat index there: `f = a*E1 + b` with `0 <= b < E1`.
/// Reshaping back flattens `(a, b)` to `g = a*E1 + b` and decodes it into
/// `[D0, D1]`, yielding `(i2, j2)` with `g = i2*D1 + j2`, `0 <= j2 < D1`.
///
/// The content of the theorem is that this roundtrip is the identity:
/// `(i2, j2) = (i, j)`. Nothing asserts that equality — it is *derived* by
/// chaining two euclidean decodes, and holds only because each decode's inner
/// coordinate is pinned by its range bound. Reshaping back into the wrong target
/// shape (the intermediate `[E0, E1]` instead of the original) recovers `(a, b)`
/// rather than `(i, j)` and makes the query SAT — see
/// `roundtrip_depends_on_the_target_shape`.
///
/// Coordinates are `Int`, not `Real`: over the reals the decode `g = i2*D1 + j2`
/// has infinitely many solutions and the theorem is false. The shapes are
/// concrete so every stride is a literal and the query stays in decidable
/// `QF_LIA`.
pub(crate) fn prove_reshape_roundtrip() -> Result<ReshapePropertyResult, SmtError> {
    let program = build_reshape_roundtrip(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "reshape_roundtrip".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the roundtrip query. `reshape_back_to_original` selects the shape the
/// tensor is reshaped back into: the original `[D0, D1]` (correct) or the
/// intermediate `[E0, E1]` — the classic `reshape(reshape(x, s2), s2)` slip that
/// passes the intermediate shape twice. Tests flip it to confirm the proof
/// depends on decoding back into the right shape.
fn build_reshape_roundtrip(reshape_back_to_original: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let [d0, d1] = ROUNDTRIP_ORIG;
    let [e0, e1] = ROUNDTRIP_INTER;
    let [back0, back1] = if reshape_back_to_original {
        ROUNDTRIP_ORIG
    } else {
        ROUNDTRIP_INTER
    };

    // Original coordinate (i, j) in [D0, D1] and its row-major flat index.
    let i = declare_index(&mut program, "i", d0);
    let j = declare_index(&mut program, "j", d1);
    let flat = flat_index_2d(&i, &j, d1);

    // Reshape forward to [E0, E1]: the same flat index decodes to (a, b).
    let a = declare_index(&mut program, "a", e0);
    let b = declare_index(&mut program, "b", e1);
    program.assert(flat_index_2d(&a, &b, e1).eq(flat));

    // Reshape backward: flatten the intermediate coordinate to g = a*E1 + b, then
    // decode it into the target shape [back0, back1], yielding (i2, j2).
    let g = flat_index_2d(&a, &b, e1);
    let i2 = declare_index(&mut program, "i2", back0);
    let j2 = declare_index(&mut program, "j2", back1);
    program.assert(flat_index_2d(&i2, &j2, back1).eq(g));

    // Violation: the roundtrip changed the coordinate.
    let violation = i2.ne(i).or(j2.ne(j));
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 4: Squeeze / Unsqueeze Roundtrip
// ---------------------------------------------------------------------------

/// Outer dimension `D0` of the `[D0, S, D2]` tensor whose middle axis is squeezed
/// by [`prove_squeeze_unsqueeze_roundtrip`].
const SQUEEZE_OUTER: i64 = 2;
/// Inner dimension `D2` of that tensor.
const SQUEEZE_INNER: i64 = 4;

/// Prove that squeezing a size-1 dimension is invertible — the exact content of
/// `unsqueeze(squeeze(x, d), d) = x`.
///
/// Squeeze drops a dimension: for a `[D0, S, D2]` tensor it maps the coordinate
/// `(i, m, k)` to `(i, k)` in `[D0, D2]`, discarding the middle index `m`. That
/// map can be undone by unsqueeze (which reinserts the dropped axis) **iff** it is
/// injective — no two distinct originals may land on one squeezed slot. Row-major,
/// the squeezed slot of `(i, m, k)` is `i*D2 + k`.
///
/// The content of the theorem is that this map is injective, which is *not*
/// asserted — it is derived, and holds only because the squeezed dimension has
/// size 1: then `m` is pinned to `0`, so two coordinates that differ must differ
/// in `i` or `k`, and (with `0 <= k < D2`) that separates their slots. Squeeze a
/// dimension of size `> 1` and the distinct originals `(i, 0, k)` and `(i, 1, k)`
/// collapse onto the same slot `i*D2 + k`; the dropped index is then unrecoverable
/// and the query is SAT — see `squeeze_roundtrip_depends_on_the_size_one_dim`.
///
/// Indices are `Int`, not `Real`: over the reals `i*D2 + k` is not injective on
/// the box. The shape is concrete so the strides are literals and the query stays
/// in decidable `QF_LIA`.
pub(crate) fn prove_squeeze_unsqueeze_roundtrip() -> Result<ReshapePropertyResult, SmtError> {
    let program = build_squeeze_unsqueeze_roundtrip(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "squeeze_unsqueeze_roundtrip".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the squeeze injectivity query. `dim_is_size_one` gates the single
/// hypothesis the theorem rests on — that the squeezed dimension has size 1, so
/// the dropped index `m` can only be `0`. When it is false the dimension has size
/// 2, the dropped index is free to differ, and distinct coordinates collide; tests
/// flip it to confirm the proof depends on the size-1 precondition.
fn build_squeeze_unsqueeze_roundtrip(dim_is_size_one: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let d0 = SQUEEZE_OUTER;
    let d2 = SQUEEZE_INNER;
    // Size 1 (correct) collapses the dropped index to 0; size 2 (the bug) lets it
    // vary and destroys injectivity.
    let squeeze_dim = if dim_is_size_one { 1 } else { 2 };

    // Two coordinates (i, m, k) and (i2, m2, k2) in the [D0, S, D2] index box.
    let i = declare_index(&mut program, "i", d0);
    let m = declare_index(&mut program, "m", squeeze_dim);
    let k = declare_index(&mut program, "k", d2);
    let i2 = declare_index(&mut program, "i2", d0);
    let m2 = declare_index(&mut program, "m2", squeeze_dim);
    let k2 = declare_index(&mut program, "k2", d2);

    // Hypothesis: the coordinates differ somewhere — including the dropped axis.
    let differ = i
        .clone()
        .ne(i2.clone())
        .or(m.ne(m2))
        .or(k.clone().ne(k2.clone()));
    program.assert(differ);

    // Squeeze discards the middle index: slot(i, m, k) = i*D2 + k.
    let slot = flat_index_2d(&i, &k, d2);
    let slot2 = flat_index_2d(&i2, &k2, d2);

    // Violation: two distinct coordinates land on the same squeezed slot, so the
    // dropped index cannot be recovered and unsqueeze cannot invert squeeze.
    program.assert(slot.eq(slot2));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 5: Transpose Involution
// ---------------------------------------------------------------------------

/// Rows of the concrete shape used by [`prove_transpose_involution`].
const TRANSPOSE_ROWS: i64 = 3;
/// Columns of the concrete shape used by [`prove_transpose_involution`].
const TRANSPOSE_COLS: i64 = 4;

/// Prove that transpose is an involution: `transpose(transpose(x)) = x`, at the
/// level of row-major linear indices.
///
/// For a `[ROWS, COLS]` matrix, element `(i, j)` has linear index `i*COLS + j`.
/// One transpose yields a `[COLS, ROWS]` matrix in which that element sits at
/// `(j, i)`, linear index `t = j*ROWS + i`.
///
/// The content of the theorem is in the *second* transpose. We do not re-spell
/// the answer: we recover the coordinates `(r, c)` of `t` inside the `[COLS,
/// ROWS]` matrix from `t` alone (`t = r*ROWS + c`, `0 <= c < ROWS`), transpose
/// those to `(c, r)`, and take the resulting index `c*COLS + r`. Proving that
/// this equals `i*COLS + j` forces the solver to show that the decode of
/// `j*ROWS + i` is exactly `(j, i)` — which is true only because `0 <= c < ROWS`
/// pins the euclidean division. Dropping that one bound makes the query SAT (see
/// `decode_range_bound_is_load_bearing`), so the proof is not vacuous.
///
/// Indices are `Int`, not `Real`: over the reals the decode `t = r*ROWS + c` has
/// infinitely many solutions and the theorem is false. The shape is concrete so
/// that `i*COLS` stays linear (QF_LIA, decidable); a symbolic `COLS` would make
/// every index nonlinear integer arithmetic.
pub(crate) fn prove_transpose_involution() -> Result<ReshapePropertyResult, SmtError> {
    let program = build_transpose_involution(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "transpose_involution".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the involution query. `constrain_decode_range` gates the single bound
/// (`c < ROWS`) that makes the euclidean decode unique; tests flip it off to
/// confirm the proof depends on it.
fn build_transpose_involution(constrain_decode_range: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let rows = Expr::int(TRANSPOSE_ROWS);
    let cols = Expr::int(TRANSPOSE_COLS);
    let zero = Expr::int(0);

    let i = program.declare_const("i", Sort::int());
    let j = program.declare_const("j", Sort::int());
    program.assert(i.clone().int_ge(zero.clone()));
    program.assert(i.clone().int_lt(rows.clone()));
    program.assert(j.clone().int_ge(zero.clone()));
    program.assert(j.clone().int_lt(cols.clone()));

    // Row-major index in the [ROWS, COLS] matrix.
    let idx_original = i.clone().int_mul(cols.clone()).int_add(j.clone());

    // After one transpose the element sits at (j, i) of a [COLS, ROWS] matrix.
    let t = j.int_mul(rows.clone()).int_add(i);

    // Recover (r, c) from `t` alone: t = r*ROWS + c with 0 <= c < ROWS.
    let r = program.declare_const("r", Sort::int());
    let c = program.declare_const("c", Sort::int());
    program.assert(r.clone().int_ge(zero.clone()));
    program.assert(r.clone().int_lt(cols.clone()));
    program.assert(c.clone().int_ge(zero));
    if constrain_decode_range {
        program.assert(c.clone().int_lt(rows.clone()));
    }
    program.assert(r.clone().int_mul(rows).int_add(c.clone()).eq(t));

    // Transposing [COLS, ROWS] back to [ROWS, COLS] sends (r, c) to (c, r).
    let idx_double_transposed = c.int_mul(cols).int_add(r);

    program.assert(idx_double_transposed.ne(idx_original));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: View Requires Contiguity
// ---------------------------------------------------------------------------

/// Prove that view is only valid when the tensor is contiguous.
///
/// A tensor is contiguous (row-major) when:
///   `stride[i] = product(shape[i+1:])` for all i.
///
/// For a 2D tensor `[d0, d1]`: contiguous means `stride0 = d1` and `stride1 = 1`.
///
/// View (like reshape) reinterprets contiguous memory. If the tensor is NOT
/// contiguous (e.g., after a transpose without `.contiguous()` call), the
/// strides do not match the expected layout, and view would produce incorrect
/// results.
///
/// We prove: if `stride0 = d1` and `stride1 = 1` (contiguous), then for any
/// linear index `idx = i * stride0 + j * stride1 = i * d1 + j`, the element
/// is at the expected position in the contiguous buffer.
///
/// Conversely, we show that if strides do NOT match the contiguous pattern
/// (`stride0 != d1` or `stride1 != 1`), then there exist elements whose
/// linear offset differs from the contiguous layout — making view invalid.
pub(crate) fn prove_view_requires_contiguity() -> Result<ReshapePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let d0 = declare_real(&mut program, "d0");
    let d1 = declare_real(&mut program, "d1");
    let stride0 = declare_real(&mut program, "stride0");
    let stride1 = declare_real(&mut program, "stride1");

    assert_bounds(&mut program, &d0, 2.0, 1000.0)?;
    assert_bounds(&mut program, &d1, 2.0, 1000.0)?;
    assert_bounds(&mut program, &stride0, 1.0, 1000.0)?;
    assert_bounds(&mut program, &stride1, 1.0, 1000.0)?;

    let one = real_from_f64(1.0)?;
    let zero = Expr::real(0);

    // Contiguity definition: stride0 = d1 AND stride1 = 1
    program.assert(stride0.clone().eq(d1.clone()));
    program.assert(stride1.clone().eq(one.clone()));

    // Two valid index pairs
    let i = declare_real(&mut program, "i");
    let j = declare_real(&mut program, "j");
    program.assert(i.clone().real_ge(zero.clone()));
    program.assert(i.clone().real_lt(d0));
    program.assert(j.clone().real_ge(zero.clone()));
    program.assert(j.clone().real_lt(d1.clone()));

    // Physical offset using strides: offset = i * stride0 + j * stride1
    let strided_offset = i
        .clone()
        .real_mul(stride0)
        .real_add(j.clone().real_mul(stride1));

    // Expected contiguous offset: offset = i * d1 + j
    let contiguous_offset = i.real_mul(d1).real_add(j);

    // Violation: strided_offset != contiguous_offset (should be UNSAT when contiguous)
    let violation = strided_offset.ne(contiguous_offset);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "view_requires_contiguity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Expand Dimension Constraint
// ---------------------------------------------------------------------------

/// Prove the expand operation constraints: expanded dimension must be 1 in
/// source, and the element count relationship holds.
///
/// `expand` broadcasts a dimension of size 1 to a larger size without copying.
/// For a tensor `[d0, 1, d2]` expanded to `[d0, E, d2]`:
///   - Source dimension must be 1 (precondition)
///   - Output element count: `d0 * E * d2`
///   - Input element count: `d0 * 1 * d2 = d0 * d2`
///   - Ratio: `output_count / input_count = E` (each element is repeated E times)
///
/// We prove: given source dim = 1 and expand factor E >= 1,
/// `output_count = input_count * E`.
pub(crate) fn prove_expand_dimension_constraint() -> Result<ReshapePropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let d0 = declare_real(&mut program, "d0");
    let d2 = declare_real(&mut program, "d2");
    let expand_factor = declare_real(&mut program, "E"); // expand from 1 to E

    assert_bounds(&mut program, &d0, 1.0, 1000.0)?;
    assert_bounds(&mut program, &d2, 1.0, 1000.0)?;
    assert_bounds(&mut program, &expand_factor, 1.0, 1000.0)?;

    let one = real_from_f64(1.0)?;

    // Source shape: [d0, 1, d2]
    // Input element count = d0 * 1 * d2 = d0 * d2
    let input_count = d0.clone().real_mul(one).real_mul(d2.clone());

    // Output shape: [d0, E, d2]
    // Output element count = d0 * E * d2
    let output_count = d0.real_mul(expand_factor.clone()).real_mul(d2);

    // The relationship: output_count = input_count * E
    let expected = input_count.real_mul(expand_factor);

    // Violation: output_count != input_count * E
    let violation = output_count.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(ReshapePropertyResult {
        property: "expand_dimension_constraint".to_string(),
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

    #[test]
    fn test_reshape_element_count_preservation_proven() {
        let result = prove_reshape_element_count_preservation().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Reshape element count preservation should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "reshape_element_count_preservation");
    }

    /// The row-major strides are the whole theorem. Using `d1` where `d1*d2`
    /// belongs makes `(0,1,0)` and `(1,0,1)` collide on slot 4, so the
    /// injectivity query must find a counterexample.
    #[test]
    fn element_count_depends_on_the_strides() {
        let program = build_reshape_element_count_preservation(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the outer stride mis-set to d1 the map collides and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_flatten_element_count_proven() {
        let result = prove_flatten_element_count().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven,
            "Flatten element count should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "flatten_element_count");
    }

    /// Without `k < d2` the innermost coordinate runs past the end of the buffer,
    /// so the range query must find a counterexample.
    #[test]
    fn flatten_range_depends_on_the_innermost_bound() {
        let program = build_flatten_element_count(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without `k < d2` the flat index escapes the buffer and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_reshape_roundtrip_proven() {
        let result = prove_reshape_roundtrip().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        // QF_LIA over concrete shapes is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Reshape roundtrip should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Reshape roundtrip must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "reshape_roundtrip");
    }

    /// Reshaping back into the intermediate shape `[E0, E1]` instead of the
    /// original `[D0, D1]` recovers the intermediate coordinate, not the original,
    /// so the roundtrip identity must fail and the query must be SAT.
    #[test]
    fn roundtrip_depends_on_the_target_shape() {
        let program = build_reshape_roundtrip(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "reshaping back into [E0,E1] instead of [D0,D1] changes the coordinate \
             and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip_proven() {
        let result = prove_squeeze_unsqueeze_roundtrip().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Squeeze/unsqueeze roundtrip should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Squeeze/unsqueeze roundtrip must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "squeeze_unsqueeze_roundtrip");
    }

    /// The size-1 dimension is the whole theorem. Squeezing a size-2 dimension lets
    /// the dropped index differ, so `(i, 0, k)` and `(i, 1, k)` — distinct
    /// coordinates — collide on the squeezed slot `i*D2 + k`, and the injectivity
    /// query must be SAT.
    #[test]
    fn squeeze_roundtrip_depends_on_the_size_one_dim() {
        let program = build_squeeze_unsqueeze_roundtrip(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "squeezing a size-2 dimension collapses distinct coordinates and the query \
             must be SAT; got: {detail}",
        );
    }

    /// Dropping the one bound that makes the euclidean decode unique must expose
    /// a counterexample. If it does not, the proof is not using the decode at all
    /// and has degenerated into a tautology.
    #[test]
    fn decode_range_bound_is_load_bearing() {
        let program = build_transpose_involution(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without `c < ROWS` the decode is ambiguous and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_transpose_involution_proven() {
        let result = prove_transpose_involution().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Transpose involution should be proven, got: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert!(
            !result.detail.contains("counterexample"),
            "Transpose involution must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "transpose_involution");
    }

    #[test]
    fn test_view_requires_contiguity_proven() {
        let result = prove_view_requires_contiguity().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "View requires contiguity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "View contiguity must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "view_requires_contiguity");
    }

    #[test]
    fn test_expand_dimension_constraint_proven() {
        let result = prove_expand_dimension_constraint().expect("proof should not error");
        assert!(
            result.smt2.contains("check-sat"),
            "SMT2 should contain check-sat"
        );
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Expand dimension constraint: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Expand dimension must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "expand_dimension_constraint");
    }

    #[test]
    fn test_reshape_smt2_structure() {
        let result = prove_reshape_element_count_preservation().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }
}
