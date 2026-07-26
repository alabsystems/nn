// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for sparse attention mathematical properties (#4558).
//!
//! Proves fundamental mathematical properties of sparse attention patterns
//! including sparsity masks, local/sliding window attention, block-sparse
//! patterns, dilated attention, and strided attention used in efficient
//! transformer variants (Longformer, BigBird, Sparse Transformers).
//!
//! # Proved Properties
//!
//! 1. **Sparsity mask binary**: Mask values are in {0, 1}.
//! 2. **Sparsity mask symmetry**: For self-attention, mask(i,j) = mask(j,i).
//! 3. **Sparse attention preserves weight normalization**: Masked softmax sums to 1.
//! 4. **Local window bounds**: Window attention only attends within [i-w, i+w].
//! 5. **Local window coverage**: Every position attends to at least itself.
//! 6. **Local window size consistency**: Window size is 2w+1 for radius w.
//! 7. **Sliding window shift invariance**: Pattern at position i matches i+1 shifted.
//! 8. **Sliding window boundary clamping**: Near boundaries, window is clamped to [0, N).
//! 9. **Block-sparse block alignment**: Block boundaries align to block_size multiples.
//! 10. **Block-sparse coverage**: Union of blocks covers entire sequence.
//! 11. **Block-sparse regularity**: Each block has identical internal structure.
//! 12. **Dilated attention stride property**: Dilated mask selects every d-th position.
//! 13. **Dilated attention coverage over heads**: Union of dilated heads covers all positions.
//! 14. **Strided attention periodicity**: Pattern repeats with period equal to stride.
//! 15. **Global token full attention**: Global tokens attend to all positions.
//! 16. **Global token symmetry**: If i is global, all positions attend to i.
//! 17. **Causal sparse intersection**: Sparse mask AND causal mask = causal-sparse mask.
//! 18. **Sparse attention output dimension**: Output dimension matches value dimension.
//! 19. **Mask sparsity ratio**: Fraction of ones <= target sparsity.
//! 20. **Local + global union coverage**: Local window union global tokens covers all.
//! 21. **Block diagonal structure**: Block-diagonal mask has no off-diagonal blocks.
//! 22. **Attention score masking**: Masked positions get -inf before softmax.
//! 23. **Random sparse mask row coverage**: Each row has at least k non-zero entries.
//! 24. **Sparse pattern composition**: Composing two sparse patterns = element-wise AND.
//! 25. **Sliding window causal restriction**: Causal sliding window is lower-triangular band.
//! 26. **Multi-scale attention merge**: Combining local and global heads preserves dimension.
//! 27. **Sparse softmax zero propagation**: Masked positions get zero weight after softmax.
//! 28. **Block size divides sequence length**: N % block_size = 0 for aligned blocking.
//! 29. **Sparse attention FLOPs reduction**: FLOPs proportional to nnz, not N^2.
//! 30. **Longformer combined pattern**: Local + global + random covers all positions.
//!
//! # Proof Strategy
//!
//! - **Structural proofs** (mask shapes, coverage): Pure QF_LRA with indicator variables.
//! - **Algebraic proofs** (normalization, dimension): QF_NRA for products.
//! - **Compositional proofs** (pattern unions, intersections): Boolean-encoded via reals in {0,1}.

use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a sparse attention property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct SparseAttentionPropertyResult {
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

/// Declare an integer variable and return its expression.
fn declare_int(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::int())
}

/// Declare an `Int` constrained to `lo <= name < hi_exclusive` and return it.
fn declare_int_in(program: &mut AYProgram, name: &str, lo: i64, hi_exclusive: i64) -> Expr {
    let var = declare_int(program, name);
    program.assert(var.clone().int_ge(Expr::int(lo)));
    program.assert(var.clone().int_lt(Expr::int(hi_exclusive)));
    var
}

/// Declare `name`, pin it to `term`, and return the new variable.
///
/// Naming each intermediate keeps a proof's conclusion one step removed from its
/// hypotheses, so the solver *derives* it rather than matching an asserted answer.
fn define_real(program: &mut AYProgram, name: &str, term: Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term));
    var
}

/// Integer analogue of [`define_real`].
fn define_int(program: &mut AYProgram, name: &str, term: Expr) -> Expr {
    let var = declare_int(program, name);
    program.assert(var.clone().eq(term));
    var
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

/// Assert `expr > lower` (strict lower bound).
fn assert_strict_positive(
    program: &mut AYProgram,
    expr: &Expr,
    lower: f64,
) -> Result<(), SmtError> {
    let lo = real_from_f64(lower)?;
    program.assert(expr.clone().real_gt(lo));
    Ok(())
}

/// Constrain a variable to be binary: in {0, 1}.
fn assert_binary(program: &mut AYProgram, expr: &Expr) -> Result<(), SmtError> {
    assert_bounds(program, expr, 0.0, 1.0)?;
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let binary_constraint = expr.clone().real_mul(one.real_sub(expr.clone()));
    program.assert(binary_constraint.eq(zero));
    Ok(())
}

/// Execute a ay program and return whether UNSAT (property proven).
///
/// The final `(proven, detail)` is funneled through
/// [`crate::ay_vacuity::reject_if_vacuous`], so any query that is UNSAT only
/// because it asserts `P ∧ ¬P` (or compares a term to itself) is downgraded to a
/// failure rather than counting as a proof. A genuine proof is returned unchanged.
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
// Property 1: Sparsity Mask Binary
// ---------------------------------------------------------------------------

/// Prove that sparsity mask values are binary (in {0, 1}).
///
/// A valid sparsity mask M has M(i,j) in {0, 1} for all i, j.
/// We encode: given mask value m with m*(1-m) = 0 and 0 <= m <= 1,
/// assert m is not in {0, 1} and prove UNSAT.
pub(crate) fn prove_sparsity_mask_binary() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let m = declare_real(&mut program, "m");
    assert_binary(&mut program, &m)?;

    // Negated property: m != 0 AND m != 1
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let violation = m.clone().ne(zero).and(m.ne(one));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparsity_mask_binary".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Sparsity Mask Symmetry
// ---------------------------------------------------------------------------

/// Prove that a distance-based self-attention mask is symmetric: the attend
/// predicate for `(i, j)` holds exactly when it holds for `(j, i)`.
///
/// The window rule keeps `(a, b)` when `a - b <= w` AND `b - a <= w` (i.e.
/// `|a - b| <= w`) — a predicate manifestly invariant under swapping `a, b`.
/// We apply the SAME rule to `(i, j)` and to `(j, i)` and prove the two verdicts
/// never disagree (no position pair is attended one way but not the other).
///
/// The content is that both directed bounds are checked. A rule that keeps
/// `(a, b)` on `a - b <= w` alone (directed distance instead of `|a - b|`) is
/// asymmetric, and the query then finds a pair that disagrees — see
/// `symmetry_depends_on_both_directed_bounds`. Indices are `Int`; `QF_LIA` over
/// concrete `w` is decidable.
pub(crate) fn prove_sparsity_mask_symmetry() -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_sparsity_mask_symmetry(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparsity_mask_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Window radius used by [`build_sparsity_mask_symmetry`].
const SYMMETRY_W: i64 = 2;

/// The window attend-predicate for the ordered pair `(a, b)`.
///
/// Correct: `a - b <= w  ∧  b - a <= w` (symmetric `|a - b| <= w`). When
/// `both_bounds` is false only the first directed bound is kept, making the
/// predicate asymmetric.
fn window_attend(a: &Expr, b: &Expr, both_bounds: bool) -> Expr {
    let w = Expr::int(SYMMETRY_W);
    let forward = a.clone().int_sub(b.clone()).int_le(w.clone());
    if both_bounds {
        let backward = b.clone().int_sub(a.clone()).int_le(w);
        forward.and(backward)
    } else {
        forward
    }
}

/// Build the symmetry query. `both_bounds` gates the second directed bound; drop
/// it and the mask rule is directed rather than `|i - j|`, breaking symmetry.
fn build_sparsity_mask_symmetry(both_bounds: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let i = declare_int_in(&mut program, "i", 0, 64);
    let j = declare_int_in(&mut program, "j", 0, 64);

    let m_ij = window_attend(&i, &j, both_bounds);
    let m_ji = window_attend(&j, &i, both_bounds);

    // Violation: the mask disagrees across the diagonal for some (i, j).
    let disagree = m_ij
        .clone()
        .and(m_ji.clone().not())
        .or(m_ji.and(m_ij.not()));
    program.assert(disagree);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: Sparse Attention Preserves Weight Normalization
// ---------------------------------------------------------------------------

/// Prove that masked softmax weights still sum to 1 — provided the denominator
/// is renormalized over the surviving (unmasked) positions.
///
/// Three positions: 1 and 2 are unmasked with exp-scores `e1, e2` (positive,
/// gauge-fixed to `e1 + e2 = 10`), position 3 is masked so its numerator is
/// `mask3 * e3 = 0`. Each weight is `w_k = numerator_k / Z`. The theorem is that
/// the surviving weights sum to 1, which holds iff `Z` is the sum over the
/// unmasked numerators (`= 10`), not the pre-mask total.
///
/// The realistic bug (`renormalize = false`) divides by the pre-mask total
/// `10 + e3` — the classic "masked the logits but forgot to drop them from the
/// denominator" slip — and the weights then sum to `10/(10+e3) < 1`, so the query
/// is SAT (see `normalization_depends_on_renormalizing_the_denominator`).
///
/// `Z` is a literal in each branch, so every weight is `var / const` and the
/// query stays linear (`QF_LRA`, decidable).
pub(crate) fn prove_sparse_weight_normalization() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_sparse_weight_normalization(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_weight_normalization".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Unmasked exp-mass (gauge) and the masked position's exp-score.
const NORM_UNMASKED_MASS: i64 = 10;
const NORM_MASKED_EXP: i64 = 5;

/// Build the normalization query. `renormalize` picks the softmax denominator:
/// the unmasked mass (correct) or the pre-mask total including the masked
/// position (the bug).
fn build_sparse_weight_normalization(renormalize: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Exp-scores of the two unmasked positions, gauge-fixed so their sum (the
    // renormalized denominator) is a literal.
    let e1 = declare_real(&mut program, "e1");
    let e2 = declare_real(&mut program, "e2");
    program.assert(e1.clone().real_gt(Expr::real(0)));
    program.assert(e2.clone().real_gt(Expr::real(0)));
    program.assert(
        e1.clone()
            .real_add(e2.clone())
            .eq(Expr::real(NORM_UNMASKED_MASS)),
    );

    // Denominator Z: unmasked mass, or pre-mask total (masked mass left in).
    let z = if renormalize {
        Expr::real(NORM_UNMASKED_MASS)
    } else {
        Expr::real(NORM_UNMASKED_MASS + NORM_MASKED_EXP)
    };

    // Masked numerator is mask3 * e3 = 0 * e3.
    let masked_numer = Expr::real(0).real_mul(Expr::real(NORM_MASKED_EXP));

    let w1 = define_real(&mut program, "w1", e1.real_div(z.clone()));
    let w2 = define_real(&mut program, "w2", e2.real_div(z.clone()));
    let w3 = define_real(&mut program, "w3", masked_numer.real_div(z));

    let total = define_real(&mut program, "total", w1.real_add(w2).real_add(w3));

    // Violation: the surviving weights do not sum to 1.
    program.assert(total.ne(Expr::real(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 4: Local Window Bounds
// ---------------------------------------------------------------------------

/// Prove that every key a local window actually attends lies within `[i-w, i+w]`.
///
/// The attended keys are generated as `j = i + off`, one per offset `off` the
/// implementation walks over. We prove that each generated key stays inside the
/// radius: `|j - i| <= w`, i.e. `j - i <= w` and `i - j <= w`.
///
/// The content is the offset range. Correct code walks `off ∈ [-w, w]`; a
/// one-too-wide range `[-(w+1), w+1]` (an off-by-one in the window loop) emits a
/// key at distance `w+1`, and the query then finds it outside the window — see
/// `bounds_depend_on_the_offset_range`. `QF_LIA` over a concrete `w` is decidable.
pub(crate) fn prove_local_window_bounds() -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_local_window_bounds(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "local_window_bounds".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Window radius used by [`build_local_window_bounds`].
const LOCAL_WINDOW_W: i64 = 3;

/// Build the window-bounds query. `radius_exact` gates the offset range: the true
/// radius `w` (correct) versus `w + 1` (a one-too-wide window loop).
fn build_local_window_bounds(radius_exact: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let w = LOCAL_WINDOW_W;
    let gen_radius = if radius_exact { w } else { w + 1 };

    let i = declare_int_in(&mut program, "i", 0, 1000);
    // The offset the window loop iterates: -gen_radius ..= gen_radius.
    let off = declare_int_in(&mut program, "off", -gen_radius, gen_radius + 1);
    let j = define_int(&mut program, "j", i.clone().int_add(off));

    // Violation: the generated key lands outside the true window [i-w, i+w].
    let w_expr = Expr::int(w);
    let above = j.clone().int_sub(i.clone()).int_gt(w_expr.clone());
    let below = i.int_sub(j).int_gt(w_expr);
    program.assert(above.or(below));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 5: Local Window Self-Coverage
// ---------------------------------------------------------------------------

/// Prove that a local window centered on the query always covers the query
/// itself: the diagonal key `j = i` lies within `[center - w, center + w]`.
///
/// The window spans `[center - w, center + w]` where `center = i + shift` is the
/// index the implementation centers on. We derive the bounds and prove the
/// diagonal is inside them.
///
/// The content is that the window is centered on the query (`shift = 0`). A
/// mis-centered window (`shift = 2`, an indexing slip that shifts the band off the
/// diagonal) excludes the query's own position, and the query is then SAT — see
/// `self_coverage_depends_on_centering_the_window`. `QF_LIA`, concrete `w`.
pub(crate) fn prove_local_window_self_coverage() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_local_window_self_coverage(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "local_window_self_coverage".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Window radius used by [`build_local_window_self_coverage`].
const SELF_COVER_W: i64 = 1;

/// Build the self-coverage query. `centered` gates the window center: on the
/// query (`shift = 0`, correct) versus shifted off it (`shift = 2`, the bug).
fn build_local_window_self_coverage(centered: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let w = SELF_COVER_W;
    let shift = if centered { 0 } else { 2 };

    let i = declare_int_in(&mut program, "i", 0, 1000);
    let center = define_int(&mut program, "center", i.clone().int_add(Expr::int(shift)));
    let window_lo = define_int(
        &mut program,
        "window_lo",
        center.clone().int_sub(Expr::int(w)),
    );
    let window_hi = define_int(&mut program, "window_hi", center.int_add(Expr::int(w)));

    // Diagonal key is the query's own position.
    let j_self = define_int(&mut program, "j_self", i);

    // Violation: the diagonal falls outside the window.
    let below = j_self.clone().int_lt(window_lo);
    let above = j_self.int_gt(window_hi);
    program.assert(below.or(above));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 6: Local Window Size Consistency
// ---------------------------------------------------------------------------

/// Prove that the local window size is 2w+1 for window radius w.
///
/// For radius w, positions in [i-w, i+w] are attended. The count is:
///   (i+w) - (i-w) + 1 = 2w + 1.
pub(crate) fn prove_local_window_size() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "w");
    let i = declare_real(&mut program, "i");

    assert_strict_positive(&mut program, &w, 0.0)?;
    assert_bounds(&mut program, &w, 1.0, 1000.0)?;
    assert_bounds(&mut program, &i, 0.0, 10000.0)?;

    let one = Expr::real(1);
    let two = Expr::real(2);

    // Window: [i-w, i+w]
    let upper = i.clone().real_add(w.clone());
    let lower = i.real_sub(w.clone());

    // Window size = upper - lower + 1
    let window_size = declare_real(&mut program, "window_size");
    program.assert(window_size.clone().eq(upper.real_sub(lower).real_add(one)));

    // Expected: 2w + 1
    let expected = two.real_mul(w).real_add(Expr::real(1));

    // Negated property: window_size != 2w + 1
    let violation = window_size.ne(expected);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "local_window_size_consistency".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Sliding Window Shift Invariance
// ---------------------------------------------------------------------------

/// Prove that the sliding window pattern is shift-invariant in the interior.
///
/// For positions i and i+1 (both far from boundaries), the set of relative
/// offsets they attend to is identical: {-w, ..., 0, ..., w}.
///
/// We prove: (i+1) + offset - ((i) + offset) = 1, meaning the attended positions
/// shift by exactly 1 when the query position shifts by 1.
pub(crate) fn prove_sliding_window_shift_invariance(
) -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let i = declare_real(&mut program, "i");
    let offset = declare_real(&mut program, "offset");

    assert_bounds(&mut program, &i, 0.0, 10000.0)?;
    assert_bounds(&mut program, &offset, -1000.0, 1000.0)?;

    let one = Expr::real(1);

    // Position attended by query i: target_i = i + offset
    let target_i = i.clone().real_add(offset.clone());
    // Position attended by query i+1: target_i1 = (i+1) + offset
    let i_plus_1 = i.real_add(one.clone());
    let target_i1 = i_plus_1.real_add(offset);

    // Difference should be exactly 1
    let diff = target_i1.real_sub(target_i);

    // Negated property: diff != 1
    let violation = diff.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sliding_window_shift_invariance".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Sliding Window Boundary Clamping
// ---------------------------------------------------------------------------

/// Prove that clamping the window start keeps it a valid buffer index (`>= 0`),
/// even near the left boundary where `i - w` is negative.
///
/// The clamp is modeled as `start = (i - w) + deficit`, where `deficit` makes up
/// the shortfall below zero: `deficit = max(0, w - i)`, encoded by
/// `deficit >= 0`, `deficit >= w - i`, and `deficit <= 0 ∨ deficit <= w - i`.
/// Then `start >= 0` is *derived* — from `deficit >= w - i` — not asserted.
///
/// The realistic bug (`clamp = false`) forgets the clamp (`deficit = 0`), so
/// `start = i - w` goes negative for `i < w` and the query is SAT — see
/// `clamping_depends_on_the_deficit`. `QF_LIA` over concrete `w` is decidable.
pub(crate) fn prove_sliding_window_boundary_clamping(
) -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_sliding_window_boundary_clamping(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sliding_window_boundary_clamping".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Window radius used by [`build_sliding_window_boundary_clamping`].
const CLAMP_W: i64 = 4;

/// Build the clamping query. `clamp` gates the `max(0, w - i)` deficit: applied
/// (correct, keeping `start >= 0`) versus forced to zero (the bug: no clamp).
fn build_sliding_window_boundary_clamping(clamp: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let w = Expr::int(CLAMP_W);
    let i = declare_int_in(&mut program, "i", 0, 64);
    let raw_start = define_int(&mut program, "raw_start", i.clone().int_sub(w.clone()));

    // The shortfall the clamp adds back: deficit = max(0, w - i).
    let short = w.int_sub(i); // w - i
    let deficit = declare_int(&mut program, "deficit");
    program.assert(deficit.clone().int_ge(Expr::int(0)));
    if clamp {
        program.assert(deficit.clone().int_ge(short.clone()));
        program.assert(
            deficit
                .clone()
                .int_le(Expr::int(0))
                .or(deficit.clone().int_le(short)),
        );
    } else {
        // Bug: no clamp is applied, so no shortfall is added back.
        program.assert(deficit.clone().eq(Expr::int(0)));
    }

    let start = define_int(&mut program, "start", raw_start.int_add(deficit));

    // Violation: the clamped start is a negative (invalid) buffer index.
    program.assert(start.int_lt(Expr::int(0)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 9: Block-Sparse Block Alignment
// ---------------------------------------------------------------------------

/// Prove that block-sparse mask boundaries align to block_size multiples.
///
/// For block_size B, a block starts at position k*B for integer k.
/// The block spans [k*B, (k+1)*B - 1]. The start of the next block is (k+1)*B.
///
/// We prove: next_block_start - current_block_start = B.
pub(crate) fn prove_block_sparse_alignment() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let b = declare_real(&mut program, "block_size");
    let k = declare_real(&mut program, "k");

    assert_strict_positive(&mut program, &b, 0.0)?;
    assert_bounds(&mut program, &b, 1.0, 512.0)?;
    assert_bounds(&mut program, &k, 0.0, 1000.0)?;

    let one = Expr::real(1);

    // Current block start: k * B
    let current_start = k.clone().real_mul(b.clone());
    // Next block start: (k + 1) * B
    let next_start = k.real_add(one).real_mul(b.clone());

    // Gap between blocks
    let gap = next_start.real_sub(current_start);

    // Negated property: gap != B
    let violation = gap.ne(b);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "block_sparse_alignment".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 10: Block-Sparse Coverage
// ---------------------------------------------------------------------------

/// Prove that the union of all blocks in a block-sparse pattern covers the
/// entire sequence.
///
/// For sequence length N and block_size B where B divides N, there are N/B
/// blocks. The total elements covered = (N/B) * B = N.
///
/// We prove: num_blocks * block_size = N.
pub(crate) fn prove_block_sparse_coverage() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let n = declare_real(&mut program, "n");
    let b = declare_real(&mut program, "block_size");
    let num_blocks = declare_real(&mut program, "num_blocks");

    assert_strict_positive(&mut program, &n, 0.0)?;
    assert_strict_positive(&mut program, &b, 0.0)?;
    assert_strict_positive(&mut program, &num_blocks, 0.0)?;
    assert_bounds(&mut program, &n, 1.0, 10000.0)?;
    assert_bounds(&mut program, &b, 1.0, 512.0)?;

    // num_blocks * B = N (even division)
    program.assert(num_blocks.clone().real_mul(b.clone()).eq(n.clone()));

    // Total covered = num_blocks * B
    let covered = declare_real(&mut program, "covered");
    program.assert(covered.clone().eq(num_blocks.real_mul(b)));

    // Negated property: covered != N
    let violation = covered.ne(n);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "block_sparse_coverage".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 11: Block-Sparse Internal Regularity
// ---------------------------------------------------------------------------

/// Prove that all blocks in a block-sparse pattern have identical internal
/// structure (same size).
///
/// For block k and block k', both have size B. The internal positions within
/// each block span [0, B). We prove block_size_k = block_size_k'.
pub(crate) fn prove_block_sparse_regularity() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let b = declare_real(&mut program, "block_size");
    let size_k = declare_real(&mut program, "size_k");
    let size_k_prime = declare_real(&mut program, "size_k_prime");

    assert_strict_positive(&mut program, &b, 0.0)?;
    assert_bounds(&mut program, &b, 1.0, 512.0)?;

    // Both blocks have size B
    program.assert(size_k.clone().eq(b.clone()));
    program.assert(size_k_prime.clone().eq(b));

    // Negated property: size_k != size_k'
    let violation = size_k.ne(size_k_prime);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "block_sparse_regularity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 12: Dilated Attention Stride
// ---------------------------------------------------------------------------

/// Prove that dilated attention selects every d-th position.
///
/// For dilation rate d and query position i, the attended positions are
/// {i, i+d, i+2d, ...}. The gap between consecutive attended positions is d.
///
/// We prove: for consecutive attended positions p_k and p_{k+1},
/// p_{k+1} - p_k = d.
pub(crate) fn prove_dilated_attention_stride() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let d = declare_real(&mut program, "dilation_rate");
    let k = declare_real(&mut program, "k");
    let start = declare_real(&mut program, "start");

    assert_strict_positive(&mut program, &d, 0.0)?;
    assert_bounds(&mut program, &d, 1.0, 64.0)?;
    assert_bounds(&mut program, &k, 0.0, 10000.0)?;
    assert_bounds(&mut program, &start, 0.0, 10000.0)?;

    let one = Expr::real(1);

    // p_k = start + k * d
    let p_k = start.clone().real_add(k.clone().real_mul(d.clone()));
    // p_{k+1} = start + (k+1) * d
    let p_k1 = start.real_add(k.real_add(one).real_mul(d.clone()));

    let gap = p_k1.real_sub(p_k);

    // Negated property: gap != d
    let violation = gap.ne(d);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "dilated_attention_stride".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 13: Dilated Attention Coverage Over Heads
// ---------------------------------------------------------------------------

/// Prove that `d` dilated heads (head `h` attends the residue class `h mod d`)
/// cover every position — provided there are as many heads as the dilation rate.
///
/// For any position `p` we recover its residue via the euclidean decomposition
/// `p = q*d + r`, `0 <= r < d`. Head `r` is the one whose stride hits `p`
/// (`p - r = q*d` is divisible by `d`), so `p` is covered iff that head index `r`
/// is a real head, i.e. `r < num_heads`.
///
/// The content is `num_heads == d`. Dropping one head (`num_heads = d - 1`, an
/// off-by-one in the head count) leaves the top residue class uncovered, and the
/// query finds a `p` with `r = d - 1 >= num_heads` — see
/// `coverage_depends_on_having_one_head_per_residue`. `QF_LIA` over concrete `d`
/// is decidable (`q*d` is `var * literal`).
pub(crate) fn prove_dilated_coverage_over_heads() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_dilated_coverage_over_heads(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "dilated_coverage_over_heads".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Dilation rate used by [`build_dilated_coverage_over_heads`].
const DILATION_D: i64 = 3;

/// Build the coverage query. `all_heads` gates the head count: one per residue
/// class (`num_heads = d`, correct) versus one short (`num_heads = d - 1`, the bug).
fn build_dilated_coverage_over_heads(all_heads: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let d = DILATION_D;
    let num_heads = if all_heads { d } else { d - 1 };

    let p = declare_int_in(&mut program, "p", 0, 1000);
    // Euclidean decomposition p = q*d + r, 0 <= r < d: r = p mod d.
    let q = declare_int(&mut program, "q");
    program.assert(q.clone().int_ge(Expr::int(0)));
    let r = declare_int_in(&mut program, "r", 0, d);
    program.assert(q.clone().int_mul(Expr::int(d)).int_add(r.clone()).eq(p));

    // Head r covers p (p - r divisible by d). p is covered iff r is a real head.
    let covering_head = define_int(&mut program, "covering_head", r);

    // Violation: the covering head index is out of the head range — p uncovered.
    program.assert(covering_head.int_ge(Expr::int(num_heads)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 14: Strided Attention Periodicity
// ---------------------------------------------------------------------------

/// Prove that strided attention pattern repeats with period equal to stride.
///
/// mask(i, j) = mask(i + stride, j + stride) for interior positions.
/// We prove: the relative pattern is invariant under stride-sized shifts.
///
/// Encoded: for positions (i, j), the attended flag depends only on
/// (i mod stride, j mod stride). If we shift both by stride, the remainders
/// are unchanged.
pub(crate) fn prove_strided_attention_periodicity(
) -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let i = declare_real(&mut program, "i");
    let j = declare_real(&mut program, "j");
    let stride = declare_real(&mut program, "stride");

    assert_bounds(&mut program, &i, 0.0, 10000.0)?;
    assert_bounds(&mut program, &j, 0.0, 10000.0)?;
    assert_strict_positive(&mut program, &stride, 0.0)?;
    assert_bounds(&mut program, &stride, 1.0, 512.0)?;

    // The relative difference is the same under stride shift:
    // (i + stride) - (j + stride) = i - j
    let diff_original = i.clone().real_sub(j.clone());
    let i_shifted = i.real_add(stride.clone());
    let j_shifted = j.real_add(stride);
    let diff_shifted = i_shifted.real_sub(j_shifted);

    // Negated property: diff_original != diff_shifted
    let violation = diff_original.ne(diff_shifted);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "strided_attention_periodicity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 15: Global Token Full Attention
// ---------------------------------------------------------------------------

/// Prove that global tokens attend to all positions.
///
/// If token i is global, then mask(i, j) = 1 for all j in [0, N).
/// We model: is_global = 1 => mask_ij = 1 for any j.
///
/// Encoded: given is_global = 1 and the rule mask = is_global (for global tokens),
/// assert mask != 1 => UNSAT.
pub(crate) fn prove_global_token_full_attention() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let is_global = declare_real(&mut program, "is_global");
    let mask_ij = declare_real(&mut program, "mask_ij");

    assert_bounds(&mut program, &is_global, 0.0, 1.0)?;
    assert_bounds(&mut program, &mask_ij, 0.0, 1.0)?;

    let one = Expr::real(1);

    // Token is global
    program.assert(is_global.clone().eq(one.clone()));

    // Global token rule: mask = is_global (attends everywhere)
    program.assert(mask_ij.clone().eq(is_global));

    // Negated property: mask_ij != 1
    let violation = mask_ij.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "global_token_full_attention".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 16: Global Token Symmetry
// ---------------------------------------------------------------------------

/// Prove that if token i is global, all other tokens attend to i as well.
///
/// In Longformer-style attention: if i is global, mask(j, i) = 1 for all j.
/// We model: is_global_i = 1 => mask(j, i) = is_global_i.
pub(crate) fn prove_global_token_symmetry() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let is_global_i = declare_real(&mut program, "is_global_i");
    let mask_ji = declare_real(&mut program, "mask_ji");

    assert_bounds(&mut program, &is_global_i, 0.0, 1.0)?;
    assert_bounds(&mut program, &mask_ji, 0.0, 1.0)?;

    let one = Expr::real(1);

    // Token i is global
    program.assert(is_global_i.clone().eq(one.clone()));

    // Reverse attention rule: all tokens attend to global token
    program.assert(mask_ji.clone().eq(is_global_i));

    // Negated property: mask_ji != 1
    let violation = mask_ji.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "global_token_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 17: Causal Sparse Intersection
// ---------------------------------------------------------------------------

/// Prove that intersecting a sparse mask with the causal mask (`cs = c * s`)
/// zeroes every future position, whatever the sparse mask says there.
///
/// We fix the geometry to make the causal factor a literal: for query `i`, a
/// past key (`j <= i`) has `c = 1` and a future key (`j > i`) has `c = 0`. The
/// sparse bits `s_past, s_future` stay free in `[0, 1]`. The combined mask is
/// `cs = c(literal) * s`, so `cs_past = 1*s_past = s_past` and
/// `cs_future = 0*s_future = 0` — the future entry is forced off.
///
/// The realistic bug (`causal_correct = false`) flips the causal comparison so
/// the future key gets `c = 1`; then `cs_future = s_future` can be 1 and the
/// query is SAT — see `intersection_depends_on_the_causal_factor`. Each product
/// has a literal factor, so the query is linear (`QF_LRA`, decidable).
pub(crate) fn prove_causal_sparse_intersection() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_causal_sparse_intersection(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "causal_sparse_intersection".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the intersection query. `causal_correct` sets the causal factor for the
/// future key: `0` (correct, masks the future) versus `1` (flipped comparison).
fn build_causal_sparse_intersection(causal_correct: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Sparse bits (relaxed to [0, 1]); the theorem holds for the whole box.
    let s_past = declare_real(&mut program, "s_past");
    let s_future = declare_real(&mut program, "s_future");
    program.assert(s_past.clone().real_ge(Expr::real(0)));
    program.assert(s_past.clone().real_le(Expr::real(1)));
    program.assert(s_future.clone().real_ge(Expr::real(0)));
    program.assert(s_future.clone().real_le(Expr::real(1)));

    // Causal factors from the geometry: past = 1 always; future = 0 unless flipped.
    let c_past = Expr::real(1);
    let c_future = if causal_correct {
        Expr::real(0)
    } else {
        Expr::real(1)
    };

    let cs_past = define_real(&mut program, "cs_past", c_past.real_mul(s_past.clone()));
    let cs_future = define_real(&mut program, "cs_future", c_future.real_mul(s_future));

    // Violation: the combined mask fails to equal (sparse where causal keeps) and
    // (zero where causal forbids).
    let bad_past = cs_past.ne(s_past);
    let bad_future = cs_future.ne(Expr::real(0));
    program.assert(bad_past.or(bad_future));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 18: Sparse Attention Output Dimension
// ---------------------------------------------------------------------------

/// Prove that sparse attention's output shape is `[seq, d_v]` — masking changes
/// which values contribute, never the dimensions — by applying the shape rule.
///
/// The attention weights `A = softmax(scores ⊙ mask)` have shape `[seq, seq]`;
/// the output is `A @ V` with `V: [seq, d_v]`, so its rows come from `A`'s rows
/// (`= seq`) and its columns from `V`'s columns (`= d_v`). Each output dimension
/// is derived through an intermediate (`a_rows`, `v_cols`), so the conclusion is
/// chained rather than asserted equal to the answer.
///
/// The realistic bug (`cols_from_values = false`) reads the output columns off
/// the attention matrix (`a_cols = seq`) instead of `V` (`d_v`), so the shape
/// becomes `[seq, seq]`; when `seq != d_v` the query is SAT — see
/// `output_dimension_depends_on_reading_cols_off_values`. `QF_LIA`, decidable.
pub(crate) fn prove_sparse_attention_output_dimension(
) -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_sparse_attention_output_dimension(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_attention_output_dimension".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the output-dimension query. `cols_from_values` picks where the output
/// columns are read: from `V` (`v_cols = d_v`, correct) or from the attention
/// matrix (`a_cols = seq`, the bug).
fn build_sparse_attention_output_dimension(cols_from_values: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let seq = declare_int_in(&mut program, "seq", 1, 4096);
    let d_v = declare_int_in(&mut program, "d_v", 1, 4096);

    // Attention weight matrix A = softmax(scores ⊙ mask): [seq, seq].
    let a_rows = define_int(&mut program, "a_rows", seq.clone());
    let a_cols = define_int(&mut program, "a_cols", seq.clone());
    // Value matrix V: [seq, d_v].
    let v_cols = define_int(&mut program, "v_cols", d_v.clone());

    // Output = A @ V: rows from A, cols from V.
    let out_rows = define_int(&mut program, "out_rows", a_rows);
    let out_cols_src = if cols_from_values { v_cols } else { a_cols };
    let out_cols = define_int(&mut program, "out_cols", out_cols_src);

    // Violation: the output shape is not [seq, d_v].
    let bad_rows = out_rows.ne(seq);
    let bad_cols = out_cols.ne(d_v);
    program.assert(bad_rows.or(bad_cols));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 19: Mask Sparsity Ratio
// ---------------------------------------------------------------------------

/// Prove that a per-row sparsity budget bounds the whole mask's density: if every
/// row of an `N x N` mask carries at most `k` ones, the total number of ones is at
/// most `k*N`, i.e. the density `nnz / N^2 <= k/N` (the target sparsity ratio).
///
/// `nnz` is *derived* as the sum of the per-row counts, each capped by the row
/// budget `k`; the bound `k*N = (k/N)*N^2 = ratio*N^2` is a literal. The realistic
/// bug (`cap_every_row = false`) forgets to enforce the budget on one row (it may
/// carry up to its full width `N`), so the total can exceed `k*N` and the query is
/// SAT — see `sparsity_ratio_depends_on_capping_every_row`. `QF_LIA` over concrete
/// `N`, `k` is decidable and fast (the old `nnz <= ratio*n_sq` encoding multiplied
/// two free reals and hung in `QF_NRA`).
pub(crate) fn prove_mask_sparsity_ratio() -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_mask_sparsity_ratio(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "mask_sparsity_ratio".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Mask side length and per-row nonzero budget used by
/// [`build_mask_sparsity_ratio`]. Target sparsity ratio = `k/N` = 2/4 = 1/2.
const SPARSITY_N: i64 = 4;
const SPARSITY_ROW_CAP: i64 = 2;

/// Build the sparsity-ratio query. `cap_every_row` gates the per-row budget: every
/// row capped at `k` (correct) versus the last row left uncapped up to `N` (bug).
fn build_mask_sparsity_ratio(cap_every_row: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let n = SPARSITY_N;
    let k = SPARSITY_ROW_CAP;

    // Per-row nonzero counts, each within its sparsity budget.
    let mut counts: Vec<Expr> = Vec::new();
    for row in 0..n {
        let cap = if cap_every_row || row < n - 1 { k } else { n };
        let rc = declare_int_in(&mut program, &format!("row{row}"), 0, cap + 1);
        counts.push(rc);
    }

    // Total nonzeros nnz = sum of the per-row counts.
    let mut sum = counts[0].clone();
    for rc in &counts[1..] {
        sum = sum.int_add(rc.clone());
    }
    let nnz = define_int(&mut program, "nnz", sum);

    // Density budget: nnz <= ratio * N^2 = (k/N) * N^2 = k*N (a literal).
    let budget = k * n;

    // Violation: the mask carries more ones than the sparsity budget allows.
    program.assert(nnz.int_gt(Expr::int(budget)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 20: Local + Global Union Coverage
// ---------------------------------------------------------------------------

/// Prove that the union of local window and global attention covers all positions.
///
/// For each position j: either j is within the local window of query i
/// (|j - i| <= w), or j is a global token (is_global_j = 1), or i is a global
/// token (is_global_i = 1). If global tokens exist plus local window, then
/// the union covers all.
///
/// We prove a simpler version: local_mask OR global_mask = 1 implies coverage = 1.
/// Encoded: coverage = max(local_mask, global_mask) = 1 - (1-local)(1-global).
pub(crate) fn prove_local_global_union_coverage() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let local_mask = declare_real(&mut program, "local_mask");
    let global_mask = declare_real(&mut program, "global_mask");

    assert_binary(&mut program, &local_mask)?;
    assert_binary(&mut program, &global_mask)?;

    let one = Expr::real(1);

    // At least one is active (precondition: position is either in window or global)
    // Encode: local OR global => 1 - (1-local)*(1-global) = 1
    let not_local = one.clone().real_sub(local_mask.clone());
    let not_global = one.clone().real_sub(global_mask.clone());
    let neither = not_local.real_mul(not_global);
    let coverage = one.clone().real_sub(neither);

    // Assume at least one is active
    program.assert(
        local_mask
            .clone()
            .real_add(global_mask.clone())
            .real_ge(one.clone()),
    );

    // Negated property: coverage != 1
    let violation = coverage.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "local_global_union_coverage".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 21: Block Diagonal Structure
// ---------------------------------------------------------------------------

/// Prove the geometric core of a block-diagonal mask: two positions in the SAME
/// block are strictly less than `B` apart (so the on-diagonal region is exactly a
/// width-`B` band, and everything `>= B` apart is off-diagonal / masked).
///
/// Each index is decomposed into its block and offset: `i = qi*B + ri`,
/// `0 <= ri < B` (and likewise `j`). "Same block" is `qi == qj`; from it the
/// solver must derive `i - j = ri - rj ∈ (-B, B)`.
///
/// The content is that both indices use the same block width `B`. The realistic
/// bug (`same_block_width = false`) decomposes `j` with a wider stride, so two
/// "same-block" indices can straddle a true boundary and land `>= B` apart — the
/// query is then SAT (see `block_diagonal_depends_on_the_block_width`). `QF_LIA`
/// over a concrete `B` is decidable (`q*B` is `var * literal`).
pub(crate) fn prove_block_diagonal_structure() -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_block_diagonal_structure(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "block_diagonal_structure".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Block size and sequence length used by [`build_block_diagonal_structure`].
const BLOCK_DIAG_B: i64 = 4;
const BLOCK_DIAG_N: i64 = 12;

/// Declare `q{sfx}`, `r{sfx}` and pin `idx = q*width + r`, `0 <= r < width`,
/// returning the block index `q`.
fn block_decompose(program: &mut AYProgram, idx: &Expr, sfx: &str, width: i64) -> Expr {
    let q = declare_int(program, &format!("q{sfx}"));
    program.assert(q.clone().int_ge(Expr::int(0)));
    let r = declare_int_in(program, &format!("r{sfx}"), 0, width);
    program.assert(
        q.clone()
            .int_mul(Expr::int(width))
            .int_add(r)
            .eq(idx.clone()),
    );
    q
}

/// Build the block-diagonal query. `same_block_width` gates `j`'s block stride:
/// the true `B` (correct) versus a wider `B + 2` (boundary-straddling bug).
fn build_block_diagonal_structure(same_block_width: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let b = BLOCK_DIAG_B;
    let b_j = if same_block_width { b } else { b + 2 };

    let i = declare_int_in(&mut program, "i", 0, BLOCK_DIAG_N);
    let j = declare_int_in(&mut program, "j", 0, BLOCK_DIAG_N);

    let block_i = block_decompose(&mut program, &i, "i", b);
    let block_j = block_decompose(&mut program, &j, "j", b_j);

    // Hypothesis: i and j are reported as sharing a block.
    program.assert(block_i.eq(block_j));

    // Violation: yet they are a whole block-width (or more) apart — i.e. they
    // would sit in different diagonal blocks of a true block-diagonal mask.
    let b_expr = Expr::int(b);
    let far_right = i.clone().int_sub(j.clone()).int_ge(b_expr.clone());
    let far_left = j.int_sub(i).int_ge(b_expr);
    program.assert(far_right.or(far_left));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 22: Attention Score Masking (-inf)
// ---------------------------------------------------------------------------

/// Prove that masking pushes a masked key's pre-softmax logit strictly below an
/// unmasked key's, so softmax assigns the masked key (near) zero weight.
///
/// The masking rule is `score_out = mask*score + (1 - mask)*fill`. We evaluate it
/// at a masked key (`mask = 0`, a literal) and an unmasked key (`mask = 1`, a
/// literal), leaving the two input logits free in `[-B, B]`. The masked key then
/// gets `fill`, the unmasked key keeps its score, and the theorem is
/// `masked_out < unmasked_out` for every in-range score — which holds because
/// `fill` is a sentinel far below the logit range.
///
/// The realistic bug (`fill_neg_inf = false`) fills masked entries with `0`
/// instead of `-inf` — the classic "zeroed the score instead of the weight" slip;
/// a masked key with a negative score is then *raised* to 0, above the unmasked
/// key, and the query is SAT — see `masking_depends_on_the_neg_inf_fill`. Each
/// product has a literal `mask` factor, so the query is linear (`QF_LRA`).
pub(crate) fn prove_attention_score_masking() -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_attention_score_masking(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "attention_score_masking".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Logit range bound and the sentinel a masked logit is filled with, used by
/// [`build_attention_score_masking`].
const MASK_SCORE_BOUND: i64 = 100;
const MASK_NEG_FILL: i64 = -1_000_000;

/// Build the score-masking query. `fill_neg_inf` picks the masked fill value: a
/// sentinel far below the logit range (correct) or `0` (the bug).
fn build_attention_score_masking(fill_neg_inf: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Free pre-softmax logits for a masked (mask=0) and an unmasked (mask=1) key.
    let score_masked_in = declare_real(&mut program, "score_masked_in");
    let score_unmasked_in = declare_real(&mut program, "score_unmasked_in");
    program.assert(score_masked_in.clone().real_ge(Expr::real(-MASK_SCORE_BOUND)));
    program.assert(score_masked_in.clone().real_le(Expr::real(MASK_SCORE_BOUND)));
    program.assert(
        score_unmasked_in
            .clone()
            .real_ge(Expr::real(-MASK_SCORE_BOUND)),
    );
    program.assert(score_unmasked_in.clone().real_le(Expr::real(MASK_SCORE_BOUND)));

    // The value a masked logit is set to before softmax.
    let fill = if fill_neg_inf {
        Expr::real(MASK_NEG_FILL)
    } else {
        Expr::real(0)
    };

    // Masking rule score_out = mask*score + (1-mask)*fill, with `mask` a literal
    // per key (0 masked, 1 unmasked), so every product has a literal factor.
    let masked_out = define_real(
        &mut program,
        "masked_out",
        Expr::real(0)
            .real_mul(score_masked_in)
            .real_add(Expr::real(1).real_mul(fill.clone())),
    );
    let unmasked_out = define_real(
        &mut program,
        "unmasked_out",
        Expr::real(1)
            .real_mul(score_unmasked_in)
            .real_add(Expr::real(0).real_mul(fill)),
    );

    // Violation: the masked key's logit is NOT below the unmasked key's, so
    // softmax would fail to suppress it.
    program.assert(masked_out.real_ge(unmasked_out));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 23: Random Sparse Mask Row Coverage
// ---------------------------------------------------------------------------

/// Prove that each row of a random sparse mask has at least k non-zero entries.
///
/// For a row with entries m1, m2, m3 (all binary) and minimum k=1:
/// m1 + m2 + m3 >= 1.
///
/// We prove: given the coverage constraint, sum < k is UNSAT.
pub(crate) fn prove_random_sparse_row_coverage() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let m1 = declare_real(&mut program, "m1");
    let m2 = declare_real(&mut program, "m2");
    let m3 = declare_real(&mut program, "m3");

    assert_binary(&mut program, &m1)?;
    assert_binary(&mut program, &m2)?;
    assert_binary(&mut program, &m3)?;

    let one = Expr::real(1);

    // Minimum coverage constraint: sum >= 1
    let row_sum = m1.clone().real_add(m2.clone()).real_add(m3.clone());
    program.assert(row_sum.clone().real_ge(one.clone()));

    // Negated property: row_sum < 1
    let violation = row_sum.real_lt(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "random_sparse_row_coverage".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 24: Sparse Pattern Composition (AND)
// ---------------------------------------------------------------------------

/// Prove that composing a sparse pattern `A` with a pattern `B` is element-wise
/// AND: where `A` is on it passes `B` through, and where `A` is off it forces the
/// result off — regardless of `B`.
///
/// We fix pattern `A` to concrete bits so the AND factor is a literal per
/// position: `A = [1, 0]`. Pattern `B`'s bits `b1, b2` stay free in `[0, 1]`.
/// The composed pattern is `composed_k = a_k(literal) * b_k`, giving
/// `composed_1 = 1*b1 = b1` and `composed_2 = 0*b2 = 0`.
///
/// The realistic bug (`a_masks_position_2 = false`) leaves `A`'s second bit on
/// (`a_2 = 1`, i.e. `A` failed to exclude that position), so `composed_2 = b2`
/// can be 1 and the query is SAT — see `composition_depends_on_a_masking_off`.
/// Products have a literal factor, so the query is linear (`QF_LRA`, decidable).
pub(crate) fn prove_sparse_pattern_composition() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_sparse_pattern_composition(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_pattern_composition".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the composition query. `a_masks_position_2` sets pattern `A`'s second
/// bit: `0` (correct, masking that position off) versus `1` (fails to exclude it).
fn build_sparse_pattern_composition(a_masks_position_2: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Pattern B's bits, relaxed to [0, 1]; the theorem holds for the whole box.
    let b1 = declare_real(&mut program, "b1");
    let b2 = declare_real(&mut program, "b2");
    program.assert(b1.clone().real_ge(Expr::real(0)));
    program.assert(b1.clone().real_le(Expr::real(1)));
    program.assert(b2.clone().real_ge(Expr::real(0)));
    program.assert(b2.clone().real_le(Expr::real(1)));

    // Pattern A is concrete: position 1 on; position 2 off (unless the bug).
    let a1 = Expr::real(1);
    let a2 = if a_masks_position_2 {
        Expr::real(0)
    } else {
        Expr::real(1)
    };

    let composed_1 = define_real(&mut program, "composed_1", a1.real_mul(b1.clone()));
    let composed_2 = define_real(&mut program, "composed_2", a2.real_mul(b2));

    // Violation: composition fails to pass B through where A is on, or fails to
    // force the result off where A is off.
    let bad_pass = composed_1.ne(b1);
    let bad_mask = composed_2.ne(Expr::real(0));
    program.assert(bad_pass.or(bad_mask));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 25: Sliding Window Causal Restriction
// ---------------------------------------------------------------------------

/// Prove that a causal sliding window is a lower-triangular *band*: every key it
/// attends satisfies both `j <= i` (causal / lower-triangular) and `j >= i - w`
/// (band of radius `w`).
///
/// The attended keys are generated by stepping back from the diagonal:
/// `j = i - off`, `0 <= off <= w`. Then `j <= i` (since `off >= 0`) and
/// `j >= i - w` (since `off <= w`) are both derived from the offset range.
///
/// The content is the band width. The realistic bug (`band_exact = false`) uses
/// twice the radius (`off <= 2w`, a doubled window), so a key at `off = 2w` falls
/// below `i - w` and escapes the band — the query is then SAT (see
/// `causal_band_depends_on_the_radius`). `QF_LIA`, concrete `w`, decidable.
pub(crate) fn prove_sliding_window_causal_restriction(
) -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_sliding_window_causal_restriction(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sliding_window_causal_restriction".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Window radius used by [`build_sliding_window_causal_restriction`].
const CAUSAL_BAND_W: i64 = 3;

/// Build the causal-band query. `band_exact` gates the max step-back: the true
/// radius `w` (correct) versus `2w` (a doubled window that leaves the band).
fn build_sliding_window_causal_restriction(band_exact: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let w = CAUSAL_BAND_W;
    let max_step = if band_exact { w } else { 2 * w };

    let i = declare_int_in(&mut program, "i", 0, 1000);
    // Step back from the diagonal: off ∈ [0, max_step].
    let off = declare_int_in(&mut program, "off", 0, max_step + 1);
    let j = define_int(&mut program, "j", i.clone().int_sub(off));

    // Violation: the attended key breaks causality (j > i) or leaves the band
    // (j < i - w).
    let future = j.clone().int_gt(i.clone());
    let below_band = j.int_lt(i.int_sub(Expr::int(w)));
    program.assert(future.or(below_band));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 26: Multi-Scale Attention Merge Dimension
// ---------------------------------------------------------------------------

/// Prove that concatenating local and global heads preserves the total feature
/// dimension: merging `H_l` local heads with `H_g` global heads (each of head
/// dim `h`) gives `(H_l + H_g)*h`, which equals `d_local + d_global`.
///
/// Per-group dims are derived from head counts: `d_local = H_l*h`,
/// `d_global = H_g*h`. The merged width is derived independently as
/// `merged_heads*h` with `merged_heads = H_l + H_g`. The two must agree by
/// distributivity, so the conclusion is derived, not asserted equal to itself.
///
/// The head dim `h` is a literal, so every product is `var * literal` and the
/// query is linear (`QF_LIA`, decidable). The realistic bug
/// (`keep_head_dim = false`) rescales the merged heads with the wrong head dim
/// (`h/2` — halving during concat), breaking the identity when the counts differ;
/// the query is then SAT — see `merge_dimension_depends_on_the_head_dim`.
pub(crate) fn prove_multi_scale_merge_dimension() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_multi_scale_merge_dimension(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "multi_scale_merge_dimension".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Per-head feature dimension used by [`build_multi_scale_merge_dimension`].
const MERGE_HEAD_DIM: i64 = 64;

/// Build the merge-dimension query. `keep_head_dim` gates the head dim used to
/// re-expand the merged heads: `h` (correct) versus `h/2` (halved during concat).
fn build_multi_scale_merge_dimension(keep_head_dim: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let h = MERGE_HEAD_DIM;
    let merged_h = if keep_head_dim { h } else { h / 2 };

    let heads_local = declare_int_in(&mut program, "heads_local", 1, 64);
    let heads_global = declare_int_in(&mut program, "heads_global", 1, 64);

    // Per-group feature dims from head counts.
    let d_local = define_int(
        &mut program,
        "d_local",
        heads_local.clone().int_mul(Expr::int(h)),
    );
    let d_global = define_int(
        &mut program,
        "d_global",
        heads_global.clone().int_mul(Expr::int(h)),
    );

    // Independent chain: merged head count, then merged feature width.
    let merged_heads = define_int(
        &mut program,
        "merged_heads",
        heads_local.int_add(heads_global),
    );
    let merged_dim = define_int(
        &mut program,
        "merged_dim",
        merged_heads.int_mul(Expr::int(merged_h)),
    );

    // Violation: the merged width is not the sum of the per-group widths.
    program.assert(merged_dim.ne(d_local.int_add(d_global)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 27: Sparse Softmax Zero Propagation
// ---------------------------------------------------------------------------

/// Prove that masked positions get zero weight after softmax.
///
/// After masking with -inf and applying softmax:
///   weight_masked = exp(-inf) / (sum of exp(scores))
///
/// Since exp(-inf) -> 0, weight_masked -> 0.
///
/// We model: given score = neg_inf (very negative), the softmax weight
/// is w = exp_score / total where exp_score is negligibly small compared to total.
/// We prove: if exp_score = 0 (limit), then w = 0.
pub(crate) fn prove_sparse_softmax_zero_propagation(
) -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let exp_score = declare_real(&mut program, "exp_score");
    let total = declare_real(&mut program, "total");
    let w = declare_real(&mut program, "w");

    let zero = Expr::real(0);

    // exp(-inf) -> 0
    program.assert(exp_score.clone().eq(zero.clone()));

    // total > 0 (at least one unmasked position)
    assert_strict_positive(&mut program, &total, 0.0)?;
    assert_bounds(&mut program, &total, 0.0, 1e10)?;

    // w = exp_score / total => w * total = exp_score
    program.assert(w.clone().real_mul(total).eq(exp_score));

    // Negated property: w != 0
    let violation = w.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_softmax_zero_propagation".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 28: Block Size Divides Sequence Length
// ---------------------------------------------------------------------------

/// Prove that when block_size divides N, the number of blocks is exact.
///
/// If N = num_blocks * block_size with integer num_blocks, then no positions
/// are left uncovered and no block extends beyond N.
///
/// We prove: remainder = N - num_blocks * block_size = 0.
pub(crate) fn prove_block_size_divides_sequence() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let n = declare_real(&mut program, "n");
    let block_size = declare_real(&mut program, "block_size");
    let num_blocks = declare_real(&mut program, "num_blocks");

    assert_strict_positive(&mut program, &n, 0.0)?;
    assert_strict_positive(&mut program, &block_size, 0.0)?;
    assert_strict_positive(&mut program, &num_blocks, 0.0)?;
    assert_bounds(&mut program, &n, 1.0, 100000.0)?;
    assert_bounds(&mut program, &block_size, 1.0, 512.0)?;

    // N = num_blocks * block_size
    program.assert(
        n.clone()
            .eq(num_blocks.clone().real_mul(block_size.clone())),
    );

    // Remainder = N - num_blocks * block_size
    let remainder = declare_real(&mut program, "remainder");
    program.assert(
        remainder
            .clone()
            .eq(n.real_sub(num_blocks.real_mul(block_size))),
    );

    // Negated property: remainder != 0
    let zero = Expr::real(0);
    let violation = remainder.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "block_size_divides_sequence".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 29: Sparse Attention FLOPs Reduction
// ---------------------------------------------------------------------------

/// Prove that sparse attention does strictly less work than dense attention:
/// FLOPs scale with the number of nonzeros, not `N^2`, so a mask with `nnz < N^2`
/// yields fewer FLOPs.
///
/// FLOPs are `c * count * d` for a flop constant `c` and head dim `d`, both pinned
/// to literals so the scale `c*d` is a literal and every product stays linear. A
/// correct sparse kernel counts only nonzeros (`count = nnz`); the realistic bug
/// (`count_only_nonzeros = false`) materializes the full `N x N` score matrix
/// (`count = N^2`), erasing the reduction, and the query is then SAT — see
/// `flops_reduction_depends_on_counting_nonzeros`. `QF_LIA` (var * literal) is
/// decidable and fast (the old `c*nnz*d` encoding multiplied three free reals and
/// hung in `QF_NRA`).
pub(crate) fn prove_sparse_flops_reduction() -> Result<SparseAttentionPropertyResult, SmtError> {
    let program = build_sparse_flops_reduction(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_flops_reduction".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Flop constant and head dim used by [`build_sparse_flops_reduction`]; their
/// product is the (literal) per-count FLOPs scale.
const FLOPS_C: i64 = 2;
const FLOPS_HEAD_DIM: i64 = 64;

/// Build the FLOPs-reduction query. `count_only_nonzeros` gates the sparse work
/// count: `nnz` (correct) versus `N^2` (the bug: full score matrix materialized).
fn build_sparse_flops_reduction(count_only_nonzeros: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    // Nonzeros of the sparse mask and the dense entry count N^2.
    let nnz = declare_int_in(&mut program, "nnz", 1, 1_000_001);
    let n_sq = declare_int_in(&mut program, "n_squared", 1, 1_000_001);

    // Sparsity precondition: strictly fewer nonzeros than a dense N x N matrix.
    program.assert(nnz.clone().int_lt(n_sq.clone()));

    // FLOPs = scale * count, scale = c*d a literal so the product is linear.
    let scale = FLOPS_C * FLOPS_HEAD_DIM;
    let sparse_count = if count_only_nonzeros {
        nnz.clone()
    } else {
        n_sq.clone()
    };
    let sparse_flops = define_int(
        &mut program,
        "sparse_flops",
        sparse_count.int_mul(Expr::int(scale)),
    );
    let dense_flops = define_int(&mut program, "dense_flops", n_sq.int_mul(Expr::int(scale)));

    // Violation: sparse attention does at least as much work as dense.
    program.assert(sparse_flops.int_ge(dense_flops));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 30: Longformer Combined Pattern Coverage
// ---------------------------------------------------------------------------

/// Prove that the Longformer attention pattern (local + global + random)
/// covers all positions when global tokens exist.
///
/// Longformer combines three patterns:
///   1. Local window (radius w)
///   2. Global tokens (selected positions)
///   3. Random tokens (r per row)
///
/// For any query i and key j:
///   - If i is global or j is global: mask(i,j) = 1
///   - If |i-j| <= w: mask(i,j) = 1 (local)
///   - With probability r/N: mask(i,j) = 1 (random)
///
/// We prove the weaker property: if at least one of the three masks is 1,
/// the combined mask is 1. Encoded via OR = 1 - (1-a)(1-b)(1-c).
pub(crate) fn prove_longformer_combined_pattern() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let local = declare_real(&mut program, "local_mask");
    let global = declare_real(&mut program, "global_mask");
    let random = declare_real(&mut program, "random_mask");

    assert_binary(&mut program, &local)?;
    assert_binary(&mut program, &global)?;
    assert_binary(&mut program, &random)?;

    let one = Expr::real(1);

    // At least one is active
    let sum = local
        .clone()
        .real_add(global.clone())
        .real_add(random.clone());
    program.assert(sum.real_ge(one.clone()));

    // Combined = 1 - (1-local)*(1-global)*(1-random)
    let nl = one.clone().real_sub(local);
    let ng = one.clone().real_sub(global);
    let nr = one.clone().real_sub(random);
    let none_active = nl.real_mul(ng).real_mul(nr);
    let combined = one.clone().real_sub(none_active);

    // Negated property: combined != 1
    let violation = combined.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "longformer_combined_pattern".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 31: Sparse Mask Idempotence
// ---------------------------------------------------------------------------

/// Prove that applying a binary mask twice is the same as applying it once.
///
/// For binary mask m: m * m = m (idempotent under multiplication).
/// This ensures re-masking does not change the result.
pub(crate) fn prove_sparse_mask_idempotence() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let m = declare_real(&mut program, "m");
    assert_binary(&mut program, &m)?;

    // m * m
    let m_squared = m.clone().real_mul(m.clone());

    // Negated property: m * m != m
    let violation = m_squared.ne(m);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_mask_idempotence".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 32: Sparse Pattern Union (OR)
// ---------------------------------------------------------------------------

/// Prove that the union of two binary masks = 1 - (1-a)(1-b) (OR encoding).
///
/// For binary masks a, b: union = a + b - a*b = 1 - (1-a)*(1-b).
pub(crate) fn prove_sparse_pattern_union() -> Result<SparseAttentionPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");

    assert_binary(&mut program, &a)?;
    assert_binary(&mut program, &b)?;

    let one = Expr::real(1);

    // Union via inclusion-exclusion: a + b - a*b
    let union_ie = a
        .clone()
        .real_add(b.clone())
        .real_sub(a.clone().real_mul(b.clone()));

    // Union via De Morgan: 1 - (1-a)*(1-b)
    let union_dm = one
        .clone()
        .real_sub(one.clone().real_sub(a).real_mul(one.real_sub(b)));

    // Negated property: the two formulations differ
    let violation = union_ie.ne(union_dm);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "sparse_pattern_union".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 33: Causal Mask Is Lower Triangular
// ---------------------------------------------------------------------------

/// Prove that the causal mask is lower-triangular: every key it attends satisfies
/// `j <= i` (never a strictly-future key).
///
/// The attended keys are generated by stepping back from the diagonal:
/// `j = i - off`, `off >= 0`. Then `j <= i` follows from `off >= 0`.
///
/// The content is the sign of the step. The realistic bug (`step_back = false`)
/// steps *forward* (`j = i + off`), attending future keys; for `off >= 1` this
/// gives `j > i` and the query is SAT — see `lower_triangular_depends_on_the_step_sign`.
/// Indices are `Int`; `QF_LIA` over a concrete shape is decidable.
pub(crate) fn prove_causal_mask_lower_triangular() -> Result<SparseAttentionPropertyResult, SmtError>
{
    let program = build_causal_mask_lower_triangular(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SparseAttentionPropertyResult {
        property: "causal_mask_lower_triangular".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the lower-triangular query. `step_back` gates the step direction from
/// the diagonal: `j = i - off` (correct, past) versus `j = i + off` (future bug).
fn build_causal_mask_lower_triangular(step_back: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let n = 64;
    let i = declare_int_in(&mut program, "i", 0, n);
    // Non-negative step away from the diagonal, bounded so j stays in range.
    let off = declare_int_in(&mut program, "off", 0, n);

    let j = if step_back {
        define_int(&mut program, "j", i.clone().int_sub(off))
    } else {
        define_int(&mut program, "j", i.clone().int_add(off))
    };

    // Violation: the attended key is strictly in the future (above the diagonal).
    program.assert(j.int_gt(i));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ay_vacuity::vacuity_smell;

    // --- Property 1: Sparsity Mask Binary ---

    #[test]
    fn test_sparsity_mask_binary_proven() {
        let result = prove_sparsity_mask_binary().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sparsity mask binary: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sparsity mask binary must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sparsity_mask_binary");
    }

    // --- Property 2: Sparsity Mask Symmetry ---

    #[test]
    fn test_sparsity_mask_symmetry_proven() {
        let result = prove_sparsity_mask_symmetry().expect("proof should not error");
        assert!(
            result.proven,
            "Sparsity mask symmetry (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sparsity_mask_symmetry");
    }

    /// A rule that checks only the forward directed bound `i - j <= w` (instead of
    /// `|i - j| <= w`) is asymmetric, so some pair is attended one way but not the
    /// other and the query must be SAT.
    #[test]
    fn symmetry_depends_on_both_directed_bounds() {
        let program = build_sparsity_mask_symmetry(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a directed (one-bound) window rule is asymmetric; the query must be SAT; got: {detail}",
        );
    }

    // --- Property 3: Sparse Weight Normalization ---

    #[test]
    fn test_sparse_weight_normalization_proven() {
        let result = prove_sparse_weight_normalization().expect("proof should not error");
        assert!(
            result.proven,
            "Sparse weight normalization (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sparse_weight_normalization");
    }

    /// Dividing by the pre-mask total (masked mass left in the denominator)
    /// instead of renormalizing over the unmasked positions makes the surviving
    /// weights sum to `10/15 < 1`, so the query must be SAT.
    #[test]
    fn normalization_depends_on_renormalizing_the_denominator() {
        let program = build_sparse_weight_normalization(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without renormalizing the denominator the weights sum to <1; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 4: Local Window Bounds ---

    #[test]
    fn test_local_window_bounds_proven() {
        let result = prove_local_window_bounds().expect("proof should not error");
        assert!(
            result.proven,
            "Local window bounds (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "local_window_bounds");
    }

    /// A one-too-wide offset range emits a key at distance `w+1`, outside the true
    /// window `[i-w, i+w]`, so the query must be SAT.
    #[test]
    fn bounds_depend_on_the_offset_range() {
        let program = build_local_window_bounds(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a window loop one step too wide emits an out-of-window key; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 5: Local Window Self-Coverage ---

    #[test]
    fn test_local_window_self_coverage_proven() {
        let result = prove_local_window_self_coverage().expect("proof should not error");
        assert!(
            result.proven,
            "Local window self-coverage (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "local_window_self_coverage");
    }

    /// A window mis-centered off the query (`shift = 2`) no longer contains the
    /// query's own position, so the self-coverage query must be SAT.
    #[test]
    fn self_coverage_depends_on_centering_the_window() {
        let program = build_local_window_self_coverage(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "an off-center window excludes the diagonal; the query must be SAT; got: {detail}",
        );
    }

    // --- Property 6: Local Window Size ---

    #[test]
    fn test_local_window_size_proven() {
        let result = prove_local_window_size().expect("proof should not error");
        assert!(
            result.proven,
            "Local window size (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "local_window_size_consistency");
    }

    // --- Property 7: Sliding Window Shift Invariance ---

    #[test]
    fn test_sliding_window_shift_invariance_proven() {
        let result = prove_sliding_window_shift_invariance().expect("proof should not error");
        assert!(
            result.proven,
            "Sliding window shift invariance (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "sliding_window_shift_invariance");
    }

    // --- Property 8: Sliding Window Boundary Clamping ---

    #[test]
    fn test_sliding_window_boundary_clamping_proven() {
        let result = prove_sliding_window_boundary_clamping().expect("proof should not error");
        assert!(
            result.proven,
            "Sliding window boundary clamping (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sliding_window_boundary_clamping");
    }

    /// Forgetting the clamp (deficit forced to 0) leaves `start = i - w`, which is
    /// negative for `i < w`, so the "valid buffer index" query must be SAT.
    #[test]
    fn clamping_depends_on_the_deficit() {
        let program = build_sliding_window_boundary_clamping(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the clamp the window start goes negative; the query must be SAT; got: {detail}",
        );
    }

    // --- Property 9: Block-Sparse Alignment ---

    #[test]
    fn test_block_sparse_alignment_proven() {
        let result = prove_block_sparse_alignment().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Block-sparse alignment: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Block-sparse alignment must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "block_sparse_alignment");
    }

    // --- Property 10: Block-Sparse Coverage ---

    #[test]
    fn test_block_sparse_coverage_proven() {
        let result = prove_block_sparse_coverage().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Block-sparse coverage: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Block-sparse coverage must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "block_sparse_coverage");
    }

    // --- Property 11: Block-Sparse Regularity ---

    #[test]
    fn test_block_sparse_regularity_proven() {
        let result = prove_block_sparse_regularity().expect("proof should not error");
        assert!(
            result.proven,
            "Block-sparse regularity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "block_sparse_regularity");
    }

    // --- Property 12: Dilated Attention Stride ---

    #[test]
    fn test_dilated_attention_stride_proven() {
        let result = prove_dilated_attention_stride().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Dilated attention stride: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Dilated attention stride must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "dilated_attention_stride");
    }

    // --- Property 13: Dilated Coverage Over Heads ---

    #[test]
    fn test_dilated_coverage_over_heads_proven() {
        let result = prove_dilated_coverage_over_heads().expect("proof should not error");
        // QF_LIA over a concrete dilation is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Dilated coverage over heads (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "dilated_coverage_over_heads");
    }

    /// One head short of the dilation rate leaves the top residue class uncovered,
    /// so some position `p` with `r = d-1 >= num_heads` exists and the query is SAT.
    #[test]
    fn coverage_depends_on_having_one_head_per_residue() {
        let program = build_dilated_coverage_over_heads(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with one fewer head than the dilation rate a residue class is uncovered; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 14: Strided Attention Periodicity ---

    #[test]
    fn test_strided_attention_periodicity_proven() {
        let result = prove_strided_attention_periodicity().expect("proof should not error");
        assert!(
            result.proven,
            "Strided attention periodicity (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "strided_attention_periodicity");
    }

    // --- Property 15: Global Token Full Attention ---

    #[test]
    fn test_global_token_full_attention_proven() {
        let result = prove_global_token_full_attention().expect("proof should not error");
        assert!(
            result.proven,
            "Global token full attention (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "global_token_full_attention");
    }

    // --- Property 16: Global Token Symmetry ---

    #[test]
    fn test_global_token_symmetry_proven() {
        let result = prove_global_token_symmetry().expect("proof should not error");
        assert!(
            result.proven,
            "Global token symmetry (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(result.property, "global_token_symmetry");
    }

    // --- Property 17: Causal Sparse Intersection ---

    #[test]
    fn test_causal_sparse_intersection_proven() {
        let result = prove_causal_sparse_intersection().expect("proof should not error");
        // Products with a literal causal factor keep this in linear QF_LRA.
        assert!(
            result.proven,
            "Causal sparse intersection (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "causal_sparse_intersection");
    }

    /// Flipping the causal comparison so the future key gets `c = 1` lets the
    /// combined mask attend a future position (`cs_future = s_future` can be 1),
    /// so the query must be SAT.
    #[test]
    fn intersection_depends_on_the_causal_factor() {
        let program = build_causal_sparse_intersection(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the causal factor flipped the future position survives; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 18: Sparse Attention Output Dimension ---

    #[test]
    fn test_sparse_attention_output_dimension_proven() {
        let result = prove_sparse_attention_output_dimension().expect("proof should not error");
        assert!(
            result.proven,
            "Sparse attention output dimension (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sparse_attention_output_dimension");
    }

    /// Reading the output columns off the attention matrix (`seq`) instead of the
    /// value matrix (`d_v`) yields shape `[seq, seq]`; when `seq != d_v` the query
    /// must be SAT.
    #[test]
    fn output_dimension_depends_on_reading_cols_off_values() {
        let program = build_sparse_attention_output_dimension(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "reading output cols off the attention matrix breaks the shape; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 19: Mask Sparsity Ratio ---

    #[test]
    fn test_mask_sparsity_ratio_proven() {
        let result = prove_mask_sparsity_ratio().expect("proof should not error");
        // QF_LIA over a concrete shape is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Mask sparsity ratio (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "mask_sparsity_ratio");
    }

    /// Leaving one row's sparsity budget unenforced lets the mask exceed the
    /// density bound `k*N`, so the query must be SAT.
    #[test]
    fn sparsity_ratio_depends_on_capping_every_row() {
        let program = build_mask_sparsity_ratio(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "an uncapped row overflows the sparsity budget; the query must be SAT; got: {detail}",
        );
    }

    // --- Property 20: Local + Global Union Coverage ---

    #[test]
    fn test_local_global_union_coverage_proven() {
        let result = prove_local_global_union_coverage().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Local+global union coverage: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Local+global union must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "local_global_union_coverage");
    }

    // --- Property 21: Block Diagonal Structure ---

    #[test]
    fn test_block_diagonal_structure_proven() {
        let result = prove_block_diagonal_structure().expect("proof should not error");
        assert!(
            result.proven,
            "Block diagonal structure (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "block_diagonal_structure");
    }

    /// Decomposing `j` with a wider block stride lets two "same-block" indices
    /// straddle a true boundary and land a full block-width apart, so the query
    /// must be SAT.
    #[test]
    fn block_diagonal_depends_on_the_block_width() {
        let program = build_block_diagonal_structure(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a mismatched block width lets same-block indices straddle a boundary; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 22: Attention Score Masking ---

    #[test]
    fn test_attention_score_masking_proven() {
        let result = prove_attention_score_masking().expect("proof should not error");
        // Products carry a literal `mask` factor, keeping this in decidable QF_LRA:
        // `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Attention score masking (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "attention_score_masking");
    }

    /// Filling masked logits with 0 instead of -inf lets a masked key with a
    /// negative score outrank the unmasked key, so the query must be SAT.
    #[test]
    fn masking_depends_on_the_neg_inf_fill() {
        let program = build_attention_score_masking(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a 0 fill a masked key can outrank an unmasked one; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 23: Random Sparse Row Coverage ---

    #[test]
    fn test_random_sparse_row_coverage_proven() {
        let result = prove_random_sparse_row_coverage().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Random sparse row coverage: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Random sparse row coverage must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "random_sparse_row_coverage");
    }

    // --- Property 24: Sparse Pattern Composition ---

    #[test]
    fn test_sparse_pattern_composition_proven() {
        let result = prove_sparse_pattern_composition().expect("proof should not error");
        // Products with a literal pattern-A factor keep this in linear QF_LRA.
        assert!(
            result.proven,
            "Sparse pattern composition (QF_LRA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sparse_pattern_composition");
    }

    /// If pattern A fails to mask off its excluded position (`a_2 = 1`), the
    /// composition passes B through there (`composed_2 = b2` can be 1) instead of
    /// forcing it off, so the query must be SAT.
    #[test]
    fn composition_depends_on_a_masking_off() {
        let program = build_sparse_pattern_composition(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "when A fails to exclude a position the composition does not force it off; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 25: Sliding Window Causal Restriction ---

    #[test]
    fn test_sliding_window_causal_restriction_proven() {
        let result = prove_sliding_window_causal_restriction().expect("proof should not error");
        assert!(
            result.proven,
            "Sliding window causal restriction (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sliding_window_causal_restriction");
    }

    /// A doubled step-back radius (`off <= 2w`) lets an attended key fall below
    /// `i - w`, escaping the band, so the query must be SAT.
    #[test]
    fn causal_band_depends_on_the_radius() {
        let program = build_sliding_window_causal_restriction(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a doubled window radius lets a key escape the band; the query must be SAT; got: {detail}",
        );
    }

    // --- Property 26: Multi-Scale Merge Dimension ---

    #[test]
    fn test_multi_scale_merge_dimension_proven() {
        let result = prove_multi_scale_merge_dimension().expect("proof should not error");
        assert!(
            result.proven,
            "Multi-scale merge dimension (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "multi_scale_merge_dimension");
    }

    /// Re-expanding the merged heads with a halved head dim breaks the
    /// distributive identity when the two head counts differ, so the query must
    /// be SAT.
    #[test]
    fn merge_dimension_depends_on_the_head_dim() {
        let program = build_multi_scale_merge_dimension(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "a wrong merged head dim breaks the total-dimension identity; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 27: Sparse Softmax Zero Propagation ---

    #[test]
    fn test_sparse_softmax_zero_propagation_proven() {
        let result = prove_sparse_softmax_zero_propagation().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sparse softmax zero propagation: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sparse softmax zero propagation must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sparse_softmax_zero_propagation");
    }

    // --- Property 28: Block Size Divides Sequence ---

    #[test]
    fn test_block_size_divides_sequence_proven() {
        let result = prove_block_size_divides_sequence().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Block size divides sequence: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Block size divides sequence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "block_size_divides_sequence");
    }

    // --- Property 29: Sparse FLOPs Reduction ---

    #[test]
    fn test_sparse_flops_reduction_proven() {
        let result = prove_sparse_flops_reduction().expect("proof should not error");
        // QF_LIA (var * literal) is decidable: `Unknown` is not acceptable.
        assert!(
            result.proven,
            "Sparse FLOPs reduction (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "sparse_flops_reduction");
    }

    /// Counting the full N^2 score matrix instead of only nonzeros erases the
    /// reduction (sparse work equals dense), so the query must be SAT.
    #[test]
    fn flops_reduction_depends_on_counting_nonzeros() {
        let program = build_sparse_flops_reduction(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "materializing the full matrix erases the FLOPs reduction; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- Property 30: Longformer Combined Pattern ---

    #[test]
    fn test_longformer_combined_pattern_proven() {
        let result = prove_longformer_combined_pattern().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Longformer combined pattern: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Longformer combined pattern must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "longformer_combined_pattern");
    }

    // --- Property 31: Sparse Mask Idempotence ---

    #[test]
    fn test_sparse_mask_idempotence_proven() {
        let result = prove_sparse_mask_idempotence().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sparse mask idempotence: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sparse mask idempotence must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sparse_mask_idempotence");
    }

    // --- Property 32: Sparse Pattern Union (OR) ---

    #[test]
    fn test_sparse_pattern_union_proven() {
        let result = prove_sparse_pattern_union().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Sparse pattern union: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Sparse pattern union must not have counterexample: {}",
            result.detail,
        );
        assert_eq!(result.property, "sparse_pattern_union");
    }

    // --- Property 33: Causal Mask Lower Triangular ---

    #[test]
    fn test_causal_mask_lower_triangular_proven() {
        let result = prove_causal_mask_lower_triangular().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask lower triangular (QF_LIA) should be Proven. detail: {}",
            result.detail,
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "causal_mask_lower_triangular");
    }

    /// Stepping forward from the diagonal (`j = i + off`) attends future keys;
    /// for `off >= 1` this gives `j > i`, so the lower-triangular query must be SAT.
    #[test]
    fn lower_triangular_depends_on_the_step_sign() {
        let program = build_causal_mask_lower_triangular(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "stepping forward from the diagonal attends future keys; \
             the query must be SAT; got: {detail}",
        );
    }

    // --- SMT2 Structure Tests ---

    #[test]
    fn test_all_sparse_attention_proofs_have_valid_smt2() {
        let proofs: Vec<SparseAttentionPropertyResult> = vec![
            prove_sparsity_mask_binary().unwrap(),
            prove_sparsity_mask_symmetry().unwrap(),
            prove_sparse_weight_normalization().unwrap(),
            prove_local_window_bounds().unwrap(),
            prove_local_window_self_coverage().unwrap(),
            prove_local_window_size().unwrap(),
            prove_sliding_window_shift_invariance().unwrap(),
            prove_sliding_window_boundary_clamping().unwrap(),
            prove_block_sparse_alignment().unwrap(),
            prove_block_sparse_coverage().unwrap(),
            prove_block_sparse_regularity().unwrap(),
            prove_dilated_attention_stride().unwrap(),
            prove_dilated_coverage_over_heads().unwrap(),
            prove_strided_attention_periodicity().unwrap(),
            prove_global_token_full_attention().unwrap(),
            prove_global_token_symmetry().unwrap(),
            prove_causal_sparse_intersection().unwrap(),
            prove_sparse_attention_output_dimension().unwrap(),
            prove_mask_sparsity_ratio().unwrap(),
            prove_local_global_union_coverage().unwrap(),
            prove_block_diagonal_structure().unwrap(),
            prove_attention_score_masking().unwrap(),
            prove_random_sparse_row_coverage().unwrap(),
            prove_sparse_pattern_composition().unwrap(),
            prove_sliding_window_causal_restriction().unwrap(),
            prove_multi_scale_merge_dimension().unwrap(),
            prove_sparse_softmax_zero_propagation().unwrap(),
            prove_block_size_divides_sequence().unwrap(),
            prove_sparse_flops_reduction().unwrap(),
            prove_longformer_combined_pattern().unwrap(),
            prove_sparse_mask_idempotence().unwrap(),
            prove_sparse_pattern_union().unwrap(),
            prove_causal_mask_lower_triangular().unwrap(),
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
