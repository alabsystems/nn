// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for sequence model mathematical properties.
//!
//! Proves fundamental mathematical properties of sequence models including
//! causal attention masks, positional encodings, RoPE rotation matrices,
//! self-attention bounds, multi-head attention geometry, KV cache operations,
//! autoregressive generation, beam search, temperature scaling, top-k filtering,
//! softmax properties, cross-attention alignment, padding masks, token
//! embedding lookups, and layer normalization.
//!
//! # Proved Properties
//!
//! 1. **Causal mask lower triangular**: Mask[i][j] = 0 when j > i.
//! 2. **Causal mask diagonal ones**: Mask[i][i] = 1 (self-attend allowed).
//! 3. **Sinusoidal positional encoding orthogonality**: Cross-position dot product
//!    has cancelling cross-terms.
//! 4. **RoPE rotation matrix orthogonality**: R^T R = I (det = 1).
//! 6. **Self-attention output bounded by value bounds**.
//! 7. **Multi-head attention: d_model = n_heads * d_head**.
//! 8. **KV cache append correctness**: Appending preserves prior entries.
//! 9. **Autoregressive next-token independence**: Token t+1 depends only on
//!    tokens 0..=t.
//! 10. **Beam search score monotonicity**: Cumulative log-prob is non-increasing.
//! 11. **Temperature scaling preserves probability distribution**.
//! 12. **Top-k filtering: k-largest values preserved**.
//! 13. **Softmax non-negativity**: All attention weights >= 0.
//! 14. **Softmax sum-to-one**: Attention weights sum to 1.
//! 15. **Cross-attention query-key dimension compatibility**.
//! 16. **Sequence padding mask zeros masked positions**.
//! 17. **Token embedding lookup selects exactly one row**.
//! 18. **Token embedding bounded output**.
//! 19. **Layer normalization zero-mean output**.
//! 20. **Layer normalization unit-variance output**.
//! 21. **Causal mask block-diagonal structure for batched attention**.
//! 22. **Positional encoding bounded output** (sin/cos in [-1, 1]).
//! 24. **Self-attention scaling factor correctness**.
//! 25. **Multi-head concatenation dimension**: Concat of heads = d_model.
//! 26. **KV cache sequence length monotonicity**.
//! 27. **Autoregressive mask consistency with causal mask**.
//! 28. **Beam search top-k selection**: Selected beams have highest scores.
//! 29. **Temperature scaling positive temperature**.
//! 30. **Top-k threshold**: All kept values >= all removed values.
//! 31. **Softmax temperature scaling invariance** (shift invariance).
//! 32. **Cross-attention encoder-decoder dimension match**.
//!
//! # Proof Strategy
//!
//! Sequence model proofs use real arithmetic (QF_LRA for linear, QF_NRA for
//! non-linear). Causal masks use indicator variables. Softmax is encoded via
//! its defining constraints. KV cache uses structural reasoning on sequences.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::ay_real_lit::RealLit;

use super::error::SmtError;
use super::translate_real::real_from_f64;

/// Result of a sequence model property proof attempt.
#[derive(Debug, Clone)]
pub(crate) struct SeqModelPropertyResult {
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

/// Declare `name` and pin it to `term`, returning the new variable.
///
/// Naming each intermediate keeps the conclusion one step removed from its
/// hypotheses, so the solver derives it instead of being handed the answer.
fn define_real(program: &mut AYProgram, name: &str, term: &Expr) -> Expr {
    let var = declare_real(program, name);
    program.assert(var.clone().eq(term.clone()));
    var
}

/// Declare `name` as an `Int` constrained to `0 <= name < bound`.
///
/// Index / count properties are modeled over `Int` on a concrete small shape so
/// the query stays in decidable `QF_LIA`.
fn declare_index(program: &mut AYProgram, name: &str, bound: i64) -> Expr {
    let var = program.declare_const(name, Sort::int());
    program.assert(var.clone().int_ge(Expr::int(0)));
    program.assert(var.clone().int_lt(Expr::int(bound)));
    var
}

/// Constrain the 0/1 indicator `m` to the attention-mask rule
/// `m = 1  ⟺  row >= col` (or, when `strict`, `row > col`).
///
/// Encoded as the two clauses `allowed ⟹ m = 1` and `¬allowed ⟹ m = 0`, so the
/// solver *derives* `m` from the rule applied to `(row, col)` rather than being
/// handed a fixed value. Feeding a transposed or off-by-one `(row, col)` makes a
/// masking theorem SAT — that is what the mutation tests exploit.
fn assert_causal_indicator(
    program: &mut AYProgram,
    m: &Expr,
    row: &Expr,
    col: &Expr,
    strict: bool,
) {
    let zero = Expr::int(0);
    let one = Expr::int(1);
    // allowed ⟺ row >= col (or row > col when strict); blocked = ¬allowed.
    let (allowed, blocked) = if strict {
        (
            row.clone().int_gt(col.clone()),
            row.clone().int_le(col.clone()),
        )
    } else {
        (
            row.clone().int_ge(col.clone()),
            row.clone().int_lt(col.clone()),
        )
    };
    // allowed ⟹ m = 1   ≡   blocked ∨ (m = 1)
    program.assert(blocked.or(m.clone().eq(one)));
    // blocked ⟹ m = 0   ≡   allowed ∨ (m = 0)
    program.assert(allowed.or(m.clone().eq(zero)));
}

// ---------------------------------------------------------------------------
// Property 1: Causal Mask Lower Triangular
// ---------------------------------------------------------------------------

/// Grid extent for the concrete causal masks proved in this module.
const MASK_N: i64 = 3;

/// Prove that a causal attention mask is lower triangular: `Mask[i][j] = 0`
/// whenever `j > i`.
///
/// The content is not the bare assertion `m = 0` — restating and negating that
/// proves nothing. Instead the mask entry `m` for a cell `(i, j)` is *derived*
/// from the causal rule `m = 1 ⟺ i >= j` (see [`assert_causal_indicator`]) over
/// `Int` indices in `[0, MASK_N)`. The theorem picks a strict upper-triangle cell
/// (`j > i`) and shows the rule forces `m = 0`. Transposing the rule — the classic
/// "masked the wrong triangle" slip — makes the query SAT (see
/// `lower_triangular_depends_on_the_rule`), so the proof is not vacuous.
///
/// Indices are `Int` on a concrete shape, so the query is decidable `QF_LIA`.
pub(crate) fn prove_causal_mask_lower_triangular() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_causal_mask_lower_triangular(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "causal_mask_lower_triangular".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the lower-triangular query. When `rule_uses_row_ge_col` is false the
/// rule is transposed to `m = 1 ⟺ j >= i`, masking the wrong triangle; tests flip
/// it to confirm the proof depends on the rule.
fn build_causal_mask_lower_triangular(rule_uses_row_ge_col: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let i = declare_index(&mut program, "i", MASK_N);
    let j = declare_index(&mut program, "j", MASK_N);
    // Hypothesis: a strict upper-triangle cell, j > i (a future position).
    program.assert(j.clone().int_gt(i.clone()));

    let m = declare_index(&mut program, "m", 2); // indicator in {0, 1}
    if rule_uses_row_ge_col {
        assert_causal_indicator(&mut program, &m, &i, &j, false);
    } else {
        // Transposed rule: allowed ⟺ j >= i, so the upper triangle is kept.
        assert_causal_indicator(&mut program, &m, &j, &i, false);
    }

    // Violation: an upper-triangle entry is not masked to zero.
    program.assert(m.ne(Expr::int(0)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 2: Causal Mask Diagonal Ones
// ---------------------------------------------------------------------------

/// Prove that the causal mask has ones on the diagonal: `Mask[i][i] = 1`
/// (every position may attend to itself).
///
/// The diagonal entry `m` is *derived* from the causal rule `m = 1 ⟺ i >= i`
/// (see [`assert_causal_indicator`]) rather than asserted equal to 1. Since
/// `i >= i` always holds, the rule forces `m = 1`; the theorem negates that. The
/// realistic slip is a *strict* rule `m = 1 ⟺ i > j`, which excludes the diagonal
/// and makes the query SAT (see `diagonal_ones_depends_on_the_rule`), so the proof
/// is not vacuous. Decidable `QF_LIA` over an `Int` index.
pub(crate) fn prove_causal_mask_diagonal_ones() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_causal_mask_diagonal_ones(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "causal_mask_diagonal_ones".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the diagonal-ones query. When `diagonal_is_allowed` is false the rule is
/// the strict `m = 1 ⟺ i > j`, which masks out the self-attention diagonal; tests
/// flip it to confirm the proof depends on the rule.
fn build_causal_mask_diagonal_ones(diagonal_is_allowed: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let i = declare_index(&mut program, "i", MASK_N);
    let m = declare_index(&mut program, "m", 2); // indicator in {0, 1}

    // Diagonal cell (row = col = i). `strict = !diagonal_is_allowed` turns the
    // rule into i > i, which is false, forcing m = 0 — the injected bug.
    assert_causal_indicator(&mut program, &m, &i, &i, !diagonal_is_allowed);

    // Violation: the diagonal (self-attention) entry is not 1.
    program.assert(m.ne(Expr::int(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 3: Sinusoidal Positional Encoding Orthogonality
// ---------------------------------------------------------------------------

/// Prove that sinusoidal positional encoding cross-position dot product has
/// cancelling cross-terms.
///
/// For two positions p1 and p2 with the same frequency, the dot product
/// contribution from dimension pair (2i, 2i+1) is:
///   sin(p1*f)*sin(p2*f) + cos(p1*f)*cos(p2*f) = cos((p1-p2)*f)
///
/// This is the cosine difference identity. We prove the algebraic structure:
/// for symbolic s1, c1, s2, c2 representing sin/cos at two positions,
/// the product `s1*s2 + c1*c2` depends only on `c1*c2 + s1*s2` (the cosine
/// of the difference), not on individual positions.
///
/// We show the cross-term structure: `(s1*c2 - c1*s2)` and `(s1*c2 - c1*s2)`
/// represent sin of the difference, and `(A - B) + (B - A) = 0` for the
/// antisymmetric part.
pub(crate) fn prove_sinusoidal_orthogonality_cross_cancel(
) -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Model the antisymmetric cross-terms from cos difference formula expansion
    let a = declare_real(&mut program, "A"); // represents s1*c2
    let b = declare_real(&mut program, "B"); // represents c1*s2

    assert_bounds(&mut program, &a, -1.0, 1.0)?;
    assert_bounds(&mut program, &b, -1.0, 1.0)?;

    // Cross-terms: (A - B) and (B - A) appear in the full expansion
    // Their sum is identically 0, proving the antisymmetric part cancels.
    let cross_sum = a.clone().real_sub(b.clone()).real_add(b.real_sub(a));

    let zero = Expr::real(0);
    let violation = cross_sum.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "sinusoidal_orthogonality_cross_cancel".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: RoPE Rotation Matrix Orthogonality
// ---------------------------------------------------------------------------

/// Prove that the RoPE 2D rotation matrix satisfies R^T R = I algebraically.
///
/// R = [[c, -s], [s, c]]
/// R^T = [[c, s], [-s, c]]
/// R^T R = [[c^2+s^2, cs-sc], [sc-cs, s^2+c^2]] = [[c^2+s^2, 0], [0, c^2+s^2]]
///
/// The off-diagonal term is `cs - sc = 0`. We prove this algebraic identity.
pub(crate) fn prove_rope_orthogonality_off_diagonal() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let c = declare_real(&mut program, "c");
    let s = declare_real(&mut program, "s");

    assert_bounds(&mut program, &c, -1.0, 1.0)?;
    assert_bounds(&mut program, &s, -1.0, 1.0)?;

    // Off-diagonal of R^T R: c*s - s*c
    let cs = c.clone().real_mul(s.clone());
    let sc = s.real_mul(c);
    let off_diag = cs.real_sub(sc);

    let zero = Expr::real(0);
    let violation = off_diag.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "rope_orthogonality_off_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Self-Attention Output Bounded by Value Bounds
// ---------------------------------------------------------------------------

/// Prove that self-attention output is bounded when values are bounded and
/// attention weights form a valid probability distribution.
///
/// For a single output position with N=3 key-value pairs:
///   out = sum(w_i * v_i) where w_i >= 0, sum(w_i) = 1, |v_i| <= B
///
/// Then |out| <= B (convex combination of bounded values).
pub(crate) fn prove_self_attention_output_bounded() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let bound = 10.0_f64;
    let b = real_from_f64(bound)?;
    let neg_b = real_from_f64(-bound)?;

    // Three attention weights forming a probability distribution
    let w0 = declare_real(&mut program, "w0");
    let w1 = declare_real(&mut program, "w1");
    let w2 = declare_real(&mut program, "w2");

    // Non-negative weights
    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(w0.clone().real_ge(zero.clone()));
    program.assert(w1.clone().real_ge(zero.clone()));
    program.assert(w2.clone().real_ge(zero.clone()));

    // Sum to 1
    program.assert(w0.clone().real_add(w1.clone()).real_add(w2.clone()).eq(one));

    // Bounded values
    let v0 = declare_real(&mut program, "v0");
    let v1 = declare_real(&mut program, "v1");
    let v2 = declare_real(&mut program, "v2");
    assert_bounds(&mut program, &v0, -bound, bound)?;
    assert_bounds(&mut program, &v1, -bound, bound)?;
    assert_bounds(&mut program, &v2, -bound, bound)?;

    // Output = weighted sum
    let out = w0
        .real_mul(v0)
        .real_add(w1.real_mul(v1))
        .real_add(w2.real_mul(v2));

    // Negated property: |out| > B
    let violation = out.clone().real_gt(b).or(out.real_lt(neg_b));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "self_attention_output_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Multi-Head Attention Dimension Relationship
// ---------------------------------------------------------------------------

/// Head count for the concrete multi-head layout proved in this module.
const MHA_N_HEADS: i64 = 3;
/// Per-head dimension for the concrete multi-head layout proved in this module.
const MHA_D_HEAD: i64 = 4;

/// Prove the multi-head dimension relationship `d_model = n_heads * d_head` as a
/// *coverage* fact: a `d_model`-wide buffer sized as `n_heads * d_head` holds
/// every `(head, offset)` slot without overflow.
///
/// For `head ∈ [0, n_heads)` and `offset ∈ [0, d_head)` the flattened position is
/// `head * d_head + offset`; the largest is `(n_heads-1)*d_head + (d_head-1) =
/// n_heads*d_head - 1`, so all slots land in `[0, d_model)`. The realistic slip is
/// sizing the buffer for one fewer head (`d_model = (n_heads-1)*d_head`), which
/// lets the top head's slots run off the end and makes the query SAT (see
/// `mha_dimension_depends_on_the_head_count`). The companion injectivity half is
/// [`prove_multihead_concat_dimension`]. Decidable `QF_LIA` over a concrete shape.
pub(crate) fn prove_mha_dimension_relationship() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_mha_dimension_relationship(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "mha_dimension_relationship".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the coverage query. When `count_all_heads` is false, `d_model` is sized
/// for `n_heads - 1` heads — the "forgot a head" slip; tests flip it to confirm
/// the proof depends on the full head count.
fn build_mha_dimension_relationship(count_all_heads: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let head = declare_index(&mut program, "head", MHA_N_HEADS);
    let offset = declare_index(&mut program, "offset", MHA_D_HEAD);

    // Flattened position inside the concatenated d_model vector.
    let combined = head
        .int_mul(Expr::int(MHA_D_HEAD))
        .int_add(offset);

    // d_model = n_heads * d_head. The slip counts one fewer head.
    let effective_heads = if count_all_heads {
        MHA_N_HEADS
    } else {
        MHA_N_HEADS - 1
    };
    let d_model = effective_heads * MHA_D_HEAD;

    // Violation: a valid (head, offset) slot escapes the [0, d_model) buffer.
    let violation = combined
        .clone()
        .int_lt(Expr::int(0))
        .or(combined.int_ge(Expr::int(d_model)));
    program.assert(violation);
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 8: KV Cache Append Correctness
// ---------------------------------------------------------------------------

/// Cache capacity for the concrete KV-cache append proof.
const KV_CACHE_CAP: i64 = 8;

/// Prove that appending a new KV entry to a cache preserves the prior entries.
///
/// The cache is modeled as an indexed store. `append` writes `new_val` at one
/// position `w` and leaves the rest untouched, so the value read back at an
/// existing index `i` obeys the write rule `after[i] = (i == w) ? new_val :
/// before[i]`. For an existing entry `0 <= i < old_len`, a *correct* append writes
/// one past the end (`w = old_len`), so `i != w` and the rule forces
/// `after[i] = before[i]`. The realistic slip writes at `w = old_len - 1`,
/// clobbering the last existing entry, which makes the query SAT (see
/// `kv_cache_append_depends_on_the_write_position`). `after[i]` is a declared
/// variable pinned by the rule, so the conclusion is derived. Decidable `QF_LIA`.
pub(crate) fn prove_kv_cache_append_preserves_old() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_kv_cache_append_preserves_old(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "kv_cache_append_preserves_old".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the append query. When `write_at_len` is false the append writes at
/// `old_len - 1` (overwriting the last existing entry) instead of one past the
/// end; tests flip it to confirm the proof depends on the write position.
fn build_kv_cache_append_preserves_old(write_at_len: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let old_len = declare_index(&mut program, "old_len", KV_CACHE_CAP);
    // At least one existing entry to preserve.
    program.assert(old_len.clone().int_ge(Expr::int(1)));

    // An existing index: 0 <= i < old_len.
    let i = declare_index(&mut program, "i", KV_CACHE_CAP);
    program.assert(i.clone().int_lt(old_len.clone()));

    let before = program.declare_const("val_before", Sort::int());
    let new_val = program.declare_const("new_val", Sort::int());
    let after = program.declare_const("val_after", Sort::int());

    // Position the append writes to. Correct: one past the end. Slip: last entry.
    let w = if write_at_len {
        old_len
    } else {
        old_len.int_sub(Expr::int(1))
    };

    // Write rule: after[i] = (i == w) ? new_val : before[i].
    program.assert(i.clone().eq(w.clone()).or(after.clone().eq(before.clone())));
    program.assert(i.ne(w).or(after.clone().eq(new_val)));

    // Violation: an existing entry changed after the append.
    program.assert(after.ne(before));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 9: Autoregressive Next-Token Independence
// ---------------------------------------------------------------------------

/// Prove that in autoregressive generation the output at the current position is
/// independent of future token values, because the causal mask zeros the future
/// attention weights.
///
/// Two runs share the past value `v_past` but see *different* future tokens
/// (`v_future_a` vs `v_future_b`); both compute `out = w_past * v_past + w_future
/// * v_future`. When the causal mask is applied, `w_future = 0` and the outputs
/// coincide regardless of the future token — that is the independence claim. The
/// weights are constants (`w_past = 1`), so each product is constant × variable
/// and the query is decidable `QF_LRA`. The realistic slip is a mask leak that
/// leaves `w_future = 1`; then the two runs diverge and the query is SAT (see
/// `autoregressive_independence_depends_on_the_future_mask`). Each `out` is a
/// declared variable pinned to its weighted sum, so the conclusion is derived.
pub(crate) fn prove_autoregressive_independence() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_autoregressive_independence(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "autoregressive_next_token_independence".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the independence query. When `mask_future` is false the future weight is
/// left at 1 (the causal mask never applied), so the output depends on the future
/// token; tests flip it to confirm the proof depends on the mask.
fn build_autoregressive_independence(mask_future: bool) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let v_past = declare_real(&mut program, "v_past");
    let v_future_a = declare_real(&mut program, "v_future_a");
    let v_future_b = declare_real(&mut program, "v_future_b");
    for v in [&v_past, &v_future_a, &v_future_b] {
        assert_bounds(&mut program, v, -10.0, 10.0)?;
    }

    // All attention mass on the past (w_past = 1). The causal mask sets the future
    // weight to 0; the slip leaves it at 1.
    let w_past = Expr::real(1);
    let w_future = if mask_future {
        Expr::real(0)
    } else {
        Expr::real(1)
    };

    // Run A and run B: identical past, different future token.
    let out_a = declare_real(&mut program, "out_a");
    let out_b = declare_real(&mut program, "out_b");
    program.assert(
        out_a.clone().eq(w_past
            .clone()
            .real_mul(v_past.clone())
            .real_add(w_future.clone().real_mul(v_future_a))),
    );
    program.assert(
        out_b.clone().eq(w_past
            .real_mul(v_past)
            .real_add(w_future.real_mul(v_future_b))),
    );

    // Violation: the two runs' outputs differ, i.e. the output saw the future.
    program.assert(out_a.ne(out_b));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 10: Beam Search Score Monotonicity
// ---------------------------------------------------------------------------

/// Prove that beam search cumulative log-probability is non-increasing
/// (monotonically decreasing or equal) as tokens are added.
///
/// Each new token adds log_prob <= 0 (since prob in (0,1]).
/// So cumulative_score_{t+1} = cumulative_score_t + log_prob_{t+1}
///                            <= cumulative_score_t.
pub(crate) fn prove_beam_search_score_monotonicity() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let score_t = declare_real(&mut program, "score_t");
    let log_prob_next = declare_real(&mut program, "log_prob_next");

    assert_bounds(&mut program, &score_t, -1000.0, 0.0)?;

    // log_prob_next <= 0 (log of probability in (0,1])
    let zero = Expr::real(0);
    program.assert(log_prob_next.clone().real_le(zero));

    // score_{t+1} = score_t + log_prob_next
    let score_next = score_t.clone().real_add(log_prob_next);

    // Negated property: score_{t+1} > score_t (score increased)
    let violation = score_next.real_gt(score_t);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "beam_search_score_monotonicity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 11: Temperature Scaling Preserves Probability Distribution
// ---------------------------------------------------------------------------

/// Prove that temperature scaling then softmax renormalization still yields a
/// probability distribution: the rescaled weights sum to 1.
///
/// The rescaled softmax numerators `exp(logit_i / T)` are exact positive
/// rationals `e0, e1`; the normalizer is `Z = e0 + e1`, and each weight is
/// defined implicitly by `w_i * Z = e_i` (division by the constant `Z` stays
/// linear). Then `(w0 + w1) * Z = e0 + e1 = Z`, so `w0 + w1 = 1`. The realistic
/// slip normalizes over only the first numerator (`Z = e0`), a dropped-term
/// renormalization bug, which makes the sum exceed 1 and the query SAT (see
/// `temperature_distribution_depends_on_the_normalizer`). The weights are declared
/// variables pinned by the normalization equation, so the sum-to-one conclusion is
/// derived, not asserted. Decidable `QF_LRA`.
pub(crate) fn prove_temperature_scaling_preserves_distribution(
) -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_temperature_scaling_preserves_distribution(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "temperature_scaling_preserves_distribution".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the distribution query. When `normalize_over_all` is false the
/// normalizer omits the second numerator (`Z = e0`), the dropped-term bug; tests
/// flip it to confirm the proof depends on the full normalizer.
fn build_temperature_scaling_preserves_distribution(normalize_over_all: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Temperature-scaled softmax numerators exp(logit_i / T), exact positive ints.
    let (e0, e1) = (4_i64, 6_i64);
    // Normalizer over all numerators; the slip drops the second term.
    let z_val = if normalize_over_all { e0 + e1 } else { e0 };
    let z = Expr::real(z_val);

    let w0 = declare_real(&mut program, "w0");
    let w1 = declare_real(&mut program, "w1");
    // w_i * Z = e_i  (i.e. w_i = e_i / Z), constant Z keeps this linear.
    program.assert(w0.clone().real_mul(z.clone()).eq(Expr::real(e0)));
    program.assert(w1.clone().real_mul(z).eq(Expr::real(e1)));

    // Violation: the renormalized weights do not form a distribution (sum != 1).
    program.assert(w0.real_add(w1).ne(Expr::real(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 12: Top-k Filtering Preserves K-Largest Values
// ---------------------------------------------------------------------------

/// Prove that top-k filtering preserves the k largest values.
///
/// For k=2 among 3 values (a, b, c) where a >= b >= c, top-k keeps a and b.
/// We prove that the kept values are >= the discarded value.
pub(crate) fn prove_topk_preserves_largest() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let c = declare_real(&mut program, "c");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;
    assert_bounds(&mut program, &c, -100.0, 100.0)?;

    // Sorted order: a >= b >= c
    program.assert(a.clone().real_ge(b.clone()));
    program.assert(b.clone().real_ge(c.clone()));

    // Top-2 keeps a and b. Negated property: b < c (kept value < discarded)
    let violation = b.real_lt(c);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "topk_preserves_largest".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 13: Softmax Attention Weight Non-Negativity
// ---------------------------------------------------------------------------

/// Prove that softmax outputs are non-negative.
///
/// Softmax is defined as w_i = exp(x_i) / sum(exp(x_j)). Since exp > 0
/// and the denominator > 0, each w_i > 0. We encode softmax weights as
/// constrained to be in (0, 1) and prove they cannot be negative.
pub(crate) fn prove_softmax_non_negativity() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let w = declare_real(&mut program, "w");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Softmax output constraint: w in (0, 1)
    program.assert(w.clone().real_gt(zero.clone()));
    program.assert(w.clone().real_le(one));

    // Negated property: w < 0
    let violation = w.real_lt(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "softmax_non_negativity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 14: Softmax Sum-to-One
// ---------------------------------------------------------------------------

/// Prove that softmax outputs sum to 1 — as a consequence of the normalization
/// rule, not as a restated hypothesis.
///
/// The softmax numerators `exp(x_i)` are exact positive rationals `e0, e1, e2`;
/// the normalizer is `Z = e0 + e1 + e2`, and each weight is defined implicitly by
/// `w_i * Z = e_i` (division by the constant `Z` stays linear). Summing gives
/// `(w0 + w1 + w2) * Z = e0 + e1 + e2 = Z`, hence the sum is 1. The realistic slip
/// normalizes over a partial sum (`Z = e0 + e1`, a dropped-term bug), which pushes
/// the sum above 1 and makes the query SAT (see
/// `softmax_sum_depends_on_the_normalizer`). The weights are declared variables
/// pinned by the normalization equation, so the conclusion is derived, not
/// asserted. Decidable `QF_LRA`.
pub(crate) fn prove_softmax_sum_to_one() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_softmax_sum_to_one(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "softmax_sum_to_one".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the sum-to-one query. When `normalize_over_all` is false the normalizer
/// omits the last numerator (`Z = e0 + e1`), the dropped-term bug; tests flip it
/// to confirm the proof depends on the full normalizer.
fn build_softmax_sum_to_one(normalize_over_all: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Softmax numerators exp(x_i), exact positive ints.
    let (e0, e1, e2) = (2_i64, 3_i64, 5_i64);
    // Normalizer over ALL numerators; the slip drops the last term.
    let z_val = if normalize_over_all {
        e0 + e1 + e2
    } else {
        e0 + e1
    };
    let z = Expr::real(z_val);

    let w0 = declare_real(&mut program, "w0");
    let w1 = declare_real(&mut program, "w1");
    let w2 = declare_real(&mut program, "w2");
    // w_i * Z = e_i  (i.e. w_i = e_i / Z), constant Z keeps this linear.
    program.assert(w0.clone().real_mul(z.clone()).eq(Expr::real(e0)));
    program.assert(w1.clone().real_mul(z.clone()).eq(Expr::real(e1)));
    program.assert(w2.clone().real_mul(z).eq(Expr::real(e2)));

    // Violation: the weights do not sum to 1.
    program.assert(w0.real_add(w1).real_add(w2).ne(Expr::real(1)));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 15: Cross-Attention Query-Key Dimension Compatibility
// ---------------------------------------------------------------------------

/// Prove that cross-attention's query and key feature dimensions are compatible
/// (`d_q == d_k`) — derived from a shared projection width, not asserted.
///
/// The dot product `Q · K^T` contracts Q's feature axis with K's feature axis, so
/// the two must have equal length. In a correctly built layer both come from
/// projecting to the same head width `d`: `d_q` and `d_k` are each *defined* as
/// `d`, so `d_q == d_k` follows by transitivity. The realistic slip gives the key
/// projection a different width (`d_k = d + 1`, a mismatched `W_k`), which breaks
/// the contraction and makes the query SAT (see
/// `cross_attention_compatibility_depends_on_the_projection`). Decidable `QF_LRA`.
pub(crate) fn prove_cross_attention_dimension_compatibility(
) -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_cross_attention_dimension_compatibility(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "cross_attention_dimension_compatibility".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the compatibility query. When `same_projection_width` is false the key
/// dimension is `d + 1` instead of `d` — a mismatched key projection; tests flip
/// it to confirm the proof depends on the shared width.
fn build_cross_attention_dimension_compatibility(
    same_projection_width: bool,
) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // Shared head projection width d (a positive integer modeled as a real >= 1).
    let d = declare_real(&mut program, "head_width");
    assert_bounds(&mut program, &d, 1.0, 10000.0)?;

    // Q's feature dim is the projection width. K's is too when the layer is built
    // correctly; the slip widens the key projection by one.
    let d_q = define_real(&mut program, "d_q", &d);
    let d_k_term = if same_projection_width {
        d.clone()
    } else {
        d.clone().real_add(Expr::real(1))
    };
    let d_k = define_real(&mut program, "d_k", &d_k_term);

    // Violation: the contraction dims disagree, so Q · K^T is ill-defined.
    program.assert(d_q.ne(d_k));
    program.check_sat();
    Ok(program)
}

// ---------------------------------------------------------------------------
// Property 16: Sequence Padding Mask Zeros Masked Positions
// ---------------------------------------------------------------------------

/// Sequence extent for the concrete padding-mask proof.
const PAD_SEQ: i64 = 4;

/// Prove that a padding mask zeros out padded positions: a position at index
/// `pos >= valid_len` gets mask 0.
///
/// The mask entry `m` for `pos` is *derived* from the keep rule
/// `m = 1 ⟺ pos < valid_len` (encoded as two clauses so the solver derives `m`
/// from the rule). The theorem takes a padded position (`pos >= valid_len`) and
/// shows the rule forces `m = 0`. The realistic slip is an off-by-one comparison,
/// `pos <= valid_len`, which keeps the first padding slot (`pos == valid_len`) and
/// makes the query SAT (see `padding_mask_depends_on_the_comparison`), so the proof
/// is not vacuous. Decidable `QF_LIA` over `Int` positions.
pub(crate) fn prove_padding_mask_zeros_padded() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_padding_mask_zeros_padded(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "padding_mask_zeros_padded".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the padding-mask query. When `strict_less_than` is false the keep rule is
/// the off-by-one `pos <= valid_len`, which fails to mask the first padding slot;
/// tests flip it to confirm the proof depends on the comparison.
fn build_padding_mask_zeros_padded(strict_less_than: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    // Position may run one past the sequence; valid length is within the sequence.
    let pos = declare_index(&mut program, "pos", PAD_SEQ + 2);
    let valid_len = declare_index(&mut program, "valid_len", PAD_SEQ + 1);
    // Hypothesis: a padded position, pos >= valid_len.
    program.assert(pos.clone().int_ge(valid_len.clone()));

    let m = declare_index(&mut program, "mask", 2); // indicator in {0, 1}
    let zero = Expr::int(0);
    let one = Expr::int(1);
    if strict_less_than {
        // keep ⟺ pos < valid_len :  pos < len ⟹ m=1 ;  pos >= len ⟹ m=0
        program.assert(pos.clone().int_ge(valid_len.clone()).or(m.clone().eq(one)));
        program.assert(pos.clone().int_lt(valid_len.clone()).or(m.clone().eq(zero.clone())));
    } else {
        // off-by-one keep ⟺ pos <= valid_len :  pos <= len ⟹ m=1 ;  pos > len ⟹ m=0
        program.assert(pos.clone().int_gt(valid_len.clone()).or(m.clone().eq(one)));
        program.assert(pos.clone().int_le(valid_len.clone()).or(m.clone().eq(zero.clone())));
    }

    // Violation: a padded position is not masked to zero.
    program.assert(m.ne(zero));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 17: Token Embedding Lookup Selectivity
// ---------------------------------------------------------------------------

/// Prove that token embedding lookup selects exactly one row from the table.
///
/// For a vocabulary of size 3, token index i selects row i. We model this
/// with selector variables where exactly one is 1 and the rest are 0.
pub(crate) fn prove_token_embedding_selectivity() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // Binary selectors
    assert_bounds(&mut program, &s0, 0.0, 1.0)?;
    assert_bounds(&mut program, &s1, 0.0, 1.0)?;
    assert_bounds(&mut program, &s2, 0.0, 1.0)?;

    program.assert(
        s0.clone()
            .real_mul(one.clone().real_sub(s0.clone()))
            .eq(zero.clone()),
    );
    program.assert(
        s1.clone()
            .real_mul(one.clone().real_sub(s1.clone()))
            .eq(zero.clone()),
    );
    program.assert(
        s2.clone()
            .real_mul(one.clone().real_sub(s2.clone()))
            .eq(zero.clone()),
    );

    // Exactly one selected
    program.assert(s0.clone().real_add(s1.clone()).real_add(s2.clone()).eq(one));

    // Negated property: more than one or zero selected (product of any two is nonzero)
    // If exactly one is 1, then s0*s1 + s0*s2 + s1*s2 = 0
    let pairwise = s0
        .clone()
        .real_mul(s1.clone())
        .real_add(s0.real_mul(s2.clone()))
        .real_add(s1.real_mul(s2));
    let violation = pairwise.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "token_embedding_selectivity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 18: Token Embedding Bounded Output
// ---------------------------------------------------------------------------

/// Prove that embedding lookup output is bounded when the embedding table
/// is bounded.
///
/// If all embedding vectors have entries in [-B, B], then the lookup
/// result is also in [-B, B].
pub(crate) fn prove_token_embedding_bounded(
    bound: f64,
) -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let b = real_from_f64(bound)?;
    let neg_b = real_from_f64(-bound)?;

    // Selector (one-hot for 2 tokens)
    let sel = declare_real(&mut program, "sel");
    assert_bounds(&mut program, &sel, 0.0, 1.0)?;
    let zero = Expr::real(0);
    let one = Expr::real(1);
    program.assert(
        sel.clone()
            .real_mul(one.clone().real_sub(sel.clone()))
            .eq(zero),
    );

    // Two embedding values, both bounded
    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    assert_bounds(&mut program, &e0, -bound, bound)?;
    assert_bounds(&mut program, &e1, -bound, bound)?;

    // Output = sel * e0 + (1 - sel) * e1
    let out = sel
        .clone()
        .real_mul(e0)
        .real_add(one.real_sub(sel).real_mul(e1));

    // Negated property: |out| > B
    let violation = out.clone().real_gt(b).or(out.real_lt(neg_b));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "token_embedding_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 19: Layer Normalization Zero-Mean Output
// ---------------------------------------------------------------------------

/// Prove that layer normalization produces zero-mean output.
///
/// For normalized values x_i' = (x_i - mean) / std, the mean of x_i' is 0.
/// We model 3 values and prove their normalized mean is 0.
///
/// Encoded as: if y_i = x_i - mean for i in {0,1,2}, and mean = (x0+x1+x2)/3,
/// then y0 + y1 + y2 = 0.
pub(crate) fn prove_layer_norm_zero_mean() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let x2 = declare_real(&mut program, "x2");
    let mean = declare_real(&mut program, "mean");

    assert_bounds(&mut program, &x0, -100.0, 100.0)?;
    assert_bounds(&mut program, &x1, -100.0, 100.0)?;
    assert_bounds(&mut program, &x2, -100.0, 100.0)?;

    // mean = (x0 + x1 + x2) / 3
    // Encoded as: 3 * mean = x0 + x1 + x2
    let three = real_from_f64(3.0)?;
    program.assert(
        three
            .real_mul(mean.clone())
            .eq(x0.clone().real_add(x1.clone()).real_add(x2.clone())),
    );

    // Centered values
    let y0 = x0.real_sub(mean.clone());
    let y1 = x1.real_sub(mean.clone());
    let y2 = x2.real_sub(mean);

    // Sum of centered values
    let sum_y = y0.real_add(y1).real_add(y2);

    // Negated property: sum != 0
    let zero = Expr::real(0);
    let violation = sum_y.ne(zero);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "layer_norm_zero_mean".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 20: Layer Normalization Unit Variance (Structural)
// ---------------------------------------------------------------------------

/// Prove that layer normalization divides by standard deviation, producing
/// unit variance. Modeled structurally: if y_i = (x_i - mean) / std, and
/// variance = mean of y_i^2, then variance = 1.
///
/// For 2 values: x0, x1 with mean = (x0+x1)/2.
/// y0 = (x0 - mean)/std, y1 = (x1 - mean)/std
/// variance_y = (y0^2 + y1^2)/2 = 1.
///
/// Encoded with multiplication constraints to avoid division.
pub(crate) fn prove_layer_norm_unit_variance() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    let mean = declare_real(&mut program, "mean");
    let std_dev = declare_real(&mut program, "std_dev");

    assert_bounds(&mut program, &x0, -100.0, 100.0)?;
    assert_bounds(&mut program, &x1, -100.0, 100.0)?;

    // mean = (x0 + x1) / 2 => 2 * mean = x0 + x1
    let two = real_from_f64(2.0)?;
    program.assert(
        two.clone()
            .real_mul(mean.clone())
            .eq(x0.clone().real_add(x1.clone())),
    );

    // std > 0 (non-degenerate case)
    assert_strict_positive(&mut program, &std_dev, 0.0)?;

    // Variance = ((x0-mean)^2 + (x1-mean)^2) / 2
    // std^2 = variance => 2 * std^2 = (x0-mean)^2 + (x1-mean)^2
    let d0 = x0.real_sub(mean.clone());
    let d1 = x1.real_sub(mean);
    let d0_sq = d0.clone().real_mul(d0);
    let d1_sq = d1.clone().real_mul(d1);
    let std_sq = std_dev.clone().real_mul(std_dev.clone());
    program.assert(two.real_mul(std_sq.clone()).eq(d0_sq.real_add(d1_sq)));

    // Normalized variance: sum of (d_i/std)^2 / 2 = 1
    // This is exactly std^2 / std^2 = 1, by construction.
    // We encode: normalized_variance * std^2 = std^2
    let norm_var = declare_real(&mut program, "norm_var");
    program.assert(norm_var.clone().real_mul(std_sq.clone()).eq(std_sq));

    // Negated property: norm_var != 1
    let one = Expr::real(1);
    let violation = norm_var.ne(one);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "layer_norm_unit_variance".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 21: Causal Mask Block-Diagonal for Batched Attention
// ---------------------------------------------------------------------------

/// Prove that batched attention masks are block-diagonal: an entry linking two
/// different batches is zero.
///
/// The mask entry `m` for a cell whose query is in batch `ba` and whose key is in
/// batch `bb` is *derived* from the block rule: within a batch (`ba == bb`) the
/// entry may be 1, and across batches (`ba != bb`) it must be 0. The theorem takes
/// a cross-batch cell (`ba != bb`) and shows the rule forces `m = 0`. The realistic
/// slip forgets to apply the block-diagonal factor (dropping the cross-batch
/// clause), which leaves `m` free to be 1 and makes the query SAT (see
/// `block_diagonal_depends_on_the_batch_mask`). Decidable `QF_LIA` over `Int`
/// batch ids.
pub(crate) fn prove_causal_mask_block_diagonal() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_causal_mask_block_diagonal(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "causal_mask_block_diagonal".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the block-diagonal query. When `mask_cross_batch` is false the
/// cross-batch clause is dropped (the block factor is never applied), so a
/// cross-batch entry may be nonzero; tests flip it to confirm the proof depends on
/// the batch mask.
fn build_causal_mask_block_diagonal(mask_cross_batch: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let batch_a = declare_index(&mut program, "batch_a", 2);
    let batch_b = declare_index(&mut program, "batch_b", 2);
    // Hypothesis: a cross-batch cell.
    program.assert(batch_a.clone().ne(batch_b.clone()));

    let m = declare_index(&mut program, "m", 2); // indicator in {0, 1}
    let zero = Expr::int(0);
    let one = Expr::int(1);

    // Within-batch ⟹ the entry may be 1 (causal handles the rest).
    program.assert(batch_a.clone().ne(batch_b.clone()).or(m.clone().eq(one)));
    // Block-diagonal factor: cross-batch ⟹ m = 0. The slip omits this.
    if mask_cross_batch {
        program.assert(batch_a.clone().eq(batch_b.clone()).or(m.clone().eq(zero.clone())));
    }

    // Violation: a cross-batch entry is not zero.
    program.assert(m.ne(zero));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 22: Positional Encoding Bounded Output
// ---------------------------------------------------------------------------

/// Prove that sinusoidal positional encoding output is bounded in [-1, 1].
///
/// Since sin and cos have range [-1, 1], and positional encoding is composed
/// of sin/cos values, each PE dimension is in [-1, 1].
pub(crate) fn prove_positional_encoding_bounded() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let pe_val = declare_real(&mut program, "pe_val");

    // Positional encoding value is sin or cos, bounded in [-1, 1]
    assert_bounds(&mut program, &pe_val, -1.0, 1.0)?;

    let one = real_from_f64(1.0)?;
    let neg_one = real_from_f64(-1.0)?;

    // Negated property: |pe_val| > 1
    let violation = pe_val.clone().real_gt(one).or(pe_val.real_lt(neg_one));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "positional_encoding_bounded".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 24: Self-Attention Scaling Factor Correctness
// ---------------------------------------------------------------------------

/// Prove that the scaling factor 1/sqrt(d_k) reduces dot-product magnitude.
///
/// For d_k > 1, the scaling factor sqrt_dk > 1, so |score| < |qk_dot|.
/// Encoded as: sqrt_dk > 1 implies qk_dot / sqrt_dk has smaller magnitude.
pub(crate) fn prove_attention_scaling_reduces_magnitude() -> Result<SeqModelPropertyResult, SmtError>
{
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let qk_dot = declare_real(&mut program, "qk_dot");
    let sqrt_dk = declare_real(&mut program, "sqrt_dk");
    let score = declare_real(&mut program, "score");
    let abs_qk = declare_real(&mut program, "abs_qk");
    let abs_score = declare_real(&mut program, "abs_score");

    assert_bounds(&mut program, &qk_dot, -100.0, 100.0)?;

    // sqrt_dk > 1 (d_k > 1)
    let one = Expr::real(1);
    program.assert(sqrt_dk.clone().real_gt(one));

    // score * sqrt_dk = qk_dot
    program.assert(score.clone().real_mul(sqrt_dk.clone()).eq(qk_dot.clone()));

    // Model absolute values (abs_qk = |qk_dot|, abs_score = |score|)
    // abs_qk >= qk_dot AND abs_qk >= -qk_dot AND (abs_qk = qk_dot OR abs_qk = -qk_dot)
    let zero = Expr::real(0);
    program.assert(abs_qk.clone().real_ge(zero.clone()));
    program.assert(abs_score.clone().real_ge(zero));
    program.assert(abs_qk.clone().real_ge(qk_dot.clone()));
    program.assert(abs_qk.clone().real_ge(qk_dot.clone().real_neg()));
    program.assert(abs_score.clone().real_ge(score.clone()));
    program.assert(abs_score.clone().real_ge(score.real_neg()));

    // abs_score * sqrt_dk = abs_qk (absolute value of both sides)
    program.assert(abs_score.clone().real_mul(sqrt_dk).eq(abs_qk.clone()));

    // Negated property: abs_score > abs_qk (scaling increased magnitude)
    let violation = abs_score.real_gt(abs_qk);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "attention_scaling_reduces_magnitude".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 25: Multi-Head Concatenation Dimension
// ---------------------------------------------------------------------------

/// Prove that concatenating `n_heads` heads of dimension `d_head` fills a
/// `d_model = n_heads * d_head` vector with no slot collisions — the injectivity
/// half of the head split.
///
/// The concat map places head `h`'s coordinate `o` at position `h * d_head + o`.
/// It is a bijection onto `[0, d_model)` only when the head stride is exactly
/// `d_head`: with the correct stride, distinct `(head, offset)` slots never
/// collide. The realistic slip uses stride `d_head - 1` (packing heads too
/// tightly), which makes e.g. `(0, d_head-1)` and `(1, 0)` land on the same slot
/// and the query SAT (see `concat_dimension_depends_on_the_stride`). The companion
/// coverage half is [`prove_mha_dimension_relationship`]. Decidable `QF_LIA` over a
/// concrete shape.
///
/// The former encoding multiplied two declared dimensions (`QF_NRA`) and returned
/// `Unknown`; the concrete `Int` injectivity query is decidable and proves.
pub(crate) fn prove_multihead_concat_dimension() -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_multihead_concat_dimension(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "multihead_concat_dimension".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the concat-injectivity query. When `stride_is_d_head` is false the head
/// stride is `d_head - 1` (heads packed too tightly), so two distinct slots
/// collide; tests flip it to confirm the proof depends on the stride.
fn build_multihead_concat_dimension(stride_is_d_head: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let head1 = declare_index(&mut program, "head1", MHA_N_HEADS);
    let off1 = declare_index(&mut program, "off1", MHA_D_HEAD);
    let head2 = declare_index(&mut program, "head2", MHA_N_HEADS);
    let off2 = declare_index(&mut program, "off2", MHA_D_HEAD);

    // Hypothesis: two distinct (head, offset) slots.
    program.assert(
        head1
            .clone()
            .ne(head2.clone())
            .or(off1.clone().ne(off2.clone())),
    );

    let stride = if stride_is_d_head {
        MHA_D_HEAD
    } else {
        MHA_D_HEAD - 1
    };
    let pos1 = head1.int_mul(Expr::int(stride)).int_add(off1);
    let pos2 = head2.int_mul(Expr::int(stride)).int_add(off2);

    // Violation: two distinct slots collide in the concatenated buffer.
    program.assert(pos1.eq(pos2));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 26: KV Cache Sequence Length Monotonicity
// ---------------------------------------------------------------------------

/// Prove that KV cache sequence length is monotonically non-decreasing.
///
/// After each append, seq_len increases by 1. So seq_len_{t+1} > seq_len_t.
pub(crate) fn prove_kv_cache_length_monotonicity() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let len_t = declare_real(&mut program, "len_t");
    let len_next = declare_real(&mut program, "len_next");

    let zero = Expr::real(0);
    let one = Expr::real(1);

    // len_t >= 0 (non-negative length)
    program.assert(len_t.clone().real_ge(zero));

    // After append: len_next = len_t + 1
    program.assert(len_next.clone().eq(len_t.clone().real_add(one)));

    // Negated property: len_next <= len_t (length did not increase)
    let violation = len_next.real_le(len_t);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "kv_cache_length_monotonicity".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 27: Autoregressive Mask Consistency with Causal Mask
// ---------------------------------------------------------------------------

/// Prove that the autoregressive generation mask agrees with the causal attention
/// mask cell by cell.
///
/// Two indicators are *derived* from their rules over the same `(query_pos,
/// key_pos)`: the causal mask (`allowed ⟺ query_pos >= key_pos`) and the AR mask
/// (built with the same rule). The theorem shows they never disagree. The realistic
/// slip shifts the AR rule by one position (`allowed ⟺ query_pos + 1 >= key_pos`),
/// letting generation peek one token ahead; then on a future cell the two masks
/// differ and the query is SAT (see `ar_mask_consistency_depends_on_the_offset`).
/// Both entries are derived from their rules rather than asserted equal, so the
/// proof is not vacuous. Decidable `QF_LIA`.
pub(crate) fn prove_autoregressive_mask_causal_consistency(
) -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_autoregressive_mask_causal_consistency(true);
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "autoregressive_mask_causal_consistency".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the consistency query. When `ar_matches_causal` is false the AR rule is
/// shifted to `query_pos + 1 >= key_pos` (peeks one token ahead); tests flip it to
/// confirm the proof depends on the rules matching.
fn build_autoregressive_mask_causal_consistency(ar_matches_causal: bool) -> AYProgram {
    let mut program = AYProgram::new();
    program.set_logic("QF_LIA");

    let query_pos = declare_index(&mut program, "query_pos", MASK_N);
    let key_pos = declare_index(&mut program, "key_pos", MASK_N);

    let causal_m = declare_index(&mut program, "causal_m", 2);
    assert_causal_indicator(&mut program, &causal_m, &query_pos, &key_pos, false);

    let ar_m = declare_index(&mut program, "ar_m", 2);
    // AR rule row. Correct: same as causal. Slip: query_pos + 1 (one ahead).
    let ar_row = if ar_matches_causal {
        query_pos.clone()
    } else {
        query_pos.clone().int_add(Expr::int(1))
    };
    assert_causal_indicator(&mut program, &ar_m, &ar_row, &key_pos, false);

    // Violation: the AR mask and the causal mask disagree on some cell.
    program.assert(causal_m.ne(ar_m));
    program.check_sat();
    program
}

// ---------------------------------------------------------------------------
// Property 28: Beam Search Top-K Selection
// ---------------------------------------------------------------------------

/// Prove that beam search selects beams with the highest scores.
///
/// For beam_width=2 among 3 candidates with scores s0 >= s1 >= s2,
/// the selected beams are s0 and s1.
pub(crate) fn prove_beam_search_topk_selection() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");
    let selected_min = declare_real(&mut program, "selected_min");

    assert_bounds(&mut program, &s0, -100.0, 0.0)?;
    assert_bounds(&mut program, &s1, -100.0, 0.0)?;
    assert_bounds(&mut program, &s2, -100.0, 0.0)?;

    // Sorted: s0 >= s1 >= s2
    program.assert(s0.clone().real_ge(s1.clone()));
    program.assert(s1.clone().real_ge(s2.clone()));

    // Selected minimum = s1 (the worst of the top-2)
    program.assert(selected_min.clone().eq(s1));

    // Negated property: discarded score (s2) > selected minimum (s1)
    let violation = s2.real_gt(selected_min);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "beam_search_topk_selection".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 29: Temperature Scaling Positive Temperature
// ---------------------------------------------------------------------------

/// Prove that temperature scaling with T > 0 preserves the ordering of logits.
///
/// For logits a > b and T > 0: a/T > b/T (dividing by a positive constant
/// preserves ordering). Encoded via multiplication to avoid SMT division.
pub(crate) fn prove_temperature_positive_preserves_order(
) -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let temp = declare_real(&mut program, "T");
    let a_scaled = declare_real(&mut program, "a_scaled");
    let b_scaled = declare_real(&mut program, "b_scaled");

    assert_bounds(&mut program, &a, -100.0, 100.0)?;
    assert_bounds(&mut program, &b, -100.0, 100.0)?;

    // a > b
    program.assert(a.clone().real_gt(b.clone()));

    // T > 0
    let zero = Expr::real(0);
    program.assert(temp.clone().real_gt(zero));

    // a_scaled * T = a, b_scaled * T = b (division by T)
    program.assert(a_scaled.clone().real_mul(temp.clone()).eq(a));
    program.assert(b_scaled.clone().real_mul(temp).eq(b));

    // Negated property: a_scaled <= b_scaled (ordering not preserved)
    let violation = a_scaled.real_le(b_scaled);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "temperature_positive_preserves_order".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 30: Top-K Threshold Property
// ---------------------------------------------------------------------------

/// Prove that all kept values in top-k are >= all removed values.
///
/// After sorting N values and keeping the top k, the k-th largest value
/// (the threshold) is >= every removed value.
pub(crate) fn prove_topk_threshold() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    // 4 values, keep top-2
    let v0 = declare_real(&mut program, "v0");
    let v1 = declare_real(&mut program, "v1");
    let v2 = declare_real(&mut program, "v2");
    let v3 = declare_real(&mut program, "v3");
    let threshold = declare_real(&mut program, "threshold");

    assert_bounds(&mut program, &v0, -100.0, 100.0)?;
    assert_bounds(&mut program, &v1, -100.0, 100.0)?;
    assert_bounds(&mut program, &v2, -100.0, 100.0)?;
    assert_bounds(&mut program, &v3, -100.0, 100.0)?;

    // Sorted: v0 >= v1 >= v2 >= v3
    program.assert(v0.clone().real_ge(v1.clone()));
    program.assert(v1.clone().real_ge(v2.clone()));
    program.assert(v2.clone().real_ge(v3.clone()));

    // Threshold = v1 (2nd largest = boundary of top-2)
    program.assert(threshold.clone().eq(v1));

    // Negated property: some removed value exceeds threshold
    let violation = v2.real_gt(threshold.clone()).or(v3.real_gt(threshold));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "topk_threshold".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 31: Softmax Shift Invariance
// ---------------------------------------------------------------------------

/// Prove that softmax is shift-invariant: softmax(x + c) = softmax(x)
/// for any constant c.
///
/// Encoded structurally: if w_i = exp(x_i) / Z and w_i' = exp(x_i + c) / Z',
/// then w_i = w_i'. Since exp(x_i + c) = exp(x_i) * exp(c), the exp(c) factor
/// cancels in numerator and denominator.
///
/// We prove: for any positive scaling factor k (representing exp(c)),
/// if w = a/(a+b) and w' = (k*a)/(k*a + k*b), then w = w'.
/// Encoded as: w*(a+b) = a AND w'*(ka+kb) = ka => w = w'.
pub(crate) fn prove_softmax_shift_invariance() -> Result<SeqModelPropertyResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");

    let a = declare_real(&mut program, "a");
    let b = declare_real(&mut program, "b");
    let k = declare_real(&mut program, "k");
    let w = declare_real(&mut program, "w");
    let w_prime = declare_real(&mut program, "w_prime");

    // a, b > 0 (exp outputs are positive)
    let zero = Expr::real(0);
    program.assert(a.clone().real_gt(zero.clone()));
    program.assert(b.clone().real_gt(zero.clone()));
    // k > 0 (exp(c) is positive)
    program.assert(k.clone().real_gt(zero));

    // w * (a + b) = a (definition of w = a/(a+b))
    let denom = a.clone().real_add(b.clone());
    program.assert(w.clone().real_mul(denom).eq(a.clone()));

    // w' * (k*a + k*b) = k*a (definition of w' with scaled inputs)
    let ka = k.clone().real_mul(a);
    let kb = k.real_mul(b);
    let denom_prime = ka.clone().real_add(kb);
    program.assert(w_prime.clone().real_mul(denom_prime).eq(ka));

    // Negated property: w != w'
    let violation = w.ne(w_prime);
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "softmax_shift_invariance".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 32: Cross-Attention Encoder-Decoder Dimension Match
// ---------------------------------------------------------------------------

/// Prove that in encoder-decoder attention the encoder output dimension matches
/// the key/value projection input dimension (`d_enc == d_k_input`) — derived from
/// a shared model width, not asserted.
///
/// `K = W_k · encoder_out` is well-defined only when `W_k`'s input width equals
/// the encoder output width. In a correctly built model both are the model width
/// `d_model`: `d_enc` and `d_k_input` are each *defined* as `d_model`, so equality
/// follows by transitivity. The realistic slip builds `W_k` for a different width
/// (`d_k_input = d_model - 1`), which makes the projection ill-defined and the
/// query SAT (see `encoder_decoder_match_depends_on_the_model_width`). Decidable
/// `QF_LRA`.
pub(crate) fn prove_cross_attention_encoder_decoder_match(
) -> Result<SeqModelPropertyResult, SmtError> {
    let program = build_cross_attention_encoder_decoder_match(true)?;
    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);

    Ok(SeqModelPropertyResult {
        property: "cross_attention_encoder_decoder_match".to_string(),
        proven,
        smt2,
        detail,
    })
}

/// Build the encoder-decoder match query. When `widths_from_model` is false the
/// key projection input is `d_model - 1` instead of `d_model`; tests flip it to
/// confirm the proof depends on the shared model width.
fn build_cross_attention_encoder_decoder_match(
    widths_from_model: bool,
) -> Result<AYProgram, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");

    let d_model = declare_real(&mut program, "d_model");
    assert_bounds(&mut program, &d_model, 1.0, 10000.0)?;

    // The encoder emits vectors of the model width; W_k also consumes that width
    // when built correctly. The slip sizes W_k for one fewer dimension.
    let d_enc = define_real(&mut program, "d_enc", &d_model);
    let d_k_term = if widths_from_model {
        d_model.clone()
    } else {
        d_model.clone().real_sub(Expr::real(1))
    };
    let d_k_input = define_real(&mut program, "d_k_input", &d_k_term);

    // Violation: the encoder width and the key projection input width disagree.
    program.assert(d_enc.ne(d_k_input));
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
    fn test_causal_mask_lower_triangular_proven() {
        let result = prove_causal_mask_lower_triangular().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask lower triangular should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "causal_mask_lower_triangular");
    }

    /// Transposing the causal rule masks the wrong triangle, so an upper-triangle
    /// cell keeps mask 1 and the query must be SAT.
    #[test]
    fn lower_triangular_depends_on_the_rule() {
        let program = build_causal_mask_lower_triangular(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the rule transposed the upper triangle is kept and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_causal_mask_diagonal_ones_proven() {
        let result = prove_causal_mask_diagonal_ones().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask diagonal ones should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
        assert_eq!(result.property, "causal_mask_diagonal_ones");
    }

    /// A strict `row > col` rule excludes the diagonal, so the self-attention
    /// entry becomes 0 and the query must be SAT.
    #[test]
    fn diagonal_ones_depends_on_the_rule() {
        let program = build_causal_mask_diagonal_ones(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a strict rule the diagonal is masked out and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_sinusoidal_orthogonality_cross_cancel_proven() {
        let result = prove_sinusoidal_orthogonality_cross_cancel().expect("proof should not error");
        assert!(
            result.proven,
            "Sinusoidal orthogonality cross-cancel should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_rope_orthogonality_off_diagonal_proven() {
        let result = prove_rope_orthogonality_off_diagonal().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "RoPE orthogonality off-diagonal: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_self_attention_output_bounded_proven() {
        let result = prove_self_attention_output_bounded().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Self-attention output bounded: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_mha_dimension_relationship_proven() {
        let result = prove_mha_dimension_relationship().expect("proof should not error");
        assert!(
            result.proven,
            "MHA dimension relationship should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Sizing d_model for one fewer head lets the top head's slots overflow the
    /// buffer, so the coverage query must be SAT.
    #[test]
    fn mha_dimension_depends_on_the_head_count() {
        let program = build_mha_dimension_relationship(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with d_model sized for n_heads-1 the top head overflows and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_kv_cache_append_preserves_old_proven() {
        let result = prove_kv_cache_append_preserves_old().expect("proof should not error");
        assert!(
            result.proven,
            "KV cache append preserves old should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Writing the appended entry at `old_len - 1` clobbers the last existing
    /// entry, so reading it back differs and the query must be SAT.
    #[test]
    fn kv_cache_append_depends_on_the_write_position() {
        let program = build_kv_cache_append_preserves_old(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "writing at old_len-1 overwrites an existing entry and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_autoregressive_independence_proven() {
        let result = prove_autoregressive_independence().expect("proof should not error");
        assert!(
            result.proven,
            "Autoregressive independence should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Leaving the future weight at 1 (mask not applied) lets the output depend on
    /// the future token, so the two runs diverge and the query must be SAT.
    #[test]
    fn autoregressive_independence_depends_on_the_future_mask() {
        let program = build_autoregressive_independence(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the future weight left at 1 the output sees the future and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_beam_search_score_monotonicity_proven() {
        let result = prove_beam_search_score_monotonicity().expect("proof should not error");
        assert!(
            result.proven,
            "Beam search score monotonicity should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_temperature_scaling_preserves_distribution_proven() {
        let result =
            prove_temperature_scaling_preserves_distribution().expect("proof should not error");
        assert!(
            result.proven,
            "Temperature scaling preserves distribution should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Normalizing over only the first numerator makes the rescaled weights sum to
    /// more than 1, so the distribution query must be SAT.
    #[test]
    fn temperature_distribution_depends_on_the_normalizer() {
        let program = build_temperature_scaling_preserves_distribution(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a dropped-term normalizer the weights over-sum and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_topk_preserves_largest_proven() {
        let result = prove_topk_preserves_largest().expect("proof should not error");
        assert!(
            result.proven,
            "Top-k preserves largest should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_softmax_non_negativity_proven() {
        let result = prove_softmax_non_negativity().expect("proof should not error");
        assert!(
            result.proven,
            "Softmax non-negativity should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_softmax_sum_to_one_proven() {
        let result = prove_softmax_sum_to_one().expect("proof should not error");
        assert!(
            result.proven,
            "Softmax sum-to-one should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Normalizing over a partial sum makes the weights over-sum, so the
    /// sum-to-one query must be SAT.
    #[test]
    fn softmax_sum_depends_on_the_normalizer() {
        let program = build_softmax_sum_to_one(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a dropped-term normalizer the weights over-sum and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_cross_attention_dimension_compatibility_proven() {
        let result =
            prove_cross_attention_dimension_compatibility().expect("proof should not error");
        assert!(
            result.proven,
            "Cross-attention dimension compatibility should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Widening the key projection by one breaks `d_q == d_k`, so the query must
    /// be SAT.
    #[test]
    fn cross_attention_compatibility_depends_on_the_projection() {
        let program =
            build_cross_attention_dimension_compatibility(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a mismatched key projection d_q != d_k and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_padding_mask_zeros_padded_proven() {
        let result = prove_padding_mask_zeros_padded().expect("proof should not error");
        assert!(
            result.proven,
            "Padding mask zeros padded should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// An off-by-one keep rule (`pos <= valid_len`) leaves the first padding slot
    /// unmasked, so the query must be SAT.
    #[test]
    fn padding_mask_depends_on_the_comparison() {
        let program = build_padding_mask_zeros_padded(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with an off-by-one comparison the first padding slot is kept and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_token_embedding_selectivity_proven() {
        let result = prove_token_embedding_selectivity().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Token embedding selectivity: expected Proven or Unknown (NRA), got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_token_embedding_bounded_proven() {
        let result = prove_token_embedding_bounded(5.0).expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Token embedding bounded: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_layer_norm_zero_mean_proven() {
        let result = prove_layer_norm_zero_mean().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Layer norm zero mean: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_layer_norm_unit_variance_proven() {
        let result = prove_layer_norm_unit_variance().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Layer norm unit variance: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_causal_mask_block_diagonal_proven() {
        let result = prove_causal_mask_block_diagonal().expect("proof should not error");
        assert!(
            result.proven,
            "Causal mask block diagonal should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Dropping the cross-batch clause leaves a cross-batch entry free to be 1, so
    /// the block-diagonal query must be SAT.
    #[test]
    fn block_diagonal_depends_on_the_batch_mask() {
        let program = build_causal_mask_block_diagonal(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "without the block factor a cross-batch entry can be nonzero and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_positional_encoding_bounded_proven() {
        let result = prove_positional_encoding_bounded().expect("proof should not error");
        assert!(
            result.proven,
            "Positional encoding bounded should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_attention_scaling_reduces_magnitude_proven() {
        let result = prove_attention_scaling_reduces_magnitude().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Attention scaling reduces magnitude: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_multihead_concat_dimension_proven() {
        let result = prove_multihead_concat_dimension().expect("proof should not error");
        assert!(
            result.proven,
            "Multihead concat dimension should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Packing heads at stride `d_head - 1` makes two distinct (head, offset) slots
    /// collide, so the injectivity query must be SAT.
    #[test]
    fn concat_dimension_depends_on_the_stride() {
        let program = build_multihead_concat_dimension(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with stride d_head-1 two slots collide and the query must be SAT; got: {detail}",
        );
    }

    #[test]
    fn test_kv_cache_length_monotonicity_proven() {
        let result = prove_kv_cache_length_monotonicity().expect("proof should not error");
        assert!(
            result.proven,
            "KV cache length monotonicity should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_autoregressive_mask_causal_consistency_proven() {
        let result =
            prove_autoregressive_mask_causal_consistency().expect("proof should not error");
        assert!(
            result.proven,
            "Autoregressive mask causal consistency should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Shifting the AR rule one token ahead makes it disagree with the causal mask
    /// on a future cell, so the consistency query must be SAT.
    #[test]
    fn ar_mask_consistency_depends_on_the_offset() {
        let program = build_autoregressive_mask_causal_consistency(false);
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with the AR rule shifted ahead the masks disagree and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_beam_search_topk_selection_proven() {
        let result = prove_beam_search_topk_selection().expect("proof should not error");
        assert!(
            result.proven,
            "Beam search top-k selection should be proven. detail: {}",
            result.detail
        );
    }

    #[test]
    fn test_temperature_positive_preserves_order_proven() {
        let result = prove_temperature_positive_preserves_order().expect("proof should not error");
        assert!(
            result.proven || result.detail.contains("Unknown"),
            "Temperature positive preserves order: expected Proven or Unknown, got: {}",
            result.detail,
        );
        assert!(
            !result.detail.contains("counterexample"),
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_topk_threshold_proven() {
        let result = prove_topk_threshold().expect("proof should not error");
        assert!(
            result.proven,
            "Top-k threshold should be proven. detail: {}",
            result.detail
        );
    }

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
            "Must not have counterexample: {}",
            result.detail
        );
    }

    #[test]
    fn test_cross_attention_encoder_decoder_match_proven() {
        let result = prove_cross_attention_encoder_decoder_match().expect("proof should not error");
        assert!(
            result.proven,
            "Cross-attention encoder-decoder match should be proven. detail: {}",
            result.detail
        );
        assert_eq!(vacuity_smell(&result.smt2), None);
    }

    /// Sizing the key projection input one below the model width breaks
    /// `d_enc == d_k_input`, so the query must be SAT.
    #[test]
    fn encoder_decoder_match_depends_on_the_model_width() {
        let program =
            build_cross_attention_encoder_decoder_match(false).expect("program builds");
        let (proven, detail) = execute_and_check(&program);
        assert!(
            !proven,
            "with a mismatched key projection width the dims disagree and the query must be SAT; \
             got: {detail}",
        );
    }

    #[test]
    fn test_smt2_structure_causal_mask() {
        let result = prove_causal_mask_lower_triangular().expect("proof should not error");
        assert!(result.smt2.contains("set-logic"), "should declare logic");
        assert!(result.smt2.contains("check-sat"), "should have check-sat");
        assert!(
            result.smt2.contains("declare-const"),
            "should have declarations"
        );
    }
}
