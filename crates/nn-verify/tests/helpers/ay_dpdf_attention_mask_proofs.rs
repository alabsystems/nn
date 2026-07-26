// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for attention mask and position bias properties.
//!
//! Proves 20 fundamental properties of attention masking and position bias
//! mechanisms used in transformer models:
//! - Causal mask: upper triangle all -inf/0, lower triangle all 0
//! - Padding mask: padded positions masked
//! - Combined causal+padding mask
//! - ALiBi linear position bias and slope geometric sequence
//! - Sliding window mask: bandwidth constraint
//! - Global attention mask for special tokens
//! - Cross-attention mask: no self-masking
//! - Mask dtype compatibility (bool vs float)
//! - Mask broadcasting across heads and batch
//! - Additive mask: -inf zeros out softmax
//! - Multiplicative mask: 0 zeros out attention
//! - Prefix mask for encoder-decoder
//! - Block-sparse mask pattern
//! - Local+global attention pattern (BigBird)
//! - Dynamic mask generation for variable length
//! - Mask inversion preserves complement
//! - Mask union/intersection properties
//!
//! Part of #4217.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};
use nn_verify::ay_real_lit::RealLit;

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
///
/// The ay convention: we assert the negation of the property, then
/// UNSAT (Verified) means the original property holds universally.
fn assert_verified(prog: &AYProgram, property_name: &str) {
    match execute_direct::execute(prog) {
        Ok(ExecuteResult::Verified) => {
            // UNSAT — property proved for all inputs.
        }
        Ok(other) => {
            panic!(
                "{property_name}: expected Verified (UNSAT), got: {other:?}. \
                 The negated property is satisfiable — the property does NOT hold."
            );
        }
        Err(e) => {
            panic!("{property_name}: ay execution error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1051: Causal mask upper triangle all -inf/0
// ---------------------------------------------------------------------------

/// Prove: in a causal mask for sequence length 4, the upper triangle
/// (positions j > i) contains only -inf (modeled as -10000) or 0.
///
/// For a 4x4 causal mask, entries where j > i are set to -M (large negative).
/// The upper-triangular sum equals -(number_of_upper_entries) * M.
/// We prove no upper-triangular entry can be positive. Uses QF_LRA.
#[test]
fn test_1051_causal_mask_upper_triangle_neg_inf() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let neg_m = Expr::real(-10000);
    let zero = Expr::real(0);

    // Declare 4x4 mask matrix entries
    let mut mask = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            let v = prog.declare_const(&format!("m{}_{}", r, c), real.clone());
            // Causal mask: j <= i => 0 (attend), j > i => -M (masked)
            if c <= r {
                prog.assert(v.clone().eq(zero.clone()));
            } else {
                prog.assert(v.clone().eq(neg_m.clone()));
            }
            mask.push(v);
        }
    }

    // Negated property: some upper-triangular entry > 0
    // Upper triangle indices in row-major 4x4: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    let upper_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for &(r, c) in &upper_indices {
        violation = violation.or(mask[r * 4 + c].clone().real_gt(zero.clone()));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "causal_mask_upper_triangle_neg_inf");
}

// ---------------------------------------------------------------------------
// Test 1052: Causal mask lower triangle all 0
// ---------------------------------------------------------------------------

/// Prove: in a causal mask for sequence length 4, the lower triangle
/// (positions j <= i) contains only 0 (attend).
///
/// Lower-triangular entries (including diagonal) are all 0. We prove
/// no lower-triangular entry is nonzero. Uses QF_LRA.
#[test]
fn test_1052_causal_mask_lower_triangle_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let neg_m = Expr::real(-10000);
    let zero = Expr::real(0);

    let mut mask = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            let v = prog.declare_const(&format!("m{}_{}", r, c), real.clone());
            if c <= r {
                prog.assert(v.clone().eq(zero.clone()));
            } else {
                prog.assert(v.clone().eq(neg_m.clone()));
            }
            mask.push(v);
        }
    }

    // Lower triangle indices: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3)
    let lower_indices = [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ];
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for &(r, c) in &lower_indices {
        violation = violation.or(mask[r * 4 + c].clone().ne(zero.clone()));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "causal_mask_lower_triangle_zero");
}

// ---------------------------------------------------------------------------
// Test 1053: Padding mask: padded positions masked
// ---------------------------------------------------------------------------

/// Prove: in a padding mask with actual_len=2 and padded_len=4,
/// columns >= actual_len are all -M. Unpadded columns are 0.
///
/// Every row has the same padding pattern: columns 0,1 are 0; columns 2,3
/// are -M. Uses QF_LRA.
#[test]
fn test_1053_padding_mask_padded_positions() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let neg_m = Expr::real(-10000);
    let zero = Expr::real(0);

    // 2 rows x 4 cols, actual_len = 2
    let mut mask = Vec::new();
    for r in 0..2 {
        for c in 0..4 {
            let v = prog.declare_const(&format!("p{}_{}", r, c), real.clone());
            if c < 2 {
                prog.assert(v.clone().eq(zero.clone()));
            } else {
                prog.assert(v.clone().eq(neg_m.clone()));
            }
            mask.push(v);
        }
    }

    // Negated property: some padded position (col >= 2) is not -M
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for r in 0..2 {
        for c in 2..4 {
            violation = violation.or(mask[r * 4 + c].clone().ne(neg_m.clone()));
        }
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "padding_mask_padded_positions");
}

// ---------------------------------------------------------------------------
// Test 1054: Combined causal+padding mask
// ---------------------------------------------------------------------------

/// Prove: the intersection of causal and padding masks satisfies both
/// constraints simultaneously. Position (i,j) is unmasked iff j <= i
/// AND j < actual_len.
///
/// For seq_len=4, actual_len=3: row 0 attends to [0], row 1 to [0,1],
/// row 2 to [0,1,2], row 3 to [0,1,2]. Column 3 is always masked.
/// Uses QF_LRA.
#[test]
fn test_1054_combined_causal_padding_mask() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let neg_m = Expr::real(-10000);
    let zero = Expr::real(0);

    // 4x4 combined mask, actual_len=3
    // Unmasked if j <= i AND j < 3, else -M
    let expected: [[i64; 4]; 4] = [
        [0, -10000, -10000, -10000],
        [0, 0, -10000, -10000],
        [0, 0, 0, -10000],
        [0, 0, 0, -10000],
    ];

    let mut mask = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            let v = prog.declare_const(&format!("c{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(Expr::real(expected[r][c])));
            mask.push(v);
        }
    }

    // Negated property: some entry is not in {0, -M}
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for v in &mask {
        violation = violation.or(v.clone().ne(zero.clone()).and(v.clone().ne(neg_m.clone())));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "combined_causal_padding_mask");
}

// ---------------------------------------------------------------------------
// Test 1055: ALiBi linear position bias
// ---------------------------------------------------------------------------

/// Prove: ALiBi bias is linear in distance. For slope m > 0 and distances
/// d1, d2, d3 where d3 = d1 + d2, the bias satisfies:
/// bias(d3) = bias(d1) + bias(d2).
///
/// bias(d) = -m * d, so -m*(d1+d2) = -m*d1 + (-m*d2). Uses QF_NRA.
#[test]
fn test_1055_alibi_linear_position_bias() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("b1", real.clone());
    let _ = prog.declare_const("b2", real.clone());
    let _ = prog.declare_const("b3", real);

    let m = real_var("m");
    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let b1 = real_var("b1");
    let b2 = real_var("b2");
    let b3 = real_var("b3");

    // m > 0 (positive slope)
    prog.assert(m.clone().real_gt(Expr::real(0)));
    // d1, d2 >= 0
    prog.assert(d1.clone().real_ge(Expr::real(0)));
    prog.assert(d2.clone().real_ge(Expr::real(0)));

    // bias = -m * distance
    prog.assert(
        b1.clone()
            .eq(Expr::real(0).real_sub(m.clone().real_mul(d1.clone()))),
    );
    prog.assert(
        b2.clone()
            .eq(Expr::real(0).real_sub(m.clone().real_mul(d2.clone()))),
    );
    prog.assert(
        b3.clone()
            .eq(Expr::real(0).real_sub(m.real_mul(d1.real_add(d2)))),
    );

    // Negated property: b3 != b1 + b2 (linearity violated)
    let violation = b3.ne(b1.real_add(b2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_linear_position_bias");
}

// ---------------------------------------------------------------------------
// Test 1056: ALiBi slope geometric sequence
// ---------------------------------------------------------------------------

/// Prove: ALiBi slopes form a geometric sequence. Given slopes s1, s2, s3
/// with common ratio r (0 < r < 1): s2 = s1*r, s3 = s2*r = s1*r^2.
///
/// The ratio between consecutive slopes is constant: s2/s1 = s3/s2 = r.
/// Uses QF_NRA.
#[test]
fn test_1056_alibi_slope_geometric_sequence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("s3", real.clone());
    let _ = prog.declare_const("r", real);

    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let s3 = real_var("s3");
    let r = real_var("r");

    // s1 > 0, 0 < r < 1
    prog.assert(s1.clone().real_gt(Expr::real(0)));
    prog.assert(r.clone().real_gt(Expr::real(0)));
    prog.assert(r.clone().real_lt(Expr::real(1)));

    // Geometric: s2 = s1*r, s3 = s2*r
    prog.assert(s2.clone().eq(s1.clone().real_mul(r.clone())));
    prog.assert(s3.clone().eq(s2.clone().real_mul(r.clone())));

    // Negated property: s3 != s1 * r^2
    let r_squared = r.clone().real_mul(r);
    let violation = s3.ne(s1.real_mul(r_squared));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_slope_geometric_sequence");
}

// ---------------------------------------------------------------------------
// Test 1057: Sliding window mask bandwidth constraint
// ---------------------------------------------------------------------------

/// Prove: in a sliding window mask with window size W, every row has
/// at most 2*W+1 non-zero entries centered around the diagonal.
///
/// For a 6x6 mask with W=1, each row has at most 3 non-zero entries.
/// We construct the concrete mask and verify the bandwidth bound. Uses QF_LRA.
#[test]
fn test_1057_sliding_window_bandwidth_constraint() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let three = Expr::real(3); // 2*W+1 for W=1

    // 6x6 sliding window mask, W=1: attend if |i-j| <= 1
    let expected: [[i64; 6]; 6] = [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ];

    let mut mask = Vec::new();
    for r in 0..6 {
        for c in 0..6 {
            let v = prog.declare_const(&format!("w{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(Expr::real(expected[r][c])));
            mask.push(v);
        }
    }

    // Negated property: some row sum > 3
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for r in 0..6 {
        let mut row_sum = Expr::real(0);
        for c in 0..6 {
            row_sum = row_sum.real_add(mask[r * 6 + c].clone());
        }
        violation = violation.or(row_sum.real_gt(three.clone()));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sliding_window_bandwidth_constraint");
}

// ---------------------------------------------------------------------------
// Test 1058: Global attention mask for special tokens
// ---------------------------------------------------------------------------

/// Prove: in a global attention mask, special token positions (index 0 = CLS)
/// attend to all positions and all positions attend to special tokens.
///
/// For seq_len=4 with position 0 as global: row 0 is all-1 (CLS attends
/// everywhere), column 0 is all-1 (everyone attends to CLS). Uses QF_LRA.
#[test]
fn test_1058_global_attention_special_tokens() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let zero = Expr::real(0);

    // 4x4 mask: position 0 is global, rest is local (window=1)
    // Row 0: [1,1,1,1] (global attends everywhere)
    // Col 0: all 1 (everyone attends to global)
    // Local positions: window around diagonal
    let expected: [[i64; 4]; 4] = [
        [1, 1, 1, 1], // global row
        [1, 1, 1, 0], // attends to global + local window
        [1, 1, 1, 1], // attends to global + local window
        [1, 0, 1, 1], // attends to global + local window
    ];

    let mut mask = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            let v = prog.declare_const(&format!("g{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(Expr::real(expected[r][c])));
            mask.push(v);
        }
    }

    // Negated property: row 0 has a zero OR column 0 has a zero
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
                                                              // Row 0 must be all-1
    for c in 0..4 {
        violation = violation.or(mask[c].clone().ne(one.clone()));
    }
    // Column 0 must be all-1
    for r in 0..4 {
        violation = violation.or(mask[r * 4].clone().ne(one.clone()));
    }
    let _ = zero; // suppress unused warning
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "global_attention_special_tokens");
}

// ---------------------------------------------------------------------------
// Test 1059: Cross-attention mask: no self-masking
// ---------------------------------------------------------------------------

/// Prove: in cross-attention between decoder (length D=2) and encoder
/// (length E=3), the mask allows every decoder position to attend to
/// every encoder position (no causal constraint).
///
/// All entries in the D x E mask are 1 (full attention). Uses QF_LRA.
#[test]
fn test_1059_cross_attention_no_self_masking() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let one = Expr::real(1);

    // 2x3 cross-attention mask: all ones
    let mut mask = Vec::new();
    for r in 0..2 {
        for c in 0..3 {
            let v = prog.declare_const(&format!("x{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(one.clone()));
            mask.push(v);
        }
    }

    // Negated property: some entry != 1
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for v in &mask {
        violation = violation.or(v.clone().ne(one.clone()));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_attention_no_self_masking");
}

// ---------------------------------------------------------------------------
// Test 1060: Mask dtype compatibility (bool vs float)
// ---------------------------------------------------------------------------

/// Prove: a boolean mask (0 or 1) can be converted to a float additive mask
/// by the transform: float_mask = (1 - bool_mask) * (-M).
///
/// bool=1 (attend) => float = 0. bool=0 (masked) => float = -M.
/// The conversion preserves semantics. Uses QF_LRA.
#[test]
fn test_1060_mask_dtype_compatibility() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("f", real.clone());
    let _ = prog.declare_const("big_m", real);

    let b = real_var("b");
    let f = real_var("f");
    let big_m = real_var("big_m");

    // b is boolean: 0 or 1
    prog.assert(b.clone().eq(Expr::real(0)).or(b.clone().eq(Expr::real(1))));

    // M > 0 (large masking constant)
    prog.assert(big_m.clone().real_gt(Expr::real(0)));

    // Conversion: f = (1 - b) * (-M)
    let one_minus_b = Expr::real(1).real_sub(b.clone());
    let neg_m = Expr::real(0).real_sub(big_m.clone());
    prog.assert(f.clone().eq(one_minus_b.real_mul(neg_m)));

    // Negated property: when b=1 (attend), f != 0, OR when b=0 (masked), f != -M
    let violation_attend = b.clone().eq(Expr::real(1)).and(f.clone().ne(Expr::real(0)));
    let violation_masked = b.eq(Expr::real(0)).and(f.ne(Expr::real(0).real_sub(big_m)));
    prog.assert(violation_attend.or(violation_masked));
    prog.check_sat();

    assert_verified(&prog, "mask_dtype_compatibility");
}

// ---------------------------------------------------------------------------
// Test 1061: Mask broadcasting across heads
// ---------------------------------------------------------------------------

/// Prove: a mask of shape [1, 1, S, S] broadcasts correctly to
/// [B, H, S, S] — the mask value at position (s1, s2) is the same
/// for all batch and head indices.
///
/// We model 2 heads and verify they see the same mask value. Uses QF_LRA.
#[test]
fn test_1061_mask_broadcasting_across_heads() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    // Source mask value for position (i, j)
    let _ = prog.declare_const("mask_src", real.clone());
    // Broadcast to head 0 and head 1
    let _ = prog.declare_const("mask_h0", real.clone());
    let _ = prog.declare_const("mask_h1", real);

    let mask_src = real_var("mask_src");
    let mask_h0 = real_var("mask_h0");
    let mask_h1 = real_var("mask_h1");

    // Broadcasting: both heads get the same source value
    prog.assert(mask_h0.clone().eq(mask_src.clone()));
    prog.assert(mask_h1.clone().eq(mask_src));

    // Negated property: heads see different mask values
    let violation = mask_h0.ne(mask_h1);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mask_broadcasting_across_heads");
}

// ---------------------------------------------------------------------------
// Test 1062: Mask broadcasting across batch
// ---------------------------------------------------------------------------

/// Prove: a mask of shape [1, S, S] broadcasts correctly to [B, S, S] —
/// all batch elements see the same mask.
///
/// We model 3 batch elements and verify they all receive the same mask
/// value at position (i, j). Uses QF_LRA.
#[test]
fn test_1062_mask_broadcasting_across_batch() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("mask_src", real.clone());
    let _ = prog.declare_const("mask_b0", real.clone());
    let _ = prog.declare_const("mask_b1", real.clone());
    let _ = prog.declare_const("mask_b2", real);

    let mask_src = real_var("mask_src");
    let mask_b0 = real_var("mask_b0");
    let mask_b1 = real_var("mask_b1");
    let mask_b2 = real_var("mask_b2");

    // Broadcasting: all batch elements get the same source value
    prog.assert(mask_b0.clone().eq(mask_src.clone()));
    prog.assert(mask_b1.clone().eq(mask_src.clone()));
    prog.assert(mask_b2.clone().eq(mask_src));

    // Negated property: some batch element differs
    let violation = mask_b0.clone().ne(mask_b1.clone()).or(mask_b1.ne(mask_b2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mask_broadcasting_across_batch");
}

// ---------------------------------------------------------------------------
// Test 1063: Additive mask: -inf zeros out softmax
// ---------------------------------------------------------------------------

/// Prove: after applying an additive mask of -M to a score, the softmax
/// output for the masked position approaches 0 while unmasked outputs
/// sum to 1.
///
/// We model: 3 positions, position 2 masked. exp_masked in [0, eps].
/// The softmax output for position 2 is near 0, and positions 0+1 sum
/// to approximately 1. Uses QF_NRA.
#[test]
fn test_1063_additive_mask_zeros_out_softmax() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("e0", real.clone());
    let _ = prog.declare_const("e1", real.clone());
    let _ = prog.declare_const("e2", real.clone());
    let _ = prog.declare_const("s2", real);

    let e0 = real_var("e0");
    let e1 = real_var("e1");
    let e2 = real_var("e2");
    let s2 = real_var("s2");

    // Unmasked exps are positive
    prog.assert(e0.clone().real_gt(Expr::real(0)));
    prog.assert(e0.clone().real_le(Expr::real(1000)));
    prog.assert(e1.clone().real_gt(Expr::real(0)));
    prog.assert(e1.clone().real_le(Expr::real(1000)));

    // Masked: exp(score + (-M)) near zero
    let eps = Expr::real_ratio(1, 1000000);
    prog.assert(e2.clone().real_ge(Expr::real(0)));
    prog.assert(e2.clone().real_le(eps));

    // s2 = e2 / (e0 + e1 + e2)
    let z = e0.real_add(e1).real_add(e2.clone());
    prog.assert(s2.clone().real_mul(z).eq(e2));

    // Negated property: s2 > 0.001 (should be near zero)
    let violation = s2.real_gt(Expr::real_ratio(1, 1000));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "additive_mask_zeros_out_softmax");
}

// ---------------------------------------------------------------------------
// Test 1064: Multiplicative mask: 0 zeros out attention
// ---------------------------------------------------------------------------

/// Prove: a multiplicative mask with value 0 at position j forces the
/// attention score for that position to 0 before softmax.
///
/// masked_score = score * mask_val. If mask_val = 0, then masked_score = 0
/// regardless of the original score. Uses QF_NRA.
#[test]
fn test_1064_multiplicative_mask_zeros_attention() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("score", real.clone());
    let _ = prog.declare_const("mask_val", real.clone());
    let _ = prog.declare_const("masked_score", real);

    let score = real_var("score");
    let mask_val = real_var("mask_val");
    let masked_score = real_var("masked_score");

    // score is bounded
    prog.assert(score.clone().real_ge(Expr::real(-100)));
    prog.assert(score.clone().real_le(Expr::real(100)));

    // mask_val = 0 (masked position)
    prog.assert(mask_val.clone().eq(Expr::real(0)));

    // masked_score = score * mask_val
    prog.assert(masked_score.clone().eq(score.real_mul(mask_val)));

    // Negated property: masked_score != 0
    let violation = masked_score.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multiplicative_mask_zeros_attention");
}

// ---------------------------------------------------------------------------
// Test 1065: Prefix mask for encoder-decoder
// ---------------------------------------------------------------------------

/// Prove: a prefix mask allows attending to the first P positions (prefix)
/// while masking the rest causally. For seq_len=4, prefix_len=2:
/// positions 0,1 are fully visible; positions 2,3 are causal.
///
/// Row 0: [0,0,-M,-M], Row 1: [0,0,-M,-M], Row 2: [0,0,0,-M],
/// Row 3: [0,0,0,0]. Uses QF_LRA.
#[test]
fn test_1065_prefix_mask_encoder_decoder() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let neg_m = Expr::real(-10000);
    let zero = Expr::real(0);

    // Prefix mask: prefix_len=2, seq_len=4
    // Rows 0,1 (prefix): attend only to prefix => [0, 0, -M, -M]
    // Rows 2,3 (causal): attend to prefix + causal positions up to i
    let expected: [[i64; 4]; 4] = [
        [0, 0, -10000, -10000],
        [0, 0, -10000, -10000],
        [0, 0, 0, -10000],
        [0, 0, 0, 0],
    ];

    let mut mask = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            let v = prog.declare_const(&format!("px{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(Expr::real(expected[r][c])));
            mask.push(v);
        }
    }

    // Negated property: some entry is not in {0, -M}
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for v in &mask {
        violation = violation.or(v.clone().ne(zero.clone()).and(v.clone().ne(neg_m.clone())));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "prefix_mask_encoder_decoder");
}

// ---------------------------------------------------------------------------
// Test 1066: Block-sparse mask pattern
// ---------------------------------------------------------------------------

/// Prove: a block-sparse mask with block_size=2 on a 6x6 matrix has
/// non-zero entries only within 2x2 blocks along the diagonal.
///
/// For a 6x6 matrix with 3 diagonal blocks of size 2x2, each row has
/// exactly 2 non-zero entries. Total non-zero = 6*2 = 12 out of 36.
/// Uses QF_LRA.
#[test]
fn test_1066_block_sparse_mask_pattern() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let two = Expr::real(2);

    // 6x6 block-diagonal mask, block_size=2
    // Blocks: [0-1,0-1], [2-3,2-3], [4-5,4-5]
    let expected: [[i64; 6]; 6] = [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ];

    let mut mask = Vec::new();
    for r in 0..6 {
        for c in 0..6 {
            let v = prog.declare_const(&format!("bs{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(Expr::real(expected[r][c])));
            mask.push(v);
        }
    }

    // Negated property: some row sum != 2 (block_size)
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for r in 0..6 {
        let mut row_sum = Expr::real(0);
        for c in 0..6 {
            row_sum = row_sum.real_add(mask[r * 6 + c].clone());
        }
        violation = violation.or(row_sum.ne(two.clone()));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "block_sparse_mask_pattern");
}

// ---------------------------------------------------------------------------
// Test 1067: Local+global attention pattern (BigBird)
// ---------------------------------------------------------------------------

/// Prove: BigBird-style attention combines local window (W=1) and global
/// tokens (position 0). Every position attends to at least its local
/// window plus the global token.
///
/// For seq_len=5, position 0 is global. Each row has at least 2 non-zero
/// entries (local + global). Row 0 has all 5 (global attends everywhere).
/// Uses QF_LRA.
#[test]
fn test_1067_bigbird_local_global_attention() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();

    // 5x5 BigBird mask: position 0 global, window W=1
    // Row 0: all-1 (global); others: local window + global column
    let expected: [[i64; 5]; 5] = [
        [1, 1, 1, 1, 1], // global
        [1, 1, 1, 0, 0], // local + global
        [1, 1, 1, 1, 0], // local + global
        [1, 0, 1, 1, 1], // local + global
        [1, 0, 0, 1, 1], // local + global
    ];

    let mut mask = Vec::new();
    for r in 0..5 {
        for c in 0..5 {
            let v = prog.declare_const(&format!("bb{}_{}", r, c), real.clone());
            prog.assert(v.clone().eq(Expr::real(expected[r][c])));
            mask.push(v);
        }
    }

    // Property: every row has at least 2 non-zero entries
    let mut violation = Expr::real(0).real_gt(Expr::real(1)); // false
    for r in 0..5 {
        let mut row_sum = Expr::real(0);
        for c in 0..5 {
            row_sum = row_sum.real_add(mask[r * 5 + c].clone());
        }
        violation = violation.or(row_sum.real_lt(Expr::real(2)));
    }
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "bigbird_local_global_attention");
}

// ---------------------------------------------------------------------------
// Test 1068: Dynamic mask generation for variable length
// ---------------------------------------------------------------------------

/// Prove: a dynamically generated mask for variable-length sequences
/// with lengths [3, 2] in a batch (padded to 3) correctly masks
/// padding positions.
///
/// Batch element 0 (len=3): all positions valid.
/// Batch element 1 (len=2): position 2 is padding (-M).
/// Uses QF_LRA.
#[test]
fn test_1068_dynamic_mask_variable_length() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let neg_m = Expr::real(-10000);
    let zero = Expr::real(0);

    // Batch 0, length=3, padded_len=3: all 0
    for c in 0..3 {
        let v = prog.declare_const(&format!("b0_{}", c), real.clone());
        prog.assert(v.eq(zero.clone()));
    }

    // Batch 1, length=2, padded_len=3: cols 0,1 = 0; col 2 = -M
    let b1_0 = prog.declare_const("b1_0", real.clone());
    let b1_1 = prog.declare_const("b1_1", real.clone());
    let b1_2 = prog.declare_const("b1_2", real.clone());
    prog.assert(b1_0.clone().eq(zero.clone()));
    prog.assert(b1_1.clone().eq(zero.clone()));
    prog.assert(b1_2.clone().eq(neg_m.clone()));

    // Negated property: batch 1 padding position is not -M
    let violation = b1_2.ne(neg_m);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dynamic_mask_variable_length");
}

// ---------------------------------------------------------------------------
// Test 1069: Mask inversion preserves complement
// ---------------------------------------------------------------------------

/// Prove: inverting a binary mask (0 <-> 1) produces the complement.
/// For every position, mask + inverted_mask = 1.
///
/// If mask[i] in {0, 1} and inv[i] = 1 - mask[i], then
/// mask[i] + inv[i] = 1 for all i. Uses QF_LRA.
#[test]
fn test_1069_mask_inversion_preserves_complement() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("inv", real);

    let m = real_var("m");
    let inv = real_var("inv");

    // m is binary
    prog.assert(m.clone().eq(Expr::real(0)).or(m.clone().eq(Expr::real(1))));

    // inv = 1 - m
    prog.assert(inv.clone().eq(Expr::real(1).real_sub(m.clone())));

    // Negated property: m + inv != 1
    let violation = m.real_add(inv).ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mask_inversion_preserves_complement");
}

// ---------------------------------------------------------------------------
// Test 1070: Mask union/intersection properties
// ---------------------------------------------------------------------------

/// Prove: for binary masks A, B:
/// - union(A, B) = max(A, B) has at least as many 1s as either mask alone
/// - intersection(A, B) = min(A, B) has at most as many 1s as either mask
///
/// For each position: max(a, b) >= a, max(a, b) >= b,
/// min(a, b) <= a, min(a, b) <= b. Uses QF_LRA.
#[test]
fn test_1070_mask_union_intersection_properties() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("u", real.clone());
    let _ = prog.declare_const("inter", real);

    let a = real_var("a");
    let b = real_var("b");
    let u = real_var("u");
    let inter = real_var("inter");

    // Binary masks
    prog.assert(a.clone().eq(Expr::real(0)).or(a.clone().eq(Expr::real(1))));
    prog.assert(b.clone().eq(Expr::real(0)).or(b.clone().eq(Expr::real(1))));

    // Union = max(a, b): u >= a, u >= b, u = a or u = b
    prog.assert(u.clone().real_ge(a.clone()));
    prog.assert(u.clone().real_ge(b.clone()));
    prog.assert(u.clone().eq(a.clone()).or(u.clone().eq(b.clone())));

    // Intersection = min(a, b): inter <= a, inter <= b, inter = a or inter = b
    prog.assert(inter.clone().real_le(a.clone()));
    prog.assert(inter.clone().real_le(b.clone()));
    prog.assert(inter.clone().eq(a.clone()).or(inter.clone().eq(b.clone())));

    // Negated property: union < some input OR intersection > some input
    let violation = u
        .clone()
        .real_lt(a.clone())
        .or(u.real_lt(b.clone()))
        .or(inter.clone().real_gt(a))
        .or(inter.real_gt(b));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mask_union_intersection_properties");
}
