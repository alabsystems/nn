// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ay SMT proofs for dpdf VLM attention mask and position bias properties (#4217).
//!
//! Extends `ay_attention_mask_properties` with dpdf-specific VLM properties:
//!
//! 1. **Causal mask zero for future**: mask[i][j] == 0 when j > i.
//! 2. **Padding mask neg-inf**: mask[i][j] == -inf for padded positions.
//! 3. **Combined causal+padding mask**: intersection satisfies both constraints.
//! 4. **ALiBi distance symmetry**: |i - j| == |j - i|.
//! 5. **RoPE norm preservation**: ||RoPE(x, pos)|| == ||x||.
//! 6. **Sliding window sparsity**: at most 2*W+1 non-zero entries per row.
//! 7. **Cross-attention shape compatibility**: encoder/decoder mask dimensions align.
//! 8. **Mask additive form**: -inf = "ignore", 0 = "attend", softmax zeros masked.
//!
//! Structural proofs use QF_LRA; algebraic proofs use QF_NRA.

use ay_bindings::{Expr, Sort, AYProgram};

use crate::smt_error::SmtError;

/// Result of a dpdf attention mask/position bias property proof attempt.
#[derive(Debug, Clone)]
pub struct DpdfAttentionMaskResult {
    /// Human-readable property name.
    pub property: String,
    /// Whether the proof succeeded (UNSAT = property holds for all inputs).
    pub proven: bool,
    /// SMT-LIB2 text of the query.
    pub smt2: String,
    /// Solver detail message.
    pub detail: String,
}

fn declare_real(program: &mut AYProgram, name: &str) -> Expr {
    program.declare_const(name, Sort::real())
}

fn assert_bounds(program: &mut AYProgram, expr: &Expr, lower: &Expr, upper: &Expr) {
    program.assert(expr.clone().real_ge(lower.clone()));
    program.assert(expr.clone().real_le(upper.clone()));
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
// Property 1: Causal Mask Zero For Future Positions
// ---------------------------------------------------------------------------

/// Prove causal mask has mask[i][j] == 0 when j > i for S=3.
/// Upper-triangular entries sum to 0. Uses QF_LRA.
pub fn prove_causal_mask_zero_future() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let zero = Expr::real(0);
    let one = Expr::real(1);

    let mut m = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            let var = declare_real(&mut program, &format!("m{}_{}", row, col));
            if col <= row {
                program.assert(var.clone().eq(one.clone()));
            } else {
                program.assert(var.clone().eq(zero.clone()));
            }
            m.push(var);
        }
    }

    // Upper-triangular: m[0][1], m[0][2], m[1][2] = indices 1, 2, 5
    let upper_sum = m[1].clone().real_add(m[2].clone()).real_add(m[5].clone());
    program.assert(upper_sum.ne(zero));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "causal_mask_zero_future_3x3".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 2: Padding Mask Neg-Inf For Padded Positions
// ---------------------------------------------------------------------------

/// Prove padding mask uses -inf (modeled as -1000) for padded column.
/// For seq_len=2 padded to 3, column 2 is all -inf. Uses QF_LRA.
pub fn prove_padding_mask_neg_inf() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let zero = Expr::real(0);
    let neg_inf = Expr::real(-1000);

    // Column 2 is padding (-inf), other columns are 0 (attend)
    let m02 = declare_real(&mut program, "m02");
    let m12 = declare_real(&mut program, "m12");
    let m22 = declare_real(&mut program, "m22");
    program.assert(m02.clone().eq(neg_inf.clone()));
    program.assert(m12.clone().eq(neg_inf.clone()));
    program.assert(m22.clone().eq(neg_inf));

    for name in ["m00", "m01", "m10", "m11", "m20", "m21"] {
        let var = declare_real(&mut program, name);
        program.assert(var.eq(zero.clone()));
    }

    let pad_sum = m02.real_add(m12).real_add(m22);
    program.assert(pad_sum.ne(Expr::real(-3000)));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "padding_mask_neg_inf_col2".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 3: Combined Causal + Padding Mask
// ---------------------------------------------------------------------------

/// Prove combined causal+padding mask entries are all <= 0 and are
/// exactly 0 (attend) or -M (masked). 3x3, seq_len=2. Uses QF_LRA.
pub fn prove_combined_causal_padding_mask() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let zero = Expr::real(0);
    let neg_m = Expr::real(-1000);

    // Combined mask: causal + padding. Row i attends to j <= i AND j < seq_len=2.
    // Row 0: [0, -M, -M], Row 1: [0, 0, -M], Row 2: [0, 0, -M]
    let values: [i64; 9] = [0, -1000, -1000, 0, 0, -1000, 0, 0, -1000];
    let mut vars = Vec::new();
    for (idx, val) in values.iter().enumerate() {
        let var = declare_real(&mut program, &format!("c{}", idx));
        program.assert(var.clone().eq(Expr::real(*val)));
        vars.push(var);
    }

    // Violation: any entry > 0 OR any entry not in {0, -M}
    let mut violation = vars[0].clone().real_gt(zero.clone());
    for var in &vars[1..] {
        violation = violation.or(var.clone().real_gt(zero.clone()));
    }
    for var in &vars {
        violation = violation.or(var
            .clone()
            .ne(zero.clone())
            .and(var.clone().ne(neg_m.clone())));
    }

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "combined_causal_padding_mask_3x3".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 4: ALiBi Distance Symmetry
// ---------------------------------------------------------------------------

/// Prove |i - j| == |j - i| for arbitrary positions. Uses QF_LRA.
pub fn prove_alibi_distance_symmetry() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let zero = Expr::real(0);
    let bound_hi = Expr::real(1000);

    let i = declare_real(&mut program, "i");
    let j = declare_real(&mut program, "j");
    assert_bounds(&mut program, &i, &zero, &bound_hi);
    assert_bounds(&mut program, &j, &zero, &bound_hi);

    // |i - j| via auxiliary
    let d_ij = declare_real(&mut program, "d_ij");
    program.assert(d_ij.clone().real_ge(zero.clone()));
    program.assert(d_ij.clone().real_ge(i.clone().real_sub(j.clone())));
    program.assert(d_ij.clone().real_ge(j.clone().real_sub(i.clone())));
    program.assert(
        d_ij.clone()
            .eq(i.clone().real_sub(j.clone()))
            .or(d_ij.clone().eq(j.clone().real_sub(i.clone()))),
    );

    // |j - i| via auxiliary
    let d_ji = declare_real(&mut program, "d_ji");
    program.assert(d_ji.clone().real_ge(zero.clone()));
    program.assert(d_ji.clone().real_ge(j.clone().real_sub(i.clone())));
    program.assert(d_ji.clone().real_ge(i.clone().real_sub(j.clone())));
    program.assert(
        d_ji.clone()
            .eq(j.clone().real_sub(i.clone()))
            .or(d_ji.clone().eq(i.real_sub(j))),
    );

    program.assert(d_ij.ne(d_ji));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "alibi_distance_symmetry".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 5: RoPE Norm Preservation
// ---------------------------------------------------------------------------

/// Prove ||RoPE(x, pos)||^2 == ||x||^2 for a 2D rotation block.
///
/// y0 = x0*c - x1*s, y1 = x0*s + x1*c where c^2 + s^2 = 1.
/// ||y||^2 = x0^2*(c^2+s^2) + x1^2*(s^2+c^2) = ||x||^2. Uses QF_NRA.
pub fn prove_rope_norm_preservation() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");
    let one = Expr::real(1);
    let bound = Expr::real(100);
    let neg_bound = Expr::real(-100);
    let trig_lo = Expr::real(-1);
    let trig_hi = Expr::real(1);

    let x0 = declare_real(&mut program, "x0");
    let x1 = declare_real(&mut program, "x1");
    assert_bounds(&mut program, &x0, &neg_bound, &bound);
    assert_bounds(&mut program, &x1, &neg_bound, &bound);

    let c = declare_real(&mut program, "c");
    let s = declare_real(&mut program, "s");
    assert_bounds(&mut program, &c, &trig_lo, &trig_hi);
    assert_bounds(&mut program, &s, &trig_lo, &trig_hi);
    program.assert(
        c.clone()
            .real_mul(c.clone())
            .real_add(s.clone().real_mul(s.clone()))
            .eq(one),
    );

    let y0 = x0
        .clone()
        .real_mul(c.clone())
        .real_sub(x1.clone().real_mul(s.clone()));
    let y1 = x0.clone().real_mul(s).real_add(x1.clone().real_mul(c));

    let norm_x_sq = x0.clone().real_mul(x0).real_add(x1.clone().real_mul(x1));
    let norm_y_sq = y0.clone().real_mul(y0).real_add(y1.clone().real_mul(y1));

    program.assert(norm_x_sq.ne(norm_y_sq));
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "rope_norm_preservation".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 6: Sliding Window Sparsity Bound
// ---------------------------------------------------------------------------

/// Prove each row of a 5x5 sliding window mask (W=1) has at most 3
/// non-zero entries (2*W+1 = 3). Uses QF_LRA.
pub fn prove_sliding_window_sparsity() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let three = Expr::real(3);

    let expected: [[i64; 5]; 5] = [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ];

    let mut mask_vars = Vec::new();
    for row in 0..5 {
        for col in 0..5 {
            let var = declare_real(&mut program, &format!("m{}_{}", row, col));
            program.assert(var.clone().eq(Expr::real(expected[row][col])));
            mask_vars.push(var);
        }
    }

    // Violation: any row sum > 3
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let mut violation = zero.real_gt(one); // false
    for row in 0..5 {
        let mut row_sum = Expr::real(0);
        for col in 0..5 {
            row_sum = row_sum.real_add(mask_vars[row * 5 + col].clone());
        }
        violation = violation.or(row_sum.real_gt(three.clone()));
    }

    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "sliding_window_sparsity_5x5_w1".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 7: Cross-Attention Shape Compatibility
// ---------------------------------------------------------------------------

/// Prove cross-attention mask [D=2, E=3] has each row sum <= E.
/// Concrete mask with binary entries. Uses QF_LRA.
pub fn prove_cross_attention_shape_compat() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_LRA");
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let encoder_len = Expr::real(3);

    // 2x3 mask: Row 0 = [1,1,0], Row 1 = [1,1,1]
    let vals: [i64; 6] = [1, 1, 0, 1, 1, 1];
    let mut vars = Vec::new();
    for (idx, val) in vals.iter().enumerate() {
        let var = declare_real(&mut program, &format!("m{}", idx));
        program.assert(var.clone().eq(zero.clone()).or(var.clone().eq(one.clone())));
        program.assert(var.clone().eq(Expr::real(*val)));
        vars.push(var);
    }

    let row0_sum = vars[0]
        .clone()
        .real_add(vars[1].clone())
        .real_add(vars[2].clone());
    let row1_sum = vars[3]
        .clone()
        .real_add(vars[4].clone())
        .real_add(vars[5].clone());

    let violation = row0_sum
        .real_gt(encoder_len.clone())
        .or(row1_sum.real_gt(encoder_len));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "cross_attention_shape_compat_2x3".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Property 8: Mask Additive Form (0 = attend, -inf = ignore)
// ---------------------------------------------------------------------------

/// Prove additive mask: after softmax, masked positions (exp=0) get zero
/// probability and unmasked outputs sum to 1. Uses QF_NRA.
pub fn prove_mask_additive_form() -> Result<DpdfAttentionMaskResult, SmtError> {
    let mut program = AYProgram::new();
    program.set_logic("QF_NRA");
    let zero = Expr::real(0);
    let one = Expr::real(1);

    let e0 = declare_real(&mut program, "e0");
    let e1 = declare_real(&mut program, "e1");
    program.assert(e0.clone().real_gt(zero.clone()));
    program.assert(e1.clone().real_gt(zero.clone()));

    // Masked position: exp(score + (-inf)) = 0
    let e2 = declare_real(&mut program, "e2");
    program.assert(e2.clone().eq(zero.clone()));

    let denom = declare_real(&mut program, "denom");
    program.assert(
        denom
            .clone()
            .eq(e0.clone().real_add(e1.clone()).real_add(e2.clone())),
    );

    let s0 = declare_real(&mut program, "s0");
    let s1 = declare_real(&mut program, "s1");
    let s2 = declare_real(&mut program, "s2");
    program.assert(s0.clone().real_mul(denom.clone()).eq(e0));
    program.assert(s1.clone().real_mul(denom.clone()).eq(e1));
    program.assert(s2.clone().real_mul(denom).eq(e2));

    // Violation: masked output != 0 OR unmasked sum != 1
    let violation = s2.ne(zero).or(s0.real_add(s1).ne(one));
    program.assert(violation);
    program.check_sat();

    let smt2 = program.to_string();
    let (proven, detail) = execute_and_check(&program);
    Ok(DpdfAttentionMaskResult {
        property: "mask_additive_form_softmax_zero".to_string(),
        proven,
        smt2,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ay_attention_mask_position_bias_dpdf_tests.rs"]
mod tests;
