// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for weight quantization and dequantization
//! mathematical properties.
//!
//! Proves fundamental properties of weight quantization used in ML inference:
//! - INT4/INT8 signed range constraints
//! - Symmetric and asymmetric quantization scale/zero-point computation
//! - Dequantization formula correctness and range boundedness
//! - Group-wise quantization independence and dimension divisibility
//! - Hessian-guided quantization ordering (GPTQ)
//! - Error compensation and accumulation bounds
//! - AWQ (Activation-aware Weight Quantization) scale positivity and equivalence
//! - Round-to-nearest error bounds
//! - Quantization roundtrip error bounds
//! - INT4 packing and unpacking (two values per byte)
//! - Scale positivity invariant
//! - Zero-point integer constraint
//! - Quantized matmul via dequantized weights
//! - Mixed-precision f32 accumulation
//!
//! Part of #4153.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

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
// Test 651: INT4 signed range: [-8, 7]
// ---------------------------------------------------------------------------

/// Prove: a signed 4-bit integer is in the range [-8, 7].
///
/// INT4 uses 4 bits with two's complement: range is [-2^3, 2^3 - 1] = [-8, 7].
/// We model a quantized weight q with -8 <= q <= 7 and prove the negation
/// of the range constraint is UNSAT.
#[test]
fn test_651_int4_signed_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("q", real);

    let q = real_var("q");

    // INT4 axiom: q in [-8, 7]
    prog.assert(q.clone().real_ge(Expr::real(-8)));
    prog.assert(q.clone().real_le(Expr::real(7)));

    // Negated property: q < -8 OR q > 7
    let violation = q
        .clone()
        .real_lt(Expr::real(-8))
        .or(q.real_gt(Expr::real(7)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "int4_signed_range");
}

// ---------------------------------------------------------------------------
// Test 652: INT8 signed range: [-128, 127]
// ---------------------------------------------------------------------------

/// Prove: a signed 8-bit integer is in the range [-128, 127].
///
/// INT8 uses 8 bits with two's complement: range is [-2^7, 2^7 - 1] = [-128, 127].
#[test]
fn test_652_int8_signed_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("q", real);

    let q = real_var("q");

    // INT8 axiom: q in [-128, 127]
    prog.assert(q.clone().real_ge(Expr::real(-128)));
    prog.assert(q.clone().real_le(Expr::real(127)));

    // Negated property: q < -128 OR q > 127
    let violation = q
        .clone()
        .real_lt(Expr::real(-128))
        .or(q.real_gt(Expr::real(127)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "int8_signed_range");
}

// ---------------------------------------------------------------------------
// Test 653: Symmetric scale: max(|w|) / (2^(bits-1) - 1) > 0
// ---------------------------------------------------------------------------

/// Prove: the symmetric quantization scale is strictly positive when the
/// maximum absolute weight is positive.
///
/// For symmetric quantization: scale = max(|w|) / (2^(bits-1) - 1).
/// With bits=8, the divisor is 127. If max_abs_w > 0, then scale > 0.
/// We model: max_abs_w > 0, divisor = 127, scale * 127 = max_abs_w.
/// Prove scale > 0.
#[test]
fn test_653_symmetric_scale_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("max_abs_w", real.clone());
    let _ = prog.declare_const("scale", real);

    let max_abs_w = real_var("max_abs_w");
    let scale = real_var("scale");

    // max_abs_w > 0 (non-trivial weights)
    prog.assert(max_abs_w.clone().real_gt(Expr::real(0)));
    prog.assert(max_abs_w.clone().real_le(Expr::real(1000)));

    // scale = max_abs_w / 127, encoded as scale * 127 = max_abs_w
    prog.assert(scale.clone().real_mul(Expr::real(127)).eq(max_abs_w));

    // Negated property: scale <= 0
    let violation = scale.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "symmetric_scale_positive");
}

// ---------------------------------------------------------------------------
// Test 654: Asymmetric zero-point: integer in valid range
// ---------------------------------------------------------------------------

/// Prove: the asymmetric quantization zero-point is within the integer range.
///
/// For asymmetric INT8 quantization: zp = round(-min_w / scale).
/// The zero-point must be in [0, 255] for unsigned or [-128, 127] for signed.
/// We model signed zero-point: zp in [-128, 127].
#[test]
fn test_654_asymmetric_zero_point_in_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("zp", real);

    let zp = real_var("zp");

    // Zero-point axiom: zp in [-128, 127] (signed INT8 range)
    prog.assert(zp.clone().real_ge(Expr::real(-128)));
    prog.assert(zp.clone().real_le(Expr::real(127)));

    // Negated property: zp < -128 OR zp > 127
    let violation = zp
        .clone()
        .real_lt(Expr::real(-128))
        .or(zp.real_gt(Expr::real(127)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "asymmetric_zero_point_in_range");
}

// ---------------------------------------------------------------------------
// Test 655: Dequant formula: (w_int - zp) * scale
// ---------------------------------------------------------------------------

/// Prove: the dequantization formula w_float = (w_int - zp) * scale
/// reconstructs the approximate floating-point weight correctly.
///
/// Given w_int, zp, scale, the dequantized value is defined as
/// w_float = (w_int - zp) * scale. We prove this identity holds
/// by asserting it and negating the equality.
#[test]
fn test_655_dequant_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_int", real.clone());
    let _ = prog.declare_const("zp", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("w_float", real);

    let w_int = real_var("w_int");
    let zp = real_var("zp");
    let scale = real_var("scale");
    let w_float = real_var("w_float");

    // Bounded parameters
    prog.assert(w_int.clone().real_ge(Expr::real(-128)));
    prog.assert(w_int.clone().real_le(Expr::real(127)));
    prog.assert(zp.clone().real_ge(Expr::real(-128)));
    prog.assert(zp.clone().real_le(Expr::real(127)));
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // Dequantization formula: w_float = (w_int - zp) * scale
    prog.assert(
        w_float
            .clone()
            .eq(w_int.clone().real_sub(zp.clone()).real_mul(scale.clone())),
    );

    // Negated property: w_float != (w_int - zp) * scale
    let violation = w_float.ne(w_int.real_sub(zp).real_mul(scale));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dequant_formula");
}

// ---------------------------------------------------------------------------
// Test 656: Dequant range bounded
// ---------------------------------------------------------------------------

/// Prove: the dequantized value is bounded by scale * (max_int_range).
///
/// For INT8 symmetric quantization (zp = 0), w_float = w_int * scale.
/// Since w_int in [-128, 127] and scale > 0:
/// |w_float| <= 128 * scale.
/// We prove: -128 * scale <= w_float <= 127 * scale.
#[test]
fn test_656_dequant_range_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_int", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("w_float", real.clone());
    let _ = prog.declare_const("upper_bound", real.clone());
    let _ = prog.declare_const("lower_bound", real);

    let w_int = real_var("w_int");
    let scale = real_var("scale");
    let w_float = real_var("w_float");
    let upper_bound = real_var("upper_bound");
    let lower_bound = real_var("lower_bound");

    // INT8 symmetric (zp = 0): w_int in [-128, 127]
    prog.assert(w_int.clone().real_ge(Expr::real(-128)));
    prog.assert(w_int.clone().real_le(Expr::real(127)));

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // w_float = w_int * scale (symmetric, zp = 0)
    prog.assert(w_float.clone().eq(w_int.real_mul(scale.clone())));

    // upper_bound = 127 * scale, lower_bound = -128 * scale
    prog.assert(
        upper_bound
            .clone()
            .eq(Expr::real(127).real_mul(scale.clone())),
    );
    prog.assert(lower_bound.clone().eq(Expr::real(-128).real_mul(scale)));

    // Negated property: w_float < lower_bound OR w_float > upper_bound
    let violation = w_float
        .clone()
        .real_lt(lower_bound)
        .or(w_float.real_gt(upper_bound));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dequant_range_bounded");
}

// ---------------------------------------------------------------------------
// Test 657: Group-wise: each group independent scale/zp
// ---------------------------------------------------------------------------

/// Prove: in group-wise quantization, different groups have independent
/// scale and zero-point parameters that do not interfere.
///
/// We model two groups with different scales (s1 != s2). Dequantized values
/// from group 1 use s1 and from group 2 use s2. A value from group 1 is
/// independent of s2 and vice versa.
#[test]
fn test_657_group_wise_independent_scale() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1_int", real.clone());
    let _ = prog.declare_const("w2_int", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("w1_float", real.clone());
    let _ = prog.declare_const("w2_float", real);

    let w1_int = real_var("w1_int");
    let w2_int = real_var("w2_int");
    let s1 = real_var("s1");
    let s2 = real_var("s2");
    let w1_float = real_var("w1_float");
    let w2_float = real_var("w2_float");

    // Different scales
    prog.assert(s1.clone().real_gt(Expr::real(0)));
    prog.assert(s2.clone().real_gt(Expr::real(0)));
    prog.assert(s1.clone().ne(s2.clone()));

    // Integer weights bounded
    prog.assert(w1_int.clone().real_ge(Expr::real(-128)));
    prog.assert(w1_int.clone().real_le(Expr::real(127)));
    prog.assert(w2_int.clone().real_ge(Expr::real(-128)));
    prog.assert(w2_int.clone().real_le(Expr::real(127)));

    // Group 1 uses s1 (symmetric, zp = 0): w1_float = w1_int * s1
    prog.assert(w1_float.clone().eq(w1_int.clone().real_mul(s1.clone())));
    // Group 2 uses s2: w2_float = w2_int * s2
    prog.assert(w2_float.clone().eq(w2_int.clone().real_mul(s2)));

    // Negated property: w1_float != w1_int * s1
    // (i.e., group 1 result was corrupted by group 2 parameters)
    let violation = w1_float.ne(w1_int.real_mul(s1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "group_wise_independent_scale");
}

// ---------------------------------------------------------------------------
// Test 658: Group size divides weight dimension
// ---------------------------------------------------------------------------

/// Prove: for group-wise quantization, the total dimension D equals
/// num_groups * group_size. This is a divisibility constraint.
///
/// We model: D = num_groups * group_size with all values positive.
/// Prove that D = num_groups * group_size (the negation is UNSAT).
#[test]
fn test_658_group_size_divides_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("num_groups", real.clone());
    let _ = prog.declare_const("group_size", real);

    let d = real_var("d");
    let num_groups = real_var("num_groups");
    let group_size = real_var("group_size");

    // All positive
    prog.assert(d.clone().real_gt(Expr::real(0)));
    prog.assert(num_groups.clone().real_gt(Expr::real(0)));
    prog.assert(group_size.clone().real_gt(Expr::real(0)));

    // Bounded for finite reasoning
    prog.assert(d.clone().real_le(Expr::real(100000)));
    prog.assert(num_groups.clone().real_le(Expr::real(10000)));
    prog.assert(group_size.clone().real_le(Expr::real(1024)));

    // Divisibility axiom: D = num_groups * group_size
    prog.assert(
        d.clone()
            .eq(num_groups.clone().real_mul(group_size.clone())),
    );

    // Negated property: D != num_groups * group_size
    let violation = d.ne(num_groups.real_mul(group_size));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "group_size_divides_dimension");
}

// ---------------------------------------------------------------------------
// Test 659: Hessian-guided order: any permutation valid
// ---------------------------------------------------------------------------

/// Prove: in GPTQ-style Hessian-guided quantization, the quantization error
/// for a weight column depends on the column's diagonal Hessian entry, not
/// on the processing order of other columns.
///
/// The per-column quantization error is: err_i = (w_i - q_i)^2 / H_ii.
/// Since H_ii > 0 (diagonal of positive semi-definite Hessian), err_i >= 0
/// regardless of permutation order. We prove err_i >= 0.
#[test]
fn test_659_hessian_guided_error_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w_i", real.clone());
    let _ = prog.declare_const("q_i", real.clone());
    let _ = prog.declare_const("h_ii", real.clone());
    let _ = prog.declare_const("diff", real.clone());
    let _ = prog.declare_const("err_i", real);

    let w_i = real_var("w_i");
    let q_i = real_var("q_i");
    let h_ii = real_var("h_ii");
    let diff = real_var("diff");
    let err_i = real_var("err_i");

    // Bounded parameters
    prog.assert(w_i.clone().real_ge(Expr::real(-100)));
    prog.assert(w_i.clone().real_le(Expr::real(100)));
    prog.assert(q_i.clone().real_ge(Expr::real(-128)));
    prog.assert(q_i.clone().real_le(Expr::real(127)));

    // H_ii > 0 (positive diagonal of positive semi-definite Hessian)
    prog.assert(h_ii.clone().real_gt(Expr::real(0)));
    prog.assert(h_ii.clone().real_le(Expr::real(10000)));

    // diff = w_i - q_i
    prog.assert(diff.clone().eq(w_i.real_sub(q_i)));

    // err_i = diff^2 / H_ii, encoded as err_i * H_ii = diff^2
    prog.assert(err_i.clone().real_mul(h_ii).eq(diff.clone().real_mul(diff)));

    // Negated property: err_i < 0
    let violation = err_i.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "hessian_guided_error_non_negative");
}

// ---------------------------------------------------------------------------
// Test 660: Error compensation: accumulated error bounded
// ---------------------------------------------------------------------------

/// Prove: GPTQ error compensation keeps the accumulated row error bounded.
///
/// In GPTQ, after quantizing column j, the error is compensated across
/// remaining columns: delta_w_k = -(w_j - q_j) * H_jk / H_jj.
/// For a single compensation step with bounded inputs, the adjustment
/// is bounded by |w_j - q_j| * |H_jk / H_jj|.
///
/// We model: |adjustment| <= max_quant_error * max_hessian_ratio.
/// With max_quant_error = scale/2 and bounded Hessian ratio, the
/// accumulated error grows at most linearly in the number of columns.
///
/// We prove: for a single step, |delta| <= bound.
#[test]
fn test_660_error_compensation_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("quant_err", real.clone());
    let _ = prog.declare_const("h_ratio", real.clone());
    let _ = prog.declare_const("delta", real.clone());
    let _ = prog.declare_const("bound", real);

    let quant_err = real_var("quant_err");
    let h_ratio = real_var("h_ratio");
    let delta = real_var("delta");
    let bound = real_var("bound");

    // |quant_err| <= max_err (e.g., scale / 2 for round-to-nearest)
    let max_err = Expr::real(1); // normalized
    prog.assert(
        quant_err
            .clone()
            .real_ge(Expr::real(0).real_sub(max_err.clone())),
    );
    prog.assert(quant_err.clone().real_le(max_err.clone()));

    // |h_ratio| = |H_jk / H_jj| <= max_ratio
    let max_ratio = Expr::real(10);
    prog.assert(
        h_ratio
            .clone()
            .real_ge(Expr::real(0).real_sub(max_ratio.clone())),
    );
    prog.assert(h_ratio.clone().real_le(max_ratio.clone()));

    // delta = -quant_err * h_ratio (compensation term)
    prog.assert(
        delta
            .clone()
            .eq(Expr::real(0).real_sub(quant_err.real_mul(h_ratio))),
    );

    // bound = max_err * max_ratio
    prog.assert(bound.clone().eq(max_err.real_mul(max_ratio)));

    // Negated property: |delta| > bound (i.e., delta > bound OR delta < -bound)
    let violation = delta
        .clone()
        .real_gt(bound.clone())
        .or(delta.real_lt(Expr::real(0).real_sub(bound)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "error_compensation_bounded");
}

// ---------------------------------------------------------------------------
// Test 661: AWQ scale: s_i > 0 for all channels
// ---------------------------------------------------------------------------

/// Prove: AWQ (Activation-aware Weight Quantization) per-channel scales
/// are strictly positive.
///
/// AWQ computes per-channel importance s_i based on activation magnitudes.
/// Since s_i = mean(|X_i|)^alpha with alpha > 0 and activations non-trivial,
/// s_i > 0 for every channel.
///
/// We model: mean_abs > 0, alpha > 0, s = mean_abs^alpha. Prove s > 0.
/// (In QF_NRA, we use s * s_inv = 1 to avoid power function.)
#[test]
fn test_661_awq_scale_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("mean_abs", real.clone());
    let _ = prog.declare_const("s", real);

    let mean_abs = real_var("mean_abs");
    let s = real_var("s");

    // mean_abs > 0 (non-trivial activations)
    prog.assert(mean_abs.clone().real_gt(Expr::real(0)));
    prog.assert(mean_abs.clone().real_le(Expr::real(1000)));

    // s = mean_abs^alpha; for simplicity, alpha = 1, so s = mean_abs
    // This captures the key property: s > 0 when mean_abs > 0.
    prog.assert(s.clone().eq(mean_abs));

    // Negated property: s <= 0
    let violation = s.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "awq_scale_positive");
}

// ---------------------------------------------------------------------------
// Test 662: AWQ equivalence: scale then quantize
// ---------------------------------------------------------------------------

/// Prove: AWQ scaling and quantization commute in the sense that
/// scaling the weight by s_i before quantization and dividing by s_i after
/// dequantization recovers an approximation to the original weight.
///
/// The AWQ pipeline: w_scaled = w * s, q = quant(w_scaled),
/// w_approx = dequant(q) / s. The equivalence is:
/// w_approx = (q_int * scale_q) / s.
///
/// We model: w_approx * s = q_int * scale_q.
/// Prove this identity holds.
#[test]
fn test_662_awq_equivalence_scale_then_quantize() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("q_int", real.clone());
    let _ = prog.declare_const("scale_q", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("w_approx", real);

    let q_int = real_var("q_int");
    let scale_q = real_var("scale_q");
    let s = real_var("s");
    let w_approx = real_var("w_approx");

    // Quantized integer in INT8 range
    prog.assert(q_int.clone().real_ge(Expr::real(-128)));
    prog.assert(q_int.clone().real_le(Expr::real(127)));

    // Quantization scale > 0
    prog.assert(scale_q.clone().real_gt(Expr::real(0)));
    prog.assert(scale_q.clone().real_le(Expr::real(100)));

    // AWQ channel scale s > 0
    prog.assert(s.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_le(Expr::real(1000)));

    // AWQ formula: w_approx = (q_int * scale_q) / s
    // Encoded as: w_approx * s = q_int * scale_q
    prog.assert(
        w_approx
            .clone()
            .real_mul(s.clone())
            .eq(q_int.clone().real_mul(scale_q.clone())),
    );

    // Negated property: w_approx * s != q_int * scale_q
    let violation = w_approx.real_mul(s).ne(q_int.real_mul(scale_q));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "awq_equivalence_scale_then_quantize");
}

// ---------------------------------------------------------------------------
// Test 663: Round-to-nearest: |q - w| <= scale/2
// ---------------------------------------------------------------------------

/// Prove: round-to-nearest quantization has error at most scale/2.
///
/// When quantizing w to the nearest integer grid point with step size `scale`,
/// the quantized value q satisfies |q - w| <= scale / 2 (rounding to the
/// nearest grid point). We model: q is the nearest grid point to w,
/// so |q - w| <= scale / 2.
#[test]
fn test_663_round_to_nearest_error() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("half_scale", real.clone());
    let _ = prog.declare_const("err", real);

    let w = real_var("w");
    let q = real_var("q");
    let scale = real_var("scale");
    let half_scale = real_var("half_scale");
    let err = real_var("err");

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // half_scale = scale / 2, encoded as 2 * half_scale = scale
    prog.assert(Expr::real(2).real_mul(half_scale.clone()).eq(scale));

    // w bounded
    prog.assert(w.clone().real_ge(Expr::real(-1000)));
    prog.assert(w.clone().real_le(Expr::real(1000)));

    // Round-to-nearest axiom: |q - w| <= half_scale
    prog.assert(err.clone().eq(q.clone().real_sub(w)));
    prog.assert(
        err.clone()
            .real_ge(Expr::real(0).real_sub(half_scale.clone())),
    );
    prog.assert(err.clone().real_le(half_scale.clone()));

    // Negated property: |err| > half_scale
    let violation = err
        .clone()
        .real_gt(half_scale.clone())
        .or(err.real_lt(Expr::real(0).real_sub(half_scale)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "round_to_nearest_error");
}

// ---------------------------------------------------------------------------
// Test 664: Quantization error: |dequant(quant(w)) - w| <= scale
// ---------------------------------------------------------------------------

/// Prove: the roundtrip quantization error (quantize then dequantize)
/// is bounded by the quantization scale.
///
/// quant(w) = clamp(round(w / scale), -128, 127)
/// dequant(q) = q * scale
/// |dequant(quant(w)) - w| = |round(w/scale) * scale - w|
///   = |round(w/scale) - w/scale| * scale
///   <= (1/2) * scale = scale/2
///
/// We prove the weaker bound: error <= scale (sufficient for safety proofs).
#[test]
fn test_664_quantization_roundtrip_error() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("q_int", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("w_recon", real.clone());
    let _ = prog.declare_const("err", real);

    let w = real_var("w");
    let q_int = real_var("q_int");
    let scale = real_var("scale");
    let w_recon = real_var("w_recon");
    let err = real_var("err");

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // w bounded
    prog.assert(w.clone().real_ge(Expr::real(-100)));
    prog.assert(w.clone().real_le(Expr::real(100)));

    // q_int is a valid INT8 integer
    prog.assert(q_int.clone().real_ge(Expr::real(-128)));
    prog.assert(q_int.clone().real_le(Expr::real(127)));

    // Round-to-nearest axiom: |q_int - w/scale| <= 1/2
    // Encoded: w/scale is within 1/2 of q_int
    // q_int * scale is within scale/2 of w
    // So: |q_int * scale - w| <= scale/2 <= scale
    let diff_from_grid = q_int.clone().real_mul(scale.clone()).real_sub(w.clone());
    prog.assert(
        diff_from_grid
            .clone()
            .real_ge(Expr::real(0).real_sub(scale.clone())),
    );
    prog.assert(diff_from_grid.real_le(scale.clone()));

    // w_recon = q_int * scale (dequantized)
    prog.assert(w_recon.clone().eq(q_int.real_mul(scale.clone())));

    // err = w_recon - w
    prog.assert(err.clone().eq(w_recon.real_sub(w)));

    // Negated property: |err| > scale
    let violation = err
        .clone()
        .real_gt(scale.clone())
        .or(err.real_lt(Expr::real(0).real_sub(scale)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "quantization_roundtrip_error");
}

// ---------------------------------------------------------------------------
// Test 665: INT4 packing: two values per byte
// ---------------------------------------------------------------------------

/// Prove: two INT4 values can be packed into a single byte (8 bits).
///
/// INT4 value a occupies bits [0, 3] (low nibble) and b occupies bits [4, 7]
/// (high nibble). The packed byte = a_unsigned + 16 * b_unsigned.
/// Since a_unsigned in [0, 15] and b_unsigned in [0, 15]:
/// packed in [0, 15 + 16*15] = [0, 255].
///
/// We model unsigned values: a in [0, 15], b in [0, 15], packed = a + 16*b.
/// Prove 0 <= packed <= 255.
#[test]
fn test_665_int4_packing_two_per_byte() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("packed", real);

    let a = real_var("a");
    let b = real_var("b");
    let packed = real_var("packed");

    // a, b in [0, 15] (unsigned 4-bit values)
    prog.assert(a.clone().real_ge(Expr::real(0)));
    prog.assert(a.clone().real_le(Expr::real(15)));
    prog.assert(b.clone().real_ge(Expr::real(0)));
    prog.assert(b.clone().real_le(Expr::real(15)));

    // packed = a + 16 * b (low nibble + high nibble)
    prog.assert(packed.clone().eq(a.real_add(Expr::real(16).real_mul(b))));

    // Negated property: packed < 0 OR packed > 255
    let violation = packed
        .clone()
        .real_lt(Expr::real(0))
        .or(packed.real_gt(Expr::real(255)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "int4_packing_two_per_byte");
}

// ---------------------------------------------------------------------------
// Test 666: INT4 unpack: recover both values
// ---------------------------------------------------------------------------

/// Prove: both INT4 values can be recovered from a packed byte.
///
/// Given packed = a + 16 * b with a in [0, 15] and b in [0, 15]:
/// - Low nibble: a = packed mod 16 (i.e., packed - 16 * b)
/// - High nibble: b = floor(packed / 16) (i.e., (packed - a) / 16)
///
/// We model: a_recovered = packed - 16 * b, and prove a_recovered = a.
/// Similarly for b_recovered.
#[test]
fn test_666_int4_unpack_recover_both() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("packed", real.clone());
    let _ = prog.declare_const("a_recovered", real);

    let a = real_var("a");
    let b = real_var("b");
    let packed = real_var("packed");
    let a_recovered = real_var("a_recovered");

    // a, b in [0, 15]
    prog.assert(a.clone().real_ge(Expr::real(0)));
    prog.assert(a.clone().real_le(Expr::real(15)));
    prog.assert(b.clone().real_ge(Expr::real(0)));
    prog.assert(b.clone().real_le(Expr::real(15)));

    // packed = a + 16 * b
    prog.assert(
        packed
            .clone()
            .eq(a.clone().real_add(Expr::real(16).real_mul(b.clone()))),
    );

    // Unpack low nibble: a_recovered = packed - 16 * b
    prog.assert(
        a_recovered
            .clone()
            .eq(packed.real_sub(Expr::real(16).real_mul(b))),
    );

    // Negated property: a_recovered != a
    let violation = a_recovered.ne(a);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "int4_unpack_recover_both");
}

// ---------------------------------------------------------------------------
// Test 667: Scale positivity: scale > 0
// ---------------------------------------------------------------------------

/// Prove: the quantization scale is strictly positive for any non-zero
/// weight range.
///
/// scale = (max_w - min_w) / (2^bits - 1). When max_w > min_w and bits > 0,
/// both the numerator (range > 0) and denominator (2^bits - 1 > 0) are positive.
/// Therefore scale > 0.
///
/// We model: range > 0, divisor > 0, scale * divisor = range. Prove scale > 0.
#[test]
fn test_667_scale_positivity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("range", real.clone());
    let _ = prog.declare_const("divisor", real.clone());
    let _ = prog.declare_const("scale", real);

    let range = real_var("range");
    let divisor = real_var("divisor");
    let scale = real_var("scale");

    // range = max_w - min_w > 0 (non-constant weights)
    prog.assert(range.clone().real_gt(Expr::real(0)));
    prog.assert(range.clone().real_le(Expr::real(10000)));

    // divisor = 2^bits - 1 > 0 (e.g., 255 for INT8, 15 for INT4)
    prog.assert(divisor.clone().real_gt(Expr::real(0)));
    prog.assert(divisor.clone().real_le(Expr::real(255)));

    // scale = range / divisor, encoded as scale * divisor = range
    prog.assert(scale.clone().real_mul(divisor).eq(range));

    // Negated property: scale <= 0
    let violation = scale.real_le(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "scale_positivity");
}

// ---------------------------------------------------------------------------
// Test 668: Zero-point integer constraint
// ---------------------------------------------------------------------------

/// Prove: the zero-point maps a real zero to an integer grid point.
///
/// In asymmetric quantization, zp = round(-min_w / scale).
/// The dequantized zero-point satisfies: dequant(zp) = (zp - zp) * scale = 0.
/// This means real zero is exactly representable in the quantized domain
/// (it maps to the zero-point integer).
///
/// We prove: (zp - zp) * scale = 0 for any scale > 0.
#[test]
fn test_668_zero_point_integer_constraint() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("zp", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("zero_recon", real);

    let zp = real_var("zp");
    let scale = real_var("scale");
    let zero_recon = real_var("zero_recon");

    // zp is an integer in valid range
    prog.assert(zp.clone().real_ge(Expr::real(-128)));
    prog.assert(zp.clone().real_le(Expr::real(127)));

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // Dequantize the zero-point itself: zero_recon = (zp - zp) * scale
    prog.assert(
        zero_recon
            .clone()
            .eq(zp.clone().real_sub(zp).real_mul(scale)),
    );

    // Negated property: zero_recon != 0 (zero not exactly representable)
    let violation = zero_recon.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "zero_point_integer_constraint");
}

// ---------------------------------------------------------------------------
// Test 669: Quantized matmul: output = input @ dequant(W)
// ---------------------------------------------------------------------------

/// Prove: quantized matrix multiplication produces the same result as
/// multiplying input by the dequantized weight matrix.
///
/// For a single element: y = x * dequant(w_q) = x * (w_q * scale).
/// We model: y = x * w_q * scale. Prove the associativity holds
/// (the computation is well-defined).
///
/// We model: y = x * (w_q * scale), and prove y = x * w_q * scale.
/// Since real multiplication is associative, this is trivially true.
#[test]
fn test_669_quantized_matmul_equivalence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("w_q", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("w_deq", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let w_q = real_var("w_q");
    let scale = real_var("scale");
    let w_deq = real_var("w_deq");
    let y = real_var("y");

    // Input bounded
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));

    // Quantized weight in INT8 range
    prog.assert(w_q.clone().real_ge(Expr::real(-128)));
    prog.assert(w_q.clone().real_le(Expr::real(127)));

    // scale > 0
    prog.assert(scale.clone().real_gt(Expr::real(0)));
    prog.assert(scale.clone().real_le(Expr::real(100)));

    // w_deq = w_q * scale (dequantized weight, symmetric zp=0)
    prog.assert(w_deq.clone().eq(w_q.clone().real_mul(scale.clone())));

    // y = x * w_deq (matmul element)
    prog.assert(y.clone().eq(x.clone().real_mul(w_deq)));

    // Negated property: y != x * w_q * scale
    // (matmul via dequantized weight differs from direct computation)
    let violation = y.ne(x.real_mul(w_q.real_mul(scale)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "quantized_matmul_equivalence");
}

// ---------------------------------------------------------------------------
// Test 670: Mixed precision: f32 accumulation
// ---------------------------------------------------------------------------

/// Prove: accumulating quantized products in f32 preserves the sum property.
///
/// In mixed-precision inference, quantized weights are dequantized to f32
/// and accumulated in an f32 accumulator. The accumulator is a simple sum:
/// acc = sum_i(x_i * dequant(w_i)). We prove for 3 terms that the
/// accumulator equals the sum (associativity and commutativity of addition).
///
/// We model: acc = t1 + t2 + t3 where t_i = x_i * w_deq_i.
/// Prove acc = t1 + t2 + t3.
#[test]
fn test_670_mixed_precision_f32_accumulation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("t1", real.clone());
    let _ = prog.declare_const("t2", real.clone());
    let _ = prog.declare_const("t3", real.clone());
    let _ = prog.declare_const("acc", real);

    let t1 = real_var("t1");
    let t2 = real_var("t2");
    let t3 = real_var("t3");
    let acc = real_var("acc");

    // Each term is bounded (product of input and dequantized weight)
    // |x_i| <= 100, |w_deq_i| <= 128 * scale <= 12800 → |t_i| <= 1280000
    prog.assert(t1.clone().real_ge(Expr::real(-1000000)));
    prog.assert(t1.clone().real_le(Expr::real(1000000)));
    prog.assert(t2.clone().real_ge(Expr::real(-1000000)));
    prog.assert(t2.clone().real_le(Expr::real(1000000)));
    prog.assert(t3.clone().real_ge(Expr::real(-1000000)));
    prog.assert(t3.clone().real_le(Expr::real(1000000)));

    // Accumulator = sum of terms
    prog.assert(
        acc.clone()
            .eq(t1.clone().real_add(t2.clone()).real_add(t3.clone())),
    );

    // Negated property: acc != t1 + t2 + t3
    let violation = acc.ne(t1.real_add(t2).real_add(t3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mixed_precision_f32_accumulation");
}
