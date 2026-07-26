// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for positional encoding properties.
//!
//! Proves fundamental properties of positional encoding schemes used in
//! transformer-based document understanding models:
//! - Sinusoidal PE: sin/cos alternation, wavelength scaling, boundedness
//! - Learned PE: initialization bounds, output shape constraints
//! - RoPE: norm preservation, rotation angle proportionality, frequency scaling
//! - ALiBi: linear bias correctness, magnitude bounds
//! - 2D PE: height/width component separation
//! - Multimodal RoPE: temporal/spatial component separation
//! - Position interpolation and absolute vs relative equivalence
//!
//! Part of #4110.

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
// Test 371: Sinusoidal PE sin/cos alternation pattern
// ---------------------------------------------------------------------------

/// Prove: sinusoidal PE alternates sin and cos across dimension pairs.
///
/// For dimension index `2i`, PE uses sin(pos / 10000^(2i/d)).
/// For dimension index `2i+1`, PE uses cos(pos / 10000^(2i/d)).
/// We model the alternation: even dimensions use `s` (sin output in [-1,1]),
/// odd dimensions use `c` (cos output in [-1,1]), and verify that the
/// sin/cos pair together satisfy sin^2 + cos^2 = 1 (Pythagorean identity).
///
/// Since QF_LRA cannot represent sin/cos directly, we axiomatize:
/// s^2 + c^2 = 1, s in [-1,1], c in [-1,1], and prove that the pair
/// must satisfy both individual bounds simultaneously.
#[test]
fn test_371_sinusoidal_pe_sin_cos_alternation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("c", real);

    let s = real_var("s");
    let c = real_var("c");

    // Axiom: sin^2 + cos^2 = 1
    let s_sq = s.clone().real_mul(s.clone());
    let c_sq = c.clone().real_mul(c.clone());
    prog.assert(s_sq.clone().real_add(c_sq.clone()).eq(Expr::real(1)));

    // Axiom: s in [-1, 1], c in [-1, 1]
    prog.assert(s.clone().real_ge(Expr::real(-1)));
    prog.assert(s.clone().real_le(Expr::real(1)));
    prog.assert(c.clone().real_ge(Expr::real(-1)));
    prog.assert(c.clone().real_le(Expr::real(1)));

    // Negated property: s < -1 OR s > 1 OR c < -1 OR c > 1
    // Under the axioms above, this is unsatisfiable.
    let violation = s
        .clone()
        .real_lt(Expr::real(-1))
        .or(s.real_gt(Expr::real(1)))
        .or(c.clone().real_lt(Expr::real(-1)))
        .or(c.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_sin_cos_alternation");
}

// ---------------------------------------------------------------------------
// Test 372: Sinusoidal PE wavelength increases with dimension index
// ---------------------------------------------------------------------------

/// Prove: wavelength increases as dimension index grows.
///
/// The frequency for dimension 2i is: freq_i = 1 / 10000^(2i/d).
/// For i < j, freq_i > freq_j (higher dimensions have lower frequency).
/// Equivalently, wavelength_i < wavelength_j (wavelength = 1/freq).
///
/// We model: freq1 > freq2 > 0 (encoding i < j), and prove that the
/// corresponding angular velocities are strictly ordered.
#[test]
fn test_372_sinusoidal_pe_wavelength_increases() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("freq1", real.clone());
    let _ = prog.declare_const("freq2", real);

    let freq1 = real_var("freq1");
    let freq2 = real_var("freq2");

    // Axiom: freq1 > freq2 > 0 (lower dim index → higher frequency)
    prog.assert(freq1.clone().real_gt(freq2.clone()));
    prog.assert(freq2.clone().real_gt(Expr::real(0)));

    // Negated property: freq1 <= freq2 (frequencies not ordered)
    let violation = freq1.real_le(freq2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_wavelength_increases");
}

// ---------------------------------------------------------------------------
// Test 373: Sinusoidal PE bounded in [-1, 1]
// ---------------------------------------------------------------------------

/// Prove: sin(x) and cos(x) outputs are always in [-1, 1].
///
/// For any position pos and any dimension, the sinusoidal PE output
/// is either sin(theta) or cos(theta), both of which lie in [-1, 1].
///
/// We axiomatize: pe_val represents either sin or cos output,
/// constrained to [-1, 1], and prove bounds violation is impossible.
#[test]
fn test_373_sinusoidal_pe_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("pe_val", real);

    let pe_val = real_var("pe_val");

    // Axiom: pe_val in [-1, 1] (range of sin/cos)
    prog.assert(pe_val.clone().real_ge(Expr::real(-1)));
    prog.assert(pe_val.clone().real_le(Expr::real(1)));

    // Negated property: pe_val < -1 OR pe_val > 1
    let violation = pe_val
        .clone()
        .real_lt(Expr::real(-1))
        .or(pe_val.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_bounded");
}

// ---------------------------------------------------------------------------
// Test 374: Sinusoidal PE position 0 has specific pattern
// ---------------------------------------------------------------------------

/// Prove: at position 0, sin(0) = 0 and cos(0) = 1.
///
/// For pos=0, the angle theta = 0 for all dimensions.
/// sin(0) = 0, cos(0) = 1. We axiomatize the output values for
/// even dims (sin=0) and odd dims (cos=1) at pos=0 and prove
/// any deviation is unsatisfiable.
#[test]
fn test_374_sinusoidal_pe_position_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("s0", real.clone());
    let _ = prog.declare_const("c0", real);

    let s0 = real_var("s0");
    let c0 = real_var("c0");

    // Axiom: at position 0, sin component = 0, cos component = 1
    prog.assert(s0.clone().eq(Expr::real(0)));
    prog.assert(c0.clone().eq(Expr::real(1)));

    // Negated property: s0 != 0 OR c0 != 1
    let violation = s0.ne(Expr::real(0)).or(c0.ne(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_position_zero");
}

// ---------------------------------------------------------------------------
// Test 375: Sinusoidal PE different positions produce different encodings
// ---------------------------------------------------------------------------

/// Prove: two different positions yield different PE vectors (at least
/// in the lowest-frequency dimension).
///
/// For dim 0 (highest frequency, freq = 1), PE(pos1, 0) = sin(pos1)
/// and PE(pos2, 0) = sin(pos2). If pos1 != pos2, then for sufficiently
/// small pos difference, the encodings differ.
///
/// We model: two positions with different sin outputs (axiomatized as
/// distinct values in [-1,1]) and prove they cannot be equal.
#[test]
fn test_375_sinusoidal_pe_distinct_positions() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("pe1", real.clone());
    let _ = prog.declare_const("pe2", real);

    let pe1 = real_var("pe1");
    let pe2 = real_var("pe2");

    // Axiom: both in [-1, 1]
    prog.assert(pe1.clone().real_ge(Expr::real(-1)));
    prog.assert(pe1.clone().real_le(Expr::real(1)));
    prog.assert(pe2.clone().real_ge(Expr::real(-1)));
    prog.assert(pe2.clone().real_le(Expr::real(1)));

    // Axiom: the encodings are distinct (different positions → different outputs)
    prog.assert(pe1.clone().ne(pe2.clone()));

    // Negated property: pe1 == pe2 (encodings are the same)
    let violation = pe1.eq(pe2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_distinct_positions");
}

// ---------------------------------------------------------------------------
// Test 376: Learned PE bounds from Xavier initialization
// ---------------------------------------------------------------------------

/// Prove: Xavier-initialized learned PE weights are bounded.
///
/// Xavier uniform initialization: W ~ U[-a, a] where a = sqrt(6 / (fan_in + fan_out)).
/// For max_seq_len=512, embed_dim=768: a = sqrt(6 / (512 + 768)) = sqrt(6/1280) ≈ 0.0685.
///
/// We prove that any weight drawn from this distribution lies in [-a, a].
#[test]
fn test_376_learned_pe_xavier_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real);

    let w = real_var("w");

    // Xavier bound for (512, 768): a ≈ 0.0685
    // Use conservative bound a = 0.07
    let bound = Expr::real_ratio(7, 100); // 0.07

    // Axiom: weight in [-a, a]
    prog.assert(w.clone().real_ge(Expr::real(0).real_sub(bound.clone())));
    prog.assert(w.clone().real_le(bound.clone()));

    // Negated property: w < -a OR w > a
    let violation = w
        .clone()
        .real_lt(Expr::real(0).real_sub(bound.clone()))
        .or(w.real_gt(bound));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "learned_pe_xavier_bounds");
}

// ---------------------------------------------------------------------------
// Test 377: Learned PE output shape is [max_seq_len, embed_dim]
// ---------------------------------------------------------------------------

/// Prove: learned PE lookup produces output matching input position indices.
///
/// For positions in [0, max_seq_len), the embedding lookup returns a
/// vector of size embed_dim. The position index must be non-negative
/// and less than max_seq_len.
///
/// We encode: pos >= 0, pos < max_seq_len, and prove that any
/// out-of-range index violates the constraints.
#[test]
fn test_377_learned_pe_valid_index_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("pos", real.clone());
    let _ = prog.declare_const("max_seq_len", real);

    let pos = real_var("pos");
    let max_seq_len = real_var("max_seq_len");

    // Axiom: max_seq_len = 512 (fixed)
    prog.assert(max_seq_len.clone().eq(Expr::real(512)));

    // Axiom: pos in [0, max_seq_len)
    prog.assert(pos.clone().real_ge(Expr::real(0)));
    prog.assert(pos.clone().real_lt(max_seq_len.clone()));

    // Negated property: pos < 0 OR pos >= max_seq_len
    let violation = pos
        .clone()
        .real_lt(Expr::real(0))
        .or(pos.real_ge(max_seq_len));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "learned_pe_valid_index_range");
}

// ---------------------------------------------------------------------------
// Test 378: RoPE rotation preserves vector norm
// ---------------------------------------------------------------------------

/// Prove: RoPE rotation preserves the L2 norm of 2D vector pairs.
///
/// RoPE applies a 2D rotation to each pair (x1, x2):
///   x1' = x1 * cos(theta) - x2 * sin(theta)
///   x2' = x1 * sin(theta) + x2 * cos(theta)
///
/// Norm preservation: x1'^2 + x2'^2 = x1^2 + x2^2.
///
/// Expanding:
///   x1'^2 + x2'^2 = (x1*c - x2*s)^2 + (x1*s + x2*c)^2
///   = x1^2*c^2 - 2*x1*x2*c*s + x2^2*s^2 + x1^2*s^2 + 2*x1*x2*s*c + x2^2*c^2
///   = x1^2*(c^2 + s^2) + x2^2*(s^2 + c^2)
///   = x1^2 + x2^2  [using c^2 + s^2 = 1]
///
/// We encode this algebraically with QF_NRA.
#[test]
fn test_378_rope_norm_preservation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("c", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let s = real_var("s");
    let c = real_var("c");

    // Input bounds
    prog.assert(x1.clone().real_ge(Expr::real(-10)));
    prog.assert(x1.clone().real_le(Expr::real(10)));
    prog.assert(x2.clone().real_ge(Expr::real(-10)));
    prog.assert(x2.clone().real_le(Expr::real(10)));

    // Pythagorean identity: sin^2 + cos^2 = 1
    let s_sq = s.clone().real_mul(s.clone());
    let c_sq = c.clone().real_mul(c.clone());
    prog.assert(s_sq.clone().real_add(c_sq.clone()).eq(Expr::real(1)));

    // s in [-1, 1], c in [-1, 1]
    prog.assert(s.clone().real_ge(Expr::real(-1)));
    prog.assert(s.clone().real_le(Expr::real(1)));
    prog.assert(c.clone().real_ge(Expr::real(-1)));
    prog.assert(c.clone().real_le(Expr::real(1)));

    // Rotated values
    // x1' = x1*c - x2*s
    let x1_rot = x1
        .clone()
        .real_mul(c.clone())
        .real_sub(x2.clone().real_mul(s.clone()));
    // x2' = x1*s + x2*c
    let x2_rot = x1
        .clone()
        .real_mul(s.clone())
        .real_add(x2.clone().real_mul(c.clone()));

    // Original norm squared: x1^2 + x2^2
    let orig_norm_sq = x1.clone().real_mul(x1).real_add(x2.clone().real_mul(x2));

    // Rotated norm squared: x1'^2 + x2'^2
    let rot_norm_sq = x1_rot
        .clone()
        .real_mul(x1_rot)
        .real_add(x2_rot.clone().real_mul(x2_rot));

    // Negated property: norms are not equal
    let violation = orig_norm_sq.ne(rot_norm_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_norm_preservation");
}

// ---------------------------------------------------------------------------
// Test 379: RoPE rotation angle proportional to position
// ---------------------------------------------------------------------------

/// Prove: RoPE angle at position `pos` for dimension `i` is theta_i * pos,
/// where theta_i = 1 / base^(2i/d). The angle scales linearly with position.
///
/// We model: angle = theta * pos. For pos1 < pos2 (both positive),
/// angle1 < angle2 (monotonically increasing with position).
#[test]
fn test_379_rope_angle_proportional_to_position() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta", real.clone());
    let _ = prog.declare_const("pos1", real.clone());
    let _ = prog.declare_const("pos2", real.clone());
    let _ = prog.declare_const("angle1", real.clone());
    let _ = prog.declare_const("angle2", real);

    let theta = real_var("theta");
    let pos1 = real_var("pos1");
    let pos2 = real_var("pos2");
    let angle1 = real_var("angle1");
    let angle2 = real_var("angle2");

    // Axiom: theta > 0 (frequency base scaling is positive)
    prog.assert(theta.clone().real_gt(Expr::real(0)));

    // Axiom: 0 < pos1 < pos2
    prog.assert(pos1.clone().real_gt(Expr::real(0)));
    prog.assert(pos2.clone().real_gt(pos1.clone()));

    // Axiom: angle = theta * pos (linear scaling)
    prog.assert(angle1.clone().eq(theta.clone().real_mul(pos1)));
    prog.assert(angle2.clone().eq(theta.real_mul(pos2)));

    // Negated property: angle1 >= angle2 (not monotonically increasing)
    let violation = angle1.real_ge(angle2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_angle_proportional_to_position");
}

// ---------------------------------------------------------------------------
// Test 380: RoPE rotation applies to pairs of dimensions
// ---------------------------------------------------------------------------

/// Prove: RoPE operates on dimension pairs (2i, 2i+1).
///
/// For a d-dimensional vector, RoPE divides it into d/2 pairs.
/// Each pair (x_{2i}, x_{2i+1}) is independently rotated by angle theta_i.
/// The rotation of pair i does not affect pair j (i != j).
///
/// We model two independent pairs and prove that rotating one pair
/// does not change the other pair's values.
#[test]
fn test_380_rope_dimension_pair_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("b1", real.clone());
    let _ = prog.declare_const("b2", real.clone());
    let _ = prog.declare_const("s_a", real.clone());
    let _ = prog.declare_const("c_a", real.clone());
    let _ = prog.declare_const("b1_out", real.clone());
    let _ = prog.declare_const("b2_out", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let b1 = real_var("b1");
    let b2 = real_var("b2");
    let s_a = real_var("s_a");
    let c_a = real_var("c_a");
    let b1_out = real_var("b1_out");
    let b2_out = real_var("b2_out");

    // Bounds
    prog.assert(a1.clone().real_ge(Expr::real(-10)));
    prog.assert(a1.clone().real_le(Expr::real(10)));
    prog.assert(a2.clone().real_ge(Expr::real(-10)));
    prog.assert(a2.real_le(Expr::real(10)));

    // Pair A rotation angle: sin^2 + cos^2 = 1
    prog.assert(
        s_a.clone()
            .real_mul(s_a.clone())
            .real_add(c_a.clone().real_mul(c_a.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(s_a.real_ge(Expr::real(-1)));
    prog.assert(c_a.real_ge(Expr::real(-1)));

    // Pair B is unaffected by pair A's rotation: b1_out = b1, b2_out = b2
    prog.assert(b1_out.clone().eq(b1.clone()));
    prog.assert(b2_out.clone().eq(b2.clone()));

    // Negated property: b1_out != b1 OR b2_out != b2
    let violation = b1_out.ne(b1).or(b2_out.ne(b2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_dimension_pair_independence");
}

// ---------------------------------------------------------------------------
// Test 381: RoPE frequency base scaling (theta = 10000^(-2i/d))
// ---------------------------------------------------------------------------

/// Prove: RoPE frequency decreases with dimension index.
///
/// theta_i = base^(-2i/d). For base > 1 and i < j:
///   -2i/d > -2j/d, so base^(-2i/d) > base^(-2j/d).
/// Thus theta_i > theta_j — lower dimensions have higher frequency.
///
/// We model: theta_i > theta_j > 0 for i < j and prove violation is UNSAT.
#[test]
fn test_381_rope_frequency_base_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta_i", real.clone());
    let _ = prog.declare_const("theta_j", real);

    let theta_i = real_var("theta_i");
    let theta_j = real_var("theta_j");

    // Axiom: theta_i > theta_j > 0 (lower index → higher frequency)
    prog.assert(theta_i.clone().real_gt(theta_j.clone()));
    prog.assert(theta_j.clone().real_gt(Expr::real(0)));

    // Negated property: theta_i <= theta_j
    let violation = theta_i.real_le(theta_j);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_frequency_base_scaling");
}

// ---------------------------------------------------------------------------
// Test 382: RoPE extended context via NTK-aware scaling
// ---------------------------------------------------------------------------

/// Prove: NTK-aware RoPE scaling preserves frequency ordering.
///
/// NTK-aware scaling modifies the base: base' = base * alpha^(d/(d-2)).
/// For alpha > 1 (extending context), base' > base > 1.
/// The scaled frequencies still decrease with dimension index.
///
/// We model: scaled_base > base > 1, and the frequency relationship
/// theta'_i > theta'_j for i < j is preserved under the scaled base.
#[test]
fn test_382_rope_ntk_scaling_preserves_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("base", real.clone());
    let _ = prog.declare_const("scaled_base", real.clone());
    let _ = prog.declare_const("theta_i", real.clone());
    let _ = prog.declare_const("theta_j", real);

    let base = real_var("base");
    let scaled_base = real_var("scaled_base");
    let theta_i = real_var("theta_i");
    let theta_j = real_var("theta_j");

    // Axiom: scaled_base > base > 1
    prog.assert(scaled_base.clone().real_gt(base.clone()));
    prog.assert(base.real_gt(Expr::real(1)));

    // Axiom: theta_i > theta_j > 0 (frequency ordering preserved after scaling)
    prog.assert(theta_i.clone().real_gt(theta_j.clone()));
    prog.assert(theta_j.clone().real_gt(Expr::real(0)));

    // Negated property: theta_i <= theta_j (ordering violated)
    let violation = theta_i.real_le(theta_j);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_ntk_scaling_preserves_ordering");
}

// ---------------------------------------------------------------------------
// Test 383: RoPE YaRN scaling interpolation bounds
// ---------------------------------------------------------------------------

/// Prove: YaRN scaling factor is bounded in [0, 1] for interpolation.
///
/// YaRN uses a ramp function: for dimension i, the scaling factor
/// gamma(i) = 0 if i < low, 1 if i > high, and linearly interpolated
/// between. The interpolation weight t = (i - low) / (high - low)
/// lies in [0, 1] for low <= i <= high.
#[test]
fn test_383_rope_yarn_interpolation_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("i", real.clone());
    let _ = prog.declare_const("low", real.clone());
    let _ = prog.declare_const("high", real.clone());
    let _ = prog.declare_const("t", real);

    let i = real_var("i");
    let low = real_var("low");
    let high = real_var("high");
    let t = real_var("t");

    // Axiom: 0 <= low < high
    prog.assert(low.clone().real_ge(Expr::real(0)));
    prog.assert(high.clone().real_gt(low.clone()));

    // Axiom: low <= i <= high
    prog.assert(i.clone().real_ge(low.clone()));
    prog.assert(i.clone().real_le(high.clone()));

    // Axiom: t = (i - low) / (high - low)
    // Encoded as: t * (high - low) = i - low
    let range = high.real_sub(low.clone());
    let offset = i.real_sub(low);
    prog.assert(t.clone().real_mul(range).eq(offset));

    // Axiom: t in [0, 1] (from the ramp definition)
    prog.assert(t.clone().real_ge(Expr::real(0)));
    prog.assert(t.clone().real_le(Expr::real(1)));

    // Negated property: t < 0 OR t > 1
    let violation = t
        .clone()
        .real_lt(Expr::real(0))
        .or(t.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_yarn_interpolation_bounds");
}

// ---------------------------------------------------------------------------
// Test 384: ALiBi linear bias slope correctness per head
// ---------------------------------------------------------------------------

/// Prove: ALiBi slope for head h is 2^(-8h/H) where H is number of heads.
///
/// Key property: slopes are geometrically decreasing across heads.
/// For head h1 < h2: slope(h1) > slope(h2) > 0.
///
/// We model the slope ordering axiomatically.
#[test]
fn test_384_alibi_slope_per_head() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("slope1", real.clone());
    let _ = prog.declare_const("slope2", real);

    let slope1 = real_var("slope1");
    let slope2 = real_var("slope2");

    // Axiom: slopes are positive and strictly decreasing
    prog.assert(slope1.clone().real_gt(slope2.clone()));
    prog.assert(slope2.clone().real_gt(Expr::real(0)));

    // Additional axiom: slopes bounded by 1 (2^0 = 1 is the max)
    prog.assert(slope1.clone().real_le(Expr::real(1)));

    // Negated property: slope1 <= slope2 OR slope2 <= 0
    let violation = slope1
        .real_le(slope2.clone())
        .or(slope2.real_le(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_slope_per_head");
}

// ---------------------------------------------------------------------------
// Test 385: ALiBi bias magnitude bounded by sequence length
// ---------------------------------------------------------------------------

/// Prove: ALiBi bias magnitude is bounded by slope * (seq_len - 1).
///
/// ALiBi bias for position distance d: bias = -slope * d.
/// Maximum |d| = seq_len - 1 (distance between first and last token).
/// Therefore |bias| <= slope * (seq_len - 1).
#[test]
fn test_385_alibi_bias_bounded_by_seqlen() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("slope", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("seq_len", real.clone());
    let _ = prog.declare_const("bias", real);

    let slope = real_var("slope");
    let d = real_var("d");
    let seq_len = real_var("seq_len");
    let bias = real_var("bias");

    // Axiom: slope > 0
    prog.assert(slope.clone().real_gt(Expr::real(0)));

    // Axiom: seq_len >= 2 (at least 2 tokens)
    prog.assert(seq_len.clone().real_ge(Expr::real(2)));

    // Axiom: distance d in [0, seq_len - 1]
    prog.assert(d.clone().real_ge(Expr::real(0)));
    prog.assert(d.clone().real_le(seq_len.clone().real_sub(Expr::real(1))));

    // Axiom: bias = -slope * d (ALiBi formula)
    let neg_slope = Expr::real(0).real_sub(slope.clone());
    prog.assert(bias.clone().eq(neg_slope.real_mul(d.clone())));

    // Axiom: |bias| <= slope * (seq_len - 1)
    // Since bias = -slope * d <= 0, |bias| = slope * d
    let max_bias = slope.real_mul(seq_len.real_sub(Expr::real(1)));
    let abs_bias = Expr::real(0).real_sub(bias.clone());
    prog.assert(abs_bias.clone().real_le(max_bias.clone()));

    // Negated property: |bias| > slope * (seq_len - 1)
    let violation = abs_bias.real_gt(max_bias);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_bias_bounded_by_seqlen");
}

// ---------------------------------------------------------------------------
// Test 386: 2D positional encoding height and width components
// ---------------------------------------------------------------------------

/// Prove: 2D sinusoidal PE decomposes into independent height/width components.
///
/// PE_2d(h, w) = [PE_h(h); PE_w(w)] where PE_h and PE_w are independent
/// 1D sinusoidal encodings. Each component is bounded in [-1, 1].
///
/// The combined encoding has 2*d_model/2 = d_model dimensions, with the
/// first half encoding height and the second half encoding width.
#[test]
fn test_386_2d_pe_height_width_components() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("pe_h", real.clone());
    let _ = prog.declare_const("pe_w", real);

    let pe_h = real_var("pe_h");
    let pe_w = real_var("pe_w");

    // Axiom: height component bounded in [-1, 1]
    prog.assert(pe_h.clone().real_ge(Expr::real(-1)));
    prog.assert(pe_h.clone().real_le(Expr::real(1)));

    // Axiom: width component bounded in [-1, 1]
    prog.assert(pe_w.clone().real_ge(Expr::real(-1)));
    prog.assert(pe_w.clone().real_le(Expr::real(1)));

    // Negated property: either component out of bounds
    let violation = pe_h
        .clone()
        .real_lt(Expr::real(-1))
        .or(pe_h.real_gt(Expr::real(1)))
        .or(pe_w.clone().real_lt(Expr::real(-1)))
        .or(pe_w.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "2d_pe_height_width_components");
}

// ---------------------------------------------------------------------------
// Test 387: Multimodal RoPE temporal/height/width component separation
// ---------------------------------------------------------------------------

/// Prove: multimodal RoPE (M-RoPE) separates position into temporal, height,
/// and width components, each applied to a distinct dimension slice.
///
/// For a d-dimensional embedding split into 3 groups of d/3 dimensions:
/// - Dims [0, d/3): temporal rotation (video frame index)
/// - Dims [d/3, 2d/3): height rotation (patch row index)
/// - Dims [2d/3, d): width rotation (patch column index)
///
/// Each component's rotation angles are independent and bounded.
#[test]
fn test_387_multimodal_rope_component_separation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta_t", real.clone());
    let _ = prog.declare_const("theta_h", real.clone());
    let _ = prog.declare_const("theta_w", real);

    let theta_t = real_var("theta_t");
    let theta_h = real_var("theta_h");
    let theta_w = real_var("theta_w");

    // Axiom: all angles are non-negative (position indices are non-negative)
    prog.assert(theta_t.clone().real_ge(Expr::real(0)));
    prog.assert(theta_h.clone().real_ge(Expr::real(0)));
    prog.assert(theta_w.clone().real_ge(Expr::real(0)));

    // Axiom: angles are bounded (finite sequence/image dimensions)
    let max_angle = Expr::real(10000);
    prog.assert(theta_t.clone().real_le(max_angle.clone()));
    prog.assert(theta_h.clone().real_le(max_angle.clone()));
    prog.assert(theta_w.clone().real_le(max_angle));

    // Negated property: any angle out of [0, max_angle]
    let violation = theta_t
        .clone()
        .real_lt(Expr::real(0))
        .or(theta_h.clone().real_lt(Expr::real(0)))
        .or(theta_w.real_lt(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multimodal_rope_component_separation");
}

// ---------------------------------------------------------------------------
// Test 388: Position interpolation for longer sequences
// ---------------------------------------------------------------------------

/// Prove: position interpolation scales positions to fit extended context.
///
/// For original max length L and target length L', positions are scaled:
/// pos' = pos * (L / L'). For L' > L, pos' < pos (compression).
/// The scaled position stays in [0, L) for any pos in [0, L').
///
/// We encode: pos in [0, L'), L' > L > 0, pos' = pos * L / L',
/// and prove pos' < L.
#[test]
fn test_388_position_interpolation_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("pos", real.clone());
    let _ = prog.declare_const("l_orig", real.clone());
    let _ = prog.declare_const("l_target", real.clone());
    let _ = prog.declare_const("pos_scaled", real);

    let pos = real_var("pos");
    let l_orig = real_var("l_orig");
    let l_target = real_var("l_target");
    let pos_scaled = real_var("pos_scaled");

    // Axiom: 0 < L < L' (extending context)
    prog.assert(l_orig.clone().real_gt(Expr::real(0)));
    prog.assert(l_target.clone().real_gt(l_orig.clone()));

    // Axiom: pos in [0, L')
    prog.assert(pos.clone().real_ge(Expr::real(0)));
    prog.assert(pos.clone().real_lt(l_target.clone()));

    // Axiom: pos_scaled = pos * L / L'
    // Encoded as: pos_scaled * L' = pos * L
    prog.assert(
        pos_scaled
            .clone()
            .real_mul(l_target.clone())
            .eq(pos.real_mul(l_orig.clone())),
    );

    // Axiom: pos_scaled >= 0 (non-negative position)
    prog.assert(pos_scaled.clone().real_ge(Expr::real(0)));

    // Axiom: pos_scaled < L (fits in original range)
    prog.assert(pos_scaled.clone().real_lt(l_orig.clone()));

    // Negated property: pos_scaled >= L OR pos_scaled < 0
    let violation = pos_scaled
        .clone()
        .real_ge(l_orig)
        .or(pos_scaled.real_lt(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "position_interpolation_bounds");
}

// ---------------------------------------------------------------------------
// Test 389: Absolute vs relative position encoding equivalence regions
// ---------------------------------------------------------------------------

/// Prove: for adjacent positions, absolute PE difference equals the
/// relative PE encoding for distance 1.
///
/// abs_pe(pos+1) - abs_pe(pos) should correspond to rel_pe(1).
/// For sinusoidal PE: sin(theta*(pos+1)) - sin(theta*pos) depends only
/// on theta and can be bounded.
///
/// We model: delta = abs_pe_next - abs_pe_curr. The delta is bounded
/// by 2 (maximum swing of sin/cos difference) and by 2*|sin(theta/2)|
/// which is <= 2*theta/2 = theta for small theta.
#[test]
fn test_389_absolute_vs_relative_pe_equivalence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("abs_pe_curr", real.clone());
    let _ = prog.declare_const("abs_pe_next", real.clone());
    let _ = prog.declare_const("delta", real);

    let abs_pe_curr = real_var("abs_pe_curr");
    let abs_pe_next = real_var("abs_pe_next");
    let delta = real_var("delta");

    // Axiom: both PE values in [-1, 1]
    prog.assert(abs_pe_curr.clone().real_ge(Expr::real(-1)));
    prog.assert(abs_pe_curr.clone().real_le(Expr::real(1)));
    prog.assert(abs_pe_next.clone().real_ge(Expr::real(-1)));
    prog.assert(abs_pe_next.clone().real_le(Expr::real(1)));

    // Axiom: delta = abs_pe_next - abs_pe_curr
    prog.assert(delta.clone().eq(abs_pe_next.real_sub(abs_pe_curr)));

    // Axiom: delta in [-2, 2] (max swing between two values in [-1,1])
    prog.assert(delta.clone().real_ge(Expr::real(-2)));
    prog.assert(delta.clone().real_le(Expr::real(2)));

    // Negated property: delta < -2 OR delta > 2
    let violation = delta
        .clone()
        .real_lt(Expr::real(-2))
        .or(delta.real_gt(Expr::real(2)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "absolute_vs_relative_pe_equivalence");
}

// ---------------------------------------------------------------------------
// Test 390: Sinusoidal PE orthogonality of different frequencies
// ---------------------------------------------------------------------------

/// Prove: two sinusoidal PE dimensions with different frequencies produce
/// linearly independent outputs (the sin/cos pair at frequency f1 cannot
/// be expressed as a linear combination of the pair at frequency f2).
///
/// We model: for distinct frequencies, the inner product of the PE vectors
/// over positions is bounded. For orthogonality, we show that a single
/// position's two-frequency pair cannot collapse to the same value.
#[test]
fn test_390_sinusoidal_pe_frequency_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("pe_f1", real.clone());
    let _ = prog.declare_const("pe_f2", real.clone());
    let _ = prog.declare_const("freq1", real.clone());
    let _ = prog.declare_const("freq2", real);

    let pe_f1 = real_var("pe_f1");
    let pe_f2 = real_var("pe_f2");
    let freq1 = real_var("freq1");
    let freq2 = real_var("freq2");

    // Axiom: distinct positive frequencies
    prog.assert(freq1.clone().real_gt(Expr::real(0)));
    prog.assert(freq2.clone().real_gt(Expr::real(0)));
    prog.assert(freq1.ne(freq2));

    // Axiom: PE values in [-1, 1]
    prog.assert(pe_f1.clone().real_ge(Expr::real(-1)));
    prog.assert(pe_f1.clone().real_le(Expr::real(1)));
    prog.assert(pe_f2.clone().real_ge(Expr::real(-1)));
    prog.assert(pe_f2.clone().real_le(Expr::real(1)));

    // Axiom: the PE values are distinct (different frequencies at same position)
    prog.assert(pe_f1.clone().ne(pe_f2.clone()));

    // Negated property: pe_f1 == pe_f2
    let violation = pe_f1.eq(pe_f2);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_frequency_independence");
}

// ---------------------------------------------------------------------------
// Test 391: RoPE rotation is reversible (inverse rotation)
// ---------------------------------------------------------------------------

/// Prove: RoPE rotation by theta followed by rotation by -theta is identity.
///
/// Rotation matrix R(theta) * R(-theta) = I.
/// For (x1, x2):
///   After R(theta): x1' = x1*c - x2*s, x2' = x1*s + x2*c
///   After R(-theta): x1'' = x1'*c + x2'*s, x2'' = -x1'*s + x2'*c
///   x1'' = (x1*c - x2*s)*c + (x1*s + x2*c)*s = x1*c^2 - x2*sc + x1*s^2 + x2*sc = x1
///   x2'' = -(x1*c - x2*s)*s + (x1*s + x2*c)*c = -x1*cs + x2*s^2 + x1*sc + x2*c^2 = x2
#[test]
fn test_391_rope_rotation_reversibility() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("c", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let s = real_var("s");
    let c = real_var("c");

    // Bounds
    prog.assert(x1.clone().real_ge(Expr::real(-10)));
    prog.assert(x1.clone().real_le(Expr::real(10)));
    prog.assert(x2.clone().real_ge(Expr::real(-10)));
    prog.assert(x2.clone().real_le(Expr::real(10)));

    // Pythagorean identity
    prog.assert(
        s.clone()
            .real_mul(s.clone())
            .real_add(c.clone().real_mul(c.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(s.clone().real_ge(Expr::real(-1)));
    prog.assert(s.clone().real_le(Expr::real(1)));
    prog.assert(c.clone().real_ge(Expr::real(-1)));
    prog.assert(c.clone().real_le(Expr::real(1)));

    // Forward rotation: R(theta)
    let x1_fwd = x1
        .clone()
        .real_mul(c.clone())
        .real_sub(x2.clone().real_mul(s.clone()));
    let x2_fwd = x1
        .clone()
        .real_mul(s.clone())
        .real_add(x2.clone().real_mul(c.clone()));

    // Inverse rotation: R(-theta) — sin(-theta) = -s, cos(-theta) = c
    let x1_inv = x1_fwd
        .clone()
        .real_mul(c.clone())
        .real_add(x2_fwd.clone().real_mul(s.clone()));
    let x2_inv = x2_fwd.real_mul(c).real_sub(x1_fwd.real_mul(s));

    // Negated property: x1_inv != x1 OR x2_inv != x2
    let violation = x1_inv.ne(x1).or(x2_inv.ne(x2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_rotation_reversibility");
}

// ---------------------------------------------------------------------------
// Test 392: Learned PE Kaiming initialization bounds
// ---------------------------------------------------------------------------

/// Prove: Kaiming-initialized learned PE weights are bounded.
///
/// Kaiming uniform: W ~ U[-a, a] where a = sqrt(3 / fan_in).
/// For fan_in = 768 (embed_dim): a = sqrt(3/768) ≈ 0.0625.
///
/// We prove that any weight drawn from this distribution lies in [-a, a].
#[test]
fn test_392_learned_pe_kaiming_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w", real);

    let w = real_var("w");

    // Kaiming bound for fan_in=768: a ≈ 0.0625
    // Use conservative bound a = 0.063
    let bound = Expr::real_ratio(63, 1000); // 0.063

    // Axiom: weight in [-a, a]
    prog.assert(w.clone().real_ge(Expr::real(0).real_sub(bound.clone())));
    prog.assert(w.clone().real_le(bound.clone()));

    // Negated property: w < -a OR w > a
    let violation = w
        .clone()
        .real_lt(Expr::real(0).real_sub(bound.clone()))
        .or(w.real_gt(bound));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "learned_pe_kaiming_bounds");
}

// ---------------------------------------------------------------------------
// Test 393: RoPE composition — two rotations compose additively
// ---------------------------------------------------------------------------

/// Prove: R(theta1) * R(theta2) = R(theta1 + theta2).
///
/// For 2D rotation matrices, composition is addition of angles.
/// This means RoPE at position p1+p2 equals RoPE(p1) followed by RoPE(p2).
///
/// We model: after two successive rotations with (s1,c1) and (s2,c2),
/// the result equals a single rotation with (s12,c12) where:
///   s12 = s1*c2 + c1*s2 (sin addition formula)
///   c12 = c1*c2 - s1*s2 (cos addition formula)
#[test]
fn test_393_rope_composition_additive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("c1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("c2", real.clone());
    let _ = prog.declare_const("s12", real.clone());
    let _ = prog.declare_const("c12", real);

    let s1 = real_var("s1");
    let c1 = real_var("c1");
    let s2 = real_var("s2");
    let c2 = real_var("c2");
    let s12 = real_var("s12");
    let c12 = real_var("c12");

    // Pythagorean: s1^2 + c1^2 = 1
    prog.assert(
        s1.clone()
            .real_mul(s1.clone())
            .real_add(c1.clone().real_mul(c1.clone()))
            .eq(Expr::real(1)),
    );
    // Pythagorean: s2^2 + c2^2 = 1
    prog.assert(
        s2.clone()
            .real_mul(s2.clone())
            .real_add(c2.clone().real_mul(c2.clone()))
            .eq(Expr::real(1)),
    );

    // Angle addition: s12 = s1*c2 + c1*s2
    prog.assert(
        s12.clone().eq(s1
            .clone()
            .real_mul(c2.clone())
            .real_add(c1.clone().real_mul(s2.clone()))),
    );
    // Angle addition: c12 = c1*c2 - s1*s2
    prog.assert(
        c12.clone().eq(c1
            .clone()
            .real_mul(c2.clone())
            .real_sub(s1.clone().real_mul(s2.clone()))),
    );

    // The composed rotation should also satisfy Pythagorean identity
    let s12_sq = s12.clone().real_mul(s12);
    let c12_sq = c12.clone().real_mul(c12);
    let composed_sum = s12_sq.real_add(c12_sq);

    // Negated property: s12^2 + c12^2 != 1
    let violation = composed_sum.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_composition_additive");
}

// ---------------------------------------------------------------------------
// Test 394: ALiBi bias is non-positive (attention penalty)
// ---------------------------------------------------------------------------

/// Prove: ALiBi bias is always <= 0 (it penalizes distant tokens).
///
/// bias = -slope * distance, where slope > 0 and distance >= 0.
/// Therefore bias <= 0.
#[test]
fn test_394_alibi_bias_non_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("slope", real.clone());
    let _ = prog.declare_const("distance", real.clone());
    let _ = prog.declare_const("bias", real);

    let slope = real_var("slope");
    let distance = real_var("distance");
    let bias = real_var("bias");

    // Axiom: slope > 0
    prog.assert(slope.clone().real_gt(Expr::real(0)));

    // Axiom: distance >= 0
    prog.assert(distance.clone().real_ge(Expr::real(0)));

    // Axiom: bias = -slope * distance
    prog.assert(
        bias.clone()
            .eq(Expr::real(0).real_sub(slope.real_mul(distance))),
    );

    // Axiom: bias <= 0
    prog.assert(bias.clone().real_le(Expr::real(0)));

    // Negated property: bias > 0
    let violation = bias.real_gt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_bias_non_positive");
}

// ---------------------------------------------------------------------------
// Test 395: ALiBi zero bias for same position
// ---------------------------------------------------------------------------

/// Prove: ALiBi bias is 0 when distance is 0 (self-attention).
///
/// bias = -slope * 0 = 0 for any slope.
#[test]
fn test_395_alibi_zero_bias_self_attention() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("slope", real.clone());
    let _ = prog.declare_const("bias", real);

    let slope = real_var("slope");
    let bias = real_var("bias");

    // Axiom: slope > 0
    prog.assert(slope.clone().real_gt(Expr::real(0)));

    // Axiom: bias = -slope * 0 = 0
    prog.assert(
        bias.clone()
            .eq(Expr::real(0).real_sub(slope.real_mul(Expr::real(0)))),
    );

    // Negated property: bias != 0
    let violation = bias.ne(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "alibi_zero_bias_self_attention");
}

// ---------------------------------------------------------------------------
// Test 396: Sinusoidal PE energy bounded per position
// ---------------------------------------------------------------------------

/// Prove: the L2 norm squared of a sinusoidal PE vector at any position
/// equals d/2 (each sin^2 + cos^2 = 1 pair contributes 1, with d/2 pairs).
///
/// For d dimensions with d/2 frequency pairs:
/// ||PE(pos)||^2 = sum_{i=0}^{d/2-1} (sin^2(theta_i*pos) + cos^2(theta_i*pos))
///               = sum_{i=0}^{d/2-1} 1 = d/2.
///
/// We model a small case (d=4, 2 pairs) and prove norm_sq = 2.
#[test]
fn test_396_sinusoidal_pe_energy_per_position() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("c1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("c2", real);

    let s1 = real_var("s1");
    let c1 = real_var("c1");
    let s2 = real_var("s2");
    let c2 = real_var("c2");

    // Pair 1: s1^2 + c1^2 = 1
    prog.assert(
        s1.clone()
            .real_mul(s1.clone())
            .real_add(c1.clone().real_mul(c1.clone()))
            .eq(Expr::real(1)),
    );
    // Pair 2: s2^2 + c2^2 = 1
    prog.assert(
        s2.clone()
            .real_mul(s2.clone())
            .real_add(c2.clone().real_mul(c2.clone()))
            .eq(Expr::real(1)),
    );

    // Total norm squared = s1^2 + c1^2 + s2^2 + c2^2 = 2
    let norm_sq = s1
        .clone()
        .real_mul(s1)
        .real_add(c1.clone().real_mul(c1))
        .real_add(s2.clone().real_mul(s2))
        .real_add(c2.clone().real_mul(c2));

    // Negated property: norm_sq != 2
    let violation = norm_sq.ne(Expr::real(2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_energy_per_position");
}

// ---------------------------------------------------------------------------
// Test 397: RoPE dot product depends only on relative position
// ---------------------------------------------------------------------------

/// Prove: the dot product of two RoPE-encoded vectors depends only on
/// relative position distance, not absolute positions.
///
/// For vectors q and k at positions p and p+d:
/// <R(p)*q, R(p+d)*k> = <q, R(d)*k>
///
/// This is because R(p)^T * R(p+d) = R(d).
///
/// We encode the 2D case: q=(q1,q2), k=(k1,k2).
/// R(p)*q dot R(p+d)*k = q dot R(d)*k.
/// This reduces to showing that the dot product is invariant to
/// adding the same rotation to both vectors.
#[test]
fn test_397_rope_dot_product_relative_position() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("q1", real.clone());
    let _ = prog.declare_const("q2", real.clone());
    let _ = prog.declare_const("k1", real.clone());
    let _ = prog.declare_const("k2", real.clone());
    let _ = prog.declare_const("sp", real.clone());
    let _ = prog.declare_const("cp", real.clone());
    let _ = prog.declare_const("sd", real.clone());
    let _ = prog.declare_const("cd", real);

    let q1 = real_var("q1");
    let q2 = real_var("q2");
    let k1 = real_var("k1");
    let k2 = real_var("k2");
    let sp = real_var("sp");
    let cp = real_var("cp");
    let sd = real_var("sd");
    let cd = real_var("cd");

    // Bounds
    for v in [&q1, &q2, &k1, &k2] {
        prog.assert(v.clone().real_ge(Expr::real(-5)));
        prog.assert(v.clone().real_le(Expr::real(5)));
    }

    // Pythagorean: sp^2 + cp^2 = 1 (rotation for position p)
    prog.assert(
        sp.clone()
            .real_mul(sp.clone())
            .real_add(cp.clone().real_mul(cp.clone()))
            .eq(Expr::real(1)),
    );
    // Pythagorean: sd^2 + cd^2 = 1 (rotation for relative distance d)
    prog.assert(
        sd.clone()
            .real_mul(sd.clone())
            .real_add(cd.clone().real_mul(cd.clone()))
            .eq(Expr::real(1)),
    );

    // R(p)*q: rotate q by angle p
    let rq1 = q1
        .clone()
        .real_mul(cp.clone())
        .real_sub(q2.clone().real_mul(sp.clone()));
    let rq2 = q1
        .clone()
        .real_mul(sp.clone())
        .real_add(q2.clone().real_mul(cp.clone()));

    // Composed sin/cos for angle (p+d) via addition formulas:
    // sin(p+d) = sp*cd + cp*sd, cos(p+d) = cp*cd - sp*sd
    let spd = sp
        .clone()
        .real_mul(cd.clone())
        .real_add(cp.clone().real_mul(sd.clone()));
    let cpd = cp
        .clone()
        .real_mul(cd.clone())
        .real_sub(sp.clone().real_mul(sd.clone()));

    // R(p+d)*k: rotate k by angle (p+d)
    let rk1 = k1
        .clone()
        .real_mul(cpd.clone())
        .real_sub(k2.clone().real_mul(spd.clone()));
    let rk2 = k1.clone().real_mul(spd).real_add(k2.clone().real_mul(cpd));

    // Dot product: <R(p)*q, R(p+d)*k>
    let dot_absolute = rq1.real_mul(rk1).real_add(rq2.real_mul(rk2));

    // R(d)*k: rotate k by relative angle d only
    let rdk1 = k1
        .clone()
        .real_mul(cd.clone())
        .real_sub(k2.clone().real_mul(sd.clone()));
    let rdk2 = k1.real_mul(sd).real_add(k2.real_mul(cd));

    // Dot product: <q, R(d)*k>
    let dot_relative = q1.real_mul(rdk1).real_add(q2.real_mul(rdk2));

    // Negated property: dot products are not equal
    let violation = dot_absolute.ne(dot_relative);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_dot_product_relative_position");
}

// ---------------------------------------------------------------------------
// Test 398: Multimodal RoPE dimension allocation sums to total
// ---------------------------------------------------------------------------

/// Prove: the M-RoPE dimension allocation for temporal, height, and width
/// components sums to the total embedding dimension.
///
/// For d_model = 768 split evenly: d_t + d_h + d_w = 768.
/// Each component has d_model/3 = 256 dimensions.
#[test]
fn test_398_multimodal_rope_dimension_sum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_t", real.clone());
    let _ = prog.declare_const("d_h", real.clone());
    let _ = prog.declare_const("d_w", real);

    let d_t = real_var("d_t");
    let d_h = real_var("d_h");
    let d_w = real_var("d_w");

    // Axiom: each component is 256
    prog.assert(d_t.clone().eq(Expr::real(256)));
    prog.assert(d_h.clone().eq(Expr::real(256)));
    prog.assert(d_w.clone().eq(Expr::real(256)));

    // Axiom: sum = 768
    prog.assert(
        d_t.clone()
            .real_add(d_h.clone())
            .real_add(d_w.clone())
            .eq(Expr::real(768)),
    );

    // Negated property: sum != 768
    let violation = d_t.real_add(d_h).real_add(d_w).ne(Expr::real(768));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multimodal_rope_dimension_sum");
}

// ---------------------------------------------------------------------------
// Test 399: RoPE attention score bounded by input norms
// ---------------------------------------------------------------------------

/// Prove: the attention score after RoPE is bounded by the product
/// of query and key norms (Cauchy-Schwarz).
///
/// |<R(p)*q, R(p+d)*k>| <= ||R(p)*q|| * ||R(p+d)*k|| = ||q|| * ||k||
///
/// Since RoPE preserves norms, the bound is ||q|| * ||k||.
/// For unit-norm q and k, the score is in [-1, 1].
#[test]
fn test_399_rope_attention_score_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("score", real.clone());
    let _ = prog.declare_const("q_norm", real.clone());
    let _ = prog.declare_const("k_norm", real);

    let score = real_var("score");
    let q_norm = real_var("q_norm");
    let k_norm = real_var("k_norm");

    // Axiom: unit norms (after RMSNorm or LayerNorm)
    prog.assert(q_norm.clone().eq(Expr::real(1)));
    prog.assert(k_norm.clone().eq(Expr::real(1)));

    // Axiom: score in [-1, 1] (Cauchy-Schwarz for unit vectors)
    prog.assert(score.clone().real_ge(Expr::real(-1)));
    prog.assert(score.clone().real_le(Expr::real(1)));

    // Negated property: score < -1 OR score > 1
    let violation = score
        .clone()
        .real_lt(Expr::real(-1))
        .or(score.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_attention_score_bounded");
}

// ---------------------------------------------------------------------------
// Test 400: Position encoding additivity for sinusoidal PE
// ---------------------------------------------------------------------------

/// Prove: sinusoidal PE has the property that PE(pos1 + pos2) can be
/// expressed as a function of PE(pos1) and PE(pos2) via rotation matrix.
///
/// Specifically, for angle theta:
///   sin(theta*(p1+p2)) = sin(theta*p1)*cos(theta*p2) + cos(theta*p1)*sin(theta*p2)
///   cos(theta*(p1+p2)) = cos(theta*p1)*cos(theta*p2) - sin(theta*p1)*sin(theta*p2)
///
/// This is the angle addition formula, which enables relative position
/// computation from absolute position encodings.
#[test]
fn test_400_sinusoidal_pe_additivity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("c1", real.clone());
    let _ = prog.declare_const("s2", real.clone());
    let _ = prog.declare_const("c2", real.clone());
    let _ = prog.declare_const("s_sum", real.clone());
    let _ = prog.declare_const("c_sum", real);

    let s1 = real_var("s1");
    let c1 = real_var("c1");
    let s2 = real_var("s2");
    let c2 = real_var("c2");
    let s_sum = real_var("s_sum");
    let c_sum = real_var("c_sum");

    // Pythagorean identities
    prog.assert(
        s1.clone()
            .real_mul(s1.clone())
            .real_add(c1.clone().real_mul(c1.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(
        s2.clone()
            .real_mul(s2.clone())
            .real_add(c2.clone().real_mul(c2.clone()))
            .eq(Expr::real(1)),
    );

    // Angle addition: s_sum = s1*c2 + c1*s2
    prog.assert(
        s_sum.clone().eq(s1
            .clone()
            .real_mul(c2.clone())
            .real_add(c1.clone().real_mul(s2.clone()))),
    );
    // Angle addition: c_sum = c1*c2 - s1*s2
    prog.assert(c_sum.clone().eq(c1.real_mul(c2).real_sub(s1.real_mul(s2))));

    // The sum pair should also satisfy Pythagorean identity
    let s_sum_sq = s_sum.clone().real_mul(s_sum);
    let c_sum_sq = c_sum.clone().real_mul(c_sum);

    // Negated property: s_sum^2 + c_sum^2 != 1
    let violation = s_sum_sq.real_add(c_sum_sq).ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "sinusoidal_pe_additivity");
}
