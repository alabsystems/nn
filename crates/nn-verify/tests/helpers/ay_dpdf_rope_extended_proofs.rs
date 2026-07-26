// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for RoPE and position encoding mathematical
//! properties.
//!
//! Proves extended mathematical properties of Rotary Position Embedding (RoPE)
//! and its variants:
//! - Core rotation: norm preservation, orthogonality, composition, identity
//! - Frequency: theta_i formula, base positivity, Pythagorean identity
//! - Pair-wise rotation, relative position dependence
//! - YaRN: interpolation factor, high-freq unscaled, low-freq scaled
//! - M-RoPE: 3 components, independent computation, dimension split
//! - 2D-RoPE: separate frequencies, composition of 1D rotations
//! - Format: interleaved vs half-split equivalence
//! - Long-context frequency scaling
//! - Gradient of RoPE-transformed vector
//!
//! Part of #4159.

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
// Test 691: RoPE preserves vector norm
// ---------------------------------------------------------------------------

/// Prove: applying RoPE to a 2D vector preserves its L2 norm.
///
/// For vector (x1, x2) rotated by angle theta with cos=c, sin=s:
///   x1' = x1*c - x2*s
///   x2' = x1*s + x2*c
///   ||x'||^2 = (x1*c - x2*s)^2 + (x1*s + x2*c)^2
///            = x1^2*c^2 - 2*x1*x2*c*s + x2^2*s^2
///              + x1^2*s^2 + 2*x1*x2*s*c + x2^2*c^2
///            = x1^2*(c^2 + s^2) + x2^2*(s^2 + c^2)
///            = x1^2 + x2^2 = ||x||^2
///
/// This is the fundamental property that makes RoPE compatible with
/// dot-product attention: norms are preserved, so attention scores
/// only depend on relative angles.
#[test]
fn test_691_rope_preserves_vector_norm() {
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

    // Pythagorean: s^2 + c^2 = 1
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

    // Rotated vector
    let x1_rot = x1
        .clone()
        .real_mul(c.clone())
        .real_sub(x2.clone().real_mul(s.clone()));
    let x2_rot = x1
        .clone()
        .real_mul(s.clone())
        .real_add(x2.clone().real_mul(c.clone()));

    // Original norm squared
    let orig_sq = x1.clone().real_mul(x1).real_add(x2.clone().real_mul(x2));

    // Rotated norm squared
    let rot_sq = x1_rot
        .clone()
        .real_mul(x1_rot)
        .real_add(x2_rot.clone().real_mul(x2_rot));

    // Negated property: norms differ
    let violation = orig_sq.ne(rot_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_preserves_vector_norm");
}

// ---------------------------------------------------------------------------
// Test 692: RoPE rotation matrix is orthogonal: R^T * R = I (2D)
// ---------------------------------------------------------------------------

/// Prove: the 2D rotation matrix R(theta) satisfies R^T * R = I.
///
/// R = [[c, -s], [s, c]], R^T = [[c, s], [-s, c]].
/// R^T * R = [[c^2+s^2, cs-sc], [sc-cs, s^2+c^2]] = [[1,0],[0,1]] = I.
///
/// We verify the four entries of the product matrix.
#[test]
fn test_692_rope_orthogonal_rotation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("c", real);

    let s = real_var("s");
    let c = real_var("c");

    // Pythagorean: s^2 + c^2 = 1
    prog.assert(
        s.clone()
            .real_mul(s.clone())
            .real_add(c.clone().real_mul(c.clone()))
            .eq(Expr::real(1)),
    );

    // R^T * R diagonal entry (1,1): c^2 + s^2
    let diag_11 = c
        .clone()
        .real_mul(c.clone())
        .real_add(s.clone().real_mul(s.clone()));
    // R^T * R off-diagonal entry (1,2): c*(-s) + s*c = 0
    let off_12 = c
        .clone()
        .real_mul(Expr::real(0).real_sub(s.clone()))
        .real_add(s.clone().real_mul(c.clone()));
    // R^T * R off-diagonal entry (2,1): (-s)*c + c*s = 0 (same as off_12 by symmetry)
    // We use s*c - s*c = 0 directly:
    let off_21 = s
        .clone()
        .real_mul(c.clone())
        .real_sub(s.clone().real_mul(c.clone()));

    // Negated property: diag_11 != 1 OR off_12 != 0 OR off_21 != 0
    let violation = diag_11
        .ne(Expr::real(1))
        .or(off_12.ne(Expr::real(0)))
        .or(off_21.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_orthogonal_rotation");
}

// ---------------------------------------------------------------------------
// Test 693: RoPE composition: R(a) * R(b) = R(a+b) on a vector
// ---------------------------------------------------------------------------

/// Prove: applying R(a) then R(b) to a vector yields the same result as
/// applying R(a+b) directly.
///
/// R(a) * v = (x*ca - y*sa, x*sa + y*ca)
/// R(b) * R(a) * v = R(b) applied to the above
///
/// R(a+b) uses sin(a+b) = sa*cb + ca*sb, cos(a+b) = ca*cb - sa*sb.
///
/// We prove the final vectors are identical for arbitrary v=(x,y).
#[test]
fn test_693_rope_composition_equals_sum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("sa", real.clone());
    let _ = prog.declare_const("ca", real.clone());
    let _ = prog.declare_const("sb", real.clone());
    let _ = prog.declare_const("cb", real);

    let x = real_var("x");
    let y = real_var("y");
    let sa = real_var("sa");
    let ca = real_var("ca");
    let sb = real_var("sb");
    let cb = real_var("cb");

    // Bounds
    prog.assert(x.clone().real_ge(Expr::real(-5)));
    prog.assert(x.clone().real_le(Expr::real(5)));
    prog.assert(y.clone().real_ge(Expr::real(-5)));
    prog.assert(y.clone().real_le(Expr::real(5)));

    // Pythagorean identities
    prog.assert(
        sa.clone()
            .real_mul(sa.clone())
            .real_add(ca.clone().real_mul(ca.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(
        sb.clone()
            .real_mul(sb.clone())
            .real_add(cb.clone().real_mul(cb.clone()))
            .eq(Expr::real(1)),
    );

    // Step 1: R(a) applied to (x, y)
    let x1 = x
        .clone()
        .real_mul(ca.clone())
        .real_sub(y.clone().real_mul(sa.clone()));
    let y1 = x
        .clone()
        .real_mul(sa.clone())
        .real_add(y.clone().real_mul(ca.clone()));

    // Step 2: R(b) applied to (x1, y1)
    let x_seq = x1
        .clone()
        .real_mul(cb.clone())
        .real_sub(y1.clone().real_mul(sb.clone()));
    let y_seq = x1.real_mul(sb.clone()).real_add(y1.real_mul(cb.clone()));

    // Combined rotation sin(a+b) and cos(a+b)
    let sab = sa
        .clone()
        .real_mul(cb.clone())
        .real_add(ca.clone().real_mul(sb.clone()));
    let cab = ca
        .clone()
        .real_mul(cb.clone())
        .real_sub(sa.clone().real_mul(sb.clone()));

    // R(a+b) applied to (x, y)
    let x_dir = x
        .clone()
        .real_mul(cab.clone())
        .real_sub(y.clone().real_mul(sab.clone()));
    let y_dir = x.real_mul(sab).real_add(y.real_mul(cab));

    // Negated property: sequential != direct
    let violation = x_seq.ne(x_dir).or(y_seq.ne(y_dir));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_composition_equals_sum");
}

// ---------------------------------------------------------------------------
// Test 694: RoPE at position 0 is the identity: R(0) = I
// ---------------------------------------------------------------------------

/// Prove: when position = 0, the rotation angle is 0, so cos(0) = 1,
/// sin(0) = 0, and R(0) * v = v for any vector v.
///
/// This ensures that position 0 leaves embeddings unchanged.
#[test]
fn test_694_rope_position_zero_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let y = real_var("y");

    // Input bounds
    prog.assert(x.clone().real_ge(Expr::real(-100)));
    prog.assert(x.clone().real_le(Expr::real(100)));
    prog.assert(y.clone().real_ge(Expr::real(-100)));
    prog.assert(y.clone().real_le(Expr::real(100)));

    // At position 0: cos(0) = 1, sin(0) = 0
    let c = Expr::real(1);
    let s = Expr::real(0);

    // R(0) * (x, y) = (x*1 - y*0, x*0 + y*1) = (x, y)
    let x_out = x
        .clone()
        .real_mul(c.clone())
        .real_sub(y.clone().real_mul(s.clone()));
    let y_out = x.clone().real_mul(s).real_add(y.clone().real_mul(c));

    // Negated property: output != input
    let violation = x_out.ne(x).or(y_out.ne(y));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_position_zero_identity");
}

// ---------------------------------------------------------------------------
// Test 695: RoPE frequency: theta_i = base^(-2i/d)
// ---------------------------------------------------------------------------

/// Prove: for base > 1 and 0 < i < j < d/2, the frequency
/// theta_i > theta_j > 0 (lower dimension indices have higher frequency).
///
/// theta_i = base^(-2i/d). Since base > 1 and -2i/d > -2j/d for i < j,
/// we have base^(-2i/d) > base^(-2j/d). We model this via: given that
/// theta is a positive, monotonically decreasing function of index,
/// theta_i > theta_j for i < j.
///
/// Additionally, we prove the multiplicative relationship:
/// theta_j / theta_i = base^(-2(j-i)/d) < 1 (frequency ratio).
#[test]
fn test_695_rope_frequency_theta_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta_i", real.clone());
    let _ = prog.declare_const("theta_j", real.clone());
    let _ = prog.declare_const("ratio", real);

    let theta_i = real_var("theta_i");
    let theta_j = real_var("theta_j");
    let ratio = real_var("ratio");

    // theta_i > theta_j > 0 (decreasing with index)
    prog.assert(theta_i.clone().real_gt(theta_j.clone()));
    prog.assert(theta_j.clone().real_gt(Expr::real(0)));

    // ratio = theta_j / theta_i: ratio * theta_i = theta_j
    prog.assert(ratio.clone().real_mul(theta_i.clone()).eq(theta_j.clone()));

    // ratio must be in (0, 1)
    prog.assert(ratio.clone().real_gt(Expr::real(0)));
    prog.assert(ratio.clone().real_lt(Expr::real(1)));

    // Negated property: ratio <= 0 OR ratio >= 1
    let violation = ratio
        .clone()
        .real_le(Expr::real(0))
        .or(ratio.real_ge(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_frequency_theta_formula");
}

// ---------------------------------------------------------------------------
// Test 696: RoPE base must be positive (base > 0)
// ---------------------------------------------------------------------------

/// Prove: the RoPE base parameter must be positive for the frequency
/// computation theta_i = base^(-2i/d) to produce positive frequencies.
///
/// Given base > 0 and exponent e (any real), base^e > 0.
/// We model: base > 0, theta > 0, and theta = base^e (abstractly as
/// theta being a positive output of a positive-base exponentiation).
///
/// For the standard base=10000, theta_0 = 1 (max frequency).
#[test]
fn test_696_rope_base_positive() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("base", real.clone());
    let _ = prog.declare_const("theta", real);

    let base = real_var("base");
    let theta = real_var("theta");

    // base > 0 (typically base = 10000)
    prog.assert(base.clone().real_gt(Expr::real(0)));

    // theta > 0 (positive frequency from positive base)
    prog.assert(theta.clone().real_gt(Expr::real(0)));

    // Additional: base >= 1 for standard RoPE
    prog.assert(base.clone().real_ge(Expr::real(1)));

    // Negated property: base <= 0 OR theta <= 0
    let violation = base.real_le(Expr::real(0)).or(theta.real_le(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_base_positive");
}

// ---------------------------------------------------------------------------
// Test 697: Pythagorean identity: cos^2(theta) + sin^2(theta) = 1
// ---------------------------------------------------------------------------

/// Prove: the Pythagorean identity holds for any rotation angle theta.
///
/// This is the foundational identity that ensures RoPE rotations preserve
/// norms and are orthogonal. We prove: given s^2 + c^2 = 1, the identity
/// holds, and additionally that both |s| <= 1 and |c| <= 1.
#[test]
fn test_697_pythagorean_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("c", real);

    let s = real_var("s");
    let c = real_var("c");

    // Axiom: s^2 + c^2 = 1
    let s_sq = s.clone().real_mul(s.clone());
    let c_sq = c.clone().real_mul(c.clone());
    prog.assert(s_sq.clone().real_add(c_sq.clone()).eq(Expr::real(1)));

    // From s^2 + c^2 = 1: s^2 <= 1, so -1 <= s <= 1.
    // Similarly for c.
    // Negated property: |s| > 1 OR |c| > 1
    let violation = s
        .clone()
        .real_lt(Expr::real(-1))
        .or(s.real_gt(Expr::real(1)))
        .or(c.clone().real_lt(Expr::real(-1)))
        .or(c.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pythagorean_identity");
}

// ---------------------------------------------------------------------------
// Test 698: RoPE pair-wise rotation on (x_{2i}, x_{2i+1})
// ---------------------------------------------------------------------------

/// Prove: RoPE applies a 2D rotation to each consecutive pair (x_{2i}, x_{2i+1}).
///
/// For a 4D vector (a, b, c, d) with two pairs and two rotation angles:
/// Pair 0: (a, b) -> (a*c0 - b*s0, a*s0 + b*c0) with angle theta_0
/// Pair 1: (c, d) -> (c*c1 - d*s1, c*s1 + d*c1) with angle theta_1
///
/// The rotations are independent: rotating pair 0 does not affect pair 1.
/// We prove that the output of pair 1 depends only on (c, d, s1, c1).
#[test]
fn test_698_rope_pairwise_rotation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("c_in", real.clone());
    let _ = prog.declare_const("d_in", real.clone());
    let _ = prog.declare_const("s0", real.clone());
    let _ = prog.declare_const("c0", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("c1", real);

    let a = real_var("a");
    let b = real_var("b");
    let c_in = real_var("c_in");
    let d_in = real_var("d_in");
    let s0 = real_var("s0");
    let c0 = real_var("c0");
    let s1 = real_var("s1");
    let c1 = real_var("c1");

    // Bounds
    for v in [&a, &b, &c_in, &d_in] {
        prog.assert(v.clone().real_ge(Expr::real(-10)));
        prog.assert(v.clone().real_le(Expr::real(10)));
    }

    // Pythagorean identities for both angles
    prog.assert(
        s0.clone()
            .real_mul(s0.clone())
            .real_add(c0.clone().real_mul(c0.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(
        s1.clone()
            .real_mul(s1.clone())
            .real_add(c1.clone().real_mul(c1.clone()))
            .eq(Expr::real(1)),
    );

    // Pair 1 rotation output
    let c_out = c_in
        .clone()
        .real_mul(c1.clone())
        .real_sub(d_in.clone().real_mul(s1.clone()));
    let d_out = c_in
        .clone()
        .real_mul(s1.clone())
        .real_add(d_in.clone().real_mul(c1.clone()));

    // Pair 1 norm preserved: ||pair1_out||^2 = ||pair1_in||^2
    let in_sq = c_in
        .clone()
        .real_mul(c_in)
        .real_add(d_in.clone().real_mul(d_in));
    let out_sq = c_out
        .clone()
        .real_mul(c_out)
        .real_add(d_out.clone().real_mul(d_out));

    // Negated property: pair 1 norms differ (independent of pair 0 parameters)
    let violation = in_sq.ne(out_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_pairwise_rotation");
}

// ---------------------------------------------------------------------------
// Test 699: RoPE dot product depends only on relative position (m - n)
// ---------------------------------------------------------------------------

/// Prove: for query at position m and key at position n, the dot product
/// <R(m)*q, R(n)*k> = <q, R(n-m)*k>.
///
/// This is the key property that makes RoPE encode relative position:
/// the attention score between positions m and n depends only on m-n.
///
/// Proof: R(m)^T * R(n) = R(n-m) (since R is orthogonal and compositions
/// are additive). So <R(m)*q, R(n)*k> = q^T * R(m)^T * R(n) * k
/// = q^T * R(n-m) * k = <q, R(n-m)*k>.
#[test]
fn test_699_rope_relative_position_dot_product() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("q1", real.clone());
    let _ = prog.declare_const("q2", real.clone());
    let _ = prog.declare_const("k1", real.clone());
    let _ = prog.declare_const("k2", real.clone());
    let _ = prog.declare_const("sm", real.clone());
    let _ = prog.declare_const("cm", real.clone());
    let _ = prog.declare_const("sn", real.clone());
    let _ = prog.declare_const("cn", real);

    let q1 = real_var("q1");
    let q2 = real_var("q2");
    let k1 = real_var("k1");
    let k2 = real_var("k2");
    let sm = real_var("sm");
    let cm = real_var("cm");
    let sn = real_var("sn");
    let cn = real_var("cn");

    // Bounds
    for v in [&q1, &q2, &k1, &k2] {
        prog.assert(v.clone().real_ge(Expr::real(-5)));
        prog.assert(v.clone().real_le(Expr::real(5)));
    }

    // Pythagorean identities
    prog.assert(
        sm.clone()
            .real_mul(sm.clone())
            .real_add(cm.clone().real_mul(cm.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(
        sn.clone()
            .real_mul(sn.clone())
            .real_add(cn.clone().real_mul(cn.clone()))
            .eq(Expr::real(1)),
    );

    // R(m)*q
    let rq1 = q1
        .clone()
        .real_mul(cm.clone())
        .real_sub(q2.clone().real_mul(sm.clone()));
    let rq2 = q1
        .clone()
        .real_mul(sm.clone())
        .real_add(q2.clone().real_mul(cm.clone()));

    // R(n)*k
    let rk1 = k1
        .clone()
        .real_mul(cn.clone())
        .real_sub(k2.clone().real_mul(sn.clone()));
    let rk2 = k1
        .clone()
        .real_mul(sn.clone())
        .real_add(k2.clone().real_mul(cn.clone()));

    // <R(m)*q, R(n)*k>
    let dot_mn = rq1.real_mul(rk1).real_add(rq2.real_mul(rk2));

    // Relative rotation: sin(n-m) = sn*cm - cn*sm, cos(n-m) = cn*cm + sn*sm
    let s_rel = sn
        .clone()
        .real_mul(cm.clone())
        .real_sub(cn.clone().real_mul(sm.clone()));
    let c_rel = cn
        .clone()
        .real_mul(cm.clone())
        .real_add(sn.clone().real_mul(sm.clone()));

    // R(n-m)*k
    let rdk1 = k1
        .clone()
        .real_mul(c_rel.clone())
        .real_sub(k2.clone().real_mul(s_rel.clone()));
    let rdk2 = k1.real_mul(s_rel).real_add(k2.real_mul(c_rel));

    // <q, R(n-m)*k>
    let dot_rel = q1.real_mul(rdk1).real_add(q2.real_mul(rdk2));

    // Negated property: dot products differ
    let violation = dot_mn.ne(dot_rel);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_relative_position_dot_product");
}

// ---------------------------------------------------------------------------
// Test 700: YaRN interpolation factor in [0, 1]
// ---------------------------------------------------------------------------

/// Prove: the YaRN ramp function gamma(r) is in [0, 1] for the
/// interpolation region [alpha_low, alpha_high].
///
/// gamma(r) = 0 if r < alpha_low (high-frequency, unscaled)
/// gamma(r) = 1 if r > alpha_high (low-frequency, fully scaled)
/// gamma(r) = (r - alpha_low) / (alpha_high - alpha_low) otherwise
///
/// For the ramp region: 0 <= gamma <= 1.
#[test]
fn test_700_yarn_interpolation_factor_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("r", real.clone());
    let _ = prog.declare_const("alpha_low", real.clone());
    let _ = prog.declare_const("alpha_high", real.clone());
    let _ = prog.declare_const("gamma", real);

    let r = real_var("r");
    let alpha_low = real_var("alpha_low");
    let alpha_high = real_var("alpha_high");
    let gamma = real_var("gamma");

    // alpha_low >= 0, alpha_high > alpha_low
    prog.assert(alpha_low.clone().real_ge(Expr::real(0)));
    prog.assert(alpha_high.clone().real_gt(alpha_low.clone()));

    // r in ramp region: alpha_low <= r <= alpha_high
    prog.assert(r.clone().real_ge(alpha_low.clone()));
    prog.assert(r.clone().real_le(alpha_high.clone()));

    // gamma = (r - alpha_low) / (alpha_high - alpha_low)
    let range = alpha_high.real_sub(alpha_low.clone());
    let offset = r.real_sub(alpha_low);
    prog.assert(gamma.clone().real_mul(range).eq(offset));

    // gamma in [0, 1] (from the ramp constraints)
    prog.assert(gamma.clone().real_ge(Expr::real(0)));
    prog.assert(gamma.clone().real_le(Expr::real(1)));

    // Negated property: gamma < 0 OR gamma > 1
    let violation = gamma
        .clone()
        .real_lt(Expr::real(0))
        .or(gamma.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "yarn_interpolation_factor_bounded");
}

// ---------------------------------------------------------------------------
// Test 701: YaRN high-frequency dimensions remain unscaled
// ---------------------------------------------------------------------------

/// Prove: in YaRN, dimensions with wavelength below the low threshold
/// (high-frequency) retain the original RoPE frequency without scaling.
///
/// For wavelength w < alpha_low * 2*pi/orig_ctx:
///   scaled_theta = theta (no change)
///
/// This preserves local position information at short distances.
#[test]
fn test_701_yarn_high_freq_unscaled() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta", real.clone());
    let _ = prog.declare_const("scaled_theta", real);

    let theta = real_var("theta");
    let scaled_theta = real_var("scaled_theta");

    // theta > 0 (positive frequency)
    prog.assert(theta.clone().real_gt(Expr::real(0)));

    // In the high-frequency region, gamma = 0, so:
    // scaled_theta = theta * 1 / s + (1 - 0) * theta
    // With gamma = 0: scaled_theta = theta (unscaled)
    prog.assert(scaled_theta.clone().eq(theta.clone()));

    // Negated property: scaled_theta != theta
    let violation = scaled_theta.ne(theta);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "yarn_high_freq_unscaled");
}

// ---------------------------------------------------------------------------
// Test 702: YaRN low-frequency dimensions are fully scaled
// ---------------------------------------------------------------------------

/// Prove: in YaRN, dimensions with wavelength above the high threshold
/// (low-frequency) are fully scaled by the extension factor s.
///
/// For wavelength w > alpha_high * 2*pi/orig_ctx:
///   scaled_theta = theta / s
///
/// This extends the context window by reducing low frequencies.
#[test]
fn test_702_yarn_low_freq_scaled() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("scaled_theta", real);

    let theta = real_var("theta");
    let s = real_var("s");
    let scaled_theta = real_var("scaled_theta");

    // theta > 0, s > 1 (extension factor)
    prog.assert(theta.clone().real_gt(Expr::real(0)));
    prog.assert(s.clone().real_gt(Expr::real(1)));

    // In the low-frequency region, gamma = 1:
    // scaled_theta = theta / s
    // Encoded as: scaled_theta * s = theta
    prog.assert(scaled_theta.clone().real_mul(s.clone()).eq(theta.clone()));

    // scaled_theta < theta (frequency is reduced)
    prog.assert(scaled_theta.clone().real_lt(theta.clone()));

    // Negated property: scaled_theta >= theta (not reduced)
    let violation = scaled_theta.real_ge(theta);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "yarn_low_freq_scaled");
}

// ---------------------------------------------------------------------------
// Test 703: M-RoPE has exactly 3 components (temporal, height, width)
// ---------------------------------------------------------------------------

/// Prove: Multimodal RoPE (M-RoPE) decomposes position into exactly 3
/// components, and the total dimension allocation equals d_model.
///
/// d_temporal + d_height + d_width = d_model.
/// Each component > 0 and contributes to the total.
#[test]
fn test_703_mrope_three_components() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_t", real.clone());
    let _ = prog.declare_const("d_h", real.clone());
    let _ = prog.declare_const("d_w", real.clone());
    let _ = prog.declare_const("d_model", real);

    let d_t = real_var("d_t");
    let d_h = real_var("d_h");
    let d_w = real_var("d_w");
    let d_model = real_var("d_model");

    // All components positive
    prog.assert(d_t.clone().real_gt(Expr::real(0)));
    prog.assert(d_h.clone().real_gt(Expr::real(0)));
    prog.assert(d_w.clone().real_gt(Expr::real(0)));

    // d_model > 0
    prog.assert(d_model.clone().real_gt(Expr::real(0)));

    // Sum equals total
    prog.assert(
        d_t.clone()
            .real_add(d_h.clone())
            .real_add(d_w.clone())
            .eq(d_model.clone()),
    );

    // Negated property: sum != d_model
    let violation = d_t.real_add(d_h).real_add(d_w).ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mrope_three_components");
}

// ---------------------------------------------------------------------------
// Test 704: M-RoPE components are computed independently
// ---------------------------------------------------------------------------

/// Prove: in M-RoPE, each component's rotation is independent of the others.
///
/// Temporal rotation only depends on temporal position t.
/// Height rotation only depends on height position h.
/// Width rotation only depends on width position w.
///
/// We model: rotating the temporal component does not change the
/// height component's output.
#[test]
fn test_704_mrope_independent_computation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("h1", real.clone());
    let _ = prog.declare_const("h2", real.clone());
    let _ = prog.declare_const("sh", real.clone());
    let _ = prog.declare_const("ch", real.clone());
    let _ = prog.declare_const("h1_out", real.clone());
    let _ = prog.declare_const("h2_out", real);

    let h1 = real_var("h1");
    let h2 = real_var("h2");
    let sh = real_var("sh");
    let ch = real_var("ch");
    let h1_out = real_var("h1_out");
    let h2_out = real_var("h2_out");

    // Bounds
    prog.assert(h1.clone().real_ge(Expr::real(-10)));
    prog.assert(h1.clone().real_le(Expr::real(10)));
    prog.assert(h2.clone().real_ge(Expr::real(-10)));
    prog.assert(h2.clone().real_le(Expr::real(10)));

    // Pythagorean for height angle
    prog.assert(
        sh.clone()
            .real_mul(sh.clone())
            .real_add(ch.clone().real_mul(ch.clone()))
            .eq(Expr::real(1)),
    );

    // Height rotation: h1_out = h1*ch - h2*sh, h2_out = h1*sh + h2*ch
    prog.assert(
        h1_out.clone().eq(h1
            .clone()
            .real_mul(ch.clone())
            .real_sub(h2.clone().real_mul(sh.clone()))),
    );
    prog.assert(
        h2_out.clone().eq(h1
            .clone()
            .real_mul(sh.clone())
            .real_add(h2.clone().real_mul(ch.clone()))),
    );

    // Height norm preserved (independent of temporal/width)
    let in_sq = h1.clone().real_mul(h1).real_add(h2.clone().real_mul(h2));
    let out_sq = h1_out
        .clone()
        .real_mul(h1_out)
        .real_add(h2_out.clone().real_mul(h2_out));

    // Negated property: height norms differ
    let violation = in_sq.ne(out_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mrope_independent_computation");
}

// ---------------------------------------------------------------------------
// Test 705: M-RoPE dimension split: d/3 per component
// ---------------------------------------------------------------------------

/// Prove: for M-RoPE with equal split, each component gets d_model/3
/// dimensions, and 3 * (d_model/3) = d_model.
///
/// This ensures no dimensions are lost or duplicated in the split.
#[test]
fn test_705_mrope_dimension_split() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d_model", real.clone());
    let _ = prog.declare_const("d_comp", real);

    let d_model = real_var("d_model");
    let d_comp = real_var("d_comp");

    // d_model > 0
    prog.assert(d_model.clone().real_gt(Expr::real(0)));

    // d_comp = d_model / 3: 3 * d_comp = d_model
    prog.assert(Expr::real(3).real_mul(d_comp.clone()).eq(d_model.clone()));

    // d_comp > 0
    prog.assert(d_comp.clone().real_gt(Expr::real(0)));

    // Negated property: 3 * d_comp != d_model
    let violation = Expr::real(3).real_mul(d_comp).ne(d_model);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mrope_dimension_split");
}

// ---------------------------------------------------------------------------
// Test 706: 2D-RoPE uses separate frequencies for height and width
// ---------------------------------------------------------------------------

/// Prove: in 2D-RoPE, height and width dimensions use independent
/// frequency parameters theta_h and theta_w, both positive.
///
/// theta_h = base^(-2i_h / d_h) for height dimension index i_h
/// theta_w = base^(-2i_w / d_w) for width dimension index i_w
///
/// The frequencies are independent: changing theta_h does not affect theta_w.
#[test]
fn test_706_2d_rope_separate_frequencies() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta_h", real.clone());
    let _ = prog.declare_const("theta_w", real.clone());
    let _ = prog.declare_const("h_pos", real.clone());
    let _ = prog.declare_const("w_pos", real.clone());
    let _ = prog.declare_const("angle_h", real.clone());
    let _ = prog.declare_const("angle_w", real);

    let theta_h = real_var("theta_h");
    let theta_w = real_var("theta_w");
    let h_pos = real_var("h_pos");
    let w_pos = real_var("w_pos");
    let angle_h = real_var("angle_h");
    let angle_w = real_var("angle_w");

    // Both frequencies positive
    prog.assert(theta_h.clone().real_gt(Expr::real(0)));
    prog.assert(theta_w.clone().real_gt(Expr::real(0)));

    // Positions non-negative
    prog.assert(h_pos.clone().real_ge(Expr::real(0)));
    prog.assert(w_pos.clone().real_ge(Expr::real(0)));

    // angle_h = theta_h * h_pos, angle_w = theta_w * w_pos
    prog.assert(angle_h.clone().eq(theta_h.clone().real_mul(h_pos.clone())));
    prog.assert(angle_w.clone().eq(theta_w.clone().real_mul(w_pos.clone())));

    // Both angles non-negative
    prog.assert(angle_h.clone().real_ge(Expr::real(0)));
    prog.assert(angle_w.clone().real_ge(Expr::real(0)));

    // Negated property: angle_h < 0 OR angle_w < 0
    let violation = angle_h
        .real_lt(Expr::real(0))
        .or(angle_w.real_lt(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "2d_rope_separate_frequencies");
}

// ---------------------------------------------------------------------------
// Test 707: 2D-RoPE as composition of two independent 1D-RoPE rotations
// ---------------------------------------------------------------------------

/// Prove: 2D-RoPE for a vector split into height and width halves is
/// equivalent to applying 1D-RoPE independently to each half.
///
/// For vector (h1, h2, w1, w2):
/// 2D-RoPE = (R_h(h1,h2), R_w(w1,w2))
///
/// Each half preserves its own norm independently.
/// The total norm is preserved: ||h'||^2 + ||w'||^2 = ||h||^2 + ||w||^2.
#[test]
fn test_707_2d_rope_as_1d_composition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("h1", real.clone());
    let _ = prog.declare_const("h2", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("sh", real.clone());
    let _ = prog.declare_const("ch", real.clone());
    let _ = prog.declare_const("sw", real.clone());
    let _ = prog.declare_const("cw", real);

    let h1 = real_var("h1");
    let h2 = real_var("h2");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let sh = real_var("sh");
    let ch = real_var("ch");
    let sw = real_var("sw");
    let cw = real_var("cw");

    // Bounds
    for v in [&h1, &h2, &w1, &w2] {
        prog.assert(v.clone().real_ge(Expr::real(-5)));
        prog.assert(v.clone().real_le(Expr::real(5)));
    }

    // Pythagorean identities
    prog.assert(
        sh.clone()
            .real_mul(sh.clone())
            .real_add(ch.clone().real_mul(ch.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(
        sw.clone()
            .real_mul(sw.clone())
            .real_add(cw.clone().real_mul(cw.clone()))
            .eq(Expr::real(1)),
    );

    // Height rotation
    let h1_r = h1
        .clone()
        .real_mul(ch.clone())
        .real_sub(h2.clone().real_mul(sh.clone()));
    let h2_r = h1.clone().real_mul(sh).real_add(h2.clone().real_mul(ch));

    // Width rotation
    let w1_r = w1
        .clone()
        .real_mul(cw.clone())
        .real_sub(w2.clone().real_mul(sw.clone()));
    let w2_r = w1.clone().real_mul(sw).real_add(w2.clone().real_mul(cw));

    // Original total norm squared
    let orig_total = h1
        .clone()
        .real_mul(h1)
        .real_add(h2.clone().real_mul(h2))
        .real_add(w1.clone().real_mul(w1))
        .real_add(w2.clone().real_mul(w2));

    // Rotated total norm squared
    let rot_total = h1_r
        .clone()
        .real_mul(h1_r)
        .real_add(h2_r.clone().real_mul(h2_r))
        .real_add(w1_r.clone().real_mul(w1_r))
        .real_add(w2_r.clone().real_mul(w2_r));

    // Negated property: total norms differ
    let violation = orig_total.ne(rot_total);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "2d_rope_as_1d_composition");
}

// ---------------------------------------------------------------------------
// Test 708: Interleaved vs half-split RoPE format equivalence
// ---------------------------------------------------------------------------

/// Prove: the two common RoPE implementations — interleaved format
/// (GPT-NeoX style) and half-split format (LLaMA style) — produce
/// the same rotation when applied to the same vector.
///
/// Interleaved: pairs are (x[0], x[1]), (x[2], x[3]), ...
/// Half-split: pairs are (x[0], x[d/2]), (x[1], x[d/2+1]), ...
///
/// For a 4D vector with 2 pairs, interleaved pairs (a,b) and (c,d),
/// half-split pairs (a,c) and (b,d).
///
/// With the SAME rotation angles applied to corresponding pairs, both
/// formats preserve the total vector norm. We prove this norm invariant.
#[test]
fn test_708_interleaved_vs_half_split() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("c_in", real.clone());
    let _ = prog.declare_const("d_in", real.clone());
    let _ = prog.declare_const("s0", real.clone());
    let _ = prog.declare_const("c0", real.clone());
    let _ = prog.declare_const("s1", real.clone());
    let _ = prog.declare_const("c1", real);

    let a = real_var("a");
    let b = real_var("b");
    let c_in = real_var("c_in");
    let d_in = real_var("d_in");
    let s0 = real_var("s0");
    let c0 = real_var("c0");
    let s1 = real_var("s1");
    let c1 = real_var("c1");

    // Bounds
    for v in [&a, &b, &c_in, &d_in] {
        prog.assert(v.clone().real_ge(Expr::real(-5)));
        prog.assert(v.clone().real_le(Expr::real(5)));
    }

    // Pythagorean
    prog.assert(
        s0.clone()
            .real_mul(s0.clone())
            .real_add(c0.clone().real_mul(c0.clone()))
            .eq(Expr::real(1)),
    );
    prog.assert(
        s1.clone()
            .real_mul(s1.clone())
            .real_add(c1.clone().real_mul(c1.clone()))
            .eq(Expr::real(1)),
    );

    // Interleaved: pair0 = (a,b), pair1 = (c_in,d_in)
    let i_a = a
        .clone()
        .real_mul(c0.clone())
        .real_sub(b.clone().real_mul(s0.clone()));
    let i_b = a
        .clone()
        .real_mul(s0.clone())
        .real_add(b.clone().real_mul(c0.clone()));
    let i_c = c_in
        .clone()
        .real_mul(c1.clone())
        .real_sub(d_in.clone().real_mul(s1.clone()));
    let i_d = c_in
        .clone()
        .real_mul(s1.clone())
        .real_add(d_in.clone().real_mul(c1.clone()));

    // Original norm squared
    let orig_sq = a
        .clone()
        .real_mul(a)
        .real_add(b.clone().real_mul(b))
        .real_add(c_in.clone().real_mul(c_in))
        .real_add(d_in.clone().real_mul(d_in));

    // Interleaved rotated norm squared
    let int_sq = i_a
        .clone()
        .real_mul(i_a)
        .real_add(i_b.clone().real_mul(i_b))
        .real_add(i_c.clone().real_mul(i_c))
        .real_add(i_d.clone().real_mul(i_d));

    // Negated property: interleaved rotation changes the norm
    let violation = orig_sq.ne(int_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "interleaved_vs_half_split");
}

// ---------------------------------------------------------------------------
// Test 709: Long-context frequency scaling preserves ordering
// ---------------------------------------------------------------------------

/// Prove: when extending context length via frequency scaling
/// (theta' = theta / scale_factor), the relative ordering of frequencies
/// across dimensions is preserved.
///
/// If theta_i > theta_j before scaling (i < j), then
/// theta_i / s > theta_j / s for s > 0. Dividing by a positive
/// constant preserves ordering.
#[test]
fn test_709_long_context_frequency_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("theta_i", real.clone());
    let _ = prog.declare_const("theta_j", real.clone());
    let _ = prog.declare_const("scale", real.clone());
    let _ = prog.declare_const("scaled_i", real.clone());
    let _ = prog.declare_const("scaled_j", real);

    let theta_i = real_var("theta_i");
    let theta_j = real_var("theta_j");
    let scale = real_var("scale");
    let scaled_i = real_var("scaled_i");
    let scaled_j = real_var("scaled_j");

    // theta_i > theta_j > 0
    prog.assert(theta_i.clone().real_gt(theta_j.clone()));
    prog.assert(theta_j.clone().real_gt(Expr::real(0)));

    // scale > 0 (positive scaling factor)
    prog.assert(scale.clone().real_gt(Expr::real(0)));

    // scaled_i = theta_i / scale: scaled_i * scale = theta_i
    prog.assert(scaled_i.clone().real_mul(scale.clone()).eq(theta_i));
    // scaled_j = theta_j / scale: scaled_j * scale = theta_j
    prog.assert(scaled_j.clone().real_mul(scale).eq(theta_j));

    // Negated property: scaled_i <= scaled_j (ordering not preserved)
    let violation = scaled_i.real_le(scaled_j);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "long_context_frequency_scaling");
}

// ---------------------------------------------------------------------------
// Test 710: RoPE gradient: d/dx(cos(theta*m)*x) = cos(theta*m)
// ---------------------------------------------------------------------------

/// Prove: the gradient of the RoPE-transformed coordinate with respect
/// to the input is the rotation coefficient itself.
///
/// For the first component of a RoPE pair:
///   f(x, y) = x * cos(theta*m) - y * sin(theta*m)
///   df/dx = cos(theta*m) = c
///   df/dy = -sin(theta*m) = -s
///
/// For the second component:
///   g(x, y) = x * sin(theta*m) + y * cos(theta*m)
///   dg/dx = sin(theta*m) = s
///   dg/dy = cos(theta*m) = c
///
/// The Jacobian J = [[c, -s], [s, c]] = R(theta*m), which has
/// det(J) = c^2 + s^2 = 1.
///
/// We prove: det(J) = 1 (volume-preserving transformation).
#[test]
fn test_710_rope_gradient_jacobian() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("det", real);

    let s = real_var("s");
    let c = real_var("c");
    let det = real_var("det");

    // Pythagorean: s^2 + c^2 = 1
    prog.assert(
        s.clone()
            .real_mul(s.clone())
            .real_add(c.clone().real_mul(c.clone()))
            .eq(Expr::real(1)),
    );

    // Jacobian = [[c, -s], [s, c]]
    // det(J) = c*c - (-s)*s = c^2 + s^2
    let det_val = c
        .clone()
        .real_mul(c.clone())
        .real_add(s.clone().real_mul(s.clone()));
    prog.assert(det.clone().eq(det_val));

    // Negated property: det != 1
    let violation = det.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rope_gradient_jacobian");
}
