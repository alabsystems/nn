// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for multi-scale feature fusion (FPN, PAN, BiFPN)
//! mathematical properties in dpdf detection models.
//!
//! Proves fundamental properties of Feature Pyramid Networks and variants:
//! - FPN top-down pathway: upsampled + lateral bounded
//! - PAN bottom-up pathway: downsampled + lateral bounded
//! - Bilinear upsample preserves value range
//! - Nearest-neighbor upsample preserves exact values
//! - 1x1 lateral convolution bounded when input bounded
//! - Element-wise addition preserves bound sum
//! - BiFPN weighted fusion: wi/(sum(wi)+eps) in [0,1]
//! - Feature concatenation along channel dim
//! - Multi-scale P3/P4/P5 pyramid consistency
//! - Stride-2 downsample spatial halving
//! - Feature normalization at each pyramid level
//! - Recursive FPN: deeper = coarser + finer bounded
//! - PANet lateral connection symmetry
//! - DetectHead input from multiple FPN levels
//! - Anchor grid generation bounded in image space
//! - Feature stride relationship: P3=8, P4=16, P5=32
//! - Cross-scale attention bounded
//! - Deformable conv sampling points in receptive field
//! - FPN output channel uniformity (all levels same channels)
//! - Skip connection identity preserves bounds exactly
//!
//! Part of #4184.

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
// Test 851: FPN top-down pathway — upsampled + lateral bounded
// ---------------------------------------------------------------------------

/// Prove: In FPN top-down pathway, the sum of an upsampled coarser feature
/// and a lateral connection from finer resolution is bounded when both are.
///
/// FPN merges features: out = upsample(coarse) + lateral(fine).
/// If |upsample(coarse)| <= C and |lateral(fine)| <= L, then
/// |out| <= C + L.
///
/// We model scalar proxy: out = u + l with |u| <= 5, |l| <= 3.
/// Prove: |out| <= 8.
#[test]
fn test_851_fpn_topdown_upsample_plus_lateral_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("u", real.clone());
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("out", real);

    let u = real_var("u");
    let l = real_var("l");
    let out = real_var("out");

    // |u| <= 5 (upsampled coarse feature)
    prog.assert(u.clone().real_ge(Expr::real(-5)));
    prog.assert(u.clone().real_le(Expr::real(5)));

    // |l| <= 3 (lateral connection)
    prog.assert(l.clone().real_ge(Expr::real(-3)));
    prog.assert(l.clone().real_le(Expr::real(3)));

    // out = u + l
    prog.assert(out.clone().eq(u.real_add(l)));

    // Negated property: |out| > 8
    let violation = out
        .clone()
        .real_gt(Expr::real(8))
        .or(out.real_lt(Expr::real(-8)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "fpn_topdown_upsample_plus_lateral_bounded");
}

// ---------------------------------------------------------------------------
// Test 852: PAN bottom-up pathway — downsampled + lateral bounded
// ---------------------------------------------------------------------------

/// Prove: In PAN bottom-up pathway, the sum of a downsampled finer feature
/// and a lateral connection from coarser resolution is bounded.
///
/// PAN reverses FPN: out = downsample(fine) + lateral(coarse).
/// Same bound arithmetic: |out| <= D + L.
///
/// We model: out = d + l with |d| <= 4, |l| <= 6.
/// Prove: |out| <= 10.
#[test]
fn test_852_pan_bottomup_downsample_plus_lateral_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("out", real);

    let d = real_var("d");
    let l = real_var("l");
    let out = real_var("out");

    // |d| <= 4 (downsampled fine feature)
    prog.assert(d.clone().real_ge(Expr::real(-4)));
    prog.assert(d.clone().real_le(Expr::real(4)));

    // |l| <= 6 (lateral from coarser level)
    prog.assert(l.clone().real_ge(Expr::real(-6)));
    prog.assert(l.clone().real_le(Expr::real(6)));

    // out = d + l
    prog.assert(out.clone().eq(d.real_add(l)));

    // Negated property: |out| > 10
    let violation = out
        .clone()
        .real_gt(Expr::real(10))
        .or(out.real_lt(Expr::real(-10)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pan_bottomup_downsample_plus_lateral_bounded");
}

// ---------------------------------------------------------------------------
// Test 853: Bilinear upsample preserves value range
// ---------------------------------------------------------------------------

/// Prove: bilinear interpolation between four corner values stays within
/// the range of the corner values.
///
/// Bilinear interpolation: out = (1-s)*(1-t)*v00 + s*(1-t)*v10
///                              + (1-s)*t*v01 + s*t*v11
/// with s, t in [0, 1]. This is a convex combination (weights sum to 1,
/// all non-negative), so out is in [min(v_ij), max(v_ij)].
///
/// We model 4 corners in [lo, hi] with convex combination weights.
/// Prove: lo <= out <= hi.
#[test]
fn test_853_bilinear_upsample_preserves_value_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("v00", real.clone());
    let _ = prog.declare_const("v10", real.clone());
    let _ = prog.declare_const("v01", real.clone());
    let _ = prog.declare_const("v11", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("t", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let v00 = real_var("v00");
    let v10 = real_var("v10");
    let v01 = real_var("v01");
    let v11 = real_var("v11");
    let s = real_var("s");
    let t = real_var("t");
    let out = real_var("out");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // All corners in [lo, hi]
    prog.assert(v00.clone().real_ge(lo.clone()));
    prog.assert(v00.clone().real_le(hi.clone()));
    prog.assert(v10.clone().real_ge(lo.clone()));
    prog.assert(v10.clone().real_le(hi.clone()));
    prog.assert(v01.clone().real_ge(lo.clone()));
    prog.assert(v01.clone().real_le(hi.clone()));
    prog.assert(v11.clone().real_ge(lo.clone()));
    prog.assert(v11.clone().real_le(hi.clone()));

    // s, t in [0, 1]
    prog.assert(s.clone().real_ge(Expr::real(0)));
    prog.assert(s.clone().real_le(Expr::real(1)));
    prog.assert(t.clone().real_ge(Expr::real(0)));
    prog.assert(t.clone().real_le(Expr::real(1)));

    // Bilinear interpolation weights:
    // w00 = (1-s)*(1-t), w10 = s*(1-t), w01 = (1-s)*t, w11 = s*t
    let one = Expr::real(1);
    let one_minus_s = one.clone().real_sub(s.clone());
    let one_minus_t = one.real_sub(t.clone());

    let term00 = one_minus_s
        .clone()
        .real_mul(one_minus_t.clone())
        .real_mul(v00);
    let term10 = s.clone().real_mul(one_minus_t).real_mul(v10);
    let term01 = one_minus_s.real_mul(t.clone()).real_mul(v01);
    let term11 = s.real_mul(t).real_mul(v11);

    // out = bilinear(v00, v10, v01, v11, s, t)
    prog.assert(
        out.clone()
            .eq(term00.real_add(term10).real_add(term01).real_add(term11)),
    );

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "bilinear_upsample_preserves_value_range");
}

// ---------------------------------------------------------------------------
// Test 854: Nearest-neighbor upsample preserves exact values
// ---------------------------------------------------------------------------

/// Prove: nearest-neighbor upsampling copies the source value exactly.
///
/// Nearest-neighbor: out = v_source (identity copy for the selected cell).
/// If v_source is in [lo, hi], then out is in [lo, hi].
///
/// We model: out = v_source. Prove: lo <= out <= hi.
#[test]
fn test_854_nearest_neighbor_upsample_preserves_exact_values() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("v_source", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let v_source = real_var("v_source");
    let out = real_var("out");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // v_source in [lo, hi]
    prog.assert(v_source.clone().real_ge(lo.clone()));
    prog.assert(v_source.clone().real_le(hi.clone()));

    // Nearest-neighbor: out = v_source
    prog.assert(out.clone().eq(v_source));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "nearest_neighbor_upsample_preserves_exact_values");
}

// ---------------------------------------------------------------------------
// Test 855: 1x1 lateral convolution bounded when input bounded
// ---------------------------------------------------------------------------

/// Prove: a 1x1 convolution (linear projection per-pixel) is bounded
/// when input and weight are bounded.
///
/// 1x1 conv with C_in=2, C_out=1: out = w1*x1 + w2*x2 + b.
/// If |x_i| <= X, |w_i| <= W, |b| <= B, then
/// |out| <= C_in * W * X + B.
///
/// For C_in=2, X=3, W=2, B=1: |out| <= 2*2*3 + 1 = 13.
#[test]
fn test_855_1x1_lateral_conv_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("out", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let b = real_var("b");
    let out = real_var("out");

    // |x_i| <= 3
    prog.assert(x1.clone().real_ge(Expr::real(-3)));
    prog.assert(x1.clone().real_le(Expr::real(3)));
    prog.assert(x2.clone().real_ge(Expr::real(-3)));
    prog.assert(x2.clone().real_le(Expr::real(3)));

    // |w_i| <= 2
    prog.assert(w1.clone().real_ge(Expr::real(-2)));
    prog.assert(w1.clone().real_le(Expr::real(2)));
    prog.assert(w2.clone().real_ge(Expr::real(-2)));
    prog.assert(w2.clone().real_le(Expr::real(2)));

    // |b| <= 1
    prog.assert(b.clone().real_ge(Expr::real(-1)));
    prog.assert(b.clone().real_le(Expr::real(1)));

    // out = w1*x1 + w2*x2 + b
    prog.assert(
        out.clone()
            .eq(w1.real_mul(x1).real_add(w2.real_mul(x2)).real_add(b)),
    );

    // Negated property: |out| > 13
    let violation = out
        .clone()
        .real_gt(Expr::real(13))
        .or(out.real_lt(Expr::real(-13)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "1x1_lateral_conv_bounded");
}

// ---------------------------------------------------------------------------
// Test 856: Element-wise addition preserves bound sum
// ---------------------------------------------------------------------------

/// Prove: element-wise addition of two bounded tensors produces output
/// bounded by the sum of bounds.
///
/// If x in [a, b] and y in [c, d], then x + y in [a+c, b+d].
///
/// We model: out = x + y. Prove: a+c <= out <= b+d.
#[test]
fn test_856_elementwise_addition_preserves_bound_sum() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("d", real);

    let x = real_var("x");
    let y = real_var("y");
    let out = real_var("out");
    let a = real_var("a");
    let b = real_var("b");
    let c = real_var("c");
    let d = real_var("d");

    // a <= b, c <= d
    prog.assert(a.clone().real_le(b.clone()));
    prog.assert(c.clone().real_le(d.clone()));

    // x in [a, b]
    prog.assert(x.clone().real_ge(a.clone()));
    prog.assert(x.clone().real_le(b.clone()));

    // y in [c, d]
    prog.assert(y.clone().real_ge(c.clone()));
    prog.assert(y.clone().real_le(d.clone()));

    // out = x + y
    prog.assert(out.clone().eq(x.real_add(y)));

    // Negated property: out < a+c OR out > b+d
    let lower = a.real_add(c);
    let upper = b.real_add(d);
    let violation = out.clone().real_lt(lower).or(out.real_gt(upper));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "elementwise_addition_preserves_bound_sum");
}

// ---------------------------------------------------------------------------
// Test 857: BiFPN weighted fusion — wi/(sum(wi)+eps) in [0,1]
// ---------------------------------------------------------------------------

/// Prove: BiFPN fast normalized fusion weights are in [0, 1].
///
/// BiFPN fusion: w_norm_i = w_i / (sum(w_j) + eps) where w_i >= 0.
/// Since w_i >= 0 and denominator = sum(w_j) + eps > 0,
/// w_norm_i >= 0. Also w_i <= sum(w_j) <= sum(w_j) + eps = denom,
/// so w_norm_i <= 1.
///
/// For n=2: w_norm_1 = w1 / (w1 + w2 + eps).
/// Prove: 0 <= w_norm_1 <= 1.
#[test]
fn test_857_bifpn_weighted_fusion_normalized_in_unit() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("eps", real.clone());
    let _ = prog.declare_const("denom", real.clone());
    let _ = prog.declare_const("w_norm", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let eps = real_var("eps");
    let denom = real_var("denom");
    let w_norm = real_var("w_norm");

    // w1, w2 >= 0 (ReLU-activated weights)
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));

    // eps > 0 (small positive constant, e.g. 1e-4)
    prog.assert(eps.clone().real_gt(Expr::real(0)));

    // denom = w1 + w2 + eps
    prog.assert(
        denom
            .clone()
            .eq(w1.clone().real_add(w2.clone()).real_add(eps)),
    );

    // w_norm * denom = w1 (i.e., w_norm = w1 / denom)
    prog.assert(w_norm.clone().real_mul(denom).eq(w1));

    // Negated property: w_norm < 0 OR w_norm > 1
    let violation = w_norm
        .clone()
        .real_lt(Expr::real(0))
        .or(w_norm.real_gt(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "bifpn_weighted_fusion_normalized_in_unit");
}

// ---------------------------------------------------------------------------
// Test 858: Feature concatenation along channel dimension
// ---------------------------------------------------------------------------

/// Prove: concatenating features from two levels along the channel dimension
/// preserves the original bounds of each channel.
///
/// For level A with value a in [lo_a, hi_a] and level B with value b in
/// [lo_b, hi_b], the concatenated output has two channels:
/// channel 0 = a (preserves [lo_a, hi_a]), channel 1 = b (preserves [lo_b, hi_b]).
///
/// We model: concat preserves identity. Prove: bounds preserved.
#[test]
fn test_858_feature_concat_channel_dim_preserves_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("out_ch0", real.clone());
    let _ = prog.declare_const("out_ch1", real.clone());
    let _ = prog.declare_const("lo_a", real.clone());
    let _ = prog.declare_const("hi_a", real.clone());
    let _ = prog.declare_const("lo_b", real.clone());
    let _ = prog.declare_const("hi_b", real);

    let a = real_var("a");
    let b = real_var("b");
    let out_ch0 = real_var("out_ch0");
    let out_ch1 = real_var("out_ch1");
    let lo_a = real_var("lo_a");
    let hi_a = real_var("hi_a");
    let lo_b = real_var("lo_b");
    let hi_b = real_var("hi_b");

    // Valid intervals
    prog.assert(lo_a.clone().real_le(hi_a.clone()));
    prog.assert(lo_b.clone().real_le(hi_b.clone()));

    // a in [lo_a, hi_a]
    prog.assert(a.clone().real_ge(lo_a.clone()));
    prog.assert(a.clone().real_le(hi_a.clone()));

    // b in [lo_b, hi_b]
    prog.assert(b.clone().real_ge(lo_b.clone()));
    prog.assert(b.clone().real_le(hi_b.clone()));

    // Concatenation: out_ch0 = a, out_ch1 = b
    prog.assert(out_ch0.clone().eq(a));
    prog.assert(out_ch1.clone().eq(b));

    // Negated property: out_ch0 not in [lo_a, hi_a] OR out_ch1 not in [lo_b, hi_b]
    let viol_a = out_ch0.clone().real_lt(lo_a).or(out_ch0.real_gt(hi_a));
    let viol_b = out_ch1.clone().real_lt(lo_b).or(out_ch1.real_gt(hi_b));
    let violation = viol_a.or(viol_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "feature_concat_channel_dim_preserves_bounds");
}

// ---------------------------------------------------------------------------
// Test 859: Multi-scale P3/P4/P5 pyramid consistency
// ---------------------------------------------------------------------------

/// Prove: in a 3-level feature pyramid, the stride relationship is consistent.
///
/// P3 stride = base_stride, P4 stride = 2 * base_stride,
/// P5 stride = 4 * base_stride. This means:
///   P4_stride = 2 * P3_stride and P5_stride = 2 * P4_stride.
///
/// We model these constraints and prove P5 = 4 * P3.
#[test]
fn test_859_multiscale_p3_p4_p5_pyramid_consistency() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p3_stride", real.clone());
    let _ = prog.declare_const("p4_stride", real.clone());
    let _ = prog.declare_const("p5_stride", real);

    let p3_stride = real_var("p3_stride");
    let p4_stride = real_var("p4_stride");
    let p5_stride = real_var("p5_stride");

    // p3_stride > 0
    prog.assert(p3_stride.clone().real_gt(Expr::real(0)));

    // p4_stride = 2 * p3_stride
    prog.assert(
        p4_stride
            .clone()
            .eq(Expr::real(2).real_mul(p3_stride.clone())),
    );

    // p5_stride = 2 * p4_stride
    prog.assert(p5_stride.clone().eq(Expr::real(2).real_mul(p4_stride)));

    // Negated property: p5_stride != 4 * p3_stride
    let violation = p5_stride.ne(Expr::real(4).real_mul(p3_stride));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "multiscale_p3_p4_p5_pyramid_consistency");
}

// ---------------------------------------------------------------------------
// Test 860: Stride-2 downsample spatial halving
// ---------------------------------------------------------------------------

/// Prove: stride-2 convolution halves the spatial dimension.
///
/// For input spatial dim H_in, output H_out = floor((H_in + 2*pad - kernel) / stride) + 1.
/// With pad=1, kernel=3, stride=2: H_out = (H_in + 2 - 3) / 2 + 1 = (H_in - 1) / 2 + 1.
/// For even H_in = 2*N: H_out = (2*N - 1)/2 + 1 = N (integer division) + 1... but
/// more simply, for the typical case H_in = 2*N with pad=1, kernel=3, stride=2:
/// H_out = N.
///
/// We model the simpler property: H_out * stride = H_in for exact halving.
/// Prove: H_out = H_in / 2.
#[test]
fn test_860_stride2_downsample_spatial_halving() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("h_in", real.clone());
    let _ = prog.declare_const("h_out", real.clone());
    let _ = prog.declare_const("stride", real);

    let h_in = real_var("h_in");
    let h_out = real_var("h_out");
    let stride = real_var("stride");

    // h_in > 0
    prog.assert(h_in.clone().real_gt(Expr::real(0)));

    // stride = 2
    prog.assert(stride.clone().eq(Expr::real(2)));

    // h_out * stride = h_in (exact spatial halving)
    prog.assert(h_out.clone().real_mul(stride).eq(h_in.clone()));

    // Negated property: h_out != h_in / 2
    // Equivalently: 2 * h_out != h_in
    let violation = Expr::real(2).real_mul(h_out).ne(h_in);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stride2_downsample_spatial_halving");
}

// ---------------------------------------------------------------------------
// Test 861: Feature normalization at each pyramid level
// ---------------------------------------------------------------------------

/// Prove: L2 normalization constrains the output norm to 1.
///
/// For a 2-element feature vector: norm = sqrt(x1^2 + x2^2).
/// Normalized: y_i = x_i / norm. Then y1^2 + y2^2 = 1.
///
/// We model: y1 = x1/norm, y2 = x2/norm, norm^2 = x1^2 + x2^2.
/// Prove: y1^2 + y2^2 = 1.
#[test]
fn test_861_feature_normalization_at_pyramid_level() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("norm_sq", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let norm_sq = real_var("norm_sq");
    let y1 = real_var("y1");
    let y2 = real_var("y2");

    // norm_sq = x1^2 + x2^2
    prog.assert(
        norm_sq.clone().eq(x1
            .clone()
            .real_mul(x1.clone())
            .real_add(x2.clone().real_mul(x2.clone()))),
    );

    // norm_sq > 0 (non-zero input)
    prog.assert(norm_sq.clone().real_gt(Expr::real(0)));

    // y_i * norm_sq = x_i * sqrt(norm_sq)
    // Equivalently: y_i^2 * norm_sq = x_i^2 (from y_i = x_i / sqrt(norm_sq))
    prog.assert(
        y1.clone()
            .real_mul(y1.clone())
            .real_mul(norm_sq.clone())
            .eq(x1.clone().real_mul(x1)),
    );
    prog.assert(
        y2.clone()
            .real_mul(y2.clone())
            .real_mul(norm_sq.clone())
            .eq(x2.clone().real_mul(x2)),
    );

    // Negated property: y1^2 + y2^2 != 1
    let y_norm_sq = y1.clone().real_mul(y1).real_add(y2.clone().real_mul(y2));
    let violation = y_norm_sq.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "feature_normalization_at_pyramid_level");
}

// ---------------------------------------------------------------------------
// Test 862: Recursive FPN — deeper = coarser + finer bounded
// ---------------------------------------------------------------------------

/// Prove: recursive FPN merge at each level is bounded by induction.
///
/// Level k: out_k = upsample(out_{k+1}) + lateral_k.
/// If |out_{k+1}| <= B_{k+1} and |lateral_k| <= L_k, then
/// |out_k| <= B_{k+1} + L_k (since upsample preserves range).
///
/// For 2 levels: |out_top| <= T, |lat_mid| <= M.
/// |out_mid| <= T + M.
/// Prove: |out_mid| <= T + M.
#[test]
fn test_862_recursive_fpn_deeper_coarser_finer_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("out_top", real.clone());
    let _ = prog.declare_const("lat_mid", real.clone());
    let _ = prog.declare_const("out_mid", real.clone());
    let _ = prog.declare_const("top_bound", real.clone());
    let _ = prog.declare_const("lat_bound", real);

    let out_top = real_var("out_top");
    let lat_mid = real_var("lat_mid");
    let out_mid = real_var("out_mid");
    let top_bound = real_var("top_bound");
    let lat_bound = real_var("lat_bound");

    // Bounds are positive
    prog.assert(top_bound.clone().real_gt(Expr::real(0)));
    prog.assert(lat_bound.clone().real_gt(Expr::real(0)));

    // |out_top| <= top_bound
    prog.assert(
        out_top
            .clone()
            .real_ge(top_bound.clone().real_mul(Expr::real(-1))),
    );
    prog.assert(out_top.clone().real_le(top_bound.clone()));

    // |lat_mid| <= lat_bound
    prog.assert(
        lat_mid
            .clone()
            .real_ge(lat_bound.clone().real_mul(Expr::real(-1))),
    );
    prog.assert(lat_mid.clone().real_le(lat_bound.clone()));

    // out_mid = out_top + lat_mid (upsample preserves range, so upsample(out_top) has same bound)
    prog.assert(out_mid.clone().eq(out_top.real_add(lat_mid)));

    // Negated property: |out_mid| > top_bound + lat_bound
    let total_bound = top_bound.real_add(lat_bound);
    let violation = out_mid
        .clone()
        .real_gt(total_bound.clone())
        .or(out_mid.real_lt(total_bound.real_mul(Expr::real(-1))));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "recursive_fpn_deeper_coarser_finer_bounded");
}

// ---------------------------------------------------------------------------
// Test 863: PANet lateral connection symmetry
// ---------------------------------------------------------------------------

/// Prove: PANet lateral connections are symmetric — the merge operation
/// (addition) is commutative, so top-down + bottom-up = bottom-up + top-down.
///
/// This means the order of FPN and PAN merges does not matter for the sum.
///
/// We model: merge_td = a + b, merge_bu = b + a.
/// Prove: merge_td = merge_bu.
#[test]
fn test_863_panet_lateral_connection_symmetry() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("merge_td", real.clone());
    let _ = prog.declare_const("merge_bu", real);

    let a = real_var("a");
    let b = real_var("b");
    let merge_td = real_var("merge_td");
    let merge_bu = real_var("merge_bu");

    // Bounded inputs
    prog.assert(a.clone().real_ge(Expr::real(-10)));
    prog.assert(a.clone().real_le(Expr::real(10)));
    prog.assert(b.clone().real_ge(Expr::real(-10)));
    prog.assert(b.clone().real_le(Expr::real(10)));

    // merge_td = a + b (top-down merge)
    prog.assert(merge_td.clone().eq(a.clone().real_add(b.clone())));

    // merge_bu = b + a (bottom-up merge)
    prog.assert(merge_bu.clone().eq(b.real_add(a)));

    // Negated property: merge_td != merge_bu
    let violation = merge_td.ne(merge_bu);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "panet_lateral_connection_symmetry");
}

// ---------------------------------------------------------------------------
// Test 864: DetectHead input from multiple FPN levels
// ---------------------------------------------------------------------------

/// Prove: when the detection head receives features from multiple FPN levels,
/// and each level is bounded, the head input for any level is bounded.
///
/// DetectHead processes each FPN level independently through shared convs.
/// For level i with feature f_i and shared weight w, bias b:
///   head_out_i = w * f_i + b.
/// If |f_i| <= F, |w| <= W, |b| <= B, then |head_out_i| <= W*F + B.
///
/// We model: head_out = w * f + b with |f| <= 4, |w| <= 1, |b| <= 0.5.
/// Prove: |head_out| <= 4.5.
#[test]
fn test_864_detecthead_input_from_multiple_fpn_levels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("f", real.clone());
    let _ = prog.declare_const("w", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("head_out", real);

    let f = real_var("f");
    let w = real_var("w");
    let b = real_var("b");
    let head_out = real_var("head_out");

    // |f| <= 4
    prog.assert(f.clone().real_ge(Expr::real(-4)));
    prog.assert(f.clone().real_le(Expr::real(4)));

    // |w| <= 1
    prog.assert(w.clone().real_ge(Expr::real(-1)));
    prog.assert(w.clone().real_le(Expr::real(1)));

    // |b| <= 0.5
    prog.assert(b.clone().real_ge(Expr::real_ratio(-1, 2)));
    prog.assert(b.clone().real_le(Expr::real_ratio(1, 2)));

    // head_out = w * f + b
    prog.assert(head_out.clone().eq(w.real_mul(f).real_add(b)));

    // Negated property: |head_out| > 4.5
    let violation = head_out
        .clone()
        .real_gt(Expr::real_ratio(9, 2))
        .or(head_out.real_lt(Expr::real_ratio(-9, 2)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "detecthead_input_from_multiple_fpn_levels");
}

// ---------------------------------------------------------------------------
// Test 865: Anchor grid generation bounded in image space
// ---------------------------------------------------------------------------

/// Prove: anchor grid coordinates are bounded within the image dimensions.
///
/// Anchor centers: cx = (grid_x + 0.5) * stride, cy = (grid_y + 0.5) * stride.
/// If grid_x in [0, W_grid-1] and stride > 0, then
/// cx in [0.5*stride, (W_grid - 0.5)*stride].
/// For image width = W_grid * stride, cx is in (0, image_width).
///
/// We model: cx = (gx + 0.5) * stride with gx in [0, G-1], stride > 0.
/// Prove: 0 < cx < G * stride.
#[test]
fn test_865_anchor_grid_bounded_in_image_space() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("gx", real.clone());
    let _ = prog.declare_const("stride", real.clone());
    let _ = prog.declare_const("grid_size", real.clone());
    let _ = prog.declare_const("cx", real.clone());
    let _ = prog.declare_const("img_width", real);

    let gx = real_var("gx");
    let stride = real_var("stride");
    let grid_size = real_var("grid_size");
    let cx = real_var("cx");
    let img_width = real_var("img_width");

    // gx in [0, grid_size - 1], grid_size >= 1
    prog.assert(grid_size.clone().real_ge(Expr::real(1)));
    prog.assert(gx.clone().real_ge(Expr::real(0)));
    prog.assert(
        gx.clone()
            .real_le(grid_size.clone().real_sub(Expr::real(1))),
    );

    // stride > 0
    prog.assert(stride.clone().real_gt(Expr::real(0)));

    // img_width = grid_size * stride
    prog.assert(img_width.clone().eq(grid_size.real_mul(stride.clone())));

    // cx = (gx + 0.5) * stride
    prog.assert(
        cx.clone()
            .eq(gx.real_add(Expr::real_ratio(1, 2)).real_mul(stride)),
    );

    // Negated property: cx <= 0 OR cx >= img_width
    let violation = cx.clone().real_le(Expr::real(0)).or(cx.real_ge(img_width));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "anchor_grid_bounded_in_image_space");
}

// ---------------------------------------------------------------------------
// Test 866: Feature stride relationship P3=8, P4=16, P5=32
// ---------------------------------------------------------------------------

/// Prove: standard FPN stride relationship is consistent.
///
/// P3_stride = 8, P4_stride = 16, P5_stride = 32.
/// Properties: P4 = 2 * P3, P5 = 2 * P4, P5 = 4 * P3.
///
/// We model concrete strides and prove the doubling relationship.
#[test]
fn test_866_feature_stride_relationship_p3_p4_p5() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p3", real.clone());
    let _ = prog.declare_const("p4", real.clone());
    let _ = prog.declare_const("p5", real);

    let p3 = real_var("p3");
    let p4 = real_var("p4");
    let p5 = real_var("p5");

    // Concrete stride values
    prog.assert(p3.clone().eq(Expr::real(8)));
    prog.assert(p4.clone().eq(Expr::real(16)));
    prog.assert(p5.clone().eq(Expr::real(32)));

    // Negated property: P4 != 2*P3 OR P5 != 2*P4 OR P5 != 4*P3
    let v1 = p4.clone().ne(Expr::real(2).real_mul(p3.clone()));
    let v2 = p5.clone().ne(Expr::real(2).real_mul(p4));
    let v3 = p5.ne(Expr::real(4).real_mul(p3));
    let violation = v1.or(v2).or(v3);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "feature_stride_relationship_p3_p4_p5");
}

// ---------------------------------------------------------------------------
// Test 867: Cross-scale attention bounded
// ---------------------------------------------------------------------------

/// Prove: cross-scale attention (attention between features at different
/// pyramid levels) produces bounded output when Q, K, V are bounded.
///
/// Cross-scale attention: out = softmax(Q*K^T / sqrt(d)) * V.
/// Since softmax produces a convex combination (weights >= 0, sum = 1)
/// and V is bounded, out is in the same range as V.
///
/// We model: out = a1*v1 + a2*v2 with a1+a2=1, a_i>=0, v_i in [lo,hi].
/// Prove: lo <= out <= hi (same as convex combination proof).
#[test]
fn test_867_cross_scale_attention_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("v2", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let v1 = real_var("v1");
    let v2 = real_var("v2");
    let out = real_var("out");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // a1, a2 >= 0 (softmax outputs)
    prog.assert(a1.clone().real_ge(Expr::real(0)));
    prog.assert(a2.clone().real_ge(Expr::real(0)));

    // a1 + a2 = 1
    prog.assert(a1.clone().real_add(a2.clone()).eq(Expr::real(1)));

    // v1, v2 in [lo, hi]
    prog.assert(v1.clone().real_ge(lo.clone()));
    prog.assert(v1.clone().real_le(hi.clone()));
    prog.assert(v2.clone().real_ge(lo.clone()));
    prog.assert(v2.clone().real_le(hi.clone()));

    // out = a1*v1 + a2*v2
    prog.assert(out.clone().eq(a1.real_mul(v1).real_add(a2.real_mul(v2))));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cross_scale_attention_bounded");
}

// ---------------------------------------------------------------------------
// Test 868: Deformable conv sampling points in receptive field
// ---------------------------------------------------------------------------

/// Prove: deformable convolution sampling points with bounded offsets
/// stay within a receptive field around the original grid position.
///
/// Base grid position: p0. Offset: delta in [-max_offset, max_offset].
/// Sampling point: p = p0 + delta.
/// Prove: |p - p0| <= max_offset, i.e., p in [p0 - max_offset, p0 + max_offset].
///
/// We model: p = p0 + delta with |delta| <= R.
/// Prove: p in [p0 - R, p0 + R].
#[test]
fn test_868_deformable_conv_sampling_in_receptive_field() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("p0", real.clone());
    let _ = prog.declare_const("delta", real.clone());
    let _ = prog.declare_const("p", real.clone());
    let _ = prog.declare_const("max_offset", real);

    let p0 = real_var("p0");
    let delta = real_var("delta");
    let p = real_var("p");
    let max_offset = real_var("max_offset");

    // max_offset > 0
    prog.assert(max_offset.clone().real_gt(Expr::real(0)));

    // |delta| <= max_offset
    prog.assert(
        delta
            .clone()
            .real_ge(max_offset.clone().real_mul(Expr::real(-1))),
    );
    prog.assert(delta.clone().real_le(max_offset.clone()));

    // p = p0 + delta
    prog.assert(p.clone().eq(p0.clone().real_add(delta)));

    // Negated property: p < p0 - max_offset OR p > p0 + max_offset
    let lower = p0.clone().real_sub(max_offset.clone());
    let upper = p0.real_add(max_offset);
    let violation = p.clone().real_lt(lower).or(p.real_gt(upper));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "deformable_conv_sampling_in_receptive_field");
}

// ---------------------------------------------------------------------------
// Test 869: FPN output channel uniformity — all levels same channels
// ---------------------------------------------------------------------------

/// Prove: FPN output channel count is uniform across all pyramid levels.
///
/// FPN uses 1x1 lateral convolutions to project each backbone level to
/// the same channel count C_fpn. For levels P3, P4, P5:
///   C_out_P3 = C_fpn, C_out_P4 = C_fpn, C_out_P5 = C_fpn.
///
/// We model: all outputs = C_fpn.
/// Prove: C_out_P3 = C_out_P4 = C_out_P5.
#[test]
fn test_869_fpn_output_channel_uniformity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("c_fpn", real.clone());
    let _ = prog.declare_const("c_p3", real.clone());
    let _ = prog.declare_const("c_p4", real.clone());
    let _ = prog.declare_const("c_p5", real);

    let c_fpn = real_var("c_fpn");
    let c_p3 = real_var("c_p3");
    let c_p4 = real_var("c_p4");
    let c_p5 = real_var("c_p5");

    // c_fpn > 0
    prog.assert(c_fpn.clone().real_gt(Expr::real(0)));

    // All levels projected to C_fpn
    prog.assert(c_p3.clone().eq(c_fpn.clone()));
    prog.assert(c_p4.clone().eq(c_fpn.clone()));
    prog.assert(c_p5.clone().eq(c_fpn));

    // Negated property: c_p3 != c_p4 OR c_p4 != c_p5
    let violation = c_p3.clone().ne(c_p4.clone()).or(c_p4.ne(c_p5));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "fpn_output_channel_uniformity");
}

// ---------------------------------------------------------------------------
// Test 870: Skip connection identity preserves bounds exactly
// ---------------------------------------------------------------------------

/// Prove: a skip (residual) connection that adds input x to output f(x)
/// preserves the sum of bounds. For the identity skip (no transform),
/// out = x + f(x). If x in [a, b] and f(x) in [c, d], then
/// out in [a+c, b+d].
///
/// Special case: if f is zero (identity block output = 0), then
/// out = x + 0 = x, preserving bounds exactly.
///
/// We model the identity skip: out = x + 0 = x.
/// Prove: if x in [lo, hi], then out in [lo, hi].
#[test]
fn test_870_skip_connection_identity_preserves_bounds_exactly() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("f_x", real.clone());
    let _ = prog.declare_const("out", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let x = real_var("x");
    let f_x = real_var("f_x");
    let out = real_var("out");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // x in [lo, hi]
    prog.assert(x.clone().real_ge(lo.clone()));
    prog.assert(x.clone().real_le(hi.clone()));

    // f(x) = 0 (identity skip: residual branch is zero)
    prog.assert(f_x.clone().eq(Expr::real(0)));

    // out = x + f(x)
    prog.assert(out.clone().eq(x.real_add(f_x)));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "skip_connection_identity_preserves_bounds_exactly");
}
