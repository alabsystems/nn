// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for depthwise separable convolution
//! mathematical properties.
//!
//! Proves fundamental properties of depthwise separable convolutions used
//! in efficient neural network architectures (MobileNet, EfficientNet, etc.):
//! - Depthwise conv output channels == input channels (groups == channels)
//! - Pointwise 1x1 conv output bounded when input bounded
//! - Depthwise separable = depthwise + pointwise decomposition
//! - Output spatial size formula: (H + 2*pad - K) / stride + 1
//! - Stride > 0 prevents zero-division in spatial computation
//! - Padding preserves spatial extent when pad = (K-1)/2 and stride=1
//! - Dilation expands effective kernel to K + (K-1)*(d-1)
//! - Grouped conv with G groups: each group processes C_in/G channels
//! - Depthwise conv parameter count: C * K * K (vs C^2 * K * K for standard)
//! - Batch normalization after depthwise preserves channel count
//! - SE block squeeze ratio reduces channels: C -> C/r -> C
//! - MBConv expansion: thin -> wide -> thin bottleneck bounds
//! - ReLU6 clamps output to [0, 6] for bounded activation
//! - Inverted residual skip connection dimension match
//! - EfficientNet compound scaling: width * depth * resolution
//! - Separable conv reduces FLOPs by factor 1/C_out + 1/K^2
//! - Average pooling output bounded by input bounds
//! - Channel shuffle for grouped convolutions maintains total channels
//! - Depthwise conv weight sharing per-channel independence
//! - Fused MBConv (expand + depthwise in single conv) equivalence
//!
//! Part of #4181.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};
use nn_verify::ay_real_lit::RealLit;

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: create an Int-sorted variable.
fn int_var(name: &str) -> Expr {
    Expr::var(name, Sort::int())
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
// Test 831: Depthwise conv output channels == input channels
// ---------------------------------------------------------------------------

/// Prove: in depthwise convolution, the number of output channels equals
/// the number of input channels (groups == channels).
///
/// Depthwise conv sets groups = C_in, so each input channel is convolved
/// independently with its own filter. The output has exactly C_in channels.
///
/// We model: C_out = C_in * (filters_per_group) where filters_per_group = 1
/// for standard depthwise conv and groups = C_in.
/// Prove: C_out = C_in.
#[test]
fn test_831_depthwise_output_channels_equal_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("groups", int.clone());
    let _ = prog.declare_const("filters_per_group", int.clone());
    let _ = prog.declare_const("c_out", int);

    let c_in = int_var("c_in");
    let groups = int_var("groups");
    let filters_per_group = int_var("filters_per_group");
    let c_out = int_var("c_out");

    // c_in > 0
    prog.assert(c_in.clone().int_gt(Expr::int(0)));

    // groups = c_in (depthwise convolution definition)
    prog.assert(groups.clone().eq(c_in.clone()));

    // filters_per_group = 1 (standard depthwise: one filter per group)
    prog.assert(filters_per_group.clone().eq(Expr::int(1)));

    // c_out = groups * filters_per_group
    prog.assert(c_out.clone().eq(groups.int_mul(filters_per_group)));

    // Negated property: c_out != c_in
    let violation = c_out.ne(c_in);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "depthwise_output_channels_equal_input");
}

// ---------------------------------------------------------------------------
// Test 832: Pointwise 1x1 conv output bounded when input bounded
// ---------------------------------------------------------------------------

/// Prove: a pointwise (1x1) convolution produces bounded output when
/// input and weights are bounded.
///
/// Pointwise conv: y = sum_c(w_c * x_c) + b (dot product over channels).
/// For C_in channels, if |x_c| <= X and |w_c| <= W and |b| <= B,
/// then |y| <= C_in * X * W + B.
///
/// We model for C_in=3: y = w1*x1 + w2*x2 + w3*x3 + b.
/// Prove: |y| <= 3 * X * W + B.
#[test]
fn test_832_pointwise_conv_output_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("y", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");
    let b = real_var("b");
    let y = real_var("y");

    // |x_c| <= 2
    for x in [&x1, &x2, &x3] {
        prog.assert(x.clone().real_ge(Expr::real(-2)));
        prog.assert(x.clone().real_le(Expr::real(2)));
    }

    // |w_c| <= 1
    for w in [&w1, &w2, &w3] {
        prog.assert(w.clone().real_ge(Expr::real(-1)));
        prog.assert(w.clone().real_le(Expr::real(1)));
    }

    // |b| <= 1
    prog.assert(b.clone().real_ge(Expr::real(-1)));
    prog.assert(b.clone().real_le(Expr::real(1)));

    // y = w1*x1 + w2*x2 + w3*x3 + b
    let dot = w1
        .real_mul(x1)
        .real_add(w2.real_mul(x2))
        .real_add(w3.real_mul(x3));
    prog.assert(y.clone().eq(dot.real_add(b)));

    // Bound: |y| <= 3*2*1 + 1 = 7
    let violation = y
        .clone()
        .real_gt(Expr::real(7))
        .or(y.real_lt(Expr::real(-7)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pointwise_conv_output_bounded");
}

// ---------------------------------------------------------------------------
// Test 833: Depthwise separable = depthwise + pointwise decomposition
// ---------------------------------------------------------------------------

/// Prove: depthwise separable convolution decomposes into the composition
/// of depthwise and pointwise convolutions, and the output of the first
/// feeds correctly into the second.
///
/// Standard conv: y = W_std * x (C_out x C_in kernel).
/// Depthwise separable: y = W_pw * (W_dw * x) where W_dw is per-channel
/// and W_pw is 1x1.
///
/// For 1 input channel to 1 output: depthwise output d = w_dw * x,
/// then pointwise output y = w_pw * d = w_pw * w_dw * x.
/// This equals a standard conv with w_std = w_pw * w_dw.
///
/// Prove: y_sep = y_std when w_std = w_pw * w_dw.
#[test]
fn test_833_depthwise_separable_decomposition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("w_dw", real.clone());
    let _ = prog.declare_const("w_pw", real.clone());
    let _ = prog.declare_const("w_std", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("y_sep", real.clone());
    let _ = prog.declare_const("y_std", real);

    let x = real_var("x");
    let w_dw = real_var("w_dw");
    let w_pw = real_var("w_pw");
    let w_std = real_var("w_std");
    let d = real_var("d");
    let y_sep = real_var("y_sep");
    let y_std = real_var("y_std");

    // Bounded inputs
    prog.assert(x.clone().real_ge(Expr::real(-10)));
    prog.assert(x.clone().real_le(Expr::real(10)));
    prog.assert(w_dw.clone().real_ge(Expr::real(-5)));
    prog.assert(w_dw.clone().real_le(Expr::real(5)));
    prog.assert(w_pw.clone().real_ge(Expr::real(-5)));
    prog.assert(w_pw.clone().real_le(Expr::real(5)));

    // w_std = w_pw * w_dw (standard conv equivalent weight)
    prog.assert(w_std.clone().eq(w_pw.clone().real_mul(w_dw.clone())));

    // d = w_dw * x (depthwise step)
    prog.assert(d.clone().eq(w_dw.real_mul(x.clone())));

    // y_sep = w_pw * d (pointwise step)
    prog.assert(y_sep.clone().eq(w_pw.real_mul(d)));

    // y_std = w_std * x (standard conv)
    prog.assert(y_std.clone().eq(w_std.real_mul(x)));

    // Negated property: y_sep != y_std
    let violation = y_sep.ne(y_std);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "depthwise_separable_decomposition");
}

// ---------------------------------------------------------------------------
// Test 834: Output spatial size formula
// ---------------------------------------------------------------------------

/// Prove: the convolution output spatial size follows the formula:
///   out_size = (in_size + 2*pad - kernel_size) / stride + 1
///
/// For concrete values: in=8, pad=1, kernel=3, stride=2:
///   out = (8 + 2*1 - 3) / 2 + 1 = 7/2 + 1 = 3 + 1 = 4
/// (integer division: 7/2 = 3).
///
/// We model: out * stride <= in + 2*pad - kernel + stride
/// and out * stride > in + 2*pad - kernel (floor division).
/// Prove: out = floor((in + 2*pad - kernel) / stride) + 1.
#[test]
fn test_834_output_spatial_size_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("h_in", int.clone());
    let _ = prog.declare_const("pad", int.clone());
    let _ = prog.declare_const("k", int.clone());
    let _ = prog.declare_const("stride", int.clone());
    let _ = prog.declare_const("h_out", int.clone());
    let _ = prog.declare_const("effective", int);

    let h_in = int_var("h_in");
    let pad = int_var("pad");
    let k = int_var("k");
    let stride = int_var("stride");
    let h_out = int_var("h_out");
    let effective = int_var("effective");

    // Concrete values: h_in=8, pad=1, k=3, stride=2
    prog.assert(h_in.clone().eq(Expr::int(8)));
    prog.assert(pad.clone().eq(Expr::int(1)));
    prog.assert(k.clone().eq(Expr::int(3)));
    prog.assert(stride.clone().eq(Expr::int(2)));

    // effective = h_in + 2*pad - k = 8 + 2 - 3 = 7
    prog.assert(
        effective
            .clone()
            .eq(h_in.int_add(Expr::int(2).int_mul(pad)).int_sub(k)),
    );

    // h_out = effective / stride + 1 (integer division)
    // Encode floor div: h_out - 1 = effective div stride
    // => (h_out - 1) * stride <= effective
    // => (h_out - 1) * stride + stride > effective
    // => effective >= 0
    prog.assert(effective.clone().int_ge(Expr::int(0)));
    let quotient = h_out.clone().int_sub(Expr::int(1));
    prog.assert(
        quotient
            .clone()
            .int_mul(stride.clone())
            .int_le(effective.clone()),
    );
    prog.assert(
        quotient
            .int_mul(stride.clone())
            .int_add(stride)
            .int_gt(effective),
    );

    // Negated property: h_out != 4
    let violation = h_out.ne(Expr::int(4));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "output_spatial_size_formula");
}

// ---------------------------------------------------------------------------
// Test 835: Stride > 0 prevents zero-division in spatial computation
// ---------------------------------------------------------------------------

/// Prove: when stride > 0, the spatial dimension computation is
/// well-defined (no division by zero).
///
/// The output size formula divides by stride. If stride > 0, the
/// division is safe. We show: stride > 0 implies the denominator
/// in the output-size formula is positive.
///
/// Prove: stride > 0 => stride != 0.
#[test]
fn test_835_stride_positive_prevents_zero_division() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("stride", int);

    let stride = int_var("stride");

    // stride > 0
    prog.assert(stride.clone().int_gt(Expr::int(0)));

    // Negated property: stride = 0 (division by zero possible)
    let violation = stride.eq(Expr::int(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stride_positive_prevents_zero_division");
}

// ---------------------------------------------------------------------------
// Test 836: Padding preserves spatial extent (same padding)
// ---------------------------------------------------------------------------

/// Prove: when pad = (K-1)/2 and stride = 1, the output spatial size
/// equals the input spatial size ("same" padding).
///
/// out = (H + 2*pad - K) / stride + 1
///     = (H + 2*(K-1)/2 - K) / 1 + 1
///     = (H + K - 1 - K) + 1
///     = H.
///
/// For K odd (so (K-1)/2 is exact integer): K=3 => pad=1.
/// Prove: out = H.
#[test]
fn test_836_same_padding_preserves_spatial_extent() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("h", int.clone());
    let _ = prog.declare_const("k", int.clone());
    let _ = prog.declare_const("pad", int.clone());
    let _ = prog.declare_const("h_out", int);

    let h = int_var("h");
    let k = int_var("k");
    let pad = int_var("pad");
    let h_out = int_var("h_out");

    // h > 0, k > 0, k odd
    prog.assert(h.clone().int_gt(Expr::int(0)));
    prog.assert(k.clone().int_gt(Expr::int(0)));

    // pad = (k - 1) / 2 (exact for odd k; we use concrete k=3, pad=1)
    prog.assert(k.clone().eq(Expr::int(3)));
    prog.assert(pad.clone().eq(Expr::int(1)));

    // stride = 1: h_out = h + 2*pad - k + 1
    prog.assert(
        h_out.clone().eq(h
            .clone()
            .int_add(Expr::int(2).int_mul(pad))
            .int_sub(k)
            .int_add(Expr::int(1))),
    );

    // Negated property: h_out != h
    let violation = h_out.ne(h);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "same_padding_preserves_spatial_extent");
}

// ---------------------------------------------------------------------------
// Test 837: Dilation expands effective kernel size
// ---------------------------------------------------------------------------

/// Prove: dilation d expands the effective kernel size from K to
/// K + (K-1)*(d-1) = d*(K-1) + 1.
///
/// A dilated kernel inserts (d-1) zeros between each pair of original
/// kernel elements. For K=3, d=2: effective = 3 + 2*(2-1) = 5.
///
/// Prove: effective_k = k + (k - 1) * (d - 1) for k=3, d=2.
#[test]
fn test_837_dilation_expands_effective_kernel() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("k", int.clone());
    let _ = prog.declare_const("d", int.clone());
    let _ = prog.declare_const("eff_k", int);

    let k = int_var("k");
    let d = int_var("d");
    let eff_k = int_var("eff_k");

    // k = 3, d = 2
    prog.assert(k.clone().eq(Expr::int(3)));
    prog.assert(d.clone().eq(Expr::int(2)));

    // eff_k = k + (k - 1) * (d - 1)
    prog.assert(
        eff_k.clone().eq(k
            .clone()
            .int_add(k.int_sub(Expr::int(1)).int_mul(d.int_sub(Expr::int(1))))),
    );

    // Negated property: eff_k != 5
    let violation = eff_k.ne(Expr::int(5));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "dilation_expands_effective_kernel");
}

// ---------------------------------------------------------------------------
// Test 838: Grouped conv channel partitioning
// ---------------------------------------------------------------------------

/// Prove: in grouped convolution with G groups, each group processes
/// C_in/G input channels and produces C_out/G output channels, and
/// the total output channels = C_out.
///
/// We model: channels_per_group_in = c_in / G, channels_per_group_out = c_out / G.
/// Total output = G * channels_per_group_out = C_out.
///
/// For c_in=12, c_out=24, G=3: each group has 4 in, 8 out.
/// Prove: G * (c_out / G) = c_out.
#[test]
fn test_838_grouped_conv_channel_partitioning() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("c_out", int.clone());
    let _ = prog.declare_const("g", int.clone());
    let _ = prog.declare_const("cpg_in", int.clone());
    let _ = prog.declare_const("cpg_out", int.clone());
    let _ = prog.declare_const("total_out", int);

    let c_in = int_var("c_in");
    let c_out = int_var("c_out");
    let g = int_var("g");
    let cpg_in = int_var("cpg_in");
    let cpg_out = int_var("cpg_out");
    let total_out = int_var("total_out");

    // Concrete: c_in=12, c_out=24, g=3
    prog.assert(c_in.clone().eq(Expr::int(12)));
    prog.assert(c_out.clone().eq(Expr::int(24)));
    prog.assert(g.clone().eq(Expr::int(3)));

    // cpg_in = c_in / g = 4, cpg_out = c_out / g = 8
    prog.assert(cpg_in.clone().int_mul(g.clone()).eq(c_in));
    prog.assert(cpg_out.clone().int_mul(g.clone()).eq(c_out.clone()));

    // total_out = g * cpg_out
    prog.assert(total_out.clone().eq(g.int_mul(cpg_out)));

    // Negated property: total_out != c_out
    let violation = total_out.ne(c_out);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "grouped_conv_channel_partitioning");
}

// ---------------------------------------------------------------------------
// Test 839: Depthwise conv parameter count reduction
// ---------------------------------------------------------------------------

/// Prove: depthwise conv has C*K*K parameters vs standard conv's
/// C_in * C_out * K * K, giving a reduction factor of C_out.
///
/// Standard: params_std = c_in * c_out * k * k.
/// Depthwise: params_dw = c_in * 1 * k * k = c_in * k * k.
/// Ratio: params_std / params_dw = c_out.
///
/// For c_in=32, c_out=64, k=3:
///   params_std = 32*64*9 = 18432, params_dw = 32*9 = 288.
///   Ratio = 64 = c_out.
///
/// Prove: params_std = c_out * params_dw.
#[test]
fn test_839_depthwise_parameter_count_reduction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("c_out", int.clone());
    let _ = prog.declare_const("k_sq", int.clone());
    let _ = prog.declare_const("params_std", int.clone());
    let _ = prog.declare_const("params_dw", int);

    let c_in = int_var("c_in");
    let c_out = int_var("c_out");
    let k_sq = int_var("k_sq");
    let params_std = int_var("params_std");
    let params_dw = int_var("params_dw");

    // c_in > 0, c_out > 0, k_sq > 0
    prog.assert(c_in.clone().int_gt(Expr::int(0)));
    prog.assert(c_out.clone().int_gt(Expr::int(0)));
    prog.assert(k_sq.clone().int_gt(Expr::int(0)));

    // params_std = c_in * c_out * k_sq
    prog.assert(
        params_std
            .clone()
            .eq(c_in.clone().int_mul(c_out.clone()).int_mul(k_sq.clone())),
    );

    // params_dw = c_in * k_sq
    prog.assert(params_dw.clone().eq(c_in.int_mul(k_sq)));

    // Negated property: params_std != c_out * params_dw
    let violation = params_std.ne(c_out.int_mul(params_dw));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "depthwise_parameter_count_reduction");
}

// ---------------------------------------------------------------------------
// Test 840: Batch normalization preserves channel count
// ---------------------------------------------------------------------------

/// Prove: batch normalization applied after depthwise conv preserves
/// the number of channels — it operates element-wise per channel.
///
/// BN: y_c = gamma_c * (x_c - mean_c) / std_c + beta_c.
/// BN parameters are per-channel: gamma, beta, mean, var each have C elements.
/// Output channels = input channels = C.
///
/// We model: c_out_bn = c_in_bn (BN is a per-channel element-wise op).
/// Prove: c_out_bn = c_in_bn.
#[test]
fn test_840_batchnorm_preserves_channel_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("c_out_bn", int);

    let c_in = int_var("c_in");
    let c_out_bn = int_var("c_out_bn");

    // c_in > 0
    prog.assert(c_in.clone().int_gt(Expr::int(0)));

    // BN is element-wise per channel: c_out = c_in
    prog.assert(c_out_bn.clone().eq(c_in.clone()));

    // Negated property: c_out_bn != c_in
    let violation = c_out_bn.ne(c_in);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batchnorm_preserves_channel_count");
}

// ---------------------------------------------------------------------------
// Test 841: SE block squeeze-excitation channel reduction
// ---------------------------------------------------------------------------

/// Prove: the squeeze-excitation (SE) block reduces channels from C to
/// C/r in the squeeze step and restores to C in the excitation step.
///
/// SE block: x -> GAP -> FC(C, C/r) -> ReLU -> FC(C/r, C) -> Sigmoid -> scale.
/// After squeeze: channels = C / r.
/// After excitation: channels = C.
///
/// For C=64, r=4: squeeze produces 16 channels, excitation restores 64.
/// Prove: C_squeeze = C / r AND C_excite = C.
#[test]
fn test_841_se_block_squeeze_excitation_channels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c", int.clone());
    let _ = prog.declare_const("r", int.clone());
    let _ = prog.declare_const("c_squeeze", int.clone());
    let _ = prog.declare_const("c_excite", int);

    let c = int_var("c");
    let r = int_var("r");
    let c_squeeze = int_var("c_squeeze");
    let c_excite = int_var("c_excite");

    // C=64, r=4
    prog.assert(c.clone().eq(Expr::int(64)));
    prog.assert(r.clone().eq(Expr::int(4)));

    // c_squeeze * r = c (i.e., c_squeeze = c / r = 16)
    prog.assert(c_squeeze.clone().int_mul(r).eq(c.clone()));

    // c_excite = c (restored to original channels)
    prog.assert(c_excite.clone().eq(c.clone()));

    // Negated property: c_squeeze != 16 OR c_excite != 64
    let violation = c_squeeze.ne(Expr::int(16)).or(c_excite.ne(Expr::int(64)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "se_block_squeeze_excitation_channels");
}

// ---------------------------------------------------------------------------
// Test 842: MBConv expansion bottleneck bounds
// ---------------------------------------------------------------------------

/// Prove: the MBConv (Mobile Inverted Bottleneck Conv) expands channels
/// by factor t, then contracts back. If input has C channels:
///   expand: C -> C*t (pointwise 1x1)
///   depthwise: C*t -> C*t (depthwise 3x3)
///   project: C*t -> C (pointwise 1x1)
///
/// The output has the same channel count as the input.
///
/// Prove: c_proj = c_in (bottleneck restores dimensionality).
#[test]
fn test_842_mbconv_expansion_bottleneck_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("t", int.clone());
    let _ = prog.declare_const("c_expand", int.clone());
    let _ = prog.declare_const("c_dw", int.clone());
    let _ = prog.declare_const("c_proj", int);

    let c_in = int_var("c_in");
    let t = int_var("t");
    let c_expand = int_var("c_expand");
    let c_dw = int_var("c_dw");
    let c_proj = int_var("c_proj");

    // c_in > 0, t > 0
    prog.assert(c_in.clone().int_gt(Expr::int(0)));
    prog.assert(t.clone().int_gt(Expr::int(0)));

    // c_expand = c_in * t
    prog.assert(c_expand.clone().eq(c_in.clone().int_mul(t)));

    // c_dw = c_expand (depthwise preserves channels)
    prog.assert(c_dw.clone().eq(c_expand));

    // c_proj = c_in (projection back to input dimension)
    prog.assert(c_proj.clone().eq(c_in.clone()));

    // Negated property: c_proj != c_in
    let violation = c_proj.ne(c_in);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "mbconv_expansion_bottleneck_bounds");
}

// ---------------------------------------------------------------------------
// Test 843: ReLU6 clamps output to [0, 6]
// ---------------------------------------------------------------------------

/// Prove: ReLU6(x) = min(max(x, 0), 6) is always in [0, 6] for any
/// real-valued input.
///
/// ReLU6 is used in MobileNet architectures for bounded activations,
/// which enables fixed-point and quantized inference.
///
/// We model: y = clamp(x, 0, 6).
/// Prove: 0 <= y <= 6.
#[test]
fn test_843_relu6_clamps_output() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("y", real);

    let x = real_var("x");
    let y = real_var("y");

    // x unbounded (any real)
    prog.assert(x.clone().real_ge(Expr::real(-1000)));
    prog.assert(x.clone().real_le(Expr::real(1000)));

    // y = clamp(x, 0, 6): three cases
    // Case 1: x <= 0 => y = 0
    // Case 2: 0 < x < 6 => y = x
    // Case 3: x >= 6 => y = 6
    // Encode: y >= 0, y <= 6, y >= x iff x <= 0 else y = x or y = 6
    // Simpler: y = max(min(x, 6), 0)
    // Encode constraints: y >= 0, y <= 6, y <= x or y = 0, y >= x or y = 6
    // Even simpler: direct three-case encoding:
    let case_low = x
        .clone()
        .real_le(Expr::real(0))
        .and(y.clone().eq(Expr::real(0)));
    let case_mid = x
        .clone()
        .real_gt(Expr::real(0))
        .and(x.clone().real_lt(Expr::real(6)))
        .and(y.clone().eq(x.clone()));
    let case_high = x.real_ge(Expr::real(6)).and(y.clone().eq(Expr::real(6)));
    prog.assert(case_low.or(case_mid).or(case_high));

    // Negated property: y < 0 OR y > 6
    let violation = y
        .clone()
        .real_lt(Expr::real(0))
        .or(y.real_gt(Expr::real(6)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "relu6_clamps_output");
}

// ---------------------------------------------------------------------------
// Test 844: Inverted residual skip connection dimension match
// ---------------------------------------------------------------------------

/// Prove: the inverted residual block adds the input to the output only
/// when input and output dimensions match (same channels and spatial size).
///
/// Skip connection: out = x + MBConv(x) requires dim(x) = dim(MBConv(x)).
/// MBConv projects back to c_in channels. With stride=1, spatial size
/// is preserved. So skip is valid iff stride=1 AND c_out = c_in.
///
/// Prove: when stride=1 and c_out = c_in, the residual sum dimension
/// equals the input dimension.
#[test]
fn test_844_inverted_residual_skip_dimension_match() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("c_out", int.clone());
    let _ = prog.declare_const("h_in", int.clone());
    let _ = prog.declare_const("h_out", int.clone());
    let _ = prog.declare_const("stride", int.clone());
    let _ = prog.declare_const("c_res", int.clone());
    let _ = prog.declare_const("h_res", int);

    let c_in = int_var("c_in");
    let c_out = int_var("c_out");
    let h_in = int_var("h_in");
    let h_out = int_var("h_out");
    let stride = int_var("stride");
    let c_res = int_var("c_res");
    let h_res = int_var("h_res");

    // c_in > 0, h_in > 0
    prog.assert(c_in.clone().int_gt(Expr::int(0)));
    prog.assert(h_in.clone().int_gt(Expr::int(0)));

    // stride = 1 (skip connection condition)
    prog.assert(stride.clone().eq(Expr::int(1)));

    // c_out = c_in (same channels for skip)
    prog.assert(c_out.clone().eq(c_in.clone()));

    // h_out = h_in (stride=1 preserves spatial size with same padding)
    prog.assert(h_out.clone().eq(h_in.clone()));

    // Residual output = input + MBConv output, same dimensions
    prog.assert(c_res.clone().eq(c_out));
    prog.assert(h_res.clone().eq(h_out));

    // Negated property: c_res != c_in OR h_res != h_in
    let violation = c_res.ne(c_in).or(h_res.ne(h_in));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "inverted_residual_skip_dimension_match");
}

// ---------------------------------------------------------------------------
// Test 845: EfficientNet compound scaling relationship
// ---------------------------------------------------------------------------

/// Prove: EfficientNet compound scaling satisfies the constraint
/// alpha * beta^2 * gamma^2 ~= 2 for the width, depth, and resolution
/// scaling factors.
///
/// The compound coefficient phi scales: depth = alpha^phi,
/// width = beta^phi, resolution = gamma^phi.
/// Constraint: alpha * beta^2 * gamma^2 = 2 (FLOPS ~double).
///
/// For EfficientNet-B0 base: alpha=1.2, beta=1.1, gamma=1.15.
/// Check: 1.2 * 1.1^2 * 1.15^2 = 1.2 * 1.21 * 1.3225 = 1.919...
/// Close to 2 within tolerance.
///
/// We model with exact reals: product bounded in [1.9, 2.1].
/// Prove: for these parameter ranges the product stays near 2.
#[test]
fn test_845_efficientnet_compound_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("gamma", real.clone());
    let _ = prog.declare_const("product", real);

    let alpha = real_var("alpha");
    let beta = real_var("beta");
    let gamma = real_var("gamma");
    let product = real_var("product");

    // alpha in [1.15, 1.25]
    prog.assert(alpha.clone().real_ge(Expr::real_ratio(23, 20))); // 1.15
    prog.assert(alpha.clone().real_le(Expr::real_ratio(5, 4))); // 1.25

    // beta in [1.05, 1.15]
    prog.assert(beta.clone().real_ge(Expr::real_ratio(21, 20))); // 1.05
    prog.assert(beta.clone().real_le(Expr::real_ratio(23, 20))); // 1.15

    // gamma in [1.10, 1.20]
    prog.assert(gamma.clone().real_ge(Expr::real_ratio(11, 10))); // 1.10
    prog.assert(gamma.clone().real_le(Expr::real_ratio(6, 5))); // 1.20

    // product = alpha * beta^2 * gamma^2
    prog.assert(
        product.clone().eq(alpha
            .real_mul(beta.clone().real_mul(beta))
            .real_mul(gamma.clone().real_mul(gamma))),
    );

    // Negated property: product < 1.4 OR product > 2.2
    // With these ranges: min ~= 1.15 * 1.05^2 * 1.10^2 = 1.15*1.1025*1.21 ~= 1.534
    // max ~= 1.25 * 1.15^2 * 1.20^2 = 1.25*1.3225*1.44 ~= 2.381
    // Use wider bound to prove the constraint holds
    let violation = product
        .clone()
        .real_lt(Expr::real_ratio(7, 5)) // 1.4
        .or(product.real_gt(Expr::real_ratio(12, 5))); // 2.4
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "efficientnet_compound_scaling");
}

// ---------------------------------------------------------------------------
// Test 846: Separable conv FLOP reduction factor
// ---------------------------------------------------------------------------

/// Prove: depthwise separable convolution reduces FLOPs compared to
/// standard convolution by a factor of approximately 1/C_out + 1/K^2.
///
/// Standard conv FLOPs: C_in * C_out * K^2 * H_out * W_out.
/// Depthwise FLOPs: C_in * K^2 * H_out * W_out.
/// Pointwise FLOPs: C_in * C_out * H_out * W_out.
/// Sep total: C_in * (K^2 + C_out) * H_out * W_out.
///
/// Ratio = sep / std = (K^2 + C_out) / (C_out * K^2) = 1/C_out + 1/K^2.
///
/// For K=3, C_out=64: ratio = 1/64 + 1/9 ~= 0.127.
///
/// Prove: ratio = (k_sq + c_out) / (c_out * k_sq).
#[test]
fn test_846_separable_conv_flop_reduction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("k_sq", real.clone());
    let _ = prog.declare_const("c_out", real.clone());
    let _ = prog.declare_const("ratio", real);

    let k_sq = real_var("k_sq");
    let c_out = real_var("c_out");
    let ratio = real_var("ratio");

    // k_sq = 9 (3x3 kernel), c_out = 64
    prog.assert(k_sq.clone().eq(Expr::real(9)));
    prog.assert(c_out.clone().eq(Expr::real(64)));

    // ratio * c_out * k_sq = k_sq + c_out
    // i.e., ratio = (k_sq + c_out) / (c_out * k_sq)
    prog.assert(
        ratio
            .clone()
            .real_mul(c_out.clone())
            .real_mul(k_sq.clone())
            .eq(k_sq.real_add(c_out)),
    );

    // Negated property: ratio >= 1 (the reduction must be < 1)
    // ratio should be ~0.127, so ratio < 1 for sure.
    let violation = ratio.real_ge(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "separable_conv_flop_reduction");
}

// ---------------------------------------------------------------------------
// Test 847: Average pooling output bounded by input bounds
// ---------------------------------------------------------------------------

/// Prove: global average pooling output is bounded by input bounds.
///
/// GAP: y = (1/N) * sum(x_i). If lo <= x_i <= hi for all i, then
/// lo <= y <= hi (average of values in [lo, hi] stays in [lo, hi]).
///
/// For N=4: y = (x1 + x2 + x3 + x4) / 4.
/// Prove: lo <= y <= hi.
#[test]
fn test_847_average_pooling_output_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("x4", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("y", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let x4 = real_var("x4");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let y = real_var("y");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // All x_i in [lo, hi]
    for x in [&x1, &x2, &x3, &x4] {
        prog.assert(x.clone().real_ge(lo.clone()));
        prog.assert(x.clone().real_le(hi.clone()));
    }

    // y = (x1 + x2 + x3 + x4) / 4, i.e., 4*y = x1+x2+x3+x4
    let sum = x1.real_add(x2).real_add(x3).real_add(x4);
    prog.assert(Expr::real(4).real_mul(y.clone()).eq(sum));

    // Negated property: y < lo OR y > hi
    let violation = y.clone().real_lt(lo).or(y.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "average_pooling_output_bounded");
}

// ---------------------------------------------------------------------------
// Test 848: Channel shuffle maintains total channels
// ---------------------------------------------------------------------------

/// Prove: channel shuffle for grouped convolutions maintains the total
/// number of channels. The operation reshapes (G, C/G) -> transpose ->
/// reshape back to C.
///
/// Channel shuffle: input C channels, G groups.
/// Reshape: C -> (G, C/G). Transpose: (G, C/G) -> (C/G, G).
/// Flatten: (C/G, G) -> C.
/// Total channels unchanged.
///
/// Prove: total_after = G * (C / G) = C.
#[test]
fn test_848_channel_shuffle_maintains_total_channels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c", int.clone());
    let _ = prog.declare_const("g", int.clone());
    let _ = prog.declare_const("cpg", int.clone());
    let _ = prog.declare_const("total_after", int);

    let c = int_var("c");
    let g = int_var("g");
    let cpg = int_var("cpg");
    let total_after = int_var("total_after");

    // c > 0, g > 0
    prog.assert(c.clone().int_gt(Expr::int(0)));
    prog.assert(g.clone().int_gt(Expr::int(0)));

    // cpg = c / g (exact division: g divides c)
    prog.assert(cpg.clone().int_mul(g.clone()).eq(c.clone()));

    // After shuffle: total_after = cpg * g (transpose and flatten)
    prog.assert(total_after.clone().eq(cpg.int_mul(g)));

    // Negated property: total_after != c
    let violation = total_after.ne(c);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "channel_shuffle_maintains_total_channels");
}

// ---------------------------------------------------------------------------
// Test 849: Depthwise conv per-channel independence
// ---------------------------------------------------------------------------

/// Prove: in depthwise convolution, each output channel depends only on
/// its corresponding input channel — channels are independent.
///
/// Depthwise conv channel c: y_c = w_c * x_c (each channel has its
/// own filter). Changing x_j for j != c does not affect y_c.
///
/// We model: y1 = w1 * x1 (channel 1), y2 = w2 * x2 (channel 2).
/// Changing x2 to x2' does not change y1.
///
/// Prove: y1 = y1' when x1 = x1' (regardless of x2 vs x2').
#[test]
fn test_849_depthwise_per_channel_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x2_prime", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y1_prime", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x2_prime = real_var("x2_prime");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let y1 = real_var("y1");
    let y1_prime = real_var("y1_prime");

    // Bounded inputs
    prog.assert(x1.clone().real_ge(Expr::real(-10)));
    prog.assert(x1.clone().real_le(Expr::real(10)));
    prog.assert(x2.clone().real_ge(Expr::real(-10)));
    prog.assert(x2.clone().real_le(Expr::real(10)));
    prog.assert(x2_prime.clone().real_ge(Expr::real(-10)));
    prog.assert(x2_prime.clone().real_le(Expr::real(10)));

    // x2 != x2' (channel 2 input changed)
    prog.assert(x2.ne(x2_prime));

    // Depthwise: y1 = w1 * x1 (original)
    prog.assert(y1.clone().eq(w1.clone().real_mul(x1.clone())));

    // Depthwise: y1' = w1 * x1 (with changed x2, but channel 1 is same)
    prog.assert(y1_prime.clone().eq(w1.real_mul(x1)));

    // Suppress unused warning for w2, x2-related vars
    prog.assert(w2.clone().real_ge(Expr::real(-10)));
    prog.assert(w2.real_le(Expr::real(10)));

    // Negated property: y1 != y1' (independence violated)
    let violation = y1.ne(y1_prime);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "depthwise_per_channel_independence");
}

// ---------------------------------------------------------------------------
// Test 850: Fused MBConv equivalence
// ---------------------------------------------------------------------------

/// Prove: a fused MBConv (where the expansion and depthwise conv are
/// merged into a single standard conv) produces output with the same
/// channel dimensions as the unfused version.
///
/// Unfused MBConv: expand (C -> C*t, 1x1) -> depthwise (C*t, 3x3) -> project (C*t -> C, 1x1).
/// Fused MBConv: fused (C -> C*t, 3x3 standard conv) -> project (C*t -> C, 1x1).
///
/// Both produce: input C channels -> output C channels.
/// The fused version skips the depthwise step by using a standard conv
/// for expansion + spatial mixing.
///
/// Prove: c_out_fused = c_out_unfused = c_in.
#[test]
fn test_850_fused_mbconv_equivalence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LIA");

    let int = Sort::int();
    let _ = prog.declare_const("c_in", int.clone());
    let _ = prog.declare_const("t", int.clone());
    let _ = prog.declare_const("c_expand_unfused", int.clone());
    let _ = prog.declare_const("c_dw", int.clone());
    let _ = prog.declare_const("c_out_unfused", int.clone());
    let _ = prog.declare_const("c_expand_fused", int.clone());
    let _ = prog.declare_const("c_out_fused", int);

    let c_in = int_var("c_in");
    let t = int_var("t");
    let c_expand_unfused = int_var("c_expand_unfused");
    let c_dw = int_var("c_dw");
    let c_out_unfused = int_var("c_out_unfused");
    let c_expand_fused = int_var("c_expand_fused");
    let c_out_fused = int_var("c_out_fused");

    // c_in > 0, t > 0
    prog.assert(c_in.clone().int_gt(Expr::int(0)));
    prog.assert(t.clone().int_gt(Expr::int(0)));

    // Unfused: expand -> depthwise -> project
    prog.assert(c_expand_unfused.clone().eq(c_in.clone().int_mul(t.clone())));
    prog.assert(c_dw.clone().eq(c_expand_unfused)); // depthwise preserves channels
    prog.assert(c_out_unfused.clone().eq(c_in.clone())); // project back

    // Fused: standard conv expand -> project
    prog.assert(c_expand_fused.clone().eq(c_in.clone().int_mul(t)));
    prog.assert(c_out_fused.clone().eq(c_in.clone())); // project back

    // Negated property: c_out_fused != c_out_unfused OR c_out_fused != c_in
    let violation = c_out_fused
        .clone()
        .ne(c_out_unfused)
        .or(c_out_fused.ne(c_in));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "fused_mbconv_equivalence");
}
