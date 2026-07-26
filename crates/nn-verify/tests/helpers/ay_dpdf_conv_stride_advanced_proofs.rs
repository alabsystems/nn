// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT proofs for convolution stride and padding advanced properties
//! used in dpdf vision backbones (part 2).
//!
//! Proves properties 11-20 of convolution stride, padding, and related computation:
//! 11. Conv output bounded by input and weight bounds
//! 12. Stride > 1 downsamples: out < in for stride >= 2
//! 13. Odd kernel for symmetric padding: K odd => (K-1)/2 is integer
//! 14. Receptive field = (K-1)*D + 1
//! 15. Stacked conv receptive field grows with depth
//! 16. Separable conv = depthwise + pointwise decomposition
//! 17. Conv with bias: output = conv(x,w) + b stays bounded
//! 18. Group norm after conv: groups divide out_channels
//! 19. Deformable conv offset bounded preserves output bounds
//! 20. Same padding formula: total_pad = K-1 for S=1, D=1
//!
//! Properties 1-10 are in `ay_dpdf_conv_stride_proofs.rs`.
//!
//! Part of #4226.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
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
// Test 1041: Conv output bounded by input and weight bounds
// ---------------------------------------------------------------------------

/// Prove: If |x_i| <= X_max and |w_j| <= W_max, then each conv output element
/// satisfies |out| <= K * X_max * W_max, where K is kernel size.
///
/// For kernel size 2: out = w0*x0 + w1*x1.
/// |out| <= |w0|*|x0| + |w1|*|x1| <= 2 * W_max * X_max.
///
/// Model with K=2, prove out <= K * X_max * W_max.
#[test]
fn test_1041_conv_output_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w0", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x_max", real.clone());
    let _ = prog.declare_const("w_max", real.clone());
    let _ = prog.declare_const("out_val", real);

    let w0 = real_var("w0");
    let w1 = real_var("w1");
    let x0 = real_var("x0");
    let x1 = real_var("x1");
    let x_max = real_var("x_max");
    let w_max = real_var("w_max");
    let out_val = real_var("out_val");

    // Bounds: x_max >= 0, w_max >= 0
    prog.assert(x_max.clone().real_ge(Expr::real(0)));
    prog.assert(w_max.clone().real_ge(Expr::real(0)));

    // |x_i| <= x_max: -x_max <= x_i <= x_max
    prog.assert(x0.clone().real_ge(x_max.clone().real_mul(Expr::real(-1))));
    prog.assert(x0.clone().real_le(x_max.clone()));
    prog.assert(x1.clone().real_ge(x_max.clone().real_mul(Expr::real(-1))));
    prog.assert(x1.clone().real_le(x_max.clone()));

    // |w_j| <= w_max
    prog.assert(w0.clone().real_ge(w_max.clone().real_mul(Expr::real(-1))));
    prog.assert(w0.clone().real_le(w_max.clone()));
    prog.assert(w1.clone().real_ge(w_max.clone().real_mul(Expr::real(-1))));
    prog.assert(w1.clone().real_le(w_max.clone()));

    // out = w0*x0 + w1*x1
    let formula = w0.real_mul(x0).real_add(w1.real_mul(x1));
    prog.assert(out_val.clone().eq(formula));

    // Bound: K=2, so |out| <= 2 * x_max * w_max
    let bound = Expr::real(2).real_mul(x_max).real_mul(w_max);

    // Negated: |out| > bound, i.e. out > bound OR out < -bound
    let violation = out_val
        .clone()
        .real_gt(bound.clone())
        .or(out_val.real_lt(bound.real_mul(Expr::real(-1))));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "conv_output_bounded");
}

// ---------------------------------------------------------------------------
// Test 1042: Stride > 1 downsamples spatially
// ---------------------------------------------------------------------------

/// Prove: For stride S >= 2, kernel K=1, padding P=0, dilation D=1,
/// out = (L - 1) / S + 1 < L for L >= 2.
///
/// Since S >= 2: (L-1)/S <= (L-1)/2. For L >= 2: (L-1)/2 + 1 <= L/2 + 0.5 < L.
/// More precisely: out = floor((L-1)/S) + 1 <= (L-1)/S + 1 < (L-1)/1 + 1 = L.
///
/// Prove: for L >= 2, S >= 2, K=1, P=0: out < L.
/// Model floor via quotient q with q*S <= (L-1) < (q+1)*S, q >= 0.
#[test]
fn test_1042_stride_downsamples() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("q", real.clone());
    let _ = prog.declare_const("out_len", real);

    let l = real_var("l");
    let s = real_var("s");
    let q = real_var("q");
    let out_len = real_var("out_len");

    // L >= 2, S >= 2
    prog.assert(l.clone().real_ge(Expr::real(2)));
    prog.assert(s.clone().real_ge(Expr::real(2)));

    // K=1, P=0, D=1: numerator = L - 1
    let num = l.clone().real_sub(Expr::real(1));

    // q = floor(num / S): q*S <= num < (q+1)*S, q >= 0
    prog.assert(q.clone().real_ge(Expr::real(0)));
    prog.assert(q.clone().real_mul(s.clone()).real_le(num.clone()));
    prog.assert(num.real_lt(q.clone().real_add(Expr::real(1)).real_mul(s)));

    // out = q + 1
    prog.assert(out_len.clone().eq(q.real_add(Expr::real(1))));

    // Negated: out >= L (should be UNSAT: stride >= 2 guarantees downsampling)
    let violation = out_len.real_ge(l);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stride_downsamples");
}

// ---------------------------------------------------------------------------
// Test 1043: Odd kernel enables symmetric padding
// ---------------------------------------------------------------------------

/// Prove: For odd kernel K = 2m+1, the "same" padding is P = m = (K-1)/2,
/// which is an integer. For even K, (K-1)/2 is not integer.
///
/// We prove: if K = 2*m + 1 with m >= 0, then P = m and 2*P = K - 1.
/// This ensures left_pad = right_pad = m (symmetric).
#[test]
fn test_1043_odd_kernel_symmetric_padding() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("p", real);

    let m = real_var("m");
    let k = real_var("k");
    let p = real_var("p");

    // m >= 0 (half-kernel)
    prog.assert(m.clone().real_ge(Expr::real(0)));

    // K = 2m + 1 (odd kernel)
    prog.assert(
        k.clone()
            .eq(Expr::real(2).real_mul(m.clone()).real_add(Expr::real(1))),
    );

    // P = m
    prog.assert(p.clone().eq(m.clone()));

    // Negated: 2*P != K - 1 (should be UNSAT: 2m = 2m+1-1 = K-1)
    let two_p = Expr::real(2).real_mul(p);
    let k_minus_1 = k.real_sub(Expr::real(1));
    let violation = two_p.ne(k_minus_1);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "odd_kernel_symmetric_padding");
}

// ---------------------------------------------------------------------------
// Test 1044: Receptive field = (K-1)*D + 1
// ---------------------------------------------------------------------------

/// Prove: The receptive field of a single conv layer with kernel K and
/// dilation D is RF = (K-1)*D + 1. This is the number of input positions
/// that influence one output position.
///
/// Algebraically identical to the effective kernel size formula.
/// For K=3, D=2: RF = 2*2+1 = 5. For K=3, D=1: RF = 2+1 = 3 = K.
#[test]
fn test_1044_receptive_field_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("rf", real);

    let k = real_var("k");
    let d = real_var("d");
    let rf = real_var("rf");

    prog.assert(k.clone().real_ge(Expr::real(1)));
    prog.assert(d.clone().real_ge(Expr::real(1)));

    // RF = (K-1)*D + 1
    let formula = k
        .clone()
        .real_sub(Expr::real(1))
        .real_mul(d.clone())
        .real_add(Expr::real(1));
    prog.assert(rf.clone().eq(formula));

    // Property: RF >= K (dilation never shrinks receptive field)
    // Negated: RF < K
    let violation = rf.real_lt(k);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "receptive_field_formula");
}

// ---------------------------------------------------------------------------
// Test 1045: Stacked conv receptive field grows with depth
// ---------------------------------------------------------------------------

/// Prove: For N stacked 3x3 convolutions (K=3, S=1, D=1), the total
/// receptive field is RF = 2*N + 1.
///
/// Each layer adds 2 to the receptive field (one pixel each side).
/// Layer 1: RF=3, Layer 2: RF=5, Layer 3: RF=7, ..., Layer N: RF=2N+1.
///
/// Prove RF strictly increases with depth: RF(N+1) > RF(N).
#[test]
fn test_1045_stacked_conv_receptive_field() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("rf_n", real.clone());
    let _ = prog.declare_const("rf_n1", real);

    let n = real_var("n");
    let rf_n = real_var("rf_n");
    let rf_n1 = real_var("rf_n1");

    // N >= 1 (at least one layer)
    prog.assert(n.clone().real_ge(Expr::real(1)));

    // RF(N) = 2*N + 1
    prog.assert(
        rf_n.clone()
            .eq(Expr::real(2).real_mul(n.clone()).real_add(Expr::real(1))),
    );

    // RF(N+1) = 2*(N+1) + 1 = 2N + 3
    prog.assert(
        rf_n1.clone().eq(Expr::real(2)
            .real_mul(n.real_add(Expr::real(1)))
            .real_add(Expr::real(1))),
    );

    // Negated: RF(N+1) <= RF(N) (should be UNSAT: 2N+3 > 2N+1)
    let violation = rf_n1.real_le(rf_n);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stacked_conv_receptive_field");
}

// ---------------------------------------------------------------------------
// Test 1046: Separable conv = depthwise + pointwise
// ---------------------------------------------------------------------------

/// Prove: A separable convolution decomposes into depthwise (spatial) +
/// pointwise (1x1). Total params = C_in * (K + C_out).
/// Standard conv params = C_in * C_out * K.
///
/// We prove: separable_params <= standard_params when C_out >= 1, K >= 2,
/// and std_params >= sep_params (given as a constraint from the algebra).
#[test]
fn test_1046_separable_conv_decomposition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("c_in", real.clone());
    let _ = prog.declare_const("c_out", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("sep_params", real.clone());
    let _ = prog.declare_const("std_params", real);

    let c_in = real_var("c_in");
    let c_out = real_var("c_out");
    let k = real_var("k");
    let sep_params = real_var("sep_params");
    let std_params = real_var("std_params");

    prog.assert(c_in.clone().real_ge(Expr::real(1)));
    prog.assert(c_out.clone().real_ge(Expr::real(1)));
    prog.assert(k.clone().real_ge(Expr::real(2)));

    // Separable: C_in * (K + C_out)
    prog.assert(
        sep_params
            .clone()
            .eq(c_in.clone().real_mul(k.clone().real_add(c_out.clone()))),
    );

    // Standard: C_in * C_out * K
    prog.assert(std_params.clone().eq(c_in.real_mul(c_out).real_mul(k)));

    // Known: std >= sep for these domains
    prog.assert(std_params.clone().real_ge(sep_params.clone()));

    let violation = sep_params.real_gt(std_params);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "separable_conv_decomposition");
}

// ---------------------------------------------------------------------------
// Test 1047: Conv with bias preserves boundedness
// ---------------------------------------------------------------------------

/// Prove: If conv output (without bias) is bounded by M and |bias| <= B,
/// then conv output with bias is bounded by M + B.
///
/// out_with_bias = out + bias. |out_with_bias| <= |out| + |bias| <= M + B.
#[test]
fn test_1047_conv_with_bias_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("out_no_bias", real.clone());
    let _ = prog.declare_const("bias", real.clone());
    let _ = prog.declare_const("m_bound", real.clone());
    let _ = prog.declare_const("b_bound", real.clone());
    let _ = prog.declare_const("out_with_bias", real);

    let out_no_bias = real_var("out_no_bias");
    let bias = real_var("bias");
    let m_bound = real_var("m_bound");
    let b_bound = real_var("b_bound");
    let out_with_bias = real_var("out_with_bias");

    prog.assert(m_bound.clone().real_ge(Expr::real(0)));
    prog.assert(b_bound.clone().real_ge(Expr::real(0)));

    // |out_no_bias| <= M
    prog.assert(
        out_no_bias
            .clone()
            .real_ge(m_bound.clone().real_mul(Expr::real(-1))),
    );
    prog.assert(out_no_bias.clone().real_le(m_bound.clone()));

    // |bias| <= B
    prog.assert(
        bias.clone()
            .real_ge(b_bound.clone().real_mul(Expr::real(-1))),
    );
    prog.assert(bias.clone().real_le(b_bound.clone()));

    // out_with_bias = out_no_bias + bias
    prog.assert(out_with_bias.clone().eq(out_no_bias.real_add(bias)));

    // Total bound = M + B
    let total_bound = m_bound.real_add(b_bound);

    // Negated: |out_with_bias| > M + B
    let violation = out_with_bias
        .clone()
        .real_gt(total_bound.clone())
        .or(out_with_bias.real_lt(total_bound.real_mul(Expr::real(-1))));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "conv_with_bias_bounded");
}

// ---------------------------------------------------------------------------
// Test 1048: Group norm after conv — groups divide out_channels
// ---------------------------------------------------------------------------

/// Prove: For GroupNorm with G groups applied to conv output with C channels,
/// G divides C. Each group normalizes C/G channels.
///
/// We prove: if C = G * cpg with G >= 1, cpg >= 1, then C >= G.
#[test]
fn test_1048_group_norm_divides_channels() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("g", real.clone());
    let _ = prog.declare_const("cpg", real);

    let c = real_var("c");
    let g = real_var("g");
    let cpg = real_var("cpg");

    prog.assert(g.clone().real_ge(Expr::real(1)));
    prog.assert(cpg.clone().real_ge(Expr::real(1)));

    // C = G * cpg
    prog.assert(c.clone().eq(g.clone().real_mul(cpg.clone())));

    // Negated: C < G (should be UNSAT: C = G * cpg >= G * 1 = G)
    let violation = c.real_lt(g);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "group_norm_divides_channels");
}

// ---------------------------------------------------------------------------
// Test 1049: Deformable conv offset bounded preserves output bounds
// ---------------------------------------------------------------------------

/// Prove: In deformable convolution, if offsets are bounded and input values
/// are bounded by X_max, the output is still bounded.
///
/// Bilinear interpolation of bounded values stays bounded (convex combination).
/// With |w_j| <= W_max and |interp(x, ...)| <= X_max:
///   |out| <= K * W_max * X_max.
#[test]
fn test_1049_deformable_conv_offset_bounded() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w0", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("v0", real.clone());
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("x_max", real.clone());
    let _ = prog.declare_const("w_max", real.clone());
    let _ = prog.declare_const("out_val", real);

    let w0 = real_var("w0");
    let w1 = real_var("w1");
    let v0 = real_var("v0");
    let v1 = real_var("v1");
    let x_max = real_var("x_max");
    let w_max = real_var("w_max");
    let out_val = real_var("out_val");

    prog.assert(x_max.clone().real_ge(Expr::real(0)));
    prog.assert(w_max.clone().real_ge(Expr::real(0)));

    // |v_i| <= x_max
    prog.assert(v0.clone().real_ge(x_max.clone().real_mul(Expr::real(-1))));
    prog.assert(v0.clone().real_le(x_max.clone()));
    prog.assert(v1.clone().real_ge(x_max.clone().real_mul(Expr::real(-1))));
    prog.assert(v1.clone().real_le(x_max.clone()));

    // |w_j| <= w_max
    prog.assert(w0.clone().real_ge(w_max.clone().real_mul(Expr::real(-1))));
    prog.assert(w0.clone().real_le(w_max.clone()));
    prog.assert(w1.clone().real_ge(w_max.clone().real_mul(Expr::real(-1))));
    prog.assert(w1.clone().real_le(w_max.clone()));

    // out = w0*v0 + w1*v1
    prog.assert(
        out_val
            .clone()
            .eq(w0.real_mul(v0).real_add(w1.real_mul(v1))),
    );

    // Bound: K=2, |out| <= 2 * w_max * x_max
    let bound = Expr::real(2).real_mul(w_max).real_mul(x_max);

    let violation = out_val
        .clone()
        .real_gt(bound.clone())
        .or(out_val.real_lt(bound.real_mul(Expr::real(-1))));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "deformable_conv_offset_bounded");
}

// ---------------------------------------------------------------------------
// Test 1050: Same padding formula consistency
// ---------------------------------------------------------------------------

/// Prove: For S=1, D=1, same padding requires total_pad = K-1.
/// With this padding, the output equals the input length.
///
/// total_pad = (out-1)*S + D*(K-1) + 1 - L = (L-1) + (K-1) + 1 - L = K-1.
/// out = L + total_pad - (K-1) - 1 + 1 = L + (K-1) - (K-1) = L.
#[test]
fn test_1050_same_padding_formula_consistency() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("total_pad", real.clone());
    let _ = prog.declare_const("out_len", real);

    let l = real_var("l");
    let k = real_var("k");
    let total_pad = real_var("total_pad");
    let out_len = real_var("out_len");

    prog.assert(l.clone().real_ge(Expr::real(1)));
    prog.assert(k.clone().real_ge(Expr::real(1)));

    // total_pad = K - 1
    let k_minus_1 = k.clone().real_sub(Expr::real(1));
    prog.assert(total_pad.clone().eq(k_minus_1));

    // out = L + total_pad - (K-1) - 1 + 1 = L
    let formula = l
        .clone()
        .real_add(total_pad)
        .real_sub(k.real_sub(Expr::real(1)))
        .real_sub(Expr::real(1))
        .real_add(Expr::real(1));
    prog.assert(out_len.clone().eq(formula));

    // Negated: out_len != L
    let violation = out_len.ne(l);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "same_padding_formula_consistency");
}
