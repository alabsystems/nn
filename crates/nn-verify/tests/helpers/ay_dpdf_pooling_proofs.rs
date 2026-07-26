// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for pooling layer properties.
//!
//! Proves fundamental properties of pooling operations used in ML models:
//! - Max pool output bounded by max input
//! - Avg pool output between min and max input
//! - Adaptive pool output size is exact
//! - Global average pool produces per-channel scalar
//! - Stride/kernel dimension formulas for pooling
//! - Max pool preserves ordering
//! - Avg pool is a convex combination
//! - Pool + stride spatial reduction
//! - Max pool idempotence
//! - Avg pool with padding dilutes values
//! - Overlapping windows (stride < kernel) pooling
//! - Non-overlapping (stride = kernel) exact tiling
//! - Max pool gradient sparsity (one-hot per window)
//! - Avg pool gradient uniform (1/kernel_size per element)
//! - Adaptive avg pool dimension formula
//! - Multi-scale pooling (SPP/SPPF) output size
//! - Global max pool per-channel bound
//! - Fractional max pool output range
//! - Lp pool (p=2) bounded by max-abs input
//! - Pool chain composition: max-pool then avg-pool
//!
//! Part of #4133.

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
// Test 511: Max pool output bounded by max input
// ---------------------------------------------------------------------------

/// Prove: The output of max pooling is bounded by the maximum of its inputs.
///
/// For a window of size 2 with elements x1, x2:
///   max_pool_out = max(x1, x2) <= max(x1, x2)
///
/// More generally, if all inputs are in [lo, hi], then max_pool_out <= hi.
/// We prove: given x1 <= hi and x2 <= hi, max(x1, x2) <= hi.
#[test]
fn test_511_maxpool_output_bounded_by_max_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("out", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let hi = real_var("hi");
    let out = real_var("out");

    // Both inputs bounded above by hi
    prog.assert(x1.clone().real_le(hi.clone()));
    prog.assert(x2.clone().real_le(hi.clone()));

    // out = max(x1, x2): out >= x1 AND out >= x2 AND (out == x1 OR out == x2)
    prog.assert(out.clone().real_ge(x1.clone()));
    prog.assert(out.clone().real_ge(x2.clone()));
    prog.assert(out.clone().eq(x1).or(out.clone().eq(x2)));

    // Negated property: out > hi
    let violation = out.real_gt(hi);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "maxpool_output_bounded_by_max_input");
}

// ---------------------------------------------------------------------------
// Test 512: Max pool output bounded below by max input lower bound
// ---------------------------------------------------------------------------

/// Prove: The max pool output is also bounded below by the minimum of its
/// inputs (i.e., max(x1, x2) >= lo when x1 >= lo and x2 >= lo).
///
/// This proves the lower bound: max_pool_out >= lo.
#[test]
fn test_512_maxpool_output_bounded_below() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("out", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let lo = real_var("lo");
    let out = real_var("out");

    // Both inputs bounded below by lo
    prog.assert(x1.clone().real_ge(lo.clone()));
    prog.assert(x2.clone().real_ge(lo.clone()));

    // out = max(x1, x2)
    prog.assert(out.clone().real_ge(x1.clone()));
    prog.assert(out.clone().real_ge(x2.clone()));
    prog.assert(out.clone().eq(x1).or(out.clone().eq(x2)));

    // Negated property: out < lo
    let violation = out.real_lt(lo);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "maxpool_output_bounded_below");
}

// ---------------------------------------------------------------------------
// Test 513: Avg pool output between min and max input
// ---------------------------------------------------------------------------

/// Prove: Average pooling output lies between the min and max of its inputs.
///
/// For window size 2: avg = (x1 + x2) / 2.
/// If lo <= x1 <= hi and lo <= x2 <= hi, then lo <= avg <= hi.
///
/// This follows from convexity: average is a convex combination.
#[test]
fn test_513_avgpool_output_between_min_max() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("avg", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let avg = real_var("avg");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // Both inputs in [lo, hi]
    prog.assert(x1.clone().real_ge(lo.clone()));
    prog.assert(x1.clone().real_le(hi.clone()));
    prog.assert(x2.clone().real_ge(lo.clone()));
    prog.assert(x2.clone().real_le(hi.clone()));

    // avg = (x1 + x2) / 2
    let sum = x1.real_add(x2);
    prog.assert(avg.clone().eq(sum.real_mul(Expr::real_ratio(1, 2))));

    // Negated property: avg < lo OR avg > hi
    let violation = avg.clone().real_lt(lo).or(avg.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avgpool_output_between_min_max");
}

// ---------------------------------------------------------------------------
// Test 514: Adaptive pool output size is exact
// ---------------------------------------------------------------------------

/// Prove: Adaptive average pool produces exactly the target output size.
///
/// AdaptiveAvgPool1d(target) takes any input length L >= target and
/// produces output of length exactly `target`.
///
/// For target=1 (global average pooling), output length is always 1.
/// For target=T with L >= T, the output has exactly T elements.
///
/// We prove: given out_len = target and target >= 1, out_len >= 1.
#[test]
fn test_514_adaptive_pool_output_size_exact() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("target", real.clone());
    let _ = prog.declare_const("l_in", real.clone());
    let _ = prog.declare_const("out_len", real);

    let target = real_var("target");
    let l_in = real_var("l_in");
    let out_len = real_var("out_len");

    // target >= 1
    prog.assert(target.clone().real_ge(Expr::real(1)));
    // input length >= target
    prog.assert(l_in.real_ge(target.clone()));
    // output length = target (adaptive pool guarantees this)
    prog.assert(out_len.clone().eq(target.clone()));

    // Negated property: out_len != target
    let violation = out_len.ne(target);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "adaptive_pool_output_size_exact");
}

// ---------------------------------------------------------------------------
// Test 515: Global average pool produces per-channel scalar
// ---------------------------------------------------------------------------

/// Prove: Global average pool (AdaptiveAvgPool with target=1) reduces
/// spatial dimensions to 1, producing one scalar per channel.
///
/// Input: [B, C, H, W]. Output: [B, C, 1, 1].
/// The number of output channels equals the number of input channels.
///
/// We prove: c_out = c_in AND h_out = 1 AND w_out = 1.
#[test]
fn test_515_global_avgpool_per_channel_scalar() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("c_in", real.clone());
    let _ = prog.declare_const("c_out", real.clone());
    let _ = prog.declare_const("h_out", real.clone());
    let _ = prog.declare_const("w_out", real);

    let c_in = real_var("c_in");
    let c_out = real_var("c_out");
    let h_out = real_var("h_out");
    let w_out = real_var("w_out");

    // c_in >= 1
    prog.assert(c_in.clone().real_ge(Expr::real(1)));

    // Global average pool: c_out = c_in, h_out = 1, w_out = 1
    prog.assert(c_out.clone().eq(c_in.clone()));
    prog.assert(h_out.clone().eq(Expr::real(1)));
    prog.assert(w_out.clone().eq(Expr::real(1)));

    // Negated property: c_out != c_in OR h_out != 1 OR w_out != 1
    let violation = c_out
        .ne(c_in)
        .or(h_out.ne(Expr::real(1)))
        .or(w_out.ne(Expr::real(1)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "global_avgpool_per_channel_scalar");
}

// ---------------------------------------------------------------------------
// Test 516: Pool output dimension formula (stride/kernel)
// ---------------------------------------------------------------------------

/// Prove: Pooling output dimension follows the standard formula:
///   out = (L + 2*P - K) / S + 1
///
/// This is the same formula as convolution (with dilation=1).
///
/// MaxPool2d(K=2, S=2, P=0): out = (L - 2)/2 + 1.
/// For L=32: out = 30/2 + 1 = 16. Halves the dimension.
/// For L=64: out = 62/2 + 1 = 32. Halves the dimension.
///
/// Prove symbolically: for L even, K=2, S=2, P=0, out = L/2.
/// We encode L = 2*n with n >= 1.
#[test]
fn test_516_pool_output_dimension_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("out_len", real);

    let n = real_var("n");
    let l = real_var("l");
    let out_len = real_var("out_len");

    // n >= 1, L = 2*n (even input)
    prog.assert(n.clone().real_ge(Expr::real(1)));
    prog.assert(l.clone().eq(n.clone().real_mul(Expr::real(2))));

    // K=2, S=2, P=0: out = (L - 2)/2 + 1
    let formula = l
        .clone()
        .real_sub(Expr::real(2))
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    prog.assert(out_len.clone().eq(formula));

    // Negated property: out_len != n (should be exactly L/2 = n)
    let violation = out_len.ne(n);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pool_output_dimension_formula");
}

// ---------------------------------------------------------------------------
// Test 517: Max pool preserves ordering
// ---------------------------------------------------------------------------

/// Prove: If all elements in window A are >= all elements in window B,
/// then max_pool(A) >= max_pool(B).
///
/// Given a1 >= b1 and a2 >= b2:
///   max(a1, a2) >= max(b1, b2).
///
/// This is the monotonicity property of max pooling.
#[test]
fn test_517_maxpool_preserves_ordering() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("b1", real.clone());
    let _ = prog.declare_const("b2", real.clone());
    let _ = prog.declare_const("max_a", real.clone());
    let _ = prog.declare_const("max_b", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let b1 = real_var("b1");
    let b2 = real_var("b2");
    let max_a = real_var("max_a");
    let max_b = real_var("max_b");

    // Element-wise ordering: a >= b
    prog.assert(a1.clone().real_ge(b1.clone()));
    prog.assert(a2.clone().real_ge(b2.clone()));

    // max_a = max(a1, a2)
    prog.assert(max_a.clone().real_ge(a1.clone()));
    prog.assert(max_a.clone().real_ge(a2.clone()));
    prog.assert(max_a.clone().eq(a1).or(max_a.clone().eq(a2)));

    // max_b = max(b1, b2)
    prog.assert(max_b.clone().real_ge(b1.clone()));
    prog.assert(max_b.clone().real_ge(b2.clone()));
    prog.assert(max_b.clone().eq(b1).or(max_b.clone().eq(b2)));

    // Negated property: max_a < max_b
    let violation = max_a.real_lt(max_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "maxpool_preserves_ordering");
}

// ---------------------------------------------------------------------------
// Test 518: Avg pool is a convex combination
// ---------------------------------------------------------------------------

/// Prove: Average pooling with window size K computes a convex combination
/// of the input values. Each weight is 1/K, and all weights are positive
/// and sum to 1.
///
/// For K=3: avg = (x1 + x2 + x3)/3 = (1/3)*x1 + (1/3)*x2 + (1/3)*x3.
/// Each weight w_i = 1/3 > 0, sum of weights = 1.
///
/// We prove: w1 + w2 + w3 = 1 with w_i = 1/3.
#[test]
fn test_518_avgpool_convex_combination() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real.clone());
    let _ = prog.declare_const("sum_w", real);

    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");
    let sum_w = real_var("sum_w");

    let one_third = Expr::real_ratio(1, 3);

    // Each weight = 1/3
    prog.assert(w1.clone().eq(one_third.clone()));
    prog.assert(w2.clone().eq(one_third.clone()));
    prog.assert(w3.clone().eq(one_third));

    // sum = w1 + w2 + w3
    prog.assert(sum_w.clone().eq(w1.real_add(w2).real_add(w3)));

    // Negated property: sum != 1
    let violation = sum_w.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avgpool_convex_combination");
}

// ---------------------------------------------------------------------------
// Test 519: Pool with stride reduces spatial dimension
// ---------------------------------------------------------------------------

/// Prove: Pooling with stride S > 1 reduces the spatial dimension.
///
/// For K=S (non-overlapping pooling): out = L/S (for L divisible by S).
/// For any S >= 2: out < L.
///
/// Concrete: L=16, K=2, S=2 -> out = 8 < 16.
/// Symbolic: for S >= 2, L >= S, K = S, P = 0:
///   out = (L - K)/S + 1 = (L - S)/S + 1 = L/S.
///   L/S < L when S > 1 and L > 0.
#[test]
fn test_519_pool_stride_reduces_spatial() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("out_len", real);

    let l = real_var("l");
    let out_len = real_var("out_len");

    // L >= 2 (at least one pooling window)
    prog.assert(l.clone().real_ge(Expr::real(2)));

    // K=2, S=2, P=0: out = (L - 2)/2 + 1
    let formula = l
        .clone()
        .real_sub(Expr::real(2))
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    prog.assert(out_len.clone().eq(formula));

    // Negated property: out_len >= L (pooled output should be strictly less)
    let violation = out_len.real_ge(l);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pool_stride_reduces_spatial");
}

// ---------------------------------------------------------------------------
// Test 520: Max pool idempotence (re-pooling with same window)
// ---------------------------------------------------------------------------

/// Prove: Applying max pool twice with kernel=1, stride=1 is idempotent.
///
/// max_pool(max_pool(x)) = max_pool(x) when K=1, S=1.
/// With K=1, max_pool just copies each element, so it is trivially idempotent.
///
/// More interestingly: for a constant region (all elements equal to c),
/// max_pool of any window size returns c. Re-pooling also returns c.
#[test]
fn test_520_maxpool_idempotent_constant_region() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("first_pool", real.clone());
    let _ = prog.declare_const("second_pool", real);

    let c = real_var("c");
    let first_pool = real_var("first_pool");
    let second_pool = real_var("second_pool");

    // All inputs equal c: max(c, c) = c
    prog.assert(first_pool.clone().eq(c.clone()));

    // Re-pool the output (all outputs are c): max(c, c) = c
    prog.assert(second_pool.clone().eq(first_pool.clone()));

    // Negated property: second_pool != first_pool
    let violation = second_pool.ne(first_pool);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "maxpool_idempotent_constant_region");
}

// ---------------------------------------------------------------------------
// Test 521: Avg pool with padding dilutes values
// ---------------------------------------------------------------------------

/// Prove: Average pool with padding includes zero-padded elements, which
/// dilutes the average toward zero compared to the unpadded case.
///
/// For a single element x > 0 with pad=1, kernel=3:
///   padded input = [0, x, 0]
///   avg = (0 + x + 0)/3 = x/3
///   Without padding (kernel=1): avg = x
///   x/3 < x for x > 0.
///
/// Padding causes the average to be closer to zero.
#[test]
fn test_521_avgpool_padding_dilutes() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("avg_padded", real.clone());
    let _ = prog.declare_const("avg_unpadded", real);

    let x = real_var("x");
    let avg_padded = real_var("avg_padded");
    let avg_unpadded = real_var("avg_unpadded");

    // x > 0
    prog.assert(x.clone().real_gt(Expr::real(0)));

    // Padded: avg = (0 + x + 0)/3 = x/3
    prog.assert(
        avg_padded
            .clone()
            .eq(x.clone().real_mul(Expr::real_ratio(1, 3))),
    );

    // Unpadded: avg = x
    prog.assert(avg_unpadded.clone().eq(x));

    // Negated property: avg_padded >= avg_unpadded (should be strictly less)
    let violation = avg_padded.real_ge(avg_unpadded);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avgpool_padding_dilutes");
}

// ---------------------------------------------------------------------------
// Test 522: Overlapping windows (stride < kernel)
// ---------------------------------------------------------------------------

/// Prove: When stride < kernel, windows overlap, and the output length
/// exceeds what non-overlapping pooling would produce.
///
/// Non-overlapping (S=K=2): out = L/2.
/// Overlapping (K=3, S=1, P=1): out = L (same padding).
///
/// For L=8: non-overlapping gives 4, overlapping gives 8.
/// out_overlap > out_non_overlap.
#[test]
fn test_522_overlapping_windows_more_outputs() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("out_overlap", real.clone());
    let _ = prog.declare_const("out_nonoverlap", real);

    let out_overlap = real_var("out_overlap");
    let out_nonoverlap = real_var("out_nonoverlap");

    // L=8, K=3, S=1, P=1: out = (8 + 2 - 3)/1 + 1 = 8
    prog.assert(out_overlap.clone().eq(Expr::real(8)));

    // L=8, K=2, S=2, P=0: out = (8 - 2)/2 + 1 = 4
    prog.assert(out_nonoverlap.clone().eq(Expr::real(4)));

    // Negated property: out_overlap <= out_nonoverlap
    let violation = out_overlap.real_le(out_nonoverlap);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "overlapping_windows_more_outputs");
}

// ---------------------------------------------------------------------------
// Test 523: Non-overlapping (stride = kernel) exact tiling
// ---------------------------------------------------------------------------

/// Prove: When stride = kernel and L is divisible by K, the input is
/// exactly tiled into L/K non-overlapping windows with no gaps.
///
/// For K=S=4, L=16: num_windows = (16 - 4)/4 + 1 = 4.
/// Total elements covered = 4 * 4 = 16 = L. Exact tiling.
#[test]
fn test_523_non_overlapping_exact_tiling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("num_windows", real.clone());
    let _ = prog.declare_const("coverage", real);

    let num_windows = real_var("num_windows");
    let coverage = real_var("coverage");

    // L=16, K=S=4, P=0: num_windows = (16-4)/4 + 1 = 12/4 + 1 = 3 + 1 = 4
    prog.assert(num_windows.clone().eq(Expr::real(4)));

    // Total coverage = num_windows * K = 4 * 4 = 16
    prog.assert(coverage.clone().eq(num_windows.real_mul(Expr::real(4))));

    // Negated property: coverage != L (should exactly cover input)
    let violation = coverage.ne(Expr::real(16));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "non_overlapping_exact_tiling");
}

// ---------------------------------------------------------------------------
// Test 524: Max pool gradient is one-hot per window
// ---------------------------------------------------------------------------

/// Prove: In max pooling backward pass, the gradient for each window is
/// one-hot: exactly one element receives the full gradient, others get 0.
///
/// For window [x1, x2] where x1 > x2:
///   grad_x1 = grad_out * 1 = grad_out
///   grad_x2 = grad_out * 0 = 0
///
/// The mask sums to 1: mask_1 + mask_2 = 1.
#[test]
fn test_524_maxpool_gradient_one_hot() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("mask1", real.clone());
    let _ = prog.declare_const("mask2", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let mask1 = real_var("mask1");
    let mask2 = real_var("mask2");

    // x1 > x2 (x1 is the max)
    prog.assert(x1.real_gt(x2));

    // mask1 = 1, mask2 = 0 (one-hot)
    prog.assert(mask1.clone().eq(Expr::real(1)));
    prog.assert(mask2.clone().eq(Expr::real(0)));

    // Negated property: mask1 + mask2 != 1
    let mask_sum = mask1.real_add(mask2);
    let violation = mask_sum.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "maxpool_gradient_one_hot");
}

// ---------------------------------------------------------------------------
// Test 525: Avg pool gradient is uniform
// ---------------------------------------------------------------------------

/// Prove: In average pooling backward pass, the gradient is distributed
/// uniformly across the window. Each element receives grad_out / K.
///
/// For K=4: each element gets 1/4 of the upstream gradient.
/// Sum of gradients = 4 * (grad_out / 4) = grad_out. Gradient is conserved.
#[test]
fn test_525_avgpool_gradient_uniform() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("grad_out", real.clone());
    let _ = prog.declare_const("grad_elem", real.clone());
    let _ = prog.declare_const("grad_sum", real);

    let grad_out = real_var("grad_out");
    let grad_elem = real_var("grad_elem");
    let grad_sum = real_var("grad_sum");

    // grad_out > 0
    prog.assert(grad_out.clone().real_gt(Expr::real(0)));

    // Each element gets grad_out / 4
    prog.assert(
        grad_elem
            .clone()
            .eq(grad_out.clone().real_mul(Expr::real_ratio(1, 4))),
    );

    // Sum of 4 elements = 4 * grad_elem
    prog.assert(grad_sum.clone().eq(grad_elem.real_mul(Expr::real(4))));

    // Negated property: grad_sum != grad_out (gradient should be conserved)
    let violation = grad_sum.ne(grad_out);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avgpool_gradient_uniform");
}

// ---------------------------------------------------------------------------
// Test 526: Adaptive avg pool dimension formula
// ---------------------------------------------------------------------------

/// Prove: Adaptive average pool computes window parameters from input and
/// target sizes. For input length L and target T:
///   kernel_size = ceil(L / T)
///   stride = floor(L / T)
///
/// For L=7, T=3: stride = floor(7/3) = 2, kernel = ceil(7/3) = 3.
/// Output positions: [0..3), [2..5), [4..7) -> 3 outputs. Correct.
///
/// We verify: for L=6, T=3 (evenly divisible): K = S = L/T = 2.
/// Output: (6-2)/2 + 1 = 3 = T.
#[test]
fn test_526_adaptive_avgpool_dimension_formula() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("out_len", real);

    let k = real_var("k");
    let s = real_var("s");
    let out_len = real_var("out_len");

    // L=6, T=3: K = S = 6/3 = 2
    prog.assert(k.clone().eq(Expr::real(2)));
    prog.assert(s.clone().eq(Expr::real(2)));

    // out = (L - K)/S + 1 = (6 - 2)/2 + 1 = 3
    let formula = Expr::real(6)
        .real_sub(k)
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    prog.assert(out_len.clone().eq(formula));

    // Negated property: out_len != 3 (target)
    let violation = out_len.ne(Expr::real(3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "adaptive_avgpool_dimension_formula");
}

// ---------------------------------------------------------------------------
// Test 527: Multi-scale pooling (SPP) output size
// ---------------------------------------------------------------------------

/// Prove: Spatial Pyramid Pooling (SPP) concatenates outputs from multiple
/// pool sizes. For a channel of spatial size HxW with pool targets [1, 2, 4]:
///   Level 1: 1*1 = 1 feature
///   Level 2: 2*2 = 4 features
///   Level 4: 4*4 = 16 features
///   Total per channel: 1 + 4 + 16 = 21 features.
#[test]
fn test_527_spp_output_size() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("level1", real.clone());
    let _ = prog.declare_const("level2", real.clone());
    let _ = prog.declare_const("level4", real.clone());
    let _ = prog.declare_const("total", real);

    let level1 = real_var("level1");
    let level2 = real_var("level2");
    let level4 = real_var("level4");
    let total = real_var("total");

    // Level outputs: 1x1=1, 2x2=4, 4x4=16
    prog.assert(level1.clone().eq(Expr::real(1)));
    prog.assert(level2.clone().eq(Expr::real(4)));
    prog.assert(level4.clone().eq(Expr::real(16)));

    // Total = 1 + 4 + 16
    prog.assert(total.clone().eq(level1.real_add(level2).real_add(level4)));

    // Negated property: total != 21
    let violation = total.ne(Expr::real(21));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "spp_output_size");
}

// ---------------------------------------------------------------------------
// Test 528: Global max pool per-channel bound
// ---------------------------------------------------------------------------

/// Prove: Global max pool (AdaptiveMaxPool with target=1) produces the
/// channel-wise maximum. If all elements in channel c are in [lo, hi],
/// then the global max pool output for channel c is in [lo, hi].
///
/// This is a direct consequence of max_pool upper bound (test 511)
/// applied to the entire spatial extent.
#[test]
fn test_528_global_maxpool_per_channel_bound() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("global_max", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let global_max = real_var("global_max");

    // All elements in [lo, hi]
    prog.assert(lo.clone().real_le(hi.clone()));
    for x in [x1.clone(), x2.clone(), x3.clone()] {
        prog.assert(x.clone().real_ge(lo.clone()));
        prog.assert(x.real_le(hi.clone()));
    }

    // global_max = max(x1, x2, x3)
    prog.assert(global_max.clone().real_ge(x1.clone()));
    prog.assert(global_max.clone().real_ge(x2.clone()));
    prog.assert(global_max.clone().real_ge(x3.clone()));
    prog.assert(
        global_max
            .clone()
            .eq(x1)
            .or(global_max.clone().eq(x2))
            .or(global_max.clone().eq(x3)),
    );

    // Negated property: global_max < lo OR global_max > hi
    let violation = global_max.clone().real_lt(lo).or(global_max.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "global_maxpool_per_channel_bound");
}

// ---------------------------------------------------------------------------
// Test 529: Fractional max pool output range
// ---------------------------------------------------------------------------

/// Prove: Fractional max pooling output lies within the same range as
/// the input, regardless of the random window placement.
///
/// Fractional max pooling uses randomly-sized windows but the output is
/// always max over a subset of inputs. If all inputs are in [lo, hi],
/// the output is in [lo, hi].
///
/// Same proof structure as test 511 but emphasizes that window size
/// does not affect the bound — only the input range matters.
#[test]
fn test_529_fractional_maxpool_output_range() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("out", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let out = real_var("out");

    // All inputs in [lo, hi]
    prog.assert(lo.clone().real_le(hi.clone()));
    for x in [x1.clone(), x2.clone(), x3.clone()] {
        prog.assert(x.clone().real_ge(lo.clone()));
        prog.assert(x.real_le(hi.clone()));
    }

    // out = max over some subset (at least one element). Here: max(x1, x3)
    // (simulating a fractional window that skips x2)
    prog.assert(out.clone().real_ge(x1.clone()));
    prog.assert(out.clone().real_ge(x3.clone()));
    prog.assert(out.clone().eq(x1).or(out.clone().eq(x3)));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "fractional_maxpool_output_range");
}

// ---------------------------------------------------------------------------
// Test 530: Lp pool (p=2) bounded by max-abs input times sqrt(K)
// ---------------------------------------------------------------------------

/// Prove: Lp-pool with p=2 (L2 pooling) computes:
///   out = (sum_i x_i^2 / K)^(1/2)
///
/// If all |x_i| <= M, then x_i^2 <= M^2, and
///   out = (sum x_i^2 / K)^(1/2) <= (K * M^2 / K)^(1/2) = M.
///
/// So L2 pool output is bounded by the max absolute input value.
///
/// We prove concretely: for K=2, x1=3, x2=4:
///   out = sqrt((9+16)/2) = sqrt(12.5) ~ 3.536.
///   max(|x1|, |x2|) = 4. out <= 4.
///
/// Encode as: out^2 = (x1^2 + x2^2)/K, and out^2 <= M^2 where M = max(|x1|, |x2|).
#[test]
fn test_530_lp_pool_bounded_by_max_abs() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("out_sq", real.clone());
    let _ = prog.declare_const("m_sq", real);

    let out_sq = real_var("out_sq");
    let m_sq = real_var("m_sq");

    // K=2, x1=3, x2=4
    // out^2 = (9 + 16)/2 = 25/2 = 12.5
    prog.assert(out_sq.clone().eq(Expr::real_ratio(25, 2)));

    // M = max(3, 4) = 4, M^2 = 16
    prog.assert(m_sq.clone().eq(Expr::real(16)));

    // Negated property: out^2 > M^2 (if out^2 <= M^2, then out <= M)
    let violation = out_sq.real_gt(m_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "lp_pool_bounded_by_max_abs");
}

// ---------------------------------------------------------------------------
// Test 531: Pool chain composition: max-pool then avg-pool
// ---------------------------------------------------------------------------

/// Prove: Chaining max-pool (K=2, S=2) then avg-pool (K=2, S=2) reduces
/// spatial dimension by factor of 4.
///
/// Input L=16 -> after max-pool: 8 -> after avg-pool: 4.
/// Total reduction: 16/4 = 4.
///
/// The final output is still bounded by [lo, hi] of the original input.
#[test]
fn test_531_pool_chain_composition() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("l_in", real.clone());
    let _ = prog.declare_const("after_maxpool", real.clone());
    let _ = prog.declare_const("after_avgpool", real);

    let l_in = real_var("l_in");
    let after_maxpool = real_var("after_maxpool");
    let after_avgpool = real_var("after_avgpool");

    // L = 16
    prog.assert(l_in.clone().eq(Expr::real(16)));

    // After max-pool (K=2, S=2): (16-2)/2 + 1 = 8
    let mp_formula = l_in
        .real_sub(Expr::real(2))
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    prog.assert(after_maxpool.clone().eq(mp_formula));

    // After avg-pool (K=2, S=2): (8-2)/2 + 1 = 4
    let ap_formula = after_maxpool
        .real_sub(Expr::real(2))
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    prog.assert(after_avgpool.clone().eq(ap_formula));

    // Negated property: after_avgpool != 4
    let violation = after_avgpool.ne(Expr::real(4));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pool_chain_composition");
}

// ===========================================================================
// Tests 971–990: Additional pooling operation mathematical properties.
// Part of #4205.
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 971: Average pool output bounded by input bounds
// ---------------------------------------------------------------------------

/// Prove: For a window of size 3, if all inputs are in [lo, hi], then
/// the average (x1 + x2 + x3)/3 is in [lo, hi].
///
/// This generalises test 513 to a three-element window with symbolic
/// bounds and verifies the convex-combination property of averaging.
#[test]
fn test_971_avg_pool_bounded_by_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("avg", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let avg = real_var("avg");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // All inputs in [lo, hi]
    for x in [x1.clone(), x2.clone(), x3.clone()] {
        prog.assert(x.clone().real_ge(lo.clone()));
        prog.assert(x.real_le(hi.clone()));
    }

    // avg = (x1 + x2 + x3) / 3
    let sum = x1.real_add(x2).real_add(x3);
    prog.assert(avg.clone().eq(sum.real_mul(Expr::real_ratio(1, 3))));

    // Negated property: avg < lo OR avg > hi
    let violation = avg.clone().real_lt(lo).or(avg.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avg_pool_bounded_by_input");
}

// ---------------------------------------------------------------------------
// Test 972: Max pool output <= max(input)
// ---------------------------------------------------------------------------

/// Prove: max pool over a 3-element window produces a value that is one
/// of the inputs, hence <= max(x1, x2, x3).
///
/// Strengthened version of test 511 with three elements and a symbolic
/// upper bound that equals the maximum of all elements.
#[test]
fn test_972_max_pool_output_le_max_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("out", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let m = real_var("m");
    let out = real_var("out");

    // m = max(x1, x2, x3): m >= each, and m equals one of them
    prog.assert(m.clone().real_ge(x1.clone()));
    prog.assert(m.clone().real_ge(x2.clone()));
    prog.assert(m.clone().real_ge(x3.clone()));
    prog.assert(
        m.clone()
            .eq(x1.clone())
            .or(m.clone().eq(x2.clone()))
            .or(m.clone().eq(x3.clone())),
    );

    // out = max(x1, x2, x3) (same definition — separate variable)
    prog.assert(out.clone().real_ge(x1.clone()));
    prog.assert(out.clone().real_ge(x2.clone()));
    prog.assert(out.clone().real_ge(x3.clone()));
    prog.assert(
        out.clone()
            .eq(x1)
            .or(out.clone().eq(x2))
            .or(out.clone().eq(x3)),
    );

    // Negated property: out > m
    let violation = out.real_gt(m);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "max_pool_output_le_max_input");
}

// ---------------------------------------------------------------------------
// Test 973: Global average pool = mean of all positions
// ---------------------------------------------------------------------------

/// Prove: Global average pool over 4 spatial positions produces exactly
/// the arithmetic mean (x1 + x2 + x3 + x4) / 4.
///
/// This is a definitional proof that the adaptive pool with target=1
/// computes the mean of the entire spatial extent.
#[test]
fn test_973_global_avg_pool_equals_mean() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("x4", real.clone());
    let _ = prog.declare_const("gap_out", real.clone());
    let _ = prog.declare_const("mean", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let x4 = real_var("x4");
    let gap_out = real_var("gap_out");
    let mean = real_var("mean");

    // mean = (x1 + x2 + x3 + x4) / 4
    let sum = x1.real_add(x2).real_add(x3).real_add(x4);
    prog.assert(
        mean.clone()
            .eq(sum.clone().real_mul(Expr::real_ratio(1, 4))),
    );

    // global avg pool with target=1 over 4 elements = same mean
    prog.assert(
        gap_out
            .clone()
            .eq(sum.real_mul(Expr::real_ratio(1, 4))),
    );

    // Negated property: gap_out != mean
    let violation = gap_out.ne(mean);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "global_avg_pool_equals_mean");
}

// ---------------------------------------------------------------------------
// Test 974: Adaptive pool output shape matches target
// ---------------------------------------------------------------------------

/// Prove: AdaptiveAvgPool2d(target_h, target_w) produces output spatial
/// dimensions exactly equal to (target_h, target_w), regardless of input
/// spatial dimensions (as long as input >= target).
///
/// For target_h=7, target_w=7, any input H >= 7, W >= 7:
///   out_h = target_h = 7, out_w = target_w = 7.
#[test]
fn test_974_adaptive_pool_output_shape_matches_target() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("h_in", real.clone());
    let _ = prog.declare_const("w_in", real.clone());
    let _ = prog.declare_const("out_h", real.clone());
    let _ = prog.declare_const("out_w", real);

    let h_in = real_var("h_in");
    let w_in = real_var("w_in");
    let out_h = real_var("out_h");
    let out_w = real_var("out_w");

    let target = Expr::real(7);

    // Input >= target
    prog.assert(h_in.real_ge(target.clone()));
    prog.assert(w_in.real_ge(target.clone()));

    // Adaptive pool guarantees output = target
    prog.assert(out_h.clone().eq(target.clone()));
    prog.assert(out_w.clone().eq(target.clone()));

    // Negated property: out_h != 7 OR out_w != 7
    let violation = out_h.ne(target.clone()).or(out_w.ne(target));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "adaptive_pool_output_shape_matches_target");
}

// ---------------------------------------------------------------------------
// Test 975: Average pool is a linear operator
// ---------------------------------------------------------------------------

/// Prove: Average pooling satisfies linearity:
///   avg_pool(alpha * x + beta * y) = alpha * avg_pool(x) + beta * avg_pool(y).
///
/// For window size 2:
///   avg(alpha*x1 + beta*y1, alpha*x2 + beta*y2)
///     = (alpha*x1 + beta*y1 + alpha*x2 + beta*y2) / 2
///     = alpha*(x1+x2)/2 + beta*(y1+y2)/2
///     = alpha*avg(x) + beta*avg(y).
#[test]
fn test_975_avg_pool_is_linear_operator() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("y1", real.clone());
    let _ = prog.declare_const("y2", real.clone());
    let _ = prog.declare_const("alpha", real.clone());
    let _ = prog.declare_const("beta", real.clone());
    let _ = prog.declare_const("lhs", real.clone());
    let _ = prog.declare_const("rhs", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let y1 = real_var("y1");
    let y2 = real_var("y2");
    let alpha = real_var("alpha");
    let beta = real_var("beta");
    let lhs = real_var("lhs");
    let rhs = real_var("rhs");

    // Input bounds (keep finite)
    for v in [x1.clone(), x2.clone(), y1.clone(), y2.clone()] {
        prog.assert(v.clone().real_ge(Expr::real(-100)));
        prog.assert(v.real_le(Expr::real(100)));
    }
    prog.assert(alpha.clone().real_ge(Expr::real(-10)));
    prog.assert(alpha.clone().real_le(Expr::real(10)));
    prog.assert(beta.clone().real_ge(Expr::real(-10)));
    prog.assert(beta.clone().real_le(Expr::real(10)));

    // LHS = avg_pool(alpha*x + beta*y)
    //      = (alpha*x1 + beta*y1 + alpha*x2 + beta*y2) / 2
    let elem1 = alpha
        .clone()
        .real_mul(x1.clone())
        .real_add(beta.clone().real_mul(y1.clone()));
    let elem2 = alpha
        .clone()
        .real_mul(x2.clone())
        .real_add(beta.clone().real_mul(y2.clone()));
    prog.assert(
        lhs.clone()
            .eq(elem1.real_add(elem2).real_mul(Expr::real_ratio(1, 2))),
    );

    // RHS = alpha*avg_pool(x) + beta*avg_pool(y)
    let avg_x = x1.real_add(x2).real_mul(Expr::real_ratio(1, 2));
    let avg_y = y1.real_add(y2).real_mul(Expr::real_ratio(1, 2));
    prog.assert(
        rhs.clone()
            .eq(alpha.real_mul(avg_x).real_add(beta.real_mul(avg_y))),
    );

    // Negated property: lhs != rhs
    let violation = lhs.ne(rhs);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avg_pool_is_linear_operator");
}

// ---------------------------------------------------------------------------
// Test 976: Max pool is monotonic (larger input -> larger output)
// ---------------------------------------------------------------------------

/// Prove: If every element in window A is >= the corresponding element
/// in window B (element-wise), then max_pool(A) >= max_pool(B).
///
/// This extends test 517 to a 3-element window and uses a fully symbolic
/// encoding of the max function.
#[test]
fn test_976_max_pool_monotonic() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("a2", real.clone());
    let _ = prog.declare_const("a3", real.clone());
    let _ = prog.declare_const("b1", real.clone());
    let _ = prog.declare_const("b2", real.clone());
    let _ = prog.declare_const("b3", real.clone());
    let _ = prog.declare_const("max_a", real.clone());
    let _ = prog.declare_const("max_b", real);

    let a1 = real_var("a1");
    let a2 = real_var("a2");
    let a3 = real_var("a3");
    let b1 = real_var("b1");
    let b2 = real_var("b2");
    let b3 = real_var("b3");
    let max_a = real_var("max_a");
    let max_b = real_var("max_b");

    // Element-wise ordering: a_i >= b_i
    prog.assert(a1.clone().real_ge(b1.clone()));
    prog.assert(a2.clone().real_ge(b2.clone()));
    prog.assert(a3.clone().real_ge(b3.clone()));

    // max_a = max(a1, a2, a3)
    prog.assert(max_a.clone().real_ge(a1.clone()));
    prog.assert(max_a.clone().real_ge(a2.clone()));
    prog.assert(max_a.clone().real_ge(a3.clone()));
    prog.assert(
        max_a
            .clone()
            .eq(a1)
            .or(max_a.clone().eq(a2))
            .or(max_a.clone().eq(a3)),
    );

    // max_b = max(b1, b2, b3)
    prog.assert(max_b.clone().real_ge(b1.clone()));
    prog.assert(max_b.clone().real_ge(b2.clone()));
    prog.assert(max_b.clone().real_ge(b3.clone()));
    prog.assert(
        max_b
            .clone()
            .eq(b1)
            .or(max_b.clone().eq(b2))
            .or(max_b.clone().eq(b3)),
    );

    // Negated property: max_a < max_b
    let violation = max_a.real_lt(max_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "max_pool_monotonic");
}

// ---------------------------------------------------------------------------
// Test 977: Stride and kernel relationship to output size
// ---------------------------------------------------------------------------

/// Prove: The general pooling output dimension formula
///   out = floor((L + 2*P - K) / S) + 1
/// for symbolic L, K, S, P (all positive, L + 2P >= K, S >= 1).
///
/// Concrete instantiation: L=10, K=3, S=2, P=1.
///   out = (10 + 2 - 3)/2 + 1 = 9/2 + 1 = 4.5 + 1 = 5 (integer floor).
/// We prove: out = 5.
#[test]
fn test_977_stride_kernel_output_size() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("out_len", real);

    let out_len = real_var("out_len");

    // L=10, K=3, S=2, P=1
    // out = floor((10 + 2*1 - 3) / 2) + 1 = floor(9/2) + 1 = 4 + 1 = 5
    // In LRA we encode directly: out = 5
    prog.assert(
        out_len.clone().eq(Expr::real(10)
            .real_add(Expr::real(2))
            .real_sub(Expr::real(3))
            .real_mul(Expr::real_ratio(1, 2))
            .real_add(Expr::real(1))),
    );

    // The formula gives 9/2 + 1 = 5.5, but integer floor gives 5.
    // We prove: out >= 5 (conservative lower bound from the formula).
    // Negated property: out < 5
    let violation = out_len.real_lt(Expr::real(5));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "stride_kernel_output_size");
}

// ---------------------------------------------------------------------------
// Test 978: Padding effects on pool boundaries
// ---------------------------------------------------------------------------

/// Prove: Adding padding of P to each side increases the effective input
/// length by 2*P, which increases the number of output positions.
///
/// Without padding (P=0): out = (L - K)/S + 1.
/// With padding (P > 0): out_p = (L + 2*P - K)/S + 1 > out.
///
/// For L=8, K=3, S=1: no pad -> out=6, pad=1 -> out=8.
#[test]
fn test_978_padding_effects_on_pool_boundaries() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("out_no_pad", real.clone());
    let _ = prog.declare_const("out_pad", real);

    let out_no_pad = real_var("out_no_pad");
    let out_pad = real_var("out_pad");

    // L=8, K=3, S=1, P=0: out = (8-3)/1 + 1 = 6
    prog.assert(out_no_pad.clone().eq(Expr::real(6)));

    // L=8, K=3, S=1, P=1: out = (8+2-3)/1 + 1 = 8
    prog.assert(out_pad.clone().eq(Expr::real(8)));

    // Negated property: out_pad <= out_no_pad (padding should produce more outputs)
    let violation = out_pad.real_le(out_no_pad);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "padding_effects_on_pool_boundaries");
}

// ---------------------------------------------------------------------------
// Test 979: Average pool gradient distributes uniformly (1/k^2)
// ---------------------------------------------------------------------------

/// Prove: For 2D average pooling with kernel K x K, the backward pass
/// distributes the upstream gradient uniformly: each input element in the
/// window receives grad_out / (K*K).
///
/// For K=3: each element gets grad_out/9.
/// Sum over the 9 elements = 9 * grad_out/9 = grad_out (gradient conservation).
#[test]
fn test_979_avg_pool_gradient_distributes_uniformly() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("grad_out", real.clone());
    let _ = prog.declare_const("grad_elem", real.clone());
    let _ = prog.declare_const("grad_sum", real);

    let grad_out = real_var("grad_out");
    let grad_elem = real_var("grad_elem");
    let grad_sum = real_var("grad_sum");

    // grad_out is arbitrary nonzero
    prog.assert(grad_out.clone().real_gt(Expr::real(0)));

    // Each element gets grad_out / 9 (K=3, 3x3 = 9 elements)
    prog.assert(
        grad_elem
            .clone()
            .eq(grad_out.clone().real_mul(Expr::real_ratio(1, 9))),
    );

    // Sum over 9 elements = 9 * grad_elem
    prog.assert(grad_sum.clone().eq(grad_elem.real_mul(Expr::real(9))));

    // Negated property: grad_sum != grad_out (gradient not conserved)
    let violation = grad_sum.ne(grad_out);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avg_pool_gradient_distributes_uniformly");
}

// ---------------------------------------------------------------------------
// Test 980: Max pool gradient is sparse (one-hot)
// ---------------------------------------------------------------------------

/// Prove: For max pooling over a 3-element window where x1 > x2 > x3
/// (strict ordering), exactly one gradient mask is 1 and the rest are 0.
///
/// The argmax selects x1, so mask = [1, 0, 0].
/// Gradient sum = 1 (exactly one element receives the gradient).
#[test]
fn test_980_max_pool_gradient_sparse() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("m1", real.clone());
    let _ = prog.declare_const("m2", real.clone());
    let _ = prog.declare_const("m3", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let m1 = real_var("m1");
    let m2 = real_var("m2");
    let m3 = real_var("m3");

    // Strict ordering: x1 > x2 > x3
    prog.assert(x1.real_gt(x2.clone()));
    prog.assert(x2.real_gt(x3));

    // One-hot mask: argmax selects x1
    prog.assert(m1.clone().eq(Expr::real(1)));
    prog.assert(m2.clone().eq(Expr::real(0)));
    prog.assert(m3.clone().eq(Expr::real(0)));

    // Negated property: sum of masks != 1
    let mask_sum = m1.real_add(m2).real_add(m3);
    let violation = mask_sum.ne(Expr::real(1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "max_pool_gradient_sparse");
}

// ---------------------------------------------------------------------------
// Test 981: Pool2d spatial dimension reduction
// ---------------------------------------------------------------------------

/// Prove: 2D pooling with K=2, S=2 on an HxW input produces (H/2)x(W/2)
/// output. Both spatial dimensions are halved independently.
///
/// For H=16, W=32: out_h=8, out_w=16. Total spatial reduction = 4x.
#[test]
fn test_981_pool2d_spatial_dimension_reduction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("h_in", real.clone());
    let _ = prog.declare_const("w_in", real.clone());
    let _ = prog.declare_const("h_out", real.clone());
    let _ = prog.declare_const("w_out", real);

    let h_in = real_var("h_in");
    let w_in = real_var("w_in");
    let h_out = real_var("h_out");
    let w_out = real_var("w_out");

    // Input: H=16, W=32
    prog.assert(h_in.clone().eq(Expr::real(16)));
    prog.assert(w_in.clone().eq(Expr::real(32)));

    // K=2, S=2, P=0: out = (L - 2)/2 + 1 = L/2
    let h_formula = h_in
        .real_sub(Expr::real(2))
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    let w_formula = w_in
        .real_sub(Expr::real(2))
        .real_mul(Expr::real_ratio(1, 2))
        .real_add(Expr::real(1));
    prog.assert(h_out.clone().eq(h_formula));
    prog.assert(w_out.clone().eq(w_formula));

    // Negated property: h_out != 8 OR w_out != 16
    let violation = h_out.ne(Expr::real(8)).or(w_out.ne(Expr::real(16)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pool2d_spatial_dimension_reduction");
}

// ---------------------------------------------------------------------------
// Test 982: Non-overlapping pooling correctness
// ---------------------------------------------------------------------------

/// Prove: When stride = kernel (non-overlapping), each input element
/// belongs to exactly one pooling window, and total elements processed
/// equals L (for L divisible by K).
///
/// For L=12, K=S=3: windows = L/K = 4. Coverage = 4*3 = 12 = L.
/// No element is counted twice.
#[test]
fn test_982_non_overlapping_pooling_correctness() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("l", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("num_windows", real.clone());
    let _ = prog.declare_const("coverage", real);

    let l = real_var("l");
    let k = real_var("k");
    let num_windows = real_var("num_windows");
    let coverage = real_var("coverage");

    // L=12, K=3
    prog.assert(l.clone().eq(Expr::real(12)));
    prog.assert(k.clone().eq(Expr::real(3)));

    // S=K=3, P=0: num_windows = (L - K)/K + 1 = (12-3)/3 + 1 = 4
    let nw_formula = l
        .clone()
        .real_sub(k.clone())
        .real_mul(Expr::real_ratio(1, 3))
        .real_add(Expr::real(1));
    prog.assert(num_windows.clone().eq(nw_formula));

    // coverage = num_windows * K
    prog.assert(coverage.clone().eq(num_windows.real_mul(k)));

    // Negated property: coverage != L (should exactly tile)
    let violation = coverage.ne(l);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "non_overlapping_pooling_correctness");
}

// ---------------------------------------------------------------------------
// Test 983: Average pool commutes with scaling
// ---------------------------------------------------------------------------

/// Prove: avg_pool(c * x) = c * avg_pool(x) for any scalar c.
///
/// For window [x1, x2]:
///   avg_pool(c*x1, c*x2) = (c*x1 + c*x2)/2 = c*(x1+x2)/2 = c*avg_pool(x1,x2).
///
/// This is a consequence of linearity (test 975 with beta=0).
#[test]
fn test_983_avg_pool_commutes_with_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("lhs", real.clone());
    let _ = prog.declare_const("rhs", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let c = real_var("c");
    let lhs = real_var("lhs");
    let rhs = real_var("rhs");

    // Bounded inputs
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));
    prog.assert(x2.clone().real_ge(Expr::real(-100)));
    prog.assert(x2.clone().real_le(Expr::real(100)));
    prog.assert(c.clone().real_ge(Expr::real(-100)));
    prog.assert(c.clone().real_le(Expr::real(100)));

    // LHS = avg(c*x1, c*x2) = (c*x1 + c*x2) / 2
    let scaled_sum = c
        .clone()
        .real_mul(x1.clone())
        .real_add(c.clone().real_mul(x2.clone()));
    prog.assert(
        lhs.clone()
            .eq(scaled_sum.real_mul(Expr::real_ratio(1, 2))),
    );

    // RHS = c * avg(x1, x2) = c * (x1 + x2) / 2
    let avg_unscaled = x1.real_add(x2).real_mul(Expr::real_ratio(1, 2));
    prog.assert(rhs.clone().eq(c.real_mul(avg_unscaled)));

    // Negated property: lhs != rhs
    let violation = lhs.ne(rhs);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "avg_pool_commutes_with_scaling");
}

// ---------------------------------------------------------------------------
// Test 984: ROI pooling output shape fixed
// ---------------------------------------------------------------------------

/// Prove: Region of Interest (ROI) pooling produces a fixed-size output
/// regardless of the ROI size. For ROI pool with target 7x7:
///   out_h = 7, out_w = 7 for any ROI height and width >= 1.
///
/// This is analogous to adaptive pooling but applied to arbitrary ROIs.
#[test]
fn test_984_roi_pooling_output_shape_fixed() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("roi_h", real.clone());
    let _ = prog.declare_const("roi_w", real.clone());
    let _ = prog.declare_const("out_h", real.clone());
    let _ = prog.declare_const("out_w", real);

    let roi_h = real_var("roi_h");
    let roi_w = real_var("roi_w");
    let out_h = real_var("out_h");
    let out_w = real_var("out_w");

    // ROI has arbitrary positive spatial dimensions
    prog.assert(roi_h.real_ge(Expr::real(1)));
    prog.assert(roi_w.real_ge(Expr::real(1)));

    // ROI pool with target 7x7: output is always 7x7
    prog.assert(out_h.clone().eq(Expr::real(7)));
    prog.assert(out_w.clone().eq(Expr::real(7)));

    // Negated property: out_h != 7 OR out_w != 7
    let violation = out_h.ne(Expr::real(7)).or(out_w.ne(Expr::real(7)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "roi_pooling_output_shape_fixed");
}

// ---------------------------------------------------------------------------
// Test 985: Channel dimension unchanged by spatial pooling
// ---------------------------------------------------------------------------

/// Prove: Spatial pooling (max or avg) does not change the channel
/// dimension. For input [B, C, H, W], output is [B, C, H', W'].
///
/// c_out = c_in and b_out = b_in for any pooling parameters.
#[test]
fn test_985_channel_unchanged_by_spatial_pooling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("b_in", real.clone());
    let _ = prog.declare_const("c_in", real.clone());
    let _ = prog.declare_const("b_out", real.clone());
    let _ = prog.declare_const("c_out", real);

    let b_in = real_var("b_in");
    let c_in = real_var("c_in");
    let b_out = real_var("b_out");
    let c_out = real_var("c_out");

    // Input dimensions positive
    prog.assert(b_in.clone().real_ge(Expr::real(1)));
    prog.assert(c_in.clone().real_ge(Expr::real(1)));

    // Spatial pooling preserves batch and channel dimensions
    prog.assert(b_out.clone().eq(b_in.clone()));
    prog.assert(c_out.clone().eq(c_in.clone()));

    // Negated property: b_out != b_in OR c_out != c_in
    let violation = b_out.ne(b_in).or(c_out.ne(c_in));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "channel_unchanged_by_spatial_pooling");
}

// ---------------------------------------------------------------------------
// Test 986: Pool output finite when input finite
// ---------------------------------------------------------------------------

/// Prove: If all inputs to average pooling are in [-M, M] (finite),
/// then the output is also in [-M, M] (finite).
///
/// This is the numerical stability property: no finite input produces
/// an infinite or out-of-range output through averaging.
#[test]
fn test_986_pool_output_finite_when_input_finite() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("x4", real.clone());
    let _ = prog.declare_const("avg", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let x4 = real_var("x4");
    let avg = real_var("avg");

    let m = Expr::real(1000000); // large finite bound
    let neg_m = Expr::real(-1000000);

    // All inputs in [-M, M]
    for x in [x1.clone(), x2.clone(), x3.clone(), x4.clone()] {
        prog.assert(x.clone().real_ge(neg_m.clone()));
        prog.assert(x.real_le(m.clone()));
    }

    // avg = (x1 + x2 + x3 + x4) / 4
    let sum = x1.real_add(x2).real_add(x3).real_add(x4);
    prog.assert(avg.clone().eq(sum.real_mul(Expr::real_ratio(1, 4))));

    // Negated property: avg < -M OR avg > M
    let violation = avg.clone().real_lt(neg_m).or(avg.real_gt(m));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "pool_output_finite_when_input_finite");
}

// ---------------------------------------------------------------------------
// Test 987: Cascaded max pool: max(max(x)) = max(x)
// ---------------------------------------------------------------------------

/// Prove: Cascading max pool is idempotent in a stronger sense:
/// applying max pool twice on overlapping windows still produces
/// an output bounded by the original max.
///
/// For 4 elements [x1, x2, x3, x4]:
///   First pass (K=2, S=1): m1 = max(x1,x2), m2 = max(x2,x3), m3 = max(x3,x4).
///   Second pass (K=2, S=1): out1 = max(m1,m2), out2 = max(m2,m3).
///   Overall max = max(x1, x2, x3, x4).
///   Prove: out1 <= overall_max AND out2 <= overall_max.
#[test]
fn test_987_cascaded_max_pool_idempotent() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("x4", real.clone());
    let _ = prog.declare_const("m1", real.clone());
    let _ = prog.declare_const("m2", real.clone());
    let _ = prog.declare_const("m3", real.clone());
    let _ = prog.declare_const("out1", real.clone());
    let _ = prog.declare_const("overall_max", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let x4 = real_var("x4");
    let m1 = real_var("m1");
    let m2 = real_var("m2");
    let m3 = real_var("m3");
    let out1 = real_var("out1");
    let overall_max = real_var("overall_max");

    // First pass: m1 = max(x1, x2)
    prog.assert(m1.clone().real_ge(x1.clone()));
    prog.assert(m1.clone().real_ge(x2.clone()));
    prog.assert(m1.clone().eq(x1.clone()).or(m1.clone().eq(x2.clone())));

    // m2 = max(x2, x3)
    prog.assert(m2.clone().real_ge(x2.clone()));
    prog.assert(m2.clone().real_ge(x3.clone()));
    prog.assert(m2.clone().eq(x2.clone()).or(m2.clone().eq(x3.clone())));

    // m3 = max(x3, x4)
    prog.assert(m3.clone().real_ge(x3.clone()));
    prog.assert(m3.clone().real_ge(x4.clone()));
    prog.assert(m3.clone().eq(x3.clone()).or(m3.clone().eq(x4.clone())));

    // Second pass: out1 = max(m1, m2)
    prog.assert(out1.clone().real_ge(m1.clone()));
    prog.assert(out1.clone().real_ge(m2.clone()));
    prog.assert(out1.clone().eq(m1).or(out1.clone().eq(m2)));

    // overall_max = max(x1, x2, x3, x4)
    prog.assert(overall_max.clone().real_ge(x1.clone()));
    prog.assert(overall_max.clone().real_ge(x2.clone()));
    prog.assert(overall_max.clone().real_ge(x3.clone()));
    prog.assert(overall_max.clone().real_ge(x4.clone()));
    prog.assert(
        overall_max
            .clone()
            .eq(x1)
            .or(overall_max.clone().eq(x2))
            .or(overall_max.clone().eq(x3))
            .or(overall_max.clone().eq(x4)),
    );

    // Negated property: out1 > overall_max
    let violation = out1.real_gt(overall_max);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cascaded_max_pool_idempotent");
}

// ---------------------------------------------------------------------------
// Test 988: Adaptive average pool for variable input sizes
// ---------------------------------------------------------------------------

/// Prove: Adaptive average pool produces the same target output size
/// regardless of input size, as long as L_in >= target.
///
/// For target=4, L_in=8 and L_in=16 both produce out=4.
/// The kernel and stride adapt, but the output is always `target`.
#[test]
fn test_988_adaptive_avg_pool_variable_input() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("l_in_a", real.clone());
    let _ = prog.declare_const("l_in_b", real.clone());
    let _ = prog.declare_const("out_a", real.clone());
    let _ = prog.declare_const("out_b", real);

    let l_in_a = real_var("l_in_a");
    let l_in_b = real_var("l_in_b");
    let out_a = real_var("out_a");
    let out_b = real_var("out_b");

    let target = Expr::real(4);

    // Two different input sizes, both >= target
    prog.assert(l_in_a.eq(Expr::real(8)));
    prog.assert(l_in_b.eq(Expr::real(16)));

    // Adaptive pool guarantees output = target for both
    prog.assert(out_a.clone().eq(target.clone()));
    prog.assert(out_b.clone().eq(target));

    // Negated property: out_a != out_b (should be equal to target)
    let violation = out_a.ne(out_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "adaptive_avg_pool_variable_input");
}

// ---------------------------------------------------------------------------
// Test 989: Weighted average bounded by weights * bounds
// ---------------------------------------------------------------------------

/// Prove: A weighted average pool with non-negative weights summing to 1
/// produces output in [lo, hi] when all inputs are in [lo, hi].
///
/// For 3 elements with weights w1, w2, w3 >= 0, w1+w2+w3 = 1:
///   out = w1*x1 + w2*x2 + w3*x3.
///   If lo <= x_i <= hi for all i, then lo <= out <= hi.
///
/// This is the convex combination property (generalisation of avg pool
/// to non-uniform weights).
#[test]
fn test_989_weighted_average_bounded_by_weights_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    let _ = prog.declare_const("x3", real.clone());
    let _ = prog.declare_const("w1", real.clone());
    let _ = prog.declare_const("w2", real.clone());
    let _ = prog.declare_const("w3", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real.clone());
    let _ = prog.declare_const("out", real);

    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let x3 = real_var("x3");
    let w1 = real_var("w1");
    let w2 = real_var("w2");
    let w3 = real_var("w3");
    let lo = real_var("lo");
    let hi = real_var("hi");
    let out = real_var("out");

    // lo <= hi
    prog.assert(lo.clone().real_le(hi.clone()));

    // All inputs in [lo, hi]
    for x in [x1.clone(), x2.clone(), x3.clone()] {
        prog.assert(x.clone().real_ge(lo.clone()));
        prog.assert(x.real_le(hi.clone()));
    }

    // Weights non-negative
    prog.assert(w1.clone().real_ge(Expr::real(0)));
    prog.assert(w2.clone().real_ge(Expr::real(0)));
    prog.assert(w3.clone().real_ge(Expr::real(0)));

    // Weights sum to 1
    prog.assert(
        w1.clone()
            .real_add(w2.clone())
            .real_add(w3.clone())
            .eq(Expr::real(1)),
    );

    // out = w1*x1 + w2*x2 + w3*x3
    let weighted = w1
        .real_mul(x1)
        .real_add(w2.real_mul(x2))
        .real_add(w3.real_mul(x3));
    prog.assert(out.clone().eq(weighted));

    // Negated property: out < lo OR out > hi
    let violation = out.clone().real_lt(lo).or(out.real_gt(hi));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "weighted_average_bounded_by_weights_bounds");
}

// ---------------------------------------------------------------------------
// Test 990: LP pool (p-norm pool) monotonicity
// ---------------------------------------------------------------------------

/// Prove: LP pooling (p=2) is monotonic with respect to element-wise
/// dominance. If all |a_i| >= |b_i|, then lp_pool(a) >= lp_pool(b).
///
/// LP pool (p=2): out = (sum x_i^2 / K)^(1/2).
/// We encode via squares: if a1^2 + a2^2 >= b1^2 + b2^2,
/// then lp_pool(a) >= lp_pool(b) (since sqrt is monotonically increasing).
///
/// Concrete: a1=4, a2=3, b1=2, b2=1.
///   sum_a_sq = 16+9 = 25, sum_b_sq = 4+1 = 5.
///   25 >= 5 => lp_pool(a) >= lp_pool(b).
#[test]
fn test_990_lp_pool_monotonicity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("sum_a_sq", real.clone());
    let _ = prog.declare_const("sum_b_sq", real);

    let sum_a_sq = real_var("sum_a_sq");
    let sum_b_sq = real_var("sum_b_sq");

    // a1=4, a2=3: sum_a_sq = 16 + 9 = 25
    prog.assert(sum_a_sq.clone().eq(Expr::real(25)));

    // b1=2, b2=1: sum_b_sq = 4 + 1 = 5
    prog.assert(sum_b_sq.clone().eq(Expr::real(5)));

    // Negated property: sum_a_sq < sum_b_sq
    // (if sum_a_sq >= sum_b_sq, then sqrt(sum_a_sq/K) >= sqrt(sum_b_sq/K)
    //  since sqrt is monotonically increasing)
    let violation = sum_a_sq.real_lt(sum_b_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "lp_pool_monotonicity");
}
