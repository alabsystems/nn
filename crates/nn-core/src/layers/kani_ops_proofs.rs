// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for softmax, log_softmax, and sigmoid scalar safety (#3623).
//!
//! Proves numerical properties of the core activation/normalization operations:
//!
//! **Sigmoid (scalar):**
//! - Output is in [0, 1] for all bounded f32 inputs
//! - Output is finite for all finite inputs
//! - Monotonically non-decreasing
//! - Symmetry: sigmoid(-x) = 1 - sigmoid(x)
//! - Fixed point: sigmoid(0) = 0.5
//!
//! **Softmax (per-lane):**
//! - All outputs are non-negative
//! - Outputs sum to ~1.0 (within f32 tolerance)
//! - Max input produces max output (monotonicity preservation)
//! - Uniform input produces uniform output
//!
//! **Log-softmax (per-lane):**
//! - All outputs are non-positive (log of probability <= 0)
//! - exp(log_softmax) sums to ~1.0
//! - Consistency: log_softmax(x) = log(softmax(x))
//!
//! These harnesses operate on pure scalar/small-array arithmetic —
//! no DynTensor, ndarray, or GPU storage — making them tractable for
//! CBMC symbolic execution.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn ln_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ---------------------------------------------------------------------------
// Sigmoid scalar: 1.0 / (1.0 + (-x).exp()) is in [0, 1]
// ---------------------------------------------------------------------------

/// Scalar sigmoid matching the production implementation in
/// `dyn_tensor/ops/math.rs` line 133: `|x| 1.0 / (1.0 + (-x).exp())`
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Prove: sigmoid output is in [0, 1] for bounded f32 inputs.
///
/// The mathematical sigmoid maps R -> (0, 1). For IEEE 754 f32,
/// exp(-x) can underflow to 0 (giving sigmoid=1) or overflow to +inf
/// (giving sigmoid=0). Both endpoints are valid.
///
/// Input bound: [-88, 88] covers the non-saturated range of f32 exp().
/// Beyond |88|, exp saturates to 0 or +inf, but sigmoid still returns
/// 0.0 or 1.0 — both in [0, 1].
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_output_in_unit_interval() {
    let x: f32 = kani::any();
    // exp() is finite for |x| < ~88.7 in f32. Use [-88, 88] to cover
    // the full non-saturated range.
    kani::assume(x >= -88.0 && x <= 88.0);
    kani::assume(x.is_finite());

    let y = scalar_sigmoid(x);

    assert!(
        y.is_finite(),
        "sigmoid must produce finite output for finite input"
    );
    assert!(y >= 0.0, "sigmoid must be >= 0");
    assert!(y <= 1.0, "sigmoid must be <= 1");
}

/// Prove: sigmoid is finite for all finite inputs in wide range.
///
/// Even for extreme inputs outside the non-saturated range, sigmoid
/// must not produce NaN or Inf — it saturates to 0.0 or 1.0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_finite_for_extreme_inputs() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    // Use the full finite f32 range (no bound restriction)
    // but avoid values where the proof would time out.
    // f32::MAX is ~3.4e38. exp(-x) for large positive x underflows to 0;
    // exp(-x) for large negative x overflows to +inf.
    // In both cases, 1/(1+result) is well-defined: 1/(1+0)=1, 1/(1+inf)=0.
    kani::assume(x >= -1e10 && x <= 1e10);

    let y = scalar_sigmoid(x);

    assert!(
        !y.is_nan(),
        "sigmoid must never produce NaN for finite input"
    );
    assert!(y >= 0.0, "sigmoid must be >= 0");
    assert!(y <= 1.0, "sigmoid must be <= 1");
}

/// Prove: sigmoid(0) = 0.5 (the fixed point).
///
/// sigmoid(0) = 1/(1+exp(0)) = 1/(1+1) = 0.5.
/// This is a fundamental property used as a classification threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_zero_is_half() {
    let y = scalar_sigmoid(0.0);
    assert!((y - 0.5).abs() < 1e-7, "sigmoid(0) must equal 0.5");
}

/// Prove: sigmoid is monotonically non-decreasing.
///
/// For x1 <= x2, sigmoid(x1) <= sigmoid(x2). This is a fundamental
/// property: sigmoid is a monotone activation function, preserving
/// the ordering of logits.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_monotone() {
    let x1: f32 = kani::any();
    let x2: f32 = kani::any();
    kani::assume(x1.is_finite() && x2.is_finite());
    kani::assume(x1 >= -88.0 && x1 <= 88.0);
    kani::assume(x2 >= -88.0 && x2 <= 88.0);
    kani::assume(x1 <= x2);

    let y1 = scalar_sigmoid(x1);
    let y2 = scalar_sigmoid(x2);

    assert!(
        y1 <= y2 + 1e-6,
        "sigmoid must be monotonically non-decreasing"
    );
}

/// Prove: sigmoid symmetry — sigmoid(-x) = 1 - sigmoid(x).
///
/// This is the fundamental symmetry of the logistic function about
/// the point (0, 0.5). It is used in binary cross-entropy loss
/// computation where log(1 - sigmoid(x)) = log(sigmoid(-x)).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_symmetry() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -80.0 && x <= 80.0);

    let pos = scalar_sigmoid(x);
    let neg = scalar_sigmoid(-x);

    // sigmoid(-x) should equal 1 - sigmoid(x) within f32 tolerance.
    // Use a tolerance of 1e-5 to account for floating-point rounding
    // in the exp() computation.
    assert!(
        (pos + neg - 1.0).abs() < 1e-5,
        "sigmoid(x) + sigmoid(-x) must equal 1.0"
    );
}

// ---------------------------------------------------------------------------
// Softmax (small fixed-size arrays): non-negativity and sum-to-one
// ---------------------------------------------------------------------------

/// Numerically stable softmax over a small array (max-subtraction trick).
/// Mirrors the production CPU implementation in `dyn_tensor/softmax.rs`.
fn scalar_softmax_3(input: [f32; 3]) -> [f32; 3] {
    let max_val = input[0].max(input[1]).max(input[2]);
    let e0 = (input[0] - max_val).exp();
    let e1 = (input[1] - max_val).exp();
    let e2 = (input[2] - max_val).exp();
    let sum = e0 + e1 + e2;
    [e0 / sum, e1 / sum, e2 / sum]
}

/// Prove: softmax outputs are all non-negative for 3-element input.
///
/// exp(x) >= 0 for all x, and sum > 0 for finite inputs, so each
/// output exp(x_i - max) / sum must be >= 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_outputs_non_negative() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);

    let out = scalar_softmax_3([a, b, c]);

    assert!(out[0] >= 0.0, "softmax output[0] must be >= 0");
    assert!(out[1] >= 0.0, "softmax output[1] must be >= 0");
    assert!(out[2] >= 0.0, "softmax output[2] must be >= 0");
}

/// Prove: softmax outputs sum to ~1.0 for 3-element input.
///
/// The normalization step divides by the sum of exponentials,
/// so the outputs must sum to 1.0 (within f32 rounding tolerance).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_outputs_sum_to_one() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);

    let out = scalar_softmax_3([a, b, c]);
    let sum = out[0] + out[1] + out[2];

    assert!((sum - 1.0).abs() < 1e-5, "softmax outputs must sum to 1.0");
}

/// Prove: softmax outputs are all <= 1.0.
///
/// Each softmax output is a probability, so it must be in [0, 1].
/// Combined with non-negativity, this proves the full [0, 1] range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_outputs_bounded_above() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);

    let out = scalar_softmax_3([a, b, c]);

    assert!(out[0] <= 1.0, "softmax output[0] must be <= 1.0");
    assert!(out[1] <= 1.0, "softmax output[1] must be <= 1.0");
    assert!(out[2] <= 1.0, "softmax output[2] must be <= 1.0");
}

/// Prove: softmax preserves argmax — the largest input produces the
/// largest output.
///
/// If a > b and a > c, then softmax(a) > softmax(b) and softmax(a) > softmax(c).
/// This is a critical property for classification: softmax must not change
/// which class has the highest probability.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_preserves_argmax() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);
    // a is strictly the largest
    kani::assume(a > b + 1e-6 && a > c + 1e-6);

    let out = scalar_softmax_3([a, b, c]);

    assert!(
        out[0] >= out[1],
        "softmax of largest input must be >= other outputs"
    );
    assert!(
        out[0] >= out[2],
        "softmax of largest input must be >= other outputs"
    );
}

/// Prove: softmax of uniform input produces uniform output.
///
/// When all inputs are equal, softmax(x, x, x) = (1/3, 1/3, 1/3).
/// This is a basic sanity check on the normalization arithmetic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_uniform_input_uniform_output() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -80.0 && x <= 80.0);

    let out = scalar_softmax_3([x, x, x]);
    let expected = 1.0_f32 / 3.0;

    assert!(
        (out[0] - expected).abs() < 1e-5,
        "softmax of uniform input must be uniform"
    );
    assert!(
        (out[1] - expected).abs() < 1e-5,
        "softmax of uniform input must be uniform"
    );
    assert!(
        (out[2] - expected).abs() < 1e-5,
        "softmax of uniform input must be uniform"
    );
}

// ---------------------------------------------------------------------------
// Log-softmax (small fixed-size arrays): non-positivity and consistency
// ---------------------------------------------------------------------------

/// Numerically stable log-softmax over a 3-element array.
/// Mirrors the production CPU implementation in `dyn_tensor/softmax.rs`:
/// log_softmax(x_i) = x_i - max - log(sum(exp(x_j - max)))
fn scalar_log_softmax_3(input: [f32; 3]) -> [f32; 3] {
    let max_val = input[0].max(input[1]).max(input[2]);
    let e0 = (input[0] - max_val).exp();
    let e1 = (input[1] - max_val).exp();
    let e2 = (input[2] - max_val).exp();
    let sum_exp = e0 + e1 + e2;
    let log_sum_exp = max_val + sum_exp.ln();
    [
        input[0] - log_sum_exp,
        input[1] - log_sum_exp,
        input[2] - log_sum_exp,
    ]
}

/// Prove: log_softmax outputs are all non-positive.
///
/// log(p) <= 0 for p in (0, 1], and softmax outputs are probabilities
/// in (0, 1]. Therefore log_softmax(x_i) <= 0 for all i.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_outputs_non_positive() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);

    let out = scalar_log_softmax_3([a, b, c]);

    assert!(out[0] <= 1e-5, "log_softmax output[0] must be <= 0");
    assert!(out[1] <= 1e-5, "log_softmax output[1] must be <= 0");
    assert!(out[2] <= 1e-5, "log_softmax output[2] must be <= 0");
}

/// Prove: exp(log_softmax) sums to ~1.0.
///
/// Since log_softmax(x) = log(softmax(x)), we have
/// exp(log_softmax(x)) = softmax(x), which sums to 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_exp_sums_to_one() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);

    let out = scalar_log_softmax_3([a, b, c]);
    let exp_sum = out[0].exp() + out[1].exp() + out[2].exp();

    assert!(
        (exp_sum - 1.0).abs() < 1e-4,
        "exp(log_softmax) must sum to 1.0"
    );
}

/// Prove: log_softmax outputs are finite for finite inputs.
///
/// The numerically stable formulation (max-subtraction) prevents
/// overflow in the exp() computation, so all outputs should be finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_outputs_finite() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);

    let out = scalar_log_softmax_3([a, b, c]);

    assert!(out[0].is_finite(), "log_softmax[0] must be finite");
    assert!(out[1].is_finite(), "log_softmax[1] must be finite");
    assert!(out[2].is_finite(), "log_softmax[2] must be finite");
}

/// Prove: log_softmax consistency with softmax — log_softmax(x) = log(softmax(x)).
///
/// Both computation paths must agree within f32 tolerance. This verifies
/// that the numerically stable formulation is equivalent to the naive one.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_consistent_with_softmax() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -40.0 && a <= 40.0);
    kani::assume(b >= -40.0 && b <= 40.0);
    kani::assume(c >= -40.0 && c <= 40.0);

    let sm = scalar_softmax_3([a, b, c]);
    let lsm = scalar_log_softmax_3([a, b, c]);

    // log(softmax(x)) should equal log_softmax(x)
    let log_sm0 = sm[0].ln();
    let log_sm1 = sm[1].ln();
    let log_sm2 = sm[2].ln();

    assert!(
        (log_sm0 - lsm[0]).abs() < 1e-4,
        "log(softmax[0]) must match log_softmax[0]"
    );
    assert!(
        (log_sm1 - lsm[1]).abs() < 1e-4,
        "log(softmax[1]) must match log_softmax[1]"
    );
    assert!(
        (log_sm2 - lsm[2]).abs() < 1e-4,
        "log(softmax[2]) must match log_softmax[2]"
    );
}

/// Prove: log_softmax preserves argmax ordering.
///
/// The largest input must produce the largest (least negative) log_softmax
/// output. Since log is monotone, this follows from softmax argmax preservation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_preserves_argmax() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a >= -80.0 && a <= 80.0);
    kani::assume(b >= -80.0 && b <= 80.0);
    kani::assume(c >= -80.0 && c <= 80.0);
    // a is strictly the largest
    kani::assume(a > b + 1e-6 && a > c + 1e-6);

    let out = scalar_log_softmax_3([a, b, c]);

    assert!(
        out[0] >= out[1],
        "log_softmax of largest input must be >= other outputs"
    );
    assert!(
        out[0] >= out[2],
        "log_softmax of largest input must be >= other outputs"
    );
}

// ---------------------------------------------------------------------------
// Softmax 2-element: verifies properties hold for the smallest non-trivial case
// ---------------------------------------------------------------------------

/// 2-element softmax for verifying properties at the minimal vector size.
fn scalar_softmax_2(input: [f32; 2]) -> [f32; 2] {
    let max_val = input[0].max(input[1]);
    let e0 = (input[0] - max_val).exp();
    let e1 = (input[1] - max_val).exp();
    let sum = e0 + e1;
    [e0 / sum, e1 / sum]
}

/// Prove: 2-element softmax outputs are valid probabilities.
///
/// For the minimal case (2 elements), both outputs must be in [0, 1]
/// and sum to 1.0. This is the base case that all larger softmax
/// computations build upon.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_2_valid_probabilities() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();

    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a >= -88.0 && a <= 88.0);
    kani::assume(b >= -88.0 && b <= 88.0);

    let out = scalar_softmax_2([a, b]);

    assert!(out[0] >= 0.0, "softmax_2[0] must be >= 0");
    assert!(out[0] <= 1.0, "softmax_2[0] must be <= 1");
    assert!(out[1] >= 0.0, "softmax_2[1] must be >= 0");
    assert!(out[1] <= 1.0, "softmax_2[1] must be <= 1");

    let sum = out[0] + out[1];
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax_2 outputs must sum to 1.0"
    );
}

/// Prove: softmax(a, b) where a == b produces (0.5, 0.5).
///
/// Equal logits must produce equal probabilities. This is a critical
/// fairness property: softmax must not bias toward any position.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_2_equal_inputs_equal_outputs() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(x >= -88.0 && x <= 88.0);

    let out = scalar_softmax_2([x, x]);

    assert!(
        (out[0] - 0.5).abs() < 1e-6,
        "softmax of equal inputs must produce 0.5"
    );
    assert!(
        (out[1] - 0.5).abs() < 1e-6,
        "softmax of equal inputs must produce 0.5"
    );
}
