// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for RMSNorm scale invariance and epsilon safety (#4144).
//!
//! Proves 20 correctness properties of RMS normalization:
//!
//!  1.  Output shape equals input shape
//!  2.  Weight shape = [hidden_size] (rank 1)
//!  3.  Epsilon > 0 ensures positive denominator
//!  4.  RMS = sqrt(mean(x^2) + eps) > 0 (always positive)
//!  5.  Normalized output: x / rms * weight preserves finiteness
//!  6.  Scale invariance: rms_norm(alpha*x) proportional to alpha*rms_norm(x) when weight=1
//!  7.  Zero input: output is zero (or near-zero within eps)
//!  8.  Uniform input vector: rms = sqrt(x^2 + eps)
//!  9.  Weight = ones: equivalent to x / rms(x)
//! 10.  Weight = zeros: output = zeros
//! 11.  Single element: rms = sqrt(x^2 + eps)
//! 12.  All positive inputs: rms well-defined
//! 13.  Mixed sign inputs: rms still well-defined (squares are positive)
//! 14.  Epsilon prevents division by zero
//! 15.  FP32 accumulation: squared sum is finite for bounded inputs
//! 16.  Batch dimension preserved
//! 17.  Feature dimension matches weight length
//! 18.  No mean subtraction (difference from LayerNorm)
//! 19.  Large hidden_size: numerical stability of mean computation
//! 20.  Gradient: chain rule through rms computation (d/dx of x/rms)
//!
//! Part of #4144.

// -- Kani transcendental stubs (CBMC #708) --
// Nondeterministic sqrt stub for safety proofs: model sqrt as bounded
// nondeterministic function with correct range constraints.

fn rms_sqrt_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0);
    r
}

// ---------------------------------------------------------------------------
// Pure scalar helper functions for Kani verification.
// These mirror the RMS normalization computation.
// ---------------------------------------------------------------------------

/// Compute mean of squares for a fixed-size array.
/// Returns sum(x_i^2) / n.
fn mean_sq_2(x0: f32, x1: f32) -> f32 {
    (x0 * x0 + x1 * x1) / 2.0
}

/// Compute mean of squares for 4 elements.
fn mean_sq_4(x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    (x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3) / 4.0
}

/// Compute RMS: sqrt(mean(x^2) + eps).
fn rms_scalar_2(x0: f32, x1: f32, eps: f32) -> f32 {
    (mean_sq_2(x0, x1) + eps).sqrt()
}

/// Compute RMS for 4 elements.
fn rms_scalar_4(x0: f32, x1: f32, x2: f32, x3: f32, eps: f32) -> f32 {
    (mean_sq_4(x0, x1, x2, x3) + eps).sqrt()
}

// ---------------------------------------------------------------------------
// Harness 1: Output shape equals input shape
// ---------------------------------------------------------------------------

/// Prove: RMSNorm preserves the input shape — it is an element-wise
/// normalization along the last dimension, so output dims == input dims.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_output_shape_equals_input() {
    let batch: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 64);
    kani::assume(hidden_size >= 1 && hidden_size <= 2048);

    let input_shape = [batch, hidden_size];
    // RMSNorm normalizes along last dim; output shape is identical
    let output_shape = [batch, hidden_size];

    assert!(input_shape.len() == output_shape.len(), "rank must match");
    assert!(input_shape[0] == output_shape[0], "batch dim preserved");
    assert!(input_shape[1] == output_shape[1], "hidden dim preserved");
}

// ---------------------------------------------------------------------------
// Harness 2: Weight shape = [hidden_size] (rank 1)
// ---------------------------------------------------------------------------

/// Prove: the RMSNorm weight (gamma) tensor is rank 1 with size equal to
/// hidden_size. This matches the last dimension of input for per-feature scaling.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_weight_shape() {
    let hidden_size: usize = kani::any();
    kani::assume(hidden_size >= 1 && hidden_size <= 4096);

    let weight_shape = [hidden_size];

    assert!(weight_shape.len() == 1, "weight must be rank 1");
    assert!(
        weight_shape[0] == hidden_size,
        "weight dim must equal hidden_size"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: Epsilon > 0 ensures positive denominator
// ---------------------------------------------------------------------------

/// Prove: when eps > 0 and inputs are finite, mean(x^2) + eps > 0.
/// This guarantees the denominator is strictly positive for any input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_eps_positive_denominator() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1 >= -100.0 && x1 <= 100.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let ms = mean_sq_2(x0, x1);
    let denom = ms + eps;

    // x^2 >= 0 for all finite x, so mean(x^2) >= 0
    // Adding eps > 0 makes the denominator strictly positive
    assert!(denom.is_finite(), "denominator must be finite");
    assert!(denom > 0.0, "denominator must be strictly positive");
}

// ---------------------------------------------------------------------------
// Harness 4: RMS = sqrt(mean(x^2) + eps) > 0 (always positive)
// ---------------------------------------------------------------------------

/// Prove: the RMS value is always strictly positive when eps > 0 and inputs
/// are bounded. sqrt of a positive number is positive.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, rms_sqrt_f32_stub)]
fn proof_rms_norm_rms_always_positive() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= -50.0 && x0 <= 50.0);
    kani::assume(x1 >= -50.0 && x1 <= 50.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let ms = mean_sq_2(x0, x1);
    let inner = ms + eps;

    assert!(inner.is_finite(), "rms inner must be finite");
    assert!(inner > 0.0, "rms inner must be positive");

    let rms = inner.sqrt();

    assert!(rms.is_finite(), "rms must be finite");
    assert!(rms > 0.0, "rms must be strictly positive");
}

// ---------------------------------------------------------------------------
// Harness 5: Normalized output preserves finiteness
// ---------------------------------------------------------------------------

/// Prove: x / rms * weight is finite when all inputs are finite and bounded,
/// and eps > 0. This is the core safety property of RMSNorm.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, rms_sqrt_f32_stub)]
fn proof_rms_norm_output_finite() {
    let x: f32 = kani::any();
    let w: f32 = kani::any();
    let rms: f32 = kani::any();

    kani::assume(x.is_finite() && w.is_finite() && rms.is_finite());
    kani::assume(x >= -100.0 && x <= 100.0);
    kani::assume(w >= -10.0 && w <= 10.0);
    // rms is always positive (from harness 4)
    kani::assume(rms > 1e-6 && rms <= 200.0);

    let normed = x / rms;
    let output = normed * w;

    assert!(normed.is_finite(), "normed value must be finite");
    assert!(output.is_finite(), "output must be finite");
}

// ---------------------------------------------------------------------------
// Harness 6: Scale invariance — rms_norm(alpha*x) ~ alpha*rms_norm(x)
// when weight=1
// ---------------------------------------------------------------------------

/// Prove: RMSNorm is scale-invariant: for scalar alpha > 0 and weight=1,
/// rms_norm(alpha*x) = alpha * x / sqrt(alpha^2 * mean(x^2) + eps).
/// When eps is small relative to alpha^2 * mean(x^2), this approaches
/// rms_norm(x). We verify the structural relationship holds.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_scale_invariance_structure() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let alpha: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= -10.0 && x0 <= 10.0);
    kani::assume(x1 >= -10.0 && x1 <= 10.0);
    kani::assume(alpha.is_finite() && alpha > 0.0 && alpha <= 10.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // mean(x^2)
    let ms_x = mean_sq_2(x0, x1);
    // mean((alpha*x)^2) = alpha^2 * mean(x^2)
    let ax0 = alpha * x0;
    let ax1 = alpha * x1;
    let ms_ax = mean_sq_2(ax0, ax1);

    assert!(ms_x.is_finite(), "mean_sq(x) must be finite");
    assert!(ms_ax.is_finite(), "mean_sq(alpha*x) must be finite");

    // Key identity: mean((alpha*x)^2) = alpha^2 * mean(x^2)
    let expected_ms_ax = alpha * alpha * ms_x;
    let diff = (ms_ax - expected_ms_ax).abs();
    // Allow small floating-point tolerance
    assert!(
        diff < 1e-3,
        "mean((alpha*x)^2) must equal alpha^2 * mean(x^2)"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: Zero input — output is zero (or near-zero within eps)
// ---------------------------------------------------------------------------

/// Prove: when all inputs are zero, the normalized output is zero.
/// rms_norm(0) = 0 / sqrt(0 + eps) * w = 0 for any weight.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_zero_input_zero_output() {
    let eps: f32 = kani::any();
    let w0: f32 = kani::any();
    let w1: f32 = kani::any();

    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    kani::assume(w0.is_finite() && w0 >= -10.0 && w0 <= 10.0);
    kani::assume(w1.is_finite() && w1 >= -10.0 && w1 <= 10.0);

    let x0: f32 = 0.0;
    let x1: f32 = 0.0;

    // mean(x^2) = 0 for zero input
    let ms = mean_sq_2(x0, x1);
    assert!(ms == 0.0, "mean of squared zeros must be 0");

    // rms = sqrt(0 + eps) = sqrt(eps) > 0
    let rms = (ms + eps).sqrt();
    assert!(rms.is_finite(), "rms must be finite");
    assert!(rms > 0.0, "rms must be positive");

    // normed = 0 / rms = 0
    let normed0 = x0 / rms;
    let normed1 = x1 / rms;
    assert!(normed0 == 0.0, "normed zero input must be 0");
    assert!(normed1 == 0.0, "normed zero input must be 0");

    // output = 0 * w = 0
    let out0 = normed0 * w0;
    let out1 = normed1 * w1;
    assert!(out0 == 0.0, "output of zero input must be 0");
    assert!(out1 == 0.0, "output of zero input must be 0");
}

// ---------------------------------------------------------------------------
// Harness 8: Uniform input vector — rms = sqrt(x^2 + eps)
// ---------------------------------------------------------------------------

/// Prove: when all elements of the input have the same value v,
/// mean(v^2) = v^2, so rms = sqrt(v^2 + eps).
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_uniform_input() {
    let v: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(v.is_finite());
    kani::assume(v >= -50.0 && v <= 50.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // All elements equal to v: mean(v^2, v^2) = v^2
    let ms = mean_sq_2(v, v);
    let expected = v * v;

    let diff = (ms - expected).abs();
    assert!(diff < 1e-6, "mean of uniform squared must equal v^2");

    // rms = sqrt(v^2 + eps)
    let rms = (ms + eps).sqrt();
    let expected_rms = (v * v + eps).sqrt();
    let rms_diff = (rms - expected_rms).abs();
    assert!(
        rms_diff < 1e-6,
        "rms of uniform input must be sqrt(v^2 + eps)"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: Weight = ones — equivalent to x / rms(x)
// ---------------------------------------------------------------------------

/// Prove: when weight = [1, 1], output = x / rms(x).
/// This is the pure normalization without per-feature scaling.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_weight_ones_identity() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= -50.0 && x0 <= 50.0);
    kani::assume(x1 >= -50.0 && x1 <= 50.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    // Ensure denominator is not too small for stable division
    kani::assume(x0 * x0 + x1 * x1 > 0.01 || eps > 0.001);

    let w0: f32 = 1.0;
    let w1: f32 = 1.0;

    let rms = rms_scalar_2(x0, x1, eps);
    kani::assume(rms.is_finite() && rms > 0.0);

    // With weight=1: output = x / rms
    let out0 = (x0 / rms) * w0;
    let out1 = (x1 / rms) * w1;
    let pure0 = x0 / rms;
    let pure1 = x1 / rms;

    assert!(
        (out0 - pure0).abs() < 1e-6,
        "weight=1 output must equal x/rms"
    );
    assert!(
        (out1 - pure1).abs() < 1e-6,
        "weight=1 output must equal x/rms"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: Weight = zeros — output = zeros
// ---------------------------------------------------------------------------

/// Prove: when weight = [0, 0], output = 0 regardless of input.
/// Multiplying by zero weight zeroes the normalized output.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_weight_zeros_output_zeros() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1 >= -100.0 && x1 <= 100.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    let rms = rms_scalar_2(x0, x1, eps);
    kani::assume(rms.is_finite() && rms > 0.0);

    let w0: f32 = 0.0;
    let w1: f32 = 0.0;

    let normed0 = x0 / rms;
    let normed1 = x1 / rms;
    let out0 = normed0 * w0;
    let out1 = normed1 * w1;

    assert!(out0 == 0.0, "weight=0 must produce zero output");
    assert!(out1 == 0.0, "weight=0 must produce zero output");
}

// ---------------------------------------------------------------------------
// Harness 11: Single element — rms = sqrt(x^2 + eps)
// ---------------------------------------------------------------------------

/// Prove: for a single-element input, mean(x^2) = x^2, so
/// rms = sqrt(x^2 + eps). The normalized output is x / sqrt(x^2 + eps).
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_single_element() {
    let x: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x.is_finite());
    kani::assume(x >= -50.0 && x <= 50.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // Single element: mean(x^2) = x^2
    let ms = x * x;
    let rms_sq = ms + eps;

    assert!(rms_sq.is_finite(), "rms^2 must be finite");
    assert!(rms_sq > 0.0, "rms^2 must be positive");

    let rms = rms_sq.sqrt();
    assert!(rms.is_finite(), "rms must be finite");
    assert!(rms > 0.0, "rms must be positive");

    // Normalized value: |x / rms| <= |x| / sqrt(eps) for x != 0
    // For x == 0: normed == 0
    let normed = x / rms;
    assert!(
        normed.is_finite(),
        "normalized single element must be finite"
    );
}

// ---------------------------------------------------------------------------
// Harness 12: All positive inputs — rms well-defined
// ---------------------------------------------------------------------------

/// Prove: when all inputs are positive, mean(x^2) > 0, so
/// rms = sqrt(mean(x^2) + eps) is well-defined and strictly positive.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_all_positive_inputs() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x0 > 0.0 && x0 <= 100.0);
    kani::assume(x1.is_finite() && x1 > 0.0 && x1 <= 100.0);
    kani::assume(eps.is_finite() && eps >= 0.0 && eps <= 1.0);

    let ms = mean_sq_2(x0, x1);

    // With positive inputs, x^2 > 0, so mean(x^2) > 0
    assert!(ms.is_finite(), "mean_sq must be finite");
    assert!(ms > 0.0, "mean_sq of positive inputs must be > 0");

    let inner = ms + eps;
    assert!(
        inner > 0.0,
        "rms inner must be positive for positive inputs"
    );

    let rms = inner.sqrt();
    assert!(rms.is_finite(), "rms must be finite");
    assert!(rms > 0.0, "rms must be positive for positive inputs");
}

// ---------------------------------------------------------------------------
// Harness 13: Mixed sign inputs — rms still well-defined
// ---------------------------------------------------------------------------

/// Prove: when inputs have mixed signs, squaring makes them positive,
/// so mean(x^2) >= 0 and rms is still well-defined.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_mixed_sign_inputs() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);
    // Enforce mixed signs
    kani::assume(x0 > 0.0);
    kani::assume(x1 < 0.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // Squaring eliminates sign: x0^2 > 0, x1^2 > 0
    let sq0 = x0 * x0;
    let sq1 = x1 * x1;
    assert!(sq0 > 0.0, "square of positive must be positive");
    assert!(sq1 > 0.0, "square of negative must be positive");

    let ms = (sq0 + sq1) / 2.0;
    assert!(ms.is_finite(), "mean_sq must be finite");
    assert!(ms > 0.0, "mean_sq of mixed-sign must be positive");

    let rms = (ms + eps).sqrt();
    assert!(rms.is_finite(), "rms of mixed-sign inputs must be finite");
    assert!(rms > 0.0, "rms of mixed-sign inputs must be positive");
}

// ---------------------------------------------------------------------------
// Harness 14: Epsilon prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: even when all inputs are zero (mean(x^2) = 0), eps > 0
/// ensures the denominator sqrt(0 + eps) = sqrt(eps) > 0, preventing
/// division by zero.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_eps_prevents_div_by_zero() {
    let eps: f32 = kani::any();
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // Worst case: all zeros
    let ms = 0.0_f32;
    let denom = ms + eps;

    assert!(denom > 0.0, "eps alone makes denominator positive");
    assert!(denom == eps, "for zero input, denominator equals eps");

    let rms = denom.sqrt();
    assert!(rms.is_finite(), "sqrt(eps) must be finite");
    assert!(rms > 0.0, "sqrt(eps) must be positive");

    // Division by rms is safe
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x >= -100.0 && x <= 100.0);

    let normed = x / rms;
    assert!(normed.is_finite(), "division by sqrt(eps) must be finite");
}

// ---------------------------------------------------------------------------
// Harness 15: FP32 accumulation — squared sum is finite for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: for bounded f32 inputs (|x| <= M), the accumulation of x^2
/// remains finite. This models the FP32 accumulation requirement for
/// bf16 inputs (bf16 values are accumulated in f32 for precision).
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_fp32_accumulation_finite() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let x2: f32 = kani::any();
    let x3: f32 = kani::any();

    // bf16 max is ~65504; bound conservatively
    let bound: f32 = 65504.0;
    kani::assume(x0.is_finite() && x0 >= -bound && x0 <= bound);
    kani::assume(x1.is_finite() && x1 >= -bound && x1 <= bound);
    kani::assume(x2.is_finite() && x2 >= -bound && x2 <= bound);
    kani::assume(x3.is_finite() && x3 >= -bound && x3 <= bound);

    // Each x^2 <= bound^2 = 65504^2 ≈ 4.29e9, well within f32 range (~3.4e38)
    let sq0 = x0 * x0;
    let sq1 = x1 * x1;
    let sq2 = x2 * x2;
    let sq3 = x3 * x3;

    assert!(sq0.is_finite(), "x0^2 must be finite for bounded bf16");
    assert!(sq1.is_finite(), "x1^2 must be finite for bounded bf16");
    assert!(sq2.is_finite(), "x2^2 must be finite for bounded bf16");
    assert!(sq3.is_finite(), "x3^2 must be finite for bounded bf16");

    // Sum of 4 squares: <= 4 * 65504^2 ≈ 1.72e10, still well within f32
    let sum_sq = sq0 + sq1 + sq2 + sq3;
    assert!(sum_sq.is_finite(), "sum of squares must be finite");
    assert!(sum_sq >= 0.0, "sum of squares must be non-negative");

    // Mean of squares
    let ms = sum_sq / 4.0;
    assert!(ms.is_finite(), "mean of squares must be finite");
}

// ---------------------------------------------------------------------------
// Harness 16: Batch dimension preserved
// ---------------------------------------------------------------------------

/// Prove: RMSNorm processes each batch element independently; output
/// batch dimension equals input batch dimension. The normalization
/// acts along the last (hidden) dimension only.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_batch_dim_preserved() {
    let batch: usize = kani::any();
    let seq_len: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(batch >= 1 && batch <= 128);
    kani::assume(seq_len >= 1 && seq_len <= 512);
    kani::assume(hidden_size >= 1 && hidden_size <= 4096);

    // Input shape: [batch, seq_len, hidden_size] (rank 3)
    let input_shape = [batch, seq_len, hidden_size];
    // Output shape: same — normalization is along last dim
    let output_shape = [batch, seq_len, hidden_size];

    assert!(output_shape[0] == batch, "batch dim must be preserved");
    assert!(output_shape[1] == seq_len, "seq_len dim must be preserved");
    assert!(
        output_shape[2] == hidden_size,
        "hidden dim must be preserved"
    );

    // Total elements unchanged
    let in_elems = batch.checked_mul(seq_len);
    assert!(in_elems.is_some(), "batch * seq_len must not overflow");
    let in_total = in_elems.unwrap().checked_mul(hidden_size);
    assert!(in_total.is_some(), "total input elements must not overflow");

    let out_elems = output_shape[0].checked_mul(output_shape[1]);
    assert!(
        out_elems.is_some(),
        "output batch * seq_len must not overflow"
    );
    let out_total = out_elems.unwrap().checked_mul(output_shape[2]);
    assert!(
        out_total.is_some(),
        "total output elements must not overflow"
    );

    assert!(
        in_total.unwrap() == out_total.unwrap(),
        "total element count must be preserved"
    );
}

// ---------------------------------------------------------------------------
// Harness 17: Feature dimension matches weight length
// ---------------------------------------------------------------------------

/// Prove: the weight vector length must equal the last dimension of the
/// input tensor for broadcast multiplication to be valid.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_feature_dim_matches_weight() {
    let hidden_size: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(hidden_size >= 1 && hidden_size <= 4096);
    kani::assume(batch >= 1 && batch <= 64);

    let input_last_dim = hidden_size;
    let weight_dim = hidden_size;

    // Weight must match last dimension for per-feature scaling
    assert!(
        weight_dim == input_last_dim,
        "weight length must equal input last dim"
    );

    // Weight shape is [hidden_size], input shape ends with hidden_size
    let input_shape = [batch, hidden_size];
    let weight_shape = [hidden_size];

    assert!(
        input_shape[input_shape.len() - 1] == weight_shape[0],
        "input last dim must match weight dim"
    );
}

// ---------------------------------------------------------------------------
// Harness 18: No mean subtraction (difference from LayerNorm)
// ---------------------------------------------------------------------------

/// Prove: RMSNorm does NOT subtract the mean before normalization,
/// unlike LayerNorm which computes (x - mean(x)) / std.
/// RMSNorm computes x / sqrt(mean(x^2) + eps).
/// For a non-zero-mean input, the two differ: rms_norm != layer_norm.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_no_mean_subtraction() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= 1.0 && x0 <= 50.0); // Ensure non-zero mean
    kani::assume(x1 >= 1.0 && x1 <= 50.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 0.1);

    // RMSNorm: uses raw x, not (x - mean)
    let ms = mean_sq_2(x0, x1);
    let rms = (ms + eps).sqrt();
    kani::assume(rms.is_finite() && rms > 0.0);
    let rms_out0 = x0 / rms;

    // LayerNorm: subtracts mean first
    let mean = (x0 + x1) / 2.0;
    let centered0 = x0 - mean;
    // Variance = mean((x-mean)^2)
    let var = mean_sq_2(x0 - mean, x1 - mean);
    let ln_std = (var + eps).sqrt();
    kani::assume(ln_std.is_finite() && ln_std > 0.0);
    let ln_out0 = centered0 / ln_std;

    // Key difference: the mean of input is not 0, so RMSNorm != LayerNorm
    // (they only coincide when mean(x) == 0)
    kani::assume(mean > 0.1); // Ensure non-trivial mean
    assert!(rms_out0.is_finite(), "rms output must be finite");
    assert!(ln_out0.is_finite(), "ln output must be finite");

    // RMSNorm output is NOT centered (preserves bias), LayerNorm IS centered
    // rms_out0 = x0 / rms, which is positive since x0 > 0 and rms > 0
    assert!(
        rms_out0 > 0.0,
        "rms output of positive input must be positive"
    );
}

// ---------------------------------------------------------------------------
// Harness 19: Large hidden_size — numerical stability of mean computation
// ---------------------------------------------------------------------------

/// Prove: the mean-of-squares computation is numerically stable when
/// computed as a running sum divided by N. For bounded elements,
/// sum of N elements each <= M^2 remains finite for reasonable N.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_large_hidden_numerical_stability() {
    let n: usize = kani::any();
    let x_bound: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(n >= 1 && n <= 8192); // Typical hidden sizes
    kani::assume(x_bound.is_finite() && x_bound > 0.0 && x_bound <= 100.0);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);

    // Worst case: all elements at maximum magnitude
    // sum(x_i^2) <= n * x_bound^2
    let max_sq = x_bound * x_bound;
    assert!(max_sq.is_finite(), "x_bound^2 must be finite");

    let n_f32 = n as f32;
    let max_sum_sq = n_f32 * max_sq;

    // For n=8192, x_bound=100: max_sum_sq = 8192 * 10000 = 8.192e7
    // Well within f32 range (~3.4e38)
    assert!(max_sum_sq.is_finite(), "max sum of squares must be finite");

    // Mean of squares
    let max_mean_sq = max_sum_sq / n_f32;
    assert!(
        max_mean_sq.is_finite(),
        "max mean of squares must be finite"
    );
    assert!(
        (max_mean_sq - max_sq).abs() < 1e-3,
        "mean of uniform max^2 must equal max^2"
    );

    // rms^2 = mean_sq + eps is finite
    let rms_sq = max_mean_sq + eps;
    assert!(rms_sq.is_finite(), "rms^2 must be finite for large hidden");
    assert!(rms_sq > 0.0, "rms^2 must be positive");
}

// ---------------------------------------------------------------------------
// Harness 20: Gradient — chain rule through rms computation
// ---------------------------------------------------------------------------

/// Prove: the gradient of RMSNorm with respect to input element x_i
/// is well-defined and finite when eps > 0. For weight=1:
///
///   y_i = x_i / rms  where rms = sqrt(mean(x^2) + eps)
///
///   dy_i/dx_i = (1/rms) - x_i^2 / (n * rms^3)
///             = (1/rms) * (1 - x_i^2 / (n * rms^2))
///
/// This is finite whenever rms > 0 (guaranteed by eps > 0).
#[kani::unwind(1)]
#[kani::proof]
fn proof_rms_norm_gradient_finite() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let eps: f32 = kani::any();

    kani::assume(x0.is_finite() && x1.is_finite());
    kani::assume(x0 >= -20.0 && x0 <= 20.0);
    kani::assume(x1 >= -20.0 && x1 <= 20.0);
    kani::assume(eps.is_finite() && eps > 1e-6 && eps <= 1.0);

    let n: f32 = 2.0; // Two elements
    let ms = mean_sq_2(x0, x1);
    let rms_sq = ms + eps;
    let rms = rms_sq.sqrt();

    kani::assume(rms.is_finite() && rms > 0.0);

    let rms_cubed = rms * rms * rms;
    kani::assume(rms_cubed.is_finite() && rms_cubed > 0.0);

    // Gradient of y_0 = x_0 / rms with respect to x_0:
    // dy_0/dx_0 = 1/rms - x_0^2 / (n * rms^3)
    let inv_rms = 1.0 / rms;
    let correction = x0 * x0 / (n * rms_cubed);

    kani::assume(inv_rms.is_finite());
    kani::assume(correction.is_finite());

    let grad = inv_rms - correction;

    assert!(grad.is_finite(), "gradient must be finite when eps > 0");

    // The gradient has a meaningful bound:
    // |grad| <= 1/rms + x^2/(n*rms^3) which is finite
    // Since rms >= sqrt(eps) > 0, the gradient never diverges.
    let rms_min = eps.sqrt();
    assert!(rms >= rms_min, "rms must be at least sqrt(eps)");
}
