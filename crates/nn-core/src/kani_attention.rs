// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for attention score invariants.
//!
//! Proves key mathematical properties of scaled dot-product attention:
//!
//! ```text
//! scores = Q @ K^T / sqrt(head_dim)
//! attn_weights = softmax(scores, dim=-1)
//! output = attn_weights @ V
//! ```
//!
//! Properties proved:
//! 1. Softmax outputs are non-negative for all finite inputs
//! 2. Softmax outputs sum to 1.0 (within f32 tolerance)
//! 3. Attention scale factor 1/sqrt(d) is finite and positive for valid head_dim
//! 4. Scaled dot-product score is bounded for bounded Q, K inputs
//! 5. Attention output is bounded by V's bounds (convex combination)
//! 6. ALiBi slopes are strictly positive and finite
//! 7. ALiBi slopes are strictly decreasing with head index
//!
//! Part of #947 (JointAttention), requested by W12 for attention score invariants.

#![cfg(kani)]

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// -- Scalar softmax for small fixed-size vectors -----------------------------

/// Softmax over a 3-element vector (smallest non-trivial size).
/// Uses the numerically stable formulation: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
fn softmax_3(x: [f32; 3]) -> [f32; 3] {
    // Find max for numerical stability
    let mut m = x[0];
    if x[1] > m {
        m = x[1];
    }
    if x[2] > m {
        m = x[2];
    }

    let e0 = (x[0] - m).exp();
    let e1 = (x[1] - m).exp();
    let e2 = (x[2] - m).exp();
    let sum = e0 + e1 + e2;

    [e0 / sum, e1 / sum, e2 / sum]
}

/// Softmax over a 2-element vector (minimal case).
fn softmax_2(x: [f32; 2]) -> [f32; 2] {
    let m = if x[0] > x[1] { x[0] } else { x[1] };
    let e0 = (x[0] - m).exp();
    let e1 = (x[1] - m).exp();
    let sum = e0 + e1;
    [e0 / sum, e1 / sum]
}

// -- Stubs for CBMC ---------------------------------------------------------

/// Nondeterministic exp stub: returns any positive finite value.
/// Sound over-approximation: exp(finite) is always positive and finite
/// (ignoring exp(>88) → +inf, which is guarded by input bound assumptions).
fn exp_stub(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > 0.0);
    result
}

// -- Proof 1: Softmax outputs are non-negative (with exp stub) ---------------

/// Proves softmax outputs are all non-negative for bounded finite inputs.
///
/// Domain: x_i ∈ [-100, 100]. Since exp() always returns positive values,
/// and softmax divides by a positive sum, all outputs must be non-negative.
///
/// Uses exp stub (nondeterministic positive finite) — sound over-approximation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_stub)]
fn attention_softmax_non_negative() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let x2: f32 = kani::any();

    kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);
    kani::assume(x2.is_finite() && x2 >= -100.0 && x2 <= 100.0);

    let out = softmax_3([x0, x1, x2]);

    assert!(out[0].is_finite(), "softmax[0] must be finite");
    assert!(out[1].is_finite(), "softmax[1] must be finite");
    assert!(out[2].is_finite(), "softmax[2] must be finite");

    assert!(out[0] >= 0.0, "softmax[0] must be non-negative");
    assert!(out[1] >= 0.0, "softmax[1] must be non-negative");
    assert!(out[2] >= 0.0, "softmax[2] must be non-negative");
}

// -- Proof 2: Softmax outputs sum to 1.0 (with exp stub) --------------------

/// Proves softmax outputs sum to 1.0 within f32 tolerance.
///
/// The sum exp(x_i - m) / sum(exp(x_j - m)) telescopes to 1.0 algebraically.
/// Under IEEE 754 f32 arithmetic, we allow tolerance of 1e-5 for rounding.
///
/// Uses exp stub — since all exp outputs are positive finite, the sum of
/// (positive / positive_sum) must be close to 1.0 within f32 precision.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_stub)]
fn attention_softmax_sums_to_one() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();

    kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);

    let out = softmax_2([x0, x1]);
    let sum = out[0] + out[1];

    assert!(sum.is_finite(), "softmax sum must be finite");
    // Algebraically exact: e0/S + e1/S = (e0+e1)/S = S/S = 1.
    // f32 rounding: the only error source is (e0/S + e1/S) vs 1.0.
    // With exp_stub giving positive finite values, the division and
    // addition introduce at most a few ULPs of error.
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax must sum to ~1.0, got {sum}"
    );
}

// -- Proof 3: Softmax each element ≤ 1.0 ------------------------------------

/// Proves each softmax output is at most 1.0.
///
/// Since softmax(x)_i = exp(x_i - m) / sum(exp(x_j - m)) and the denominator
/// includes the numerator as a term, each output ≤ 1.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_stub)]
fn attention_softmax_at_most_one() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let x2: f32 = kani::any();

    kani::assume(x0.is_finite() && x0 >= -100.0 && x0 <= 100.0);
    kani::assume(x1.is_finite() && x1 >= -100.0 && x1 <= 100.0);
    kani::assume(x2.is_finite() && x2 >= -100.0 && x2 <= 100.0);

    let out = softmax_3([x0, x1, x2]);

    // Each element = e_i / (e_0 + e_1 + e_2). Since all e_j > 0,
    // the denominator > e_i, so the ratio < 1.0.
    // With f32 rounding, allow 1 ULP above 1.0.
    assert!(out[0] <= 1.0 + 1e-7, "softmax[0] must be <= 1");
    assert!(out[1] <= 1.0 + 1e-7, "softmax[1] must be <= 1");
    assert!(out[2] <= 1.0 + 1e-7, "softmax[2] must be <= 1");
}

// -- Proof 4: Attention scale factor is finite and positive ------------------

/// Proves 1/sqrt(head_dim) is finite and positive for valid head dimensions.
///
/// Domain: head_dim ∈ [1, 1024] (covers all practical attention configurations:
/// typical values are 32, 64, 128).
/// JointAttention uses `1.0 / (head_dim as f64).sqrt()` as scale.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn attention_scale_factor_valid() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 1024);

    let scale_f64 = 1.0_f64 / (head_dim as f64).sqrt();
    let scale = scale_f64 as f32;

    assert!(
        scale.is_finite(),
        "scale must be finite for head_dim={head_dim}"
    );
    assert!(
        scale > 0.0,
        "scale must be positive for head_dim={head_dim}"
    );
    // Scale decreases with head_dim: 1/sqrt(1) = 1.0 to 1/sqrt(1024) ≈ 0.031
    assert!(scale <= 1.0, "scale must be <= 1.0 for head_dim >= 1");
}

// -- Proof 5: Scaled dot-product is bounded for bounded Q, K ----------------

/// Proves Q·K scaled score is bounded for bounded inputs.
///
/// For a single query-key dot product (1D):
///   score = (sum_i Q_i * K_i) / sqrt(d)
///
/// With |Q_i|, |K_i| <= B and d terms:
///   |score| <= d * B^2 / sqrt(d) = B^2 * sqrt(d)
///
/// For d=2, B=10: |score| <= 100 * sqrt(2) ≈ 141.4
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn attention_scaled_dot_product_bounded() {
    let q0: f32 = kani::any();
    let q1: f32 = kani::any();
    let k0: f32 = kani::any();
    let k1: f32 = kani::any();

    kani::assume(q0.is_finite() && q0 >= -10.0 && q0 <= 10.0);
    kani::assume(q1.is_finite() && q1 >= -10.0 && q1 <= 10.0);
    kani::assume(k0.is_finite() && k0 >= -10.0 && k0 <= 10.0);
    kani::assume(k1.is_finite() && k1 >= -10.0 && k1 <= 10.0);

    // Dot product
    let dot = q0 * k0 + q1 * k1;
    // Scale by 1/sqrt(head_dim=2)
    let scale = 1.0_f32 / 2.0_f32.sqrt();
    let score = dot * scale;

    assert!(score.is_finite(), "scaled score must be finite");
    // Bound: |dot| <= 2 * 10 * 10 = 200, |score| = |dot| / sqrt(2) <= 200/1.414 ≈ 141.4
    assert!(score.abs() <= 142.0, "scaled score must be bounded");
}

// -- Proof 6: Attention output bounded by V bounds (convex combination) ------

/// Proves attention output is bounded by V bounds when weights are valid probabilities.
///
/// output = sum_j(w_j * v_j) where w_j >= 0, sum(w_j) = 1, and |v_j| <= V_max.
/// Then |output| <= V_max (convex combination of bounded values is bounded).
///
/// This is the fundamental guarantee: attention cannot amplify V's magnitude.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn attention_output_bounded_by_values() {
    let w0: f32 = kani::any();
    let w1: f32 = kani::any();
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();

    // Weights form a valid probability distribution
    kani::assume(w0.is_finite() && w0 >= 0.0 && w0 <= 1.0);
    kani::assume(w1.is_finite() && w1 >= 0.0 && w1 <= 1.0);
    // Allow small tolerance for sum-to-1 (matching softmax f32 precision)
    kani::assume((w0 + w1 - 1.0).abs() < 1e-5);

    // Values are bounded
    let v_max = 10.0_f32;
    kani::assume(v0.is_finite() && v0 >= -v_max && v0 <= v_max);
    kani::assume(v1.is_finite() && v1 >= -v_max && v1 <= v_max);

    let output = w0 * v0 + w1 * v1;

    assert!(output.is_finite(), "attention output must be finite");
    // Convex combination: |w0*v0 + w1*v1| <= w0*|v0| + w1*|v1| <= (w0+w1)*v_max
    // With w0+w1 ≈ 1.0 (within 1e-5), the bound is v_max + small epsilon
    assert!(
        output.abs() <= v_max + 0.01,
        "attention output must be bounded by v_max"
    );
}

// ALiBi and structural proofs (7-11) extracted to kani_attention_alibi.rs.
#[cfg(kani)]
#[path = "kani_attention_alibi.rs"]
mod kani_attention_alibi;
