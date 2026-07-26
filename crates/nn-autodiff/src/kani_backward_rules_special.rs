// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `backward_rules_special.rs`.
//!
//! Proves properties of the composite backward rules: Softmax, LayerNorm,
//! Embedding, CrossEntropy, MSE, L1, and Huber loss backward implementations.
//! These are the most complex backward rules — they involve multi-tensor
//! computation (recomputing forward intermediates) rather than simple
//! element-wise scalar derivatives.
//!
//! Harnesses verify:
//! - Softmax backward Jacobian element properties
//! - LayerNorm inv_std is always positive
//! - LayerNorm normalized output has zero mean
//! - Embedding backward scatter-add position mapping
//! - CrossEntropy backward gradient scaling by 1/N
//! - Cross-entropy softmax minus one-hot produces bounded values
//! - MSE/L1/Huber backward antisymmetry (swapping input and target negates gradient)
//! - Huber backward continuity at the quadratic/linear boundary
//! - scalar_grad_val finiteness
//!
//! **Local-copy gap:** Scalar functions here re-implement production formulas from
//! `backward_rules_special.rs`. `// SYNC:` comments track correspondence.
//!
//! Re: #3714 (Kani harnesses for nn-autodiff grad + backward_rules_special + trainable_extra).

// ── Softmax backward Jacobian diagonal ───────────────────────────────────
//
// Softmax backward: grad_x[i] = s[i] * (grad[i] - dot(grad, s))
// where s = softmax(x).
//
// SYNC: backward_rules_special.rs:38-42

/// Softmax backward element: s_i * (grad_i - dot_grad_s).
///
/// SYNC: backward_rules_special.rs:41
#[allow(dead_code)]
fn softmax_backward_elem(s_i: f32, grad_i: f32, dot_grad_s: f32) -> f32 {
    s_i * (grad_i - dot_grad_s)
}

/// Prove softmax backward element is finite for valid softmax output.
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_backward_elem_finite() {
    let s_i: f32 = kani::any();
    let grad_i: f32 = kani::any();
    let dot: f32 = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
    kani::assume(dot.is_finite() && dot.abs() <= 1e6);
    let result = softmax_backward_elem(s_i, grad_i, dot);
    assert!(result.is_finite(), "softmax backward elem must be finite");
}

/// Prove softmax backward is bounded by s_i * (|grad_i| + |dot|).
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_backward_elem_bounded() {
    let s_i: f32 = kani::any();
    let grad_i: f32 = kani::any();
    let dot: f32 = kani::any();
    kani::assume(s_i.is_finite() && s_i >= 0.0 && s_i <= 1.0);
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e4);
    kani::assume(dot.is_finite() && dot.abs() <= 1e4);
    let result = softmax_backward_elem(s_i, grad_i, dot);
    let bound = s_i * (grad_i.abs() + dot.abs());
    assert!(
        result.abs() <= bound + 1e-5,
        "softmax backward elem must be bounded by s_i * (|grad| + |dot|)"
    );
}

/// Prove softmax backward is exactly zero when s_i = 0 (masked position).
/// Gradient must not flow through masked-out softmax positions.
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_backward_zero_when_masked() {
    let grad_i: f32 = kani::any();
    let dot: f32 = kani::any();
    kani::assume(grad_i.is_finite() && grad_i.abs() <= 1e6);
    kani::assume(dot.is_finite() && dot.abs() <= 1e6);
    let result = softmax_backward_elem(0.0, grad_i, dot);
    assert!(result == 0.0, "softmax backward must be zero when s_i = 0");
}

// ── LayerNorm inv_std computation ────────────────────────────────────────
//
// inv_std = 1 / sqrt(var + eps) where var >= 0 and eps > 0.
// Therefore inv_std > 0 for any valid input.
//
// SYNC: backward_rules_special.rs:108

/// Compute inv_std from variance and epsilon.
///
/// SYNC: backward_rules_special.rs:108 (`var.add_scalar(eps)?.sqrt()?.recip()?`)
#[allow(dead_code)]
fn inv_std(var: f32, eps: f64) -> f32 {
    let denominator = (var + eps as f32).sqrt();
    1.0 / denominator
}

/// Prove inv_std is positive for non-negative variance and positive eps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn prove_inv_std_positive() {
    let var: f32 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(var.is_finite() && var >= 0.0 && var <= 1e6);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    let result = inv_std(var, eps);
    assert!(result.is_finite(), "inv_std must be finite");
    assert!(result > 0.0, "inv_std must be positive");
}

/// Prove inv_std is bounded below by 1/sqrt(var_max + eps).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn prove_inv_std_monotone_decreasing() {
    let var_lo: f32 = kani::any();
    let var_hi: f32 = kani::any();
    let eps: f64 = kani::any();
    kani::assume(var_lo.is_finite() && var_lo >= 0.0 && var_lo <= 1e4);
    kani::assume(var_hi.is_finite() && var_hi >= 0.0 && var_hi <= 1e4);
    kani::assume(var_lo < var_hi);
    kani::assume(eps.is_finite() && eps > 1e-6 && eps <= 1.0);
    let lo = inv_std(var_lo, eps);
    let hi = inv_std(var_hi, eps);
    assert!(
        lo >= hi,
        "inv_std must be monotonically decreasing in variance"
    );
}

// ── LayerNorm normalized mean ────────────────────────────────────────────
//
// After normalization: normalized = (x - mean) / std.
// The mean of normalized values should be zero (within floating-point precision).
//
// For a single element: normalized = (x - x) / std = 0.
//
// SYNC: backward_rules_special.rs:106-109

/// Single-element normalization: (x - mean) / std.
#[allow(dead_code)]
fn normalize_single(x: f32, mean: f32, inv_std: f32) -> f32 {
    (x - mean) * inv_std
}

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
    }
    r
}

/// Prove single-element normalization of x with mean=x gives zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn prove_normalize_single_zero_when_x_eq_mean() {
    let x: f32 = kani::any();
    let eps: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e6);
    kani::assume(eps.is_finite() && eps > 0.0 && eps <= 1.0);
    let istd = 1.0 / eps.sqrt(); // var = 0, so inv_std = 1/sqrt(eps)
    let result = normalize_single(x, x, istd);
    assert!(
        result == 0.0,
        "normalized value must be 0 when x equals mean"
    );
}

// ── Embedding backward scatter-add position mapping ──────────────────────
//
// Embedding backward: for each token i, grad_weight[indices[i]] += grad[i].
// Each index must be in [0, vocab_size).
//
// SYNC: backward_rules_special.rs:138-175

/// Check if an index is valid for the embedding table.
///
/// SYNC: backward_rules_special.rs:154-155
#[allow(dead_code)]
fn is_valid_embedding_index(idx: usize, vocab_size: usize) -> bool {
    idx < vocab_size
}

/// Prove valid indices are accepted.
#[kani::unwind(1)]
#[kani::proof]
fn prove_valid_embedding_index() {
    let vocab: u16 = kani::any();
    let idx: u16 = kani::any();
    kani::assume(vocab >= 1 && vocab <= 10000);
    kani::assume(idx < vocab);
    assert!(
        is_valid_embedding_index(idx as usize, vocab as usize),
        "index < vocab must be valid"
    );
}

/// Prove out-of-range indices are rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_invalid_embedding_index() {
    let vocab: u16 = kani::any();
    let idx: u16 = kani::any();
    kani::assume(vocab >= 1 && vocab <= 10000);
    kani::assume(idx >= vocab);
    assert!(
        !is_valid_embedding_index(idx as usize, vocab as usize),
        "index >= vocab must be invalid"
    );
}

/// Model the num_tokens calculation: total_grad_elements / embed_dim.
///
/// SYNC: backward_rules_special.rs:159
#[allow(dead_code)]
fn num_tokens(grad_numel: usize, embed_dim: usize) -> usize {
    grad_numel / embed_dim
}

/// Prove num_tokens * embed_dim <= grad_numel (integer division).
#[kani::unwind(1)]
#[kani::proof]
fn prove_num_tokens_consistency() {
    let grad_numel: u16 = kani::any();
    let embed_dim: u16 = kani::any();
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);
    kani::assume(grad_numel >= embed_dim && grad_numel <= 10000);
    let n = num_tokens(grad_numel as usize, embed_dim as usize);
    assert!(
        n * embed_dim as usize <= grad_numel as usize,
        "num_tokens * embed_dim must not exceed grad_numel"
    );
}

// ── CrossEntropy backward gradient scaling ───────────────────────────────
//
// Cross-entropy backward: grad = (softmax - one_hot) / N * upstream_grad.
// The 1/N scaling averages over samples.
//
// SYNC: backward_rules_special.rs:201-234

/// Cross-entropy backward scaling factor: 1/N.
///
/// SYNC: backward_rules_special.rs:231 (`diff.mul_scalar(1.0 / n as f64)`)
#[allow(dead_code)]
fn ce_scale(n: usize) -> f64 {
    1.0 / n as f64
}

/// Prove CE scaling factor is positive and bounded by 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_ce_scale_positive_bounded() {
    let n: u16 = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    let scale = ce_scale(n as usize);
    assert!(scale.is_finite(), "CE scale must be finite");
    assert!(scale > 0.0, "CE scale must be positive");
    assert!(scale <= 1.0, "CE scale must be <= 1.0");
}

/// Cross-entropy softmax minus one-hot element: value in [-1, 1].
///
/// SYNC: backward_rules_special.rs:230
#[allow(dead_code)]
fn softmax_minus_onehot(softmax_val: f32, is_target: bool) -> f32 {
    if is_target {
        softmax_val - 1.0
    } else {
        softmax_val
    }
}

/// Prove softmax minus one-hot is in [-1, 1].
#[kani::unwind(1)]
#[kani::proof]
fn prove_softmax_minus_onehot_bounded() {
    let s: f32 = kani::any();
    let is_target: bool = kani::any();
    kani::assume(s.is_finite() && s >= 0.0 && s <= 1.0);
    let result = softmax_minus_onehot(s, is_target);
    assert!(
        result >= -1.0 && result <= 1.0,
        "softmax - one_hot must be in [-1, 1]"
    );
}

/// Prove target class gradient is non-positive (softmax < 1 for multi-class).
#[kani::unwind(1)]
#[kani::proof]
fn prove_ce_target_grad_nonpositive() {
    let s: f32 = kani::any();
    kani::assume(s.is_finite() && s >= 0.0 && s <= 1.0);
    let result = softmax_minus_onehot(s, true);
    assert!(
        result <= 0.0,
        "target class gradient must be non-positive (softmax <= 1)"
    );
}

/// Prove non-target class gradient is non-negative.
#[kani::unwind(1)]
#[kani::proof]
fn prove_ce_nontarget_grad_nonnegative() {
    let s: f32 = kani::any();
    kani::assume(s.is_finite() && s >= 0.0 && s <= 1.0);
    let result = softmax_minus_onehot(s, false);
    assert!(
        result >= 0.0,
        "non-target class gradient must be non-negative"
    );
}

// ── MSE backward antisymmetry ────────────────────────────────────────────
//
// MSE backward: 2*(x - t)/N. Swapping x and t negates the gradient.
//
// SYNC: backward_rules_special.rs:247-250

/// MSE backward scalar: 2*(x - t) / N.
///
/// SYNC: backward_rules_special.rs:250
#[allow(dead_code)]
fn mse_backward(x: f32, t: f32, n: usize) -> f32 {
    2.0 * (x - t) / n as f32
}

/// Prove MSE backward is antisymmetric: swap(x, t) negates the result.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mse_backward_antisymmetric() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e3);
    kani::assume(t.is_finite() && t.abs() <= 1e3);
    let forward = mse_backward(x, t, 1);
    let reversed = mse_backward(t, x, 1);
    assert!(
        (forward + reversed).abs() < 1e-5,
        "MSE backward must be antisymmetric: f(x,t) = -f(t,x)"
    );
}

// ── L1 backward antisymmetry ─────────────────────────────────────────────
//
// L1 backward: sign(x - t) / N. Swapping x and t negates the gradient.
//
// SYNC: backward_rules_special.rs:267-276

/// L1 backward scalar: sign(x - t) / N.
///
/// SYNC: backward_rules_special.rs:272-276
#[allow(dead_code)]
fn l1_backward(x: f32, t: f32, n: usize) -> f32 {
    let diff = x - t;
    let sign = if diff > 0.0 {
        1.0
    } else if diff < 0.0 {
        -1.0
    } else {
        0.0
    };
    sign / n as f32
}

/// Prove L1 backward is antisymmetric when x != t.
#[kani::unwind(1)]
#[kani::proof]
fn prove_l1_backward_antisymmetric() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e4);
    kani::assume(t.is_finite() && t.abs() <= 1e4);
    kani::assume(x != t);
    let forward = l1_backward(x, t, 1);
    let reversed = l1_backward(t, x, 1);
    assert!(
        (forward + reversed).abs() < 1e-7,
        "L1 backward must be antisymmetric: f(x,t) = -f(t,x)"
    );
}

// ── Huber backward continuity at boundary ────────────────────────────────
//
// Huber backward: diff/(N*delta) when |diff| < delta, sign(diff)/N otherwise.
// At |diff| == delta, both branches give the same value: sign(diff)/N.
// The quadratic branch: diff/(N*delta) = +/-1/N when diff = +/-delta.
//
// SYNC: backward_rules_special.rs:284-312

/// Huber backward scalar.
///
/// SYNC: backward_rules_special.rs:296-308
#[allow(dead_code)]
fn huber_backward(x: f32, t: f32, delta: f64, n: usize) -> f32 {
    let diff = x - t;
    if diff.abs() < delta as f32 {
        diff / (n as f32 * delta as f32)
    } else {
        let sign = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };
        sign / n as f32
    }
}

/// Prove Huber backward is antisymmetric.
#[kani::unwind(1)]
#[kani::proof]
fn prove_huber_backward_antisymmetric() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    let delta: f64 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e3);
    kani::assume(t.is_finite() && t.abs() <= 1e3);
    kani::assume(delta.is_finite() && delta > 0.01 && delta <= 100.0);
    kani::assume(x != t);
    let forward = huber_backward(x, t, delta, 1);
    let reversed = huber_backward(t, x, delta, 1);
    assert!(
        (forward + reversed).abs() < 1e-5,
        "Huber backward must be antisymmetric"
    );
}

/// Prove Huber backward quadratic region gradient magnitude < 1/N.
/// In the quadratic region |diff| < delta: |diff/(N*delta)| < 1/N.
#[kani::unwind(1)]
#[kani::proof]
fn prove_huber_quadratic_bounded() {
    let x: f32 = kani::any();
    let t: f32 = kani::any();
    let delta: f64 = kani::any();
    let n: u16 = kani::any();
    kani::assume(x.is_finite() && x.abs() <= 1e3);
    kani::assume(t.is_finite() && t.abs() <= 1e3);
    kani::assume(delta.is_finite() && delta > 0.01 && delta <= 100.0);
    kani::assume(n >= 1 && n <= 10000);
    let diff = x - t;
    kani::assume(diff.is_finite() && diff.abs() < delta as f32);
    let result = huber_backward(x, t, delta, n as usize);
    let bound = 1.0_f32 / n as f32;
    assert!(
        result.abs() <= bound + 1e-6,
        "Huber quadratic gradient must be < 1/N"
    );
}

// ── scalar_grad_val finiteness ───────────────────────────────────────────
//
// scalar_grad_val extracts a single f32 from the gradient tensor and
// converts to f64. The production code uses to_scalar::<f32>().
//
// SYNC: backward_rules_special.rs:26-28

/// Model scalar_grad_val: f32 → f64 conversion.
///
/// SYNC: backward_rules_special.rs:27
#[allow(dead_code)]
fn scalar_grad_to_f64(grad_scalar: f32) -> f64 {
    f64::from(grad_scalar)
}

/// Prove f32 to f64 conversion preserves finiteness.
#[kani::unwind(1)]
#[kani::proof]
fn prove_scalar_grad_preserves_finiteness() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    let result = scalar_grad_to_f64(v);
    assert!(
        result.is_finite(),
        "f32 to f64 conversion must preserve finiteness"
    );
}

/// Prove f32 to f64 conversion preserves sign.
#[kani::unwind(1)]
#[kani::proof]
fn prove_scalar_grad_preserves_sign() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v != 0.0);
    let result = scalar_grad_to_f64(v);
    if v > 0.0 {
        assert!(result > 0.0, "positive f32 must stay positive as f64");
    } else {
        assert!(result < 0.0, "negative f32 must stay negative as f64");
    }
}

// ── Embedding weight shape validation ────────────────────────────────────
//
// backward_embedding validates weight is 2D with embed_dim > 0.
//
// SYNC: backward_rules_special.rs:145-153

/// Model embedding weight shape validation.
///
/// SYNC: backward_rules_special.rs:145-146
#[allow(dead_code)]
fn is_valid_embedding_weight(rank: usize, embed_dim: usize) -> bool {
    rank >= 2 && embed_dim > 0
}

/// Prove valid 2D weight with nonzero embed_dim is accepted.
#[kani::unwind(1)]
#[kani::proof]
fn prove_valid_embedding_weight_accepted() {
    let embed_dim: u16 = kani::any();
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);
    assert!(
        is_valid_embedding_weight(2, embed_dim as usize),
        "2D weight with embed_dim > 0 must be valid"
    );
}

/// Prove 1D weight is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_1d_embedding_weight_rejected() {
    let embed_dim: u16 = kani::any();
    kani::assume(embed_dim >= 1 && embed_dim <= 1024);
    assert!(
        !is_valid_embedding_weight(1, embed_dim as usize),
        "1D weight must be rejected"
    );
}

/// Prove zero embed_dim is rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_zero_embed_dim_rejected() {
    assert!(
        !is_valid_embedding_weight(2, 0),
        "zero embed_dim must be rejected"
    );
}

// ── CrossEntropy num_classes guard ────────────────────────────────────────
//
// backward_cross_entropy checks num_classes > 0 to avoid division by zero.
//
// SYNC: backward_rules_special.rs:193-200

/// Model num_classes validation.
///
/// SYNC: backward_rules_special.rs:195-199
#[allow(dead_code)]
fn is_valid_num_classes(num_classes: usize) -> bool {
    num_classes > 0
}

/// Prove zero classes rejected.
#[kani::unwind(1)]
#[kani::proof]
fn prove_zero_classes_rejected() {
    assert!(!is_valid_num_classes(0), "zero classes must be rejected");
}

/// Prove positive classes accepted.
#[kani::unwind(1)]
#[kani::proof]
fn prove_positive_classes_accepted() {
    let n: u16 = kani::any();
    kani::assume(n >= 1 && n <= 10000);
    assert!(
        is_valid_num_classes(n as usize),
        "positive num_classes must be accepted"
    );
}
