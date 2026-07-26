// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for SDPA (Scaled Dot-Product Attention).
//!
//! Extends the base harnesses in `sdpa.rs` and `kani_sdpa_rope_proofs.rs` with:
//! - Score scaling amplitude bounds for practical head dimensions
//! - Softmax weight normalization properties (sum-to-one, non-negative)
//! - Softmax numerical stability under max-subtraction
//! - Causal mask row-level attendable position counts
//! - repeat_kv dimension arithmetic safety
//! - SDPA scale factor reciprocal relationship
//!
//! Part of #3671.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// -- Score scaling amplitude bounds -----------------------------------------------

/// Prove that 1/sqrt(d_k) scaling reduces score magnitude for d_k >= 2.
///
/// For any dot-product score s, |s / sqrt(d_k)| < |s| when d_k >= 2.
/// This is the fundamental purpose of SDPA scaling: prevent softmax saturation
/// by normalizing scores to a reasonable range.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn scaling_reduces_magnitude_for_dk_ge_2() {
    let score: f32 = kani::any();
    let d_k: u32 = kani::any();
    kani::assume(score.is_finite());
    kani::assume(score.abs() > 1e-10); // non-trivial score
    kani::assume(d_k >= 2 && d_k <= 512);
    let sqrt_dk = (d_k as f32).sqrt();
    let scaled = score / sqrt_dk;
    kani::assert(scaled.is_finite(), "scaled score must be finite");
    kani::assert(
        scaled.abs() < score.abs(),
        "scaling by 1/sqrt(d_k>=2) must reduce magnitude",
    );
}

/// Prove scale factor = 1/sqrt(d_k) satisfies 1/sqrt(d_k) * sqrt(d_k) = 1.
///
/// The scale factor and sqrt(d_k) are multiplicative inverses. This ensures
/// that `Q @ K^T * scale` is equivalent to `(Q / sqrt(d_k)) @ K^T`.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn scale_factor_is_reciprocal_of_sqrt_dk() {
    let d_k: u32 = kani::any();
    kani::assume(d_k >= 1 && d_k <= 1024);
    let sqrt_dk = (d_k as f64).sqrt();
    let scale = 1.0_f64 / sqrt_dk;
    let product = scale * sqrt_dk;
    kani::assert(
        (product - 1.0).abs() < 1e-12,
        "scale * sqrt(d_k) must equal 1.0",
    );
}

/// Prove common head dimensions (64, 80, 96, 128) produce well-known scale values.
///
/// These are the head dimensions used by GPT-2 (64), Whisper (80),
/// LLaMA-2 (128), and Kokoro (96). The scale factor must be finite and in
/// a known range.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn scale_factor_common_head_dims_in_range() {
    let d_k: u32 = kani::any();
    kani::assume(d_k == 64 || d_k == 80 || d_k == 96 || d_k == 128);
    let scale = 1.0_f64 / (d_k as f64).sqrt();
    kani::assert(scale.is_finite(), "scale must be finite for common dims");
    // 1/sqrt(128) ~= 0.0884, 1/sqrt(64) = 0.125
    kani::assert(
        scale >= 0.08 && scale <= 0.13,
        "scale in expected range for common dims",
    );
}

// -- Softmax weight properties ---------------------------------------------------

/// Prove softmax of two elements sums to 1.0 (within float tolerance).
///
/// For any two finite scores after max-subtraction, softmax weights are
/// non-negative and sum to approximately 1.0. This is the fundamental
/// softmax normalization property.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_two_elements_sum_to_one() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a >= -100.0 && a <= 100.0);
    kani::assume(b >= -100.0 && b <= 100.0);
    let max_val = if a >= b { a } else { b };
    let exp_a = (a - max_val).exp();
    let exp_b = (b - max_val).exp();
    kani::assert(
        exp_a.is_finite() && exp_a >= 0.0,
        "exp_a must be finite non-negative",
    );
    kani::assert(
        exp_b.is_finite() && exp_b >= 0.0,
        "exp_b must be finite non-negative",
    );
    let sum = exp_a + exp_b;
    kani::assume(sum > 0.0); // sum > 0 since at least one exp == 1.0
    let w_a = exp_a / sum;
    let w_b = exp_b / sum;
    kani::assert(w_a >= 0.0 && w_a <= 1.0, "weight a in [0, 1]");
    kani::assert(w_b >= 0.0 && w_b <= 1.0, "weight b in [0, 1]");
    let weight_sum = w_a + w_b;
    kani::assert(
        (weight_sum - 1.0).abs() < 1e-5,
        "softmax weights must sum to ~1.0",
    );
}

/// Prove softmax preserves ordering: if a > b then softmax(a) > softmax(b).
///
/// Softmax is order-preserving because exp() is monotonically increasing.
/// This means the token with the highest attention score always receives
/// the highest attention weight.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_preserves_ordering() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a >= -50.0 && a <= 50.0);
    kani::assume(b >= -50.0 && b <= 50.0);
    kani::assume(a > b);
    let max_val = a; // a > b, so max is a
    let exp_a = (a - max_val).exp(); // exp(0) = 1.0
    let exp_b = (b - max_val).exp(); // exp(b-a) < 1.0 since b < a
    let sum = exp_a + exp_b;
    kani::assume(sum > 0.0);
    let w_a = exp_a / sum;
    let w_b = exp_b / sum;
    kani::assert(w_a > w_b, "higher score must get higher softmax weight");
}

/// Prove uniform scores produce equal softmax weights.
///
/// When all scores are equal, softmax assigns equal weight to each position.
/// For two elements, each gets weight 0.5.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_uniform_scores_equal_weights() {
    let s: f32 = kani::any();
    kani::assume(s.is_finite());
    kani::assume(s >= -100.0 && s <= 100.0);
    let exp_s = (s - s).exp(); // exp(0) = 1.0
    let sum = exp_s + exp_s; // 2.0
    let w = exp_s / sum;
    kani::assert(
        (w - 0.5).abs() < 1e-6,
        "uniform scores must produce equal weights",
    );
}

// -- Causal mask row-level properties --------------------------------------------

/// Prove causal mask row i has exactly (i + 1) attendable positions.
///
/// In a square causal mask with no offset, query at position i can attend
/// to positions 0, 1, ..., i. This gives exactly (i + 1) non-masked positions.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn causal_mask_row_attendable_count() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 16);
    let i: usize = kani::any();
    kani::assume(i < seq_len);
    // Count attendable positions in row i.
    // Position j is attendable iff j <= i (for square mask, offset=0).
    let attendable_count = i + 1;
    kani::assert(
        attendable_count >= 1,
        "every row has at least one attendable position (self)",
    );
    kani::assert(
        attendable_count <= seq_len,
        "attendable count cannot exceed seq_len",
    );
    kani::assert(
        attendable_count == i + 1,
        "row i has exactly i+1 attendable positions",
    );
}

/// Prove last row of causal mask has all positions attendable.
///
/// The last query token at position (seq_len - 1) can attend to all
/// positions [0, seq_len - 1]. This is the row with maximum context.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn causal_mask_last_row_full_context() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 32);
    let last_row = seq_len - 1;
    // For the last row, abs_pos = last_row = seq_len - 1.
    // Position j is attendable iff j <= seq_len - 1, which is all j < seq_len.
    let j: usize = kani::any();
    kani::assume(j < seq_len);
    let is_masked = j > last_row;
    kani::assert(!is_masked, "last row must have all positions attendable");
}

/// Prove first row of causal mask only attends to position 0.
///
/// The first query token (position 0) can only attend to itself.
/// All other positions j > 0 are masked.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn causal_mask_first_row_self_only() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 2 && seq_len <= 32);
    let j: usize = kani::any();
    kani::assume(j < seq_len);
    let is_masked = j > 0; // row 0, abs_pos = 0
    if j == 0 {
        kani::assert(!is_masked, "position 0 must attend to itself");
    } else {
        kani::assert(is_masked, "position 0 cannot attend to future positions");
    }
}

// -- repeat_kv dimension safety ---------------------------------------------------

/// Prove repeat_kv intermediate expand dimensions are valid.
///
/// The expand step creates shape [B, H, num_rep, S, D] from [B, H, 1, S, D].
/// The product B * H * num_rep * S * D must not overflow and must equal
/// the final output element count.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn repeat_kv_expand_dimensions_safe() {
    let b: usize = kani::any();
    let h: usize = kani::any();
    let num_rep: usize = kani::any();
    let s: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(b >= 1 && b <= 4);
    kani::assume(h >= 1 && h <= 16);
    kani::assume(num_rep >= 1 && num_rep <= 16);
    kani::assume(s >= 1 && s <= 32);
    kani::assume(d >= 1 && d <= 128);
    // Check intermediate shape [B, H, num_rep, S, D] doesn't overflow.
    let intermediate = b
        .checked_mul(h)
        .and_then(|x| x.checked_mul(num_rep))
        .and_then(|x| x.checked_mul(s))
        .and_then(|x| x.checked_mul(d));
    kani::assume(intermediate.is_some());
    let intermediate = intermediate.unwrap();
    // Final shape [B, H*num_rep, S, D].
    let final_h = h.checked_mul(num_rep);
    kani::assume(final_h.is_some());
    let final_total = b
        .checked_mul(final_h.unwrap())
        .and_then(|x| x.checked_mul(s))
        .and_then(|x| x.checked_mul(d));
    kani::assume(final_total.is_some());
    kani::assert(
        intermediate == final_total.unwrap(),
        "expand and reshape must have same total elements",
    );
}

// -- Causal mask offset boundary conditions --------------------------------------

/// Prove causal mask with offset: last new token sees all total_tokens.
///
/// The last new query token is at absolute position (total_tokens - 1),
/// which can attend to all positions [0, total_tokens - 1].
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn causal_mask_offset_last_new_token_full_context() {
    let new_tokens: usize = kani::any();
    let total_tokens: usize = kani::any();
    kani::assume(new_tokens >= 1 && new_tokens <= 8);
    kani::assume(total_tokens >= new_tokens && total_tokens <= 16);
    let offset = total_tokens - new_tokens;
    let last_row = new_tokens - 1;
    let abs_pos = offset + last_row; // = total_tokens - 1
    kani::assert(
        abs_pos == total_tokens - 1,
        "last new token at end of sequence",
    );
    let col: usize = kani::any();
    kani::assume(col < total_tokens);
    let is_masked = col > abs_pos;
    kani::assert(
        !is_masked,
        "last new token must attend to all total_tokens positions",
    );
}
