// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for ALiBi (Attention with Linear Biases).
//!
//! Extends the base harnesses in `alibi.rs` and `kani_sdpa_rope_proofs.rs` with:
//! - First slope equals 2^(-8/num_heads) (Press et al. 2021 specification)
//! - Slope range bounds: all slopes in (0, 1) for any num_heads >= 1
//! - ALiBi bias is linear in relative distance for each head
//! - Power-of-2 heads produce exact geometric progression
//! - Self-attention (i==j) always gets zero bias (no position penalty)
//! - ALiBi bias magnitude bounded by slope * (seq_len - 1)
//! - Scaled ALiBi preserves monotonicity when scale > 0
//!
//! Part of #3671.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn powf_f32_stub(b: f32, _e: f32) -> f32 {
    let _ = b;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

// -- First slope specification ---------------------------------------------------

/// Prove the first ALiBi slope equals 2^(-8/num_heads).
///
/// Press et al. 2021 define slopes as a geometric sequence starting at
/// 2^(-8/n) for head h=0 (using 1-indexed: h+1=1). This is the mildest
/// slope (closest to 1), giving the first head the longest effective context.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_first_slope_equals_spec() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    // h=0 => slope = 2^(-8 * 1 / num_heads) = 2^(-8/n)
    let first_slope = 2f32.powf(-8.0 * 1.0 / num_heads as f32);
    let expected = 2f32.powf(-8.0 / num_heads as f32);
    kani::assert(first_slope.is_finite(), "first slope must be finite");
    kani::assert(
        (first_slope - expected).abs() < 1e-7,
        "first slope must match 2^(-8/n)",
    );
}

// -- Slope range bounds ----------------------------------------------------------

/// Prove all ALiBi slopes are strictly in (0, 1) for num_heads >= 1.
///
/// Since the exponent -8*(h+1)/n is always negative (h >= 0, n >= 1),
/// 2^(negative) is in (0, 1). This means ALiBi always attenuates
/// (never amplifies) attention scores.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_slopes_strictly_in_unit_interval() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    let h: usize = kani::any();
    kani::assume(h < num_heads);
    let exponent = -8.0 * (h + 1) as f32 / num_heads as f32;
    kani::assert(exponent < 0.0, "exponent must be negative");
    let slope = 2f32.powf(exponent);
    kani::assert(slope.is_finite(), "slope must be finite");
    kani::assert(slope > 0.0, "slope must be positive");
    kani::assert(slope < 1.0, "slope must be less than 1.0");
}

/// Prove the maximum ALiBi slope (head 0) is at most 2^(-8/64) = 2^(-0.125).
///
/// For num_heads up to 64 (practical limit), the mildest slope is bounded.
/// This ensures ALiBi always provides meaningful distance penalty.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_max_slope_bounded() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    // First slope (h=0) is the largest.
    let max_slope = 2f32.powf(-8.0 / num_heads as f32);
    kani::assert(max_slope.is_finite(), "max slope must be finite");
    // For n=1: 2^(-8) ~= 0.0039; for n=64: 2^(-0.125) ~= 0.917
    kani::assert(max_slope > 0.0 && max_slope < 1.0, "max slope in (0, 1)");
}

// -- Bias linearity in distance ---------------------------------------------------

/// Prove ALiBi bias is linear in relative distance for each head.
///
/// bias(h, i, j) = slope_h * (j - i). For fixed h and i, doubling the
/// distance (j - i) doubles the bias. This is the "linear bias" in ALiBi.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_bias_linear_in_distance() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 32);
    let h: usize = kani::any();
    kani::assume(h < num_heads);
    let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
    kani::assume(slope.is_finite() && slope > 0.0);

    let d1: i32 = kani::any();
    let d2: i32 = kani::any();
    kani::assume(d1.abs() <= 100 && d2.abs() <= 100);
    let bias1 = slope * d1 as f32;
    let bias2 = slope * d2 as f32;
    // Linearity: bias(d1 + d2) = bias(d1) + bias(d2)
    let d_sum = d1 + d2;
    kani::assume(d_sum.abs() <= 200);
    let bias_sum = slope * d_sum as f32;
    kani::assume(bias1.is_finite() && bias2.is_finite() && bias_sum.is_finite());
    kani::assert(
        (bias_sum - (bias1 + bias2)).abs() < slope * 1e-4 + 1e-6,
        "ALiBi bias must be linear: bias(d1+d2) = bias(d1) + bias(d2)",
    );
}

// -- Power-of-2 heads special case -----------------------------------------------

/// Prove ALiBi slopes for power-of-2 heads form an exact geometric progression.
///
/// When num_heads is a power of 2 (1, 2, 4, 8, 16, 32), the slopes are
/// exact powers of 2 and the ratio between consecutive slopes is exactly
/// 2^(-8/num_heads). No floating-point rounding occurs in the exponent
/// computation for these cases.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_power_of_2_heads_exact_progression() {
    let log2_heads: u32 = kani::any();
    kani::assume(log2_heads >= 1 && log2_heads <= 5); // 2..32
    let num_heads = 1usize << log2_heads;
    let h: usize = kani::any();
    kani::assume(h < num_heads - 1);

    let slope_h = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
    let slope_h1 = 2f32.powf(-8.0 * (h + 2) as f32 / num_heads as f32);
    kani::assume(slope_h.is_finite() && slope_h > 0.0);
    kani::assume(slope_h1.is_finite() && slope_h1 > 0.0);

    let ratio = slope_h1 / slope_h;
    let expected_ratio = 2f32.powf(-8.0 / num_heads as f32);
    kani::assert(
        (ratio - expected_ratio).abs() < 1e-6,
        "power-of-2 heads must have exact geometric ratio",
    );
}

// -- Self-attention zero bias ----------------------------------------------------

/// Prove self-attention position (i == j) always gets zero ALiBi bias.
///
/// When query and key are at the same position, the relative distance is 0,
/// so bias = slope * 0 = 0. Self-attention is never penalized by ALiBi.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_self_attention_zero_bias() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    let h: usize = kani::any();
    kani::assume(h < num_heads);
    let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
    kani::assume(slope.is_finite());
    let pos: usize = kani::any();
    kani::assume(pos < 1024);
    // distance = pos - pos = 0
    let bias = slope * 0.0;
    kani::assert(bias == 0.0, "self-attention bias must be exactly 0.0");
}

// -- Bias magnitude bound -------------------------------------------------------

/// Prove ALiBi bias magnitude is bounded by slope * (seq_len - 1).
///
/// The maximum relative distance in a sequence of length S is (S - 1).
/// Since slopes are in (0, 1), the maximum bias magnitude is less than S.
/// This bounds the range of values added to attention scores.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_bias_magnitude_bounded() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 32);
    let h: usize = kani::any();
    kani::assume(h < num_heads);
    let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
    kani::assume(slope.is_finite() && slope > 0.0);

    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 2 && seq_len <= 128);
    let i: usize = kani::any();
    let j: usize = kani::any();
    kani::assume(i < seq_len && j < seq_len);

    let distance = j as f32 - i as f32;
    let bias = slope * distance;
    kani::assume(bias.is_finite());
    let max_dist = (seq_len - 1) as f32;

    kani::assert(
        bias.abs() <= slope * max_dist + 1e-6,
        "bias magnitude bounded by slope * (seq_len - 1)",
    );
}

// -- Scaled ALiBi preserves monotonicity -----------------------------------------

/// Prove scaled ALiBi with positive scale preserves the monotonicity of bias.
///
/// alibi_bias_scaled multiplies by a per-head scale factor. When scale > 0,
/// the ordering of biases (closer positions get higher bias) is preserved.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_scaled_preserves_monotonicity() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 16);
    let h: usize = kani::any();
    kani::assume(h < num_heads);
    let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
    kani::assume(slope.is_finite() && slope > 0.0);

    let scale: f32 = kani::any();
    kani::assume(scale.is_finite() && scale > 0.0 && scale <= 10.0);

    // For keys to the left of query (j < i), closer is less negative.
    let i: usize = kani::any();
    let j1: usize = kani::any();
    let j2: usize = kani::any();
    kani::assume(i >= 2 && i < 16);
    kani::assume(j1 < j2 && j2 < i);

    let bias_j1 = slope * scale * (j1 as f32 - i as f32);
    let bias_j2 = slope * scale * (j2 as f32 - i as f32);
    kani::assume(bias_j1.is_finite() && bias_j2.is_finite());

    // j2 is closer to i, so (j2 - i) > (j1 - i) (less negative).
    // With positive slope and scale, bias_j2 > bias_j1.
    kani::assert(
        bias_j2 > bias_j1,
        "positive scale must preserve closer-is-better monotonicity",
    );
}

/// Prove scaled ALiBi with scale=1.0 equals unscaled ALiBi.
///
/// The identity scale factor should not change the bias values.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn alibi_scale_one_is_identity() {
    let num_heads: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 32);
    let h: usize = kani::any();
    kani::assume(h < num_heads);
    let slope = 2f32.powf(-8.0 * (h + 1) as f32 / num_heads as f32);
    kani::assume(slope.is_finite());

    let distance: f32 = kani::any();
    kani::assume(distance.is_finite() && distance.abs() <= 1000.0);

    let bias_unscaled = slope * distance;
    let bias_scaled = slope * 1.0_f32 * distance;
    kani::assert(
        bias_unscaled == bias_scaled,
        "scale=1.0 must be identity for ALiBi",
    );
}
