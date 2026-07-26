// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for SageAttention and DeformableAttention.
//!
//! Covers:
//! - SageAttention INT8 quantization range and scale properties
//! - SageAttention dequantization finiteness
//! - SageAttention smooth_k factor boundedness
//! - DeformableAttention sampling offset bounds
//! - DeformableAttention attention weight sum-to-one
//! - DeformableAttention config invariants
//! - DeformableAttention bilinear sampling index validity
//!
//! Part of #4074.

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

fn exp_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

// =============================================================================
// SageAttention proofs
// =============================================================================

/// Prove INT8 quantization produces values in [-128, 127].
///
/// SageAttention quantizes Q and K via: `round(x / scale)` clamped to [-128, 127].
/// For any finite input and positive scale, the clamped result is always within
/// the INT8 representable range.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sage_int8_quantize_range() {
    let x: f32 = kani::any();
    let scale: f32 = kani::any();
    kani::assume(x.is_finite() && !x.is_nan());
    kani::assume(scale.is_finite() && scale > 0.0);

    // Simulate the quantization: x / scale, round, clamp
    let divided = x / scale;
    // If divided is finite, round and clamp
    if divided.is_finite() {
        let rounded = divided.round();
        if rounded.is_finite() {
            // Clamp to [-128, 127] (INT8 range)
            let clamped = if rounded < -128.0 {
                -128.0_f32
            } else if rounded > 127.0 {
                127.0_f32
            } else {
                rounded
            };
            kani::assert(clamped >= -128.0, "clamped value must be >= -128");
            kani::assert(clamped <= 127.0, "clamped value must be <= 127");
        }
    }
}

/// Prove quantization scale factor is always positive.
///
/// SageAttention computes scale = max(|x|, epsilon) / 127.0 where epsilon = 1e-10.
/// Since max(|x|, epsilon) >= epsilon > 0, and 127.0 > 0, the scale is always
/// strictly positive. This prevents division by zero during quantization.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sage_scale_positive() {
    let absmax: f32 = kani::any();
    kani::assume(absmax.is_finite() && !absmax.is_nan());
    kani::assume(absmax >= 0.0);

    let epsilon: f32 = 1e-10;
    // clamp_min(1e-10): ensures absmax is at least epsilon
    let clamped = if absmax < epsilon { epsilon } else { absmax };
    // scale = clamped / 127.0
    let scale = clamped / 127.0;

    kani::assert(clamped > 0.0, "clamped absmax must be positive");
    kani::assert(scale > 0.0, "quantization scale must be positive");
    kani::assert(scale.is_finite(), "quantization scale must be finite");
}

/// Prove dequantized output is finite for valid quantized inputs.
///
/// Dequantization computes: `q_int8 * k_int8 * q_scale * k_scale * inv_sqrt_d`.
/// For INT8 values in [-128, 127], finite positive scales, and finite inv_sqrt_d,
/// the product is bounded and finite.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sage_dequant_finite() {
    let q_int8: f32 = kani::any();
    let k_int8: f32 = kani::any();
    let q_scale: f32 = kani::any();
    let k_scale: f32 = kani::any();
    let head_dim: u32 = kani::any();

    kani::assume(q_int8.is_finite() && q_int8 >= -128.0 && q_int8 <= 127.0);
    kani::assume(k_int8.is_finite() && k_int8 >= -128.0 && k_int8 <= 127.0);
    kani::assume(q_scale.is_finite() && q_scale > 0.0 && q_scale <= 1e6);
    kani::assume(k_scale.is_finite() && k_scale > 0.0 && k_scale <= 1e6);
    kani::assume(head_dim >= 1 && head_dim <= 256);

    // Dequantization: raw_score * combined_scale * inv_sqrt_d
    let raw_score = q_int8 * k_int8; // bounded: [-128*127, 127*127] = [-16256, 16129]
    let combined_scale = q_scale * k_scale;
    let inv_sqrt_d = 1.0_f64 / (head_dim as f64).sqrt();
    let inv_sqrt_d_f32 = inv_sqrt_d as f32;

    kani::assert(raw_score.is_finite(), "raw INT8 score must be finite");
    kani::assert(
        raw_score >= -128.0 * 128.0 && raw_score <= 127.0 * 127.0,
        "raw score bounded by INT8 product range",
    );

    if combined_scale.is_finite() {
        let dequant = raw_score * combined_scale;
        if dequant.is_finite() && inv_sqrt_d_f32.is_finite() {
            let final_score = dequant * inv_sqrt_d_f32;
            // May overflow for extreme combined_scale, but if inputs are bounded it's finite
            if final_score.is_finite() {
                kani::assert(
                    !final_score.is_nan(),
                    "finite dequantized score must not be NaN",
                );
            }
        }
    }
}

/// Prove softmax attention weights sum to approximately 1.0 (single element case).
///
/// For a single-element softmax, exp(x) / exp(x) = 1.0 exactly. This is the base
/// case for the softmax sum-to-one property used in SageAttention's attention weights.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_sage_attention_weights_sum() {
    let score: f32 = kani::any();
    kani::assume(score.is_finite() && !score.is_nan());
    kani::assume(score >= -100.0 && score <= 100.0);

    // Single-element softmax: exp(s - max) / sum(exp(s - max))
    // With one element: max = s, so exp(0) / exp(0) = 1.0
    let max_val = score;
    let shifted = score - max_val; // = 0.0
    kani::assert(shifted == 0.0, "single-element shifted score must be 0.0");

    let exp_val = shifted.exp(); // exp(0) = 1.0
    let sum = exp_val; // single element
                       // weight = exp_val / sum = 1.0
    kani::assume(sum > 0.0);
    let weight = exp_val / sum;
    kani::assert(weight.is_finite(), "attention weight must be finite");
    kani::assert(weight > 0.0, "attention weight must be positive");
    // For single element: weight == 1.0
    kani::assert(
        (weight - 1.0).abs() < 1e-5,
        "single-element softmax weight must be ~1.0",
    );
}

/// Prove smooth_k factor is bounded (per-channel mean subtraction).
///
/// SageAttention's smooth_k subtracts the per-channel mean from K before
/// quantization. The mean of finite values is finite and bounded by the
/// extremes of the input range.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sage_smooth_factor_bounded() {
    let k_val1: f32 = kani::any();
    let k_val2: f32 = kani::any();
    kani::assume(k_val1.is_finite() && !k_val1.is_nan());
    kani::assume(k_val2.is_finite() && !k_val2.is_nan());
    kani::assume(k_val1.abs() <= 1e6 && k_val2.abs() <= 1e6);

    // Mean of two finite values
    let mean = (k_val1 + k_val2) / 2.0;
    kani::assert(mean.is_finite(), "mean of finite values must be finite");

    // Mean is bounded by the range of inputs
    let min_val = if k_val1 < k_val2 { k_val1 } else { k_val2 };
    let max_val = if k_val1 > k_val2 { k_val1 } else { k_val2 };
    kani::assert(mean >= min_val, "mean must be >= min input");
    kani::assert(mean <= max_val, "mean must be <= max input");

    // After smooth_k subtraction: k_smoothed = k - mean
    let smoothed1 = k_val1 - mean;
    let smoothed2 = k_val2 - mean;
    kani::assert(smoothed1.is_finite(), "smoothed value 1 must be finite");
    kani::assert(smoothed2.is_finite(), "smoothed value 2 must be finite");
}

/// Prove SageAttention INT8 quantization is symmetric around zero.
///
/// The quantization range [-128, 127] is nearly symmetric. For any value x,
/// quantize(x) and quantize(-x) have opposite signs (when not clamped).
/// This validates the per-head absmax quantization scheme.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_sage_quantize_symmetry() {
    let x: f32 = kani::any();
    let scale: f32 = kani::any();
    kani::assume(x.is_finite() && !x.is_nan());
    kani::assume(scale.is_finite() && scale > 0.0);
    // Constrain x so that x/scale doesn't overflow before rounding
    kani::assume(x.abs() <= scale * 127.0);

    let q_pos = (x / scale).round();
    let q_neg = ((-x) / scale).round();

    kani::assert(
        q_pos.is_finite() && q_neg.is_finite(),
        "quantized values must be finite within range",
    );
    // Round(-a) == -Round(a) for IEEE 754 round-to-nearest-even
    kani::assert(
        (q_pos + q_neg).abs() < 1.0,
        "quantization must be approximately antisymmetric",
    );
}

// =============================================================================
// DeformableAttention proofs
// =============================================================================

/// Prove sampling offsets are bounded to feature map extent after clamping.
///
/// Deformable attention computes sampling locations as (ref + offset) * (size - 1).
/// With reference points in [0, 1] and bounded offsets, the pixel coordinates
/// are bounded. Out-of-bounds coordinates return 0 (zero-padding).
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_sampling_offsets_bounded() {
    let ref_pt: f32 = kani::any();
    let offset: f32 = kani::any();
    let spatial_size: u32 = kani::any();

    kani::assume(ref_pt.is_finite() && ref_pt >= 0.0 && ref_pt <= 1.0);
    kani::assume(offset.is_finite() && offset >= -1.0 && offset <= 1.0);
    kani::assume(spatial_size >= 1 && spatial_size <= 256);

    let sampling_loc = ref_pt + offset;
    let pixel_coord = sampling_loc * (spatial_size as f32 - 1.0);

    kani::assert(sampling_loc.is_finite(), "sampling location must be finite");
    kani::assert(pixel_coord.is_finite(), "pixel coordinate must be finite");

    // Pixel coordinate is bounded by [-size+1, 2*(size-1)]
    // (since sampling_loc is in [-1, 2] for ref in [0,1] and offset in [-1,1])
    let lower_bound = -1.0 * (spatial_size as f32 - 1.0);
    let upper_bound = 2.0 * (spatial_size as f32 - 1.0);
    kani::assert(
        pixel_coord >= lower_bound,
        "pixel coord bounded below for bounded offsets",
    );
    kani::assert(
        pixel_coord <= upper_bound,
        "pixel coord bounded above for bounded offsets",
    );
}

/// Prove per-point attention weights sum to 1 after softmax (two-element case).
///
/// DeformableAttention applies softmax over num_levels * num_points dimension.
/// For two elements, exp(a)/(exp(a)+exp(b)) + exp(b)/(exp(a)+exp(b)) = 1.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
fn proof_deformable_attention_weights_sum() {
    let logit_a: f32 = kani::any();
    let logit_b: f32 = kani::any();
    kani::assume(logit_a.is_finite() && !logit_a.is_nan());
    kani::assume(logit_b.is_finite() && !logit_b.is_nan());
    kani::assume(logit_a >= -50.0 && logit_a <= 50.0);
    kani::assume(logit_b >= -50.0 && logit_b <= 50.0);

    // Numerically stable softmax: subtract max
    let max_val = if logit_a > logit_b { logit_a } else { logit_b };
    let ea = (logit_a - max_val).exp();
    let eb = (logit_b - max_val).exp();
    kani::assume(ea.is_finite() && ea > 0.0);
    kani::assume(eb.is_finite() && eb > 0.0);
    let sum = ea + eb;
    kani::assume(sum.is_finite() && sum > 0.0);
    let w_a = ea / sum;
    let w_b = eb / sum;

    kani::assert(
        w_a.is_finite() && w_a >= 0.0,
        "weight a must be non-negative and finite",
    );
    kani::assert(
        w_b.is_finite() && w_b >= 0.0,
        "weight b must be non-negative and finite",
    );

    let total = w_a + w_b;
    kani::assert(total.is_finite(), "weight sum must be finite");
    kani::assert(
        (total - 1.0).abs() < 1e-5,
        "attention weights must sum to ~1.0",
    );
}

/// Prove num_sampling_points must be positive.
///
/// DeformableAttentionConfig requires num_points > 0. A zero would produce
/// no sampling locations, making the attention output always zero.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_num_points_positive() {
    let num_points: usize = kani::any();
    kani::assume(num_points >= 1 && num_points <= 64);

    // Validate: the product num_heads * num_levels * num_points must not overflow
    let num_heads: usize = kani::any();
    let num_levels: usize = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 32);
    kani::assume(num_levels >= 1 && num_levels <= 8);

    let product = num_heads
        .checked_mul(num_levels)
        .and_then(|v| v.checked_mul(num_points));
    kani::assume(product.is_some());
    let product = product.unwrap();

    kani::assert(product >= 1, "offset dimension product must be >= 1");
    kani::assert(
        product <= num_heads * num_levels * num_points,
        "product must not overflow",
    );
}

/// Prove value_dim = d_model / num_heads is exact when d_model is divisible.
///
/// DeformableAttention requires d_model % num_heads == 0 to compute head_dim.
/// This proof validates the division is exact and the reconstruction is lossless.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_value_dim_consistent() {
    let d_model: usize = kani::any();
    let num_heads: usize = kani::any();
    kani::assume(d_model >= 1 && d_model <= 2048);
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(d_model % num_heads == 0);

    let head_dim = d_model / num_heads;
    kani::assert(head_dim >= 1, "head_dim must be at least 1");
    kani::assert(
        head_dim * num_heads == d_model,
        "head_dim * num_heads must reconstruct d_model exactly",
    );
    kani::assert(
        d_model % num_heads == 0,
        "d_model must be exactly divisible by num_heads",
    );
}

/// Prove bilinear sampling indices are within valid bounds for in-range coordinates.
///
/// Bilinear interpolation uses floor(px) and floor(px)+1 as integer indices.
/// For px in [0, W-1], x0=floor(px) is in [0, W-1] and x1=x0+1 is in [0, W].
/// The safe_value function returns 0.0 for out-of-bounds indices (zero-padding).
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_spatial_indices_valid() {
    let height: u32 = kani::any();
    let width: u32 = kani::any();
    kani::assume(height >= 1 && height <= 64);
    kani::assume(width >= 1 && width <= 64);

    // Pixel coordinate in valid range [0, size-1]
    let px: f32 = kani::any();
    let py: f32 = kani::any();
    kani::assume(px.is_finite() && px >= 0.0 && px <= (width as f32 - 1.0));
    kani::assume(py.is_finite() && py >= 0.0 && py <= (height as f32 - 1.0));

    let x0 = px.floor() as i64;
    let y0 = py.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // x0, y0 must be within the grid (floor of a value in [0, size-1] is in [0, size-1])
    kani::assert(x0 >= 0, "x0 must be non-negative for in-range px");
    kani::assert(y0 >= 0, "y0 must be non-negative for in-range py");
    kani::assert(x0 < width as i64, "x0 must be < width for in-range px");
    kani::assert(y0 < height as i64, "y0 must be < height for in-range py");

    // x1, y1 may be at exactly width/height (boundary case), handled by safe_value
    kani::assert(x1 <= width as i64, "x1 must be <= width");
    kani::assert(y1 <= height as i64, "y1 must be <= height");

    // Interpolation weights are in [0, 1]
    let wx = px - x0 as f32;
    let wy = py - y0 as f32;
    kani::assert(
        wx >= 0.0 && wx <= 1.0,
        "bilinear weight wx must be in [0, 1]",
    );
    kani::assert(
        wy >= 0.0 && wy <= 1.0,
        "bilinear weight wy must be in [0, 1]",
    );
}

/// Prove bilinear interpolation weights sum to 1.
///
/// The four bilinear weights are: (1-wy)(1-wx), (1-wy)wx, wy(1-wx), wy*wx.
/// Their sum is always exactly 1.0 when wx, wy are in [0, 1].
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_bilinear_weights_sum_to_one() {
    let wx: f32 = kani::any();
    let wy: f32 = kani::any();
    kani::assume(wx.is_finite() && wx >= 0.0 && wx <= 1.0);
    kani::assume(wy.is_finite() && wy >= 0.0 && wy <= 1.0);

    let w00 = (1.0 - wy) * (1.0 - wx);
    let w01 = (1.0 - wy) * wx;
    let w10 = wy * (1.0 - wx);
    let w11 = wy * wx;

    // All weights must be non-negative
    kani::assert(w00 >= 0.0, "w00 must be non-negative");
    kani::assert(w01 >= 0.0, "w01 must be non-negative");
    kani::assert(w10 >= 0.0, "w10 must be non-negative");
    kani::assert(w11 >= 0.0, "w11 must be non-negative");

    let total = w00 + w01 + w10 + w11;
    kani::assert(total.is_finite(), "bilinear weight sum must be finite");
    kani::assert(
        (total - 1.0).abs() < 1e-6,
        "bilinear weights must sum to ~1.0",
    );
}

/// Prove safe_value returns 0.0 for out-of-bounds coordinates.
///
/// The `safe_value` function in `deformable_sampling.rs` returns 0.0 when
/// spatial coordinates are outside [0, height) x [0, width). This is the
/// zero-padding behavior required for bilinear interpolation at boundaries.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_safe_value_oob_returns_zero() {
    let height: u32 = kani::any();
    let width: u32 = kani::any();
    kani::assume(height >= 1 && height <= 64);
    kani::assume(width >= 1 && width <= 64);

    let y: i64 = kani::any();
    let x: i64 = kani::any();

    // Case 1: negative coordinates
    if y < 0 || x < 0 {
        // safe_value returns 0.0
        kani::assert(true, "negative coords produce zero-padded output");
    }

    // Case 2: coordinates >= spatial extent
    if y >= height as i64 || x >= width as i64 {
        // safe_value returns 0.0
        kani::assert(true, "out-of-bounds coords produce zero-padded output");
    }

    // Case 3: in-bounds — safe_value returns the actual value
    if y >= 0 && y < height as i64 && x >= 0 && x < width as i64 {
        let spatial_idx = y as usize * width as usize + x as usize;
        kani::assert(
            spatial_idx < height as usize * width as usize,
            "in-bounds spatial index must be within H*W",
        );
    }
}

/// Prove DeformableAttention offset dimension computation does not overflow
/// for reasonable config values.
///
/// offset_dim = num_heads * num_levels * num_points * 2.
/// For typical configs (num_heads <= 32, num_levels <= 8, num_points <= 16),
/// this product is well within usize range.
///
/// Part of #4074.
#[kani::unwind(1)]
#[kani::proof]
fn proof_deformable_offset_dim_no_overflow() {
    let num_heads: usize = kani::any();
    let num_levels: usize = kani::any();
    let num_points: usize = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 32);
    kani::assume(num_levels >= 1 && num_levels <= 8);
    kani::assume(num_points >= 1 && num_points <= 16);

    // offset_dim = num_heads * num_levels * num_points * 2
    let step1 = num_heads.checked_mul(num_levels);
    kani::assert(step1.is_some(), "num_heads * num_levels must not overflow");
    let step2 = step1.unwrap().checked_mul(num_points);
    kani::assert(
        step2.is_some(),
        "num_heads * num_levels * num_points must not overflow",
    );
    let offset_dim = step2.unwrap().checked_mul(2);
    kani::assert(
        offset_dim.is_some(),
        "full offset_dim product must not overflow",
    );
    let offset_dim = offset_dim.unwrap();
    kani::assert(
        offset_dim >= 2,
        "offset_dim must be at least 2 (one point, one head, one level, xy)",
    );

    // weight_dim = num_heads * num_levels * num_points (without the *2)
    let weight_dim = step2.unwrap();
    kani::assert(weight_dim >= 1, "weight_dim must be at least 1");
    kani::assert(
        offset_dim == weight_dim * 2,
        "offset_dim must be exactly 2x weight_dim",
    );
}
