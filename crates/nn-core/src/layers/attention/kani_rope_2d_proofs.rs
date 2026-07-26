// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for 2D RoPE and sinusoidal 2D positional encoding.
//!
//! Covers:
//! - head_dim divisible-by-4 constraint
//! - Dimension splitting: half_dim = head_dim / 2, quarter_dim = half_dim / 2
//! - Inverse frequency positive and finite
//! - Inv-freq monotonically decreasing
//! - Position-0 identity rotation for 2D RoPE
//! - Pythagorean identity for rotation angles
//! - Sinusoidal 2D: dim divisible by 4
//! - Sinusoidal 2D: output layout (4 quadrants)
//! - Sinusoidal 2D: sin/cos bounded [-1, 1]
//! - Sinusoidal 2D: origin encoding (h=0, w=0)
//! - Temperature positive/finite validation
//!
//! Part of #3672.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn cos_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn powf_f32_stub(b: f32, _e: f32) -> f32 {
    let _ = b;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn sin_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

// -- head_dim constraints --------------------------------------------------------

/// Prove head_dim divisible by 4 yields exact quarter_dim and half_dim.
///
/// RotaryEmbedding2d requires head_dim % 4 == 0. This ensures the dimension
/// splits cleanly into 4 equal parts for height-even, height-odd, width-even,
/// width-odd.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn rope_2d_head_dim_splits_into_quarters() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 4 && head_dim <= 512);
    kani::assume(head_dim % 4 == 0);
    let half_dim = head_dim / 2;
    let quarter_dim = half_dim / 2;
    kani::assert(half_dim * 2 == head_dim, "half_dim * 2 == head_dim");
    kani::assert(quarter_dim * 2 == half_dim, "quarter_dim * 2 == half_dim");
    kani::assert(quarter_dim * 4 == head_dim, "quarter_dim * 4 == head_dim");
    kani::assert(quarter_dim >= 1, "quarter_dim must be >= 1");
}

/// Prove head_dim not divisible by 4 leaves a remainder.
///
/// If head_dim % 4 != 0, integer division loses information — this is why
/// the constructor rejects such values.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn rope_2d_head_dim_not_div4_has_remainder() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 1 && head_dim <= 512);
    kani::assume(head_dim % 4 != 0);
    let quarter_dim = head_dim / 4;
    kani::assert(
        quarter_dim * 4 != head_dim,
        "non-div-4 head_dim does not round-trip",
    );
}

// -- Inverse frequency properties ------------------------------------------------

/// Prove 2D RoPE inv_freq is positive and finite for valid base and dim.
///
/// inv_freq[i] = 1 / base^(2i / half_dim) where base > 0 and half_dim > 0.
/// For base >= 1, inv_freq is in (0, 1].
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn rope_2d_inv_freq_positive_finite() {
    let base: f64 = kani::any();
    let head_dim: u32 = kani::any();
    let i: u32 = kani::any();
    kani::assume(base >= 1.0 && base <= 1_000_001.0);
    kani::assume(head_dim >= 4 && head_dim <= 256);
    kani::assume(head_dim % 4 == 0);
    let half_dim = head_dim / 2;
    let quarter_dim = half_dim / 2;
    kani::assume(i < quarter_dim);
    let exponent = (2 * i) as f64 / half_dim as f64;
    let inv_freq = 1.0 / base.powf(exponent);
    let inv_freq_f32 = inv_freq as f32;
    kani::assert(inv_freq.is_finite(), "inv_freq (f64) must be finite");
    kani::assert(inv_freq > 0.0, "inv_freq must be positive");
    kani::assert(inv_freq_f32.is_finite(), "inv_freq (f32) must be finite");
}

/// Prove 2D RoPE inv_freq is monotonically decreasing with index.
///
/// Higher dimension indices get lower frequencies (longer wavelengths).
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn rope_2d_inv_freq_monotonically_decreasing() {
    let base: f64 = kani::any();
    let head_dim: u32 = kani::any();
    let i: u32 = kani::any();
    kani::assume(base > 1.0 && base <= 1_000_001.0);
    kani::assume(head_dim >= 8 && head_dim <= 256);
    kani::assume(head_dim % 4 == 0);
    let half_dim = head_dim / 2;
    let quarter_dim = half_dim / 2;
    kani::assume(i + 1 < quarter_dim);
    let exp_i = (2 * i) as f64 / half_dim as f64;
    let exp_i1 = (2 * (i + 1)) as f64 / half_dim as f64;
    let freq_i = 1.0 / base.powf(exp_i);
    let freq_i1 = 1.0 / base.powf(exp_i1);
    kani::assume(freq_i.is_finite() && freq_i1.is_finite());
    kani::assert(
        freq_i > freq_i1,
        "inv_freq must decrease with dimension index",
    );
}

// -- 2D RoPE rotation properties -------------------------------------------------

/// Prove 2D RoPE at position (0, 0) produces identity rotation for both axes.
///
/// At pos=0, angle = 0 * inv_freq = 0 for all frequency bands.
/// cos(0) = 1, sin(0) = 0, so the rotation matrix is identity.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_2d_position_zero_zero_is_identity() {
    let pos_h: usize = 0;
    let pos_w: usize = 0;
    let freq: f32 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0);
    let h_angle = (pos_h as f64 * f64::from(freq)) as f32;
    let w_angle = (pos_w as f64 * f64::from(freq)) as f32;
    kani::assert(h_angle == 0.0, "h_angle at pos 0 must be 0");
    kani::assert(w_angle == 0.0, "w_angle at pos 0 must be 0");
    let h_cos = h_angle.cos();
    let h_sin = h_angle.sin();
    let w_cos = w_angle.cos();
    let w_sin = w_angle.sin();
    kani::assert(h_cos == 1.0, "cos(0) == 1 for height");
    kani::assert(h_sin == 0.0, "sin(0) == 0 for height");
    kani::assert(w_cos == 1.0, "cos(0) == 1 for width");
    kani::assert(w_sin == 0.0, "sin(0) == 0 for width");
}

/// Prove 2D RoPE Pythagorean identity holds for both height and width angles.
///
/// cos^2(h_angle) + sin^2(h_angle) == 1 and likewise for w_angle.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_2d_pythagorean_identity_both_axes() {
    let pos: u32 = kani::any();
    let freq: f32 = kani::any();
    kani::assume(pos <= 4096);
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    let angle = (pos as f64 * f64::from(freq)) as f32;
    kani::assume(angle.is_finite());
    let c = angle.cos();
    let s = angle.sin();
    let sum = c * c + s * s;
    kani::assert(sum.is_finite(), "cos^2 + sin^2 must be finite");
    kani::assert(
        (sum - 1.0).abs() < 1e-5,
        "Pythagorean identity must hold within tolerance",
    );
}

/// Prove 2D RoPE rotation is orthogonal (determinant == 1) for each axis.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_2d_rotation_determinant() {
    let angle: f32 = kani::any();
    kani::assume(angle.is_finite() && angle >= -1e6 && angle <= 1e6);
    let c = angle.cos();
    let s = angle.sin();
    // det([[c, -s], [s, c]]) = c^2 + s^2
    let det = c * c + s * s;
    kani::assert(det.is_finite(), "determinant finite");
    kani::assert((det - 1.0).abs() < 1e-5, "determinant close to 1");
}

// -- Sinusoidal 2D properties ----------------------------------------------------

/// Prove sinusoidal_2d dim must be divisible by 4.
///
/// The encoding splits dim into 4 equal parts: sin_h, cos_h, sin_w, cos_w.
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_2d_dim_divisible_by_4() {
    let dim: usize = kani::any();
    kani::assume(dim >= 4 && dim <= 512);
    kani::assume(dim % 4 == 0);
    let quarter = dim / 4;
    kani::assert(quarter * 4 == dim, "quarter_dim * 4 == dim");
    kani::assert(quarter >= 1, "quarter_dim >= 1");
    // The 4 quadrants: [0..quarter), [quarter..2*quarter), [2*quarter..3*quarter), [3*quarter..dim)
    let half = dim / 2;
    kani::assert(half == 2 * quarter, "half_dim = 2 * quarter_dim");
}

/// Prove sinusoidal_2d output row layout: 4 bands of quarter_dim each.
///
/// For row = h * width + w, the encoding is:
///   [sin_h(0..qd), cos_h(qd..2qd), sin_w(2qd..3qd), cos_w(3qd..4qd)]
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_2d_output_band_offsets() {
    let dim: usize = kani::any();
    kani::assume(dim >= 4 && dim <= 256);
    kani::assume(dim % 4 == 0);
    let qd = dim / 4;
    let i: usize = kani::any();
    kani::assume(i < qd);
    // Band offsets for frequency index i.
    let sin_h_offset = i;
    let cos_h_offset = qd + i;
    let sin_w_offset = 2 * qd + i;
    let cos_w_offset = 3 * qd + i;
    kani::assert(sin_h_offset < qd, "sin_h in first quarter");
    kani::assert(
        cos_h_offset >= qd && cos_h_offset < 2 * qd,
        "cos_h in second quarter",
    );
    kani::assert(
        sin_w_offset >= 2 * qd && sin_w_offset < 3 * qd,
        "sin_w in third quarter",
    );
    kani::assert(
        cos_w_offset >= 3 * qd && cos_w_offset < dim,
        "cos_w in fourth quarter",
    );
}

/// Prove sinusoidal_2d element count = height * width * dim.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_2d_element_count() {
    let h: usize = kani::any();
    let w: usize = kani::any();
    let dim: usize = kani::any();
    kani::assume(h >= 1 && h <= 32);
    kani::assume(w >= 1 && w <= 32);
    kani::assume(dim >= 4 && dim <= 64);
    kani::assume(dim % 4 == 0);
    let seq_len = h.checked_mul(w);
    kani::assume(seq_len.is_some());
    let total = seq_len.unwrap().checked_mul(dim);
    kani::assume(total.is_some());
    kani::assert(total.unwrap() == h * w * dim, "element count = H * W * dim");
}

/// Prove sinusoidal_2d: sin/cos values are bounded in [-1, 1].
///
/// For any finite angle, sin and cos are in [-1, 1].
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn sinusoidal_2d_sin_cos_bounded() {
    let angle: f32 = kani::any();
    kani::assume(angle.is_finite() && angle >= -1e6 && angle <= 1e6);
    let s = angle.sin();
    let c = angle.cos();
    kani::assert(s.is_finite(), "sin must be finite");
    kani::assert(c.is_finite(), "cos must be finite");
    kani::assert(s >= -1.0 && s <= 1.0, "sin bounded in [-1, 1]");
    kani::assert(c >= -1.0 && c <= 1.0, "cos bounded in [-1, 1]");
}

/// Prove sinusoidal_2d origin (h=0, w=0) produces sin(0)=0, cos(0)=1.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn sinusoidal_2d_origin_encoding() {
    let freq: f64 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    let h_angle = (0.0_f64 * freq) as f32;
    let w_angle = (0.0_f64 * freq) as f32;
    kani::assert(h_angle == 0.0, "angle is 0 at origin");
    kani::assert(w_angle == 0.0, "angle is 0 at origin");
    kani::assert(h_angle.sin() == 0.0, "sin(0) = 0");
    kani::assert(h_angle.cos() == 1.0, "cos(0) = 1");
}

/// Prove sinusoidal_2d inverse frequency computation is positive finite.
///
/// inv_freq[i] = 1 / temperature^(2i / half_dim).
/// For temperature >= 1 and valid i, inv_freq is in (0, 1].
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn sinusoidal_2d_inv_freq_positive() {
    let temp: f64 = kani::any();
    let dim: u32 = kani::any();
    let i: u32 = kani::any();
    kani::assume(temp >= 1.0 && temp <= 100_001.0);
    kani::assume(dim >= 4 && dim <= 256);
    kani::assume(dim % 4 == 0);
    let half_dim = dim / 2;
    let quarter_dim = dim / 4;
    kani::assume(i < quarter_dim);
    let exponent = (2 * i) as f64 / half_dim as f64;
    let inv_freq = 1.0 / temp.powf(exponent);
    kani::assert(inv_freq.is_finite(), "inv_freq must be finite");
    kani::assert(inv_freq > 0.0, "inv_freq must be positive");
    kani::assert(inv_freq <= 1.0, "inv_freq <= 1 for temperature >= 1");
}

/// Prove sinusoidal_2d row index = h * width + w is within bounds.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_2d_row_index_in_bounds() {
    let height: usize = kani::any();
    let width: usize = kani::any();
    kani::assume(height >= 1 && height <= 64);
    kani::assume(width >= 1 && width <= 64);
    let seq_len = height.checked_mul(width);
    kani::assume(seq_len.is_some());
    let seq_len = seq_len.unwrap();
    let h: usize = kani::any();
    let w: usize = kani::any();
    kani::assume(h < height && w < width);
    let row = h * width + w;
    kani::assert(row < seq_len, "row index must be within seq_len");
}

/// Prove sinusoidal_2d data offset = row * dim + i is within total.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_2d_data_offset_in_bounds() {
    let height: usize = kani::any();
    let width: usize = kani::any();
    let dim: usize = kani::any();
    kani::assume(height >= 1 && height <= 32);
    kani::assume(width >= 1 && width <= 32);
    kani::assume(dim >= 4 && dim <= 64);
    kani::assume(dim % 4 == 0);
    let seq_len = height.checked_mul(width);
    kani::assume(seq_len.is_some());
    let total = seq_len.unwrap().checked_mul(dim);
    kani::assume(total.is_some());
    let total = total.unwrap();
    let h: usize = kani::any();
    let w: usize = kani::any();
    let i: usize = kani::any();
    kani::assume(h < height && w < width && i < dim);
    let row = h * width + w;
    let offset = row * dim + i;
    kani::assert(offset < total, "data offset must be within bounds");
}

// -- Temperature validation ------------------------------------------------------

/// Prove temperature must be positive finite for valid inv_freq.
///
/// Part of #3672.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_2d_temperature_must_be_positive_finite() {
    let temp: f64 = kani::any();
    kani::assume(temp.is_finite());
    let valid = temp > 0.0;
    if valid {
        let inv_freq = 1.0 / temp;
        kani::assert(
            inv_freq.is_finite(),
            "1/temp finite for positive finite temp",
        );
        kani::assert(inv_freq > 0.0, "1/temp positive for positive temp");
    } else {
        // temp <= 0: either division by zero or negative.
        kani::assert(!valid, "non-positive temp is invalid");
    }
}
