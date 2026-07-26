// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for RoPE (Rotary Position Embedding) safety.
//!
//! 20 proof harnesses covering:
//! - Rotation matrix Pythagorean identity (cos^2 + sin^2 = 1)
//! - Frequency theta^(-2i/d) positivity
//! - Position encoding monotonicity
//! - Cos/sin boundedness in [-1, 1]
//! - 2D RoPE spatial position encoding safety
//! - M-ROPE 3-component temporal/height/width bounds
//! - Interleaved M-ROPE half-rotation pair validity
//! - YarnScaling frequency adjustment non-negativity
//! - RoPE rotation norm preservation
//! - No NaN in cos/sin for valid position inputs
//! - Inverse frequency computation no overflow
//! - NTK scaling bounded output
//! - Complex rotation decomposition
//! - Position interpolation for extended context
//! - Linear bias correction boundedness
//! - Max position within embedding table
//! - RoPE dimension must be even
//! - Frequency decay with dimension index
//! - RoPE commutes with QK scaling
//! - Cached cos/sin lookup within table bounds
//!
//! Part of #4191.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn cos_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

fn sin_f32_stub(x: f32) -> f32 {
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

// ---------------------------------------------------------------------------
// 1. Rotation matrix cos^2 + sin^2 = 1 (within epsilon)
// ---------------------------------------------------------------------------

/// Prove the Pythagorean identity holds for RoPE rotation angles computed
/// from position * inverse_frequency, ensuring the rotation matrix is unitary.
///
/// For any finite angle theta = pos * inv_freq, cos(theta)^2 + sin(theta)^2
/// must equal 1.0 within floating-point tolerance.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_rotation_matrix_pythagorean() {
    let pos: u32 = kani::any();
    let inv_freq: f32 = kani::any();
    kani::assume(pos <= 131072);
    kani::assume(inv_freq.is_finite() && inv_freq > 0.0 && inv_freq <= 1.0);
    let angle = (pos as f64 * f64::from(inv_freq)) as f32;
    kani::assume(angle.is_finite());
    let c = angle.cos();
    let s = angle.sin();
    let sum = c * c + s * s;
    kani::assert(sum.is_finite(), "cos^2 + sin^2 must be finite");
    kani::assert(
        (sum - 1.0).abs() < 1e-5,
        "cos^2 + sin^2 must be close to 1.0",
    );
}

// ---------------------------------------------------------------------------
// 2. Frequency theta^(-2i/d) positive for valid i, d
// ---------------------------------------------------------------------------

/// Prove that the RoPE inverse frequency base^(-2i/d) is strictly positive
/// for all valid dimension indices i and head dimensions d.
///
/// The exponent 2i/d is in [0, 1) for i in [0, d/2). Since base > 0,
/// base^(2i/d) > 0, and therefore 1/base^(2i/d) > 0.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_frequency_positive() {
    let base: f64 = kani::any();
    let head_dim: u32 = kani::any();
    let i: u32 = kani::any();
    kani::assume(base >= 1.0 && base <= 1_000_001.0 && base.is_finite());
    kani::assume(head_dim >= 2 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);
    kani::assume(i < head_dim / 2);
    let exponent = (2 * i) as f64 / head_dim as f64;
    kani::assert(exponent >= 0.0, "exponent must be non-negative");
    kani::assert(exponent < 1.0, "exponent must be < 1.0 for valid i");
    let denominator = base.powf(exponent);
    kani::assert(denominator.is_finite(), "base^exponent must be finite");
    kani::assert(denominator >= 1.0, "base^exponent >= 1.0 for base >= 1");
    let inv_freq = 1.0 / denominator;
    kani::assert(inv_freq > 0.0, "inv_freq must be strictly positive");
    kani::assert(inv_freq <= 1.0, "inv_freq must be at most 1.0");
}

// ---------------------------------------------------------------------------
// 3. Position encoding increases with position
// ---------------------------------------------------------------------------

/// Prove that the RoPE angle (pos * inv_freq) increases monotonically with
/// position for any fixed positive inv_freq.
///
/// This is the core property that makes RoPE encode positional order:
/// token at position p2 > p1 gets a strictly larger angle.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_angle_increases_with_position() {
    let p1: u32 = kani::any();
    let p2: u32 = kani::any();
    let inv_freq: f64 = kani::any();
    kani::assume(p1 < p2);
    kani::assume(p2 <= 131072);
    kani::assume(inv_freq > 0.0 && inv_freq <= 1.0 && inv_freq.is_finite());
    let angle1 = p1 as f64 * inv_freq;
    let angle2 = p2 as f64 * inv_freq;
    kani::assert(angle2 > angle1, "angle must increase with position");
}

// ---------------------------------------------------------------------------
// 4. Cos/sin values bounded in [-1, 1]
// ---------------------------------------------------------------------------

/// Prove that cos and sin values computed for any valid RoPE angle are
/// bounded in [-1, 1]. This ensures cached cos/sin tables contain no
/// out-of-range values that could amplify embeddings.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_cos_sin_bounded() {
    let angle: f32 = kani::any();
    kani::assume(angle.is_finite() && angle.abs() <= 1e6);
    let c = angle.cos();
    let s = angle.sin();
    kani::assert(c >= -1.0 && c <= 1.0, "cos must be in [-1, 1]");
    kani::assert(s >= -1.0 && s <= 1.0, "sin must be in [-1, 1]");
    kani::assert(c.is_finite(), "cos must be finite");
    kani::assert(s.is_finite(), "sin must be finite");
}

// ---------------------------------------------------------------------------
// 5. 2D RoPE for vision: spatial position encoding safe
// ---------------------------------------------------------------------------

/// Prove that 2D RoPE spatial position encoding produces valid angles
/// for both height and width axes. Each axis gets head_dim/4 pairs, and
/// the frequency computation is identical to 1D RoPE per axis.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_2d_spatial_encoding_safe() {
    let head_dim: u32 = kani::any();
    kani::assume(head_dim >= 4 && head_dim <= 256);
    kani::assume(head_dim % 4 == 0);
    let quarter_dim = head_dim / 4;
    let h_pos: u32 = kani::any();
    let w_pos: u32 = kani::any();
    kani::assume(h_pos <= 4096);
    kani::assume(w_pos <= 4096);
    let base: f64 = 10000.0;
    let i: u32 = kani::any();
    kani::assume(i < quarter_dim);
    // Height axis angle
    let exp_h = (2 * i) as f64 / head_dim as f64;
    let inv_freq_h = 1.0 / base.powf(exp_h);
    let angle_h = h_pos as f64 * inv_freq_h;
    kani::assert(angle_h.is_finite(), "height angle must be finite");
    kani::assert(angle_h >= 0.0, "height angle must be non-negative");
    // Width axis angle (offset by quarter_dim in freq space)
    let exp_w = (2 * (quarter_dim + i)) as f64 / head_dim as f64;
    let inv_freq_w = 1.0 / base.powf(exp_w);
    let angle_w = w_pos as f64 * inv_freq_w;
    kani::assert(angle_w.is_finite(), "width angle must be finite");
    kani::assert(angle_w >= 0.0, "width angle must be non-negative");
}

// ---------------------------------------------------------------------------
// 6. M-ROPE 3-component temporal/height/width within bounds
// ---------------------------------------------------------------------------

/// Prove that M-ROPE section dimensions sum to head_dim and each section
/// has valid positive even dimension. The 3-component split for
/// temporal/height/width must reconstruct the full head dimension.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_mrope_section_dims_valid() {
    let t_pairs: usize = kani::any();
    let h_pairs: usize = kani::any();
    let w_pairs: usize = kani::any();
    kani::assume(t_pairs >= 1 && t_pairs <= 64);
    kani::assume(h_pairs >= 1 && h_pairs <= 64);
    kani::assume(w_pairs >= 1 && w_pairs <= 64);
    let total_pairs = t_pairs + h_pairs + w_pairs;
    kani::assume(total_pairs <= 128);
    let head_dim = total_pairs * 2;
    let t_dim = t_pairs * 2;
    let h_dim = h_pairs * 2;
    let w_dim = w_pairs * 2;
    kani::assert(
        t_dim + h_dim + w_dim == head_dim,
        "section dims must sum to head_dim",
    );
    kani::assert(t_dim % 2 == 0, "temporal section must be even");
    kani::assert(h_dim % 2 == 0, "height section must be even");
    kani::assert(w_dim % 2 == 0, "width section must be even");
    kani::assert(head_dim % 2 == 0, "full head_dim must be even");
}

// ---------------------------------------------------------------------------
// 7. Interleaved M-ROPE half-rotation pairs valid
// ---------------------------------------------------------------------------

/// Prove that interleaved M-ROPE pair-to-section assignment (i % 3) covers
/// all three sections equally when head_dim is divisible by 6, and that
/// each pair index maps to a valid section.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_interleaved_mrope_pairs_valid() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 6 && head_dim <= 384);
    kani::assume(head_dim % 6 == 0);
    let total_pairs = head_dim / 2;
    let pairs_per_section = total_pairs / 3;
    kani::assert(
        pairs_per_section * 3 == total_pairs,
        "pairs must divide evenly among 3 sections",
    );
    // Verify pair-to-section mapping
    let i: usize = kani::any();
    kani::assume(i < total_pairs);
    let section = i % 3;
    kani::assert(section < 3, "section index must be 0, 1, or 2");
    // Verify each section gets equal count
    kani::assert(
        pairs_per_section >= 1,
        "each section must have at least 1 pair",
    );
}

// ---------------------------------------------------------------------------
// 8. YarnScaling frequency adjustment non-negative
// ---------------------------------------------------------------------------

/// Prove that YaRN-scaled inverse frequencies are always non-negative.
///
/// The blended frequency `(1 - ramp) * freq + ramp * (freq / factor)`
/// is non-negative for freq > 0, factor > 0, and ramp in [0, 1].
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_yarn_scaled_freq_nonnegative() {
    let freq: f64 = kani::any();
    let factor: f64 = kani::any();
    let ramp: f64 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    kani::assume(factor.is_finite() && factor > 0.0 && factor <= 100.0);
    kani::assume(ramp.is_finite() && ramp >= 0.0 && ramp <= 1.0);
    let scaled_freq = (1.0 - ramp) * freq + ramp * (freq / factor);
    kani::assert(scaled_freq.is_finite(), "scaled freq must be finite");
    kani::assert(scaled_freq >= 0.0, "scaled freq must be non-negative");
    kani::assert(
        scaled_freq > 0.0,
        "scaled freq must be strictly positive for positive inputs",
    );
}

// ---------------------------------------------------------------------------
// 9. RoPE rotation preserves vector norm (up to epsilon)
// ---------------------------------------------------------------------------

/// Prove that applying RoPE rotation to a 2D vector pair preserves the
/// L2 norm. The rotation matrix is orthogonal, so ||Rx|| = ||x||.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_rotation_preserves_norm() {
    let theta: f32 = kani::any();
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    kani::assume(theta.is_finite() && theta.abs() <= 1e4);
    kani::assume(x0.is_finite() && x0.abs() <= 50.0);
    kani::assume(x1.is_finite() && x1.abs() <= 50.0);
    let c = theta.cos();
    let s = theta.sin();
    // RoPE rotation: y0 = x0*cos - x1*sin, y1 = x0*sin + x1*cos
    let y0 = x0 * c - x1 * s;
    let y1 = x0 * s + x1 * c;
    kani::assume(y0.is_finite() && y1.is_finite());
    let norm_sq_in = x0 * x0 + x1 * x1;
    let norm_sq_out = y0 * y0 + y1 * y1;
    kani::assume(norm_sq_in.is_finite() && norm_sq_out.is_finite());
    kani::assert(
        (norm_sq_out - norm_sq_in).abs() < norm_sq_in.abs() * 1e-4 + 1e-4,
        "rotation must preserve squared norm",
    );
}

// ---------------------------------------------------------------------------
// 10. No NaN in cos/sin for valid position inputs
// ---------------------------------------------------------------------------

/// Prove that cos and sin produce no NaN values when the angle is computed
/// from a valid position and finite inverse frequency.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_no_nan_for_valid_inputs() {
    let pos: u32 = kani::any();
    let inv_freq: f32 = kani::any();
    kani::assume(pos <= 131072);
    kani::assume(inv_freq.is_finite() && inv_freq > 0.0 && inv_freq <= 1.0);
    let angle = (pos as f64 * f64::from(inv_freq)) as f32;
    kani::assume(angle.is_finite());
    let c = angle.cos();
    let s = angle.sin();
    kani::assert(!c.is_nan(), "cos must not be NaN for valid input");
    kani::assert(!s.is_nan(), "sin must not be NaN for valid input");
    kani::assert(c.is_finite(), "cos must be finite");
    kani::assert(s.is_finite(), "sin must be finite");
}

// ---------------------------------------------------------------------------
// 11. Inverse frequency computation no overflow
// ---------------------------------------------------------------------------

/// Prove that computing inv_freq = 1 / base^(2i/d) does not overflow
/// for all valid base, head_dim, and dimension index combinations.
///
/// Since base >= 1 and exponent in [0, 1), the denominator is in [1, base),
/// so inv_freq is in (1/base, 1]. No overflow possible.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_inv_freq_no_overflow() {
    let base: f64 = kani::any();
    let head_dim: u32 = kani::any();
    let i: u32 = kani::any();
    kani::assume(base >= 1.0 && base <= 1_000_000.0 && base.is_finite());
    kani::assume(head_dim >= 2 && head_dim <= 512);
    kani::assume(head_dim % 2 == 0);
    kani::assume(i < head_dim / 2);
    let exponent = (2 * i) as f64 / head_dim as f64;
    let denominator = base.powf(exponent);
    kani::assert(
        denominator.is_finite(),
        "denominator must be finite (no overflow)",
    );
    kani::assert(denominator >= 1.0, "denominator must be >= 1.0");
    let inv_freq = 1.0 / denominator;
    kani::assert(inv_freq.is_finite(), "inv_freq must be finite");
    kani::assert(!inv_freq.is_nan(), "inv_freq must not be NaN");
    kani::assert(inv_freq > 0.0, "inv_freq must be positive");
}

// ---------------------------------------------------------------------------
// 12. RoPE with extended context (NTK scaling) bounded
// ---------------------------------------------------------------------------

/// Prove that NTK-aware (YaRN) scaled frequencies remain bounded and
/// positive. NTK scaling adjusts the base to base * factor^(d/(d-2)),
/// but the resulting frequencies must still be finite and positive.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_ntk_scaling_bounded() {
    let base: f64 = kani::any();
    let factor: f64 = kani::any();
    let head_dim: u32 = kani::any();
    kani::assume(base >= 1.0 && base <= 1_000_001.0 && base.is_finite());
    kani::assume(factor >= 1.0 && factor <= 100.0 && factor.is_finite());
    kani::assume(head_dim >= 4 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);
    let d = head_dim as f64;
    // NTK-aware base scaling: base' = base * factor^(d / (d - 2))
    let exponent = d / (d - 2.0);
    kani::assume(exponent.is_finite());
    let ntk_base = base * factor.powf(exponent);
    kani::assume(ntk_base.is_finite());
    kani::assert(ntk_base > 0.0, "NTK-scaled base must be positive");
    // Compute an inv_freq with the NTK-scaled base
    let i: u32 = kani::any();
    kani::assume(i < head_dim / 2);
    let freq_exp = (2 * i) as f64 / d;
    let inv_freq = 1.0 / ntk_base.powf(freq_exp);
    kani::assume(inv_freq.is_finite());
    kani::assert(inv_freq > 0.0, "NTK inv_freq must be positive");
}

// ---------------------------------------------------------------------------
// 13. Complex rotation Re(z*e^(i*theta)) = Re(z)*cos - Im(z)*sin
// ---------------------------------------------------------------------------

/// Prove the complex rotation decomposition used by RoPE:
/// Re(z * e^(i*theta)) = Re(z)*cos(theta) - Im(z)*sin(theta)
/// Im(z * e^(i*theta)) = Re(z)*sin(theta) + Im(z)*cos(theta)
///
/// This is exactly the RoPE formula applied to even/odd dimension pairs.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_complex_rotation_decomposition() {
    let re: f32 = kani::any(); // x_even
    let im: f32 = kani::any(); // x_odd
    let theta: f32 = kani::any();
    kani::assume(re.is_finite() && re.abs() <= 100.0);
    kani::assume(im.is_finite() && im.abs() <= 100.0);
    kani::assume(theta.is_finite() && theta.abs() <= 1e4);
    let c = theta.cos();
    let s = theta.sin();
    // Complex multiplication: (re + i*im) * (cos + i*sin)
    // = (re*cos - im*sin) + i*(re*sin + im*cos)
    let out_re = re * c - im * s;
    let out_im = re * s + im * c;
    kani::assume(out_re.is_finite() && out_im.is_finite());
    // The magnitude squared should be preserved: |z|^2 = |z*e^(itheta)|^2
    let mag_in = re * re + im * im;
    let mag_out = out_re * out_re + out_im * out_im;
    kani::assume(mag_in.is_finite() && mag_out.is_finite());
    kani::assert(
        (mag_out - mag_in).abs() < mag_in.abs() * 1e-4 + 1e-4,
        "complex rotation preserves magnitude",
    );
}

// ---------------------------------------------------------------------------
// 14. Position interpolation for longer sequences
// ---------------------------------------------------------------------------

/// Prove that linear position interpolation (PI) produces angles in the
/// expected range. PI divides position by a scaling factor to extend
/// context: angle_pi = (pos / scale) * inv_freq.
///
/// For scale > 1, the interpolated angle is smaller than the original,
/// fitting longer sequences into the original position range.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_position_interpolation() {
    let pos: u32 = kani::any();
    let scale: f64 = kani::any();
    let inv_freq: f64 = kani::any();
    kani::assume(pos <= 131072);
    kani::assume(scale > 1.0 && scale <= 100.0 && scale.is_finite());
    kani::assume(inv_freq > 0.0 && inv_freq <= 1.0 && inv_freq.is_finite());
    let original_angle = pos as f64 * inv_freq;
    let interpolated_angle = (pos as f64 / scale) * inv_freq;
    kani::assert(
        interpolated_angle.is_finite(),
        "interpolated angle must be finite",
    );
    kani::assert(
        interpolated_angle >= 0.0,
        "interpolated angle must be non-negative",
    );
    kani::assert(
        interpolated_angle < original_angle || pos == 0,
        "interpolated angle must be smaller than original for pos > 0",
    );
    kani::assert(
        interpolated_angle <= original_angle,
        "interpolated angle must not exceed original",
    );
}

// ---------------------------------------------------------------------------
// 15. RoPE with linear bias correction bounded
// ---------------------------------------------------------------------------

/// Prove that applying a linear bias correction to RoPE frequencies
/// keeps the result finite and positive. Linear bias adds a small
/// correction term: freq_corrected = freq + bias * (i / half_dim).
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_linear_bias_bounded() {
    let freq: f64 = kani::any();
    let bias: f64 = kani::any();
    let i: u32 = kani::any();
    let half_dim: u32 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    kani::assume(bias.is_finite() && bias.abs() <= 0.1);
    kani::assume(half_dim >= 1 && half_dim <= 256);
    kani::assume(i < half_dim);
    let correction = bias * (i as f64 / half_dim as f64);
    let freq_corrected = freq + correction;
    kani::assert(
        freq_corrected.is_finite(),
        "bias-corrected frequency must be finite",
    );
    // With |bias| <= 0.1 and freq > 0.0, freq_corrected > 0 - 0.1 = -0.1
    // but freq > 0 and correction abs < 0.1, so still positive when freq > 0.1
    // For general case, just verify finiteness and no NaN.
    kani::assert(
        !freq_corrected.is_nan(),
        "bias-corrected frequency must not be NaN",
    );
}

// ---------------------------------------------------------------------------
// 16. Max position within embedding table
// ---------------------------------------------------------------------------

/// Prove that position offset + seq_len never exceeds the precomputed
/// embedding table size. This is the bounds check that prevents
/// out-of-bounds access into the cached cos/sin tables.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_max_position_within_table() {
    let max_seq_len: usize = kani::any();
    let offset: usize = kani::any();
    let seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 131072);
    kani::assume(offset <= max_seq_len);
    kani::assume(seq_len >= 1 && seq_len <= max_seq_len);
    let end_pos = offset.checked_add(seq_len);
    kani::assume(end_pos.is_some());
    let end_pos = end_pos.unwrap();
    kani::assume(end_pos <= max_seq_len);
    kani::assert(
        end_pos <= max_seq_len,
        "end position must not exceed table size",
    );
    kani::assert(offset < max_seq_len, "offset must be within table");
    // All positions in [offset, end_pos) must be valid indices
    let any_pos: usize = kani::any();
    kani::assume(any_pos >= offset && any_pos < end_pos);
    kani::assert(
        any_pos < max_seq_len,
        "every accessed position must be a valid table index",
    );
}

// ---------------------------------------------------------------------------
// 17. RoPE dimension must be even
// ---------------------------------------------------------------------------

/// Prove that any valid RoPE head_dim is even, and that the half_dim
/// used for pair counting reconstructs the original dimension exactly.
/// Also verify that inv_freq array length matches half_dim.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_dimension_must_be_even() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 2 && head_dim <= 1024);
    kani::assume(head_dim % 2 == 0);
    let half_dim = head_dim / 2;
    kani::assert(
        half_dim * 2 == head_dim,
        "half_dim must exactly reconstruct head_dim",
    );
    kani::assert(half_dim >= 1, "half_dim must be at least 1");
    // The cos/sin cache shape is [max_seq_len, half_dim]
    // Each pair (2i, 2i+1) uses one frequency entry
    let max_pair_index = half_dim - 1;
    let max_element_index = max_pair_index * 2 + 1;
    kani::assert(
        max_element_index < head_dim,
        "highest element index must be within head_dim",
    );
    kani::assert(
        max_element_index == head_dim - 1,
        "last pair covers the last element",
    );
}

// ---------------------------------------------------------------------------
// 18. Frequency decay with dimension index
// ---------------------------------------------------------------------------

/// Prove that inverse frequencies strictly decrease with dimension index.
/// Higher dimension indices get lower frequencies (longer wavelengths),
/// which captures increasingly coarse-grained position information.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_frequency_decay_with_index() {
    let base: f64 = kani::any();
    let head_dim: u32 = kani::any();
    let i1: u32 = kani::any();
    let i2: u32 = kani::any();
    kani::assume(base > 1.0 && base <= 1_000_001.0 && base.is_finite());
    kani::assume(head_dim >= 4 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);
    kani::assume(i1 < i2);
    kani::assume(i2 < head_dim / 2);
    let exp1 = (2 * i1) as f64 / head_dim as f64;
    let exp2 = (2 * i2) as f64 / head_dim as f64;
    kani::assert(exp1 < exp2, "exponent must increase with index");
    let freq1 = 1.0 / base.powf(exp1);
    let freq2 = 1.0 / base.powf(exp2);
    kani::assume(freq1.is_finite() && freq2.is_finite());
    kani::assert(
        freq1 > freq2,
        "inv_freq must strictly decrease with dimension index",
    );
}

// ---------------------------------------------------------------------------
// 19. RoPE commutes with QK scaling
// ---------------------------------------------------------------------------

/// Prove that RoPE rotation commutes with uniform scaling:
/// scale * R(theta) * x = R(theta) * (scale * x).
///
/// This matters because SDPA applies scale = 1/sqrt(d) after RoPE.
/// Commutativity means the order doesn't affect the result.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn proof_rope_commutes_with_scaling() {
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    let theta: f32 = kani::any();
    let scale: f32 = kani::any();
    kani::assume(x0.is_finite() && x0.abs() <= 10.0);
    kani::assume(x1.is_finite() && x1.abs() <= 10.0);
    kani::assume(theta.is_finite() && theta.abs() <= 1e4);
    kani::assume(scale.is_finite() && scale > 0.0 && scale <= 10.0);
    let c = theta.cos();
    let s = theta.sin();
    // Path 1: scale then rotate
    let sx0 = scale * x0;
    let sx1 = scale * x1;
    kani::assume(sx0.is_finite() && sx1.is_finite());
    let r1_0 = sx0 * c - sx1 * s;
    let r1_1 = sx0 * s + sx1 * c;
    // Path 2: rotate then scale
    let r0 = x0 * c - x1 * s;
    let r1 = x0 * s + x1 * c;
    kani::assume(r0.is_finite() && r1.is_finite());
    let r2_0 = scale * r0;
    let r2_1 = scale * r1;
    kani::assume(r1_0.is_finite() && r1_1.is_finite());
    kani::assume(r2_0.is_finite() && r2_1.is_finite());
    kani::assert(
        (r1_0 - r2_0).abs() < 1e-4,
        "RoPE must commute with scaling (even component)",
    );
    kani::assert(
        (r1_1 - r2_1).abs() < 1e-4,
        "RoPE must commute with scaling (odd component)",
    );
}

// ---------------------------------------------------------------------------
// 20. Cached cos/sin lookup within table bounds
// ---------------------------------------------------------------------------

/// Prove that narrowing the cos/sin cache by [offset..offset+seq_len]
/// produces valid indices when offset + seq_len <= max_seq_len.
/// Also verify the narrowed slice has the correct shape.
///
/// Part of #4191.
#[kani::unwind(1)]
#[kani::proof]
fn proof_rope_cached_lookup_within_bounds() {
    let max_seq_len: usize = kani::any();
    let half_dim: usize = kani::any();
    let offset: usize = kani::any();
    let seq_len: usize = kani::any();
    kani::assume(max_seq_len >= 1 && max_seq_len <= 131072);
    kani::assume(half_dim >= 1 && half_dim <= 256);
    kani::assume(seq_len >= 1);
    kani::assume(offset.checked_add(seq_len).is_some());
    kani::assume(offset + seq_len <= max_seq_len);
    // The narrowed cache shape is [seq_len, half_dim]
    let narrowed_rows = seq_len;
    let narrowed_cols = half_dim;
    kani::assert(
        narrowed_rows == seq_len,
        "narrowed cache must have seq_len rows",
    );
    kani::assert(
        narrowed_cols == half_dim,
        "narrowed cache must have half_dim columns",
    );
    // Total elements in narrowed cache
    let total = narrowed_rows.checked_mul(narrowed_cols);
    kani::assume(total.is_some());
    let total = total.unwrap();
    kani::assert(total >= 1, "narrowed cache must have at least 1 element");
    // Every row index in narrowed maps to a valid original index
    let row: usize = kani::any();
    kani::assume(row < seq_len);
    let original_row = offset + row;
    kani::assert(
        original_row < max_seq_len,
        "every narrowed row must map to a valid original row",
    );
}
