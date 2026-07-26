// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced Kani proof harnesses for RoPE (Rotary Position Embedding).
//!
//! Extends the base harnesses in `kani_sdpa_rope_proofs.rs` with:
//! - RoPE rotation inverse (applying -theta reverses the rotation)
//! - Frequency wavelength relationship (wavelength = 2*pi / inv_freq)
//! - Rotation norm preservation for concrete vector pairs
//! - YaRN ramp function clamping properties
//! - YaRN frequency blending invariants
//! - Half-RoPE dimension divisibility requirements
//! - 2D RoPE head_dim divisible by 4 constraint
//! - RoPE rotation additivity (rotation by theta1+theta2 = compose)
//!
//! Part of #3671.

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

// -- RoPE rotation inverse --------------------------------------------------------

/// Prove RoPE rotation is invertible: rotating by theta then by -theta
/// recovers the original vector.
///
/// If R(theta) is the rotation matrix, then R(-theta) = R(theta)^(-1).
/// For any input pair (x_even, x_odd), applying rotation theta and then
/// rotation -theta must recover the original values.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_rotation_inverse() {
    let theta: f32 = kani::any();
    let x_even: f32 = kani::any();
    let x_odd: f32 = kani::any();
    kani::assume(theta.is_finite() && theta.abs() <= 1e4);
    kani::assume(x_even.is_finite() && x_even.abs() <= 1e3);
    kani::assume(x_odd.is_finite() && x_odd.abs() <= 1e3);

    let c = theta.cos();
    let s = theta.sin();
    // Forward rotation by theta:
    let y_even = x_even * c - x_odd * s;
    let y_odd = x_even * s + x_odd * c;
    kani::assume(y_even.is_finite() && y_odd.is_finite());

    // Reverse rotation by -theta (cos(-t)=cos(t), sin(-t)=-sin(t)):
    let z_even = y_even * c - y_odd * (-s);
    let z_odd = y_even * (-s) + y_odd * c;
    kani::assume(z_even.is_finite() && z_odd.is_finite());

    // Must recover original values within float tolerance.
    kani::assert(
        (z_even - x_even).abs() < 1e-2,
        "inverse rotation must recover x_even",
    );
    kani::assert(
        (z_odd - x_odd).abs() < 1e-2,
        "inverse rotation must recover x_odd",
    );
}

/// Prove RoPE rotation by pi swaps and negates the pair.
///
/// cos(pi) = -1, sin(pi) = 0, so rotation by pi gives:
/// y_even = -x_even, y_odd = -x_odd. This is a 180-degree rotation.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_rotation_by_pi_negates() {
    let x_even: f32 = kani::any();
    let x_odd: f32 = kani::any();
    kani::assume(x_even.is_finite() && x_odd.is_finite());
    kani::assume(x_even.abs() <= 1e6 && x_odd.abs() <= 1e6);

    let theta = std::f32::consts::PI;
    let c = theta.cos();
    let s = theta.sin();
    let y_even = x_even * c - x_odd * s;
    let y_odd = x_even * s + x_odd * c;

    // cos(pi) ~= -1, sin(pi) ~= 0
    kani::assert(
        (y_even + x_even).abs() < x_even.abs() * 1e-5 + 1e-5,
        "rotation by pi: y_even ~= -x_even",
    );
    kani::assert(
        (y_odd + x_odd).abs() < x_odd.abs() * 1e-5 + 1e-5,
        "rotation by pi: y_odd ~= -x_odd",
    );
}

// -- Frequency and wavelength relationship ----------------------------------------

/// Prove RoPE wavelength = 2*pi / inv_freq is monotonically increasing.
///
/// Higher dimension indices have smaller inv_freq (lower frequency) and
/// therefore longer wavelengths. This is the key design: low-indexed dims
/// capture short-range position info, high-indexed dims capture long-range.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn rope_wavelength_monotonically_increasing() {
    let base: f64 = kani::any();
    let head_dim: u32 = kani::any();
    let i: u32 = kani::any();
    kani::assume(base > 1.0 && base <= 1_000_001.0);
    kani::assume(head_dim >= 4 && head_dim <= 256);
    kani::assume(head_dim % 2 == 0);
    kani::assume(i + 1 < head_dim / 2);

    let exp_i = (2 * i) as f64 / head_dim as f64;
    let exp_i1 = (2 * (i + 1)) as f64 / head_dim as f64;
    let freq_i = 1.0 / base.powf(exp_i);
    let freq_i1 = 1.0 / base.powf(exp_i1);
    kani::assume(freq_i.is_finite() && freq_i > 0.0);
    kani::assume(freq_i1.is_finite() && freq_i1 > 0.0);

    let wavelength_i = 2.0 * std::f64::consts::PI / freq_i;
    let wavelength_i1 = 2.0 * std::f64::consts::PI / freq_i1;
    kani::assert(
        wavelength_i < wavelength_i1,
        "wavelength must increase with dimension index",
    );
}

/// Prove first RoPE frequency (i=0) equals 1.0 (base^0 = 1, inv_freq = 1/1 = 1).
///
/// At dimension index 0, the exponent is 0, so base^0 = 1 and inv_freq[0] = 1.
/// This is the highest frequency in the RoPE spectrum.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn rope_first_frequency_is_one() {
    let base: f64 = kani::any();
    kani::assume(base > 0.0 && base.is_finite());
    let head_dim: u32 = kani::any();
    kani::assume(head_dim >= 2 && head_dim <= 512);
    kani::assume(head_dim % 2 == 0);

    let exponent = 0.0_f64 / head_dim as f64; // i=0 => exponent=0
    let inv_freq = 1.0 / base.powf(exponent);
    kani::assert(
        (inv_freq - 1.0).abs() < 1e-12,
        "inv_freq[0] must equal 1.0 for any base",
    );
}

// -- RoPE norm preservation for vector pairs ------------------------------------

/// Prove RoPE rotation preserves squared norm of a 2D vector pair.
///
/// For input pair (x0, x1), the output pair (y0, y1) after rotation by theta
/// has ||y||^2 = ||x||^2. This follows from the orthogonality of the
/// rotation matrix, and ensures RoPE doesn't change embedding magnitude.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_rotation_preserves_squared_norm() {
    let theta: f32 = kani::any();
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    kani::assume(theta.is_finite() && theta.abs() <= 1e4);
    kani::assume(x0.is_finite() && x0.abs() <= 100.0);
    kani::assume(x1.is_finite() && x1.abs() <= 100.0);

    let c = theta.cos();
    let s = theta.sin();
    let y0 = x0 * c - x1 * s;
    let y1 = x0 * s + x1 * c;
    kani::assume(y0.is_finite() && y1.is_finite());

    let norm_in = x0 * x0 + x1 * x1;
    let norm_out = y0 * y0 + y1 * y1;
    kani::assume(norm_in.is_finite() && norm_out.is_finite());

    kani::assert(
        (norm_out - norm_in).abs() < norm_in.abs() * 1e-4 + 1e-4,
        "rotation must preserve squared norm",
    );
}

// -- YaRN ramp function properties ------------------------------------------------

/// Prove YaRN ramp function output is clamped to [0, 1].
///
/// The YaRN ramp blends between high-frequency (no scaling) and low-frequency
/// (linear interpolation): `ramp = clamp((wavelen - low) / range, 0, 1)`.
/// The clamp guarantees the blend factor is always valid.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn yarn_ramp_clamped_to_unit_interval() {
    let wavelen: f64 = kani::any();
    let low_freq_wavelen: f64 = kani::any();
    let high_freq_wavelen: f64 = kani::any();
    kani::assume(wavelen.is_finite() && wavelen > 0.0);
    kani::assume(low_freq_wavelen.is_finite() && low_freq_wavelen > 0.0);
    kani::assume(high_freq_wavelen.is_finite() && high_freq_wavelen > low_freq_wavelen);

    let wavelen_range = (high_freq_wavelen - low_freq_wavelen).max(1e-12);
    let raw_ramp = (wavelen - low_freq_wavelen) / wavelen_range;
    let ramp = raw_ramp.clamp(0.0, 1.0);

    kani::assert(ramp >= 0.0, "ramp must be non-negative");
    kani::assert(ramp <= 1.0, "ramp must be at most 1.0");
    kani::assert(ramp.is_finite(), "ramp must be finite");
}

/// Prove YaRN blended frequency is between original and scaled frequencies.
///
/// `scaled_freq = (1 - ramp) * freq + ramp * (freq / factor)`.
/// Since ramp in [0, 1] and factor > 1, the blended frequency is between
/// freq/factor (fully scaled) and freq (unscaled).
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn yarn_blended_freq_bounded() {
    let freq: f64 = kani::any();
    let factor: f64 = kani::any();
    let ramp: f64 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    kani::assume(factor.is_finite() && factor > 1.0 && factor <= 100.0);
    kani::assume(ramp >= 0.0 && ramp <= 1.0 && ramp.is_finite());

    let scaled_freq = (1.0 - ramp) * freq + ramp * (freq / factor);
    let freq_min = freq / factor;

    kani::assert(scaled_freq.is_finite(), "blended freq must be finite");
    kani::assert(
        scaled_freq >= freq_min - 1e-12,
        "blended freq >= freq/factor",
    );
    kani::assert(scaled_freq <= freq + 1e-12, "blended freq <= original freq");
}

/// Prove YaRN ramp=0 preserves original frequency (high-freq dimensions).
///
/// When the wavelength is below the low-frequency threshold, ramp=0 and
/// the frequency is unchanged. These are high-frequency (short-range) dims.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn yarn_ramp_zero_preserves_freq() {
    let freq: f64 = kani::any();
    let factor: f64 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    kani::assume(factor.is_finite() && factor > 1.0);

    let ramp = 0.0_f64;
    let blended = (1.0 - ramp) * freq + ramp * (freq / factor);
    kani::assert(
        (blended - freq).abs() < 1e-12,
        "ramp=0 must preserve original frequency",
    );
}

/// Prove YaRN ramp=1 applies full interpolation (low-freq dimensions).
///
/// When the wavelength exceeds the high-frequency threshold, ramp=1 and
/// the frequency is divided by the scaling factor. These are low-frequency
/// (long-range) dims that need interpolation for extended context.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn yarn_ramp_one_applies_full_interpolation() {
    let freq: f64 = kani::any();
    let factor: f64 = kani::any();
    kani::assume(freq.is_finite() && freq > 0.0 && freq <= 1.0);
    kani::assume(factor.is_finite() && factor > 1.0 && factor <= 100.0);

    let ramp = 1.0_f64;
    let blended = (1.0 - ramp) * freq + ramp * (freq / factor);
    let expected = freq / factor;
    kani::assert(
        (blended - expected).abs() < 1e-12,
        "ramp=1 must apply freq/factor interpolation",
    );
}

// -- Half-RoPE dimension constraints --------------------------------------------

/// Prove Half-RoPE head_dim divisible by 4 ensures rope_dim is even.
///
/// HalfRotaryEmbedding rotates only the first head_dim/2 elements.
/// That half must itself be even for RoPE pairing. So head_dim must be
/// divisible by 4: head_dim/2 = rope_dim, and rope_dim % 2 == 0.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn half_rope_dim_divisibility() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 4 && head_dim <= 512);
    kani::assume(head_dim % 4 == 0);
    let rope_dim = head_dim / 2;
    let half_rope_dim = rope_dim / 2;
    kani::assert(rope_dim % 2 == 0, "rope_dim must be even for pairing");
    kani::assert(
        half_rope_dim * 2 == rope_dim,
        "half_rope_dim splits cleanly",
    );
    kani::assert(head_dim == 4 * half_rope_dim, "head_dim = 4 * quarter");
}

// -- 2D RoPE dimension constraints -----------------------------------------------

/// Prove 2D RoPE head_dim divisible by 4 yields clean quarter splits.
///
/// RotaryEmbedding2d splits head_dim: half for height, half for width.
/// Each half is further split into pairs for rotation. So head_dim must
/// be divisible by 4, giving quarter_dim = head_dim / 4 pairs per axis.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
fn rope_2d_quarter_dim_splits_cleanly() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 4 && head_dim <= 512);
    kani::assume(head_dim % 4 == 0);
    let half_dim = head_dim / 2;
    let quarter_dim = half_dim / 2;
    kani::assert(half_dim * 2 == head_dim, "half splits cleanly");
    kani::assert(quarter_dim * 2 == half_dim, "quarter splits cleanly");
    kani::assert(quarter_dim * 4 == head_dim, "quarter_dim * 4 = head_dim");
    kani::assert(quarter_dim >= 1, "quarter_dim must be at least 1");
}

// -- RoPE rotation additivity (composition) ---------------------------------------

/// Prove composing two RoPE rotations is equivalent to a single rotation by the sum.
///
/// R(theta1) * R(theta2) = R(theta1 + theta2) for rotation matrices.
/// This means rotating by pos1 * freq then by pos2 * freq is the same as
/// rotating by (pos1 + pos2) * freq in one step.
///
/// Part of #3671.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::cos, cos_f32_stub)]
#[kani::stub(f32::sin, sin_f32_stub)]
fn rope_rotation_additivity() {
    let theta1: f32 = kani::any();
    let theta2: f32 = kani::any();
    let x0: f32 = kani::any();
    let x1: f32 = kani::any();
    kani::assume(theta1.is_finite() && theta1.abs() <= 100.0);
    kani::assume(theta2.is_finite() && theta2.abs() <= 100.0);
    kani::assume(x0.is_finite() && x0.abs() <= 10.0);
    kani::assume(x1.is_finite() && x1.abs() <= 10.0);

    // Two-step rotation: first by theta1, then by theta2.
    let c1 = theta1.cos();
    let s1 = theta1.sin();
    let y0 = x0 * c1 - x1 * s1;
    let y1 = x0 * s1 + x1 * c1;
    kani::assume(y0.is_finite() && y1.is_finite());

    let c2 = theta2.cos();
    let s2 = theta2.sin();
    let z0 = y0 * c2 - y1 * s2;
    let z1 = y0 * s2 + y1 * c2;
    kani::assume(z0.is_finite() && z1.is_finite());

    // Single-step rotation by theta1 + theta2.
    let theta_sum = theta1 + theta2;
    kani::assume(theta_sum.is_finite());
    let cs = theta_sum.cos();
    let ss = theta_sum.sin();
    let w0 = x0 * cs - x1 * ss;
    let w1 = x0 * ss + x1 * cs;
    kani::assume(w0.is_finite() && w1.is_finite());

    kani::assert(
        (z0 - w0).abs() < 5e-3,
        "composed rotation must match single rotation (even)",
    );
    kani::assert(
        (z1 - w1).abs() < 5e-3,
        "composed rotation must match single rotation (odd)",
    );
}
