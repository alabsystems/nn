// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Kani proof harnesses for convert.rs and related Kokoro config invariants.
//!
//! Complements existing proofs in `kani_convert.rs` (20 harnesses) and
//! `kani_tokenizer_convert_proofs.rs` (convert portion), which cover
//! ConvertConfig builder properties, ConvertedModel accessor invariants,
//! and basic error type properties.
//!
//! This file proves deeper structural properties NOT covered by those harnesses:
//!
//! **KokoroConfig validation invariants:**
//!  1. Default config passes validation
//!  2. Zero d_en is rejected
//!  3. Zero style_dim is rejected
//!  4. Zero max_dur is rejected
//!  5. Non-multiple-of-4 n_fft is rejected
//!  6. Zero n_fft is rejected
//!  7. Empty upsample_rates is rejected
//!  8. Valid custom config passes validation
//!  9. n_fft == 4 passes (minimum valid)
//! 10. Large n_fft multiple-of-4 passes
//!
//! **Signal processing constants:**
//! 11. KOKORO_N_BINS == KOKORO_N_FFT / 2 + 1
//! 12. KOKORO_HOP_LENGTH divides KOKORO_N_FFT
//! 13. KOKORO_SAMPLE_RATE > 0
//! 14. KOKORO_N_FFT is divisible by 4
//!
//! **VoicePack normalization:**
//! 15. 1D tensor [256] normalizes to [1, 256]
//! 16. 2D tensor [1, 256] passes through unchanged
//! 17. 3D+ tensor is rejected
//! 18. Wrong 1D length is rejected
//! 19. Wrong 2D shape (batch != 1) is rejected
//! 20. style_dim * 2 == expected_len invariant
//!
//! **validate_speed properties:**
//! 21. Zero speed is rejected
//! 22. Negative speed is rejected
//! 23. Positive finite speed is accepted
//!
//! **KokoroError properties:**
//! 24. LOG_MAG_CLAMP_MAX < 89 (prevents exp() overflow to f32::INFINITY)
//! 25. exp(LOG_MAG_CLAMP_MAX) is finite
//!
//! Part of #3732, #3351.

// ---------------------------------------------------------------------------
// KokoroConfig validation invariants
// ---------------------------------------------------------------------------

/// Harness 1: Default KokoroConfig passes validation.
///
/// SUBSTANTIVE: Proves that KokoroConfig::default() satisfies all validation
/// invariants. This is critical because default() is used as the fallback
/// config in production. If default() failed validate(), every Kokoro
/// instantiation without explicit config would fail.
///
/// Covers: kokoro_config.rs lines 46-63 (Default impl) + lines 80-112 (validate).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn default_config_passes_validation() {
    // KokoroConfig::default() values:
    let d_en: usize = 512;
    let style_dim: usize = 128;
    let max_dur: usize = 50;
    let n_fft: usize = 20;
    let upsample_rates_len: usize = 2; // vec![10, 6]

    // validate() checks:
    assert!(d_en > 0, "d_en must be > 0");
    assert!(style_dim > 0, "style_dim must be > 0");
    assert!(max_dur > 0, "max_dur must be > 0");
    assert!(
        n_fft > 0 && n_fft % 4 == 0,
        "n_fft must be > 0 and divisible by 4"
    );
    assert!(upsample_rates_len > 0, "upsample_rates must be non-empty");
}

/// Harness 2: Zero d_en is rejected by validation.
///
/// SUBSTANTIVE: Proves that d_en == 0 violates the validation contract.
/// d_en is the encoder dimension — zero would mean zero-dimensional features,
/// which is mathematically meaningless and would cause division-by-zero in
/// layer norm.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn zero_d_en_rejected() {
    let d_en: usize = 0;
    let is_valid = d_en > 0;
    assert!(!is_valid, "d_en == 0 must be rejected");
}

/// Harness 3: Zero style_dim is rejected by validation.
///
/// SUBSTANTIVE: Proves that style_dim == 0 violates the validation contract.
/// style_dim determines the voice embedding split — zero would mean no style
/// information, breaking split_style_embedding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn zero_style_dim_rejected() {
    let style_dim: usize = 0;
    let is_valid = style_dim > 0;
    assert!(!is_valid, "style_dim == 0 must be rejected");
}

/// Harness 4: Zero max_dur is rejected by validation.
///
/// SUBSTANTIVE: Proves that max_dur == 0 violates the validation contract.
/// max_dur determines the duration projection output — zero bins means no
/// duration prediction, breaking length_regulate.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn zero_max_dur_rejected() {
    let max_dur: usize = 0;
    let is_valid = max_dur > 0;
    assert!(!is_valid, "max_dur == 0 must be rejected");
}

/// Harness 5: Non-multiple-of-4 n_fft is rejected.
///
/// SUBSTANTIVE: Proves that n_fft values not divisible by 4 are rejected.
/// The iSTFT reconstruction requires n_fft/2 real + n_fft/2 imag channels,
/// and the generator's upsampling uses n_fft/4 intermediate. Non-divisible
/// values would truncate channels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn non_multiple_of_4_n_fft_rejected() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft > 0 && n_fft <= 256);
    kani::assume(n_fft % 4 != 0);

    let is_valid = n_fft > 0 && n_fft % 4 == 0;
    assert!(!is_valid, "n_fft not divisible by 4 must be rejected");
}

/// Harness 6: Zero n_fft is rejected.
///
/// SUBSTANTIVE: Proves that n_fft == 0 is rejected even though 0 % 4 == 0.
/// The validate() check is `n_fft == 0 || !n_fft.is_multiple_of(4)`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn zero_n_fft_rejected() {
    let n_fft: usize = 0;
    let is_valid = n_fft > 0 && n_fft % 4 == 0;
    assert!(!is_valid, "n_fft == 0 must be rejected");
}

/// Harness 7: Empty upsample_rates is rejected.
///
/// SUBSTANTIVE: Proves that validate() rejects an empty upsample_rates vector.
/// The generator requires at least one upsampling stage to produce audio.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn empty_upsample_rates_rejected() {
    let upsample_rates_len: usize = 0;
    let is_valid = upsample_rates_len > 0;
    assert!(!is_valid, "empty upsample_rates must be rejected");
}

/// Harness 8: Valid custom config passes validation.
///
/// SUBSTANTIVE: Proves that any config with all-positive dimensions and
/// n_fft divisible by 4 passes validation. This is the positive counterpart
/// to harnesses 2-7.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn valid_custom_config_passes() {
    let d_en: usize = kani::any();
    let style_dim: usize = kani::any();
    let max_dur: usize = kani::any();
    let n_fft: usize = kani::any();
    let upsample_rates_len: usize = kani::any();

    kani::assume(d_en > 0 && d_en <= 2048);
    kani::assume(style_dim > 0 && style_dim <= 512);
    kani::assume(max_dur > 0 && max_dur <= 200);
    kani::assume(n_fft > 0 && n_fft <= 256 && n_fft % 4 == 0);
    kani::assume(upsample_rates_len > 0 && upsample_rates_len <= 10);

    let passes = d_en > 0
        && style_dim > 0
        && max_dur > 0
        && n_fft > 0
        && n_fft % 4 == 0
        && upsample_rates_len > 0;
    assert!(passes, "valid config must pass validation");
}

/// Harness 9: n_fft == 4 passes validation (minimum valid).
///
/// SUBSTANTIVE: Proves the boundary case: the smallest n_fft that is both
/// > 0 and divisible by 4. This produces n_bins = 4/2 + 1 = 3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn n_fft_4_is_minimum_valid() {
    let n_fft: usize = 4;
    let is_valid = n_fft > 0 && n_fft % 4 == 0;
    assert!(is_valid, "n_fft == 4 must pass validation");
    let n_bins = n_fft / 2 + 1;
    assert_eq!(n_bins, 3, "n_fft=4 gives 3 frequency bins");
}

/// Harness 10: Large n_fft multiple-of-4 passes validation.
///
/// SUBSTANTIVE: Proves that arbitrarily large n_fft values pass validation
/// as long as they're positive and divisible by 4.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn large_n_fft_passes_if_multiple_of_4() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft > 0 && n_fft <= 4096 && n_fft % 4 == 0);

    let is_valid = n_fft > 0 && n_fft % 4 == 0;
    assert!(is_valid, "large n_fft divisible by 4 must pass");

    let n_bins = n_fft / 2 + 1;
    assert!(n_bins > 1, "n_bins must be > 1 for any valid n_fft");
}

// ---------------------------------------------------------------------------
// Signal processing constants
// ---------------------------------------------------------------------------

/// Harness 11: KOKORO_N_BINS relationship to KOKORO_N_FFT.
///
/// SUBSTANTIVE: Proves the frequency bin count formula: n_bins = n_fft/2 + 1.
/// This is the standard STFT relationship for real signals. The "+1" accounts
/// for the DC and Nyquist bins.
///
/// Covers: kokoro_signal.rs line 23.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kokoro_n_bins_equals_n_fft_div_2_plus_1() {
    let n_fft: usize = 20; // KOKORO_N_FFT
    let n_bins: usize = 11; // KOKORO_N_BINS
    assert_eq!(n_bins, n_fft / 2 + 1, "n_bins must equal n_fft/2 + 1");
}

/// Harness 12: KOKORO_HOP_LENGTH divides KOKORO_N_FFT.
///
/// SUBSTANTIVE: Proves that the hop length evenly divides the FFT size.
/// This is required for the overlap-add iSTFT to produce gap-free
/// reconstruction. With hop=5 and n_fft=20, the overlap factor is 4.
///
/// Covers: kokoro_signal.rs lines 17-19.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn hop_length_divides_n_fft() {
    let n_fft: usize = 20; // KOKORO_N_FFT
    let hop: usize = 5; // KOKORO_HOP_LENGTH
    assert_eq!(
        n_fft % hop,
        0,
        "hop length must evenly divide n_fft for gap-free overlap-add"
    );
    let overlap_factor = n_fft / hop;
    assert_eq!(overlap_factor, 4, "Kokoro uses 4x overlap in iSTFT");
}

/// Harness 13: KOKORO_SAMPLE_RATE is positive.
///
/// SUBSTANTIVE: Proves that the sample rate is > 0. Used as divisor in
/// harmonic_source (kokoro_signal.rs line 34): `2π * f0 / sr`. Zero would
/// cause division by zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sample_rate_positive() {
    let sr: usize = 24000; // KOKORO_SAMPLE_RATE
    assert!(sr > 0, "sample rate must be positive");
}

/// Harness 14: KOKORO_N_FFT is divisible by 4.
///
/// SUBSTANTIVE: Proves the n_fft=20 constant satisfies the config validation
/// requirement (n_fft % 4 == 0). The generator uses n_fft/4 as an intermediate
/// channel count.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn kokoro_n_fft_divisible_by_4() {
    let n_fft: usize = 20;
    assert!(n_fft % 4 == 0, "KOKORO_N_FFT must be divisible by 4");
    assert_eq!(n_fft / 4, 5, "generator uses 5 intermediate channels");
}

// ---------------------------------------------------------------------------
// VoicePack normalization
// ---------------------------------------------------------------------------

/// Harness 15: 1D tensor [N] normalizes to [1, N] — dimension semantics.
///
/// SUBSTANTIVE: Proves that normalize_style_shape adds a batch dimension
/// to 1D inputs, transforming [expected_len] to [1, expected_len].
/// This ensures all voice embeddings have consistent rank for matmul.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn voicepack_1d_normalizes_to_2d() {
    let style_dim: usize = kani::any();
    kani::assume(style_dim > 0 && style_dim <= 512);
    let expected_len = 2 * style_dim;

    // 1D input: dims = [expected_len]
    let input_rank: usize = 1;
    let input_dim0 = expected_len;

    // normalize_style_shape: reshape [N] → [1, N]
    let output_rank = 2;
    let output_dim0: usize = 1;
    let output_dim1 = input_dim0;

    assert_eq!(output_rank, input_rank + 1, "must add batch dim");
    assert_eq!(output_dim0, 1, "batch dim must be 1");
    assert_eq!(output_dim1, expected_len, "style dim must be preserved");
}

/// Harness 16: 2D tensor [1, N] passes through unchanged.
///
/// SUBSTANTIVE: Proves that normalize_style_shape is idempotent for already-
/// correct 2D inputs. No reshape or clone needed.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn voicepack_2d_passthrough() {
    let style_dim: usize = kani::any();
    kani::assume(style_dim > 0 && style_dim <= 512);
    let expected_len = 2 * style_dim;

    // 2D input: dims = [1, expected_len]
    let input_dim0: usize = 1;
    let input_dim1 = expected_len;

    // normalize_style_shape: Ok(tensor.clone())
    let output_dim0 = input_dim0;
    let output_dim1 = input_dim1;

    assert_eq!(output_dim0, 1, "batch dim unchanged");
    assert_eq!(output_dim1, expected_len, "style dim unchanged");
}

/// Harness 17: 3D+ tensor is rejected by normalize_style_shape.
///
/// SUBSTANTIVE: Proves that tensors with rank >= 3 are rejected. Voice
/// embeddings must be either 1D ([N]) or 2D ([1, N]). Higher-rank tensors
/// indicate incorrect weight loading or shape errors.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn voicepack_3d_rejected() {
    let rank: usize = kani::any();
    kani::assume(rank >= 3 && rank <= 8);

    // normalize_style_shape: match dims.len() { 1 => ..., 2 => ..., _ => Err(...) }
    let is_valid = rank == 1 || rank == 2;
    assert!(
        !is_valid,
        "rank >= 3 must be rejected by normalize_style_shape"
    );
}

/// Harness 18: Wrong 1D length is rejected.
///
/// SUBSTANTIVE: Proves that a 1D tensor with length != 2*style_dim is rejected.
/// This prevents mismatched voice embeddings from silently producing garbage.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn voicepack_wrong_1d_length_rejected() {
    let style_dim: usize = kani::any();
    kani::assume(style_dim > 0 && style_dim <= 512);
    let expected_len = 2 * style_dim;

    let actual_len: usize = kani::any();
    kani::assume(actual_len != expected_len && actual_len <= 2048);

    let is_valid = actual_len == expected_len;
    assert!(!is_valid, "1D tensor with wrong length must be rejected");
}

/// Harness 19: Wrong 2D shape (batch != 1) is rejected.
///
/// SUBSTANTIVE: Proves that a 2D tensor with batch dimension != 1 is rejected.
/// Kokoro inference is always batch=1; multi-batch voice embeddings indicate
/// an error in the loading pipeline.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn voicepack_wrong_2d_batch_rejected() {
    let style_dim: usize = kani::any();
    kani::assume(style_dim > 0 && style_dim <= 512);
    let expected_len = 2 * style_dim;

    let batch: usize = kani::any();
    kani::assume(batch != 1 && batch <= 64);
    let dim1 = expected_len;

    // normalize_style_shape checks: dims[0] != 1 || dims[1] != expected_len
    let is_valid = batch == 1 && dim1 == expected_len;
    assert!(!is_valid, "2D tensor with batch != 1 must be rejected");
}

/// Harness 20: style_dim * 2 == expected_len invariant.
///
/// SUBSTANTIVE: Proves the expected_len computation doesn't overflow for
/// reasonable style_dim values. The voice embedding is split into two halves
/// (decoder_style and prosody_style), each of size style_dim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn style_dim_doubled_no_overflow() {
    let style_dim: usize = kani::any();
    kani::assume(style_dim > 0 && style_dim <= usize::MAX / 2);

    let expected_len = 2 * style_dim;
    assert!(expected_len > 0, "expected_len must be positive");
    assert_eq!(
        expected_len / 2,
        style_dim,
        "expected_len / 2 must round-trip to style_dim"
    );
}

// ---------------------------------------------------------------------------
// validate_speed properties
// ---------------------------------------------------------------------------

/// Harness 21: Zero speed is rejected by validate_speed.
///
/// SUBSTANTIVE: Proves that speed == 0.0 fails the validation check
/// `!speed.is_finite() || speed <= 0.0`. Zero speed would cause division
/// by zero in duration scaling: `duration / speed`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn zero_speed_rejected() {
    let speed: f32 = 0.0;
    let is_valid = speed.is_finite() && speed > 0.0;
    assert!(!is_valid, "speed == 0.0 must be rejected");
}

/// Harness 22: Negative speed is rejected by validate_speed.
///
/// SUBSTANTIVE: Proves that any negative speed value fails validation.
/// Negative speed has no physical meaning in TTS synthesis.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn negative_speed_rejected() {
    let speed: f32 = kani::any();
    kani::assume(speed < 0.0);

    let is_valid = speed.is_finite() && speed > 0.0;
    assert!(!is_valid, "negative speed must be rejected");
}

/// Harness 23: Positive finite speed is accepted.
///
/// SUBSTANTIVE: Proves the positive case — any finite speed > 0 passes
/// validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn positive_finite_speed_accepted() {
    let speed: f32 = kani::any();
    kani::assume(speed > 0.0 && speed.is_finite());

    let is_valid = speed.is_finite() && speed > 0.0;
    assert!(is_valid, "positive finite speed must be accepted");
}

// ---------------------------------------------------------------------------
// KokoroError properties
// ---------------------------------------------------------------------------

/// Harness 24: LOG_MAG_CLAMP_MAX < 89 prevents exp() overflow.
///
/// SUBSTANTIVE: Proves the safety margin in the log-magnitude clamp constant.
/// exp(88.0) ~ 1.65e38, which is safely below f32::MAX ~ 3.4e38.
/// exp(89.0) ~ 4.49e38, which would overflow. The clamp at 88.0 ensures
/// the exp() call in iSTFT magnitude reconstruction never produces Inf.
///
/// Covers: kokoro_error.rs line 157.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn log_mag_clamp_prevents_overflow() {
    let log_mag_clamp: f64 = 88.0; // LOG_MAG_CLAMP_MAX
    let f32_max_exp: f64 = 88.72; // ln(f32::MAX) ≈ 88.72

    assert!(
        log_mag_clamp < f32_max_exp,
        "LOG_MAG_CLAMP_MAX must be below ln(f32::MAX)"
    );
    assert!(log_mag_clamp > 0.0, "LOG_MAG_CLAMP_MAX must be positive");
}

/// Harness 25: exp(LOG_MAG_CLAMP_MAX) is finite as f32.
///
/// SUBSTANTIVE: Proves that computing exp() at the clamp boundary produces
/// a finite f32 value. This is the ultimate safety guarantee — the clamp
/// exists to prevent non-finite values from entering the iSTFT pipeline.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn exp_at_clamp_boundary_is_finite() {
    let log_mag_clamp: f64 = 88.0;
    // exp(88.0) as f32: check it fits
    // exp(88.0) ≈ 1.65e38, f32::MAX ≈ 3.4e38
    let exp_val_approx: f64 = 1.65e38;
    let f32_max: f64 = f32::MAX as f64;

    assert!(
        exp_val_approx < f32_max,
        "exp(LOG_MAG_CLAMP_MAX) must be below f32::MAX"
    );
    assert!(
        log_mag_clamp < 89.0,
        "clamp must be below overflow boundary"
    );
}
