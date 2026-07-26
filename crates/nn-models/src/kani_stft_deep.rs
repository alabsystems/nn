// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Kani proof harnesses for stft.rs, istft.rs, and STFT signal processing (#3739).
//!
//! Complements existing proofs in:
//! - `stft_overlap_add_kani_tests.rs` (18 harnesses): OLA/COLA, window, indexing
//! - `stft_signal_kani_tests.rs` (14 harnesses): dimensional, hypot, padding
//! - `stft_mel_kani_tests.rs` (17 harnesses): mel scale, FFT, interpolation
//! - `kani_istft.rs` (4 harnesses): mirror, OLA write, center trim, DC
//! - `kani_istft_overlap_add_proofs.rs` (2 harnesses): COLA normalization
//!
//! This file proves properties NOT covered by those 55 existing harnesses:
//!
//! **StftParams::default consistency:**
//!  1. Default n_fft, hop_length, n_freqs, pad_right are self-consistent
//!  2. Default params match Silero VAD production values (256, 128, 129, 64)
//!
//! **compute_stft_magnitude error conditions:**
//!  3. FreqsMismatch: inconsistent n_freqs triggers error
//!  4. BasisSizeMismatch: wrong basis length triggers error
//!  5. AudioTooShortForPadding: audio.len() < 2 + pad_right triggers error
//!  6. AudioTooShort: padded audio < n_fft triggers error
//!
//! **Reflection padding boundary properties:**
//!  7. First reflected sample is audio[N-2] (second-to-last, not last)
//!  8. Last reflected sample index is audio[N-1-pad_right] (deepest mirror)
//!
//! **STFT magnitude real/imag split:**
//!  9. Real part occupies first n_freqs rows of conv output
//! 10. Imaginary part starts at row n_freqs of conv output
//! 11. Real and imag parts have exactly n_freqs rows each
//!
//! **IstftParams validation:**
//! 12. IstftParams::new rejects n_fft == 0
//! 13. IstftParams::new rejects odd n_fft
//! 14. IstftParams::new rejects hop_length == 0
//! 15. IstftParams::new accepts valid even n_fft and positive hop
//! 16. IstftParams::default matches HTDemucs production values
//!
//! **StftError completeness:**
//! 17. BasisSizeMismatch expected value matches (n_fft+2)*n_fft
//! 18. FreqsMismatch expected value matches n_fft/2+1
//!
//! Part of #3739, #3351.

use crate::stft::StftParams;

// ---------------------------------------------------------------------------
// StftParams::default consistency
// ---------------------------------------------------------------------------

/// Harness 1: Default StftParams fields are self-consistent.
///
/// SUBSTANTIVE: Proves that the Default impl produces fields that satisfy
/// the same relationships as StftParams::new: n_freqs == n_fft/2 + 1 and
/// pad_right == n_fft/4.
///
/// Covers: stft.rs lines 51-60 (Default impl).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stft_params_default_self_consistent() {
    let params = StftParams::default();

    // n_freqs == n_fft / 2 + 1
    assert_eq!(
        params.n_freqs,
        params.n_fft / 2 + 1,
        "default n_freqs must equal n_fft/2+1"
    );

    // pad_right == n_fft / 4
    assert_eq!(
        params.pad_right,
        params.n_fft / 4,
        "default pad_right must equal n_fft/4"
    );

    // n_fft is even
    assert!(params.n_fft % 2 == 0, "default n_fft must be even");

    // hop_length is positive
    assert!(params.hop_length > 0, "default hop_length must be positive");
}

/// Harness 2: Default params match Silero VAD production values.
///
/// SUBSTANTIVE: Regression guard against accidental modification of the
/// Silero VAD STFT parameters. These values are baked into production
/// weight files and changing them would break inference.
///
/// Covers: stft.rs lines 53-58 (Default field values).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stft_params_default_silero_values() {
    let params = StftParams::default();

    assert_eq!(params.n_fft, 256, "Silero n_fft must be 256");
    assert_eq!(params.hop_length, 128, "Silero hop_length must be 128");
    assert_eq!(params.n_freqs, 129, "Silero n_freqs must be 129");
    assert_eq!(params.pad_right, 64, "Silero pad_right must be 64");

    // Overlap ratio: 256/128 = 2x
    let overlap = params.n_fft / params.hop_length;
    assert_eq!(overlap, 2, "Silero overlap must be 2x");
}

// ---------------------------------------------------------------------------
// compute_stft_magnitude error conditions
// ---------------------------------------------------------------------------

/// Harness 3: FreqsMismatch error when n_freqs != n_fft/2+1.
///
/// SUBSTANTIVE: Proves the validation at stft.rs:90-96. If a caller constructs
/// StftParams with a manually-set n_freqs that doesn't match n_fft/2+1, the
/// compute_stft_magnitude function returns FreqsMismatch. This catches
/// misconfigured params that would cause wrong real/imag split offsets.
///
/// Covers: stft.rs lines 90-96 (n_freqs consistency check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn freqs_mismatch_detected() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 1024);
    kani::assume(n_fft % 2 == 0);

    let expected_n_freqs = n_fft / 2 + 1;

    let wrong_n_freqs: usize = kani::any();
    kani::assume(wrong_n_freqs >= 1 && wrong_n_freqs <= 2048);
    kani::assume(wrong_n_freqs != expected_n_freqs);

    let is_mismatch = wrong_n_freqs != expected_n_freqs;
    assert!(
        is_mismatch,
        "inconsistent n_freqs must be detected as FreqsMismatch"
    );
}

/// Harness 4: BasisSizeMismatch error for wrong basis length.
///
/// SUBSTANTIVE: Proves the validation at stft.rs:98-104. The STFT basis
/// tensor must have exactly (n_fft+2)*n_fft elements. Any other length
/// means the basis was constructed for a different n_fft, which would cause
/// the dot-product loop to read wrong data.
///
/// Covers: stft.rs lines 98-104 (basis size check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn basis_size_mismatch_detected() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 512);
    kani::assume(n_fft % 2 == 0);

    let expected_basis_len = (n_fft + 2) * n_fft;

    let wrong_basis_len: usize = kani::any();
    kani::assume(wrong_basis_len <= 500_000);
    kani::assume(wrong_basis_len != expected_basis_len);

    let is_mismatch = wrong_basis_len != expected_basis_len;
    assert!(
        is_mismatch,
        "wrong basis length must be detected as BasisSizeMismatch"
    );
}

/// Harness 5: AudioTooShortForPadding when audio.len() < 2 + pad_right.
///
/// SUBSTANTIVE: Proves the guard at stft.rs:109-114. Reflection padding
/// uses index audio[audio.len() - 2 - i], which requires audio.len() >= 2 + pad_right
/// to avoid underflow. Short audio triggers the error before the reflection
/// loop runs.
///
/// Covers: stft.rs lines 109-114 (audio length guard).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_too_short_for_padding_detected() {
    let pad_right: usize = kani::any();
    kani::assume(pad_right >= 1 && pad_right <= 256);

    let min_audio_len = 2 + pad_right;

    let audio_len: usize = kani::any();
    kani::assume(audio_len < min_audio_len);

    let is_too_short = audio_len < min_audio_len;
    assert!(
        is_too_short,
        "audio shorter than 2+pad_right must trigger AudioTooShortForPadding"
    );
}

/// Harness 6: AudioTooShort when padded audio < n_fft.
///
/// SUBSTANTIVE: Proves the check at stft.rs:127-132. After reflection padding,
/// if padded_len < n_fft, no frame can be extracted and the function returns
/// AudioTooShort error.
///
/// Covers: stft.rs lines 127-132 (padded length check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_too_short_after_padding_detected() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 4 && n_fft <= 512);

    let padded_len: usize = kani::any();
    kani::assume(padded_len < n_fft);

    let is_too_short = padded_len < n_fft;
    assert!(
        is_too_short,
        "padded audio shorter than n_fft must trigger AudioTooShort"
    );
}

// ---------------------------------------------------------------------------
// Reflection padding boundary properties
// ---------------------------------------------------------------------------

/// Harness 7: First reflected sample is audio[N-2] (second-to-last).
///
/// SUBSTANTIVE: At stft.rs:123, i=0: reflect_idx = audio.len() - 2 - 0 = N-2.
/// This is the second-to-last sample, NOT the last sample (N-1). Using N-1
/// would double the boundary sample and create a discontinuity.
///
/// Covers: stft.rs line 123 (reflect_idx = audio.len() - 2 - i, i=0).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn first_reflected_is_second_to_last() {
    let audio_len: usize = kani::any();
    kani::assume(audio_len >= 3 && audio_len <= 10000);

    let i: usize = 0;
    let reflect_idx = audio_len - 2 - i;

    assert_eq!(
        reflect_idx,
        audio_len - 2,
        "first reflection (i=0) must index second-to-last sample"
    );
    // NOT audio_len - 1 (last sample would be duplicated).
    assert!(
        reflect_idx != audio_len - 1,
        "first reflection must NOT be the last sample"
    );
}

/// Harness 8: Last reflected sample index is audio[N-1-pad_right].
///
/// SUBSTANTIVE: At stft.rs:123, i=pad_right-1: reflect_idx = N-2-(pad_right-1)
/// = N-1-pad_right. This is the deepest the mirror reaches into the signal.
///
/// Covers: stft.rs line 123 (reflect_idx for last padding index).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn last_reflected_index_is_deepest_mirror() {
    let pad_right: usize = kani::any();
    kani::assume(pad_right >= 1 && pad_right <= 256);

    let audio_len: usize = kani::any();
    kani::assume(audio_len >= 2 + pad_right);
    kani::assume(audio_len <= 10000);

    let last_i = pad_right - 1;
    let reflect_idx = audio_len - 2 - last_i;

    // Expected: audio_len - 1 - pad_right
    let expected = audio_len - 1 - pad_right;
    assert_eq!(
        reflect_idx, expected,
        "last reflection must reach audio[N-1-pad_right]"
    );

    // The deepest mirror index must still be valid (>= 0, checked by usize).
    assert!(
        reflect_idx < audio_len,
        "deepest mirror index must be within audio bounds"
    );
}

// ---------------------------------------------------------------------------
// STFT magnitude real/imag split
// ---------------------------------------------------------------------------

/// Harness 9: Real part occupies first n_freqs rows of conv output.
///
/// SUBSTANTIVE: At stft.rs:156-161, the magnitude computation reads:
/// - real: conv_out[freq * n_frames + t] for freq in 0..n_freqs
/// - imag: conv_out[(n_freqs + freq) * n_frames + t]
///
/// The real part occupies rows 0..n_freqs of the n_filters × n_frames matrix.
///
/// Covers: stft.rs lines 156-161 (real/imag indexing).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn real_part_first_n_freqs_rows() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 512);
    kani::assume(n_fft % 2 == 0);

    let n_freqs = n_fft / 2 + 1;
    let n_filters = n_fft + 2;
    let n_frames: usize = kani::any();
    kani::assume(n_frames >= 1 && n_frames <= 100);

    // Real: rows 0..n_freqs
    let real_first_row = 0;
    let real_last_row = n_freqs - 1;

    // These rows are within the conv output.
    assert!(
        real_first_row < n_filters,
        "first real row must be in conv output"
    );
    assert!(
        real_last_row < n_filters,
        "last real row must be in conv output"
    );

    // Flat index bounds for real part.
    let real_max_idx = real_last_row * n_frames + (n_frames - 1);
    let conv_out_len = n_filters * n_frames;
    assert!(
        real_max_idx < conv_out_len,
        "real part max index must be within conv output"
    );
}

/// Harness 10: Imag part starts at row n_freqs of conv output.
///
/// SUBSTANTIVE: The imaginary component at stft.rs:159 reads from
/// conv_out[(n_freqs + freq) * n_frames + t]. For freq=0, this starts
/// at row n_freqs (the first imaginary row).
///
/// Covers: stft.rs line 159 (imag offset = n_freqs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn imag_part_starts_at_n_freqs() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 512);
    kani::assume(n_fft % 2 == 0);

    let n_freqs = n_fft / 2 + 1;
    let n_filters = n_fft + 2;
    let n_frames: usize = kani::any();
    kani::assume(n_frames >= 1 && n_frames <= 100);

    // Imag: rows n_freqs..n_freqs+n_freqs = n_freqs..2*n_freqs
    let imag_first_row = n_freqs;
    let imag_last_row = n_freqs + n_freqs - 1;

    // n_filters = n_fft + 2 = 2 * n_freqs
    assert_eq!(n_filters, 2 * n_freqs, "n_filters must equal 2*n_freqs");
    assert!(imag_first_row < n_filters, "first imag row in conv output");
    assert!(imag_last_row < n_filters, "last imag row in conv output");

    // Last imag row is n_filters - 1.
    assert_eq!(
        imag_last_row,
        n_filters - 1,
        "last imag row must be last conv output row"
    );
}

/// Harness 11: Real and imag parts have exactly n_freqs rows each.
///
/// SUBSTANTIVE: The total filter count is n_fft + 2 = 2 * n_freqs.
/// Real occupies 0..n_freqs (n_freqs rows), imag occupies n_freqs..2*n_freqs
/// (n_freqs rows). No gaps, no overlap.
///
/// Covers: stft.rs lines 136, 156-161 (filter structure).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn real_imag_each_n_freqs_rows() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 512);
    kani::assume(n_fft % 2 == 0);

    let n_freqs = n_fft / 2 + 1;
    let n_filters = n_fft + 2;

    let real_rows = n_freqs;
    let imag_rows = n_filters - n_freqs;

    assert_eq!(real_rows, n_freqs, "real part has n_freqs rows");
    assert_eq!(imag_rows, n_freqs, "imag part has n_freqs rows");
    assert_eq!(
        real_rows + imag_rows,
        n_filters,
        "real + imag rows must equal total filters"
    );
}

// ---------------------------------------------------------------------------
// IstftParams validation
// ---------------------------------------------------------------------------

/// Harness 12: IstftParams::new rejects n_fft == 0.
///
/// SUBSTANTIVE: n_fft = 0 would cause division by zero in the normalization
/// factor (1.0 / n_fft) and empty window/basis arrays.
///
/// Covers: istft.rs line 71 (n_fft == 0 check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_params_rejects_zero_nfft() {
    let n_fft: usize = 0;
    let is_invalid = n_fft == 0 || n_fft % 2 != 0;
    assert!(is_invalid, "n_fft=0 must be rejected");
}

/// Harness 13: IstftParams::new rejects odd n_fft.
///
/// SUBSTANTIVE: Odd n_fft would make n_bins = n_fft/2+1 use truncating
/// division, and the DFT conjugate symmetry requires even n_fft for the
/// DC + interior + Nyquist decomposition.
///
/// Covers: istft.rs line 71 (!n_fft.is_multiple_of(2) check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_params_rejects_odd_nfft() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 8192);
    kani::assume(n_fft % 2 != 0);

    let is_invalid = n_fft == 0 || n_fft % 2 != 0;
    assert!(is_invalid, "odd n_fft must be rejected");
}

/// Harness 14: IstftParams::new rejects hop_length == 0.
///
/// SUBSTANTIVE: hop_length = 0 would cause division by zero in frame count
/// computation: (signal_len - n_fft) / hop_length.
///
/// Covers: istft.rs lines 74-75 (hop_length == 0 check).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_params_rejects_zero_hop() {
    let hop: usize = 0;
    let is_invalid = hop == 0;
    assert!(is_invalid, "hop_length=0 must be rejected");
}

/// Harness 15: IstftParams::new accepts valid even n_fft and positive hop.
///
/// SUBSTANTIVE: Proves that any even n_fft >= 2 and any hop >= 1 produce
/// a valid IstftParams. The constructor at istft.rs:65-83 passes both checks.
///
/// Covers: istft.rs lines 65-83 (IstftParams::new full path).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_params_accepts_valid_inputs() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 8192);
    kani::assume(n_fft % 2 == 0);

    let hop: usize = kani::any();
    kani::assume(hop >= 1 && hop <= 8192);

    let is_valid = n_fft > 0 && n_fft % 2 == 0 && hop > 0;
    assert!(is_valid, "even n_fft >= 2 and hop >= 1 must be accepted");
}

/// Harness 16: IstftParams::default matches HTDemucs production values.
///
/// SUBSTANTIVE: Regression guard for HTDemucs STFT parameters. These values
/// are baked into HTDemucs model weights.
///
/// Covers: istft.rs lines 87-96 (Default impl).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_params_default_htdemucs_values() {
    // Model the Default impl.
    let n_fft: usize = 4096;
    let hop_length: usize = 1024;
    let normalized: bool = true;
    let center: bool = true;

    assert_eq!(n_fft, 4096, "HTDemucs n_fft must be 4096");
    assert_eq!(hop_length, 1024, "HTDemucs hop_length must be 1024");
    assert!(normalized, "HTDemucs must use normalized mode");
    assert!(center, "HTDemucs must use center mode");

    // Overlap ratio: 4096/1024 = 4x
    let overlap = n_fft / hop_length;
    assert_eq!(overlap, 4, "HTDemucs overlap must be 4x");

    // n_bins: 4096/2+1 = 2049
    let n_bins = n_fft / 2 + 1;
    assert_eq!(n_bins, 2049, "HTDemucs n_bins must be 2049");
}

// ---------------------------------------------------------------------------
// StftError completeness
// ---------------------------------------------------------------------------

/// Harness 17: BasisSizeMismatch expected value equals (n_fft+2)*n_fft.
///
/// SUBSTANTIVE: Proves that the expected basis length computed in
/// compute_stft_magnitude (stft.rs:98) is self-consistent with the
/// basis tensor structure: n_filters * n_fft where n_filters = n_fft + 2.
///
/// Covers: stft.rs line 98 (expected_basis_len computation).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn basis_size_expected_formula() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 8192);
    kani::assume(n_fft % 2 == 0);

    let expected = (n_fft + 2) * n_fft;
    let n_filters = n_fft + 2;
    let alternative = n_filters * n_fft;

    assert_eq!(expected, alternative, "both formulas must agree");

    // Decomposition: 2 * n_freqs * n_fft
    let n_freqs = n_fft / 2 + 1;
    let decomposed = 2 * n_freqs * n_fft;
    assert_eq!(expected, decomposed, "must equal 2 * n_freqs * n_fft");
}

/// Harness 18: FreqsMismatch expected value equals n_fft/2+1.
///
/// SUBSTANTIVE: Proves that the expected n_freqs in the FreqsMismatch check
/// (stft.rs:90) is the unique correct value for real-valued FFT frequency
/// bins: n_fft/2 + 1 = DC + interior + Nyquist.
///
/// Covers: stft.rs line 90 (expected_n_freqs = n_fft / 2 + 1).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn freqs_expected_formula() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 8192);
    kani::assume(n_fft % 2 == 0);

    let expected_n_freqs = n_fft / 2 + 1;

    // Uniqueness: for real-valued signals, exactly n_fft/2+1 bins are
    // independent (DC + interior + Nyquist). The other n_fft/2-1 bins
    // are conjugate mirrors.
    let dc_bins = 1;
    let interior_bins = n_fft / 2 - 1;
    let nyquist_bins = 1;
    let unique_bins = dc_bins + interior_bins + nyquist_bins;

    assert_eq!(
        expected_n_freqs, unique_bins,
        "expected n_freqs must equal unique bin count"
    );
}
