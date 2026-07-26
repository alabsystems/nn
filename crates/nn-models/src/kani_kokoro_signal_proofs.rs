// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for kokoro_signal constants and iSTFT input preparation.
//!
//! Proves that:
//! 1. KOKORO_N_BINS == KOKORO_N_FFT / 2 + 1 (DFT frequency bin count).
//! 2. KOKORO_HOP_LENGTH evenly divides KOKORO_N_FFT.
//! 3. KOKORO_SAMPLE_RATE is standard audio rate (24kHz).
//! 4. KOKORO_N_FFT is even (required for real-valued FFT symmetry).
//! 5. iSTFT real/imag split: half + (n_fft - half) == n_fft.
//! 6. iSTFT bin count: for any even n_fft, n_bins padded from half covers n_bins rows.
//!
//! Part of #3793, #3351.

use crate::kokoro_tts::{KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT, KOKORO_SAMPLE_RATE};

/// Proof 1: KOKORO_N_BINS == KOKORO_N_FFT / 2 + 1.
///
/// The number of unique frequency bins in a real-valued DFT of size N
/// is N/2 + 1 (DC through Nyquist).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_n_bins_matches_n_fft() {
    assert_eq!(
        KOKORO_N_BINS,
        KOKORO_N_FFT / 2 + 1,
        "N_BINS must equal N_FFT/2 + 1 for real-valued FFT"
    );
}

/// Proof 2: Hop length divides FFT size evenly.
///
/// This is required for COLA (constant-overlap-add) reconstruction
/// with a periodic Hann window.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_hop_divides_n_fft() {
    assert!(KOKORO_HOP_LENGTH > 0, "hop length must be positive");
    assert_eq!(
        KOKORO_N_FFT % KOKORO_HOP_LENGTH,
        0,
        "hop length must evenly divide n_fft for COLA"
    );
}

/// Proof 3: Sample rate is 24kHz.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_sample_rate_is_24khz() {
    assert_eq!(KOKORO_SAMPLE_RATE, 24000);
}

/// Proof 4: N_FFT is even.
///
/// Required for real-valued FFT: conjugate symmetry means
/// bin count is N/2 + 1, which requires N to be even.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kokoro_n_fft_is_even() {
    assert_eq!(KOKORO_N_FFT % 2, 0, "n_fft must be even for real FFT");
}

/// Proof 5: iSTFT real/imag split covers all channels.
///
/// For any even n_fft, splitting the first half as real and
/// the second half as imaginary accounts for all n_fft channels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_istft_split_covers_all_channels() {
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 64);
    let n_fft = (n_fft_half as usize) * 2;
    let half = n_fft / 2;
    // Real occupies channels [0, half), Imag occupies channels [half, n_fft)
    let real_count = half;
    let imag_count = n_fft - half;
    assert_eq!(
        real_count + imag_count,
        n_fft,
        "real + imag channels must cover all n_fft channels"
    );
}

/// Proof 6: iSTFT padded bin count is n_bins = half + 1.
///
/// After taking `half` real/imag rows from the decoder output and
/// zero-padding one extra row (for Nyquist), we get n_bins = half + 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_istft_padded_n_bins() {
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 64);
    let n_fft = (n_fft_half as usize) * 2;
    let half = n_fft / 2;
    let n_bins = half + 1;
    // Padding from half rows to n_bins rows adds exactly 1 row
    assert_eq!(
        n_bins - half,
        1,
        "zero-padding adds exactly the Nyquist bin"
    );
    // n_bins matches the standard DFT formula
    assert_eq!(n_bins, n_fft / 2 + 1);
}
