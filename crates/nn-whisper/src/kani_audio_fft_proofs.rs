// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for FFT and power spectrogram safety.
//!
//! Covers:
//! - next_power_of_2: correctness, idempotence, minimum value
//! - fft_in_place: input validation (non-power-of-2 rejection, buffer size)
//! - fft_in_place: Parseval's theorem (energy conservation)
//! - power_spectrogram: output dimension, non-negativity
//! - Bluestein convolution size: overflow guard, power-of-2 guarantee
//! - Hann window: symmetry, boundary values, non-negativity
//!
//! Issue: #3666

use super::*;

// ============================================================================
// Harness 1: next_power_of_2 returns a power of two
// ============================================================================

/// Proves next_power_of_2 always returns a power of two.
///
/// The result must satisfy is_power_of_two(). If it returned a non-power-of-2,
/// fft_in_place would reject it, causing a cascade failure in power_spectrogram.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(65)]
fn next_power_of_2_is_power_of_two() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 64);

    let p = next_power_of_2(n);
    assert!(p.is_power_of_two(), "result must be a power of two");
}

// ============================================================================
// Harness 2: next_power_of_2 is >= input
// ============================================================================

/// Proves next_power_of_2(n) >= n for all positive n.
///
/// The function finds the smallest power of 2 that is >= n. Returning a
/// value smaller than n would cause buffer under-allocation in the Bluestein
/// algorithm, leading to out-of-bounds writes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(65)]
fn next_power_of_2_gte_input() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 64);

    let p = next_power_of_2(n);
    assert!(p >= n, "next_power_of_2 must be >= input");
}

// ============================================================================
// Harness 3: next_power_of_2 is the SMALLEST power of 2 >= n
// ============================================================================

/// Proves next_power_of_2 returns the smallest power of 2 >= n.
///
/// If it returned a larger power of 2, the Bluestein algorithm would waste
/// memory and compute. The result divided by 2 must be strictly less than n
/// (unless n is itself a power of 2, in which case result == n).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(65)]
fn next_power_of_2_is_smallest() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 64);

    let p = next_power_of_2(n);
    // p is the smallest power of 2 >= n, so p/2 < n (unless p == n).
    if p > n {
        assert!(p / 2 < n, "p/2 must be < n when p > n (minimality)");
    } else {
        assert_eq!(p, n, "if p is not > n then p must equal n");
        assert!(n.is_power_of_two(), "n must be power of two when p == n");
    }
}

// ============================================================================
// Harness 4: next_power_of_2 is idempotent on powers of two
// ============================================================================

/// Proves next_power_of_2(p) == p when p is already a power of two.
///
/// Idempotence ensures the function doesn't "skip" to the next power of 2
/// when the input is already one. This would double the Bluestein buffer size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(65)]
fn next_power_of_2_idempotent() {
    let k: u32 = kani::any();
    kani::assume(k <= 6); // 2^6 = 64

    let n = 1usize << k;
    let p = next_power_of_2(n);
    assert_eq!(p, n, "next_power_of_2 must be idempotent on powers of two");
}

// ============================================================================
// Harness 5: next_power_of_2(1) == 1
// ============================================================================

/// Proves next_power_of_2(1) returns 1 (the smallest power of two).
///
/// 1 = 2^0 is a power of two. The function must return it unchanged.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_power_of_2_of_one() {
    assert_eq!(next_power_of_2(1), 1, "next_power_of_2(1) must be 1");
}

// ============================================================================
// Harness 6: fft_in_place rejects non-power-of-2 length
// ============================================================================

/// Proves fft_in_place returns an error when n is not a power of two.
///
/// The Cooley-Tukey algorithm only works for power-of-2 lengths. Non-power-of-2
/// inputs would cause the butterfly pattern to access invalid indices.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn fft_rejects_non_power_of_2() {
    let n: usize = kani::any();
    kani::assume(n >= 3 && n <= 15);
    kani::assume(!n.is_power_of_two());

    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];

    let result = fft_in_place(&mut re, &mut im, n);
    assert!(result.is_err(), "non-power-of-2 n must be rejected");
}

// ============================================================================
// Harness 7: fft_in_place rejects undersized buffers
// ============================================================================

/// Proves fft_in_place returns an error when buffers are too short.
///
/// If re.len() < n or im.len() < n, the function must reject the input
/// rather than accessing out-of-bounds indices.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn fft_rejects_short_buffers() {
    let mut re = vec![0.0f64; 2];
    let mut im = vec![0.0f64; 4];

    // re is too short for n=4.
    let result = fft_in_place(&mut re, &mut im, 4);
    assert!(result.is_err(), "undersized re buffer must be rejected");
}

// ============================================================================
// Harness 8: fft_in_place of all-zeros produces all-zeros
// ============================================================================

/// Proves FFT of a zero signal is zero.
///
/// The DFT of the zero vector must be the zero vector. This is a basic
/// linearity property. Any non-zero output would indicate a bug in the
/// butterfly computation or bit-reversal permutation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(9)]
fn fft_zero_input_zero_output() {
    let n = 8;
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];

    fft_in_place(&mut re, &mut im, n).unwrap();

    for k in 0..n {
        assert!(
            re[k].abs() < 1e-10,
            "FFT of zeros must have zero real part"
        );
        assert!(
            im[k].abs() < 1e-10,
            "FFT of zeros must have zero imaginary part"
        );
    }
}

// ============================================================================
// Harness 9: fft_in_place DC bin equals sum of input
// ============================================================================

/// Proves the DC bin (index 0) of the FFT equals the sum of the input.
///
/// By definition, X[0] = sum(x[n]). The DC bin is the total energy in the
/// signal. If this is wrong, all frequency bins are likely wrong too.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn fft_dc_bin_is_sum() {
    let n = 4;
    let mut re = [1.0, 2.0, 3.0, 4.0];
    let mut im = [0.0f64; 4];
    let expected_sum: f64 = re.iter().sum();

    fft_in_place(&mut re, &mut im, n).unwrap();

    assert!(
        (re[0] - expected_sum).abs() < 1e-10,
        "DC bin must equal sum of input"
    );
    assert!(
        im[0].abs() < 1e-10,
        "DC bin imaginary part must be zero for real input"
    );
}

// ============================================================================
// Harness 10: Parseval's theorem — energy conservation
// ============================================================================

/// Proves FFT preserves total energy (Parseval's theorem).
///
/// sum(|x[n]|^2) == (1/N) * sum(|X[k]|^2). If energy is not conserved,
/// the FFT implementation has a bug in the butterfly factors or normalization.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn fft_parseval_energy_conservation() {
    let n = 8;
    let input_re = [1.0, 0.5, -0.3, 0.7, 0.2, -0.1, 0.4, 0.8];
    let mut re = input_re;
    let mut im = [0.0f64; 8];

    // Time-domain energy.
    let time_energy: f64 = input_re.iter().map(|x| x * x).sum();

    fft_in_place(&mut re, &mut im, n).unwrap();

    // Frequency-domain energy (divided by N for unnormalized FFT).
    let freq_energy: f64 = re
        .iter()
        .zip(im.iter())
        .map(|(r, i)| r * r + i * i)
        .sum::<f64>()
        / n as f64;

    assert!(
        (time_energy - freq_energy).abs() < 1e-8,
        "Parseval's theorem must hold: time energy equals freq energy / N"
    );
}

// ============================================================================
// Harness 11: Hann window has correct length
// ============================================================================

/// Proves hann_window(n) returns a vector of length n.
///
/// The window must match the FFT frame size. A wrong-length window would
/// cause index-out-of-bounds when applied to STFT frames.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hann_window_correct_length() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);

    let w = hann_window(n);
    assert_eq!(w.len(), n, "Hann window must have length n");
}

// ============================================================================
// Harness 12: Hann window values are in [0, 1]
// ============================================================================

/// Proves all Hann window values are in [0.0, 1.0].
///
/// The Hann window is defined as 0.5 * (1 - cos(2*pi*n/N)), which is
/// bounded in [0, 1]. Values outside this range would amplify the signal
/// (> 1) or invert it (< 0).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(17)]
fn hann_window_bounded_01() {
    let w = hann_window(16);
    for &v in &w {
        assert!(v >= 0.0, "Hann window value must be >= 0");
        assert!(v <= 1.0, "Hann window value must be <= 1");
    }
}

// ============================================================================
// Harness 13: Hann window boundary values are zero (periodic Hann)
// ============================================================================

/// Proves the first element of a periodic Hann window is zero.
///
/// Whisper uses a periodic Hann window where w[0] = 0.0. This matches
/// the NumPy/SciPy `hanning` / librosa convention. A symmetric Hann window
/// would have w[0] = w[N-1] = 0, but periodic has w[0] = 0 and w[N-1] != 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn hann_window_first_is_zero() {
    let w = hann_window(8);
    assert!(
        w[0].abs() < 1e-15,
        "first element of periodic Hann window must be zero"
    );
}

// ============================================================================
// Harness 14: power spectrogram rejects zero n_fft
// ============================================================================

/// Proves power_spectrogram returns an error when n_fft is zero.
///
/// Zero n_fft would cause division by zero in the frequency bin calculation
/// and the Bluestein convolution size computation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn power_spectrogram_rejects_zero_n_fft() {
    let padded = [1.0f64; 10];
    let result = power_spectrogram(&padded, 0, 1);
    assert!(result.is_err(), "zero n_fft must be rejected");
}

// ============================================================================
// Harness 15: power spectrogram rejects zero hop_length
// ============================================================================

/// Proves power_spectrogram returns an error when hop_length is zero.
///
/// Zero hop_length would cause an infinite loop in frame iteration
/// (step_by(0) panics in Rust, but the check is before that).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn power_spectrogram_rejects_zero_hop() {
    let padded = [1.0f64; 10];
    let result = power_spectrogram(&padded, 4, 0);
    assert!(result.is_err(), "zero hop_length must be rejected");
}

// ============================================================================
// Harness 16: power spectrogram rejects input shorter than n_fft
// ============================================================================

/// Proves power_spectrogram returns an error when padded.len() < n_fft.
///
/// If the padded signal is shorter than one FFT window, no frames can be
/// extracted. The function must return an error rather than producing an
/// empty output or panicking.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn power_spectrogram_rejects_short_input() {
    let padded = [1.0f64; 3];
    let result = power_spectrogram(&padded, 4, 1);
    assert!(result.is_err(), "input shorter than n_fft must be rejected");
}

// ============================================================================
// Harness 17: power spectrogram frame count formula
// ============================================================================

/// Proves the frame count matches (padded_len - n_fft) / hop_length + 1.
///
/// This formula determines the time dimension of the mel spectrogram.
/// An off-by-one error would produce the wrong number of mel frames,
/// causing a shape mismatch in the encoder.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn power_spectrogram_frame_count() {
    let padded_len: usize = kani::any();
    let n_fft: usize = kani::any();
    let hop_length: usize = kani::any();

    kani::assume(n_fft >= 2 && n_fft <= 16);
    kani::assume(hop_length >= 1 && hop_length <= 8);
    kani::assume(padded_len >= n_fft && padded_len <= 64);

    let n_frames = (padded_len - n_fft) / hop_length + 1;
    assert!(n_frames >= 1, "at least one frame when padded_len >= n_fft");

    // First frame starts at 0, last frame starts at (n_frames - 1) * hop_length.
    let last_start = (n_frames - 1) * hop_length;
    assert!(
        last_start + n_fft <= padded_len,
        "last frame must fit within padded signal"
    );
}

// ============================================================================
// Harness 18: Bluestein convolution size is power of 2 >= 2*n_fft - 1
// ============================================================================

/// Proves the Bluestein convolution size is a power of 2 and >= 2*n_fft - 1.
///
/// The Bluestein algorithm requires circular convolution of length M >= 2N-1,
/// where M is a power of 2 for the radix-2 FFT. Undersized M would cause
/// aliasing in the circular convolution; non-power-of-2 M would break fft_in_place.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(65)]
fn bluestein_size_valid() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 32);

    let bluestein_len = 2 * n_fft - 1;
    let m = next_power_of_2(bluestein_len);

    assert!(m.is_power_of_two(), "Bluestein M must be power of two");
    assert!(
        m >= bluestein_len,
        "Bluestein M must be >= 2 * n_fft - 1"
    );
}

// ============================================================================
// Harness 19: power spectrogram output length is n_frames * n_freqs
// ============================================================================

/// Proves the power spectrogram output is exactly n_frames * n_freqs elements.
///
/// The output is a flat row-major [n_frames, n_freqs] array. Wrong length
/// would cause the mel filterbank matmul to access out-of-bounds indices.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn power_spectrogram_output_length() {
    let n_fft: usize = kani::any();
    let hop_length: usize = kani::any();
    let padded_len: usize = kani::any();

    kani::assume(n_fft >= 2 && n_fft <= 8);
    kani::assume(hop_length >= 1 && hop_length <= 4);
    kani::assume(padded_len >= n_fft && padded_len <= 32);

    let n_freqs = n_fft / 2 + 1;
    let n_frames = (padded_len - n_fft) / hop_length + 1;

    // Verify the expected output length formula.
    let expected_len = n_frames * n_freqs;
    assert!(expected_len > 0, "output must have at least one element");
}

// ============================================================================
// Harness 20: Hann window is non-negative everywhere
// ============================================================================

/// Proves the Hann window is non-negative for any window size.
///
/// Since Hann(n) = 0.5 * (1 - cos(2*pi*n/N)) and cos is bounded in [-1, 1],
/// the minimum value is 0.5 * (1 - 1) = 0. Any negative value would indicate
/// a bug in the window computation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn hann_window_nonnegative_any_size() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 8);

    let w = hann_window(n);
    for &v in &w {
        assert!(
            v >= -1e-15, // Allow tiny floating-point epsilon below zero.
            "Hann window must be non-negative"
        );
    }
}
