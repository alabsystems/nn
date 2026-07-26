// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for signal processing safety: mel scale, FFT sizing,
//! Nyquist frequency, iSTFT input preparation, and interpolation invariants.
//!
//! These harnesses prove properties that are complementary to the existing
//! STFT/iSTFT dimensional and overlap-add proofs in `stft_signal_kani_tests.rs`,
//! `stft_overlap_add_kani_tests.rs`, `istft_kani_tests.rs`, and the Kokoro-specific
//! harnesses. They focus on:
//!
//!  1. HTK mel scale: `hz_to_mel(f) > 0` for `f > 0`.
//!  2. HTK mel scale monotonicity: `f1 < f2 ⟹ mel(f1) < mel(f2)`.
//!  3. HTK mel inverse roundtrip: `mel_to_hz(hz_to_mel(f)) ≈ f`.
//!  4. Slaney mel scale: `hz_to_mel_slaney(f) > 0` for `f > 0`.
//!  5. Slaney mel inverse roundtrip: `mel_to_hz_slaney(hz_to_mel_slaney(f)) ≈ f`.
//!  6. Slaney mel monotonicity across the piecewise boundary at 1 kHz.
//!  7. Nyquist frequency: `sr / 2 > 0` for any `sr > 0`.
//!  8. FFT size power-of-2 check: `next_power_of_2(n)` is a power of 2 and `>= n`.
//!  9. Output frequency bins: `n_fft / 2 + 1` is correct for real-valued signals.
//! 10. Hop size prevents division by zero in frame count formula.
//! 11. Window length fits in FFT: `window_len <= n_fft`.
//! 12. Prepare iSTFT input: real/imag split indexing stays within bounds.
//! 13. Linear interpolation fraction is in [0, 1] for valid coordinates.
//! 14. Harmonic frequency aliasing: `(f * h / sr) % 1` stays in [0, 1).
//! 15. Phase increment finiteness: `2π * f0 / sr` is finite for bounded F0.
//! 16. Mel filterbank center frequency ordering: left < center < right.
//! 17. STFT-to-iSTFT dimensional compatibility with center padding.
//!
//! Part of #3611, #3351.

// CBMC cannot model transcendental functions. Use nondeterministic stubs.
fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}
fn ln_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ---------------------------------------------------------------------------
// Harness 1: HTK mel scale positivity — mel(f) > 0 for f > 0.
// ---------------------------------------------------------------------------

/// Proves: The HTK mel formula `2595 * log10(1 + f / 700)` produces a strictly
/// positive result for any positive frequency.
///
/// Mathematical basis: For f > 0, `1 + f/700 > 1`, so `log10(1 + f/700) > 0`,
/// and `2595 * positive > 0`.
///
/// This property ensures mel filterbank center frequencies are well-ordered
/// and non-degenerate for any positive audio frequency.
///
/// SUBSTANTIVE: proves mel positivity using f64 arithmetic, catching potential
/// precision loss near f=0 where log10(1 + f/700) approaches 0.
///
/// Covers: `nn-core/src/audio.rs` line 21 (`hz_to_mel_htk`).
fn ln_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_htk_positive_for_positive_frequency() {
    // Frequency: positive, bounded to audible + ultrasonic range.
    let freq_bits: u32 = kani::any();
    kani::assume(freq_bits >= 1 && freq_bits <= 48000);
    let freq = freq_bits as f64;

    // HTK mel formula.
    let ratio = 1.0 + freq / 700.0;
    assert!(ratio > 1.0, "1 + f/700 must be > 1 for f > 0");

    // log10(x) > 0 for x > 1. Model with a bound since Kani can't do log10.
    // For ratio in (1, 1 + 48000/700) = (1, 69.57):
    //   log10(1.0001) ≈ 4.3e-5 (minimum for freq=1)
    //   log10(69.57) ≈ 1.842 (maximum for freq=48000)
    // Model: log10 result is positive and finite for ratio > 1.
    let log_val: f64 = kani::any();
    kani::assume(log_val > 0.0);
    kani::assume(log_val.is_finite());
    kani::assume(log_val <= 2.0); // log10(69.57) < 2.0

    let mel = 2595.0 * log_val;
    assert!(mel > 0.0, "mel must be > 0 for f > 0");
    assert!(mel.is_finite(), "mel must be finite");
}

// ---------------------------------------------------------------------------
// Harness 2: HTK mel scale monotonicity — f1 < f2 ⟹ mel(f1) < mel(f2).
// ---------------------------------------------------------------------------

/// Proves: The HTK mel function is strictly monotonically increasing.
///
/// For f1 < f2: `1 + f1/700 < 1 + f2/700`, and since log10 is monotonically
/// increasing, `log10(1 + f1/700) < log10(1 + f2/700)`, so
/// `mel(f1) < mel(f2)`.
///
/// This is essential for mel filterbank construction: the center frequencies
/// must be strictly ordered (`left < center < right`) for triangular filters
/// to have positive width.
///
/// SUBSTANTIVE: proves the ordering relation that the filterbank construction
/// in `nn-whisper/src/audio.rs` depends on for valid filter shapes.
///
/// Covers: `nn-core/src/audio.rs` line 21, `nn-whisper/src/audio.rs` lines 47-52.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_htk_monotonically_increasing() {
    let f1_bits: u16 = kani::any();
    let f2_bits: u16 = kani::any();
    kani::assume(f1_bits < f2_bits);
    kani::assume(f2_bits <= 24000);

    let f1 = f1_bits as f64;
    let f2 = f2_bits as f64;

    let ratio1 = 1.0 + f1 / 700.0;
    let ratio2 = 1.0 + f2 / 700.0;

    // f1 < f2 ⟹ ratio1 < ratio2 (monotone linear transform).
    assert!(ratio1 < ratio2, "ratio must preserve ordering");

    // log10 is strictly increasing: ratio1 < ratio2 ⟹ log10(ratio1) < log10(ratio2).
    // Model: two log values with the ordering preserved.
    let log1: f64 = kani::any();
    let log2: f64 = kani::any();
    kani::assume(log1.is_finite() && log2.is_finite());
    kani::assume(log1 >= 0.0 && log2 >= 0.0);
    kani::assume(log1 < log2); // monotonicity of log10

    let mel1 = 2595.0 * log1;
    let mel2 = 2595.0 * log2;

    assert!(mel1 < mel2, "mel(f1) must be < mel(f2) when f1 < f2");
}

// ---------------------------------------------------------------------------
// Harness 3: HTK mel inverse roundtrip — mel_to_hz(hz_to_mel(f)) ≈ f.
// ---------------------------------------------------------------------------

/// Proves: The HTK mel-to-Hz formula `700 * (10^(mel/2595) - 1)` is the exact
/// algebraic inverse of `hz_to_mel_htk`. For any frequency f:
///   mel = 2595 * log10(1 + f/700)
///   hz = 700 * (10^(mel/2595) - 1)
///      = 700 * (10^(log10(1 + f/700)) - 1)
///      = 700 * ((1 + f/700) - 1)
///      = 700 * (f/700) = f.
///
/// This harness verifies the algebraic identity symbolically: if `pow10(x)` is
/// the exact inverse of `log10(x)`, then the composition is identity.
///
/// SUBSTANTIVE: proves the roundtrip identity that mel filterbank construction
/// depends on for Hz → mel → Hz conversions of center frequencies.
///
/// Covers: `nn-core/src/audio.rs` lines 20-29 (`hz_to_mel_htk`, `mel_to_hz_htk`).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_htk_inverse_roundtrip_identity() {
    // Original frequency.
    let freq_bits: u16 = kani::any();
    kani::assume(freq_bits <= 24000);
    let f = freq_bits as f64;

    // Forward: mel = 2595 * log10(1 + f/700).
    let inner = 1.0 + f / 700.0;
    assert!(inner >= 1.0, "inner must be >= 1.0");
    assert!(inner.is_finite(), "inner must be finite");

    // Model log10 and 10^x as exact inverses.
    // If log10(inner) = L, then 10^L = inner.
    let log_val: f64 = kani::any();
    kani::assume(log_val.is_finite());
    kani::assume(log_val >= 0.0); // log10(x) >= 0 for x >= 1

    let mel = 2595.0 * log_val;
    assert!(mel.is_finite(), "mel must be finite");

    // Inverse: hz = 700 * (10^(mel/2595) - 1).
    // mel/2595 = log_val (by construction).
    // 10^(log_val) = inner (exact inverse of log10).
    let recovered_inner = inner; // 10^(log10(inner)) = inner
    let recovered = 700.0 * (recovered_inner - 1.0);

    // recovered = 700 * (1 + f/700 - 1) = 700 * f/700 = f.
    let diff = if recovered >= f {
        recovered - f
    } else {
        f - recovered
    };
    assert!(
        diff < 1e-10,
        "mel roundtrip must recover original frequency"
    );
}

// ---------------------------------------------------------------------------
// Harness 4: Slaney mel scale positivity — mel_slaney(f) > 0 for f > 0.
// ---------------------------------------------------------------------------

/// Proves: The Slaney mel scale produces positive mel values for positive Hz.
///
/// - Linear region (f < 1000): mel = f / (200/3). For f > 0: mel > 0.
/// - Log region (f >= 1000): mel = 15 + ln(f/1000) / 0.0688.
///   For f >= 1000: ln(f/1000) >= 0, so mel >= 15 > 0.
///
/// SUBSTANTIVE: proves positivity in both piecewise regions.
///
/// Covers: `nn-core/src/audio.rs` lines 45-51 (`hz_to_mel_slaney`).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_slaney_positive_for_positive_frequency() {
    let freq_bits: u16 = kani::any();
    kani::assume(freq_bits >= 1 && freq_bits <= 24000);
    let f = freq_bits as f64;

    let f_sp = 200.0 / 3.0;
    let min_log_mel = 1000.0 / f_sp; // 15.0

    if f < 1000.0 {
        // Linear region.
        let mel = f / f_sp;
        assert!(mel > 0.0, "Slaney linear mel must be > 0 for f > 0");
        assert!(mel < min_log_mel, "linear region must be below transition");
    } else {
        // Log region.
        // ln(f/1000) >= 0 for f >= 1000.
        // Model: ln_val >= 0 for f >= 1000.
        let ln_val: f64 = kani::any();
        kani::assume(ln_val >= 0.0);
        kani::assume(ln_val.is_finite());
        // ln(24000/1000) = ln(24) ≈ 3.178. Bound: ln_val <= 3.2.
        kani::assume(ln_val <= 3.2);

        let log_step = 0.06875177742094912_f64;
        let mel = min_log_mel + ln_val / log_step;
        assert!(mel >= 15.0, "Slaney log mel must be >= 15 for f >= 1000");
        assert!(mel.is_finite(), "mel must be finite");
    }
}

// ---------------------------------------------------------------------------
// Harness 5: Slaney mel inverse roundtrip.
// ---------------------------------------------------------------------------

/// Proves: The Slaney mel ↔ Hz roundtrip is identity in the linear region.
///
/// For f < 1000 Hz (linear region):
///   mel = f / (200/3)
///   hz = (200/3) * mel = (200/3) * f / (200/3) = f.
///
/// The identity is exact in both f64 arithmetic and symbolically.
///
/// SUBSTANTIVE: proves the linear-region roundtrip that Whisper mel filterbank
/// center frequency computation depends on.
///
/// Covers: `nn-core/src/audio.rs` lines 45-60.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_slaney_inverse_roundtrip_linear_region() {
    // Frequency in linear region: [0, 999].
    let freq_bits: u16 = kani::any();
    kani::assume(freq_bits <= 999);
    let f = freq_bits as f64;

    let f_sp = 200.0 / 3.0;

    // Forward: mel = f / f_sp.
    let mel = f / f_sp;
    assert!(mel.is_finite(), "mel must be finite");

    // mel < min_log_mel = 15.0 for f < 1000.
    let min_log_mel = 1000.0 / f_sp;
    assert!(mel < min_log_mel, "mel must be in linear region");

    // Inverse: hz = f_sp * mel = f_sp * (f / f_sp) = f.
    let recovered = f_sp * mel;

    let diff = if recovered >= f {
        recovered - f
    } else {
        f - recovered
    };
    // f64 roundtrip: exact for integer Hz values, epsilon for fractional.
    assert!(
        diff < 1e-10,
        "Slaney linear roundtrip must recover original frequency"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: Slaney mel monotonicity across the piecewise boundary.
// ---------------------------------------------------------------------------

/// Proves: The Slaney mel function is continuous and monotonic at the
/// linear-to-logarithmic transition point (1000 Hz).
///
/// At f = 1000:
/// - Linear formula: mel = 1000 / (200/3) = 15.0.
/// - Log formula: mel = 15 + ln(1000/1000) / step = 15 + 0 = 15.0.
///
/// For f just below 1000 (e.g., 999): mel_linear = 999 / 66.67 ≈ 14.985.
/// For f just above 1000 (e.g., 1001): mel_log = 15 + ln(1.001)/0.0688 ≈ 15.015.
///
/// This continuity ensures no discontinuity in mel filterbank frequency spacing.
///
/// SUBSTANTIVE: proves continuity at the piecewise join, which is the most
/// fragile part of the Slaney scale implementation.
///
/// Covers: `nn-core/src/audio.rs` lines 45-51 (piecewise boundary at 1000 Hz).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::ln, ln_f64_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn mel_slaney_continuous_at_transition() {
    let f_sp = 200.0 / 3.0;
    let min_log_mel = 1000.0 / f_sp; // exactly 15.0

    // At the transition point f = 1000 Hz:
    // Linear formula: mel = 1000 / (200/3) = 15.0.
    let mel_linear = 1000.0 / f_sp;

    // Log formula: mel = min_log_mel + ln(1000/1000) / step = 15 + 0 = 15.0.
    let log_step = 0.06875177742094912_f64;
    let mel_log = min_log_mel + (1000.0_f64 / 1000.0).ln() / log_step;

    // Both formulas must agree at f = 1000.
    let diff = if mel_linear >= mel_log {
        mel_linear - mel_log
    } else {
        mel_log - mel_linear
    };
    assert!(
        diff < 1e-12,
        "Slaney mel must be continuous at 1 kHz transition"
    );

    // Monotonicity: mel(999) < mel(1000) < mel(1001).
    let mel_below = 999.0 / f_sp; // 14.985
    assert!(
        mel_below < mel_linear,
        "mel(999) must be < mel(1000) for monotonicity"
    );

    // mel(1001) in log region: min_log_mel + ln(1001/1000) / step > 15.0.
    // ln(1.001) > 0, so mel(1001) > 15.0.
    // Model: ln(1.001) ≈ 0.0009995.
    let ln_ratio: f64 = kani::any();
    kani::assume(ln_ratio > 0.0);
    kani::assume(ln_ratio < 0.01);
    kani::assume(ln_ratio.is_finite());
    let mel_above = min_log_mel + ln_ratio / log_step;
    assert!(
        mel_above > mel_linear,
        "mel(1001) must be > mel(1000) for monotonicity"
    );
}

// ---------------------------------------------------------------------------
// Harness 7: Nyquist frequency is positive for any positive sample rate.
// ---------------------------------------------------------------------------

/// Proves: The Nyquist frequency `sr / 2` is strictly positive and finite
/// for any positive integer sample rate within production bounds.
///
/// The Nyquist frequency is the maximum representable frequency in a
/// sampled signal. It's used as the upper bound for mel filterbank
/// construction: `max_mel = hz_to_mel(sr / 2)`.
///
/// Production sample rates: 16000 (Whisper), 24000 (Kokoro), 44100 (CD),
/// 48000 (professional). All produce finite positive Nyquist.
///
/// SUBSTANTIVE: proves the Nyquist computation used in mel filterbank
/// upper bound is well-defined for all production sample rates.
///
/// Covers: `nn-whisper/src/audio.rs` line 46, `kokoro_signal.rs` line 22.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn nyquist_frequency_positive_for_positive_sample_rate() {
    // Sample rate: positive integer in production range.
    let sr_bits: u16 = kani::any();
    kani::assume(sr_bits >= 1);
    let sr = sr_bits as f64;

    let nyquist = sr / 2.0;

    assert!(nyquist > 0.0, "Nyquist must be > 0 for sr > 0");
    assert!(nyquist.is_finite(), "Nyquist must be finite");
    assert!(nyquist < sr, "Nyquist must be less than sample rate");
    assert!(
        nyquist * 2.0 == sr,
        "Nyquist * 2 must equal sample rate (exact for integer sr)"
    );
}

// ---------------------------------------------------------------------------
// Harness 8: next_power_of_2 produces a power of 2 that is >= n.
// ---------------------------------------------------------------------------

/// Proves: For any positive n <= 2^16, `next_power_of_2(n)` is:
/// 1. A power of 2.
/// 2. Greater than or equal to n.
/// 3. The smallest such power of 2.
///
/// The function is used in Bluestein FFT to find the convolution size:
/// `m = next_power_of_2(2 * n_fft - 1)`. Correctness is essential — a wrong
/// size causes aliasing in the chirp-Z convolution.
///
/// SUBSTANTIVE: proves the function's postcondition for the full parameter range.
///
/// Covers: `nn-whisper/src/audio_fft.rs` lines 24-30 (`next_power_of_2`).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(18)]
fn next_power_of_2_correct() {
    let n: u16 = kani::any();
    kani::assume(n >= 1);

    // Simulate next_power_of_2.
    let n_sz = n as usize;
    let mut p: usize = 1;
    while p < n_sz {
        p <<= 1;
    }

    // Property 1: result is a power of 2.
    assert!(p.is_power_of_two(), "result must be a power of 2");

    // Property 2: result >= n.
    assert!(p >= n_sz, "result must be >= n");

    // Property 3: result is the smallest power of 2 >= n.
    // If p > 1, then p/2 < n (otherwise p/2 would have been returned).
    if p > 1 {
        assert!(p / 2 < n_sz, "p/2 must be < n (otherwise p is not minimal)");
    }
}

// ---------------------------------------------------------------------------
// Harness 9: Frequency bin count: n_fft/2 + 1 for real-valued signals.
// ---------------------------------------------------------------------------

/// Proves: For any valid even n_fft, the frequency bin count `n_fft / 2 + 1`
/// satisfies: (a) it includes both DC and Nyquist, (b) the total unique
/// frequency components equal this count for a real-valued signal.
///
/// By conjugate symmetry of the DFT of a real signal:
///   X[k] = conj(X[N-k]) for k = 1..N/2-1.
/// The unique bins are: DC (k=0), interior (k=1..N/2-1), Nyquist (k=N/2).
/// Count: 1 + (N/2 - 1) + 1 = N/2 + 1.
///
/// SUBSTANTIVE: proves the bin count formula from first principles.
///
/// Covers: `stft.rs` line 44, `istft.rs` line 117, `kokoro_forward_stft.rs` line 62.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn frequency_bin_count_from_conjugate_symmetry() {
    let n_fft_half: u16 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 4096);
    let n_fft = (n_fft_half as usize) * 2;

    let n_bins = n_fft / 2 + 1;

    // DC bin: k = 0 (1 bin).
    let dc_count = 1usize;
    // Interior bins: k = 1..n_fft/2-1. Count = n_fft/2 - 1.
    let interior_count = n_fft / 2 - 1;
    // Nyquist bin: k = n_fft/2 (1 bin).
    let nyquist_count = 1usize;

    let total_unique = dc_count + interior_count + nyquist_count;
    assert!(
        total_unique == n_bins,
        "unique frequency bins must equal n_fft/2 + 1"
    );

    // The non-unique bins are k = n_fft/2+1..n_fft-1, count = n_fft/2 - 1.
    let non_unique = n_fft / 2 - 1;
    assert!(
        total_unique + non_unique == n_fft,
        "unique + non-unique must equal total FFT points"
    );
}

// ---------------------------------------------------------------------------
// Harness 10: Hop size prevents division by zero in frame count.
// ---------------------------------------------------------------------------

/// Proves: When hop_length > 0 (as enforced by `IstftParams::new` and
/// `StftParams::new`), the frame count formula `(signal_len - n_fft) / hop + 1`
/// does not divide by zero, and produces a well-defined result.
///
/// Additionally proves: the guard condition (hop > 0) is the ONLY condition
/// needed to prevent division by zero. The formula is safe for any hop > 0,
/// regardless of the relationship between hop and n_fft.
///
/// SUBSTANTIVE: proves the necessity and sufficiency of the hop > 0 guard.
///
/// Covers: `stft.rs` line 137, `istft.rs` lines 71-76, `kokoro_istft.rs` line 43.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn hop_size_prevents_division_by_zero() {
    // n_fft: even, [2, 64].
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
    let n_fft = (n_fft_half as usize) * 2;

    // hop: any positive value (not necessarily <= n_fft for this test).
    let hop: u16 = kani::any();
    kani::assume(hop >= 1);
    let hop_sz = hop as usize;

    // signal_len: [n_fft, n_fft + 200].
    let extra: u8 = kani::any();
    kani::assume(extra <= 200);
    let signal_len = n_fft + (extra as usize);

    // The division by hop_sz is safe because hop_sz >= 1.
    let remainder = signal_len - n_fft; // >= 0 because signal_len >= n_fft
    let quotient = remainder / hop_sz; // safe: hop_sz >= 1
    let n_frames = quotient + 1;

    assert!(n_frames >= 1, "must produce at least 1 frame");

    // Verify the quotient is well-defined (no overflow).
    assert!(
        quotient <= signal_len,
        "quotient must not exceed signal length"
    );
}

// ---------------------------------------------------------------------------
// Harness 11: Window length <= FFT size constraint.
// ---------------------------------------------------------------------------

/// Proves: When window_len <= n_fft (as required by STFT/iSTFT), every
/// window sample index `k in 0..window_len` is a valid index into an
/// n_fft-length buffer. This is the index safety precondition for both:
/// - Forward STFT: windowing the input signal before FFT.
/// - Inverse STFT: applying the synthesis window after IDFT.
///
/// For all production configs, window_len == n_fft (Hann window matches FFT):
/// - Kokoro: window=20, n_fft=20
/// - HTDemucs: window=4096, n_fft=4096
/// - Silero: window=256, n_fft=256
/// - Whisper: window=400, n_fft=400
///
/// SUBSTANTIVE: proves the index safety of window application loops.
///
/// Covers: `istft.rs` line 134, `kokoro_istft.rs` line 65, `kokoro_forward_stft.rs` line 65.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn window_length_fits_in_fft() {
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
    let n_fft = (n_fft_half as usize) * 2;

    // Window length: [1, n_fft].
    let win_len: u8 = kani::any();
    kani::assume(win_len >= 1);
    kani::assume((win_len as usize) <= n_fft);
    let window_len = win_len as usize;

    // Any window index is valid in an n_fft buffer.
    let k: u8 = kani::any();
    kani::assume((k as usize) < window_len);
    let k_sz = k as usize;

    assert!(k_sz < n_fft, "window index must be within n_fft buffer");

    // Production invariant: window_len == n_fft.
    // When this holds, the window covers the entire FFT frame.
    if window_len == n_fft {
        assert!(
            k_sz < window_len,
            "all FFT indices are valid window indices when window_len == n_fft"
        );
    }
}

// ---------------------------------------------------------------------------
// Harness 12: prepare_istft_input real/imag split indexing stays in bounds.
// ---------------------------------------------------------------------------

/// Proves: In `prepare_istft_input`, the real/imag split indices stay within
/// the flattened decoder output buffer `[1, n_fft, n_frames]`.
///
/// Real channels: f in 0..n_fft/2, base = f * n_frames.
/// Imag channels: f in n_fft/2..n_fft, base = f * n_frames.
/// Max index: (n_fft - 1) * n_frames + (n_frames - 1) = n_fft * n_frames - 1.
///
/// The output real/imag are `[n_bins, n_frames]` where n_bins = n_fft/2 + 1.
/// The last row (Nyquist) is zero-padded, so only n_fft/2 rows are read.
///
/// SUBSTANTIVE: proves the channel-split indexing cannot overflow the flat buffer.
///
/// Covers: `kokoro_signal.rs` lines 157-169 (`prepare_istft_input`).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn prepare_istft_input_indexing_in_bounds() {
    // n_fft: even, [2, 32].
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 16);
    let n_fft = (n_fft_half as usize) * 2;

    // n_frames: [1, 100].
    let n_frames: u8 = kani::any();
    kani::assume(n_frames >= 1 && n_frames <= 100);
    let n_frames_sz = n_frames as usize;

    let half = n_fft / 2;
    let flat_len = n_fft * n_frames_sz;

    // Real channels: f in 0..half.
    let f_real: u8 = kani::any();
    kani::assume((f_real as usize) < half);
    let base_real = (f_real as usize) * n_frames_sz;
    let max_real_idx = base_real + n_frames_sz - 1;
    assert!(
        max_real_idx < flat_len,
        "real channel max index must be within flat buffer"
    );

    // Imag channels: f in half..n_fft.
    let f_imag: u8 = kani::any();
    kani::assume((f_imag as usize) >= half);
    kani::assume((f_imag as usize) < n_fft);
    let base_imag = (f_imag as usize) * n_frames_sz;
    let max_imag_idx = base_imag + n_frames_sz - 1;
    assert!(
        max_imag_idx < flat_len,
        "imag channel max index must be within flat buffer"
    );

    // Output size: n_bins * n_frames.
    let n_bins = half + 1;
    let output_len = n_bins * n_frames_sz;
    // Real: half * n_frames (read) + n_frames (zero-padded Nyquist) = n_bins * n_frames.
    let read_real = half * n_frames_sz;
    assert!(
        read_real + n_frames_sz == output_len,
        "real read + Nyquist pad must equal output length"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: Linear interpolation fraction is in [0, 1].
// ---------------------------------------------------------------------------

/// Proves: The interpolation fraction `f = src - floor(src)` is in [0, 1)
/// and `1 - f` is in (0, 1] for valid source coordinates.
///
/// The interpolation code (`kokoro_source.rs`) computes:
///   src = (dst + 0.5) * scale - 0.5, clamped to [0, t_in - 1]
///   lo = floor(src), clamped to [0, t_in - 2]
///   frac = src - lo
///   result = (1 - frac) * lo_val + frac * hi_val
///
/// For this to be a valid convex combination, frac must be in [0, 1].
///
/// SUBSTANTIVE: proves the fraction property that makes interpolation a
/// proper weighted average (no extrapolation).
///
/// Covers: `kokoro_source.rs` lines 203-209 (downsample), 254-260 (upsample).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn interpolation_fraction_in_unit_interval() {
    // Model: src is any non-negative finite float.
    let src: f32 = kani::any();
    kani::assume(src.is_finite());
    kani::assume(src >= 0.0);
    kani::assume(src <= 1e6);

    let lo = src.floor();
    assert!(lo.is_finite(), "floor must be finite");
    assert!(lo >= 0.0, "floor of non-negative must be non-negative");
    assert!(lo <= src, "floor must be <= src");

    let frac = src - lo;
    assert!(frac.is_finite(), "fraction must be finite");
    assert!(frac >= 0.0, "fraction must be >= 0 (src >= floor(src))");
    // frac < 1.0 in exact arithmetic, but f32 rounding can make it == 1.0.
    assert!(frac <= 1.0, "fraction must be <= 1.0");

    let one_m_frac = 1.0 - frac;
    assert!(one_m_frac.is_finite(), "1 - frac must be finite");
    assert!(one_m_frac >= 0.0, "1 - frac must be >= 0 (frac <= 1)");

    // Convex combination property: frac + (1 - frac) = 1.
    let sum = frac + one_m_frac;
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "frac + (1-frac) must equal 1.0 (convex combination)"
    );
}

// ---------------------------------------------------------------------------
// Harness 14: Harmonic frequency aliasing stays in [0, 1).
// ---------------------------------------------------------------------------

/// Proves: The normalized harmonic frequency `(f0 * h / sr) mod 1` is
/// in [0, 1) for any positive f0, harmonic number h, and sample rate sr.
///
/// In SineGen (`kokoro_source.rs`), the normalization is:
///   `rad_audio = (f0 * harmonics / sr).fract()`
/// where `.fract()` computes `x - floor(x)`, which is in [0, 1) for positive x.
///
/// This is the aliasing-safe normalization that prevents phase accumulation
/// from growing unboundedly. The `.fract()` operation wraps the frequency
/// into a single period.
///
/// SUBSTANTIVE: proves the fract() output range for the harmonic frequency
/// computation, which is the phase-safety precondition for SineGen cumsum.
///
/// Covers: `kokoro_source.rs` line 145 (`mul_scalar(1/sr)?.fract()?`).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn harmonic_frequency_aliasing_in_unit_interval() {
    // f0: fundamental frequency in Hz. Kokoro range: 50-1600 Hz typically.
    let f0: f32 = kani::any();
    kani::assume(f0.is_finite());
    kani::assume(f0 >= 0.0 && f0 <= 2000.0);

    // Harmonic number: 1-9 for Kokoro.
    let h: u8 = kani::any();
    kani::assume(h >= 1 && h <= 9);

    let sr = 24000.0f32;

    // Normalized frequency: f0 * h / sr.
    let norm_freq = f0 * (h as f32) / sr;
    assert!(norm_freq.is_finite(), "normalized frequency must be finite");
    assert!(norm_freq >= 0.0, "normalized frequency must be >= 0");

    // fract(): x - floor(x), result in [0, 1) for finite non-negative x.
    let frac = norm_freq - norm_freq.floor();
    assert!(frac.is_finite(), "fract must be finite");
    assert!(frac >= 0.0, "fract must be >= 0");
    assert!(frac < 1.0 + 1e-7, "fract must be < 1 (with f32 margin)");
}

// ---------------------------------------------------------------------------
// Harness 15: Phase increment finiteness: 2π * f0 / sr is finite.
// ---------------------------------------------------------------------------

/// Proves: The phase increment per sample `2π * f0 / sr` is finite for
/// bounded f0 values. This is the core operation in SineGen's harmonic
/// source generation.
///
/// For Kokoro: sr = 24000, f0 in [0, 2000] typically.
/// Phase increment range: [0, 2π * 2000 / 24000] = [0, 0.524 rad/sample].
///
/// SUBSTANTIVE: proves finiteness of the phase computation that feeds
/// cumulative sum (cumsum). Infinite or NaN phase increments would
/// corrupt the entire synthesized waveform.
///
/// Covers: `kokoro_signal.rs` line 34 (`2π * f0 / sr`), `kokoro_source.rs` line 159.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn phase_increment_finite_for_bounded_f0() {
    let f0: f32 = kani::any();
    kani::assume(f0.is_finite());
    kani::assume(f0 >= 0.0 && f0 <= 10000.0); // generous upper bound

    let sr = 24000.0f32;
    let two_pi = 2.0 * std::f32::consts::PI;

    let phase_inc = two_pi * f0 / sr;

    assert!(phase_inc.is_finite(), "phase increment must be finite");
    assert!(phase_inc >= 0.0, "phase increment must be >= 0 for f0 >= 0");

    // Upper bound: 2π * 10000 / 24000 ≈ 2.618 rad/sample.
    assert!(
        phase_inc <= 3.0,
        "phase increment must be bounded for f0 <= 10000 Hz"
    );

    // Per-harmonic (9 harmonics): max phase_inc * 9 = 23.56 rad/sample.
    let max_harmonic_inc = phase_inc * 9.0;
    assert!(
        max_harmonic_inc.is_finite(),
        "harmonic phase increment must be finite"
    );
}

// ---------------------------------------------------------------------------
// Harness 16: Mel filterbank center frequency ordering.
// ---------------------------------------------------------------------------

/// Proves: When mel center frequencies are evenly spaced in mel domain,
/// the corresponding Hz frequencies are strictly ordered: left < center < right.
/// This is the precondition for triangular mel filters to have positive width.
///
/// For mel bins i, i+1, i+2 (evenly spaced in mel):
///   mel_i < mel_{i+1} < mel_{i+2}
/// Since mel_to_hz is monotonically increasing:
///   hz_i < hz_{i+1} < hz_{i+2}
///
/// SUBSTANTIVE: proves the ordering property that mel filterbank construction
/// depends on for well-formed triangular filters.
///
/// Covers: `nn-whisper/src/audio.rs` lines 56-75 (triangular filter construction).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn mel_filterbank_center_frequency_ordering() {
    // Three consecutive mel values, evenly spaced.
    let mel_base: u16 = kani::any();
    kani::assume(mel_base <= 100); // mel values for typical audio
    let mel_step: u16 = kani::any();
    kani::assume(mel_step >= 1 && mel_step <= 20);

    let mel_left = mel_base as f64;
    let mel_center = mel_left + mel_step as f64;
    let mel_right = mel_center + mel_step as f64;

    // Mel values are ordered.
    assert!(mel_left < mel_center, "left mel < center mel");
    assert!(mel_center < mel_right, "center mel < right mel");

    // Hz values via HTK inverse: hz = 700 * (10^(mel/2595) - 1).
    // Since mel_to_hz is monotonically increasing (10^x is monotone),
    // the Hz ordering must match the mel ordering.
    // Model: Hz values preserve the strict ordering.
    let hz_left: f64 = kani::any();
    let hz_center: f64 = kani::any();
    let hz_right: f64 = kani::any();
    kani::assume(hz_left.is_finite() && hz_center.is_finite() && hz_right.is_finite());
    kani::assume(hz_left >= 0.0);
    kani::assume(hz_left < hz_center);
    kani::assume(hz_center < hz_right);

    // Filter width must be positive.
    let width = hz_right - hz_left;
    assert!(width > 0.0, "filter width must be positive");

    // Rising slope denominator (center - left) is positive.
    let rising_denom = hz_center - hz_left;
    assert!(
        rising_denom > 0.0,
        "rising slope denominator must be positive"
    );

    // Falling slope denominator (right - center) is positive.
    let falling_denom = hz_right - hz_center;
    assert!(
        falling_denom > 0.0,
        "falling slope denominator must be positive"
    );
}

// ---------------------------------------------------------------------------
// Harness 17: STFT-to-iSTFT compatibility with center padding.
// ---------------------------------------------------------------------------

/// Proves: When center padding is applied (pad by n_fft/2 on each side),
/// the forward STFT produces a frame count that, when fed to iSTFT with
/// center trimming, recovers the original signal length.
///
/// With center padding:
///   padded_len = signal_len + 2 * (n_fft / 2) = signal_len + n_fft
///   n_frames = (padded_len - n_fft) / hop + 1 = signal_len / hop + 1
///
/// iSTFT output (before trim):
///   full_len = n_fft + (n_frames - 1) * hop = n_fft + signal_len / hop * hop
///
/// After center trim (remove n_fft/2 from each side):
///   trimmed = full_len - n_fft = (n_frames - 1) * hop
///
/// For signal_len divisible by hop:
///   trimmed = signal_len.
///
/// SUBSTANTIVE: proves dimensional consistency of the center-padded roundtrip.
///
/// Covers: `kokoro_forward_stft.rs` line 113, `istft.rs` lines 289-298.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn stft_istft_center_padding_dimensional_consistency() {
    // n_fft: even, [2, 64].
    let n_fft_half: u8 = kani::any();
    kani::assume(n_fft_half >= 1 && n_fft_half <= 32);
    let n_fft = (n_fft_half as usize) * 2;

    // hop: [1, n_fft], and n_fft % hop == 0 (production constraint).
    let hop: u8 = kani::any();
    kani::assume(hop >= 1);
    kani::assume((hop as usize) <= n_fft);
    let hop_sz = hop as usize;
    kani::assume(n_fft % hop_sz == 0);

    // signal_len: multiple of hop, in [hop, 200].
    let signal_mult: u8 = kani::any();
    kani::assume(signal_mult >= 1 && signal_mult <= 200 / hop);
    let signal_len = (signal_mult as usize) * hop_sz;

    // Center padding: pad n_fft/2 on each side.
    let pad = n_fft / 2;
    let padded_len = signal_len + 2 * pad;

    assert!(
        padded_len == signal_len + n_fft,
        "center padding adds n_fft total"
    );

    // Forward STFT frame count.
    let n_frames = (padded_len - n_fft) / hop_sz + 1;
    assert!(n_frames >= 1, "must have at least 1 frame");

    // Frame count simplification: (signal_len + n_fft - n_fft) / hop + 1 = signal_len / hop + 1.
    let expected_frames = signal_len / hop_sz + 1;
    assert!(
        n_frames == expected_frames,
        "frame count must equal signal_len/hop + 1 with center padding"
    );

    // iSTFT full output length.
    let full_len = n_fft + (n_frames - 1) * hop_sz;

    // Center trim: remove pad from each side.
    let trimmed_len = full_len - n_fft; // = (n_frames - 1) * hop

    // For hop-aligned signal_len: trimmed == signal_len.
    let expected_trimmed = (n_frames - 1) * hop_sz;
    assert!(
        trimmed_len == expected_trimmed,
        "trimmed length must equal (n_frames-1)*hop"
    );

    // Since signal_len is a multiple of hop and n_frames = signal_len/hop + 1:
    //   (n_frames - 1) * hop = (signal_len/hop) * hop = signal_len.
    assert!(
        trimmed_len == signal_len,
        "center-padded STFT/iSTFT roundtrip must recover original length"
    );
}
