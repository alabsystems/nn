// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio-level quality metrics for speech processing evaluation.
//!
//! - **SNR** (Signal-to-Noise Ratio): ratio of signal power to noise power in dB.
//! - **PESQ-lite**: simplified perceptual quality approximation based on spectral
//!   distortion and loudness alignment (not full ITU-T P.862).

use std::f32::consts::PI;

/// Compute Signal-to-Noise Ratio (SNR) between reference and degraded audio.
///
/// SNR = 10 * log10( sum(reference^2) / sum((reference - hypothesis)^2) )
///
/// Both slices must have the same length. Returns the SNR in decibels (dB).
///
/// # Edge cases
///
/// - Returns `f32::INFINITY` when the signals are identical (zero noise).
/// - Returns `f32::NEG_INFINITY` when the reference is all zeros (zero signal power).
/// - Returns `f32::NAN` for empty slices.
///
/// # Panics
///
/// Panics if `reference` and `hypothesis` have different lengths.
///
/// # Examples
///
/// ```
/// use nn_whisper::audio_snr;
///
/// let reference = [1.0_f32, 0.0, -1.0, 0.0];
/// let same = [1.0_f32, 0.0, -1.0, 0.0];
/// assert_eq!(audio_snr(&reference, &same), f32::INFINITY);
///
/// let noisy = [1.1_f32, 0.1, -0.9, 0.1];
/// let snr = audio_snr(&reference, &noisy);
/// assert!(snr > 0.0); // signal stronger than noise
/// ```
#[must_use]
pub fn audio_snr(reference: &[f32], hypothesis: &[f32]) -> f32 {
    assert_eq!(
        reference.len(),
        hypothesis.len(),
        "audio_snr: reference and hypothesis must have the same length"
    );

    if reference.is_empty() {
        return f32::NAN;
    }

    let signal_power: f64 = reference.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let noise_power: f64 = reference
        .iter()
        .zip(hypothesis.iter())
        .map(|(&r, &h)| {
            let diff = f64::from(r) - f64::from(h);
            diff * diff
        })
        .sum();

    if noise_power == 0.0 {
        return f32::INFINITY;
    }
    if signal_power == 0.0 {
        return f32::NEG_INFINITY;
    }

    (10.0 * (signal_power / noise_power).log10()) as f32
}

/// Simplified perceptual audio quality approximation.
///
/// Produces a score loosely modeled on PESQ (ITU-T P.862) by combining:
/// 1. **Spectral distortion**: average log-spectral distance across short-time
///    frames using a simple DFT.
/// 2. **Loudness alignment**: RMS level difference penalty.
///
/// The output is mapped to a 1.0 -- 4.5 scale (similar to MOS/PESQ range):
/// - 4.5 = imperceptible distortion
/// - 1.0 = severely degraded
///
/// This is NOT a standards-compliant PESQ implementation. It provides a fast,
/// dependency-free approximation useful for regression testing and relative
/// quality comparisons within the nn pipeline.
///
/// # Arguments
///
/// - `reference`: clean reference audio samples (mono, normalized to [-1, 1]).
/// - `hypothesis`: degraded/synthesized audio samples (same length as reference).
/// - `sample_rate`: sample rate in Hz (e.g., 16000, 22050, 44100).
///
/// # Panics
///
/// Panics if `reference` and `hypothesis` have different lengths.
///
/// # Examples
///
/// ```
/// use nn_whisper::pesq_approximation;
///
/// let reference: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
/// let same = reference.clone();
/// let score = pesq_approximation(&reference, &same, 16000);
/// assert!(score >= 4.0, "identical signals should score high: {score}");
/// ```
#[must_use]
pub fn pesq_approximation(reference: &[f32], hypothesis: &[f32], sample_rate: usize) -> f32 {
    assert_eq!(
        reference.len(),
        hypothesis.len(),
        "pesq_approximation: reference and hypothesis must have the same length"
    );

    if reference.is_empty() {
        return 1.0; // No data => worst score.
    }

    // Frame parameters: 20ms frames with 10ms hop (standard for speech).
    let frame_len = (sample_rate as f64 * 0.020).round() as usize;
    let hop_len = (sample_rate as f64 * 0.010).round() as usize;

    if frame_len == 0 || hop_len == 0 || reference.len() < frame_len {
        // Signal too short for even one frame — fall back to simple SNR mapping.
        let snr = audio_snr(reference, hypothesis);
        return snr_to_pesq_scale(snr);
    }

    // 1. Spectral distortion: average log-spectral distance across frames.
    let mut total_lsd = 0.0_f64;
    let mut frame_count = 0usize;
    let n_fft = frame_len.next_power_of_two();

    let mut pos = 0;
    while pos + frame_len <= reference.len() {
        let ref_frame = &reference[pos..pos + frame_len];
        let hyp_frame = &hypothesis[pos..pos + frame_len];

        let ref_mag = frame_magnitude_spectrum(ref_frame, n_fft);
        let hyp_mag = frame_magnitude_spectrum(hyp_frame, n_fft);

        let lsd = log_spectral_distance(&ref_mag, &hyp_mag);
        if lsd.is_finite() {
            total_lsd += lsd;
            frame_count += 1;
        }

        pos += hop_len;
    }

    let avg_lsd = if frame_count > 0 {
        total_lsd / frame_count as f64
    } else {
        0.0
    };

    // 2. Loudness alignment: RMS level difference in dB.
    let ref_rms = rms(reference);
    let hyp_rms = rms(hypothesis);
    let loudness_diff_db = if ref_rms > 1e-10 && hyp_rms > 1e-10 {
        (20.0 * (hyp_rms / ref_rms).log10()).abs()
    } else {
        20.0 // Large penalty for silence mismatch.
    };

    // 3. Combine into PESQ-like score.
    // Higher LSD and loudness diff => lower quality.
    // Empirical mapping: LSD of 0 => 4.5, LSD of ~25 dB => 1.0.
    let lsd_score = 4.5 - (avg_lsd as f32 / 8.0).min(3.5);
    let loudness_penalty = (loudness_diff_db as f32 / 20.0).min(1.0);
    

    (lsd_score - loudness_penalty * 0.5).clamp(1.0, 4.5)
}

/// Map an SNR value to the 1.0 -- 4.5 PESQ-like scale.
fn snr_to_pesq_scale(snr: f32) -> f32 {
    if !snr.is_finite() {
        if snr > 0.0 { 4.5 } else { 1.0 }
    } else {
        // Linear map: SNR <= 0 dB => 1.0, SNR >= 35 dB => 4.5
        (1.0 + (snr / 35.0) * 3.5).clamp(1.0, 4.5)
    }
}

/// Compute magnitude spectrum of a single frame via naive DFT.
///
/// Returns `n_fft / 2 + 1` magnitude bins. Applies a Hann window before transform.
fn frame_magnitude_spectrum(frame: &[f32], n_fft: usize) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let frame_len = frame.len();
    let mut magnitudes = vec![0.0_f64; n_bins];

    for k in 0..n_bins {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (n, &sample) in frame.iter().enumerate() {
            // Hann window: 0.5 * (1 - cos(2*pi*n / (N-1)))
            let window = if frame_len > 1 {
                0.5 * (1.0 - (2.0 * f64::from(PI) * n as f64 / (frame_len - 1) as f64).cos())
            } else {
                1.0
            };
            let x = f64::from(sample) * window;
            let angle = -2.0 * f64::from(PI) * k as f64 * n as f64 / n_fft as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes[k] = re.hypot(im);
    }

    magnitudes
}

/// Log-spectral distance between two magnitude spectra in dB.
///
/// LSD = sqrt( mean( (10*log10(|X|^2) - 10*log10(|Y|^2))^2 ) )
///
/// Floor at -80 dB to avoid log(0).
fn log_spectral_distance(ref_mag: &[f64], hyp_mag: &[f64]) -> f64 {
    let floor = 1e-8_f64; // -80 dB floor
    let n = ref_mag.len().min(hyp_mag.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let ref_power_db = 10.0 * (ref_mag[i] * ref_mag[i]).max(floor).log10();
        let hyp_power_db = 10.0 * (hyp_mag[i] * hyp_mag[i]).max(floor).log10();
        let diff = ref_power_db - hyp_power_db;
        sum_sq += diff * diff;
    }

    (sum_sq / n as f64).sqrt()
}

/// Root mean square of a signal.
fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SNR tests ----

    #[test]
    fn test_snr_identical_signals() {
        let signal = [1.0_f32, -1.0, 0.5, -0.5];
        assert_eq!(audio_snr(&signal, &signal), f32::INFINITY);
    }

    #[test]
    fn test_snr_zero_reference() {
        let zeros = [0.0_f32; 4];
        let noise = [0.1_f32, -0.1, 0.1, -0.1];
        assert_eq!(audio_snr(&zeros, &noise), f32::NEG_INFINITY);
    }

    #[test]
    fn test_snr_empty_slices() {
        assert!(audio_snr(&[], &[]).is_nan());
    }

    #[test]
    fn test_snr_known_value() {
        // Signal: [1, 0, -1, 0], power = 2
        // Noise:  [0.1, 0.1, -0.1, 0.1], power = 0.04
        // SNR = 10 * log10(2 / 0.04) = 10 * log10(50) ~ 16.99 dB
        let reference = [1.0_f32, 0.0, -1.0, 0.0];
        let noisy = [1.1_f32, 0.1, -0.9, 0.1];
        let snr = audio_snr(&reference, &noisy);
        assert!((snr - 16.9897).abs() < 0.01, "SNR was {snr}");
    }

    #[test]
    fn test_snr_negative_for_high_noise() {
        // Signal power < noise power => negative SNR.
        let reference = [0.1_f32, -0.1];
        let hypothesis = [1.0_f32, -1.0];
        let snr = audio_snr(&reference, &hypothesis);
        assert!(snr < 0.0, "Expected negative SNR for high noise, got {snr}");
    }

    #[test]
    fn test_snr_symmetry_property() {
        // SNR(ref, hyp) is NOT symmetric in general, but noise power is symmetric
        // in the difference. Verify a basic swap changes the result.
        let a = [1.0_f32, 0.0];
        let b = [0.5_f32, 0.5];
        let snr_ab = audio_snr(&a, &b);
        let snr_ba = audio_snr(&b, &a);
        // These should differ because signal power differs.
        assert!((snr_ab - snr_ba).abs() > 0.01);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_snr_mismatched_lengths() {
        let _ = audio_snr(&[1.0], &[1.0, 2.0]);
    }

    // ---- PESQ approximation tests ----

    #[test]
    fn test_pesq_identical_signals() {
        let signal: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let score = pesq_approximation(&signal, &signal, 16000);
        assert!(
            score >= 4.0,
            "Identical signals should score near max: {score}"
        );
    }

    #[test]
    fn test_pesq_severely_degraded() {
        let signal: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        // White noise as hypothesis — should score low.
        let noise: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 7.37 + 3.14).sin() * 0.8)
            .collect();
        let score = pesq_approximation(&signal, &noise, 16000);
        assert!(score < 3.0, "Noisy signal should score low: {score}");
    }

    #[test]
    fn test_pesq_empty_signals() {
        assert!((pesq_approximation(&[], &[], 16000) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pesq_score_range() {
        // Score must always be in [1.0, 4.5].
        let signal: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.02).sin()).collect();
        let degraded: Vec<f32> = signal.iter().map(|x| x * 0.5 + 0.1).collect();
        let score = pesq_approximation(&signal, &degraded, 16000);
        assert!(
            (1.0..=4.5).contains(&score),
            "Score out of range: {score}"
        );
    }

    #[test]
    fn test_pesq_quality_ordering() {
        // More distorted signal should score lower.
        let signal: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();

        let mild: Vec<f32> = signal.iter().map(|&x| x + 0.01).collect();
        let heavy: Vec<f32> = signal.iter().map(|&x| x * 0.3 + 0.2).collect();

        let score_mild = pesq_approximation(&signal, &mild, 16000);
        let score_heavy = pesq_approximation(&signal, &heavy, 16000);

        assert!(
            score_mild > score_heavy,
            "Mild degradation ({score_mild}) should score higher than heavy ({score_heavy})"
        );
    }

    #[test]
    fn test_pesq_short_signal_fallback() {
        // Signal shorter than one frame (20ms at 16kHz = 320 samples).
        let short_ref = [0.5_f32; 100];
        let short_hyp = [0.5_f32; 100];
        let score = pesq_approximation(&short_ref, &short_hyp, 16000);
        assert!(
            (1.0..=4.5).contains(&score),
            "Short signal score out of range: {score}"
        );
    }

    #[test]
    fn test_pesq_different_sample_rates() {
        let signal: Vec<f32> = (0..22050).map(|i| (i as f32 * 0.01).sin()).collect();
        let same = signal.clone();

        let score_16k = pesq_approximation(&signal[..16000], &same[..16000], 16000);
        let score_22k = pesq_approximation(&signal, &same, 22050);

        // Both identical signals should score high regardless of sample rate.
        assert!(score_16k >= 4.0, "16kHz identical: {score_16k}");
        assert!(score_22k >= 4.0, "22kHz identical: {score_22k}");
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_pesq_mismatched_lengths() {
        let _ = pesq_approximation(&[1.0; 100], &[1.0; 200], 16000);
    }

    // ---- Internal helper tests ----

    #[test]
    fn test_rms_of_sine() {
        // RMS of sin wave over full period = 1/sqrt(2) ~ 0.7071
        let n = 10000;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
            .collect();
        let r = rms(&signal);
        assert!(
            (r - 1.0 / 2.0_f64.sqrt()).abs() < 0.01,
            "RMS of sine: {r}"
        );
    }

    #[test]
    fn test_rms_empty() {
        assert!((rms(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_to_pesq_scale_mapping() {
        assert!((snr_to_pesq_scale(f32::INFINITY) - 4.5).abs() < 1e-6);
        assert!((snr_to_pesq_scale(f32::NEG_INFINITY) - 1.0).abs() < 1e-6);
        assert!((snr_to_pesq_scale(0.0) - 1.0).abs() < 1e-6);
        assert!((snr_to_pesq_scale(35.0) - 4.5).abs() < 1e-6);
        // Midpoint: 17.5 dB => 1.0 + (17.5/35)*3.5 = 1.0 + 1.75 = 2.75
        assert!((snr_to_pesq_scale(17.5) - 2.75).abs() < 1e-6);
    }

    #[test]
    fn test_log_spectral_distance_identical() {
        let mag = vec![1.0, 2.0, 3.0, 4.0];
        let lsd = log_spectral_distance(&mag, &mag);
        assert!(lsd.abs() < 1e-10, "LSD of identical spectra: {lsd}");
    }

    #[test]
    fn test_log_spectral_distance_different() {
        let ref_mag = vec![1.0, 2.0, 3.0];
        let hyp_mag = vec![0.5, 1.0, 1.5];
        let lsd = log_spectral_distance(&ref_mag, &hyp_mag);
        // Each bin has 6.02 dB difference (half magnitude = -6.02 dB power).
        // LSD should be ~ 6.02 dB.
        assert!(lsd > 5.0 && lsd < 7.0, "LSD: {lsd}");
    }
}
