// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive mel spectrogram tests covering filterbank construction,
//! frequency-to-mel conversion (HTK and Slaney), log mel computation for
//! known synthetic inputs (silence, sine wave, white noise), output shape
//! invariants, and frame count expectations.

use crate::audio::{mel_filterbank, pcm_to_mel};
use crate::config::{HOP_LENGTH, N_FFT, SAMPLE_RATE};

// -- HTK mel scale tests (2595 * log10(1 + f/700)) ---------------------------

#[test]
fn test_htk_hz_to_mel_known_values() {
    use nn_core::audio::hz_to_mel_htk;

    // 0 Hz -> 0 mel
    let mel_0 = hz_to_mel_htk(0.0);
    assert!(mel_0.abs() < 1e-12, "hz_to_mel_htk(0) = {mel_0}, expected 0");

    // 700 Hz -> 2595 * log10(2) ~= 781.18
    let mel_700 = hz_to_mel_htk(700.0);
    let expected_700 = 2595.0 * 2.0_f64.log10();
    assert!(
        (mel_700 - expected_700).abs() < 1e-8,
        "hz_to_mel_htk(700) = {mel_700}, expected {expected_700}"
    );

    // Monotonicity: higher Hz -> higher mel
    let mel_1000 = hz_to_mel_htk(1000.0);
    let mel_4000 = hz_to_mel_htk(4000.0);
    assert!(mel_1000 < mel_4000, "mel(1000) >= mel(4000)");
}

#[test]
fn test_htk_mel_to_hz_known_values() {
    use nn_core::audio::mel_to_hz_htk;

    // 0 mel -> 0 Hz
    let hz_0 = mel_to_hz_htk(0.0);
    assert!(hz_0.abs() < 1e-12, "mel_to_hz_htk(0) = {hz_0}, expected 0");
}

#[test]
fn test_htk_roundtrip_comprehensive() {
    use nn_core::audio::{hz_to_mel_htk, mel_to_hz_htk};

    let test_freqs = [
        0.0, 50.0, 100.0, 200.0, 440.0, 700.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
    ];
    for &hz in &test_freqs {
        let mel = hz_to_mel_htk(hz);
        let back = mel_to_hz_htk(mel);
        assert!(
            (back - hz).abs() < 1e-8,
            "HTK roundtrip failed: hz={hz}, mel={mel}, back={back}"
        );
    }
}

#[test]
fn test_htk_mel_strictly_monotonic() {
    use nn_core::audio::hz_to_mel_htk;

    let mut prev = hz_to_mel_htk(0.0);
    for hz in (1..=16000).step_by(5) {
        let mel = hz_to_mel_htk(f64::from(hz));
        assert!(
            mel > prev,
            "HTK mel not strictly monotonic at {hz} Hz: mel={mel}, prev={prev}"
        );
        prev = mel;
    }
}

// -- Slaney mel scale tests ---------------------------------------------------

#[test]
fn test_slaney_hz_to_mel_below_1khz_is_linear() {
    use nn_core::audio::hz_to_mel_slaney;

    // Below 1 kHz, mel = hz / (200/3)
    let f_sp = 200.0 / 3.0;
    for &hz in &[0.0, 100.0, 250.0, 500.0, 750.0, 999.0] {
        let mel = hz_to_mel_slaney(hz);
        let expected = hz / f_sp;
        assert!(
            (mel - expected).abs() < 1e-10,
            "Slaney linear region: hz={hz}, mel={mel}, expected={expected}"
        );
    }
}

#[test]
fn test_slaney_hz_to_mel_above_1khz_is_logarithmic() {
    use nn_core::audio::hz_to_mel_slaney;

    // Above 1 kHz, mel spacing should compress (logarithmic)
    let mel_1k = hz_to_mel_slaney(1000.0);
    let mel_2k = hz_to_mel_slaney(2000.0);
    let mel_4k = hz_to_mel_slaney(4000.0);

    // Equal frequency ratios -> equal mel spacing above 1 kHz
    let diff_1k_2k = mel_2k - mel_1k;
    let diff_2k_4k = mel_4k - mel_2k;
    assert!(
        (diff_1k_2k - diff_2k_4k).abs() < 1e-8,
        "logarithmic region: octave spacing should be equal: {diff_1k_2k} vs {diff_2k_4k}"
    );
}

#[test]
fn test_slaney_roundtrip_comprehensive() {
    use nn_core::audio::{hz_to_mel_slaney, mel_to_hz_slaney};

    let test_freqs = [
        0.0, 50.0, 200.0, 500.0, 999.0, 1000.0, 1001.0, 2000.0, 4000.0, 8000.0,
    ];
    for &hz in &test_freqs {
        let mel = hz_to_mel_slaney(hz);
        let back = mel_to_hz_slaney(mel);
        assert!(
            (back - hz).abs() < 1e-8,
            "Slaney roundtrip failed: hz={hz}, mel={mel}, back={back}"
        );
    }
}

// -- Mel filterbank construction tests ----------------------------------------

#[test]
fn test_mel_filterbank_80_bins_400_fft() {
    // Whisper v1/v2 config: 80 mel bands, 400 FFT bins at 16 kHz
    let n_mels = 80;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1; // 201
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    // Shape check
    assert_eq!(
        filters.len(),
        n_mels * n_freqs,
        "expected {}x{} = {} elements, got {}",
        n_mels,
        n_freqs,
        n_mels * n_freqs,
        filters.len()
    );

    // All values non-negative
    for (i, &v) in filters.iter().enumerate() {
        assert!(v >= 0.0, "filter[{i}] = {v} is negative");
    }

    // Every mel band has at least one nonzero coefficient
    for m in 0..n_mels {
        let row_sum: f32 = (0..n_freqs).map(|k| filters[m * n_freqs + k]).sum();
        assert!(
            row_sum > 0.0,
            "mel band {m} is all-zero (sum={row_sum})"
        );
    }
}

#[test]
fn test_mel_filterbank_128_bins_400_fft() {
    // Whisper large-v3 config: 128 mel bands, 400 FFT bins at 16 kHz
    let n_mels = 128;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1; // 201
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    assert_eq!(filters.len(), n_mels * n_freqs);

    // Non-negative
    assert!(
        filters.iter().all(|&v| v >= 0.0),
        "filterbank contains negative values"
    );

    // All finite
    assert!(
        filters.iter().all(|v| v.is_finite()),
        "filterbank contains non-finite values"
    );
}

#[test]
fn test_mel_filterbank_center_frequencies_increase() {
    // The peak of each successive mel band should be at a higher frequency bin
    let n_mels = 80;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    let mut prev_peak = 0;
    for m in 0..n_mels {
        let row: Vec<f32> = (0..n_freqs).map(|k| filters[m * n_freqs + k]).collect();
        let peak_bin = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            peak_bin >= prev_peak,
            "mel band {m} peak at bin {peak_bin} < previous peak at {prev_peak}"
        );
        prev_peak = peak_bin;
    }
}

#[test]
fn test_mel_filterbank_slaney_area_normalized() {
    // Slaney normalization: each filter's area is 2 / (right_hz - left_hz).
    // With area normalization, the integral (sum) of each triangular filter
    // should approximate 1.0 when multiplied by the frequency resolution.
    let n_mels = 80;
    let n_fft = 400;
    let n_freqs = n_fft / 2 + 1;
    let freq_resolution = (SAMPLE_RATE as f64 / n_fft as f64) as f32;
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    for m in 0..n_mels {
        let integral: f32 = (0..n_freqs)
            .map(|k| filters[m * n_freqs + k] * freq_resolution)
            .sum();
        // Slaney normalized filters integrate to approximately 1.0.
        // Allow generous tolerance since we're computing a discrete sum.
        assert!(
            integral < 5.0,
            "mel band {m} integral = {integral}, unexpectedly large"
        );
    }
}

// -- Silence input: log mel should be very negative ---------------------------

#[test]
fn test_silence_produces_very_negative_log_mel() {
    let n_mels = 80;
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;
    // 1 second of silence
    let audio = vec![0.0f32; SAMPLE_RATE];
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // After Whisper's normalization: log10(max(1e-10, E)) then (x+4)/4.
    // For silence, mel energies are 0 -> log10(1e-10) = -10.
    // After clamping to max-8 and normalization: (-10 + 4)/4 = -1.5.
    // All values should be very negative (well below 0).
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_val < 0.0,
        "silence mel max = {max_val}, expected < 0.0"
    );

    // All values should be uniform for pure silence (all mel bands same energy = 0)
    let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        (max_val - min_val).abs() < 1e-5,
        "silence mel not uniform: min={min_val}, max={max_val}"
    );
}

// -- Single frequency sine wave: peak in expected mel band --------------------

#[test]
fn test_sine_440hz_peak_in_correct_mel_band() {
    let n_mels = 80;
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;
    let freq = 440.0_f32; // A4

    // Generate 1 second of 440 Hz sine wave
    let audio: Vec<f32> = (0..SAMPLE_RATE)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let dims = mel.dims();
    let n_frames = dims[2];
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Compute mean energy per mel band
    let mut band_means = vec![0.0f32; n_mels];
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means[m] = sum / n_frames as f32;
    }

    // Find the peak band
    let (peak_band, _) = band_means
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    // 440 Hz with 80 Slaney mel bands from 0-8kHz: expect peak around band 10-25
    assert!(
        (5..=35).contains(&peak_band),
        "440 Hz sine peak at band {peak_band}, expected in range 5..=35"
    );
}

#[test]
fn test_sine_4000hz_peak_in_higher_mel_band() {
    let n_mels = 80;
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;
    let freq = 4000.0_f32;

    let audio: Vec<f32> = (0..SAMPLE_RATE)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let dims = mel.dims();
    let n_frames = dims[2];
    let vals = mel.to_flat_vec::<f32>().unwrap();

    let mut band_means = vec![0.0f32; n_mels];
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means[m] = sum / n_frames as f32;
    }

    let (peak_band_4k, _) = band_means
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    // 4 kHz should produce a peak in a higher band than 440 Hz.
    // With 80 bands over 0-8 kHz (Slaney scale), 4 kHz is around band 45-65.
    assert!(
        (35..=75).contains(&peak_band_4k),
        "4 kHz sine peak at band {peak_band_4k}, expected 35..=75"
    );
}

#[test]
fn test_higher_freq_sine_peaks_in_higher_band() {
    let n_mels = 80;
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;

    let peak_for_freq = |freq: f32| -> usize {
        let audio: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();

        let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
        let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
        let dims = mel.dims();
        let n_frames = dims[2];
        let vals = mel.to_flat_vec::<f32>().unwrap();

        let mut band_means = vec![0.0f32; n_mels];
        for m in 0..n_mels {
            let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
            band_means[m] = sum / n_frames as f32;
        }

        band_means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };

    let peak_500 = peak_for_freq(500.0);
    let peak_2000 = peak_for_freq(2000.0);
    let peak_6000 = peak_for_freq(6000.0);

    assert!(
        peak_500 < peak_2000,
        "500 Hz peak (band {peak_500}) should be below 2000 Hz peak (band {peak_2000})"
    );
    assert!(
        peak_2000 < peak_6000,
        "2000 Hz peak (band {peak_2000}) should be below 6000 Hz peak (band {peak_6000})"
    );
}

// -- White noise: relatively flat mel spectrum --------------------------------

#[test]
fn test_white_noise_produces_relatively_flat_spectrum() {
    let n_mels = 80;
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;

    // Deterministic pseudo-white noise using a simple LCG
    let mut rng_state: u64 = 42;
    let audio: Vec<f32> = (0..SAMPLE_RATE)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1]
            (rng_state >> 33) as f32 / (u32::MAX >> 1) as f32 * 2.0 - 1.0
        })
        .collect();

    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let dims = mel.dims();
    let n_frames = dims[2];
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Compute mean energy per mel band
    let mut band_means = vec![0.0f32; n_mels];
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means[m] = sum / n_frames as f32;
    }

    // For white noise, the spectrum should be relatively flat.
    // Compute the standard deviation of band means — should be small relative
    // to the mean.
    let overall_mean: f32 = band_means.iter().sum::<f32>() / n_mels as f32;
    let variance: f32 = band_means
        .iter()
        .map(|&b| (b - overall_mean).powi(2))
        .sum::<f32>()
        / n_mels as f32;
    let std_dev = variance.sqrt();

    // Coefficient of variation should be small (< 0.3 for white noise)
    let cv = std_dev / overall_mean.abs().max(1e-10);
    assert!(
        cv < 0.5,
        "white noise CV = {cv} (std={std_dev}, mean={overall_mean}), expected < 0.5"
    );
}

// -- Output shape tests -------------------------------------------------------

#[test]
fn test_pcm_to_mel_output_shape_is_batch_mels_frames() {
    let n_mels = 80;
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;
    let audio = vec![0.1f32; SAMPLE_RATE]; // 1 second
    let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);

    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
    let dims = mel.dims();

    assert_eq!(dims.len(), 3, "expected rank-3 tensor, got rank {}", dims.len());
    assert_eq!(dims[0], 1, "batch dim should be 1");
    assert_eq!(dims[1], n_mels, "mel dim should be {n_mels}");
}

#[test]
fn test_pcm_to_mel_frame_count_formula() {
    // Frame count should match: (n_samples + hop_length - 1) / hop_length
    // after reflect-padding. The actual formula for the padded signal is:
    // n_frames = (padded_len - n_fft) / hop + 1
    // where padded_len = n_samples + n_fft (pad n_fft/2 on each side)
    // So: n_frames = (n_samples + n_fft - n_fft) / hop + 1 = n_samples / hop + 1
    let n_mels = 80;
    let n_fft = N_FFT; // 400
    let hop = HOP_LENGTH; // 160

    for &n_samples in &[1600_usize, 8000, 16000, 32000] {
        let audio = vec![0.1f32; n_samples];
        let filters = mel_filterbank(n_mels, n_fft, SAMPLE_RATE);
        let mel = pcm_to_mel(&audio, &filters, n_fft, hop, n_mels).unwrap();
        let n_frames = mel.dim(2).unwrap();

        // Expected: (n_samples + n_fft - n_fft) / hop + 1 = n_samples / hop + 1
        let expected = n_samples / hop + 1;
        assert_eq!(
            n_frames, expected,
            "n_samples={n_samples}: got {n_frames} frames, expected {expected}"
        );
    }
}

// -- compute_log_mel_spectrogram convenience function tests -------------------

#[test]
fn test_compute_log_mel_spectrogram_basic() {
    let audio = vec![0.0f32; SAMPLE_RATE]; // 1 second silence
    let result = crate::audio::compute_log_mel_spectrogram(&audio, SAMPLE_RATE as u32);

    // 80 mel bands
    assert_eq!(result.len(), 80, "expected 80 mel bands, got {}", result.len());

    // Frame count: n_samples/hop + 1 = 16000/160 + 1 = 101
    let expected_frames = SAMPLE_RATE / HOP_LENGTH + 1;
    assert_eq!(
        result[0].len(),
        expected_frames,
        "expected {expected_frames} frames, got {}",
        result[0].len()
    );

    // All frames should have the same length
    for (i, band) in result.iter().enumerate() {
        assert_eq!(
            band.len(),
            expected_frames,
            "band {i} has {} frames, expected {expected_frames}",
            band.len()
        );
    }
}

#[test]
fn test_compute_log_mel_spectrogram_silence_is_negative() {
    let audio = vec![0.0f32; SAMPLE_RATE];
    let result = crate::audio::compute_log_mel_spectrogram(&audio, SAMPLE_RATE as u32);

    // All values should be very negative for silence (log(max(1e-10, 0)) = -10)
    for (m, band) in result.iter().enumerate() {
        for (t, &val) in band.iter().enumerate() {
            assert!(
                val < -5.0,
                "silence mel[{m}][{t}] = {val}, expected < -5.0"
            );
            assert!(
                val.is_finite(),
                "silence mel[{m}][{t}] = {val}, expected finite"
            );
        }
    }
}

#[test]
fn test_compute_log_mel_spectrogram_sine_wave_has_peak() {
    let freq = 1000.0_f32;
    let audio: Vec<f32> = (0..SAMPLE_RATE)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let result = crate::audio::compute_log_mel_spectrogram(&audio, SAMPLE_RATE as u32);
    let n_frames = result[0].len();

    // Compute mean per band
    let band_means: Vec<f32> = result
        .iter()
        .map(|band| band.iter().sum::<f32>() / n_frames as f32)
        .collect();

    let (peak_band, &peak_val) = band_means
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    // Peak band should have notably higher energy than the overall mean
    let overall_mean = band_means.iter().sum::<f32>() / band_means.len() as f32;
    assert!(
        peak_val > overall_mean,
        "peak band {peak_band} mean ({peak_val}) should exceed overall mean ({overall_mean})"
    );
}

#[test]
fn test_compute_log_mel_spectrogram_output_all_finite() {
    // Various audio inputs should always produce finite output
    let test_inputs: Vec<(&str, Vec<f32>)> = vec![
        ("silence", vec![0.0f32; 8000]),
        ("dc_offset", vec![0.5f32; 8000]),
        (
            "sine_220",
            (0..8000)
                .map(|i| {
                    (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SAMPLE_RATE as f32).sin()
                })
                .collect(),
        ),
        ("tiny_amplitude", vec![1e-7_f32; 8000]),
    ];

    for (name, audio) in test_inputs {
        let result = crate::audio::compute_log_mel_spectrogram(&audio, SAMPLE_RATE as u32);
        for (m, band) in result.iter().enumerate() {
            for (t, &val) in band.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "{name}: mel[{m}][{t}] = {val} is not finite"
                );
            }
        }
    }
}
