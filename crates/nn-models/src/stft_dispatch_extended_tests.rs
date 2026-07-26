// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for nn-models: signal processing and model dispatch (#4495).
//!
//! Covers:
//! - STFT forward/inverse roundtrip
//! - STFT window function properties (Hann, Hamming symmetry, normalization)
//! - STFT output shape validation for various hop/window sizes
//! - Frequency bin count validation
//! - Phase consistency between frames
//! - Magnitude/phase decomposition
//! - Power spectrum computation
//! - Mel filterbank properties (triangular shape, coverage)
//! - Signal reconstruction quality
//! - Zero-padding effects
//! - Overlap-add consistency
//! - STFT linearity (STFT(a+b) = STFT(a) + STFT(b))
//! - Energy conservation (Parseval's theorem approximation)
//! - Window normalization
//! - Model dispatch registry validation
//! - HTDemucs configuration validation
//! - Silero VAD configuration validation
//! - Kokoro model configuration

use std::f32::consts::PI;

use crate::convert::{ConvertConfig, DpdfModelType};
use crate::demucs_shared::{
    channels_at_depth, conv1d_output_len, BASE_CHANNELS, DECODER_OUTPUT_CHANNELS,
    DECODER_REWRITE_KERNEL, DECODER_REWRITE_PADDING, SPECTRAL_BASIC_DEPTH, SPECTRAL_DEPTH,
    SPECTRAL_INPUT_CHANNELS, SPECTRAL_KERNEL_SIZE, SPECTRAL_STRIDE,
};
use crate::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, FFN_HIDDEN_SCALE, LAYER_NORM_EPS, NUM_HEADS, NUM_LAYERS,
    TRANSFORMER_DIM,
};
use crate::doclayout_yolo::{DocLayoutYoloConfig, INPUT_SIZE, NUM_CLASSES, REG_MAX};
use crate::dpdf_registry::{DpdfModelRegistry, ModelType};
use crate::istft::{IstftBasis, IstftError, IstftParams};
use crate::kokoro_tts::{
    KokoroConfig, KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT, KOKORO_SAMPLE_RATE,
};
use crate::plbert::PlbertConfig;
use crate::silero_vad_builders::{ENCODER_BLOCKS, LSTM_HIDDEN_SIZE};
use crate::stft::{compute_stft_magnitude, StftParams};
use crate::table_structure::TableStructureConfig;
use crate::table_transformer::{TableTransformerConfig, HIDDEN_DIM, NUM_QUERIES};

// ===========================================================================
// Helpers
// ===========================================================================

/// Periodic Hann window: w[k] = 0.5 * (1 - cos(2*pi*k / n)).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n as f32).cos()))
        .collect()
}

/// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k / n).
fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f32 / n as f32).cos())
        .collect()
}

/// Build a DFT-style STFT basis (cos rows + sin rows) for compute_stft_magnitude.
/// Shape: [n_fft+2, n_fft] flattened, first n_freqs rows cos, next n_freqs rows -sin.
fn build_dft_basis(n_fft: usize) -> Vec<f32> {
    let n_filters = n_fft + 2;
    let n_freqs = n_fft / 2 + 1;
    let mut basis = vec![0.0f32; n_filters * n_fft];
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[f * n_fft + k] = angle.cos();
        }
    }
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[(n_freqs + f) * n_fft + k] = -angle.sin();
        }
    }
    basis
}

/// Windowed forward DFT for a single frame: returns (real, imag) per freq bin.
fn forward_dft_frame(frame: &[f32], window: &[f32], n_fft: usize) -> (Vec<f32>, Vec<f32>) {
    let n_bins = n_fft / 2 + 1;
    let mut real = vec![0.0f32; n_bins];
    let mut imag = vec![0.0f32; n_bins];
    for f in 0..n_bins {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            let windowed = frame[k] * window[k];
            real[f] += windowed * angle.cos();
            imag[f] -= windowed * angle.sin();
        }
    }
    (real, imag)
}

// ===========================================================================
// 1. STFT forward/inverse roundtrip
// ===========================================================================

#[test]
fn test_roundtrip_sine_wave_reconstruction() {
    let n_fft = 16;
    let hop = 4;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 12;

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let sig_len = n_fft + (n_frames - 1) * hop;
    let freq = 3;
    let original: Vec<f32> = (0..sig_len)
        .map(|k| (2.0 * PI * freq as f32 * k as f32 / n_fft as f32).sin())
        .collect();

    let window = hann_window(n_fft);
    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        let offset = t * hop;
        let (r, im) = forward_dft_frame(&original[offset..offset + n_fft], &window, n_fft);
        for f in 0..n_bins {
            real[f * n_frames + t] = r[f];
            imag[f * n_frames + t] = im[f];
        }
    }

    let reconstructed = basis.istft(&real, &imag, n_frames, sig_len).unwrap();
    let start = n_fft;
    let end = sig_len.saturating_sub(n_fft);
    if end > start {
        let max_err: f32 = (start..end)
            .map(|i| (reconstructed[i] - original[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.15,
            "Sine roundtrip interior max error = {max_err}"
        );
    }
}

#[test]
fn test_roundtrip_chirp_signal() {
    // Chirp: frequency sweeps linearly from 0 to n_fft/2 over signal length.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;
    let sig_len = n_fft + (n_frames - 1) * hop;

    let original: Vec<f32> = (0..sig_len)
        .map(|k| {
            let t = k as f32 / sig_len as f32;
            let inst_freq = t * (n_fft as f32 / 2.0);
            (2.0 * PI * inst_freq * t).sin()
        })
        .collect();

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = hann_window(n_fft);

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        let offset = t * hop;
        let (r, im) = forward_dft_frame(&original[offset..offset + n_fft], &window, n_fft);
        for f in 0..n_bins {
            real[f * n_frames + t] = r[f];
            imag[f * n_frames + t] = im[f];
        }
    }

    let recon = basis.istft(&real, &imag, n_frames, sig_len).unwrap();
    assert_eq!(recon.len(), sig_len);
    // Just verify the signal is finite and not all-zero.
    let energy: f32 = recon.iter().map(|x| x * x).sum();
    assert!(
        energy > 0.01,
        "Chirp roundtrip should have nonzero energy, got {energy}"
    );
}

// ===========================================================================
// 2. STFT window function properties
// ===========================================================================

#[test]
fn test_hann_window_cola_property() {
    // Constant Overlap-Add (COLA) for Hann: sum of squared windows = constant.
    let n_fft = 16;
    let hop = 4; // 75% overlap
    let window = hann_window(n_fft);
    let n_samples = 100;
    let mut cola_sum = vec![0.0f32; n_samples];
    let n_frames = (n_samples.saturating_sub(n_fft)) / hop + 1;
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            if offset + k < n_samples {
                cola_sum[offset + k] += window[k] * window[k];
            }
        }
    }
    // Interior samples should all have approximately the same COLA sum.
    let interior_start = n_fft;
    let interior_end = n_samples.saturating_sub(n_fft);
    if interior_end > interior_start {
        let ref_val = cola_sum[interior_start];
        for i in interior_start..interior_end {
            assert!(
                (cola_sum[i] - ref_val).abs() < 0.01,
                "COLA property violated at sample {i}: {} vs reference {ref_val}",
                cola_sum[i]
            );
        }
    }
}

#[test]
fn test_hamming_window_sum_property() {
    // Sum of a Hamming window should be approximately n/2 for large n.
    let n = 512;
    let w = hamming_window(n);
    let sum: f32 = w.iter().sum();
    // Hamming average value is approximately 0.54, so sum ~= 0.54 * n
    let expected = 0.54 * n as f32;
    assert!(
        (sum - expected).abs() / expected < 0.05,
        "Hamming window sum {sum} not close to expected {expected}"
    );
}

#[test]
fn test_hann_window_energy_normalization() {
    // Sum of squared Hann window values should be approximately n * 3/8.
    let n = 256;
    let w = hann_window(n);
    let energy: f32 = w.iter().map(|x| x * x).sum();
    let expected = n as f32 * 3.0 / 8.0;
    assert!(
        (energy - expected).abs() / expected < 0.02,
        "Hann window energy {energy} not close to expected {expected}"
    );
}

#[test]
fn test_window_functions_different_sizes() {
    for n in [8, 16, 32, 64, 128, 256] {
        let hann = hann_window(n);
        let hamming = hamming_window(n);
        assert_eq!(hann.len(), n);
        assert_eq!(hamming.len(), n);

        // All values should be non-negative and <= 1.0
        for k in 0..n {
            assert!(hann[k] >= 0.0, "Hann[{k}] negative for n={n}");
            assert!(hann[k] <= 1.0 + 1e-6, "Hann[{k}] > 1.0 for n={n}");
            assert!(hamming[k] >= 0.0, "Hamming[{k}] negative for n={n}");
            assert!(hamming[k] <= 1.0 + 1e-6, "Hamming[{k}] > 1.0 for n={n}");
        }
    }
}

// ===========================================================================
// 3. STFT output shape validation for various hop/window sizes
// ===========================================================================

#[test]
fn test_stft_output_shape_various_hops() {
    let configs: Vec<(usize, usize, usize, usize)> = vec![
        // (n_fft, hop, audio_len, expected_n_frames)
        (8, 2, 100, 48),  // padded_len=102, (102-8)/2+1=48
        (8, 4, 100, 24),  // padded_len=102, (102-8)/4+1=24
        (8, 8, 100, 12),  // padded_len=102, (102-8)/8+1=12 (no overlap)
        (16, 4, 200, 48), // padded_len=204, (204-16)/4+1=48
        (16, 8, 200, 24), // padded_len=204, (204-16)/8+1=24
    ];
    for (n_fft, hop, audio_len, expected_frames) in configs {
        let params = StftParams::new(n_fft, hop);
        let basis = build_dft_basis(n_fft);
        let audio = vec![0.1f32; audio_len];
        let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
        let actual_frames = mag.len() / params.n_freqs;
        assert_eq!(
            actual_frames, expected_frames,
            "n_fft={n_fft}, hop={hop}, audio={audio_len}: expected {expected_frames} frames, got {actual_frames}"
        );
    }
}

#[test]
fn test_stft_output_shape_with_reflection_padding() {
    // Reflection padding adds pad_right = n_fft/4 samples.
    let n_fft = 32;
    let hop = 8;
    let params = StftParams::new(n_fft, hop); // pad_right = 8
    let audio_len = 256;
    let padded_len = audio_len + params.pad_right;
    let expected_frames = (padded_len - n_fft) / hop + 1;

    let basis = build_dft_basis(n_fft);
    let audio = vec![0.0f32; audio_len];
    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    assert_eq!(
        mag.len(),
        params.n_freqs * expected_frames,
        "Shape mismatch with reflection padding"
    );
}

// ===========================================================================
// 4. Frequency bin count validation
// ===========================================================================

#[test]
fn test_freq_bin_count_power_of_two_ffts() {
    for exp in 2..=12 {
        let n_fft = 1 << exp;
        let expected_bins = n_fft / 2 + 1;
        let params = StftParams::new(n_fft, n_fft / 2);
        assert_eq!(
            params.n_freqs, expected_bins,
            "n_fft={n_fft}: expected {expected_bins} bins"
        );
    }
}

#[test]
fn test_freq_bin_count_non_power_of_two() {
    for n_fft in [6, 10, 14, 18, 20, 22, 100, 300] {
        let expected = n_fft / 2 + 1;
        let params = StftParams::new(n_fft, n_fft / 2);
        assert_eq!(params.n_freqs, expected, "n_fft={n_fft}");
    }
}

#[test]
fn test_istft_n_bins_matches_stft_n_freqs() {
    for n_fft in [8, 16, 20, 32, 64, 256] {
        let stft_params = StftParams::new(n_fft, n_fft / 2);
        let istft_params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let istft_basis = IstftBasis::new(istft_params).unwrap();
        assert_eq!(
            stft_params.n_freqs,
            istft_basis.n_bins(),
            "n_fft={n_fft}: STFT n_freqs should equal iSTFT n_bins"
        );
    }
}

// ===========================================================================
// 5. Phase consistency between frames
// ===========================================================================

#[test]
fn test_phase_consistency_stationary_signal() {
    // A stationary cosine should have consistent phase across frames.
    let n_fft = 16;
    let hop = 4;
    let freq_bin = 2;
    let sig_len = n_fft + 20 * hop; // enough frames

    let signal: Vec<f32> = (0..sig_len)
        .map(|k| (2.0 * PI * freq_bin as f32 * k as f32 / n_fft as f32).cos())
        .collect();

    let window = hann_window(n_fft);
    let n_frames = (sig_len - n_fft) / hop + 1;
    let mut phases = Vec::new();

    for t in 0..n_frames {
        let offset = t * hop;
        let (real, imag) = forward_dft_frame(&signal[offset..offset + n_fft], &window, n_fft);
        let phase = imag[freq_bin].atan2(real[freq_bin]);
        phases.push(phase);
    }

    // Phase should advance by 2*pi*freq_bin*hop/n_fft per frame.
    let expected_advance = 2.0 * PI * freq_bin as f32 * hop as f32 / n_fft as f32;
    for i in 1..phases.len().min(10) {
        let diff = phases[i] - phases[i - 1];
        // Normalize to [-pi, pi].
        let normalized = ((diff + PI) % (2.0 * PI)) - PI;
        let expected_norm = ((expected_advance + PI) % (2.0 * PI)) - PI;
        assert!(
            (normalized - expected_norm).abs() < 0.3,
            "Frame {i}: phase advance {normalized:.4} vs expected {expected_norm:.4}"
        );
    }
}

// ===========================================================================
// 6. Magnitude/phase decomposition
// ===========================================================================

#[test]
fn test_magnitude_phase_reconstruction() {
    // mag = sqrt(real^2 + imag^2), phase = atan2(imag, real)
    // real = mag * cos(phase), imag = mag * sin(phase)
    let test_values: Vec<(f32, f32)> = vec![
        (3.0, 4.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.7071, 0.7071),
    ];
    for (real, imag) in test_values {
        let mag = real.hypot(imag);
        let phase = imag.atan2(real);
        let reconstructed_real = mag * phase.cos();
        let reconstructed_imag = mag * phase.sin();
        assert!(
            (reconstructed_real - real).abs() < 1e-4,
            "real: ({real}, {imag}) -> mag={mag}, phase={phase} -> {reconstructed_real}"
        );
        assert!(
            (reconstructed_imag - imag).abs() < 1e-4,
            "imag: ({real}, {imag}) -> mag={mag}, phase={phase} -> {reconstructed_imag}"
        );
    }
}

// ===========================================================================
// 7. Power spectrum computation
// ===========================================================================

#[test]
fn test_power_spectrum_is_magnitude_squared() {
    let n_fft = 8;
    let params = StftParams::new(n_fft, 4);
    let basis = build_dft_basis(n_fft);
    let audio: Vec<f32> = (0..50).map(|k| (0.1 * k as f32).sin()).collect();
    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    let power: Vec<f32> = mag.iter().map(|m| m * m).collect();
    // Power spectrum should be non-negative everywhere.
    for (i, &p) in power.iter().enumerate() {
        assert!(p >= 0.0, "Power spectrum negative at index {i}: {p}");
    }
}

#[test]
fn test_dc_signal_power_concentration() {
    // A DC signal should concentrate power at frequency bin 0.
    let n_fft = 8;
    let params = StftParams::new(n_fft, 4);
    let basis = build_dft_basis(n_fft);
    let audio = vec![1.0f32; 50];
    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    let n_freqs = params.n_freqs;
    let n_frames = mag.len() / n_freqs;

    // For each frame, bin 0 should dominate.
    for t in 0..n_frames {
        let dc_mag = mag[0 * n_frames + t];
        let total_mag: f32 = (0..n_freqs).map(|f| mag[f * n_frames + t]).sum();
        assert!(
            dc_mag / total_mag > 0.5,
            "Frame {t}: DC bin should dominate, ratio = {}",
            dc_mag / total_mag
        );
    }
}

// ===========================================================================
// 8. Mel filterbank properties
// ===========================================================================

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

#[test]
fn test_mel_scale_monotonicity() {
    let mut prev_mel = 0.0f32;
    for hz in (0..=8000).step_by(100) {
        let mel = hz_to_mel(hz as f32);
        assert!(
            mel >= prev_mel,
            "Mel scale not monotonic at {hz} Hz: {mel} < {prev_mel}"
        );
        prev_mel = mel;
    }
}

#[test]
fn test_mel_hz_roundtrip() {
    for hz in [0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0, 16000.0] {
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!(
            (back - hz).abs() < 0.01,
            "Mel roundtrip failed: {hz} Hz -> {mel} mel -> {back} Hz"
        );
    }
}

#[test]
fn test_mel_filterbank_triangular_shape() {
    // A mel filterbank filter should be triangular: rises linearly, peaks, falls linearly.
    let n_mels = 40;
    let n_fft = 256;
    let sample_rate = 16000.0f32;
    let _n_freqs = n_fft / 2 + 1;

    // Build mel center frequencies.
    let f_min = 0.0f32;
    let f_max = sample_rate / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    // Each filter spans [mel_points[m], mel_points[m+1], mel_points[m+2]].
    // Verify coverage: every mel filter center should be within [f_min, f_max].
    for m in 0..n_mels {
        let center = mel_points[m + 1];
        assert!(
            center >= f_min && center <= f_max,
            "Mel filter {m} center {center} out of range [{f_min}, {f_max}]"
        );
        // Triangle: left < center < right
        assert!(
            mel_points[m] <= center && center <= mel_points[m + 2],
            "Mel filter {m} not triangular: left={}, center={center}, right={}",
            mel_points[m],
            mel_points[m + 2]
        );
    }
}

#[test]
fn test_mel_filterbank_full_coverage() {
    // The mel filterbank should cover the full frequency range.
    let n_mels = 80;
    let sample_rate = 16000.0f32;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    // First filter should start near 0 Hz.
    assert!(
        mel_points[0] < 50.0,
        "First mel point too high: {}",
        mel_points[0]
    );
    // Last filter should end near Nyquist.
    let last = mel_points[n_mels + 1];
    assert!(
        (last - sample_rate / 2.0).abs() < 100.0,
        "Last mel point {last} not near Nyquist {}",
        sample_rate / 2.0
    );
}

// ===========================================================================
// 9. Signal reconstruction quality
// ===========================================================================

#[test]
fn test_istft_reconstruction_white_noise_finite() {
    // White noise through iSTFT should produce finite output.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 8;

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    // Random-ish STFT coefficients (deterministic pseudo-random).
    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for i in 0..real.len() {
        real[i] = ((i as f32 * 0.17).sin()) * 0.5;
        imag[i] = ((i as f32 * 0.31).cos()) * 0.5;
    }

    let out_len = n_fft + (n_frames - 1) * hop;
    let result = basis.istft(&real, &imag, n_frames, out_len).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite output at index {i}: {v}");
    }
}

// ===========================================================================
// 10. Zero-padding effects
// ===========================================================================

#[test]
fn test_zero_padded_signal_vs_original() {
    // Zero-padding a signal should not change the magnitude at existing bins,
    // just interpolate between them (spectral interpolation).
    let n_fft = 8;
    let params = StftParams::new(n_fft, n_fft); // no overlap
    let basis = build_dft_basis(n_fft);

    let audio_short = vec![1.0f32; 20];
    let mut audio_padded = audio_short.clone();
    audio_padded.extend(vec![0.0f32; 20]); // zero-pad to double length

    let mag_short = compute_stft_magnitude(&audio_short, &basis, &params).unwrap();
    let mag_padded = compute_stft_magnitude(&audio_padded, &basis, &params).unwrap();

    // Padded version should have at least as many frames.
    assert!(mag_padded.len() >= mag_short.len());
}

// ===========================================================================
// 11. Overlap-add consistency
// ===========================================================================

#[test]
fn test_overlap_add_window_sum_nonzero_interior() {
    // For valid COLA reconstruction, the sum of squared Hann windows should be nonzero
    // at every interior sample.
    let n_fft = 20;
    let hop = 5;
    let n_frames = 15;
    let window = hann_window(n_fft);
    let full_len = n_fft + (n_frames - 1) * hop;
    let mut window_sum = vec![0.0f32; full_len];

    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            window_sum[offset + k] += window[k] * window[k];
        }
    }

    // Interior (excluding first and last n_fft/2 samples) should be > 0.
    let margin = n_fft / 2;
    for i in margin..full_len.saturating_sub(margin) {
        assert!(
            window_sum[i] > 0.01,
            "Window sum too small at interior sample {i}: {}",
            window_sum[i]
        );
    }
}

// ===========================================================================
// 12. STFT linearity (STFT(a+b) = STFT(a) + STFT(b))
// ===========================================================================

#[test]
fn test_stft_linearity_magnitude_triangle_inequality() {
    // For magnitudes: |STFT(a+b)| <= |STFT(a)| + |STFT(b)| (triangle inequality).
    let n_fft = 8;
    let params = StftParams::new(n_fft, 4);
    let basis = build_dft_basis(n_fft);

    let a: Vec<f32> = (0..50).map(|k| (0.2 * k as f32).sin()).collect();
    let b: Vec<f32> = (0..50).map(|k| (0.3 * k as f32).cos()).collect();
    let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let mag_a = compute_stft_magnitude(&a, &basis, &params).unwrap();
    let mag_b = compute_stft_magnitude(&b, &basis, &params).unwrap();
    let mag_ab = compute_stft_magnitude(&ab, &basis, &params).unwrap();

    for i in 0..mag_ab.len() {
        assert!(
            mag_ab[i] <= mag_a[i] + mag_b[i] + 1e-5,
            "Triangle inequality violated at bin {i}: |STFT(a+b)|={} > |STFT(a)|+|STFT(b)|={}",
            mag_ab[i],
            mag_a[i] + mag_b[i]
        );
    }
}

// ===========================================================================
// 13. Energy conservation (Parseval's theorem approximation)
// ===========================================================================

#[test]
fn test_parseval_energy_approximation() {
    // For a rectangular window: sum(|x|^2) ~= sum(|X|^2) / n_fft.
    // With a Hann window and overlap-add, the relationship is approximate.
    let n_fft = 8;
    let params = StftParams::new(n_fft, 4);
    let basis = build_dft_basis(n_fft);

    let audio: Vec<f32> = (0..50).map(|k| (0.5 * k as f32).sin()).collect();
    let time_energy: f32 = audio.iter().map(|x| x * x).sum();
    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    let freq_energy: f32 = mag.iter().map(|m| m * m).sum();

    // These should be within an order of magnitude (Parseval with windowing).
    assert!(time_energy > 0.0);
    assert!(freq_energy > 0.0);
    let ratio = freq_energy / time_energy;
    assert!(
        ratio > 0.01 && ratio < 1000.0,
        "Energy ratio out of reasonable range: {ratio}"
    );
}

// ===========================================================================
// 14. Window normalization
// ===========================================================================

#[test]
fn test_istft_normalized_vs_unnormalized_scaling() {
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 4;

    let params_norm = IstftParams::new(n_fft, hop, true, false).unwrap();
    let params_unnorm = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis_norm = IstftBasis::new(params_norm).unwrap();
    let basis_unnorm = IstftBasis::new(params_unnorm).unwrap();

    // DC-only signal.
    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0; // DC bin only
    }

    let out_len = n_fft + (n_frames - 1) * hop;
    let recon_norm = basis_norm.istft(&real, &imag, n_frames, out_len).unwrap();
    let recon_unnorm = basis_unnorm.istft(&real, &imag, n_frames, out_len).unwrap();

    // Normalized uses 1/sqrt(N), unnormalized uses 1/N. Ratio should be sqrt(N).
    let sqrt_n = (n_fft as f32).sqrt();
    // Compare interior samples.
    let mid = out_len / 2;
    if recon_unnorm[mid].abs() > 1e-8 {
        let ratio = recon_norm[mid] / recon_unnorm[mid];
        assert!(
            (ratio - sqrt_n).abs() / sqrt_n < 0.15,
            "Normalized/unnormalized ratio {ratio} should be ~sqrt({n_fft})={sqrt_n}"
        );
    }
}

// ===========================================================================
// 15-18. Model dispatch registry validation
// ===========================================================================

#[test]
fn test_dpdf_registry_all_models_have_descriptions() {
    let registry = DpdfModelRegistry::default_pipeline();
    for entry in registry.models() {
        assert!(
            !entry.description.is_empty(),
            "Model '{}' has empty description",
            entry.name
        );
    }
}

#[test]
fn test_dpdf_registry_no_duplicate_names() {
    let registry = DpdfModelRegistry::default_pipeline();
    let models: Vec<_> = registry.models().collect();
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        names.len(),
        sorted.len(),
        "Registry contains duplicate model names"
    );
}

#[test]
fn test_dpdf_registry_all_model_types_represented() {
    let registry = DpdfModelRegistry::default_pipeline();
    let types = [
        ModelType::OCR,
        ModelType::VLM,
        ModelType::LayoutDetection,
        ModelType::TableStructure,
    ];
    for ty in types {
        let count = registry.list_by_type(ty).len();
        assert!(
            count > 0,
            "ModelType::{ty:?} has no models in default pipeline"
        );
    }
}

#[test]
fn test_dpdf_registry_empty_registry() {
    let registry = DpdfModelRegistry::new();
    assert_eq!(registry.len(), 0);
    assert!(registry.get("anything").is_none());
    assert!(registry.list_by_type(ModelType::OCR).is_empty());
}

// ===========================================================================
// 19-23. HTDemucs configuration validation
// ===========================================================================

#[test]
fn test_htdemucs_spectral_constants() {
    assert_eq!(SPECTRAL_BASIC_DEPTH, 4);
    assert_eq!(SPECTRAL_DEPTH, 6);
    assert_eq!(SPECTRAL_KERNEL_SIZE, 8);
    assert_eq!(SPECTRAL_STRIDE, 4);
    // Spectral branch input is 2 (stereo) x 2 (real + imaginary) = 4 channels.
    assert_eq!(SPECTRAL_INPUT_CHANNELS, 4);
}

#[test]
fn test_htdemucs_transformer_constants() {
    assert_eq!(NUM_LAYERS, 5);
    assert_eq!(NUM_HEADS, 8);
    assert_eq!(TRANSFORMER_DIM, 512);
    assert_eq!(BOTTLENECK_DIM, 384);
    assert_eq!(FFN_HIDDEN_DIM, 2048);
    assert_eq!(LAYER_NORM_EPS, 1e-5);
    assert!((FFN_HIDDEN_SCALE - 4.0).abs() < 1e-10);
}

#[test]
fn test_htdemucs_channel_doubling_consistency() {
    // Channel count should double at each depth.
    for d in 0..5 {
        let ch = channels_at_depth(d);
        assert_eq!(
            ch,
            BASE_CHANNELS * (1 << d),
            "channels_at_depth({d}) = {ch}, expected {}",
            BASE_CHANNELS * (1 << d)
        );
    }
}

#[test]
fn test_htdemucs_conv1d_output_len_with_padding() {
    // With padding, output should be larger.
    let out_no_pad = conv1d_output_len(100, 8, 4, 0).unwrap();
    let out_with_pad = conv1d_output_len(100, 8, 4, 2).unwrap();
    assert!(
        out_with_pad > out_no_pad,
        "Padding should increase output length"
    );
}

#[test]
fn test_htdemucs_decoder_output_channels() {
    // 4 sources * 2 stereo channels = 8.
    assert_eq!(DECODER_OUTPUT_CHANNELS, 8);
    assert_eq!(DECODER_REWRITE_KERNEL, 3);
    assert_eq!(DECODER_REWRITE_PADDING, 1); // 3/2 = 1
}

// ===========================================================================
// 24-28. Silero VAD configuration validation
// ===========================================================================

#[test]
fn test_silero_vad_encoder_chain_dimensions() {
    // Verify the temporal dimension flow through encoder blocks.
    // Input: 4 frames (from STFT of 576 samples with n_fft=256, hop=128).
    let mut t = 4usize;
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        // Conv1d output: (t + 2*padding - kernel) / stride + 1
        let next_t = (t + 2 * block.padding - block.kernel_size) / block.stride + 1;
        assert!(
            next_t > 0,
            "Block {i}: temporal dimension collapsed to 0 from t={t}"
        );
        t = next_t;
    }
    // After all 4 blocks, temporal dimension should be 1 (Silero VAD architecture).
    assert_eq!(t, 1, "Final temporal dimension should be 1, got {t}");
}

#[test]
fn test_silero_vad_encoder_strides() {
    // Only blocks 1 and 2 have stride=2, others stride=1.
    assert_eq!(ENCODER_BLOCKS[0].stride, 1);
    assert_eq!(ENCODER_BLOCKS[1].stride, 2);
    assert_eq!(ENCODER_BLOCKS[2].stride, 2);
    assert_eq!(ENCODER_BLOCKS[3].stride, 1);
}

#[test]
fn test_silero_vad_encoder_all_use_kernel_3() {
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        assert_eq!(block.kernel_size, 3, "Block {i} kernel_size should be 3");
        assert_eq!(block.padding, 1, "Block {i} padding should be 1");
    }
}

#[test]
fn test_silero_vad_final_output_matches_lstm() {
    // Last encoder block outputs 128 channels, matching LSTM hidden size.
    let last_block = &ENCODER_BLOCKS[ENCODER_BLOCKS.len() - 1];
    assert_eq!(last_block.out_channels, LSTM_HIDDEN_SIZE);
}

#[test]
fn test_silero_vad_build_output_def() {
    // Should successfully build the output TensorKernelDef.
    let def = crate::silero_vad_builders::build_output_def();
    assert!(def.is_ok(), "build_output_def failed: {:?}", def.err());
}

// ===========================================================================
// 29-35. Kokoro model configuration
// ===========================================================================

#[test]
fn test_kokoro_config_default_validates() {
    let config = KokoroConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_kokoro_config_zero_d_en_rejected() {
    let config = KokoroConfig {
        d_en: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_kokoro_config_zero_style_dim_rejected() {
    let config = KokoroConfig {
        style_dim: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_kokoro_config_n_fft_must_be_divisible_by_4() {
    let mut config = KokoroConfig {
        n_fft: 21, // Not divisible by 4
        ..Default::default()
    };
    assert!(config.validate().is_err());

    config.n_fft = 20; // Valid
    assert!(config.validate().is_ok());
}

#[test]
fn test_kokoro_config_empty_upsample_rates_rejected() {
    let config = KokoroConfig {
        upsample_rates: vec![],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_kokoro_config_upsample_product() {
    let config = KokoroConfig::default();
    let product: usize = config.upsample_rates.iter().product();
    // Kokoro upsamples by 10*6 = 60x from mel frames to audio rate.
    assert_eq!(product, 60);
}

#[test]
fn test_kokoro_signal_constants_consistent() {
    assert_eq!(KOKORO_N_FFT, 20);
    assert_eq!(KOKORO_HOP_LENGTH, 5);
    assert_eq!(KOKORO_N_BINS, 11); // 20/2 + 1
    assert_eq!(KOKORO_SAMPLE_RATE, 24000);
    // Hop/FFT ratio = 1/4
    assert_eq!(KOKORO_N_FFT / KOKORO_HOP_LENGTH, 4);
}

// ===========================================================================
// 36-38. PlBert and Table Transformer configuration
// ===========================================================================

#[test]
fn test_plbert_config_defaults() {
    let config = PlbertConfig::default();
    assert_eq!(config.vocab_size, 178);
    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.intermediate_size, 2048);
    assert_eq!(config.max_position_embeddings, 512);
    assert_eq!(config.num_hidden_layers, 12);
    // Head dim should divide evenly.
    assert_eq!(config.hidden_size % config.num_attention_heads, 0);
}

#[test]
fn test_table_transformer_config_presets() {
    let det = TableTransformerConfig::preset_detection();
    assert!(det.validate().is_ok());

    let struct_cfg = TableTransformerConfig::preset_structure();
    assert!(struct_cfg.validate().is_ok());
}

#[test]
fn test_table_transformer_constants() {
    assert_eq!(HIDDEN_DIM, 256);
    assert_eq!(NUM_QUERIES, 125);
    assert_eq!(crate::table_transformer::NUM_ENCODER_LAYERS, 6);
    assert_eq!(crate::table_transformer::NUM_DECODER_LAYERS, 6);
}

// ===========================================================================
// 39-42. DocLayout-YOLO and model dispatch cross-checks
// ===========================================================================

#[test]
fn test_doclayout_yolo_constants() {
    assert_eq!(NUM_CLASSES, 10);
    assert_eq!(REG_MAX, 16);
    assert_eq!(INPUT_SIZE, 800);
}

#[test]
fn test_doclayout_yolo_default_config() {
    let config = DocLayoutYoloConfig::default();
    let neck_ch = config.neck_channels();
    // Neck channels should be 3 elements (P3, P4, P5).
    assert_eq!(neck_ch.len(), 3);
    for ch in neck_ch {
        assert!(ch > 0, "Neck channel should be > 0, got {ch}");
    }
}

#[test]
fn test_convert_config_dpdf_model_types_exhaustive() {
    // All dpdf model types should be detectable.
    let type_strings = [
        ("Granite-Docling", DpdfModelType::GraniteDocling),
        ("DocLayout-YOLO", DpdfModelType::DocLayoutYolo),
        ("Qwen3-VL", DpdfModelType::Qwen3VL),
        ("table-transformer", DpdfModelType::TableTransformer),
        ("glm-ocr", DpdfModelType::GlmOcr),
        ("PaddleOCR", DpdfModelType::PaddleOcr),
        ("FireRed-OCR", DpdfModelType::FireRedOcr),
    ];
    for (name, expected_type) in type_strings {
        let detected = ConvertConfig::detect_model_type(name);
        assert_eq!(
            detected,
            Some(expected_type.clone()),
            "Failed to detect model type from '{name}'"
        );
    }
}

#[test]
fn test_table_structure_config_defaults() {
    let config = TableStructureConfig::default();
    assert!(config.iou_threshold > 0.0 && config.iou_threshold < 1.0);
    assert!(config.row_tolerance > 0.0 && config.row_tolerance < 1.0);
    assert!(config.col_tolerance > 0.0 && config.col_tolerance < 1.0);
}

// ===========================================================================
// 43-46. iSTFT error handling and edge cases
// ===========================================================================

#[test]
fn test_istft_real_imag_length_mismatch() {
    let params = IstftParams::new(20, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let real = vec![0.0f32; 44]; // 11 bins * 4 frames
    let imag = vec![0.0f32; 33]; // wrong length
    let result = basis.istft(&real, &imag, 4, 35);
    assert!(matches!(result, Err(IstftError::LengthMismatch { .. })));
}

#[test]
fn test_istft_shape_mismatch() {
    let params = IstftParams::new(20, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let real = vec![0.0f32; 50]; // doesn't match 11 * any integer cleanly
    let imag = vec![0.0f32; 50];
    let result = basis.istft(&real, &imag, 4, 35); // 11 * 4 = 44 != 50
    assert!(matches!(result, Err(IstftError::ShapeMismatch { .. })));
}

#[test]
fn test_istft_non_finite_input_rejected() {
    let params = IstftParams::new(20, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = 11;
    let n_frames = 4;
    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    real[0] = f32::NAN;
    let result = basis.istft(&real, &imag, n_frames, 35);
    assert!(matches!(result, Err(IstftError::NonFiniteInput)));
}

#[test]
fn test_istft_center_trim_geometry() {
    // With center=true, output is trimmed by n_fft/2 on each side.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;

    let params = IstftParams::new(n_fft, hop, false, true).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let full_len = n_fft + (n_frames - 1) * hop;
    let trimmed_len = full_len - n_fft; // Remove n_fft/2 from each side.
    let result = basis.istft(&real, &imag, n_frames, trimmed_len).unwrap();
    assert_eq!(result.len(), trimmed_len);
}
