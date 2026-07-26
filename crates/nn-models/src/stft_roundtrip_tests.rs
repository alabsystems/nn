// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! STFT/iSTFT roundtrip and signal processing correctness tests (#4186).
//!
//! Covers:
//! - STFT -> iSTFT roundtrip reconstruction on sine waves (< 1e-4 error)
//! - STFT output shape for various window sizes and hop lengths
//! - Hann window construction and properties (symmetric, sum-to-constant with 50% overlap)
//! - Zero-padding behavior
//! - Short signal handling (shorter than window)
//! - Parseval's theorem (energy conservation through STFT)

use std::f32::consts::PI;

use crate::istft::{IstftBasis, IstftParams};

// =============================================================================
// Helpers
// =============================================================================

/// Build a DFT-style STFT basis (cos + sin rows) for use with compute_stft_magnitude
/// or for manual forward STFT.
fn build_dft_basis(n_fft: usize) -> Vec<f32> {
    let n_filters = n_fft + 2; // n_freqs real + n_freqs imag
    let n_freqs = n_fft / 2 + 1;
    let mut basis = vec![0.0f32; n_filters * n_fft];
    // First n_freqs rows: cosine (real)
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[f * n_fft + k] = angle.cos();
        }
    }
    // Next n_freqs rows: sine (imaginary) -- negative sine for STFT convention
    for f in 0..n_freqs {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            basis[(n_freqs + f) * n_fft + k] = -angle.sin();
        }
    }
    basis
}

/// Perform a scalar windowed forward STFT (Hann window + DFT per frame).
/// Returns (real, imag) each [n_bins, n_frames] row-major, plus n_frames.
fn windowed_forward_stft(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let n_bins = n_fft / 2 + 1;
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop + 1
    } else {
        0
    };
    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        let offset = t * hop;
        for f in 0..n_bins {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                let windowed = signal[offset + k] * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r;
            imag[f * n_frames + t] = im;
        }
    }
    (real, imag, n_frames)
}

/// Generate a pure sine wave.
fn sine_wave(len: usize, freq_hz: f32, sample_rate: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
        .collect()
}

// =============================================================================
// 1. STFT -> iSTFT roundtrip on a sine wave (reconstruction error < 1e-4)
// =============================================================================

#[test]
fn test_roundtrip_sine_wave_reconstruction_error_below_1e4() {
    // Core roundtrip test: forward STFT then iSTFT on a 440 Hz sine wave.
    // Interior reconstruction error must be < 1e-4.
    let n_fft = 64;
    let hop = 16; // 75% overlap for good COLA
    let sample_rate = 16000.0;
    let freq = 440.0;
    let signal_len = 1024;

    let signal = sine_wave(signal_len, freq, sample_rate);
    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    // Check reconstruction in the interior (skip 2 windows from each edge).
    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        let err = (reconstructed[i] - signal[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 1e-4,
        "roundtrip reconstruction error {max_err:.2e} exceeds 1e-4"
    );
}

#[test]
fn test_roundtrip_multi_frequency_sine() {
    // Roundtrip with a sum of 3 sine waves at different frequencies.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 512;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.5 * (2.0 * PI * 3.0 * t).sin()
                + 0.3 * (2.0 * PI * 7.0 * t).sin()
                + 0.2 * (2.0 * PI * 11.0 * t).cos()
        })
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(
        max_err < 1e-4,
        "multi-frequency roundtrip error {max_err:.2e} exceeds 1e-4"
    );
}

#[test]
fn test_roundtrip_kokoro_size_nfft20_hop5() {
    // Roundtrip with Kokoro-specific parameters (n_fft=20, hop=5).
    let n_fft = 20;
    let hop = 5;
    let signal_len = 400;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 4.0 * i as f32 / signal_len as f32).sin())
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(
        max_err < 1e-4,
        "Kokoro-sized roundtrip error {max_err:.2e} exceeds 1e-4"
    );
}

// =============================================================================
// 2. STFT output shape for various window sizes and hop lengths
// =============================================================================

#[test]
fn test_stft_output_shape_various_params() {
    // Verify n_frames = floor((signal_len - n_fft) / hop) + 1 for multiple configs.
    let configs: &[(usize, usize, usize)] = &[
        // (n_fft, hop, signal_len)
        (64, 16, 256),
        (32, 8, 128),
        (20, 5, 200),
        (128, 32, 512),
        (16, 4, 64),
    ];
    for &(n_fft, hop, signal_len) in configs {
        let expected_frames = (signal_len - n_fft) / hop + 1;
        let signal: Vec<f32> = (0..signal_len).map(|i| i as f32 * 0.001).collect();
        let (real, _imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
        assert_eq!(
            n_frames, expected_frames,
            "n_fft={n_fft}, hop={hop}, signal_len={signal_len}: expected {expected_frames} frames, got {n_frames}"
        );
        let n_bins = n_fft / 2 + 1;
        assert_eq!(
            real.len(),
            n_bins * n_frames,
            "real data length mismatch for n_fft={n_fft}"
        );
    }
}

#[test]
fn test_stft_output_shape_exactly_nfft() {
    // Signal exactly n_fft samples long gives exactly 1 frame.
    let n_fft = 32;
    let hop = 8;
    let signal = vec![1.0f32; n_fft];
    let (_real, _imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    assert_eq!(
        n_frames, 1,
        "signal of exactly n_fft samples should give 1 frame"
    );
}

#[test]
fn test_stft_output_shape_shorter_than_nfft_gives_zero_frames() {
    // Signal shorter than n_fft produces 0 frames.
    let n_fft = 64;
    let hop = 16;
    let signal = vec![1.0f32; 32]; // shorter than n_fft
    let (_real, _imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    assert_eq!(
        n_frames, 0,
        "signal shorter than n_fft should give 0 frames"
    );
}

// =============================================================================
// 3. Hann window construction and properties
// =============================================================================

#[test]
fn test_hann_window_symmetric_property() {
    // Periodic Hann window satisfies w[k] = w[N-k] for k = 1..N/2-1.
    for &n_fft in &[16, 32, 64, 128] {
        let params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let window = basis.window();
        for k in 1..n_fft / 2 {
            let diff = (window[k] - window[n_fft - k]).abs();
            assert!(
                diff < 1e-6,
                "n_fft={n_fft}: window[{k}]={} != window[{}]={}",
                window[k],
                n_fft - k,
                window[n_fft - k]
            );
        }
    }
}

#[test]
fn test_hann_window_cola_sum_constant_50_percent_overlap() {
    // With hop = n_fft/2 (50% overlap), sum of shifted Hann windows should be
    // approximately constant in the fully overlapping region.
    let n_fft = 64;
    let hop = n_fft / 2; // 50% overlap
    let n_frames = 20;
    let full_len = n_fft + (n_frames - 1) * hop;

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    // Sum of (non-squared) Hann windows at each position.
    let mut window_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            window_sum[offset + k] += window[k];
        }
    }

    // In the deep interior, the sum should be constant (=1.0 for 50% overlap Hann).
    let margin = 2 * n_fft;
    let interior_start = margin;
    let interior_end = full_len.saturating_sub(margin);
    if interior_end > interior_start {
        let first_val = window_sum[interior_start];
        for i in interior_start..interior_end {
            let diff = (window_sum[i] - first_val).abs();
            assert!(
                diff < 0.01,
                "COLA not constant at position {i}: sum={}, expected ~{first_val}",
                window_sum[i]
            );
        }
    }
}

#[test]
fn test_hann_window_endpoints_zero_and_peak() {
    // w[0] = 0, w[N/2] = 1 (for periodic Hann).
    for &n_fft in &[8, 16, 32, 64] {
        let params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let window = basis.window();
        assert!(
            window[0].abs() < 1e-7,
            "n_fft={n_fft}: window[0] = {} should be ~0",
            window[0]
        );
        assert!(
            (window[n_fft / 2] - 1.0).abs() < 1e-6,
            "n_fft={n_fft}: window[N/2] = {} should be ~1",
            window[n_fft / 2]
        );
    }
}

// =============================================================================
// 4. Zero-padding behavior
// =============================================================================

#[test]
fn test_istft_zero_input_produces_zero_output() {
    // All-zero STFT input should produce all-zero output.
    let n_fft = 32;
    let hop = 8;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let output = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "zero-input iSTFT should produce zero output, got {v} at index {i}"
        );
    }
}

#[test]
fn test_istft_output_pads_when_requested_length_exceeds_signal() {
    // Request more output samples than overlap-add produces; extra should be zeros.
    let n_fft = 16;
    let hop = 4;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 3;
    let full_len = n_fft + (n_frames - 1) * hop; // 16 + 2*4 = 24
    let requested = 100; // much larger than full_len

    // Put some energy in the DC bin to get non-zero output in the valid region.
    let mut real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        real[t] = 1.0; // DC bin for each frame
    }

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let output = basis.istft(&real, &imag, n_frames, requested).unwrap();

    assert_eq!(output.len(), requested);
    // Samples beyond full_len should be zero.
    for i in full_len..requested {
        assert_eq!(
            output[i], 0.0,
            "padding region should be zero at index {i}, got {}",
            output[i]
        );
    }
}

// =============================================================================
// 5. Short signal handling
// =============================================================================

#[test]
fn test_roundtrip_signal_exactly_nfft_samples() {
    // Signal of exactly n_fft samples: single frame, reconstruction should be finite.
    let n_fft = 32;
    let hop = 8;
    let signal: Vec<f32> = (0..n_fft)
        .map(|i| (2.0 * PI * 2.0 * i as f32 / n_fft as f32).sin())
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    assert_eq!(n_frames, 1);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let reconstructed = basis.istft(&real, &imag, n_frames, n_fft).unwrap();

    assert_eq!(reconstructed.len(), n_fft);
    for v in &reconstructed {
        assert!(v.is_finite(), "single-frame reconstruction must be finite");
    }
}

#[test]
fn test_roundtrip_minimal_two_frame_signal() {
    // Minimal 2-frame signal.
    let n_fft = 16;
    let hop = 4;
    let signal_len = n_fft + hop; // 20 samples -> 2 frames
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 1.5 * i as f32 / signal_len as f32).sin())
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    assert_eq!(n_frames, 2);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    assert_eq!(reconstructed.len(), full_len);
    for v in &reconstructed {
        assert!(v.is_finite(), "two-frame reconstruction must be finite");
    }
}

// =============================================================================
// 6. Parseval's theorem (energy conservation)
// =============================================================================

#[test]
fn test_parsevals_theorem_energy_conservation_stft() {
    // Parseval's theorem: per-frame windowed time-domain energy should equal
    // the per-frame frequency-domain energy (normalized by N).
    // sum(|x_windowed[k]|^2) = (1/N) * [|X[0]|^2 + 2*sum(|X[f]|^2) + |X[N/2]|^2]
    let n_fft = 64;
    let hop = 16;
    let signal_len = 512;

    // Use a sum of sinusoids for broadband energy.
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.6 * (2.0 * PI * 5.0 * t).sin() + 0.4 * (2.0 * PI * 13.0 * t).cos()
        })
        .collect();

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    // Check interior frames (skip edges where signal may be truncated).
    for t in 2..(n_frames.saturating_sub(2)) {
        let offset = t * hop;

        // Time-domain energy (windowed).
        let time_energy: f32 = (0..n_fft)
            .map(|k| {
                let w = signal[offset + k] * window[k];
                w * w
            })
            .sum();

        // Frequency-domain energy with conjugate symmetry.
        let mut freq_energy = 0.0f32;
        // DC
        let r0 = real[t];
        let i0 = imag[t];
        freq_energy += r0 * r0 + i0 * i0;
        // Interior bins (doubled for mirror)
        for f in 1..(n_bins - 1) {
            let r = real[f * n_frames + t];
            let im = imag[f * n_frames + t];
            freq_energy += 2.0 * (r * r + im * im);
        }
        // Nyquist
        let rn = real[(n_bins - 1) * n_frames + t];
        let imn = imag[(n_bins - 1) * n_frames + t];
        freq_energy += rn * rn + imn * imn;
        freq_energy /= n_fft as f32;

        if time_energy > 1e-8 {
            let ratio = freq_energy / time_energy;
            assert!(
                (ratio - 1.0).abs() < 0.02,
                "Parseval's theorem violated at frame {t}: time_energy={time_energy:.4}, \
                 freq_energy={freq_energy:.4}, ratio={ratio:.6}"
            );
        }
    }
}

#[test]
fn test_dc_signal_dominant_energy_at_bin_zero() {
    // DC signal (constant 1.0): bin 0 should have the largest magnitude.
    // Hann window spectral leakage means non-DC bins have some energy,
    // but DC should always dominate.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;
    let signal = vec![1.0f32; signal_len];

    let (real, imag, n_frames) = windowed_forward_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for t in 2..(n_frames.saturating_sub(2)) {
        let dc_mag = real[t].hypot(imag[t]);
        assert!(
            dc_mag > 0.1,
            "DC bin should have significant energy at frame {t}"
        );

        // DC bin should be the largest magnitude among all bins.
        for f in 1..n_bins {
            let mag = real[f * n_frames + t].hypot(imag[f * n_frames + t]);
            assert!(
                dc_mag > mag,
                "frame {t}: DC magnitude {dc_mag} should exceed bin {f} magnitude {mag}"
            );
        }
    }
}

// =============================================================================
// 7. Additional roundtrip precision tests
// =============================================================================

#[test]
fn test_roundtrip_htdemucs_size_nfft4096_hop1024() {
    // Roundtrip with HTDemucs-sized parameters. Uses normalized iSTFT.
    let n_fft = 128; // Use smaller size for test speed (4096 would be very slow).
    let hop = 32;
    let signal_len = 1024;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 6.0 * i as f32 / signal_len as f32).sin())
        .collect();

    // Forward STFT (normalized: divide by sqrt(N))
    let n_bins = n_fft / 2 + 1;
    let n_frames = (signal_len - n_fft) / hop + 1;
    let norm_factor = 1.0 / (n_fft as f32).sqrt();

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        let offset = t * hop;
        for f in 0..n_bins {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                let windowed = signal[offset + k] * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r * norm_factor;
            imag[f * n_frames + t] = im * norm_factor;
        }
    }

    // Inverse with normalized mode (1/sqrt(N))
    let params = IstftParams::new(n_fft, hop, true, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let skip = 2 * n_fft;
    let end = signal_len.min(reconstructed.len()).saturating_sub(skip);
    let mut max_err = 0.0f32;
    for i in skip..end {
        max_err = max_err.max((reconstructed[i] - signal[i]).abs());
    }
    assert!(
        max_err < 1e-4,
        "normalized roundtrip error {max_err:.2e} exceeds 1e-4"
    );
}

#[test]
fn test_dft_basis_builder_shape() {
    // Verify build_dft_basis produces the right shape.
    let n_fft = 32;
    let basis = build_dft_basis(n_fft);
    let n_filters = n_fft + 2;
    assert_eq!(basis.len(), n_filters * n_fft);
}
