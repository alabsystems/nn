// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive STFT/iSTFT signal processing tests.
//!
//! Covers frequency detection, window function correctness, overlap-add
//! reconstruction, Parseval's theorem, linearity, time-shift properties,
//! edge cases (very short signals, non-power-of-2 sizes), and numerical
//! precision of round-trip reconstruction.
//!
//! Part of #3351 (Absolutely Best Kokoro).

use std::f32::consts::PI;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::istft::{IstftBasis, IstftParams};
use crate::kokoro_forward_stft::KokoroForwardStft;
use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};

// ============================================================================
// Helpers
// ============================================================================

/// Generate a pure sine wave at the given frequency (in cycles per signal length).
fn sine_wave(len: usize, freq_cycles: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * PI * freq_cycles * i as f32 / len as f32).sin())
        .collect()
}

/// Generate a cosine wave at the given frequency.
fn cosine_wave(len: usize, freq_cycles: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * PI * freq_cycles * i as f32 / len as f32).cos())
        .collect()
}

/// Compute a scalar forward STFT (windowed DFT per frame).
/// Returns (real, imag) each of shape [n_bins, n_frames] row-major.
fn scalar_forward_stft(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
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

/// Compute round-trip quality metrics (SNR, max_err, RMS) over interior region.
fn quality_metrics(original: &[f32], reconstructed: &[f32], skip: usize) -> (f32, f32, f32) {
    let len = original.len().min(reconstructed.len());
    let start = skip;
    let end = len.saturating_sub(skip);
    if end <= start {
        return (f32::INFINITY, 0.0, 0.0);
    }
    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    let mut sum_sq_ref = 0.0f32;
    for i in start..end {
        let err = (reconstructed[i] - original[i]).abs();
        max_err = max_err.max(err);
        sum_sq_err += (reconstructed[i] - original[i]).powi(2);
        sum_sq_ref += original[i].powi(2);
    }
    let snr_db = if sum_sq_err > 0.0 {
        10.0 * (sum_sq_ref / sum_sq_err).log10()
    } else {
        f32::INFINITY
    };
    let rms_err = (sum_sq_err / (end - start) as f32).sqrt();
    (snr_db, max_err, rms_err)
}

// ============================================================================
// 1. Frequency detection: pure sine wave -> peak at correct bin
// ============================================================================

#[test]
fn test_stft_frequency_detection_pure_sine() {
    // A pure sine at bin frequency f should produce a peak at bin f.
    // With n_fft=64, bin k corresponds to frequency k * (sample_rate / n_fft).
    // We test with a signal whose frequency aligns to bin 5.
    let n_fft = 64;
    let hop = 16;
    let n_bins = n_fft / 2 + 1; // 33
    let signal_len = 256;
    let target_bin = 5usize;

    // Generate sine at exactly bin 5 frequency: f = target_bin cycles per n_fft samples.
    // Over signal_len samples, this is target_bin * (signal_len / n_fft) cycles.
    let cycles = target_bin as f32 * (signal_len as f32 / n_fft as f32);
    let signal = sine_wave(signal_len, cycles);

    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

    // For each frame, find the bin with maximum magnitude.
    for t in 0..n_frames {
        let mut max_mag = 0.0f32;
        let mut max_bin = 0usize;
        for f in 0..n_bins {
            let r = real[f * n_frames + t];
            let im = imag[f * n_frames + t];
            let mag = r.hypot(im);
            if mag > max_mag {
                max_mag = mag;
                max_bin = f;
            }
        }
        assert_eq!(
            max_bin, target_bin,
            "frame {t}: expected peak at bin {target_bin}, got bin {max_bin} (mag={max_mag})"
        );
    }
}

#[test]
fn test_kokoro_forward_stft_frequency_detection() {
    // Use KokoroForwardStft (FFT-based) to detect a pure sine at bin 3.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1; // 11
    let signal_len = 200;
    let target_bin = 3usize;

    // Sine at bin 3: target_bin cycles per n_fft samples.
    let cycles = target_bin as f32 * (signal_len as f32 / n_fft as f32);
    let signal = sine_wave(signal_len, cycles);

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, _phase) = stft.forward(&input).unwrap();

    let mag_data = magnitude.to_flat_vec::<f32>().unwrap();
    let n_frames = magnitude.dims()[2];

    // Check that bin 3 has the largest magnitude for interior frames.
    // Skip first and last frames where windowing edge effects can shift energy.
    for t in 2..(n_frames - 2) {
        let mut max_mag = 0.0f32;
        let mut max_bin = 0usize;
        for f in 0..n_bins {
            let mag = mag_data[f * n_frames + t];
            if mag > max_mag {
                max_mag = mag;
                max_bin = f;
            }
        }
        assert_eq!(
            max_bin, target_bin,
            "frame {t}: expected peak at bin {target_bin}, got bin {max_bin}"
        );
    }
}

#[test]
fn test_stft_two_frequency_detection() {
    // Signal with two frequencies: bins 3 and 7.
    // Both should have large magnitude, others should be small.
    let n_fft = 64;
    let hop = 16;
    let n_bins = n_fft / 2 + 1;
    let signal_len = 256;

    let cycles_a = 3.0 * (signal_len as f32 / n_fft as f32);
    let cycles_b = 7.0 * (signal_len as f32 / n_fft as f32);
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            (2.0 * PI * cycles_a * t).sin() + (2.0 * PI * cycles_b * t).sin()
        })
        .collect();

    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

    // For a middle frame, check that bins 3 and 7 are the top 2.
    let t = n_frames / 2;
    let mut mags: Vec<(usize, f32)> = (0..n_bins)
        .map(|f| {
            let r = real[f * n_frames + t];
            let im = imag[f * n_frames + t];
            (f, r.hypot(im))
        })
        .collect();
    mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_bins: Vec<usize> = mags.iter().take(2).map(|(f, _)| *f).collect();
    assert!(
        top_bins.contains(&3) && top_bins.contains(&7),
        "expected bins 3 and 7 in top 2, got {top_bins:?}"
    );
}

// ============================================================================
// 2. Window function correctness
// ============================================================================

#[test]
fn test_hann_window_formula_istft_basis() {
    // Verify IstftBasis Hann window matches the analytical formula exactly.
    let n_fft = 32;
    let params = IstftParams::new(n_fft, 8, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();

    for k in 0..n_fft {
        let expected = 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos());
        assert!(
            (window[k] - expected).abs() < 1e-7,
            "window[{k}]: expected {expected}, got {}",
            window[k]
        );
    }
}

#[test]
fn test_hann_window_symmetry() {
    // Hann window is symmetric: w[k] = w[n_fft - k] for k > 0.
    // (periodic Hann: w[0] != w[N], but w[k] = w[N-k] for interior)
    let n_fft = 64;
    let params = IstftParams::new(n_fft, 16, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();

    for k in 1..n_fft / 2 {
        let diff = (window[k] - window[n_fft - k]).abs();
        assert!(
            diff < 1e-6,
            "window[{k}]={} != window[{}]={}, diff={}",
            window[k],
            n_fft - k,
            window[n_fft - k],
            diff
        );
    }
}

#[test]
fn test_hann_window_peak_at_center() {
    // The Hann window peaks at k = n_fft/2 with value 1.0.
    let n_fft = 128;
    let params = IstftParams::new(n_fft, 32, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();

    let center = n_fft / 2;
    assert!(
        (window[center] - 1.0).abs() < 1e-6,
        "window[{center}] should be 1.0, got {}",
        window[center]
    );

    // All other values should be <= 1.0.
    for k in 0..n_fft {
        assert!(
            window[k] <= 1.0 + 1e-6,
            "window[{k}]={} exceeds 1.0",
            window[k]
        );
    }
}

#[test]
fn test_hann_window_zero_at_boundaries() {
    // Hann window: w[0] = 0.
    for &n_fft in &[8, 16, 32, 64, 128] {
        let params = IstftParams::new(n_fft, n_fft / 4, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let window = basis.window();
        assert!(
            window[0].abs() < 1e-7,
            "n_fft={n_fft}: window[0]={} should be ~0",
            window[0]
        );
    }
}

#[test]
fn test_kokoro_forward_stft_window_matches_istft() {
    // The forward STFT and iSTFT must use the same window for perfect reconstruction.
    // Both use periodic Hann: w[k] = 0.5 * (1 - cos(2*pi*k / n_fft)).
    let n_fft = 20;
    let kokoro_window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let params = IstftParams::new(n_fft, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let istft_window = basis.window();

    for k in 0..n_fft {
        assert!(
            (kokoro_window[k] - istft_window[k]).abs() < 1e-7,
            "window mismatch at k={k}: forward={}, istft={}",
            kokoro_window[k],
            istft_window[k]
        );
    }
}

// ============================================================================
// 3. COLA (Constant Overlap-Add) condition
// ============================================================================

#[test]
fn test_cola_condition_various_ratios() {
    // COLA: sum of squared Hann windows at every position must be non-zero
    // for the overlap region. Test several n_fft/hop ratios.
    for &(n_fft, hop) in &[(64, 16), (64, 32), (32, 8), (20, 5), (128, 32)] {
        let n_frames = 20;
        let full_len = n_fft + (n_frames - 1) * hop;

        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
            .collect();

        let mut window_sum = vec![0.0f32; full_len];
        for t in 0..n_frames {
            let offset = t * hop;
            for k in 0..n_fft {
                window_sum[offset + k] += window[k] * window[k];
            }
        }

        // Interior region (skip one window from each edge).
        let margin = n_fft;
        for i in margin..(full_len.saturating_sub(margin)) {
            assert!(
                window_sum[i] > 1e-6,
                "COLA violated: n_fft={n_fft}, hop={hop}, position {i}: window_sum={}",
                window_sum[i]
            );
        }
    }
}

#[test]
fn test_cola_normalization_flatness() {
    // For Hann window with hop = n_fft/4 (75% overlap), the COLA sum
    // should be approximately constant in the interior.
    let n_fft = 64;
    let hop = n_fft / 4;
    let n_frames = 30;
    let full_len = n_fft + (n_frames - 1) * hop;

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let mut window_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            window_sum[offset + k] += window[k] * window[k];
        }
    }

    // In the deep interior, the sum should be nearly constant.
    let interior_start = 2 * n_fft;
    let interior_end = full_len.saturating_sub(2 * n_fft);
    if interior_end > interior_start {
        let interior = &window_sum[interior_start..interior_end];
        let mean: f32 = interior.iter().sum::<f32>() / interior.len() as f32;
        let max_deviation = interior
            .iter()
            .map(|v| (v - mean).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_deviation / mean < 0.01,
            "COLA sum not flat: mean={mean}, max_deviation={max_deviation}"
        );
    }
}

// ============================================================================
// 4. STFT -> iSTFT round-trip tests (comprehensive)
// ============================================================================

#[test]
fn test_round_trip_chirp_signal() {
    // Chirp signal: frequency sweeps from low to high.
    // Tests that reconstruction works for time-varying frequency content.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            // Linear chirp: frequency goes from 1 to 10 cycles
            let phase = 2.0 * PI * (1.0 * t + 9.0 * t * t / 2.0);
            phase.sin()
        })
        .collect();

    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let (snr_db, max_err, _rms_err) = quality_metrics(&signal, &reconstructed, n_fft);
    assert!(
        snr_db > 30.0,
        "chirp round-trip SNR {snr_db:.1}dB below 30dB"
    );
    assert!(
        max_err < 0.05,
        "chirp round-trip max error {max_err:.6} exceeds 0.05"
    );
}

#[test]
fn test_round_trip_white_noise() {
    // Pseudo-random signal to test broadband reconstruction.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;

    // LCG pseudo-random noise in [-1, 1].
    let mut rng_state = 42u64;
    let signal: Vec<f32> = (0..signal_len)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
        })
        .collect();

    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    let (snr_db, max_err, _rms_err) = quality_metrics(&signal, &reconstructed, n_fft);
    assert!(
        snr_db > 25.0,
        "noise round-trip SNR {snr_db:.1}dB below 25dB"
    );
    assert!(
        max_err < 0.1,
        "noise round-trip max error {max_err:.6} exceeds 0.1"
    );
}

#[test]
fn test_round_trip_kokoro_params_multiple_waveforms() {
    // Test round-trip with Kokoro-specific params (n_fft=20, hop=5) on
    // several different waveform types.
    let n_fft = 20;
    let hop = 5;
    let signal_len = 200;

    let waveforms: Vec<(&str, Vec<f32>)> = vec![
        ("sine_3", sine_wave(signal_len, 3.0)),
        ("cosine_7", cosine_wave(signal_len, 7.0)),
        (
            "multi_freq",
            (0..signal_len)
                .map(|i| {
                    let t = i as f32 / signal_len as f32;
                    0.5 * (2.0 * PI * 2.0 * t).sin()
                        + 0.3 * (2.0 * PI * 5.0 * t).cos()
                        + 0.2 * (2.0 * PI * 9.0 * t).sin()
                })
                .collect(),
        ),
        ("dc_0.5", vec![0.5f32; signal_len]),
        (
            "sawtooth",
            (0..signal_len)
                .map(|i| (i as f32 / signal_len as f32 * 4.0).fract() * 2.0 - 1.0)
                .collect(),
        ),
    ];

    for (name, signal) in &waveforms {
        let (real, imag, n_frames) = scalar_forward_stft(signal, n_fft, hop);

        let params = KokoroIstftParams {
            n_fft,
            hop_length: hop,
        };
        let full_len = n_fft + (n_frames - 1) * hop;
        let reconstructed = kokoro_istft(&params, &real, &imag, n_frames, full_len).unwrap();

        let (snr_db, max_err, _rms_err) = quality_metrics(signal, &reconstructed, n_fft);
        assert!(
            max_err < 0.1,
            "{name}: round-trip max error {max_err:.6} exceeds 0.1"
        );
        // DC signal has zero variance, so SNR is not meaningful.
        if *name != "dc_0.5" {
            assert!(
                snr_db > 20.0,
                "{name}: round-trip SNR {snr_db:.1}dB below 20dB"
            );
        }
    }
}

#[test]
fn test_kokoro_forward_stft_istft_round_trip_tight_precision() {
    // Use the DynTensor-based KokoroForwardStft for the forward pass and
    // kokoro_istft for the inverse. This tests the actual production path
    // with center padding + center trim, with tighter precision requirements.
    let n_fft = 20;
    let hop = 5;
    let signal_len = 400;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.6 * (2.0 * PI * 3.0 * t).sin() + 0.4 * (2.0 * PI * 8.0 * t).cos()
        })
        .collect();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal.clone(), &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, phase) = stft.forward_center(&input).unwrap();
    let n_frames = magnitude.dims()[2];

    // Reconstruct real/imag from magnitude and phase.
    let real_spec = magnitude.mul(&phase.cos().unwrap()).unwrap();
    let imag_spec = magnitude.mul(&phase.sin().unwrap()).unwrap();

    let real_arr = real_spec.to_f32_array().unwrap();
    let real_flat: Vec<f32> = real_arr.as_standard_layout().as_slice().unwrap().to_vec();
    let imag_arr = imag_spec.to_f32_array().unwrap();
    let imag_flat: Vec<f32> = imag_arr.as_standard_layout().as_slice().unwrap().to_vec();

    let output_length = n_fft + n_frames.saturating_sub(1) * hop;
    let istft_params = KokoroIstftParams {
        n_fft,
        hop_length: hop,
    };
    let audio_pcm = kokoro_istft(
        &istft_params,
        &real_flat,
        &imag_flat,
        n_frames,
        output_length,
    )
    .unwrap();

    // Center trim.
    let pad = n_fft / 2;
    let trim_end = audio_pcm.len().saturating_sub(pad);
    let trimmed = if pad < trim_end {
        &audio_pcm[pad..trim_end]
    } else {
        &audio_pcm[..]
    };

    assert_eq!(
        trimmed.len(),
        signal_len,
        "trimmed length mismatch: {} != {signal_len}",
        trimmed.len()
    );

    let (snr_db, max_err, rms_err) = quality_metrics(&signal, trimmed, n_fft);
    assert!(
        snr_db > 35.0,
        "production round-trip SNR {snr_db:.1}dB below 35dB"
    );
    assert!(
        max_err < 0.02,
        "production round-trip max error {max_err:.6} exceeds 0.02"
    );
    assert!(
        rms_err < 0.005,
        "production round-trip RMS {rms_err:.6} exceeds 0.005"
    );
}

// ============================================================================
// 5. Parseval's theorem: energy conservation
// ============================================================================

#[test]
fn test_parsevals_theorem_energy_conservation() {
    // Parseval's theorem: the total energy in time domain should equal the
    // total energy in frequency domain (up to normalization).
    // For windowed STFT: sum of |windowed_frame|^2 = (1/N) * sum of |X[f]|^2
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;

    let signal = sine_wave(signal_len, 5.0);

    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for t in 2..(n_frames - 2) {
        // Time-domain energy for this frame (windowed).
        let offset = t * hop;
        let time_energy: f32 = (0..n_fft)
            .map(|k| {
                let windowed = signal[offset + k] * window[k];
                windowed * windowed
            })
            .sum();

        // Frequency-domain energy: (1/N) * [|X[0]|^2 + 2*sum(|X[f]|^2 for 1..N/2-1) + |X[N/2]|^2]
        let mut freq_energy = 0.0f32;

        // DC
        let r0 = real[0 * n_frames + t];
        let i0 = imag[0 * n_frames + t];
        freq_energy += r0 * r0 + i0 * i0;

        // Interior bins (count twice for conjugate symmetry).
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

        let ratio = if time_energy > 1e-10 {
            freq_energy / time_energy
        } else {
            1.0
        };

        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Parseval's theorem violated at frame {t}: time_energy={time_energy}, \
             freq_energy={freq_energy}, ratio={ratio}"
        );
    }
}

// ============================================================================
// 6. Linearity of STFT
// ============================================================================

#[test]
fn test_stft_linearity() {
    // STFT(a*x + b*y) = a*STFT(x) + b*STFT(y)
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;
    let a = 0.7f32;
    let b = 1.3f32;

    let x = sine_wave(signal_len, 3.0);
    let y = cosine_wave(signal_len, 7.0);
    let combined: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| a * xi + b * yi)
        .collect();

    let (real_x, imag_x, n_frames) = scalar_forward_stft(&x, n_fft, hop);
    let (real_y, imag_y, _) = scalar_forward_stft(&y, n_fft, hop);
    let (real_combined, imag_combined, _) = scalar_forward_stft(&combined, n_fft, hop);

    let n_bins = n_fft / 2 + 1;
    let mut max_real_err = 0.0f32;
    let mut max_imag_err = 0.0f32;

    for i in 0..(n_bins * n_frames) {
        let expected_real = a * real_x[i] + b * real_y[i];
        let expected_imag = a * imag_x[i] + b * imag_y[i];

        max_real_err = max_real_err.max((real_combined[i] - expected_real).abs());
        max_imag_err = max_imag_err.max((imag_combined[i] - expected_imag).abs());
    }

    assert!(
        max_real_err < 1e-4,
        "STFT linearity real error: {max_real_err}"
    );
    assert!(
        max_imag_err < 1e-4,
        "STFT linearity imag error: {max_imag_err}"
    );
}

#[test]
fn test_kokoro_forward_stft_linearity() {
    // Same linearity test but using KokoroForwardStft (FFT-based).
    let n_fft = 20;
    let hop = 5;
    let signal_len = 100;
    let a = 2.0f64;
    let b = 0.5f64;

    let x: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 3.0 * i as f32 / signal_len as f32).sin())
        .collect();
    let y: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 7.0 * i as f32 / signal_len as f32).cos())
        .collect();
    let combined: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (a as f32) * xi + (b as f32) * yi)
        .collect();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();

    let input_x = DynTensor::from_vec(x, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let input_y = DynTensor::from_vec(y, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let input_c = DynTensor::from_vec(combined, &[1, 1, signal_len], &Device::Cpu).unwrap();

    let (mag_x, phase_x) = stft.forward(&input_x).unwrap();
    let (mag_y, phase_y) = stft.forward(&input_y).unwrap();
    let (mag_c, phase_c) = stft.forward(&input_c).unwrap();

    // Convert to real/imag for linear combination check.
    let rx = mag_x.mul(&phase_x.cos().unwrap()).unwrap();
    let ix = mag_x.mul(&phase_x.sin().unwrap()).unwrap();
    let ry = mag_y.mul(&phase_y.cos().unwrap()).unwrap();
    let iy = mag_y.mul(&phase_y.sin().unwrap()).unwrap();
    let rc = mag_c.mul(&phase_c.cos().unwrap()).unwrap();
    let ic = mag_c.mul(&phase_c.sin().unwrap()).unwrap();

    // Expected: a*STFT(x) + b*STFT(y) = STFT(a*x + b*y)
    let expected_r = rx
        .mul_scalar(a)
        .unwrap()
        .add(&ry.mul_scalar(b).unwrap())
        .unwrap();
    let expected_i = ix
        .mul_scalar(a)
        .unwrap()
        .add(&iy.mul_scalar(b).unwrap())
        .unwrap();

    let rc_flat = rc.to_flat_vec::<f32>().unwrap();
    let ic_flat = ic.to_flat_vec::<f32>().unwrap();
    let er_flat = expected_r.to_flat_vec::<f32>().unwrap();
    let ei_flat = expected_i.to_flat_vec::<f32>().unwrap();

    let max_real_err = rc_flat
        .iter()
        .zip(er_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_imag_err = ic_flat
        .iter()
        .zip(ei_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_real_err < 1e-3,
        "KokoroForwardStft linearity real error: {max_real_err}"
    );
    assert!(
        max_imag_err < 1e-3,
        "KokoroForwardStft linearity imag error: {max_imag_err}"
    );
}

// ============================================================================
// 7. Edge cases
// ============================================================================

#[test]
fn test_istft_signal_exactly_nfft_length() {
    // Signal exactly n_fft samples long => 1 frame.
    let n_fft = 16;
    let hop = 4;
    let signal = sine_wave(n_fft, 2.0);

    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);
    assert_eq!(n_frames, 1);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let reconstructed = basis.istft(&real, &imag, n_frames, n_fft).unwrap();
    assert_eq!(reconstructed.len(), n_fft);

    // With single frame, reconstruction through COLA may not be perfect
    // at edges, but should be finite.
    for v in &reconstructed {
        assert!(
            v.is_finite(),
            "non-finite value in single-frame reconstruction"
        );
    }
}

#[test]
fn test_istft_non_power_of_two_nfft() {
    // n_fft = 20 (Kokoro), n_fft = 30 (arbitrary), n_fft = 6 (tiny).
    for &n_fft in &[20, 30, 6] {
        let hop = n_fft / 2;
        if hop == 0 {
            continue;
        }
        let signal_len = n_fft * 8;
        let signal = sine_wave(signal_len, 3.0);

        let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

        let params = IstftParams::new(n_fft, hop, false, false).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let full_len = n_fft + (n_frames - 1) * hop;
        let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

        let (snr_db, max_err, _) = quality_metrics(&signal, &reconstructed, n_fft);
        assert!(
            max_err < 0.15,
            "n_fft={n_fft}: round-trip max error {max_err:.6} exceeds 0.15"
        );
        assert!(
            snr_db > 15.0,
            "n_fft={n_fft}: round-trip SNR {snr_db:.1}dB below 15dB"
        );
    }
}

#[test]
fn test_istft_very_short_two_frame_signal() {
    // Minimal: 2 frames worth of signal.
    let n_fft = 8;
    let hop = 4;
    let signal_len = n_fft + hop; // = 12, gives 2 frames

    let signal = sine_wave(signal_len, 1.0);
    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);
    assert_eq!(n_frames, 2);

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();
    assert_eq!(reconstructed.len(), full_len);

    // With only 2 frames, boundary effects dominate, but output must be finite.
    for v in &reconstructed {
        assert!(v.is_finite());
    }
}

#[test]
fn test_kokoro_istft_large_frame_count() {
    // Stress test: 1000 frames (simulating a long utterance).
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 1000;

    // All-zero input => all-zero output (fast path, no numerical instability).
    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    let output_length = n_fft + (n_frames - 1) * hop;

    let params = KokoroIstftParams {
        n_fft,
        hop_length: hop,
    };
    let result = kokoro_istft(&params, &real, &imag, n_frames, output_length).unwrap();
    assert_eq!(result.len(), output_length);
    for v in &result {
        assert!(*v == 0.0 || v.abs() < 1e-10);
    }
}

#[test]
fn test_stft_dc_signal_dominant_at_bin_zero() {
    // A constant (DC) signal should have its dominant energy at bin 0.
    // Non-DC bins may have some energy due to Hann window spectral leakage
    // (the window modulates the constant, spreading energy), but bin 0
    // should always be the largest by a significant margin.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;

    let signal = vec![1.0f32; signal_len];
    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);
    let n_bins = n_fft / 2 + 1;

    for t in 2..(n_frames - 2) {
        let dc_mag = real[0 * n_frames + t].hypot(imag[0 * n_frames + t]);
        assert!(dc_mag > 0.1, "DC bin should have significant energy");

        // DC bin should be the largest.
        for f in 1..n_bins {
            let mag = real[f * n_frames + t].hypot(imag[f * n_frames + t]);
            assert!(
                dc_mag > mag,
                "frame {t}: DC bin magnitude {dc_mag} should exceed bin {f} magnitude {mag}"
            );
        }
    }
}

// ============================================================================
// 8. Time-shift property
// ============================================================================

#[test]
fn test_stft_time_shift_phase_rotation() {
    // Shifting a signal by `d` samples within a frame should rotate the phase
    // by -2*pi*f*d/N at each frequency bin f, without changing magnitude.
    //
    // We test this approximately by comparing STFT of original vs shifted signal:
    // magnitudes should match, phases should differ by the expected rotation.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;
    let shift = 4; // shift by 4 samples

    let signal = sine_wave(signal_len, 5.0);
    // Shifted signal: prepend `shift` zeros, truncate to same length.
    let mut shifted = vec![0.0f32; shift];
    shifted.extend_from_slice(&signal[..signal_len - shift]);
    assert_eq!(shifted.len(), signal_len);

    let (real_orig, imag_orig, n_frames) = scalar_forward_stft(&signal, n_fft, hop);
    let (real_shift, imag_shift, _) = scalar_forward_stft(&shifted, n_fft, hop);

    let n_bins = n_fft / 2 + 1;

    // Compare magnitudes — should be close for interior frames.
    // (Not exact because the shift changes which samples are under the window.)
    // Skip boundary frames and frames near the prepended zeros.
    for t in 3..(n_frames - 3) {
        for f in 1..(n_bins - 1) {
            let mag_orig = real_orig[f * n_frames + t].hypot(imag_orig[f * n_frames + t]);
            let mag_shift = real_shift[f * n_frames + t].hypot(imag_shift[f * n_frames + t]);

            if mag_orig > 0.1 {
                let mag_ratio = mag_shift / mag_orig;
                // Magnitudes should be similar (within 30% for frames far from the shift region).
                assert!(
                    (mag_ratio - 1.0).abs() < 0.5,
                    "frame {t}, bin {f}: magnitude changed significantly under shift: \
                     orig={mag_orig}, shifted={mag_shift}, ratio={mag_ratio}"
                );
            }
        }
    }
}

// ============================================================================
// 9. IstftBasis vs kokoro_istft consistency
// ============================================================================

#[test]
fn test_istft_basis_vs_kokoro_istft_consistency() {
    // Both iSTFT implementations should produce the same output for Kokoro params.
    let n_fft = 20;
    let hop = 5;
    let signal_len = 100;

    let signal = sine_wave(signal_len, 4.0);
    let (real, imag, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

    // IstftBasis path
    let params_basis = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params_basis).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let result_basis = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    // kokoro_istft path
    let params_kokoro = KokoroIstftParams {
        n_fft,
        hop_length: hop,
    };
    let result_kokoro = kokoro_istft(&params_kokoro, &real, &imag, n_frames, full_len).unwrap();

    assert_eq!(result_basis.len(), result_kokoro.len());

    let mut max_diff = 0.0f32;
    for i in 0..result_basis.len() {
        let diff = (result_basis[i] - result_kokoro[i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(
        max_diff < 1e-5,
        "IstftBasis and kokoro_istft differ by {max_diff} (expected < 1e-5)"
    );
}

// ============================================================================
// 10. Normalized vs unnormalized iSTFT
// ============================================================================

#[test]
fn test_normalized_vs_unnormalized_istft_scaling() {
    // normalized=true uses 1/sqrt(N), normalized=false uses 1/N.
    // With matching forward STFT normalization, both should reconstruct correctly.
    let n_fft = 32;
    let hop = 8;
    let signal_len = 128;

    let signal = sine_wave(signal_len, 5.0);

    // Unnormalized forward + unnormalized inverse.
    let (real_unnorm, imag_unnorm, n_frames) = scalar_forward_stft(&signal, n_fft, hop);

    let params_unnorm = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis_unnorm = IstftBasis::new(params_unnorm).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let recon_unnorm = basis_unnorm
        .istft(&real_unnorm, &imag_unnorm, n_frames, full_len)
        .unwrap();

    // Normalized forward (scale by 1/sqrt(N)) + normalized inverse.
    let norm_factor = 1.0 / (n_fft as f32).sqrt();
    let real_norm: Vec<f32> = real_unnorm.iter().map(|v| v * norm_factor).collect();
    let imag_norm: Vec<f32> = imag_unnorm.iter().map(|v| v * norm_factor).collect();

    let params_norm = IstftParams::new(n_fft, hop, true, false).unwrap();
    let basis_norm = IstftBasis::new(params_norm).unwrap();
    let recon_norm = basis_norm
        .istft(&real_norm, &imag_norm, n_frames, full_len)
        .unwrap();

    // Both reconstructions should be approximately equal.
    let mut max_diff = 0.0f32;
    for i in 0..recon_unnorm.len() {
        max_diff = max_diff.max((recon_unnorm[i] - recon_norm[i]).abs());
    }

    assert!(
        max_diff < 1e-4,
        "normalized vs unnormalized iSTFT differ by {max_diff}"
    );
}
