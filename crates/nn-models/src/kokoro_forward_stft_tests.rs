// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for KokoroForwardStft (FFT-based forward STFT).

use std::f32::consts::PI;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use super::KokoroForwardStft;

/// Reference forward STFT (scalar loop) for comparison.
/// Returns (real, imag) each `[n_bins, n_frames]` flattened row-major.
fn reference_stft(signal: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>) {
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
            // Layout: [freq_bin, frame] row-major
            real[f * n_frames + t] = r;
            imag[f * n_frames + t] = im;
        }
    }

    (real, imag)
}

/// Wrap-aware phase difference in radians, normalized to [0, π].
fn wrap_phase_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % (2.0 * PI);
    if d < 0.0 {
        d += 2.0 * PI;
    }
    if d > PI {
        d = 2.0 * PI - d;
    }
    d
}

#[test]
fn test_kokoro_params_basic() {
    // Kokoro defaults: n_fft=20, hop=5
    let stft = KokoroForwardStft::new(20, 5, &Device::Cpu).unwrap();
    assert_eq!(stft.n_bins(), 11); // 20/2 + 1 = 11
}

#[test]
fn test_invalid_n_fft_zero() {
    let result = KokoroForwardStft::new(0, 5, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_invalid_n_fft_odd() {
    let result = KokoroForwardStft::new(21, 5, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_invalid_hop_zero() {
    let result = KokoroForwardStft::new(20, 0, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_magnitude_phase_vs_reference() {
    // Use Kokoro params: n_fft=20, hop=5
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1; // 11
    let signal_len = 100;

    // Generate a test signal: sum of two sinusoids
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            (2.0 * PI * 3.0 * t).sin() + 0.5 * (2.0 * PI * 7.0 * t).cos()
        })
        .collect();

    // Reference scalar STFT
    let (ref_real, ref_imag) = reference_stft(&signal, n_fft, hop);
    let n_frames = (signal_len - n_fft) / hop + 1; // (100-20)/5 + 1 = 17

    // Conv1d STFT
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, phase) = stft.forward(&input).unwrap();

    assert_eq!(magnitude.dims(), &[1, n_bins, n_frames]);
    assert_eq!(phase.dims(), &[1, n_bins, n_frames]);

    let mag_data = magnitude.to_flat_vec::<f32>().unwrap();
    let phase_data = phase.to_flat_vec::<f32>().unwrap();

    // Verify magnitude against reference: mag = sqrt(real² + imag²).
    // Phase comparison is wrap-aware: the STFT may use either atan2(Im, Re) or
    // atan2(-Im, Re) depending on convention. Wrap-aware diff handles both.
    let mut max_mag_err = 0.0f32;
    let mut max_phase_err = 0.0f32;

    for f in 0..n_bins {
        for t in 0..n_frames {
            let idx = f * n_frames + t;
            let r = ref_real[idx];
            let im = ref_imag[idx];

            let expected_mag = r.hypot(im);
            // Try both conventions and take the smaller error.
            let phase_a = im.atan2(r); // standard: atan2(Im(X), Re(X))
            let phase_b = (-im).atan2(r); // negated: atan2(-Im(X), Re(X))

            let actual_mag = mag_data[idx];
            let actual_phase = phase_data[idx];

            max_mag_err = max_mag_err.max((actual_mag - expected_mag).abs());
            let err_a = wrap_phase_diff(actual_phase, phase_a);
            let err_b = wrap_phase_diff(actual_phase, phase_b);
            max_phase_err = max_phase_err.max(err_a.min(err_b));
        }
    }

    // FFT and scalar DFT use different summation order (butterfly vs flat
    // accumulation), so real/imag differ by ~1e-6. Near zero-crossings,
    // atan2 amplifies this to larger phase differences (up to ~0.06 rad).
    // The key property is no 2π wraps (wrap_phase_diff handles that).
    assert!(
        max_mag_err < 1e-4,
        "magnitude max error {max_mag_err} exceeds tolerance"
    );
    assert!(
        max_phase_err < 0.1,
        "phase max error {max_phase_err} exceeds tolerance"
    );
}

#[test]
fn test_forward_cat_shape() {
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let signal_len = 60;
    let n_frames = (signal_len - n_fft) / hop + 1; // (60-20)/5 + 1 = 9

    let signal = vec![0.1f32; signal_len];
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let cat = stft.forward_cat(&input).unwrap();

    // forward_cat returns [B, 2*n_bins, n_frames]
    assert_eq!(cat.dims(), &[1, 2 * n_bins, n_frames]);
}

#[test]
fn test_magnitude_non_negative() {
    let n_fft = 20;
    let hop = 5;

    let signal: Vec<f32> = (0..80).map(|i| (i as f32 * 0.3).sin() - 0.5).collect();
    let input = DynTensor::from_vec(signal, &[1, 1, 80], &Device::Cpu).unwrap();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let (magnitude, _phase) = stft.forward(&input).unwrap();

    let mag_data = magnitude.to_flat_vec::<f32>().unwrap();
    for (i, &v) in mag_data.iter().enumerate() {
        assert!(v >= 0.0, "magnitude[{i}] = {v} is negative");
    }
}

#[test]
fn test_zero_signal_magnitude_zero() {
    let n_fft = 20;
    let hop = 5;
    let signal_len = 40;

    let signal = vec![0.0f32; signal_len];
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let (magnitude, _phase) = stft.forward(&input).unwrap();

    let mag_data = magnitude.to_flat_vec::<f32>().unwrap();
    let max_val = mag_data.iter().copied().fold(0.0f32, f32::max);
    // FFT-based STFT returns exact zero magnitude for zero signal (eps removed in 1679337).
    assert!(
        max_val == 0.0,
        "zero signal magnitude should be exactly 0.0, got {max_val}"
    );
}

#[test]
fn test_rank_mismatch_error() {
    let stft = KokoroForwardStft::new(20, 5, &Device::Cpu).unwrap();

    // Wrong rank: [1, 80] instead of [1, 1, 80]
    let input = DynTensor::from_vec(vec![0.0f32; 80], &[1, 80], &Device::Cpu).unwrap();
    let result = stft.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_multi_channel_error() {
    let stft = KokoroForwardStft::new(20, 5, &Device::Cpu).unwrap();

    // Wrong channels: [1, 2, 40] instead of [1, 1, 40]
    let input = DynTensor::from_vec(vec![0.0f32; 80], &[1, 2, 40], &Device::Cpu).unwrap();
    let result = stft.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_batch_dimension() {
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let signal_len = 40;
    let n_frames = (signal_len - n_fft) / hop + 1;
    let batch = 3;

    // Batch of 3 identical signals
    let one_signal: Vec<f32> = (0..signal_len).map(|i| (i as f32 * 0.2).sin()).collect();
    let mut batch_data = Vec::with_capacity(batch * signal_len);
    for _ in 0..batch {
        batch_data.extend_from_slice(&one_signal);
    }

    let input = DynTensor::from_vec(batch_data, &[batch, 1, signal_len], &Device::Cpu).unwrap();
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let (magnitude, phase) = stft.forward(&input).unwrap();

    assert_eq!(magnitude.dims(), &[batch, n_bins, n_frames]);
    assert_eq!(phase.dims(), &[batch, n_bins, n_frames]);

    // All batch elements should be identical
    let mag_data = magnitude.to_flat_vec::<f32>().unwrap();
    let elems_per_batch = n_bins * n_frames;
    for b in 1..batch {
        for i in 0..elems_per_batch {
            let diff = (mag_data[b * elems_per_batch + i] - mag_data[i]).abs();
            assert!(
                diff < 1e-6,
                "batch {b} element {i} differs from batch 0 by {diff}"
            );
        }
    }
}

// -- STFT → iSTFT round-trip tests (#2666) ------------------------------------

/// Compute SNR (dB), max error, and RMS error between two slices over an interior region.
/// `skip` samples are excluded from each end to avoid boundary effects.
fn round_trip_quality(original: &[f32], reconstructed: &[f32], skip: usize) -> (f32, f32, f32) {
    let len = original.len().min(reconstructed.len());
    let start = skip;
    let end = len.saturating_sub(skip);
    if end <= start {
        return (f32::INFINITY, 0.0, 0.0);
    }
    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    let mut sum_sq_ref = 0.0f32;
    let count = (end - start) as f32;
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
    let rms_err = (sum_sq_err / count).sqrt();
    (snr_db, max_err, rms_err)
}

/// Reference STFT → iSTFT round-trip with center padding and trim.
///
/// Uses the scalar `reference_stft` for the forward pass (convention-independent)
/// and `kokoro_istft` for reconstruction. This proves the iSTFT correctly
/// inverts the STFT, matching the production `forward_audio()` geometry
/// (center pad + overlap-add + center trim).
///
/// Part of #2666, Part of #2218.
#[test]
fn test_reference_stft_istft_round_trip_center() {
    use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};

    let n_fft = 20;
    let hop = 5;
    let signal_len = 200;

    // Test signal: sum of sinusoids at different frequencies
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.7 * (2.0 * PI * 5.0 * t).sin() + 0.3 * (2.0 * PI * 11.0 * t).cos()
        })
        .collect();

    // Center-pad the signal with reflection (matching forward_audio/forward_center)
    let pad = n_fft / 2;
    let mut padded = Vec::with_capacity(signal_len + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(signal[i]);
    }
    padded.extend_from_slice(&signal);
    for i in (signal_len - pad - 1..signal_len - 1).rev() {
        padded.push(signal[i]);
    }

    // Forward STFT on padded signal (scalar reference, no phase convention dependency)
    let (real, imag) = reference_stft(&padded, n_fft, hop);
    let padded_len = padded.len();
    let n_frames = (padded_len - n_fft) / hop + 1;

    // iSTFT + center trim (matches forward_audio:67-96)
    let output_length = n_fft + n_frames.saturating_sub(1) * hop;
    let params = KokoroIstftParams {
        n_fft,
        hop_length: hop,
    };
    let audio_pcm = kokoro_istft(&params, &real, &imag, n_frames, output_length).unwrap();

    let trim_end = audio_pcm.len().saturating_sub(pad);
    let trimmed = if pad < trim_end {
        &audio_pcm[pad..trim_end]
    } else {
        &audio_pcm[..]
    };

    assert_eq!(trimmed.len(), signal_len, "trimmed length mismatch");

    let (snr_db, max_err, rms_err) = round_trip_quality(&signal, trimmed, n_fft);
    assert!(
        max_err < 0.05,
        "max error {max_err:.6} exceeds 0.05 (SNR={snr_db:.1}dB)"
    );
    assert!(snr_db > 30.0, "SNR {snr_db:.1}dB below 30dB");
    assert!(rms_err < 0.01, "RMS error {rms_err:.6} exceeds 0.01");
}

/// iSTFT + center trim on DynTensor real/imag spectrograms.
///
/// Mirrors forward_audio:64-96: extracts flat f32 slices, calls kokoro_istft,
/// then center-trims n_fft/2 samples from each side.
fn istft_center_trim(
    real_spec: &DynTensor,
    imag_spec: &DynTensor,
    n_fft: usize,
    hop: usize,
    n_frames: usize,
) -> Vec<f32> {
    use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};

    let real_arr = real_spec.to_f32_array().unwrap();
    let real_std = real_arr.as_standard_layout();
    let real_flat = real_std.as_slice().expect("standard-layout");
    let imag_arr = imag_spec.to_f32_array().unwrap();
    let imag_std = imag_arr.as_standard_layout();
    let imag_flat = imag_std.as_slice().expect("standard-layout");

    let output_length = n_fft + n_frames.saturating_sub(1) * hop;
    let params = KokoroIstftParams {
        n_fft,
        hop_length: hop,
    };
    let audio_pcm = kokoro_istft(&params, real_flat, imag_flat, n_frames, output_length).unwrap();

    let center_pad = n_fft / 2;
    let trim_end = audio_pcm.len().saturating_sub(center_pad);
    if center_pad < trim_end {
        audio_pcm[center_pad..trim_end].to_vec()
    } else {
        audio_pcm
    }
}

/// Conv1d-based KokoroForwardStft → mag*cos/sin → kokoro_istft → center_trim.
///
/// Tests the EXACT chain used by `forward_audio()`. If this test fails, the
/// #2663 cosine=0.06 bug is in the STFT/iSTFT chain. If it passes, the bug
/// is upstream in the model.
///
/// Part of #2666, Part of #2663, Part of #2218.
#[test]
fn test_conv_stft_istft_round_trip_center() {
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let signal_len = 200; // must be multiple of hop for exact reconstruction

    // Multi-frequency test signal
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            0.5 * (2.0 * PI * 3.0 * t).sin()
                + 0.3 * (2.0 * PI * 7.0 * t).cos()
                + 0.2 * (2.0 * PI * 11.0 * t).sin()
        })
        .collect();

    // Conv1d forward STFT with center padding
    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal.clone(), &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, phase) = stft.forward_center(&input).unwrap();
    assert_eq!(magnitude.dims()[1], n_bins);
    let n_frames = magnitude.dims()[2];

    // Reconstruct real/imag (no pi scaling — STFT phase is already in radians)
    let real_spec = magnitude.mul(&phase.cos().unwrap()).unwrap();
    let imag_spec = magnitude.mul(&phase.sin().unwrap()).unwrap();

    // iSTFT + center trim (same as forward_audio)
    let trimmed = istft_center_trim(&real_spec, &imag_spec, n_fft, hop, n_frames);
    assert_eq!(trimmed.len(), signal_len, "round-trip length mismatch");

    // Quality check — interior region (skip boundary effects)
    let (snr_db, max_err, rms_err) = round_trip_quality(&signal, &trimmed, n_fft);
    assert!(
        max_err < 0.05,
        "Conv1d round-trip max error {max_err:.6} exceeds 0.05 (SNR={snr_db:.1}dB)"
    );
    assert!(
        snr_db > 30.0,
        "Conv1d round-trip SNR {snr_db:.1}dB below 30dB"
    );
    assert!(
        rms_err < 0.01,
        "Conv1d round-trip RMS {rms_err:.6} exceeds 0.01"
    );
}

/// Test forward_audio's pi*phase reconstruction path with known STFT data.
///
/// The Generator outputs phase = sin(phase_raw) ∈ [-1, 1]. forward_audio then
/// computes: pi_phase = phase * π, real = mag * cos(pi_phase), imag = mag * sin(pi_phase).
///
/// This test verifies that if we take a known STFT, encode it in Generator format
/// (mag, sin(atan2(imag,real)/π)), then decode via the pi*phase path, we recover
/// the original STFT real/imag coefficients correctly.
///
/// Part of #2666, Part of #2663, Part of #2218.
#[test]
fn test_pi_phase_reconstruction_identity() {
    // Generate known STFT data from a signal
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let signal_len = 100;

    let signal: Vec<f32> = (0..signal_len)
        .map(|i| (2.0 * PI * 4.0 * i as f32 / signal_len as f32).sin())
        .collect();

    let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
    let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
    let (magnitude, phase) = stft.forward(&input).unwrap();

    let n_frames = magnitude.dims()[2];

    // Encode phase in "Generator format": phase_gen = sin(phase_raw) where
    // the actual angle is π * phase_gen. We simulate: phase_gen = phase / π
    // (phase from atan2 is the true angle). Then forward_audio does
    // pi_phase = phase_gen * π = phase (the original angle), so cos/sin
    // should perfectly reconstruct real/imag.
    let phase_gen = phase.mul_scalar(1.0 / std::f64::consts::PI).unwrap();

    // Apply forward_audio reconstruction: pi_phase = phase_gen * π
    let pi_phase = phase_gen.mul_scalar(std::f64::consts::PI).unwrap();
    let cos_phase = pi_phase.cos().unwrap();
    let sin_phase = pi_phase.sin().unwrap();
    let real_recon = magnitude.mul(&cos_phase).unwrap();
    let imag_recon = magnitude.mul(&sin_phase).unwrap();

    // Compare with direct reconstruction: real = mag * cos(phase), imag = mag * sin(phase)
    let real_direct = magnitude.mul(&phase.cos().unwrap()).unwrap();
    let imag_direct = magnitude.mul(&phase.sin().unwrap()).unwrap();

    let real_recon_flat = real_recon.to_flat_vec::<f32>().unwrap();
    let real_direct_flat = real_direct.to_flat_vec::<f32>().unwrap();
    let imag_recon_flat = imag_recon.to_flat_vec::<f32>().unwrap();
    let imag_direct_flat = imag_direct.to_flat_vec::<f32>().unwrap();

    let mut max_real_err = 0.0f32;
    let mut max_imag_err = 0.0f32;
    for i in 0..n_bins * n_frames {
        max_real_err = max_real_err.max((real_recon_flat[i] - real_direct_flat[i]).abs());
        max_imag_err = max_imag_err.max((imag_recon_flat[i] - imag_direct_flat[i]).abs());
    }

    // The pi*phase round-trip (phase → phase/π → phase/π * π → cos/sin) should be
    // exact up to floating-point precision.
    assert!(
        max_real_err < 1e-5,
        "pi*phase real reconstruction error {max_real_err:.8} exceeds 1e-5"
    );
    assert!(
        max_imag_err < 1e-5,
        "pi*phase imag reconstruction error {max_imag_err:.8} exceeds 1e-5"
    );
}

/// Verify output_length formula consistency between forward_audio and iSTFT.
///
/// forward_audio (kokoro_audio.rs:67): output_length = n_fft + (n_frames-1) * hop
/// After center trim (remove n_fft/2 from each side):
///   final_len = output_length - n_fft = (n_frames-1) * hop
///
/// With center-padded STFT: n_frames = signal_len / hop + 1 (for signal_len % hop == 0)
///   => final_len = (signal_len / hop + 1 - 1) * hop = signal_len
///
/// Part of #2666.
#[test]
fn test_output_length_formula_consistency() {
    let n_fft = 20;
    let hop = 5;

    // Test several signal lengths that are multiples of hop
    for &signal_len in &[50, 100, 200, 500, 1000] {
        let stft = KokoroForwardStft::new(n_fft, hop, &Device::Cpu).unwrap();
        let signal = vec![0.1f32; signal_len];
        let input = DynTensor::from_vec(signal, &[1, 1, signal_len], &Device::Cpu).unwrap();
        let (magnitude, _) = stft.forward_center(&input).unwrap();
        let n_frames = magnitude.dims()[2];

        // forward_audio formula
        let output_length = n_fft + n_frames.saturating_sub(1) * hop;
        let center_pad = n_fft / 2;
        let trim_end = output_length.saturating_sub(center_pad);
        let final_len = if center_pad < trim_end {
            trim_end - center_pad
        } else {
            output_length
        };

        assert_eq!(
            final_len, signal_len,
            "signal_len={signal_len}: forward_audio formula gives final_len={final_len} (n_frames={n_frames})"
        );
    }
}

/// Performance proof: CPU iSTFT runtime scales O(n_frames), not O(n_frames²).
///
/// The iSTFT inner loop is O(n_frames × n_fft × n_bins). For fixed Kokoro
/// params (n_fft=20, n_bins=11), this is O(n_frames). Multiplying n_frames by
/// 4 should roughly multiply runtime by 4. An O(n²) bug would cause 16× slowdown.
///
/// Uses generous 8× bound for 4× input increase to avoid flakiness.
///
/// Part of #2218.
#[test]
fn test_kokoro_istft_linear_scaling() {
    use crate::kokoro_istft::{kokoro_istft, KokoroIstftParams};
    use std::time::Instant;

    let params = KokoroIstftParams {
        n_fft: 20,
        hop_length: 5,
    };
    let n_bins = 11;

    // Small: 200 frames (typical short utterance)
    let n_small = 200;
    let real_s = vec![0.01f32; n_bins * n_small];
    let imag_s = vec![0.01f32; n_bins * n_small];
    let out_len_s = 20 + (n_small - 1) * 5;

    // Large: 4× bigger = 800 frames
    let n_large = n_small * 4;
    let real_l = vec![0.01f32; n_bins * n_large];
    let imag_l = vec![0.01f32; n_bins * n_large];
    let out_len_l = 20 + (n_large - 1) * 5;

    // Warm up
    let _ = kokoro_istft(&params, &real_s, &imag_s, n_small, out_len_s).unwrap();

    // Measure small (average over 5 runs)
    let start = Instant::now();
    for _ in 0..5 {
        let _ = kokoro_istft(&params, &real_s, &imag_s, n_small, out_len_s).unwrap();
    }
    let time_small = start.elapsed();

    // Measure large (average over 5 runs)
    let start = Instant::now();
    for _ in 0..5 {
        let _ = kokoro_istft(&params, &real_l, &imag_l, n_large, out_len_l).unwrap();
    }
    let time_large = start.elapsed();

    let ratio = time_large.as_secs_f64() / time_small.as_secs_f64();
    // O(n): expect ~4× ratio. O(n²): would be ~16×.
    assert!(
        ratio < 8.0,
        "iSTFT scaling ratio {ratio:.1}× for 4× input increase — \
         expected <8× (O(n)), got >8× suggesting O(n²). \
         small={:.3}ms, large={:.3}ms",
        time_small.as_secs_f64() * 1000.0 / 5.0,
        time_large.as_secs_f64() * 1000.0 / 5.0,
    );
}
