// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU-side helpers for HTDemucs: normalization and shape computation.
//!
//! Extracted from `htdemucs.rs` to keep the main module under 500 lines.
//!
//! Part of #779 and #831.

use crate::cache::PipelineCache;
use crate::istft::IstftBasis;
use crate::istft_gpu::IstftGpuBasis;

use super::{HTDemucsError, AUDIO_CHANNELS, NUM_SOURCES, OUTPUT_CHANNELS};

// ---------------------------------------------------------------------------
// Normalization helpers (CPU-side, matching Python HTDemucs)
// ---------------------------------------------------------------------------

/// Normalize stereo audio: compute mono mean and std, normalize all channels.
///
/// Python reference: `mix = mix / mix.std() * ref_scale` with `ref_scale =0.1`.
/// We use the simpler `(x - mean) / std` normalization with epsilon for stability.
///
/// Returns `(normalized, mean, std)`.
///
/// Returns `NormalizeOverflow` if any normalized output is non-finite (can happen
/// when input values are finite but very large and std_val is clamped to epsilon).
pub(super) fn normalize_audio(
    audio: &[f32],
    t: usize,
) -> Result<(Vec<f32>, f32, f32), HTDemucsError> {
    // Guard: t == 0 produces division by zero in mean and variance computation.
    if t == 0 {
        return Err(HTDemucsError::ZeroLengthAudio);
    }
    // Compute mono signal for statistics.
    let mut mono_sum = 0.0f64;
    for i in 0..t {
        let mut sample = 0.0f64;
        for c in 0..AUDIO_CHANNELS {
            sample += f64::from(audio[c * t + i]);
        }
        mono_sum += sample / AUDIO_CHANNELS as f64;
    }
    let mean = (mono_sum / t as f64) as f32;

    // Compute std on mono signal.
    let mut var_sum = 0.0f64;
    for i in 0..t {
        let mut sample = 0.0f64;
        for c in 0..AUDIO_CHANNELS {
            sample += f64::from(audio[c * t + i]);
        }
        let mono_val = (sample / AUDIO_CHANNELS as f64) as f32;
        let diff = f64::from(mono_val - mean);
        var_sum += diff * diff;
    }
    // NaN.max(x) returns NaN in Rust, so .max(1e-8) alone does not protect
    // against NaN from corrupt input. Guard explicitly (defense-in-depth).
    let raw_std = (var_sum / t as f64).sqrt() as f32;
    let std_val = if raw_std.is_finite() {
        raw_std.max(1e-8)
    } else {
        1e-8
    };

    // Normalize all channels.
    let mut normalized = Vec::with_capacity(audio.len());
    for &v in audio {
        normalized.push((v - mean) / std_val);
    }

    // Check output finiteness — (v - mean) / std_val can overflow to Inf
    // when v is finite but very large and std_val is clamped to epsilon.
    let non_finite_count = normalized.iter().filter(|v| !v.is_finite()).count();
    if non_finite_count > 0 {
        return Err(HTDemucsError::NormalizeOverflow {
            count: non_finite_count,
        });
    }

    Ok((normalized, mean, std_val))
}

/// Denormalize output: multiply by std, add mean.
///
/// Output shape `[OUTPUT_CHANNELS, T]`: each source channel gets the same
/// mean/std applied (matching Python HTDemucs which normalizes the mix and
/// applies inverse to all sources).
///
/// Returns `DenormalizeLengthMismatch` if `data.len() < OUTPUT_CHANNELS * t`.
pub(super) fn denormalize_output(
    data: &[f32],
    t: usize,
    mean: f32,
    std_val: f32,
) -> Result<Vec<f32>, HTDemucsError> {
    let total = OUTPUT_CHANNELS * t;
    if data.len() < total {
        return Err(HTDemucsError::DenormalizeLengthMismatch {
            actual: data.len(),
            expected: total,
        });
    }
    let mut out = Vec::with_capacity(total);
    for &v in data.iter().take(total) {
        out.push(v * std_val + mean);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Shape computation helpers
// ---------------------------------------------------------------------------

/// Compute the bottleneck temporal dimension after 4 encoder blocks.
///
/// Each block: pad to stride(4) multiple → Conv1d(k=8, s=4, p=2).
pub(super) fn compute_bottleneck_t(audio_t: usize) -> usize {
    const KERNEL: usize = 8;
    const STRIDE: usize = 4;
    const PADDING: usize = KERNEL / 4;

    let mut t = audio_t;
    for _ in 0..4 {
        // Pad to stride multiple.
        if !t.is_multiple_of(STRIDE) {
            t += STRIDE - (t % STRIDE);
        }
        // Conv1d output: (padded + 2*padding - kernel) / stride + 1
        t = (t + 2 * PADDING - KERNEL) / STRIDE + 1;
    }
    t
}

/// Compute encoder input lengths at each depth (used by decoder for trim).
pub(super) fn compute_encoder_input_lengths(audio_t: usize) -> Vec<usize> {
    const KERNEL: usize = 8;
    const STRIDE: usize = 4;
    const PADDING: usize = KERNEL / 4;

    let mut lengths = Vec::with_capacity(4);
    let mut t = audio_t;
    for _ in 0..4 {
        lengths.push(t);
        // Pad to stride multiple.
        if !t.is_multiple_of(STRIDE) {
            t += STRIDE - (t % STRIDE);
        }
        // Conv1d output.
        t = (t + 2 * PADDING - KERNEL) / STRIDE + 1;
    }
    lengths
}

/// Compute spectral encoder input frequency dims at each depth.
///
/// Spectral Conv1d(k=8, s=4, p=2) downsample along frequency axis:
/// e.g., 2048 → 512 → 128 → 32 → 8.
pub(super) fn compute_spectral_encoder_freqs(initial_f: usize) -> Vec<usize> {
    const KERNEL: usize = 8;
    const STRIDE: usize = 4;
    const PADDING: usize = KERNEL / 4;

    let mut freqs = Vec::with_capacity(4);
    let mut f = initial_f;
    for _ in 0..4 {
        freqs.push(f);
        f = (f + 2 * PADDING - KERNEL) / STRIDE + 1;
    }
    freqs
}

// ---------------------------------------------------------------------------
// Spectral reconstruction: spectral decoder output → iSTFT → waveform
// ---------------------------------------------------------------------------

/// Reconstruct time-domain audio from spectral decoder output via iSTFT.
///
/// The spectral decoder outputs `[16, F, T]` (flattened) where 16 =
/// NUM_SOURCES(4) × AUDIO_CHANNELS(2) × 2 (real + imaginary).
///
/// This function:
/// 1. Denormalizes the spectral output (reverse mean/std normalization)
/// 2. Splits into 8 source-channel pairs (4 sources × 2 stereo channels)
/// 3. Runs iSTFT on each (real, imag) pair
/// 4. Returns `[OUTPUT_CHANNELS, audio_t]` (flattened) — same layout as
///    the temporal decoder output, ready for element-wise summation.
///
/// Reference: dvoice `model.rs:376-421` (`denormalize_and_combine`).
pub(super) fn spectral_reconstruct(
    spectral_decoded: &[f32],
    basis: &IstftBasis,
    stft_f: usize,
    stft_t: usize,
    audio_t: usize,
    mean: f32,
    std_val: f32,
) -> Result<Vec<f32>, HTDemucsError> {
    // Spectral decoder output layout: [16, F, T] row-major.
    // 16 channels = NUM_SOURCES * AUDIO_CHANNELS * 2 (interleaved real/imag).
    // Channel ordering: for source s, audio channel c:
    //   real = channel_idx(s, c, 0) = s * AUDIO_CHANNELS * 2 + c * 2
    //   imag = channel_idx(s, c, 1) = s * AUDIO_CHANNELS * 2 + c * 2 + 1
    let total_ch = NUM_SOURCES * AUDIO_CHANNELS * 2; // 16
    let expected_len = total_ch * stft_f * stft_t;
    if spectral_decoded.len() != expected_len {
        return Err(HTDemucsError::DenormalizeLengthMismatch {
            actual: spectral_decoded.len(),
            expected: expected_len,
        });
    }

    let ft = stft_f * stft_t;
    let mut output = vec![0.0f32; OUTPUT_CHANNELS * audio_t];

    for source in 0..NUM_SOURCES {
        for ch in 0..AUDIO_CHANNELS {
            let real_ch = source * AUDIO_CHANNELS * 2 + ch * 2;
            let imag_ch = real_ch + 1;

            // Extract [F, T] slices for real and imaginary parts.
            let real_start = real_ch * ft;
            let imag_start = imag_ch * ft;
            let real_slice = &spectral_decoded[real_start..real_start + ft];
            let imag_slice = &spectral_decoded[imag_start..imag_start + ft];

            // Denormalize: reverse the input mean/std normalization.
            let real_denorm: Vec<f32> = real_slice.iter().map(|&v| v * std_val + mean).collect();
            let imag_denorm: Vec<f32> = imag_slice.iter().map(|&v| v * std_val + mean).collect();

            // iSTFT expects [n_bins, n_frames] — same as [F, T].
            let waveform = basis.istft(&real_denorm, &imag_denorm, stft_t, audio_t)?;

            // Accumulate into output at the correct source-channel position.
            let out_ch = source * AUDIO_CHANNELS + ch;
            let out_start = out_ch * audio_t;
            for (i, &v) in waveform.iter().enumerate() {
                output[out_start + i] += v;
            }
        }
    }

    // Validate output finiteness.
    crate::check_non_finite_err(&output, |count| HTDemucsError::NonFiniteIntermediate {
        stage: "spectral_reconstruct",
        count,
    })?;

    Ok(output)
}

/// GPU-accelerated variant of [`spectral_reconstruct`].
///
/// Same logic as the CPU version but runs each per-source-channel iSTFT on the
/// GPU via [`IstftGpuBasis::gpu_istft_from_cpu`]. This eliminates 8 CPU iSTFT
/// calls (4 sources × 2 stereo channels) and replaces them with GPU kernel
/// dispatches.
///
/// Part of #1393, Stage 5 of #1370.
pub(super) fn spectral_reconstruct_gpu(
    spectral_decoded: &[f32],
    gpu_basis: &IstftGpuBasis,
    cache: &PipelineCache,
    stft_f: usize,
    stft_t: usize,
    audio_t: usize,
    mean: f32,
    std_val: f32,
) -> Result<Vec<f32>, HTDemucsError> {
    let total_ch = NUM_SOURCES * AUDIO_CHANNELS * 2; // 16
    let expected_len = total_ch * stft_f * stft_t;
    if spectral_decoded.len() != expected_len {
        return Err(HTDemucsError::DenormalizeLengthMismatch {
            actual: spectral_decoded.len(),
            expected: expected_len,
        });
    }

    let ft = stft_f * stft_t;
    let mut output = vec![0.0f32; OUTPUT_CHANNELS * audio_t];

    for source in 0..NUM_SOURCES {
        for ch in 0..AUDIO_CHANNELS {
            let real_ch = source * AUDIO_CHANNELS * 2 + ch * 2;
            let imag_ch = real_ch + 1;

            let real_start = real_ch * ft;
            let imag_start = imag_ch * ft;
            let real_slice = &spectral_decoded[real_start..real_start + ft];
            let imag_slice = &spectral_decoded[imag_start..imag_start + ft];

            // Denormalize: reverse the input mean/std normalization.
            let real_denorm: Vec<f32> = real_slice.iter().map(|&v| v * std_val + mean).collect();
            let imag_denorm: Vec<f32> = imag_slice.iter().map(|&v| v * std_val + mean).collect();

            // GPU iSTFT: upload real/imag to GPU and run Metal compute kernels.
            let waveform =
                gpu_basis.gpu_istft_from_cpu(cache, &real_denorm, &imag_denorm, stft_t, audio_t)?;

            let out_ch = source * AUDIO_CHANNELS + ch;
            let out_start = out_ch * audio_t;
            for (i, &v) in waveform.iter().enumerate() {
                output[out_start + i] += v;
            }
        }
    }

    crate::check_non_finite_err(&output, |count| HTDemucsError::NonFiniteIntermediate {
        stage: "spectral_reconstruct_gpu",
        count,
    })?;

    Ok(output)
}
