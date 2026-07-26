// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio preprocessing utilities for Whisper inference.
//!
//! Handles conversion from arbitrary audio formats to Whisper's expected
//! input: mono, 16 kHz, f32, 30-second padded chunks.
//!
//! # Pipeline
//!
//! 1. Stereo → mono (channel averaging)
//! 2. Resample to 16 kHz (linear interpolation)
//! 3. Normalize to [-1, 1]
//! 4. Pad or trim to 30 seconds (480,000 samples)

use crate::config::{N_SAMPLES, SAMPLE_RATE};
use crate::WhisperError;
use nn_core::Result;

/// Convert stereo (interleaved) audio to mono by averaging left and right channels.
///
/// Input: interleaved `[L, R, L, R, ...]` samples.
/// Output: mono samples of length `input.len() / 2`.
///
/// # Errors
///
/// Returns error if input length is odd (incomplete stereo pair).
pub fn stereo_to_mono(interleaved: &[f32]) -> Result<Vec<f32>> {
    if !interleaved.len().is_multiple_of(2) {
        return Err(WhisperError::AudioFormat {
            reason: format!(
                "stereo_to_mono: odd sample count {} (expected even for stereo)",
                interleaved.len()
            ),
        }
        .into());
    }
    Ok(interleaved
        .chunks_exact(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect())
}

/// Resample audio from `source_rate` to `target_rate` using linear interpolation.
///
/// This is a simple resampler suitable for speech (where spectral precision
/// beyond 8 kHz is not critical). For production use with music, a polyphase
/// filter would be preferred.
///
/// # Arguments
/// * `audio` - Source audio samples.
/// * `source_rate` - Source sample rate in Hz.
/// * `target_rate` - Target sample rate in Hz (typically 16000 for Whisper).
///
/// # Errors
///
/// Returns error if either rate is zero.
pub fn resample(audio: &[f32], source_rate: usize, target_rate: usize) -> Result<Vec<f32>> {
    if source_rate == 0 {
        return Err(WhisperError::AudioFormat {
            reason: "resample: source_rate must be > 0".into(),
        }
        .into());
    }
    if target_rate == 0 {
        return Err(WhisperError::AudioFormat {
            reason: "resample: target_rate must be > 0".into(),
        }
        .into());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    if source_rate == target_rate {
        return Ok(audio.to_vec());
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = ((audio.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx + 1 < audio.len() {
            output.push(audio[idx] * (1.0 - frac) + audio[idx + 1] * frac);
        } else if idx < audio.len() {
            output.push(audio[idx]);
        }
    }

    Ok(output)
}

/// Normalize audio to [-1, 1] by dividing by the peak absolute value.
///
/// If the audio is silent (all zeros or peak below `1e-10`), returns the
/// input unchanged to avoid division by near-zero.
///
/// # Arguments
/// * `audio` - Input audio samples.
///
/// # Returns
/// Normalized audio where `max(|sample|) <= 1.0`.
#[must_use]
pub fn normalize_audio(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    let peak = audio
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    // Avoid amplifying near-silence.
    if peak < 1e-10 {
        return audio.to_vec();
    }

    let scale = 1.0 / peak;
    audio.iter().map(|&s| s * scale).collect()
}

/// Pad or trim audio to exactly 30 seconds at 16 kHz (`N_SAMPLES` = 480,000).
///
/// - If shorter: zero-pad on the right (silence after speech).
/// - If longer: truncate to `N_SAMPLES`.
/// - If exactly `N_SAMPLES`: return as-is (clone).
///
/// This matches AI Provider Whisper's `pad_or_trim()` behavior.
#[must_use]
pub fn pad_or_trim(audio: &[f32]) -> Vec<f32> {
    if audio.len() >= N_SAMPLES {
        audio[..N_SAMPLES].to_vec()
    } else {
        let mut buf = vec![0.0f32; N_SAMPLES];
        buf[..audio.len()].copy_from_slice(audio);
        buf
    }
}

/// Full preprocessing pipeline: stereo detection, resampling, normalization,
/// padding.
///
/// # Arguments
/// * `audio` - Raw audio samples (mono or interleaved stereo).
/// * `channels` - Number of channels (1 = mono, 2 = stereo).
/// * `sample_rate` - Source sample rate in Hz.
///
/// # Returns
/// Mono, 16 kHz, normalized, 30-second padded audio ready for mel spectrogram.
///
/// # Errors
/// Returns error on invalid channel count or sample rate.
pub fn preprocess_audio(audio: &[f32], channels: usize, sample_rate: usize) -> Result<Vec<f32>> {
    if channels == 0 || channels > 2 {
        return Err(WhisperError::AudioFormat {
            reason: format!(
                "preprocess_audio: unsupported channel count {channels} (expected 1 or 2)"
            ),
        }
        .into());
    }

    // Step 1: Convert to mono.
    let mono = if channels == 2 {
        stereo_to_mono(audio)?
    } else {
        audio.to_vec()
    };

    // Step 2: Resample to 16 kHz.
    let resampled = resample(&mono, sample_rate, SAMPLE_RATE)?;

    // Step 3: Normalize.
    let normalized = normalize_audio(&resampled);

    // Step 4: Pad or trim to 30 seconds.
    Ok(pad_or_trim(&normalized))
}

#[cfg(test)]
#[path = "audio_processing_tests.rs"]
mod tests;
