// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio preprocessing for Whisper: mel spectrogram from raw PCM.
//!
//! Implements the full pipeline: PCM → reflect-pad → STFT → power spectrogram
//! → mel filterbank → log-mel normalization. Output matches AI Provider Whisper /
//! candle-transformers `pcm_to_mel()`.
//!
//! # Mel Scale
//!
//! Uses the **Slaney** scale (librosa default, `htk=False`), which is piecewise
//! linear below 1 kHz and logarithmic above. This matches the precomputed
//! `mel_filters.npz` shipped with AI Provider Whisper.

use crate::WhisperError;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};

// -- Slaney mel scale (delegated to nn-core) --------------------------------

use nn_core::audio::{hz_to_mel_slaney as hz_to_mel, mel_to_hz_slaney as mel_to_hz};

// -- Mel filterbank -----------------------------------------------------------

/// Generate a mel filterbank matrix.
///
/// Returns a flat `[n_mels, n_freqs]` row-major matrix where
/// `n_freqs = n_fft / 2 + 1`. Uses Slaney scale with area normalization.
///
/// # Arguments
/// * `n_mels` — Number of mel frequency bins (e.g., 128 for Whisper v3).
/// * `n_fft` — FFT size (e.g., 400 for Whisper).
/// * `sample_rate` — Audio sample rate in Hz (e.g., 16000).
#[must_use]
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f64;

    // Linear frequency bins matching FFT output.
    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * sr / n_fft as f64).collect();

    // n_mels + 2 center frequencies: evenly spaced in mel domain.
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(sr / 2.0);
    let hz_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for i in 0..n_mels {
        let left = hz_points[i];
        let center = hz_points[i + 1];
        let right = hz_points[i + 2];

        // Triangular filter: rising slope left→center, falling slope center→right.
        for j in 0..n_freqs {
            let f = freqs[j];
            let rising = if center > left {
                (f - left) / (center - left)
            } else {
                0.0
            };
            let falling = if right > center {
                (right - f) / (right - center)
            } else {
                0.0
            };
            filters[i * n_freqs + j] = rising.min(falling).max(0.0) as f32;
        }

        // Slaney area normalization: 2 / (right_hz - left_hz).
        let enorm = (2.0 / (right - left)) as f32;
        for j in 0..n_freqs {
            filters[i * n_freqs + j] *= enorm;
        }
    }

    filters
}

// -- FFT primitives (extracted to audio_fft.rs) -------------------------------

#[path = "audio_fft.rs"]
mod fft;
use fft::power_spectrogram;
#[cfg(test)]
use fft::{fft_in_place, hann_window, next_power_of_2};

// -- pcm_to_mel ---------------------------------------------------------------

/// Convert raw PCM audio to a log-mel spectrogram suitable for Whisper.
///
/// # Pipeline
///
/// 1. Reflect-pad `n_fft / 2` samples on each side
/// 2. STFT with periodic Hann window → power spectrogram
/// 3. Multiply by mel filterbank matrix
/// 4. Log10 with floor at 1e-10
/// 5. Clamp to `max - 8.0` (80 dB dynamic range)
/// 6. Affine normalize: `(x + 4.0) / 4.0`
///
/// # Arguments
/// * `audio` — Raw PCM samples (f32, mono, at `sample_rate`).
/// * `mel_filters` — Flat `[n_mels, n_freqs]` filterbank from [`mel_filterbank`].
/// * `n_fft` — FFT window size (400 for Whisper).
/// * `hop_length` — Hop between STFT frames (160 for Whisper).
/// * `n_mels` — Number of mel bins (128 for Whisper v3).
///
/// # Returns
/// `DynTensor` of shape `[1, n_mels, n_frames]`.
pub fn pcm_to_mel(
    audio: &[f32],
    mel_filters: &[f32],
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> Result<DynTensor> {
    if n_fft == 0 {
        return Err(WhisperError::AudioFormat {
            reason: "pcm_to_mel: n_fft must be > 0".into(),
        }
        .into());
    }
    if hop_length == 0 {
        return Err(WhisperError::AudioFormat {
            reason: "pcm_to_mel: hop_length must be > 0".into(),
        }
        .into());
    }
    let n_freqs = n_fft / 2 + 1;
    if mel_filters.len() != n_mels * n_freqs {
        return Err(WhisperError::AudioFormat {
            reason: format!(
                "mel_filters length {} != n_mels({}) * n_freqs({})",
                mel_filters.len(),
                n_mels,
                n_freqs
            ),
        }
        .into());
    }
    if audio.is_empty() {
        return Err(WhisperError::EmptyAudio {
            stage: "pcm_to_mel",
        }
        .into());
    }
    // Reject NaN/Inf PCM samples — they would be silently absorbed by
    // the log10 floor (NaN.max(1e-10) → 1e-10 in Rust) producing a
    // valid-looking but corrupted mel spectrogram.
    let non_finite_count = audio.iter().filter(|v| !v.is_finite()).count();
    if non_finite_count > 0 {
        return Err(TensorError::NonFiniteData {
            name: "pcm_to_mel audio input".into(),
            count: non_finite_count,
        });
    }

    // Step 1: Reflect-pad n_fft/2 on each side.
    let pad = n_fft / 2;
    let padded_len = audio.len() + 2 * pad;
    let mut padded = vec![0.0f64; padded_len];
    // Left reflect: audio[1], audio[2], ..., audio[pad] (no boundary duplication).
    for i in 0..pad {
        padded[pad - 1 - i] = f64::from(audio[(i + 1).min(audio.len() - 1)]);
    }
    // Center: copy audio.
    for (i, &s) in audio.iter().enumerate() {
        padded[pad + i] = f64::from(s);
    }
    // Right reflect: mirror from end of audio.
    for i in 0..pad {
        let src_idx = audio.len().saturating_sub(2 + i);
        padded[pad + audio.len() + i] = f64::from(audio[src_idx]);
    }

    // Step 2: STFT → power spectrogram [n_frames, n_freqs].
    let power = power_spectrogram(&padded, n_fft, hop_length)?;
    let n_frames = power.len() / n_freqs;

    // Step 3: Mel filterbank multiplication → [n_mels, n_frames].
    // mel_spec[m, t] = sum_k(filters[m, k] * power[t, k])
    let mut mel_spec = vec![0.0f32; n_mels * n_frames];
    for m in 0..n_mels {
        for t in 0..n_frames {
            let mut sum = 0.0f64;
            for k in 0..n_freqs {
                sum += f64::from(mel_filters[m * n_freqs + k]) * power[t * n_freqs + k];
            }
            mel_spec[m * n_frames + t] = sum as f32;
        }
    }

    // Step 4: log10 with floor at 1e-10.
    for v in &mut mel_spec {
        *v = v.max(1e-10).log10();
    }

    // Step 5: Clamp to max - 8.0 (80 dB dynamic range).
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max_val - 8.0;
    for v in &mut mel_spec {
        *v = v.max(floor);
    }

    // Step 6: Affine normalize: (x + 4.0) / 4.0.
    for v in &mut mel_spec {
        *v = (*v + 4.0) / 4.0;
    }

    DynTensor::from_vec(mel_spec, &[1, n_mels, n_frames], &Device::Cpu)
}

/// Convenience: compute mel spectrogram with standard Whisper parameters.
///
/// Uses n_fft=400, hop_length=160, n_mels=128, sample_rate=16000.
/// Generates the filterbank internally.
///
/// Audio is zero-padded (or truncated) to exactly 30 seconds (`N_SAMPLES` =
/// 480,000 samples at 16 kHz) before mel computation, matching AI Provider Whisper's
/// preprocessing. Without this padding, short audio produces fewer encoder
/// positions than the model expects (`max_source_positions` = 1500), causing
/// distributional shift and immediate EOT in the decoder.
///
/// # Returns
/// `DynTensor` of shape `[1, 128, N_FRAMES]` where `N_FRAMES` = 3000.
pub fn whisper_mel_spectrogram(audio: &[f32]) -> Result<DynTensor> {
    whisper_mel_spectrogram_for_config(audio, super::config::NUM_MEL_BINS)
}

/// Compute mel spectrogram with configurable mel bin count.
///
/// Uses n_fft=400, hop_length=160, sample_rate=16000 (standard Whisper),
/// but accepts a custom `num_mel_bins` matching the model config.
///
/// Audio is zero-padded (or truncated) to exactly 30 seconds (`N_SAMPLES`)
/// before mel computation. See [`whisper_mel_spectrogram`] for rationale.
///
/// Use this for models with non-default mel bins (e.g., whisper-tiny uses 80,
/// large-v3-turbo uses 128).
///
/// # Returns
/// `DynTensor` of shape `[1, num_mel_bins, N_FRAMES]` where `N_FRAMES` = 3000.
/// Output is clipped to exactly `N_FRAMES` to match AI Provider Whisper's
/// `mel_spec[:, :, :N_FRAMES]` convention.
pub fn whisper_mel_spectrogram_for_config(audio: &[f32], num_mel_bins: usize) -> Result<DynTensor> {
    let padded = pad_or_trim_to_n_samples(audio);
    let filters = mel_filterbank(
        num_mel_bins,
        super::config::N_FFT,
        super::config::SAMPLE_RATE,
    );
    let mel = pcm_to_mel(
        &padded,
        &filters,
        super::config::N_FFT,
        super::config::HOP_LENGTH,
        num_mel_bins,
    )?;

    // Clip to exactly N_FRAMES (3000) to match AI Provider Whisper's
    // `mel_spec[:, :, :N_FRAMES]`. The STFT produces 3001 frames for
    // 30-second audio due to the +1 in frame count calculation:
    //   (padded_len - n_fft) / hop_length + 1 = (480400 - 400) / 160 + 1 = 3001
    // Without this clip, the encoder's stride-2 conv produces 1501 positions,
    // exceeding max_source_positions=1500 and causing an out-of-bounds error
    // in the positional embedding narrow().
    let n_frames = mel.dim(2)?;
    let target_frames = super::config::N_FRAMES;
    if n_frames > target_frames {
        mel.narrow(2, 0, target_frames)
    } else {
        Ok(mel)
    }
}

/// Pad or truncate audio to exactly `N_SAMPLES` (30 seconds at 16 kHz).
///
/// - If shorter: zero-pad on the right (silence after speech).
/// - If longer: truncate to `N_SAMPLES`.
/// - If exactly `N_SAMPLES`: return as-is (no allocation).
///
/// This matches AI Provider Whisper's `load_audio()` / `pad_or_trim()` behavior.
fn pad_or_trim_to_n_samples(audio: &[f32]) -> std::borrow::Cow<'_, [f32]> {
    use super::config::N_SAMPLES;
    if audio.len() == N_SAMPLES {
        std::borrow::Cow::Borrowed(audio)
    } else if audio.len() > N_SAMPLES {
        std::borrow::Cow::Borrowed(&audio[..N_SAMPLES])
    } else {
        let mut buf = vec![0.0f32; N_SAMPLES];
        buf[..audio.len()].copy_from_slice(audio);
        std::borrow::Cow::Owned(buf)
    }
}

// -- Standalone log-mel spectrogram (no Whisper-specific padding) -------------

/// Compute a log mel spectrogram from raw PCM audio.
///
/// Returns `Vec<Vec<f32>>` with shape `[n_mels][n_frames]` where:
/// - `n_mels` = 80 mel bands
/// - `n_frames` = `n_samples / hop_length + 1`
///
/// Uses standard Whisper parameters:
/// - 25 ms window (400 samples at 16 kHz)
/// - 10 ms hop (160 samples)
/// - 80 mel filterbanks (Slaney scale)
/// - Log transform: `log(max(1e-10, mel_energy))`
///
/// Unlike [`whisper_mel_spectrogram`], this function does NOT pad/truncate
/// to 30 seconds or clip to `N_FRAMES`. It returns the natural frame count
/// for the given audio length.
///
/// # Arguments
/// * `audio` — Raw PCM samples (f32, mono).
/// * `sample_rate` — Audio sample rate in Hz (e.g., 16000).
///
/// # Panics
///
/// Panics if `audio` is empty or contains non-finite values.
pub fn compute_log_mel_spectrogram(audio: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
    let n_mels = 80;
    let n_fft = super::config::N_FFT;
    let hop = super::config::HOP_LENGTH;
    let sr = sample_rate as usize;

    let filters = mel_filterbank(n_mels, n_fft, sr);
    let mel_tensor = pcm_to_mel(audio, &filters, n_fft, hop, n_mels)
        .expect("compute_log_mel_spectrogram: pcm_to_mel failed");

    let dims = mel_tensor.dims();
    let n_frames = dims[2];
    let flat = mel_tensor
        .to_flat_vec::<f32>()
        .expect("compute_log_mel_spectrogram: to_flat_vec failed");

    // pcm_to_mel returns normalized values: log10 + clamp + affine.
    // Undo the affine normalization to get raw log10 values:
    // normalized = (log10_val + 4.0) / 4.0
    // log10_val = normalized * 4.0 - 4.0
    let mut result = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let band: Vec<f32> = (0..n_frames)
            .map(|t| {
                let normalized = flat[m * n_frames + t];
                normalized * 4.0 - 4.0
            })
            .collect();
        result.push(band);
    }
    result
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_audio_proofs.rs"]
mod kani_audio_proofs;
