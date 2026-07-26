// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU-side STFT for Silero VAD.
//!
//! Computes a short-time Fourier transform magnitude spectrogram on CPU.
//! This is intentionally not a GPU operation — the input is a single 576-sample
//! audio chunk, and the STFT is a single Conv1d with the STFT basis filter.
//! Computing on CPU avoids adding reflection padding to the kernel IR and
//! simplifies the first end-to-end model validation (Part of #761, Direction 4).
//!
//! The output is uploaded to Metal for the encoder blocks.

/// STFT parameters for Silero VAD 16kHz.
///
/// - n_fft = 256 (kernel size)
/// - hop_length = 128 (stride)
/// - n_freqs = n_fft/2 + 1 = 129
/// - The STFT basis tensor shape is `[n_fft + 2, 1, n_fft]` = `[258, 1, 256]`
///   where the first 129 rows are cosine (real) components and the next 129
///   rows are sine (imaginary) components.
#[must_use]
#[non_exhaustive]
pub struct StftParams {
    /// FFT size (kernel width for the Conv1d).
    pub n_fft: usize,
    /// Hop length (stride for the Conv1d).
    pub hop_length: usize,
    /// Number of frequency bins = n_fft/2 + 1.
    pub n_freqs: usize,
    /// Right-side reflection padding length.
    pub pad_right: usize,
}

impl StftParams {
    /// Create STFT parameters with the given FFT size and hop length.
    ///
    /// Computes derived fields: `n_freqs = n_fft / 2 + 1`, `pad_right = n_fft / 4`.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            n_freqs: n_fft / 2 + 1,
            pad_right: n_fft / 4,
        }
    }
}

/// Default STFT parameters for Silero VAD 16kHz.
impl Default for StftParams {
    fn default() -> Self {
        Self {
            n_fft: 256,
            hop_length: 128,
            n_freqs: 129,
            pad_right: 64,
        }
    }
}

/// Compute STFT magnitude spectrogram on CPU.
///
/// # Arguments
///
/// * `audio` — input audio samples, shape `[num_samples]` (e.g., 576 for
///   Silero VAD with 512 new + 64 context samples)
/// * `stft_basis` — STFT basis filter, flattened from `[n_fft+2, 1, n_fft]`.
///   First `n_freqs * n_fft` values are cosine (real) components,
///   next `n_freqs * n_fft` are sine (imaginary) components.
/// * `params` — STFT parameters (n_fft, hop_length, n_freqs, pad_right)
///
/// # Returns
///
/// Flattened magnitude spectrogram `[n_freqs, n_frames]` in row-major order.
/// For Silero VAD: `[129, 4]` = 516 floats.
///
/// # Errors
///
/// Returns `Err` if the basis size doesn't match `(n_fft + 2) * n_fft` or
/// if the padded audio length is shorter than `n_fft`.
pub fn compute_stft_magnitude(
    audio: &[f32],
    stft_basis: &[f32],
    params: &StftParams,
) -> Result<Vec<f32>, StftError> {
    // Validate n_freqs consistency: must equal n_fft/2 + 1 for real FFT.
    // A caller could construct StftParams with inconsistent n_freqs, causing
    // the magnitude split to use the wrong offset for real/imaginary components.
    let expected_n_freqs = params.n_fft / 2 + 1;
    if params.n_freqs != expected_n_freqs {
        return Err(StftError::FreqsMismatch {
            expected: expected_n_freqs,
            actual: params.n_freqs,
        });
    }

    let expected_basis_len = (params.n_fft + 2) * params.n_fft;
    if stft_basis.len() != expected_basis_len {
        return Err(StftError::BasisSizeMismatch {
            expected: expected_basis_len,
            actual: stft_basis.len(),
        });
    }

    // Step 1: Reflection pad on the right.
    // Reflection requires audio.len() >= 2 + pad_right to avoid underflow:
    // the mirror index `audio.len() - 2 - i` must stay >= 0 for i in 0..pad_right.
    let min_audio_len = 2 + params.pad_right;
    if audio.len() < min_audio_len {
        return Err(StftError::AudioTooShortForPadding {
            audio_len: audio.len(),
            min_len: min_audio_len,
        });
    }

    let padded_len = audio.len() + params.pad_right;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(audio);
    // Reflection: mirror the last `pad_right` samples (excluding the boundary).
    // For audio[..N] with pad_right=P: append audio[N-2], audio[N-3], ..., audio[N-1-P]
    for i in 0..params.pad_right {
        let reflect_idx = audio.len() - 2 - i;
        padded.push(audio[reflect_idx]);
    }

    if padded.len() < params.n_fft {
        return Err(StftError::AudioTooShort {
            padded_len: padded.len(),
            n_fft: params.n_fft,
        });
    }

    // Step 2: Conv1d with STFT basis — dot product at each hop position.
    // Output shape: [n_fft+2, n_frames] where n_frames = floor((padded_len - n_fft) / hop) + 1
    let n_filters = params.n_fft + 2; // 258 for n_fft=256
    let n_frames = (padded.len() - params.n_fft) / params.hop_length + 1;

    // Compute Conv1d: output[f][t] = sum_{k=0}^{n_fft-1} basis[f*n_fft + k] * padded[t*hop + k]
    let mut conv_out = vec![0.0f32; n_filters * n_frames];
    for f in 0..n_filters {
        let basis_offset = f * params.n_fft;
        for t in 0..n_frames {
            let audio_offset = t * params.hop_length;
            let mut sum = 0.0f32;
            for k in 0..params.n_fft {
                sum += stft_basis[basis_offset + k] * padded[audio_offset + k];
            }
            conv_out[f * n_frames + t] = sum;
        }
    }

    // Step 3: Split into real (first n_freqs rows) and imag (next n_freqs rows),
    // then compute magnitude = sqrt(real² + imag²).
    let mut magnitude = Vec::with_capacity(params.n_freqs * n_frames);
    for freq in 0..params.n_freqs {
        for t in 0..n_frames {
            let real = conv_out[freq * n_frames + t];
            let imag = conv_out[(params.n_freqs + freq) * n_frames + t];
            magnitude.push(real.hypot(imag));
        }
    }

    Ok(magnitude)
}

/// Errors from STFT computation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StftError {
    /// STFT basis tensor size doesn't match expected `(n_fft+2) * n_fft`.
    #[error("STFT basis size mismatch: expected {expected} elements, got {actual}")]
    BasisSizeMismatch { expected: usize, actual: usize },

    /// `n_freqs` doesn't match `n_fft / 2 + 1`.
    #[error("n_freqs mismatch: expected {expected} (n_fft/2+1), got {actual}")]
    FreqsMismatch { expected: usize, actual: usize },

    /// Audio (after padding) is shorter than the FFT window.
    #[error("Padded audio too short: {padded_len} samples < n_fft={n_fft}")]
    AudioTooShort { padded_len: usize, n_fft: usize },

    /// Audio too short for reflection padding.
    /// Reflection padding requires at least `2 + pad_right` samples to avoid
    /// index underflow when mirroring the trailing boundary.
    #[error("audio too short for reflection padding: {audio_len} samples < minimum {min_len}")]
    AudioTooShortForPadding { audio_len: usize, min_len: usize },
}

impl From<StftError> for nn_core::TensorError {
    fn from(e: StftError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Cpu,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

#[cfg(test)]
#[path = "stft_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "stft_overlap_add_kani_tests.rs"]
mod kani_stft_ola_proofs;

#[cfg(kani)]
#[path = "stft_signal_kani_tests.rs"]
mod kani_stft_signal_proofs;

#[cfg(kani)]
#[path = "stft_mel_kani_tests.rs"]
mod kani_stft_mel_proofs;
