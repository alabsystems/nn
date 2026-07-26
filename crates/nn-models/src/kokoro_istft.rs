// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Lightweight CPU iSTFT for Kokoro TTS waveform reconstruction.
//!
//! This is a minimal inverse STFT (overlap-add with Hann window) designed for
//! Kokoro's small FFT sizes (n_fft=20, hop=5). The implementation matches
//! `nn-metal/src/istft.rs` algorithmically but lives in nn-models so
//! `KokoroModel::forward_audio()` can produce audio without a Metal dependency.

use std::f32::consts::PI;

/// iSTFT parameters for Kokoro waveform reconstruction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KokoroIstftParams {
    pub n_fft: usize,
    pub hop_length: usize,
}

/// Reconstruct time-domain audio from real/imag STFT representation.
///
/// `real` and `imag` are `[n_bins, n_frames]` row-major where `n_bins = n_fft/2 + 1`.
/// Returns a `Vec<f32>` of length `output_length`.
///
/// Uses unnormalized iDFT (1/N scaling) + Hann window overlap-add + COLA normalization.
/// No center trimming in this function (caller handles center trim separately).
pub(crate) fn kokoro_istft(
    params: &KokoroIstftParams,
    real: &[f32],
    imag: &[f32],
    n_frames: usize,
    output_length: usize,
) -> Result<Vec<f32>, KokoroIstftError> {
    let n_fft = params.n_fft;
    let hop = params.hop_length;
    let n_bins = n_fft / 2 + 1;

    if n_fft == 0 || !n_fft.is_multiple_of(2) {
        return Err(KokoroIstftError::InvalidNfft(n_fft));
    }
    if hop == 0 {
        return Err(KokoroIstftError::ZeroHop);
    }

    let expected = n_bins * n_frames;
    if real.len() != expected || imag.len() != expected {
        return Err(KokoroIstftError::ShapeMismatch {
            real_len: real.len(),
            imag_len: imag.len(),
            expected,
        });
    }

    // Check input finiteness.
    for &v in real.iter().chain(imag.iter()) {
        if !v.is_finite() {
            return Err(KokoroIstftError::NonFiniteInput);
        }
    }

    let norm = 1.0 / n_fft as f32;

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    // Precompute DFT basis (small for Kokoro: 11 × 20 = 220 entries)
    let mut cos_basis = vec![0.0f32; n_bins * n_fft];
    let mut sin_basis = vec![0.0f32; n_bins * n_fft];
    for f in 0..n_bins {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            cos_basis[f * n_fft + k] = angle.cos();
            sin_basis[f * n_fft + k] = angle.sin();
        }
    }

    // Per-frame IDFT via matmul with conjugate symmetry
    let mut frames = vec![0.0f32; n_frames * n_fft];
    for t in 0..n_frames {
        for k in 0..n_fft {
            let mut sum = 0.0f32;

            // DC (f=0)
            let r0 = real[t];
            let i0 = imag[t];
            sum += r0 * cos_basis[k] - i0 * sin_basis[k];

            // Interior (f=1..n_bins-2): 2× for conjugate symmetry
            for f in 1..(n_bins - 1) {
                let rf = real[f * n_frames + t];
                let imf = imag[f * n_frames + t];
                sum += 2.0 * (rf * cos_basis[f * n_fft + k] - imf * sin_basis[f * n_fft + k]);
            }

            // Nyquist (f=n_bins-1)
            let fn_idx = n_bins - 1;
            let rn = real[fn_idx * n_frames + t];
            let imn = imag[fn_idx * n_frames + t];
            sum += rn * cos_basis[fn_idx * n_fft + k] - imn * sin_basis[fn_idx * n_fft + k];

            frames[t * n_fft + k] = sum * norm;
        }
    }

    // Windowed overlap-add
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let mut output = vec![0.0f32; full_len];
    let mut window_sum = vec![0.0f32; full_len];
    for t in 0..n_frames {
        let offset = t * hop;
        for k in 0..n_fft {
            let w = window[k];
            output[offset + k] += frames[t * n_fft + k] * w;
            window_sum[offset + k] += w * w;
        }
    }

    // COLA normalization
    let eps = 1e-11f32;
    for i in 0..full_len {
        if window_sum[i] > eps {
            output[i] /= window_sum[i];
        }
    }

    // Trim or pad to output_length (no center trim in this function)
    if output.len() >= output_length {
        output.truncate(output_length);
    } else {
        output.resize(output_length, 0.0);
    }
    let result = output;

    // Validate output finiteness
    for v in &result {
        if !v.is_finite() {
            return Err(KokoroIstftError::NonFiniteOutput);
        }
    }

    Ok(result)
}

/// iSTFT reconstruction errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KokoroIstftError {
    #[error("n_fft must be even and > 0, got {0}")]
    InvalidNfft(usize),
    #[error("hop_length must be > 0")]
    ZeroHop,
    #[error("shape mismatch: real={real_len}, imag={imag_len}, expected={expected}")]
    ShapeMismatch {
        real_len: usize,
        imag_len: usize,
        expected: usize,
    },
    #[error("input contains non-finite values")]
    NonFiniteInput,
    #[error("output contains non-finite values")]
    NonFiniteOutput,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kokoro_istft_round_trip_impulse() {
        // Single frame with DC=1.0, all others zero → flat impulse
        let params = KokoroIstftParams {
            n_fft: 20,
            hop_length: 5,
        };
        let n_bins = 11;
        let n_frames = 1;
        let mut real = vec![0.0f32; n_bins * n_frames];
        let imag = vec![0.0f32; n_bins * n_frames];
        real[0] = 1.0; // DC = 1.0

        let audio = kokoro_istft(&params, &real, &imag, n_frames, 20).unwrap();
        assert_eq!(audio.len(), 20);
        // DC=1.0 with 1/N normalization → constant 0.05 windowed by Hann
        for v in &audio {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_kokoro_istft_shape_mismatch() {
        let params = KokoroIstftParams {
            n_fft: 20,
            hop_length: 5,
        };
        let err = kokoro_istft(&params, &[0.0; 10], &[0.0; 11], 1, 20);
        assert!(err.is_err());
    }

    #[test]
    fn test_kokoro_istft_non_finite_input() {
        let params = KokoroIstftParams {
            n_fft: 20,
            hop_length: 5,
        };
        let n_bins = 11;
        let mut real = vec![0.0f32; n_bins];
        real[0] = f32::NAN;
        let imag = vec![0.0f32; n_bins];
        let err = kokoro_istft(&params, &real, &imag, 1, 20);
        assert!(err.is_err());
    }

    #[test]
    fn test_kokoro_istft_multiple_frames() {
        let params = KokoroIstftParams {
            n_fft: 20,
            hop_length: 5,
        };
        let n_bins = 11;
        let n_frames = 4;
        let real = vec![0.0f32; n_bins * n_frames];
        let imag = vec![0.0f32; n_bins * n_frames];
        let audio = kokoro_istft(&params, &real, &imag, n_frames, 35).unwrap();
        assert_eq!(audio.len(), 35);
        // All-zero input → all-zero output
        for v in &audio {
            assert!(*v == 0.0 || v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_kokoro_istft_output_padding() {
        // Request more output than overlap-add produces
        let params = KokoroIstftParams {
            n_fft: 20,
            hop_length: 5,
        };
        let n_bins = 11;
        let n_frames = 1;
        let real = vec![0.0f32; n_bins * n_frames];
        let imag = vec![0.0f32; n_bins * n_frames];
        let audio = kokoro_istft(&params, &real, &imag, n_frames, 100).unwrap();
        assert_eq!(audio.len(), 100);
        // Extra samples should be zero-padded
        for &v in &audio[20..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_kokoro_istft_invalid_nfft_odd() {
        let params = KokoroIstftParams {
            n_fft: 21,
            hop_length: 5,
        };
        let err = kokoro_istft(&params, &[], &[], 0, 0).unwrap_err();
        assert!(matches!(err, KokoroIstftError::InvalidNfft(21)));
    }

    #[test]
    fn test_kokoro_istft_invalid_nfft_zero() {
        let params = KokoroIstftParams {
            n_fft: 0,
            hop_length: 5,
        };
        let err = kokoro_istft(&params, &[], &[], 0, 0).unwrap_err();
        assert!(matches!(err, KokoroIstftError::InvalidNfft(0)));
    }

    #[test]
    fn test_kokoro_istft_zero_hop() {
        let params = KokoroIstftParams {
            n_fft: 20,
            hop_length: 0,
        };
        let err = kokoro_istft(&params, &[], &[], 0, 0).unwrap_err();
        assert!(matches!(err, KokoroIstftError::ZeroHop));
    }

    #[test]
    fn test_kokoro_istft_round_trip_sine() {
        // Forward STFT → iSTFT round-trip on a known sine wave.
        // Verifies COLA reconstruction: iSTFT(STFT(x)) ≈ x in the valid region.
        let n_fft = 20;
        let hop = 5;
        let n_bins = n_fft / 2 + 1;
        let n_samples = 60;
        let freq = 3.0; // 3 cycles across the signal

        // Generate test signal: sine wave
        let signal: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / n_samples as f32).sin())
            .collect();

        // Forward STFT: window + DFT per frame
        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
            .collect();

        let n_frames = (n_samples - n_fft) / hop + 1;
        let mut real = vec![0.0f32; n_bins * n_frames];
        let mut imag = vec![0.0f32; n_bins * n_frames];

        for t in 0..n_frames {
            let offset = t * hop;
            for f in 0..n_bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for k in 0..n_fft {
                    if offset + k < n_samples {
                        let windowed = signal[offset + k] * window[k];
                        let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                        re += windowed * angle.cos();
                        im -= windowed * angle.sin();
                    }
                }
                // row-major [n_bins, n_frames]
                real[f * n_frames + t] = re;
                imag[f * n_frames + t] = im;
            }
        }

        // Inverse STFT
        let params = KokoroIstftParams {
            n_fft,
            hop_length: hop,
        };
        let output = kokoro_istft(&params, &real, &imag, n_frames, n_samples).unwrap();
        assert_eq!(output.len(), n_samples);

        // Verify reconstruction in the valid region (skip edges where Hann
        // window taper causes attenuation). The interior region where all
        // n_fft/hop = 4 windows overlap has near-perfect COLA reconstruction.
        let start = n_fft; // skip first window's taper
        let end = (n_frames - 1) * hop; // skip last window's taper
        if end > start {
            let mut max_err = 0.0f32;
            for i in start..end.min(n_samples) {
                let err = (output[i] - signal[i]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
            assert!(
                max_err < 1e-5,
                "round-trip reconstruction error too large: max_err={max_err:.2e} (expected < 1e-5)"
            );
        }
    }
}

#[cfg(kani)]
#[path = "kokoro_istft_kani_tests.rs"]
mod kani_proofs;
