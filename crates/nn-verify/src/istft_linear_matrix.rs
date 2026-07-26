// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! iSTFT linear weight matrix for CROWN bound propagation.
//!
//! The inverse STFT is a fully linear transform: DFT matmul, Hann windowing,
//! overlap-add, and COLA normalization are all linear operations. This module
//! precomputes the entire transform as a single weight matrix `W` such that
//! `audio = W @ [real; imag]`.
//!
//! CROWN propagation through a `LinearLayer` with this matrix is *exact* —
//! no relaxation, no bound explosion. This enables tight audio-domain bounds
//! for the Kokoro TTS pipeline.
//!
//! Part of #2916: CROWN through iSTFT — prove audio [-1,1].

use std::f32::consts::PI;

/// Error from iSTFT weight matrix construction.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IstftMatrixError {
    /// n_fft must be even and > 0.
    #[error("n_fft must be even and > 0, got {n_fft}")]
    InvalidNfft { n_fft: usize },
    /// hop_length must be > 0.
    #[error("hop_length must be > 0")]
    ZeroHop,
    /// n_frames must be > 0.
    #[error("n_frames must be > 0")]
    ZeroFrames,
    /// output_length must be > 0.
    #[error("output_length must be > 0")]
    ZeroOutput,
    /// output_length exceeds available trimmed signal.
    #[error("output_length {output_length} exceeds available signal {available}")]
    OutputTooLong {
        output_length: usize,
        available: usize,
    },
}

/// Result of building the iSTFT weight matrix.
pub struct IstftWeightMatrix {
    /// Weight matrix, row-major `[output_length, 2 * n_bins * n_frames]`.
    pub weights: Vec<f32>,
    /// Number of output audio samples (rows).
    pub output_length: usize,
    /// Input dimension: `2 * n_bins * n_frames` (real ++ imag, flattened).
    pub input_dim: usize,
}

/// Build the iSTFT linear weight matrix.
///
/// Input vector layout: `[real_0, real_1, ..., real_{n_bins*n_frames-1},
///                        imag_0, imag_1, ..., imag_{n_bins*n_frames-1}]`
/// where `real[f * n_frames + t]` is frequency bin `f`, frame `t` (row-major).
///
/// Output: audio samples `[output_length]`.
///
/// Returns weight matrix as row-major `Vec<f32>` of shape
/// `[output_length, 2 * n_bins * n_frames]`.
pub fn build_istft_weight_matrix(
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    output_length: usize,
    normalized: bool,
    center: bool,
) -> Result<IstftWeightMatrix, IstftMatrixError> {
    if n_fft == 0 || !n_fft.is_multiple_of(2) {
        return Err(IstftMatrixError::InvalidNfft { n_fft });
    }
    if hop == 0 {
        return Err(IstftMatrixError::ZeroHop);
    }
    if n_frames == 0 {
        return Err(IstftMatrixError::ZeroFrames);
    }
    if output_length == 0 {
        return Err(IstftMatrixError::ZeroOutput);
    }

    let n_bins = n_fft / 2 + 1;
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;

    let trim_left = if center { n_fft / 2 } else { 0 };
    let trim_right = if center { n_fft / 2 } else { 0 };
    let trimmed_len = full_len.saturating_sub(trim_left + trim_right);

    if output_length > trimmed_len {
        return Err(IstftMatrixError::OutputTooLong {
            output_length,
            available: trimmed_len,
        });
    }

    // Normalization factor (matches IstftBasis).
    let norm = if normalized {
        1.0 / (n_fft as f32).sqrt()
    } else {
        1.0 / n_fft as f32
    };

    // Precompute DFT basis.
    let mut cos_basis = vec![0.0f32; n_bins * n_fft];
    let mut sin_basis = vec![0.0f32; n_bins * n_fft];
    for f in 0..n_bins {
        for k in 0..n_fft {
            let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
            cos_basis[f * n_fft + k] = angle.cos();
            sin_basis[f * n_fft + k] = angle.sin();
        }
    }

    // Hann window.
    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    // COLA window_sum for the full output.
    let mut window_sum = vec![0.0f32; full_len];
    for frame in 0..n_frames {
        let offset = frame * hop;
        for k in 0..n_fft {
            window_sum[offset + k] += window[k] * window[k];
        }
    }

    let eps = 1e-11f32;
    let input_dim = 2 * n_bins * n_frames;
    let mut weights = vec![0.0f32; output_length * input_dim];

    // For each output audio sample, compute its linear dependence on each
    // input spectral coefficient.
    for out_t in 0..output_length {
        let t_full = out_t + trim_left;
        let cola = if window_sum[t_full] > eps {
            1.0 / window_sum[t_full]
        } else {
            0.0
        };

        let row_offset = out_t * input_dim;

        for frame in 0..n_frames {
            let offset = frame * hop;
            if t_full < offset || t_full >= offset + n_fft {
                continue;
            }
            let k = t_full - offset;
            let w = window[k];
            let scale = norm * w * cola;

            for f in 0..n_bins {
                // Conjugate symmetry factor: DC and Nyquist count once,
                // interior frequencies count twice.
                let sym = if f == 0 || f == n_bins - 1 { 1.0 } else { 2.0 };

                let cos_val = cos_basis[f * n_fft + k];
                let sin_val = sin_basis[f * n_fft + k];

                // real[f * n_frames + frame] contributes: sym * cos * scale
                let real_idx = f * n_frames + frame;
                weights[row_offset + real_idx] += sym * cos_val * scale;

                // imag[f * n_frames + frame] contributes: -sym * sin * scale
                let imag_idx = n_bins * n_frames + f * n_frames + frame;
                weights[row_offset + imag_idx] += -sym * sin_val * scale;
            }
        }
    }

    Ok(IstftWeightMatrix {
        weights,
        output_length,
        input_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the weight matrix produces identical output to IstftBasis::istft().
    #[test]
    fn test_matrix_matches_istft_basis() {
        use nn_models::istft::{IstftBasis, IstftParams};

        let n_fft = 20;
        let hop = 5;
        let n_frames = 10;
        let output_length = (n_frames - 1) * hop; // 45, standard Kokoro formula

        let params = IstftParams::new(n_fft, hop, false, true).unwrap();
        let basis = IstftBasis::new(params).unwrap();
        let n_bins = n_fft / 2 + 1; // 11

        // Generate deterministic test input.
        let input_len = n_bins * n_frames;
        let mut real = vec![0.0f32; input_len];
        let mut imag = vec![0.0f32; input_len];
        for i in 0..input_len {
            real[i] = (i as f32 * 0.1).sin() * 0.5;
            imag[i] = (i as f32 * 0.07 + 1.0).cos() * 0.3;
        }

        // Reference: IstftBasis::istft()
        let reference = basis.istft(&real, &imag, n_frames, output_length).unwrap();

        // Matrix path: W @ [real; imag]
        let mat =
            build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, true).unwrap();
        assert_eq!(mat.output_length, output_length);
        assert_eq!(mat.input_dim, 2 * n_bins * n_frames);

        let mut flat_input = Vec::with_capacity(mat.input_dim);
        flat_input.extend_from_slice(&real);
        flat_input.extend_from_slice(&imag);

        let mut matrix_output = vec![0.0f32; output_length];
        for row in 0..output_length {
            let mut sum = 0.0f32;
            for col in 0..mat.input_dim {
                sum += mat.weights[row * mat.input_dim + col] * flat_input[col];
            }
            matrix_output[row] = sum;
        }

        // Compare with tight tolerance (f32 arithmetic).
        let max_err = reference
            .iter()
            .zip(matrix_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "matrix output diverges from IstftBasis: max_err={max_err}"
        );
        eprintln!("iSTFT matrix vs basis max error: {max_err:.2e} (output_length={output_length})");
    }

    /// Test with Kokoro-specific parameters.
    #[test]
    fn test_kokoro_parameters() {
        let n_fft = 20;
        let hop = 5;
        let n_frames = 50; // typical short utterance

        // Kokoro: center=true, normalized=false
        let output_length = (n_frames - 1) * hop; // 245

        let mat =
            build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, true).unwrap();
        let n_bins = n_fft / 2 + 1;
        assert_eq!(mat.input_dim, 2 * n_bins * n_frames); // 2 * 11 * 50 = 1100
        assert_eq!(mat.output_length, 245);

        // Matrix should not be all-zero (sanity).
        let nonzero = mat.weights.iter().filter(|&&w| w.abs() > 1e-10).count();
        assert!(nonzero > 0, "weight matrix should have nonzero entries");
        eprintln!(
            "Kokoro iSTFT matrix: {}x{}, {} nonzero entries ({:.1}% sparsity)",
            mat.output_length,
            mat.input_dim,
            nonzero,
            100.0 * (1.0 - nonzero as f64 / (mat.output_length * mat.input_dim) as f64)
        );
    }

    /// Verify error cases.
    #[test]
    fn test_invalid_params() {
        assert!(build_istft_weight_matrix(0, 5, 10, 45, false, true).is_err());
        assert!(build_istft_weight_matrix(3, 5, 10, 45, false, true).is_err()); // odd
        assert!(build_istft_weight_matrix(20, 0, 10, 45, false, true).is_err());
        assert!(build_istft_weight_matrix(20, 5, 0, 45, false, true).is_err());
        assert!(build_istft_weight_matrix(20, 5, 10, 0, false, true).is_err());
        assert!(build_istft_weight_matrix(20, 5, 10, 10000, false, true).is_err());
        // too long
    }

    /// Verify normalized mode matches IstftBasis.
    #[test]
    fn test_normalized_mode() {
        use nn_models::istft::{IstftBasis, IstftParams};

        let n_fft = 20;
        let hop = 5;
        let n_frames = 8;
        let n_bins = n_fft / 2 + 1;
        let output_length = (n_frames - 1) * hop;

        let params = IstftParams::new(n_fft, hop, true, true).unwrap();
        let basis = IstftBasis::new(params).unwrap();

        let input_len = n_bins * n_frames;
        let mut real = vec![0.0f32; input_len];
        let mut imag = vec![0.0f32; input_len];
        for i in 0..input_len {
            real[i] = ((i as f32) * 0.13).cos() * 0.4;
            imag[i] = ((i as f32) * 0.09 + 0.5).sin() * 0.2;
        }

        let reference = basis.istft(&real, &imag, n_frames, output_length).unwrap();
        let mat =
            build_istft_weight_matrix(n_fft, hop, n_frames, output_length, true, true).unwrap();

        let mut flat_input = Vec::with_capacity(mat.input_dim);
        flat_input.extend_from_slice(&real);
        flat_input.extend_from_slice(&imag);

        let mut matrix_output = vec![0.0f32; output_length];
        for row in 0..output_length {
            let mut sum = 0.0f32;
            for col in 0..mat.input_dim {
                sum += mat.weights[row * mat.input_dim + col] * flat_input[col];
            }
            matrix_output[row] = sum;
        }

        let max_err = reference
            .iter()
            .zip(matrix_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "normalized mode: max_err={max_err}");
        eprintln!("Normalized mode max error: {max_err:.2e}");
    }
}
