// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Inverse Short-Time Fourier Transform (iSTFT) via overlap-add.
//!
//! Reconstructs a time-domain signal from complex STFT representation
//! (real + imaginary parts). Uses DFT-via-matmul for the per-frame inverse
//! transform, matching the dvoice `overlap_add_core` implementation.
//!
//! Three dvoice models need iSTFT:
//! - HTDemucs spectral decoder (n_fft=4096, hop=1024, normalized)
//! - Kokoro-82M Generator (n_fft=20, hop=5)
//! - CosyVoice3 CausalHiFT (n_fft=16, hop=4)
//!
//! The matmul approach has no FFT library dependency and is simple to verify.
//! GPU FFT acceleration can replace the inner matmul later without API changes.

use std::f32::consts::PI;

/// Pre-computed iSTFT basis and overlap-add parameters.
#[must_use]
pub struct IstftBasis {
    /// Cosine DFT basis, shape `[n_bins, n_fft]` row-major.
    /// `cos_basis[f * n_fft + k] = cos(2 * pi * f * k / n_fft)`
    cos_basis: Vec<f32>,
    /// Sine DFT basis, shape `[n_bins, n_fft]` row-major.
    /// `sin_basis[f * n_fft + k] = sin(2 * pi * f * k / n_fft)`
    sin_basis: Vec<f32>,
    /// Hann window, shape `[n_fft]`.
    window: Vec<f32>,
    /// iSTFT parameters.
    pub params: IstftParams,
}

/// iSTFT configuration.
#[derive(Debug, Clone, Copy)]
#[must_use]
#[non_exhaustive]
pub struct IstftParams {
    /// FFT size. Must be even and > 0.
    pub n_fft: usize,
    /// Hop length (stride between frames). Must be > 0.
    pub hop_length: usize,
    /// Whether to apply `1/sqrt(N)` normalization (matching `torch.stft(normalized=True)`).
    /// If false, uses `1/N` normalization.
    pub normalized: bool,
    /// Whether to center-trim output by `n_fft/2` on each side.
    pub center: bool,
}

impl IstftParams {
    /// Create iSTFT parameters.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT size. Must be even and > 0.
    /// * `hop_length` - Hop length (stride between frames). Must be > 0.
    /// * `normalized` - Whether to apply `1/sqrt(N)` normalization.
    /// * `center` - Whether to center-trim output by `n_fft/2` on each side.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `n_fft` is 0 or odd, or `hop_length` is 0.
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        normalized: bool,
        center: bool,
    ) -> Result<Self, IstftError> {
        if n_fft == 0 || !n_fft.is_multiple_of(2) {
            return Err(IstftError::OddNfft { n_fft });
        }
        if hop_length == 0 {
            return Err(IstftError::ZeroHopLength);
        }
        Ok(Self {
            n_fft,
            hop_length,
            normalized,
            center,
        })
    }
}

/// HTDemucs defaults: n_fft=4096, hop=1024, normalized, centered.
impl Default for IstftParams {
    fn default() -> Self {
        Self {
            n_fft: 4096,
            hop_length: 1024,
            normalized: true,
            center: true,
        }
    }
}

impl IstftBasis {
    /// Create iSTFT basis for given parameters.
    ///
    /// Precomputes cos/sin DFT matrices and Hann window.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `n_fft` is 0, odd, or `hop_length` is 0.
    pub fn new(params: IstftParams) -> Result<Self, IstftError> {
        if params.n_fft == 0 || !params.n_fft.is_multiple_of(2) {
            return Err(IstftError::OddNfft {
                n_fft: params.n_fft,
            });
        }
        if params.hop_length == 0 {
            return Err(IstftError::ZeroHopLength);
        }

        let n_fft = params.n_fft;
        let n_bins = n_fft / 2 + 1;

        // Precompute DFT basis matrices.
        // cos_basis[f, k] = cos(2*pi*f*k / n_fft)
        // sin_basis[f, k] = sin(2*pi*f*k / n_fft)
        let mut cos_basis = Vec::with_capacity(n_bins * n_fft);
        let mut sin_basis = Vec::with_capacity(n_bins * n_fft);

        for f in 0..n_bins {
            for k in 0..n_fft {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                cos_basis.push(angle.cos());
                sin_basis.push(angle.sin());
            }
        }

        // Hann window: w[k] = 0.5 * (1 - cos(2*pi*k / n_fft))
        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
            .collect();

        Ok(Self {
            cos_basis,
            sin_basis,
            window,
            params,
        })
    }

    /// Number of frequency bins (`n_fft / 2 + 1`).
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.params.n_fft / 2 + 1
    }

    /// Cosine DFT basis matrix, shape `[n_bins, n_fft]` row-major.
    #[must_use]
    pub fn cos_basis(&self) -> &[f32] {
        &self.cos_basis
    }

    /// Sine DFT basis matrix, shape `[n_bins, n_fft]` row-major.
    #[must_use]
    pub fn sin_basis(&self) -> &[f32] {
        &self.sin_basis
    }

    /// Hann window, shape `[n_fft]`.
    #[must_use]
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Reconstruct time-domain signal from complex STFT representation.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part of STFT, shape `[n_bins, n_frames]` row-major (length = `n_bins * n_frames`).
    /// * `imag` - Imaginary part of STFT, same shape as `real`.
    /// * `n_frames` - Number of STFT frames.
    /// * `output_length` - Desired output length (signal is trimmed or zero-padded to this length).
    ///
    /// # Returns
    ///
    /// Time-domain signal of length `output_length`.
    ///
    /// # Errors
    ///
    /// Returns `Err` on length mismatch, shape mismatch, or non-finite output.
    pub fn istft(
        &self,
        real: &[f32],
        imag: &[f32],
        n_frames: usize,
        output_length: usize,
    ) -> Result<Vec<f32>, IstftError> {
        let n_fft = self.params.n_fft;
        let hop = self.params.hop_length;
        let n_bins = self.n_bins();

        // Validate inputs.
        if real.len() != imag.len() {
            return Err(IstftError::LengthMismatch {
                real_len: real.len(),
                imag_len: imag.len(),
            });
        }
        let expected_len = n_bins * n_frames;
        if real.len() != expected_len {
            return Err(IstftError::ShapeMismatch {
                actual: real.len(),
                n_bins,
                n_frames,
            });
        }

        // Check input finiteness.
        for &v in real.iter().chain(imag.iter()) {
            if !v.is_finite() {
                return Err(IstftError::NonFiniteInput);
            }
        }

        // Normalization factor.
        let norm = if self.params.normalized {
            1.0 / (n_fft as f32).sqrt()
        } else {
            1.0 / n_fft as f32
        };

        // Step 1: Per-frame IDFT via matmul.
        // For each frame t, reconstruct n_fft time-domain samples:
        //   frame[k] = norm * sum_{f=0}^{n_bins-1} (real[f,t]*cos[f,k] - imag[f,t]*sin[f,k])
        //
        // For frequencies f > 0 and f < n_fft/2, the DFT has conjugate symmetry,
        // so we need to add the contribution of the mirror frequencies.
        // real[n_fft - f] = real[f], imag[n_fft - f] = -imag[f]
        // This means: full_sum = dc_term + nyquist_term + 2 * sum_{f=1}^{n_bins-2}(...)
        let mut frames = vec![0.0f32; n_frames * n_fft];

        for t in 0..n_frames {
            for k in 0..n_fft {
                let mut sum = 0.0f32;

                // DC component (f=0): no mirror frequency.
                let r0 = real[t];
                let i0 = imag[t];
                sum += r0 * self.cos_basis[k] - i0 * self.sin_basis[k];

                // Interior frequencies (f=1..n_bins-2): count twice for conjugate symmetry.
                for f in 1..(n_bins - 1) {
                    let rf = real[f * n_frames + t];
                    let imf = imag[f * n_frames + t];
                    let cos_val = self.cos_basis[f * n_fft + k];
                    let sin_val = self.sin_basis[f * n_fft + k];
                    sum += 2.0 * (rf * cos_val - imf * sin_val);
                }

                // Nyquist component (f=n_fft/2): no mirror frequency.
                let fn_idx = n_bins - 1;
                let rn = real[fn_idx * n_frames + t];
                let imn = imag[fn_idx * n_frames + t];
                sum += rn * self.cos_basis[fn_idx * n_fft + k]
                    - imn * self.sin_basis[fn_idx * n_fft + k];

                frames[t * n_fft + k] = sum * norm;
            }
        }

        // Step 2: Windowed overlap-add.
        let full_len = n_fft + (n_frames.saturating_sub(1)) * hop;
        let mut output = vec![0.0f32; full_len];
        let mut window_sum = vec![0.0f32; full_len];

        for t in 0..n_frames {
            let offset = t * hop;
            for k in 0..n_fft {
                let w = self.window[k];
                output[offset + k] += frames[t * n_fft + k] * w;
                window_sum[offset + k] += w * w;
            }
        }

        // Step 3: COLA normalization.
        let eps = 1e-11f32;
        for i in 0..full_len {
            if window_sum[i] > eps {
                output[i] /= window_sum[i];
            }
        }

        // Optional center trim.
        let trimmed = if self.params.center {
            let trim = n_fft / 2;
            if full_len > 2 * trim {
                output[trim..full_len - trim].to_vec()
            } else {
                // Signal too short for center trimming — return empty.
                Vec::new()
            }
        } else {
            output
        };

        // Trim or pad to output_length.
        let result = if trimmed.len() >= output_length {
            trimmed[..output_length].to_vec()
        } else {
            let mut padded = trimmed;
            padded.resize(output_length, 0.0);
            padded
        };

        // Validate output finiteness.
        for v in &result {
            if !v.is_finite() {
                return Err(IstftError::NonFiniteOutput);
            }
        }

        Ok(result)
    }
}

/// Errors from iSTFT computation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IstftError {
    /// Real and imaginary arrays have different lengths.
    #[error("real and imag length mismatch: real={real_len}, imag={imag_len}")]
    LengthMismatch { real_len: usize, imag_len: usize },

    /// Data length doesn't match expected `n_bins * n_frames`.
    #[error("data length {actual} doesn't match n_bins={n_bins} * n_frames={n_frames}")]
    ShapeMismatch {
        actual: usize,
        n_bins: usize,
        n_frames: usize,
    },

    /// n_fft must be even and > 0.
    #[error("n_fft must be even and > 0, got {n_fft}")]
    OddNfft { n_fft: usize },

    /// hop_length must be > 0.
    #[error("hop_length must be > 0")]
    ZeroHopLength,

    /// Input contains non-finite values.
    #[error("input contains non-finite values")]
    NonFiniteInput,

    /// Output contains non-finite values.
    #[error("output contains non-finite values")]
    NonFiniteOutput,
}

impl From<IstftError> for nn_core::TensorError {
    fn from(e: IstftError) -> Self {
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
#[path = "istft_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "istft_kani_tests.rs"]
mod kani_proofs;
