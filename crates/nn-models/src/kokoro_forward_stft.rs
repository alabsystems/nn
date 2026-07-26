// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Forward STFT for Kokoro TTS harmonic source generation.
//!
//! Computes a short-time Fourier transform on DynTensors using `rustfft` for
//! the DFT computation. Used by SourceModule to transform the multi-harmonic
//! excitation signal into magnitude + phase representation for the Generator's
//! noise injection path.
//!
//! Uses FFT (Cooley-Tukey via `rustfft`) instead of Conv1d to match PyTorch's
//! `torch.stft` more closely. Conv1d-based DFT uses matrix multiply (flat
//! dot product), while both PyTorch and `rustfft` use butterfly-structured FFT.
//! This reduces atan2 phase wrapping at the ±π boundary from ~1% to ~0.002%
//! of bins (measured empirically), fixing the har_source 2π error that was
//! the sole root cause of AC1=0.2 (#2691).
//!
//! Part of #2507, #2218, #2691.

use std::f32::consts::PI;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Device, Result, TensorError};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Forward STFT using `rustfft` for FFT computation.
///
/// Pre-computes the Hann window. The FFT plan is created per-forward call
/// (planning is O(1) for repeated sizes in `FftPlanner`'s cache).
pub struct KokoroForwardStft {
    /// Hann window coefficients, length `n_fft`.
    window: Vec<f32>,
    /// FFT size.
    n_fft: usize,
    /// Hop length (stride between frames).
    hop_length: usize,
    /// Number of frequency bins = n_fft / 2 + 1.
    n_bins: usize,
}

impl KokoroForwardStft {
    /// Create a forward STFT with given parameters.
    ///
    /// # Arguments
    /// - `n_fft`: FFT size (must be even, > 0). Kokoro default: 20.
    /// - `hop_length`: Stride between frames. Kokoro default: 5 (= n_fft/4).
    /// - `_device`: Device (currently unused — FFT runs on CPU).
    pub fn new(n_fft: usize, hop_length: usize, _device: &Device) -> Result<Self> {
        if n_fft == 0 || !n_fft.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "n_fft must be even and > 0",
            });
        }
        if hop_length == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "hop_length must be > 0",
            });
        }

        let n_bins = n_fft / 2 + 1;

        // Hann window (periodic): w[k] = 0.5 * (1 - cos(2π * k / n_fft))
        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
            .collect();

        Ok(Self {
            window,
            n_fft,
            hop_length,
            n_bins,
        })
    }

    /// Number of frequency bins (n_fft / 2 + 1).
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Compute forward STFT: signal → (magnitude, phase).
    ///
    /// # Arguments
    /// - `signal`: `[B, 1, T_audio]` — single-channel audio.
    ///
    /// # Returns
    /// `(magnitude, phase)` each `[B, n_bins, n_frames]`.
    ///
    /// Magnitude is non-negative. Phase is in radians `(-π, π]`.
    pub fn forward(&self, signal: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        self.forward_inner(signal, false)
    }

    /// Compute forward STFT with center padding, matching `torch.stft(center=True)`.
    ///
    /// Reflection-pads the signal by `n_fft/2` on each side before the DFT.
    pub fn forward_center(&self, signal: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        self.forward_inner(signal, true)
    }

    fn forward_inner(&self, signal: &DynTensor, center: bool) -> Result<(DynTensor, DynTensor)> {
        let dims = signal.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }

        let padded = if center {
            let pad = self.n_bins - 1; // n_fft / 2
            signal.reflection_pad1d(pad, pad)?
        } else {
            signal.clone()
        };

        let padded_dims = padded.dims();
        let batch = padded_dims[0];
        let t_padded = padded_dims[2];

        if t_padded < self.n_fft {
            return Err(TensorError::ValueOutOfRange {
                description: "signal too short for n_fft",
            });
        }
        let n_frames = (t_padded - self.n_fft) / self.hop_length + 1;

        let signal_flat = padded.to_flat_vec::<f32>()?;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); self.n_fft];

        // Output: [B, n_bins, n_frames] stored as [B][n_bins][n_frames].
        let total = batch * self.n_bins * n_frames;
        let mut mag_flat = vec![0.0f32; total];
        let mut phase_flat = vec![0.0f32; total];

        for b in 0..batch {
            let batch_start = b * t_padded;
            for frame in 0..n_frames {
                let sig_start = batch_start + frame * self.hop_length;

                // Fill buffer with windowed signal.
                for i in 0..self.n_fft {
                    let val = signal_flat[sig_start + i];
                    fft_buf[i] = Complex::new(val * self.window[i], 0.0);
                }

                fft.process(&mut fft_buf);

                // Extract positive frequencies (DC through Nyquist).
                for (freq, c) in fft_buf.iter().enumerate().take(self.n_bins) {
                    let mag = c.re.hypot(c.im);
                    let phase = c.im.atan2(c.re);
                    let idx = (b * self.n_bins + freq) * n_frames + frame;
                    mag_flat[idx] = mag;
                    phase_flat[idx] = phase;
                }
            }
        }

        let shape = [batch, self.n_bins, n_frames];
        let dev = signal.device();
        let magnitude = DynTensor::from_vec(mag_flat, &shape, &dev)?;
        let phase = DynTensor::from_vec(phase_flat, &shape, &dev)?;

        Ok((magnitude, phase))
    }

    /// Compute forward STFT and return concatenated `[mag, phase]`.
    ///
    /// # Returns
    /// `[B, 2 * n_bins, n_frames]` — magnitude then phase channels.
    pub fn forward_cat(&self, signal: &DynTensor) -> Result<DynTensor> {
        let (magnitude, phase) = self.forward(signal)?;
        DynTensor::cat(&[&magnitude, &phase], 1)
    }

    /// Like `forward_cat` but with center padding (matching `torch.stft(center=True)`).
    pub fn forward_cat_center(&self, signal: &DynTensor) -> Result<DynTensor> {
        let (magnitude, phase) = self.forward_center(signal)?;
        DynTensor::cat(&[&magnitude, &phase], 1)
    }
}

#[cfg(test)]
#[path = "kokoro_forward_stft_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kokoro_forward_stft_kani_tests.rs"]
mod kani_proofs;
