// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Short-Time Fourier Transform computation.
//!
//! Provides STFT magnitude and phase spectrograms for 1-D signals,
//! used by the spectral comparison engine in `spectral.rs`.

use crate::error::ReftestError;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

/// Supported window functions for STFT analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WindowFn {
    /// Hann window (raised cosine). Default choice — good frequency resolution.
    #[default]
    Hann,
    /// Hamming window. Slightly lower sidelobe leakage than Hann.
    Hamming,
    /// Rectangular window (no windowing). Highest frequency leakage.
    Rectangular,
}

/// Generate a window of length `n`.
pub(crate) fn make_window(kind: WindowFn, n: usize) -> Vec<f32> {
    match kind {
        WindowFn::Hann => (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f32 / n as f32;
                0.5 * (1.0 - t.cos())
            })
            .collect(),
        WindowFn::Hamming => (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f32 / n as f32;
                0.54 - 0.46 * t.cos()
            })
            .collect(),
        WindowFn::Rectangular => vec![1.0; n],
    }
}

// ---------------------------------------------------------------------------
// STFT configuration
// ---------------------------------------------------------------------------

/// Configuration for the Short-Time Fourier Transform.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StftConfig {
    /// FFT size. Must be a power of 2. Default: 1024.
    pub n_fft: usize,
    /// Hop length between successive frames. Default: 256.
    pub hop_length: usize,
    /// Window function applied to each frame.
    pub window: WindowFn,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            window: WindowFn::default(),
        }
    }
}

impl StftConfig {
    /// Number of frequency bins: `n_fft / 2 + 1`.
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

// ---------------------------------------------------------------------------
// STFT computation
// ---------------------------------------------------------------------------

/// Compute the magnitude spectrogram of a 1-D signal.
///
/// Returns a flat `Vec<f32>` of shape `[n_freqs, n_frames]` in row-major order,
/// where `n_freqs = n_fft / 2 + 1`.
pub fn stft_magnitude(signal: &[f32], config: &StftConfig) -> Result<StftResult, ReftestError> {
    let (mag, _phase) = stft_full(signal, config)?;
    Ok(mag)
}

/// Result of an STFT computation.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Flat data in row-major `[n_freqs, n_frames]`.
    pub data: Vec<f32>,
    /// Number of frequency bins.
    pub n_freqs: usize,
    /// Number of time frames.
    pub n_frames: usize,
}

impl StftResult {
    /// Access element at `(freq, frame)`.
    #[must_use]
    pub fn get(&self, freq: usize, frame: usize) -> f32 {
        self.data[freq * self.n_frames + frame]
    }
}

/// Compute both magnitude and phase spectrograms.
///
/// Returns `(magnitude, phase)` where each is `[n_freqs, n_frames]`.
pub fn stft_full(
    signal: &[f32],
    config: &StftConfig,
) -> Result<(StftResult, StftResult), ReftestError> {
    if signal.is_empty() {
        return Err(ReftestError::EmptyTensor("spectral input".into()));
    }
    if config.n_fft == 0 || config.hop_length == 0 {
        return Err(ReftestError::SpectralConfig(
            "n_fft and hop_length must be > 0".into(),
        ));
    }
    if !config.n_fft.is_power_of_two() {
        return Err(ReftestError::SpectralConfig(
            "n_fft must be a power of 2".into(),
        ));
    }

    let n_fft = config.n_fft;
    let hop = config.hop_length;
    let n_freqs = config.n_freqs();

    // Number of frames: how many full hops fit.
    let n_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop + 1
    } else {
        // Signal shorter than FFT window — still produce 1 zero-padded frame.
        1
    };

    let window = make_window(config.window, n_fft);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut mag_data = vec![0.0f32; n_freqs * n_frames];
    let mut phase_data = vec![0.0f32; n_freqs * n_frames];
    let mut buffer = vec![Complex::new(0.0f32, 0.0f32); n_fft];

    for frame in 0..n_frames {
        let start = frame * hop;

        // Fill buffer with windowed signal, zero-pad if needed.
        for (i, sample) in buffer.iter_mut().enumerate() {
            let sig_val = if start + i < signal.len() {
                signal[start + i]
            } else {
                0.0
            };
            *sample = Complex::new(sig_val * window[i], 0.0);
        }

        fft.process(&mut buffer);

        // Extract positive frequencies (DC through Nyquist).
        for freq in 0..n_freqs {
            let c = buffer[freq];
            let mag = c.re.hypot(c.im);
            let phase = c.im.atan2(c.re);
            mag_data[freq * n_frames + frame] = mag;
            phase_data[freq * n_frames + frame] = phase;
        }
    }

    Ok((
        StftResult {
            data: mag_data,
            n_freqs,
            n_frames,
        },
        StftResult {
            data: phase_data,
            n_freqs,
            n_frames,
        },
    ))
}
