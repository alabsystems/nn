// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro signal processing helpers: harmonic source, har_source building,
//! iSTFT preparation, and audio constants.
//!
//! Extracted from `kokoro_tts.rs` for 500-line compliance.
//! Part of #2507, #2218.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Result, TensorError};

// -- iSTFT reconstruction constants -------------------------------------------

/// Kokoro-82M FFT size (ISTFTNet decoder).
pub const KOKORO_N_FFT: usize = 20;
/// Kokoro-82M hop length (ISTFTNet decoder).
pub const KOKORO_HOP_LENGTH: usize = 5;
/// Kokoro-82M audio sampling rate in Hz.
pub const KOKORO_SAMPLE_RATE: usize = 24000;
/// Number of frequency bins: `n_fft / 2 + 1 = 11`.
pub const KOKORO_N_BINS: usize = KOKORO_N_FFT / 2 + 1;

// -- Harmonic source generation -----------------------------------------------

/// Harmonic source module: generates waveform from F0 pitch contour.
///
/// `f0`: `[B, 1, T]` — fundamental frequency in Hz.
/// `sampling_rate`: audio sampling rate (e.g., 24000).
/// Returns: `[B, 1, T]` harmonic source signal.
pub fn harmonic_source(f0: &DynTensor, sampling_rate: f32) -> Result<DynTensor> {
    // Phase accumulation: phase = cumsum(2π * f0 / sr)
    let scale = 2.0 * std::f64::consts::PI / f64::from(sampling_rate);
    let phase_inc = f0.mul_scalar(scale)?;
    let phase = phase_inc.cumsum(2)?;
    // Harmonic signal: sin(phase)
    phase.sin()
}

/// Build harmonic source tensor from F0 and energy predictions.
///
/// Generates harmonic signal from F0 pitch contour, expands to `n_bins`
/// frequency bins alongside energy, concatenates, and zero-pads to
/// `total_samples`. Both CPU and GPU callers use this shared function.
///
/// Returns `[B, 2*n_bins, total_samples]` tensor.
pub fn build_har_source(
    f0: &DynTensor,
    energy: &DynTensor,
    n_bins: usize,
    total_samples: usize,
    sample_rate: f32,
) -> Result<DynTensor> {
    let batch = f0.dim(0)?;
    let har = harmonic_source(f0, sample_rate)?;
    let t_har = har.dim(2)?;
    let usable = t_har.min(total_samples);

    let har_trimmed = har.narrow(2, 0, usable)?;
    let har_expanded = har_trimmed.expand([batch, n_bins, usable])?;
    let energy_trimmed = energy.narrow(2, 0, usable)?;
    let energy_expanded = energy_trimmed.expand([batch, n_bins, usable])?;

    let mut har_source = DynTensor::cat(&[&har_expanded, &energy_expanded], 1)?;
    if usable < total_samples {
        let pad = DynTensor::zeros(
            &[batch, 2 * n_bins, total_samples - usable],
            DType::F32,
            &har_source.device(),
        )?;
        har_source = DynTensor::cat(&[&har_source, &pad], 2)?;
    }
    Ok(har_source)
}

/// Build harmonic source from a pre-computed SourceModule excitation signal.
///
/// Like `build_har_source` but uses a pre-computed audio-rate signal instead of
/// generating one from F0. The signal is expanded to `n_bins` frequency bins
/// and concatenated with energy, matching the Generator's expected format.
///
/// `source_signal`: `[B, 1, T_audio]` from SourceModule (audio rate).
/// `energy`: `[B, 1, T_energy]` from F0EnergyPredictor (2× mel rate).
/// Returns `[B, 2*n_bins, total_samples]`.
pub fn build_har_from_source(
    source_signal: &DynTensor,
    energy: &DynTensor,
    n_bins: usize,
    total_samples: usize,
) -> Result<DynTensor> {
    let batch = source_signal.dim(0)?;
    let t_src = source_signal.dim(2)?;
    let t_en = energy.dim(2)?;
    let usable = t_src.min(total_samples);

    let src_trimmed = source_signal.narrow(2, 0, usable)?;
    let src_exp = src_trimmed.expand([batch, n_bins, usable])?;
    // Energy is at lower rate — trim to usable length.
    let en_usable = t_en.min(usable);
    let en_trimmed = energy.narrow(2, 0, en_usable)?;
    let en_exp = en_trimmed.expand([batch, n_bins, en_usable])?;
    // Pad energy to match source length.
    let en_padded = if en_usable < usable {
        let pad = DynTensor::zeros(
            &[batch, n_bins, usable - en_usable],
            DType::F32,
            &en_exp.device(),
        )?;
        DynTensor::cat(&[&en_exp, &pad], 2)?
    } else {
        en_exp
    };
    let mut har_source = DynTensor::cat(&[&src_exp, &en_padded], 1)?;
    if usable < total_samples {
        let pad = DynTensor::zeros(
            &[batch, 2 * n_bins, total_samples - usable],
            DType::F32,
            &har_source.device(),
        )?;
        har_source = DynTensor::cat(&[&har_source, &pad], 2)?;
    }
    Ok(har_source)
}

// -- iSTFT input preparation --------------------------------------------------

/// Split decoder output `[1, n_fft, T]` into real/imag for iSTFT.
///
/// Returns `(real, imag, n_frames)` where each is `[n_bins, n_frames]` flat.
/// First `n_fft/2` channels → real, second `n_fft/2` channels → imag,
/// both zero-padded to `n_bins` (DC/Nyquist imag = 0).
///
/// Note: `to_f32_array()` CPU extraction is architecturally necessary here —
/// the iSTFT reconstruction (DFT matmul + overlap-add) is CPU-only in the
/// uncompiled path. The compiled pipeline has a GPU iSTFT via `IstftBasis`
/// (see `istft_linear_matrix.rs` and `compiled_kokoro_bridges.rs`).
pub fn prepare_istft_input(decoder_output: &DynTensor) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let dims = decoder_output.dims();
    if dims.len() != 3 {
        return Err(TensorError::RankMismatch {
            expected: 3,
            actual: dims.len(),
        });
    }
    if dims[0] != 1 {
        return Err(TensorError::shape_mismatch(vec![1, 0, 0], dims.to_vec()));
    }
    let n_fft = dims[1];
    let n_frames = dims[2];
    let half = n_fft / 2;
    let n_bins = half + 1;
    let dec_cpu = decoder_output.to_device(&nn_core::Device::Cpu)?;
    let dec_arr = dec_cpu.to_f32_array()?;
    let flat: Vec<f32> = dec_arr.iter().copied().collect();

    let mut real = Vec::with_capacity(n_bins * n_frames);
    for f in 0..half {
        let base = f * n_frames;
        real.extend_from_slice(&flat[base..base + n_frames]);
    }
    real.resize(n_bins * n_frames, 0.0); // Nyquist pad

    let mut imag = Vec::with_capacity(n_bins * n_frames);
    for f in half..n_fft {
        let base = f * n_frames;
        imag.extend_from_slice(&flat[base..base + n_frames]);
    }
    imag.resize(n_bins * n_frames, 0.0); // Nyquist imag = 0 (real signal symmetry)

    Ok((real, imag, n_frames))
}

#[cfg(test)]
#[path = "kokoro_signal_tests.rs"]
mod tests;
