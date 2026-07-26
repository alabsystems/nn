// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Differentiable audio losses for TTS training.
//!
//! Provides multi-resolution STFT loss, mel-spectrogram MSE loss, and feature
//! matching loss — all on [`TrackedTensor`] for gradient tape integration.
//!
//! These are the **training path** differentiable surrogates for the
//! non-differentiable evaluation metrics in `nn-tts-verify`. The evaluation
//! metrics (PESQ, STOI, MCD) identify WHERE quality is weak; these losses
//! provide gradients to fix it.
//!
//! References:
//! - Kong et al. 2020, "HiFi-GAN" (NeurIPS) — multi-res STFT loss
//! - Yamamoto et al. 2020, "Parallel WaveGAN" (ICASSP) — multi-res spectrogram
//! - Arik et al. 2019, "Neural Voice Cloning" (NeurIPS) — magnitude-only STFT

use crate::error::{AutodiffError, Result};
use crate::TrackedTensor;
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use std::sync::Arc;

/// Numerical stability epsilon for log and sqrt operations.
const EPS: f64 = 1e-8;

// ---------------------------------------------------------------------------
// DFT basis construction (constant, not tracked)
// ---------------------------------------------------------------------------

/// Build a real DFT basis matrix of shape `[n_bins, fft_size]`.
///
/// Each row `k` contains `cos(2π k n / N)` for n in 0..N (real part)
/// followed by `-sin(2π k n / N)` (imaginary part), but since we only
/// need magnitude, we produce separate real and imaginary matrices.
///
/// Returns `(cos_basis, sin_basis)`, each `[n_bins, fft_size]`.
fn dft_basis(fft_size: usize, device: &Device) -> Result<(Arc<TrackedTensor>, Arc<TrackedTensor>)> {
    let n_bins = fft_size / 2 + 1;
    let n = fft_size;
    let mut cos_data = Vec::with_capacity(n_bins * n);
    let mut sin_data = Vec::with_capacity(n_bins * n);

    for k in 0..n_bins {
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) * (i as f64) / (n as f64);
            cos_data.push(angle.cos() as f32);
            sin_data.push(-(angle.sin()) as f32);
        }
    }

    // Compute on CPU, then move to target device (constant leaf — no gradients).
    // Fixes nn#4582: mixed-device errors when signal is on Metal GPU.
    let cos_t = DynTensor::new(&cos_data, &[n_bins, n], &Device::Cpu)?.to_device(device)?;
    let sin_t = DynTensor::new(&sin_data, &[n_bins, n], &Device::Cpu)?.to_device(device)?;

    Ok((
        Arc::new(TrackedTensor::from_tensor(cos_t)),
        Arc::new(TrackedTensor::from_tensor(sin_t)),
    ))
}

/// Build a Hann window of length `win_size` as a constant TrackedTensor.
///
/// Computed on CPU, then moved to `device` (constant leaf — no gradients).
/// Fixes nn#4582: mixed-device errors when signal is on Metal GPU.
fn hann_window(win_size: usize, device: &Device) -> Result<Arc<TrackedTensor>> {
    let data: Vec<f32> = nn_core::audio::hann_window(win_size)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let t = DynTensor::new(&data, &[1, win_size], &Device::Cpu)?.to_device(device)?;
    Ok(Arc::new(TrackedTensor::from_tensor(t)))
}

/// Build a mel filterbank matrix of shape `[n_mels, n_bins]`.
///
/// Uses HTK-style mel scale: `mel = 2595 * log10(1 + hz / 700)`.
///
/// Computed on CPU, then moved to `device` (constant leaf — no gradients).
/// Fixes nn#4582: mixed-device errors when signal is on Metal GPU.
fn mel_filterbank(
    n_mels: usize,
    n_bins: usize,
    sample_rate: u32,
    device: &Device,
) -> Result<Arc<TrackedTensor>> {
    let sr = f64::from(sample_rate);
    let fft_size = (n_bins - 1) * 2;

    use nn_core::audio::{hz_to_mel_htk as hz_to_mel, mel_to_hz_htk as mel_to_hz};

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sr / 2.0);

    // n_mels + 2 evenly spaced points on the mel scale
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|hz| hz * fft_size as f64 / sr)
        .collect();

    let mut fb = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..n_bins {
            let kf = k as f64;
            let val = if kf >= left && kf <= center && center > left {
                ((kf - left) / (center - left)) as f32
            } else if kf > center && kf <= right && right > center {
                ((right - kf) / (right - center)) as f32
            } else {
                0.0
            };
            fb[m * n_bins + k] = val;
        }
    }

    let t = DynTensor::new(&fb, &[n_mels, n_bins], &Device::Cpu)?.to_device(device)?;
    Ok(Arc::new(TrackedTensor::from_tensor(t)))
}

// ---------------------------------------------------------------------------
// Frame extraction
// ---------------------------------------------------------------------------

/// Extract overlapping frames from a 1-D signal using unfold.
///
/// Input shape: `[length]` or `[1, length]`.
/// Output shape: `[n_frames, fft_size]`.
///
/// Uses a single `unfold(dim=0, size=fft_size, step=hop_size)` call instead
/// of O(n_frames) narrow() + cat() operations. On GPU, this is a single
/// Metal kernel dispatch instead of ~87K dispatches for typical audio lengths.
fn extract_frames(
    signal: &Arc<TrackedTensor>,
    fft_size: usize,
    hop_size: usize,
) -> Result<Arc<TrackedTensor>> {
    let dims = signal.dims();
    // Normalize to [length]
    let flat = if dims.len() == 2 && dims[0] == 1 {
        signal.squeeze(0)?
    } else if dims.len() == 1 {
        Arc::clone(signal)
    } else {
        return Err(AutodiffError::ShapeMismatch {
            expected: vec![dims.iter().product()],
            got: dims.to_vec(),
        });
    };

    let length = flat.dims()[0];
    if length < fft_size {
        return Err(AutodiffError::InvalidConfig {
            op: "extract_frames",
            reason: format!("Signal length {length} < fft_size {fft_size}"),
        });
    }

    // Single unfold replaces O(n_frames) narrow() + unsqueeze() + cat().
    // [length].unfold(0, fft_size, hop_size) -> [n_frames, fft_size]
    flat.unfold(0, fft_size, hop_size)
}

// ---------------------------------------------------------------------------
// STFT magnitude computation
// ---------------------------------------------------------------------------

/// Compute windowed STFT magnitude spectrogram.
///
/// Input: 1-D signal `[length]` or `[1, length]`.
/// Output: `[n_frames, n_bins]` where `n_bins = fft_size / 2 + 1`.
fn stft_magnitude(
    signal: &Arc<TrackedTensor>,
    fft_size: usize,
    hop_size: usize,
) -> Result<Arc<TrackedTensor>> {
    // Infer target device from signal tensor.
    let device = signal.tensor().device();

    // 1. Extract overlapping frames: [n_frames, fft_size]
    let frames = extract_frames(signal, fft_size, hop_size)?;

    // 2. Apply Hann window: element-wise multiply
    let window = hann_window(fft_size, &device)?;
    let windowed = frames.mul(&window)?; // broadcast [n_frames, fft_size] * [1, fft_size]

    // 3. DFT via matrix multiply with basis
    let (cos_basis, sin_basis) = dft_basis(fft_size, &device)?;
    // cos_basis: [n_bins, fft_size], windowed: [n_frames, fft_size]
    // We need windowed @ cos_basis^T = [n_frames, n_bins]
    let cos_basis_t = cos_basis.transpose(0, 1)?; // [fft_size, n_bins]
    let sin_basis_t = sin_basis.transpose(0, 1)?;

    let real = windowed.matmul(&cos_basis_t)?; // [n_frames, n_bins]
    let imag = windowed.matmul(&sin_basis_t)?;

    // 4. Magnitude: sqrt(real^2 + imag^2 + eps)
    let real_sq = real.sqr()?;
    let imag_sq = imag.sqr()?;
    let sum = real_sq.add(&imag_sq)?;
    let mag = sum.add_scalar(EPS)?.sqrt()?;

    Ok(mag)
}

// ---------------------------------------------------------------------------
// Public loss functions
// ---------------------------------------------------------------------------

/// Single-resolution STFT loss: spectral convergence + log spectral distance.
///
/// Returns a scalar loss value (sum of spectral convergence and log spectral
/// distance). Both components are differentiable.
///
/// - **Spectral convergence**: `|| |S_c| - |S_r| ||_F / || |S_r| ||_F`
/// - **Log spectral distance**: `mean(|log|S_c| - log|S_r||)`
pub fn stft_loss(
    candidate: &Arc<TrackedTensor>,
    reference: &Arc<TrackedTensor>,
    fft_size: usize,
    hop_size: usize,
) -> Result<Arc<TrackedTensor>> {
    let cand_mag = stft_magnitude(candidate, fft_size, hop_size)?;
    let ref_mag = stft_magnitude(reference, fft_size, hop_size)?;

    // Spectral convergence: ||ref - cand||_F / ||ref||_F
    let diff = cand_mag.sub(&ref_mag)?;
    let diff_sq = diff.sqr()?;
    // Sum over all elements (frames × bins)
    let mut diff_sum = diff_sq;
    for d in (0..diff_sum.dims().len()).rev() {
        diff_sum = diff_sum.sum_keepdim(d)?;
    }
    let ref_sq = ref_mag.sqr()?;
    let mut ref_sum = ref_sq;
    for d in (0..ref_sum.dims().len()).rev() {
        ref_sum = ref_sum.sum_keepdim(d)?;
    }
    // sc = sqrt(diff_sum / (ref_sum + eps))
    let sc = diff_sum.div(&ref_sum.add_scalar(EPS)?)?.sqrt()?;

    // Log spectral distance: mean(|log(cand_mag) - log(ref_mag)|)
    let log_cand = cand_mag.add_scalar(EPS)?.log()?;
    let log_ref = ref_mag.add_scalar(EPS)?.log()?;
    let log_diff = log_cand.sub(&log_ref)?.abs()?;
    let mut lsd = log_diff;
    for d in (0..lsd.dims().len()).rev() {
        lsd = lsd.mean_keepdim(d)?;
    }

    // Total = sc + lsd (both are scalar-ish after reduction)
    sc.add(&lsd)
}

/// Multi-resolution STFT loss.
///
/// Computes [`stft_loss`] at multiple FFT sizes and averages.
/// Standard FFT sizes: `[512, 1024, 2048]` with hop = fft_size / 4.
///
/// This is the TrackedTensor equivalent of
/// `nn_tts_verify::compute_multi_res_stft`, but differentiable.
///
/// # Arguments
///
/// * `candidate` — Synthesized audio, shape `[length]` or `[1, length]`.
/// * `reference` — Reference audio, same shape.
/// * `fft_sizes` — FFT sizes for multi-resolution analysis.
///
/// # Returns
///
/// Scalar loss (average across resolutions).
pub fn multi_res_stft_loss(
    candidate: &Arc<TrackedTensor>,
    reference: &Arc<TrackedTensor>,
    fft_sizes: &[usize],
) -> Result<Arc<TrackedTensor>> {
    if fft_sizes.is_empty() {
        return Err(AutodiffError::InvalidConfig {
            op: "multi_res_stft_loss",
            reason: "fft_sizes must not be empty".to_string(),
        });
    }

    let mut total: Option<Arc<TrackedTensor>> = None;
    for &fft_size in fft_sizes {
        let hop_size = fft_size / 4;
        let loss = stft_loss(candidate, reference, fft_size, hop_size)?;
        total = Some(match total {
            Some(acc) => acc.add(&loss)?,
            None => loss,
        });
    }

    let total = total.ok_or(AutodiffError::InvalidConfig {
        op: "multi_res_stft_loss",
        reason: "accumulator empty after non-empty fft_sizes".to_string(),
    })?;
    let n = fft_sizes.len() as f64;
    total.mul_scalar(1.0 / n)
}

/// Mel-spectrogram MSE loss.
///
/// Computes STFT magnitude → mel filterbank projection → log → MSE.
/// Standard in TTS training (Yamamoto et al. 2020).
///
/// # Arguments
///
/// * `candidate` — Synthesized audio, shape `[length]` or `[1, length]`.
/// * `reference` — Reference audio, same shape.
/// * `n_mels` — Number of mel bands. Default: 80.
/// * `fft_size` — FFT size. Default: 1024.
/// * `sample_rate` — Audio sample rate in Hz.
pub fn mel_spectrogram_loss(
    candidate: &Arc<TrackedTensor>,
    reference: &Arc<TrackedTensor>,
    n_mels: usize,
    fft_size: usize,
    sample_rate: u32,
) -> Result<Arc<TrackedTensor>> {
    let hop_size = fft_size / 4;
    let n_bins = fft_size / 2 + 1;

    let cand_mag = stft_magnitude(candidate, fft_size, hop_size)?;
    let ref_mag = stft_magnitude(reference, fft_size, hop_size)?;

    // Mel filterbank: [n_mels, n_bins] — on same device as signal
    let device = candidate.tensor().device();
    let fb = mel_filterbank(n_mels, n_bins, sample_rate, &device)?;
    let fb_t = fb.transpose(0, 1)?; // [n_bins, n_mels]

    // Project to mel: [n_frames, n_bins] @ [n_bins, n_mels] = [n_frames, n_mels]
    let cand_mel = cand_mag.matmul(&fb_t)?;
    let ref_mel = ref_mag.matmul(&fb_t)?;

    // Log mel (with eps for stability)
    let log_cand = cand_mel.add_scalar(EPS)?.log()?;
    let log_ref = ref_mel.add_scalar(EPS)?.log()?;

    // MSE between log mel spectrograms
    log_cand.mse_loss(&log_ref)
}

/// Feature matching loss: L1 distance between intermediate feature lists.
///
/// Used for discriminator feature matching in GAN-based vocoders
/// (Kumar et al. 2019, "MelGAN").
///
/// # Arguments
///
/// * `candidate_features` — Intermediate features from candidate (e.g.,
///   discriminator hidden layers).
/// * `reference_features` — Corresponding features from reference.
///
/// # Returns
///
/// Scalar loss: average L1 distance across feature layers.
pub fn feature_matching_loss(
    candidate_features: &[Arc<TrackedTensor>],
    reference_features: &[Arc<TrackedTensor>],
) -> Result<Arc<TrackedTensor>> {
    if candidate_features.is_empty() || reference_features.is_empty() {
        return Err(AutodiffError::InvalidConfig {
            op: "feature_matching_loss",
            reason: "feature lists must not be empty".to_string(),
        });
    }
    if candidate_features.len() != reference_features.len() {
        return Err(AutodiffError::ShapeMismatch {
            expected: vec![candidate_features.len()],
            got: vec![reference_features.len()],
        });
    }

    let mut total: Option<Arc<TrackedTensor>> = None;
    for (cand, refr) in candidate_features.iter().zip(reference_features.iter()) {
        let loss = cand.l1_loss(refr)?;
        total = Some(match total {
            Some(acc) => acc.add(&loss)?,
            None => loss,
        });
    }

    let total = total.ok_or(AutodiffError::InvalidConfig {
        op: "multi_feature_loss",
        reason: "accumulator empty after non-empty features".to_string(),
    })?;
    let n = candidate_features.len() as f64;
    total.mul_scalar(1.0 / n)
}

#[cfg(test)]
#[path = "audio_losses_tests.rs"]
mod tests;
