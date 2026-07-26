// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated Kokoro audio reconstruction.
//!
//! Provides [`kokoro_forward_audio_gpu`] which runs the Kokoro TTS forward pass
//! and reconstructs PCM audio via GPU iSTFT, keeping the spectrogram on GPU
//! throughout the pipeline (avoiding the GPU→CPU→GPU round-trip in the
//! CPU `forward_audio()` path).
//!
//! Part of #2230.

use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::istft_gpu::IstftGpuBasis;

use nn_core::dyn_tensor::DynTensor;
use nn_models::istft::{IstftBasis, IstftParams};
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_tts::KokoroModel;

/// Run Kokoro TTS forward pass with GPU-accelerated iSTFT audio reconstruction.
///
/// Equivalent to [`KokoroModel::forward_audio()`] but uses GPU iSTFT for the
/// spectrogram-to-waveform conversion. The spectrogram stays on GPU from the
/// Generator output through the iSTFT, avoiding the CPU readback in the
/// standard path.
///
/// # Arguments
///
/// * `model` - Kokoro model (weights already loaded).
/// * `input_ids` - `[B, T]` token indices.
/// * `style` - `[B, 256]` voice embedding.
/// * `speed` - Speaking rate multiplier (1.0 = normal).
/// * `cache` - Metal pipeline cache.
/// * `istft_cache` - Caller-owned cache for the GPU iSTFT basis. Pass
///   `&mut None` on first call; subsequent calls with the same config
///   reuse the cached basis (DFT matrices + Hann window depend only on
///   `n_fft` and `hop_length`, both fixed per `KokoroConfig`).
///
/// # Returns
///
/// `[1, 1, T_audio]` PCM audio at 24kHz.
pub fn kokoro_forward_audio_gpu(
    model: &KokoroModel,
    input_ids: &DynTensor,
    style: &DynTensor,
    speed: f32,
    cache: &PipelineCache,
    istft_cache: &mut Option<IstftGpuBasis>,
) -> Result<DynTensor, KokoroError> {
    let (magnitude, phase) = model.forward(input_ids, style, speed)?;

    // Generator outputs phase = sin(phase_raw) ∈ [-1, 1].
    // PyTorch Kokoro: S = mag * exp(j * phase) — phase used directly as radians.
    // NOT scaled by π (empirically verified: π scaling drops cosine 0.97→0.86).

    let n_bins = magnitude.dim(1).map_err(kokoro_tensor_err)?;
    let n_frames = magnitude.dim(2).map_err(kokoro_tensor_err)?;

    let n_fft = model.config().n_fft;
    let hop = n_fft / 4;
    // Center-trimmed length matches CPU kokoro_audio.rs and
    // torch.istft(center=True): (n_frames - 1) * hop. (#2706)
    let output_length = n_frames.saturating_sub(1) * hop;

    // Validate bin count.
    let expected_bins = n_fft / 2 + 1;
    if n_bins != expected_bins {
        return Err(KokoroError::IstftBinMismatch {
            actual: n_bins,
            expected: expected_bins,
            n_fft,
        });
    }

    // Reuse cached GPU iSTFT basis or build on first call (#2486).
    // Basis depends only on (n_fft, hop), both fixed per KokoroConfig.
    if istft_cache.is_none() {
        let istft_params =
            IstftParams::new(n_fft, hop, false, true).map_err(|e| kokoro_tensor_err(e.into()))?;
        let cpu_basis = IstftBasis::new(istft_params).map_err(|e| kokoro_tensor_err(e.into()))?;
        let basis = IstftGpuBasis::from_basis(&cpu_basis).map_err(kokoro_tensor_err)?;
        *istft_cache = Some(basis);
    }
    let gpu_basis = istft_cache.as_ref().ok_or_else(|| {
        kokoro_tensor_err(nn_core::TensorError::Unsupported(
            "iSTFT cache not initialized".into(),
        ))
    })?;

    // Fused polar→iSTFT: single dispatch from (magnitude, phase) → PCM.
    // Combines polar-to-rect + IDFT + overlap-add into one kernel (#3351).
    let mag_data = magnitude
        .gpu_data::<MetalTensorData>()
        .map_err(kokoro_tensor_err)?;
    let phase_data = phase
        .gpu_data::<MetalTensorData>()
        .map_err(kokoro_tensor_err)?;

    let audio_pcm = gpu_basis
        .gpu_istft_from_polar(
            cache,
            mag_data.buffer(),
            mag_data.byte_offset,
            phase_data.buffer(),
            phase_data.byte_offset,
            n_frames,
            output_length,
        )
        .map_err(kokoro_tensor_err)?;

    // Wrap as [1, 1, T_audio] DynTensor on CPU. gpu_istft_from_polar returns
    // Vec<f32> (CPU readback from fused kernel). Creating on GPU would trigger
    // a pointless CPU→GPU upload. Fix: #2634.
    let audio_len = audio_pcm.len();
    let audio = DynTensor::new(&audio_pcm, &[1, 1, audio_len], &nn_core::Device::Cpu)
        .map_err(kokoro_tensor_err)?;
    nn_core::layers::check_output_finite(&audio, "gpu_istft_output").map_err(kokoro_tensor_err)?;
    let audio = audio.clamp(-1.0, 1.0).map_err(kokoro_tensor_err)?;

    Ok(audio)
}

/// Convert TensorError to KokoroError via the transparent Tensor variant.
fn kokoro_tensor_err(e: nn_core::TensorError) -> KokoroError {
    KokoroError::Tensor(e)
}

#[cfg(test)]
#[path = "kokoro_audio_gpu_tests.rs"]
mod tests;
