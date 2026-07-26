// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge helpers for [`CompiledKokoro`] pipeline.
//!
//! Contains `build_harmonic_source` (GPU-native, #2909) and `gpu_istft` —
//! the non-compiled stages between compiled GPU segments.
//!
//! Extracted from `compiled_kokoro_segments.rs` for 450-line compliance (#2744).
//!
//! Part of #2487, #2218, #2744.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use nn_core::layers::check_output_finite;
use nn_models::kokoro_error::KokoroError;

use crate::cache::PipelineCache;
use crate::stft_gpu::StftGpuBasis;

use super::{gpu, seg_cache_miss, CompiledKokoro, CompiledKokoroError};

impl CompiledKokoro {
    /// Build harmonic source from F0 predictions.
    ///
    /// When `SourceModule` weights are loaded, uses the full 9-harmonic SineGen +
    /// Linear + tanh excitation path (matching CPU model). Returns
    /// `MissingSourceModule` error when weights are absent (#2667).
    ///
    /// **Compiled segments (#1815 D5):** SineGen is split into two compiled
    /// segments around the eager `cumsum_kahan` barrier:
    /// - Seg5a (sinegen_pre): F0 → rad_frames + voiced mask (~7 compiled ops)
    /// - Eager: cumsum_kahan (1 dispatch, custom Metal kernel not in TraceOp)
    /// - Seg5b (sinegen_post): phase → excitation via interp + sin + linear + tanh
    ///
    /// Replaces ~25 eager SineGen + SourceModule encodings with ~15 compiled
    /// dispatches + 1 eager cumsum. Forward STFT remains eager (~12 dispatches).
    ///
    /// Part of #2487, #2523, #2785, #2218, #2928, #1815.
    pub(super) fn build_harmonic_source(
        &mut self,
        f0_gpu: &DynTensor,
        _energy_gpu: &DynTensor,
        n_fft: usize,
        upsample_rates_product: usize,
        cache: &PipelineCache,
    ) -> Result<DynTensor> {
        if self.shared.source_module.is_none() {
            return Err(KokoroError::MissingSourceModule.into());
        }

        let hop_length = n_fft / 4;
        let source_upsample = upsample_rates_product * hop_length;

        // f0: [B, 1, 2T] → [B, 2T, 1] for SineGen (channel-last).
        let f0_frames = f0_gpu.transpose(1, 2)?;
        let f0_dev = f0_frames.to_device(&gpu())?;
        let t_frames = f0_dev.dim(1)?;

        // Seg5a: compile + execute pre-cumsum (single-output: rad_frames).
        self.ensure_seg_sinegen_pre(t_frames, &f0_dev, source_upsample, cache)?;
        let rad_frames = {
            let seg = self
                .seg_sinegen_pre
                .get(t_frames)
                .ok_or_else(|| seg_cache_miss("sinegen_pre"))?;
            seg.execute_dyn_no_fence(cache, &[&f0_dev]).map_err(|e| {
                CompiledKokoroError::SegmentExecutionFailed {
                    segment: "sinegen_pre",
                    source: Box::new(e),
                }
            })?
        };

        // Eager: Kahan-compensated cumsum (1 dispatch, custom Metal kernel).
        let cum_gpu = rad_frames.cumsum_kahan(1)?;

        // Phase continuity: add carried-over terminal phase from previous
        // streaming chunk. Without this, cumsum resets to zero at each chunk
        // boundary, creating audible clicking/popping artifacts. The offset
        // is in normalized-frequency units (cycles), matching cumsum output.
        let cum_gpu = if let Some(ref offset) = self.sinegen_last_cumphase {
            cum_gpu.broadcast_add(offset)?
        } else {
            cum_gpu
        };

        // Save terminal cumulative phase for the next streaming chunk.
        // Extract the last frame: cum_gpu[:, -1, :] → [1, 1, n_ch].
        // fract() keeps values in [0, 1) to prevent precision loss over many
        // chunks (sin is periodic, so fract preserves phase continuity).
        {
            let n_ch = cum_gpu.dim(2)?;
            let last_idx = DynTensor::from_vec_u32(
                vec![(t_frames - 1) as u32],
                &[1],
                &cum_gpu.device(),
            )?;
            let last_frame = cum_gpu.index_select(&last_idx, 1)?;
            let last_frame_frac = last_frame.fract()?;
            self.sinegen_last_cumphase = Some(last_frame_frac);
            let _ = n_ch; // used only to clarify shape semantics above
        }

        // Voiced mask threshold for compiled segment.
        let sm = self
            .shared
            .source_module
            .as_ref()
            .ok_or(KokoroError::MissingSourceModule)?;
        let voiced_threshold = f64::from(sm.sine_gen().voiced_threshold());

        // Seg5b: compile + execute post-cumsum (single-output: excitation).
        // Voiced mask (unsqueeze→expand→reshape→gt→to_dtype) is now folded
        // into the compiled segment — f0_dev passed as input, threshold compiled in.
        self.ensure_seg_sinegen_post(
            t_frames, &cum_gpu, &f0_dev, source_upsample, voiced_threshold, cache,
        )?;
        let excitation = {
            let seg = self
                .seg_sinegen_post
                .get(t_frames)
                .ok_or_else(|| seg_cache_miss("sinegen_post"))?;
            seg.execute_dyn_no_fence(cache, &[&cum_gpu, &f0_dev])
                .map_err(|e| CompiledKokoroError::SegmentExecutionFailed {
                    segment: "sinegen_post",
                    source: Box::new(e),
                })?
        };
        check_output_finite(&excitation, "compiled:source_module")?;

        // excitation is already [B, 1, T_audio] — transpose folded into
        // sinegen_post compiled segment (#1815).

        // GPU mixed-radix FFT for forward STFT (#2928 D2).
        let stft_result = self.shared.stft_basis.get_or_init(|| {
            StftGpuBasis::new(n_fft, hop_length).map_err(|e| format!("STFT basis init: {e}"))
        });
        let stft_basis =
            stft_result
                .as_ref()
                .map_err(|e| CompiledKokoroError::BasisInitFailed {
                    component: "STFT",
                    source: Box::new(TensorError::Unsupported(e.clone())),
                })?;
        let har = if n_fft == 20 {
            stft_basis.forward_cat_center_fft(&excitation, cache)?
        } else {
            stft_basis.forward_cat_center(&excitation, cache)?
        };
        check_output_finite(&har, "compiled:forward_stft")?;
        Ok(har)
    }

    /// Run GPU iSTFT on magnitude + phase to produce PCM audio (GPU-resident).
    ///
    /// Uses the fused polar→iSTFT kernel that combines polar-to-rect
    /// conversion, IDFT, and overlap-add into a single Metal dispatch.
    /// Output stays on GPU — no flush or CPU readback. The caller transfers
    /// to CPU when needed (e.g., in `step_verify` via `to_device(&cpu())`).
    ///
    /// The `IstftGpuBasis` (DFT matrices + Hann window) is cached on first call
    /// since it depends only on `n_fft` and `hop_length`, both fixed per config.
    pub(super) fn gpu_istft(
        &mut self,
        magnitude: &DynTensor,
        phase: &DynTensor,
        n_fft: usize,
        cache: &PipelineCache,
    ) -> std::result::Result<DynTensor, CompiledKokoroError> {
        use crate::dyn_tensor_metal::MetalTensorData;
        use crate::istft_gpu::IstftGpuBasis;
        use nn_models::istft::{IstftBasis, IstftParams};

        let mag_dims = magnitude.dims();
        let n_frames = mag_dims[2];
        let hop = n_fft / 4;
        // Center-trimmed length matches CPU kokoro_audio.rs and
        // torch.istft(center=True): (n_frames - 1) * hop. (#2706)
        let output_length = n_frames.saturating_sub(1) * hop;

        // Shared iSTFT basis: first caller initializes via OnceLock, all
        // subsequent callers (including clone_dispatch instances) share it.
        // Depends only on (n_fft, hop), both fixed per config. (#2740)
        let istft_result = self.shared.istft_basis.get_or_init(|| {
            let istft_params = IstftParams::new(n_fft, hop, false, true)
                .map_err(|e| format!("iSTFT params: {e}"))?;
            let cpu_basis =
                IstftBasis::new(istft_params).map_err(|e| format!("iSTFT basis: {e}"))?;
            IstftGpuBasis::from_basis(&cpu_basis).map_err(|e| format!("GPU iSTFT upload: {e}"))
        });
        let gpu_basis =
            istft_result
                .as_ref()
                .map_err(|e| CompiledKokoroError::BasisInitFailed {
                    component: "iSTFT",
                    source: Box::new(TensorError::Unsupported(e.clone())),
                })?;

        // Fused polar→iSTFT: single dispatch from (magnitude, phase) → PCM.
        // GPU-resident output — no flush, no CPU readback. Center trimming
        // done via zero-copy byte-offset adjustment inside gpu_istft_from_polar_gpu.
        // Generator outputs phase = sin(phase_raw) ∈ [-1, 1].
        // PyTorch Kokoro: S = mag * exp(j * phase) — phase used directly as radians.
        // NOT scaled by π (empirically verified: π scaling drops cosine 0.97→0.86).
        let mag_data = magnitude.gpu_data::<MetalTensorData>()?;
        let phase_data = phase.gpu_data::<MetalTensorData>()?;

        gpu_basis
            .gpu_istft_from_polar_gpu(
                cache,
                mag_data.buffer(),
                mag_data.byte_offset(),
                phase_data.buffer(),
                phase_data.byte_offset(),
                n_frames,
                output_length,
            )
            .map_err(|e| CompiledKokoroError::GpuIstftFailed {
                source: Box::new(e),
            })
    }
}
