// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated forward STFT via Metal compute shaders.
//!
//! Provides [`StftGpuBasis`] which pre-uploads windowed DFT basis matrices to
//! GPU buffers, then dispatches a single Metal kernel to compute magnitude and
//! phase per STFT frame. Output stays on GPU as `DynTensor` — no CPU readback.
//!
//! Eliminates the GPU→CPU→GPU roundtrip in `build_harmonic_source` where the
//! CPU `KokoroForwardStft` forced a flush of all pending GPU work to read the
//! SourceModule output, computed the FFT on CPU, then uploaded the result back
//! to GPU. With this GPU implementation, the harmonic source pipeline stays
//! entirely on GPU between the `step_regulate` 4-byte readback (#2911)
//! and the terminal `step_istft` readback — achieving the "2 sync point" target.
//!
//! Part of #2218.

#[path = "stft_gpu_msl.rs"]
mod msl;

use std::f32::consts::PI;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::global_metal_context;

/// Pre-uploaded GPU buffers for forward STFT.
///
/// Holds both the DFT-matmul basis (for legacy `forward_cat_center`) and the
/// raw Hann window (for `forward_cat_center_fft`). The FFT path eliminates
/// the phase wrapping that causes -21% amplitude regression (#2928).
///
/// Created via [`StftGpuBasis::new`]. Persistent — upload once, reuse across
/// forward passes.
#[derive(Debug)]
pub(crate) struct StftGpuBasis {
    /// Window-weighted cosine DFT basis, [n_bins × n_fft] row-major.
    windowed_cos_buf: MetalBuffer,
    /// Window-weighted sine DFT basis, [n_bins × n_fft] row-major.
    windowed_sin_buf: MetalBuffer,
    /// Raw Hann window [n_fft] for the FFT kernel (not pre-multiplied).
    window_buf: MetalBuffer,
    n_fft: usize,
    n_bins: usize,
    hop_length: usize,
}

fn metal_err(e: impl std::fmt::Display) -> TensorError {
    crate::metal_backend::metal_err(format!("GPU forward STFT: {e}"))
}

impl StftGpuBasis {
    /// Create and upload forward STFT basis to GPU.
    ///
    /// Pre-computes `window[k] * cos(2π*f*k/N)` and `window[k] * sin(2π*f*k/N)`
    /// for all (f, k) pairs and uploads to GPU buffers.
    ///
    /// # Arguments
    ///
    /// * `n_fft` — FFT size (must be even, > 0). Kokoro default: 20.
    /// * `hop_length` — Stride between frames. Kokoro default: 5.
    pub(crate) fn new(n_fft: usize, hop_length: usize) -> Result<Self> {
        if n_fft == 0 || !n_fft.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "GPU STFT: n_fft must be even and > 0",
            });
        }
        if hop_length == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "GPU STFT: hop_length must be > 0",
            });
        }

        let ctx = global_metal_context().map_err(metal_err)?;
        let n_bins = n_fft / 2 + 1;

        // Hann window: w[k] = 0.5 * (1 - cos(2π * k / n_fft))
        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
            .collect();

        // Pre-multiply DFT basis with window.
        let mut windowed_cos = Vec::with_capacity(n_bins * n_fft);
        let mut windowed_sin = Vec::with_capacity(n_bins * n_fft);

        for f in 0..n_bins {
            for (k, w) in window.iter().enumerate().take(n_fft) {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                windowed_cos.push(w * angle.cos());
                windowed_sin.push(w * angle.sin());
            }
        }

        let windowed_cos_buf = ctx.create_buffer(&windowed_cos).map_err(metal_err)?;
        let windowed_sin_buf = ctx.create_buffer(&windowed_sin).map_err(metal_err)?;
        let window_buf = ctx.create_buffer(&window).map_err(metal_err)?;

        Ok(Self {
            windowed_cos_buf,
            windowed_sin_buf,
            window_buf,
            n_fft,
            n_bins,
            hop_length,
        })
    }

    /// Forward STFT with center padding, returning concatenated `[mag, phase]`.
    ///
    /// Input: `signal` `[B, 1, T_audio]` on GPU (batch=1 for Kokoro).
    /// Output: `[B, 2*n_bins, n_frames]` on GPU — magnitude then phase channels.
    ///
    /// Center padding: reflection-pads signal by `n_fft/2` on each side,
    /// matching `torch.stft(center=True)`.
    ///
    /// All computation stays on GPU — no CPU readback, no flush.
    pub(crate) fn forward_cat_center(
        &self,
        signal: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<DynTensor> {
        let dims = signal.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        let batch = dims[0];
        if batch != 1 {
            return Err(TensorError::Unsupported(
                "GPU forward STFT: only batch=1 supported".into(),
            ));
        }

        // Center padding via reflection (GPU op — stays on GPU).
        let pad = self.n_bins - 1; // = n_fft / 2
        let padded = signal.reflection_pad1d(pad, pad)?;

        let t_padded = padded.dims()[2];
        if t_padded < self.n_fft {
            return Err(TensorError::ValueOutOfRange {
                description: "GPU STFT: signal too short for n_fft",
            });
        }
        let n_frames = (t_padded - self.n_fft) / self.hop_length + 1;

        // Get GPU buffer for the padded signal (must account for byte_offset).
        // Arena allocation and zero-copy narrow can produce non-zero offsets.
        // Without this, the kernel reads from the wrong position (#2928).
        let padded_data = padded.gpu_data::<MetalTensorData>()?;
        let padded_buf = padded_data.buffer();
        let padded_off = padded_data.byte_offset();

        let ctx = global_metal_context().map_err(metal_err)?;

        // Allocate a single contiguous buffer for [mag; phase] — eliminates
        // the DynTensor::cat dispatch by writing both halves into one buffer
        // at different offsets. Saves 1 dispatch + 1 intermediate buffer.
        let out_numel = self.n_bins * n_frames;
        let out_bytes = out_numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("output size overflow"))?;
        let combined_bytes = out_bytes
            .checked_mul(2)
            .ok_or_else(|| metal_err("combined output size overflow"))?;
        let (combined_buf, combined_off) =
            crate::arena::arena_alloc_or_create(ctx, combined_bytes).map_err(metal_err)?;
        let mag_off = combined_off;
        let phase_off = combined_off + out_bytes;

        // Compile kernel pipeline.
        let dft_msl = msl::dft_msl();
        let pipeline = KernelPipeline::from_msl(cache, &dft_msl, "stft_dft_f32", 1, false)
            .map_err(metal_err)?;

        // Dispatch parameters.
        let n_bins_u32 = crate::to_u32(self.n_bins, "stft n_bins")?;
        let n_frames_u32 = crate::to_u32(n_frames, "stft n_frames")?;
        let n_fft_u32 = crate::to_u32(self.n_fft, "stft n_fft")?;
        let hop_u32 = crate::to_u32(self.hop_length, "stft hop")?;

        let tg_x = 16u32.min(n_bins_u32);
        let tg_y = 16u32.min(n_frames_u32);

        // Encode into the lazy GPU batch — no flush triggered.
        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, padded_buf, padded_off);
                enc.set_buffer(1, &self.windowed_cos_buf);
                enc.set_buffer(2, &self.windowed_sin_buf);
                enc.set_buffer_with_offset(3, &combined_buf, mag_off);
                enc.set_buffer_with_offset(4, &combined_buf, phase_off);
                enc.set_bytes(5, &n_bins_u32);
                enc.set_bytes(6, &n_frames_u32);
                enc.set_bytes(7, &n_fft_u32);
                enc.set_bytes(8, &hop_u32);
                enc.encode(
                    pipeline.pipeline(),
                    [n_bins_u32, n_frames_u32, 1],
                    [tg_x, tg_y, 1],
                )?;
                enc.end_encoding();
                Ok(())
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Return as [B, 2*n_bins, n_frames] — magnitude in [0..n_bins, :]
        // and phase in [n_bins..2*n_bins, :]. Single contiguous buffer,
        // no cat dispatch needed.
        let combined_shape = vec![batch, 2 * self.n_bins, n_frames];
        let combined_storage =
            MetalTensorData::from_arena_alloc(combined_buf, combined_off);
        DynTensor::from_gpu_storage(
            combined_shape,
            DType::F32,
            Arc::new(combined_storage),
            Device::metal(),
        )
    }

    /// Forward STFT via GPU mixed-radix FFT (Good-Thomas PFA, n_fft=20 only).
    ///
    /// Same API as [`forward_cat_center`] but uses butterfly FFT instead of
    /// DFT-matmul. Eliminates ~4.8% phase wrapping at ±π atan2 boundary
    /// that causes -21% amplitude deficit through trained Generator weights (#2928).
    ///
    /// All computation stays on GPU — no CPU readback, no flush.
    pub(crate) fn forward_cat_center_fft(
        &self,
        signal: &DynTensor,
        cache: &PipelineCache,
    ) -> Result<DynTensor> {
        if self.n_fft != 20 {
            return Err(TensorError::Unsupported(format!(
                "GPU FFT STFT: only n_fft=20 supported (Good-Thomas 4×5), got {}",
                self.n_fft,
            )));
        }

        let dims = signal.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        let batch = dims[0];
        if batch != 1 {
            return Err(TensorError::Unsupported(
                "GPU forward STFT: only batch=1 supported".into(),
            ));
        }

        let pad = self.n_bins - 1;
        let padded = signal.reflection_pad1d(pad, pad)?;

        let t_padded = padded.dims()[2];
        if t_padded < self.n_fft {
            return Err(TensorError::ValueOutOfRange {
                description: "GPU STFT: signal too short for n_fft",
            });
        }
        let n_frames = (t_padded - self.n_fft) / self.hop_length + 1;

        let padded_data = padded.gpu_data::<MetalTensorData>()?;
        let padded_buf = padded_data.buffer();
        let padded_off = padded_data.byte_offset();

        let ctx = global_metal_context().map_err(metal_err)?;

        let out_numel = self.n_bins * n_frames;
        let out_bytes = out_numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("output size overflow"))?;

        // Allocate a single contiguous buffer for [mag; phase] — eliminates
        // the DynTensor::cat dispatch by writing both halves into one buffer
        // at different offsets. Saves 1 dispatch + 1 intermediate buffer.
        let combined_bytes = out_bytes
            .checked_mul(2)
            .ok_or_else(|| metal_err("combined output size overflow"))?;
        let (combined_buf, combined_off) =
            crate::arena::arena_alloc_or_create(ctx, combined_bytes).map_err(metal_err)?;
        let mag_off = combined_off;
        let phase_off = combined_off + out_bytes;

        let fft_msl = msl::fft_msl();
        let pipeline = KernelPipeline::from_msl(cache, &fft_msl, "stft_fft_f32", 1, false)
            .map_err(metal_err)?;

        let n_bins_u32 = crate::to_u32(self.n_bins, "stft n_bins")?;
        let n_frames_u32 = crate::to_u32(n_frames, "stft n_frames")?;
        let hop_u32 = crate::to_u32(self.hop_length, "stft hop")?;

        // 1D grid: one thread per frame.
        let tg_size = 256u32.min(n_frames_u32);

        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, padded_buf, padded_off);
                enc.set_buffer(1, &self.window_buf);
                enc.set_buffer_with_offset(2, &combined_buf, mag_off);
                enc.set_buffer_with_offset(3, &combined_buf, phase_off);
                enc.set_bytes(4, &n_bins_u32);
                enc.set_bytes(5, &n_frames_u32);
                enc.set_bytes(6, &hop_u32);
                enc.encode(pipeline.pipeline(), [n_frames_u32, 1, 1], [tg_size, 1, 1])?;
                enc.end_encoding();
                Ok(())
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Return as [B, 2*n_bins, n_frames] — magnitude in [0..n_bins, :]
        // and phase in [n_bins..2*n_bins, :]. Single contiguous buffer,
        // no cat dispatch needed.
        let combined_shape = vec![batch, 2 * self.n_bins, n_frames];
        let combined_storage =
            MetalTensorData::from_arena_alloc(combined_buf, combined_off);
        DynTensor::from_gpu_storage(
            combined_shape,
            DType::F32,
            Arc::new(combined_storage),
            Device::metal(),
        )
    }
}

/// MSL source for pre-compilation: forward STFT DFT kernel.
pub(crate) fn stft_dft_msl_source() -> String {
    msl::dft_msl()
}

/// MSL source for pre-compilation: forward STFT FFT kernel.
pub(crate) fn stft_fft_msl_source() -> String {
    msl::fft_msl()
}

#[cfg(test)]
#[path = "stft_gpu_tests.rs"]
mod tests;
