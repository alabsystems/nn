// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated iSTFT via Metal compute shaders.
//!
//! Provides [`IstftGpuBasis`] which pre-uploads the DFT basis matrices and Hann
//! window to GPU buffers, then dispatches two Metal kernels:
//! 1. Per-frame IDFT (matmul with basis)
//! 2. Windowed overlap-add + COLA normalization
//!
//! GPU iSTFT output matches CPU [`IstftBasis::istft`] within f32 tolerance.
//!
//! Part of #1393, Stage 5 of #1370.

#[path = "istft_gpu_msl.rs"]
mod msl;

use std::mem::size_of;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::context::MetalContext;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::global_metal_context;

use nn_models::istft::IstftBasis;

/// Pre-uploaded GPU buffers for iSTFT basis matrices.
///
/// Created from an [`IstftBasis`] via [`IstftGpuBasis::from_basis`].
/// Persistent — upload once, reuse across forward passes.
#[derive(Debug)]
pub struct IstftGpuBasis {
    cos_buf: MetalBuffer,
    sin_buf: MetalBuffer,
    window_buf: MetalBuffer,
    /// Cached parameters from the basis.
    n_fft: usize,
    n_bins: usize,
    hop_length: usize,
    normalized: bool,
    center: bool,
}

/// Helper to map errors to TensorError::BackendFailure with iSTFT context.
fn metal_err(e: impl std::fmt::Display) -> nn_core::TensorError {
    crate::metal_backend::metal_err(format!("GPU iSTFT: {e}"))
}

impl IstftGpuBasis {
    /// Upload iSTFT basis matrices to GPU buffers.
    ///
    /// # Errors
    ///
    /// Returns `Err` if Metal context is not initialized or buffer creation fails.
    pub fn from_basis(basis: &IstftBasis) -> nn_core::Result<Self> {
        let ctx = global_metal_context().map_err(metal_err)?;
        let params = &basis.params;

        let n_fft = params.n_fft;
        let n_bins = n_fft / 2 + 1;

        // Upload cos/sin basis matrices and Hann window.
        let cos_buf = Self::upload_f32(ctx, basis.cos_basis())?;
        let sin_buf = Self::upload_f32(ctx, basis.sin_basis())?;
        let window_buf = Self::upload_f32(ctx, basis.window())?;

        Ok(Self {
            cos_buf,
            sin_buf,
            window_buf,
            n_fft,
            n_bins,
            hop_length: params.hop_length,
            normalized: params.normalized,
            center: params.center,
        })
    }

    /// Upload f32 slice to a new GPU buffer.
    fn upload_f32(ctx: &MetalContext, data: &[f32]) -> nn_core::Result<MetalBuffer> {
        ctx.create_buffer(data).map_err(metal_err)
    }

    /// Run GPU iSTFT on the given real/imag STFT representation.
    ///
    /// # Arguments
    ///
    /// * `cache` - Pipeline cache for compiled Metal kernels.
    /// * `real_buf` - Real part, shape `[n_bins, n_frames]` row-major, as a GPU buffer.
    /// * `imag_buf` - Imaginary part, same shape, as a GPU buffer.
    /// * `n_frames` - Number of STFT frames.
    /// * `output_length` - Desired output signal length.
    ///
    /// # Returns
    ///
    /// Time-domain signal of length `output_length` (read back to CPU).
    pub fn gpu_istft(
        &self,
        cache: &PipelineCache,
        real_buf: &MetalBuffer,
        real_offset: usize,
        imag_buf: &MetalBuffer,
        imag_offset: usize,
        n_frames: usize,
        output_length: usize,
    ) -> nn_core::Result<Vec<f32>> {
        let ctx = global_metal_context().map_err(metal_err)?;
        let n_fft = self.n_fft;
        let hop = self.hop_length;

        // --- Kernel 1: Per-frame IDFT ---
        let norm: f32 = if self.normalized {
            1.0 / (n_fft as f32).sqrt()
        } else {
            1.0 / n_fft as f32
        };

        let idft_numel = n_frames
            .checked_mul(n_fft)
            .ok_or_else(|| metal_err("IDFT size overflow"))?;
        let idft_bytes = idft_numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("IDFT buffer bytes overflow"))?;
        let (frames_buf, frames_off) =
            crate::arena::arena_alloc_or_create(ctx, idft_bytes).map_err(metal_err)?;

        let idft_msl = msl::idft_msl();
        let idft_pipeline = KernelPipeline::from_msl(cache, &idft_msl, "istft_idft_f32", 1, false)
            .map_err(metal_err)?;

        // Dispatch: 2D grid [n_fft, n_frames].
        let n_bins_u32 = crate::to_u32(self.n_bins, "istft n_bins")?;
        let n_frames_u32 = crate::to_u32(n_frames, "istft n_frames")?;
        let n_fft_u32 = crate::to_u32(n_fft, "istft n_fft")?;

        // Threadgroup size: use 16x16 = 256 for 2D grid.
        let tg_x = 16u32.min(n_fft_u32);
        let tg_y = 16u32.min(n_frames_u32);

        // --- Kernel 2: Overlap-add + COLA ---
        let full_len = n_fft + n_frames.saturating_sub(1) * hop;
        let full_len_u32 = crate::to_u32(full_len, "istft full_len")?;
        let hop_u32 = crate::to_u32(hop, "istft hop")?;

        let ola_bytes = full_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("OLA buffer bytes overflow"))?;
        let (ola_buf, ola_off) =
            crate::arena::arena_alloc_or_create(ctx, ola_bytes).map_err(metal_err)?;

        let ola_msl = msl::overlap_add_msl();
        let ola_pipeline =
            KernelPipeline::from_msl(cache, &ola_msl, "istft_overlap_add_f32", 1, false)
                .map_err(metal_err)?;

        let tg_size = 256u32.min(full_len_u32);

        // Helper: encode both iSTFT kernels into any encoder-like target.
        macro_rules! encode_idft {
            ($enc:expr) => {{
                $enc.set_buffer_with_offset(0, real_buf, real_offset);
                $enc.set_buffer_with_offset(1, imag_buf, imag_offset);
                $enc.set_buffer(2, &self.cos_buf);
                $enc.set_buffer(3, &self.sin_buf);
                $enc.set_buffer_with_offset(4, &frames_buf, frames_off);
                $enc.set_bytes(5, &n_bins_u32);
                $enc.set_bytes(6, &n_frames_u32);
                $enc.set_bytes(7, &n_fft_u32);
                $enc.set_bytes(8, &norm);
                $enc.encode(
                    idft_pipeline.pipeline(),
                    [n_fft_u32, n_frames_u32, 1],
                    [tg_x, tg_y, 1],
                )
            }};
        }

        macro_rules! encode_ola {
            ($enc:expr) => {{
                $enc.set_buffer_with_offset(0, &frames_buf, frames_off);
                $enc.set_buffer(1, &self.window_buf);
                $enc.set_buffer_with_offset(2, &ola_buf, ola_off);
                $enc.set_bytes(3, &n_frames_u32);
                $enc.set_bytes(4, &n_fft_u32);
                $enc.set_bytes(5, &hop_u32);
                $enc.set_bytes(6, &full_len_u32);
                $enc.encode(
                    ola_pipeline.pipeline(),
                    [full_len_u32, 1, 1],
                    [tg_size, 1, 1],
                )
            }};
        }

        // Lazy batch (#2009): encode both iSTFT kernels into the thread-local
        // lazy batch, then flush before CPU readback.
        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> Result<(), crate::error::MetalError> {
                let enc1 = batch.new_encoder()?;
                encode_idft!(enc1)?;
                enc1.end_encoding();

                let enc2 = batch.new_encoder()?;
                encode_ola!(enc2)?;
                enc2.end_encoding();
                Ok(())
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Lazy batch (#2009): flush before CPU readback.
        crate::gpu_scope::flush()?;

        // --- Read back to CPU ---
        let raw_output = ola_buf
            .contents_at_offset::<f32>(ola_off, full_len)
            .map_err(metal_err)?;

        // Center trim (same logic as CPU iSTFT).
        let trimmed: &[f32] = if self.center {
            let trim = n_fft / 2;
            if full_len > 2 * trim {
                &raw_output[trim..full_len - trim]
            } else {
                &[]
            }
        } else {
            raw_output
        };

        // Trim or pad to output_length.
        let result = if trimmed.len() >= output_length {
            trimmed[..output_length].to_vec()
        } else {
            let mut padded = trimmed.to_vec();
            padded.resize(output_length, 0.0);
            padded
        };

        // Validate output finiteness.
        let non_finite_count = result.iter().filter(|v| !v.is_finite()).count();
        if non_finite_count > 0 {
            return Err(metal_err(format!(
                "output contains {non_finite_count} non-finite values"
            )));
        }

        Ok(result)
    }

    /// Fused polar→iSTFT: single-dispatch path from (magnitude, phase) to PCM.
    ///
    /// Combines polar-to-rectangular conversion, per-frame IDFT, windowed
    /// overlap-add, and COLA normalization into one Metal compute kernel.
    /// Eliminates 2 intermediate dispatches and 3 intermediate GPU buffers
    /// compared to the separate `gpu_polar_to_rect` + `gpu_istft` path.
    ///
    /// # Arguments
    ///
    /// * `cache` - Pipeline cache for compiled Metal kernels.
    /// * `mag_buf` - Magnitude, shape `[n_bins, n_frames]` row-major.
    /// * `mag_offset` - Byte offset into `mag_buf`.
    /// * `phase_buf` - Phase (radians), same shape.
    /// * `phase_offset` - Byte offset into `phase_buf`.
    /// * `n_frames` - Number of STFT frames.
    /// * `output_length` - Desired output signal length.
    ///
    /// # Returns
    ///
    /// Time-domain signal of length `output_length` (read back to CPU).
    ///
    /// Part of iSTFT fusion (#3351).
    pub fn gpu_istft_from_polar(
        &self,
        cache: &PipelineCache,
        mag_buf: &MetalBuffer,
        mag_offset: usize,
        phase_buf: &MetalBuffer,
        phase_offset: usize,
        n_frames: usize,
        output_length: usize,
    ) -> nn_core::Result<Vec<f32>> {
        let ctx = global_metal_context().map_err(metal_err)?;
        let n_fft = self.n_fft;
        let hop = self.hop_length;

        let norm: f32 = if self.normalized {
            1.0 / (n_fft as f32).sqrt()
        } else {
            1.0 / n_fft as f32
        };

        let full_len = n_fft + n_frames.saturating_sub(1) * hop;
        let full_len_u32 = crate::to_u32(full_len, "istft full_len")?;
        let n_bins_u32 = crate::to_u32(self.n_bins, "istft n_bins")?;
        let n_frames_u32 = crate::to_u32(n_frames, "istft n_frames")?;
        let n_fft_u32 = crate::to_u32(n_fft, "istft n_fft")?;
        let hop_u32 = crate::to_u32(hop, "istft hop")?;

        let out_bytes = full_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("fused iSTFT output buffer bytes overflow"))?;
        let (out_buf, out_off) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        let fused_msl = msl::fused_polar_istft_msl();
        let fused_pipeline = KernelPipeline::from_msl(
            cache,
            &fused_msl,
            "istft_fused_polar_f32",
            2,
            false,
        )
        .map_err(metal_err)?;

        let tg_size = 256u32.min(full_len_u32);

        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, mag_buf, mag_offset);
                enc.set_buffer_with_offset(1, phase_buf, phase_offset);
                enc.set_buffer(2, &self.cos_buf);
                enc.set_buffer(3, &self.sin_buf);
                enc.set_buffer(4, &self.window_buf);
                enc.set_buffer_with_offset(5, &out_buf, out_off);
                enc.set_bytes(6, &n_bins_u32);
                enc.set_bytes(7, &n_frames_u32);
                enc.set_bytes(8, &n_fft_u32);
                enc.set_bytes(9, &hop_u32);
                enc.set_bytes(10, &full_len_u32);
                enc.set_bytes(11, &norm);
                enc.encode(
                    fused_pipeline.pipeline(),
                    [full_len_u32, 1, 1],
                    [tg_size, 1, 1],
                )
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Flush before CPU readback.
        crate::gpu_scope::flush()?;

        // Read back and center-trim.
        let raw_output = out_buf
            .contents_at_offset::<f32>(out_off, full_len)
            .map_err(metal_err)?;

        let trimmed: &[f32] = if self.center {
            let trim = n_fft / 2;
            if full_len > 2 * trim {
                &raw_output[trim..full_len - trim]
            } else {
                &[]
            }
        } else {
            raw_output
        };

        let result = if trimmed.len() >= output_length {
            trimmed[..output_length].to_vec()
        } else {
            let mut padded = trimmed.to_vec();
            padded.resize(output_length, 0.0);
            padded
        };

        let non_finite_count = result.iter().filter(|v| !v.is_finite()).count();
        if non_finite_count > 0 {
            return Err(metal_err(format!(
                "fused iSTFT output contains {non_finite_count} non-finite values"
            )));
        }

        Ok(result)
    }

    /// Fused polar→iSTFT returning a GPU-resident `DynTensor` (no flush, no readback).
    ///
    /// Same kernel as [`gpu_istft_from_polar`] but keeps the output on GPU.
    /// The caller is responsible for flushing the lazy batch before reading
    /// the result (e.g., via `to_device(&cpu())`). Center trimming is done
    /// via byte-offset adjustment (zero-copy).
    ///
    /// Returns `[1, 1, output_length]` F32 tensor on `Device::metal()`.
    ///
    /// Part of eliminating the iSTFT CPU bridge (#3351).
    pub fn gpu_istft_from_polar_gpu(
        &self,
        cache: &PipelineCache,
        mag_buf: &MetalBuffer,
        mag_offset: usize,
        phase_buf: &MetalBuffer,
        phase_offset: usize,
        n_frames: usize,
        output_length: usize,
    ) -> nn_core::Result<DynTensor> {
        let ctx = global_metal_context().map_err(metal_err)?;
        let n_fft = self.n_fft;
        let hop = self.hop_length;

        let norm: f32 = if self.normalized {
            1.0 / (n_fft as f32).sqrt()
        } else {
            1.0 / n_fft as f32
        };

        let full_len = n_fft + n_frames.saturating_sub(1) * hop;
        let full_len_u32 = crate::to_u32(full_len, "istft full_len")?;
        let n_bins_u32 = crate::to_u32(self.n_bins, "istft n_bins")?;
        let n_frames_u32 = crate::to_u32(n_frames, "istft n_frames")?;
        let n_fft_u32 = crate::to_u32(n_fft, "istft n_fft")?;
        let hop_u32 = crate::to_u32(hop, "istft hop")?;

        let out_bytes = full_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("fused iSTFT output buffer bytes overflow"))?;
        let (out_buf, out_off) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        let fused_msl = msl::fused_polar_istft_msl();
        let fused_pipeline = KernelPipeline::from_msl(
            cache,
            &fused_msl,
            "istft_fused_polar_f32",
            2,
            false,
        )
        .map_err(metal_err)?;

        let tg_size = 256u32.min(full_len_u32);

        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                enc.set_buffer_with_offset(0, mag_buf, mag_offset);
                enc.set_buffer_with_offset(1, phase_buf, phase_offset);
                enc.set_buffer(2, &self.cos_buf);
                enc.set_buffer(3, &self.sin_buf);
                enc.set_buffer(4, &self.window_buf);
                enc.set_buffer_with_offset(5, &out_buf, out_off);
                enc.set_bytes(6, &n_bins_u32);
                enc.set_bytes(7, &n_frames_u32);
                enc.set_bytes(8, &n_fft_u32);
                enc.set_bytes(9, &hop_u32);
                enc.set_bytes(10, &full_len_u32);
                enc.set_bytes(11, &norm);
                enc.encode(
                    fused_pipeline.pipeline(),
                    [full_len_u32, 1, 1],
                    [tg_size, 1, 1],
                )
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Center trim via byte-offset adjustment (zero-copy on GPU).
        // For center=true with Kokoro params: trimmed_len == output_length
        // (full_len - n_fft == (n_frames-1)*hop == output_length).
        let (trimmed_off, trimmed_len) = if self.center {
            let trim = n_fft / 2;
            if full_len > 2 * trim {
                (out_off + trim * size_of::<f32>(), full_len - 2 * trim)
            } else {
                (out_off, 0)
            }
        } else {
            (out_off, full_len)
        };

        let final_len = trimmed_len.min(output_length);

        let storage = MetalTensorData::from_arena_alloc(out_buf, trimmed_off);
        DynTensor::from_gpu_storage(
            vec![1, 1, final_len],
            DType::F32,
            Arc::new(storage),
            Device::metal(),
        )
    }

    /// Run GPU iSTFT from CPU f32 slices (uploads real/imag to GPU first).
    ///
    /// Convenience wrapper for testing and integration — production paths
    /// should use `gpu_istft` with pre-uploaded GPU buffers.
    pub fn gpu_istft_from_cpu(
        &self,
        cache: &PipelineCache,
        real: &[f32],
        imag: &[f32],
        n_frames: usize,
        output_length: usize,
    ) -> nn_core::Result<Vec<f32>> {
        let ctx = global_metal_context().map_err(metal_err)?;

        // Validate inputs match CPU iSTFT expectations.
        let expected_len = self.n_bins * n_frames;
        if real.len() != expected_len || imag.len() != expected_len {
            return Err(nn_core::TensorError::InvalidShape(format!(
                "GPU iSTFT: expected real/imag length {expected_len} (n_bins={} * n_frames={}), \
                 got real={}, imag={}",
                self.n_bins,
                n_frames,
                real.len(),
                imag.len(),
            )));
        }

        // Check input finiteness.
        for &v in real.iter().chain(imag.iter()) {
            if !v.is_finite() {
                return Err(metal_err("input contains non-finite values"));
            }
        }

        // Upload to GPU.
        let real_buf = Self::upload_f32(ctx, real)?;
        let imag_buf = Self::upload_f32(ctx, imag)?;

        self.gpu_istft(cache, &real_buf, 0, &imag_buf, 0, n_frames, output_length)
    }
}

/// MSL source for pre-compilation: iSTFT inverse DFT kernel.
pub(crate) fn istft_idft_msl_source() -> String {
    msl::idft_msl()
}

/// MSL source for pre-compilation: iSTFT overlap-add kernel.
pub(crate) fn istft_overlap_add_msl_source() -> String {
    msl::overlap_add_msl()
}

/// MSL source for pre-compilation: fused polar→iSTFT kernel.
pub(crate) fn istft_fused_polar_msl_source() -> String {
    msl::fused_polar_istft_msl()
}

#[cfg(test)]
#[path = "istft_gpu_tests.rs"]
mod tests;

#[cfg(all(test, feature = "bench"))]
#[path = "istft_gpu_bench.rs"]
mod bench;
