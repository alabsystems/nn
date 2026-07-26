// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU cumulative sum (prefix scan) dispatch for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal_data_ops.rs` for the 500-line limit.
//! MSL kernel sources extracted to `dyn_tensor_metal_cumsum_msl.rs`.
//!
//! Implements Blelloch parallel prefix sum on Metal:
//! - axis_size <= 256: single-threadgroup (one pass)
//! - 256 < axis_size <= 65536: multi-pass (three passes via CommandBatch)
//! - axis_size > 65536: returns `Unsupported` (CPU fallback)

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_cumsum_msl.rs"]
mod msl;

#[path = "dyn_tensor_metal_cumsum_kahan_msl.rs"]
mod kahan_msl;

impl super::MetalDynBackend {
    // ===== cumsum =====
    //
    // output[i] = sum(input[0..=i]) along one axis.
    // Single-threadgroup Blelloch for axis_size <= 256, multi-pass for larger.

    /// Maximum axis size for the single-threadgroup Blelloch prefix sum.
    pub(super) const CUMSUM_BLOCK_SIZE: usize = 256;

    /// Maximum axis size supported on GPU (256 blocks × 256 elements = 65536).
    /// Larger axes fall back to CPU.
    pub(super) const CUMSUM_MAX_AXIS: usize = Self::CUMSUM_BLOCK_SIZE * Self::CUMSUM_BLOCK_SIZE;

    /// GPU-native cumulative sum (prefix scan) along one axis.
    ///
    /// - axis_size <= 256: single-threadgroup Blelloch (one pass).
    /// - 256 < axis_size <= 65536: multi-pass Blelloch (three passes).
    /// - axis_size > 65536: falls back to CPU (returns `Unsupported`).
    ///
    /// **WARNING: f32-only accumulation.** This GPU kernel accumulates in f32,
    /// while the CPU path (`cumsum_cpu`) uses f64 to match PyTorch's
    /// `torch.cumsum` behavior. For long sequences (>100 frames), sequential
    /// f32 accumulation drifts ~9.5e-6 per frame; in Kokoro SineGen this is
    /// amplified by 2*pi*300 to ~18e-3 rad phase error, causing STFT 2*pi
    /// wraps (#2691). Callers needing PyTorch-exact cumsum should use CPU
    /// dispatch or pre-accumulate in f64 before GPU transfer.
    pub(super) fn gpu_cumsum(x: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_cumsum")?;

        let shape = x.dims();
        let ndim = shape.len();

        check_dim(dim, ndim)?;

        let axis_size = shape[dim];
        if axis_size > Self::CUMSUM_MAX_AXIS {
            return Err(TensorError::Unsupported(format!(
                "gpu_cumsum: axis_size {axis_size} > max {} (use CPU)",
                Self::CUMSUM_MAX_AXIS,
            )));
        }

        if axis_size == 0 {
            return DynTensor::zeros(shape, DType::F32, &Device::metal());
        }

        // outer = product of dims before `dim`, inner = product of dims after `dim`
        let outer = checked_dim_product(&shape[..dim])?;
        let inner = checked_dim_product(&shape[dim + 1..])?;
        let total_slices =
            outer
                .checked_mul(inner)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: shape.to_vec(),
                })?;

        if total_slices == 0 {
            return DynTensor::zeros(shape, DType::F32, &Device::metal());
        }

        if axis_size <= Self::CUMSUM_BLOCK_SIZE {
            Self::gpu_cumsum_single(x, axis_size, inner, total_slices)
        } else {
            Self::gpu_cumsum_multipass(x, axis_size, inner, total_slices)
        }
    }

    /// Single-threadgroup Blelloch prefix sum (axis_size <= 256).
    fn gpu_cumsum_single(
        x: &DynTensor,
        axis_size: usize,
        inner: usize,
        total_slices: usize,
    ) -> Result<DynTensor> {
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let shape = x.dims();
        let out_shape = shape.to_vec();
        let total_elems = checked_dim_product(&out_shape)?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            let msl_src = msl::single_pass_msl();
            let pipeline = KernelPipeline::from_msl(cache, &msl_src, "cumsum_f32", 1, false)
                .map_err(metal_err)?;

            let out_bytes = total_elems.checked_mul(size_of::<f32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            // One threadgroup per slice, 256 threads per threadgroup
            let axis_size_u32 = crate::to_u32(axis_size, "cumsum axis_size")?;
            let inner_u32 = crate::to_u32(inner, "cumsum inner")?;
            let total_slices_u32 = crate::to_u32(total_slices, "cumsum total_slices")?;

            let plan = DispatchMode::PerSliceReduction {
                outer: total_slices_u32,
                reduce: axis_size_u32,
                threads: 256,
                shared_bytes: 256 * size_of::<f32>() as u32, // f32 per thread
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(vec![axis_size_u32, inner_u32]);

            pipeline
                .dispatch_buffers_with_all_offsets(
                    ctx,
                    &[&x_data.buffer],
                    &[x_data.byte_offset],
                    &out_buf,
                    out_offset,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(out_shape, DType::F32, Arc::new(storage), Device::metal())
        })
    }

    /// Multi-pass Blelloch prefix sum for axis_size > 256 (up to 65536).
    ///
    /// Three-pass algorithm using `CommandBatch` for single GPU sync:
    /// 1. **Block scan**: Each threadgroup scans a 256-element chunk, storing
    ///    per-element inclusive prefix sums and each chunk's total.
    /// 2. **Scan block sums**: Single threadgroup scans the chunk totals.
    /// 3. **Propagate**: Each element adds its chunk's scanned prefix to get
    ///    the global inclusive prefix sum.
    fn gpu_cumsum_multipass(
        x: &DynTensor,
        axis_size: usize,
        inner: usize,
        total_slices: usize,
    ) -> Result<DynTensor> {
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let shape = x.dims();
        let bs = Self::CUMSUM_BLOCK_SIZE;
        let num_blocks = axis_size.div_ceil(bs);

        let out_shape = shape.to_vec();
        let total_elems = checked_dim_product(&out_shape)?;
        let total_block_sums =
            total_slices
                .checked_mul(num_blocks)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: shape.to_vec(),
                })?;

        let ctx = Self::ctx()?;
        super::with_pipeline_cache(|cache| {
            // Compile all 3 kernels
            let pass1_src = msl::block_scan_msl(bs);
            let pass2_src = msl::scan_block_sums_msl(bs);

            let p1 = KernelPipeline::from_msl(cache, &pass1_src, "cumsum_block_scan", 1, false)
                .map_err(metal_err)?;
            let p2 =
                KernelPipeline::from_msl(cache, &pass2_src, "cumsum_scan_block_sums", 1, false)
                    .map_err(metal_err)?;
            let p3 =
                KernelPipeline::from_msl(cache, msl::PROPAGATE_MSL, "cumsum_propagate", 1, false)
                    .map_err(metal_err)?;

            // Allocate buffers (arena-aware).
            let out_bytes = total_elems.checked_mul(size_of::<f32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                }
            })?;
            let block_sum_bytes =
                total_block_sums
                    .checked_mul(size_of::<f32>())
                    .ok_or_else(|| TensorError::DimensionOverflow {
                        dims: vec![total_block_sums],
                    })?;
            let (out_buf, out_off) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;
            let out_arena_gen = crate::arena::last_alloc_generation();
            let (block_sums_buf, bs_off) =
                crate::arena::arena_alloc_or_create(ctx, block_sum_bytes).map_err(metal_err)?;
            let (scanned_sums_buf, ss_off) =
                crate::arena::arena_alloc_or_create(ctx, block_sum_bytes).map_err(metal_err)?;

            // Pre-compute all u32 constants before the encoding closure
            // (to_u32 returns TensorError, not MetalError).
            let axis_size_u32 = crate::to_u32(axis_size, "cumsum axis_size")?;
            let inner_u32 = crate::to_u32(inner, "cumsum inner")?;
            let num_blocks_u32 = crate::to_u32(num_blocks, "cumsum num_blocks")?;
            let bs_u32 = crate::to_u32(bs, "cumsum block_size")?;
            let total_slices_u32 = crate::to_u32(total_slices, "cumsum total_slices")?;
            let total_groups = total_slices * num_blocks;
            let total_groups_u32 = crate::to_u32(total_groups, "cumsum total_groups")?;
            let total_threads = total_slices * axis_size;
            let total_threads_u32 = crate::to_u32(total_threads, "cumsum total_threads")?;
            let tg = 256u32.min(total_threads_u32);
            let groups = total_threads_u32.div_ceil(tg);

            // Helper: encode all 3 passes into a CommandBatch.
            let encode_all_passes =
                |batch: &crate::dispatch::CommandBatch| -> std::result::Result<(), crate::error::MetalError> {
                    // --- Pass 1: block-level prefix scan ---
                    {
                        let enc = batch.new_encoder()?;
                        enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                        enc.set_buffer_with_offset(1, &out_buf, out_off);
                        enc.set_buffer_with_offset(2, &block_sums_buf, bs_off);
                        enc.set_bytes(3, &axis_size_u32);
                        enc.set_bytes(4, &inner_u32);
                        enc.set_bytes(5, &num_blocks_u32);
                        enc.set_threadgroup_memory_length(0, (bs * size_of::<f32>()) as u64);
                        enc.encode_threadgroups(
                            p1.pipeline(),
                            [total_groups_u32, 1, 1],
                            [bs_u32, 1, 1],
                        )?;
                        enc.end_encoding();
                    }

                    // --- Pass 2: scan block sums ---
                    {
                        let enc = batch.new_encoder()?;
                        enc.set_buffer_with_offset(0, &block_sums_buf, bs_off);
                        enc.set_buffer_with_offset(1, &scanned_sums_buf, ss_off);
                        enc.set_bytes(2, &num_blocks_u32);
                        enc.set_threadgroup_memory_length(0, (bs * size_of::<f32>()) as u64);
                        enc.encode_threadgroups(
                            p2.pipeline(),
                            [total_slices_u32, 1, 1],
                            [bs_u32, 1, 1],
                        )?;
                        enc.end_encoding();
                    }

                    // --- Pass 3: propagate scanned block sums ---
                    {
                        let enc = batch.new_encoder()?;
                        enc.set_buffer_with_offset(0, &out_buf, out_off);
                        enc.set_buffer_with_offset(1, &scanned_sums_buf, ss_off);
                        enc.set_bytes(2, &axis_size_u32);
                        enc.set_bytes(3, &inner_u32);
                        enc.set_bytes(4, &num_blocks_u32);
                        enc.set_bytes(5, &bs_u32);
                        enc.encode_threadgroups(p3.pipeline(), [groups, 1, 1], [tg, 1, 1])?;
                        enc.end_encoding();
                    }

                    Ok(())
                };

            // Lazy batch (#2009): encode into the thread-local lazy batch.
            crate::gpu_scope::get_or_create_batch()?;
            let scope_result =
                crate::gpu_scope::encode_into_lazy_batch(|batch| encode_all_passes(batch));
            match scope_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(metal_err(e)),
                Err(e) => return Err(e),
            }

            // Use out_arena_gen captured at alloc site, not last_alloc_generation()
            // which would reflect scanned_sums_buf (3rd alloc, not 1st).
            let storage = match out_arena_gen {
                Some(g) => MetalTensorData::view_arena(out_buf.alias(), out_off, g),
                None if out_off > 0 => MetalTensorData::view(out_buf.alias(), out_off),
                None => MetalTensorData::new(out_buf),
            };
            DynTensor::from_gpu_storage(out_shape, DType::F32, Arc::new(storage), Device::metal())
        })
    }

    /// Maximum axis size for Kahan-compensated sequential cumsum.
    /// Above this, fall back to the Blelloch parallel scan or CPU f64.
    pub(super) const KAHAN_MAX_AXIS: usize = 1024;

    /// Kahan-compensated f32 cumulative sum along one axis (#2909).
    ///
    /// Error bound: O(nε) vs O(n²ε) for naive f32 accumulation. Not equivalent
    /// to f64 (O(n²ε₆₄)) but sufficient for SineGen phase precision where
    /// worst-case phase error drops from ~2.3 rad (naive) to ~0.014 rad (Kahan).
    ///
    /// Uses one GPU thread per slice (sequential scan with Kahan compensation).
    /// Intended for small axis sizes (e.g. SineGen T_frames=126).
    ///
    /// Returns `Unsupported` if axis_size > `KAHAN_MAX_AXIS`.
    pub(crate) fn cumsum_kahan_gpu(x: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "cumsum_kahan_gpu")?;

        let shape = x.dims();
        let ndim = shape.len();
        check_dim(dim, ndim)?;

        let axis_size = shape[dim];
        if axis_size > Self::KAHAN_MAX_AXIS {
            return Err(TensorError::Unsupported(format!(
                "cumsum_kahan_gpu: axis_size {axis_size} > max {} (use Blelloch or CPU)",
                Self::KAHAN_MAX_AXIS,
            )));
        }

        if axis_size == 0 {
            return DynTensor::zeros(shape, DType::F32, &Device::metal());
        }

        let outer = checked_dim_product(&shape[..dim])?;
        let inner = checked_dim_product(&shape[dim + 1..])?;
        let total_slices =
            outer
                .checked_mul(inner)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: shape.to_vec(),
                })?;

        if total_slices == 0 {
            return DynTensor::zeros(shape, DType::F32, &Device::metal());
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let out_shape = shape.to_vec();
        let total_elems = checked_dim_product(&out_shape)?;
        let ctx = Self::ctx()?;

        super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(
                cache,
                kahan_msl::CUMSUM_KAHAN_F32_MSL,
                "cumsum_kahan_f32",
                1,
                false,
            )
            .map_err(metal_err)?;

            let out_bytes = total_elems.checked_mul(size_of::<f32>()).ok_or_else(|| {
                TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                }
            })?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

            let total_slices_u32 = crate::to_u32(total_slices, "kahan total_slices")?;
            let axis_size_u32 = crate::to_u32(axis_size, "kahan axis_size")?;
            let inner_u32 = crate::to_u32(inner, "kahan inner")?;

            // One thread per slice. Grid3D uses dispatchThreads (total thread count).
            let tg_size = 64u32.min(total_slices_u32);

            let plan = DispatchMode::Grid3D {
                grid: [total_slices_u32, 1, 1],
                threads: [tg_size, 1, 1],
            }
            .plan()
            .map_err(metal_err)?
            .with_output_elems(total_elems)
            .with_constants(vec![total_slices_u32, axis_size_u32, inner_u32]);

            pipeline
                .dispatch_buffers_with_all_offsets(
                    ctx,
                    &[&x_data.buffer],
                    &[x_data.byte_offset],
                    &out_buf,
                    out_offset,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(out_shape, DType::F32, Arc::new(storage), Device::metal())
        })
    }
}

/// MSL sources for pre-compilation of cumsum kernels.
pub(crate) fn cumsum_single_pass_msl_source() -> String {
    msl::single_pass_msl()
}
pub(crate) fn cumsum_propagate_msl_source() -> &'static str {
    msl::PROPAGATE_MSL
}
pub(crate) fn cumsum_block_scan_msl_source(block_size: usize) -> String {
    msl::block_scan_msl(block_size)
}
pub(crate) fn cumsum_scan_block_sums_msl_source(block_size: usize) -> String {
    msl::scan_block_sums_msl(block_size)
}

/// Maximum axis size for the single-threadgroup Blelloch prefix sum.
pub(crate) const CUMSUM_BLOCK_SIZE: usize = 256;
/// Maximum axis size supported on GPU.
pub(crate) const CUMSUM_MAX_AXIS: usize = CUMSUM_BLOCK_SIZE * CUMSUM_BLOCK_SIZE;
