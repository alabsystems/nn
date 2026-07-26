// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simdgroup_matrix GEMM kernel for Apple Silicon with adaptive tile selection.
//!
//! Uses `simdgroup_matrix<float, 8, 8>` hardware multiply-accumulate with
//! shape-aware tile routing (#3479):
//!
//! | Config | BM×BN | BK | Output/TG | When |
//! |--------|-------|----|-----------|------|
//! | Small  | 32×32 | 32 | 1,024     | M or N < 64, or few threadgroups |
//! | Large  | 64×64 | 32 | 4,096     | M ≥ 64, N ≥ 64, ≥16 TGs |
//!
//! The Large config produces 4x output per threadgroup, reducing dispatch
//! overhead and improving L1/L2 cache utilization. BK=32 (matching Small)
//! enabled by 2-pass output write — `pass_out[32×65]` replaces `tile_out[64×65]`.
//!
//! Broadcast RHS is supported for the Linear::forward() `[B,M,K] × [K,N]` pattern.
//!
//! Requires Apple Silicon with simdgroup_matrix support (A14+/M1+).
//!
//! Issue: #1518 (original), #3479 (adaptive tile selection)

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_matmul_simd_msl.rs"]
mod msl_kernels;
use msl_kernels::{SIMD_GEMM_64_F16_MSL, SIMD_GEMM_64_MSL, SIMD_GEMM_F16_MSL, SIMD_GEMM_MSL};

// ---------------------------------------------------------------------------
// Tile configuration (#3479)
// ---------------------------------------------------------------------------

/// GEMM tile configuration for simdgroup kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmTileConfig {
    /// Output tile rows (BM).
    pub(crate) bm: u32,
    /// Output tile columns (BN).
    pub(crate) bn: u32,
}

impl GemmTileConfig {
    /// Classic 32×32 tile with BK=32. 1,024 output elements per TG.
    pub(crate) const SMALL: Self = Self { bm: 32, bn: 32 };

    /// Large 64×64 tile with BK=32. 4,096 output elements per TG.
    /// 2-pass output write keeps TG memory under 32 KB (25,088 bytes f32).
    pub(crate) const LARGE: Self = Self { bm: 64, bn: 64 };
}

/// Select the optimal GEMM tile configuration for given dimensions.
///
/// Routes to LARGE (64×64, BK=32) when both output dimensions are large enough
/// to fill the tile and there are enough threadgroups for good GPU occupancy.
/// Falls back to SMALL (32×32, BK=32) for small shapes or low TG counts.
///
/// Threshold: ≥32 threadgroups ensures ~80% core utilization on M4 Max (40 cores).
/// Each LARGE TG produces 4x output vs SMALL, so 32 LARGE TGs ≈ 128 SMALL TGs.
///
/// Issue: #3479 (adaptive GEMM tile selection)
pub(crate) fn select_tile_config(m: usize, _k: usize, n: usize, _batch: usize) -> GemmTileConfig {
    let tgs_64 = m.div_ceil(64) * n.div_ceil(64);
    if m >= 64 && n >= 64 && tgs_64 >= 32 {
        GemmTileConfig::LARGE
    } else {
        GemmTileConfig::SMALL
    }
}

/// Returns true if the simdgroup GEMM kernel should be used for these dims.
///
/// Criteria (from design doc `designs/2026-03-08-simdgroup-matmul-dispatch-strategy.md`):
/// - All dimensions must be multiples of 8 (simdgroup_matrix requirement)
/// - M×N must be ≥ 16,384 (compute must dominate dispatch overhead)
/// - K must be ≥ 128 (amortize shared memory tiling cost)
///
/// Shared by DynTensor matmul routing, LinearActivation NativeOp, and
/// compare_op matmul. Single source of truth for simdgroup eligibility.
pub(crate) fn should_use_simdgroup(m: usize, k: usize, n: usize) -> bool {
    m.is_multiple_of(8) && k.is_multiple_of(8) && n.is_multiple_of(8) && m * n >= 16_384 && k >= 128
}

/// Minimum threadgroup count for F16 simdgroup to outperform F32.
/// Empirically determined: 768 TGs → 1.73x, 192 TGs → 0.96x.
/// Conservative threshold: 384 TGs (~9.6 TGs/core on M4 Max 40-core).
pub(crate) const F16_MIN_THREADGROUPS: usize = 384;

/// Returns true if F16 simdgroup GEMM should be preferred over F32 for these dims.
///
/// F16 requires higher occupancy than F32 because the 2x ALU throughput from
/// `simdgroup_multiply_accumulate(float, half, half, float)` is only realized
/// when ALU is the execution bottleneck, not when the GPU is under-saturated.
/// Part of #2981.
pub(crate) fn should_use_f16_simdgroup(m: usize, k: usize, n: usize, batch: usize) -> bool {
    if !should_use_simdgroup(m, k, n) {
        return false;
    }
    // Scale threshold by tile area: LARGE (4096) needs fewer TGs than SMALL (1024)
    // to saturate the GPU. F16_MIN_THREADGROUPS * 1024 / tile_area normalizes.
    let tile = select_tile_config(m, k, n, batch);
    let tg_count = m.div_ceil(tile.bm as usize) * n.div_ceil(tile.bn as usize) * batch.max(1);
    let tile_area = (tile.bm as usize) * (tile.bn as usize);
    let threshold = F16_MIN_THREADGROUPS * 1024 / tile_area;
    tg_count >= threshold
}

impl super::MetalDynBackend {
    /// Dispatch simdgroup_matrix GEMM kernel, selecting f32 or f16 variant.
    ///
    /// F32 tensors use `simd_gemm_f32` with float buffers.
    /// BF16/F16 tensors use `simd_gemm_f16` with half buffers and float
    /// accumulators for mixed-precision (half inputs, float MAC, half output).
    ///
    /// Supports 2D, 3D (batched), and 3D×2D (broadcast RHS) matmul.
    /// Returns GPU-resident DynTensor with shape `out_shape`.
    ///
    /// Issue: #1518 (f32), #1670 (f16/bf16)
    pub(super) fn gpu_matmul_simdgroup(lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        Self::gpu_matmul_simdgroup_inner(lhs, rhs, None)
    }

    /// Like `gpu_matmul_simdgroup` but forces a specific tile config (for benchmarking).
    #[cfg(test)]
    pub(super) fn gpu_matmul_simdgroup_forced(
        lhs: &DynTensor,
        rhs: &DynTensor,
        tile: GemmTileConfig,
    ) -> Result<DynTensor> {
        Self::gpu_matmul_simdgroup_inner(lhs, rhs, Some(tile))
    }

    fn gpu_matmul_simdgroup_inner(
        lhs: &DynTensor,
        rhs: &DynTensor,
        tile_override: Option<GemmTileConfig>,
    ) -> Result<DynTensor> {
        Self::validate_same_float_dtype(lhs, rhs, "gpu_matmul_simdgroup")?;

        let dtype = lhs.dtype();
        let is_half = dtype == DType::BF16 || dtype == DType::F16;
        let bytes_per_elem: usize = if is_half { 2 } else { 4 };

        let l_shape = lhs.dims();
        let r_shape = rhs.dims();
        let l_ndim = l_shape.len();
        let r_ndim = r_shape.len();

        if l_ndim < 2 || r_ndim < 2 {
            return Err(TensorError::InvalidShape(format!(
                "simdgroup matmul requires >= 2D, got {l_shape:?} and {r_shape:?}"
            )));
        }

        let m = l_shape[l_ndim - 2];
        let k = l_shape[l_ndim - 1];
        let n = r_shape[r_ndim - 1];

        let k_rhs = r_shape[r_ndim - 2];
        if k != k_rhs {
            return Err(TensorError::InvalidShape(format!(
                "simdgroup matmul K mismatch: {l_shape:?} (K={k}) @ {r_shape:?} (K={k_rhs})"
            )));
        }

        // Compute batch count and broadcast flag.
        let (batch_count, broadcast_rhs) = if l_ndim == 2 && r_ndim == 2 {
            (1usize, false)
        } else if l_ndim >= 3 && r_ndim == 2 {
            let batch: usize = checked_dim_product(&l_shape[..l_ndim - 2])?;
            (batch, true)
        } else if l_ndim >= 3 && r_ndim >= 3 {
            let l_batch: usize = checked_dim_product(&l_shape[..l_ndim - 2])?;
            let r_batch: usize = checked_dim_product(&r_shape[..r_ndim - 2])?;
            if l_batch != r_batch {
                return Err(TensorError::InvalidShape(format!(
                    "simdgroup matmul batch mismatch: {l_shape:?} (batch={l_batch}) @ {r_shape:?} (batch={r_batch})"
                )));
            }
            (l_batch, false)
        } else {
            return Err(TensorError::InvalidShape(format!(
                "simdgroup matmul unsupported rank combination: {l_shape:?} @ {r_shape:?}"
            )));
        };

        let mut out_shape: Vec<usize> = l_shape[..l_ndim - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let total_output = batch_count
            .checked_mul(m)
            .and_then(|v| v.checked_mul(n))
            .ok_or(TensorError::DimensionOverflow {
                dims: vec![batch_count, m, n],
            })?;
        let buffer_bytes = total_output.checked_mul(bytes_per_elem).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: out_shape.clone(),
            }
        })?;

        let m_u32 = u32::try_from(m).map_err(|_| TensorError::ValueOutOfRange {
            description: "simd_gemm: M exceeds u32::MAX for Metal dispatch",
        })?;
        let n_u32 = u32::try_from(n).map_err(|_| TensorError::ValueOutOfRange {
            description: "simd_gemm: N exceeds u32::MAX for Metal dispatch",
        })?;
        let k_u32 = u32::try_from(k).map_err(|_| TensorError::ValueOutOfRange {
            description: "simd_gemm: K exceeds u32::MAX for Metal dispatch",
        })?;
        let batch_u32 = u32::try_from(batch_count).map_err(|_| TensorError::ValueOutOfRange {
            description: "simd_gemm: batch_count exceeds u32::MAX for Metal dispatch",
        })?;
        let bcast_flag: u32 = if broadcast_rhs { 1 } else { 0 };

        // Select tile config: use override if provided, else auto-select (#3479).
        let tile = tile_override.unwrap_or_else(|| select_tile_config(m, k, n, batch_count));

        // Select kernel variant based on tile config + dtype.
        let (msl_source, kernel_name) = match (tile, is_half) {
            (GemmTileConfig::LARGE, false) => (SIMD_GEMM_64_MSL, "simd_gemm_64_f32"),
            (GemmTileConfig::LARGE, true) => (SIMD_GEMM_64_F16_MSL, "simd_gemm_64_f16"),
            (_, false) => (SIMD_GEMM_MSL, "simd_gemm_f32"),
            (_, true) => (SIMD_GEMM_F16_MSL, "simd_gemm_f16"),
        };

        // Threadgroup memory depends on tile config and dtype.
        let tg_bytes = tg_memory_bytes(tile, is_half);

        let lhs_data = lhs.gpu_data::<MetalTensorData>()?;
        let rhs_data = rhs.gpu_data::<MetalTensorData>()?;

        super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(cache, msl_source, kernel_name, 2, false)
                .map_err(metal_err)?;

            let ctx = Self::ctx()?;
            let (out_buf, out_offset) =
                crate::arena::arena_alloc_or_create(ctx, buffer_bytes).map_err(metal_err)?;

            let grid_x = n_u32.div_ceil(tile.bn);
            let grid_y = m_u32.div_ceil(tile.bm);
            let grid_z = batch_u32;

            let plan = DispatchMode::Grid3D {
                grid: [grid_x, grid_y, grid_z],
                threads: [32, 4, 1],
            }
            .plan()
            .map_err(metal_err)?
            .with_output_elems(total_output)
            .with_constants(vec![m_u32, n_u32, k_u32, batch_u32, bcast_flag])
            .with_use_threadgroups(true)
            .with_threadgroup_memory_bytes(Some(tg_bytes));

            pipeline
                .dispatch_buffers_with_all_offsets(
                    ctx,
                    &[&lhs_data.buffer, &rhs_data.buffer],
                    &[lhs_data.byte_offset, rhs_data.byte_offset],
                    &out_buf,
                    out_offset,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
            DynTensor::from_gpu_storage(out_shape, dtype, Arc::new(storage), Device::metal())
        })
    }
}

/// Compute threadgroup memory bytes for a given tile config and element type.
///
/// Direct register-to-device writes eliminate tile_out/pass_out for f32 kernels.
/// Edge tiles reuse As (32×32) or Bs (64×64) — no additional TG memory.
/// F16 64×64 keeps pass_out for float→half conversion (can't simdgroup_store float to half).
pub(crate) fn tg_memory_bytes(tile: GemmTileConfig, is_half: bool) -> u64 {
    let (bm, bn) = (u64::from(tile.bm), u64::from(tile.bn));
    match (tile, is_half) {
        // SMALL 32×32, BK=32: As[32×33] + Bs[32×33] (tile_out eliminated)
        (GemmTileConfig::SMALL, false) => 2 * bm * (bm + 1) * 4, // 8,448
        // SMALL 32×32 f16: As[32×33]h + Bs[32×33]h + tile_out[32×33]f (keeps conversion buf)
        (GemmTileConfig::SMALL, true) => 2 * bm * (bm + 1) * 2 + bm * (bm + 1) * 4, // 8,448
        // LARGE 64×64 f32: As[64×33] + Bs[32×65] (pass_out eliminated)
        (GemmTileConfig::LARGE, false) => {
            // As[64×33]f + Bs[32×65]f = 16,768
            bm * 33 * 4 + 32 * (bn + 1) * 4
        }
        // LARGE 64×64 f16: As[64×33]h + Bs[32×65]h + pass_out[32×65]f (keeps conversion buf)
        (GemmTileConfig::LARGE, true) => {
            // As[64×33]h + Bs[32×65]h + pass_out[32×65]f = 16,704
            bm * 33 * 2 + 32 * (bn + 1) * 2 + 32 * (bn + 1) * 4
        }
        // Fallback: treat as SMALL
        (_, false) => 2 * bm * (bm + 1) * 4,
        (_, true) => 2 * bm * (bm + 1) * 2 + bm * (bm + 1) * 4,
    }
}

/// Encode a 2D F32 simdgroup matmul into the current lazy command buffer batch.
///
/// Used by the LSTM precomputed input projection path (#3491): computes
/// `input[S*B, I] @ weight_ih_t[I, 4H] + bias` as a single GEMM dispatch
/// in the lazy batch, without triggering a flush. The recurrence kernel then
/// reads pre-projected values instead of computing the input dot product.
///
/// Uses adaptive tile selection (#3479): 64×64 tiles for large shapes,
/// 32×32 for small.
///
/// Returns a `MetalTensorData` pointing to the arena-allocated output.
pub(crate) fn encode_simdgroup_matmul_into_batch(
    a_data: &MetalTensorData,
    b_data: &MetalTensorData,
    m: usize,
    k: usize,
    n: usize,
) -> Result<MetalTensorData> {
    let total_output = m
        .checked_mul(n)
        .ok_or(TensorError::DimensionOverflow { dims: vec![m, n] })?;
    let buffer_bytes = total_output
        .checked_mul(size_of::<f32>())
        .ok_or(TensorError::DimensionOverflow { dims: vec![m, n] })?;

    let m_u32 = u32::try_from(m).map_err(|_| TensorError::ValueOutOfRange {
        description: "encode_simdgroup_matmul: M exceeds u32::MAX",
    })?;
    let n_u32 = u32::try_from(n).map_err(|_| TensorError::ValueOutOfRange {
        description: "encode_simdgroup_matmul: N exceeds u32::MAX",
    })?;
    let k_u32 = u32::try_from(k).map_err(|_| TensorError::ValueOutOfRange {
        description: "encode_simdgroup_matmul: K exceeds u32::MAX",
    })?;
    let batch_u32: u32 = 1;
    let bcast_flag: u32 = 0;

    let tile = select_tile_config(m, k, n, 1);
    let (msl_source, kernel_name) = match tile {
        GemmTileConfig::LARGE => (SIMD_GEMM_64_MSL, "simd_gemm_64_f32"),
        _ => (SIMD_GEMM_MSL, "simd_gemm_f32"),
    };
    let tg_bytes = tg_memory_bytes(tile, false);

    super::with_pipeline_cache(|cache| {
        let pipeline = KernelPipeline::from_msl(cache, msl_source, kernel_name, 2, false)
            .map_err(metal_err)?;

        let ctx = super::MetalDynBackend::ctx()?;
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(ctx, buffer_bytes).map_err(metal_err)?;

        let grid_x = n_u32.div_ceil(tile.bn);
        let grid_y = m_u32.div_ceil(tile.bm);

        let plan = DispatchMode::Grid3D {
            grid: [grid_x, grid_y, batch_u32],
            threads: [32, 4, 1],
        }
        .plan()
        .map_err(metal_err)?
        .with_output_elems(total_output)
        .with_constants(vec![m_u32, n_u32, k_u32, batch_u32, bcast_flag])
        .with_use_threadgroups(true)
        .with_threadgroup_memory_bytes(Some(tg_bytes));

        pipeline
            .dispatch_buffers_with_all_offsets(
                ctx,
                &[&a_data.buffer, &b_data.buffer],
                &[a_data.byte_offset, b_data.byte_offset],
                &out_buf,
                out_offset,
                &plan,
            )
            .map_err(metal_err)?;

        Ok(MetalTensorData::from_arena_alloc(out_buf, out_offset))
    })
}

/// MSL source for pre-compilation: simdgroup GEMM f32 kernel (32×32).
pub(crate) fn simd_gemm_f32_msl_source() -> &'static str {
    SIMD_GEMM_MSL
}

/// MSL source for pre-compilation: simdgroup GEMM f16 kernel (32×32).
pub(crate) fn simd_gemm_f16_msl_source() -> &'static str {
    SIMD_GEMM_F16_MSL
}

/// MSL source for pre-compilation: simdgroup GEMM f32 kernel (64×64).
pub(crate) fn simd_gemm_64_f32_msl_source() -> &'static str {
    SIMD_GEMM_64_MSL
}

/// MSL source for pre-compilation: simdgroup GEMM f16 kernel (64×64).
pub(crate) fn simd_gemm_64_f16_msl_source() -> &'static str {
    SIMD_GEMM_64_F16_MSL
}
