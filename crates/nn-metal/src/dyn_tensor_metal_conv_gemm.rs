// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d via im2col + simdgroup GEMM (#3002).
//!
//! Transforms Conv1d into matrix multiplication for large convolutions:
//! 1. **im2col**: Unfold input `[C_in, L_in]` → `[C_in*K, L_out]` (1 dispatch per batch)
//! 2. **GEMM**: weight `[C_out, C_in*K]` × col `[C_in*K, L_out]` → `[C_out, L_out]` (1 dispatch)
//! 3. **Bias**: add `[C_out]` broadcast (uses DynTensor broadcast add)
//!
//! The GEMM step reuses the existing simdgroup_matrix 32×32 tiled kernel from
//! `dyn_tensor_metal_matmul_simd.rs`, which provides ~1.3x speedup over naive
//! per-element Conv1d for large channel counts (512+) via tiled shared-memory
//! weight reuse.
//!
//! **Constraints:**
//! - groups must be 1 (standard convolution, not grouped/depthwise)
//! - F32 and F16/BF16 supported (simdgroup GEMM has both variants)
//!
//! Part of #3002.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;

use super::matmul_simd::{should_use_simdgroup, F16_MIN_THREADGROUPS};

use super::MetalTensorData;

use crate::simdgroup_tile_select;

#[path = "dyn_tensor_metal_conv_gemm_msl.rs"]
mod msl;

/// Minimum total GEMM FLOPs (M*K*N) to justify im2col overhead.
/// Below this, the naive per-element conv is faster due to im2col memory cost.
/// C_out=256+, C_in*K=768+, L_out=100+ → 20M+, clear win for GEMM.
const MIN_GEMM_FLOPS: usize = 2_000_000;

/// Minimum C_in * L_out product for the direct K=3 kernel to beat im2col+GEMM.
/// Below this threshold the overhead of per-element boundary checks in the
/// direct kernel dominates. For Kokoro shapes (C_in=256+, L_out=50+), this
/// is easily met.
const DIRECT_K3_MIN_WORK: usize = 8_192;

impl super::MetalDynBackend {
    /// Returns true if Conv1d should use the direct sliding-window kernel
    /// instead of im2col + GEMM. The direct kernel avoids the im2col buffer
    /// allocation and blit, saving 1 dispatch per Conv1d.
    ///
    /// Conditions:
    /// - kernel_size == 3, stride == 1, dilation == 1 (sliding-window pattern)
    /// - groups == 1
    /// - F32/F16/BF16 dtype
    /// - Sufficient work to amortize per-element boundary checks
    /// - Known Kokoro generator shape for shape-specific tile optimization
    ///
    /// Issue: #4264
    pub(super) fn should_use_direct_conv1d_k3(
        in_shape: &[usize],
        k_shape: &[usize],
        out_len: usize,
        groups: usize,
        stride: usize,
        dilation: usize,
        dtype: DType,
    ) -> bool {
        if groups != 1
            || stride != 1
            || dilation != 1
            || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return false;
        }
        if in_shape.len() != 3 || k_shape.len() != 3 {
            return false;
        }
        let c_out = k_shape[0];
        let c_in = k_shape[1];
        let k_size = k_shape[2];
        if k_size != 3 {
            return false;
        }
        // Check sufficient work for direct kernel to outperform im2col+GEMM.
        if c_in.saturating_mul(out_len) < DIRECT_K3_MIN_WORK {
            return false;
        }
        // Prefer direct kernel for known Kokoro shapes.
        simdgroup_tile_select::is_kokoro_conv1d_shape(c_out, c_in, k_size)
    }

    /// Direct sliding-window Conv1d for K=3, stride=1, dilation=1.
    ///
    /// Computes Conv1d without im2col by reading the 3 input positions directly
    /// in the GEMM tile loop. Saves 1 dispatch + 1 temporary buffer per Conv1d.
    ///
    /// Input: `[B, C_in, L_in]`, Kernel: `[C_out, C_in, 3]`
    /// Output: `[B, C_out, L_out]`
    ///
    /// Issue: #4264
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gpu_direct_conv1d_k3(
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        out_shape: &[usize],
    ) -> Result<DynTensor> {
        let in_shape = input.dims();
        let k_shape = kernel.dims();
        let dtype = input.dtype();
        let elem_bytes = dtype.size_bytes();

        let batch = in_shape[0];
        let c_in = in_shape[1];
        let l_in = in_shape[2];
        let c_out = k_shape[0];
        let l_out = out_shape[2];

        let output_per_batch = c_out * l_out;
        let total_output = batch * output_per_batch;
        let output_bytes =
            total_output
                .checked_mul(elem_bytes)
                .ok_or(TensorError::DimensionOverflow {
                    dims: vec![batch, c_out, l_out],
                })?;

        let is_half = matches!(dtype, DType::F16 | DType::BF16);
        let (kernel_msl, kernel_name) = if is_half {
            (msl::DIRECT_CONV1D_K3_F16_MSL, "direct_conv1d_k3_f16")
        } else {
            (msl::DIRECT_CONV1D_K3_F32_MSL, "direct_conv1d_k3_f32")
        };

        let input_data = input.gpu_data::<MetalTensorData>()?;
        // Weight [C_out, C_in, 3] is contiguous; same layout as [C_out, C_in*3].
        let weight_data = kernel.gpu_data::<MetalTensorData>()?;

        let c_out_u32 = crate::to_u32(c_out, "direct_conv1d c_out")?;
        let c_in_u32 = crate::to_u32(c_in, "direct_conv1d c_in")?;
        let l_in_u32 = crate::to_u32(l_in, "direct_conv1d l_in")?;
        let l_out_u32 = crate::to_u32(l_out, "direct_conv1d l_out")?;
        let pad_u32 = crate::to_u32(padding, "direct_conv1d padding")?;

        // TM=32, TN=32 tile sizes match the MSL kernel constants.
        let grid_x = l_out_u32.div_ceil(32);
        let grid_y = c_out_u32.div_ceil(32);

        let ctx = Self::ctx()?;

        let result_buf = super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(
                cache,
                kernel_msl,
                kernel_name,
                2, // 2 input buffers: input + weight
                false,
            )
            .map_err(metal_err)?;

            let out_buf = ctx.create_buffer_zeroed(output_bytes).map_err(metal_err)?;

            let plan = DispatchMode::Grid3D {
                grid: [grid_x, grid_y, 1],
                threads: [128, 1, 1],
            }
            .plan()
            .map_err(metal_err)?
            .with_output_elems(output_per_batch)
            .with_constants(vec![c_out_u32, c_in_u32, l_in_u32, l_out_u32, pad_u32])
            .with_use_threadgroups(true);

            // Dispatch once per batch element.
            let in_batch_bytes = c_in * l_in * elem_bytes;
            let out_batch_bytes = output_per_batch * elem_bytes;

            for b in 0..batch {
                let in_offset = input_data.byte_offset + b * in_batch_bytes;
                let out_offset = b * out_batch_bytes;

                pipeline
                    .dispatch_buffers_with_all_offsets(
                        ctx,
                        &[&input_data.buffer, &weight_data.buffer],
                        &[in_offset, weight_data.byte_offset],
                        &out_buf,
                        out_offset,
                        &plan,
                    )
                    .map_err(metal_err)?;
            }

            Ok::<_, TensorError>(out_buf)
        })?;

        // Wrap result in DynTensor.
        let storage = MetalTensorData::new(result_buf);
        let gemm_result = DynTensor::from_gpu_storage(
            if batch == 1 {
                vec![c_out, l_out]
            } else {
                vec![batch, c_out, l_out]
            },
            dtype,
            Arc::new(storage),
            Device::metal(),
        )?;

        // Add bias if present.
        let biased = if let Some(bias_t) = bias {
            if batch == 1 {
                let bias_col = bias_t.reshape([c_out, 1])?;
                gemm_result.add(&bias_col)?
            } else {
                let bias_bcast = bias_t.reshape([1, c_out, 1])?;
                gemm_result.add(&bias_bcast)?
            }
        } else {
            gemm_result
        };

        biased.reshape(out_shape)
    }

    /// Returns true if Conv1d should use the im2col + GEMM path.
    ///
    /// Conditions:
    /// - groups == 1 (no grouped/depthwise convolution)
    /// - F32/F16/BF16 dtype (im2col + simdgroup GEMM have both f32 and f16 variants)
    /// - GEMM dimensions large enough to justify im2col overhead
    pub(super) fn should_use_conv1d_gemm(
        in_shape: &[usize],
        k_shape: &[usize],
        out_len: usize,
        groups: usize,
        dtype: DType,
    ) -> bool {
        if groups != 1 || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return false;
        }
        if in_shape.len() != 3 || k_shape.len() != 3 {
            return false;
        }
        let c_out = k_shape[0];
        let c_in_k = k_shape[1] * k_shape[2]; // C_in * K_size (groups=1)
        if c_out * c_in_k * out_len < MIN_GEMM_FLOPS {
            return false;
        }
        // F16/BF16 simdgroup GEMM regresses at low occupancy (~0.96x at 192 TGs).
        // Fall back to naive conv1d when threadgroup count is insufficient (#3315).
        if matches!(dtype, DType::F16 | DType::BF16) {
            let batch = in_shape[0];
            let tg_count = c_out.div_ceil(32) * out_len.div_ceil(32) * batch.max(1);
            if tg_count < F16_MIN_THREADGROUPS {
                return false;
            }
        }
        true
    }

    /// Conv1d via im2col + simdgroup GEMM.
    ///
    /// Input: `[B, C_in, L_in]`, Kernel: `[C_out, C_in, K]`
    /// Output: `[B, C_out, L_out]`
    ///
    /// Only called when `should_use_conv1d_gemm` returns true.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gpu_conv1d_gemm(
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        out_shape: &[usize],
    ) -> Result<DynTensor> {
        let in_shape = input.dims();
        let k_shape = kernel.dims();
        let dtype = input.dtype();
        let elem_bytes = dtype.size_bytes();

        let batch = in_shape[0];
        let c_in = in_shape[1];
        let l_in = in_shape[2];
        let c_out = k_shape[0];
        let k_size = k_shape[2];
        let l_out = out_shape[2];

        let col_rows = c_in * k_size; // C_in * K

        // Step 1: im2col — unfold input per batch into [C_in*K, L_out].
        let col_per_batch = col_rows * l_out;
        let col_total_elems = batch * col_per_batch;
        let col_bytes =
            col_total_elems
                .checked_mul(elem_bytes)
                .ok_or(TensorError::DimensionOverflow {
                    dims: vec![batch, col_rows, l_out],
                })?;

        // Select im2col MSL kernel variant based on dtype.
        let is_half = matches!(dtype, DType::F16 | DType::BF16);
        let (im2col_msl, im2col_name) = if is_half {
            (msl::IM2COL_1D_F16_MSL, "im2col_1d_f16")
        } else {
            (msl::IM2COL_1D_F32_MSL, "im2col_1d_f32")
        };

        let input_data = input.gpu_data::<MetalTensorData>()?;
        let ctx = Self::ctx()?;

        let col_buf: crate::buffer::MetalBuffer = super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(
                cache,
                im2col_msl,
                im2col_name,
                1, // 1 input buffer
                false,
            )
            .map_err(metal_err)?;

            let col_buf = ctx.create_buffer_zeroed(col_bytes).map_err(metal_err)?;

            let total_u32 = crate::to_u32(col_per_batch, "im2col total")?;
            let c_in_u32 = crate::to_u32(c_in, "im2col c_in")?;
            let k_u32 = crate::to_u32(k_size, "im2col k")?;
            let l_in_u32 = crate::to_u32(l_in, "im2col l_in")?;
            let l_out_u32 = crate::to_u32(l_out, "im2col l_out")?;
            let stride_u32 = crate::to_u32(stride, "im2col stride")?;
            let pad_u32 = crate::to_u32(padding, "im2col padding")?;
            let dil_u32 = crate::to_u32(dilation, "im2col dilation")?;

            let tg_size = 256u32;
            let num_tgs = total_u32.div_ceil(tg_size);

            // Constants layout matches MSL buffer(2..9):
            // total, C_in, K_sz, L_in, L_out, stride, padding, dilation
            let plan = DispatchMode::Grid3D {
                grid: [num_tgs, 1, 1],
                threads: [tg_size, 1, 1],
            }
            .plan()
            .map_err(metal_err)?
            .with_output_elems(col_per_batch)
            .with_constants(vec![
                total_u32, c_in_u32, k_u32, l_in_u32, l_out_u32, stride_u32, pad_u32, dil_u32,
            ])
            .with_use_threadgroups(true);

            // Dispatch once per batch element. The MSL kernel operates on a
            // single [C_in, L_in] → [C_in*K, L_out] slice. Byte offsets select
            // the batch slice in input and output buffers.
            let in_batch_bytes = c_in * l_in * elem_bytes;
            let col_batch_bytes = col_per_batch * elem_bytes;

            for b in 0..batch {
                let in_offset = input_data.byte_offset + b * in_batch_bytes;
                let out_offset = b * col_batch_bytes;

                pipeline
                    .dispatch_buffers_with_all_offsets(
                        ctx,
                        &[&input_data.buffer],
                        &[in_offset],
                        &col_buf,
                        out_offset,
                        &plan,
                    )
                    .map_err(metal_err)?;
            }

            Ok::<_, TensorError>(col_buf)
        })?;

        // Step 2: GEMM — weight [C_out, C_in*K] × col → out
        //
        // Weight [C_out, C_in, K] is contiguous and has the same memory layout as
        // [C_out, C_in*K] when groups=1. Reshape is zero-copy.
        let weight_2d = kernel.reshape([c_out, col_rows])?;

        // Route GEMM: simdgroup when dims conform (alignment, M*N, K thresholds),
        // naive otherwise. Fixes #3315 — Conv1d previously called simdgroup
        // unconditionally, bypassing the occupancy gate that standalone matmul uses.
        let use_simdgroup = should_use_simdgroup(c_out, col_rows, l_out);
        let matmul_fn = if use_simdgroup {
            Self::gpu_matmul_simdgroup
        } else {
            Self::gpu_matmul_naive
        };

        let gemm_result = if batch == 1 {
            // 2D GEMM: [C_out, col_rows] × [col_rows, L_out] → [C_out, L_out]
            let col_storage = MetalTensorData::new(col_buf);
            let col_2d = DynTensor::from_gpu_storage(
                vec![col_rows, l_out],
                dtype,
                Arc::new(col_storage),
                Device::metal(),
            )?;
            matmul_fn(&weight_2d, &col_2d)?
        } else {
            // Batched: per-batch GEMM then stack.
            let col_storage = MetalTensorData::new(col_buf);
            let col_3d = DynTensor::from_gpu_storage(
                vec![batch, col_rows, l_out],
                dtype,
                Arc::new(col_storage),
                Device::metal(),
            )?;

            let mut results = Vec::with_capacity(batch);
            for b in 0..batch {
                let col_b = col_3d.narrow(0, b, 1)?.reshape([col_rows, l_out])?;
                results.push(matmul_fn(&weight_2d, &col_b)?);
            }
            DynTensor::stack(&results, 0)?
        };

        // Step 3: Add bias if present via DynTensor broadcast add.
        let biased = if let Some(bias_t) = bias {
            if batch == 1 {
                // result: [C_out, L_out], bias: [C_out] → [C_out, 1]
                let bias_col = bias_t.reshape([c_out, 1])?;
                gemm_result.add(&bias_col)?
            } else {
                // result: [B, C_out, L_out], bias: [C_out] → [1, C_out, 1]
                let bias_bcast = bias_t.reshape([1, c_out, 1])?;
                gemm_result.add(&bias_bcast)?
            }
        } else {
            gemm_result
        };

        // Reshape to [B, C_out, L_out].
        biased.reshape(out_shape)
    }
}
