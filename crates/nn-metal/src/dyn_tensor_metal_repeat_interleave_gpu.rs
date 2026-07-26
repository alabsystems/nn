// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native repeat_interleave: computes prefix-sum offsets on GPU.
//!
//! Eliminates the CPU sync bottleneck in [`DynTensor::repeat_interleave`] by
//! keeping the counts tensor on GPU. Only one scalar readback (the total count)
//! is needed to allocate the output buffer.
//!
//! Used by Kokoro TTS `length_regulate` where this eliminates 4 forced GPU
//! flushes per synthesis call. Part of #2616, #2218.
//!
//! Supports counts arrays up to 256 elements (single-threadgroup Blelloch).
//! Larger falls back to the CPU-counts path.

use std::mem::size_of;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

/// Maximum counts length for GPU-native prefix sum (single threadgroup).
pub(crate) const MAX_GPU_PREFIX_SUM: usize = 256;

impl super::MetalDynBackend {
    /// Dispatch Blelloch prefix-sum on a 1-D f32 counts tensor, flush, and
    /// read back the total. Returns `(offsets_buffer, total_repeats)`.
    ///
    /// The offsets buffer contains `dim_size+1` u32 values:
    ///   `offsets[i] = sum(counts[0..i])` (exclusive prefix sum)
    ///   `offsets[dim_size] = total` (sentinel for binary search + readback)
    ///
    /// Used by both `gpu_repeat_interleave_from_gpu` and the compiled
    /// step-regulate path (#2911) which shares one prefix-sum across
    /// multiple scatter dispatches.
    pub(super) fn gpu_prefix_sum_offsets(
        counts: &DynTensor,
        dim_size: usize,
    ) -> Result<(crate::MetalBuffer, usize)> {
        let offsets_buf = Self::dispatch_prefix_sum_only(counts, dim_size)?;
        crate::gpu_scope::flush()?;
        let total_repeats = Self::read_prefix_sum_total(&offsets_buf, dim_size)?;
        Ok((offsets_buf, total_repeats))
    }

    /// Dispatch Blelloch prefix-sum without flushing or reading back.
    ///
    /// Returns the GPU-resident offsets buffer. Callers must ensure the
    /// command buffer completes before reading the scalar total.
    pub(super) fn dispatch_prefix_sum_only(
        counts: &DynTensor,
        dim_size: usize,
    ) -> Result<crate::MetalBuffer> {
        Self::validate_f32_buffer(counts, "dispatch_prefix_sum_only")?;

        if dim_size == 0 {
            return Err(TensorError::InvalidShape(
                "dispatch_prefix_sum_only: dim_size must be > 0".into(),
            ));
        }
        if dim_size > MAX_GPU_PREFIX_SUM {
            return Err(TensorError::Unsupported(format!(
                "dispatch_prefix_sum_only: dim_size {dim_size} > {MAX_GPU_PREFIX_SUM}",
            )));
        }

        let ctx = Self::ctx()?;
        let counts_data = counts.gpu_data::<MetalTensorData>()?;

        let offsets_len = dim_size + 1;
        let offsets_bytes = offsets_len
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| metal_err("prefix sum offsets bytes overflow"))?;
        let offsets_buf = ctx.create_buffer_zeroed(offsets_bytes).map_err(metal_err)?;

        super::with_pipeline_cache(|cache| -> Result<()> {
            let pipeline =
                KernelPipeline::from_msl(cache, PREFIX_SUM_MSL, "ri_prefix_sum_u32", 1, false)
                    .map_err(metal_err)?;

            let plan = DispatchMode::PerSliceReduction {
                outer: 1,
                reduce: dim_size as u32,
                threads: MAX_GPU_PREFIX_SUM as u32,
                shared_bytes: (MAX_GPU_PREFIX_SUM * size_of::<u32>()) as u32,
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(vec![crate::to_u32(dim_size, "ri_prefix_sum dim_size")?]);

            pipeline
                .dispatch_buffers_with_all_offsets(
                    ctx,
                    &[&counts_data.buffer],
                    &[counts_data.byte_offset],
                    &offsets_buf,
                    0,
                    &plan,
                )
                .map_err(metal_err)?;
            Ok(())
        })?;

        Ok(offsets_buf)
    }

    /// Read the total repeats scalar from a completed prefix-sum offsets buffer.
    pub(super) fn read_prefix_sum_total(
        offsets_buf: &crate::MetalBuffer,
        dim_size: usize,
    ) -> Result<usize> {
        let byte_offset = dim_size * size_of::<u32>();
        let slice: &[u32] = offsets_buf
            .contents_at_offset(byte_offset, 1)
            .map_err(metal_err)?;
        Ok(slice[0] as usize)
    }

    /// Binary-search scatter using pre-computed GPU-resident offsets.
    ///
    /// Scatter `x` along `dim` using `offsets_buf` from
    /// [`gpu_prefix_sum_offsets`]. Output shape = `x.shape` with
    /// `shape[dim]` replaced by `total_repeats`.
    pub(super) fn gpu_scatter_with_offsets(
        x: &DynTensor,
        dim: usize,
        offsets_buf: &crate::MetalBuffer,
        dim_size: usize,
        total_repeats: usize,
    ) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_scatter_with_offsets")?;
        let shape = x.dims();
        check_dim(dim, shape.len())?;

        let mut out_shape = shape.to_vec();
        out_shape[dim] = total_repeats;
        let total_elems = checked_dim_product(&out_shape)?;
        let inner = checked_dim_product(&shape[dim + 1..])?;
        let x_data = x.gpu_data::<MetalTensorData>()?;

        Self::dispatch_raw_msl(
            SCATTER_MSL,
            "repeat_interleave_f32_gpu",
            2, // param_count: input + offsets
            &[&x_data.buffer, offsets_buf],
            &[x_data.byte_offset, 0],
            total_elems,
            &out_shape,
            x.dtype(),
            vec![
                crate::to_u32(total_elems, "ri_scatter total_elems")?,
                crate::to_u32(dim_size, "ri_scatter dim_size")?,
                crate::to_u32(total_repeats, "ri_scatter total_repeats")?,
                crate::to_u32(inner, "ri_scatter inner")?,
            ],
        )
    }

    /// GPU-native repeat_interleave with GPU-resident counts tensor.
    ///
    /// Delegates to [`gpu_prefix_sum_offsets`] + [`gpu_scatter_with_offsets`].
    /// Validates counts for NaN/Inf/negative/non-integer before GPU dispatch
    /// to match CPU path behavior (ops_ext/repeat_interleave_validate_counts).
    pub(super) fn gpu_repeat_interleave_from_gpu(
        x: &DynTensor,
        dim: usize,
        counts: &DynTensor,
    ) -> Result<DynTensor> {
        Self::validate_f32_buffer(x, "gpu_repeat_interleave_from_gpu")?;
        Self::validate_f32_buffer(counts, "gpu_repeat_interleave_from_gpu:counts")?;

        let shape = x.dims();
        let ndim = shape.len();
        check_dim(dim, ndim)?;

        let dim_size = shape[dim];
        if counts.dims().len() != 1 || counts.dims()[0] != dim_size {
            return Err(TensorError::InvalidShape(format!(
                "gpu_repeat_interleave_from_gpu: counts shape {:?} must be [{dim_size}]",
                counts.dims(),
            )));
        }
        if dim_size == 0 {
            let mut out_shape = shape.to_vec();
            out_shape[dim] = 0;
            return DynTensor::zeros(&out_shape, x.dtype(), &Device::metal());
        }

        // Validate counts for NaN/Inf/negative/non-integer before GPU dispatch.
        // Flush pending GPU work so we can read the counts buffer. The counts
        // tensor is small (≤256 elements) so this readback is cheap.
        // Without this, NaN/negative counts silently map to 0 repeats on GPU
        // (MSL floor(NaN)=NaN, NaN > 0 = false → 0), diverging from the CPU
        // path which returns Err. Part of #2218 F1.
        crate::gpu_scope::flush()?;
        let counts_data = counts.gpu_data::<MetalTensorData>()?;
        let counts_slice: &[f32] = counts_data
            .buffer
            .contents_at_offset(counts_data.byte_offset, dim_size)
            .map_err(metal_err)?;
        for (i, &v) in counts_slice.iter().enumerate() {
            if !v.is_finite() || v < 0.0 || v != v.trunc() {
                return Err(TensorError::InvalidShape(format!(
                    "gpu_repeat_interleave_from_gpu: counts[{i}] = {v} \
                     must be a non-negative integer"
                )));
            }
        }

        let (offsets_buf, total_repeats) = Self::gpu_prefix_sum_offsets(counts, dim_size)?;

        if total_repeats == 0 {
            let mut out_shape = shape.to_vec();
            out_shape[dim] = 0;
            return DynTensor::zeros(&out_shape, x.dtype(), &Device::metal());
        }

        Self::gpu_scatter_with_offsets(x, dim, &offsets_buf, dim_size, total_repeats)
    }
}

// Free functions gpu_prefix_sum_offsets / gpu_scatter_with_offsets removed —
// callers use native_bridges.rs delegates → MetalDynBackend methods above.

/// MSL: fused f32→u32 cast + Blelloch exclusive prefix sum.
///
/// Input: `float counts[dim_size]` at buffer(0)
/// Output: `uint offsets[dim_size+1]` at buffer(1) — exclusive prefix sums
///         with `offsets[dim_size] = total` as sentinel.
/// Constant: `uint dim_size` at buffer(2)
///
/// Single threadgroup of 256 threads. Converts f32→u32 via `floor + max(0)`,
/// then computes exclusive prefix sum via Blelloch up-sweep/down-sweep.
const PREFIX_SUM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void ri_prefix_sum_u32(
    device const float* counts   [[buffer(0)]],
    device uint* offsets         [[buffer(1)]],
    device const uint& dim_size  [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint shared[256];

    // Load: convert f32 to u32 (floor + clamp to 0)
    uint val = 0;
    if (lid < dim_size) {
        float fv = counts[lid];
        fv = floor(fv);
        val = (fv > 0.0f) ? uint(fv) : 0u;
    }
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch up-sweep (reduce)
    for (uint s = 1; s < 256u; s *= 2) {
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < 256u) {
            shared[idx] += shared[idx - s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block total, then clear last for exclusive scan
    uint block_total = shared[255];
    if (lid == 0) {
        shared[255] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch down-sweep → exclusive prefix sum
    for (uint s = 128u; s > 0; s /= 2) {
        uint idx = (lid + 1) * s * 2 - 1;
        if (idx < 256u) {
            uint tmp = shared[idx];
            shared[idx] += shared[idx - s];
            shared[idx - s] = tmp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write exclusive prefix sums: offsets[i] = sum(counts[0..i])
    if (lid < dim_size) {
        offsets[lid] = shared[lid];
    }
    // Thread 0 writes sentinel: offsets[dim_size] = total
    if (lid == 0) {
        offsets[dim_size] = block_total;
    }
}
"#;

/// MSL: binary-search scatter using GPU-resident u32 offsets.
///
/// Same algorithm as the existing `repeat_interleave_f32` kernel but with
/// the offsets buffer already on GPU (from `ri_prefix_sum_u32`).
const SCATTER_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void repeat_interleave_f32_gpu(
    device const float* input      [[buffer(0)]],
    device const uint*  offsets    [[buffer(1)]],
    device float* output           [[buffer(2)]],
    device const uint& total_els   [[buffer(3)]],
    device const uint& dim_size    [[buffer(4)]],
    device const uint& out_dim     [[buffer(5)]],
    device const uint& inner_sz    [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;

    uint out_axis_inner = out_dim * inner_sz;
    uint outer_idx = tid / out_axis_inner;
    uint rem = tid % out_axis_inner;
    uint out_axis_idx = rem / inner_sz;
    uint inner_idx = rem % inner_sz;

    // Binary search offsets to find source element index.
    uint lo = 0, hi = dim_size;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= out_axis_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    uint src_idx = lo;

    uint in_offset = outer_idx * (dim_size * inner_sz) + src_idx * inner_sz + inner_idx;
    output[tid] = input[in_offset];
}
"#;
