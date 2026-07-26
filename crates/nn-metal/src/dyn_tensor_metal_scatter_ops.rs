// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU scatter, scatter_add, and index_add dispatch for [`MetalDynBackend`].
//!
//! Uses raw MSL kernels for indexed writes. scatter_add and index_add use
//! `atomic<float>` for concurrent accumulation; scatter uses non-atomic
//! overwrite (last-write-wins for duplicate indices).
//! Extracted from `dyn_tensor_metal_data_ops.rs` for the 500-line limit.
//!
//! Design: `designs/2026-03-05-metal-gpu-data-ops.md` (Direction 4)
//! Issues: #1178, #1949, #3942

use std::mem::size_of;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

impl super::MetalDynBackend {
    // ===== scatter_add =====
    //
    // output = self.clone(); output[index[i]] += src[i]  (atomic add).
    // Each source thread atomically adds its value to the output at the
    // indexed position. Metal `atomic<float>` requires Metal 3.0 (Apple Silicon).

    /// GPU-native scatter_add: atomic accumulation at indexed positions.
    ///
    /// Output starts as a clone of `self`. Each source element atomically adds
    /// to the output at the position specified by `index`. Uses `atomic_float`
    /// and `atomic_fetch_add_explicit` in MSL.
    pub(super) fn gpu_scatter_add(
        x: &DynTensor,
        dim: usize,
        index: &DynTensor,
        src: &DynTensor,
    ) -> Result<DynTensor> {
        // Lazy batch (#2009): flush pending GPU work before CPU readback
        // via clone_buffer_range. Without this, empty-src path reads stale data.
        crate::gpu_scope::flush()?;
        Self::validate_f32_buffer(x, "gpu_scatter_add")?;
        Self::validate_f32_buffer(src, "gpu_scatter_add(src)")?;

        let x_shape = x.dims();
        let src_shape = src.dims();
        let ndim = x_shape.len();

        check_dim(dim, ndim)?;

        let total_src = checked_dim_product(src_shape)?;
        let x_numel = checked_dim_product(x_shape)?;
        let x_bytes = x_numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("buffer bytes overflow"))?;

        if total_src == 0 {
            // No source elements to scatter — return a clone of self on GPU.
            let x_data = x.gpu_data::<MetalTensorData>()?;
            let ctx = Self::ctx()?;
            let cloned = ctx
                .clone_buffer_range(&x_data.buffer, x_data.byte_offset, x_bytes)
                .map_err(metal_err)?;
            let storage = MetalTensorData::new(cloned);
            return DynTensor::from_gpu_storage(
                x_shape.to_vec(),
                x.dtype(),
                Arc::new(storage),
                Device::metal(),
            );
        }

        // Compute outer/inner strides for the decomposition:
        //   outer_idx = tid / (src_dim_size * inner)
        //   inner_idx = tid % inner
        let src_dim_size = src_shape[dim];
        let out_dim_size = x_shape[dim];
        let inner = checked_dim_product(&src_shape[dim + 1..])?;

        let x_data = x.gpu_data::<MetalTensorData>()?;

        // Ensure src is on GPU.
        let gpu_src = if src.device().is_gpu() {
            src.clone()
        } else {
            src.to_device(&Device::metal())?
        };
        let src_data = gpu_src.gpu_data::<MetalTensorData>()?;

        // Convert indices to CPU U32 for validation + GPU upload.
        let cpu_u32_idx = {
            let cpu_idx = index.to_device(&Device::Cpu)?;
            cpu_idx.to_dtype(DType::U32)?
        };

        // Host-side OOB validation: match CPU scatter_add error behavior.
        {
            let indices = cpu_u32_idx.to_flat_vec::<u32>()?;
            for &idx in &indices {
                if (idx as usize) >= out_dim_size {
                    return Err(TensorError::ValueOutOfRange {
                        description: "scatter_add: index out of bounds for target dimension",
                    });
                }
            }
        }

        // Upload validated indices to GPU.
        let idx_u32 = cpu_u32_idx.to_device(&Device::metal())?;
        let idx_data = idx_u32.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;

        let msl = r#"
#include <metal_stdlib>
using namespace metal;

typedef atomic<float> atomic_float;

kernel void scatter_add_f32(
    device const float*  src        [[buffer(0)]],
    device const uint*   indices    [[buffer(1)]],
    device atomic_float* output     [[buffer(2)]],
    device const uint&   total_src  [[buffer(3)]],
    device const uint&   src_dim_sz [[buffer(4)]],
    device const uint&   out_dim_sz [[buffer(5)]],
    device const uint&   inner_sz   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_src) return;

    uint inner = tid % inner_sz;
    uint outer = tid / (src_dim_sz * inner_sz);
    uint dst_idx = indices[tid];

    // OOB guard: skip writes with invalid scatter indices.
    if (dst_idx >= out_dim_sz) return;

    uint out_offset = outer * (out_dim_sz * inner_sz) + dst_idx * inner_sz + inner;
    atomic_fetch_add_explicit(&output[out_offset], src[tid], memory_order_relaxed);
}
"#;

        // Clone self's logical data as the output (scatter accumulates into it).
        // Use clone_buffer_range to respect byte_offset from narrow views (#1964).
        let out_buf = ctx
            .clone_buffer_range(&x_data.buffer, x_data.byte_offset, x_bytes)
            .map_err(metal_err)?;
        let out_shape = x_shape.to_vec();

        super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(cache, msl, "scatter_add_f32", 2, false)
                .map_err(metal_err)?;

            let plan = DispatchMode::Elementwise {
                total: crate::to_u32(total_src, "scatter_add total_src")?,
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(vec![
                crate::to_u32(total_src, "scatter_add total_src")?,
                crate::to_u32(src_dim_size, "scatter_add src_dim_size")?,
                crate::to_u32(out_dim_size, "scatter_add out_dim_size")?,
                crate::to_u32(inner, "scatter_add inner")?,
            ]);

            pipeline
                .dispatch_buffers_with_offsets(
                    ctx,
                    &[&src_data.buffer, &idx_data.buffer],
                    &[src_data.byte_offset, idx_data.byte_offset],
                    &out_buf,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::new(out_buf);
            DynTensor::from_gpu_storage(out_shape, x.dtype(), Arc::new(storage), Device::metal())
        })
    }

    // ===== index_add =====
    //
    // output = self.clone(); output[index[i], ...] += src[i, ...]  (atomic add).
    // Unlike scatter_add, `index` is 1D — it maps src's `dim` axis to self's
    // `dim` axis. Non-dim axes must match between src and self.

    /// GPU-native index_add: atomic accumulation at 1D-indexed positions.
    ///
    /// Output starts as a clone of `self`. For each `i` in `0..index.len()`,
    /// atomically adds `src[..., i, ...]` to `output[..., index[i], ...]` along
    /// the specified dimension. Uses `atomic_float` with `atomic_fetch_add_explicit`.
    pub(super) fn gpu_index_add(
        x: &DynTensor,
        dim: usize,
        index: &DynTensor,
        src: &DynTensor,
    ) -> Result<DynTensor> {
        // Lazy batch (#2009): flush pending GPU work before CPU readback
        // via clone_buffer_range. Without this, empty-src path reads stale data.
        crate::gpu_scope::flush()?;
        Self::validate_f32_buffer(x, "gpu_index_add")?;
        Self::validate_f32_buffer(src, "gpu_index_add(src)")?;

        let x_shape = x.dims();
        let src_shape = src.dims();
        let ndim = x_shape.len();

        check_dim(dim, ndim)?;

        let total_src = checked_dim_product(src_shape)?;
        let x_numel = checked_dim_product(x_shape)?;
        let x_bytes = x_numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("buffer bytes overflow"))?;
        if total_src == 0 {
            // No source elements — return a clone of self on GPU.
            let x_data = x.gpu_data::<MetalTensorData>()?;
            let ctx = Self::ctx()?;
            let cloned = ctx
                .clone_buffer_range(&x_data.buffer, x_data.byte_offset, x_bytes)
                .map_err(metal_err)?;
            let storage = MetalTensorData::new(cloned);
            return DynTensor::from_gpu_storage(
                x_shape.to_vec(),
                x.dtype(),
                Arc::new(storage),
                Device::metal(),
            );
        }

        let src_dim_size = src_shape[dim];
        let out_dim_size = x_shape[dim];
        let inner = checked_dim_product(&src_shape[dim + 1..])?;

        let x_data = x.gpu_data::<MetalTensorData>()?;

        let gpu_src = if src.device().is_gpu() {
            src.clone()
        } else {
            src.to_device(&Device::metal())?
        };
        let src_data = gpu_src.gpu_data::<MetalTensorData>()?;

        // index is 1D — convert to CPU U32 for validation, then upload.
        let cpu_u32_idx = {
            let cpu_idx = index.to_device(&Device::Cpu)?;
            cpu_idx.to_dtype(DType::U32)?
        };

        {
            let indices = cpu_u32_idx.to_flat_vec::<u32>()?;
            for &idx in &indices {
                if (idx as usize) >= out_dim_size {
                    return Err(TensorError::ValueOutOfRange {
                        description: "index_add: index out of bounds for target dimension",
                    });
                }
            }
        }

        let idx_u32 = cpu_u32_idx.to_device(&Device::metal())?;
        let idx_data = idx_u32.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;

        // index_add kernel: index is 1D. For each source element at flat position
        // `tid`, decompose into (outer, src_dim_pos, inner). Look up the dest dim
        // position from `indices[src_dim_pos]`.
        let msl = r#"
#include <metal_stdlib>
using namespace metal;

typedef atomic<float> atomic_float;

kernel void index_add_f32(
    device const float*  src        [[buffer(0)]],
    device const uint*   indices    [[buffer(1)]],
    device atomic_float* output     [[buffer(2)]],
    device const uint&   total_src  [[buffer(3)]],
    device const uint&   src_dim_sz [[buffer(4)]],
    device const uint&   out_dim_sz [[buffer(5)]],
    device const uint&   inner_sz   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_src) return;

    uint inner = tid % inner_sz;
    uint src_dim_pos = (tid / inner_sz) % src_dim_sz;
    uint outer = tid / (src_dim_sz * inner_sz);

    uint dst_idx = indices[src_dim_pos];

    // OOB guard: skip writes with invalid indices.
    if (dst_idx >= out_dim_sz) return;

    uint out_offset = outer * (out_dim_sz * inner_sz) + dst_idx * inner_sz + inner;
    atomic_fetch_add_explicit(&output[out_offset], src[tid], memory_order_relaxed);
}
"#;

        // Clone self's logical data as the output (index_add accumulates into it).
        // Use clone_buffer_range to respect byte_offset from narrow views (#1964).
        let out_buf = ctx
            .clone_buffer_range(&x_data.buffer, x_data.byte_offset, x_bytes)
            .map_err(metal_err)?;
        let out_shape = x_shape.to_vec();

        super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(cache, msl, "index_add_f32", 2, false)
                .map_err(metal_err)?;

            let plan = DispatchMode::Elementwise {
                total: crate::to_u32(total_src, "index_add total_src")?,
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(vec![
                crate::to_u32(total_src, "index_add total_src")?,
                crate::to_u32(src_dim_size, "index_add src_dim_size")?,
                crate::to_u32(out_dim_size, "index_add out_dim_size")?,
                crate::to_u32(inner, "index_add inner")?,
            ]);

            pipeline
                .dispatch_buffers_with_offsets(
                    ctx,
                    &[&src_data.buffer, &idx_data.buffer],
                    &[src_data.byte_offset, idx_data.byte_offset],
                    &out_buf,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::new(out_buf);
            DynTensor::from_gpu_storage(out_shape, x.dtype(), Arc::new(storage), Device::metal())
        })
    }

    // ===== scatter (overwrite) =====
    //
    // output = self.clone(); output[index[i]] = src[i]  (non-atomic overwrite).
    // Unlike scatter_add, this uses direct writes. For duplicate indices,
    // last-write-wins (non-deterministic across threads for same position).

    /// GPU-native scatter (overwrite): write `src` values into `self` at indexed positions.
    ///
    /// Output starts as a clone of `self`. Each source element overwrites the
    /// output at the position specified by `index`. Non-atomic writes — duplicate
    /// indices result in last-write-wins behavior.
    pub(super) fn gpu_scatter(
        x: &DynTensor,
        dim: usize,
        index: &DynTensor,
        src: &DynTensor,
    ) -> Result<DynTensor> {
        // Lazy batch (#2009): flush pending GPU work before CPU readback
        // via clone_buffer_range. Without this, empty-src path reads stale data.
        crate::gpu_scope::flush()?;
        Self::validate_f32_buffer(x, "gpu_scatter")?;
        Self::validate_f32_buffer(src, "gpu_scatter(src)")?;

        let x_shape = x.dims();
        let src_shape = src.dims();
        let ndim = x_shape.len();

        check_dim(dim, ndim)?;

        let total_src = checked_dim_product(src_shape)?;
        let x_numel = checked_dim_product(x_shape)?;
        let x_bytes = x_numel
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| metal_err("buffer bytes overflow"))?;

        if total_src == 0 {
            let x_data = x.gpu_data::<MetalTensorData>()?;
            let ctx = Self::ctx()?;
            let cloned = ctx
                .clone_buffer_range(&x_data.buffer, x_data.byte_offset, x_bytes)
                .map_err(metal_err)?;
            let storage = MetalTensorData::new(cloned);
            return DynTensor::from_gpu_storage(
                x_shape.to_vec(),
                x.dtype(),
                Arc::new(storage),
                Device::metal(),
            );
        }

        let src_dim_size = src_shape[dim];
        let out_dim_size = x_shape[dim];
        let inner = checked_dim_product(&src_shape[dim + 1..])?;

        let x_data = x.gpu_data::<MetalTensorData>()?;

        let gpu_src = if src.device().is_gpu() {
            src.clone()
        } else {
            src.to_device(&Device::metal())?
        };
        let src_data = gpu_src.gpu_data::<MetalTensorData>()?;

        let cpu_u32_idx = {
            let cpu_idx = index.to_device(&Device::Cpu)?;
            cpu_idx.to_dtype(DType::U32)?
        };

        {
            let indices = cpu_u32_idx.to_flat_vec::<u32>()?;
            for &idx in &indices {
                if (idx as usize) >= out_dim_size {
                    return Err(TensorError::ValueOutOfRange {
                        description: "scatter: index out of bounds for target dimension",
                    });
                }
            }
        }

        let idx_u32 = cpu_u32_idx.to_device(&Device::metal())?;
        let idx_data = idx_u32.gpu_data::<MetalTensorData>()?;

        let ctx = Self::ctx()?;

        let msl = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scatter_f32(
    device const float*  src        [[buffer(0)]],
    device const uint*   indices    [[buffer(1)]],
    device float*        output     [[buffer(2)]],
    device const uint&   total_src  [[buffer(3)]],
    device const uint&   src_dim_sz [[buffer(4)]],
    device const uint&   out_dim_sz [[buffer(5)]],
    device const uint&   inner_sz   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_src) return;

    uint inner = tid % inner_sz;
    uint outer = tid / (src_dim_sz * inner_sz);
    uint dst_idx = indices[tid];

    // OOB guard: skip writes with invalid scatter indices.
    if (dst_idx >= out_dim_sz) return;

    uint out_offset = outer * (out_dim_sz * inner_sz) + dst_idx * inner_sz + inner;
    output[out_offset] = src[tid];
}
"#;

        let out_buf = ctx
            .clone_buffer_range(&x_data.buffer, x_data.byte_offset, x_bytes)
            .map_err(metal_err)?;
        let out_shape = x_shape.to_vec();

        super::with_pipeline_cache(|cache| {
            let pipeline = KernelPipeline::from_msl(cache, msl, "scatter_f32", 2, false)
                .map_err(metal_err)?;

            let plan = DispatchMode::Elementwise {
                total: crate::to_u32(total_src, "scatter total_src")?,
            }
            .plan()
            .map_err(metal_err)?
            .with_constants(vec![
                crate::to_u32(total_src, "scatter total_src")?,
                crate::to_u32(src_dim_size, "scatter src_dim_size")?,
                crate::to_u32(out_dim_size, "scatter out_dim_size")?,
                crate::to_u32(inner, "scatter inner")?,
            ]);

            pipeline
                .dispatch_buffers_with_offsets(
                    ctx,
                    &[&src_data.buffer, &idx_data.buffer],
                    &[src_data.byte_offset, idx_data.byte_offset],
                    &out_buf,
                    &plan,
                )
                .map_err(metal_err)?;

            let storage = MetalTensorData::new(out_buf);
            DynTensor::from_gpu_storage(out_shape, x.dtype(), Arc::new(storage), Device::metal())
        })
    }
}
