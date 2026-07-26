// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native index_select implementation for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal_conv_ops.rs` — `gpu_index_select` is a
//! selection/embedding operation, not a convolution. Placed in its own file
//! for conceptual clarity and to keep conv_ops focused on convolution.
//!
//! Part of #1098 (code structure extraction).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native index_select along any dimension.
    ///
    /// Uses a raw MSL kernel with native `uint*` indices to preserve precision
    /// for indices > 2^24 (f32 has only 24-bit mantissa). This is separate
    /// from the `emit_embedding_kernel` path used by `execute_tensor_dispatch`
    /// contract tests (which passes all-f32 buffers).
    ///
    /// The kernel decomposes each output element into (outer, index, inner)
    /// coordinates relative to the selected dimension, supporting arbitrary
    /// `dim` values (not just dim=0). This eliminates CPU fallback for
    /// non-dim-0 index_select used by Vocos GAN and other models (#1997).
    ///
    /// For non-f32 dtypes, returns `None` to fall back to CPU.
    pub(super) fn gpu_index_select(
        x: &DynTensor,
        ids: &DynTensor,
        dim: usize,
    ) -> Option<Result<DynTensor>> {
        if Self::validate_f32_buffer(x, "gpu_index_select").is_err() {
            return crate::gpu_fallback("index_select", "non-f32 dtype not supported on Metal");
        }

        let x_shape = x.dims();
        if x_shape.is_empty() {
            return crate::gpu_fallback("index_select", "rank 0 not supported on Metal");
        }
        if dim >= x_shape.len() {
            return Some(Err(TensorError::InvalidShape(format!(
                "index_select: dim {dim} out of range for rank {}",
                x_shape.len(),
            ))));
        }

        let x_data = match x.gpu_data::<MetalTensorData>() {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        let num_indices = match crate::metal_backend::checked_dim_product(ids.dims()) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let dim_size = x_shape[dim];

        // inner_size = product of dims after `dim`
        let inner_size = match crate::metal_backend::checked_dim_product(&x_shape[dim + 1..]) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };

        // total output elements = outer_size * num_indices * inner_size
        let outer_size = match crate::metal_backend::checked_dim_product(&x_shape[..dim]) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let total = match outer_size
            .checked_mul(num_indices)
            .and_then(|v| v.checked_mul(inner_size))
        {
            Some(v) => v,
            None => {
                return Some(Err(TensorError::DimensionOverflow {
                    dims: x_shape.to_vec(),
                }))
            }
        };

        // Fast path: if indices are already GPU U32, use them directly.
        // This avoids a GPU→CPU→GPU round-trip that causes a flush per call.
        // The MSL kernel clamps OOB rows as defense-in-depth.
        // Part of #2958: eliminates 4 flushes in SineGen (interp_{down,up}sample).
        let gpu_u32_fast = ids.device().is_metal() && ids.dtype() == nn_core::DType::U32;

        // Slow path: holds the validated GPU U32 tensor to extend its lifetime.
        let _ids_owned;
        let ids_data = if gpu_u32_fast {
            match ids.gpu_data::<MetalTensorData>() {
                Ok(d) => d,
                Err(e) => return Some(Err(e)),
            }
        } else {
            // Convert to CPU U32 for host-side OOB validation + GPU upload.
            let cpu_u32_ids = {
                let cpu_ids = match ids.to_device(&nn_core::Device::Cpu) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e)),
                };
                match cpu_ids.to_dtype(nn_core::DType::U32) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e)),
                }
            };

            // Host-side OOB validation: match CPU index_select error behavior.
            {
                let indices = match cpu_u32_ids.to_flat_vec::<u32>() {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                for &idx in &indices {
                    if (idx as usize) >= dim_size {
                        return Some(Err(TensorError::EmbeddingIndexOutOfRange {
                            index: idx as usize,
                            vocab_size: dim_size,
                        }));
                    }
                }
            }

            // Upload validated indices to GPU.
            _ids_owned = match cpu_u32_ids.to_device(&nn_core::Device::metal()) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            };
            match _ids_owned.gpu_data::<MetalTensorData>() {
                Ok(d) => d,
                Err(e) => return Some(Err(e)),
            }
        };

        // Build output shape: replace x_shape[dim] with num_indices.
        let mut out_shape = Vec::with_capacity(x_shape.len());
        out_shape.extend_from_slice(&x_shape[..dim]);
        out_shape.push(num_indices);
        out_shape.extend_from_slice(&x_shape[dim + 1..]);

        // Generalized index_select MSL kernel.
        //
        // For each output element at flat position `tid`, we decompose into:
        //   outer_idx = tid / (num_indices * inner_size)
        //   index_idx = (tid / inner_size) % num_indices
        //   inner_idx = tid % inner_size
        //
        // The source element in the input tensor is at:
        //   src = outer_idx * (dim_size * inner_size) + row * inner_size + inner_idx
        //
        // where `row = indices[index_idx]`.
        let msl = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_u32(
    device const uint*  indices    [[buffer(0)]],
    device const float* src        [[buffer(1)]],
    device float*       dst        [[buffer(2)]],
    device const uint&  total_els  [[buffer(3)]],
    device const uint&  inner_sz   [[buffer(4)]],
    device const uint&  dim_sz     [[buffer(5)]],
    device const uint&  n_indices  [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;
    uint inner_idx  = tid % inner_sz;
    uint index_idx  = (tid / inner_sz) % n_indices;
    uint outer_idx  = tid / (n_indices * inner_sz);
    uint row = indices[index_idx];
    // Defense-in-depth: clamp OOB rows (host already validated).
    if (row >= dim_sz) row = dim_sz - 1;
    uint src_offset = outer_idx * (dim_sz * inner_sz) + row * inner_sz + inner_idx;
    dst[tid] = src[src_offset];
}
"#;

        Some(Self::dispatch_raw_msl(
            msl,
            "index_select_u32",
            2, // param_count: indices + src
            &[&ids_data.buffer, &x_data.buffer],
            &[ids_data.byte_offset, x_data.byte_offset],
            total,
            &out_shape,
            x.dtype(),
            vec![
                match crate::to_u32(total, "index_select total") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                match crate::to_u32(inner_size, "index_select inner_size") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                match crate::to_u32(dim_size, "index_select dim_size") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                match crate::to_u32(num_indices, "index_select num_indices") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
            ],
        ))
    }

    /// GPU-native index_select without OOB validation.
    ///
    /// Caller guarantees all index values are `< x.dims()[dim]`. The MSL kernel
    /// clamps OOB as defense-in-depth but clamping masks bugs silently.
    ///
    /// Supports GPU U32 indices (direct) and GPU F32 indices (inline cast to uint
    /// in MSL — no separate conversion pass, no CPU readback).
    ///
    /// Part of #2653: eliminates ~40ms of GPU flushes per Kokoro synthesis.
    pub(super) fn gpu_index_select_unchecked(
        x: &DynTensor,
        ids: &DynTensor,
        dim: usize,
    ) -> Option<Result<DynTensor>> {
        if Self::validate_f32_buffer(x, "gpu_index_select_unchecked").is_err() {
            return crate::gpu_fallback(
                "index_select_unchecked",
                "non-f32 dtype not supported on Metal",
            );
        }

        let x_shape = x.dims();
        if x_shape.is_empty() {
            return crate::gpu_fallback("index_select_unchecked", "rank 0 not supported");
        }
        if dim >= x_shape.len() {
            return Some(Err(TensorError::InvalidShape(format!(
                "index_select_unchecked: dim {dim} out of range for rank {}",
                x_shape.len(),
            ))));
        }

        // Require GPU-resident indices (U32 or F32).
        if !ids.device().is_metal() {
            return crate::gpu_fallback(
                "index_select_unchecked",
                "indices must be on Metal device",
            );
        }
        let is_u32 = ids.dtype() == nn_core::DType::U32;
        let is_f32 = ids.dtype() == nn_core::DType::F32;
        if !is_u32 && !is_f32 {
            return crate::gpu_fallback(
                "index_select_unchecked",
                "only U32 and F32 indices supported",
            );
        }

        let x_data = match x.gpu_data::<MetalTensorData>() {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };
        let ids_data = match ids.gpu_data::<MetalTensorData>() {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        let num_indices = match crate::metal_backend::checked_dim_product(ids.dims()) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let dim_size = x_shape[dim];
        let inner_size = match crate::metal_backend::checked_dim_product(&x_shape[dim + 1..]) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let outer_size = match crate::metal_backend::checked_dim_product(&x_shape[..dim]) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let total = match outer_size
            .checked_mul(num_indices)
            .and_then(|v| v.checked_mul(inner_size))
        {
            Some(v) => v,
            None => {
                return Some(Err(TensorError::DimensionOverflow {
                    dims: x_shape.to_vec(),
                }))
            }
        };

        let mut out_shape = Vec::with_capacity(x_shape.len());
        out_shape.extend_from_slice(&x_shape[..dim]);
        out_shape.push(num_indices);
        out_shape.extend_from_slice(&x_shape[dim + 1..]);

        // Select MSL kernel based on index dtype.
        let (msl, kernel_name) = if is_u32 {
            (INDEX_SELECT_U32_MSL, "index_select_u32_unchecked")
        } else {
            (INDEX_SELECT_F32_MSL, "index_select_f32_unchecked")
        };

        Some(Self::dispatch_raw_msl(
            msl,
            kernel_name,
            2, // param_count: indices + src
            &[&ids_data.buffer, &x_data.buffer],
            &[ids_data.byte_offset, x_data.byte_offset],
            total,
            &out_shape,
            x.dtype(),
            vec![
                match crate::to_u32(total, "index_select_unchecked total") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                match crate::to_u32(inner_size, "index_select_unchecked inner_size") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                match crate::to_u32(dim_size, "index_select_unchecked dim_size") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
                match crate::to_u32(num_indices, "index_select_unchecked num_indices") {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                },
            ],
        ))
    }
}

/// MSL kernel for index_select with U32 indices (no host-side OOB validation).
const INDEX_SELECT_U32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_u32_unchecked(
    device const uint*  indices    [[buffer(0)]],
    device const float* src        [[buffer(1)]],
    device float*       dst        [[buffer(2)]],
    device const uint&  total_els  [[buffer(3)]],
    device const uint&  inner_sz   [[buffer(4)]],
    device const uint&  dim_sz     [[buffer(5)]],
    device const uint&  n_indices  [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;
    uint inner_idx  = tid % inner_sz;
    uint index_idx  = (tid / inner_sz) % n_indices;
    uint outer_idx  = tid / (n_indices * inner_sz);
    uint row = indices[index_idx];
    if (row >= dim_sz) row = dim_sz - 1;
    uint src_offset = outer_idx * (dim_sz * inner_sz) + row * inner_sz + inner_idx;
    dst[tid] = src[src_offset];
}
"#;

/// MSL kernel for index_select with F32 indices (inline cast to uint, no flush).
const INDEX_SELECT_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void index_select_f32_unchecked(
    device const float* indices    [[buffer(0)]],
    device const float* src        [[buffer(1)]],
    device float*       dst        [[buffer(2)]],
    device const uint&  total_els  [[buffer(3)]],
    device const uint&  inner_sz   [[buffer(4)]],
    device const uint&  dim_sz     [[buffer(5)]],
    device const uint&  n_indices  [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;
    uint inner_idx  = tid % inner_sz;
    uint index_idx  = (tid / inner_sz) % n_indices;
    uint outer_idx  = tid / (n_indices * inner_sz);
    uint row = uint(indices[index_idx]);
    if (row >= dim_sz) row = dim_sz - 1;
    uint src_offset = outer_idx * (dim_sz * inner_sz) + row * inner_sz + inner_idx;
    dst[tid] = src[src_offset];
}
"#;
