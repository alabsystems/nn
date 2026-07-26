// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native unfold (sliding window extraction) for [`MetalDynBackend`].
//!
//! Single-dispatch replacement for O(n_frames) narrow() calls in STFT framing.
//! For `[B, C, T]` with `unfold(2, fft_size, hop_size)` → `[B, C, n_frames, fft_size]`
//! in one GPU kernel instead of 87K narrow dispatches (#1945).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

/// Generate MSL kernel source for unfold with compile-time baked strides.
///
/// Since rank is known at kernel generation time, all stride values are embedded
/// directly as MSL constants. This avoids runtime array buffer passing and
/// produces optimal GPU code with no indirection.
///
/// Buffer layout:
/// - `buffer(0)`: input float data
/// - `buffer(1)`: output float data (allocated by caller)
/// - `buffer(2)`: total_out (u32 constant via dispatch plan)
fn generate_unfold_msl(in_strides: &[u32], out_strides: &[u32], dim: usize, step: u32) -> String {
    let _in_rank = in_strides.len();
    let out_rank = out_strides.len(); // in_rank + 1
    let last_axis = out_rank - 1;

    // Build the unrolled index mapping body.
    // For each output axis d, compute coord = remaining / out_strides[d] and
    // map it to the input flat index.
    let mut body = String::new();
    for d in 0..out_rank {
        let os = out_strides[d];
        if d == dim {
            // Window index: maps to input[..., w*step, ...] at the unfold dimension.
            let is = in_strides[dim];
            let stride_product = u64::from(step) * u64::from(is);
            body.push_str(&format!(
                "    {{ uint coord = remaining / {os}u; remaining = remaining % {os}u; \
                 in_idx += coord * {stride_product}u; }}\n"
            ));
        } else if d == last_axis {
            // Within-window position k: maps to input's unfold dim stride.
            let is = in_strides[dim];
            body.push_str(&format!(
                "    {{ uint coord = remaining / {os}u; remaining = remaining % {os}u; \
                 in_idx += coord * {is}u; }}\n"
            ));
        } else {
            // Regular axis: direct mapping to the same input axis.
            // For d < dim: input axis d. For dim < d < last_axis: input axis d
            // (output inserted trailing axis, not before dim).
            let is = in_strides[d];
            body.push_str(&format!(
                "    {{ uint coord = remaining / {os}u; remaining = remaining % {os}u; \
                 in_idx += coord * {is}u; }}\n"
            ));
        }
    }

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void unfold_f32(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    device const uint&  total   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= total) return;

    uint remaining = tid;
    uint in_idx = 0;

{body}
    output[tid] = input[in_idx];
}}
"#
    )
}

impl super::super::MetalDynBackend {
    /// GPU-native unfold: extract overlapping sliding windows along a dimension.
    ///
    /// For input shape `[d0, ..., d_dim, ..., dN]`, output shape is
    /// `[d0, ..., n_windows, ..., dN, size]` where the `dim` axis is replaced
    /// by `n_windows = (d_dim - size) / step + 1` and `size` is appended as
    /// a trailing dimension.
    ///
    /// Each output thread at flat index `tid` decomposes into multi-dim coordinates,
    /// computes the source element at `w*step + k` for the unfold dimension, and
    /// copies it to the output buffer. One GPU dispatch replaces O(n_windows) narrow
    /// calls.
    ///
    /// The MSL kernel is generated per-(rank, shape, dim, step) with all stride
    /// values baked as compile-time constants — no runtime array buffers needed.
    ///
    /// Supports byte-offset input buffers from zero-copy narrow views (#1945).
    pub(crate) fn gpu_unfold(
        x: &DynTensor,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_unfold")?;
        let shape = x.dims();
        let rank = shape.len();

        // Defense-in-depth: validate parameters (also validated at DynTensor level).
        check_dim(dim, rank)?;
        if size == 0 || step == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_unfold: size and step must be > 0".into(),
            ));
        }
        let dim_size = shape[dim];
        if size > dim_size {
            return Err(TensorError::InvalidShape(format!(
                "gpu_unfold: size ({size}) exceeds dimension size ({dim_size})"
            )));
        }
        let n_windows = (dim_size - size) / step + 1;

        // Build output shape: replace dim with n_windows, append size at end.
        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape[dim] = n_windows;
        out_shape.push(size);

        let total_out = checked_dim_product(&out_shape)?;
        if total_out == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_unfold: output has zero elements".into(),
            ));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;

        // Compute input strides (row-major) for the source element lookup.
        let mut in_strides = vec![1u32; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1]
                .checked_mul(crate::to_u32(shape[i + 1], "gpu_unfold in_stride")?)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: shape.to_vec(),
                })?;
        }

        // Compute output strides (row-major) for flat index decomposition.
        let out_rank = out_shape.len(); // rank + 1
        let mut out_strides = vec![1u32; out_rank];
        for i in (0..out_rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1]
                .checked_mul(crate::to_u32(out_shape[i + 1], "gpu_unfold out_stride")?)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: out_shape.clone(),
                })?;
        }

        let step_u32 = crate::to_u32(step, "gpu_unfold step")?;
        let msl = generate_unfold_msl(&in_strides, &out_strides, dim, step_u32);

        let ctx = Self::ctx()?;

        // Allocate output buffer (arena-aware).
        let out_bytes = total_out.checked_mul(size_of::<f32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: out_shape.clone(),
            }
        })?;
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        super::super::with_pipeline_cache(|cache| {
            // param_count=1 (input buffer). Output at buffer(1), constants at buffer(2+).
            let pipeline =
                KernelPipeline::from_msl(cache, &msl, "unfold_f32", 1, true).map_err(metal_err)?;

            let total_u32 = crate::to_u32(total_out, "gpu_unfold total_out")?;
            let plan = DispatchMode::Elementwise { total: total_u32 }
                .plan()
                .map_err(metal_err)?
                .with_constants(vec![total_u32]);

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
            DynTensor::from_gpu_storage(out_shape, x.dtype(), Arc::new(storage), Device::metal())
        })
    }
}
