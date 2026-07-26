// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU top-k dispatch for [`MetalDynBackend`].
//!
//! Returns (values_f32, indices_u32) via raw MSL kernel. Each output lane
//! uses register-based insertion sort to find the k largest elements,
//! avoiding GPU→CPU round-trip for autoregressive decode sampling.
//!
//! Supports k ≤ 64 (register allocation limit); returns `None` for larger k
//! so the caller falls back to CPU. Covers all dvoice use cases:
//! k=50 (top-k sampling), k=2 (MoE routing).
//!
//! Part of #1331: eliminates GPU→CPU round-trip for topk.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, DType, Device, Result, TensorError};

use crate::kernel_dispatch::KernelPipeline;

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

/// Maximum k for the register-based kernel. Beyond this, CPU fallback.
const MAX_GPU_K: usize = 64;

/// Generate MSL source for the top-k kernel.
///
/// The kernel uses a register-based insertion sort: each thread maintains a
/// sorted buffer of k elements and scans the full dimension, inserting any
/// value larger than the current k-th largest. For typical parameters
/// (dim=32768, k=50) this is ~1.6M comparisons per lane — dominated by
/// memory bandwidth, not compute.
fn topk_msl(k: usize) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

kernel void topk_f32(
    device const float* input     [[buffer(0)]],
    device float*       out_vals  [[buffer(1)]],
    device uint*        out_idxs  [[buffer(2)]],
    device const uint&  num_lanes [[buffer(3)]],
    device const uint&  dim_size  [[buffer(4)]],
    device const uint&  inner_sz  [[buffer(5)]],
    device const uint&  k_val     [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= num_lanes) return;

    uint inner_idx = tid % inner_sz;
    uint outer_idx = tid / inner_sz;
    // Base offset into the input for this lane. Elements along the target
    // dimension are spaced `inner_sz` apart in row-major layout.
    uint base = outer_idx * dim_size * inner_sz + inner_idx;

    // Register-based sorted top-k buffer (insertion sort).
    float top_vals[{k}];
    uint  top_idxs[{k}];
    for (uint i = 0; i < k_val; i++) {{
        top_vals[i] = -HUGE_VALF;
        top_idxs[i] = 0;
    }}

    for (uint i = 0; i < dim_size; i++) {{
        float val = input[base + i * inner_sz];
        if (val > top_vals[k_val - 1]) {{
            // Insert into sorted position (descending).
            uint pos = k_val - 1;
            while (pos > 0 && val > top_vals[pos - 1]) {{
                top_vals[pos] = top_vals[pos - 1];
                top_idxs[pos] = top_idxs[pos - 1];
                pos--;
            }}
            top_vals[pos] = val;
            top_idxs[pos] = i;
        }}
    }}

    // Write output in row-major order matching the output tensor layout.
    // For non-last-dim topk, elements are strided: the k results for each
    // lane are interleaved with results from other lanes sharing the same
    // outer index. Position: outer_idx * k * inner_sz + i * inner_sz + inner_idx.
    for (uint i = 0; i < k_val; i++) {{
        uint out_pos = outer_idx * k_val * inner_sz + i * inner_sz + inner_idx;
        out_vals[out_pos] = top_vals[i];
        out_idxs[out_pos] = top_idxs[i];
    }}
}}
"#
    )
}

/// Dispatch the topk MSL kernel and return (values, indices) tensors.
///
/// Uses raw encoder calls because the kernel has two output buffers
/// (values at buffer(1), indices at buffer(2)) which the standard
/// `dispatch_buffers` single-output API cannot express.
fn dispatch_topk(
    x_data: &MetalTensorData,
    out_shape: &[usize],
    out_dtype: DType,
    num_lanes: usize,
    dim_size: usize,
    inner: usize,
    k: usize,
) -> Result<(DynTensor, DynTensor)> {
    let ctx = super::MetalDynBackend::ctx()?;
    let out_numel = checked_dim_product(out_shape)?;

    super::with_pipeline_cache(|cache| {
        let msl = topk_msl(k);
        let pipeline =
            KernelPipeline::from_msl(cache, &msl, "topk_f32", 1, false).map_err(metal_err)?;

        // Two output buffers: values (f32) and indices (u32), each out_numel elements.
        let val_bytes = out_numel.checked_mul(size_of::<f32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: out_shape.to_vec(),
            }
        })?;
        let idx_bytes = out_numel.checked_mul(size_of::<u32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: out_shape.to_vec(),
            }
        })?;
        let (val_buf, val_offset) =
            crate::arena::arena_alloc_or_create(ctx, val_bytes).map_err(metal_err)?;
        let val_arena_gen = crate::arena::last_alloc_generation();
        let (idx_buf, idx_offset) =
            crate::arena::arena_alloc_or_create(ctx, idx_bytes).map_err(metal_err)?;
        let idx_arena_gen = crate::arena::last_alloc_generation();

        // Manual dispatch: bind buffers according to MSL kernel signature.
        // buffer(0): input, buffer(1): out_vals, buffer(2): out_idxs,
        // buffer(3-6): constants (num_lanes, dim_size, inner_sz, k_val)
        let num_lanes_u32 = crate::to_u32(num_lanes, "topk num_lanes")?;
        let dim_size_u32 = crate::to_u32(dim_size, "topk dim_size")?;
        let inner_u32 = crate::to_u32(inner, "topk inner")?;
        let k_u32 = crate::to_u32(k, "topk k")?;
        let tg_size = 256u32.min(num_lanes_u32);

        macro_rules! encode_topk {
            ($enc:expr) => {{
                $enc.set_buffer_with_offset(0, &x_data.buffer, x_data.byte_offset);
                $enc.set_buffer_with_offset(1, &val_buf, val_offset);
                $enc.set_buffer_with_offset(2, &idx_buf, idx_offset);
                $enc.set_bytes(3, &num_lanes_u32);
                $enc.set_bytes(4, &dim_size_u32);
                $enc.set_bytes(5, &inner_u32);
                $enc.set_bytes(6, &k_u32);
                $enc.encode(pipeline.pipeline(), [num_lanes_u32, 1, 1], [tg_size, 1, 1])
            }};
        }

        // Lazy batch (#2009): encode into the thread-local lazy batch.
        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                encode_topk!(enc)?;
                enc.end_encoding();
                Ok(())
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Stamp arena generation for stale-read detection (#2328).
        let val_storage = match val_arena_gen {
            Some(g) => MetalTensorData::view_arena(val_buf.alias(), val_offset, g),
            None if val_offset > 0 => MetalTensorData::view(val_buf.alias(), val_offset),
            None => MetalTensorData::new(val_buf),
        };
        let idx_storage = match idx_arena_gen {
            Some(g) => MetalTensorData::view_arena(idx_buf.alias(), idx_offset, g),
            None if idx_offset > 0 => MetalTensorData::view(idx_buf.alias(), idx_offset),
            None => MetalTensorData::new(idx_buf),
        };
        let values = DynTensor::from_gpu_storage(
            out_shape.to_vec(),
            out_dtype,
            Arc::new(val_storage),
            Device::metal(),
        )?;
        let indices = DynTensor::from_gpu_storage(
            out_shape.to_vec(),
            DType::U32,
            Arc::new(idx_storage),
            Device::metal(),
        )?;
        Ok((values, indices))
    })
}

impl super::MetalDynBackend {
    /// GPU-native top-k: returns (values, indices) along `dim`.
    ///
    /// Returns `None` for k > 64 (CPU fallback). Results are sorted
    /// descending by value within each lane.
    pub(super) fn gpu_topk(
        x: &DynTensor,
        dim: usize,
        k: usize,
    ) -> Option<Result<(DynTensor, DynTensor)>> {
        if k > MAX_GPU_K {
            return crate::gpu_fallback("topk", "k > 64 exceeds register limit");
        }
        Some(Self::gpu_topk_inner(x, dim, k))
    }

    fn gpu_topk_inner(x: &DynTensor, dim: usize, k: usize) -> Result<(DynTensor, DynTensor)> {
        // Lazy batch (#2009): flush pending GPU work so the NaN check
        // below (sum_all → to_scalar) reads committed data.
        crate::gpu_scope::flush().map_err(|e| TensorError::InvalidShape(e.to_string()))?;
        Self::validate_f32_buffer(x, "gpu_topk")?;
        let shape = x.dims();
        let ndim = shape.len();
        check_dim(dim, ndim)?;
        let dim_size = shape[dim];
        if k == 0 || k > dim_size {
            return Err(TensorError::ValueOutOfRange {
                description: "gpu_topk: k must be in [1, dim_size]",
            });
        }

        // AC3: NaN rejection matching CPU behavior.
        // Fast path: GPU sum reduction detects NaN (NaN propagates through
        // addition). If sum is finite or ±Inf, there are no NaN values.
        // Edge case: sum can be NaN from Inf + (-Inf) cancellation without
        // any actual NaN in the input. In that case, fall back to CPU check
        // to distinguish real NaN from Inf cancellation.
        let sum_scalar = x.sum_all()?.to_scalar::<f32>()?;
        if sum_scalar.is_nan() {
            // Slow path: sum was NaN, could be real NaN or Inf+(-Inf).
            // Transfer to CPU to check definitively. This path is rare in
            // practice (requires both +Inf and -Inf or actual NaN).
            let cpu_arr = x.to_device(&Device::Cpu)?.to_f32_array()?;
            if cpu_arr.iter().any(|v| v.is_nan()) {
                return Err(TensorError::ValueOutOfRange {
                    description: "topk: input contains NaN",
                });
            }
        }

        // Output shape: same as input but dim replaced by k.
        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape[dim] = k;

        // Number of independent lanes to process.
        let inner = checked_dim_product(&shape[dim + 1..])?;
        let outer = checked_dim_product(&shape[..dim])?;
        let num_lanes = outer
            .checked_mul(inner)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        dispatch_topk(x_data, &out_shape, x.dtype(), num_lanes, dim_size, inner, k)
    }
}
