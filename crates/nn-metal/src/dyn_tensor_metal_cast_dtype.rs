// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native dtype conversion kernels for [`MetalDynBackend`].
//!
//! Eliminates the GPU→CPU→GPU round-trip for cross-byte-width float conversions
//! (F32↔F16, F32↔BF16). Uses raw MSL kernels that read one element type and
//! write another.
//!
//! Before this: `to_dtype(F16)` on a GPU F32 tensor transferred to CPU, converted
//! element-by-element, then transferred back. For a 1M-element tensor, that's
//! ~4MB GPU→CPU + CPU conversion + ~2MB CPU→GPU. With this kernel, it's a single
//! GPU dispatch reading `float*` and writing `half*` (or vice versa).
//!
//! Impact: Every softmax auto-upcast (BF16/F16→F32→BF16/F16), every precision-
//! sensitive unary op, and every norm auto-upcast now stays on GPU.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

/// MSL kernel: F32 → F16 element-wise conversion.
///
/// Each thread reads one `float`, casts to `half`, writes output.
/// Apple Metal `half` maps to IEEE 754 binary16 (same as Rust `f16`).
/// BF16 tensors use `half` buffers on Apple GPUs (no native bf16 ALU).
const CAST_F32_TO_F16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cast_f32_to_f16(
    device const float* input  [[buffer(0)]],
    device half*        output [[buffer(1)]],
    device const uint&  total  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    output[tid] = half(input[tid]);
}
"#;

/// MSL kernel: F16 → F32 element-wise conversion.
///
/// Each thread reads one `half`, casts to `float`, writes output.
const CAST_F16_TO_F32_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cast_f16_to_f32(
    device const half*  input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    device const uint&  total  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    output[tid] = float(input[tid]);
}
"#;

impl super::MetalDynBackend {
    /// GPU-native dtype cast for cross-byte-width float conversions.
    ///
    /// Returns `Some(Ok(...))` for F32↔F16/BF16 conversions.
    /// Returns `None` for unsupported conversion pairs (caller uses CPU fallback).
    pub(super) fn gpu_cast_dtype(x: &DynTensor, target_dtype: DType) -> Option<Result<DynTensor>> {
        // Only handle cross-byte-width float conversions.
        let src = x.dtype();
        if src == target_dtype {
            return Some(Ok(x.clone()));
        }
        if !src.is_float() || !target_dtype.is_float() {
            return None;
        }

        // Determine direction. Both BF16 and F16 use 2-byte `half` Metal buffers.
        let (msl, entry, dst_bytes) = match (src, target_dtype) {
            // F32 → F16 or BF16
            (DType::F32, DType::F16 | DType::BF16) => {
                (CAST_F32_TO_F16_MSL, "cast_f32_to_f16", 2usize)
            }
            // F16 or BF16 → F32
            (DType::F16 | DType::BF16, DType::F32) => {
                (CAST_F16_TO_F32_MSL, "cast_f16_to_f32", 4usize)
            }
            // F64 → stored as F32 internally, so F64↔F16 goes via F32 on CPU.
            // Same-byte-width (BF16↔F16) handled by zero-copy relabel upstream.
            _ => return None,
        };

        Some(gpu_cast_inner(x, target_dtype, msl, entry, dst_bytes))
    }
}

/// Inner implementation: compile MSL, allocate output buffer, dispatch.
fn gpu_cast_inner(
    x: &DynTensor,
    target_dtype: DType,
    msl: &str,
    entry_point: &str,
    dst_bytes: usize,
) -> Result<DynTensor> {
    let shape = x.dims();
    let total = checked_dim_product(shape)?;
    if total == 0 {
        return DynTensor::zeros(shape, target_dtype, &Device::metal());
    }

    let x_data = x.gpu_data::<MetalTensorData>()?;
    let ctx = super::MetalDynBackend::ctx()?;

    super::with_pipeline_cache(|cache| {
        use crate::dispatch_plan::DispatchMode;
        use crate::kernel_dispatch::KernelPipeline;

        let pipeline =
            KernelPipeline::from_msl(cache, msl, entry_point, 1, false).map_err(metal_err)?;

        let out_bytes =
            total
                .checked_mul(dst_bytes)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: shape.to_vec(),
                })?;
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)?;

        let plan = DispatchMode::Elementwise {
            total: crate::to_u32(total, "cast_dtype total")?,
        }
        .plan()
        .map_err(metal_err)?
        .with_constants(vec![crate::to_u32(total, "cast_dtype total_const")?]);

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
        DynTensor::from_gpu_storage(
            shape.to_vec(),
            target_dtype,
            Arc::new(storage),
            Device::metal(),
        )
    })
}

#[cfg(test)]
#[path = "dyn_tensor_metal_cast_dtype_tests.rs"]
mod tests;
