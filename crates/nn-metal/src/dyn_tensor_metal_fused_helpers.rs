// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for fused GPU kernel dispatch files.
//!
//! Eliminates duplication across the 9+ `dyn_tensor_metal_*_fused.rs` files
//! for common patterns: eps validation, output allocation, lazy batch submit.
//!
//! Part of #2441 (code_structure).

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::metal_backend::metal_err;

use super::MetalTensorData;

/// Validate eps for normalization kernels.
///
/// Rejects non-finite or zero eps. `eps=0.0` with `variance=0.0` produces
/// `rsqrt(0)=Inf` on Metal GPU, so `eps` must be strictly positive.
///
/// Returns the validated `f32` eps value.
pub(super) fn validate_eps(eps: f64, kernel_name: &str) -> Result<f32> {
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 <= 0.0 {
        return Err(TensorError::InvalidShape(format!(
            "{kernel_name}: eps must be finite and > 0, got {eps}"
        )));
    }
    Ok(eps_f32)
}

/// Allocate output buffer from arena, with checked overflow.
///
/// Computes `total_elems * elem_bytes` with overflow check, then allocates
/// from the arena (or buffer pool if arena is bypassed).
pub(super) fn alloc_output(
    ctx: &MetalContext,
    total_elems: usize,
    elem_bytes: usize,
    dims: &[usize],
) -> Result<(MetalBuffer, usize)> {
    let out_bytes =
        total_elems
            .checked_mul(elem_bytes)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            })?;
    crate::arena::arena_alloc_or_create(ctx, out_bytes).map_err(metal_err)
}

/// Submit an encode closure via lazy batch.
///
/// Handles `get_or_create_batch` + `encode_into_lazy_batch` + nested Result
/// unwrapping. Call this, then `build_output` to construct the DynTensor.
///
/// Split from `build_output` because the encode closure borrows the output
/// buffer; ownership transfers to `build_output` after the closure completes.
pub(super) fn submit_encode(
    encode: impl FnOnce(
        &crate::dispatch::CommandBatch,
    ) -> std::result::Result<(), crate::error::MetalError>,
) -> Result<()> {
    crate::gpu_scope::get_or_create_batch()?;
    let scope_result = crate::gpu_scope::encode_into_lazy_batch(|batch| encode(batch));
    match scope_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(metal_err(e)),
        Err(e) => Err(e),
    }
}

/// Build output DynTensor from arena-allocated buffer.
///
/// Call after `submit_encode` — the encode closure's borrow on `out_buf`
/// has ended, so ownership can transfer to the storage.
pub(super) fn build_output(
    out_buf: MetalBuffer,
    out_offset: usize,
    dims: &[usize],
    dtype: DType,
) -> Result<DynTensor> {
    let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
    DynTensor::from_gpu_storage(dims.to_vec(), dtype, Arc::new(storage), Device::metal())
}
