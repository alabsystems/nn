// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for `GpuBackend` trait implementation.
//!
//! Extracted from `dyn_tensor_metal_backend_impl.rs` for the 500-line limit.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Result};

/// Return `true` (CPU fallback) if the tensor is not a float dtype.
///
/// IR-based shape ops (narrow, transpose, permute, cat, expand) use
/// `dispatch_def` which supports F32 (float*) and BF16/F16 (half*) buffers.
/// Integer GPU tensors (U32 from argmax/topk, I64 token IDs) have no MSL
/// type mapping in dispatch_def and must fall back to CPU. Returning `None`
/// from the `GpuBackend` trait triggers graceful CPU fallback; returning
/// `Some(Err)` propagates as a hard error.
pub(super) fn needs_non_float_fallback(x: &DynTensor) -> bool {
    !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
}

/// Return `true` (CPU fallback) if the tensor is not F32.
///
/// Used by two categories of GPU ops that cannot handle bf16/f16 buffers:
///
/// 1. **Raw MSL kernels** (gather, topk, scatter, cumsum, argreduce,
///    repeat_interleave) — hardcoded `float*` buffer types (#1668).
/// 2. **IR ops with dtype-mismatch** (compare, where_cond, slice_set)
///    — `dispatch_def` uses a single dtype for both input AND output buffers,
///    so bf16 compare would produce bf16 masks instead of f32 (#1646);
///    slice_set reads/writes as f32 (#1671).
///
/// Note: norm ops (layer_norm, rms_norm, group_norm) use
/// `gpu_norm_with_dtype_promotion` instead of this fallback (#1699).
pub(super) fn needs_f32_fallback(x: &DynTensor) -> bool {
    x.dtype() != DType::F32
}

/// Promote a GPU tensor to F32 if needed. No-op for F32 tensors.
pub(super) fn promote_to_f32(x: &DynTensor) -> Result<DynTensor> {
    if x.dtype() == DType::F32 {
        Ok(x.clone())
    } else {
        x.to_dtype(DType::F32)
    }
}

/// Ensure a parameter tensor matches the target dtype. No-op if already matching.
///
/// Used by fused MSL kernels (RmsNorm, GroupNorm, Snake) that accept half I/O
/// with float accumulators (#3294). Converts weight/bias to match input dtype
/// instead of promoting everything to F32.
pub(super) fn ensure_matching_dtype(param: &DynTensor, target: DType) -> Result<DynTensor> {
    if param.dtype() == target {
        Ok(param.clone())
    } else {
        param.to_dtype(target)
    }
}

/// Minimum reduction dimension for routing to fused MSL norm kernels.
///
/// Below this threshold, the decomposed TensorBlockBuilder path (with F32
/// promotion) is used instead. At `TG_SIZE=256`, `hidden_dim < 256` means
/// threadgroup occupancy < 50%, making the fused kernel slower than the
/// IR-compiled version.
///
/// Kokoro channel dimensions: 48, 96, 192, 384, 512.
/// - 48, 96, 192 → decomposed (< 256)
/// - 384, 512 → fused (>= 256)
///
/// Source: R1 design D5/D7, #3348.
pub(super) const FUSED_NORM_MIN_REDUCTION: usize = 256;

/// Run a fused GPU norm kernel with dtype promotion for BF16/F16 inputs (#1699).
///
/// If the input is already F32, runs the kernel directly. For BF16/F16,
/// promotes input to F32, runs the kernel, and casts the result back to
/// the original dtype. This avoids the CPU round-trip path where recip()
/// finiteness checks reject near-zero variance as Inf.
pub(super) fn gpu_norm_with_dtype_promotion(
    x: &DynTensor,
    kernel_fn: impl FnOnce(DynTensor) -> Result<DynTensor>,
) -> Result<DynTensor> {
    let orig_dtype = x.dtype();
    if orig_dtype == DType::F32 {
        return kernel_fn(x.clone());
    }
    // BF16/F16: promote → compute in F32 → cast back.
    let x32 = x.to_dtype(DType::F32)?;
    let result32 = kernel_fn(x32)?;
    result32.to_dtype(orig_dtype)
}
