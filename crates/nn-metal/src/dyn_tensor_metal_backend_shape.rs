// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`GpuShapeOps`] implementation for `MetalDynBackend`.
//!
//! 7 shape methods: narrow, transpose, permute, cat, expand, unfold, slice_set.
//! Extracted from `dyn_tensor_metal_backend_impl.rs` (#1917).

use nn_core::dyn_tensor::{DynTensor, GpuShapeOps};
use nn_core::Result;

use super::helpers::needs_non_float_fallback;
use super::MetalDynBackend;

impl GpuShapeOps for MetalDynBackend {
    fn narrow(
        &self,
        x: &DynTensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Option<Result<DynTensor>> {
        // Integer GPU tensors (U32 from argmax/topk) have no dispatch_def mapping.
        if needs_non_float_fallback(x) {
            return None;
        }
        Some(Self::gpu_narrow(x, dim, start, len))
    }

    fn transpose(&self, x: &DynTensor, d1: usize, d2: usize) -> Option<Result<DynTensor>> {
        if needs_non_float_fallback(x) {
            return None;
        }
        Some(Self::gpu_transpose(x, d1, d2))
    }

    fn permute(&self, x: &DynTensor, dims: &[usize]) -> Option<Result<DynTensor>> {
        if needs_non_float_fallback(x) {
            return None;
        }
        Some(Self::gpu_permute(x, dims))
    }

    fn cat(&self, tensors: &[&DynTensor], dim: usize) -> Option<Result<DynTensor>> {
        // Any non-float tensor in the list requires CPU fallback.
        if tensors.iter().any(|t| needs_non_float_fallback(t)) {
            return None;
        }
        Some(Self::gpu_cat(tensors, dim))
    }

    fn expand(&self, x: &DynTensor, new_dims: &[usize]) -> Option<Result<DynTensor>> {
        if needs_non_float_fallback(x) {
            return None;
        }
        Some(Self::gpu_expand(x, new_dims))
    }

    fn unfold(
        &self,
        x: &DynTensor,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Option<Result<DynTensor>> {
        if needs_non_float_fallback(x) {
            return None;
        }
        Some(Self::gpu_unfold(x, dim, size, step))
    }

    fn slice_set(
        &self,
        dst: &DynTensor,
        dim: usize,
        offset: usize,
        src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        // Integer tensors (U32, I64, etc.) fall back to CPU.
        // Float tensors (F32, BF16, F16) stay on GPU — gpu_slice_set uses
        // byte-level copies that handle any element width (#1711).
        if needs_non_float_fallback(dst) || needs_non_float_fallback(src) {
            return None;
        }
        Some(Self::gpu_slice_set(dst, dim, offset, src))
    }

    fn pad(
        &self,
        x: &DynTensor,
        padding: &[usize],
        value: f64,
    ) -> Option<Result<DynTensor>> {
        if needs_non_float_fallback(x) {
            return None;
        }
        Some(Self::gpu_pad(x, padding, value))
    }
}
