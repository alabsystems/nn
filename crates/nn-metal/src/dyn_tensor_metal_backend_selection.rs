// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`GpuSelectionOps`] implementation for `MetalDynBackend`.
//!
//! 11 selection/comparison methods: index_select, gather, compare, compare_tensor,
//! where_cond, scatter_add, cumsum, repeat_interleave, argmax, argmin, topk.
//! Extracted from `dyn_tensor_metal_backend_impl.rs` (#1917).

use nn_core::dyn_tensor::{CompareOp, DynTensor, GpuSelectionOps};
use nn_core::Result;

use super::helpers::needs_f32_fallback;
use super::MetalDynBackend;

impl GpuSelectionOps for MetalDynBackend {
    fn index_select(
        &self,
        x: &DynTensor,
        ids: &DynTensor,
        dim: usize,
    ) -> Option<Result<DynTensor>> {
        Self::gpu_index_select(x, ids, dim)
    }

    fn index_select_unchecked(
        &self,
        x: &DynTensor,
        ids: &DynTensor,
        dim: usize,
    ) -> Option<Result<DynTensor>> {
        Self::gpu_index_select_unchecked(x, ids, dim)
    }

    fn gather(&self, x: &DynTensor, ids: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) {
            return None;
        }
        Some(Self::gpu_gather(x, ids, dim))
    }

    fn compare(&self, x: &DynTensor, op: CompareOp, val: f64) -> Option<Result<DynTensor>> {
        // Compare ops produce F32 masks (0.0/1.0). dispatch_def uses out_dtype for
        // both input and output buffer types, so bf16 input → bf16 output mask.
        // gpu_where_cond rejects non-F32 masks. Fall back to CPU for bf16 (#1646).
        if needs_f32_fallback(x) {
            return None;
        }
        Some(Self::gpu_compare(x, op, val))
    }

    fn compare_tensor(
        &self,
        lhs: &DynTensor,
        op: CompareOp,
        rhs: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        // Compare ops produce F32 masks. Same reasoning as compare() above.
        if needs_f32_fallback(lhs) || needs_f32_fallback(rhs) {
            return None;
        }
        Some(Self::gpu_compare_tensor(lhs, op, rhs))
    }

    fn where_cond(
        &self,
        mask: &DynTensor,
        on_true: &DynTensor,
        on_false: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(on_true) || needs_f32_fallback(on_false) {
            return None;
        }
        Some(Self::gpu_where_cond(mask, on_true, on_false))
    }

    fn index_add(
        &self,
        x: &DynTensor,
        dim: usize,
        index: &DynTensor,
        src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) || needs_f32_fallback(src) {
            return None;
        }
        Some(Self::gpu_index_add(x, dim, index, src))
    }

    fn scatter_add(
        &self,
        x: &DynTensor,
        dim: usize,
        index: &DynTensor,
        src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) || needs_f32_fallback(src) {
            return None;
        }
        Some(Self::gpu_scatter_add(x, dim, index, src))
    }

    fn cumsum(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) {
            return None;
        }
        // WARNING: GPU cumsum accumulates in f32 only, diverging from CPU f64
        // accumulation. See gpu_cumsum() doc comment for precision implications.
        // Fall back to CPU for axis sizes > 65536 (multi-pass Blelloch limit)
        if dim < x.rank() && x.dims()[dim] > 256 * 256 {
            return crate::gpu_fallback("cumsum", "axis size > 65536 exceeds Blelloch limit");
        }
        Some(Self::gpu_cumsum(x, dim))
    }

    fn cumsum_kahan(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) {
            return None;
        }
        if dim < x.rank() && x.dims()[dim] > Self::KAHAN_MAX_AXIS {
            return crate::gpu_fallback("cumsum_kahan", "axis size exceeds Kahan sequential limit");
        }
        Some(Self::cumsum_kahan_gpu(x, dim))
    }

    fn repeat_interleave(
        &self,
        x: &DynTensor,
        dim: usize,
        counts: &[usize],
    ) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) {
            return None;
        }
        Some(Self::gpu_repeat_interleave(x, dim, counts))
    }

    fn repeat_interleave_from_gpu(
        &self,
        x: &DynTensor,
        dim: usize,
        counts: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) || needs_f32_fallback(counts) {
            return None;
        }
        // GPU-native path limited to counts.len() <= 256 (single-threadgroup
        // Blelloch prefix sum). Larger falls back to CPU-counts path.
        if dim < x.rank() && x.dims()[dim] > 256 {
            return None;
        }
        Some(Self::gpu_repeat_interleave_from_gpu(x, dim, counts))
    }

    fn argmax(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        if !x.device().is_gpu() || needs_f32_fallback(x) {
            return None;
        }
        Some(Self::gpu_argmax(x, dim))
    }

    fn argmin(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        if !x.device().is_gpu() || needs_f32_fallback(x) {
            return None;
        }
        Some(Self::gpu_argmin(x, dim))
    }

    fn topk(&self, x: &DynTensor, dim: usize, k: usize) -> Option<Result<(DynTensor, DynTensor)>> {
        if needs_f32_fallback(x) {
            return None;
        }
        Self::gpu_topk(x, dim, k)
    }

    fn scatter(
        &self,
        x: &DynTensor,
        dim: usize,
        index: &DynTensor,
        src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        if needs_f32_fallback(x) || needs_f32_fallback(src) {
            return None;
        }
        Some(Self::gpu_scatter(x, dim, index, src))
    }

    fn sort(
        &self,
        x: &DynTensor,
        dim: usize,
        descending: bool,
    ) -> Option<Result<(DynTensor, DynTensor)>> {
        if needs_f32_fallback(x) {
            return None;
        }
        // Bitonic sort requires power-of-2 padded axis; fall back to CPU for
        // very large axes where the padding waste is excessive.
        if dim < x.rank() && x.dims()[dim] > 65536 {
            return None;
        }
        Some(Self::gpu_sort(x, dim, descending))
    }
}
