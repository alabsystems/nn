// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU selection and comparison operation sub-trait for [`GpuBackend`](super::GpuBackend)
//! decomposition.
//!
//! Contains 12 selection/comparison methods extracted from the monolithic `GpuBackend`
//! trait: index_select, gather, compare, compare_tensor, where_cond, index_add,
//! scatter_add, cumsum, repeat_interleave, argmax, argmin, topk. All methods are
//! optional (default `None` → CPU fallback).

use super::DynTensor;
use crate::Result;

// CompareOp is defined in the parent module (gpu.rs) via gpu_ops.rs.
// From this child module of gpu, we access it through super.
pub(super) use super::CompareOp;

/// GPU selection and comparison operations.
///
/// All methods return `Option<Result<...>>` — `None` triggers CPU fallback,
/// `Some(Ok(..))` returns the GPU result, `Some(Err(e))` propagates.
pub trait GpuSelectionOps: Send + Sync {
    /// Select elements along `dim` using 1-D U32 index tensor on GPU.
    fn index_select(
        &self,
        _x: &DynTensor,
        _ids: &DynTensor,
        _dim: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Select elements along `dim` without OOB validation (caller guarantees valid indices).
    ///
    /// Skips the GPU→CPU readback that `index_select` uses for bounds checking.
    /// The MSL kernel clamps OOB indices as defense-in-depth, but callers MUST
    /// guarantee indices are in-range — clamping masks bugs silently.
    ///
    /// Supports F32 indices (cast to uint inline in MSL) and U32 indices.
    fn index_select_unchecked(
        &self,
        _x: &DynTensor,
        _ids: &DynTensor,
        _dim: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Gather elements along `dim` using N-D U32 index tensor on GPU.
    fn gather(&self, _x: &DynTensor, _ids: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Element-wise comparison against scalar, returning F32 (0.0/1.0) mask on GPU.
    ///
    /// Returns F32 instead of U8 to avoid GPU→CPU→GPU round-trip (#1323).
    /// `where_cond` accepts both F32 and U8 masks.
    fn compare(&self, _x: &DynTensor, _op: CompareOp, _val: f64) -> Option<Result<DynTensor>> {
        None
    }

    /// Element-wise comparison between two tensors on GPU, returning F32 (0.0/1.0) mask.
    ///
    /// Both tensors must be f32 with the same shape. Broadcasting is not supported
    /// on GPU — callers must broadcast to matching shapes before dispatch.
    fn compare_tensor(
        &self,
        _lhs: &DynTensor,
        _op: CompareOp,
        _rhs: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Conditional select on GPU: `if mask[i] != 0 { on_true[i] } else { on_false[i] }`.
    ///
    /// Accepts both F32 (0.0/1.0, from `compare`) and U8 masks.
    fn where_cond(
        &self,
        _mask: &DynTensor,
        _on_true: &DynTensor,
        _on_false: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Index-add on GPU: accumulate `src` into `self` along `dim` using 1-D `index`.
    ///
    /// `output[..., index[i], ...] += src[..., i, ...]` (where the indexed axis is `dim`).
    /// `index` must be 1-D U32 with length equal to `src.dims()[dim]`.
    fn index_add(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _index: &DynTensor,
        _src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Scatter-add on GPU: accumulate `src` into `self` at positions given by `index`.
    fn scatter_add(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _index: &DynTensor,
        _src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Cumulative sum along a dimension on GPU.
    fn cumsum(&self, _x: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Kahan-compensated cumulative sum along a dimension on GPU (#2909).
    ///
    /// Error bound: O(nε) vs O(n²ε) for naive f32. Intended for small axis
    /// sizes where sequential Kahan compensation is sufficient (e.g. SineGen
    /// T_frames=126). Falls back to `None` on backends without support.
    fn cumsum_kahan(&self, _x: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Repeat-interleave along `dim` using integer repeat counts on GPU.
    fn repeat_interleave(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _counts: &[usize],
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// GPU-native repeat-interleave: keeps counts on GPU, avoids CPU sync.
    ///
    /// Takes a GPU-resident f32 counts tensor and computes the prefix-sum
    /// offsets entirely on GPU. Only one scalar readback (total count) is
    /// needed to allocate the output buffer.
    ///
    /// Returns `None` to fall back to the CPU-counts path.
    fn repeat_interleave_from_gpu(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _counts: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Argmax along `dim` on GPU. Returns U32 index tensor (shape with `dim` removed).
    fn argmax(&self, _x: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Argmin along `dim` on GPU. Returns U32 index tensor (shape with `dim` removed).
    fn argmin(&self, _x: &DynTensor, _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Top-k values and indices along `dim` on GPU. Returns `(values_f32, indices_u32)`.
    /// Both outputs have the same shape as input except `dim` is replaced by `k`.
    /// Results are sorted descending by value within each slice.
    /// Supports k ≤ 64 (register-based insertion sort); returns `None` for larger k.
    fn topk(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _k: usize,
    ) -> Option<Result<(DynTensor, DynTensor)>> {
        None
    }

    /// Scatter (overwrite) on GPU: write `src` into `self` at positions given by `index`.
    ///
    /// For dim=1: `output[i][index[i][j][k]][k] = src[i][j][k]`
    ///
    /// Unlike `scatter_add`, this overwrites rather than accumulates.
    /// Returns `None` to fall back to CPU.
    fn scatter(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _index: &DynTensor,
        _src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Sort values and indices along `dim` on GPU. Returns `(values, indices_u32)`.
    ///
    /// Both outputs have the same shape as input. When `descending` is true,
    /// largest values come first. Uses bitonic sort for GPU-efficient parallel sorting.
    /// Returns `None` for axis sizes exceeding GPU sort limits.
    fn sort(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _descending: bool,
    ) -> Option<Result<(DynTensor, DynTensor)>> {
        None
    }
}
