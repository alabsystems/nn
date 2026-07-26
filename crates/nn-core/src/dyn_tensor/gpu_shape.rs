// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU shape operation sub-trait for [`GpuBackend`](super::GpuBackend) decomposition.
//!
//! Contains 7 shape-manipulation methods extracted from the monolithic `GpuBackend`
//! trait: narrow, transpose, permute, cat, expand, unfold, slice_set. All methods
//! are optional (default `None` → CPU fallback).

use super::DynTensor;
use crate::Result;

/// GPU shape operations: narrow, transpose, permute, cat, expand, unfold, slice_set.
///
/// All methods return `Option<Result<DynTensor>>` — `None` triggers CPU fallback,
/// `Some(Ok(t))` returns the GPU result, `Some(Err(e))` propagates the error.
pub trait GpuShapeOps: Send + Sync {
    /// Narrow (slice) along a dimension on GPU.
    fn narrow(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Transpose two dimensions on GPU.
    fn transpose(&self, _x: &DynTensor, _d1: usize, _d2: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Permute dimensions on GPU.
    fn permute(&self, _x: &DynTensor, _dims: &[usize]) -> Option<Result<DynTensor>> {
        None
    }

    /// Concatenate tensors along a dimension on GPU.
    fn cat(&self, _tensors: &[&DynTensor], _dim: usize) -> Option<Result<DynTensor>> {
        None
    }

    /// Expand tensor to a larger size using broadcast semantics on GPU.
    fn expand(&self, _x: &DynTensor, _new_dims: &[usize]) -> Option<Result<DynTensor>> {
        None
    }

    /// Extract overlapping sliding windows along a dimension on GPU.
    ///
    /// Returns a tensor with an additional trailing dimension of size `size`.
    /// For input shape `[d0, ..., d_dim, ..., dN]`, output shape is
    /// `[d0, ..., n_windows, ..., dN, size]` where `n_windows = (d_dim - size) / step + 1`.
    ///
    /// This is the core primitive for STFT framing — replaces O(n_frames) narrow()
    /// calls with a single GPU dispatch (#1945).
    fn unfold(
        &self,
        _x: &DynTensor,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Write `src` into a slice of `dst` along `dim` starting at `offset` on GPU.
    /// Returns a new tensor with the slice region overwritten.
    fn slice_set(
        &self,
        _dst: &DynTensor,
        _dim: usize,
        _offset: usize,
        _src: &DynTensor,
    ) -> Option<Result<DynTensor>> {
        None
    }

    /// Pad tensor with a constant value on GPU.
    ///
    /// `padding` follows PyTorch's `F.pad()` convention: pairs of
    /// `[left_last, right_last, left_2nd_last, right_2nd_last, ...]`.
    /// Returns `None` to fall back to CPU.
    fn pad(&self, _x: &DynTensor, _padding: &[usize], _value: f64) -> Option<Result<DynTensor>> {
        None
    }
}
