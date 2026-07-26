// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced slicing and multi-dim flip operations for [`DynTensor`].
//!
//! - `slice_assign`: non-consuming slice set (returns new tensor with src inserted).
//! - `flip_dims`: reverse along multiple dimensions at once.

use crate::dyn_tensor::ops::binary::check_same_device;
use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

impl DynTensor {
    /// Return a new tensor with `src` values inserted at `[start..start+src_size]`
    /// along dimension `dim`.
    ///
    /// Like PyTorch `tensor[..., start:start+n, ...] = src` but non-mutating:
    /// returns a new tensor. The original tensor is not modified.
    ///
    /// All dimensions of `self` and `src` must match except dimension `dim`,
    /// where `src.dims()[dim]` elements starting at `start` are overwritten.
    ///
    /// # Errors
    ///
    /// - `dim` out of range
    /// - `src` rank mismatch
    /// - Non-dim dimensions mismatch
    /// - `start + src.dims()[dim]` exceeds `self.dims()[dim]`
    /// - Device mismatch
    pub fn slice_assign(&self, dim: usize, start: usize, src: &Self) -> Result<Self> {
        if dim >= self.rank() {
            return Err(TensorError::DimensionOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        if src.rank() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: src.rank(),
            });
        }
        check_same_device(self, src)?;
        if self.dtype() != src.dtype() {
            return Err(TensorError::dtype_mismatch(self.dtype(), src.dtype()));
        }
        let src_dim_size = src.dims()[dim];
        let end = start.checked_add(src_dim_size).ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "slice_assign: start {start} + src dim size {src_dim_size} overflows usize"
            ))
        })?;
        if end > self.dims()[dim] {
            return Err(TensorError::InvalidShape(format!(
                "slice_assign: start({start}) + src.dim({dim})={src_dim_size} = {end} \
                 exceeds self.dim({dim})={}",
                self.dims()[dim]
            )));
        }
        // Check non-dim dimensions match.
        for (d, (s, t)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if d != dim && s != t {
                return Err(TensorError::shape_mismatch(
                    self.dims().to_vec(),
                    src.dims().to_vec(),
                ));
            }
        }
        // Delegate to clone + slice_set_into (consuming).
        self.clone().slice_set_into(dim, start, src)
    }

    /// Reverse elements along multiple dimensions simultaneously.
    ///
    /// Like PyTorch `torch.flip(tensor, dims)`. Applies flip sequentially
    /// along each specified dimension.
    ///
    /// # Errors
    ///
    /// - Any dim out of range for tensor rank.
    pub fn flip_dims(&self, dims: &[usize]) -> Result<Self> {
        for &d in dims {
            if d >= self.rank() {
                return Err(TensorError::DimensionOutOfRange {
                    dim: d,
                    rank: self.rank(),
                });
            }
        }
        if dims.is_empty() {
            return Ok(self.clone());
        }
        let mut result = self.clone();
        for &d in dims {
            result = result.flip(d)?;
        }
        Ok(result)
    }
}
