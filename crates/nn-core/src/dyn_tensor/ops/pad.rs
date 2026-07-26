// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tuple-based padding operations for [`DynTensor`].
//!
//! Provides `pad_with_value`, `pad_zeros_nd`, and `pad_reflect` with
//! per-dimension `(before, after)` tuple padding specification. These
//! complement the existing PyTorch-convention `pad()` (flat array) and
//! candle-convention `pad_with_zeros(dim, left, right)` (single-dim).

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

impl DynTensor {
    /// Pad tensor with a constant value, specifying `(before, after)` per dimension.
    ///
    /// `pads[i]` gives `(before, after)` padding for dimension `i`.
    /// Length of `pads` must equal the tensor rank.
    ///
    /// # Examples
    ///
    /// ```text
    /// // 1D: [1.0, 2.0] with pads=[(1, 2)] → [0.0, 1.0, 2.0, 0.0, 0.0]
    /// // 2D: [[1,2],[3,4]] with pads=[(1,0),(0,1)] → [[0,0,0],[1,2,0],[3,4,0]]
    /// ```
    pub fn pad_with_value(&self, pads: &[(usize, usize)], value: f64) -> Result<Self> {
        if pads.len() != self.rank() {
            return Err(TensorError::InvalidShape(format!(
                "pad_with_value: pads length {} != tensor rank {}",
                pads.len(),
                self.rank()
            )));
        }
        // No-op shortcut
        if pads.iter().all(|&(l, r)| l == 0 && r == 0) {
            return Ok(self.clone());
        }
        // Delegate to the existing pad() method by converting tuple format to
        // PyTorch's flat convention: [left_last, right_last, left_2nd_last, ...].
        let rank = self.rank();
        let mut pytorch_padding = Vec::with_capacity(rank * 2);
        for i in (0..rank).rev() {
            pytorch_padding.push(pads[i].0); // left for dim i
            pytorch_padding.push(pads[i].1); // right for dim i
        }
        self.pad(&pytorch_padding, value)
    }

    /// Pad tensor with zeros, specifying `(before, after)` per dimension.
    ///
    /// Convenience wrapper for `pad_with_value(pads, 0.0)`.
    /// `pads[i]` gives `(before, after)` zero-padding for dimension `i`.
    /// Length of `pads` must equal the tensor rank.
    pub fn pad_zeros_nd(&self, pads: &[(usize, usize)]) -> Result<Self> {
        self.pad_with_value(pads, 0.0)
    }

    /// Reflection-pad the tensor, specifying `(before, after)` per dimension.
    ///
    /// `pads[i]` gives `(before, after)` reflection padding for dimension `i`.
    /// Length of `pads` must equal the tensor rank. Padding for each dimension
    /// must be strictly less than the corresponding dimension size.
    ///
    /// Reflects values at boundaries (excluding the boundary element itself),
    /// matching PyTorch's reflection padding semantics.
    ///
    /// # Examples
    ///
    /// ```text
    /// // [a, b, c, d, e] with pads=[(2, 1)]
    /// // → [c, b, a, b, c, d, e, d]
    /// ```
    pub fn pad_reflect(&self, pads: &[(usize, usize)]) -> Result<Self> {
        if pads.len() != self.rank() {
            return Err(TensorError::InvalidShape(format!(
                "pad_reflect: pads length {} != tensor rank {}",
                pads.len(),
                self.rank()
            )));
        }
        // Validate: padding must be < dim size for each dimension.
        for (d, &(before, after)) in pads.iter().enumerate() {
            let dim_size = self.dims()[d];
            if before >= dim_size {
                return Err(TensorError::InvalidShape(format!(
                    "pad_reflect: before padding {before} >= dim {d} size {dim_size} \
                     (reflection requires padding < dim size)"
                )));
            }
            if after >= dim_size {
                return Err(TensorError::InvalidShape(format!(
                    "pad_reflect: after padding {after} >= dim {d} size {dim_size} \
                     (reflection requires padding < dim size)"
                )));
            }
        }
        // No-op shortcut
        if pads.iter().all(|&(l, r)| l == 0 && r == 0) {
            return Ok(self.clone());
        }
        // Apply reflection padding dimension by dimension using narrow + flip + cat.
        // This works on both CPU and GPU since those primitives are already dispatched.
        let mut result = self.clone();
        for (dim, &(before, after)) in pads.iter().enumerate() {
            if before == 0 && after == 0 {
                continue;
            }
            result = reflect_pad_dim(&result, dim, before, after)?;
        }
        Ok(result)
    }
}

/// Reflect-pad a single dimension: `[before reflected | original | after reflected]`.
fn reflect_pad_dim(
    tensor: &DynTensor,
    dim: usize,
    before: usize,
    after: usize,
) -> Result<DynTensor> {
    let mut parts: Vec<DynTensor> = Vec::with_capacity(3);
    if before > 0 {
        // Elements at indices [1..=before], reversed.
        let left_slice = tensor.narrow(dim, 1, before)?;
        parts.push(left_slice.flip(dim)?);
    }
    parts.push(tensor.clone());
    if after > 0 {
        let dim_size = tensor.dims()[dim];
        // Elements at indices [dim_size-after-1..dim_size-1), reversed.
        let right_slice = tensor.narrow(dim, dim_size - after - 1, after)?;
        parts.push(right_slice.flip(dim)?);
    }
    let refs: Vec<&DynTensor> = parts.iter().collect();
    DynTensor::cat(&refs, dim)
}
