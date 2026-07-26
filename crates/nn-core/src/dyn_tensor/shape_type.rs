// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Candle-compatible [`Shape`] type and DynTensor shape compatibility methods.
//!
//! A thin wrapper around `Vec<usize>` that provides `.dims()`, `.elem_count()`,
//! `.rank()` matching candle's `Shape` API. DynTensor gains `.shape()`,
//! `.broadcast_as()`, and `.elem_count()` for dvoice candle migration.

use std::ops::Deref;

use super::DynTensor;
use crate::tensor::checked_dim_product;
use crate::Result;

/// Candle-compatible shape type wrapping a dimension vector.
///
/// Returned by [`DynTensor::shape()`]. Provides `.dims()`, `.elem_count()`,
/// `.rank()` matching candle's `Shape` API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a Shape from dimension sizes.
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    /// Dimension sizes as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Clone dimensions to a Vec (candle compat: `Shape::to_vec()`).
    #[must_use]
    pub fn to_vec(&self) -> Vec<usize> {
        self.0.clone()
    }

    /// Total number of elements with checked arithmetic.
    ///
    /// Returns an error if the dimension product overflows `usize`.
    pub fn checked_elem_count(&self) -> Result<usize> {
        checked_dim_product(&self.0)
    }

    /// Total number of elements (product of dimensions).
    ///
    /// Saturates to `usize::MAX` on overflow instead of wrapping.
    /// Callers performing allocation should prefer [`checked_elem_count`](Self::checked_elem_count).
    #[must_use]
    pub fn elem_count(&self) -> usize {
        checked_dim_product(&self.0).unwrap_or(usize::MAX)
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }
}

// ---------------------------------------------------------------------------
// From conversions for Shape (candle compatibility)
// ---------------------------------------------------------------------------

impl From<usize> for Shape {
    fn from(d0: usize) -> Self {
        Self(vec![d0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from((d0, d1): (usize, usize)) -> Self {
        Self(vec![d0, d1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((d0, d1, d2): (usize, usize, usize)) -> Self {
        Self(vec![d0, d1, d2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((d0, d1, d2, d3): (usize, usize, usize, usize)) -> Self {
        Self(vec![d0, d1, d2, d3])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(dims: &Vec<usize>) -> Self {
        Self(dims.clone())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

// Fixed-size array references: `&[N; M]` does NOT auto-coerce to `&[usize]`
// through `Into<Shape>`, so we need explicit impls for common sizes.
impl From<&[usize; 0]> for Shape {
    fn from(dims: &[usize; 0]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 1]> for Shape {
    fn from(dims: &[usize; 1]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 2]> for Shape {
    fn from(dims: &[usize; 2]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 3]> for Shape {
    fn from(dims: &[usize; 3]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 4]> for Shape {
    fn from(dims: &[usize; 4]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 5]> for Shape {
    fn from(dims: &[usize; 5]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize; 6]> for Shape {
    fn from(dims: &[usize; 6]) -> Self {
        Self(dims.to_vec())
    }
}

impl Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl DynTensor {
    /// Total number of elements (candle compatibility alias for [`numel`](Self::numel)).
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.numel()
    }

    /// Return the shape as a [`Shape`] value (candle compatibility).
    #[must_use]
    pub fn shape(&self) -> Shape {
        Shape::from_dims(&self.dims)
    }

    /// Broadcast tensor to match the given shape (candle compatibility).
    ///
    /// Delegates to [`expand`](Self::expand). Dimensions of size 1 can be
    /// broadcast to any size; other dimensions must match exactly.
    /// Accepts `&Shape`, `Shape`, `&[usize]`, or `Vec<usize>`.
    pub fn broadcast_as(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        self.expand(shape.as_ref())
    }

    /// Broadcast tensor by prepending new dimensions on the left.
    ///
    /// Inserts the dimensions from `left_shape` before the tensor's existing
    /// dimensions, then broadcasts. Matching candle's `Tensor::broadcast_left`.
    ///
    /// ```text
    /// [T, C].broadcast_left(B) → broadcast_as([B, T, C])
    /// [C].broadcast_left((B, T)) → broadcast_as([B, T, C])
    /// ```
    pub fn broadcast_left<S: Into<Shape>>(&self, left_shape: S) -> Result<Self> {
        let left = left_shape.into();
        let mut dims = left.to_vec();
        dims.extend_from_slice(self.dims());
        self.unsqueeze_to_rank(dims.len())?.expand(&dims)
    }

    /// Unsqueeze leading dimensions until the tensor has `target_rank` dims.
    ///
    /// Prepends dimensions of size 1 to make the rank match. Used by
    /// `broadcast_left` to align shapes before expand.
    fn unsqueeze_to_rank(&self, target_rank: usize) -> Result<Self> {
        let current_rank = self.rank();
        if target_rank < current_rank {
            return Err(crate::TensorError::InvalidShape(format!(
                "unsqueeze_to_rank: target rank {target_rank} < current rank {current_rank}"
            )));
        }
        if target_rank == current_rank {
            return Ok(self.clone());
        }
        let leading = target_rank - current_rank;
        let mut new_dims = vec![1; leading];
        new_dims.extend_from_slice(self.dims());
        self.reshape(&new_dims)
    }
}
