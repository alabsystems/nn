// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-dimensional indexing for [`DynTensor`] matching candle's `.i(...)` API.
//!
//! ```ignore
//! // NOTE: ignore — requires constructing a tensor and uses top-level nn crate imports
//! use nn::{DynTensor, IndexOp};
//! // Select row 0, columns 2..5:
//! let sub = tensor.i((0, 2..5))?;
//! // Select batch 0, all positions, last token:
//! let sub = tensor.i((0, .., tensor.dim(2)? - 1))?;
//! ```

use super::DynTensor;
use crate::{Result, TensorError};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Specifies how to index a single dimension.
#[non_exhaustive]
pub enum TensorIndexer {
    /// Select a single index, removing the dimension.
    Select(usize),
    /// Narrow to a contiguous range, preserving the dimension.
    /// Fields: `(start, len)`. A `len` of `usize::MAX` means "to end of dim".
    Narrow(usize, usize),
    /// Keep the entire dimension (equivalent to `..`).
    Full,
}

/// Trait for `.i(...)` indexing. Implemented for single values, ranges, tuples.
///
/// This matches candle's `IndexOp` API pattern. Import this trait to use `.i()`
/// on `DynTensor`.
pub trait IndexOp<T> {
    fn i(&self, index: T) -> Result<DynTensor>;
}

// -- Conversions from Rust range types to TensorIndexer -----------------------

impl From<usize> for TensorIndexer {
    fn from(i: usize) -> Self {
        Self::Select(i)
    }
}

impl From<Range<usize>> for TensorIndexer {
    fn from(r: Range<usize>) -> Self {
        Self::Narrow(r.start, r.end.saturating_sub(r.start))
    }
}

impl From<RangeFull> for TensorIndexer {
    fn from(_: RangeFull) -> Self {
        Self::Full
    }
}

impl From<RangeFrom<usize>> for TensorIndexer {
    fn from(r: RangeFrom<usize>) -> Self {
        // Resolved to Narrow at apply time using dim size
        Self::Narrow(r.start, usize::MAX)
    }
}

impl From<RangeTo<usize>> for TensorIndexer {
    fn from(r: RangeTo<usize>) -> Self {
        Self::Narrow(0, r.end)
    }
}

impl From<RangeInclusive<usize>> for TensorIndexer {
    fn from(r: RangeInclusive<usize>) -> Self {
        let start = *r.start();
        let end = *r.end();
        if end < start {
            // Empty range (e.g. 5..=2): len = 0, will fail bounds check at apply time.
            return Self::Narrow(start, 0);
        }
        // end >= start, so end - start + 1 >= 1 and cannot overflow
        // because end <= usize::MAX implies end - start <= usize::MAX - 1
        // so end - start + 1 <= usize::MAX.
        Self::Narrow(start, end - start + 1)
    }
}

impl From<RangeToInclusive<usize>> for TensorIndexer {
    fn from(r: RangeToInclusive<usize>) -> Self {
        // Saturating add: ..=usize::MAX would overflow without this.
        Self::Narrow(0, r.end.saturating_add(1))
    }
}

impl DynTensor {
    /// Apply a sequence of indexers to this tensor.
    ///
    /// Each `Select` removes a dimension; `Narrow`/`Full` preserve it.
    fn apply_indexers(&self, indexers: &[TensorIndexer]) -> Result<Self> {
        let mut current = self.clone();
        let mut current_dim = 0usize;

        for indexer in indexers {
            let rank = current.rank();
            if current_dim >= rank {
                return Err(TensorError::DimensionOutOfRange {
                    dim: current_dim,
                    rank,
                });
            }
            match indexer {
                TensorIndexer::Select(idx) => {
                    let dim_size = current.dims()[current_dim];
                    if *idx >= dim_size {
                        return Err(TensorError::InvalidShape(format!(
                            "index {idx} out of bounds for dim {current_dim} of size {dim_size}"
                        )));
                    }
                    current = current.narrow(current_dim, *idx, 1)?.squeeze(current_dim)?;
                    // Don't increment current_dim — the squeezed dim is gone.
                }
                TensorIndexer::Narrow(start, len) => {
                    let dim_size = current.dims()[current_dim];
                    let actual_len = if *len == usize::MAX {
                        dim_size.saturating_sub(*start)
                    } else {
                        *len
                    };
                    if *start + actual_len > dim_size {
                        return Err(TensorError::InvalidShape(format!(
                            "narrow range {}..{} out of bounds for dim {current_dim} of size {dim_size}",
                            start,
                            start + actual_len
                        )));
                    }
                    current = current.narrow(current_dim, *start, actual_len)?;
                    current_dim += 1;
                }
                TensorIndexer::Full => {
                    current_dim += 1;
                }
            }
        }
        Ok(current)
    }
}

// -- IndexOp for single value (1-dim indexing) --------------------------------

impl<I: Into<TensorIndexer>> IndexOp<I> for DynTensor {
    fn i(&self, index: I) -> Result<DynTensor> {
        self.apply_indexers(&[index.into()])
    }
}

// -- IndexOp for 2-tuples -----------------------------------------------------

impl<I0, I1> IndexOp<(I0, I1)> for DynTensor
where
    I0: Into<TensorIndexer>,
    I1: Into<TensorIndexer>,
{
    fn i(&self, index: (I0, I1)) -> Result<DynTensor> {
        self.apply_indexers(&[index.0.into(), index.1.into()])
    }
}

// -- IndexOp for 3-tuples -----------------------------------------------------

impl<I0, I1, I2> IndexOp<(I0, I1, I2)> for DynTensor
where
    I0: Into<TensorIndexer>,
    I1: Into<TensorIndexer>,
    I2: Into<TensorIndexer>,
{
    fn i(&self, index: (I0, I1, I2)) -> Result<DynTensor> {
        self.apply_indexers(&[index.0.into(), index.1.into(), index.2.into()])
    }
}

// -- IndexOp for 4-tuples -----------------------------------------------------

impl<I0, I1, I2, I3> IndexOp<(I0, I1, I2, I3)> for DynTensor
where
    I0: Into<TensorIndexer>,
    I1: Into<TensorIndexer>,
    I2: Into<TensorIndexer>,
    I3: Into<TensorIndexer>,
{
    fn i(&self, index: (I0, I1, I2, I3)) -> Result<DynTensor> {
        self.apply_indexers(&[
            index.0.into(),
            index.1.into(),
            index.2.into(),
            index.3.into(),
        ])
    }
}

// -- IndexOp for 5-tuples -----------------------------------------------------

impl<I0, I1, I2, I3, I4> IndexOp<(I0, I1, I2, I3, I4)> for DynTensor
where
    I0: Into<TensorIndexer>,
    I1: Into<TensorIndexer>,
    I2: Into<TensorIndexer>,
    I3: Into<TensorIndexer>,
    I4: Into<TensorIndexer>,
{
    fn i(&self, index: (I0, I1, I2, I3, I4)) -> Result<DynTensor> {
        self.apply_indexers(&[
            index.0.into(),
            index.1.into(),
            index.2.into(),
            index.3.into(),
            index.4.into(),
        ])
    }
}
