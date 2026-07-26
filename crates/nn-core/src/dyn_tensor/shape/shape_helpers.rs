// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for shape operations on [`DynTensor`].

use super::DynTensor;
use crate::{Result, TensorError};
use ndarray::SliceInfoElem;

/// Validate arguments for `slice_set`. Returns the end index on success.
pub(super) fn validate_slice_set_args(
    dst: &DynTensor,
    dim: usize,
    offset: usize,
    src: &DynTensor,
) -> Result<usize> {
    crate::check_dim(dim, dst.rank())?;
    if dst.rank() != src.rank() {
        return Err(TensorError::RankMismatch {
            expected: dst.rank(),
            actual: src.rank(),
        });
    }
    for (i, (s, d)) in dst.dims().iter().zip(src.dims()).enumerate() {
        if i != dim && s != d {
            return Err(TensorError::shape_mismatch(
                dst.dims().to_vec(),
                src.dims().to_vec(),
            ));
        }
    }
    let src_len = src.dims()[dim];
    let end = offset.checked_add(src_len).ok_or_else(|| {
        TensorError::InvalidShape(format!("slice_set overflow: {offset}+{src_len}"))
    })?;
    if end > dst.dims()[dim] {
        return Err(TensorError::InvalidShape(format!(
            "slice_set [{offset}..{end}) exceeds dim {dim} size {}",
            dst.dims()[dim]
        )));
    }
    Ok(end)
}

/// Build the ndarray `SliceInfoElem` vector for `narrow(dim, start, end)`.
pub(super) fn build_narrow_slice(
    rank: usize,
    dim: usize,
    start: usize,
    end: usize,
) -> Result<Vec<SliceInfoElem>> {
    build_slice_info(rank, dim, start, end)
}

/// Build the ndarray `SliceInfoElem` vector for writing `[offset..end]` along `dim`.
pub(super) fn build_slice_info(
    rank: usize,
    dim: usize,
    offset: usize,
    end: usize,
) -> Result<Vec<SliceInfoElem>> {
    let ioffset = isize::try_from(offset).map_err(|_| {
        TensorError::InvalidShape(format!("slice_set offset {offset} exceeds isize::MAX"))
    })?;
    let iend = isize::try_from(end).map_err(|_| {
        TensorError::InvalidShape(format!("slice_set end {end} exceeds isize::MAX"))
    })?;
    Ok((0..rank)
        .map(|d| {
            if d == dim {
                SliceInfoElem::Slice {
                    start: ioffset,
                    end: Some(iend),
                    step: 1,
                }
            } else {
                SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }
            }
        })
        .collect())
}
