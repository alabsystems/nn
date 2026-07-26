// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Free-function wrappers for scatter, gather, and index_select operations.
//!
//! These delegate to the corresponding [`DynTensor`] methods in the `selection`
//! module, providing a functional API convenient for attention mechanisms and
//! embedding lookups.

use crate::dyn_tensor::DynTensor;
use crate::Result;

/// Gather elements from `input` along `dim` using an N-D index tensor.
///
/// `output[i][j][k] = input[i][index[i][j][k]][k]` when `dim=1`.
///
/// `index` must have the same rank as `input` and dtype U32.
/// Output shape equals the shape of `index`.
/// Matches PyTorch `torch.gather` / candle `Tensor::gather`.
pub fn gather(input: &DynTensor, dim: usize, index: &DynTensor) -> Result<DynTensor> {
    input.gather(index, dim)
}

/// Scatter values from `src` into a copy of `input` at positions given by
/// `index` along `dim` (overwrite semantics).
///
/// `output[i][index[i][j][k]][k] = src[i][j][k]` when `dim=1`.
///
/// `index` must have the same rank and shape as `src` (U32 dtype).
/// Output has the same shape as `input`.
/// Matches PyTorch `Tensor.scatter_`.
pub fn scatter(
    input: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<DynTensor> {
    input.scatter(dim, index, src)
}

/// Scatter-add: accumulate values from `src` into a copy of `input` at
/// positions given by `index` along `dim`.
///
/// `output[i][index[i][j][k]][k] += src[i][j][k]` when `dim=1`.
///
/// `index` must have the same rank and shape as `src` (U32 dtype).
/// Output has the same shape as `input`.
/// Matches PyTorch `Tensor.scatter_add_`.
pub fn scatter_add(
    input: &DynTensor,
    dim: usize,
    index: &DynTensor,
    src: &DynTensor,
) -> Result<DynTensor> {
    input.scatter_add(dim, index, src)
}

/// Select slices from `input` along `dim` using a 1-D index tensor.
///
/// Output shape: same as `input` with `dims[dim]` replaced by `index.len()`.
/// Matches PyTorch `torch.index_select` / candle `Tensor::index_select`.
///
/// `index` must be rank-1 with dtype U32 or I64.
pub fn index_select(input: &DynTensor, dim: usize, index: &DynTensor) -> Result<DynTensor> {
    input.index_select(index, dim)
}

#[cfg(test)]
#[path = "tests_scatter_gather.rs"]
mod tests_scatter_gather;
