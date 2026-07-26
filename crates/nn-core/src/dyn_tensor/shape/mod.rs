// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape manipulation operations for [`DynTensor`].
//!
//! Reshape, narrow, slice_set, unsqueeze, squeeze, transpose, permute, contiguous, chunk.
//! Cat and stack live in `cat.rs`. Flip, to_device, repeat live in `device_and_repeat.rs`.

use super::{checked_dim_product, gpu_backend_dispatch, trace, Dim, DynTensor, TensorStorage};
use crate::dyn_tensor::trace::TraceOp;
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn};

mod cat;
#[path = "shape_helpers.rs"]
mod helpers;

#[path = "shape_f32_ops.rs"]
mod f32_ops;

// Narrow and slice_set extracted for 500-line compliance.
#[path = "shape_narrow_slice_set.rs"]
mod narrow_slice_set;

// Unfold (sliding window extraction) for STFT framing (#1945).
#[path = "shape_unfold.rs"]
mod unfold;

// Constant padding (F.pad equivalent).
#[path = "shape_pad.rs"]
mod pad;

// Advanced reshape: repeat_interleave_n, tile_numpy.
#[path = "reshape_advanced.rs"]
mod reshape_advanced;

// dispatch_cpu_typed! macro is defined in parent module (dyn_tensor/mod.rs).

impl DynTensor {
    /// Reshape to new dimensions. Total element count must match.
    pub fn reshape(&self, new_dims: impl AsRef<[usize]>) -> Result<Self> {
        let new_dims = new_dims.as_ref();
        let new_numel = checked_dim_product(new_dims)?;
        let self_numel = self.checked_numel()?;
        if new_numel != self_numel {
            return Err(TensorError::DataLengthMismatch {
                expected: self_numel,
                actual: new_numel,
            });
        }
        // Wrap dispatch_cpu_typed! in a closure so its internal `return`
        // statements exit the closure, not reshape(). Without this, the macro's
        // `return` for FloatStorage paths bypasses the trace recording below.
        // Auto-dequantize quantized tensors before reshape.
        if self.is_quantized() {
            return self.dequantize()?.reshape(new_dims);
        }
        let mut result = match &self.storage {
            TensorStorage::Cpu(_) => (|| {
                dispatch_cpu_typed!(
                    self,
                    |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                        match arr.to_shape(IxDyn(new_dims)) {
                            Ok(view) => Ok(view.to_owned()),
                            Err(_) => {
                                let flat: Vec<_> = arr.iter().copied().collect();
                                Ok(ArrayD::from_shape_vec(IxDyn(new_dims), flat)?)
                            }
                        }
                    },
                    "reshape"
                )
            })(),
            TensorStorage::Gpu { .. } => {
                // GPU reshape is just metadata change — same buffer, new dims
                Ok(Self {
                    dims: new_dims.to_vec(),
                    dtype: self.dtype,
                    storage: self.storage.clone(),
                    trace_node_id: None,
                })
            }
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Reshape {
                    target_shape: new_dims.to_vec(),
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, dim: impl Dim) -> Result<Self> {
        // unsqueeze inserts a new dimension, so valid range is 0..=rank
        let dim = dim.to_index(self.rank() + 1)?;
        let mut new_dims = self.dims.clone();
        new_dims.insert(dim, 1);
        let mut result = self.reshape(&new_dims)?;
        // Override reshape's trace op with the more specific Unsqueeze.
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Unsqueeze { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Remove a dimension of size 1 at the given position.
    pub fn squeeze(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        if self.dims[dim] != 1 {
            return Err(TensorError::InvalidShape(format!(
                "squeeze dim {dim} has size {} (expected 1)",
                self.dims[dim]
            )));
        }
        let mut new_dims = self.dims.clone();
        new_dims.remove(dim);
        let mut result = self.reshape(&new_dims)?;
        // Override reshape's trace op with the more specific Squeeze.
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Squeeze { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Transpose the last two dimensions (candle `.t()` compat).
    ///
    /// Shorthand for `self.transpose(rank - 2, rank - 1)`.
    pub fn t(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rank,
            });
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Transpose two dimensions.
    ///
    /// # GPU dispatch
    ///
    /// Float GPU tensors use native Metal kernel dispatch via
    /// [`GpuBackend::transpose`]. Non-float GPU tensors (U32, I64) fall back
    /// to CPU round-trip: GPU→CPU transfer, transpose, CPU→GPU transfer.
    pub fn transpose(&self, d1: impl Dim, d2: impl Dim) -> Result<Self> {
        let rank = self.rank();
        let d1 = d1.to_index(rank)?;
        let d2 = d2.to_index(rank)?;
        if d1 == d2 {
            return Ok(self.clone());
        }
        let mut result = if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.transpose(self, d1, d2)) {
                result?
            } else {
                let cpu = self.to_device(&Device::Cpu)?;
                let transposed = cpu.transpose(d1, d2)?;
                transposed.to_device(&self.device())?
            }
        } else {
            let mut axes: Vec<usize> = (0..rank).collect();
            axes.swap(d1, d2);
            // Closure wrapper: dispatch_cpu_typed! uses `return` that would
            // bypass trace recording below. Same fix as reshape().
            (|| {
                dispatch_cpu_typed!(
                    self,
                    |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                        Ok(arr
                            .clone()
                            .permuted_axes(IxDyn(&axes))
                            .as_standard_layout()
                            .to_owned())
                    },
                    "transpose"
                )
            })()?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Transpose { dim0: d1, dim1: d2 },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Permute dimensions.
    ///
    /// # GPU dispatch
    ///
    /// Float GPU tensors use native Metal kernel dispatch via
    /// [`GpuBackend::permute`]. Non-float GPU tensors (U32, I64) fall back
    /// to CPU round-trip: GPU→CPU transfer, permute, CPU→GPU transfer.
    pub fn permute(&self, dims: impl AsRef<[usize]>) -> Result<Self> {
        let dims = dims.as_ref();
        let rank = self.rank();
        if dims.len() != rank {
            return Err(TensorError::RankMismatch {
                expected: rank,
                actual: dims.len(),
            });
        }
        let mut seen = vec![false; rank];
        for &d in dims {
            if d >= rank {
                return Err(TensorError::DimensionOutOfRange { dim: d, rank });
            }
            if seen[d] {
                return Err(TensorError::InvalidShape(format!(
                    "duplicate axis {d} in permutation"
                )));
            }
            seen[d] = true;
        }
        let mut result = if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.permute(self, dims)) {
                result?
            } else {
                let cpu = self.to_device(&Device::Cpu)?;
                let permuted = cpu.permute(dims)?;
                permuted.to_device(&self.device())?
            }
        } else {
            // Closure wrapper: dispatch_cpu_typed! uses `return` that would
            // bypass trace recording below. Same fix as reshape().
            (|| {
                dispatch_cpu_typed!(
                    self,
                    |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                        Ok(arr
                            .clone()
                            .permuted_axes(IxDyn(dims))
                            .as_standard_layout()
                            .to_owned())
                    },
                    "permute"
                )
            })()?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Permute {
                    axes: dims.to_vec(),
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Return a contiguous tensor (no-op for CPU ndarray, copies if needed).
    ///
    /// Propagates `trace_node_id` since contiguous is a layout-only identity
    /// operation — no new [`TraceOp`] variant is needed (#2357).
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_quantized() {
            return self.dequantize()?.contiguous();
        }
        let mut result = match &self.storage {
            TensorStorage::Cpu(_) => {
                // Closure wrapper: dispatch_cpu_typed! uses `return` that would
                // bypass trace_node_id propagation below. Same fix as transpose().
                (|| {
                    dispatch_cpu_typed!(
                        self,
                        |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                            if arr.is_standard_layout() {
                                Ok(arr.clone())
                            } else {
                                Ok(arr.as_standard_layout().to_owned())
                            }
                        },
                        "contiguous"
                    )
                })()?
            }
            TensorStorage::Gpu { .. } => self.clone(),
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        };
        // Propagate trace ID: contiguous() is identity in the computation graph.
        result.trace_node_id = self.trace_node_id;
        Ok(result)
    }

    /// Split tensor into chunks along a dimension.
    pub fn chunk(&self, chunks: usize, dim: impl Dim) -> Result<Vec<Self>> {
        if chunks == 0 {
            return Err(TensorError::InvalidShape("chunk count must be > 0".into()));
        }
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dims[dim];
        let chunk_size = dim_size.div_ceil(chunks);
        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;
        while start < dim_size {
            let len = chunk_size.min(dim_size - start);
            result.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(result)
    }

    /// Split tensor into pieces of given sizes along a dimension.
    ///
    /// Unlike `chunk()` which divides into N equal-ish pieces, `split()` takes
    /// an explicit list of sizes. The sum of sizes must equal the dimension size.
    ///
    /// Matches PyTorch's `torch.split(tensor, split_size_or_sections, dim)` when
    /// called with a list of sizes.
    pub fn split(&self, sizes: impl AsRef<[usize]>, dim: impl Dim) -> Result<Vec<Self>> {
        let sizes = sizes.as_ref();
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dims[dim];
        let total: usize = sizes.iter().sum();
        if total != dim_size {
            return Err(TensorError::InvalidShape(format!(
                "split: sizes sum {total} != dim {dim} size {dim_size}"
            )));
        }
        let mut result = Vec::with_capacity(sizes.len());
        let mut start = 0;
        for &s in sizes {
            result.push(self.narrow(dim, start, s)?);
            start += s;
        }
        Ok(result)
    }

    /// Split tensor into parts of a uniform size along a dimension.
    ///
    /// The last part may be smaller if the dimension size is not evenly
    /// divisible by `split_size`.
    ///
    /// Matches PyTorch's `torch.split(tensor, split_size, dim)` when
    /// `split_size` is an integer.
    pub fn split_uniform(&self, split_size: usize, dim: impl Dim) -> Result<Vec<Self>> {
        if split_size == 0 {
            return Err(TensorError::InvalidShape(
                "split_uniform: split_size must be > 0".into(),
            ));
        }
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dims[dim];
        let num_full = dim_size / split_size;
        let remainder = dim_size % split_size;
        let num_parts = num_full + if remainder > 0 { 1 } else { 0 };
        let mut result = Vec::with_capacity(num_parts);
        let mut start = 0;
        while start < dim_size {
            let len = split_size.min(dim_size - start);
            result.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(result)
    }
}

// Submodule uses dispatch_cpu_typed! from parent module (dyn_tensor/mod.rs).
#[path = "device_and_repeat.rs"]
mod device_and_repeat;

#[cfg(test)]
#[path = "shape_dtype_tests.rs"]
mod dtype_tests;

#[cfg(test)]
#[path = "shape_dtype_tests_cat.rs"]
mod dtype_cat_tests;

#[cfg(kani)]
#[path = "kani_shape_mod_proofs.rs"]
mod kani_shape_mod_proofs;

#[cfg(kani)]
#[path = "kani_shape_unfold_proofs.rs"]
mod kani_shape_unfold_proofs;

#[cfg(kani)]
#[path = "kani_cat_pad_proofs.rs"]
mod kani_cat_pad_proofs;

#[cfg(test)]
#[path = "tests_tile_pad.rs"]
mod tests_tile_pad;

#[cfg(test)]
#[path = "tests_pad.rs"]
mod tests_pad;

#[cfg(test)]
#[path = "tests_chunk_split_ext.rs"]
mod tests_chunk_split_ext;

#[cfg(test)]
#[path = "tests_stack_chunk.rs"]
mod tests_stack_chunk;
