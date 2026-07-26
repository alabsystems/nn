// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-mutating `index_put` for [`DynTensor`].
//!
//! Writes `src` values into `self` at positions given by a 1-D index tensor
//! along a specified dimension. Returns a new tensor (does not mutate `self`).
//! Duplicate indices: last write wins (non-accumulating).

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, TensorError};

impl DynTensor {
    /// Write `src` values into `self` at index positions along `dim`.
    ///
    /// Returns a new tensor — does **not** mutate `self`.
    ///
    /// - `index`: 1-D tensor of U32 or I64 indices into dimension `dim`.
    /// - `src`: tensor whose shape matches `self` except along `dim`, where
    ///   its size must equal `index.len()`.
    /// - Duplicate indices: last write wins (non-accumulating).
    ///
    /// Matches PyTorch's `tensor.index_copy_(dim, index, source)` semantics
    /// but is non-mutating.
    pub fn index_put(&self, dim: usize, index: &Self, src: &Self) -> Result<Self> {
        // Validate dimension
        if dim >= self.rank() {
            return Err(TensorError::DimensionOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        // Index must be 1-D
        if index.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: index.rank(),
            });
        }
        // Index dtype must be U32 or I64
        if !matches!(index.dtype(), DType::U32 | DType::I64) {
            return Err(TensorError::dtype_mismatch(DType::U32, index.dtype()));
        }
        let n_indices = index.dims()[0];
        // src must have same rank as self
        if src.rank() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: src.rank(),
            });
        }
        // src shape must match self except along dim, where src.dims[dim] == n_indices
        for (d, (&self_d, &src_d)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if d == dim {
                if src_d != n_indices {
                    return Err(TensorError::InvalidShape(format!(
                        "index_put: src dim {dim} size {src_d} != index length {n_indices}"
                    )));
                }
            } else if self_d != src_d {
                return Err(TensorError::InvalidShape(format!(
                    "index_put: src dim {d} size {src_d} != self dim {d} size {self_d}"
                )));
            }
        }

        // Extract index values as usize
        let indices: Vec<usize> = if index.dtype() == DType::U32 {
            let u32_arr = index.to_vec1::<u32>()?;
            u32_arr.iter().map(|&v| v as usize).collect()
        } else {
            let i64_arr = index.to_vec1::<i64>()?;
            i64_arr
                .iter()
                .map(|&v| {
                    if v < 0 {
                        Err(TensorError::InvalidShape(format!(
                            "index_put: negative index {v}"
                        )))
                    } else {
                        Ok(v as usize)
                    }
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Bounds check
        let dim_size = self.dims()[dim];
        for &idx in &indices {
            if idx >= dim_size {
                return Err(TensorError::InvalidShape(format!(
                    "index_put: index {idx} out of bounds for dim {dim} size {dim_size}"
                )));
            }
        }

        // CPU implementation: GPU falls back to CPU round-trip
        let (self_cpu, src_cpu) = if self.device().is_gpu() {
            (self.to_device(&Device::Cpu)?, src.to_device(&Device::Cpu)?)
        } else {
            (self.clone(), src.clone())
        };

        let dst_arr = self_cpu.to_f32_array()?;
        let src_arr = src_cpu.to_f32_array()?;
        let mut out = dst_arr;

        // Write src slices into dst at indexed positions along dim
        let rank = self.rank();
        if rank == 1 {
            // Fast path for 1-D
            for (src_i, &dst_i) in indices.iter().enumerate() {
                out[dst_i] = src_arr[src_i];
            }
        } else {
            // General N-D path: iterate over all positions in src, map dim index
            let src_shape = src.dims();
            let total_src_elems: usize = src_shape.iter().product();
            let mut src_coord = vec![0usize; rank];
            for flat_i in 0..total_src_elems {
                // Compute src coordinate from flat index
                let mut remainder = flat_i;
                for d in (0..rank).rev() {
                    src_coord[d] = remainder % src_shape[d];
                    remainder /= src_shape[d];
                }
                // Map to dst coordinate: replace dim coordinate with index lookup
                let mut dst_coord = src_coord.clone();
                dst_coord[dim] = indices[src_coord[dim]];
                out[dst_coord.as_slice()] = src_arr[src_coord.as_slice()];
            }
        }

        let mut result = Self::from_f32_result(out, self.dtype)?;

        // Transfer back to GPU if needed
        if self.device().is_gpu() {
            result = result.to_device(&self.device())?;
        }

        // Record trace op
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, index, src])?;
            if let Some(id) = trace::record_op(
                TraceOp::IndexPut { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }

        Ok(result)
    }
}
