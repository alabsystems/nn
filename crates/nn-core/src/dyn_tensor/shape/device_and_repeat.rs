// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Device transfer, flip, and repeat ops for [`DynTensor`].
//!
//! Extracted from `shape/mod.rs` for 500-line compliance (#1280 Direction 1).

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend, DynTensor, TensorStorage};
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, SliceInfoElem};

impl DynTensor {
    /// Reverse elements along a dimension.
    ///
    /// `flip(0)` on `[seq, batch, features]` reverses the sequence order.
    /// Uses ndarray reverse-stride slice for O(n) single-pass copy.
    pub fn flip(&self, dim: impl crate::dyn_tensor::Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let dim_size = self.dims[dim];
        if dim_size <= 1 {
            return Ok(self.clone());
        }
        let tracing = trace::is_tracing();
        let mut result = if self.device().is_gpu() {
            // GPU path: decompose as index_select with reversed indices.
            // Only the small index vector (CPU U32) is created; data stays on GPU.
            if dim_size > u32::MAX as usize {
                return Err(TensorError::InvalidShape(format!(
                    "flip dim {dim} size {dim_size} exceeds u32::MAX"
                )));
            }
            let compute = || {
                let reversed: Vec<u32> = (0..dim_size as u32).rev().collect();
                let ids = Self::from_vec_u32(reversed, &[dim_size], &Device::Cpu)?;
                self.index_select(&ids, dim)
            };
            // Suppress tracing during GPU decomposition: index_select's internal
            // trace_input_ids fails because the locally-created ids tensor has no
            // trace ID. The composite Flip op is recorded below instead (#2414).
            if tracing {
                trace::with_trace_suppressed(compute)?
            } else {
                compute()?
            }
        } else {
            // Use ndarray slice with step=-1 to reverse along the target axis.
            let slice_info: Vec<SliceInfoElem> = (0..self.rank())
                .map(|d| {
                    if d == dim {
                        SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: -1,
                        }
                    } else {
                        SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        }
                    }
                })
                .collect();
            // Closure wrapper: dispatch_cpu_typed! uses `return` that would
            // bypass trace recording below. Same fix as reshape/transpose.
            (|| {
                dispatch_cpu_typed!(
                    self,
                    |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                        let sliced = arr.slice(slice_info.as_slice());
                        Ok(sliced.as_standard_layout().to_owned())
                    },
                    "flip"
                )
            })()?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Flip { dim },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Transfer tensor to a different device.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device() == *device {
            return Ok(self.clone());
        }
        if device.is_gpu() && self.device().is_cpu() {
            let backend = gpu_backend()?;
            backend.to_gpu(self)
        } else if device.is_cpu() && self.device().is_gpu() {
            let backend = gpu_backend()?;
            backend.to_cpu(self)
        } else {
            Err(TensorError::device_transfer(self.device(), *device))
        }
    }

    /// Tile (repeat) the tensor along each dimension.
    ///
    /// `repeats` specifies how many times to repeat along each dimension.
    /// Length must equal the tensor's rank.
    /// Matches `torch.Tensor.repeat()` / candle `Tensor::repeat()`.
    ///
    /// Example: `[1, 2].repeat(&[3, 2])` → `[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]`
    /// with shape `[3, 4]`.
    pub fn repeat(&self, repeats: impl AsRef<[usize]>) -> Result<Self> {
        let repeats = repeats.as_ref();
        if repeats.len() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                actual: repeats.len(),
            });
        }
        // No-op if all repeats are 1.
        if repeats.iter().all(|&r| r == 1) {
            return Ok(self.clone());
        }
        // Zero repeat in any dim → empty tensor.
        if repeats.contains(&0) {
            let out_dims: Vec<usize> = self
                .dims()
                .iter()
                .zip(repeats.iter())
                .map(|(&d, &r)| d * r)
                .collect();
            return Self::zeros(&out_dims, self.dtype(), &self.device());
        }
        // Strategy: for each dim with repeat > 1, reshape to insert a repeat
        // axis, broadcast, and flatten back. This is efficient for GPU too
        // since expand uses broadcast semantics.
        let mut result = self.clone();
        for (dim, &rep) in repeats.iter().enumerate() {
            if rep == 1 {
                continue;
            }
            let cur_size = result.dims()[dim];
            // [.., cur_size, ..] → [.., 1, cur_size, ..] → [.., rep, cur_size, ..]
            // → [.., rep * cur_size, ..]
            let r = result.unsqueeze(dim)?;
            let mut expand_dims = r.dims().to_vec();
            expand_dims[dim] = rep;
            let expanded = r.expand(&expand_dims)?;
            let mut flat_dims = result.dims().to_vec();
            flat_dims[dim] = cur_size * rep;
            result = expanded.reshape(&flat_dims)?;
        }
        Ok(result)
    }

    /// Circular shift along specified dimensions.
    ///
    /// Each `(shift, dim)` pair shifts the tensor circularly along `dim` by
    /// `shift` positions. Positive shift moves elements toward higher indices
    /// (wrapping around to the beginning). Matches PyTorch's `torch.roll`.
    ///
    /// # Example
    ///
    /// `[1, 2, 3, 4].roll(&[1], &[0])` -> `[4, 1, 2, 3]`
    pub fn roll(&self, shifts: &[i64], dims: &[usize]) -> Result<Self> {
        if shifts.len() != dims.len() {
            return Err(TensorError::InvalidShape(format!(
                "roll: shifts length {} != dims length {}",
                shifts.len(),
                dims.len()
            )));
        }
        if self.rank() == 0 {
            return Ok(self.clone());
        }
        for &d in dims {
            if d >= self.rank() {
                return Err(TensorError::InvalidShape(format!(
                    "roll: dim {d} out of range for rank {}",
                    self.rank()
                )));
            }
        }
        // Apply shifts sequentially.
        let mut result = self.clone();
        for (&shift, &dim) in shifts.iter().zip(dims.iter()) {
            let dim_size = result.dims()[dim];
            if dim_size == 0 {
                continue;
            }
            // Normalize shift to [0, dim_size).
            let shift = ((shift % dim_size as i64) + dim_size as i64) as usize % dim_size;
            if shift == 0 {
                continue;
            }
            // Roll by concatenating [narrow(dim_size - shift ..), narrow(.. dim_size - shift)].
            let split = dim_size - shift;
            let tail = result.narrow(dim, split, shift)?;
            let head = result.narrow(dim, 0, split)?;
            result = Self::cat(&[&tail, &head], dim)?;
        }
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Roll {
                    shifts: shifts.to_vec(),
                    dims: dims.to_vec(),
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

    /// Tile (repeat) the tensor along each dimension.
    ///
    /// Alias for [`repeat`](Self::repeat). Matches PyTorch `torch.tile()`.
    pub fn tile(&self, repeats: impl AsRef<[usize]>) -> Result<Self> {
        self.repeat(repeats)
    }
}
