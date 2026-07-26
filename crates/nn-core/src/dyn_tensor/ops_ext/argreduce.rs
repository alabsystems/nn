// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Argmax and argmin operations for [`DynTensor`].
//!
//! Extracted from `ops_ext/mod.rs` for file-size compliance.

use super::{to_cpu, to_orig};
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, Dim, DynTensor};
use crate::{Result, TensorError};

impl DynTensor {
    /// Index of the maximum value along a dimension.
    ///
    /// Returns a U32 tensor of indices (same shape minus the reduced dim).
    /// Matches candle's `Tensor::argmax(dim)` return type.
    pub fn argmax(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.argreduce_impl(dim, true)
    }

    /// Index of the minimum value along a dimension.
    ///
    /// Returns a U32 tensor of indices (same shape minus the reduced dim).
    pub fn argmin(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.argreduce_impl(dim, false)
    }

    /// Index of the maximum value along a dimension, keeping the reduced dim as size 1.
    ///
    /// Returns a U32 tensor of indices with the reduced dimension preserved as size 1.
    /// Matches candle's `Tensor::argmax_keepdim(dim)`.
    pub fn argmax_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.argreduce_impl(dim, true)?.unsqueeze(dim)
    }

    /// Index of the minimum value along a dimension, keeping the reduced dim as size 1.
    ///
    /// Returns a U32 tensor of indices with the reduced dimension preserved as size 1.
    /// Matches candle's `Tensor::argmin_keepdim(dim)`.
    pub fn argmin_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.argreduce_impl(dim, false)?.unsqueeze(dim)
    }

    fn argreduce_impl(&self, dim: usize, find_max: bool) -> Result<Self> {
        crate::check_dim(dim, self.rank())?;
        let dim_size = self.dim(dim)?;
        if dim_size == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: dim,
                operation: "argmax/argmin",
            });
        }
        if dim_size > u32::MAX as usize {
            return Err(TensorError::InvalidShape(format!(
                "argmax/argmin dim {dim} size {dim_size} exceeds u32::MAX"
            )));
        }
        let mut result = self.argreduce_dispatch(dim, find_max)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            let op = if find_max {
                TraceOp::Argmax { dim }
            } else {
                TraceOp::Argmin { dim }
            };
            if let Some(id) = trace::record_op(op, &input_ids, result.dims(), result.dtype()) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Dispatch argmax/argmin to GPU or CPU. Separated for trace recording.
    fn argreduce_dispatch(&self, dim: usize, find_max: bool) -> Result<Self> {
        if self.device().is_gpu() {
            let dispatch_fn = if find_max {
                |b: &dyn crate::dyn_tensor::gpu::GpuFullBackend, x: &Self, d: usize| {
                    b.argmax(x, d)
                }
            } else {
                |b: &dyn crate::dyn_tensor::gpu::GpuFullBackend, x: &Self, d: usize| {
                    b.argmin(x, d)
                }
            };
            if let Some(result) = gpu_backend_dispatch(|b| dispatch_fn(b, self, dim)) {
                return result;
            }
        }
        let (cpu_self, device) = to_cpu(self)?;
        let arr = cpu_self.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: "argmax/argmin input".into(),
                count: nan_count,
            });
        }
        let axis = ndarray::Axis(dim);
        let indices = arr.map_axis(axis, |lane| {
            let mut best_idx = 0u32;
            let mut best_val = if find_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            for (i, &v) in lane.iter().enumerate() {
                let better = if find_max { v > best_val } else { v < best_val };
                if better {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            best_idx
        });
        to_orig(Self::from_cpu_u32(indices)?, &device)
    }
}
