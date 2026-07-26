// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reduction operations for [`DynTensor`] — keepdim, all-reduce, variance.
//!
//! Extracted from `ops/mod.rs` for file-size compliance.

use super::super::dim::Dim;
use super::super::{gpu_backend, trace, DynTensor, ReduceOp, TensorStorage};
use crate::dyn_tensor::trace::TraceOp;
use crate::{Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn};

/// Convert a ReduceOp to its corresponding TraceOp with dim/keepdim metadata.
pub(super) fn reduce_op_to_trace_op(op: ReduceOp, dim: usize, keepdim: bool) -> TraceOp {
    match op {
        ReduceOp::Sum => TraceOp::ReduceSum { dim, keepdim },
        ReduceOp::Mean => TraceOp::ReduceMean { dim, keepdim },
        ReduceOp::Max => TraceOp::ReduceMax { dim, keepdim },
        ReduceOp::Min => TraceOp::ReduceMin { dim, keepdim },
    }
}

impl DynTensor {
    /// Mean along a dimension, keeping the reduced dim as size 1.
    pub fn mean_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_impl(ReduceOp::Mean, dim, true)
    }

    /// Sum along a dimension, keeping the reduced dim as size 1.
    pub fn sum_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_impl(ReduceOp::Sum, dim, true)
    }

    /// Mean over all elements → scalar tensor.
    ///
    /// Returns an error if the tensor has zero elements (would produce NaN via 0/0).
    /// The result preserves the device of the input tensor.
    pub fn mean_all(&self) -> Result<Self> {
        let n = self.elem_count();
        if n == 0 {
            return Err(TensorError::InvalidShape(
                "mean_all requires non-empty tensor".into(),
            ));
        }
        // GPU-native: sum_all uses GPU reduce, div_scalar uses broadcast_div on GPU.
        // No GPU→CPU transfer needed.
        self.sum_all()?.div_scalar(n as f64)
    }

    /// Sum over all elements → scalar tensor.
    ///
    /// The result preserves the device of the input tensor.
    pub fn sum_all(&self) -> Result<Self> {
        self.reduce_all_impl(ReduceOp::Sum, 0.0, |a, b| a + b)
    }

    /// Max over all elements → scalar tensor.
    ///
    /// Returns an error if the tensor has zero elements or contains NaN.
    /// The result preserves the device of the input tensor.
    pub fn max_all(&self) -> Result<Self> {
        if self.elem_count() == 0 {
            return Err(TensorError::InvalidShape(
                "max_all requires non-empty tensor".into(),
            ));
        }
        self.reject_nan("max_all")?;
        self.reduce_all_impl(ReduceOp::Max, f32::NEG_INFINITY, f32::max)
    }

    /// Min over all elements → scalar tensor.
    ///
    /// Returns an error if the tensor has zero elements or contains NaN.
    /// The result preserves the device of the input tensor.
    pub fn min_all(&self) -> Result<Self> {
        if self.elem_count() == 0 {
            return Err(TensorError::InvalidShape(
                "min_all requires non-empty tensor".into(),
            ));
        }
        self.reject_nan("min_all")?;
        self.reduce_all_impl(ReduceOp::Min, f32::INFINITY, f32::min)
    }

    /// Reject tensors containing NaN values.
    ///
    /// Matches the `topk()` NaN rejection pattern.
    ///
    /// # GPU dispatch
    ///
    /// No native GPU NaN-scan kernel exists. GPU tensors are transferred
    /// to CPU via [`to_device`](DynTensor::to_device) for the NaN check.
    /// This is a full readback of the tensor data. Called only on
    /// error-path guards (`max_all`, `min_all`), so the cost is
    /// acceptable for correctness.
    fn reject_nan(&self, op_name: &str) -> Result<()> {
        let cpu_tensor;
        let deq_tensor;
        let t = match &self.storage {
            TensorStorage::Cpu(_) => self,
            TensorStorage::Gpu { .. } => {
                cpu_tensor = self.to_device(&Device::Cpu)?;
                &cpu_tensor
            }
            TensorStorage::Quantized(_) => {
                deq_tensor = self.dequantize()?;
                &deq_tensor
            }
        };
        // Promote to f32 for NaN check (bf16/f16 NaN maps to f32 NaN).
        let arr = t.to_f32_array()?;
        let nan_count = arr.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            return Err(TensorError::NonFiniteData {
                name: format!("{op_name} input"),
                count: nan_count,
            });
        }
        Ok(())
    }

    /// Shared implementation for reduce-all (max_all, min_all).
    ///
    /// GPU path: iteratively reduce dims on GPU until rank 1, then transfer
    /// the small [N] result to CPU for the final scalar fold. This avoids
    /// transferring the full tensor to CPU.
    fn reduce_all_impl(&self, op: ReduceOp, init: f32, fold: fn(f32, f32) -> f32) -> Result<Self> {
        let device = self.device();
        let input_dtype = self.dtype;
        if device.is_gpu() {
            // O(1) via Arc::clone on TensorStorage — no data copy.
            let mut t = self.clone();
            while t.rank() > 1 {
                let last = t.rank() - 1;
                t = t.reduce_impl(op, last, false)?;
            }
            let cpu_t = t.to_device(&Device::Cpu)?;
            let vals = cpu_t.to_f32_array()?;
            let result = vals.iter().copied().fold(init, fold);
            let cpu_scalar =
                Self::from_f32_result(ArrayD::from_elem(IxDyn(&[]), result), input_dtype)?;
            return cpu_scalar.to_device(&device);
        }
        // Reductions always accumulate in f32 for numerical precision (#1646 D3).
        // Result dtype matches input dtype (bf16 in → bf16 out).
        let arr = self.to_f32_array()?;
        let result = arr.iter().copied().fold(init, fold);
        Self::from_f32_result(ArrayD::from_elem(IxDyn(&[]), result), input_dtype)
    }

    /// Max along a dimension, keeping the reduced dim as size 1.
    pub fn max_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_impl(ReduceOp::Max, dim, true)
    }

    /// Min along a dimension, keeping the reduced dim as size 1.
    pub fn min_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_impl(ReduceOp::Min, dim, true)
    }

    /// Variance along a dimension, keeping the reduced dim as size 1.
    ///
    /// Computes population variance: `mean((x - mean(x))^2)`.
    /// Matches candle's `Tensor::var_keepdim(dim)`.
    pub fn var_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let mean = self.mean_keepdim(dim)?;
        let centered = self.broadcast_sub(&mean)?;
        centered.sqr()?.mean_keepdim(dim)
    }

    /// Compensated (Kahan) sum along a dimension for near-f64 precision (#1814).
    ///
    /// On GPU: uses `PrecisionTier::Strict` Kahan-compensated MSL kernel.
    /// On CPU: uses Kahan compensated summation instead of naive sum.
    /// For Max/Min: delegates to the standard path (Kahan does not apply).
    pub fn sum_compensated(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_compensated_impl(ReduceOp::Sum, dim, false)
    }

    /// Compensated (Kahan) sum keepdim for near-f64 precision (#1814).
    pub fn sum_compensated_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_compensated_impl(ReduceOp::Sum, dim, true)
    }

    /// Compensated (Kahan) mean along a dimension for near-f64 precision (#1814).
    pub fn mean_compensated(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_compensated_impl(ReduceOp::Mean, dim, false)
    }

    /// Compensated (Kahan) mean keepdim for near-f64 precision (#1814).
    pub fn mean_compensated_keepdim(&self, dim: impl Dim) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        self.reduce_compensated_impl(ReduceOp::Mean, dim, true)
    }

    /// Internal: reduce along dim with keepdim control. pub(crate) for ops_ext.
    pub(crate) fn reduce_impl(&self, op: ReduceOp, dim: usize, keepdim: bool) -> Result<Self> {
        if self.is_quantized() {
            return self.dequantize()?.reduce_impl(op, dim, keepdim);
        }
        crate::check_dim(dim, self.rank())?;
        let mut result = match &self.storage {
            TensorStorage::Cpu(_) => self.cpu_reduce(op, dim, keepdim),
            TensorStorage::Gpu { .. } => {
                let backend = gpu_backend()?;
                backend.reduce_op(op, self, dim, keepdim)
            }
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            let trace_op = reduce_op_to_trace_op(op, dim, keepdim);
            if let Some(id) = trace::record_op(trace_op, &input_ids, result.dims(), result.dtype())
            {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Internal: compensated reduce along dim with keepdim control.
    fn reduce_compensated_impl(&self, op: ReduceOp, dim: usize, keepdim: bool) -> Result<Self> {
        if self.is_quantized() {
            return self.dequantize()?.reduce_compensated_impl(op, dim, keepdim);
        }
        crate::check_dim(dim, self.rank())?;
        let mut result = match &self.storage {
            TensorStorage::Cpu(_) => self.cpu_reduce_compensated(op, dim, keepdim),
            TensorStorage::Gpu { .. } => {
                let backend = gpu_backend()?;
                backend.reduce_op_compensated(op, self, dim, keepdim)
            }
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        }?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            let trace_op = reduce_op_to_trace_op(op, dim, keepdim);
            if let Some(id) = trace::record_op(trace_op, &input_ids, result.dims(), result.dtype())
            {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    fn cpu_reduce(&self, op: ReduceOp, dim: usize, keepdim: bool) -> Result<Self> {
        // Reductions always accumulate in f32 for numerical precision (#1646 D3).
        // Result dtype matches input dtype (bf16 in → bf16 out).
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let axis = ndarray::Axis(dim);
        let reduced = match op {
            ReduceOp::Sum => arr.sum_axis(axis),
            ReduceOp::Mean => arr
                .mean_axis(axis)
                .ok_or(TensorError::ZeroLengthDimension {
                    axis: dim,
                    operation: "mean",
                })?,
            ReduceOp::Max => arr.map_axis(axis, |lane| {
                lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            }),
            ReduceOp::Min => arr.map_axis(axis, |lane| {
                lane.iter().copied().fold(f32::INFINITY, f32::min)
            }),
        };
        if keepdim {
            let mut new_dims: Vec<usize> = reduced.shape().to_vec();
            new_dims.insert(dim, 1);
            let reshaped = reduced.into_shape_with_order(IxDyn(&new_dims))?;
            Self::from_f32_result(reshaped, input_dtype)
        } else {
            Self::from_f32_result(reduced, input_dtype)
        }
    }

    /// CPU Kahan compensated reduction (#1814).
    ///
    /// Uses Kahan summation for Sum/Mean (provides ~2x mantissa bits of
    /// precision from f32 arithmetic). Falls back to standard reduce for
    /// Max/Min where Kahan does not apply.
    fn cpu_reduce_compensated(&self, op: ReduceOp, dim: usize, keepdim: bool) -> Result<Self> {
        // Max/Min: Kahan does not apply.
        if matches!(op, ReduceOp::Max | ReduceOp::Min) {
            return self.cpu_reduce(op, dim, keepdim);
        }
        let input_dtype = self.dtype;
        let arr = self.to_f32_array()?;
        let axis = ndarray::Axis(dim);
        let dim_size = arr.shape()[dim];
        if dim_size == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: dim,
                operation: if matches!(op, ReduceOp::Mean) {
                    "mean_compensated"
                } else {
                    "sum_compensated"
                },
            });
        }
        let reduced = arr.map_axis(axis, |lane| {
            // Kahan compensated summation: tracks running error in `comp`.
            let mut sum = 0.0_f32;
            let mut comp = 0.0_f32;
            for &val in lane.iter() {
                let y = val - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
            if matches!(op, ReduceOp::Mean) {
                sum / dim_size as f32
            } else {
                sum
            }
        });
        if keepdim {
            let mut new_dims: Vec<usize> = reduced.shape().to_vec();
            new_dims.insert(dim, 1);
            let reshaped = reduced.into_shape_with_order(IxDyn(&new_dims))?;
            Self::from_f32_result(reshaped, input_dtype)
        } else {
            Self::from_f32_result(reduced, input_dtype)
        }
    }
}
