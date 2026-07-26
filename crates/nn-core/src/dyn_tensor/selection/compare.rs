// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comparison operations for [`DynTensor`].
//!
//! Extracted from `selection/mod.rs` for file-size compliance.
//! Dtype conversion (`to_dtype`) lives in `dtype_convert.rs`.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, CompareOp, DynTensor};
use crate::{DType, Device, Result, TensorError};

impl DynTensor {
    // -- Comparison Ops ----------------------------------------------------------
    //
    // CPU: returns U8 mask (1/0).
    // GPU: returns F32 mask (1.0/0.0) to avoid round-trip for where_cond (#1323).
    // `where_cond` accepts both U8 and F32 masks.

    /// Element-wise equality comparison.
    ///
    /// CPU: returns U8 tensor (1/0). GPU: returns F32 tensor (1.0/0.0) (#1323).
    pub fn eq(&self, val: f64) -> Result<Self> {
        self.compare_scalar(CompareOp::Eq, val)
    }

    /// Element-wise not-equal comparison.
    ///
    /// CPU: returns U8 tensor (1/0). GPU: returns F32 tensor (1.0/0.0) (#1323).
    pub fn ne(&self, val: f64) -> Result<Self> {
        self.compare_scalar(CompareOp::Ne, val)
    }

    /// Element-wise greater-than-or-equal comparison.
    ///
    /// CPU: returns U8 tensor (1/0). GPU: returns F32 tensor (1.0/0.0) (#1323).
    pub fn ge(&self, val: f64) -> Result<Self> {
        self.compare_scalar(CompareOp::Ge, val)
    }

    /// Element-wise greater-than comparison.
    ///
    /// CPU: returns U8 tensor (1/0). GPU: returns F32 tensor (1.0/0.0) (#1323).
    pub fn gt(&self, val: f64) -> Result<Self> {
        self.compare_scalar(CompareOp::Gt, val)
    }

    /// Element-wise less-than comparison.
    ///
    /// CPU: returns U8 tensor (1/0). GPU: returns F32 tensor (1.0/0.0) (#1323).
    pub fn lt(&self, val: f64) -> Result<Self> {
        self.compare_scalar(CompareOp::Lt, val)
    }

    /// Element-wise less-than-or-equal comparison.
    ///
    /// CPU: returns U8 tensor (1/0). GPU: returns F32 tensor (1.0/0.0) (#1323).
    pub fn le(&self, val: f64) -> Result<Self> {
        self.compare_scalar(CompareOp::Le, val)
    }

    /// Shared implementation for element-wise comparison ops.
    ///
    /// CPU: dispatches by dtype (F32 as f32, I64 as i64, U32 as u32), returns U8 mask.
    /// GPU: dispatches via `GpuBackend::compare()`, returns F32 mask (1.0/0.0) to
    /// avoid GPU→CPU→GPU round-trip for the common `compare → where_cond` pattern
    /// (#1323). `where_cond` accepts both U8 and F32 masks.
    fn compare_scalar(&self, op: CompareOp, val: f64) -> Result<Self> {
        // Suppress internal tracing (GPU→CPU fallback calls recursively).
        let mut result = if trace::is_tracing() {
            trace::with_trace_suppressed(|| self.compare_scalar_compute(op, val))?
        } else {
            self.compare_scalar_compute(op, val)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Compare { op, value: val },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    fn compare_scalar_compute(&self, op: CompareOp, val: f64) -> Result<Self> {
        if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.compare(self, op, val)) {
                return result;
            }
            let cpu = self.to_device(&Device::Cpu)?;
            return cpu
                .compare_scalar_compute(op, val)?
                .to_device(&self.device());
        }
        match self.dtype {
            DType::I64 => {
                let arr = self.as_cpu_i64()?;
                if !val.is_finite() {
                    return Err(TensorError::ValueOutOfRange {
                        description: "compare_scalar: comparison value is not finite",
                    });
                }
                let mask = arr.mapv(|x| Self::apply_cmp_f64(op, x as f64, val));
                Self::from_cpu_u8(mask)
            }
            DType::U32 => {
                let arr = self.as_cpu_u32()?;
                if !val.is_finite() {
                    return Err(TensorError::ValueOutOfRange {
                        description: "compare_scalar: comparison value is not finite",
                    });
                }
                let mask = arr.mapv(|x| Self::apply_cmp_f64(op, f64::from(x), val));
                Self::from_cpu_u8(mask)
            }
            DType::U8 => {
                let arr = self.as_cpu_u8()?;
                if !val.is_finite() {
                    return Err(TensorError::ValueOutOfRange {
                        description: "compare_scalar: comparison value is not finite",
                    });
                }
                let mask = arr.mapv(|x| Self::apply_cmp_f64(op, f64::from(x), val));
                Self::from_cpu_u8(mask)
            }
            DType::F32 | DType::F16 | DType::BF16 | DType::F64 => {
                let arr = self.to_f32_array()?;
                let v = crate::dyn_tensor::checked_f64_to_f32(val, "compare_scalar()")?;
                let mask = arr.mapv(|x| match op {
                    CompareOp::Eq => u8::from(x == v),
                    CompareOp::Ne => u8::from(x != v),
                    CompareOp::Ge => u8::from(x >= v),
                    CompareOp::Gt => u8::from(x > v),
                    CompareOp::Lt => u8::from(x < v),
                    CompareOp::Le => u8::from(x <= v),
                });
                Self::from_cpu_u8(mask)
            }
            DType::I32 | DType::Bool => Err(TensorError::Unsupported(format!(
                "compare_scalar: dtype {} not supported",
                self.dtype
            ))),
        }
    }

    /// Apply a comparison op on two f64 values (used for integer-vs-fractional comparisons).
    fn apply_cmp_f64(op: CompareOp, a: f64, b: f64) -> u8 {
        match op {
            CompareOp::Eq => u8::from(a == b),
            CompareOp::Ne => u8::from(a != b),
            CompareOp::Ge => u8::from(a >= b),
            CompareOp::Gt => u8::from(a > b),
            CompareOp::Lt => u8::from(a < b),
            CompareOp::Le => u8::from(a <= b),
        }
    }

    // -- Tensor-vs-tensor comparison ops ----------------------------------------
    //
    // CPU: returns U8 mask (1/0).
    // GPU: returns F32 mask (1.0/0.0) to avoid round-trip for where_cond (#1323).
    // Matches candle's `tensor.eq(&other)`, `tensor.gt(&other)`, etc.

    /// Broadcast element-wise equality comparison with another tensor.
    ///
    /// CPU: returns U8 (1/0). GPU: returns F32 (1.0/0.0) (#1323).
    pub fn broadcast_eq(&self, other: &Self) -> Result<Self> {
        self.compare_tensor(CompareOp::Eq, other)
    }

    /// Broadcast element-wise not-equal comparison with another tensor.
    ///
    /// CPU: returns U8 (1/0). GPU: returns F32 (1.0/0.0) (#1323).
    pub fn broadcast_ne(&self, other: &Self) -> Result<Self> {
        self.compare_tensor(CompareOp::Ne, other)
    }

    /// Broadcast element-wise greater-than-or-equal with another tensor.
    ///
    /// CPU: returns U8 (1/0). GPU: returns F32 (1.0/0.0) (#1323).
    pub fn broadcast_ge(&self, other: &Self) -> Result<Self> {
        self.compare_tensor(CompareOp::Ge, other)
    }

    /// Broadcast element-wise greater-than with another tensor.
    ///
    /// CPU: returns U8 (1/0). GPU: returns F32 (1.0/0.0) (#1323).
    pub fn broadcast_gt(&self, other: &Self) -> Result<Self> {
        self.compare_tensor(CompareOp::Gt, other)
    }

    /// Broadcast element-wise less-than with another tensor.
    ///
    /// CPU: returns U8 (1/0). GPU: returns F32 (1.0/0.0) (#1323).
    pub fn broadcast_lt(&self, other: &Self) -> Result<Self> {
        self.compare_tensor(CompareOp::Lt, other)
    }

    /// Broadcast element-wise less-than-or-equal with another tensor.
    ///
    /// CPU: returns U8 (1/0). GPU: returns F32 (1.0/0.0) (#1323).
    pub fn broadcast_le(&self, other: &Self) -> Result<Self> {
        self.compare_tensor(CompareOp::Le, other)
    }

    /// Shared implementation for tensor-vs-tensor comparison.
    ///
    /// Both tensors must be f32 (the common case in ML). Broadcasting follows
    /// NumPy rules via ndarray. GPU: returns F32 mask via native dispatch (#1357,
    /// #1323). CPU fallback when GPU dispatch returns None.
    fn compare_tensor(&self, op: CompareOp, other: &Self) -> Result<Self> {
        // Suppress internal tracing (expand/broadcast ops create noise).
        let mut result = if trace::is_tracing() {
            trace::with_trace_suppressed(|| self.compare_tensor_compute(op, other))?
        } else {
            self.compare_tensor_compute(op, other)?
        };
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self, other])?;
            if let Some(id) = trace::record_op(
                TraceOp::CompareTensor { op },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    fn compare_tensor_compute(&self, op: CompareOp, other: &Self) -> Result<Self> {
        if self.device() != other.device() {
            return Err(TensorError::Unsupported(format!(
                "compare_tensor: mixed devices {} vs {}",
                self.device(),
                other.device()
            )));
        }
        let original_device = self.device();
        if original_device.is_gpu() {
            let (lhs_gpu, rhs_gpu) = if self.dims() == other.dims() {
                (self.clone(), other.clone())
            } else {
                let out_shape =
                    crate::dyn_tensor::ops::broadcast_output_shape(self.dims(), other.dims())?;
                (self.expand(&out_shape)?, other.expand(&out_shape)?)
            };
            if let Some(result) = gpu_backend_dispatch(|b| b.compare_tensor(&lhs_gpu, op, &rhs_gpu))
            {
                return result;
            }
        }
        let lhs = if original_device.is_gpu() {
            self.to_device(&Device::Cpu)?
        } else {
            self.clone()
        };
        let rhs = if original_device.is_gpu() {
            other.to_device(&Device::Cpu)?
        } else {
            other.clone()
        };
        let result = compare_tensor_cpu(&lhs, op, &rhs)?;
        if original_device.is_gpu() {
            result.to_device(&original_device)
        } else {
            Ok(result)
        }
    }
}

/// CPU comparison of two f32 tensors with NumPy-style broadcasting.
fn compare_tensor_cpu(lhs: &DynTensor, op: CompareOp, rhs: &DynTensor) -> Result<DynTensor> {
    let lhs_arr = lhs.to_f32_array()?;
    let rhs_arr = rhs.to_f32_array()?;
    let apply = |a: f32, b: f32| -> u8 {
        match op {
            CompareOp::Eq => u8::from(a == b),
            CompareOp::Ne => u8::from(a != b),
            CompareOp::Ge => u8::from(a >= b),
            CompareOp::Gt => u8::from(a > b),
            CompareOp::Lt => u8::from(a < b),
            CompareOp::Le => u8::from(a <= b),
        }
    };
    // Same-shape fast path avoids broadcast overhead.
    let mask = if lhs_arr.shape() == rhs_arr.shape() {
        ndarray::Zip::from(&lhs_arr)
            .and(&rhs_arr)
            .map_collect(|&a, &b| apply(a, b))
    } else {
        let out_shape =
            crate::dyn_tensor::ops::broadcast_output_shape(lhs_arr.shape(), rhs_arr.shape())?;
        let lhs_b = lhs_arr
            .broadcast(ndarray::IxDyn(&out_shape))
            .ok_or_else(|| TensorError::shape_mismatch(lhs.dims().to_vec(), rhs.dims().to_vec()))?;
        let rhs_b = rhs_arr
            .broadcast(ndarray::IxDyn(&out_shape))
            .ok_or_else(|| TensorError::shape_mismatch(lhs.dims().to_vec(), rhs.dims().to_vec()))?;
        ndarray::Zip::from(&lhs_b)
            .and(&rhs_b)
            .map_collect(|&a, &b| apply(a, b))
    };
    DynTensor::from_cpu_u8(mask)
}
