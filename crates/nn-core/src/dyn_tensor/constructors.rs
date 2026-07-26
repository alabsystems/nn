// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Constructors for [`DynTensor`]: `new`, `zeros`, `ones`, `from_vec`, `full`, `arange`.

use super::{DynTensor, TensorStorage};
use crate::dyn_tensor::Shape;
use crate::tensor::checked_dim_product;
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

impl DynTensor {
    // -- Constructors ---------------------------------------------------------

    /// Create a tensor from a flat f32 slice with explicit dimensions.
    ///
    /// Accepts `&[usize]`, tuples `(d0, d1)` / `(d0, d1, d2)` / `(d0, d1, d2, d3)`,
    /// `Vec<usize>`, or [`Shape`] for the `dims` parameter (candle compatibility).
    pub fn new(data: &[f32], dims: impl Into<Shape>, device: &Device) -> Result<Self> {
        let shape = dims.into();
        let dims = shape.dims();
        let expected = checked_dim_product(dims)?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        let arr = ArrayD::from_shape_vec(IxDyn(dims), data.to_vec())?;
        let t = Self {
            dims: dims.to_vec(),
            dtype: DType::F32,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        };
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a zero-filled tensor.
    ///
    /// Float dtypes (F32, BF16, F16, F64) all create f32 storage labeled as F32,
    /// matching `full()`. The `dtype` parameter is accepted for candle API
    /// compatibility. Integer dtypes (U32, U8, I64) create native-typed storage.
    ///
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`] for dimensions.
    pub fn zeros(dims: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Self> {
        let shape = dims.into();
        let dims = shape.dims();
        checked_dim_product(dims)?;
        let t = Self::cpu_zeros_typed(dims, dtype)?;
        let mut result = if device.is_gpu() {
            t.to_device(device)?
        } else {
            t
        };
        Self::maybe_register_constant(&mut result, 0.0);
        Ok(result)
    }

    /// Create a one-filled tensor.
    ///
    /// Float dtypes (F32, BF16, F16, F64) all create f32 storage labeled as F32,
    /// matching `full()`. The `dtype` parameter is accepted for candle API
    /// compatibility. Integer dtypes (U32, U8, I64) create native-typed storage.
    ///
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`] for dimensions.
    pub fn ones(dims: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Self> {
        let shape = dims.into();
        let dims = shape.dims();
        checked_dim_product(dims)?;
        let t = Self::cpu_ones_typed(dims, dtype)?;
        let mut result = if device.is_gpu() {
            t.to_device(device)?
        } else {
            t
        };
        Self::maybe_register_constant(&mut result, 1.0);
        Ok(result)
    }

    /// Create a tensor from a flat f32 slice with explicit dimensions.
    ///
    /// Alias for [`new`](Self::new) — matches candle's `Tensor::from_slice`.
    pub fn from_slice(data: &[f32], dims: impl Into<Shape>, device: &Device) -> Result<Self> {
        Self::new(data, dims, device)
    }

    /// Create a tensor from a `Vec<f32>` with explicit dimensions.
    ///
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`] for dimensions.
    pub fn from_vec(data: Vec<f32>, dims: impl Into<Shape>, device: &Device) -> Result<Self> {
        let shape = dims.into();
        let dims = shape.dims();
        let expected = checked_dim_product(dims)?;
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        let arr = ArrayD::from_shape_vec(IxDyn(dims), data)?;
        let t = Self {
            dims: dims.to_vec(),
            dtype: DType::F32,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        };
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a constant-filled tensor.
    ///
    /// Float dtypes (F32, BF16, F16, F64) create f32 storage labeled as F32,
    /// matching the DynTensor dtype/storage invariant. Integer dtypes (U32, U8, I64)
    /// create native-typed storage with range-checked conversion from f64.
    ///
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`] for dimensions.
    pub fn full(dims: impl Into<Shape>, val: f64, dtype: DType, device: &Device) -> Result<Self> {
        let shape = dims.into();
        let dims = shape.dims();
        checked_dim_product(dims)?;
        let t = match dtype {
            DType::U32 => {
                if val < 0.0 || val > f64::from(u32::MAX) || val.fract() != 0.0 || !val.is_finite()
                {
                    return Err(TensorError::DtypeConversion {
                        source_dtype: DType::F64,
                        target_dtype: DType::U32,
                        reason: format!("value {val} cannot be represented as U32"),
                    });
                }
                Self::from_cpu_u32(ArrayD::from_elem(IxDyn(dims), val as u32))?
            }
            DType::U8 => {
                if val < 0.0 || val > f64::from(u8::MAX) || val.fract() != 0.0 || !val.is_finite() {
                    return Err(TensorError::DtypeConversion {
                        source_dtype: DType::F64,
                        target_dtype: DType::U8,
                        reason: format!("value {val} cannot be represented as U8"),
                    });
                }
                Self::from_cpu_u8(ArrayD::from_elem(IxDyn(dims), val as u8))?
            }
            DType::I64 => {
                if val < i64::MIN as f64
                    || val >= i64::MAX as f64
                    || val.fract() != 0.0
                    || !val.is_finite()
                {
                    return Err(TensorError::DtypeConversion {
                        source_dtype: DType::F64,
                        target_dtype: DType::I64,
                        reason: format!("value {val} cannot be represented as I64"),
                    });
                }
                Self::from_cpu_i64(ArrayD::from_elem(IxDyn(dims), val as i64))?
            }
            // Float dtypes use native FloatStorage.
            DType::F32 | DType::F16 | DType::BF16 => {
                let fs = super::FloatStorage::full(dims, val, dtype)?;
                let actual_dtype = fs.dtype();
                Self {
                    dims: dims.to_vec(),
                    dtype: actual_dtype,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                }
            }
            DType::F64 => {
                // F64 demoted to F32 (no f64 storage in DynTensor).
                let fs = super::FloatStorage::full(dims, val, DType::F32)?;
                Self {
                    dims: dims.to_vec(),
                    dtype: DType::F32,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                }
            }
            DType::I32 | DType::Bool => {
                return Err(TensorError::Unsupported(format!(
                    "full(): dtype {dtype} not supported"
                )));
            }
        };
        let mut result = if device.is_gpu() {
            t.to_device(device)?
        } else {
            t
        };
        Self::maybe_register_constant(&mut result, val);
        Ok(result)
    }

    /// Register a Constant trace node if tracing is active.
    ///
    /// Called from `full()` (and transitively `scalar_like()`) so that
    /// downstream binary ops can reference this tensor's trace node ID
    /// instead of failing with "no trace ID during active trace".
    fn maybe_register_constant(tensor: &mut Self, value: f64) {
        if super::trace::is_tracing() {
            if let Some(id) = super::trace::record_op(
                super::trace::TraceOp::Constant { value },
                &[],
                tensor.dims(),
                tensor.dtype(),
            ) {
                tensor.set_trace_id(id);
            }
        }
    }

    /// Create a zero-filled tensor with the same shape, dtype, and device (candle compat).
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.dims(), self.dtype(), &self.device())
    }

    /// Create a one-filled tensor with the same shape, dtype, and device (candle compat).
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.dims(), self.dtype(), &self.device())
    }

    /// Create a constant-filled tensor with the same shape, dtype, and device.
    ///
    /// Matches PyTorch's `torch.full_like(self, val)`.
    pub fn full_like(&self, val: f64) -> Result<Self> {
        Self::full(self.dims(), val, self.dtype(), &self.device())
    }

    // cat() and stack() are in dyn_tensor_shape.rs

    /// Create a 1-D f32 tensor with values from start (inclusive) to end (exclusive), step 1.
    ///
    /// Returns an f32 tensor. For integer index/ID sequences, prefer
    /// [`arange_u32`](Self::arange_u32) which returns a native U32 tensor —
    /// f32 loses precision for integer values > 2^24 (16,777,216).
    pub fn arange(start: f64, end: f64, device: &Device) -> Result<Self> {
        Self::arange_step(start, end, 1.0, device)
    }

    /// Create a 1-D f32 tensor with values from start (inclusive) to end (exclusive) with step.
    ///
    /// Matches `torch.arange(start, end, step)` / candle `Tensor::arange_step`.
    /// For integer index/ID sequences, prefer [`arange_u32`](Self::arange_u32).
    pub fn arange_step(start: f64, end: f64, step: f64, device: &Device) -> Result<Self> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "arange_step: start, end, and step must be finite",
            });
        }
        if step == 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "arange_step: step must be non-zero",
            });
        }
        if (step > 0.0 && end <= start) || (step < 0.0 && end >= start) {
            return Self::from_vec(vec![], &[0], device);
        }
        let n_f64 = ((end - start) / step).ceil();
        // Guard against overflow: (end - start) can be ±Inf for large finite inputs
        // (e.g., start=-MAX, end=MAX), and the division can overflow to Inf for
        // tiny steps. The `as usize` cast saturates Inf → usize::MAX, which would
        // attempt an impossible allocation.
        if !n_f64.is_finite() || n_f64 < 0.0 || n_f64 > isize::MAX as f64 {
            return Err(TensorError::InvalidShape(format!(
                "arange_step: result length {n_f64} exceeds maximum"
            )));
        }
        let n = n_f64 as usize;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let val = start + i as f64 * step;
                super::checked_f64_to_f32(val, "arange_step element")
            })
            .collect::<Result<Vec<f32>>>()?;
        Self::from_vec(data, &[n], device)
    }

    /// Create a typed zero-filled CPU tensor.
    ///
    /// Float dtypes (F32, BF16, F16) use native [`FloatStorage`] — bf16/f16
    /// tensors are stored at half precision, not converted to f32. F64 is
    /// stored as F32 (f64 storage not supported in DynTensor).
    /// Integer dtypes (U32, U8, I64) create native-typed storage.
    pub(crate) fn cpu_zeros_typed(dims: &[usize], dtype: DType) -> Result<Self> {
        use super::FloatStorage;
        match dtype {
            DType::U32 => Self::from_cpu_u32(ArrayD::<u32>::zeros(IxDyn(dims))),
            DType::U8 => Self::from_cpu_u8(ArrayD::<u8>::zeros(IxDyn(dims))),
            DType::I64 => Self::from_cpu_i64(ArrayD::<i64>::zeros(IxDyn(dims))),
            DType::F32 | DType::F16 | DType::BF16 => {
                // F64 requests are demoted to F32 (no f64 storage in DynTensor).
                let fs = FloatStorage::zeros(dims, dtype)?;
                let actual_dtype = fs.dtype();
                Ok(Self {
                    dims: dims.to_vec(),
                    dtype: actual_dtype,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                })
            }
            DType::F64 => {
                let fs = FloatStorage::zeros(dims, DType::F32)?;
                Ok(Self {
                    dims: dims.to_vec(),
                    dtype: DType::F32,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                })
            }
            DType::I32 | DType::Bool => Err(TensorError::Unsupported(format!(
                "zeros: dtype {dtype} not supported"
            ))),
        }
    }

    /// Create a typed one-filled CPU tensor.
    ///
    /// Float dtypes (F32, BF16, F16) use native [`FloatStorage`] — bf16/f16
    /// tensors are stored at half precision, not converted to f32. F64 is
    /// stored as F32 (f64 storage not supported in DynTensor).
    /// Integer dtypes (U32, U8, I64) create native-typed storage.
    pub(crate) fn cpu_ones_typed(dims: &[usize], dtype: DType) -> Result<Self> {
        use super::FloatStorage;
        match dtype {
            DType::U32 => Self::from_cpu_u32(ArrayD::<u32>::ones(IxDyn(dims))),
            DType::U8 => Self::from_cpu_u8(ArrayD::<u8>::ones(IxDyn(dims))),
            DType::I64 => Self::from_cpu_i64(ArrayD::<i64>::ones(IxDyn(dims))),
            DType::F32 | DType::F16 | DType::BF16 => {
                let fs = FloatStorage::ones(dims, dtype)?;
                let actual_dtype = fs.dtype();
                Ok(Self {
                    dims: dims.to_vec(),
                    dtype: actual_dtype,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                })
            }
            DType::F64 => {
                let fs = FloatStorage::ones(dims, DType::F32)?;
                Ok(Self {
                    dims: dims.to_vec(),
                    dtype: DType::F32,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                })
            }
            DType::I32 | DType::Bool => Err(TensorError::Unsupported(format!(
                "ones: dtype {dtype} not supported"
            ))),
        }
    }
}
