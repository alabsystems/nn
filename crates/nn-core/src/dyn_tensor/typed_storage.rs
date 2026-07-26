// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integer typed storage helpers and constructors for [`DynTensor`].
//!
//! Provides `as_cpu_u32`, `as_cpu_u8`, `as_cpu_i64`, `from_cpu_u32`,
//! `from_cpu_u8`, `from_cpu_i64`, `from_vec_u8`, `from_vec_u32`,
//! `from_vec_i64`, and `arange_u32`.

use crate::dyn_tensor::{DynTensor, Shape, TensorStorage};
use crate::tensor::checked_dim_product;
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

impl DynTensor {
    /// Get the underlying CPU u32 ndarray view. Returns error if not CPU U32.
    ///
    /// Zero-copy: returns a view into the internal ndarray storage.
    /// For a flat `Vec<u32>`, use [`to_flat_vec_u32`](Self::to_flat_vec_u32) instead.
    pub fn as_cpu_u32(&self) -> Result<ndarray::ArrayViewD<'_, u32>> {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                let arr = any
                    .downcast_ref::<ArrayD<u32>>()
                    .ok_or(TensorError::dtype_mismatch(DType::U32, self.dtype))?;
                Ok(arr.view())
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            TensorStorage::Quantized(_) => Err(TensorError::dtype_mismatch(DType::U32, self.dtype)),
        }
    }

    /// Create DynTensor from an owned CPU u32 ndarray.
    pub fn from_cpu_u32(arr: ArrayD<u32>) -> Result<Self> {
        let dims = arr.shape().to_vec();
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype: DType::U32,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        })
    }

    /// Get the underlying CPU u8 ndarray view. Returns error if not CPU U8.
    ///
    /// Zero-copy: returns a view into the internal ndarray storage.
    pub fn as_cpu_u8(&self) -> Result<ndarray::ArrayViewD<'_, u8>> {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                let arr = any
                    .downcast_ref::<ArrayD<u8>>()
                    .ok_or(TensorError::dtype_mismatch(DType::U8, self.dtype))?;
                Ok(arr.view())
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            TensorStorage::Quantized(_) => Err(TensorError::dtype_mismatch(DType::U8, self.dtype)),
        }
    }

    /// Create DynTensor from an owned CPU u8 ndarray.
    pub fn from_cpu_u8(arr: ArrayD<u8>) -> Result<Self> {
        let dims = arr.shape().to_vec();
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype: DType::U8,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        })
    }

    // -- I64 Storage ----------------------------------------------------------

    /// Get the underlying CPU i64 ndarray view. Returns error if not CPU I64.
    ///
    /// Zero-copy: returns a view into the internal ndarray storage.
    /// For token ID extraction (Embedding, Qwen3, Whisper), prefer
    /// [`to_flat_vec_i64`](Self::to_flat_vec_i64) for a flat `Vec<i64>`.
    pub fn as_cpu_i64(&self) -> Result<ndarray::ArrayViewD<'_, i64>> {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                let arr = any
                    .downcast_ref::<ArrayD<i64>>()
                    .ok_or(TensorError::dtype_mismatch(DType::I64, self.dtype))?;
                Ok(arr.view())
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            TensorStorage::Quantized(_) => Err(TensorError::dtype_mismatch(DType::I64, self.dtype)),
        }
    }

    /// Create DynTensor from an owned CPU i64 ndarray.
    pub fn from_cpu_i64(arr: ArrayD<i64>) -> Result<Self> {
        let dims = arr.shape().to_vec();
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype: DType::I64,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        })
    }

    // -- I64 Constructors -----------------------------------------------------

    /// Create a tensor from a flat i64 slice with explicit dimensions.
    ///
    /// Used for token IDs, embedding lookups, diffusion timesteps, and
    /// other integer index tensors matching candle's I64 usage patterns.
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`].
    pub fn from_vec_i64(data: Vec<i64>, dims: impl Into<Shape>, device: &Device) -> Result<Self> {
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
        let t = Self::from_cpu_i64(arr)?;
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    // -- U8 Constructors (boolean masks) --------------------------------------

    /// Create a tensor from a flat u8 slice with explicit dimensions.
    ///
    /// Used for boolean mask tensors (from `ge`, `gt`, etc.) that need to be
    /// constructed directly in user code or tests.
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`].
    pub fn from_vec_u8(data: Vec<u8>, dims: impl Into<Shape>, device: &Device) -> Result<Self> {
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
            dtype: DType::U8,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        };
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    // -- U32 Constructors -----------------------------------------------------

    /// Create a tensor from a flat u32 slice with explicit dimensions.
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`].
    pub fn from_vec_u32(data: Vec<u32>, dims: impl Into<Shape>, device: &Device) -> Result<Self> {
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
            dtype: DType::U32,
            storage: TensorStorage::Cpu(Arc::new(arr)),
            trace_node_id: None,
        };
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a 1-D u32 tensor with values from start (inclusive) to end (exclusive).
    pub fn arange_u32(start: u32, end: u32, device: &Device) -> Result<Self> {
        if end <= start {
            return Self::from_vec_u32(vec![], &[0], device);
        }
        let data: Vec<u32> = (start..end).collect();
        let n = data.len();
        Self::from_vec_u32(data, &[n], device)
    }

    /// Create a 1-D i64 tensor with values from start (inclusive) to end (exclusive).
    ///
    /// Used for token ID ranges, sequence position indices, and other
    /// integer index generation matching candle's `Tensor::arange` with I64.
    pub fn arange_i64(start: i64, end: i64, device: &Device) -> Result<Self> {
        if end <= start {
            return Self::from_vec_i64(vec![], &[0], device);
        }
        let data: Vec<i64> = (start..end).collect();
        let n = data.len();
        Self::from_vec_i64(data, &[n], device)
    }
}
