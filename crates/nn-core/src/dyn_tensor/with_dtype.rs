// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generic type-to-dtype mapping for candle API compatibility.
//!
//! The [`WithDType`] trait maps Rust scalar types (f32, u32, u8, i64) to
//! [`DType`] values and provides generic extraction from [`DynTensor`].
//! This enables candle-compatible `.to_vec1::<f32>()` syntax instead of
//! the typed `.to_vec1_f32()` suffix variants.
//!
//! Covers 533+ dvoice call sites using `.to_vec1()`, `.to_vec2()`,
//! `.to_vec3()`, and `.to_scalar()`.

use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, Shape, TensorError};

/// Maps a Rust scalar type to its [`DType`] discriminant and provides
/// flat data extraction from a CPU [`DynTensor`].
///
/// Matches candle's `WithDType` trait so that code like
/// `tensor.to_vec1::<f32>()` works identically in both frameworks.
pub trait WithDType: Copy + Send + Sync + 'static {
    /// The [`DType`] variant for this scalar.
    const DTYPE: DType;

    /// Extract flat data from a CPU tensor. Returns error if the tensor
    /// is on GPU or has a mismatched dtype.
    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>>;

    /// Construct a [`DynTensor`] from a typed `Vec<Self>`.
    /// Delegates to the appropriate type-specific constructor.
    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor>;
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;

    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>> {
        let arr = tensor.to_f32_array()?;
        Ok(arr.iter().copied().collect())
    }

    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor> {
        DynTensor::from_vec(data, dims, device)
    }
}

impl WithDType for u32 {
    const DTYPE: DType = DType::U32;

    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>> {
        let arr = tensor.as_cpu_u32()?;
        Ok(arr.iter().copied().collect())
    }

    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor> {
        DynTensor::from_vec_u32(data, dims, device)
    }
}

impl WithDType for u8 {
    const DTYPE: DType = DType::U8;

    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>> {
        let arr = tensor.as_cpu_u8()?;
        Ok(arr.iter().copied().collect())
    }

    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor> {
        DynTensor::from_vec_u8(data, dims, device)
    }
}

impl WithDType for i64 {
    const DTYPE: DType = DType::I64;

    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>> {
        let arr = tensor.as_cpu_i64()?;
        Ok(arr.iter().copied().collect())
    }

    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor> {
        DynTensor::from_vec_i64(data, dims, device)
    }
}

impl WithDType for half::f16 {
    const DTYPE: DType = DType::F16;

    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>> {
        let arr = tensor.as_cpu_f16()?;
        Ok(arr.iter().copied().collect())
    }

    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor> {
        DynTensor::from_vec_f16(data, dims, device)
    }
}

impl WithDType for half::bf16 {
    const DTYPE: DType = DType::BF16;

    fn extract_flat(tensor: &DynTensor) -> Result<Vec<Self>> {
        let arr = tensor.as_cpu_bf16()?;
        Ok(arr.iter().copied().collect())
    }

    fn from_vec_to_tensor(
        data: Vec<Self>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<DynTensor> {
        DynTensor::from_vec_bf16(data, dims, device)
    }
}

impl DynTensor {
    /// Construct a tensor from a typed `Vec<T>`, where `T` determines the dtype.
    ///
    /// Generic replacement for `from_vec` (f32), `from_vec_u32`, `from_vec_i64`,
    /// `from_vec_u8`, `from_vec_f16`, `from_vec_bf16`.
    ///
    /// # Examples
    /// ```ignore
    /// let t = DynTensor::from_typed_vec::<f32>(vec![1.0, 2.0], &[2], &cpu)?;
    /// let t = DynTensor::from_typed_vec::<u32>(vec![1, 2, 3], &[3], &cpu)?;
    /// ```
    pub fn from_typed_vec<T: WithDType>(
        data: Vec<T>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<Self> {
        T::from_vec_to_tensor(data, dims, device)
    }

    /// Extract 1-D data as `Vec<S>`. Copies to CPU if needed.
    ///
    /// Matches candle's `Tensor::to_vec1::<f32>()` API.
    ///
    /// # Errors
    /// Returns [`TensorError::RankMismatch`] if the tensor is not 1-D.
    /// Returns [`TensorError::DTypeMismatch`] if the dtype doesn't match `S`.
    pub fn to_vec1<S: WithDType>(&self) -> Result<Vec<S>> {
        if self.device().is_gpu() {
            return self.to_device(&Device::Cpu)?.to_vec1::<S>();
        }
        if self.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.rank(),
            });
        }
        S::extract_flat(self)
    }

    /// Extract 2-D data as `Vec<Vec<S>>`. Copies to CPU if needed.
    ///
    /// Matches candle's `Tensor::to_vec2::<f32>()` API.
    pub fn to_vec2<S: WithDType>(&self) -> Result<Vec<Vec<S>>> {
        if self.device().is_gpu() {
            return self.to_device(&Device::Cpu)?.to_vec2::<S>();
        }
        let (d0, d1) = self.dims2()?;
        let flat = S::extract_flat(self)?;
        Ok((0..d0)
            .map(|i| flat[i * d1..(i + 1) * d1].to_vec())
            .collect())
    }

    /// Extract 3-D data as `Vec<Vec<Vec<S>>>`. Copies to CPU if needed.
    ///
    /// Matches candle's `Tensor::to_vec3::<f32>()` API.
    pub fn to_vec3<S: WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        if self.device().is_gpu() {
            return self.to_device(&Device::Cpu)?.to_vec3::<S>();
        }
        let (d0, d1, d2) = self.dims3()?;
        let flat = S::extract_flat(self)?;
        Ok((0..d0)
            .map(|i| {
                (0..d1)
                    .map(|j| {
                        let start = (i * d1 + j) * d2;
                        flat[start..start + d2].to_vec()
                    })
                    .collect()
            })
            .collect())
    }

    /// Extract all elements as a flat `Vec<S>`, regardless of rank.
    /// Copies to CPU if needed.
    ///
    /// This is the generic replacement for the deprecated `to_flat_vec_f32()`,
    /// `to_flat_vec_u32()`, etc.
    ///
    /// # Errors
    /// Returns [`TensorError::DTypeMismatch`] if the dtype doesn't match `S`.
    pub fn to_flat_vec<S: WithDType>(&self) -> Result<Vec<S>> {
        if self.device().is_gpu() {
            return self.to_device(&Device::Cpu)?.to_flat_vec::<S>();
        }
        S::extract_flat(self)
    }

    /// Extract a single scalar value. Copies to CPU if needed.
    ///
    /// Matches candle's `Tensor::to_scalar::<f32>()` API.
    ///
    /// # Errors
    /// Returns error if the tensor has more than 1 element or dtype mismatch.
    pub fn to_scalar<S: WithDType>(&self) -> Result<S> {
        if self.device().is_gpu() {
            return self.to_device(&Device::Cpu)?.to_scalar::<S>();
        }
        if self.numel() != 1 {
            return Err(TensorError::InvalidShape(format!(
                "to_scalar requires 1 element, got {}",
                self.numel()
            )));
        }
        let flat = S::extract_flat(self)?;
        flat.into_iter().next().ok_or_else(|| {
            TensorError::InvalidShape("to_scalar: empty after numel==1 check".into())
        })
    }
}
