// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convenience methods for [`DynTensor`] matching candle-core API.
//!
//! `.get(i)`, `to_vec1_u32`, `to_scalar_u32`, `reshape_as`, `expand_as`.
//! Note: `.t()`, `to_vec2_f32`, `to_vec3_f32` live in dyn_tensor_shape.rs and
//! dyn_tensor.rs respectively.

use crate::dyn_tensor::{DynTensor, FloatStorage, TensorStorage};
use crate::{Device, Result, TensorError};
use ndarray::ArrayD;

impl DynTensor {
    /// Select a single index along dimension 0, removing that dimension.
    /// Matches candle's `.get(i)`.
    ///
    /// A tensor of shape `[A, B, C]` with `.get(i)` returns shape `[B, C]`.
    pub fn get(&self, index: usize) -> Result<Self> {
        self.narrow(0, index, 1)?.squeeze(0)
    }

    /// Check whether the tensor data is contiguous (C-style row-major layout).
    ///
    /// Matches candle's `Tensor::is_contiguous()`. Returns `true` for CPU
    /// tensors in standard layout and for all GPU tensors (GPU buffers are
    /// always contiguous).
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                // Check ArcArray<f32> first (from zero-copy narrow), then ArrayD<f32>.
                if let Some(arr) = any.downcast_ref::<ndarray::ArcArray<f32, ndarray::IxDyn>>() {
                    return arr.is_standard_layout();
                }
                if let Some(arr) = any.downcast_ref::<ArrayD<f32>>() {
                    return arr.is_standard_layout();
                }
                if let Some(fs) = any.downcast_ref::<FloatStorage>() {
                    return match fs {
                        FloatStorage::F32(arr) => arr.is_standard_layout(),
                        FloatStorage::F16(arr) => arr.is_standard_layout(),
                        FloatStorage::BF16(arr) => arr.is_standard_layout(),
                    };
                }
                if let Some(arr) = any.downcast_ref::<ArrayD<u32>>() {
                    return arr.is_standard_layout();
                }
                if let Some(arr) = any.downcast_ref::<ArrayD<u8>>() {
                    return arr.is_standard_layout();
                }
                if let Some(arr) = any.downcast_ref::<ArrayD<i64>>() {
                    return arr.is_standard_layout();
                }
                // Unknown dtype — assume non-contiguous (safe default).
                // Returning false triggers a copy via as_standard_layout(),
                // which is correct for safety: non-contiguous data passed to
                // reshape/matmul/conv would silently corrupt output.
                false
            }
            TensorStorage::Gpu { .. } => true,
            // Quantized storage is logically contiguous (block layout is sequential).
            TensorStorage::Quantized(_) => true,
        }
    }

    /// No-op gradient detach for candle API compatibility.
    ///
    /// `DynTensor` does not track gradients (that is `Var` / `TrackedTensor`
    /// in nn-autodiff), so this returns a clone. Enables mechanical
    /// find-and-replace migration from candle where `.detach()` is common.
    #[must_use]
    pub fn detach(&self) -> Self {
        self.clone()
    }

    /// Reshape to the same dimensions as `other`.
    ///
    /// Convenience wrapper matching PyTorch's `Tensor.reshape_as(other)`.
    pub fn reshape_as(&self, other: &Self) -> Result<Self> {
        self.reshape(other.dims())
    }

    /// Expand (broadcast) to the same dimensions as `other`.
    ///
    /// Convenience wrapper matching PyTorch's `Tensor.expand_as(other)`.
    pub fn expand_as(&self, other: &Self) -> Result<Self> {
        self.expand(other.dims())
    }

    /// Extract 1D u32 data. Copies to CPU if needed.
    #[deprecated(since = "0.1.0", note = "use `.to_vec1::<u32>()` instead")]
    pub fn to_vec1_u32(&self) -> Result<Vec<u32>> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_vec1_u32();
        }
        if self.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.rank(),
            });
        }
        let arr = self.as_cpu_u32()?;
        Ok(arr.iter().copied().collect())
    }

    /// Extract a scalar u32 value. Copies to CPU if needed.
    #[deprecated(since = "0.1.0", note = "use `.to_scalar::<u32>()` instead")]
    pub fn to_scalar_u32(&self) -> Result<u32> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_scalar_u32();
        }
        if self.numel() != 1 {
            return Err(TensorError::InvalidShape(format!(
                "to_scalar_u32 requires 1 element, got {}",
                self.numel()
            )));
        }
        let arr = self.as_cpu_u32()?;
        arr.iter().next().copied().ok_or_else(|| {
            TensorError::InvalidShape("to_scalar_u32: empty after numel==1 check".into())
        })
    }
}
