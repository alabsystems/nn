// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Data extraction accessors for [`DynTensor`].
//!
//! Provides `to_vec1_f32`, `to_flat_vec_f32`, `to_flat_vec_u32`,
//! `to_flat_vec_u8`, `to_flat_vec_i64`, `to_scalar_f32`, `to_vec2_f32`,
//! `to_vec3_f32`, and `flatten_all`.
//!
//! **Canonical API:** Prefer the generic `WithDType` accessors in `with_dtype.rs`
//! (e.g., `to_vec1::<f32>()`, `to_scalar::<f32>()`) for new code. These
//! typed-suffix variants (`to_vec1_f32`, etc.) are retained for convenience
//! and backward compatibility. Both produce identical results.

use super::DynTensor;
use crate::{Device, Result, TensorError};

impl DynTensor {
    /// Extract as a flat Vec<f32>. Copies data to CPU if needed.
    ///
    /// Works on all float dtypes (F32, F16, BF16) — converts to f32 on demand.
    #[deprecated(since = "0.1.0", note = "use to_vec1::<f32>() instead")]
    pub fn to_vec1_f32(&self) -> Result<Vec<f32>> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_vec1_f32();
        }
        if self.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.rank(),
            });
        }
        // Try zero-copy f32 first, then convert-on-demand for f16/bf16.
        match self.as_cpu_f32() {
            Ok(arr) => Ok(arr.iter().copied().collect()),
            Err(_) => {
                let arr = self.to_f32_array()?;
                Ok(arr.iter().copied().collect())
            }
        }
    }

    /// Extract as a flat Vec<f32> regardless of rank.
    ///
    /// Works on all float dtypes (F32, F16, BF16) — converts to f32 on demand.
    #[deprecated(since = "0.1.0", note = "use to_flat_vec::<f32>() instead")]
    pub fn to_flat_vec_f32(&self) -> Result<Vec<f32>> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_flat_vec::<f32>();
        }
        // Try zero-copy f32 first, then convert-on-demand for f16/bf16.
        match self.as_cpu_f32() {
            Ok(arr) => Ok(arr.iter().copied().collect()),
            Err(_) => {
                let arr = self.to_f32_array()?;
                Ok(arr.iter().copied().collect())
            }
        }
    }

    /// Extract as a flat Vec<u32> regardless of rank.
    ///
    /// Reads u32 storage natively — no F32 intermediate, so values >= 2^24
    /// are preserved exactly.
    #[deprecated(since = "0.1.0", note = "use to_flat_vec::<u32>() instead")]
    pub fn to_flat_vec_u32(&self) -> Result<Vec<u32>> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_flat_vec_u32();
        }
        let arr = self.as_cpu_u32()?;
        Ok(arr.iter().copied().collect())
    }

    /// Extract as a flat `Vec<u8>` regardless of rank.
    ///
    /// Reads u8 storage natively — used for boolean mask tensors.
    #[deprecated(since = "0.1.0", note = "use to_flat_vec::<u8>() instead")]
    pub fn to_flat_vec_u8(&self) -> Result<Vec<u8>> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_flat_vec_u8();
        }
        let arr = self.as_cpu_u8()?;
        Ok(arr.iter().copied().collect())
    }

    /// Extract as a flat `Vec<i64>` regardless of rank.
    ///
    /// Reads i64 storage natively — used for token IDs (Embedding, Qwen3, Whisper).
    #[deprecated(since = "0.1.0", note = "use to_flat_vec::<i64>() instead")]
    pub fn to_flat_vec_i64(&self) -> Result<Vec<i64>> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_flat_vec_i64();
        }
        let arr = self.as_cpu_i64()?;
        Ok(arr.iter().copied().collect())
    }

    /// Extract as a scalar f32.
    ///
    /// Works on all float dtypes (F32, F16, BF16) — converts to f32 on demand.
    #[deprecated(since = "0.1.0", note = "use to_scalar::<f32>() instead")]
    pub fn to_scalar_f32(&self) -> Result<f32> {
        if self.device().is_gpu() {
            #[allow(deprecated)]
            return self.to_device(&Device::Cpu)?.to_scalar_f32();
        }
        if self.numel() != 1 {
            return Err(TensorError::InvalidShape(format!(
                "to_scalar requires 1 element, got {}",
                self.numel()
            )));
        }
        // Try zero-copy f32 first, then convert-on-demand for f16/bf16.
        match self.as_cpu_f32() {
            Ok(arr) => arr.iter().next().copied().ok_or_else(|| {
                TensorError::InvalidShape("to_scalar: empty array after numel==1 check".into())
            }),
            Err(_) => {
                let arr = self.to_f32_array()?;
                arr.iter().next().copied().ok_or_else(|| {
                    TensorError::InvalidShape("to_scalar: empty array after numel==1 check".into())
                })
            }
        }
    }

    /// Extract as a nested `Vec<Vec<f32>>` for a 2-D tensor (candle compat).
    #[deprecated(since = "0.1.0", note = "use to_vec2::<f32>() instead")]
    #[allow(deprecated)]
    pub fn to_vec2_f32(&self) -> Result<Vec<Vec<f32>>> {
        let (d0, d1) = self.dims2()?;
        let flat = self.to_flat_vec::<f32>()?;
        Ok((0..d0)
            .map(|i| flat[i * d1..(i + 1) * d1].to_vec())
            .collect())
    }

    /// Extract as a nested `Vec<Vec<Vec<f32>>>` for a 3-D tensor (candle compat).
    #[deprecated(since = "0.1.0", note = "use to_vec3::<f32>() instead")]
    #[allow(deprecated)]
    pub fn to_vec3_f32(&self) -> Result<Vec<Vec<Vec<f32>>>> {
        let (d0, d1, d2) = self.dims3()?;
        let flat = self.to_flat_vec::<f32>()?;
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

    /// Flatten all dimensions into a 1-D tensor.
    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape([self.checked_numel()?])
    }
}
