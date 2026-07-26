// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Built-in backends for [`VarBuilder`]: zeros (testing) and tensor map.
//!
//! These backends live in nn-core (no Metal dependency). The production
//! safetensors backend lives in nn-metal.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, TensorError};

use super::{TensorBackend, VarBuilder};

// -- ZerosBackend -------------------------------------------------------------

/// Zeros backend — returns zero tensors of the requested shape.
///
/// Used extensively in dvoice tests (~50 call sites).
/// `get_unchecked` returns a scalar zero since no shape is specified.
pub struct ZerosBackend;

impl TensorBackend for ZerosBackend {
    fn get(&self, dims: &[usize], _name: &str, dtype: DType, device: &Device) -> Result<DynTensor> {
        DynTensor::zeros(dims, dtype, device)
    }

    fn get_unchecked(&self, _name: &str, dtype: DType, device: &Device) -> Result<DynTensor> {
        // Without shape info, return a scalar zero (0-D tensor).
        DynTensor::zeros(&[], dtype, device)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

// -- TensorMapBackend ---------------------------------------------------------

/// In-memory tensor map backend.
///
/// Used in dvoice for manual weight construction (~15 call sites).
/// Stores tensors by string key. Shape and dtype are validated at get time.
pub struct TensorMapBackend {
    tensors: HashMap<String, DynTensor>,
}

impl TensorMapBackend {
    /// Create from a HashMap of named tensors.
    pub fn new(tensors: HashMap<String, DynTensor>) -> Self {
        Self { tensors }
    }
}

impl TensorBackend for TensorMapBackend {
    fn get(&self, dims: &[usize], name: &str, dtype: DType, device: &Device) -> Result<DynTensor> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound {
                name: name.to_string(),
            })?;
        // Validate shape.
        if t.dims() != dims {
            return Err(TensorError::shape_mismatch(
                dims.to_vec(),
                t.dims().to_vec(),
            ));
        }
        // Defense-in-depth: reject NaN/Inf weight data at load time (#943).
        check_weight_finite(t, name)?;
        convert_tensor(t, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<DynTensor> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| TensorError::TensorNotFound {
                name: name.to_string(),
            })?;
        // Defense-in-depth: reject NaN/Inf weight data at load time (#943).
        check_weight_finite(t, name)?;
        convert_tensor(t, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

/// Validate that a weight tensor contains no NaN/Inf values.
///
/// Matches the `SafeTensorsBackend` finiteness guard from #943.
/// Uses `any_non_finite()` which is zero-copy O(n) scan for CPU f32 tensors (no allocation).
fn check_weight_finite(t: &DynTensor, name: &str) -> Result<()> {
    if t.any_non_finite()? {
        // Count non-finite for the error message (only on the error path).
        let count = match t.as_cpu_f32() {
            Ok(view) => view.iter().filter(|v| !v.is_finite()).count(),
            Err(_) => {
                let data = t.to_f32_array()?;
                data.iter().filter(|v| !v.is_finite()).count()
            }
        };
        return Err(TensorError::NonFiniteData {
            name: name.to_string(),
            count,
        });
    }
    Ok(())
}

/// Convert a tensor's dtype and device if needed.
///
/// Uses `DynTensor::to_dtype()` for float-to-float conversion (F32, BF16, F16, F64)
/// and `to_device()` for CPU↔GPU transfer.
fn convert_tensor(t: &DynTensor, dtype: DType, device: &Device) -> Result<DynTensor> {
    let t = if t.dtype() != dtype {
        t.to_dtype(dtype)?
    } else {
        t.clone()
    };
    if t.device() != *device {
        t.to_device(device)
    } else {
        Ok(t)
    }
}

// -- Convenience constructors on VarBuilder -----------------------------------

impl VarBuilder {
    /// Create a zeros VarBuilder (for tests). Matches candle's `VarBuilder::zeros()`.
    ///
    /// Every `.get(dims, name)` call returns a zero tensor with the requested shape.
    pub fn zeros(dtype: DType, device: &Device) -> Self {
        Self::from_backend(Arc::new(ZerosBackend), dtype, *device)
    }

    /// Create from an in-memory tensor map. Matches candle's `VarBuilder::from_tensors()`.
    ///
    /// Keys should include full hierarchical paths (e.g., `"encoder.conv.weight"`).
    pub fn from_tensors(
        tensors: HashMap<String, DynTensor>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        Self::from_backend(Arc::new(TensorMapBackend::new(tensors)), dtype, *device)
    }
}
