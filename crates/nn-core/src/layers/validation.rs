// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for nn layer constructors.
//!
//! Deduplicates common validation patterns across nn layer constructors (#1205).

use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, TensorError};

/// Validate that `num_heads > 0`.
pub(crate) fn validate_heads(num_heads: usize, _layer: &str) -> Result<()> {
    if num_heads == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "num_heads must be > 0",
        });
    }
    Ok(())
}

/// Validate epsilon for normalization layers: must be finite and non-negative.
pub(crate) fn validate_eps(eps: f64, _layer: &str) -> Result<()> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(TensorError::ValueOutOfRange {
            description: "eps must be finite and non-negative",
        });
    }
    Ok(())
}

/// Validate that `a` is evenly divisible by `b`.
pub(crate) fn validate_divisible(
    a: usize,
    b: usize,
    _a_name: &str,
    _b_name: &str,
    _layer: &str,
) -> Result<()> {
    if !a.is_multiple_of(b) {
        return Err(TensorError::ValueOutOfRange {
            description: "dimension not evenly divisible",
        });
    }
    Ok(())
}

// -- bf16/f16 CPU round-trip helper for decomposed norm fallback (#1672) ------
//
// Non-f32 GPU tensors must round-trip through CPU for decomposed norm chains
// because gpu_relabel_dtype doesn't convert buffer data — a 2-byte bf16 buffer
// reinterpreted as f32 causes size mismatch. This helper deduplicates the
// pattern used by LayerNorm, GroupNorm, RmsNorm, and InstanceNorm fallback paths.

/// Helper for moving non-f32 GPU tensors to CPU before decomposed norm ops.
///
/// Usage:
/// ```ignore
/// let rt = CpuRoundTrip::new(x);
/// let x_cpu = rt.prepare(x)?;
/// let w_cpu = rt.prepare_param(&self.weight)?;
/// // ... compute norm on CPU tensors ...
/// rt.restore(result)
/// ```
pub(crate) struct CpuRoundTrip {
    need_roundtrip: bool,
    device: Device,
}

impl CpuRoundTrip {
    /// Detect whether a CPU round-trip is needed for the given input tensor.
    pub(crate) fn new(x: &DynTensor) -> Self {
        let device = x.device();
        let need_roundtrip = x.dtype() != DType::F32 && device.is_gpu();
        Self {
            need_roundtrip,
            device,
        }
    }

    /// Move input tensor to CPU if needed, otherwise clone.
    pub(crate) fn prepare(&self, x: &DynTensor) -> Result<DynTensor> {
        if self.need_roundtrip {
            x.to_device(&Device::Cpu)
        } else {
            Ok(x.clone())
        }
    }

    /// Move a parameter tensor (weight/bias) to CPU if needed, otherwise clone.
    pub(crate) fn prepare_param(&self, param: &DynTensor) -> Result<DynTensor> {
        if self.need_roundtrip && param.device().is_gpu() {
            param.to_device(&Device::Cpu)
        } else {
            Ok(param.clone())
        }
    }

    /// Move result back to the original device if a round-trip was performed.
    pub(crate) fn restore(&self, result: DynTensor) -> Result<DynTensor> {
        if self.need_roundtrip {
            result.to_device(&self.device)
        } else {
            Ok(result)
        }
    }
}

/// Validate that a weight tensor contains no NaN/Inf values.
///
/// Uses `any_non_finite()` (zero-copy for CPU f32), counts non-finite
/// elements only on the error path.
pub(crate) fn validate_weight_finite(t: &DynTensor, name: &str) -> Result<()> {
    if t.any_non_finite()? {
        let count = match t.as_cpu_f32() {
            Ok(view) => view.iter().filter(|v| !v.is_finite()).count(),
            Err(_) => {
                // GPU tensors: transfer to CPU before reading f32 data.
                let cpu_t = t.to_device(&Device::Cpu)?;
                let data = cpu_t.to_f32_array()?;
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

#[cfg(test)]
#[path = "validation_tests.rs"]
mod tests;
