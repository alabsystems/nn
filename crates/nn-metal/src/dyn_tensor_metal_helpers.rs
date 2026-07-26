// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for Metal DynTensor backend: broadcast, validation, non-finite scan.
//!
//! GPU ↔ CPU transfer methods (`gpu_to_cpu`, `cpu_to_gpu`) are in the
//! `transfer` submodule (`dyn_tensor_metal_transfer.rs`).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Result, TensorError};
use nn_dsl::ScalarType;

use super::MetalTensorData;

#[path = "dyn_tensor_metal_transfer.rs"]
mod transfer;
use transfer::validated_elem_range;

impl super::MetalDynBackend {
    /// Compute NumPy-style broadcast shape for two inputs.
    pub(super) fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
        let ndim = a.len().max(b.len());
        let mut out = vec![0usize; ndim];
        for i in 0..ndim {
            let da = if i < ndim - a.len() {
                1
            } else {
                a[i - (ndim - a.len())]
            };
            let db = if i < ndim - b.len() {
                1
            } else {
                b[i - (ndim - b.len())]
            };
            if da == db {
                out[i] = da;
            } else if da == 1 {
                out[i] = db;
            } else if db == 1 {
                out[i] = da;
            } else {
                return Err(TensorError::InvalidShape(format!(
                    "broadcast mismatch: {a:?} vs {b:?} at dim {i}"
                )));
            }
        }
        Ok(out)
    }

    /// Validate that a GPU tensor has a float dtype supported for Metal dispatch.
    ///
    /// Accepts F32, BF16, and F16. F32 uses 4 bytes/element, BF16/F16 use
    /// 2 bytes/element (f16 encoding) in the Metal buffer (#1646 D7).
    /// Returns `DtypeMismatch` for integer and other non-float dtypes.
    pub(super) fn validate_f32(tensor: &DynTensor, op_name: &str) -> Result<()> {
        match tensor.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => Ok(()),
            other => Err(TensorError::dtype_mismatch(DType::F32, other)),
        }?;
        let _ = op_name; // reserved for future diagnostics
        Ok(())
    }

    /// Validate that two GPU tensors have the same float dtype for Metal dispatch.
    ///
    /// `dispatch_def` uses a single dtype for ALL input buffers. If one tensor
    /// is BF16 (2-byte f16 buffer) and the other is F32 (4-byte float buffer),
    /// dispatching with either dtype will reinterpret the other buffer with the
    /// wrong byte width, causing silent data corruption.
    pub(super) fn validate_same_float_dtype(
        a: &DynTensor,
        b: &DynTensor,
        op_name: &str,
    ) -> Result<()> {
        Self::validate_f32(a, op_name)?;
        Self::validate_f32(b, op_name)?;
        if a.dtype() != b.dtype() {
            return Err(TensorError::dtype_mismatch(a.dtype(), b.dtype()));
        }
        Ok(())
    }

    /// Validate that a GPU tensor is strictly F32 for raw MSL kernel dispatch.
    ///
    /// Raw MSL kernels (topk, scatter, gather, index_select) use hardcoded
    /// `float` buffer types. BF16/F16 tensors now have 2-byte f16 buffers
    /// (#1646 D7) and cannot be read as `float*`. These ops must reject
    /// BF16/F16 and let the caller fall back to CPU or convert to F32.
    pub(super) fn validate_f32_buffer(tensor: &DynTensor, op_name: &str) -> Result<()> {
        if tensor.dtype() != DType::F32 {
            return Err(TensorError::dtype_mismatch(DType::F32, tensor.dtype()));
        }
        let _ = op_name;
        Ok(())
    }

    /// Count non-finite (NaN/Inf) elements in a GPU tensor by reading the
    /// Metal buffer directly. Returns a scalar count without constructing a
    /// full CPU `DynTensor`. Used by `check_output_finite` (#1320).
    ///
    /// F32 buffers are 4 bytes/element, scanned as f32.
    /// BF16/F16 buffers are 2 bytes/element (f16 encoding), converted to f32
    /// for finiteness check (#1646 D7).
    /// Non-float dtypes (U32, U8, I64) are always finite and return 0.
    ///
    /// Byte-offset aware: zero-copy views (#1945) share the parent buffer
    /// with `byte_offset > 0`. Scans only the view's element range.
    pub(super) fn gpu_count_non_finite(x: &DynTensor) -> Result<usize> {
        // Lazy batch (#2009): flush pending GPU work before CPU readback.
        crate::gpu_scope::flush()?;
        let data = x.gpu_data::<MetalTensorData>()?;
        let numel = x.checked_numel()?;
        let byte_offset = data.byte_offset;
        match x.dtype() {
            // BF16/F16: read 2-byte f16 Metal buffer, check via f32 (#1646 D7).
            DType::BF16 | DType::F16 => {
                let u16s = data.buffer.contents::<u16>().map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let (start, end) = validated_elem_range(byte_offset, 2, numel, u16s.len())?;
                Ok(u16s[start..end]
                    .iter()
                    .filter(|&&bits| !half::f16::from_bits(bits).to_f32().is_finite())
                    .count())
            }
            // F32: read 4-byte f32 Metal buffer directly.
            DType::F32 => {
                let floats = data.buffer.contents::<f32>().map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        e.to_string(),
                    )
                })?;
                let (start, end) = validated_elem_range(byte_offset, 4, numel, floats.len())?;
                Ok(floats[start..end].iter().filter(|v| !v.is_finite()).count())
            }
            DType::U32 | DType::U8 | DType::I64 => Ok(0),
            other => Err(TensorError::dtype_mismatch(DType::F32, other)),
        }
    }
}

/// Convert a DType to the MSL ScalarType for kernel generation.
///
/// F32 → ScalarType::F32 (float buffers), BF16/F16 → ScalarType::F16 (half buffers).
/// Matches the dispatch_def_inner mapping at dyn_tensor_metal_dispatch.rs.
pub(super) fn scalar_type_for_dtype(dtype: DType) -> ScalarType {
    match dtype {
        DType::BF16 | DType::F16 => ScalarType::F16,
        _ => ScalarType::F32,
    }
}

/// Retype a KernelDef's params and return type to match the dispatch dtype.
///
/// Kernel builders default to ScalarType::F32. When dispatching BF16/F16 tensors,
/// the MSL wrapper emits `float*` buffers (from F32 params) but the actual GPU
/// data is `half*` (2 bytes) — causing NaN/garbage reads. This function fixes
/// the KernelDef to use the correct ScalarType before it enters the cache.
pub(super) fn retype_kernel(
    mut def: nn_dsl::ir::KernelDef,
    stype: ScalarType,
) -> nn_dsl::ir::KernelDef {
    if stype == ScalarType::F32 {
        return def;
    }
    for p in &mut def.params {
        p.ty = stype;
    }
    def.return_type = stype;
    def
}

#[cfg(test)]
#[path = "dyn_tensor_metal_helpers_tests.rs"]
mod tests;
