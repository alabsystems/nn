// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SafeTensors backend for [`VarBuilder`] — wraps [`WeightMap`] (mmap).
//!
//! This is the production weight-loading backend for dvoice and other
//! consumers that load safetensors files via memory-mapped Metal buffers.
//!
//! The backend reads raw bytes from `WeightMap` and constructs [`DynTensor`]
//! values. Supports F32, BF16, and F16 stored dtypes. When `requested_dtype`
//! matches the stored dtype, data loads natively without f32 conversion (#1646).
//! Otherwise BF16/F16 are converted to F32 (matching candle behavior).
//!
//! See `designs/2026-03-03-var-builder-weight-loading.md` (D5 Direction 2).

use std::sync::Arc;

use nn_core::var_builder::TensorBackend;
use nn_core::{DType, Device, DynTensor, Result as TensorResult, TensorError, VarBuilder};

use crate::safetensors::WeightMap;

/// SafeTensors backend for VarBuilder — wraps [`WeightMap`].
///
/// Loads tensors from a memory-mapped safetensors file. Each `get()` call
/// reads raw bytes and constructs a CPU `DynTensor`. When the requested dtype
/// matches the stored dtype, bf16/f16 data loads natively (#1646 D2).
/// If the caller requests a GPU device, the tensor is transferred via
/// `DynTensor::to_device()`.
///
/// Supports F32, BF16, and F16 stored tensors.
/// Other dtypes (F64, I32, etc.) return `DTypeMismatch`.
pub struct SafeTensorsBackend {
    weight_map: WeightMap,
}

impl SafeTensorsBackend {
    /// Create from a loaded `WeightMap`.
    pub fn new(weight_map: WeightMap) -> Self {
        Self { weight_map }
    }
}

impl TensorBackend for SafeTensorsBackend {
    fn get(
        &self,
        dims: &[usize],
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> TensorResult<DynTensor> {
        let info = self
            .weight_map
            .tensor_info(name)
            .map_err(|_| TensorError::TensorNotFound {
                name: name.to_string(),
            })?;

        // Validate shape matches expected dims.
        if info.shape.as_slice() != dims {
            return Err(TensorError::shape_mismatch(
                dims.to_vec(),
                info.shape.clone(),
            ));
        }

        load_tensor_from_weight_map(
            &self.weight_map,
            name,
            &info.shape,
            info.dtype,
            dtype,
            device,
        )
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> TensorResult<DynTensor> {
        let info = self
            .weight_map
            .tensor_info(name)
            .map_err(|_| TensorError::TensorNotFound {
                name: name.to_string(),
            })?;

        load_tensor_from_weight_map(
            &self.weight_map,
            name,
            &info.shape,
            info.dtype,
            dtype,
            device,
        )
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.weight_map.tensor_info(name).is_ok()
    }

    fn tensor_names(&self) -> Vec<String> {
        self.weight_map.tensor_names().map(String::from).collect()
    }
}

/// Load a tensor from WeightMap bytes and construct DynTensor.
///
/// Supports F32, BF16, and F16 stored dtypes. When `requested_dtype` matches
/// the stored dtype, data is loaded natively without f32 conversion (D2 of
/// #1646). When `requested_dtype` differs, data is converted to the requested
/// dtype via f32 as intermediate.
///
/// Integer dtypes (U32, U8, I64) return `DTypeMismatch`.
pub(crate) fn load_tensor_from_weight_map(
    weight_map: &WeightMap,
    name: &str,
    shape: &[usize],
    stored_dtype: DType,
    requested_dtype: DType,
    device: &Device,
) -> TensorResult<DynTensor> {
    if !requested_dtype.is_float() {
        return Err(TensorError::dtype_mismatch(requested_dtype, DType::F32));
    }

    let data = weight_map
        .tensor_data(name)
        .map_err(|e| TensorError::Unsupported(e.to_string()))?;

    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorError::DimensionOverflow {
            dims: shape.to_vec(),
        })?;

    let bytes_per_elem = stored_dtype.size_bytes();
    let expected_bytes =
        numel
            .checked_mul(bytes_per_elem)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;

    if data.len() != expected_bytes {
        return Err(TensorError::DataLengthMismatch {
            expected: numel,
            actual: data.len() / bytes_per_elem,
        });
    }

    // Native loading: when requested dtype matches stored dtype, load directly
    // into native storage without f32 intermediate (#1646 D2).
    match (stored_dtype, requested_dtype) {
        (DType::BF16, DType::BF16) => {
            return load_bf16_native(data, name, shape, device);
        }
        (DType::F16, DType::F16) => {
            return load_f16_native(data, name, shape, device);
        }
        _ => {} // fall through to f32-intermediate path
    }

    // Fast path: F32 → F32 GPU — create Metal buffer directly from mmap bytes
    // without any intermediate Vec<f32> allocation. Saves ~2× weight-file-size
    // of peak RSS during loading (#3079). NaN check runs on mmap bytes in-place.
    if stored_dtype == DType::F32 && requested_dtype == DType::F32 && device.is_gpu() {
        return load_f32_gpu_direct(data, name, shape);
    }

    let f32_data = match stored_dtype {
        DType::F32 => {
            // Read f32 via chunks_exact to avoid alignment issues with mmap (#940).
            if data.len() % 4 != 0 {
                return Err(TensorError::Unsupported(format!(
                    "f32 data length {} not aligned to 4 bytes for tensor {name}",
                    data.len()
                )));
            }
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect::<Vec<f32>>()
        }
        DType::BF16 => {
            // Convert bf16 → f32 (requested dtype is F32 or F16, not BF16).
            data.chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect::<Vec<f32>>()
        }
        DType::F16 => {
            // Convert f16 → f32 (requested dtype is F32 or BF16, not F16).
            data.chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect::<Vec<f32>>()
        }
        other => {
            return Err(TensorError::dtype_mismatch(DType::F32, other));
        }
    };

    // Defense-in-depth: reject NaN/Inf in weight data at load time (#943).
    let bad = f32_data.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(TensorError::NonFiniteData {
            name: name.to_string(),
            count: bad,
        });
    }

    let t = DynTensor::new(&f32_data, shape, device)?;

    // If requested dtype is bf16/f16 but stored as f32, convert after loading.
    match requested_dtype {
        DType::BF16 | DType::F16 => t.to_dtype(requested_dtype),
        _ => Ok(t),
    }
}

/// Load bf16 data directly into native `FloatStorage::BF16` — no f32 intermediate.
fn load_bf16_native(
    data: &[u8],
    name: &str,
    shape: &[usize],
    device: &Device,
) -> TensorResult<DynTensor> {
    let bf16_vec: Vec<half::bf16> = data
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
        .collect();

    // Defense-in-depth: reject NaN/Inf (#943). Check via f32 conversion for
    // half types since bf16::is_finite() requires the check.
    let bad = bf16_vec.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(TensorError::NonFiniteData {
            name: name.to_string(),
            count: bad,
        });
    }

    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), bf16_vec)
        .map_err(|e| TensorError::InvalidShape(e.to_string()))?;
    let t = DynTensor::from_cpu_bf16(arr)?;
    if device.is_gpu() {
        t.to_device(device)
    } else {
        Ok(t)
    }
}

/// Load f16 data directly into native `FloatStorage::F16` — no f32 intermediate.
fn load_f16_native(
    data: &[u8],
    name: &str,
    shape: &[usize],
    device: &Device,
) -> TensorResult<DynTensor> {
    let f16_vec: Vec<half::f16> = data
        .chunks_exact(2)
        .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
        .collect();

    // Defense-in-depth: reject NaN/Inf (#943).
    let bad = f16_vec.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(TensorError::NonFiniteData {
            name: name.to_string(),
            count: bad,
        });
    }

    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), f16_vec)
        .map_err(|e| TensorError::InvalidShape(e.to_string()))?;
    let t = DynTensor::from_cpu_f16(arr)?;
    if device.is_gpu() {
        t.to_device(device)
    } else {
        Ok(t)
    }
}

/// Load F32 tensor directly to GPU from mmap bytes — zero intermediate allocation.
///
/// Validates bytes for NaN/Inf in-place (no `Vec<f32>` heap allocation), then
/// creates a Metal buffer directly from the raw mmap bytes. On little-endian
/// ARM64, the mmap bytes ARE valid f32 representations in native byte order.
///
/// Eliminates 2 intermediate `Vec<f32>` copies (one in `load_tensor_from_weight_map`
/// and one in `DynTensor::new`'s `.to_vec()`) — saves ~2× tensor size in peak RSS
/// during weight loading. For Kokoro-82M (~328 MB weights), this saves ~656 MB
/// peak RSS.
///
/// Part of #3079.
fn load_f32_gpu_direct(data: &[u8], name: &str, shape: &[usize]) -> TensorResult<DynTensor> {
    use crate::MetalTensorData;

    if !data.len().is_multiple_of(4) {
        return Err(TensorError::Unsupported(format!(
            "f32 data length {} not aligned to 4 bytes for tensor {name}",
            data.len()
        )));
    }

    // Defense-in-depth: reject NaN/Inf directly on mmap bytes (#943).
    // No heap allocation — iterates over mmap pages in-place.
    let bad = data
        .chunks_exact(4)
        .filter(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
        .count();
    if bad > 0 {
        return Err(TensorError::NonFiniteData {
            name: name.to_string(),
            count: bad,
        });
    }

    // Create Metal buffer directly from mmap bytes. On Apple Silicon (little-endian
    // ARM64), safetensors little-endian f32 bytes are valid native f32 representation.
    // Metal's newBufferWithData copies the bytes into GPU-accessible shared memory.
    // u8: bytemuck::NoUninit, so create_buffer accepts raw byte slices.
    let ctx = crate::metal_backend::global_metal_context().map_err(|e| {
        TensorError::backend_failure(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::DispatchFailed,
            e.to_string(),
        )
    })?;
    let buffer = ctx.create_buffer(data).map_err(|e| {
        TensorError::backend_failure(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::DispatchFailed,
            e.to_string(),
        )
    })?;

    let storage = MetalTensorData::new(buffer);
    DynTensor::from_gpu_storage(
        shape.to_vec(),
        DType::F32,
        Arc::new(storage),
        Device::metal(),
    )
}

// -- Sharded backend (extracted to var_builder_safetensors_sharded.rs, #1377) --

#[path = "var_builder_safetensors_sharded.rs"]
mod sharded;
use sharded::ShardedSafeTensorsBackend;

// -- Convenience constructors -------------------------------------------------

/// Create a `VarBuilder` backed by a safetensors `WeightMap`.
///
/// Matches candle's `VarBuilder::from_mmaped_safetensors` pattern.
/// The `WeightMap` must already be loaded (via `WeightMap::load()` or
/// `WeightMap::load_global()`).
///
/// # Example
///
/// ```no_run
/// # use nn_core::{DType, Device};
/// # use nn_metal::{WeightMap, SafeTensorsBackend};
/// # use nn_metal::var_builder_from_weight_map;
/// let wm = unsafe { WeightMap::load_global(std::path::Path::new("model.safetensors")).expect("load") };
/// let vb = var_builder_from_weight_map(wm, DType::F32, &Device::Cpu);
/// let weight = vb.pp("encoder").get(&[512, 256], "weight").expect("load weight");
/// ```
pub fn var_builder_from_weight_map(
    weight_map: WeightMap,
    dtype: DType,
    device: &Device,
) -> VarBuilder {
    VarBuilder::from_backend(
        Arc::new(SafeTensorsBackend::new(weight_map)),
        dtype,
        *device,
    )
}

// Mmap convenience constructors and MetalVarBuilderExt trait extracted to
// var_builder_safetensors_mmap.rs (#1572) to keep files under 500 lines.
#[path = "var_builder_safetensors_mmap.rs"]
mod mmap;
pub use mmap::{from_mmaped_safetensors, from_mmaped_safetensors_with_ctx, MetalVarBuilderExt};

#[cfg(test)]
#[path = "var_builder_safetensors_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "var_builder_safetensors_error_tests.rs"]
mod error_tests;
