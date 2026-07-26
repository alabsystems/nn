// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dynamic-rank tensor for imperative model code.
//!
//! [`DynTensor`] is the candle-compatible tensor type that enables ergonomic
//! model porting. Unlike [`Tensor<D, T, B>`](crate::Tensor) which has
//! compile-time rank, `DynTensor` determines rank at runtime — matching
//! candle's `Tensor` and enabling `reshape`, `squeeze`, `unsqueeze` to
//! return the same type.
//!
//! CPU operations use ndarray. GPU operations dispatch through the registered
//! [`GpuBackend`] trait (provided by nn-metal or other backend crates).
//!
//! ## Tracing infrastructure
//!
//! `DynTensor` supports computation graph tracing via [`trace::trace_graph`].
//! When tracing is active, operations record [`TraceOp`](trace::TraceOp)
//! nodes (99 variants) into a [`ComputationGraph`](trace::ComputationGraph).
//! The resulting graph feeds into nn-verify's `trace_to_graph` translator for
//! NY verification and nn-dsl's `trace_compile` for compiled model
//! dispatch.
//!
//! ## `dim` parameter ordering convention
//!
//! Methods use two conventions for the `dim` parameter position:
//!
//! **dim-first** (PyTorch convention, preferred for new methods):
//! `narrow`, `slice_set`, `scatter_add`, `index_add`, `topk`,
//! `repeat_interleave`, `pad_with_zeros`, `flatten`.
//!
//! **dim-last** (candle compatibility):
//! `index_select`, `gather`, `cat`, `stack`, `chunk`.
//!
//! New methods MUST use dim-first ordering unless matching an existing candle API.

use crate::tensor::checked_dim_product;
use crate::{DType, Device, Result, TensorError};
use std::any::Any;
use std::sync::Arc;

/// Candle-compatible dimension specifier for negative indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum D {
    /// Last dimension (Python's -1).
    Minus1,
    /// Second-to-last dimension (Python's -2).
    Minus2,
}

impl D {
    /// Resolve to a concrete axis index given a tensor rank.
    pub fn resolve(self, rank: usize) -> Result<usize> {
        let needed = match self {
            Self::Minus1 => 1,
            Self::Minus2 => 2,
        };
        if rank < needed {
            return Err(TensorError::RankMismatch {
                expected: needed,
                actual: rank,
            });
        }
        Ok(rank - needed)
    }
}

/// Type-erased tensor storage.
///
/// CPU storage holds an `Arc<ArrayD<T>>` behind `dyn Any`. GPU storage holds
/// backend-specific data (e.g., `MetalTensorData`) behind `dyn Any`, plus
/// the device it lives on. Quantized storage holds block-quantized bytes
/// (Q4_0/Q4_1/Q8_0) that dequantize to f32 on demand.
#[derive(Clone)]
pub(crate) enum TensorStorage {
    Cpu(Arc<dyn Any + Send + Sync>),
    Gpu {
        data: Arc<dyn Any + Send + Sync>,
        device: Device,
    },
    /// Block-quantized storage (GGUF/GGML formats). Dequantizes to f32 on
    /// demand when arithmetic operations are performed.
    Quantized(Arc<QuantizedStorage>),
}

pub(crate) mod gpu;
pub(crate) use gpu::{gpu_backend, gpu_backend_dispatch, gpu_backend_dispatch_pair};
pub use gpu::{
    gpu_backend_flush, register_gpu_backend, BinaryOp, CompareOp, GpuBackend, GpuFullBackend,
    GpuNnOps, GpuSelectionOps, GpuShapeOps, ReduceOp, UnaryOp,
};

#[doc(hidden)]
pub mod safetensors_write;
pub use safetensors_write::{
    load_safetensors, load_safetensors_from_bytes, save_safetensors, tensors_to_safetensors_bytes,
};

// -- DynTensor ----------------------------------------------------------------

/// Dynamic-rank tensor for imperative model code.
///
/// Owns its shape and dtype at the top level. Storage holds only the data
/// buffer. This matches candle's `Tensor` type, enabling find-and-replace
/// migration from candle to nn.
///
/// # Device transparency
///
/// Operations check the storage variant and dispatch to CPU (ndarray) or
/// GPU (registered backend) automatically. Mixed-device operations return
/// `DeviceMismatch` errors.
#[derive(Clone)]
pub struct DynTensor {
    dims: Vec<usize>,
    dtype: DType,
    pub(crate) storage: TensorStorage,
    /// Trace node ID when this tensor was created during graph tracing.
    /// `None` when tracing is inactive (the common case).
    pub(crate) trace_node_id: Option<trace::NodeId>,
}

// Compile-time assertion: DynTensor must be Send + Sync to support
// Arc<SharedModel> patterns in multi-threaded inference (#1952).
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<DynTensor>();
    }
};

impl DynTensor {
    // -- Shape queries --------------------------------------------------------

    /// Dimension sizes as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the single dimension of a 1-D tensor.
    pub fn dims1(&self) -> Result<usize> {
        if self.dims.len() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: self.dims.len(),
            });
        }
        Ok(self.dims[0])
    }

    /// Get dimensions of a 2-D tensor.
    pub fn dims2(&self) -> Result<(usize, usize)> {
        if self.dims.len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.dims.len(),
            });
        }
        Ok((self.dims[0], self.dims[1]))
    }

    /// Get dimensions of a 3-D tensor.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        if self.dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: self.dims.len(),
            });
        }
        Ok((self.dims[0], self.dims[1], self.dims[2]))
    }

    /// Get dimensions of a 4-D tensor.
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        if self.dims.len() != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: self.dims.len(),
            });
        }
        Ok((self.dims[0], self.dims[1], self.dims[2], self.dims[3]))
    }

    /// Get dimensions of a 5-D tensor.
    pub fn dims5(&self) -> Result<(usize, usize, usize, usize, usize)> {
        if self.dims.len() != 5 {
            return Err(TensorError::RankMismatch {
                expected: 5,
                actual: self.dims.len(),
            });
        }
        Ok((
            self.dims[0],
            self.dims[1],
            self.dims[2],
            self.dims[3],
            self.dims[4],
        ))
    }

    /// Size of a specific dimension.
    ///
    /// Accepts both `usize` and [`D`] (e.g., `D::Minus1`).
    pub fn dim(&self, d: impl Dim) -> Result<usize> {
        let idx = d.to_index(self.rank())?;
        self.dims.get(idx).copied().ok_or_else(|| {
            TensorError::InvalidShape(format!(
                "dimension {idx} out of range for rank {}",
                self.dims.len()
            ))
        })
    }

    /// Number of dimensions (rank).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Element data type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device this tensor lives on.
    ///
    /// Quantized tensors are always CPU-resident (dequantize before GPU transfer).
    #[must_use]
    pub fn device(&self) -> Device {
        match &self.storage {
            TensorStorage::Cpu(_) | TensorStorage::Quantized(_) => Device::Cpu,
            TensorStorage::Gpu { device, .. } => *device,
        }
    }

    /// Total number of elements (checked arithmetic).
    ///
    /// Returns an error if the dimension product overflows `usize`.
    /// Callers performing allocation should use this method.
    pub fn checked_numel(&self) -> Result<usize> {
        checked_dim_product(&self.dims)
    }

    /// Total number of elements.
    ///
    /// # Note
    ///
    /// Saturates to `usize::MAX` on overflow instead of panicking or wrapping.
    /// Prefer [`checked_numel()`](Self::checked_numel) for allocation-size
    /// calculations — it returns `Err` on overflow rather than a sentinel value.
    #[must_use]
    pub fn numel(&self) -> usize {
        checked_dim_product(&self.dims).unwrap_or(usize::MAX)
    }

    // Storage accessors and GPU helpers extracted to dyn_tensor_storage.rs.
}

#[path = "dyn_tensor_storage.rs"]
mod dyn_tensor_storage;

/// Convert f64 to f32, returning an error if the value is finite in f64
/// but overflows to infinity in f32.
///
/// This guards against silent data corruption when user-supplied f64
/// parameters (e.g., clamp bounds, ELU alpha, exponent) overflow the
/// f32 range. NaN and infinity inputs pass through unchanged — only
/// finite→non-finite transitions are rejected.
pub(crate) fn checked_f64_to_f32(val: f64, param_name: &str) -> Result<f32> {
    let val_f32 = val as f32;
    if !val_f32.is_finite() && val.is_finite() {
        return Err(TensorError::InvalidBounds(format!(
            "{param_name}: value {val} overflows f32 (becomes {val_f32})"
        )));
    }
    Ok(val_f32)
}

impl std::fmt::Debug for DynTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTensor")
            .field("dims", &self.dims)
            .field("dtype", &self.dtype)
            .field("device", &self.device())
            .field("numel", &self.numel())
            .finish_non_exhaustive()
    }
}

/// Dispatch a CPU tensor operation across all supported dtypes (f32, f16, bf16, u32, u8, i64).
///
/// Extracts `TensorStorage::Cpu` (returning `Unsupported` on GPU), then tries
/// each dtype via `downcast_ref`. The `$label` parameter customises the error
/// message (e.g. `"narrow"`, `"index_select"`).
///
/// Float storage is handled via [`FloatStorage`] (native f32/f16/bf16),
/// `ArcArray<f16/bf16, IxDyn>` (from `narrow_half_zero_copy`, #1856),
/// and legacy `ArcArray<f32, IxDyn>` / `ArrayD<f32>` paths.
macro_rules! dispatch_cpu_typed {
    ($self:expr, $op:expr, $label:expr) => {{
        // Auto-dequantize quantized tensors before CPU dispatch.
        // Dequantize produces a CPU f32 tensor, so we apply the op directly
        // to the f32 array without recursing through the macro.
        if let TensorStorage::Quantized(ref qs) = $self.storage {
            let arr = qs.dequantize()?;
            let result = ($op)(&arr)?;
            return DynTensor::from_cpu_f32(result);
        }
        let any = match &$self.storage {
            TensorStorage::Cpu(a) => a,
            TensorStorage::Gpu { .. } => {
                return Err(TensorError::Unsupported(
                    concat!($label, " called on GPU tensor").into(),
                ));
            }
            TensorStorage::Quantized(_) => unreachable!("handled above"),
        };
        // Integer types first (unchanged).
        // Use fully-qualified ndarray types for macro hygiene — callers
        // should not need `use ndarray::{ArrayD, ArcArray, IxDyn}` for
        // this macro to work (#1968 AC1).
        if let Some(arr) = any.downcast_ref::<ndarray::ArrayD<u32>>() {
            let result = ($op)(arr)?;
            return DynTensor::from_cpu_u32(result);
        }
        if let Some(arr) = any.downcast_ref::<ndarray::ArrayD<u8>>() {
            let result = ($op)(arr)?;
            return DynTensor::from_cpu_u8(result);
        }
        if let Some(arr) = any.downcast_ref::<ndarray::ArrayD<i64>>() {
            let result = ($op)(arr)?;
            return DynTensor::from_cpu_i64(result);
        }
        // FloatStorage (new native path for f32/f16/bf16).
        if let Some(fs) = any.downcast_ref::<$crate::dyn_tensor::float_storage::FloatStorage>() {
            match fs {
                $crate::dyn_tensor::float_storage::FloatStorage::F32(arr) => {
                    let result = ($op)(arr)?;
                    return DynTensor::from_cpu_f32(result);
                }
                $crate::dyn_tensor::float_storage::FloatStorage::F16(arr) => {
                    let result = ($op)(arr)?;
                    return DynTensor::from_cpu_f16(result);
                }
                $crate::dyn_tensor::float_storage::FloatStorage::BF16(arr) => {
                    let result = ($op)(arr)?;
                    return DynTensor::from_cpu_bf16(result);
                }
            }
        }
        // Legacy f32 paths (ArcArray from zero-copy narrow, ArrayD from old constructors).
        if let Some(arc_arr) = any.downcast_ref::<ndarray::ArcArray<f32, ndarray::IxDyn>>() {
            let owned: ndarray::ArrayD<f32> = arc_arr.to_owned();
            let result = ($op)(&owned)?;
            return DynTensor::from_cpu_f32(result);
        }
        // ArcArray<f16/bf16> from narrow_half_zero_copy (#1856).
        if let Some(arc_arr) = any.downcast_ref::<ndarray::ArcArray<half::f16, ndarray::IxDyn>>() {
            let owned: ndarray::ArrayD<half::f16> = arc_arr.to_owned();
            let result = ($op)(&owned)?;
            return DynTensor::from_cpu_f16(result);
        }
        if let Some(arc_arr) = any.downcast_ref::<ndarray::ArcArray<half::bf16, ndarray::IxDyn>>() {
            let owned: ndarray::ArrayD<half::bf16> = arc_arr.to_owned();
            let result = ($op)(&owned)?;
            return DynTensor::from_cpu_bf16(result);
        }
        let arr = any
            .downcast_ref::<ndarray::ArrayD<f32>>()
            .ok_or(TensorError::dtype_mismatch(DType::F32, $self.dtype))?;
        let result = ($op)(arr)?;
        DynTensor::from_cpu_f32(result)
    }};
}

pub(crate) mod float_storage;
pub(crate) use float_storage::FloatStorage;

pub mod quantized;
pub use quantized::{QuantType, QuantizedStorage};

pub mod quantized_matmul;
pub use quantized_matmul::{
    quantized_linear, quantized_matmul_q4_0, quantized_matmul_q8_0, QuantizedMatmulError,
};

pub mod trace;

mod shape_type;
pub use shape_type::Shape;
mod constructors;
#[doc(hidden)]
pub mod indexing;
#[doc(hidden)]
pub mod ops;
mod selection;
mod shape;
mod typed_storage;
pub use indexing::{IndexOp, TensorIndexer};
pub(crate) mod conv;
pub use conv::{
    conv1d_out_len, conv2d_out_len, conv3d_out_len, conv_transpose1d_out_len,
    conv_transpose2d_out_len, Conv1dParams, Conv2dParams, Conv3dParams, ConvTranspose1dParams,
    ConvTranspose2dParams,
};
mod ops_ext;
#[allow(deprecated)]
pub use ops::softmax_last_dim;
pub use ops::{einsum, EinsumNotation};
pub use ops_ext::GridSamplePaddingMode;
mod accessors;
mod with_dtype;
pub use with_dtype::WithDType;
pub mod dim;
mod softmax;
pub use dim::Dim;
#[cfg(test)]
#[path = "broadcast_edge_tests.rs"]
mod broadcast_edge_tests;
#[cfg(test)]
mod broadcast_tests;
#[cfg(test)]
mod candle_compat_tests;
mod convenience;
#[cfg(test)]
mod convenience_tests;
#[cfg(test)]
mod convenience_tests_dim;
#[cfg(test)]
mod convenience_tests_dim_reduction;
#[cfg(test)]
#[path = "creation_shape_tests.rs"]
mod creation_shape_tests;
#[cfg(test)]
mod dvoice_compat_tests;
#[cfg(test)]
#[path = "dtype_conversion_tests.rs"]
mod dtype_conversion_tests;
#[cfg(kani)]
#[path = "kani_broadcast_proofs.rs"]
mod kani_broadcast_proofs;
#[cfg(kani)]
#[path = "kani_cat_split_dpdf_extended.rs"]
mod kani_cat_split_dpdf_extended;
#[cfg(kani)]
#[path = "kani_cat_split_proofs.rs"]
mod kani_cat_split_proofs;
#[cfg(kani)]
#[path = "kani_constructor_accessor_proofs.rs"]
mod kani_constructor_accessor_proofs;
#[cfg(kani)]
#[path = "kani_dtype_cast_proofs.rs"]
mod kani_dtype_cast_proofs;
#[cfg(kani)]
#[path = "kani_dtype_conversion_proofs.rs"]
mod kani_dtype_conversion_proofs;
#[cfg(kani)]
#[path = "kani_dyn_tensor_ops_proofs.rs"]
mod kani_dyn_tensor_ops_proofs;
#[cfg(kani)]
#[path = "kani_float_storage_proofs.rs"]
mod kani_float_storage_proofs;
#[cfg(kani)]
#[path = "kani_gpu_dispatch_proofs.rs"]
mod kani_gpu_dispatch_proofs;
#[cfg(kani)]
#[path = "kani_indexing_proofs.rs"]
mod kani_indexing_proofs;
#[cfg(kani)]
#[path = "kani_ops_proofs.rs"]
mod kani_ops_proofs;
#[cfg(kani)]
#[path = "kani_ops_proofs_ext.rs"]
mod kani_ops_proofs_ext;
#[cfg(kani)]
#[path = "kani_shape_ops_proofs.rs"]
mod kani_shape_ops_proofs;
#[cfg(kani)]
mod kani_shape_proofs;
#[cfg(test)]
mod ops_edge_case_tests;
#[cfg(test)]
mod quantized_tests;
#[cfg(feature = "training")]
mod random;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_helpers;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_creation;
#[cfg(test)]
mod tests_reshape_advanced;
#[cfg(test)]
mod with_dtype_tests;
