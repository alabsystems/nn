// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Typed storage accessors and GPU helpers for [`DynTensor`].
//!
//! Extracted from `mod.rs` for 500-line compliance.

use crate::tensor::checked_dim_product;
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArcArray, ArrayD, IxDyn};
use std::any::Any;
use std::sync::Arc;

use super::{DynTensor, FloatStorage, Shape, TensorStorage};

impl DynTensor {
    /// Get the underlying CPU f32 ndarray view. Returns error if not CPU f32.
    ///
    /// This is the zero-copy path for consumers (e.g. NY) that store
    /// data as `ndarray::ArrayD<f32>`. No allocation or copy occurs.
    ///
    /// Handles `FloatStorage::F32`, `ArcArray<f32, IxDyn>` (from zero-copy narrow),
    /// and `ArrayD<f32>` (legacy constructors) transparently.
    ///
    /// For f16/bf16 tensors, use [`to_f32_array()`](Self::to_f32_array) which
    /// converts on demand, or the dtype-specific [`as_cpu_f16()`](Self::as_cpu_f16)
    /// / [`as_cpu_bf16()`](Self::as_cpu_bf16) accessors.
    pub fn as_cpu_f32(&self) -> Result<ndarray::ArrayViewD<'_, f32>> {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                // Try FloatStorage first (new native path).
                if let Some(fs) = any.downcast_ref::<FloatStorage>() {
                    return fs.as_f32_view();
                }
                // Try ArcArray (produced by zero-copy narrow).
                if let Some(arc_arr) = any.downcast_ref::<ArcArray<f32, IxDyn>>() {
                    return Ok(arc_arr.view());
                }
                // Fall back to ArrayD (produced by legacy constructors).
                let arr = any
                    .downcast_ref::<ArrayD<f32>>()
                    .ok_or(TensorError::dtype_mismatch(DType::F32, self.dtype))?;
                Ok(arr.view())
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            TensorStorage::Quantized(_) => Err(TensorError::Unsupported(
                "as_cpu_f32 on quantized tensor — call .dequantize() first".into(),
            )),
        }
    }

    /// Get the underlying CPU f16 ndarray view. Returns error if not CPU f16.
    ///
    /// Handles both `FloatStorage::F16` and `ArcArray<f16>` storage
    /// (the latter from `narrow_half_zero_copy`, #1856).
    pub fn as_cpu_f16(&self) -> Result<ndarray::ArrayViewD<'_, half::f16>> {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                // Try ArcArray<f16> first (from narrow_half_zero_copy).
                if let Some(arc_arr) = any.downcast_ref::<ArcArray<half::f16, IxDyn>>() {
                    return Ok(arc_arr.view());
                }
                let fs = any
                    .downcast_ref::<FloatStorage>()
                    .ok_or(TensorError::dtype_mismatch(DType::F16, self.dtype))?;
                fs.as_f16_view()
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            TensorStorage::Quantized(_) => Err(TensorError::Unsupported(
                "as_cpu_f16 on quantized tensor — call .dequantize() first".into(),
            )),
        }
    }

    /// Get the underlying CPU bf16 ndarray view. Returns error if not CPU bf16.
    ///
    /// Handles both `FloatStorage::BF16` and `ArcArray<bf16>` storage
    /// (the latter from `narrow_half_zero_copy`, #1856).
    pub fn as_cpu_bf16(&self) -> Result<ndarray::ArrayViewD<'_, half::bf16>> {
        match &self.storage {
            TensorStorage::Cpu(any) => {
                // Try ArcArray<bf16> first (from narrow_half_zero_copy).
                if let Some(arc_arr) = any.downcast_ref::<ArcArray<half::bf16, IxDyn>>() {
                    return Ok(arc_arr.view());
                }
                let fs = any
                    .downcast_ref::<FloatStorage>()
                    .ok_or(TensorError::dtype_mismatch(DType::BF16, self.dtype))?;
                fs.as_bf16_view()
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            TensorStorage::Quantized(_) => Err(TensorError::Unsupported(
                "as_cpu_bf16 on quantized tensor — call .dequantize() first".into(),
            )),
        }
    }

    /// Convert any float tensor to an owned `ArrayD<f32>`.
    ///
    /// GPU tensors are automatically transferred to CPU first (matching
    /// the pattern in `to_vec1`, `to_flat_vec`, etc.). Clones for F32
    /// storage (O(n) copy). Allocates and converts element-wise for
    /// f16/bf16. For zero-copy F32 access, use `as_cpu_f32()` which
    /// returns a borrowed view but fails on f16/bf16 and GPU tensors.
    pub fn to_f32_array(&self) -> Result<ArrayD<f32>> {
        if self.device().is_gpu() {
            return self.to_device(&Device::Cpu)?.to_f32_array();
        }
        match &self.storage {
            TensorStorage::Cpu(any) => {
                // Try FloatStorage first (new native path).
                if let Some(fs) = any.downcast_ref::<FloatStorage>() {
                    return Ok(fs.to_f32_array());
                }
                // Try ArcArray<f32> (from zero-copy narrow).
                if let Some(arc_arr) = any.downcast_ref::<ArcArray<f32, IxDyn>>() {
                    return Ok(arc_arr.to_owned());
                }
                // Try ArcArray<f16/bf16> (from narrow_half_zero_copy, #1856).
                if let Some(arc_arr) = any.downcast_ref::<ArcArray<half::f16, IxDyn>>() {
                    return Ok(arc_arr.mapv(half::f16::to_f32));
                }
                if let Some(arc_arr) = any.downcast_ref::<ArcArray<half::bf16, IxDyn>>() {
                    return Ok(arc_arr.mapv(half::bf16::to_f32));
                }
                // Try ArrayD (legacy constructors).
                let arr = any
                    .downcast_ref::<ArrayD<f32>>()
                    .ok_or(TensorError::dtype_mismatch(DType::F32, self.dtype))?;
                Ok(arr.clone())
            }
            TensorStorage::Gpu { .. } => Err(TensorError::Unsupported(
                "CPU operation on GPU tensor — call .to_device(&Device::Cpu) first".into(),
            )),
            // Auto-dequantize: expand compressed blocks to f32.
            TensorStorage::Quantized(qs) => qs.dequantize(),
        }
    }

    /// Create a DynTensor from an owned CPU f32 ndarray (zero-copy).
    ///
    /// The array is consumed, converted to `ArcArray` (shared-backing for
    /// zero-copy narrow), and wrapped in `Arc` — no data copy occurs.
    /// This is the zero-copy path for consumers (e.g. NY) that
    /// produce `ndarray::ArrayD<f32>` and need DynTensor for GPU dispatch.
    pub fn from_cpu_f32(arr: ArrayD<f32>) -> Result<Self> {
        let dims = arr.shape().to_vec();
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype: DType::F32,
            storage: TensorStorage::Cpu(Arc::new(arr.into_shared())),
            trace_node_id: None,
        })
    }

    /// Create a DynTensor from an owned CPU f16 ndarray.
    ///
    /// Stores natively as `FloatStorage::F16` — no f32 conversion.
    pub fn from_cpu_f16(arr: ArrayD<half::f16>) -> Result<Self> {
        let dims = arr.shape().to_vec();
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype: DType::F16,
            storage: TensorStorage::Cpu(Arc::new(FloatStorage::F16(arr))),
            trace_node_id: None,
        })
    }

    /// Create a DynTensor from an owned CPU bf16 ndarray.
    ///
    /// Stores natively as `FloatStorage::BF16` — no f32 conversion.
    pub fn from_cpu_bf16(arr: ArrayD<half::bf16>) -> Result<Self> {
        let dims = arr.shape().to_vec();
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype: DType::BF16,
            storage: TensorStorage::Cpu(Arc::new(FloatStorage::BF16(arr))),
            trace_node_id: None,
        })
    }

    /// Create a tensor from a flat f16 slice with explicit dimensions.
    ///
    /// Stores natively as `FloatStorage::F16` — no f32 conversion.
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`].
    pub fn from_vec_f16(
        data: Vec<half::f16>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<Self> {
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
        let t = Self::from_cpu_f16(arr)?;
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a tensor from a flat bf16 slice with explicit dimensions.
    ///
    /// Stores natively as `FloatStorage::BF16` — no f32 conversion.
    /// Accepts `&[usize]`, tuples, `Vec<usize>`, or [`Shape`].
    pub fn from_vec_bf16(
        data: Vec<half::bf16>,
        dims: impl Into<Shape>,
        device: &Device,
    ) -> Result<Self> {
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
        let t = Self::from_cpu_bf16(arr)?;
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a DynTensor from an f32 computation result, converting to the
    /// target dtype if needed. Used by ops that promote bf16/f16 to f32 for
    /// computation and need to convert the result back (#1646 D3).
    ///
    /// For F32 target: equivalent to `from_cpu_f32()` (zero overhead).
    /// For F16/BF16 target: converts via `FloatStorage::from_f32_array()`.
    pub fn from_f32_result(arr: ArrayD<f32>, target_dtype: DType) -> Result<Self> {
        match target_dtype {
            DType::F16 | DType::BF16 => {
                let dims = arr.shape().to_vec();
                checked_dim_product(&dims)?;
                let fs = FloatStorage::from_f32_array(arr, target_dtype);
                Ok(Self {
                    dims,
                    dtype: target_dtype,
                    storage: TensorStorage::Cpu(Arc::new(fs)),
                    trace_node_id: None,
                })
            }
            // F32 and F64 are valid float targets — store as f32 (F64 is
            // downcast, matching the "all float data is f32 internally"
            // invariant documented in design doc).
            DType::F32 | DType::F64 => Self::from_cpu_f32(arr),
            // Integer and boolean targets are caller bugs — f32 computation
            // results should never be labeled as integer/bool tensors.
            DType::I32 | DType::I64 | DType::U32 | DType::U8 | DType::Bool => {
                Err(TensorError::dtype_mismatch(DType::F32, target_dtype))
            }
        }
    }

    /// Construct from raw parts (for backend crates building GPU tensors).
    pub fn from_gpu_storage(
        dims: Vec<usize>,
        dtype: DType,
        data: Arc<dyn Any + Send + Sync>,
        device: Device,
    ) -> Result<Self> {
        checked_dim_product(&dims)?;
        Ok(Self {
            dims,
            dtype,
            storage: TensorStorage::Gpu { data, device },
            trace_node_id: None,
        })
    }

    /// Create a GPU tensor that shares the same buffer but with a different
    /// dtype label. Zero-copy — no data movement.
    ///
    /// # Safety invariant
    ///
    /// Caller must ensure `new_dtype` has the same GPU buffer byte width as
    /// `self.dtype`. BF16↔F16 (both 2-byte Metal `half`) is safe. F32↔BF16
    /// or F32↔F16 (4-byte vs 2-byte) would cause dispatch to misinterpret
    /// buffer data — use CPU round-trip for those conversions instead.
    ///
    /// The `to_dtype()` implementation enforces this via `same_gpu_byte_width()`.
    pub(crate) fn gpu_relabel_dtype(&self, new_dtype: DType) -> Result<Self> {
        match &self.storage {
            TensorStorage::Gpu { data, device } => Ok(Self {
                dims: self.dims.clone(),
                dtype: new_dtype,
                storage: TensorStorage::Gpu {
                    data: Arc::clone(data),
                    device: *device,
                },
                trace_node_id: None,
            }),
            TensorStorage::Cpu(_) | TensorStorage::Quantized(_) => Err(TensorError::Unsupported(
                "gpu_relabel_dtype called on non-GPU tensor".into(),
            )),
        }
    }

    /// Downcast GPU storage to a concrete type. Used by backend crates.
    pub fn gpu_data<T: 'static>(&self) -> Result<&T> {
        match &self.storage {
            TensorStorage::Gpu { data, .. } => data
                .downcast_ref::<T>()
                .ok_or_else(|| TensorError::Unsupported("GPU storage type mismatch".into())),
            TensorStorage::Cpu(_) | TensorStorage::Quantized(_) => Err(TensorError::Unsupported(
                "gpu_data called on non-GPU tensor".into(),
            )),
        }
    }
}
