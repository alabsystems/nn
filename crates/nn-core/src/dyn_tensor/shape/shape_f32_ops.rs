// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Float-specific shape operations: zero-copy narrow and in-place slice_set.
//!
//! These functions handle `ArcArray<f32, IxDyn>`, `ArrayD<f32>`, and
//! `FloatStorage` storage variants for copy-on-write semantics in KV
//! cache operations. BF16/F16 zero-copy narrow is supported via
//! `narrow_half_zero_copy` (#1856).

use super::{DynTensor, TensorStorage};
use crate::dyn_tensor::FloatStorage;
use crate::{DType, Result, TensorError};
use half::{bf16, f16};
use ndarray::{ArcArray, ArrayD, Axis, IxDyn, Slice};
use std::any::Any;
use std::sync::Arc;

/// Shared helper: zero-copy slice on an `ArrayD<f32>` reference.
fn slice_array_d(arr: &ArrayD<f32>, dim: usize, start: usize, len: usize) -> DynTensor {
    let shared: ArcArray<f32, IxDyn> = arr.to_shared();
    let sliced = shared.slice_axis_move(
        Axis(dim),
        Slice::from(start as isize..(start + len) as isize),
    );
    let dims = sliced.shape().to_vec();
    DynTensor {
        dims,
        dtype: DType::F32,
        storage: TensorStorage::Cpu(Arc::new(sliced)),
        trace_node_id: None,
    }
}

/// Zero-copy narrow for f32 tensors via ArcArray shared-backing slice.
///
/// Converts `ArrayD<f32>`, `ArcArray<f32, IxDyn>`, or `FloatStorage::F32`
/// storage into an `ArcArray` view that shares the parent's data allocation.
/// `slice_axis_move` adjusts offset and strides without copying any elements.
///
/// Returns `Ok(None)` if the storage is not f32 (caller should use fallback).
pub(super) fn narrow_f32_zero_copy(
    any: &Arc<dyn Any + Send + Sync>,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<Option<DynTensor>> {
    // Try existing ArcArray storage (from a previous narrow).
    if let Some(arc_arr) = any.downcast_ref::<ArcArray<f32, IxDyn>>() {
        let sliced = arc_arr.clone().slice_axis_move(
            Axis(dim),
            Slice::from(start as isize..(start + len) as isize),
        );
        let dims = sliced.shape().to_vec();
        return Ok(Some(DynTensor {
            dims,
            dtype: DType::F32,
            storage: TensorStorage::Cpu(Arc::new(sliced)),
            trace_node_id: None,
        }));
    }
    // Try ArrayD storage (from legacy constructors). Convert to ArcArray first.
    if let Some(arr) = any.downcast_ref::<ArrayD<f32>>() {
        return Ok(Some(slice_array_d(arr, dim, start, len)));
    }
    // Try FloatStorage::F32 (from new native constructors: zeros/ones/full).
    if let Some(FloatStorage::F32(arr)) = any.downcast_ref::<FloatStorage>() {
        return Ok(Some(slice_array_d(arr, dim, start, len)));
    }
    // Not f32 (FloatStorage::F16/BF16, or integer types).
    Ok(None)
}

/// Zero-copy narrow for f16/bf16 tensors stored in `FloatStorage` (#1856).
///
/// Converts `FloatStorage::F16`/`FloatStorage::BF16` into an `ArcArray` view
/// that shares the parent's data allocation via `slice_axis_move`. Same
/// zero-copy semantics as `narrow_f32_zero_copy`.
///
/// Also handles `ArcArray<f16>` and `ArcArray<bf16>` storage produced by a
/// prior `narrow_half_zero_copy` call (chained narrows stay zero-copy).
///
/// Returns `Ok(None)` if the storage is not a half-precision type.
pub(super) fn narrow_half_zero_copy(
    any: &Arc<dyn Any + Send + Sync>,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<Option<DynTensor>> {
    let slice = Slice::from(start as isize..(start + len) as isize);

    // Try existing ArcArray<f16> (from a previous narrow_half_zero_copy).
    if let Some(arc_arr) = any.downcast_ref::<ArcArray<f16, IxDyn>>() {
        let sliced = arc_arr.clone().slice_axis_move(Axis(dim), slice);
        let dims = sliced.shape().to_vec();
        return Ok(Some(DynTensor {
            dims,
            dtype: DType::F16,
            storage: TensorStorage::Cpu(Arc::new(sliced)),
            trace_node_id: None,
        }));
    }
    // Try existing ArcArray<bf16> (from a previous narrow_half_zero_copy).
    if let Some(arc_arr) = any.downcast_ref::<ArcArray<bf16, IxDyn>>() {
        let sliced = arc_arr.clone().slice_axis_move(Axis(dim), slice);
        let dims = sliced.shape().to_vec();
        return Ok(Some(DynTensor {
            dims,
            dtype: DType::BF16,
            storage: TensorStorage::Cpu(Arc::new(sliced)),
            trace_node_id: None,
        }));
    }
    // Try FloatStorage::F16/BF16 (from constructors: zeros/ones/full/from_raw).
    match any.downcast_ref::<FloatStorage>() {
        Some(FloatStorage::F16(arr)) => {
            let shared: ArcArray<f16, IxDyn> = arr.to_shared();
            let sliced = shared.slice_axis_move(Axis(dim), slice);
            let dims = sliced.shape().to_vec();
            Ok(Some(DynTensor {
                dims,
                dtype: DType::F16,
                storage: TensorStorage::Cpu(Arc::new(sliced)),
                trace_node_id: None,
            }))
        }
        Some(FloatStorage::BF16(arr)) => {
            let shared: ArcArray<bf16, IxDyn> = arr.to_shared();
            let sliced = shared.slice_axis_move(Axis(dim), slice);
            let dims = sliced.shape().to_vec();
            Ok(Some(DynTensor {
                dims,
                dtype: DType::BF16,
                storage: TensorStorage::Cpu(Arc::new(sliced)),
                trace_node_id: None,
            }))
        }
        _ => Ok(None), // F32 handled by narrow_f32_zero_copy; integer types use fallback.
    }
}

/// Perform slice_set for f32 storage, handling both ArcArray and ArrayD backing.
///
/// ArcArray backing arises from zero-copy `narrow()`. ArrayD backing arises from
/// constructors. Both are mutated via `slice_mut().assign()`, then stored back
/// as `ArcArray` (shared-backing for future narrow operations).
///
/// When the outer `Arc` has refcount=1 (sole owner) AND the inner `ArcArray`
/// has refcount=1 (no outstanding narrow views), the mutation is truly in-place:
/// only the slice region is written, not the full buffer. This makes KV cache
/// `append()` O(write_region) per step instead of O(buffer_size).
pub(super) fn slice_set_f32(
    storage_arc: &mut Arc<dyn Any + Send + Sync>,
    src: &DynTensor,
    slice_info: &[ndarray::SliceInfoElem],
    dtype: DType,
) -> Result<()> {
    let placeholder: ArcArray<f32, IxDyn> = ArrayD::<f32>::zeros(IxDyn(&[])).into_shared();
    let taken = std::mem::replace(storage_arc, Arc::new(placeholder));
    let src_view = src
        .as_cpu_f32()
        .map_err(|_| TensorError::dtype_mismatch(dtype, src.dtype()))?;
    // Check type before consuming `taken` (Arc::downcast consumes the Arc).
    let is_arc_array = (*taken).is::<ArcArray<f32, IxDyn>>();
    let is_array_d = (*taken).is::<ArrayD<f32>>();
    let is_float_storage = (*taken).is::<FloatStorage>();

    if is_arc_array {
        let concrete = taken.downcast::<ArcArray<f32, IxDyn>>().map_err(|_| {
            TensorError::InvalidShape("slice_set_f32: downcast to ArcArray failed".into())
        })?;
        // Arc::try_unwrap avoids cloning the outer Arc when refcount=1.
        let arc_arr = match Arc::try_unwrap(concrete) {
            Ok(owned) => owned,
            Err(shared) => shared.as_ref().clone(),
        };
        // ArcArray::into_owned() is O(1) when the ArcArray's internal
        // refcount is 1 (no outstanding narrow views sharing the backing
        // data). When refcount > 1, it copies — but that only happens
        // when a narrow view is still alive, which is correct (COW).
        let mut arr = arc_arr.into_owned();
        arr.slice_mut(slice_info).assign(&src_view);
        *storage_arc = Arc::new(arr.into_shared());
    } else if is_array_d {
        let concrete = taken.downcast::<ArrayD<f32>>().map_err(|_| {
            TensorError::InvalidShape("slice_set_f32: downcast to ArrayD failed".into())
        })?;
        let mut arr = match Arc::try_unwrap(concrete) {
            Ok(owned) => owned,
            Err(shared) => shared.as_ref().clone(),
        };
        arr.slice_mut(slice_info).assign(&src_view);
        *storage_arc = Arc::new(arr.into_shared());
    } else if is_float_storage {
        // FloatStorage::F32 from new native constructors (zeros/ones/full).
        let concrete = taken.downcast::<FloatStorage>().map_err(|_| {
            TensorError::InvalidShape("slice_set_f32: downcast to FloatStorage failed".into())
        })?;
        let fs = match Arc::try_unwrap(concrete) {
            Ok(owned) => owned,
            Err(shared) => shared.as_ref().clone(),
        };
        match fs {
            FloatStorage::F32(mut arr) => {
                arr.slice_mut(slice_info).assign(&src_view);
                *storage_arc = Arc::new(arr.into_shared());
            }
            other => {
                return Err(TensorError::dtype_mismatch(DType::F32, other.dtype()));
            }
        }
    } else {
        return Err(TensorError::dtype_mismatch(dtype, dtype));
    }
    Ok(())
}

/// Perform slice_set for f16/bf16 tensors.
///
/// Handles both `FloatStorage` and `ArcArray<f16/bf16>` storage types (the
/// latter from `narrow_half_zero_copy`, #1856). The source tensor must have
/// the same dtype.
pub(super) fn slice_set_half(
    storage_arc: &mut Arc<dyn Any + Send + Sync>,
    src: &DynTensor,
    slice_info: &[ndarray::SliceInfoElem],
    dtype: DType,
) -> Result<()> {
    let placeholder: FloatStorage = FloatStorage::F32(ArrayD::<f32>::zeros(IxDyn(&[])));
    let taken = std::mem::replace(storage_arc, Arc::new(placeholder));

    let src_any = match &src.storage {
        TensorStorage::Cpu(a) => a,
        _ => {
            return Err(TensorError::Unsupported(
                "slice_set_half: GPU src not handled".into(),
            ))
        }
    };

    // Helper: get src f16 view from either FloatStorage or ArcArray<f16>.
    let get_src_f16 = |src_a: &Arc<dyn Any + Send + Sync>| -> Result<ArrayD<f16>> {
        if let Some(arc_arr) = src_a.downcast_ref::<ArcArray<f16, IxDyn>>() {
            return Ok(arc_arr.to_owned());
        }
        let fs = src_a
            .downcast_ref::<FloatStorage>()
            .ok_or(TensorError::dtype_mismatch(DType::F16, src.dtype()))?;
        Ok(fs.as_f16_view()?.to_owned())
    };
    // Helper: get src bf16 view from either FloatStorage or ArcArray<bf16>.
    let get_src_bf16 = |src_a: &Arc<dyn Any + Send + Sync>| -> Result<ArrayD<bf16>> {
        if let Some(arc_arr) = src_a.downcast_ref::<ArcArray<bf16, IxDyn>>() {
            return Ok(arc_arr.to_owned());
        }
        let fs = src_a
            .downcast_ref::<FloatStorage>()
            .ok_or(TensorError::dtype_mismatch(DType::BF16, src.dtype()))?;
        Ok(fs.as_bf16_view()?.to_owned())
    };

    // Try ArcArray<f16> destination (from narrow_half_zero_copy).
    if dtype == DType::F16 {
        if let Ok(concrete) = taken.clone().downcast::<ArcArray<f16, IxDyn>>() {
            let arc_arr = match Arc::try_unwrap(concrete) {
                Ok(owned) => owned,
                Err(shared) => shared.as_ref().clone(),
            };
            let mut arr = arc_arr.into_owned();
            let src_data = get_src_f16(src_any)?;
            arr.slice_mut(slice_info).assign(&src_data);
            *storage_arc = Arc::new(arr.into_shared());
            return Ok(());
        }
    }
    // Try ArcArray<bf16> destination (from narrow_half_zero_copy).
    if dtype == DType::BF16 {
        if let Ok(concrete) = taken.clone().downcast::<ArcArray<bf16, IxDyn>>() {
            let arc_arr = match Arc::try_unwrap(concrete) {
                Ok(owned) => owned,
                Err(shared) => shared.as_ref().clone(),
            };
            let mut arr = arc_arr.into_owned();
            let src_data = get_src_bf16(src_any)?;
            arr.slice_mut(slice_info).assign(&src_data);
            *storage_arc = Arc::new(arr.into_shared());
            return Ok(());
        }
    }

    // FloatStorage destination (from constructors: zeros/ones/full/from_raw).
    let concrete = taken
        .downcast::<FloatStorage>()
        .map_err(|_| TensorError::dtype_mismatch(dtype, dtype))?;
    let fs = match Arc::try_unwrap(concrete) {
        Ok(owned) => owned,
        Err(shared) => shared.as_ref().clone(),
    };

    match (fs, dtype) {
        (FloatStorage::F16(mut arr), DType::F16) => {
            let src_data = get_src_f16(src_any)?;
            arr.slice_mut(slice_info).assign(&src_data);
            *storage_arc = Arc::new(FloatStorage::F16(arr));
        }
        (FloatStorage::BF16(mut arr), DType::BF16) => {
            let src_data = get_src_bf16(src_any)?;
            arr.slice_mut(slice_info).assign(&src_data);
            *storage_arc = Arc::new(FloatStorage::BF16(arr));
        }
        (other_fs, _) => {
            return Err(TensorError::dtype_mismatch(dtype, other_fs.dtype()));
        }
    }
    Ok(())
}
