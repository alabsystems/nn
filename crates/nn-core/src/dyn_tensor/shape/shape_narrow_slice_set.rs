// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Narrow and slice_set operations for [`DynTensor`].
//!
//! Extracted from `shape/mod.rs` to stay under the 500-line limit.

use super::f32_ops::{narrow_f32_zero_copy, narrow_half_zero_copy, slice_set_f32, slice_set_half};
use super::helpers::{build_narrow_slice, build_slice_info, validate_slice_set_args};
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, trace, Dim, DynTensor, TensorStorage};
use crate::{DType, Device, Result, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

impl DynTensor {
    /// Narrow (slice) along a dimension.
    ///
    /// For CPU f32 tensors, returns a **zero-copy view** sharing the same
    /// backing data via `ArcArray`. The returned DynTensor points into the
    /// parent's memory — no data is copied. This makes KvCache append returns
    /// O(1) instead of O(S) per step.
    ///
    /// For BF16/F16 CPU tensors, also returns a zero-copy view via
    /// `narrow_half_zero_copy` (#1856).
    ///
    /// For integer CPU and GPU tensors, falls back to copying behavior.
    ///
    /// # GPU dispatch
    ///
    /// Float GPU tensors use native Metal kernel dispatch via
    /// [`GpuBackend::narrow`]. Dim-0 narrow uses CPU-side memcpy (no kernel
    /// launch). Non-float GPU tensors (U32, I64 from argmax/topk) fall back
    /// to CPU round-trip: GPU→CPU transfer, narrow, CPU→GPU transfer.
    pub fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let end = start.checked_add(len).ok_or_else(|| {
            TensorError::InvalidShape(format!("narrow({dim}, {start}, {len}) overflows usize"))
        })?;
        if end > self.dims[dim] {
            return Err(TensorError::InvalidShape(format!(
                "narrow({dim}, {start}, {len}) exceeds dim size {}",
                self.dims[dim]
            )));
        }
        let mut result = self.narrow_dispatch(dim, start, len, end)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[self])?;
            if let Some(id) = trace::record_op(
                TraceOp::Narrow {
                    dim,
                    start,
                    length: len,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Internal dispatch for narrow — separated so trace recording can wrap all paths.
    fn narrow_dispatch(&self, dim: usize, start: usize, len: usize, end: usize) -> Result<Self> {
        // Try native GPU dispatch; fall back to CPU round-trip.
        if self.device().is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.narrow(self, dim, start, len)) {
                return result;
            }
            let cpu = self.to_device(&Device::Cpu)?;
            let result = cpu.narrow(dim, start, len)?;
            return result.to_device(&self.device());
        }
        // Zero-copy path for f32: use ArcArray shared-backing slice.
        if let TensorStorage::Cpu(any) = &self.storage {
            if let Some(arc_view) = narrow_f32_zero_copy(any, dim, start, len)? {
                return Ok(arc_view);
            }
        }
        // Zero-copy path for f16/bf16: use ArcArray shared-backing slice (#1856).
        if let TensorStorage::Cpu(any) = &self.storage {
            if let Some(arc_view) = narrow_half_zero_copy(any, dim, start, len)? {
                return Ok(arc_view);
            }
        }
        // Fallback for integer types: copy via slice.
        let slice_info = build_narrow_slice(self.rank(), dim, start, end)?;
        dispatch_cpu_typed!(
            self,
            |arr: &ArrayD<_>| -> Result<ArrayD<_>> {
                Ok(arr.slice(slice_info.as_slice()).to_owned())
            },
            "narrow"
        )
    }

    /// Write `src` into `self` at `[offset..offset+src.dim(dim)]` along `dim`.
    ///
    /// Consumes `self`. When the backing storage is uniquely owned (no
    /// outstanding views from `narrow()`), mutates in-place — only the slice
    /// region is written, not the full buffer. This makes KV cache appends
    /// O(1) per token instead of O(buffer_size).
    ///
    /// Supports all CPU dtypes (f32, u32, u8, i64).
    ///
    /// Accepts both `usize` and [`D`](crate::D) (e.g., `D::Minus1`).
    ///
    /// # GPU dispatch
    ///
    /// Tries native Metal dispatch via [`GpuBackend::slice_set`]. If the
    /// backend returns `None` (e.g., non-float dtype), both `self` and `src`
    /// are transferred to CPU, the operation runs on CPU, and the result is
    /// transferred back to the original GPU device.
    pub fn slice_set_into(self, dim: impl Dim, offset: usize, src: &Self) -> Result<Self> {
        let dim = dim.to_index(self.rank())?;
        let end = validate_slice_set_args(&self, dim, offset, src)?;
        if self.dtype != src.dtype {
            return Err(TensorError::dtype_mismatch(self.dtype, src.dtype));
        }
        if self.device().is_gpu() || src.device().is_gpu() {
            return self.slice_set_gpu_path(dim, offset, src);
        }
        self.slice_set_cpu_path(dim, offset, end, src)
    }

    /// GPU dispatch for slice_set_into. Separated for function-size compliance.
    fn slice_set_gpu_path(self, dim: usize, offset: usize, src: &Self) -> Result<Self> {
        if let Some(gpu_result) = gpu_backend_dispatch(|b| b.slice_set(&self, dim, offset, src)) {
            let mut result = gpu_result?;
            if trace::is_tracing() {
                let input_ids = Self::trace_input_ids(&[&self, src])?;
                if let Some(id) = trace::record_op(
                    TraceOp::SliceSet { dim, start: offset },
                    &input_ids,
                    result.dims(),
                    result.dtype(),
                ) {
                    result.set_trace_id(id);
                }
            }
            return Ok(result);
        }
        let dev = if self.device().is_gpu() {
            self.device()
        } else {
            src.device()
        };
        // Capture trace IDs before self is consumed by the closure.
        let self_trace_id = self.trace_id();
        let src_trace_id = src.trace_id();
        // Suppress tracing during CPU round-trip (CPU copies lack trace IDs).
        let mut result = trace::with_trace_suppressed(|| {
            self.to_device(&Device::Cpu)?
                .slice_set_into(dim, offset, &src.to_device(&Device::Cpu)?)?
                .to_device(&dev)
        })?;
        if trace::is_tracing() {
            if let (Some(sid), Some(rid)) = (self_trace_id, src_trace_id) {
                if let Some(id) = trace::record_op(
                    TraceOp::SliceSet { dim, start: offset },
                    &[sid, rid],
                    result.dims(),
                    result.dtype(),
                ) {
                    result.set_trace_id(id);
                }
            }
        }
        Ok(result)
    }

    /// CPU path for slice_set_into. Separated for function-size compliance.
    fn slice_set_cpu_path(
        mut self,
        dim: usize,
        offset: usize,
        end: usize,
        src: &Self,
    ) -> Result<Self> {
        let slice_info = build_slice_info(self.rank(), dim, offset, end)?;
        let storage_arc = match &mut self.storage {
            TensorStorage::Cpu(arc) => arc,
            TensorStorage::Gpu { .. } => {
                return Err(TensorError::Unsupported(
                    "slice_set: GPU tensor not handled".into(),
                ))
            }
            TensorStorage::Quantized(_) => {
                return Err(TensorError::Unsupported(
                    "slice_set: quantized tensor — call .dequantize() first".into(),
                ))
            }
        };
        let src_any = match &src.storage {
            TensorStorage::Cpu(a) => a,
            TensorStorage::Gpu { .. } => {
                return Err(TensorError::Unsupported(
                    "slice_set: GPU src not handled".into(),
                ))
            }
            TensorStorage::Quantized(_) => {
                return Err(TensorError::Unsupported(
                    "slice_set: quantized src — call .dequantize() first".into(),
                ))
            }
        };
        macro_rules! do_slice_set {
            ($T:ty) => {{
                let placeholder = ArrayD::<$T>::zeros(IxDyn(&[]));
                let taken = std::mem::replace(storage_arc, Arc::new(placeholder));
                let concrete: Arc<ArrayD<$T>> = taken
                    .downcast()
                    .map_err(|_| TensorError::dtype_mismatch(self.dtype, self.dtype))?;
                let mut arr = match Arc::try_unwrap(concrete) {
                    Ok(owned) => owned,
                    Err(shared) => shared.as_ref().clone(),
                };
                let s = src_any
                    .downcast_ref::<ArrayD<$T>>()
                    .ok_or(TensorError::dtype_mismatch(self.dtype, src.dtype))?;
                arr.slice_mut(slice_info.as_slice()).assign(s);
                *storage_arc = Arc::new(arr);
            }};
        }
        match self.dtype {
            DType::U32 => do_slice_set!(u32),
            DType::U8 => do_slice_set!(u8),
            DType::I64 => do_slice_set!(i64),
            DType::F32 | DType::F64 => slice_set_f32(storage_arc, src, &slice_info, self.dtype)?,
            DType::F16 => slice_set_half(storage_arc, src, &slice_info, DType::F16)?,
            DType::BF16 => slice_set_half(storage_arc, src, &slice_info, DType::BF16)?,
            DType::I32 | DType::Bool => {
                return Err(TensorError::Unsupported(format!(
                    "slice_set: dtype {} not supported",
                    self.dtype
                )))
            }
        }
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&[&self, src])?;
            if let Some(id) = trace::record_op(
                TraceOp::SliceSet { dim, start: offset },
                &input_ids,
                self.dims(),
                self.dtype(),
            ) {
                self.set_trace_id(id);
            }
        }
        Ok(self)
    }

    /// Deprecated alias for [`slice_set_into`](Self::slice_set_into).
    #[deprecated(
        since = "0.1.0",
        note = "renamed to slice_set_into for consistency with scatter_add_into/index_add_into"
    )]
    pub fn slice_set(self, dim: impl Dim, offset: usize, src: &Self) -> Result<Self> {
        self.slice_set_into(dim, offset, src)
    }
}
