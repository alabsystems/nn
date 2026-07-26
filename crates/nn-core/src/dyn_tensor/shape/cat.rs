// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Concatenation and stacking operations for [`DynTensor`].
//!
//! Extracted from `shape/mod.rs` for file-size compliance.

use std::borrow::Borrow;

use ndarray::ArrayD;

use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, trace, Dim, DynTensor};
use crate::{DType, Device, Result, TensorError};

impl DynTensor {
    /// Concatenate tensors along a dimension.
    ///
    /// Accepts both `&[DynTensor]` (candle convention) and `&[&DynTensor]`.
    /// GPU tensors are auto-transferred to CPU for the operation, then the
    /// result is placed on the original device.
    pub fn cat<T: Borrow<Self>>(tensors: &[T], dim: impl Dim) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidShape(
                "cat requires at least one tensor".into(),
            ));
        }
        let first = tensors[0].borrow();
        let rank = first.rank();
        let original_device = first.device();
        let original_dtype = first.dtype();
        let dim = dim.to_index(rank)?;
        for t in tensors.iter().skip(1) {
            let t = t.borrow();
            if t.dtype() != original_dtype {
                return Err(TensorError::dtype_mismatch(original_dtype, t.dtype()));
            }
            if t.rank() != rank {
                return Err(TensorError::RankMismatch {
                    expected: rank,
                    actual: t.rank(),
                });
            }
            for d in 0..rank {
                if d != dim && t.dims()[d] != first.dims()[d] {
                    return Err(TensorError::shape_mismatch(
                        first.dims().to_vec(),
                        t.dims().to_vec(),
                    ));
                }
            }
        }
        // Build a borrowed slice for GPU dispatch and CPU paths.
        let refs: Vec<&Self> = tensors.iter().map(Borrow::borrow).collect();
        let num_inputs = refs.len();
        let mut result = Self::cat_dispatch(&refs, dim, original_device, original_dtype)?;
        if trace::is_tracing() {
            let input_ids = Self::trace_input_ids(&refs)?;
            if let Some(id) = trace::record_op(
                TraceOp::Cat { dim, num_inputs },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Internal dispatch for cat — separated so trace recording wraps all paths.
    fn cat_dispatch(
        refs: &[&Self],
        dim: usize,
        original_device: Device,
        original_dtype: DType,
    ) -> Result<Self> {
        // Try native GPU dispatch for cat.
        if original_device.is_gpu() {
            if let Some(result) = gpu_backend_dispatch(|b| b.cat(refs, dim)) {
                return result;
            }
        }
        // Transfer GPU tensors to CPU for ndarray concat.
        let cpu_tensors: Vec<Self> = if original_device.is_gpu() {
            refs.iter()
                .map(|t| t.to_device(&Device::Cpu))
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        let cpu_refs: Vec<&Self> = if original_device.is_gpu() {
            cpu_tensors.iter().collect()
        } else {
            refs.to_vec()
        };
        let cpu_result = cat_cpu(&cpu_refs, dim, original_dtype)?;
        if original_device.is_gpu() {
            cpu_result.to_device(&original_device)
        } else {
            Ok(cpu_result)
        }
    }

    /// Stack tensors along a new dimension.
    ///
    /// Accepts both `&[DynTensor]` (candle convention) and `&[&DynTensor]`.
    /// GPU tensors stay on GPU: decomposes into unsqueeze (metadata-only) + cat
    /// (has native GPU dispatch). No GPU→CPU transfer.
    pub fn stack<T: Borrow<Self>>(tensors: &[T], dim: impl Dim) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidShape(
                "stack requires at least one tensor".into(),
            ));
        }
        let first = tensors[0].borrow();
        for t in tensors.iter().skip(1) {
            let t = t.borrow();
            if t.dims() != first.dims() {
                return Err(TensorError::shape_mismatch(
                    first.dims().to_vec(),
                    t.dims().to_vec(),
                ));
            }
        }
        // stack inserts a new dimension, so valid range is 0..=rank
        let new_rank = first.rank() + 1;
        let dim = dim.to_index(new_rank)?;
        // Decompose stack into unsqueeze + cat.
        // unsqueeze is metadata-only (no data copy on GPU).
        // cat has gpu_backend_dispatch for native GPU execution.
        let unsqueezed: Vec<Self> = tensors
            .iter()
            .map(|t| t.borrow().unsqueeze(dim))
            .collect::<Result<_>>()?;
        Self::cat(&unsqueezed, dim)
    }
}

/// CPU concatenation with dtype dispatch.
///
/// Extracts typed ndarray views for each tensor, performs `ndarray::concatenate`,
/// and wraps the result in a `DynTensor` with the correct dtype.
fn cat_cpu(tensors: &[&DynTensor], dim: usize, dtype: DType) -> Result<DynTensor> {
    /// Typed helper: extract views, concatenate, wrap result.
    fn cat_typed<T: Clone + 'static>(
        tensors: &[&DynTensor],
        dim: usize,
        extract: fn(&DynTensor) -> Result<ndarray::ArrayViewD<'_, T>>,
        wrap: fn(ArrayD<T>) -> Result<DynTensor>,
    ) -> Result<DynTensor> {
        let arrays: Vec<ndarray::ArrayViewD<'_, T>> =
            tensors.iter().map(|t| extract(t)).collect::<Result<_>>()?;
        let views: Vec<ndarray::ArrayViewD<'_, T>> = arrays.iter().map(|a| a.view()).collect();
        let result = ndarray::concatenate(ndarray::Axis(dim), &views)?;
        wrap(result)
    }

    match dtype {
        DType::U32 => cat_typed(tensors, dim, DynTensor::as_cpu_u32, DynTensor::from_cpu_u32),
        DType::U8 => cat_typed(tensors, dim, DynTensor::as_cpu_u8, DynTensor::from_cpu_u8),
        DType::I64 => cat_typed(tensors, dim, DynTensor::as_cpu_i64, DynTensor::from_cpu_i64),
        // Float dtypes use native FloatStorage since #1646.
        DType::F32 | DType::F64 => {
            cat_typed(tensors, dim, DynTensor::as_cpu_f32, DynTensor::from_cpu_f32)
        }
        DType::F16 => cat_typed(tensors, dim, DynTensor::as_cpu_f16, DynTensor::from_cpu_f16),
        DType::BF16 => cat_typed(
            tensors,
            dim,
            DynTensor::as_cpu_bf16,
            DynTensor::from_cpu_bf16,
        ),
        DType::I32 | DType::Bool => Err(TensorError::Unsupported(format!(
            "cat: dtype {dtype} not supported"
        ))),
    }
}
