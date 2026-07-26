// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for compiled model execution.
//!
//! Extracted from `compiled_model_execute.rs` to keep files under 450 lines.
//! Contains output normalization, dtype casting, autocast boundary helpers,
//! and DynTensor conversion utilities.

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_slice::GpuSlice;

use super::CompiledModelError;

/// Build a `DispatchFailed` error for a native op step.
///
/// Replaces the per-function `let make_err = |reason: String| -> TensorError { ... }`
/// closures that were duplicated 23× across 10 executor files.
pub(super) fn native_dispatch_err(step_idx: usize, reason: String) -> TensorError {
    CompiledModelError::DispatchFailed { step_idx, reason }.into()
}

/// Blit-copy a `GpuSlice` to offset 0 when it sits at a non-zero arena offset.
///
/// Arena-allocated intermediate buffers may have `byte_offset > 0`. Callers of
/// `execute()` expect data at offset 0, so we normalize here. See #2217.
///
/// `out_bytes`: the exact output byte count (product of shape * element_size).
/// Must not use `buffer.len() - offset` since that gives remaining arena space.
pub(super) fn normalize_output_to_offset_zero(
    cache: &PipelineCache,
    slice: &GpuSlice,
    step_idx: usize,
    out_bytes: usize,
) -> Result<MetalBuffer> {
    if slice.byte_offset() == 0 && slice.buffer().len() == out_bytes {
        return Ok(slice.buffer().alias());
    }
    // Either non-zero offset or oversized buffer — blit-copy to exact-size buffer.
    let ctx = cache.context();
    let fresh = ctx.create_buffer_zeroed(out_bytes).map_err(|e| {
        TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("output normalize alloc: {e}"),
        })
    })?;
    crate::gpu_scope::encode_into_lazy_batch(|batch| {
        batch.blit_copy(slice.buffer(), slice.byte_offset(), &fresh, 0, out_bytes)
    })
    .map_err(|e| {
        TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("output normalize batch scope: {e}"),
        })
    })?
    .map_err(|e| {
        TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("output normalize blit: {e}"),
        })
    })?;
    Ok(fresh)
}

/// Blit-copy a step's output into the planned contiguous buffer and return
/// a `GpuSlice` referencing the planned region. Restores the arena checkpoint
/// to reclaim the temporary allocation. (#2913)
///
/// Returns `None` if the step has no planned offset (i.e., it's an input or
/// passthrough that doesn't consume arena space).
pub(super) fn relocate_to_planned_buffer(
    planned_buf: &MetalBuffer,
    slice: &GpuSlice,
    offset: usize,
    size: usize,
    step_idx: usize,
) -> Result<GpuSlice> {
    // Pre-validate source bounds to give a detailed error before the blit.
    let src_end = slice.byte_offset().checked_add(size);
    if src_end.map_or(true, |end| end > slice.buffer().len()) {
        return Err(TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!(
                "relocate source bounds: src_offset={} + size={} > src_buf_len={} \
                 (2x ratio={:.1}, likely F32/F16 dtype mismatch in buffer planner)",
                slice.byte_offset(),
                size,
                slice.buffer().len(),
                size as f64 / slice.buffer().len().max(1) as f64,
            ),
        }));
    }
    // Pre-validate destination bounds.
    let dst_end = offset.checked_add(size);
    if dst_end.map_or(true, |end| end > planned_buf.len()) {
        return Err(TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!(
                "relocate dest bounds: dst_offset={} + size={} > planned_buf_len={}",
                offset,
                size,
                planned_buf.len(),
            ),
        }));
    }
    // Ensure a lazy batch exists — NativeOps may have flushed it.
    // Use blit-specific path: increments TOTAL_BLITS, not TOTAL_ENCODINGS.
    crate::gpu_scope::ensure_batch_for_blit()?;
    crate::gpu_scope::encode_into_lazy_batch(|batch| {
        batch.blit_copy(
            slice.buffer(),
            slice.byte_offset(),
            planned_buf,
            offset,
            size,
        )
    })
    .map_err(|e| {
        TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("planned buffer blit scope: {e}"),
        })
    })?
    .map_err(|e| {
        TensorError::from(CompiledModelError::DispatchFailed {
            step_idx,
            reason: format!("planned buffer blit: {e}"),
        })
    })?;
    Ok(GpuSlice::new(planned_buf.alias(), offset))
}

/// Arm the planned-buffer redirect before NativeOp execution (#3448).
///
/// When the step has a planned offset and a non-zero size, arms the
/// thread-local redirect so `arena_alloc_or_create` returns the planned
/// buffer region for the first matching allocation.
///
/// Returns an RAII guard that clears the redirect on drop — ensures
/// cleanup even when `execute_native_op` returns `Err` via `?`.
pub(super) fn arm_native_op_redirect(
    planned_offset: Option<usize>,
    planned_buf: &Option<MetalBuffer>,
    step_sizes: &[usize],
    step_idx: usize,
) -> Option<crate::arena::PlannedRedirectGuard> {
    if let (Some(off), Some(ref pb)) = (planned_offset, planned_buf) {
        let size = step_sizes.get(step_idx).copied().unwrap_or(0);
        if size > 0 {
            return Some(crate::arena::arm_planned_redirect_guard(pb, off, size));
        }
    }
    None
}

/// Arm the dispatch normalization skip before IR Dispatch execution (#4264).
///
/// When the step has a planned offset and a non-zero size, arms the
/// thread-local `SKIP_DISPATCH_NORMALIZATION` flag so `dispatch_execute_plan`
/// returns the output `GpuSlice` with its original byte offset instead of
/// blit-copying to a fresh zero-offset buffer. This MUST be paired with
/// `arm_native_op_redirect` — otherwise the output will have a non-zero
/// offset but won't be in the planned buffer, and the relocation blit in
/// `run_steps_inner` will fire anyway.
///
/// Returns an RAII guard that clears the flag on drop.
pub(super) fn arm_dispatch_normalization_skip(
    planned_offset: Option<usize>,
    planned_buf: &Option<MetalBuffer>,
    step_sizes: &[usize],
    step_idx: usize,
) -> Option<crate::tensor_dispatch::SkipNormalizationGuard> {
    if let (Some(_off), Some(ref _pb)) = (planned_offset, planned_buf) {
        let size = step_sizes.get(step_idx).copied().unwrap_or(0);
        if size > 0 {
            return Some(crate::tensor_dispatch::arm_skip_dispatch_normalization());
        }
    }
    None
}

/// Cast a GPU buffer between ScalarTypes (e.g., F32 → F16 or F16 → F32).
///
/// Uses `DynTensor::to_dtype()` which dispatches a GPU element-wise conversion
/// kernel. Returns the input slice unchanged if `from` and `to` are identical.
///
/// `num_elements`: total element count in the buffer (byte_len / from_element_size).
/// The shape used for the DynTensor wrapper is flat `[num_elements]` — the actual
/// tensor shape doesn't matter for element-wise conversion.
///
/// Part of F16 mixed-precision pipeline (Tier 1).
pub(super) fn cast_slice_dtype(
    _cache: &PipelineCache,
    slice: &GpuSlice,
    num_elements: usize,
    from: nn_dsl::ir::ScalarType,
    to: nn_dsl::ir::ScalarType,
) -> Result<GpuSlice> {
    if from == to || num_elements == 0 {
        return Ok(slice.alias());
    }

    let from_dt: DType = from.into();
    let to_dt: DType = to.into();

    let storage = MetalTensorData::view(slice.buffer().alias(), slice.byte_offset());
    let tensor = DynTensor::from_gpu_storage(
        vec![num_elements],
        from_dt,
        Arc::new(storage),
        Device::metal(),
    )?;
    let converted = tensor.to_dtype(to_dt)?;
    let gpu_data = converted.gpu_data::<MetalTensorData>().map_err(|e| {
        TensorError::from(CompiledModelError::DispatchFailed {
            step_idx: 0,
            reason: format!("dtype cast GPU data extract: {e}"),
        })
    })?;
    Ok(gpu_data.as_gpu_slice())
}

/// Cast an F32 external input GpuSlice to F16 for mixed-precision InputForward.
///
/// `num_elements` must come from `step_numel(step_idx)`, not from `buffer().len()`,
/// because relocated slices in the planned buffer have oversized allocations.
pub(super) fn cast_input_f32_to_f16(
    cache: &PipelineCache,
    slice: &GpuSlice,
    num_elements: usize,
) -> Result<GpuSlice> {
    cast_slice_dtype(
        cache,
        slice,
        num_elements,
        nn_dsl::ir::ScalarType::F32,
        nn_dsl::ir::ScalarType::F16,
    )
}

/// Execute a NativeOp step with mixed-precision awareness.
///
/// **D5b (Tier 2):** NativeOps with parameterized MSL (D4) now handle F16
/// directly — their executors use `model.step_dtype()` to wrap buffers with
/// the correct dtype. For these ops, no boundary cast is needed.
///
/// **LSTM (D6):** Stays F32 (`step_scalar_types[i]` is F32 even in mixed
/// precision). When inputs are F16, boundary casts are still required:
/// cast F16 inputs → F32, execute LSTM, cast F32 output → F16.
///
/// `buffer_dtypes` tracks the actual dtype of each buffer.
pub(super) fn execute_native_op_mixed(
    model: &super::CompiledModel,
    op: &nn_dsl::NativeOpKind,
    step_idx: usize,
    buffers: &mut [Option<GpuSlice>],
    buffer_dtypes: &mut [nn_dsl::ir::ScalarType],
    cache: &PipelineCache,
) -> Result<()> {
    use nn_dsl::ir::ScalarType;

    let step_st = model.step_scalar_type(step_idx);

    // D5b: NativeOps that accept F16 directly (step_scalar_type != F32).
    // No boundary cast — execute_native_op wraps buffers at the right dtype.
    if step_st != ScalarType::F32 {
        let output = model.execute_native_op(op, step_idx, buffers, cache)?;
        buffers[step_idx] = Some(output);
        buffer_dtypes[step_idx] = step_st;
        return Ok(());
    }

    // F32 NativeOp (LSTM): cast F16 inputs → F32, execute, cast output → F16.
    let edges = model.edge_map_for(step_idx);
    let mut saved: Vec<(usize, Option<GpuSlice>)> = Vec::new();
    for &src in edges {
        if buffer_dtypes[src] != ScalarType::F32 {
            if let Some(ref s) = buffers[src] {
                // Use effective_numel: RuntimeOp buffers use buffer geometry,
                // planned-buffer steps use pre-computed trace-time numel. (#3121)
                let n = model.effective_numel(src, s, buffer_dtypes[src]);
                let f32_slice = cast_slice_dtype(cache, s, n, buffer_dtypes[src], ScalarType::F32)?;
                saved.push((src, buffers[src].take()));
                buffers[src] = Some(f32_slice);
            }
        }
    }

    let output = model.execute_native_op(op, step_idx, buffers, cache)?;

    // Restore original F16 buffers.
    for (src, orig) in saved {
        buffers[src] = orig;
    }

    // Cast F32 output → F16 for downstream consumers.
    // Use effective_numel: RuntimeOp outputs use buffer geometry. (#3121)
    let n = model.effective_numel(step_idx, &output, ScalarType::F32);
    let f16_out = cast_slice_dtype(cache, &output, n, ScalarType::F32, ScalarType::F16)?;
    buffers[step_idx] = Some(f16_out);
    buffer_dtypes[step_idx] = ScalarType::F16;

    Ok(())
}

/// Cast input buffers for a Dispatch step at F16↔F32 autocast boundaries.
///
/// When a step expects dtype `step_dt` but some input buffers have a different
/// dtype (tracked in `buffer_dtypes`), cast those buffers to `step_dt`.
/// Returns saved originals for restoration after dispatch.
///
/// Part of #3085 (per-op autocast).
pub(super) fn cast_autocast_inputs(
    model: &super::CompiledModel,
    cache: &PipelineCache,
    step_idx: usize,
    step_dt: nn_dsl::ir::ScalarType,
    buffers: &mut [Option<GpuSlice>],
    buffer_dtypes: &[nn_dsl::ir::ScalarType],
) -> Result<Vec<(usize, Option<GpuSlice>)>> {
    use nn_dsl::ir::ScalarType;

    let edges = model.edge_map_for(step_idx);
    let mut saved: Vec<(usize, Option<GpuSlice>)> = Vec::new();
    for &src in edges {
        // Skip already-cast duplicates (e.g., x*x).
        if saved.iter().any(|(s, _)| *s == src) {
            continue;
        }
        let src_dt = buffer_dtypes.get(src).copied().unwrap_or(ScalarType::F32);
        if src_dt != step_dt {
            if let Some(ref s) = buffers[src] {
                // Use effective_numel: RuntimeOp buffers use buffer geometry,
                // planned-buffer steps use pre-computed trace-time numel. (#3121)
                let n = model.effective_numel(src, s, src_dt);
                let cast = cast_slice_dtype(cache, s, n, src_dt, step_dt)?;
                saved.push((src, buffers[src].take()));
                buffers[src] = Some(cast);
            }
        }
    }
    Ok(saved)
}

/// Restore buffers that were temporarily replaced by [`cast_autocast_inputs`].
pub(super) fn restore_autocast_inputs(
    buffers: &mut [Option<GpuSlice>],
    saved: Vec<(usize, Option<GpuSlice>)>,
) {
    for (src, original) in saved {
        buffers[src] = original;
    }
}

/// Validate that a buffer has enough bytes for the given shape and dtype.
///
/// Defense-in-depth: catches trace compiler bugs that produce shape/buffer
/// mismatches before they cause silent out-of-bounds GPU reads (#3298).
fn validate_buffer_capacity(
    buf_len: usize,
    byte_offset: usize,
    shape: &[usize],
    dtype: DType,
    context: &str,
) -> Result<()> {
    let elem_count = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let expected = elem_count.and_then(|n| n.checked_mul(dtype.size_bytes()));
    let available = buf_len.saturating_sub(byte_offset);
    match expected {
        Some(req) if available >= req => Ok(()),
        Some(req) => Err(TensorError::from(CompiledModelError::DispatchFailed {
            step_idx: 0,
            reason: format!(
                "{context}: buffer capacity {available} < required {req} \
                 (shape={shape:?}, dtype={dtype:?})",
            ),
        })),
        None => Err(TensorError::from(CompiledModelError::DispatchFailed {
            step_idx: 0,
            reason: format!("{context}: shape product overflow (shape={shape:?}, dtype={dtype:?})"),
        })),
    }
}

/// Wrap a `GpuSlice` as a `DynTensor` for use with eager-path kernels.
///
/// Replaces the per-function `wrap_dyn` / `slice_to_dyn` closures that were
/// duplicated 15× across 7 executor files.
///
/// Validates that the buffer has enough bytes for the requested shape × dtype
/// before wrapping. Returns `DispatchFailed` if the buffer is too small (#3298).
pub(super) fn slice_to_dyn(slice: &GpuSlice, shape: &[usize], dtype: DType) -> Result<DynTensor> {
    validate_buffer_capacity(
        slice.buffer().len(),
        slice.byte_offset(),
        shape,
        dtype,
        "slice_to_dyn",
    )?;
    let storage = MetalTensorData::view(slice.buffer().alias(), slice.byte_offset());
    DynTensor::from_gpu_storage(shape.to_vec(), dtype, Arc::new(storage), Device::metal())
}

/// Extract a `GpuSlice` from a `DynTensor` result.
///
/// Replaces the per-function `.gpu_data::<MetalTensorData>().map_err(...)?.as_gpu_slice()`
/// blocks that were duplicated 18× across 7 executor files.
pub(super) fn dyn_to_slice(tensor: &DynTensor, step_idx: usize, op_name: &str) -> Result<GpuSlice> {
    let data = tensor
        .gpu_data::<MetalTensorData>()
        .map_err(|_| native_dispatch_err(step_idx, format!("{op_name}: output not GPU tensor")))?;
    Ok(data.as_gpu_slice())
}

/// Wrap a pre-uploaded weight `MetalBuffer` as a `DynTensor`.
///
/// Replaces the per-function `weight_to_dyn` closures that were duplicated
/// across 8 executor files.
///
/// Validates that the weight buffer has enough bytes for the requested
/// shape × dtype before wrapping (#3298).
pub(super) fn weight_to_dyn(
    weights: &HashMap<String, MetalBuffer>,
    name: &str,
    shape: &[usize],
    dtype: DType,
    step_idx: usize,
    op_name: &str,
) -> Result<DynTensor> {
    let buf = weights.get(name).ok_or_else(|| {
        native_dispatch_err(step_idx, format!("{op_name}: missing weight '{name}'"))
    })?;
    validate_buffer_capacity(buf.len(), 0, shape, dtype, &format!("weight '{name}'"))?;
    let storage = MetalTensorData::new(buf.alias());
    DynTensor::from_gpu_storage(shape.to_vec(), dtype, Arc::new(storage), Device::metal())
}

#[cfg(test)]
#[path = "compiled_model_execute_helpers_tests.rs"]
mod tests;
