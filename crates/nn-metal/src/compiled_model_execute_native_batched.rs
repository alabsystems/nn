// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Batched linear projection executor for `CompiledModel`.
//!
//! Implements `BatchedLinearProjection` and `ProjectionSlice` NativeOp
//! execution. The batched projection does a single matmul with concatenated
//! weights, narrows the first projection as the step output, and stashes
//! the full intermediate in a thread-local for `ProjectionSlice` steps.
//!
//! Part of #3269.

use std::cell::RefCell;
use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

use super::CompiledModel;
use super::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};

thread_local! {
    /// Thread-local stash for full batched matmul intermediates.
    ///
    /// Keyed by step index of the `BatchedLinearProjection` step.
    /// `ProjectionSlice` steps read from here and narrow their portion.
    /// Cleared at the start of each forward pass by `clear_projection_temps`.
    static PROJECTION_TEMPS: RefCell<HashMap<usize, DynTensor>> =
        RefCell::new(HashMap::new());
}

/// Clear all stashed batched projection intermediates.
///
/// Called at the start of `run_steps_inner` to prevent stale data from
/// a prior forward pass leaking into the current one.
pub(crate) fn clear_projection_temps() {
    PROJECTION_TEMPS.with(|t| t.borrow_mut().clear());
}

/// Execute a `NativeOpKind::BatchedLinearProjection` step.
///
/// 1. Matmul: `input @ weight_t` (+bias) → `[..batch, total_out]`
/// 2. Narrow first projection → step output `[..batch, proj_sizes[0]]`
/// 3. Stash full output in `PROJECTION_TEMPS` for `ProjectionSlice` steps.
///
/// Under autocast, inputs arrive pre-cast to F16 by `cast_autocast_inputs`.
/// DynTensor matmul at F16 routes to `simd_gemm_f16` (F32 accumulators)
/// when dims qualify, so F32 accumulation precision is preserved. (#3281)
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_batched_linear_projection(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    in_features: usize,
    total_out_features: usize,
    projection_sizes: &[usize],
    has_bias: bool,
    input_shape: &[usize],
    _cache: &PipelineCache,
) -> Result<GpuSlice> {
    let dtype = model.step_dtype(step_idx);
    let step_weights = &model.def.weight_buffers[step_idx];

    // Resolve the graph input (shared hidden state).
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;

    let input_tensor = slice_to_dyn(&input_slice, input_shape, dtype)?;

    // Load pre-transposed weight: [in_features, total_out].
    let weight_t = weight_to_dyn(
        step_weights,
        "weight_t",
        &[in_features, total_out_features],
        dtype,
        step_idx,
        "BatchedLinearProjection",
    )?;

    // Matmul: input @ weight_t → [..batch, total_out].
    // Under F16, DynTensor matmul routes to simd_gemm_f16 with F32 accumulators.
    let full_output = input_tensor.matmul(&weight_t).map_err(|e| {
        native_dispatch_err(step_idx, format!("BatchedLinearProjection matmul: {e}"))
    })?;

    let full_output = if has_bias {
        let bias = weight_to_dyn(
            step_weights,
            "bias",
            &[total_out_features],
            dtype,
            step_idx,
            "BatchedLinearProjection",
        )?;
        full_output.broadcast_add(&bias).map_err(|e| {
            native_dispatch_err(step_idx, format!("BatchedLinearProjection bias_add: {e}"))
        })?
    } else {
        full_output
    };

    // Narrow the first projection from the full output.
    let ndim = full_output.rank();
    let last_dim = ndim.checked_sub(1).ok_or_else(|| {
        native_dispatch_err(step_idx, "BatchedLinearProjection: 0-rank output".into())
    })?;
    let first_proj_size = projection_sizes.first().copied().unwrap_or(0);
    let first_proj = full_output
        .narrow(last_dim, 0, first_proj_size)
        .map_err(|e| {
            native_dispatch_err(
                step_idx,
                format!("BatchedLinearProjection narrow first: {e}"),
            )
        })?;

    // Stash the full output for ProjectionSlice steps to read.
    PROJECTION_TEMPS.with(|t| t.borrow_mut().insert(step_idx, full_output));

    dyn_to_slice(&first_proj, step_idx, "BatchedLinearProjection")
}

/// Execute a `NativeOpKind::ProjectionSlice` step.
///
/// Reads the stashed full output from `PROJECTION_TEMPS` and narrows
/// to extract this projection's slice.
pub(super) fn execute_native_projection_slice(
    step_idx: usize,
    source_step: usize,
    dim: usize,
    start: usize,
    length: usize,
    _output_shape: &[usize],
) -> Result<GpuSlice> {
    let full_output = PROJECTION_TEMPS.with(|t| t.borrow().get(&source_step).cloned());
    let full_output = full_output.ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!(
                "ProjectionSlice: no stashed output for source step {source_step} \
                 (BatchedLinearProjection must execute before ProjectionSlice)"
            ),
        )
    })?;

    let narrowed = full_output.narrow(dim, start, length).map_err(|e| {
        native_dispatch_err(
            step_idx,
            format!("ProjectionSlice narrow(dim={dim}, start={start}, len={length}): {e}"),
        )
    })?;

    dyn_to_slice(&narrowed, step_idx, "ProjectionSlice")
}
