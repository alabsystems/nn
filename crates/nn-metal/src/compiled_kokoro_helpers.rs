// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the [`CompiledKokoro`](super::CompiledKokoro) pipeline.
//!
//! Extracted from `compiled_kokoro.rs` per Wave 4 D2 design (#2575).
//! Contains device constructors, segment error helpers, and the shared
//! multi-output validator used by both `synthesize()` and the step API.

use nn_core::dyn_tensor::trace::{record_input, ComputationGraph, NodeId};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, TensorError};
use nn_models::kokoro_error::validate_speed;

use super::{CompiledKokoro, CompiledKokoroError, StyleSplit};

pub(super) fn cpu() -> Device {
    Device::Cpu
}

pub(super) fn gpu() -> Device {
    Device::Metal { device_id: 0 }
}

/// Return the device where model weights live.
///
/// Trace inputs must be on this device so forward passes inside
/// `trace_graph` don't hit device-mismatch errors. After weight release
/// (#3079), `model` is `None` — returns CPU since tracing is no longer
/// possible and all compiled segments already have GPU buffers.
pub(super) fn model_device(model: Option<&nn_models::kokoro_tts::KokoroModel>) -> Device {
    model.map_or(Device::Cpu, |m| {
        m.plbert().word_embeddings().weight().device()
    })
}

/// Set the last traced node as the primary output.
///
/// Returns an error if tracing produced an empty graph or if the graph's
/// output bookkeeping is internally inconsistent.
pub(super) fn set_last_output(graph: &mut ComputationGraph) -> nn_core::Result<()> {
    let last_id = graph.nodes().last().map(nn_core::dyn_tensor::trace::TraceNode::id).ok_or_else(|| {
        TensorError::InvalidShape("trace bug: graph has no nodes to mark as primary output".into())
    })?;
    if !graph.set_primary_output(last_id) {
        return Err(TensorError::InvalidShape(format!(
            "trace bug: output node {last_id} not found in graph"
        )));
    }
    Ok(())
}

/// Record a trace input, returning `Err` if tracing is not active.
///
/// Replaces `record_input(...).expect("invariant: tracing active")` with
/// proper `Result` propagation (#2962). Inside `trace_graph()` closures
/// tracing is always active, but `.expect()` is forbidden in production.
pub(super) fn record_input_or_err(dims: &[usize], dtype: DType) -> nn_core::Result<NodeId> {
    record_input(dims, dtype).ok_or_else(|| CompiledKokoroError::TracingNotActive.into())
}

pub(super) fn seg_compile_err(segment: &'static str, source: TensorError) -> TensorError {
    CompiledKokoroError::SegmentCompileFailed {
        segment,
        source: Box::new(source),
    }
    .into()
}

pub(super) fn seg_cache_miss(segment: &'static str) -> TensorError {
    CompiledKokoroError::SegmentCacheMiss { segment }.into()
}

/// Clone a tensor and register it as a trace input.
///
/// Reduces the 3-line `clone` + `set_trace_id` + `record_input_or_err`
/// pattern to a single call. Used inside `trace_graph()` closures where
/// every input tensor must have a trace node ID.
///
/// Part of #2972.
pub(super) fn trace_input(t: &DynTensor) -> nn_core::Result<DynTensor> {
    let mut cloned = t.clone();
    let id = record_input_or_err(cloned.dims(), DType::F32)?;
    cloned.set_trace_id(id);
    Ok(cloned)
}

/// Compute Generator cache key / output length with overflow checks.
///
/// Generator segment 4 is keyed by `2 * t_mel * upsample_factor`.
pub(super) fn generator_total_samples(
    t_mel: usize,
    upsample_factor: usize,
) -> nn_core::Result<usize> {
    t_mel
        .checked_mul(2)
        .and_then(|v| v.checked_mul(upsample_factor))
        .ok_or_else(|| TensorError::DimensionOverflow {
            dims: vec![2, t_mel, upsample_factor],
        })
}

/// Shared front-end validation and setup for Kokoro synthesis entrypoints.
pub(super) fn prepare_synthesis_inputs(
    kokoro: &CompiledKokoro,
    input_ids: &DynTensor,
    style: &DynTensor,
    speed: f32,
) -> Result<StyleSplit, CompiledKokoroError> {
    validate_speed(speed).map_err(|_| CompiledKokoroError::InvalidSpeed { value: speed })?;
    validate_input_ids(input_ids, kokoro.config().plbert.max_position_embeddings)?;

    // Reclaim buffer pool entries from previous synthesis call (#3079 D3).
    crate::arena::pool_reclaim();

    kokoro.split_style(style)
}

/// Validate input_ids shape: must be rank 2, non-empty, and within PlBert context.
///
/// Both `synthesize()` and `synthesize_with_timing()` need this check.
/// Without it, `step_encode`'s `ids.dims()[1]` panics with an opaque
/// index-out-of-bounds instead of a meaningful error.
///
/// Part of F18, #2218.
pub(super) fn validate_input_ids(
    input_ids: &DynTensor,
    max_position_embeddings: usize,
) -> Result<(), CompiledKokoroError> {
    if input_ids.dims().len() < 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: input_ids.dims().len(),
        }
        .into());
    }
    let seq_len = input_ids.dims()[1];
    if seq_len == 0 {
        return Err(TensorError::Unsupported("input_ids has zero sequence length".into()).into());
    }
    if seq_len > max_position_embeddings {
        return Err(CompiledKokoroError::InvalidInput(format!(
            "input_ids seq_len {seq_len} exceeds max_position_embeddings {max_position_embeddings}"
        )));
    }
    Ok(())
}

/// Validate that a segment produced the expected number of outputs.
///
/// Returns `Err(OutputCountMismatch)` unless the output count exactly matches
/// `expected`. Used by both `synthesize()` (returns `CompiledKokoroError`)
/// and the step API (returns `TensorError`, auto-converted via `?`).
pub(super) fn check_multi_output(
    outputs: &[DynTensor],
    expected: usize,
    segment: &'static str,
) -> Result<(), CompiledKokoroError> {
    if outputs.len() != expected {
        return Err(CompiledKokoroError::OutputCountMismatch {
            segment,
            expected,
            actual: outputs.len(),
        });
    }
    Ok(())
}
