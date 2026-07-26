// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch and input resolution helpers for compiled model execution.
//!
//! Extracted from `compiled_model_execute.rs` for 450-line compliance.
//! Contains `resolve_input_slice` and `execute_dispatch`.

use std::collections::HashMap;

use nn_core::{Result, TensorError};

use nn_dsl::PrecisionContract;

use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;
use crate::tensor_dispatch::execute_tensor_dispatch_to_buffer_gpu_only;

use super::super::{CompiledModel, CompiledModelError};

impl CompiledModel {
    /// Resolve the input slice for a step by looking up the edge map.
    pub(super) fn resolve_input_slice(
        &self,
        step_idx: usize,
        input_idx: usize,
        buffers: &[Option<GpuSlice>],
    ) -> Result<GpuSlice> {
        let meta = self.def.step_metas.get(step_idx).ok_or_else(|| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx,
                reason: "no step_metas entry".into(),
            })
        })?;

        let &src_step = meta.edges.get(input_idx).ok_or_else(|| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx,
                reason: format!("no input at index {input_idx}"),
            })
        })?;

        buffers[src_step]
            .as_ref()
            .map(GpuSlice::alias)
            .ok_or_else(|| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!("input step {src_step} has no buffer"),
                })
            })
    }

    /// Execute a single Dispatch step: build input map and dispatch to GPU (#2339, #2501).
    pub(super) fn execute_dispatch<'a>(
        &'a self,
        cache: &PipelineCache,
        def: &nn_dsl::TensorKernelDef,
        step_idx: usize,
        buffers: &[Option<GpuSlice>],
        scratch: &mut HashMap<&'a str, GpuSlice>,
    ) -> Result<GpuSlice> {
        scratch.clear();

        // Use cached input names computed at build time instead of scanning
        // IR nodes on every forward pass. (#2501)
        let input_names = &self.def.input_name_cache[step_idx];
        let graph_inputs = &self
            .def.step_metas
            .get(step_idx)
            .ok_or_else(|| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: "no step_metas entry for dispatch step".into(),
                })
            })?
            .edges;
        let mut graph_input_idx = 0;

        let step_weights = &self.def.weight_buffers[step_idx];
        for input_name in input_names {
            if let Some(weight_buf) = step_weights.get(input_name.as_str()) {
                scratch.insert(input_name.as_str(), GpuSlice::from_ref(weight_buf, 0));
            } else if let Some(&src_step) = graph_inputs.get(graph_input_idx) {
                let slice = buffers[src_step].as_ref().ok_or_else(|| {
                    TensorError::from(CompiledModelError::DispatchFailed {
                        step_idx,
                        reason: format!(
                            "input '{input_name}' references step {src_step} with no buffer"
                        ),
                    })
                })?;
                scratch.insert(input_name.as_str(), slice.alias());
                graph_input_idx += 1;
            } else {
                return Err(TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx,
                    reason: format!("input '{input_name}' is neither a weight nor a graph edge"),
                }));
            }
        }

        // Guard: all graph edges must be consumed. If not, the positional
        // counter diverged from the edge map — a build_edge_map / def_input_names
        // ordering mismatch. See #2379.
        if graph_input_idx != graph_inputs.len() {
            return Err(TensorError::from(CompiledModelError::DispatchFailed {
                step_idx,
                reason: format!(
                    "graph input count mismatch: consumed {} but edge_map has {} \
                     (def_input_names / build_edge_map ordering divergence)",
                    graph_input_idx,
                    graph_inputs.len()
                ),
            }));
        }

        let dtype = self.step_scalar_type(step_idx);
        dispatch_gpu_typed(cache, def, step_idx, scratch, dtype, self.def.precision)
    }
}

/// Dispatch a GPU-only input map to the tensor dispatch engine.
///
/// Uses the GPU-only dispatch path that accepts `&HashMap<&str, GpuSlice>`
/// directly, eliminating the per-call `HashMap<&str, DispatchInput<E>>`
/// allocation and reusing internal buffer/offset HashMaps via thread-local
/// `DispatchContext` (~1,470 HashMap allocs/fwd eliminated).
///
/// Part of #3079 (dispatch transient allocation elimination).
fn dispatch_gpu_typed(
    cache: &PipelineCache,
    def: &nn_dsl::TensorKernelDef,
    step_idx: usize,
    gpu_inputs: &HashMap<&str, GpuSlice>,
    dtype: nn_dsl::ir::ScalarType,
    contract: Option<PrecisionContract>,
) -> Result<GpuSlice> {
    use nn_dsl::ir::ScalarType;

    // Metal has no native bf16 compute — bf16 is stored as f16 on GPU.
    // Remap BF16 → F16 for MSL codegen, matching the old per-type dispatch.
    let effective_dtype = match dtype {
        ScalarType::BF16 => ScalarType::F16,
        other => other,
    };

    execute_tensor_dispatch_to_buffer_gpu_only(cache, def, effective_dtype, gpu_inputs, contract)
        .map_err(|e| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx,
                reason: e.to_string(),
            })
        })
}
