// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ICB replay wiring for the compiled model execution loop.
//!
//! Bridges step-indexed buffer arrays (`Vec<Option<GpuSlice>>`) to
//! kernel-local `TensorNodeId`-keyed buffer maps for ICB encoding and replay.
//!
//! Extracted from `compiled_model_execute_steps.rs` (500-line compliance).
//! Part of #3259 (D4).

use std::collections::HashMap;

use nn_core::Result;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::gpu_slice::GpuSlice;

// Module hierarchy: icb_replay ⊂ steps ⊂ execute ⊂ compiled_model.
// `super::super::super` reaches compiled_model where icb and CompiledModel live.
use super::super::super::icb::{
    encode_icb_from_segment, update_icb_bindings, IcbSegment, IcbStepBindings,
    IndirectCommandBuffer,
};
use super::super::super::{CompiledModel, CompiledModelError};

impl CompiledModel {
    /// Attempt to replay (or first-pass encode) an ICB for the segment.
    ///
    /// On success: populates `buffers[step_idx]` for all steps in the segment,
    /// returns `Ok(true)` to skip those steps in the main loop.
    /// On failure: returns `Ok(false)` to fall back to normal per-step dispatch.
    ///
    /// First forward pass: builds per-step bindings, encodes the ICB, caches it.
    /// Subsequent passes: updates variable buffer bindings, replays the cached ICB.
    ///
    /// Part of #3259 (D4).
    pub(super) fn try_replay_icb(
        &self,
        cache: &PipelineCache,
        seg_idx: usize,
        seg: &IcbSegment,
        buffers: &mut [Option<GpuSlice>],
        planned_buf: Option<&MetalBuffer>,
    ) -> Result<bool> {
        // ICB requires the planned buffer for stable output allocation.
        let pb = match planned_buf {
            Some(b) => b,
            None => return Ok(false),
        };

        // Verify all segment steps have planned offsets (required for stable
        // output buffer bindings across forward passes).
        let step_offsets = &self.def.buffer_plan.step_offsets;
        for step_idx in seg.start..=seg.end {
            if step_offsets.get(step_idx).and_then(|o| *o).is_none() {
                return Ok(false);
            }
        }

        // Pre-allocate output buffers in the planned buffer.
        for (step_idx, buf) in buffers
            .iter_mut()
            .enumerate()
            .take(seg.end + 1)
            .skip(seg.start)
        {
            if let Some(Some(offset)) = step_offsets.get(step_idx) {
                *buf = Some(GpuSlice::new(pb.alias(), *offset));
            }
        }

        // Build per-step bindings: map kernel-local TensorNodeIds to actual
        // Metal buffers (weights, activation inputs, planned buffer outputs).
        let per_step_bindings = match self.build_icb_step_bindings(seg, buffers, pb) {
            Some(b) => b,
            None => return Ok(false),
        };

        let icbs = self.cached_icbs.borrow();
        let has_cached = matches!(icbs.get(seg_idx), Some(Some(_)));
        drop(icbs);

        if has_cached {
            // Subsequent pass: update variable bindings and replay.
            let icbs = self.cached_icbs.borrow();
            if let Some(Some(icb)) = icbs.get(seg_idx) {
                update_icb_bindings(icb, seg, &per_step_bindings)
                    .map_err(|e| icb_err(seg.start, "update bindings", e))?;
                execute_icb(self, icb, seg, buffers)?;
            }
            return Ok(true);
        }

        // First pass: encode ICB from pre-compiled codegen + runtime buffers.
        let encoded = encode_icb_from_segment(cache, seg, &per_step_bindings)
            .map_err(|e| icb_err(seg.start, "encode", e))?;

        let icb = match encoded {
            Some(icb) => icb,
            None => return Ok(false), // Unsupported step type in segment.
        };

        execute_icb(self, &icb, seg, buffers)?;

        // Cache the ICB for subsequent passes.
        let mut icbs = self.cached_icbs.borrow_mut();
        if seg_idx < icbs.len() {
            icbs[seg_idx] = Some(icb);
        }

        Ok(true)
    }

    /// Build per-step `IcbStepBindings` for all steps in a segment.
    ///
    /// Maps kernel-local TensorNodeIds to actual Metal buffers by matching
    /// input names from the expanded kernel def against weight buffers and
    /// graph edges. Returns `None` if any binding cannot be resolved.
    fn build_icb_step_bindings<'a>(
        &'a self,
        seg: &IcbSegment,
        buffers: &'a [Option<GpuSlice>],
        planned_buf: &'a MetalBuffer,
    ) -> Option<Vec<IcbStepBindings<'a>>> {
        use nn_dsl::TensorOpKind;

        let step_offsets = &self.def.buffer_plan.step_offsets;
        let mut result = Vec::with_capacity(seg.step_codegen.len());

        for (offset, codegen) in seg.step_codegen.iter().enumerate() {
            let step_idx = seg.start + offset;
            let mut nodes = HashMap::new();

            let def = &codegen.expanded;
            let meta = self.def.step_metas.get(step_idx)?;
            let step_weights = &self.def.weight_buffers[step_idx];
            let input_names = &self.def.input_name_cache[step_idx];
            let mut graph_input_idx = 0;

            for input_name in input_names {
                // Find the TensorNodeId for this input name in the kernel IR.
                let node_id = def.nodes.iter().find_map(|node| match &node.kind {
                    TensorOpKind::Input { name, .. } if name == input_name => Some(node.id),
                    _ => None,
                })?;

                if let Some(weight_buf) = step_weights.get(input_name.as_str()) {
                    nodes.insert(node_id, (weight_buf, 0usize));
                } else if let Some(&src_step) = meta.edges.get(graph_input_idx) {
                    let slice = buffers[src_step].as_ref()?;
                    nodes.insert(node_id, (slice.buffer(), slice.byte_offset()));
                    graph_input_idx += 1;
                } else {
                    return None;
                }
            }

            // Map output TensorNodeId to planned buffer at step's offset.
            let planned_offset = step_offsets.get(step_idx).and_then(|o| *o)?;
            nodes.insert(codegen.effective_output, (planned_buf, planned_offset));

            result.push(IcbStepBindings { nodes });
        }

        Some(result)
    }

    /// Collect all unique Metal buffers referenced by an ICB segment.
    pub(super) fn collect_icb_resources(
        &self,
        seg: &IcbSegment,
        buffers: &[Option<GpuSlice>],
    ) -> Vec<MetalBuffer> {
        let mut seen = std::collections::HashSet::new();
        let mut resources = Vec::new();

        // Output + input buffers from the segment.
        for step_idx in seg.start..=seg.end {
            if let Some(Some(slice)) = buffers.get(step_idx) {
                let ptr = std::ptr::from_ref(slice.buffer().inner()) as usize;
                if seen.insert(ptr) {
                    resources.push(slice.buffer().alias());
                }
            }
        }

        // Input buffers from edges outside the segment.
        for (offset, _codegen) in seg.step_codegen.iter().enumerate() {
            let step_idx = seg.start + offset;
            if let Some(meta) = self.def.step_metas.get(step_idx) {
                for &src_step in &meta.edges {
                    if src_step < seg.start || src_step > seg.end {
                        if let Some(Some(slice)) = buffers.get(src_step) {
                            let ptr = std::ptr::from_ref(slice.buffer().inner()) as usize;
                            if seen.insert(ptr) {
                                resources.push(slice.buffer().alias());
                            }
                        }
                    }
                }
            }
        }

        // Weight buffers.
        for step_idx in seg.start..=seg.end {
            for buf in self.def.weight_buffers[step_idx].values() {
                let ptr = std::ptr::from_ref(buf.inner()) as usize;
                if seen.insert(ptr) {
                    resources.push(buf.alias());
                }
            }
        }

        resources
    }
}

/// Execute an ICB: declare resources and dispatch via the lazy command batch.
fn execute_icb(
    model: &CompiledModel,
    icb: &IndirectCommandBuffer,
    seg: &IcbSegment,
    buffers: &[Option<GpuSlice>],
) -> Result<()> {
    let resources = model.collect_icb_resources(seg, buffers);
    let resource_refs: Vec<&MetalBuffer> = resources.iter().collect();

    // Use the lazy batch infrastructure (same as normal GPU dispatches).
    let result =
        crate::gpu_scope::encode_custom_dispatch(|batch| icb.execute(batch, &resource_refs))?;
    result.map_err(|e| icb_err(seg.start, "execute", e))?;
    Ok(())
}

/// Wrap Metal errors as CompiledModelError::DispatchFailed.
fn icb_err(step_idx: usize, phase: &str, err: crate::error::MetalError) -> nn_core::TensorError {
    nn_core::TensorError::from(CompiledModelError::DispatchFailed {
        step_idx,
        reason: format!("ICB {phase}: {err}"),
    })
}
