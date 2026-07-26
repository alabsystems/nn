// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ICB runtime encoding: create `IndirectCommandBuffer` on first forward pass.
//!
//! Uses pre-compiled codegen from [`IcbSegment::step_codegen`] plus actual
//! runtime buffer bindings to encode all dispatch commands into an ICB.
//! Subsequent forward passes replay the ICB without re-encoding.
//!
//! D4 refactor: per-step buffer resolution (TensorNodeIds are kernel-local,
//! so a single global HashMap would cause cross-step ID collisions).
//!
//! Part of #3259 (D3, D4).

use std::collections::HashMap;

use nn_dsl::{DispatchStep, TensorNodeId};

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::dispatch_plan::DispatchMode;
use crate::error::MetalError;
use crate::kernel_source::KernelSource;

use super::analysis::IcbSegment;
use super::IndirectCommandBuffer;

/// Maximum buffer bindings per ICB command (inputs + output + constants).
const MAX_ICB_BINDINGS: usize = 16;

/// Per-step buffer binding: maps kernel-local TensorNodeIds to GPU slices.
///
/// Each compiled step has its own TensorNodeId namespace (starting from 0),
/// so bindings must be resolved per-step, not globally across a segment.
pub(crate) struct IcbStepBindings<'a> {
    /// Maps TensorNodeId → (MetalBuffer ref, byte_offset).
    pub(crate) nodes: HashMap<TensorNodeId, (&'a MetalBuffer, usize)>,
}

/// Encode a pre-compiled segment into an `IndirectCommandBuffer`.
///
/// `per_step_bindings[i]` corresponds to `segment.step_codegen[i]`.
/// Each binding map uses kernel-local TensorNodeIds for that step's IR.
pub(crate) fn encode_icb_from_segment(
    cache: &PipelineCache,
    segment: &IcbSegment,
    per_step_bindings: &[IcbStepBindings<'_>],
) -> Result<Option<IndirectCommandBuffer>, MetalError> {
    if per_step_bindings.len() != segment.step_codegen.len() {
        return Err(MetalError::DispatchFailed(format!(
            "ICB: binding count {} != codegen count {}",
            per_step_bindings.len(),
            segment.step_codegen.len(),
        )));
    }

    let total_commands = count_gpu_commands(segment);
    if total_commands == 0 {
        return Ok(None);
    }

    let device = cache.context().device();
    let mut icb = IndirectCommandBuffer::new(device, total_commands, MAX_ICB_BINDINGS)?;

    let mut cmd_idx = 0;
    for (step_offset, codegen) in segment.step_codegen.iter().enumerate() {
        let bindings = &per_step_bindings[step_offset];
        for step in &codegen.plan {
            match step {
                DispatchStep::Elementwise {
                    kernel_name,
                    inputs,
                    output,
                    total_elements,
                    ..
                } => {
                    encode_elementwise_cmd(
                        cache,
                        &mut icb,
                        cmd_idx,
                        &codegen.msl,
                        kernel_name,
                        inputs,
                        *output,
                        *total_elements,
                        &bindings.nodes,
                    )?;
                    cmd_idx += 1;
                }
                DispatchStep::Broadcast {
                    kernel_name,
                    input,
                    output,
                    total_elements,
                    ..
                } => {
                    encode_elementwise_cmd(
                        cache,
                        &mut icb,
                        cmd_idx,
                        &codegen.msl,
                        kernel_name,
                        &[*input],
                        *output,
                        *total_elements,
                        &bindings.nodes,
                    )?;
                    cmd_idx += 1;
                }
                DispatchStep::Reshape { .. } => {}
                _ => {
                    return Ok(None);
                }
            }
        }
    }

    Ok(Some(icb))
}

fn count_gpu_commands(segment: &IcbSegment) -> usize {
    let mut count = 0;
    for codegen in &segment.step_codegen {
        for step in &codegen.plan {
            match step {
                DispatchStep::Elementwise { .. } | DispatchStep::Broadcast { .. } => count += 1,
                DispatchStep::Reshape { .. } => {}
                _ => return 0,
            }
        }
    }
    count
}

#[allow(clippy::too_many_arguments)]
fn encode_elementwise_cmd(
    cache: &PipelineCache,
    icb: &mut IndirectCommandBuffer,
    cmd_idx: usize,
    msl: &str,
    kernel_name: &str,
    inputs: &[TensorNodeId],
    output: TensorNodeId,
    total_elements: usize,
    node_map: &HashMap<TensorNodeId, (&MetalBuffer, usize)>,
) -> Result<(), MetalError> {
    let source = KernelSource::new(msl, kernel_name);
    let pipeline = cache.get_or_compile_icb(&source)?;

    let total_u32 = u32::try_from(total_elements)
        .map_err(|_| MetalError::DispatchSizeOverflow(total_elements))?;
    let plan = DispatchMode::Elementwise { total: total_u32 }.plan()?;

    let mut buf_bindings: Vec<(usize, &MetalBuffer, usize)> = Vec::with_capacity(inputs.len() + 1);
    for (i, &node_id) in inputs.iter().enumerate() {
        let &(buf, offset) = node_map.get(&node_id).ok_or_else(|| {
            MetalError::DispatchFailed(format!("ICB: missing buffer {node_id:?}"))
        })?;
        buf_bindings.push((i, buf, offset));
    }
    let &(out_buf, out_offset) = node_map
        .get(&output)
        .ok_or_else(|| MetalError::DispatchFailed(format!("ICB: missing output {output:?}")))?;
    buf_bindings.push((inputs.len(), out_buf, out_offset));

    let grid = plan.grid();
    let threads = plan.threads();
    let tg_grid = if plan.use_threadgroups() {
        grid
    } else {
        [
            grid[0].div_ceil(threads[0].max(1)),
            grid[1].div_ceil(threads[1].max(1)),
            grid[2].div_ceil(threads[2].max(1)),
        ]
    };
    icb.encode_command(cmd_idx, &pipeline, &buf_bindings, tg_grid, threads, false)?;
    Ok(())
}

/// Update variable buffer bindings on an existing ICB for a new forward pass.
pub(crate) fn update_icb_bindings(
    icb: &IndirectCommandBuffer,
    segment: &IcbSegment,
    per_step_bindings: &[IcbStepBindings<'_>],
) -> Result<(), MetalError> {
    let mut cmd_idx = 0;
    for (step_offset, codegen) in segment.step_codegen.iter().enumerate() {
        let bindings = match per_step_bindings.get(step_offset) {
            Some(b) => b,
            None => return Ok(()),
        };
        for step in &codegen.plan {
            match step {
                DispatchStep::Elementwise { output, inputs, .. } => {
                    for (i, &node_id) in inputs.iter().enumerate() {
                        if let Some(&(buf, offset)) = bindings.nodes.get(&node_id) {
                            icb.update_buffer(cmd_idx, i, buf, offset)?;
                        }
                    }
                    if let Some(&(buf, offset)) = bindings.nodes.get(output) {
                        icb.update_buffer(cmd_idx, inputs.len(), buf, offset)?;
                    }
                    cmd_idx += 1;
                }
                DispatchStep::Broadcast { output, input, .. } => {
                    if let Some(&(buf, offset)) = bindings.nodes.get(input) {
                        icb.update_buffer(cmd_idx, 0, buf, offset)?;
                    }
                    if let Some(&(buf, offset)) = bindings.nodes.get(output) {
                        icb.update_buffer(cmd_idx, 1, buf, offset)?;
                    }
                    cmd_idx += 1;
                }
                DispatchStep::Reshape { .. } => {}
                _ => return Ok(()),
            }
        }
    }
    Ok(())
}
