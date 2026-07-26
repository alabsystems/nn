// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for tensor-level kernels (reductions, broadcasts).
//! Dispatch planning + types for threadgroup-parallel reduction kernels.
//! MSL string emission lives in [`super::codegen_msl_tensor_emit`].

use crate::ir::ScalarType;
use crate::tensor_ir::{TensorKernelDef, TensorNodeId};

#[path = "codegen_msl_tensor_error.rs"]
mod error;
pub use error::TensorMSLCodegenError;

#[path = "dispatch_step.rs"]
mod step;
// Items re-exported as `pub` from lib.rs — `pub` visibility required here.
#[allow(unreachable_pub)]
pub use step::{
    tiled_transpose_2d_params, BinaryBroadcastInfo, BroadcastSide, Conv1dParams, Conv2dParams,
    ConvTranspose1dParams, DispatchStep, SimdgroupLinearParams, SimdgroupMatMulParams,
    TiledLinearParams, TiledMatMulParams, TILED_GEMM_TILE, TILED_TRANSPOSE_TILE_SIZE,
};

/// Default threadgroup size for reduction kernels.
///
/// 256 is the Metal maximum and works on all Apple GPUs. Fixed for now;
/// the design doc notes this can be made configurable later if needed.
pub(crate) const REDUCE_THREADGROUP_SIZE: usize = 256;
const _: () = assert!(
    REDUCE_THREADGROUP_SIZE.is_power_of_two(),
    "Tree reduction requires power-of-2 threadgroup size"
);

// MSL emission functions (emit_reduce_kernel, emit_broadcast_kernel,
// emit_tensor_msl, emit_tensor_msl_with_contract) live in
// codegen_msl_tensor_emit.rs and are re-exported via lib.rs.

#[path = "codegen_msl_tensor_expand.rs"]
mod expand;

#[path = "codegen_msl_tensor_ops.rs"]
mod ops;

#[path = "codegen_msl_tensor_dispatch_activation.rs"]
mod activation;

#[path = "codegen_msl_tensor_dispatch.rs"]
mod dispatch;

/// Look up a node's shape by id, returning `InvalidNodeRef` on out-of-bounds.
fn node_shape(
    kernel: &TensorKernelDef,
    id: TensorNodeId,
) -> Result<&[usize], TensorMSLCodegenError> {
    Ok(&kernel
        .nodes
        .get(id.index())
        .ok_or(TensorMSLCodegenError::InvalidNodeRef {
            node_id: id,
            graph_len: kernel.nodes.len(),
        })?
        .shape)
}

/// Compute total element count from a shape, returning `ShapeProductOverflow` on overflow.
fn shape_total(shape: &[usize]) -> Result<usize, TensorMSLCodegenError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorMSLCodegenError::ShapeProductOverflow {
            shape: shape.to_vec(),
        })
}

/// Build a Metal dispatch plan from a tensor kernel definition.
///
/// Walks the tensor IR in topological order and emits one [`DispatchStep`] per
/// kernel launch (element-wise, reduce, or broadcast).
///
/// Returns `(steps, effective_output)` where `effective_output` is the output
/// node ID after norm/attention expansion. When no expansion occurs, this equals
/// `kernel.output`. When composite ops are expanded, IDs are remapped and the
/// effective output may differ from the original.
#[must_use = "returns a Result that may contain an error"]
pub fn build_dispatch_plan(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
) -> Result<(Vec<DispatchStep>, TensorNodeId), TensorMSLCodegenError> {
    let (steps, effective_output, _expanded) = build_dispatch_plan_full(kernel, dtype)?;
    Ok((steps, effective_output))
}

/// Build dispatch plan and return the expanded kernel.
///
/// MSL emission and GPU dispatch need the expanded kernel to look up node
/// shapes in the post-expansion graph (expanded node IDs differ from original
/// ones when composite ops like LayerNorm, Attention are decomposed).
pub fn build_dispatch_plan_full(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
) -> Result<(Vec<DispatchStep>, TensorNodeId, TensorKernelDef), TensorMSLCodegenError> {
    kernel.validate()?;

    // Expand composite ops (norms, attention, LSTM) into decomposed primitives
    // before dispatch planning. This closes the topology divergence between
    // the verified graph (native ops) and the executed graph (decomposed
    // ops). See #667, #812, #2306.
    let effective = if expand::has_norm_ops(kernel)
        || expand::has_attention_ops(kernel)
        || expand::has_lstm_ops(kernel)
    {
        let expanded = expand::expand_norm_ops(kernel);
        expanded.validate()?;
        expanded
    } else {
        kernel.clone()
    };

    let effective_output = effective.output;

    let mut steps = Vec::new();

    for node in &effective.nodes {
        if let Some(step) = dispatch::build_step_for_node(&effective, node, dtype)? {
            steps.push(step);
        }
    }

    // Peephole: fuse Broadcast + BinaryAdd/BinaryMul pairs into broadcast-aware
    // binary ops. Saves ~8-12 dispatches in norm expansions (#1815 Tier 4 D2).
    fuse_broadcast_binary_ops(&mut steps);

    Ok((steps, effective_output, effective))
}

/// Fuse Broadcast + BinaryAdd/BinaryMul pairs into broadcast-aware binary ops.
///
/// When a Broadcast output feeds into exactly one BinaryAdd or BinaryMul as
/// either left or right operand, the broadcast is absorbed into the binary op
/// kernel (modular indexing replaces the separate copy). The Broadcast step
/// becomes a no-op Reshape.
#[allow(clippy::type_complexity)]
fn fuse_broadcast_binary_ops(plan: &mut [DispatchStep]) {
    use crate::tensor_ir::TensorNodeId;

    // Collect broadcast steps: (index, output_id, input_id, input_shape, output_shape, alignment).
    let broadcasts: Vec<(
        usize,
        TensorNodeId,
        TensorNodeId,
        Vec<usize>,
        Vec<usize>,
        crate::tensor_ir::BroadcastAlignment,
    )> = plan
        .iter()
        .enumerate()
        .filter_map(|(i, step)| {
            if let DispatchStep::Broadcast {
                input,
                output,
                input_shape,
                output_shape,
                alignment,
                ..
            } = step
            {
                Some((
                    i,
                    *output,
                    *input,
                    input_shape.clone(),
                    output_shape.clone(),
                    *alignment,
                ))
            } else {
                None
            }
        })
        .collect();

    for (bcast_idx, bcast_output, bcast_input, input_shape, output_shape, alignment) in broadcasts {
        // Count total consumers of this broadcast output.
        let consumer_count = plan.iter().filter(|s| s.uses_input(bcast_output)).count();
        if consumer_count != 1 {
            continue; // Only fuse single-consumer broadcasts
        }

        // Find the consumer: must be BinaryAdd or BinaryMul.
        let consumer = plan.iter().enumerate().find_map(|(i, step)| match step {
            DispatchStep::BinaryAdd { left, right, .. } => {
                if *left == bcast_output {
                    Some((i, BroadcastSide::Left))
                } else if *right == bcast_output {
                    Some((i, BroadcastSide::Right))
                } else {
                    None
                }
            }
            DispatchStep::BinaryMul { left, right, .. } => {
                if *left == bcast_output {
                    Some((i, BroadcastSide::Left))
                } else if *right == bcast_output {
                    Some((i, BroadcastSide::Right))
                } else {
                    None
                }
            }
            _ => None,
        });

        let Some((consumer_idx, side)) = consumer else {
            continue;
        };

        let info = BinaryBroadcastInfo {
            side,
            input_shape,
            output_shape,
            alignment,
        };

        // Set broadcast info and rewire input to the original small tensor.
        match &mut plan[consumer_idx] {
            DispatchStep::BinaryAdd {
                left,
                right,
                broadcast,
                ..
            } => {
                *broadcast = Some(info);
                match side {
                    BroadcastSide::Left => *left = bcast_input,
                    BroadcastSide::Right => *right = bcast_input,
                }
            }
            DispatchStep::BinaryMul {
                left,
                right,
                broadcast,
                ..
            } => {
                *broadcast = Some(info);
                match side {
                    BroadcastSide::Left => *left = bcast_input,
                    BroadcastSide::Right => *right = bcast_input,
                }
            }
            _ => continue,
        }

        // Replace the Broadcast step with a no-op Reshape.
        plan[bcast_idx] = DispatchStep::Reshape {
            input: bcast_input,
            output: bcast_output,
        };
    }
}

#[cfg(kani)]
#[path = "codegen_kani_dispatch.rs"]
mod kani_dispatch_proofs;

#[cfg(kani)]
#[path = "kani_codegen_msl_tensor_dispatch.rs"]
mod kani_codegen_msl_tensor_dispatch;

#[cfg(kani)]
#[path = "kani_codegen_msl_tensor_dispatch_ext.rs"]
mod kani_codegen_msl_tensor_dispatch_ext;

#[cfg(test)]
#[path = "codegen_msl_tensor_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "codegen_msl_tensor_tests_conv.rs"]
mod conv_tests;

#[cfg(test)]
#[path = "codegen_msl_tensor_tests_emit.rs"]
mod emit_tests;
#[cfg(test)]
#[path = "codegen_msl_tensor_tests_norm.rs"]
mod norm_tests;
