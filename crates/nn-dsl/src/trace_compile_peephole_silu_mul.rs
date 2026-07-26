// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pass 14: fuse Silu + Mul → SiluMul Dispatch.
//!
//! Detects the pattern:
//!   Step i:   Dispatch { kernel: "silu" }
//!   Step i+1: Dispatch { kernel: "mul" }
//! where the mul consumes the silu's output (adjacency + single-use).
//!
//! Replaces both with a single Dispatch using a fused `silu_mul` kernel
//! that computes `silu(gate) * up` in one pass. The fused kernel goes
//! through normal IR → MSL codegen (not a NativeOp).
//!
//! This pattern occurs in every SwiGLU MLP block (Qwen3, GLM5, etc.).
//! Saves 1 dispatch per transformer layer.
//!
//! Part of #3521.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::silu_mul::build_silu_mul_kernel;
use crate::tensor_block_builder::TensorBlockBuilder;

use super::super::{CompiledKernel, CompiledStep};

/// Fuse adjacent Silu + Mul pairs into single SiluMul dispatches.
///
/// Scans for pairs where:
/// 1. Step i is a Dispatch with kernel name "silu" (single-use output)
/// 2. Step i+1 is a Dispatch with kernel name "mul"
/// 3. The mul step consumes the silu step's output (graph topology check)
///
/// The silu step becomes IdentityPassthrough and the mul step becomes
/// a fused silu_mul Dispatch with `external_node_ids` set to [gate, up].
pub(super) fn fuse_silu_mul(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
    graph: &ComputationGraph,
) {
    let len = steps.len();
    if len < 2 {
        return;
    }

    let graph_nodes = graph.nodes();

    let mut i = 0;
    while i + 1 < len {
        if try_fuse_silu_mul(steps, i, use_counts, graph_nodes) {
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// Try to fuse steps[i] (silu) with steps[i+1] (mul).
fn try_fuse_silu_mul(
    steps: &mut [CompiledStep],
    i: usize,
    use_counts: &[usize],
    graph_nodes: &[nn_core::dyn_tensor::trace::TraceNode],
) -> bool {
    // Step[i] must be Dispatch with kernel name "silu".
    let silu_shape = match &steps[i] {
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "silu" => {
            kernel.output_shape().map(<[usize]>::to_vec)
        }
        _ => None,
    };
    let silu_shape = match silu_shape {
        Some(shape) => shape,
        None => return false,
    };

    // Fan-out: silu output must have exactly 1 consumer.
    if use_counts.get(i).copied().unwrap_or(0) != 1 {
        return false;
    }

    // Step[i+1] must be Dispatch with kernel name "mul".
    let is_mul = matches!(
        &steps[i + 1],
        CompiledStep::Dispatch { kernel, .. } if kernel.name() == "mul"
    );
    if !is_mul {
        return false;
    }

    // Use graph topology to verify mul consumes silu output and find
    // the gate and up input node IDs.
    let silu_node = match graph_nodes.get(i) {
        Some(n) => n,
        None => return false,
    };
    let mul_node = match graph_nodes.get(i + 1) {
        Some(n) => n,
        None => return false,
    };

    let silu_id = silu_node.id();
    let mul_inputs = mul_node.inputs();

    // Mul must have exactly 2 inputs, one of which is the silu output.
    if mul_inputs.len() != 2 {
        return false;
    }
    let silu_pos = if mul_inputs[0] == silu_id {
        Some(0)
    } else if mul_inputs[1] == silu_id {
        Some(1)
    } else {
        None
    };
    let silu_pos = match silu_pos {
        Some(p) => p,
        None => return false,
    };

    // "up" is the mul input that is NOT the silu output.
    let up_id = mul_inputs[1 - silu_pos];

    // "gate" is the silu's own input.
    let silu_inputs = silu_node.inputs();
    if silu_inputs.is_empty() {
        return false;
    }
    let gate_id = silu_inputs[0];

    // Build the fused silu_mul kernel via TensorBlockBuilder.
    let scalar_kernel = match build_silu_mul_kernel() {
        Ok(k) => k,
        Err(_) => return false,
    };

    let mut b = TensorBlockBuilder::new("silu_mul");
    let gate_in = b.add_input("gate", &silu_shape);
    let up_in = b.add_input("up", &silu_shape);
    let fused_out = b.add_elementwise(scalar_kernel, &[gate_in, up_in], &silu_shape);
    let fused_def = match b.build(fused_out) {
        Ok(def) => def,
        Err(_) => return false,
    };

    // Replace silu with IdentityPassthrough.
    steps[i] = CompiledStep::IdentityPassthrough;

    // Replace mul with fused silu_mul Dispatch.
    // external_node_ids = [gate, up] overrides graph edge resolution
    // to ensure the kernel receives inputs in the correct order.
    steps[i + 1] = CompiledStep::Dispatch {
        kernel: CompiledKernel::new(fused_def),
        weight_data: HashMap::new(),
        external_node_ids: Some(vec![gate_id, up_id]),
    };

    true
}

#[cfg(test)]
#[path = "trace_compile_peephole_silu_mul_tests.rs"]
mod tests;
