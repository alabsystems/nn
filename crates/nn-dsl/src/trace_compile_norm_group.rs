// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GroupNorm compilation helper for `trace_compile`.
//!
//! Extracted from `trace_compile_norm.rs` to keep files under 450 lines.
//! Decomposes GroupNorm into reshape -> instance_norm -> reshape -> affine.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::tensor_ir::TensorIRError;

use super::super::super::{resolve_input_shape, CompiledKernel, CompiledStep};
use super::super::{add_weight, build_op_with_weights};
use super::validate_eps;

/// Compile GroupNorm by decomposing into reshape -> instance_norm -> reshape -> affine.
///
/// GroupNorm(num_groups=G): reshape [B, C, *spatial] -> [B*G, C/G, *spatial],
/// apply instance_norm, reshape back, then apply per-channel affine.
///
/// For num_groups=1 with 2D input, uses the optimized `add_group_norm_g1` path.
pub(in crate::trace_compile) fn compile_group_norm(
    node: &TraceNode,
    graph: &ComputationGraph,
    num_groups: usize,
    eps: f64,
    weight: &WeightRef,
    bias: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = validate_eps(eps, "group_norm")?;
    let input_shape = resolve_input_shape(node, 0, graph)?;

    // GroupNorm requires at least 2D input: [C, *spatial] or [B, C, *spatial]
    if input_shape.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("GroupNorm with {}D input (need >= 2D)", input_shape.len()),
        });
    }

    let channels = if input_shape.len() == 2 {
        input_shape[0] // [C, T]
    } else {
        input_shape[1] // [B, C, *spatial]
    };

    if num_groups == 0 || channels % num_groups != 0 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!(
                "GroupNorm: channels ({channels}) not divisible by num_groups ({num_groups})"
            ),
        });
    }

    // For num_groups == 1 with 2D input, use the optimized builder path.
    if num_groups == 1 && input_shape.len() == 2 {
        let time_len = input_shape[1];
        let (def, weight_data) = build_op_with_weights("group_norm", node, |b, wd| {
            let input = b.add_input("input_0", input_shape);
            let eps_node = b.add_input("eps", &[1]);
            wd.insert(
                "eps".to_string(),
                WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
            );
            let gamma = add_weight(b, wd, "weight", weight);
            let beta = add_weight(b, wd, "bias", bias);
            b.add_group_norm_g1(input, eps_node, Some(gamma), Some(beta), channels, time_len)
        })?;
        return Ok(CompiledStep::Dispatch {
            kernel: CompiledKernel::new(def),
            weight_data,
            external_node_ids: super::super::graph_input_ids(node, 1),
        });
    }

    // General case: decompose GroupNorm using reshape + instance_norm + affine.
    let (def, weight_data) = build_op_with_weights("group_norm", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let eps_node = b.add_input("eps", &[1]);
        wd.insert(
            "eps".to_string(),
            WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
        );

        let channels_per_group = channels / num_groups;

        // Build reshaped shape: merge groups into batch dim
        let reshape_to = if input_shape.len() == 2 {
            // [C, T] -> [G, C/G, T]
            let t = input_shape[1];
            vec![num_groups, channels_per_group, t]
        } else {
            // [B, C, *spatial] -> [B*G, C/G, *spatial]
            let batch = input_shape[0];
            let spatial: &[usize] = &input_shape[2..];
            let mut shape = vec![batch * num_groups, channels_per_group];
            shape.extend_from_slice(spatial);
            shape
        };

        let reshaped = b.add_reshape(input, &reshape_to);

        // GroupNorm reduces over ALL dimensions within a group (C/G + spatial).
        // Flatten [B*G, C/G, *spatial] -> [B*G, C/G * prod(spatial)] so a single
        // instance_norm on axis 1 covers the entire group.
        let group_batch = reshape_to[0];
        let group_numel: usize = reshape_to[1..].iter().product();
        let flat_shape = vec![group_batch, group_numel];
        let flattened = b.add_reshape(reshaped, &flat_shape);

        let normed_flat = b.add_instance_norm(flattened, eps_node, 1, None, None, &flat_shape);

        // Reshape back through intermediate shape then to original
        let normed = b.add_reshape(normed_flat, &reshape_to);
        let back = b.add_reshape(normed, input_shape);

        // Per-channel affine: gamma * x + beta
        // Weight/bias are [C]. For [B, C, *spatial], reshape to [1, C, 1, 1, ...]
        // so broadcast_left aligns C to the channel dimension.
        let gamma = add_weight(b, wd, "weight", weight);
        let beta = add_weight(b, wd, "bias", bias);

        let out_shape = node.output_shape();
        let ndim = out_shape.len();

        if ndim == 2 {
            // [C, T]: broadcast_left of [C] -> [C, T] works directly.
            let gamma_bc = b.add_broadcast_left(gamma, out_shape);
            let scaled = b.add_binary_mul(back, gamma_bc, out_shape);
            let beta_bc = b.add_broadcast_left(beta, out_shape);
            b.add_binary_add(scaled, beta_bc, out_shape)
        } else {
            // [B, C, *spatial]: reshape [C] -> [1, C, 1, ...] then broadcast.
            let n_spatial = ndim - 2;
            let mut bc_shape = Vec::with_capacity(ndim);
            bc_shape.push(1);
            bc_shape.push(channels);
            bc_shape.extend(std::iter::repeat_n(1, n_spatial));

            let gamma_r = b.add_reshape(gamma, &bc_shape);
            let gamma_bc = b.add_broadcast_left(gamma_r, out_shape);
            let scaled = b.add_binary_mul(back, gamma_bc, out_shape);

            let beta_r = b.add_reshape(beta, &bc_shape);
            let beta_bc = b.add_broadcast_left(beta_r, out_shape);
            b.add_binary_add(scaled, beta_bc, out_shape)
        }
    })?;

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::super::graph_input_ids(node, 1),
    })
}
