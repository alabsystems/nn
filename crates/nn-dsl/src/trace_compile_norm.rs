// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Normalization compilation helpers for `trace_compile`.
//!
//! Extracted from `trace_compile_ops.rs` to keep files under 500 lines.
//! These functions lower LayerNorm, RmsNorm, InstanceNorm, and BatchNorm
//! `TraceOp` variants into `TensorKernelDef` dispatch plans via
//! `TensorBlockBuilder`.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::tensor_ir::TensorIRError;

use super::super::{resolve_input_shape, CompiledKernel, CompiledStep};
use super::{add_weight, build_op_with_weights};

#[path = "trace_compile_norm_group.rs"]
mod group;
pub(in crate::trace_compile) use group::compile_group_norm;

// -- Helpers ------------------------------------------------------------------

/// Validate that an eps value is finite and positive after f64 -> f32 cast.
///
/// Per design doc convention, all constant-fold paths must check finiteness.
/// The f64 -> f32 cast can produce Inf for values > f32::MAX or 0.0 for
/// subnormal f64 values.
pub(super) fn validate_eps(eps: f64, op_name: &str) -> Result<f32, TensorIRError> {
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 <= 0.0 {
        return Err(TensorIRError::NonFiniteConstant {
            name: format!("{op_name}.eps"),
            value: eps,
        });
    }
    Ok(eps_f32)
}

// -- Normalization ------------------------------------------------------------

pub(in crate::trace_compile) fn compile_layer_norm(
    node: &TraceNode,
    graph: &ComputationGraph,
    eps: f64,
    weight: &WeightRef,
    bias: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = validate_eps(eps, "layer_norm")?;
    let input_shape = resolve_input_shape(node, 0, graph)?;

    // Emit a NativeOp for rank >= 2 input (the common case: [B, T, C] or [B, C]).
    // Uses the decomposed GPU dispatch path (`gpu_layer_norm`).
    if input_shape.len() >= 2 {
        let hidden_dim = *input_shape.last().unwrap_or(&0);
        let mut weight_data = std::collections::HashMap::new();
        weight_data.insert("weight".to_string(), weight.clone());
        weight_data.insert("bias".to_string(), bias.clone());
        return Ok(CompiledStep::NativeOp {
            op: super::super::NativeOpKind::LayerNorm {
                eps: eps_f32,
                input_shape: input_shape.to_vec(),
                hidden_dim,
            },
            weight_data,
        });
    }

    // Fallback for rank < 2 (unusual): use the IR decomposition path.
    let ndim = input_shape.len();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };
    let (def, weight_data) = build_op_with_weights("layer_norm", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let eps_node = b.add_input("eps", &[1]);
        wd.insert(
            "eps".to_string(),
            WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
        );
        let w = add_weight(b, wd, "weight", weight);
        let bi = add_weight(b, wd, "bias", bias);
        b.add_layer_norm(input, eps_node, axis, w, bi, node.output_shape())
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

pub(in crate::trace_compile) fn compile_rms_norm(
    node: &TraceNode,
    graph: &ComputationGraph,
    eps: f64,
    weight: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = validate_eps(eps, "rms_norm")?;
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let ndim = input_shape.len();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };
    let (def, weight_data) = build_op_with_weights("rms_norm", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let eps_node = b.add_input("eps", &[1]);
        wd.insert(
            "eps".to_string(),
            WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
        );
        let w = add_weight(b, wd, "weight", weight);
        b.add_rms_norm(input, eps_node, axis, w, node.output_shape())
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

pub(in crate::trace_compile) fn compile_instance_norm(
    node: &TraceNode,
    graph: &ComputationGraph,
    eps: f64,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = validate_eps(eps, "instance_norm")?;
    let input_shape = resolve_input_shape(node, 0, graph)?;

    // Emit a NativeOp for rank >= 3 input (the common case: [B, C, T]).
    // The fused kernel uses threadgroup parallel reduction — single Metal
    // dispatch instead of the 7-dispatch IR decomposition. Part of #2472.
    if input_shape.len() >= 3 {
        return Ok(CompiledStep::NativeOp {
            op: super::super::NativeOpKind::InstanceNorm {
                eps: eps_f32,
                input_shape: input_shape.to_vec(),
            },
            weight_data: std::collections::HashMap::new(),
        });
    }

    // Fallback for rank < 3 (unusual): use the IR decomposition path.
    let ndim = input_shape.len();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };
    let (def, weight_data) = build_op_with_weights("instance_norm", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let eps_node = b.add_input("eps", &[1]);
        wd.insert(
            "eps".to_string(),
            WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
        );
        b.add_instance_norm(input, eps_node, axis, None, None, node.output_shape())
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

/// Compile AdaLayerNorm: `(1 + gamma) * LayerNorm(x, weight, bias, eps) + beta`.
///
/// Tensor inputs from trace graph: `[x, gamma, beta]`.
///
/// For rank >= 3 (the common `[B, T, C]` case): emits a `NativeOp` with a
/// fused Metal kernel — single dispatch instead of ~6-7. Part of #2482.
///
/// For rank < 3 (unusual): falls back to IR decomposition via
/// `TensorBlockBuilder`.
pub(in crate::trace_compile) fn compile_ada_layer_norm(
    node: &TraceNode,
    graph: &ComputationGraph,
    eps: f64,
    norm_weight: &WeightRef,
    norm_bias: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = validate_eps(eps, "ada_layer_norm")?;
    let x_shape = resolve_input_shape(node, 0, graph)?;

    // Emit a NativeOp for rank >= 3 (the common case: [B, T, C]).
    // The fused kernel uses threadgroup parallel reduction — single Metal
    // dispatch instead of the ~6-7 dispatch IR decomposition. Part of #2482.
    if x_shape.len() >= 3 {
        let hidden_dim = *x_shape.last().unwrap_or(&0);
        let mut weight_data = std::collections::HashMap::new();
        weight_data.insert("norm_weight".to_string(), norm_weight.clone());
        weight_data.insert("norm_bias".to_string(), norm_bias.clone());
        return Ok(CompiledStep::NativeOp {
            op: super::super::NativeOpKind::AdaLayerNorm {
                eps: eps_f32,
                input_shape: x_shape.to_vec(),
                hidden_dim,
            },
            weight_data,
        });
    }

    // Fallback for rank < 3 (unusual): use the IR decomposition path.
    let gamma_shape = resolve_input_shape(node, 1, graph)?;
    let beta_shape = resolve_input_shape(node, 2, graph)?;
    let ndim = x_shape.len();
    let axis = if ndim > 0 { ndim - 1 } else { 0 };
    let output_shape = node.output_shape();

    let (def, weight_data) = build_op_with_weights("ada_layer_norm", node, |b, wd| {
        let x = b.add_input("input_0", x_shape);
        let gamma = b.add_input("input_1", gamma_shape);
        let beta = b.add_input("input_2", beta_shape);

        // Scalar weights
        let eps_node = b.add_input("eps", &[1]);
        wd.insert(
            "eps".to_string(),
            WeightRef::new(vec![eps_f32], vec![1]).expect("valid eps scalar"),
        );

        // LayerNorm weights
        let w = add_weight(b, wd, "norm_weight", norm_weight);
        let bi = add_weight(b, wd, "norm_bias", norm_bias);

        // Step 1: LayerNorm(x, weight, bias, eps) → affine_normed [B, T, C]
        let affine_normed = b.add_layer_norm(x, eps_node, axis, w, bi, output_shape);

        // Step 2: (1 + gamma) * affine_normed + beta → output [B, T, C]
        let ones_node = b.add_input("ones", &[1]);
        wd.insert(
            "ones".to_string(),
            WeightRef::new(vec![1.0f32], vec![1]).expect("valid scalar"),
        );
        let ones_bc = b.add_broadcast(ones_node, gamma_shape);
        let scale = b.add_binary_add(ones_bc, gamma, gamma_shape);
        let scale_bc = b.add_broadcast(scale, output_shape);
        let scaled = b.add_binary_mul(affine_normed, scale_bc, output_shape);
        let beta_bc = b.add_broadcast(beta, output_shape);
        b.add_binary_add(scaled, beta_bc, output_shape)
    })?;

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 3),
    })
}

/// Compile BatchNorm (eval mode) by precomputing per-channel scale and offset.
///
/// `output = (x - mean) / sqrt(var + eps) * weight + bias`
/// Precomputed: `scale = weight / sqrt(var + eps)`, `offset = bias - mean * scale`.
/// Then: `output = x * scale + offset` (per-channel broadcast).
pub(in crate::trace_compile) fn compile_batch_norm(
    node: &TraceNode,
    graph: &ComputationGraph,
    eps: f64,
    weight: &WeightRef,
    bias: &WeightRef,
    running_mean: &WeightRef,
    running_var: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = validate_eps(eps, "batch_norm")?;
    let input_shape = resolve_input_shape(node, 0, graph)?;

    if input_shape.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("BatchNorm with {}D input (need >= 2D)", input_shape.len()),
        });
    }

    let channels = input_shape[1];

    // Precompute scale and offset from static weights.
    let w_data = weight.data();
    let b_data = bias.data();
    let mean_data = running_mean.data();
    let var_data = running_var.data();

    let mut scale = vec![0.0_f32; channels];
    let mut offset = vec![0.0_f32; channels];
    for c in 0..channels {
        let inv_std = 1.0 / (var_data[c] + eps_f32).sqrt();
        scale[c] = w_data[c] * inv_std;
        offset[c] = b_data[c] - mean_data[c] * scale[c];
        if !scale[c].is_finite() || !offset[c].is_finite() {
            return Err(TensorIRError::NonFiniteConstant {
                name: format!("batch_norm precomputed channel {c}"),
                value: f64::from(scale[c]),
            });
        }
    }

    let scale_ref =
        WeightRef::new(scale, vec![channels]).map_err(|_| TensorIRError::UnsupportedTraceOp {
            name: "batch_norm: invalid precomputed scale".into(),
        })?;
    let offset_ref =
        WeightRef::new(offset, vec![channels]).map_err(|_| TensorIRError::UnsupportedTraceOp {
            name: "batch_norm: invalid precomputed offset".into(),
        })?;

    let (def, weight_data) = build_op_with_weights("batch_norm", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let s = add_weight(b, wd, "bn_scale", &scale_ref);
        let o = add_weight(b, wd, "bn_offset", &offset_ref);

        let out_shape = node.output_shape();
        let ndim = out_shape.len();

        // Reshape [C] → [1, C, 1, ...] for per-channel broadcast.
        let n_spatial = ndim - 2;
        let mut bc_shape = Vec::with_capacity(ndim);
        bc_shape.push(1);
        bc_shape.push(channels);
        bc_shape.extend(std::iter::repeat_n(1, n_spatial));

        let s_r = b.add_reshape(s, &bc_shape);
        let s_bc = b.add_broadcast_left(s_r, out_shape);
        let scaled = b.add_binary_mul(input, s_bc, out_shape);

        let o_r = b.add_reshape(o, &bc_shape);
        let o_bc = b.add_broadcast_left(o_r, out_shape);
        b.add_binary_add(scaled, o_bc, out_shape)
    })?;

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}
