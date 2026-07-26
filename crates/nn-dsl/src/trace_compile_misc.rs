// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Miscellaneous structural op compilation helpers for `trace_compile`.
//!
//! Extracted from `trace_compile.rs` to keep files under 450 lines.
//! Contains concatenation (`compile_cat`), axis reordering (`compile_transpose`,
//! `compile_permute`), expand (`compile_expand`), and clamping (`compile_clamp`)
//! -- ops that are not element-wise (trace_compile_ops), not normalization
//! (trace_compile_norm), and not fusion (trace_compile_fusion).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::ir::{BinOpKind, CompareOpKind, MinMaxKind};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::{binop_kernel, compare_select_kernel, minmax_kernel};
use crate::tensor_ir::TensorIRError;

use super::{build_single_op, resolve_input_shape, CompiledKernel, CompiledStep, NativeOpKind};

// -- Cat (variable-input concatenation) ---------------------------------------

pub(super) fn compile_cat(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    let n = node.inputs().len();

    // Single-input cat is identity — no data movement needed.
    if n == 1 {
        return Ok(CompiledStep::IdentityPassthrough);
    }

    let mut b = TensorBlockBuilder::new("cat");
    let mut input_ids = Vec::with_capacity(n);
    for i in 0..n {
        let input_shape = resolve_input_shape(node, i, graph)?;
        let id = b.add_input(&format!("input_{i}"), input_shape);
        input_ids.push(id);
    }
    let output = b.add_concat(&input_ids, dim, node.output_shape());
    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, n),
    })
}

// -- Transpose (axis swap via permutation) ------------------------------------

pub(super) fn compile_transpose(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim0: usize,
    dim1: usize,
) -> Result<CompiledStep, TensorIRError> {
    // Build axes permutation: identity with dim0 and dim1 swapped.
    let ndim = node.output_shape().len();
    if dim0 >= ndim || dim1 >= ndim {
        return Err(TensorIRError::TransposeDimOutOfBounds { dim0, dim1, ndim });
    }

    // Identity transpose (same axis) is a no-op — zero-copy passthrough.
    if dim0 == dim1 {
        return Ok(CompiledStep::Passthrough {
            op_name: "transpose".into(),
            output_shape: node.output_shape().to_vec(),
        });
    }

    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(dim0, dim1);
    build_single_op("transpose", node, graph, 1, |b, inputs| {
        b.add_transpose(inputs[0], &axes, node.output_shape())
    })
}

// -- Permute (full-axes reorder — NOT metadata-only like Reshape) -------------

pub(super) fn compile_permute(
    node: &TraceNode,
    graph: &ComputationGraph,
    axes: &[usize],
) -> Result<CompiledStep, TensorIRError> {
    let ndim = node.output_shape().len();
    // Validate axes: must be a valid permutation of 0..ndim.
    if axes.len() != ndim {
        return Err(TensorIRError::InvalidPermuteAxes {
            axes: axes.to_vec(),
            ndim,
            reason: format!("expected {} axes, got {}", ndim, axes.len()),
        });
    }
    let mut seen = vec![false; ndim];
    for &a in axes {
        if a >= ndim {
            return Err(TensorIRError::InvalidPermuteAxes {
                axes: axes.to_vec(),
                ndim,
                reason: format!("axis {a} out of bounds for rank {ndim}"),
            });
        }
        if seen[a] {
            return Err(TensorIRError::InvalidPermuteAxes {
                axes: axes.to_vec(),
                ndim,
                reason: format!("duplicate axis {a}"),
            });
        }
        seen[a] = true;
    }

    // Identity permutation [0, 1, ..., n-1] is a no-op — zero-copy passthrough.
    let is_identity = axes.iter().enumerate().all(|(i, &a)| a == i);
    if is_identity {
        return Ok(CompiledStep::Passthrough {
            op_name: "permute".into(),
            output_shape: node.output_shape().to_vec(),
        });
    }

    build_single_op("permute", node, graph, 1, |b, inputs| {
        b.add_transpose(inputs[0], axes, node.output_shape())
    })
}

// -- Expand (broadcast -- physical data movement, NOT metadata-only) -----------

pub(super) fn compile_expand(
    node: &TraceNode,
    graph: &ComputationGraph,
    target_shape: &[usize],
) -> Result<CompiledStep, TensorIRError> {
    // Identity expand (input already matches target shape) is a no-op.
    let input_shape = resolve_input_shape(node, 0, graph)?;
    if input_shape == target_shape {
        return Ok(CompiledStep::Passthrough {
            op_name: "expand".into(),
            output_shape: target_shape.to_vec(),
        });
    }

    build_single_op("expand", node, graph, 1, |b, inputs| {
        b.add_broadcast(inputs[0], target_shape)
    })
}

// -- Clamp (decomposed into max/min with broadcast constants) -----------------

pub(super) fn compile_clamp(
    node: &TraceNode,
    graph: &ComputationGraph,
    min_val: &Option<f64>,
    max_val: &Option<f64>,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = TensorBlockBuilder::new("clamp");
    let input = b.add_input("input_0", input_shape);
    let mut weight_data = HashMap::new();
    let mut current = input;

    if let Some(lo) = min_val {
        let lo_f32 = *lo as f32;
        if !lo_f32.is_finite() {
            return Err(TensorIRError::NonFiniteConstant {
                name: "clamp_min".into(),
                value: *lo,
            });
        }
        let lo_node = b.add_input("clamp_min", &[1]);
        weight_data.insert(
            "clamp_min".to_string(),
            WeightRef::new(vec![lo_f32], vec![1]).expect("valid finite scalar"),
        );
        let lo_bc = b.add_broadcast(lo_node, node.output_shape());
        let kernel = minmax_kernel("max", MinMaxKind::Max);
        current = b.add_elementwise(kernel, &[current, lo_bc], node.output_shape());
    }
    if let Some(hi) = max_val {
        let hi_f32 = *hi as f32;
        if !hi_f32.is_finite() {
            return Err(TensorIRError::NonFiniteConstant {
                name: "clamp_max".into(),
                value: *hi,
            });
        }
        let hi_node = b.add_input("clamp_max", &[1]);
        weight_data.insert(
            "clamp_max".to_string(),
            WeightRef::new(vec![hi_f32], vec![1]).expect("valid finite scalar"),
        );
        let hi_bc = b.add_broadcast(hi_node, node.output_shape());
        let kernel = minmax_kernel("min", MinMaxKind::Min);
        current = b.add_elementwise(kernel, &[current, hi_bc], node.output_shape());
    }

    let def = b.build(current)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- Cumsum (native GPU Blelloch prefix scan) ---------------------------------

/// Compile `cumsum(x, dim)` as a native GPU operation.
///
/// Emits `NativeOp::Cumsum` which delegates to the existing Blelloch
/// parallel prefix sum kernel (`gpu_cumsum`) at execution time. This
/// replaces the previous O(N) narrow+add+concat IR decomposition with
/// a single GPU dispatch.
pub(super) fn compile_cumsum(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;

    Ok(CompiledStep::NativeOp {
        op: NativeOpKind::Cumsum {
            dim,
            input_shape: input_shape.to_vec(),
        },
        weight_data: HashMap::new(),
    })
}

// -- RepeatInterleave (decomposed into reshape + broadcast + reshape) ----------

/// Compile `repeat_interleave(x, repeats, dim)` for uniform repeats.
///
/// Decomposition (unsqueeze + expand + reshape):
///   Step 1: reshape  [..., S, ...]          → [..., S, 1, ...]
///   Step 2: broadcast [..., S, 1, ...]      → [..., S, R, ...]
///   Step 3: reshape  [..., S, R, ...]       → [..., S*R, ...]
///
/// Only supports uniform repeats (all counts equal). Variable-length
/// repeats return `UnsupportedTraceOp`.
pub(super) fn compile_repeat_interleave(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let output_shape = node.output_shape();
    let s = input_shape[dim];

    // Two-input repeat_interleave (tensor + counts): ALWAYS emit RuntimeOp.
    // The counts tensor is data-dependent — even when total repeats happen to
    // divide evenly, individual counts may be non-uniform. Fixes #2452.
    if node.inputs().len() >= 2 {
        let counts_shape = resolve_input_shape(node, 1, graph)?.to_vec();
        return Ok(CompiledStep::RuntimeOp {
            op: super::RuntimeOpKind::RepeatInterleave {
                dim,
                input_shape: input_shape.to_vec(),
                counts_shape,
            },
        });
    }

    // Single-input repeat_interleave: derive uniform repeat count from shapes.
    let out_dim = output_shape[dim];
    if s == 0 || !out_dim.is_multiple_of(s) {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("repeat_interleave: output dim {out_dim} not divisible by input dim {s}"),
        });
    }
    let repeats = out_dim / s;

    let mut b = TensorBlockBuilder::new("repeat_interleave");
    let input = b.add_input("input_0", input_shape);

    // Step 1: Unsqueeze — insert dim of size 1 after target dim.
    let mut unsqueezed_shape = input_shape.to_vec();
    unsqueezed_shape.insert(dim + 1, 1);
    let unsqueezed = b.add_reshape(input, &unsqueezed_shape);

    // Step 2: Expand — broadcast the new dim to `repeats`.
    let mut expanded_shape = input_shape.to_vec();
    expanded_shape.insert(dim + 1, repeats);
    let expanded = b.add_broadcast(unsqueezed, &expanded_shape);

    // Step 3: Reshape — merge dim and dim+1 back into one.
    let output = b.add_reshape(expanded, output_shape);

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- WhereCond (ternary select, decomposed) -----------------------------------

/// Compile `WhereCond` as `mask * on_true + (1 - mask) * on_false`.
///
/// GPU masks are F32 0.0/1.0 (from compare ops), so the decomposition
/// is exact. Three graph inputs: mask, on_true, on_false.
pub(super) fn compile_where_cond(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    let mask_shape = resolve_input_shape(node, 0, graph)?;
    let true_shape = resolve_input_shape(node, 1, graph)?;
    let false_shape = resolve_input_shape(node, 2, graph)?;
    let out_shape = node.output_shape();

    let mut b = TensorBlockBuilder::new("where_cond");
    let mask = b.add_input("input_0", mask_shape);
    let on_true = b.add_input("input_1", true_shape);
    let on_false = b.add_input("input_2", false_shape);

    // Broadcast all inputs to output shape.
    let mask_bc = b.add_broadcast(mask, out_shape);
    let true_bc = b.add_broadcast(on_true, out_shape);
    let false_bc = b.add_broadcast(on_false, out_shape);

    // one_const (scalar weight = 1.0, broadcast to output shape)
    let one = b.add_input("one_const", &[1]);
    let one_bc = b.add_broadcast(one, out_shape);

    // inv_mask = 1.0 - mask
    let sub_kernel = binop_kernel("sub", BinOpKind::Sub);
    let inv_mask = b.add_elementwise(sub_kernel, &[one_bc, mask_bc], out_shape);

    // masked_true = mask * on_true
    let masked_true = b.add_binary_mul(mask_bc, true_bc, out_shape);

    // masked_false = (1 - mask) * on_false
    let masked_false = b.add_binary_mul(inv_mask, false_bc, out_shape);

    // output = masked_true + masked_false
    let output = b.add_binary_add(masked_true, masked_false, out_shape);

    let def = b.build(output)?;
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "one_const".to_string(),
        WeightRef::new(vec![1.0f32], vec![1]).expect("valid scalar"),
    );
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 3),
    })
}

// -- Flip (reverse elements along dim via index_select) -----------------------

/// Compile `Flip { dim }` as a single `index_select` with reversed indices.
///
/// This generates 1 Metal dispatch per flip instead of N narrow + 1 cat (N+1
/// dispatches), where N = input_shape[dim]. For bidirectional LSTM models
/// this is a major optimization: e.g. Kokoro f0_energy alone had 156 Metal
/// dispatches from flip ops, reduced to ~20 with index_select.
pub(super) fn compile_flip(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let n = input_shape[dim];

    // Flip on a single-element dimension is identity — no data movement.
    if n <= 1 {
        return Ok(CompiledStep::IdentityPassthrough);
    }

    // Build reversed index tensor [n-1, n-2, ..., 1, 0] as f32.
    // IndexSelect indices are stored as f32 and cast to uint in MSL.
    let reversed_indices: Vec<f32> = (0..n).rev().map(|i| i as f32).collect();
    let idx_weight =
        WeightRef::new(reversed_indices, vec![n]).map_err(TensorIRError::WeightData)?;

    // Include `n` in the weight name so the shared-weight alias system
    // (keyed by `(step_idx, name)`) doesn't reuse a buffer from a different
    // shape variant. Without this, a cached model compiled at n=32 would
    // alias its 128-byte buffer for a model needing n=56 (224 bytes),
    // causing "GPU buffer size mismatch" (#3234).
    let idx_name = format!("flip_indices_{n}");
    let mut b = TensorBlockBuilder::new("flip");
    let input = b.add_input("input_0", input_shape);
    let indices = b.add_input(&idx_name, &[n]);
    let output = b.add_index_select(input, indices, dim, node.output_shape());

    let def = b.build(output)?;
    let mut weight_data = HashMap::new();
    weight_data.insert(idx_name, idx_weight);
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- Compare (element-wise scalar comparison producing mask) ------------------

/// Compile `TraceOp::Compare { op, value }` as an elementwise
/// `select(x <op> threshold, 1.0, 0.0)` kernel.
///
/// SineGen uses `gt(threshold)` for voiced mask generation. Without this,
/// trace compilation fails with `UnsupportedTraceOp: compare` (#3214).
pub(super) fn compile_compare(
    node: &TraceNode,
    graph: &ComputationGraph,
    op: CompareOpKind,
    value: f64,
) -> Result<CompiledStep, TensorIRError> {
    let val_f32 = value as f32;
    if !val_f32.is_finite() {
        return Err(TensorIRError::NonFiniteConstant {
            name: "compare_threshold".into(),
            value,
        });
    }

    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = TensorBlockBuilder::new("compare");
    let input = b.add_input("input_0", input_shape);
    let threshold_node = b.add_input("compare_threshold", &[1]);
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "compare_threshold".to_string(),
        WeightRef::new(vec![val_f32], vec![1]).expect("valid finite scalar"),
    );
    let threshold_bc = b.add_broadcast(threshold_node, node.output_shape());
    let kernel = compare_select_kernel("compare", op);
    let output = b.add_elementwise(kernel, &[input, threshold_bc], node.output_shape());

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 1),
    })
}

// -- Padding ops (extracted to trace_compile_pad.rs) --------------------------

#[path = "trace_compile_pad.rs"]
mod trace_compile_pad;
pub(in crate::trace_compile) use trace_compile_pad::{
    compile_constant_pad_nd, compile_reflection_pad1d,
};
