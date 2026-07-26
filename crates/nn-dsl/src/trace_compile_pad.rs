// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Padding op compilation helpers for `trace_compile`.
//!
//! Extracted from `trace_compile_misc.rs` to keep files under 450 lines.
//! Contains `compile_reflection_pad1d` and `compile_constant_pad_nd`.
//! Part of #2745 (import-compile gap).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorIRError;

use super::{resolve_input_shape, CompiledKernel, CompiledStep};

// -- ReflectionPad1d (decomposed into narrow + flip + cat) --------------------

/// Compile `ReflectionPad1d { pad_left, pad_right }` by mirroring boundary
/// values on the last dimension. Matches PyTorch `nn.ReflectionPad1d`.
///
/// Decomposition (same as `DynTensor::reflection_pad1d`):
///   left_pad  = flip(narrow(x, last, 1, pad_left), last)
///   right_pad = flip(narrow(x, last, dim_len - pad_right - 1, pad_right), last)
///   output    = cat([left_pad, x, right_pad], last)
pub(in crate::trace_compile) fn compile_reflection_pad1d(
    node: &TraceNode,
    graph: &ComputationGraph,
    pad_left: usize,
    pad_right: usize,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let ndim = input_shape.len();
    let last = ndim - 1;
    let dim_len = input_shape[last];

    if pad_left >= dim_len || pad_right >= dim_len {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!(
                "reflection_pad1d: padding ({pad_left}, {pad_right}) >= input size {dim_len}"
            ),
        });
    }

    let mut b = TensorBlockBuilder::new("reflection_pad1d");
    let input = b.add_input("input_0", input_shape);

    // Helper: build a flip(narrow(input, last, start, len)) sub-graph.
    let flip_narrow = |b: &mut TensorBlockBuilder,
                       src: crate::tensor_ir::TensorNodeId,
                       start: usize,
                       len: usize|
     -> crate::tensor_ir::TensorNodeId {
        let mut slice_shape = input_shape.to_vec();
        slice_shape[last] = len;
        let narrowed = b.add_narrow(src, last, start, len, &slice_shape);

        // Flip along last dim: narrow each position in reverse, concat.
        if len <= 1 {
            return narrowed;
        }
        let mut single = slice_shape.clone();
        single[last] = 1;
        let slices: Vec<_> = (0..len)
            .rev()
            .map(|i| b.add_narrow(narrowed, last, i, 1, &single))
            .collect();
        b.add_concat(&slices, last, &slice_shape)
    };

    let mut cat_parts = Vec::with_capacity(3);

    if pad_left > 0 {
        cat_parts.push(flip_narrow(&mut b, input, 1, pad_left));
    }

    cat_parts.push(input);

    if pad_right > 0 {
        cat_parts.push(flip_narrow(
            &mut b,
            input,
            dim_len - pad_right - 1,
            pad_right,
        ));
    }

    let output = if cat_parts.len() == 1 {
        cat_parts[0]
    } else {
        b.add_concat(&cat_parts, last, node.output_shape())
    };

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::super::graph_input_ids(node, 1),
    })
}

// -- ConstantPadNd (decomposed into constant-fill + cat) ----------------------

/// Compile `ConstantPadNd { padding, value }` by concatenating constant tensors
/// with the input along each padded dimension.
///
/// Padding is in PyTorch reverse order: `[last_left, last_right, ..., first_left, first_right]`.
pub(in crate::trace_compile) fn compile_constant_pad_nd(
    node: &TraceNode,
    graph: &ComputationGraph,
    padding: &[usize],
    value: f64,
) -> Result<CompiledStep, TensorIRError> {
    let val_f32 = value as f32;
    if !val_f32.is_finite() {
        return Err(TensorIRError::NonFiniteConstant {
            name: "constant_pad_nd value".into(),
            value,
        });
    }

    let input_shape = resolve_input_shape(node, 0, graph)?;
    let ndim = input_shape.len();
    let n_padded_dims = padding.len() / 2;

    // Parse padding pairs (PyTorch reverse order).
    let mut pad_pairs = vec![(0usize, 0usize); ndim];
    for i in 0..n_padded_dims {
        let dim = ndim - 1 - i;
        pad_pairs[dim] = (padding[2 * i], padding[2 * i + 1]);
    }

    let mut b = TensorBlockBuilder::new("constant_pad_nd");
    let mut weight_data = HashMap::new();
    let input = b.add_input("input_0", input_shape);

    let mut current = input;
    let mut current_shape = input_shape.to_vec();

    for dim in (0..ndim).rev() {
        let (pl, pr) = pad_pairs[dim];
        if pl == 0 && pr == 0 {
            continue;
        }

        let mut cat_parts = Vec::with_capacity(3);

        if pl > 0 {
            let mut pad_shape = current_shape.clone();
            pad_shape[dim] = pl;
            let pad_name = format!("pad_left_d{dim}");
            let pad_data = vec![val_f32; pad_shape.iter().product::<usize>()];
            let weight = WeightRef::new(pad_data, pad_shape.clone()).map_err(|_| {
                TensorIRError::NonFiniteConstant {
                    name: pad_name.clone(),
                    value,
                }
            })?;
            let pad_node = b.add_input(&pad_name, &pad_shape);
            weight_data.insert(pad_name, weight);
            cat_parts.push(pad_node);
        }

        cat_parts.push(current);

        if pr > 0 {
            let mut pad_shape = current_shape.clone();
            pad_shape[dim] = pr;
            let pad_name = format!("pad_right_d{dim}");
            let pad_data = vec![val_f32; pad_shape.iter().product::<usize>()];
            let weight = WeightRef::new(pad_data, pad_shape.clone()).map_err(|_| {
                TensorIRError::NonFiniteConstant {
                    name: pad_name.clone(),
                    value,
                }
            })?;
            let pad_node = b.add_input(&pad_name, &pad_shape);
            weight_data.insert(pad_name, weight);
            cat_parts.push(pad_node);
        }

        current_shape[dim] += pl + pr;

        current = if cat_parts.len() == 1 {
            cat_parts[0]
        } else {
            b.add_concat(&cat_parts, dim, &current_shape)
        };
    }

    let def = b.build(current)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::super::graph_input_ids(node, 1),
    })
}
