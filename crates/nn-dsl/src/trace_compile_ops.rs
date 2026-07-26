// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-op compilation helpers for `trace_compile`.
//!
//! Extracted from `trace_compile.rs` to keep files under 500 lines.
//! These functions lower individual `TraceOp` variants into
//! `TensorKernelDef` dispatch plans via `TensorBlockBuilder`.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::ir::{BinOpKind, CompareOpKind, UnaryFnKind};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::{binop_kernel, compare_select_kernel, square_kernel, unary_kernel};
use crate::tensor_ir::{ReduceOp, TensorIRError, TensorKernelDef, TensorNodeId};

use super::{resolve_input_shape, CompiledKernel, CompiledStep};

// -- Helper types and compilation functions ---------------------------------

/// Distinguishes binary ops that have dedicated builder methods from those
/// using the scalar `add_elementwise` path.
pub(super) enum BinaryMethod {
    BuilderAdd,
    BuilderMul,
}

/// Distinguishes activation ops with dedicated builder methods.
pub(super) enum ActivationKind {
    Relu,
    Gelu,
    GeluErf,
    Sigmoid,
    Tanh,
}

/// Build a single-op `TensorKernelDef` with the given builder closure.
///
/// Handles the common pattern: create builder, add inputs for each trace
/// input, run the builder closure, build the def.
pub(super) fn build_single_op<F>(
    name: &str,
    node: &TraceNode,
    graph: &ComputationGraph,
    num_inputs: usize,
    f: F,
) -> Result<CompiledStep, TensorIRError>
where
    F: FnOnce(&mut TensorBlockBuilder, &[TensorNodeId]) -> TensorNodeId,
{
    let mut b = TensorBlockBuilder::new(name);
    let mut input_ids = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        let input_shape = resolve_input_shape(node, i, graph)?;
        let id = b.add_input(&format!("input_{i}"), input_shape);
        input_ids.push(id);
    }
    let output = f(&mut b, &input_ids);
    let def = b.build(output)?;
    // Populate external_node_ids from graph topology so that downstream
    // fusion passes (auto-fuse, partition codegen) can resolve graph-level
    // input edges without fabricating placeholder IDs. Part of #4283.
    let ext_ids: Vec<u64> = node.inputs().iter().take(num_inputs).copied().collect();
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: if ext_ids.is_empty() {
            None
        } else {
            Some(ext_ids)
        },
    })
}

/// Build a single-op def with explicit input shapes from trace graph context.
pub(super) fn build_op_with_weights<F>(
    name: &str,
    _node: &TraceNode,
    f: F,
) -> Result<(TensorKernelDef, HashMap<String, WeightRef>), TensorIRError>
where
    F: FnOnce(&mut TensorBlockBuilder, &mut HashMap<String, WeightRef>) -> TensorNodeId,
{
    let mut b = TensorBlockBuilder::new(name);
    let mut weight_data = HashMap::new();
    let output = f(&mut b, &mut weight_data);
    let def = b.build(output)?;
    Ok((def, weight_data))
}

/// Add a weight tensor as an input node and record it in weight_data.
pub(super) fn add_weight(
    b: &mut TensorBlockBuilder,
    weight_data: &mut HashMap<String, WeightRef>,
    name: &str,
    w: &WeightRef,
) -> TensorNodeId {
    let id = b.add_input(name, w.shape());
    weight_data.insert(name.to_string(), w.clone());
    id
}

/// Extract graph-level input node IDs for non-weight inputs.
///
/// Used by compilation helpers that construct Dispatch steps directly
/// (without `build_single_op`) to populate `external_node_ids`.
/// Part of #4283.
pub(super) fn graph_input_ids(node: &TraceNode, num_graph_inputs: usize) -> Option<Vec<u64>> {
    let ids: Vec<u64> = node
        .inputs()
        .iter()
        .take(num_graph_inputs)
        .copied()
        .collect();
    if ids.is_empty() {
        None
    } else {
        Some(ids)
    }
}

// -- Binary ops (extracted to trace_compile_binary.rs) ------------------------

#[path = "trace_compile_binary.rs"]
mod trace_compile_binary;
pub(super) use trace_compile_binary::{
    compile_binary_elementwise, compile_binary_fn_elementwise, compile_binary_op,
};

// -- Unary ops ----------------------------------------------------------------

pub(super) fn compile_unary_elementwise(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    op: UnaryFnKind,
) -> Result<CompiledStep, TensorIRError> {
    let kernel = unary_kernel(name, op);
    build_single_op(name, node, graph, 1, |b, inputs| {
        b.add_elementwise(kernel.clone(), &[inputs[0]], node.output_shape())
    })
}

pub(super) fn compile_sqr(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    let kernel = square_kernel();
    build_single_op("sqr", node, graph, 1, |b, inputs| {
        b.add_elementwise(kernel.clone(), &[inputs[0]], node.output_shape())
    })
}

pub(super) fn compile_neg(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    // Negate via subtract from zero: 0 - x.
    // There's no dedicated negate kernel, so we build a sub(zero, x) graph.
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let mut b = TensorBlockBuilder::new("neg");
    let input = b.add_input("input_0", input_shape);
    let zero = b.add_input("zero", &[1]);
    let zero_bc = b.add_broadcast(zero, node.output_shape());
    let kernel = binop_kernel("sub", BinOpKind::Sub);
    let output = b.add_elementwise(kernel, &[zero_bc, input], node.output_shape());
    let def = b.build(output)?;

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "zero".to_string(),
        WeightRef::new(vec![0.0f32], vec![1]).expect("valid scalar"),
    );

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: graph_input_ids(node, 1),
    })
}

/// Compile `Powf { exponent }` using the eager GPU semantics from
/// `DynTensor::gpu_powf`.
///
/// The magnitude path is always `exp(exponent * log(abs(x)))`. For integer
/// exponents we restore the sign for odd powers; for non-integer exponents we
/// replace negative-base results with NaN. This matches `f32::powf` for the
/// representable cases and avoids the old `log(x)` domain bug on negative
/// inputs. (#2751)
pub(super) fn compile_powf(
    node: &TraceNode,
    graph: &ComputationGraph,
    exponent: f64,
) -> Result<CompiledStep, TensorIRError> {
    let exp_f32 = exponent as f32;
    if !exp_f32.is_finite() {
        return Err(TensorIRError::NonFiniteConstant {
            name: "powf_exponent".into(),
            value: exponent,
        });
    }
    if exp_f32 == 0.0 {
        return Ok(CompiledStep::ConstantValue {
            value: 1.0,
            shape: node.output_shape().to_vec(),
        });
    }
    if exp_f32 == 1.0 {
        return Ok(CompiledStep::IdentityPassthrough);
    }

    let input_shape = resolve_input_shape(node, 0, graph)?;
    let out_shape = node.output_shape();
    let mut b = TensorBlockBuilder::new("powf");
    let input = b.add_input("input_0", input_shape);

    // abs(x)
    let abs_kernel = unary_kernel("abs", UnaryFnKind::Abs);
    let abs_x = b.add_elementwise(abs_kernel, &[input], out_shape);

    // log(abs(x))
    let log_kernel = unary_kernel("log", UnaryFnKind::Log);
    let log_x = b.add_elementwise(log_kernel, &[abs_x], out_shape);

    // exponent * log(abs(x))
    let exp_const = b.add_input("exponent", &[1]);
    let exp_bc = b.add_broadcast(exp_const, out_shape);
    let mul_kernel = binop_kernel("mul", BinOpKind::Mul);
    let scaled = b.add_elementwise(mul_kernel, &[exp_bc, log_x], out_shape);

    // exp(exponent * log(abs(x))))
    let exp_kernel = unary_kernel("exp", UnaryFnKind::Exp);
    let mut weight_data = HashMap::new();
    weight_data.insert(
        "exponent".to_string(),
        WeightRef::new(vec![exp_f32], vec![1]).expect("valid finite scalar"),
    );

    let abs_pow = b.add_elementwise(exp_kernel, &[scaled], out_shape);
    let is_integer = exp_f32 == exp_f32.floor();
    let output = if is_integer {
        // Match eager GPU fallback: if parity cannot be represented exactly
        // beyond 2^24, treat the exponent as even.
        let can_determine_parity = exp_f32.abs() <= (1i64 << 24) as f32;
        let is_even = !can_determine_parity || (exp_f32 as i64) % 2 == 0;
        if is_even {
            abs_pow
        } else {
            let zero_const = b.add_input("zero_const", &[1]);
            let zero_bc = b.add_broadcast(zero_const, out_shape);
            let neg_mask = b.add_elementwise(
                compare_select_kernel("cmp_lt", CompareOpKind::Lt),
                &[input, zero_bc],
                out_shape,
            );
            weight_data.insert(
                "zero_const".to_string(),
                WeightRef::new(vec![0.0], vec![1]).expect("valid scalar"),
            );

            let neg_abs_pow = b.add_elementwise(
                binop_kernel("sub", BinOpKind::Sub),
                &[zero_bc, abs_pow],
                out_shape,
            );

            let one_const = b.add_input("one_const", &[1]);
            let one_bc = b.add_broadcast(one_const, out_shape);
            weight_data.insert(
                "one_const".to_string(),
                WeightRef::new(vec![1.0], vec![1]).expect("valid scalar"),
            );

            let inv_mask = b.add_elementwise(
                binop_kernel("sub", BinOpKind::Sub),
                &[one_bc, neg_mask],
                out_shape,
            );
            let masked_true = b.add_binary_mul(neg_mask, neg_abs_pow, out_shape);
            let masked_false = b.add_binary_mul(inv_mask, abs_pow, out_shape);
            b.add_binary_add(masked_true, masked_false, out_shape)
        }
    } else {
        let zero_const = b.add_input("zero_const", &[1]);
        let zero_bc = b.add_broadcast(zero_const, out_shape);
        let neg_mask = b.add_elementwise(
            compare_select_kernel("cmp_lt", CompareOpKind::Lt),
            &[input, zero_bc],
            out_shape,
        );
        weight_data.insert(
            "zero_const".to_string(),
            WeightRef::new(vec![0.0], vec![1]).expect("valid scalar"),
        );

        let neg_one_const = b.add_input("neg_one_const", &[1]);
        let neg_one_bc = b.add_broadcast(neg_one_const, out_shape);
        weight_data.insert(
            "neg_one_const".to_string(),
            WeightRef::new(vec![-1.0], vec![1]).expect("valid scalar"),
        );
        let nan_fill = b.add_elementwise(
            unary_kernel("log", UnaryFnKind::Log),
            &[neg_one_bc],
            out_shape,
        );

        let one_const = b.add_input("one_const", &[1]);
        let one_bc = b.add_broadcast(one_const, out_shape);
        weight_data.insert(
            "one_const".to_string(),
            WeightRef::new(vec![1.0], vec![1]).expect("valid scalar"),
        );

        let inv_mask = b.add_elementwise(
            binop_kernel("sub", BinOpKind::Sub),
            &[one_bc, neg_mask],
            out_shape,
        );
        let masked_true = b.add_binary_mul(neg_mask, nan_fill, out_shape);
        let masked_false = b.add_binary_mul(inv_mask, abs_pow, out_shape);
        b.add_binary_add(masked_true, masked_false, out_shape)
    };
    let def = b.build(output)?;

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: graph_input_ids(node, 1),
    })
}

// -- Activations --------------------------------------------------------------

pub(super) fn compile_activation(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    kind: ActivationKind,
) -> Result<CompiledStep, TensorIRError> {
    build_single_op(name, node, graph, 1, |b, inputs| match kind {
        ActivationKind::Relu => b.add_relu(inputs[0], node.output_shape()),
        ActivationKind::Gelu => b.add_gelu(inputs[0], node.output_shape()),
        ActivationKind::GeluErf => b.add_gelu_erf(inputs[0], node.output_shape()),
        ActivationKind::Sigmoid => b.add_sigmoid(inputs[0], node.output_shape()),
        ActivationKind::Tanh => b.add_tanh(inputs[0], node.output_shape()),
    })
}

// -- Reductions ---------------------------------------------------------------

pub(super) fn compile_reduce(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
    op: ReduceOp,
    dim: usize,
    keepdim: bool,
) -> Result<CompiledStep, TensorIRError> {
    build_single_op(name, node, graph, 1, |b, inputs| {
        b.add_reduce(inputs[0], op, dim, keepdim, node.output_shape())
    })
}

// -- MatMul -------------------------------------------------------------------

pub(super) fn compile_matmul(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    build_single_op("matmul", node, graph, 2, |b, inputs| {
        b.add_matmul(inputs[0], inputs[1], false, None, node.output_shape())
    })
}

// -- Linear -------------------------------------------------------------------

pub(super) fn compile_linear(
    node: &TraceNode,
    graph: &ComputationGraph,
    weight: &WeightRef,
    bias: &Option<WeightRef>,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let (def, weight_data) = build_op_with_weights("linear", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let w = add_weight(b, wd, "weight", weight);
        let bi = bias.as_ref().map(|bw| add_weight(b, wd, "bias", bw));
        b.add_linear(input, w, bi, node.output_shape())
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: graph_input_ids(node, 1),
    })
}

// -- Convolutions (extracted to trace_compile_conv.rs) ------------------------

#[path = "trace_compile_conv.rs"]
mod trace_compile_conv;
pub(super) use trace_compile_conv::*;

// -- Normalization (extracted to trace_compile_norm.rs) -----------------------

#[path = "trace_compile_norm.rs"]
mod trace_compile_norm;
pub(super) use trace_compile_norm::*;

// -- Attention / Softmax ------------------------------------------------------

pub(super) fn compile_softmax(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    let dim_i32 = i32::try_from(dim).map_err(|_| TensorIRError::SoftmaxDimOverflow { dim })?;
    build_single_op("softmax", node, graph, 1, |b, inputs| {
        b.add_softmax(inputs[0], dim_i32, node.output_shape())
    })
}

// -- Embedding ----------------------------------------------------------------

pub(super) fn compile_embedding(
    node: &TraceNode,
    graph: &ComputationGraph,
    weight: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let (def, weight_data) = build_op_with_weights("embedding", node, |b, wd| {
        let input = b.add_input("input_0", input_shape);
        let w = add_weight(b, wd, "weight", weight);
        b.add_embedding(input, w, node.output_shape())
    })?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: graph_input_ids(node, 1),
    })
}

// -- LSTM (extracted to trace_compile_lstm.rs) --------------------------------

#[path = "trace_compile_lstm.rs"]
mod trace_compile_lstm;
pub(super) use trace_compile_lstm::*;

// -- Shape ops with dispatch --------------------------------------------------

pub(super) fn compile_narrow(
    node: &TraceNode,
    graph: &ComputationGraph,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<CompiledStep, TensorIRError> {
    // Zero-copy path: when all dimensions before the narrow axis have size 1,
    // the narrow produces a contiguous byte range that can be expressed as a
    // buffer offset (no memcpy needed). Matches the runtime optimization in
    // dyn_tensor_metal_shape_ops_narrow.rs::is_narrow_contiguous(). #2780.
    let input_shape = resolve_input_shape(node, 0, graph)?;

    let resolved_length = input_shape
        .get(dim)
        .copied()
        .filter(|&axis_len| start <= axis_len)
        .map(|axis_len| {
            let remaining = axis_len - start;
            if length == usize::MAX || length > remaining {
                remaining
            } else {
                length
            }
        })
        .unwrap_or(length);

    // Full-range narrow (start=0, length=dim_size) is identity — no data movement.
    if start == 0 && dim < input_shape.len() && resolved_length == input_shape[dim] {
        return Ok(CompiledStep::IdentityPassthrough);
    }

    let is_contiguous = input_shape[..dim].iter().all(|&s| s == 1);
    if is_contiguous {
        // Byte offset = start * product(dims_after_narrow_axis) * sizeof(f32).
        let trailing: usize = input_shape[dim + 1..]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| TensorIRError::ShapeOverflow {
                shape: input_shape.to_vec(),
            })?;
        let byte_offset = start
            .checked_mul(trailing)
            .and_then(|v| v.checked_mul(4)) // f32 = 4 bytes
            .ok_or_else(|| TensorIRError::ShapeOverflow {
                shape: input_shape.to_vec(),
            })?;
        return Ok(CompiledStep::NarrowView {
            byte_offset,
            output_shape: node.output_shape().to_vec(),
            source_step: None,
        });
    }

    build_single_op("narrow", node, graph, 1, |b, inputs| {
        b.add_narrow(inputs[0], dim, start, resolved_length, node.output_shape())
    })
}

// -- Activations (decomposed) + Binary min/max (extracted) -------------------

#[path = "trace_compile_activations.rs"]
mod trace_compile_activations;
pub(super) use trace_compile_activations::*;

#[path = "trace_compile_adain.rs"]
mod trace_compile_adain;
pub(super) use trace_compile_adain::*;

// Kani proofs for NarrowView byte_offset overflow (#2218).
#[cfg(kani)]
#[path = "trace_compile_ops_narrow_kani.rs"]
mod narrow_kani;

// Kani proofs for trace_compile_ops per-op compilation helpers (#3704).
#[cfg(kani)]
#[path = "kani_trace_compile_ops.rs"]
mod kani_trace_compile_ops;

// Kani proofs for trace_compile_ops extended coverage (#3745).
#[cfg(kani)]
#[path = "kani_trace_compile_ops_3745.rs"]
mod kani_trace_compile_ops_3745;
