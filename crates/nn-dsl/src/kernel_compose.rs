// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Scalar-level KernelDef composition for elementwise chain fusion.
//!
//! Given a chain of elementwise trace ops, builds a single composed
//! `KernelDef` where each op contributes a few scalar IR nodes. External
//! inputs (values from outside the chain) become kernel parameters.
//!
//! This produces a single `DispatchStep::Elementwise` → single GPU kernel
//! launch, unlike the prior approach which created multiple tensor-level
//! nodes (each generating a separate GPU dispatch).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId as TraceNodeId, TraceNode, TraceOp};

use crate::ir::{
    BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId,
    Param, ScalarType, UnaryFnKind,
};
use crate::tensor_ir::TensorIRError;

/// Build a single composed scalar `KernelDef` from a chain of elementwise
/// trace ops.
///
/// Each trace op contributes scalar IR nodes. External inputs (values from
/// outside the chain) become kernel parameters. Returns the composed kernel
/// and the ordered list of external trace `NodeId`s (for edge_map patching
/// in the executor).
pub(crate) fn build_fused_scalar_kernel(
    chain: &[TraceNode],
    graph: &ComputationGraph,
) -> Result<(KernelDef, Vec<TraceNodeId>), TensorIRError> {
    let mut params: Vec<Param> = Vec::new();
    let mut nodes: Vec<IRNode> = Vec::new();

    // Map trace NodeId → scalar IR NodeId for resolved inputs.
    let mut trace_to_ir: HashMap<TraceNodeId, NodeId> = HashMap::new();
    // External input trace NodeIds in kernel param order.
    let mut external_ids: Vec<TraceNodeId> = Vec::new();

    for trace_node in chain {
        let trace_inputs = trace_node.inputs();
        let num_inputs = super::op_input_count(trace_node.op());

        // Resolve each input to a scalar IR NodeId.
        let mut ir_inputs: Vec<NodeId> = Vec::with_capacity(num_inputs);
        for idx in 0..num_inputs {
            if let Some(&trace_id) = trace_inputs.get(idx) {
                if let Some(&ir_id) = trace_to_ir.get(&trace_id) {
                    ir_inputs.push(ir_id);
                } else {
                    // Check if the external input is a Constant or scalar
                    // ConstantWeight that can be inlined as a Literal IR node,
                    // eliminating a GPU buffer binding.
                    let inlined = graph
                        .node(trace_id)
                        .and_then(|ext_node| match ext_node.op() {
                            TraceOp::Constant { value } => Some(*value),
                            TraceOp::ConstantWeight { weight }
                                if weight.data().len() == 1 && !weight.is_placeholder() =>
                            {
                                Some(f64::from(weight.data()[0]))
                            }
                            _ => None,
                        });

                    if let Some(const_val) = inlined {
                        let ir_id = emit_literal(&mut nodes, const_val);
                        trace_to_ir.insert(trace_id, ir_id);
                        ir_inputs.push(ir_id);
                    } else {
                        // External input — add as a kernel parameter.
                        let param_idx = params.len();
                        params.push(Param::new(format!("p{param_idx}"), ScalarType::F32));
                        let ir_id = NodeId::new(nodes.len());
                        nodes.push(IRNode::new(ir_id, IRNodeKind::Param(param_idx)));
                        external_ids.push(trace_id);
                        trace_to_ir.insert(trace_id, ir_id);
                        ir_inputs.push(ir_id);
                    }
                }
            }
        }

        // Emit scalar IR nodes for this trace op.
        let result_id = emit_trace_op(&mut nodes, trace_node.op(), &ir_inputs)?;
        trace_to_ir.insert(trace_node.id(), result_id);
    }

    let last_id = chain
        .last()
        .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
            name: "empty fused chain".into(),
        })?
        .id();
    let output = *trace_to_ir
        .get(&last_id)
        .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
            name: "fused chain output not in IR map".into(),
        })?;

    let name = format!("fused_{}_x{}", chain[0].op().canonical_name(), chain.len());

    let kernel = KernelDef::new(name, params, ScalarType::F32, nodes, output);
    kernel
        .validate()
        .map_err(|e| TensorIRError::UnsupportedTraceOp {
            name: format!("fused kernel validation failed: {e}"),
        })?;

    Ok((kernel, external_ids))
}

/// Emit scalar IR nodes for a single trace op, returning the output NodeId.
fn emit_trace_op(
    nodes: &mut Vec<IRNode>,
    op: &TraceOp,
    inputs: &[NodeId],
) -> Result<NodeId, TensorIRError> {
    let result = match op {
        // Unary math
        TraceOp::Exp => emit_unary(nodes, UnaryFnKind::Exp, inputs[0]),
        TraceOp::Log => emit_unary(nodes, UnaryFnKind::Log, inputs[0]),
        TraceOp::Sqrt => emit_unary(nodes, UnaryFnKind::Sqrt, inputs[0]),
        TraceOp::Abs => emit_unary(nodes, UnaryFnKind::Abs, inputs[0]),
        TraceOp::Recip => emit_unary(nodes, UnaryFnKind::Recip, inputs[0]),
        TraceOp::Sin => emit_unary(nodes, UnaryFnKind::Sin, inputs[0]),
        TraceOp::Cos => emit_unary(nodes, UnaryFnKind::Cos, inputs[0]),
        TraceOp::Floor => emit_unary(nodes, UnaryFnKind::Floor, inputs[0]),
        TraceOp::Round => emit_unary(nodes, UnaryFnKind::Round, inputs[0]),
        TraceOp::Fract => emit_unary(nodes, UnaryFnKind::Fract, inputs[0]),
        TraceOp::Tanh => emit_unary(nodes, UnaryFnKind::Tanh, inputs[0]),

        // Square: x * x
        TraceOp::Sqr => emit_binop(nodes, BinOpKind::Mul, inputs[0], inputs[0]),

        // Neg: 0 - x (literal zero, no weight tensor needed)
        TraceOp::Neg => {
            let zero = emit_literal(nodes, 0.0);
            emit_binop(nodes, BinOpKind::Sub, zero, inputs[0])
        }

        // Activations
        TraceOp::Relu => emit_relu(nodes, inputs[0]),
        TraceOp::Gelu => emit_gelu(nodes, inputs[0]),
        TraceOp::GeluErf => emit_gelu_erf(nodes, inputs[0]),
        TraceOp::Sigmoid => emit_sigmoid(nodes, inputs[0]),
        TraceOp::Silu => emit_silu(nodes, inputs[0]),

        // Parameterized activations
        TraceOp::LeakyRelu { slope } => emit_leaky_relu(nodes, inputs[0], *slope),
        TraceOp::Elu { alpha } => emit_elu(nodes, inputs[0], *alpha),

        // Clamp
        TraceOp::Clamp { min, max } => emit_clamp(nodes, inputs[0], *min, *max),

        // Power: use eager GPU semantics based on exp(exponent * log(abs(x))).
        TraceOp::Powf { exponent } => emit_powf(nodes, inputs[0], *exponent),

        // Binary ops
        TraceOp::Add => emit_binop(nodes, BinOpKind::Add, inputs[0], inputs[1]),
        TraceOp::Sub => emit_binop(nodes, BinOpKind::Sub, inputs[0], inputs[1]),
        TraceOp::Mul => emit_binop(nodes, BinOpKind::Mul, inputs[0], inputs[1]),
        TraceOp::Div => emit_binop(nodes, BinOpKind::Div, inputs[0], inputs[1]),

        // Binary min/max
        TraceOp::Maximum => emit_minmax(nodes, MinMaxKind::Max, inputs[0], inputs[1]),
        TraceOp::Minimum => emit_minmax(nodes, MinMaxKind::Min, inputs[0], inputs[1]),

        // Binary trigonometric
        TraceOp::Atan2 => emit_binary_fn(nodes, BinaryFnKind::Atan2, inputs[0], inputs[1]),

        other => {
            return Err(TensorIRError::UnsupportedTraceOp {
                name: other.canonical_name().to_string(),
            });
        }
    };
    Ok(result)
}

// --- IR node emitters ---

fn emit_literal(nodes: &mut Vec<IRNode>, value: f64) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(id, IRNodeKind::Literal(value)));
    id
}

fn emit_unary(nodes: &mut Vec<IRNode>, op: UnaryFnKind, input: NodeId) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(id, IRNodeKind::UnaryFn { op, input }));
    id
}

fn emit_binop(nodes: &mut Vec<IRNode>, op: BinOpKind, lhs: NodeId, rhs: NodeId) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(id, IRNodeKind::BinOp { op, lhs, rhs }));
    id
}

fn emit_minmax(nodes: &mut Vec<IRNode>, op: MinMaxKind, lhs: NodeId, rhs: NodeId) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(id, IRNodeKind::MinMax { op, lhs, rhs }));
    id
}

/// `relu(x) = max(x, 0)`
fn emit_relu(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let zero = emit_literal(nodes, 0.0);
    emit_minmax(nodes, MinMaxKind::Max, x, zero)
}

/// `sigmoid(x) = 1 / (1 + exp(-x))`
fn emit_sigmoid(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let zero = emit_literal(nodes, 0.0);
    let neg_x = emit_binop(nodes, BinOpKind::Sub, zero, x);
    let exp_neg = emit_unary(nodes, UnaryFnKind::Exp, neg_x);
    let one = emit_literal(nodes, 1.0);
    let denom = emit_binop(nodes, BinOpKind::Add, one, exp_neg);
    emit_binop(nodes, BinOpKind::Div, one, denom)
}

/// `gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
fn emit_gelu(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let half = emit_literal(nodes, 0.5);
    let half_x = emit_binop(nodes, BinOpKind::Mul, half, x);

    // x³ = x * x * x
    let x2 = emit_binop(nodes, BinOpKind::Mul, x, x);
    let x3 = emit_binop(nodes, BinOpKind::Mul, x2, x);

    let coeff = emit_literal(nodes, 0.044715);
    let coeff_x3 = emit_binop(nodes, BinOpKind::Mul, coeff, x3);
    let inner = emit_binop(nodes, BinOpKind::Add, x, coeff_x3);

    // sqrt(2/π) ≈ 0.7978845608028654
    let sqrt_2_pi = emit_literal(nodes, 0.797_884_560_802_865_4);
    let scaled = emit_binop(nodes, BinOpKind::Mul, sqrt_2_pi, inner);
    let tanh_val = emit_unary(nodes, UnaryFnKind::Tanh, scaled);

    let one = emit_literal(nodes, 1.0);
    let one_plus = emit_binop(nodes, BinOpKind::Add, one, tanh_val);
    emit_binop(nodes, BinOpKind::Mul, half_x, one_plus)
}

/// `gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`
/// where erf uses the A&S 7.1.26 polynomial approximation.
fn emit_gelu_erf(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let erf_val = emit_erf_polynomial(nodes, x);
    let one = emit_literal(nodes, 1.0);
    let one_plus_erf = emit_binop(nodes, BinOpKind::Add, one, erf_val);
    let half = emit_literal(nodes, 0.5);
    let half_x = emit_binop(nodes, BinOpKind::Mul, half, x);
    emit_binop(nodes, BinOpKind::Mul, half_x, one_plus_erf)
}

/// Abramowitz & Stegun 7.1.26 erf polynomial: `erf(x/sqrt(2))`.
///
/// Coefficients match `erf_f32()` in `dyn_tensor/ops/math.rs` and
/// `build_erf_graph()` in `dyn_tensor_metal_kernels_complex.rs`.
fn emit_erf_polynomial(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let inv_sqrt2 = emit_literal(nodes, std::f64::consts::FRAC_1_SQRT_2);
    let u = emit_binop(nodes, BinOpKind::Mul, x, inv_sqrt2);

    // sign = (u >= 0) ? 1 : -1
    let zero = emit_literal(nodes, 0.0);
    let pos_one = emit_literal(nodes, 1.0);
    let neg_one = emit_literal(nodes, -1.0);
    let cond = emit_compare(nodes, CompareOpKind::Ge, u, zero);
    let sign = emit_select(nodes, cond, pos_one, neg_one);

    // ax = abs(u)
    let ax = emit_unary(nodes, UnaryFnKind::Abs, u);

    // t = 1 / (1 + p * ax)
    let p = emit_literal(nodes, 0.327_591_1);
    let p_ax = emit_binop(nodes, BinOpKind::Mul, p, ax);
    let one_plus = emit_binop(nodes, BinOpKind::Add, pos_one, p_ax);
    let t = emit_unary(nodes, UnaryFnKind::Recip, one_plus);

    // Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    let a5 = emit_literal(nodes, 1.061_405_4);
    let a4 = emit_literal(nodes, -1.453_152);
    let a3 = emit_literal(nodes, 1.421_413_8);
    let a2 = emit_literal(nodes, -0.284_496_74);
    let a1 = emit_literal(nodes, 0.254_829_6);

    let h = emit_binop(nodes, BinOpKind::Mul, a5, t);
    let h = emit_binop(nodes, BinOpKind::Add, h, a4);
    let h = emit_binop(nodes, BinOpKind::Mul, h, t);
    let h = emit_binop(nodes, BinOpKind::Add, h, a3);
    let h = emit_binop(nodes, BinOpKind::Mul, h, t);
    let h = emit_binop(nodes, BinOpKind::Add, h, a2);
    let h = emit_binop(nodes, BinOpKind::Mul, h, t);
    let h = emit_binop(nodes, BinOpKind::Add, h, a1);
    let h = emit_binop(nodes, BinOpKind::Mul, h, t);

    // poly * exp(-u*u)
    let neg_u_sq = emit_binop(nodes, BinOpKind::Mul, u, u);
    let neg_u_sq = emit_binop(nodes, BinOpKind::Sub, zero, neg_u_sq);
    let exp_val = emit_unary(nodes, UnaryFnKind::Exp, neg_u_sq);
    let poly_exp = emit_binop(nodes, BinOpKind::Mul, h, exp_val);

    // erf = sign * (1 - poly_exp)
    let one_minus = emit_binop(nodes, BinOpKind::Sub, pos_one, poly_exp);
    emit_binop(nodes, BinOpKind::Mul, sign, one_minus)
}

/// `silu(x) = x * sigmoid(x)`
fn emit_silu(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let sig = emit_sigmoid(nodes, x);
    emit_binop(nodes, BinOpKind::Mul, x, sig)
}

/// `leaky_relu(x, slope) = x > 0 ? x : slope * x`
fn emit_leaky_relu(nodes: &mut Vec<IRNode>, x: NodeId, slope: f64) -> NodeId {
    let zero = emit_literal(nodes, 0.0);
    let cond = emit_compare(nodes, CompareOpKind::Gt, x, zero);
    let slope_lit = emit_literal(nodes, slope);
    let neg_branch = emit_binop(nodes, BinOpKind::Mul, slope_lit, x);
    emit_select(nodes, cond, x, neg_branch)
}

/// `elu(x, alpha) = x > 0 ? x : alpha * (exp(x) - 1)`
fn emit_elu(nodes: &mut Vec<IRNode>, x: NodeId, alpha: f64) -> NodeId {
    let zero = emit_literal(nodes, 0.0);
    let cond = emit_compare(nodes, CompareOpKind::Gt, x, zero);
    let exp_x = emit_unary(nodes, UnaryFnKind::Exp, x);
    let one = emit_literal(nodes, 1.0);
    let exp_minus_1 = emit_binop(nodes, BinOpKind::Sub, exp_x, one);
    let alpha_lit = emit_literal(nodes, alpha);
    let neg_branch = emit_binop(nodes, BinOpKind::Mul, alpha_lit, exp_minus_1);
    emit_select(nodes, cond, x, neg_branch)
}

/// `clamp(x, min, max)` — handles optional min/max bounds.
fn emit_clamp(nodes: &mut Vec<IRNode>, x: NodeId, min: Option<f64>, max: Option<f64>) -> NodeId {
    let mut result = x;
    if let Some(min_val) = min {
        let min_lit = emit_literal(nodes, min_val);
        result = emit_minmax(nodes, MinMaxKind::Max, result, min_lit);
    }
    if let Some(max_val) = max {
        let max_lit = emit_literal(nodes, max_val);
        result = emit_minmax(nodes, MinMaxKind::Min, result, max_lit);
    }
    result
}

/// Scalar powf lowering matching the eager GPU path.
fn emit_powf(nodes: &mut Vec<IRNode>, x: NodeId, exponent: f64) -> NodeId {
    if exponent == 0.0 {
        return emit_literal(nodes, 1.0);
    }
    if exponent == 1.0 {
        return x;
    }

    let abs_x = emit_unary(nodes, UnaryFnKind::Abs, x);
    let log_x = emit_unary(nodes, UnaryFnKind::Log, abs_x);
    let exp_lit = emit_literal(nodes, exponent);
    let scaled = emit_binop(nodes, BinOpKind::Mul, exp_lit, log_x);
    let abs_pow = emit_unary(nodes, UnaryFnKind::Exp, scaled);

    if exponent.is_finite() && exponent == exponent.floor() {
        let can_determine_parity = exponent.abs() <= (1i64 << 24) as f64;
        let is_even = !can_determine_parity || (exponent as i64) % 2 == 0;
        if is_even {
            return abs_pow;
        }
        let zero = emit_literal(nodes, 0.0);
        let neg_cond = emit_compare(nodes, CompareOpKind::Lt, x, zero);
        let neg_abs_pow = emit_binop(nodes, BinOpKind::Sub, zero, abs_pow);
        return emit_select(nodes, neg_cond, neg_abs_pow, abs_pow);
    }

    let zero = emit_literal(nodes, 0.0);
    let neg_cond = emit_compare(nodes, CompareOpKind::Lt, x, zero);
    let neg_one = emit_literal(nodes, -1.0);
    let nan = emit_unary(nodes, UnaryFnKind::Log, neg_one);
    emit_select(nodes, neg_cond, nan, abs_pow)
}

fn emit_compare(nodes: &mut Vec<IRNode>, op: CompareOpKind, lhs: NodeId, rhs: NodeId) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(id, IRNodeKind::Compare { op, lhs, rhs }));
    id
}

fn emit_select(
    nodes: &mut Vec<IRNode>,
    cond: NodeId,
    then_val: NodeId,
    else_val: NodeId,
) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(
        id,
        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        },
    ));
    id
}

fn emit_binary_fn(nodes: &mut Vec<IRNode>, op: BinaryFnKind, lhs: NodeId, rhs: NodeId) -> NodeId {
    let id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(id, IRNodeKind::BinaryFn { op, lhs, rhs }));
    id
}

#[cfg(test)]
#[path = "kernel_compose_tests.rs"]
mod tests;
