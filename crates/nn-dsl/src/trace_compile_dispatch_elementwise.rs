// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Elementwise dispatch: binary, unary, min/max, activation, and reduction ops.
//!
//! Part of the category-dispatch refactor (#2305). Workers adding a new
//! elementwise op only touch this file, not the shared `compile_node` hub.

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp};
use nn_core::dyn_tensor::CompareOp;

use crate::ir::{BinOpKind, BinaryFnKind, CompareOpKind, MinMaxKind, UnaryFnKind};
use crate::tensor_ir::{ReduceOp, TensorIRError};

use super::super::trace_compile_misc::compile_compare;
use super::super::trace_compile_ops::{
    compile_activation, compile_adain_leaky_relu, compile_adain_snake, compile_binary_elementwise,
    compile_binary_fn_elementwise, compile_binary_minmax, compile_binary_op, compile_elu,
    compile_leaky_relu, compile_neg, compile_powf, compile_reduce, compile_silu,
    compile_snake_tensor, compile_softplus, compile_sqr, compile_unary_elementwise, ActivationKind,
    BinaryMethod,
};
use super::super::CompiledStep;

/// Try to compile an elementwise trace op. Returns `None` for non-elementwise ops.
pub(in crate::trace_compile) fn try_compile(
    node: &TraceNode,
    graph: &ComputationGraph,
) -> Option<Result<CompiledStep, TensorIRError>> {
    match node.op() {
        // -- Binary element-wise ops ------------------------------------------
        TraceOp::Add => Some(compile_binary_op(
            node,
            graph,
            "add",
            BinaryMethod::BuilderAdd,
        )),
        TraceOp::Mul => Some(compile_binary_op(
            node,
            graph,
            "mul",
            BinaryMethod::BuilderMul,
        )),
        TraceOp::Sub => Some(compile_binary_elementwise(
            node,
            graph,
            "sub",
            BinOpKind::Sub,
        )),
        TraceOp::Div => Some(compile_binary_elementwise(
            node,
            graph,
            "div",
            BinOpKind::Div,
        )),

        // -- Unary element-wise ops -------------------------------------------
        TraceOp::Sqrt => Some(compile_unary_elementwise(
            node,
            graph,
            "sqrt",
            UnaryFnKind::Sqrt,
        )),
        TraceOp::Exp => Some(compile_unary_elementwise(
            node,
            graph,
            "exp",
            UnaryFnKind::Exp,
        )),
        TraceOp::Log => Some(compile_unary_elementwise(
            node,
            graph,
            "log",
            UnaryFnKind::Log,
        )),
        TraceOp::Abs => Some(compile_unary_elementwise(
            node,
            graph,
            "abs",
            UnaryFnKind::Abs,
        )),
        TraceOp::Recip => Some(compile_unary_elementwise(
            node,
            graph,
            "recip",
            UnaryFnKind::Recip,
        )),
        TraceOp::Sin => Some(compile_unary_elementwise(
            node,
            graph,
            "sin",
            UnaryFnKind::Sin,
        )),
        TraceOp::Cos => Some(compile_unary_elementwise(
            node,
            graph,
            "cos",
            UnaryFnKind::Cos,
        )),
        TraceOp::Floor => Some(compile_unary_elementwise(
            node,
            graph,
            "floor",
            UnaryFnKind::Floor,
        )),
        TraceOp::Round => Some(compile_unary_elementwise(
            node,
            graph,
            "round",
            UnaryFnKind::Round,
        )),
        TraceOp::Fract => Some(compile_unary_elementwise(
            node,
            graph,
            "fract",
            UnaryFnKind::Fract,
        )),
        TraceOp::Tanh => Some(compile_activation(
            node,
            graph,
            "tanh",
            ActivationKind::Tanh,
        )),
        TraceOp::Sqr => Some(compile_sqr(node, graph)),
        TraceOp::Neg => Some(compile_neg(node, graph)),

        // -- Binary min/max ---------------------------------------------------
        TraceOp::Maximum => Some(compile_binary_minmax(
            node,
            graph,
            "maximum",
            MinMaxKind::Max,
        )),
        TraceOp::Minimum => Some(compile_binary_minmax(
            node,
            graph,
            "minimum",
            MinMaxKind::Min,
        )),

        // -- Binary functions (function-call syntax) ----------------------------
        TraceOp::Atan2 => Some(compile_binary_fn_elementwise(
            node,
            graph,
            "atan2",
            BinaryFnKind::Atan2,
        )),

        // -- Activations ------------------------------------------------------
        TraceOp::Relu => Some(compile_activation(
            node,
            graph,
            "relu",
            ActivationKind::Relu,
        )),
        TraceOp::Gelu => Some(compile_activation(
            node,
            graph,
            "gelu",
            ActivationKind::Gelu,
        )),
        TraceOp::GeluErf => Some(compile_activation(
            node,
            graph,
            "gelu_erf",
            ActivationKind::GeluErf,
        )),
        TraceOp::Sigmoid => Some(compile_activation(
            node,
            graph,
            "sigmoid",
            ActivationKind::Sigmoid,
        )),
        TraceOp::Silu => Some(compile_silu(node, graph)),
        TraceOp::Elu { alpha } => {
            let a = *alpha as f32;
            if !a.is_finite() {
                return Some(Err(TensorIRError::NonFiniteConstant {
                    name: "Elu alpha".into(),
                    value: *alpha,
                }));
            }
            Some(compile_elu(node, graph, a))
        }
        TraceOp::LeakyRelu { slope } => {
            let s = *slope as f32;
            if !s.is_finite() {
                return Some(Err(TensorIRError::NonFiniteConstant {
                    name: "LeakyRelu slope".into(),
                    value: *slope,
                }));
            }
            Some(compile_leaky_relu(node, graph, s))
        }
        TraceOp::Softplus => Some(compile_softplus(node, graph)),
        TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor { alpha }) => {
            Some(compile_snake_tensor(node, graph, alpha))
        }
        TraceOp::KokoroFused(KokoroFusedOp::AdainSnake { alpha, eps }) => {
            Some(compile_adain_snake(node, graph, alpha, *eps))
        }
        TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }) => {
            Some(compile_adain_leaky_relu(node, graph, *eps, *slope))
        }
        TraceOp::Activation { kind } => Some(compile_named_activation(node, graph, kind.as_str())),

        // -- Powf (decomposed) ------------------------------------------------
        // Special-case common exponents to avoid exp(e*log(x)) which
        // produces NaN for negative inputs (log domain error). (#2751)
        TraceOp::Powf { exponent } if *exponent == 2.0 => Some(compile_sqr(node, graph)),
        TraceOp::Powf { exponent } if *exponent == 0.5 => Some(compile_unary_elementwise(
            node,
            graph,
            "sqrt",
            UnaryFnKind::Sqrt,
        )),
        TraceOp::Powf { exponent } => Some(compile_powf(node, graph, *exponent)),

        // -- Reductions -------------------------------------------------------
        TraceOp::ReduceSum { dim, keepdim } => Some(compile_reduce(
            node,
            graph,
            "reduce_sum",
            ReduceOp::Sum,
            *dim,
            *keepdim,
        )),
        TraceOp::ReduceMean { dim, keepdim } => Some(compile_reduce(
            node,
            graph,
            "reduce_mean",
            ReduceOp::Mean,
            *dim,
            *keepdim,
        )),
        TraceOp::ReduceMax { dim, keepdim } => Some(compile_reduce(
            node,
            graph,
            "reduce_max",
            ReduceOp::Max,
            *dim,
            *keepdim,
        )),
        TraceOp::ReduceMin { dim, keepdim } => Some(compile_reduce(
            node,
            graph,
            "reduce_min",
            ReduceOp::Min,
            *dim,
            *keepdim,
        )),

        // -- Scalar comparison (produces 0.0/1.0 mask) ------------------------
        TraceOp::Compare { op, value } => {
            let ir_op = match *op {
                CompareOp::Eq => CompareOpKind::Eq,
                CompareOp::Ne => CompareOpKind::Ne,
                CompareOp::Lt => CompareOpKind::Lt,
                CompareOp::Le => CompareOpKind::Le,
                CompareOp::Gt => CompareOpKind::Gt,
                CompareOp::Ge => CompareOpKind::Ge,
                _ => {
                    return Some(Err(TensorIRError::UnsupportedTraceOp {
                        name: format!("compare:{op:?}"),
                    }))
                }
            };
            Some(compile_compare(node, graph, ir_op, *value))
        }

        _ => None,
    }
}

/// Compile a `TraceOp::Activation { name }` by dispatching on the name string.
fn compile_named_activation(
    node: &TraceNode,
    graph: &ComputationGraph,
    name: &str,
) -> Result<CompiledStep, TensorIRError> {
    match name {
        "Gelu" | "gelu" => compile_activation(node, graph, "gelu", ActivationKind::Gelu),
        "GeluErf" | "gelu_erf" => {
            compile_activation(node, graph, "gelu_erf", ActivationKind::GeluErf)
        }
        "Relu" | "relu" => compile_activation(node, graph, "relu", ActivationKind::Relu),
        "Sigmoid" | "sigmoid" => {
            compile_activation(node, graph, "sigmoid", ActivationKind::Sigmoid)
        }
        "Silu" | "silu" => compile_silu(node, graph),
        // LeakyRelu/Elu through the generic Activation path use hardcoded defaults
        // (slope=0.01, alpha=1.0) which silently produce wrong results when the model
        // uses non-default parameters (e.g. Kokoro uses LeakyReLU(0.1)). Reject here
        // to force tracing through the dedicated TraceOp::LeakyRelu { slope } variant
        // which carries the actual parameter. See #2267.
        "LeakyRelu" | "leaky_relu" => Err(TensorIRError::UnsupportedTraceOp {
            name: "activation:LeakyRelu — use TraceOp::LeakyRelu { slope } instead of \
                   generic Activation to preserve the actual slope parameter"
                .into(),
        }),
        "Tanh" | "tanh" => compile_activation(node, graph, "tanh", ActivationKind::Tanh),
        "Exp" | "exp" => compile_unary_elementwise(node, graph, "exp", UnaryFnKind::Exp),
        "Log" | "log" => compile_unary_elementwise(node, graph, "log", UnaryFnKind::Log),
        // Same as LeakyRelu above: reject generic Elu to force TraceOp::Elu { alpha }.
        // See #2267.
        "Elu" | "elu" => Err(TensorIRError::UnsupportedTraceOp {
            name: "activation:Elu — use TraceOp::Elu { alpha } instead of \
                   generic Activation to preserve the actual alpha parameter"
                .into(),
        }),
        _ => Err(TensorIRError::UnsupportedTraceOp {
            name: format!("activation:{name}"),
        }),
    }
}
