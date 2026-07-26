// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Auto-generate fused Metal kernels from traced subgraphs.
//!
//! Provides the public API for composing a chain of elementwise `TraceOp`s
//! into a single `KernelDef`, then generating MSL source via the existing
//! codegen pipeline. This is the foundation for auto-fusing arbitrary traced
//! subgraphs into single GPU kernel launches.
//!
//! # Architecture
//!
//! ```text
//! [TraceOp chain]  →  compose_trace_ops_to_kernel_ir()  →  KernelDef
//!                                                              ↓
//!                      auto_fuse_to_msl()               →  MSL source
//!                                                              ↓
//!                      AutoFusedKernel                   →  {KernelDef, MSL, metadata}
//! ```
//!
//! The scalar-level composition reuses [`kernel_compose::emit_trace_op`] for
//! each op, inlining IR nodes into a single `KernelDef`. External inputs
//! (values from outside the chain) become kernel parameters and buffer
//! bindings in the generated MSL.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_dsl::auto_fuse_codegen::{auto_fuse_to_msl, FuseableOp};
//! use nn_core::dyn_tensor::trace::TraceOp;
//!
//! // Chain: exp → relu → add(_, external_y)
//! let ops = vec![
//!     FuseableOp::unary(TraceOp::Exp),
//!     FuseableOp::unary(TraceOp::Relu),
//!     FuseableOp::binary_second_external(TraceOp::Add),
//! ];
//! let fused = auto_fuse_to_msl(&ops, "nn_fused_kernel")?;
//! println!("MSL:\n{}", fused.msl_source);
//! // fused.kernel_def: KernelDef with 2 params (x, external_y)
//! // fused.num_external_inputs: 2
//! ```
//!
//! Part of #3518.

use crate::codegen_msl::emit_msl;
use crate::ir::{
    BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId,
    Param, ScalarType, UnaryFnKind,
};
use crate::tensor_ir::TensorIRError;

/// Describes how an op in a fuseable chain connects to its inputs.
///
/// Each op in the chain either:
/// - Takes the previous chain output as its single input (unary).
/// - Takes the previous chain output + a new external input (binary, LHS from chain).
/// - Takes a new external input + previous chain output (binary, RHS from chain).
/// - Is the first op, taking one or two external inputs.
#[derive(Debug, Clone)]
pub struct FuseableOp {
    /// The trace operation to fuse.
    pub op: TraceOp,
    /// Input wiring: which inputs come from the chain vs. external.
    pub wiring: OpWiring,
}

/// How a fuseable op's inputs are wired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpWiring {
    /// Single input from the previous chain output (or first external input
    /// if this is the first op in the chain).
    Unary,
    /// Binary op: LHS = previous chain output, RHS = new external input.
    BinarySecondExternal,
    /// Binary op: LHS = new external input, RHS = previous chain output.
    BinaryFirstExternal,
    /// Binary op: both inputs are new external inputs (only valid for
    /// the first op in the chain, or when no chain output exists yet).
    BinaryBothExternal,
}

impl FuseableOp {
    /// Create a unary fuseable op (input from previous chain output).
    #[must_use]
    pub fn unary(op: TraceOp) -> Self {
        Self {
            op,
            wiring: OpWiring::Unary,
        }
    }

    /// Create a binary fuseable op where the second input is external.
    ///
    /// LHS = previous chain output, RHS = new external buffer.
    #[must_use]
    pub fn binary_second_external(op: TraceOp) -> Self {
        Self {
            op,
            wiring: OpWiring::BinarySecondExternal,
        }
    }

    /// Create a binary fuseable op where the first input is external.
    ///
    /// LHS = new external buffer, RHS = previous chain output.
    #[must_use]
    pub fn binary_first_external(op: TraceOp) -> Self {
        Self {
            op,
            wiring: OpWiring::BinaryFirstExternal,
        }
    }

    /// Create a binary fuseable op where both inputs are external.
    ///
    /// Only valid as the first op in a chain.
    #[must_use]
    pub fn binary_both_external(op: TraceOp) -> Self {
        Self {
            op,
            wiring: OpWiring::BinaryBothExternal,
        }
    }
}

/// A fused kernel with MSL source, ready for Metal compilation.
#[derive(Debug, Clone)]
pub struct AutoFusedKernel {
    /// The composed scalar kernel definition.
    pub kernel_def: KernelDef,
    /// Complete MSL source (scalar helper + kernel wrapper).
    pub msl_source: String,
    /// Number of external buffer inputs (kernel parameters).
    pub num_external_inputs: usize,
    /// Kernel entry point name (for Metal pipeline creation).
    /// Format: `{name}_kernel`.
    pub entry_point: String,
}

/// Compose a chain of fuseable elementwise ops into a single `KernelDef`.
///
/// Each op contributes scalar IR nodes to the composed kernel. External
/// inputs become kernel parameters (buffer bindings in the MSL wrapper).
///
/// # Errors
///
/// Returns `TensorIRError::UnsupportedTraceOp` if any op in the chain
/// cannot be lowered to scalar IR (e.g., non-elementwise ops).
pub fn compose_trace_ops_to_kernel_ir(
    ops: &[FuseableOp],
    name: &str,
) -> Result<KernelDef, TensorIRError> {
    if ops.is_empty() {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: "empty fuseable op chain".into(),
        });
    }

    let mut params: Vec<Param> = Vec::new();
    let mut nodes: Vec<IRNode> = Vec::new();
    let mut prev_output: Option<NodeId> = None;

    for (step_idx, fuseable) in ops.iter().enumerate() {
        let expected_inputs = op_input_count(&fuseable.op);

        // Resolve inputs based on wiring.
        let ir_inputs = resolve_inputs(
            &fuseable.wiring,
            expected_inputs,
            prev_output,
            step_idx,
            &mut params,
            &mut nodes,
        )?;

        // Emit scalar IR nodes for this trace op.
        let result_id = emit_trace_op(&mut nodes, &fuseable.op, &ir_inputs)?;
        prev_output = Some(result_id);
    }

    let output = prev_output.ok_or_else(|| TensorIRError::UnsupportedTraceOp {
        name: "no output produced from fuseable chain".into(),
    })?;

    let kernel = KernelDef::new(name.to_string(), params, ScalarType::F32, nodes, output);
    kernel
        .validate()
        .map_err(|e| TensorIRError::UnsupportedTraceOp {
            name: format!("auto-fused kernel validation failed: {e}"),
        })?;

    Ok(kernel)
}

/// Compose a chain of fuseable ops and generate MSL source code.
///
/// End-to-end pipeline: `[FuseableOp]` → `KernelDef` → MSL source.
/// Returns an [`AutoFusedKernel`] with the kernel definition, MSL source,
/// and metadata needed for Metal pipeline creation.
///
/// # Errors
///
/// Returns `TensorIRError` if composition or MSL generation fails.
pub fn auto_fuse_to_msl(ops: &[FuseableOp], name: &str) -> Result<AutoFusedKernel, TensorIRError> {
    let kernel_def = compose_trace_ops_to_kernel_ir(ops, name)?;
    let num_external_inputs = kernel_def.params.len();
    let entry_point = format!("{name}_kernel");

    let msl_source = emit_msl(&kernel_def).map_err(|e| TensorIRError::UnsupportedTraceOp {
        name: format!("MSL generation failed for auto-fused kernel '{name}': {e}"),
    })?;

    Ok(AutoFusedKernel {
        kernel_def,
        msl_source,
        num_external_inputs,
        entry_point,
    })
}

// ---------------------------------------------------------------------------
// Internal: input resolution
// ---------------------------------------------------------------------------

/// Resolve IR inputs for a single op based on its wiring configuration.
fn resolve_inputs(
    wiring: &OpWiring,
    expected_inputs: usize,
    prev_output: Option<NodeId>,
    step_idx: usize,
    params: &mut Vec<Param>,
    nodes: &mut Vec<IRNode>,
) -> Result<Vec<NodeId>, TensorIRError> {
    let mut ir_inputs = Vec::with_capacity(expected_inputs);

    match (wiring, expected_inputs) {
        (OpWiring::Unary, 1) => {
            // Single input: chain output if available, else new external.
            let input = match prev_output {
                Some(id) => id,
                None => add_external_param(params, nodes),
            };
            ir_inputs.push(input);
        }
        (OpWiring::BinarySecondExternal, 2) => {
            // LHS = chain output (or external), RHS = new external.
            let lhs = match prev_output {
                Some(id) => id,
                None => add_external_param(params, nodes),
            };
            let rhs = add_external_param(params, nodes);
            ir_inputs.push(lhs);
            ir_inputs.push(rhs);
        }
        (OpWiring::BinaryFirstExternal, 2) => {
            // LHS = new external, RHS = chain output (or external).
            let lhs = add_external_param(params, nodes);
            let rhs = match prev_output {
                Some(id) => id,
                None => add_external_param(params, nodes),
            };
            ir_inputs.push(lhs);
            ir_inputs.push(rhs);
        }
        (OpWiring::BinaryBothExternal, 2) => {
            if prev_output.is_some() && step_idx > 0 {
                return Err(TensorIRError::UnsupportedTraceOp {
                    name: "BinaryBothExternal only valid for first op in chain".into(),
                });
            }
            let lhs = add_external_param(params, nodes);
            let rhs = add_external_param(params, nodes);
            ir_inputs.push(lhs);
            ir_inputs.push(rhs);
        }
        _ => {
            return Err(TensorIRError::UnsupportedTraceOp {
                name: format!(
                    "wiring {wiring:?} incompatible with {expected_inputs}-input op at step {step_idx}"
                ),
            });
        }
    }

    Ok(ir_inputs)
}

/// Add a new external parameter and its corresponding Param IR node.
fn add_external_param(params: &mut Vec<Param>, nodes: &mut Vec<IRNode>) -> NodeId {
    let param_idx = params.len();
    params.push(Param::new(format!("p{param_idx}"), ScalarType::F32));
    let ir_id = NodeId::new(nodes.len());
    nodes.push(IRNode::new(ir_id, IRNodeKind::Param(param_idx)));
    ir_id
}

// ---------------------------------------------------------------------------
// Internal: op classification and IR emission
// ---------------------------------------------------------------------------

/// Returns the number of external inputs for a `TraceOp`.
pub(crate) fn op_input_count(op: &TraceOp) -> usize {
    match op {
        TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Mul
        | TraceOp::Div
        | TraceOp::Maximum
        | TraceOp::Minimum
        | TraceOp::Atan2 => 2,
        _ => 1,
    }
}

/// Emit scalar IR nodes for a single trace op, returning the output NodeId.
///
/// This is a self-contained version of the logic in `kernel_compose.rs`,
/// used by the standalone auto-fuse codegen path.
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

        // Neg: 0 - x
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

        // Power
        TraceOp::Powf { exponent } => emit_powf(nodes, inputs[0], *exponent),

        // Softplus: ln(1 + exp(x))
        TraceOp::Softplus => emit_softplus(nodes, inputs[0]),

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
                name: format!("auto-fuse does not support op: {}", other.canonical_name()),
            });
        }
    };
    Ok(result)
}

// ---------------------------------------------------------------------------
// IR node emitters (mirrors kernel_compose.rs for self-containment)
// ---------------------------------------------------------------------------

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

/// `gelu(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
fn emit_gelu(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let half = emit_literal(nodes, 0.5);
    let half_x = emit_binop(nodes, BinOpKind::Mul, half, x);

    let x2 = emit_binop(nodes, BinOpKind::Mul, x, x);
    let x3 = emit_binop(nodes, BinOpKind::Mul, x2, x);

    let coeff = emit_literal(nodes, 0.044715);
    let coeff_x3 = emit_binop(nodes, BinOpKind::Mul, coeff, x3);
    let inner = emit_binop(nodes, BinOpKind::Add, x, coeff_x3);

    let sqrt_2_pi = emit_literal(nodes, 0.797_884_560_802_865_4);
    let scaled = emit_binop(nodes, BinOpKind::Mul, sqrt_2_pi, inner);
    let tanh_val = emit_unary(nodes, UnaryFnKind::Tanh, scaled);

    let one = emit_literal(nodes, 1.0);
    let one_plus = emit_binop(nodes, BinOpKind::Add, one, tanh_val);
    emit_binop(nodes, BinOpKind::Mul, half_x, one_plus)
}

/// `gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`
fn emit_gelu_erf(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let erf_val = emit_erf_polynomial(nodes, x);
    let one = emit_literal(nodes, 1.0);
    let one_plus_erf = emit_binop(nodes, BinOpKind::Add, one, erf_val);
    let half = emit_literal(nodes, 0.5);
    let half_x = emit_binop(nodes, BinOpKind::Mul, half, x);
    emit_binop(nodes, BinOpKind::Mul, half_x, one_plus_erf)
}

/// A&S 7.1.26 erf polynomial: `erf(x/sqrt(2))`.
fn emit_erf_polynomial(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let inv_sqrt2 = emit_literal(nodes, std::f64::consts::FRAC_1_SQRT_2);
    let u = emit_binop(nodes, BinOpKind::Mul, x, inv_sqrt2);

    let zero = emit_literal(nodes, 0.0);
    let pos_one = emit_literal(nodes, 1.0);
    let neg_one = emit_literal(nodes, -1.0);
    let cond = emit_compare(nodes, CompareOpKind::Ge, u, zero);
    let sign = emit_select(nodes, cond, pos_one, neg_one);

    let ax = emit_unary(nodes, UnaryFnKind::Abs, u);

    let p = emit_literal(nodes, 0.327_591_1);
    let p_ax = emit_binop(nodes, BinOpKind::Mul, p, ax);
    let one_plus = emit_binop(nodes, BinOpKind::Add, pos_one, p_ax);
    let t = emit_unary(nodes, UnaryFnKind::Recip, one_plus);

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

    let neg_u_sq = emit_binop(nodes, BinOpKind::Mul, u, u);
    let neg_u_sq = emit_binop(nodes, BinOpKind::Sub, zero, neg_u_sq);
    let exp_val = emit_unary(nodes, UnaryFnKind::Exp, neg_u_sq);
    let poly_exp = emit_binop(nodes, BinOpKind::Mul, h, exp_val);

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

/// `clamp(x, min, max)` with optional bounds.
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

/// `softplus(x) = ln(1 + exp(x))`
fn emit_softplus(nodes: &mut Vec<IRNode>, x: NodeId) -> NodeId {
    let exp_x = emit_unary(nodes, UnaryFnKind::Exp, x);
    let one = emit_literal(nodes, 1.0);
    let one_plus_exp = emit_binop(nodes, BinOpKind::Add, one, exp_x);
    emit_unary(nodes, UnaryFnKind::Log, one_plus_exp)
}

/// Scalar powf lowering matching the eager GPU semantics.
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

use nn_core::dyn_tensor::trace::TraceOp;

#[cfg(test)]
#[path = "auto_fuse_codegen_tests.rs"]
mod tests;
