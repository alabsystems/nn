// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for composed elementwise kernels from `KernelDef` IR.
//!
//! Translates the same `KernelDef` scalar IR that MSL codegen uses into
//! HIP `__global__` kernel functions. The scalar math is standard C++ —
//! the only differences are type names, math intrinsics, and kernel wrapper
//! syntax.
//!
//! Part of #2241 (HIP codegen Phase 4: deferred ops).

use crate::codegen_hip::hip_type;
use crate::HipCodegenError;
use nn_dsl::codegen_shared::powi_stmts;
use nn_dsl::ir::BinaryFnKind;
use nn_dsl::{
    BinOpKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, ScalarType,
    UnaryFnKind,
};
use std::borrow::Cow;

/// Emit a complete HIP C++ source for an elementwise kernel defined by IR.
///
/// Generates a scalar helper function from the IR nodes, then wraps it in
/// a `__global__` kernel that dispatches one thread per element.
///
/// For kernels with many parameters, all inputs are passed as direct pointer
/// arguments (HIP has no Metal buffer-index limit).
pub fn emit_elementwise_hip(kernel: &KernelDef) -> Result<String, HipCodegenError> {
    let ret = hip_type(kernel.return_type)?;
    let helper_name = format!("_nn_{}", kernel.name);
    let scalar = emit_scalar_fn(kernel, ret, &helper_name)?;
    let wrapper = emit_hip_kernel_wrapper(kernel, ret, &helper_name)?;
    Ok(format!("{scalar}\n\n{wrapper}\n"))
}

/// Emit the scalar helper function from IR nodes.
fn emit_scalar_fn(kernel: &KernelDef, ret: &str, fn_name: &str) -> Result<String, HipCodegenError> {
    let params = kernel
        .params
        .iter()
        .map(|p| Ok(format!("{} {}", hip_type(p.ty)?, p.name)))
        .collect::<Result<Vec<String>, HipCodegenError>>()?;

    let mut body_lines = Vec::new();
    for node in &kernel.nodes {
        if let Some(line) = emit_node(node, kernel, ret)? {
            body_lines.push(line);
        }
    }

    let output_ref = node_ref(kernel.output, kernel)?;
    Ok(format!(
        "__device__ {ret} {fn_name}({params}) {{\n{body}\n    return {output_ref};\n}}",
        params = params.join(", "),
        body = body_lines.join("\n"),
    ))
}

/// Emit a `__global__` kernel wrapper that reads per-element inputs and
/// writes the scalar function result to the output buffer.
fn emit_hip_kernel_wrapper(
    kernel: &KernelDef,
    ret: &str,
    scalar_fn_name: &str,
) -> Result<String, HipCodegenError> {
    let mut params = Vec::new();
    for param in &kernel.params {
        params.push(format!(
            "    const {}* __restrict__ {}",
            hip_type(param.ty)?,
            param.name,
        ));
    }
    params.push(format!("    {ret}* __restrict__ out"));
    params.push("    const unsigned int total".to_string());

    let call_args: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{}[tid]", p.name))
        .collect();

    Ok(format!(
        "extern \"C\" __global__ void {name}_kernel(\n{params}\n) {{\n    \
         unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;\n    \
         if (tid >= total) return;\n    \
         out[tid] = {scalar_fn_name}({call_args});\n}}",
        name = kernel.name,
        params = params.join(",\n"),
        call_args = call_args.join(", "),
    ))
}

/// Emit a single IR node as a HIP C++ statement.
fn emit_node(
    node: &IRNode,
    kernel: &KernelDef,
    ret: &str,
) -> Result<Option<String>, HipCodegenError> {
    let tid = node.id.index();
    match &node.kind {
        IRNodeKind::Param(_) => Ok(None),
        IRNodeKind::Literal(v) => {
            let safe_v = clamp_literal(kernel.return_type, *v);
            Ok(Some(format!(
                "    {ret} t{tid} = {};",
                format_hip_f64(safe_v)
            )))
        }
        IRNodeKind::BinOp { op, lhs, rhs } => {
            let l = node_ref(*lhs, kernel)?;
            let r = node_ref(*rhs, kernel)?;
            let op_str = binop_str(*op)?;
            Ok(Some(format!("    {ret} t{tid} = {l} {op_str} {r};")))
        }
        IRNodeKind::Compare { op, lhs, rhs } => {
            let l = node_ref(*lhs, kernel)?;
            let r = node_ref(*rhs, kernel)?;
            let cmp = compare_str(*op)?;
            Ok(Some(format!("    bool t{tid} = {l} {cmp} {r};")))
        }
        IRNodeKind::UnaryFn { op, input } => {
            let arg = node_ref(*input, kernel)?;
            match hip_unary_fn(*op)? {
                HipUnaryOp::Named(name) => Ok(Some(format!("    {ret} t{tid} = {name}({arg});"))),
                HipUnaryOp::Reciprocal => Ok(Some(format!("    {ret} t{tid} = {ret}(1) / {arg};"))),
            }
        }
        IRNodeKind::Powi { base, exp } => {
            let b = node_ref(*base, kernel)?;
            Ok(Some(powi_stmts(&b, *exp, ret, tid)))
        }
        IRNodeKind::Clamp { input, min, max } => {
            let x = node_ref(*input, kernel)?;
            let lo = node_ref(*min, kernel)?;
            let hi = node_ref(*max, kernel)?;
            // HIP does not have a generic `clamp` built-in for all types;
            // use fminf/fmaxf composition for float.
            Ok(Some(format!(
                "    {ret} t{tid} = fminf(fmaxf({x}, {lo}), {hi});"
            )))
        }
        IRNodeKind::MinMax { op, lhs, rhs } => {
            let a = node_ref(*lhs, kernel)?;
            let b = node_ref(*rhs, kernel)?;
            let fn_name = match op {
                MinMaxKind::Min => "fminf",
                MinMaxKind::Max => "fmaxf",
                _ => {
                    return Err(HipCodegenError::UnsupportedIRVariant {
                        variant_desc: "MinMaxKind",
                    })
                }
            };
            Ok(Some(format!("    {ret} t{tid} = {fn_name}({a}, {b});")))
        }
        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => {
            let cond_ref = node_ref(*cond, kernel)?;
            let then_ref = node_ref(*then_val, kernel)?;
            let else_ref = node_ref(*else_val, kernel)?;
            Ok(Some(format!(
                "    {ret} t{tid} = ({cond_ref} ? {then_ref} : {else_ref});"
            )))
        }
        IRNodeKind::SumReduce { inputs } => {
            if let Some((first, rest)) = inputs.split_first() {
                let mut expr = node_ref(*first, kernel)?.into_owned();
                for node_id in rest {
                    expr.push_str(" + ");
                    expr.push_str(&node_ref(*node_id, kernel)?);
                }
                Ok(Some(format!("    {ret} t{tid} = {expr};")))
            } else {
                Ok(Some(format!("    {ret} t{tid} = {ret}(0);")))
            }
        }
        IRNodeKind::BinaryFn { op, lhs, rhs } => {
            let a = node_ref(*lhs, kernel)?;
            let b = node_ref(*rhs, kernel)?;
            let fn_name = match op {
                BinaryFnKind::Atan2 => "atan2f",
                _ => {
                    return Err(HipCodegenError::UnsupportedIRVariant {
                        variant_desc: "BinaryFnKind",
                    })
                }
            };
            Ok(Some(format!("    {ret} t{tid} = {fn_name}({a}, {b});")))
        }
        // Error for future #[non_exhaustive] variants — never silently
        // produce wrong kernel code.
        _ => Err(HipCodegenError::UnsupportedIRVariant {
            variant_desc: "IRNodeKind",
        }),
    }
}

/// Resolve a node reference to a variable name or parameter name.
fn node_ref(id: NodeId, kernel: &KernelDef) -> Result<Cow<'_, str>, HipCodegenError> {
    let node = kernel.nodes.get(id.index()).ok_or_else(|| {
        HipCodegenError::InvalidParameter(format!("invalid node ref: {}", id.index()))
    })?;
    match &node.kind {
        IRNodeKind::Param(idx) => {
            let param = kernel.params.get(*idx).ok_or_else(|| {
                HipCodegenError::InvalidParameter(format!(
                    "invalid param ref: {} (have {})",
                    idx,
                    kernel.params.len()
                ))
            })?;
            Ok(Cow::Borrowed(&param.name))
        }
        _ => Ok(Cow::Owned(format!("t{}", id.index()))),
    }
}

// --- Helper functions ---

fn binop_str(op: BinOpKind) -> Result<&'static str, HipCodegenError> {
    match op {
        BinOpKind::Add => Ok("+"),
        BinOpKind::Sub => Ok("-"),
        BinOpKind::Mul => Ok("*"),
        BinOpKind::Div => Ok("/"),
        _ => Err(HipCodegenError::UnsupportedIRVariant {
            variant_desc: "BinOpKind",
        }),
    }
}

fn compare_str(op: CompareOpKind) -> Result<&'static str, HipCodegenError> {
    match op {
        CompareOpKind::Lt => Ok("<"),
        CompareOpKind::Le => Ok("<="),
        CompareOpKind::Gt => Ok(">"),
        CompareOpKind::Ge => Ok(">="),
        CompareOpKind::Eq => Ok("=="),
        CompareOpKind::Ne => Ok("!="),
        _ => Err(HipCodegenError::UnsupportedIRVariant {
            variant_desc: "CompareOpKind",
        }),
    }
}

enum HipUnaryOp {
    Named(&'static str),
    Reciprocal,
}

fn hip_unary_fn(op: UnaryFnKind) -> Result<HipUnaryOp, HipCodegenError> {
    match op {
        UnaryFnKind::Sin => Ok(HipUnaryOp::Named("sinf")),
        UnaryFnKind::Cos => Ok(HipUnaryOp::Named("cosf")),
        UnaryFnKind::Sqrt => Ok(HipUnaryOp::Named("sqrtf")),
        UnaryFnKind::Rsqrt => Ok(HipUnaryOp::Named("rsqrtf")),
        UnaryFnKind::Exp => Ok(HipUnaryOp::Named("expf")),
        UnaryFnKind::Abs => Ok(HipUnaryOp::Named("fabsf")),
        UnaryFnKind::Recip => Ok(HipUnaryOp::Reciprocal),
        UnaryFnKind::Tanh => Ok(HipUnaryOp::Named("tanhf")),
        UnaryFnKind::Log => Ok(HipUnaryOp::Named("logf")),
        UnaryFnKind::Floor => Ok(HipUnaryOp::Named("floorf")),
        UnaryFnKind::Round => Ok(HipUnaryOp::Named("rintf")),
        _ => Err(HipCodegenError::UnsupportedIRVariant {
            variant_desc: "UnaryFnKind",
        }),
    }
}

/// Clamp a literal value to the representable range of the target type.
fn clamp_literal(ty: ScalarType, v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    match ty {
        ScalarType::F32 => {
            let max = f64::from(f32::MAX);
            v.clamp(-max, max)
        }
        ScalarType::F16 | ScalarType::BF16 => {
            let max: f64 = 65504.0;
            v.clamp(-max, max)
        }
        _ => v,
    }
}

/// Format an f64 literal for HIP C++ source.
fn format_hip_f64(v: f64) -> String {
    if v.is_nan() {
        "nanf(\"\")".to_string()
    } else if v == f64::INFINITY {
        "HUGE_VALF".to_string()
    } else if v == f64::NEG_INFINITY {
        "(-HUGE_VALF)".to_string()
    } else {
        format!("{v:.8}f")
    }
}

// `powi_stmts` imported from `nn_dsl::codegen_shared` (shared with MSL backend).
// Part of #3338.

#[cfg(test)]
#[path = "codegen_hip_tensor_emit_elementwise_tests.rs"]
mod tests;
