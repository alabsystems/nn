// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL (Metal Shading Language) code generation from KernelIR.
//!
//! Emits a standalone MSL source file containing:
//! 1. A scalar helper function (the kernel math)
//! 2. A `[[kernel]]` compute function that dispatches per-element
//!
//! Pure mapping helpers (type conversions, operator strings, literal formatting,
//! powi expansion) live in [`codegen_msl_helpers`](super::codegen_msl_helpers).

#[path = "codegen_msl_helpers.rs"]
mod helpers;

// Re-export pub(crate) items so callers continue using `codegen_msl::msl_type` etc.
pub(crate) use helpers::{
    clamp_literal_for_type, compare_op, format_float, msl_accumulator_type, msl_fn, msl_type,
    validate_buffer_count, wrapper_out_buffer_index, wrapper_total_buffer_index, MslUnaryOp,
    MAX_METAL_BUFFER_INDEX, MSL_PRELUDE,
};

// Re-exported as `pub` so the dispatch layer (nn-metal) can use the same
// threshold to decide between direct-binding and packed-buffer encoding.
// Part of #1649.
use crate::codegen_shared::powi_stmts;
use helpers::msl_binop;
pub use helpers::MAX_DIRECT_BINDING_INPUTS;
use std::borrow::Cow;

use crate::{
    ir::{BinaryFnKind, IRError, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, ScalarType},
    precision::{PrecisionContract, PrecisionTier},
};

/// MSL reserved words that cannot be used as kernel/parameter identifiers.
///
/// Previously in `ir_validate.rs` (moved here as part of #586: backend-specific
/// reserved words belong in the backend codegen module, not the backend-agnostic IR).
const MSL_RESERVED: &[&str] = &[
    "thread",
    "device",
    "constant",
    "kernel",
    "threadgroup",
    "fragment",
    "vertex",
    "using",
    "namespace",
    "metal",
    "float",
    "half",
    "int",
    "uint",
    "bool",
    "void",
    "true",
    "false",
    "if",
    "else",
    "for",
    "while",
    "do",
    "return",
    "switch",
    "case",
    "break",
    "continue",
    "struct",
    "class",
    "enum",
    "union",
    "typedef",
    "static",
    "inline",
    "volatile",
    "const",
    "extern",
    "sizeof",
    "new",
    "delete",
];

/// Validate that a kernel's name and parameter names do not collide with MSL reserved words.
///
/// Called at MSL emit time, not during IR construction. This keeps the IR
/// backend-agnostic while still catching reserved word conflicts before
/// generating invalid MSL. (Part of #586.)
fn validate_msl_identifiers(kernel: &KernelDef) -> Result<(), IRError> {
    if MSL_RESERVED.contains(&kernel.name.as_str()) {
        return Err(IRError::InvalidIdentifier {
            name: kernel.name.clone(),
            context: "kernel name",
            reason: "is an MSL reserved word",
        });
    }
    for param in &kernel.params {
        if MSL_RESERVED.contains(&param.name.as_str()) {
            return Err(IRError::InvalidIdentifier {
                name: param.name.clone(),
                context: "parameter name",
                reason: "is an MSL reserved word",
            });
        }
    }
    Ok(())
}

/// Emit a complete MSL source file for an element-wise kernel.
///
/// Each parameter becomes a `device const T*` buffer. The kernel dispatches
/// one thread per element, reading `param[tid]` and writing `out[tid]`.
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[must_use = "MSL source string is computed but not used"]
pub fn emit_msl(kernel: &KernelDef) -> Result<String, IRError> {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    emit_msl_with_contract(kernel, contract)
}

/// Emit a complete MSL source file with an explicit precision contract.
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed or exceeds buffer limits.
/// For kernels with >28 parameters, use [`emit_msl_packed_with_contract`] instead.
#[must_use = "MSL source string is computed but not used"]
pub fn emit_msl_with_contract(
    kernel: &KernelDef,
    contract: PrecisionContract,
) -> Result<String, IRError> {
    kernel.validate()?;
    validate_msl_identifiers(kernel)?;
    validate_buffer_count(kernel.params.len())?;
    // Prefix the scalar helper function name with `_nn_` to avoid collisions
    // with MSL built-in functions (e.g., `rsqrt`, `sin`, `exp`). The kernel
    // entry point remains `{name}_kernel`.
    let helper_name = format!("_nn_{}", kernel.name);
    let scalar = emit_scalar_fn_inner(kernel, contract, &helper_name)?;
    let wrapper = emit_kernel_wrapper(kernel, &helper_name);
    Ok(format!("{MSL_PRELUDE}{scalar}\n\n{wrapper}\n"))
}

/// Emit a packed MSL source file for kernels with >28 parameters.
///
/// Instead of binding each parameter as a separate `[[buffer(i)]]`, the packed
/// variant reads all parameters from a single contiguous buffer using an offsets
/// array. This uses 4 buffer slots regardless of parameter count:
/// - `buffer(0)`: packed_inputs (all params concatenated element-wise)
/// - `buffer(1)`: offsets (element offset per parameter, `constant uint*`)
/// - `buffer(2)`: output
/// - `buffer(3)`: total element count
///
/// Part of #1649.
#[must_use = "MSL source string is computed but not used"]
pub(crate) fn emit_msl_packed_with_contract(
    kernel: &KernelDef,
    contract: PrecisionContract,
) -> Result<String, IRError> {
    kernel.validate()?;
    validate_msl_identifiers(kernel)?;
    // No buffer count validation — packed variant handles any param count.
    let helper_name = format!("_nn_{}", kernel.name);
    let scalar = emit_scalar_fn_inner(kernel, contract, &helper_name)?;
    let wrapper = emit_packed_kernel_wrapper(kernel, &helper_name);
    Ok(format!("{MSL_PRELUDE}{scalar}\n\n{wrapper}\n"))
}

/// Emit just the scalar helper function (useful for testing).
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[must_use = "MSL scalar function string is computed but not used"]
pub fn emit_scalar_fn(kernel: &KernelDef) -> Result<String, IRError> {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, kernel.return_type);
    emit_scalar_fn_with_contract(kernel, contract)
}

/// Emit just the scalar helper function with an explicit precision contract.
///
/// # Errors
///
/// Returns `IRError` if the kernel IR is malformed.
#[must_use = "MSL scalar function string is computed but not used"]
pub(crate) fn emit_scalar_fn_with_contract(
    kernel: &KernelDef,
    contract: PrecisionContract,
) -> Result<String, IRError> {
    kernel.validate()?;
    validate_msl_identifiers(kernel)?;
    emit_scalar_fn_inner(kernel, contract, &kernel.name)
}

/// Inner implementation shared by the public entry points (called after
/// validation).
///
/// `fn_name` controls the emitted function name. The public `emit_scalar_fn`
/// API passes `kernel.name` directly; `emit_msl_with_contract` passes a
/// prefixed name (e.g., `_nn_rsqrt`) to avoid collisions with MSL built-in
/// functions like `rsqrt`, `sin`, `exp`, etc.
fn emit_scalar_fn_inner(
    kernel: &KernelDef,
    contract: PrecisionContract,
    fn_name: &str,
) -> Result<String, IRError> {
    let ret = msl_type(kernel.return_type);
    let acc = msl_accumulator_type(kernel.return_type);
    let use_acc = acc != ret;
    let params: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{} {}", msl_type(p.ty), p.name))
        .collect();

    let mut body_lines = Vec::new();

    // Float-accumulator mode: promote half inputs to float for precision.
    if use_acc {
        for p in &kernel.params {
            body_lines.push(format!("    {acc} {}_f = {acc}({});", p.name, p.name));
        }
    }

    for node in &kernel.nodes {
        if let Some(line) = emit_node(node, kernel, contract, acc, use_acc)? {
            body_lines.push(line);
        }
    }

    let output_ref = node_ref(kernel.output, kernel, use_acc)?;
    let return_stmt = if use_acc {
        format!("return {ret}({output_ref});")
    } else {
        format!("return {output_ref};")
    };
    Ok(format!(
        "{ret} {fn_name}({params}) {{\n{body}\n    {return_stmt}\n}}",
        params = params.join(", "),
        body = body_lines.join("\n"),
    ))
}

// Wrapper emission extracted to codegen_msl_wrapper.rs (500-line limit).
#[path = "codegen_msl_wrapper.rs"]
mod wrapper;
use wrapper::{emit_kernel_wrapper, emit_packed_kernel_wrapper};

fn emit_node(
    node: &IRNode,
    kernel: &KernelDef,
    contract: PrecisionContract,
    int_type: &str,
    use_acc: bool,
) -> Result<Option<String>, IRError> {
    let tid = node.id.index();
    match &node.kind {
        IRNodeKind::Param(_) => Ok(None), // params are function arguments
        IRNodeKind::Literal(v) => {
            // Float-accumulator mode: clamp to F32 range (not F16).
            let clamp_ty = if use_acc {
                ScalarType::F32
            } else {
                kernel.return_type
            };
            let safe_v = clamp_literal_for_type(*v, clamp_ty);
            Ok(Some(format!(
                "    {int_type} t{tid} = {};",
                format_float(safe_v)
            )))
        }
        IRNodeKind::BinOp { op, lhs, rhs } => {
            let l = node_ref(*lhs, kernel, use_acc)?;
            let r = node_ref(*rhs, kernel, use_acc)?;
            let op_str = msl_binop(*op);
            Ok(Some(format!("    {int_type} t{tid} = {l} {op_str} {r};")))
        }
        IRNodeKind::Compare { op, lhs, rhs } => {
            let l = node_ref(*lhs, kernel, use_acc)?;
            let r = node_ref(*rhs, kernel, use_acc)?;
            let cmp = compare_op(*op);
            Ok(Some(format!("    bool t{tid} = {l} {cmp} {r};")))
        }
        IRNodeKind::UnaryFn { op, input } => {
            let arg = node_ref(*input, kernel, use_acc)?;
            match msl_fn(*op, contract.tier) {
                MslUnaryOp::Named(fn_name) => {
                    Ok(Some(format!("    {int_type} t{tid} = {fn_name}({arg});")))
                }
                MslUnaryOp::Negation => Ok(Some(format!("    {int_type} t{tid} = -({arg});"))),
                MslUnaryOp::Reciprocal => Ok(Some(format!(
                    "    {int_type} t{tid} = {int_type}(1) / {arg};"
                ))),
            }
        }
        IRNodeKind::Powi { base, exp } => {
            let b = node_ref(*base, kernel, use_acc)?;
            Ok(Some(powi_stmts(&b, *exp, int_type, tid)))
        }
        IRNodeKind::Clamp { input, min, max } => {
            let x = node_ref(*input, kernel, use_acc)?;
            let lo = node_ref(*min, kernel, use_acc)?;
            let hi = node_ref(*max, kernel, use_acc)?;
            Ok(Some(format!(
                "    {int_type} t{tid} = clamp({x}, {lo}, {hi});"
            )))
        }
        IRNodeKind::MinMax { op, lhs, rhs } => {
            let a = node_ref(*lhs, kernel, use_acc)?;
            let b = node_ref(*rhs, kernel, use_acc)?;
            let fn_name = match op {
                MinMaxKind::Min => "min",
                MinMaxKind::Max => "max",
            };
            Ok(Some(format!(
                "    {int_type} t{tid} = {fn_name}({a}, {b});"
            )))
        }
        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => {
            let cond_ref = node_ref(*cond, kernel, use_acc)?;
            let then_ref = node_ref(*then_val, kernel, use_acc)?;
            let else_ref = node_ref(*else_val, kernel, use_acc)?;
            Ok(Some(format!(
                "    {int_type} t{tid} = ({cond_ref} ? {then_ref} : {else_ref});"
            )))
        }
        IRNodeKind::SumReduce { inputs } => {
            if let Some((first, rest)) = inputs.split_first() {
                let mut expr = node_ref(*first, kernel, use_acc)?.into_owned();
                for node in rest {
                    expr.push_str(" + ");
                    expr.push_str(&node_ref(*node, kernel, use_acc)?);
                }
                Ok(Some(format!("    {int_type} t{tid} = {expr};")))
            } else {
                // Kernel validation rejects empty reductions, but keep a stable
                // fallback for direct emitter calls that skip validation.
                Ok(Some(format!("    {int_type} t{tid} = {int_type}(0);")))
            }
        }
        IRNodeKind::BinaryFn { op, lhs, rhs } => {
            let a = node_ref(*lhs, kernel, use_acc)?;
            let b = node_ref(*rhs, kernel, use_acc)?;
            let fn_name = match op {
                BinaryFnKind::Atan2 => "atan2",
            };
            Ok(Some(format!(
                "    {int_type} t{tid} = {fn_name}({a}, {b});"
            )))
        }
    }
}

fn node_ref(id: NodeId, kernel: &KernelDef, use_acc: bool) -> Result<Cow<'_, str>, IRError> {
    let node = kernel
        .nodes
        .get(id.index())
        .ok_or(IRError::InvalidNodeRef(id))?;
    match &node.kind {
        IRNodeKind::Param(idx) => {
            let param = kernel
                .params
                .get(*idx)
                .ok_or(IRError::InvalidParamRef(*idx, kernel.params.len()))?;
            if use_acc {
                Ok(Cow::Owned(format!("{}_f", param.name)))
            } else {
                Ok(Cow::Borrowed(&param.name))
            }
        }
        IRNodeKind::Literal(_)
        | IRNodeKind::BinOp { .. }
        | IRNodeKind::Compare { .. }
        | IRNodeKind::UnaryFn { .. }
        | IRNodeKind::Powi { .. }
        | IRNodeKind::Clamp { .. }
        | IRNodeKind::MinMax { .. }
        | IRNodeKind::Select { .. }
        | IRNodeKind::SumReduce { .. }
        | IRNodeKind::BinaryFn { .. } => Ok(Cow::Owned(format!("t{}", id.index()))),
    }
}

#[cfg(test)]
#[path = "codegen_msl_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "codegen_msl_tests_ops.rs"]
mod tests_ops;

#[cfg(test)]
#[path = "codegen_msl_tests_clamp.rs"]
mod tests_clamp;
