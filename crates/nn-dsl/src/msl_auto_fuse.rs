// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Auto-generated MSL codegen for fused elementwise subgraphs.
//!
//! Given a composed `KernelDef` (from `kernel_compose::build_fused_scalar_kernel`)
//! and metadata about input shapes / output shape, generates a single Metal
//! compute kernel with broadcast-aware buffer indexing.
//!
//! This is the codegen half of the auto-fusion pipeline:
//! 1. `trace_compile_fusion` detects fusible chains in the trace graph
//! 2. `kernel_compose` builds a composed scalar `KernelDef`
//! 3. **This module** generates MSL source from that `KernelDef` + shape metadata
//!
//! The existing `codegen_msl::emit_msl()` generates MSL from a `KernelDef`
//! with flat `param[tid]` indexing (all buffers same size). This module adds
//! broadcast-aware indexing: inputs with shapes smaller than the output use
//! modular index remapping, matching `build_broadcast_index_body()`.
//!
//! Part of #3518.

use crate::codegen_msl::{
    self, clamp_literal_for_type, compare_op, format_float, msl_fn, MslUnaryOp, MSL_PRELUDE,
};
use crate::codegen_msl_structural::safe_msl_uint;
use crate::codegen_shared::{powi_stmts, row_major_strides};
use crate::ir::{
    BinOpKind, BinaryFnKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, ScalarType,
};
use crate::precision::{PrecisionContract, PrecisionTier};
use crate::tensor_ir::BroadcastAlignment;

/// Metadata for generating a fused MSL kernel from a composed `KernelDef`.
///
/// Each external input (kernel parameter) has a shape that may differ from
/// the output shape. Inputs with smaller shapes require broadcast indexing.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FusedKernelMeta {
    /// Shape of each external input, in kernel parameter order.
    /// `input_shapes[i]` corresponds to `kernel.params[i]`.
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shape of the fused kernel (all inputs broadcast to this).
    pub output_shape: Vec<usize>,
    /// Broadcast alignment for inputs with fewer dimensions.
    pub alignment: BroadcastAlignment,
    /// Scalar dtype for buffers.
    pub dtype: ScalarType,
}

impl FusedKernelMeta {
    /// Create metadata for a fused kernel.
    #[must_use]
    pub fn new(
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        alignment: BroadcastAlignment,
        dtype: ScalarType,
    ) -> Self {
        Self {
            input_shapes,
            output_shape,
            alignment,
            dtype,
        }
    }

    /// Total elements in the output tensor.
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.output_shape.iter().product()
    }
}

/// Result of MSL codegen for a fused elementwise subgraph.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FusedMslResult {
    /// Complete MSL source (prelude + kernel function).
    pub msl_source: String,
    /// Metal kernel function name (for pipeline creation).
    pub kernel_name: String,
    /// Number of buffer arguments: inputs + output + total.
    pub buffer_count: usize,
    /// Threadgroup size used for dispatch.
    pub threadgroup_size: usize,
}

impl FusedMslResult {
    pub(crate) fn new(
        msl_source: String,
        kernel_name: String,
        buffer_count: usize,
        threadgroup_size: usize,
    ) -> Self {
        Self {
            msl_source,
            kernel_name,
            buffer_count,
            threadgroup_size,
        }
    }
}

/// Default threadgroup size for elementwise fused kernels.
pub(crate) const FUSED_THREADGROUP_SIZE: usize = 256;

/// Generate MSL source for a fused elementwise subgraph.
///
/// Takes a composed `KernelDef` (built by `kernel_compose`) and metadata
/// about input/output shapes. Produces a single Metal compute kernel where:
/// - Each external input gets a `device const T*` buffer binding
/// - Inputs matching the output shape use flat `tid` indexing
/// - Inputs with smaller shapes use broadcast modular indexing
/// - One output buffer and one total-elements constant
///
/// # Errors
///
/// Returns an error if:
/// - The `KernelDef` fails validation
/// - Input shape count does not match parameter count
/// - Shape dimensions exceed `u32::MAX`
/// - Buffer count exceeds Metal hardware limit (31 slots)
#[must_use = "returns a Result that may contain an error"]
pub fn generate_fused_msl(
    kernel: &KernelDef,
    meta: &FusedKernelMeta,
) -> Result<FusedMslResult, FusedMslError> {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, meta.dtype);
    generate_fused_msl_with_contract(kernel, meta, contract)
}

/// Generate MSL source with an explicit precision contract.
///
/// # Errors
///
/// Returns `FusedMslError` if codegen fails.
#[must_use = "returns a Result that may contain an error"]
pub fn generate_fused_msl_with_contract(
    kernel: &KernelDef,
    meta: &FusedKernelMeta,
    contract: PrecisionContract,
) -> Result<FusedMslResult, FusedMslError> {
    // Validate inputs.
    kernel
        .validate()
        .map_err(|e| FusedMslError::IrValidation(e.to_string()))?;

    if meta.input_shapes.len() != kernel.params.len() {
        return Err(FusedMslError::ShapeParamMismatch {
            shapes: meta.input_shapes.len(),
            params: kernel.params.len(),
        });
    }

    let param_count = kernel.params.len();
    // buffers: param_count inputs + 1 output + 1 total = param_count + 2
    let buffer_count = param_count + 2;
    if buffer_count > (codegen_msl::MAX_METAL_BUFFER_INDEX + 1) {
        return Err(FusedMslError::BufferLimitExceeded {
            required: buffer_count,
            max: codegen_msl::MAX_METAL_BUFFER_INDEX + 1,
        });
    }

    let kernel_name = format!("{}_kernel", kernel.name);
    let t = codegen_msl::msl_type(meta.dtype);
    let acc_t = codegen_msl::msl_accumulator_type(meta.dtype);
    let use_acc = acc_t != t;

    // Determine which inputs need broadcast indexing.
    let needs_broadcast: Vec<bool> = meta
        .input_shapes
        .iter()
        .map(|s| s != &meta.output_shape)
        .collect();

    // Emit the scalar computation body.
    let scalar_body = emit_scalar_body(kernel, contract, acc_t, use_acc)?;

    // Emit the kernel function.
    let mut msl = String::with_capacity(2048);
    msl.push_str(MSL_PRELUDE);

    // Kernel signature.
    msl.push_str(&format!("[[kernel]] void {kernel_name}(\n"));
    for (i, param) in kernel.params.iter().enumerate() {
        msl.push_str(&format!(
            "    device const {t}* {name} [[buffer({i})]],\n",
            name = param.name,
        ));
    }
    let out_idx = param_count;
    let total_idx = param_count + 1;
    msl.push_str(&format!("    device {t}* out [[buffer({out_idx})]],\n"));
    msl.push_str(&format!(
        "    constant uint& total [[buffer({total_idx})]],\n"
    ));
    msl.push_str("    uint tid [[thread_position_in_grid]]\n");
    msl.push_str(") {\n");
    msl.push_str("    if (tid >= total) return;\n");

    // Emit broadcast index computations for inputs that need them.
    for (i, param) in kernel.params.iter().enumerate() {
        if needs_broadcast[i] {
            let idx_var = format!("{}_idx", param.name);
            let index_body = build_broadcast_index(
                &meta.input_shapes[i],
                &meta.output_shape,
                meta.alignment,
                &idx_var,
            )?;
            msl.push_str(&index_body);
        }
    }

    // Emit accumulator promotions for half-precision inputs.
    if use_acc {
        for (i, param) in kernel.params.iter().enumerate() {
            let read_expr = if needs_broadcast[i] {
                format!("{name}[{name}_idx]", name = param.name)
            } else {
                format!("{name}[tid]", name = param.name)
            };
            msl.push_str(&format!(
                "    {acc_t} {name}_f = {acc_t}({read_expr});\n",
                name = param.name,
            ));
        }
    } else {
        // Read inputs into local variables.
        for (i, param) in kernel.params.iter().enumerate() {
            let read_expr = if needs_broadcast[i] {
                format!("{name}[{name}_idx]", name = param.name)
            } else {
                format!("{name}[tid]", name = param.name)
            };
            msl.push_str(&format!(
                "    {t} {name}_v = {read_expr};\n",
                name = param.name,
            ));
        }
    }

    // Emit the scalar computation.
    msl.push_str(&scalar_body);

    // Write output.
    let output_ref = node_ref_fused(kernel.output, kernel, use_acc)?;
    if use_acc {
        msl.push_str(&format!("    out[tid] = {t}({output_ref});\n"));
    } else {
        msl.push_str(&format!("    out[tid] = {output_ref};\n"));
    }
    msl.push_str("}\n");

    Ok(FusedMslResult::new(
        msl,
        kernel_name,
        buffer_count,
        FUSED_THREADGROUP_SIZE,
    ))
}

/// Emit the scalar computation body for the fused kernel.
///
/// Generates MSL statements for each non-Param IR node. Parameters are
/// read via `{name}_v` (float mode) or `{name}_f` (accumulator mode).
fn emit_scalar_body(
    kernel: &KernelDef,
    contract: PrecisionContract,
    int_type: &str,
    use_acc: bool,
) -> Result<String, FusedMslError> {
    let mut body = String::new();
    for node in &kernel.nodes {
        if let Some(line) = emit_node_fused(node, kernel, contract, int_type, use_acc)? {
            body.push_str(&line);
            body.push('\n');
        }
    }
    Ok(body)
}

/// Emit a single IR node as MSL, returning `None` for Param nodes (already read).
fn emit_node_fused(
    node: &IRNode,
    kernel: &KernelDef,
    contract: PrecisionContract,
    int_type: &str,
    use_acc: bool,
) -> Result<Option<String>, FusedMslError> {
    let tid = node.id.index();
    match &node.kind {
        IRNodeKind::Param(_) => Ok(None),
        IRNodeKind::Literal(v) => {
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
            let l = node_ref_fused(*lhs, kernel, use_acc)?;
            let r = node_ref_fused(*rhs, kernel, use_acc)?;
            let op_str = match op {
                BinOpKind::Add => "+",
                BinOpKind::Sub => "-",
                BinOpKind::Mul => "*",
                BinOpKind::Div => "/",
            };
            Ok(Some(format!("    {int_type} t{tid} = {l} {op_str} {r};")))
        }
        IRNodeKind::Compare { op, lhs, rhs } => {
            let l = node_ref_fused(*lhs, kernel, use_acc)?;
            let r = node_ref_fused(*rhs, kernel, use_acc)?;
            let cmp = compare_op(*op);
            Ok(Some(format!("    bool t{tid} = {l} {cmp} {r};")))
        }
        IRNodeKind::UnaryFn { op, input } => {
            let arg = node_ref_fused(*input, kernel, use_acc)?;
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
            let b = node_ref_fused(*base, kernel, use_acc)?;
            Ok(Some(powi_stmts(&b, *exp, int_type, tid)))
        }
        IRNodeKind::Clamp { input, min, max } => {
            let x = node_ref_fused(*input, kernel, use_acc)?;
            let lo = node_ref_fused(*min, kernel, use_acc)?;
            let hi = node_ref_fused(*max, kernel, use_acc)?;
            Ok(Some(format!(
                "    {int_type} t{tid} = clamp({x}, {lo}, {hi});"
            )))
        }
        IRNodeKind::MinMax { op, lhs, rhs } => {
            let a = node_ref_fused(*lhs, kernel, use_acc)?;
            let b = node_ref_fused(*rhs, kernel, use_acc)?;
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
            let cond_ref = node_ref_fused(*cond, kernel, use_acc)?;
            let then_ref = node_ref_fused(*then_val, kernel, use_acc)?;
            let else_ref = node_ref_fused(*else_val, kernel, use_acc)?;
            Ok(Some(format!(
                "    {int_type} t{tid} = ({cond_ref} ? {then_ref} : {else_ref});"
            )))
        }
        IRNodeKind::SumReduce { inputs } => {
            if let Some((first, rest)) = inputs.split_first() {
                let mut expr = node_ref_fused(*first, kernel, use_acc)?;
                for nid in rest {
                    expr.push_str(" + ");
                    expr.push_str(&node_ref_fused(*nid, kernel, use_acc)?);
                }
                Ok(Some(format!("    {int_type} t{tid} = {expr};")))
            } else {
                Ok(Some(format!("    {int_type} t{tid} = {int_type}(0);")))
            }
        }
        IRNodeKind::BinaryFn { op, lhs, rhs } => {
            let a = node_ref_fused(*lhs, kernel, use_acc)?;
            let b = node_ref_fused(*rhs, kernel, use_acc)?;
            let fn_name = match op {
                BinaryFnKind::Atan2 => "atan2",
            };
            Ok(Some(format!(
                "    {int_type} t{tid} = {fn_name}({a}, {b});"
            )))
        }
    }
}

/// Reference a node by its ID, returning the variable name used in the
/// fused kernel body.
///
/// For Param nodes: returns `{name}_v` (float mode) or `{name}_f` (acc mode).
/// For computed nodes: returns `t{index}`.
fn node_ref_fused(id: NodeId, kernel: &KernelDef, use_acc: bool) -> Result<String, FusedMslError> {
    let node = kernel
        .nodes
        .get(id.index())
        .ok_or_else(|| FusedMslError::InvalidNodeRef(id.index()))?;
    match &node.kind {
        IRNodeKind::Param(idx) => {
            let param = kernel
                .params
                .get(*idx)
                .ok_or(FusedMslError::InvalidParamRef(*idx))?;
            if use_acc {
                Ok(format!("{}_f", param.name))
            } else {
                Ok(format!("{}_v", param.name))
            }
        }
        _ => Ok(format!("t{}", id.index())),
    }
}

/// Build MSL statements that compute a broadcast index variable from `tid`.
///
/// Similar to `build_broadcast_index_body` in `codegen_msl_structural_spatial`
/// but writes to a named variable instead of the fixed `in_idx`.
fn build_broadcast_index(
    input_shape: &[usize],
    output_shape: &[usize],
    alignment: BroadcastAlignment,
    idx_var: &str,
) -> Result<String, FusedMslError> {
    let rank = output_shape.len();
    let input_rank = input_shape.len();

    let mut body = String::new();
    body.push_str(&format!("    uint {idx_var} = 0;\n"));
    body.push_str(&format!("    {{ uint remainder_{idx_var} = tid;\n"));

    let out_strides =
        row_major_strides(output_shape).ok_or_else(|| FusedMslError::ShapeOverflow {
            context: "output strides".into(),
        })?;

    let offset = match alignment {
        BroadcastAlignment::Left => 0,
        BroadcastAlignment::Right => rank.saturating_sub(input_rank),
    };

    let in_strides =
        row_major_strides(input_shape).ok_or_else(|| FusedMslError::ShapeOverflow {
            context: "input strides".into(),
        })?;

    for (i, &stride) in out_strides.iter().enumerate() {
        let s = safe_msl_uint(stride).map_err(|e| FusedMslError::MslCodegen(e.to_string()))?;
        body.push_str(&format!(
            "    uint bc_{idx_var}_{i} = remainder_{idx_var} / {s};\n\
             \x20   remainder_{idx_var} = remainder_{idx_var} % {s};\n",
        ));
        let input_idx = if i >= offset { i - offset } else { continue };
        if input_idx < input_rank && input_shape[input_idx] > 1 {
            let in_s = safe_msl_uint(in_strides[input_idx])
                .map_err(|e| FusedMslError::MslCodegen(e.to_string()))?;
            body.push_str(&format!("    {idx_var} += bc_{idx_var}_{i} * {in_s};\n"));
        }
    }
    body.push_str("    }\n");
    Ok(body)
}

#[path = "msl_auto_fuse_error.rs"]
mod error;
pub use error::FusedMslError;

#[cfg(test)]
#[path = "msl_auto_fuse_tests.rs"]
mod tests;
