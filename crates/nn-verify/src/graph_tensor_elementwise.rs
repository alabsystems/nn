// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Elementwise tensor node translation: inline scalar kernel into tensor graph.
//!
//! Extracted from `graph_tensor.rs` to stay under the 500-line file limit (#479).

use ny_propagate::layers::{MulConstantLayer, PowConstantLayer, SiLULayer, SnakeLayer};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::ir::{BinOpKind, IRNodeKind, KernelDef};
use nn_dsl::tensor_ir::TensorNodeId;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, translate_node, NodeValue, ParamBinding, TranslationContext};
use crate::util::get_value;

use super::TensorNodeValue;

/// Translate an Elementwise tensor node by inlining the scalar kernel's
/// IR translation with a tensor-node-specific prefix.
///
/// Each scalar param maps to its tensor input's NY node. The
/// scalar kernel's IR nodes are translated using the existing `translate_node`
/// machinery with a `t{id}_` prefix to avoid name collisions.
pub(super) fn translate_elementwise_inline(
    ctx: &super::TensorTranslationContext<'_>,
    tensor_node_id: TensorNodeId,
    scalar_kernel: &KernelDef,
    tensor_inputs: &[TensorNodeId],
    tensor_node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // Fast-path: collapse a decomposed GroupNorm(g=1) `centered * rsqrt` apex
    // into NY's native InstanceNorm1d, which enforces the sound `|z| <= sqrt(n-1)`
    // clamp. The decomposed primitive product otherwise bounds the normalized
    // output by ~`max|centered|/sqrt(eps)`, which compounds exponentially through
    // deep DConv stacks. See graph_tensor_group_norm_fusion.rs for the soundness
    // argument and the exact subgraph matched.
    if let Some(result) = super::group_norm_fusion::try_decomposed_group_norm(
        ctx,
        tensor_node_id,
        scalar_kernel,
        tensor_inputs,
        tensor_node_values,
        graph,
    )? {
        return Ok(result);
    }

    // Fast-path: use native NY layers for known kernels.
    // Native layers produce tighter bounds than decomposition because they
    // exploit mathematical properties (e.g., Snake's monotonicity, SiLU's
    // known derivative bounds). See graph_native.rs for the scalar-path
    // equivalent and graph_translate_native.rs tests for bound comparisons.
    if let Some(result) = try_native_elementwise(
        tensor_node_id,
        scalar_kernel,
        tensor_inputs,
        tensor_node_values,
        graph,
    )? {
        return Ok(result);
    }

    let prefix = format!("t{}_", tensor_node_id.index());

    // Build scalar ParamBindings from tensor input values
    let bindings: Vec<ParamBinding> = tensor_inputs
        .iter()
        .map(|tid| {
            Ok(
                match get_value(tensor_node_values, tid.index(), "Elementwise input binding")? {
                    TensorNodeValue::Variable(_) => ParamBinding::Variable,
                    TensorNodeValue::Constant(val) => ParamBinding::Constant(val.get()),
                    // Constant-fold a single-element weight tensor (e.g. Snake's
                    // `alpha` or GroupNorm's `eps`, bound as `ConstantTensor([1])`)
                    // into a scalar constant binding. This is EXACT: the lone
                    // element is a known finite value (finiteness already checked
                    // in graph_tensor_helpers.rs) broadcast across the operand, so
                    // the scalar kernel sees the same constant at every position —
                    // no interval or relaxation is introduced.
                    TensorNodeValue::WeightTensor(arr) if arr.len() == 1 => {
                        let scalar = *arr.iter().next().expect("len()==1 has one element");
                        ParamBinding::Constant(scalar)
                    }
                    // A genuinely multi-element weight tensor cannot be represented
                    // by a single broadcast scalar binding; reject (fail-closed).
                    TensorNodeValue::WeightTensor(_) => {
                        return Err(VerifyError::UnsupportedOp(
                            "weight tensor cannot be used as elementwise input".into(),
                        ));
                    }
                },
            )
        })
        .collect::<Result<_, VerifyError>>()?;

    let num_variables = bindings
        .iter()
        .filter(|b| matches!(b, ParamBinding::Variable))
        .count();

    // Map each variable param to its tensor input's NY node name.
    // This replaces the SliceLayer approach used by kernel_to_graph_multi.
    let param_node_names: Vec<Option<String>> = tensor_inputs
        .iter()
        .map(|tid| {
            Ok(
                match get_value(tensor_node_values, tid.index(), "Elementwise input name")? {
                    TensorNodeValue::Variable(name) => Some(name.clone()),
                    TensorNodeValue::Constant(_) | TensorNodeValue::WeightTensor(_) => None,
                },
            )
        })
        .collect::<Result<_, VerifyError>>()?;

    // Translate scalar kernel nodes with prefixed names
    let ctx = TranslationContext {
        prefix: &prefix,
        bindings: &bindings,
        num_variables,
        param_node_names: &param_node_names,
        all_nodes: &scalar_kernel.nodes,
    };
    let mut scalar_values: Vec<NodeValue> = Vec::with_capacity(scalar_kernel.nodes.len());
    for node in &scalar_kernel.nodes {
        let value = translate_node(&ctx, node.id.index(), &scalar_values, graph)?;
        scalar_values.push(value);
    }

    // Convert output NodeValue → TensorNodeValue
    match get_value(
        &scalar_values,
        scalar_kernel.output.index(),
        "Elementwise scalar output",
    )? {
        NodeValue::Constant(val) => Ok(TensorNodeValue::Constant(*val)),
        NodeValue::Variable(name) => Ok(TensorNodeValue::Variable(name.clone())),
    }
}

/// Try to emit a native NY layer for a known elementwise kernel.
///
/// Returns `Ok(Some(value))` if the kernel was translated via a native layer,
/// `Ok(None)` if decomposition should be used instead.
///
/// This mirrors the `try_native_layer` logic in `graph_native.rs` but operates
/// on tensor-level nodes rather than standalone scalar graphs. The benefit is
/// tighter bounds: Snake's monotonicity gives exact IBP bounds `[f(l), f(u)]`,
/// while the decomposed Sin→Pow→Mul→Add path loses this because Sin alone is
/// non-monotone. SiLU's dedicated layer similarly exploits its known derivative
/// bounds for tighter CROWN relaxation.
fn try_native_elementwise(
    tensor_node_id: TensorNodeId,
    scalar_kernel: &KernelDef,
    tensor_inputs: &[TensorNodeId],
    tensor_node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<Option<TensorNodeValue>, VerifyError> {
    let node_name = format!("t{}", tensor_node_id.index());

    // Snake: f(x, alpha) = x + (1/alpha) * sin²(alpha * x)
    // Requires: 2 inputs (variable x, constant alpha), alpha > 0.
    if scalar_kernel.name == "snake" && tensor_inputs.len() == 2 {
        let x_val = get_value(tensor_node_values, tensor_inputs[0].index(), "Snake x")?;
        let alpha_val = get_value(tensor_node_values, tensor_inputs[1].index(), "Snake alpha")?;

        if let (TensorNodeValue::Variable(input_name), TensorNodeValue::Constant(alpha)) =
            (x_val, alpha_val)
        {
            if let Ok(snake) = SnakeLayer::new(alpha.get()) {
                add_unary_node(&node_name, Layer::Snake(snake), input_name, graph);
                return Ok(Some(TensorNodeValue::Variable(node_name)));
            }
            // alpha <= 0 or non-finite: fall through to decomposed path
        }
    }

    // SiLU-Mul: silu_mul(x, up) = silu(x) * up
    // Requires: 2 inputs (variable x, constant up).
    // Emits SiLULayer (x * sigmoid(x)) + optional MulConstant(up).
    if scalar_kernel.name == "silu_mul" && tensor_inputs.len() == 2 {
        let x_val = get_value(tensor_node_values, tensor_inputs[0].index(), "SiLU x")?;
        let up_val = get_value(tensor_node_values, tensor_inputs[1].index(), "SiLU up")?;

        if let (TensorNodeValue::Variable(input_name), TensorNodeValue::Constant(up)) =
            (x_val, up_val)
        {
            let up_f32 = up.get();
            if !up_f32.is_finite() {
                return Ok(None);
            }

            let silu_name = format!("t{}_silu", tensor_node_id.index());
            add_unary_node(&silu_name, Layer::SiLU(SiLULayer::new()), input_name, graph);

            if (up_f32 - 1.0).abs() > f32::EPSILON {
                // Multiply by up when up != 1.0
                add_unary_node(
                    &node_name,
                    Layer::MulConstant(MulConstantLayer::scalar(up_f32)),
                    &silu_name,
                    graph,
                );
                return Ok(Some(TensorNodeValue::Variable(node_name)));
            }
            // up ≈ 1.0: SiLU output is the result, no MulConstant needed.
            // Matches graph_native.rs:101 which skips MulConstant for up=1.
            return Ok(Some(TensorNodeValue::Variable(silu_name)));
        }
    }

    // Square: square(x) = x * x  (a self-multiply BinOp::Mul lhs==rhs).
    // Requires: 1 variable input.
    //
    // Why native vs. decomposition: the generic `x * x` path lowers to a NY
    // `MulBinary` of `x` with itself, whose interval IBP `[a,b]*[a,b]` returns a
    // NEGATIVE lower bound `ab < 0` whenever the input straddles zero (a<0<b).
    // For GroupNorm(g=1)/LayerNorm variance this makes `var = sum(centered²)`
    // appear to have a negative lower bound, so `sqrt(var+eps)` looks like it
    // touches a negative domain and trips the `SqrtNegativeDomain` heuristic
    // flag — a phantom: the true variance is provably >= 0.
    //
    // `PowConstant(2)`'s IBP is square-aware: for a U-shaped x², it returns a
    // lower bound of 0 when the interval spans zero and `min(l²,u²)` otherwise,
    // i.e. always >= 0. This is a STRICT TIGHTENING of the generic product (x²
    // genuinely is non-negative), so it never excludes a reachable value and
    // never loosens any bound; it merely removes the spurious negative domain.
    if scalar_kernel.name == "square" && tensor_inputs.len() == 1 {
        // Confirm the kernel body is the canonical self-multiply x*x before
        // treating it as a square (guards against an unrelated kernel that
        // happens to share the name).
        let is_self_square = matches!(
            scalar_kernel
                .nodes
                .iter()
                .find(|n| n.id == scalar_kernel.output)
                .map(|n| &n.kind),
            Some(IRNodeKind::BinOp {
                op: BinOpKind::Mul,
                lhs,
                rhs,
            }) if lhs == rhs
        );

        let x_val = get_value(tensor_node_values, tensor_inputs[0].index(), "Square x")?;
        if is_self_square {
            if let TensorNodeValue::Variable(input_name) = x_val {
                add_unary_node(
                    &node_name,
                    Layer::PowConstant(PowConstantLayer::square()),
                    input_name,
                    graph,
                );
                return Ok(Some(TensorNodeValue::Variable(node_name)));
            }
            // Constant/weight x: fall through to decomposition (constant-folds).
        }
    }

    Ok(None)
}
