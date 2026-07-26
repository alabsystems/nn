// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Decomposed-GroupNorm(g=1) → native InstanceNorm1d fusion for verification.
//!
//! `TensorBlockBuilder::add_group_norm_g1` decomposes GroupNorm(groups=1) into
//! dispatchable primitives (`Reshape -> mean/sub/square/var/rsqrt/mul -> Reshape`)
//! because the kernel-codegen path (`build_dispatch_plan`) cannot handle the
//! native `InstanceNorm1d` op. For VERIFICATION, however, propagating IBP through
//! that primitive chain is catastrophically loose: the `centered * rsqrt` product
//! is bounded by the 4-corner interval product, which drops the joint constraint
//! that ties a large `centered` to a small `rsqrt` (large deviation ⇒ large
//! variance ⇒ large std ⇒ small reciprocal-std). The normalized output is then
//! bounded by ~`max|centered| / sqrt(eps)` instead of the exact ceiling
//! `|z_i| <= sqrt(n-1)`.
//!
//! GroupNorm(g=1) over `[C, T]` is — after the builder's own `reshape [C,T] ->
//! [1, C*T]` — *exactly* InstanceNorm1d over the last axis of a 1-channel
//! `[1, C*T]` tensor: both subtract the single mean over all `n = C*T` elements
//! and divide by the single std over the same `n` elements. NY's native
//! `InstanceNorm1dLayer` IBP/CROWN already enforce the sound `|z_i| <= sqrt(n-1)`
//! clamp (+ a sound f32 margin). So when we detect the decomposed subgraph at its
//! `mul` apex we emit the native layer instead, reusing that proven clamp.
//!
//! Soundness: the substitution computes the **same mathematical function** (mean-
//! subtract / std-divide over the `n`-element group). The native layer's IBP is a
//! sound over-approximation of that function (it is the verifier's own,
//! proptest-validated GroupNorm/InstanceNorm enclosure). The per-channel affine
//! `gamma * z + beta` that follows in the decomposed IR is left untouched (the
//! native layer here is non-affine), so the end-to-end function is unchanged. The
//! enclosure is tighter-or-equal everywhere (it is the intersection of the same
//! reachable set with the `sqrt(n-1)` envelope), never wider.

use ny_propagate::layers::InstanceNorm1dLayer;
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::ir::{BinOpKind, IRNodeKind, KernelDef, UnaryFnKind};
use nn_dsl::tensor_ir::{TensorNode, TensorNodeId, TensorOpKind};

use crate::graph::add_unary_node;
use crate::util::get_value;

use super::{TensorNodeValue, TensorTranslationContext};

/// Result of matching a decomposed GroupNorm(g=1) subgraph at its `mul` apex.
struct DecomposedGroupNormMatch {
    /// The `Reshape [C,T] -> [1, C*T]` node feeding the normalization (the ny
    /// node whose bounds become the InstanceNorm1d input).
    reshaped: TensorNodeId,
    /// Group size n = C*T (the normalized axis length).
    group_size: usize,
    /// The `eps` input node (a constant scalar `[1]` tensor).
    eps: TensorNodeId,
}

/// Return the `IRNodeKind` of the scalar-kernel output node, if the kernel is a
/// single-op elementwise (the shape produced by `binop_kernel`/`unary_kernel`).
fn scalar_kernel_output_kind(kernel: &KernelDef) -> Option<&IRNodeKind> {
    kernel
        .nodes
        .iter()
        .find(|n| n.id == kernel.output)
        .map(|n| &n.kind)
}

/// Is this Elementwise node a single scalar op of the given binary kind?
fn is_binop_kernel(kind: &TensorOpKind, want: BinOpKind) -> Option<&[TensorNodeId]> {
    if let TensorOpKind::Elementwise { kernel, inputs } = kind {
        if let Some(IRNodeKind::BinOp { op, .. }) = scalar_kernel_output_kind(kernel) {
            if *op == want {
                return Some(inputs);
            }
        }
    }
    None
}

/// Is this Elementwise node a single scalar unary op of the given kind?
fn is_unary_kernel(kind: &TensorOpKind, want: UnaryFnKind) -> Option<&[TensorNodeId]> {
    if let TensorOpKind::Elementwise { kernel, inputs } = kind {
        if let Some(IRNodeKind::UnaryFn { op, .. }) = scalar_kernel_output_kind(kernel) {
            if *op == want {
                return Some(inputs);
            }
        }
    }
    None
}

/// Is this Elementwise node the canonical `square(x) = x * x` self-multiply?
fn is_square_kernel(kind: &TensorOpKind) -> Option<&[TensorNodeId]> {
    if let TensorOpKind::Elementwise { kernel, inputs } = kind {
        if let Some(IRNodeKind::BinOp {
            op: BinOpKind::Mul,
            lhs,
            rhs,
        }) = scalar_kernel_output_kind(kernel)
        {
            if lhs == rhs {
                return Some(inputs);
            }
        }
    }
    None
}

fn node(all_nodes: &[TensorNode], id: TensorNodeId) -> Option<&TensorNode> {
    all_nodes.get(id.index())
}

/// Match the `Broadcast(Reduce(Mean, axis=1))` mean/var pattern, returning the
/// reduce's *input* node id (the tensor being averaged).
fn match_broadcast_mean(all_nodes: &[TensorNode], bc_id: TensorNodeId) -> Option<TensorNodeId> {
    let bc = node(all_nodes, bc_id)?;
    let TensorOpKind::Broadcast { input: red_id, .. } = &bc.kind else {
        return None;
    };
    let red = node(all_nodes, *red_id)?;
    match &red.kind {
        TensorOpKind::Reduce {
            op: nn_dsl::tensor_ir::ReduceOp::Mean,
            input,
            axis: 1,
            keepdim: false,
        } => Some(*input),
        _ => None,
    }
}

/// Try to match the decomposed GroupNorm(g=1) subgraph rooted at a `mul`
/// Elementwise node (the `centered * rsqrt` apex produced by
/// `add_group_norm_g1`). Returns the data needed to emit a native
/// InstanceNorm1d, or `None` if the structure does not match exactly.
///
/// Expected DAG (ids relative to the `mul` apex):
/// ```text
/// reshaped = Reshape(input, [1, n])
/// mean     = Reduce(Mean, reshaped, axis=1)
/// mean_bc  = Broadcast(mean, [1, n])
/// centered = sub(reshaped, mean_bc)
/// sq       = square(centered)
/// var      = Reduce(Mean, sq, axis=1)
/// var_bc   = Broadcast(var, [1, n])
/// eps_bc   = Broadcast(eps, [1, n])
/// var_eps  = add(var_bc, eps_bc)
/// rsqrt    = rsqrt(var_eps)
/// mul      = mul(centered, rsqrt)          ← apex
/// ```
fn match_decomposed_group_norm(
    all_nodes: &[TensorNode],
    mul_inputs: &[TensorNodeId],
) -> Option<DecomposedGroupNormMatch> {
    if mul_inputs.len() != 2 {
        return None;
    }
    // Operand order from the builder is (centered, rsqrt). Accept either order
    // defensively: one operand must be a `sub`, the other a `rsqrt`.
    let (centered_id, rsqrt_id) = {
        let a = mul_inputs[0];
        let b = mul_inputs[1];
        let a_is_sub = is_binop_kernel(&node(all_nodes, a)?.kind, BinOpKind::Sub).is_some();
        let b_is_rsqrt = is_unary_kernel(&node(all_nodes, b)?.kind, UnaryFnKind::Rsqrt).is_some();
        if a_is_sub && b_is_rsqrt {
            (a, b)
        } else {
            let a_is_rsqrt =
                is_unary_kernel(&node(all_nodes, a)?.kind, UnaryFnKind::Rsqrt).is_some();
            let b_is_sub = is_binop_kernel(&node(all_nodes, b)?.kind, BinOpKind::Sub).is_some();
            if a_is_rsqrt && b_is_sub {
                (b, a)
            } else {
                return None;
            }
        }
    };

    // centered = sub(reshaped, mean_bc)
    let sub_inputs = is_binop_kernel(&node(all_nodes, centered_id)?.kind, BinOpKind::Sub)?;
    if sub_inputs.len() != 2 {
        return None;
    }
    let reshaped_id = sub_inputs[0];
    let mean_bc_id = sub_inputs[1];

    // reshaped = Reshape(input, [1, n]) with a 2-D [1, n] shape.
    let reshaped = node(all_nodes, reshaped_id)?;
    let TensorOpKind::Reshape { target_shape, .. } = &reshaped.kind else {
        return None;
    };
    if target_shape.len() != 2 || target_shape[0] != 1 {
        return None;
    }
    let group_size = target_shape[1];
    if group_size < 2 {
        return None;
    }

    // mean_bc = Broadcast(Reduce(Mean, reshaped, axis=1)); reduce input == reshaped.
    if match_broadcast_mean(all_nodes, mean_bc_id)? != reshaped_id {
        return None;
    }

    // rsqrt = rsqrt(var_eps)
    let rsqrt_inputs = is_unary_kernel(&node(all_nodes, rsqrt_id)?.kind, UnaryFnKind::Rsqrt)?;
    if rsqrt_inputs.len() != 1 {
        return None;
    }
    // var_eps = add(var_bc, eps_bc)
    let add_inputs = is_binop_kernel(&node(all_nodes, rsqrt_inputs[0])?.kind, BinOpKind::Add)?;
    if add_inputs.len() != 2 {
        return None;
    }
    let var_bc_id = add_inputs[0];
    let eps_bc_id = add_inputs[1];

    // var_bc = Broadcast(Reduce(Mean, sq, axis=1)); sq = square(centered).
    let sq_id = match_broadcast_mean(all_nodes, var_bc_id)?;
    let sq_inputs = is_square_kernel(&node(all_nodes, sq_id)?.kind)?;
    if sq_inputs.len() != 1 || sq_inputs[0] != centered_id {
        return None;
    }

    // eps_bc = Broadcast(eps_input); the eps input is a scalar [1] tensor.
    let eps_bc = node(all_nodes, eps_bc_id)?;
    let TensorOpKind::Broadcast { input: eps_id, .. } = &eps_bc.kind else {
        return None;
    };
    let eps_node = node(all_nodes, *eps_id)?;
    if eps_node.shape != [1] {
        return None;
    }

    Some(DecomposedGroupNormMatch {
        reshaped: reshaped_id,
        group_size,
        eps: *eps_id,
    })
}

/// If the `mul` Elementwise node at `inputs` is the apex of a decomposed
/// GroupNorm(g=1), emit a native `InstanceNorm1dLayer` (1 channel, group_size
/// time-steps) wired to the reshaped input's NY node, and return its value.
/// Returns `Ok(None)` to fall through to the decomposed primitive translation.
pub(super) fn try_decomposed_group_norm(
    ctx: &TensorTranslationContext<'_>,
    tensor_node_id: TensorNodeId,
    scalar_kernel: &KernelDef,
    inputs: &[TensorNodeId],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<Option<TensorNodeValue>, crate::error::VerifyError> {
    // Apex must itself be a `mul`.
    let mul_kind = TensorOpKind::Elementwise {
        kernel: scalar_kernel.clone(),
        inputs: inputs.to_vec(),
    };
    if is_binop_kernel(&mul_kind, BinOpKind::Mul).is_none() {
        return Ok(None);
    }

    let Some(m) = match_decomposed_group_norm(ctx.all_nodes, inputs) else {
        return Ok(None);
    };

    // The reshaped node's NY value must be a propagatable Variable; constants
    // would already have been folded upstream.
    let TensorNodeValue::Variable(input_name) = get_value(
        node_values,
        m.reshaped.index(),
        "GroupNorm-fusion reshaped input",
    )?
    else {
        return Ok(None);
    };

    // Resolve eps to a finite scalar. The eps input is bound as ConstantScalar /
    // ConstantTensor([1]); a Variable or non-finite eps is not fusable here.
    let eps_val = match get_value(node_values, m.eps.index(), "GroupNorm-fusion eps")? {
        TensorNodeValue::Constant(v) => v.get(),
        TensorNodeValue::WeightTensor(arr) if arr.len() == 1 => {
            *arr.iter().next().expect("len()==1 has one element")
        }
        TensorNodeValue::Variable(_) | TensorNodeValue::WeightTensor(_) => return Ok(None),
    };
    if !eps_val.is_finite() || eps_val < 0.0 {
        return Ok(None);
    }

    // GroupNorm(g=1) over [C,T] reshaped to [1, C*T] == InstanceNorm1d over the
    // last axis of a 1-channel tensor with time_len = group_size. Non-affine: the
    // builder applies per-channel gamma/beta in later IR ops, which we leave intact.
    let norm_mode = ctx.norm_mode;
    let layer = InstanceNorm1dLayer::new_default(1, eps_val)?
        .with_forward_mode(norm_mode.forward_mode())
        .with_crown_mode(norm_mode.crown_mode());
    let _ = m.group_size; // already encoded by the reshaped [1, n] input shape.

    let node_name = format!("t{}", tensor_node_id.index());
    add_unary_node(&node_name, Layer::InstanceNorm1d(layer), input_name, graph);
    Ok(Some(TensorNodeValue::Variable(node_name)))
}
