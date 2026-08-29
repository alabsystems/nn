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
//!
//! `tests::fusion_is_orders_of_magnitude_tighter_than_the_primitive_chain`
//! measures that on the real translation path (`C=4, T=8, n=32`, Demucs'
//! `GROUP_NORM_EPS = 1e-5`): **1.1x** tighter at typical variance and **56.8x**
//! tighter on a near-constant group, where the fused form sits at its
//! `2*sqrt(n-1) = 11.14` clamp while the primitive chain reaches `632` —
//! i.e. `2 * max|centered| / sqrt(eps)`, exactly the failure mode described above.
//!
//! ## Why `InstanceNorm1d` and not NY's `Layer::GroupNorm` — measured, not assumed
//!
//! NY does have a native `GroupNormLayer` (conservative IBP, forward-mode IBP,
//! scalar and batched CROWN, and a decomposed grouped-centered fallback), and nn
//! never emits it: this fusion emits `InstanceNorm1d`, and `nn-dsl`'s
//! `compile_group_norm` lowers the `g > 1` case to a flat
//! `InstanceNorm1d` over `[B*G, (C/G)*spatial]`. That is a real observation — but
//! it costs NOTHING in bound tightness, which is the only reason it would matter.
//!
//! `tests::native_group_norm_g1_and_instance_norm_agree_bound_for_bound` pins the
//! measurement: `GroupNormLayer::new_default(1, 1, eps)` and
//! `InstanceNorm1dLayer::new_default(1, eps)` on the same `[1, n]` box return
//! **bit-identical** lower and upper bounds at `n = 32, 128, 192`. They are the
//! same algorithm — the same `sqrt(n-1) + delta` z-clamp with the same
//! `(n+2)*ulp/std_lower` f32 margin, the same 4-corner `centered/std` ratio, the
//! same directed rounding — because with one channel a group IS the channel.
//!
//! So switching this call site to `Layer::GroupNorm` would be a rename, not a
//! tightening, and it would be a rename in the *wrong* direction: by NY's own
//! reduction table (`GroupNorm` with `num_groups = C` IS InstanceNorm), the layer
//! we attach to the flattened `[1, n]` tensor is most accurately an InstanceNorm.
//! The `g > 1` case is likewise already exact — normalizing each `[B*G]` row over
//! its `(C/G)*spatial` elements is group normalization by definition. Emit
//! `Layer::GroupNorm` here only if that equality measurement ever breaks.

use nn_dsl::ir::{BinOpKind, IRNodeKind, KernelDef, UnaryFnKind};
use nn_dsl::tensor_ir::{TensorNode, TensorNodeId, TensorOpKind};
use ny_propagate::layers::InstanceNorm1dLayer;
use ny_propagate::{GraphNetwork, Layer};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
    use ndarray::{ArrayD, IxDyn};
    use nn_dsl::ir::{IRNode, NodeId, Param, ScalarType};
    use nn_dsl::tensor_block_builder::TensorBlockBuilder;
    use nn_dsl::tensor_ir::{ReduceOp, TensorKernelDef};
    use ny_propagate::layers::GroupNormLayer;
    use ny_propagate::BoundPropagation;

    fn total_width(t: &BoundedTensor) -> f64 {
        let (lo, hi) = t.lower_upper();
        lo.iter()
            .zip(hi.iter())
            .map(|(&l, &u)| f64::from(u) - f64::from(l))
            .sum()
    }

    fn max_width(t: &BoundedTensor) -> f64 {
        let (lo, hi) = t.lower_upper();
        lo.iter()
            .zip(hi.iter())
            .map(|(&l, &u)| f64::from(u) - f64::from(l))
            .fold(0.0f64, f64::max)
    }

    /// A DConv-shaped input box: `[C, T]` centred on a deterministic mixed-sign
    /// pattern scaled by `spread`, with per-element radius `radius`. A small
    /// `spread` gives a near-constant group (variance -> 0), the regime where
    /// `rsqrt(var + eps)` reaches ~`1/sqrt(eps)`.
    fn scaled_input_box(shape: &[usize], spread: f32, radius: f32) -> BoundedTensor {
        let n: usize = shape.iter().product();
        let centre: Vec<f32> = (0..n)
            .map(|i| (((i * 41 % 197) as f32) / 98.0 - 1.0) * spread)
            .collect();
        let lower: Vec<f32> = centre.iter().map(|&c| c - radius).collect();
        let upper: Vec<f32> = centre.iter().map(|&c| c + radius).collect();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(shape), upper).expect("upper"),
        )
        .expect("valid box")
    }

    fn input_box(shape: &[usize], radius: f32) -> BoundedTensor {
        scaled_input_box(shape, 1.0, radius)
    }

    // -- Premise check: is NY's native GroupNormLayer actually reachable, and
    //    does it beat the InstanceNorm1d this module emits instead? --

    /// MEASUREMENT (task 1). The audit's premise is HALF right: nn genuinely
    /// never emits `Layer::GroupNorm` — this fusion emits
    /// `Layer::InstanceNorm1d` and the `g > 1` path in `nn-dsl`'s
    /// `compile_group_norm` emits a flat `InstanceNorm1d` too. But the claimed
    /// COST is zero. GroupNorm(g=1) over the flattened `[1, n]` normalizes
    /// exactly the `n` elements InstanceNorm1d(1 channel) normalizes, and NY's
    /// two IBP implementations are the same algorithm (identical `max_norm`
    /// `sqrt(n-1) + delta` clamp, identical 4-corner ratio, identical directed
    /// rounding). This measures it rather than asserting it: bound-for-bound
    /// equality on a shared DConv-shaped fixture.
    ///
    /// Consequence: switching to `Layer::GroupNorm` here would buy no tightness,
    /// so the fusion stays on `InstanceNorm1d`. If this ever stops being an
    /// equality, the layer choice is worth revisiting.
    #[test]
    fn native_group_norm_g1_and_instance_norm_agree_bound_for_bound() {
        for (channels, time_len) in [(4usize, 8usize), (8, 16), (48, 4)] {
            let n = channels * time_len;
            let eps = 1e-5f32;
            let flat = input_box(&[1, n], 0.05);

            let gn = GroupNormLayer::new_default(1, 1, eps).expect("GroupNorm(1 channel, 1 group)");
            let inn = InstanceNorm1dLayer::new_default(1, eps).expect("InstanceNorm1d(1 channel)");

            let gn_out = gn.propagate_ibp(&flat).expect("GroupNorm IBP");
            let in_out = inn.propagate_ibp(&flat).expect("InstanceNorm1d IBP");

            let (glo, ghi) = gn_out.lower_upper();
            let (ilo, ihi) = in_out.lower_upper();
            assert_eq!(glo.shape(), ilo.shape());
            for (i, (((&a, &b), &c), &d)) in glo
                .iter()
                .zip(ghi.iter())
                .zip(ilo.iter())
                .zip(ihi.iter())
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    c.to_bits(),
                    "C={channels} T={time_len} index {i}: GroupNorm lower {a} != InstanceNorm {c}"
                );
                assert_eq!(
                    b.to_bits(),
                    d.to_bits(),
                    "C={channels} T={time_len} index {i}: GroupNorm upper {b} != InstanceNorm {d}"
                );
            }
            println!(
                "C={channels} T={time_len} n={n}: GroupNorm(g=1) width={:.6} == InstanceNorm1d width={:.6}",
                total_width(&gn_out),
                total_width(&in_out)
            );
        }
    }

    // -- The fusion's own before/after: native norm vs the primitive chain --

    fn binop_kernel(name: &str, op: BinOpKind) -> KernelDef {
        KernelDef::new(
            name,
            vec![
                Param::new("a", ScalarType::F32),
                Param::new("b", ScalarType::F32),
            ],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
                IRNode::new(
                    NodeId::new(2),
                    IRNodeKind::BinOp {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
            ],
            NodeId::new(2),
        )
    }

    fn unary_kernel(name: &str, op: UnaryFnKind) -> KernelDef {
        KernelDef::new(
            name,
            vec![Param::new("x", ScalarType::F32)],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(
                    NodeId::new(1),
                    IRNodeKind::UnaryFn {
                        op,
                        input: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        )
    }

    fn square_kernel() -> KernelDef {
        KernelDef::new(
            "square",
            vec![Param::new("x", ScalarType::F32)],
            ScalarType::F32,
            vec![
                IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
                IRNode::new(
                    NodeId::new(1),
                    IRNodeKind::BinOp {
                        op: BinOpKind::Mul,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        )
    }

    /// Reproduces `TensorBlockBuilder::add_group_norm_g1`'s non-affine body.
    ///
    /// `fusable == false` flips ONE cosmetic detail — the mean reduce keeps its
    /// axis (`keepdim = true`) and is broadcast back from `[1, 1]` instead of
    /// `[1]` — which computes the IDENTICAL function but makes
    /// `match_broadcast_mean` (which pattern-matches `keepdim: false`) refuse,
    /// so the translation falls through to the primitive chain. Every other
    /// node, including the `square` kernel and the `mul` apex, is untouched, so
    /// the two graphs differ only by the fusion.
    fn group_norm_g1_kernel(channels: usize, time_len: usize, fusable: bool) -> TensorKernelDef {
        let n = channels * time_len;
        let flat = vec![1usize, n];
        let mut b = TensorBlockBuilder::new("gn_g1");
        let x = b.add_input("x", &[channels, time_len]);
        let eps = b.add_input("eps", &[1]);

        let reshaped = b.add_reshape(x, &flat);
        let mean_bc = if fusable {
            let mean = b.add_reduce(reshaped, ReduceOp::Mean, 1, false, &[1]);
            b.add_broadcast_left(mean, &flat)
        } else {
            let mean = b.add_reduce(reshaped, ReduceOp::Mean, 1, true, &[1, 1]);
            b.add_broadcast(mean, &flat)
        };
        let centered = b.add_elementwise(
            binop_kernel("sub", BinOpKind::Sub),
            &[reshaped, mean_bc],
            &flat,
        );
        let sq = b.add_elementwise(square_kernel(), &[centered], &flat);
        let var = b.add_reduce(sq, ReduceOp::Mean, 1, false, &[1]);
        let var_bc = b.add_broadcast_left(var, &flat);
        let eps_bc = b.add_broadcast_left(eps, &flat);
        let var_eps = b.add_elementwise(
            binop_kernel("add", BinOpKind::Add),
            &[var_bc, eps_bc],
            &flat,
        );
        let rsqrt = b.add_elementwise(unary_kernel("rsqrt", UnaryFnKind::Rsqrt), &[var_eps], &flat);
        let normed = b.add_elementwise(
            binop_kernel("mul", BinOpKind::Mul),
            &[centered, rsqrt],
            &flat,
        );
        let out = b.add_reshape(normed, &[channels, time_len]);
        b.build(out).expect("valid GroupNorm(g=1) kernel")
    }

    /// BEFORE/AFTER measurement for the fusion itself. This is the number the
    /// module docs claim: propagating IBP through the decomposed
    /// `centered * rsqrt` product loses the joint constraint that ties a large
    /// deviation to a large std, so the normalized output is bounded by roughly
    /// `max|centered| / sqrt(eps)` instead of the exact `sqrt(n-1)` ceiling.
    /// Demucs/HTDemucs DConv is the beneficiary, and its `add_group_norm_g1`
    /// call site is exactly this shape.
    #[test]
    fn fusion_is_orders_of_magnitude_tighter_than_the_primitive_chain() {
        let (channels, time_len) = (4usize, 8usize);
        let n = channels * time_len;
        let bindings = [
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(1e-5),
        ];

        let fused =
            tensor_kernel_to_graph(&group_norm_g1_kernel(channels, time_len, true), &bindings)
                .expect("fused translation");
        let decomposed =
            tensor_kernel_to_graph(&group_norm_g1_kernel(channels, time_len, false), &bindings)
                .expect("decomposed translation");

        // The fused graph really does carry NY's native normalization layer,
        // and the unfused one really does not — otherwise the comparison below
        // would be measuring nothing.
        let has_native_norm = |g: &GraphNetwork| {
            g.node_names().iter().any(|nm| {
                matches!(
                    g.node(nm).map(ny_propagate::GraphNode::layer),
                    Some(Layer::InstanceNorm1d(_) | Layer::GroupNorm(_))
                )
            })
        };
        assert!(
            has_native_norm(&fused),
            "fused path must emit a native norm layer"
        );
        assert!(
            !has_native_norm(&decomposed),
            "the unfused fixture must fall through to the primitive chain"
        );

        // Two regimes. `spread` scales the per-element centre pattern, so a
        // small spread means a near-constant group: variance close to zero, and
        // therefore an `rsqrt(var + eps)` interval reaching ~1/sqrt(eps). That
        // is the regime the decomposed product cannot bound — it multiplies the
        // largest deviation by the largest reciprocal-std as if they were
        // independent — and it is the regime a deep DConv stack lands in.
        let z_ceiling = 2.0 * ((n - 1) as f64).sqrt();
        let mut low_variance_ratio = 0.0f64;
        for (spread, radius, label) in [
            (1.0f32, 0.05f32, "typical variance"),
            (0.02, 0.5, "near-constant group (low variance)"),
        ] {
            let input = scaled_input_box(&[channels, time_len], spread, radius);
            let fused_out = fused.propagate_ibp(&input).expect("fused IBP");
            let dec_out = decomposed.propagate_ibp(&input).expect("decomposed IBP");

            let fused_max = max_width(&fused_out);
            let dec_max = max_width(&dec_out);
            let ratio = dec_max / fused_max.max(f64::MIN_POSITIVE);
            println!(
                "GroupNorm(g=1) C={channels} T={time_len} n={n} [{label}]: \
                 fused max width={fused_max:.6} decomposed max width={dec_max:.6} \
                 (decomposed is {ratio:.1}x looser)"
            );

            // The native layer's sound `|z| <= sqrt(n-1)` clamp caps the
            // half-width at sqrt(n-1) in EVERY regime; the primitive chain has
            // no such ceiling.
            assert!(
                fused_max <= z_ceiling * 1.01,
                "[{label}] fused width {fused_max} must respect the sqrt(n-1) ceiling {z_ceiling}"
            );
            assert!(
                fused_max <= dec_max + 1e-6,
                "[{label}] the fusion must never be looser than the chain it replaces: \
                 fused={fused_max} decomposed={dec_max}"
            );

            // SOUNDNESS: the fused bounds must enclose the exact GroupNorm(g=1)
            // evaluation at the box centre.
            let (blo, bhi) = input.lower_upper();
            let centre = (blo.to_owned() + bhi.to_owned()).mapv(|v| v / 2.0);
            let mean: f32 = centre.iter().sum::<f32>() / (n as f32);
            let var: f32 =
                centre.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (n as f32);
            let inv_std = 1.0f32 / (var + 1e-5).sqrt();
            let (flo, fhi) = fused_out.lower_upper();
            for (i, ((&v, &l), &u)) in centre.iter().zip(flo.iter()).zip(fhi.iter()).enumerate() {
                let z = (v - mean) * inv_std;
                assert!(
                    l <= z + 1e-4 && z <= u + 1e-4,
                    "[{label}] index {i}: fused bound [{l}, {u}] must enclose the exact \
                     normalized value {z}"
                );
            }

            if spread < 0.5 {
                low_variance_ratio = ratio;
            }
        }

        assert!(
            low_variance_ratio > 10.0,
            "in the low-variance regime the primitive chain should be >10x looser; got \
             {low_variance_ratio}x"
        );
    }
}
