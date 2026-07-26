// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expand native composite ops into decomposed primitives for MSL codegen.
//!
//! Native tensor ops (`InstanceNorm1d`, `RmsNorm`, `AdaIN1d`, `Attention`, `Lstm`)
//! map directly to NY layers for tight verification bounds. However,
//! MSL codegen requires these to be expressed as sequences of primitive ops
//! (Reduce/Broadcast/Elementwise/MatMul/Softmax/Linear/Narrow) that map to
//! individual Metal kernel dispatches.
//!
//! This module expands native ops in-place before dispatch planning, closing
//! the topology divergence between the verified graph and the executed graph
//! (see #667, #812, #2306).

use crate::ir::BinOpKind;
use crate::tensor_builders::{binop_kernel, broadcast_node, elementwise_node};
use crate::tensor_ir::{
    AttentionMask, BroadcastAlignment, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};
use std::collections::HashMap;

#[path = "codegen_msl_tensor_expand_norm.rs"]
mod norm;
use norm::{emit_affine_transform, emit_instance_norm_core, emit_rms_norm_core};

#[path = "codegen_msl_tensor_expand_lstm.rs"]
mod lstm;
use lstm::emit_lstm_cell;

/// Mutable expansion state passed through helper functions.
pub(super) struct ExpandState {
    pub(super) nodes: Vec<TensorNode>,
    pub(super) id_map: HashMap<usize, usize>,
    next_id: usize,
}

impl ExpandState {
    pub(super) fn alloc(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub(super) fn push(&mut self, node: TensorNode) {
        self.nodes.push(node);
    }

    /// Look up the shape of a node by its new (remapped) ID.
    fn node_shape(&self, new_id: usize) -> &[usize] {
        &self.nodes[new_id].shape
    }
}

/// Expand native composite ops into decomposed primitives for dispatch.
///
/// Returns a new `TensorKernelDef` where `InstanceNorm1d`, `RmsNorm`,
/// `AdaIN1d`, `LayerNorm`, `Attention(Standard)`, and `Lstm` nodes have been
/// replaced by their equivalent primitive sequences. The output is suitable
/// for `build_dispatch_plan`.
pub(crate) fn expand_norm_ops(def: &TensorKernelDef) -> TensorKernelDef {
    let mut st = ExpandState {
        nodes: Vec::new(),
        id_map: HashMap::new(),
        next_id: 0,
    };

    for node in &def.nodes {
        expand_single_node(&mut st, node);
    }

    let mapped_output = st.id_map[&def.output.index()];
    TensorKernelDef {
        name: def.name.clone(),
        nodes: st.nodes,
        output: TensorNodeId::new(mapped_output),
    }
}

/// Expand a single node, either decomposing a composite op or passing through.
fn expand_single_node(st: &mut ExpandState, node: &TensorNode) {
    let old_id = node.id.index();

    match &node.kind {
        TensorOpKind::InstanceNorm1d {
            input,
            eps,
            axis,
            gamma,
            beta,
        } => expand_instance_norm(st, old_id, &node.shape, input, eps, *axis, gamma, beta),

        TensorOpKind::RmsNorm {
            input,
            eps,
            axis,
            weight,
        } => expand_rms_norm(st, old_id, &node.shape, input, eps, *axis, weight),

        TensorOpKind::AdaIN1d {
            input,
            eps,
            axis,
            style_gamma,
            style_beta,
        } => expand_adain1d(
            st,
            old_id,
            &node.shape,
            input,
            eps,
            *axis,
            style_gamma,
            style_beta,
        ),

        TensorOpKind::LayerNorm {
            input,
            eps,
            axis,
            weight,
            bias,
        } => expand_layer_norm(st, old_id, &node.shape, input, eps, *axis, weight, bias),

        TensorOpKind::Attention {
            q,
            k,
            v,
            mask,
            scale,
        } if *mask == AttentionMask::Standard => {
            expand_attention_standard(st, old_id, &node.shape, q, k, v, *scale);
        }

        TensorOpKind::Lstm {
            input,
            hidden_state,
            cell_state,
            weight_ih,
            weight_hh,
            bias,
        } => expand_lstm(
            st,
            old_id,
            &node.shape,
            input,
            hidden_state,
            cell_state,
            weight_ih,
            weight_hh,
            bias,
        ),

        _ => passthrough_node(st, old_id, node),
    }
}

/// Pass through a non-composite node, remapping its input IDs.
fn passthrough_node(st: &mut ExpandState, old_id: usize, node: &TensorNode) {
    let new_kind = node.kind.remap_ids(&st.id_map);
    let new_id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(new_id),
        new_kind,
        node.shape.clone(),
    ));
    st.id_map.insert(old_id, new_id);
}

/// Expand `InstanceNorm1d` to primitives, with optional affine transform.
fn expand_instance_norm(
    st: &mut ExpandState,
    old_id: usize,
    shape: &[usize],
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    gamma: &Option<TensorNodeId>,
    beta: &Option<TensorNodeId>,
) {
    let (mi, me) = (st.id_map[&input.index()], st.id_map[&eps.index()]);
    let full = shape.to_vec();
    let mut reduced = full.clone();
    reduced.remove(axis);
    let normed = emit_instance_norm_core(st, mi, me, axis, &full, &reduced);
    let out = if let Some(g) = gamma {
        let mg = st.id_map[&g.index()];
        let mb = beta.map(|b| st.id_map[&b.index()]);
        emit_affine_transform(st, normed, mg, mb, &full, axis)
    } else {
        normed
    };
    st.id_map.insert(old_id, out);
}

/// Expand `RmsNorm` to primitives: `x * rsqrt(mean(x²) + eps) * weight`.
fn expand_rms_norm(
    st: &mut ExpandState,
    old_id: usize,
    shape: &[usize],
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    weight: &TensorNodeId,
) {
    let (mi, me, mw) = (
        st.id_map[&input.index()],
        st.id_map[&eps.index()],
        st.id_map[&weight.index()],
    );
    let full = shape.to_vec();
    let mut reduced = full.clone();
    reduced.remove(axis);
    let out = emit_rms_norm_core(st, mi, me, mw, axis, &full, &reduced);
    st.id_map.insert(old_id, out);
}

/// Expand `AdaIN1d` to primitives: instance-norm core + style affine.
fn expand_adain1d(
    st: &mut ExpandState,
    old_id: usize,
    shape: &[usize],
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    style_gamma: &TensorNodeId,
    style_beta: &TensorNodeId,
) {
    let (mi, me) = (st.id_map[&input.index()], st.id_map[&eps.index()]);
    let (mg, mb) = (
        st.id_map[&style_gamma.index()],
        st.id_map[&style_beta.index()],
    );
    let full = shape.to_vec();
    let mut reduced = full.clone();
    reduced.remove(axis);
    let normed = emit_instance_norm_core(st, mi, me, axis, &full, &reduced);
    let out = emit_affine_transform(st, normed, mg, Some(mb), &full, axis);
    st.id_map.insert(old_id, out);
}

/// Expand `LayerNorm` to primitives: instance-norm core + right-aligned affine.
///
/// Unlike InstanceNorm where gamma/beta are `[C]` (channel axis), LayerNorm's
/// gamma/beta are `[hidden]` (norm axis = last axis). Use `BroadcastAlignment::Right`
/// to align `[hidden]` against `[B, hidden]`, matching the decomposed builder.
fn expand_layer_norm(
    st: &mut ExpandState,
    old_id: usize,
    shape: &[usize],
    input: &TensorNodeId,
    eps: &TensorNodeId,
    axis: usize,
    weight: &TensorNodeId,
    bias: &TensorNodeId,
) {
    let (mi, me) = (st.id_map[&input.index()], st.id_map[&eps.index()]);
    let (mw, mb) = (st.id_map[&weight.index()], st.id_map[&bias.index()]);
    let full = shape.to_vec();
    let mut reduced = full.clone();
    reduced.remove(axis);
    let normed = emit_instance_norm_core(st, mi, me, axis, &full, &reduced);

    // Right-aligned broadcast: gamma [hidden] → [B, hidden]
    let gb = st.alloc();
    st.push(broadcast_node(gb, mw, &full, BroadcastAlignment::Right));
    let scaled = st.alloc();
    st.push(elementwise_node(
        scaled,
        binop_kernel("mul", BinOpKind::Mul),
        &[normed, gb],
        &full,
    ));
    // Right-aligned broadcast: beta [hidden] → [B, hidden]
    let bb = st.alloc();
    st.push(broadcast_node(bb, mb, &full, BroadcastAlignment::Right));
    let out = st.alloc();
    st.push(elementwise_node(
        out,
        binop_kernel("add", BinOpKind::Add),
        &[scaled, bb],
        &full,
    ));
    st.id_map.insert(old_id, out);
}

/// Expand `Lstm` to 16 decomposed primitive ops via `emit_lstm_cell`.
fn expand_lstm(
    st: &mut ExpandState,
    old_id: usize,
    shape: &[usize],
    input: &TensorNodeId,
    hidden_state: &TensorNodeId,
    cell_state: &TensorNodeId,
    weight_ih: &TensorNodeId,
    weight_hh: &TensorNodeId,
    bias: &Option<TensorNodeId>,
) {
    let mi = st.id_map[&input.index()];
    let mh = st.id_map[&hidden_state.index()];
    let mc = st.id_map[&cell_state.index()];
    let mwih = st.id_map[&weight_ih.index()];
    let mwhh = st.id_map[&weight_hh.index()];
    let mbias = bias.map(|b| st.id_map[&b.index()]);
    let out = emit_lstm_cell(st, mi, mh, mc, mwih, mwhh, mbias, shape);
    st.id_map.insert(old_id, out);
}

/// Returns `true` if the def contains any native norm ops that need expansion.
pub(crate) fn has_norm_ops(def: &TensorKernelDef) -> bool {
    def.nodes.iter().any(|n| {
        matches!(
            n.kind,
            TensorOpKind::InstanceNorm1d { .. }
                | TensorOpKind::RmsNorm { .. }
                | TensorOpKind::AdaIN1d { .. }
                | TensorOpKind::LayerNorm { .. }
        )
    })
}

/// Returns `true` if the def contains any `Lstm` ops that need expansion.
pub(crate) fn has_lstm_ops(def: &TensorKernelDef) -> bool {
    def.nodes
        .iter()
        .any(|n| matches!(n.kind, TensorOpKind::Lstm { .. }))
}

/// Returns `true` if the def contains any `Attention(Standard)` ops that need expansion.
pub(crate) fn has_attention_ops(def: &TensorKernelDef) -> bool {
    def.nodes.iter().any(|n| {
        matches!(
            n.kind,
            TensorOpKind::Attention {
                mask: AttentionMask::Standard,
                ..
            }
        )
    })
}

/// Expand `Attention(Standard)` into `MatMul(Q, K^T, scale) → Softmax → MatMul(attn, V)`.
///
/// Q `[*, T, D]`, K `[*, T_kv, D]`, V `[*, T_kv, D_v]`.
/// Scores = Q @ K^T * scale → `[*, T, T_kv]`.
/// Attn = softmax(scores, -1) → `[*, T, T_kv]`.
/// Output = attn @ V → `[*, T, D_v]`.
fn expand_attention_standard(
    st: &mut ExpandState,
    old_id: usize,
    out_shape: &[usize],
    q: &TensorNodeId,
    k: &TensorNodeId,
    v: &TensorNodeId,
    scale: Option<f32>,
) {
    let mq = st.id_map[&q.index()];
    let mk = st.id_map[&k.index()];
    let mv = st.id_map[&v.index()];

    let q_shape = st.node_shape(mq).to_vec();
    let k_shape = st.node_shape(mk).to_vec();

    // Q: [*, T, D], K: [*, T_kv, D] → scores: [*, T, T_kv]
    let rank = q_shape.len();
    let d_k = q_shape[rank - 1];
    let t_kv = k_shape[rank - 2];

    // Auto-scale: 1/sqrt(d_k) if None
    let effective_scale = scale.unwrap_or(1.0 / (d_k as f32).sqrt());

    // Scores shape: batch dims from Q + [T, T_kv]
    let mut scores_shape = q_shape[..rank - 1].to_vec();
    scores_shape.push(t_kv);

    // Step 1: scores = MatMul(Q, K, transpose_right=true, scale)
    let scores_id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(scores_id),
        TensorOpKind::MatMul {
            left: TensorNodeId::new(mq),
            right: TensorNodeId::new(mk),
            transpose_right: true,
            scale: Some(effective_scale),
        },
        scores_shape.clone(),
    ));

    // Step 2: attn_weights = Softmax(scores, axis=-1)
    let attn_id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(attn_id),
        TensorOpKind::Softmax {
            input: TensorNodeId::new(scores_id),
            axis: i32::try_from(rank).expect("attention rank fits i32") - 1,
        },
        scores_shape,
    ));

    // Step 3: output = MatMul(attn_weights, V, transpose_right=false)
    let out_id = st.alloc();
    st.push(TensorNode::new(
        TensorNodeId::new(out_id),
        TensorOpKind::MatMul {
            left: TensorNodeId::new(attn_id),
            right: TensorNodeId::new(mv),
            transpose_right: false,
            scale: None,
        },
        out_shape.to_vec(),
    ));

    st.id_map.insert(old_id, out_id);
}

#[cfg(test)]
#[path = "codegen_msl_tensor_expand_tests.rs"]
mod tests;
