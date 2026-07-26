// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention tensor op translation to NY `SelfAttentionLayer`.
//!
//! Maps `TensorOpKind::Attention { q, k, v, mask, scale }` to
//! `Layer::SelfAttention(SelfAttentionLayer::new(mask, scale))`.
//!
//! NY's `SelfAttentionLayer` decomposes to:
//!   softmax(Q @ K^T * scale, mask) @ V
//!
//! **Self-attention:** Q, K, V are all Variable → ternary IBP propagation.
//!
//! **Cross-attention:** Q is Variable, K/V are `WeightTensor` (constant KV input).
//! Constants are injected as a `Linear(zeros, constant_as_bias) -> Reshape` subgraph,
//! producing degenerate bounds (`lower == upper == constant`) with the constant's
//! *own* shape. This is the established constant-injection idiom (cf. LSTM initial
//! states in `graph_tensor_lstm.rs`). Because the zero-weight `Linear` ignores its
//! parent's values *and* the trailing `Reshape` fixes the output shape from the
//! constant itself, the injected K/V are fully decoupled from Q's shape. This
//! supports genuine asymmetric cross-attention where `KV_SEQ != Q_SEQ` (e.g. DETR
//! decoder cross-attention over an encoder memory of a different length). See #830.
//!
//! **Constant query:** Q may itself be a `WeightTensor` — e.g. DETR/Qwen learned
//! object-query / query embeddings, which are fixed model parameters rather than a
//! function of the variable input. Such a constant Q is injected via the *same*
//! `inject_constant_via_zero_mul` path as constant K/V (degenerate bounds equal to
//! the exact query), so NY's `SelfAttention` still propagates sound bounds. The
//! injected constant needs a variable parent to seed a valid graph edge; we use
//! whichever of K/V is `Variable` as that anchor.
//!
//! **Fully-constant attention:** If Q, K, *and* V are all constant (e.g. the DETR
//! decoder self-attention over fixed object queries before any encoder memory has
//! entered the graph), the entire attention output is itself a constant. We fold it
//! eagerly to the exact forward result `softmax((Q·Kᵀ)·scale, mask)·V` and return a
//! `WeightTensor`, exactly mirroring the constant-fold path in `graph_tensor_linear.rs`
//! for a Linear over a constant input. This is sound: the output is a precisely-known
//! point value (the true model output), so degenerate bounds around it are exact — no
//! over-approximation is needed or made. Downstream nodes (Reshape / residual-Add /
//! the next Linear) consume the `WeightTensor` via the same constant-fold idioms.
//!
//! See `designs/2026-03-02-attention-composition.md`.

use ny_core::nan_propagating_max;
use ny_propagate::layers::{
    AttentionMask as GcAttentionMask, LinearLayer, ReshapeLayer, SelfAttentionLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::{AttentionMask, TensorNode, TensorNodeId};
use ndarray::{Array1, Array2, ArrayD, Axis};

use super::TensorNodeValue;
use crate::error::VerifyError;
use crate::util::get_value;

/// Convert nn-dsl `AttentionMask` to NY `AttentionMask`.
///
/// Returns `Err` for unknown variants (AttentionMask is `#[non_exhaustive]`).
fn convert_mask(mask: &AttentionMask) -> Result<GcAttentionMask, VerifyError> {
    match mask {
        AttentionMask::Standard => Ok(GcAttentionMask::Standard),
        AttentionMask::Causal => Ok(GcAttentionMask::Causal),
        // AttentionMask is #[non_exhaustive] — reject unknown variants.
        other => Err(VerifyError::UnsupportedOp(format!(
            "unsupported AttentionMask variant: {other:?}"
        ))),
    }
}

/// Inject a constant `WeightTensor` as a NY graph node with the constant's *own*
/// shape, fully decoupled from the parent node's shape.
///
/// Creates a small subgraph chained off `parent_name`:
/// 1. `Reshape([-1])` — flatten the parent to 1-D `[parent_total]` (cheap; shares
///    no values with the output, it only seeds a valid graph edge).
/// 2. `Linear(zeros[1, parent_total], bias=[0.0])` — collapse to a scalar `[1]`.
///    The zero weight makes the output `0.0` regardless of the parent's bounds,
///    so the constant never depends on the variable input. Using a `[1,
///    parent_total]` weight (rather than `[const_total, parent_total]`) keeps the
///    injected weight matrices tiny even when `parent_total` is large.
/// 3. `Linear(zeros[const_total, 1], bias=constant_flat)` — expand the scalar to
///    the flattened constant. Again the zero weight means the output is always
///    exactly `constant_flat` (degenerate bounds `lower == upper == constant`).
/// 4. `Reshape(const_shape)` — restore the constant's N-D shape.
///
/// Soundness: every value-carrying weight is zero, so the injected node produces
/// `lower == upper == constant` (an exact degenerate interval) with the
/// constant's shape, *independent of `parent_shape`*. This is the same idiom used
/// for LSTM initial states (`graph_tensor_lstm.rs::inject_constant_state`).
///
/// Decoupling from `parent_shape` is what enables asymmetric cross-attention
/// (`KV_SEQ != Q_SEQ`): NY's `SelfAttentionLayer` only requires Q and K to share
/// the contraction (head) dim and K/V to share `KV_SEQ`; the query and key/value
/// sequence lengths may differ. Those remaining real shape constraints are still
/// enforced soundly by NY's MatMul at propagation time (structured
/// `ShapeMismatch`, never a panic). We keep a minimal rank + batch-dim sanity
/// check here so a genuinely malformed constant (wrong rank / wrong number of
/// heads) is still rejected up front with a clear message. See #830.
///
/// # Errors
///
/// Returns `VerifyError::UnsupportedOp` if the constant's rank or leading batch
/// dimensions (all axes except the last two) disagree with `parent_shape`, or
/// `VerifyError::InternalTranslationError` if the zero-weight `Linear` fails to
/// construct.
fn inject_constant_via_zero_mul(
    name: &str,
    constant: &ArrayD<f32>,
    parent_name: &str,
    parent_shape: &[usize],
    graph: &mut GraphNetwork,
) -> Result<String, VerifyError> {
    let const_shape = constant.shape();
    // Sound sanity check: the query/key/value sequence length (axis -2) and the
    // value head dim (axis -1) may legitimately differ between Q and constant
    // K/V, so we do NOT compare those. We only require the rank and the leading
    // batch dimensions (e.g. the head count) to agree — a mismatch there is a
    // genuinely malformed spec, not valid asymmetric attention. The remaining
    // (contraction / KV_SEQ) constraints are enforced soundly by NY's MatMul.
    // Attention operands are always rank >= 2 ([..heads, seq, head_dim]); guard the
    // `len() - 2` slices below so a degenerate rank-0/1 shape can't underflow.
    let rank_ok = const_shape.len() == parent_shape.len();
    let batch_ok = const_shape.len() >= 2
        && parent_shape.len() >= 2
        && parent_shape[..parent_shape.len() - 2] == const_shape[..const_shape.len() - 2];
    if !rank_ok || !batch_ok {
        return Err(VerifyError::UnsupportedOp(format!(
            "cross-attention constant '{name}' shape {const_shape:?} is incompatible with \
             parent '{parent_name}' shape {parent_shape:?} (rank and leading batch dimensions \
             must match; sequence length and value head dim may differ)"
        )));
    }

    let parent_total: usize = parent_shape.iter().product();
    let const_flat: Vec<f32> = constant.iter().copied().collect();
    let const_total = const_flat.len();

    // Guard against degenerate (empty) shapes that would make the zero-weight
    // Linear ill-formed. NY's MatMul would reject these downstream too, but a
    // clear up-front error is friendlier.
    if parent_total == 0 || const_total == 0 {
        return Err(VerifyError::UnsupportedOp(format!(
            "cross-attention constant '{name}' or parent '{parent_name}' has an empty shape \
             (parent {parent_shape:?}, constant {const_shape:?})"
        )));
    }

    // 1. Flatten parent to 1-D [parent_total] so the collapse Linear sees a known
    //    `in_features`. Reshape only seeds a valid edge; its values are discarded
    //    by the zero weight below.
    let flat_name = format!("{name}_flat");
    graph.add_node(GraphNode::new(
        flat_name.clone(),
        Layer::Reshape(ReshapeLayer::new(vec![-1])),
        vec![parent_name.to_string()],
    ));

    // 2. Collapse to scalar [1] with a zero weight: output is always 0.0,
    //    decoupling the constant from the variable parent entirely.
    let collapse = LinearLayer::new(Array2::zeros((1, parent_total)), Some(Array1::zeros(1)))
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("cross-attention constant '{name}' collapse Linear: {e}"),
        })?;
    let scalar_name = format!("{name}_scalar");
    graph.add_node(GraphNode::new(
        scalar_name.clone(),
        Layer::Linear(collapse),
        vec![flat_name],
    ));

    // 3. Expand the scalar to the flattened constant via a zero weight + constant
    //    bias: output is always exactly `const_flat` (degenerate bounds).
    let expand = LinearLayer::new(
        Array2::zeros((const_total, 1)),
        Some(Array1::from_vec(const_flat)),
    )
    .map_err(|e| VerifyError::InternalTranslationError {
        context: format!("cross-attention constant '{name}' expand Linear: {e}"),
    })?;
    let const_1d_name = format!("{name}_1d");
    graph.add_node(GraphNode::new(
        const_1d_name.clone(),
        Layer::Linear(expand),
        vec![scalar_name],
    ));

    // 4. Restore the constant's N-D shape.
    let target: Vec<i64> = const_shape.iter().map(|&d| d as i64).collect();
    graph.add_node(GraphNode::new(
        name.to_string(),
        Layer::Reshape(ReshapeLayer::new(target)),
        vec![const_1d_name],
    ));
    Ok(name.to_string())
}

/// Exact forward attention for a single 2-D head: `softmax((q·kᵀ)·scale, mask)·v`.
///
/// `q` is `[Sq, D]`, `k` is `[Sk, D]`, `v` is `[Sk, Dv]`; the result is `[Sq, Dv]`.
/// The softmax is the ordinary (numerically-stabilized, max-subtracted) forward
/// softmax over the key axis — i.e. the exact value the real model computes. For
/// `Causal`, row `i` attends only to keys `j <= i` (matching NY's
/// `CausalSoftmaxLayer::eval_row`); masked positions contribute exactly zero.
///
/// This is an *exact* point computation (no interval widening): all inputs are
/// precisely-known constants, so the output is the precisely-known true model
/// output. That is what makes folding sound — see `fold_constant_attention`.
fn attention_head_2d(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    scale: f32,
    causal: bool,
) -> Result<Array2<f32>, VerifyError> {
    let seq_q = q.shape()[0];
    let seq_k = k.shape()[0];
    let dv = v.shape()[1];

    // scores = (q · kᵀ) * scale  →  [Sq, Sk]
    let mut scores = q.dot(&k.t());
    scores *= scale;

    let mut out = Array2::<f32>::zeros((seq_q, dv));
    for i in 0..seq_q {
        // Active key range: all keys for Standard, j <= i for Causal (no sliding
        // window here — translate_attention never sets one). Mirrors NY's
        // `active_range`: active_end = min(i+1, seq_k) under causal masking.
        let active_end = if causal {
            (i + 1).min(seq_k)
        } else {
            seq_k
        };
        // Stabilized softmax over the active range (max-subtraction, like NY and
        // every real softmax kernel). NaN-propagating max so a NaN score is not
        // silently dropped.
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..active_end {
            max_val = nan_propagating_max(max_val, scores[[i, j]]);
        }
        let mut sum_exp = 0.0_f32;
        let mut probs = vec![0.0_f32; active_end];
        for j in 0..active_end {
            let e = (scores[[i, j]] - max_val).exp();
            probs[j] = e;
            sum_exp += e;
        }
        // Exact softmax denominator (no epsilon): the true model normalizes by the
        // plain sum of exponentials. A degenerate range (active_end == 0) cannot
        // occur — row i always includes key i for causal, and seq_k >= 1 otherwise.
        let inv_sum = 1.0 / sum_exp;
        // out[i, :] = sum_j probs[j] * v[j, :]
        for j in 0..active_end {
            let w = probs[j] * inv_sum;
            for d in 0..dv {
                out[[i, d]] += w * v[[j, d]];
            }
        }
    }
    // Finiteness guard (same as the Linear constant-fold path).
    for &val in out.iter() {
        if !val.is_finite() {
            return Err(VerifyError::NonFiniteConstant {
                value: val,
                context: "constant-attention fold".to_string(),
            });
        }
    }
    Ok(out)
}

/// Fold a fully-constant attention (`Q`, `K`, `V` all `WeightTensor`) to its exact
/// forward output `softmax((Q·Kᵀ)·scale, mask)·V`, returned as a `WeightTensor`.
///
/// Soundness: every operand is a precisely-known constant, so the result is the
/// precisely-known true model output — a degenerate point. Returning it as a
/// `WeightTensor` (consumed by the same constant-fold idioms used elsewhere, e.g.
/// `graph_tensor_linear.rs`) is therefore exact, not an over-approximation. This is
/// the attention analogue of constant-folding a `Linear` over a constant input.
///
/// Supports rank-2 `[S, D]` and rank-`n` batched `[..batch, S, D]` operands (the
/// leading batch axes are the attention heads). Q/K/V must share rank and leading
/// batch dims; K and V must share the key sequence length; Q and K must share the
/// head (contraction) dim — all enforced here with clear errors.
fn fold_constant_attention(
    node_id: TensorNodeId,
    q: &ArrayD<f32>,
    k: &ArrayD<f32>,
    v: &ArrayD<f32>,
    mask: &AttentionMask,
    scale: Option<f32>,
) -> Result<TensorNodeValue, VerifyError> {
    let causal = match mask {
        AttentionMask::Standard => false,
        AttentionMask::Causal => true,
        // AttentionMask is #[non_exhaustive]; reject unknown variants rather than
        // silently treating them as standard.
        other => {
            return Err(VerifyError::UnsupportedOp(format!(
                "constant-attention fold: unsupported AttentionMask variant: {other:?}"
            )));
        }
    };

    let qs = q.shape();
    let ks = k.shape();
    let vs = v.shape();
    // Attention operands are rank >= 2 ([..heads, seq, head_dim]).
    if qs.len() < 2 || qs.len() != ks.len() || qs.len() != vs.len() {
        return Err(VerifyError::UnsupportedOp(format!(
            "constant-attention fold t{}: Q/K/V must share rank >= 2; got Q={qs:?}, K={ks:?}, V={vs:?}",
            node_id.index()
        )));
    }
    let rank = qs.len();
    // Leading batch (head) dims must match across all three.
    if qs[..rank - 2] != ks[..rank - 2] || qs[..rank - 2] != vs[..rank - 2] {
        return Err(VerifyError::UnsupportedOp(format!(
            "constant-attention fold t{}: leading batch/head dims must match; got Q={qs:?}, K={ks:?}, V={vs:?}",
            node_id.index()
        )));
    }
    let (seq_q, d_q) = (qs[rank - 2], qs[rank - 1]);
    let (seq_k, d_k) = (ks[rank - 2], ks[rank - 1]);
    let (seq_v, d_v) = (vs[rank - 2], vs[rank - 1]);
    // Q and K share the contraction (head) dim; K and V share the key seq length.
    if d_q != d_k {
        return Err(VerifyError::UnsupportedOp(format!(
            "constant-attention fold t{}: Q head dim {d_q} != K head dim {d_k}",
            node_id.index()
        )));
    }
    if seq_k != seq_v {
        return Err(VerifyError::UnsupportedOp(format!(
            "constant-attention fold t{}: K key-seq {seq_k} != V key-seq {seq_v}",
            node_id.index()
        )));
    }
    if seq_q == 0 || seq_k == 0 || d_q == 0 || d_v == 0 {
        return Err(VerifyError::UnsupportedOp(format!(
            "constant-attention fold t{}: empty operand shape Q={qs:?}, K={ks:?}, V={vs:?}",
            node_id.index()
        )));
    }

    // Resolve scale exactly as NY does (`resolve_scale`): explicit scale, else
    // 1/sqrt(head_dim) from Q's last dim, with the same f32 exact-integer guard.
    let scale = match scale {
        Some(s) => s,
        None => {
            if d_q > (1 << 24) {
                return Err(VerifyError::UnsupportedOp(format!(
                    "constant-attention fold t{}: head_dim {d_q} exceeds f32 exact integer range",
                    node_id.index()
                )));
            }
            1.0 / (d_q as f32).sqrt()
        }
    };

    // Flatten the leading batch/head axes so we can iterate 2-D heads uniformly.
    // `to_shape` (vs `into_shape_with_order`) tolerates non-contiguous strides by
    // cloning as needed — robust to whatever layout the upstream constant-folds
    // produced. Mirrors the batched-MatMul reshape in `graph_tensor_matmul.rs`.
    let num_heads: usize = qs[..rank - 2].iter().product::<usize>().max(1);
    let q3 =
        q.to_shape((num_heads, seq_q, d_q))
            .map_err(|e| VerifyError::InternalTranslationError {
                context: format!("constant-attention fold: reshape Q to heads: {e}"),
            })?;
    let k3 =
        k.to_shape((num_heads, seq_k, d_k))
            .map_err(|e| VerifyError::InternalTranslationError {
                context: format!("constant-attention fold: reshape K to heads: {e}"),
            })?;
    let v3 =
        v.to_shape((num_heads, seq_v, d_v))
            .map_err(|e| VerifyError::InternalTranslationError {
                context: format!("constant-attention fold: reshape V to heads: {e}"),
            })?;

    let mut out = ndarray::Array3::<f32>::zeros((num_heads, seq_q, d_v));
    for h in 0..num_heads {
        let qh = q3.index_axis(Axis(0), h).to_owned();
        let kh = k3.index_axis(Axis(0), h).to_owned();
        let vh = v3.index_axis(Axis(0), h).to_owned();
        let oh = attention_head_2d(&qh, &kh, &vh, scale, causal)?;
        out.slice_mut(ndarray::s![h, .., ..]).assign(&oh);
    }

    // Restore the original leading batch/head shape: [..batch, seq_q, d_v].
    let mut out_shape: Vec<usize> = qs[..rank - 2].to_vec();
    out_shape.push(seq_q);
    out_shape.push(d_v);
    let out = out
        .into_dyn()
        .into_shape_with_order(ndarray::IxDyn(&out_shape))
        .map_err(|e| VerifyError::InternalTranslationError {
            context: format!("constant-attention fold: restore output shape: {e}"),
        })?;
    Ok(TensorNodeValue::WeightTensor(out))
}

/// Translate a `TensorOpKind::Attention` node to a NY `SelfAttentionLayer`.
///
/// Q, K, and V can each be `Variable` (self-attention) or `WeightTensor` (a constant
/// query / key / value, e.g. a learned object-query embedding or constant KV memory).
/// Any `WeightTensor` operand is injected as a `Linear(zeros, constant) -> Reshape`
/// subgraph that produces degenerate bounds (`lower == upper == constant`) with the
/// constant's *own* shape, decoupled from the variable anchor. At least one of Q/K/V
/// must be `Variable` to seed the injected constants' graph edges; if all three are
/// constant the attention output is itself a constant, which we do not fold here.
///
/// **Shape constraint:** A constant operand may have a different sequence length than
/// the variable anchor (asymmetric cross-attention is supported, e.g. DETR decoder
/// cross-attention, or a constant query of a different length than the K/V memory).
/// Only the rank and leading batch dimensions (e.g. head count) must match the anchor;
/// the remaining real constraints (matching head dim, matching KV_SEQ between K and V)
/// are enforced soundly by NY's MatMul at propagation time. See #830.
///
/// `scale` is forwarded to NY. When `None`, NY auto-infers
/// `1/sqrt(d_k)` from Q's last dimension at propagation time.
pub(super) fn translate_attention(
    node_id: TensorNodeId,
    q_id: &TensorNodeId,
    k_id: &TensorNodeId,
    v_id: &TensorNodeId,
    mask: &AttentionMask,
    scale: Option<f32>,
    node_values: &[TensorNodeValue],
    all_nodes: &[TensorNode],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // First, try to de-fuse a standard multi-head self-attention into per-head
    // 2-D primitives so the Q@Kᵀ score reaches NY's tight zonotope bound (the
    // fused SelfAttentionLayer uses plain IBP for the score). Only fires for the
    // exact projection pattern with a shared variable base, no projection bias,
    // and within the CROWN size budget; on ANY mismatch it returns None and we
    // keep the fused node below (sound, just looser). The de-fused graph computes
    // the identical function — see `graph_tensor_attention_defuse.rs`.
    if let Some(val) = super::attention_defuse::try_defuse_mha(
        node_id, q_id, k_id, v_id, mask, scale, node_values, all_nodes, graph,
    )? {
        return Ok(val);
    }

    let q_val = get_value(node_values, q_id.index(), "Attention Q")?;
    let k_val = get_value(node_values, k_id.index(), "Attention K")?;
    let v_val = get_value(node_values, v_id.index(), "Attention V")?;

    // Pick a variable "anchor" among Q/K/V. Any `WeightTensor` operand is injected
    // as a constant via `inject_constant_via_zero_mul`, which needs a variable
    // parent to seed a valid graph edge (its values are discarded by a zero weight,
    // so the choice of anchor does not affect the injected constant's bounds — only
    // its leading batch dims are sanity-checked against the anchor). We prefer the
    // operand with the most leading batch context: Q, then K, then V. The anchor's
    // node shape (its own, possibly asymmetric, shape) is used only for the
    // rank/batch-dim sanity check on injected constants — never a seq-length
    // constraint (#830).
    let anchor: Option<(&str, &[usize])> = if let TensorNodeValue::Variable(name) = q_val {
        Some((name, &all_nodes[q_id.index()].shape))
    } else if let TensorNodeValue::Variable(name) = k_val {
        Some((name, &all_nodes[k_id.index()].shape))
    } else if let TensorNodeValue::Variable(name) = v_val {
        Some((name, &all_nodes[v_id.index()].shape))
    } else {
        None
    };

    let (anchor_name, anchor_shape) = match anchor {
        Some(a) => a,
        None => {
            // No variable among Q/K/V. If all three are constant tensors, the whole
            // attention output is a precisely-known constant — fold it exactly to a
            // WeightTensor (the attention analogue of the Linear constant-fold).
            // Otherwise (e.g. a scalar Constant operand) keep a clear error.
            if let (
                TensorNodeValue::WeightTensor(qa),
                TensorNodeValue::WeightTensor(ka),
                TensorNodeValue::WeightTensor(va),
            ) = (q_val, k_val, v_val)
            {
                return fold_constant_attention(node_id, qa, ka, va, mask, scale);
            }
            return Err(VerifyError::UnsupportedOp(format!(
                "Attention requires at least one of Q/K/V to be Variable, or all three to be \
                 constant WeightTensors (which are folded); got Q={q_val:?}, K={k_val:?}, \
                 V={v_val:?}"
            )));
        }
    };
    // Own the anchor name so the immutable borrow of `node_values` does not conflict
    // with the `&mut graph` injections below.
    let anchor_name = anchor_name.to_string();
    let anchor_shape = anchor_shape.to_vec();

    // Q can be Variable or WeightTensor (a constant query, e.g. a learned
    // object-query / query-embedding). A constant Q is injected with degenerate
    // bounds equal to the exact query; NY's SelfAttention then propagates sound
    // bounds. Reject other value kinds (e.g. scalar Constant) as before.
    let q_name = match q_val {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::WeightTensor(arr) => {
            let name = format!("t{}_q_const", node_id.index());
            inject_constant_via_zero_mul(&name, arr, &anchor_name, &anchor_shape, graph)?
        }
        _ => {
            return Err(VerifyError::UnsupportedOp(format!(
                "Attention Q must be Variable or WeightTensor; got Q={q_val:?}"
            )));
        }
    };

    // K can be Variable or WeightTensor (cross-attention constant KV).
    let k_name = match k_val {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::WeightTensor(arr) => {
            let name = format!("t{}_k_const", node_id.index());
            inject_constant_via_zero_mul(&name, arr, &anchor_name, &anchor_shape, graph)?
        }
        _ => {
            return Err(VerifyError::UnsupportedOp(format!(
                "Attention K must be Variable or WeightTensor; got K={k_val:?}"
            )));
        }
    };

    // V can be Variable or WeightTensor.
    let v_name = match v_val {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::WeightTensor(arr) => {
            let name = format!("t{}_v_const", node_id.index());
            inject_constant_via_zero_mul(&name, arr, &anchor_name, &anchor_shape, graph)?
        }
        _ => {
            return Err(VerifyError::UnsupportedOp(format!(
                "Attention V must be Variable or WeightTensor; got V={v_val:?}"
            )));
        }
    };

    let node_name = format!("t{}_attention", node_id.index());
    let gc_mask = convert_mask(mask)?;
    let layer = Layer::SelfAttention(SelfAttentionLayer::new(gc_mask, scale));
    graph.add_node(GraphNode::new(
        node_name.clone(),
        layer,
        vec![q_name, k_name, v_name],
    ));
    Ok(TensorNodeValue::Variable(node_name))
}

#[cfg(test)]
#[path = "graph_tensor_attention_tests.rs"]
mod tests;
