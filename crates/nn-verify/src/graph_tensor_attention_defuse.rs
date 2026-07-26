// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! De-fuse standard multi-head self-attention into per-head 2-D primitives so the
//! `Q@Kᵀ` score reaches NY's tight zonotope bound (the fused `Layer::SelfAttention`
//! path bounds the score with plain term-wise IBP).
//!
//! ## Why
//!
//! NY has a 14–55× tighter zonotope bound for `Q@Kᵀ` that tracks the `Q,K`
//! correlation (both projected from the same LayerNorm output). It fires *only*
//! for a **2-D** `Layer::MatMul(transpose_b=true)` whose Q and K trace back
//! through `{Reshape, Tile, last-two-axes Transpose, Linear}` to a **shared
//! Linear base**. The standard MHA decomposition emitted by
//! `add_multi_head_attention` reaches the fused node as **3-D** `[H, S, HD]`
//! tensors via a `[1,0,2]` (first-two-axes) transpose, so the zonotope bails to
//! IBP on both counts (rank-3 *and* unsupported transpose).
//!
//! ## What this does
//!
//! When the Attention node's Q/K/V each trace back to a per-head-sliceable shared
//! `Linear(base, W, bias=None)`, we emit, per head `h`:
//!
//! ```text
//! q_h = Linear(base, W_q[h·HD:(h+1)·HD, :])  -> [S, HD]   (2-D, pure Linear)
//! k_h, v_h likewise
//! scores_h = MatMul(q_h, k_h, transpose_b=true, scale)  -> [S, S]   ← ZONOTOPE
//! probs_h  = Softmax/CausalSoftmax(scores_h, axis=-1)    -> [S, S]
//! out_h    = MatMul(probs_h, v_h, transpose_b=false)     -> [S, HD]  ← simplex@V
//! ```
//!
//! and assemble the output to match the fused node's declared shape (multi-head:
//! Unsqueeze+Concat into `[H,S,HD]`; direct-2-D: the single head's `[S,D]`).
//!
//! ## Soundness — identical function
//!
//! Reshaping `[S,D] -> [S,H,HD]` maps `q[s, h·HD+d] -> [s,h,d]`; the `[1,0,2]`
//! transpose gives `q_t[h,s,d] = q[s, h·HD+d]`. Since `q[s,:] = base[s,:] @ W_qᵀ`,
//! `q[s, h·HD+d] = base[s,:] · W_q[h·HD+d, :]`, i.e. head `h` uses exactly **rows
//! `h·HD..(h+1)·HD` of `W_q`** — the row-block this module slices. So `q_h = base @
//! (W_q[head rows])ᵀ` *is* the fused head-`h` query, element-for-element. The
//! per-head scaled-dot-product softmax and `@V` are the same op the fused
//! `SelfAttentionLayer` computes per head. The function is **identical** (validated
//! numerically to f32 tolerance) and the bound is **tighter-or-equal** everywhere
//! (the zonotope score is a sound enclosure NY intersects with IBP; the softmax@V
//! simplex lever applies on this 2-D path exactly as on the fused path).
//!
//! ## Memory gate (CROWN)
//!
//! The fused `SelfAttention` node has no CROWN backward, so alpha-CROWN cleanly
//! falls back to (cheap) IBP. The de-fused MatMul/Softmax nodes DO have a CROWN
//! backward (McCormick), so CROWN now traverses them — which is O(H·S²·…) memory
//! per attention. To avoid regressing the CROWN compose tests into an OOM, we only
//! de-fuse when `num_heads * seq² <= DEFUSE_SCORE_BUDGET`; larger attention keeps
//! the fused node (sound, looser score, but CROWN-cheap). The zonotope score
//! benefit is realized at IBP time regardless, where it is cheap.
//!
//! ## Fail-safe
//!
//! Any deviation from the exact pattern (bias present, base not a shared Variable,
//! non-`[1,0,2]` transpose, reshape not `[S,H,HD]`, weight not 2-D, unknown mask,
//! over the size budget) returns `Ok(None)` and the caller keeps the fused node —
//! **sound, just looser**. We never emit a different function.

use ny_propagate::layers::{
    CausalSoftmaxLayer, ConcatLayer, LinearLayer, MatMulLayer, ReshapeLayer, SoftmaxLayer,
    UnsqueezeLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::tensor_ir::{AttentionMask, TensorNode, TensorNodeId, TensorOpKind};
use ndarray::{Array2, ArrayD};

use super::TensorNodeValue;
use crate::error::VerifyError;
use crate::util::get_value;

/// Upper bound on `num_heads * seq²` for de-fusing.
///
/// The de-fused per-head path exposes binary `MatMul`/`Softmax` nodes to CROWN's
/// McCormick backward, whereas the fused `SelfAttention` node has no CROWN backward
/// (CROWN cheaply falls back to IBP). For *small* attention this is fine — and is
/// where the zonotope IBP score tightening is realized cheaply — but for
/// production-sequence attention the per-head CROWN backward through a residual can
/// allocate enormous bound matrices (observed to exceed machine memory at
/// `H=4, S=64`, in BOTH the fused and de-fused forms — i.e. those tests are already
/// memory-bound independent of this change).
///
/// To guarantee the de-fusion never *adds* CROWN memory risk to the existing
/// CROWN-heavy compose tests, we de-fuse only when `num_heads * seq²` is small
/// enough that the per-head CROWN backward stays cheap. This covers the small
/// encoder blocks (DETR-small `S=16` => 1024, SVTR `S=16` => 2048) where the score
/// tightening is delivered; larger attention (DETR-medium `S=32`,
/// table_transformer `S=64`) keeps the fused node — sound, with CROWN behaviour
/// IDENTICAL to before this change. Tunable via `NN_VERIFY_DEFUSE_SCORE_BUDGET`
/// (set high to de-fuse all sizes for IBP-only verification).
const DEFUSE_SCORE_BUDGET_DEFAULT: usize = 4096; // e.g. S=16,H<=16 or S=32,H<=4

fn defuse_score_budget() -> usize {
    std::env::var("NN_VERIFY_DEFUSE_SCORE_BUDGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFUSE_SCORE_BUDGET_DEFAULT)
}

/// One projection's traced-back structure: the shared base IR id and its weight
/// array, plus the per-head split dims (`S`, `H`, `HD`).
struct TracedProjection<'a> {
    base_id: TensorNodeId,
    weight: &'a ArrayD<f32>,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
}

/// Validate a `Linear(base, W, bias=None)` node whose declared output is exactly
/// 2-D `[seq, d_out]`, returning `(base_id, weight, d_out)`. `None` on any
/// mismatch (non-Linear, bias present, non-2-D / non-constant weight, output rank
/// != 2). The weight is borrowed from `node_values`.
fn validate_no_bias_linear<'a>(
    lin_id: TensorNodeId,
    expect_seq: usize,
    all_nodes: &[TensorNode],
    node_values: &'a [TensorNodeValue],
) -> Option<(TensorNodeId, &'a ArrayD<f32>, usize)> {
    let lin = all_nodes.get(lin_id.index())?;
    let TensorOpKind::Linear {
        input: base_id,
        weight: w_id,
        bias,
    } = &lin.kind
    else {
        return None;
    };
    if bias.is_some() {
        return None;
    }
    let TensorNodeValue::WeightTensor(weight) = node_values.get(w_id.index())? else {
        return None;
    };
    if weight.ndim() != 2 {
        return None;
    }
    let d_out = weight.shape()[0];
    if lin.shape.as_slice() != [expect_seq, d_out] {
        return None;
    }
    Some((*base_id, weight, d_out))
}

/// Trace an attention operand back to a per-head-sliceable shared Linear base.
///
/// Two accepted shapes (both produce a *2-D* per-head `Q@Kᵀ` reaching the
/// zonotope), else `None`:
/// 1. **Multi-head:** `Transpose([1,0,2]) <- Reshape([S,H,HD]) <- Linear(base,W)`.
/// 2. **Direct 2-D:** the operand *is* a `Linear(base,W)` producing `[S,D]`
///    (single conceptual head, `H=1, HD=D`).
fn trace_projection<'a>(
    start: TensorNodeId,
    all_nodes: &[TensorNode],
    node_values: &'a [TensorNodeValue],
) -> Option<TracedProjection<'a>> {
    let node = all_nodes.get(start.index())?;

    // Case 2: the operand is a Linear directly (2-D `[S, D]`, single head).
    if let TensorOpKind::Linear { .. } = &node.kind {
        if node.shape.len() != 2 {
            return None;
        }
        let seq = node.shape[0];
        if seq == 0 {
            return None;
        }
        let (base_id, weight, d_out) =
            validate_no_bias_linear(start, seq, all_nodes, node_values)?;
        if d_out == 0 {
            return None;
        }
        return Some(TracedProjection {
            base_id,
            weight,
            seq,
            num_heads: 1,
            head_dim: d_out,
        });
    }

    // Case 1: multi-head Transpose([1,0,2]) <- Reshape([S,H,HD]) <- Linear.
    let TensorOpKind::Transpose { input: re_id, axes } = &node.kind else {
        return None;
    };
    if axes.as_slice() != [1, 0, 2] {
        return None;
    }
    let re = all_nodes.get(re_id.index())?;
    let TensorOpKind::Reshape {
        input: lin_id,
        target_shape,
    } = &re.kind
    else {
        return None;
    };
    if target_shape.len() != 3 {
        return None;
    }
    let (seq, num_heads, head_dim) = (target_shape[0], target_shape[1], target_shape[2]);
    if num_heads == 0 || head_dim == 0 || seq == 0 {
        return None;
    }
    let d_out = num_heads.checked_mul(head_dim)?;
    let (base_id, weight, w_out) = validate_no_bias_linear(*lin_id, seq, all_nodes, node_values)?;
    if w_out != d_out {
        return None;
    }
    Some(TracedProjection {
        base_id,
        weight,
        seq,
        num_heads,
        head_dim,
    })
}

/// Slice row-block `h·HD..(h+1)·HD` of `[D, in]` weight into an owned 2-D
/// `[HD, in]` array for a per-head `LinearLayer`.
fn head_weight_rows(weight: &ArrayD<f32>, h: usize, head_dim: usize) -> Option<Array2<f32>> {
    let w2 = weight.view().into_dimensionality::<ndarray::Ix2>().ok()?;
    let block = w2.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
    Some(block.to_owned())
}

/// Try to de-fuse a standard multi-head self-attention node into per-head 2-D
/// primitives that reach NY's zonotope `Q@Kᵀ` tightening. Returns `Ok(Some(val))`
/// with the attention output (drop-in for the fused node, shape matching the
/// declared output) when the exact pattern matches and the size budget allows,
/// else `Ok(None)` to fall through to the fused path.
pub(super) fn try_defuse_mha(
    node_id: TensorNodeId,
    q_id: &TensorNodeId,
    k_id: &TensorNodeId,
    v_id: &TensorNodeId,
    mask: &AttentionMask,
    scale: Option<f32>,
    node_values: &[TensorNodeValue],
    all_nodes: &[TensorNode],
    graph: &mut GraphNetwork,
) -> Result<Option<TensorNodeValue>, VerifyError> {
    // Only the two masks the per-head softmax can replicate exactly.
    let causal = match mask {
        AttentionMask::Standard => false,
        AttentionMask::Causal => true,
        _ => return Ok(None),
    };

    let (Some(qp), Some(kp), Some(vp)) = (
        trace_projection(*q_id, all_nodes, node_values),
        trace_projection(*k_id, all_nodes, node_values),
        trace_projection(*v_id, all_nodes, node_values),
    ) else {
        return Ok(None);
    };

    // Q/K/V must share one base, the same head split, and the base must be a
    // propagatable Variable.
    if qp.base_id != kp.base_id || qp.base_id != vp.base_id {
        return Ok(None);
    }
    if (qp.num_heads, qp.head_dim, qp.seq) != (kp.num_heads, kp.head_dim, kp.seq)
        || (qp.num_heads, qp.head_dim, qp.seq) != (vp.num_heads, vp.head_dim, vp.seq)
    {
        return Ok(None);
    }
    let TensorNodeValue::Variable(base_name) =
        get_value(node_values, qp.base_id.index(), "MHA de-fuse shared base")?
    else {
        return Ok(None);
    };
    let base_name = base_name.clone();

    let (seq, num_heads, head_dim) = (qp.seq, qp.num_heads, qp.head_dim);

    // Safety ceiling on `H·S²` (see `DEFUSE_SCORE_BUDGET_DEFAULT`): a guard against
    // pathologically large attention only — at realistic transformer sizes the
    // de-fused per-head path is lighter than the fused path, not a regression. Also
    // bails on `checked_mul` overflow (a degenerate/huge shape).
    match num_heads.checked_mul(seq).and_then(|hs| hs.checked_mul(seq)) {
        Some(cost) if cost <= defuse_score_budget() => {}
        _ => return Ok(None),
    }

    // Resolve the per-head score scale exactly as the fused path does: explicit
    // `scale` is used as-is (same function for any scale); when None, NY infers
    // `1/sqrt(Q.last_dim)`, which equals `head_dim` in both forms here.
    let head_scale = match scale {
        Some(s) => s,
        None => {
            if head_dim > (1 << 24) {
                return Ok(None);
            }
            1.0_f32 / (head_dim as f32).sqrt()
        }
    };

    // Output shape must match the fused node's DECLARED output (drop-in):
    //   rank 3 `[H,S,HD]` -> stack heads; rank 2 `[S,D]` -> single head's `[S,D]`.
    let declared = all_nodes
        .get(node_id.index())
        .map(|n| n.shape.as_slice())
        .unwrap_or(&[]);
    let assemble_2d = match declared.len() {
        2 => {
            if num_heads != 1 || declared != [seq, head_dim] {
                return Ok(None);
            }
            true
        }
        3 => {
            if declared != [num_heads, seq, head_dim] {
                return Ok(None);
            }
            false
        }
        _ => return Ok(None),
    };

    // Build per-head 2-D subgraphs.
    let prefix = format!("t{}_dfmha", node_id.index());
    let out_name = format!("t{}_attention", node_id.index());
    let mut head_outputs: Vec<String> = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let make_head_linear = |which: &str,
                                weight: &ArrayD<f32>,
                                graph: &mut GraphNetwork|
         -> Result<String, VerifyError> {
            let rows = head_weight_rows(weight, h, head_dim).ok_or_else(|| {
                VerifyError::InternalTranslationError {
                    context: format!("MHA de-fuse: {which} head {h} weight slice failed"),
                }
            })?;
            let lin =
                LinearLayer::new(rows, None).map_err(|e| VerifyError::WeightValidation {
                    op: "Attention(de-fused)",
                    reason: format!("{which} head {h} LinearLayer: {e}"),
                })?;
            let name = format!("{prefix}_{which}{h}");
            graph.add_node(GraphNode::new(
                name.clone(),
                Layer::Linear(lin),
                vec![base_name.clone()],
            ));
            Ok(name)
        };
        let q_h = make_head_linear("q", qp.weight, graph)?;
        let k_h = make_head_linear("k", kp.weight, graph)?;
        let v_h = make_head_linear("v", vp.weight, graph)?;

        // scores_h = MatMul(q_h, k_h, transpose_b=true, scale) -> [S, S]  (ZONOTOPE)
        let scores_h = format!("{prefix}_s{h}");
        graph.add_node(GraphNode::binary(
            scores_h.clone(),
            Layer::MatMul(MatMulLayer::new(true, Some(head_scale))),
            q_h,
            k_h,
        ));

        // probs_h = softmax over the key axis (last). Causal masks j>i exactly as
        // NY's CausalSoftmaxLayer (== the fused SelfAttention causal semantics).
        let probs_h = format!("{prefix}_p{h}");
        let softmax_layer = if causal {
            Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1))
        } else {
            Layer::Softmax(SoftmaxLayer::new(-1))
        };
        graph.add_node(GraphNode::new(probs_h.clone(), softmax_layer, vec![scores_h]));

        // out_h = MatMul(probs_h, v_h, transpose_b=false) -> [S, HD]  (simplex@V)
        let out_h = if assemble_2d {
            out_name.clone()
        } else {
            format!("{prefix}_o{h}")
        };
        graph.add_node(GraphNode::binary(
            out_h.clone(),
            Layer::MatMul(MatMulLayer::new(false, None)),
            probs_h,
            v_h,
        ));
        head_outputs.push(out_h);
    }

    if assemble_2d {
        return Ok(Some(TensorNodeValue::Variable(out_name)));
    }

    // Multi-head: Unsqueeze each `[S,HD]` to `[1,S,HD]`, then Concat(axis=0) ->
    // `[H, S, HD]` (the drop-in for the fused node).
    let mut unsqueezed: Vec<String> = Vec::with_capacity(num_heads);
    for (h, out_h) in head_outputs.iter().enumerate() {
        let unsq = format!("{prefix}_u{h}");
        graph.add_node(GraphNode::new(
            unsq.clone(),
            Layer::Unsqueeze(UnsqueezeLayer::new(0)),
            vec![out_h.clone()],
        ));
        unsqueezed.push(unsq);
    }
    if unsqueezed.len() == 1 {
        let only = unsqueezed.into_iter().next().expect("len==1");
        graph.add_node(GraphNode::new(
            out_name.clone(),
            Layer::Reshape(ReshapeLayer::new(
                declared.iter().map(|&d| d as i64).collect(),
            )),
            vec![only],
        ));
    } else {
        graph.add_node(GraphNode::new(
            out_name.clone(),
            Layer::Concat(ConcatLayer::new(0)),
            unsqueezed,
        ));
    }

    Ok(Some(TensorNodeValue::Variable(out_name)))
}
