// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention, composite, and accumulation op compilation for `trace_compile`.
//!
//! Contains compile functions for:
//! - **Sdpa** — decomposed into MatMul(Q, K^T, scale) [+ mask] → Softmax → MatMul(attn, V)
//! - **SwiGlu** — IdentityPassthrough (inner ops traced individually)
//! - **MultiHeadAttention** — IdentityPassthrough (inner ops traced individually)
//! - **RotaryEmbedding** — unsupported (needs pre-computed position tables)
//! - **ScatterAdd/IndexAdd** — unsupported (needs atomic GPU operations)

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, WeightRef};

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorIRError;

use super::{
    add_weight, resolve_input_shape, AttentionLayout, CompiledKernel, CompiledStep, NativeOpKind,
};

// -- Sdpa (Scaled Dot-Product Attention) --------------------------------------

/// Compile `Sdpa { scale }` by decomposing into MatMul + [Mask] + Softmax + MatMul.
///
/// SDPA = softmax(Q @ K^T * scale [+ mask]) @ V
///
/// Inputs: Q `[*, T, D_q]`, K `[*, T_kv, D_q]`, V `[*, T_kv, D_v]`,
///         optional mask `[*, T, T_kv]` (broadcastable).
/// Output: `[*, T, D_v]`.
///
/// Decomposition (3 inputs, no mask):
///   1. scores = MatMul(Q, K^T, scale) → `[*, T, T_kv]`
///   2. attn = Softmax(scores, -1)     → `[*, T, T_kv]`
///   3. output = MatMul(attn, V)       → `[*, T, D_v]`
///
/// Decomposition (4 inputs, with mask):
///   1. scores = MatMul(Q, K^T, scale) → `[*, T, T_kv]`
///   2. scores = scores + broadcast(mask) → `[*, T, T_kv]`
///   3. attn = Softmax(scores, -1)     → `[*, T, T_kv]`
///   4. output = MatMul(attn, V)       → `[*, T, D_v]`
pub(super) fn compile_sdpa(
    node: &TraceNode,
    graph: &ComputationGraph,
    scale: f64,
) -> Result<CompiledStep, TensorIRError> {
    let n_inputs = node.inputs().len();
    if !(3..=4).contains(&n_inputs) {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("sdpa: expected 3 or 4 inputs (Q, K, V [, mask]), got {n_inputs}"),
        });
    }

    let scale_f32 = scale as f32;
    if !scale_f32.is_finite() {
        return Err(TensorIRError::NonFiniteConstant {
            name: "Sdpa scale".into(),
            value: scale,
        });
    }

    let q_shape = resolve_input_shape(node, 0, graph)?;
    let k_shape = resolve_input_shape(node, 1, graph)?;
    let v_shape = resolve_input_shape(node, 2, graph)?;
    let out_shape = node.output_shape();

    // Validate rank: Q, K, V must have at least 2 dimensions.
    if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!(
                "sdpa: rank too low (Q={}, K={}, V={})",
                q_shape.len(),
                k_shape.len(),
                v_shape.len()
            ),
        });
    }

    // Try fused Flash Attention path for eligible cases (#2434).
    // Eligible: 3 inputs (no mask), 4D shapes, head_dim <= 128 (kernel MAX_D),
    // GQA-compatible heads (H_q % H_kv == 0), matching batch/head_dim.
    if n_inputs == 3
        && q_shape.len() == 4
        && k_shape.len() == 4
        && v_shape.len() == 4
        && q_shape[3] > 0
        && q_shape[3] <= 128
        && q_shape[0] == k_shape[0]
        && k_shape[1] > 0
        && q_shape[1] % k_shape[1] == 0
        && k_shape[3] == q_shape[3]
        && v_shape[3] == q_shape[3]
    {
        return Ok(CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention {
                scale: scale_f32,
                causal: false,
                q_shape: q_shape.to_vec(),
                k_shape: k_shape.to_vec(),
                output_shape: out_shape.to_vec(),
                input_layout: AttentionLayout::default(),
            },
            weight_data: HashMap::new(),
        });
    }

    let mut b = TensorBlockBuilder::new("sdpa");
    let q = b.add_input("input_0", q_shape);
    let k = b.add_input("input_1", k_shape);
    let v = b.add_input("input_2", v_shape);

    // Step 1: scores = Q @ K^T * scale → [*, T, T_kv]
    let mut scores_shape = q_shape.to_vec();
    let t_kv = k_shape[k_shape.len() - 2];
    *scores_shape.last_mut().expect("non-empty shape") = t_kv;
    let mut scores = b.add_matmul(q, k, true, Some(scale_f32), &scores_shape);

    // Step 1b (optional): scores += mask — apply attention mask for causal/padding.
    // When 4 inputs are traced, input_3 is the additive mask (e.g. causal mask with
    // -inf entries for masked positions). Fixes #2284.
    if node.inputs().len() >= 4 {
        let mask_shape = resolve_input_shape(node, 3, graph)?;
        let mask = b.add_input("input_3", mask_shape);
        let mask_bc = b.add_broadcast(mask, &scores_shape);
        scores = b.add_binary_add(scores, mask_bc, &scores_shape);
    }

    // Step 2: attn = softmax(scores, dim=-1) → [*, T, T_kv]
    let softmax_axis = scores_shape.len() as i32 - 1;
    let attn = b.add_softmax(scores, softmax_axis, &scores_shape);

    // Step 3: output = attn @ V → [*, T, D_v]
    let output = b.add_matmul(attn, v, false, None, out_shape);

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: super::graph_input_ids(node, n_inputs),
    })
}

// -- SdpaCausal (Scaled Dot-Product Attention with causal masking) -------------

/// Compile `SdpaCausal { scale }` — causal attention without an explicit mask.
///
/// Inputs: Q `[*, T, D]`, K `[*, T, D]`, V `[*, T, D]` (always 3, S_q == S_kv).
/// Output: `[*, T, D]`.
///
/// For eligible 4D cases (head_dim <= 128, GQA-compatible), emits
/// `NativeOpKind::FlashAttention { causal: true }`.
///
/// Falls back to decomposed MatMul + causal mask + Softmax + MatMul otherwise.
pub(super) fn compile_sdpa_causal(
    node: &TraceNode,
    graph: &ComputationGraph,
    scale: f64,
) -> Result<CompiledStep, TensorIRError> {
    let n_inputs = node.inputs().len();
    if n_inputs != 3 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("sdpa_causal: expected 3 inputs (Q, K, V), got {n_inputs}"),
        });
    }

    let scale_f32 = scale as f32;
    if !scale_f32.is_finite() {
        return Err(TensorIRError::NonFiniteConstant {
            name: "SdpaCausal scale".into(),
            value: scale,
        });
    }

    let q_shape = resolve_input_shape(node, 0, graph)?;
    let k_shape = resolve_input_shape(node, 1, graph)?;
    let v_shape = resolve_input_shape(node, 2, graph)?;
    let out_shape = node.output_shape();

    if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!(
                "sdpa_causal: rank too low (Q={}, K={}, V={})",
                q_shape.len(),
                k_shape.len(),
                v_shape.len()
            ),
        });
    }

    // Fused Flash Attention path: 4D, head_dim <= 128, GQA-compatible.
    if q_shape.len() == 4
        && k_shape.len() == 4
        && v_shape.len() == 4
        && q_shape[3] > 0
        && q_shape[3] <= 128
        && q_shape[0] == k_shape[0]
        && k_shape[1] > 0
        && q_shape[1] % k_shape[1] == 0
        && k_shape[3] == q_shape[3]
        && v_shape[3] == q_shape[3]
    {
        return Ok(CompiledStep::NativeOp {
            op: NativeOpKind::FlashAttention {
                scale: scale_f32,
                causal: true,
                q_shape: q_shape.to_vec(),
                k_shape: k_shape.to_vec(),
                output_shape: out_shape.to_vec(),
                input_layout: AttentionLayout::default(),
            },
            weight_data: HashMap::new(),
        });
    }

    // Decomposed fallback: MatMul(Q, K^T, scale) + causal mask + Softmax + MatMul(attn, V).
    let mut b = TensorBlockBuilder::new("sdpa_causal");
    let mut weight_data = HashMap::new();
    let q = b.add_input("input_0", q_shape);
    let k = b.add_input("input_1", k_shape);
    let v = b.add_input("input_2", v_shape);

    let mut scores_shape = q_shape.to_vec();
    let t_kv = k_shape[k_shape.len() - 2];
    *scores_shape.last_mut().expect("non-empty shape") = t_kv;
    let scores = b.add_matmul(q, k, true, Some(scale_f32), &scores_shape);

    // Causal mask as embedded weight (upper-triangular -inf).
    let t_q = q_shape[q_shape.len() - 2];
    let mask_data: Vec<f32> = (0..t_q * t_kv)
        .map(|idx| {
            let row = idx / t_kv;
            let col = idx % t_kv;
            if col > row {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let mask_ref = WeightRef::new(mask_data, vec![t_q, t_kv]).map_err(|_| {
        TensorIRError::UnsupportedTraceOp {
            name: "sdpa_causal: causal mask shape mismatch".into(),
        }
    })?;
    let mask = add_weight(&mut b, &mut weight_data, "causal_mask", &mask_ref);
    let mask_bc = b.add_broadcast(mask, &scores_shape);
    let masked_scores = b.add_binary_add(scores, mask_bc, &scores_shape);

    let softmax_axis = scores_shape.len() as i32 - 1;
    let attn = b.add_softmax(masked_scores, softmax_axis, &scores_shape);

    let output = b.add_matmul(attn, v, false, None, out_shape);

    let def = b.build(output)?;
    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 3),
    })
}

// -- SwiGlu (composite — inner ops traced individually) -----------------------

/// Compile `TraceOp::SwiGlu` as `IdentityPassthrough`.
///
/// SwiGlu's `traced_forward` wrapper was removed so inner ops (Linear, SiLU, Mul)
/// are traced individually. The SwiGlu variant should not appear in compilation
/// traces. This arm prevents `UnsupportedTraceOp` if it does appear, but the
/// passthrough semantics mean the output aliases the input — which is only
/// correct when SwiGlu is a composite marker (not the actual computation).
pub(super) fn compile_swiglu(
    _node: &TraceNode,
    _graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    Ok(CompiledStep::IdentityPassthrough)
}

// -- MultiHeadAttention (composite — inner ops traced individually) -----------

/// Compile `TraceOp::MultiHeadAttention` as `IdentityPassthrough`.
///
/// Per `designs/2026-03-14-attention-trace-integration.md` Step 4:
/// MHA uses `traced_forward` which suppresses internal ops (Linear projections,
/// reshape, SDPA). The compiled plan uses the inner steps, so the MHA node is
/// a marker that passes through.
///
/// For compilation (not verification), models should trace without
/// `traced_forward` so inner ops appear individually and compile correctly.
pub(super) fn compile_mha(
    _node: &TraceNode,
    _graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    Ok(CompiledStep::IdentityPassthrough)
}

// -- RotaryEmbedding ----------------------------------------------------------

/// Compile `TraceOp::RotaryEmbedding` as a `NativeOpKind::RotaryEmbedding`.
///
/// The trace captures pre-narrowed cos/sin caches as `WeightRef` data inside
/// `TraceOp::RotaryEmbedding { cos_cache, sin_cache, head_dim, offset }`.
/// These are compile-time constants (position embeddings for the traced
/// sequence length). The compiled model executor delegates to
/// `MetalDynBackend::gpu_rope(x, cos, sin)` which applies the fused rotation
/// in a single dispatch graph.
///
/// Input 0: `[..., S, D]` where D = head_dim (must be even).
/// Weights: `"cos_cache"` `[S, D/2]`, `"sin_cache"` `[S, D/2]`.
/// Output: same shape as input.
///
/// Part of #3526.
pub(super) fn compile_rope(
    node: &TraceNode,
    graph: &ComputationGraph,
    head_dim: usize,
    cos_cache: &WeightRef,
    sin_cache: &WeightRef,
) -> Result<CompiledStep, TensorIRError> {
    let input_shape = resolve_input_shape(node, 0, graph)?;
    let rank = input_shape.len();

    if rank < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("rope: input rank must be >= 2, got {rank}"),
        });
    }

    let last_dim = input_shape[rank - 1];
    if last_dim != head_dim {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("rope: input last dim ({last_dim}) does not match head_dim ({head_dim})"),
        });
    }

    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("rope: head_dim must be a positive even number, got {head_dim}"),
        });
    }

    let mut weight_data = HashMap::new();
    weight_data.insert("cos_cache".to_string(), cos_cache.clone());
    weight_data.insert("sin_cache".to_string(), sin_cache.clone());

    Ok(CompiledStep::NativeOp {
        op: NativeOpKind::RotaryEmbedding {
            head_dim,
            input_shape: input_shape.to_vec(),
        },
        weight_data,
    })
}

// -- ScatterAdd (unsupported — needs atomic GPU operations) -------------------

/// Compile `TraceOp::ScatterAdd` — currently unsupported.
///
/// Scatter-add is a data-dependent write: for each index `i`, it adds
/// `src[i]` to `self[index[i]]` along `dim`. Multiple indices can map to
/// the same destination, requiring atomic add on GPU. The current DispatchStep
/// infrastructure does not support atomics.
pub(super) fn compile_scatter_add(
    _node: &TraceNode,
    _graph: &ComputationGraph,
    _dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    Err(TensorIRError::UnsupportedTraceOp {
        name: "scatter_add (requires atomic GPU operations — not yet supported)".into(),
    })
}

// -- IndexAdd (unsupported — needs atomic GPU operations) ---------------------

/// Compile `TraceOp::IndexAdd` — currently unsupported.
///
/// Index-add is similar to scatter-add: adds `src` into `self` along `dim`
/// at positions given by `index`. Requires atomic add for correctness when
/// multiple indices collide.
pub(super) fn compile_index_add(
    _node: &TraceNode,
    _graph: &ComputationGraph,
    _dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    Err(TensorIRError::UnsupportedTraceOp {
        name: "index_add (requires atomic GPU operations — not yet supported)".into(),
    })
}

pub(super) fn compile_index_put(
    _node: &TraceNode,
    _graph: &ComputationGraph,
    _dim: usize,
) -> Result<CompiledStep, TensorIRError> {
    Err(TensorIRError::UnsupportedTraceOp {
        name: "index_put (requires scatter GPU operations)".into(),
    })
}
