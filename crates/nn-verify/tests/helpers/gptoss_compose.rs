// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for gpt-oss-20b NY composition tests.
//!
//! Part of #4271: gpt-oss NY compose verification.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{ReduceOp, TensorKernelDef};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

pub(super) const SEQ_LEN: usize = 4;
pub(super) const HIDDEN_DIM: usize = 16;
pub(super) const NUM_HEADS: usize = 4;
pub(super) const NUM_KV_HEADS: usize = 2;
pub(super) const HEAD_DIM: usize = 8;
pub(super) const HALF_DIM: usize = HEAD_DIM / 2;
pub(super) const NUM_EXPERTS: usize = 4;
pub(super) const TOP_K: usize = 2;
pub(super) const INTERMEDIATE: usize = HIDDEN_DIM;
pub(super) const SWIGLU_LIMIT: f32 = 7.0;
pub(super) const SLIDING_WINDOW: usize = 2;
const WEIGHT_MAG: f32 = 0.02;
const RMS_EPS: f32 = 1e-5;
pub(super) const ATTN_DIM: usize = NUM_HEADS * HEAD_DIM;

// 19. Attention Sink Bias
//
// Models the StreamingLLM attention sink applied to position 0:
//   scores [SEQ_LEN, SEQ_LEN] + sink_bias [1, SEQ_LEN] broadcast → biased → softmax
pub(super) fn build_attn_sink_bias() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_attn_sink_bias");
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let bias_shape = [1, SEQ_LEN];

    let scores = b.add_input("scores", &scores_shape);
    let sink_bias = b.add_input("sink_bias", &bias_shape);

    let bias_bc = b.add_broadcast(sink_bias, &scores_shape);
    let biased = b.add_binary_add(scores, bias_bc, &scores_shape);
    let output = b.add_softmax(biased, 1, &scores_shape);

    b.build(output).expect("valid attn_sink_bias sub-graph")
}

pub(super) fn attn_sink_bias_bindings() -> Vec<TensorParamBinding> {
    // sink_bias: bias value at column 0 only, zero elsewhere
    let mut bias_data = vec![0.0f32; SEQ_LEN];
    bias_data[0] = 2.0; // positive bias for sink token at position 0
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[1, SEQ_LEN]), bias_data).unwrap(),
        ),
    ]
}

// 20. YaRN Frequency Modulation
//
// Models the YaRN frequency scaling applied to RoPE:
//   base_freq [SEQ_LEN, HALF_DIM] * scale_factors [HALF_DIM] broadcast → scaled_freq
pub(super) fn build_yarn_freq_mod() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_yarn_freq_mod");
    let freq_shape = [SEQ_LEN, HALF_DIM];
    let scale_shape = [HALF_DIM];

    let base_freq = b.add_input("base_freq", &freq_shape);
    let scale_factors = b.add_input("scale_factors", &scale_shape);

    let scale_bc = b.add_broadcast(scale_factors, &freq_shape);
    let output = b.add_binary_mul(base_freq, scale_bc, &freq_shape);

    b.build(output).expect("valid yarn_freq_mod sub-graph")
}

pub(super) fn yarn_freq_mod_bindings() -> Vec<TensorParamBinding> {
    // scale_factors precomputed from yarn_factor and original_max
    let scale_data: Vec<f32> = (0..HALF_DIM)
        .map(|i| {
            let t = i as f32 / HALF_DIM.max(1) as f32;
            // YaRN interpolation: scale between 1.0 and yarn_factor
            1.0 + t * 0.5 // gentle ramp from 1.0 to 1.5
        })
        .collect();
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[HALF_DIM]), scale_data).unwrap(),
        ),
    ]
}

// 21. Top-k Expert Selection
//
// Models softmax router → narrow(top_k) → sum → broadcast_div (renormalization).
// We model this as softmax → narrow → reduce_sum → broadcast → mul as a proxy
// for top-k selection and weight renormalization.
pub(super) fn build_topk_expert_select() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_topk_expert_select");
    let logit_shape = [SEQ_LEN, NUM_EXPERTS];
    let topk_shape = [SEQ_LEN, TOP_K];
    let sum_shape = [SEQ_LEN, 1];

    let logits = b.add_input("router_logits", &logit_shape);

    // softmax over experts
    let probs = b.add_softmax(logits, 1, &logit_shape);
    // narrow to first TOP_K experts (proxy for top-k selection)
    let topk_probs = b.add_narrow(probs, 1, 0, TOP_K, &topk_shape);
    // sum for renormalization denominator
    let topk_sum = b.add_reduce(topk_probs, ReduceOp::Sum, 1, true, &sum_shape);
    // broadcast sum back to topk shape
    let sum_bc = b.add_broadcast(topk_sum, &topk_shape);
    // renormalize: topk_probs * sum (proxy for division by sum)
    let output = b.add_binary_mul(topk_probs, sum_bc, &topk_shape);

    b.build(output).expect("valid topk_expert_select sub-graph")
}

pub(super) fn topk_expert_select_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

// 22. Clamped SwiGLU Activation
//
// Models the full SwiGLU: gate * sigmoid(gate) * up.
// Clamp is omitted since TensorBlockBuilder lacks binary max/min ops.
// gate [SEQ_LEN, INTERMEDIATE], up [SEQ_LEN, INTERMEDIATE]
//   → sigmoid(gate) → mul(gate, sigmoid) → mul(silu_gate, up)
pub(super) fn build_clamped_swiglu() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_clamped_swiglu");
    let inter_shape = [SEQ_LEN, INTERMEDIATE];

    let gate = b.add_input("gate", &inter_shape);
    let up = b.add_input("up", &inter_shape);

    let gate_sig = b.add_sigmoid(gate, &inter_shape);
    let silu_gate = b.add_binary_mul(gate, gate_sig, &inter_shape);
    let output = b.add_binary_mul(silu_gate, up, &inter_shape);

    b.build(output).expect("valid clamped_swiglu sub-graph")
}

pub(super) fn clamped_swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable, TensorParamBinding::Variable]
}

// 23. GQA KV Repeat
//
// Models the GQA KV head repetition for multi-head attention:
//   kv [SEQ_LEN, HEAD_DIM] → concat(kv, kv) along the head/feature dim (axis 1)
//   → [SEQ_LEN, 2*HEAD_DIM]
// repeat_kv duplicates the KV head along the feature axis (NOT the sequence
// axis); concat on axis 1 is the correct proxy for repeat_kv with factor=2.
pub(super) fn build_gqa_kv_repeat() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_gqa_kv_repeat");
    let kv_shape = [SEQ_LEN, HEAD_DIM];
    let repeated_shape = [SEQ_LEN, 2 * HEAD_DIM];

    let kv = b.add_input("kv", &kv_shape);

    let output = b.add_concat(&[kv, kv], 1, &repeated_shape);

    b.build(output).expect("valid gqa_kv_repeat sub-graph")
}

pub(super) fn gqa_kv_repeat_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

fn deterministic_weights(size: usize, mag: f32) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let t = i as f32 / size.max(1) as f32;
            mag * (2.0 * t - 1.0)
        })
        .collect()
}

fn ones_vec(size: usize) -> Vec<f32> {
    vec![1.0f32; size]
}

// 1. Embedding + RMSNorm + Q projection
pub(super) fn build_embed_rmsnorm_proj() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_embed_rmsnorm_proj");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let out_shape = [SEQ_LEN, ATTN_DIM];

    let input = b.add_input("hidden", &hidden_shape);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let proj_weight = b.add_input("proj_weight", &[HIDDEN_DIM, ATTN_DIM]);

    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &hidden_shape);
    let projected = b.add_matmul(normed, proj_weight, false, None, &out_shape);

    b.build(projected)
        .expect("valid embed_rmsnorm_proj sub-graph")
}

pub(super) fn embed_rmsnorm_proj_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), RMS_EPS)),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM]), ones_vec(HIDDEN_DIM)).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, ATTN_DIM]),
                deterministic_weights(HIDDEN_DIM * ATTN_DIM, WEIGHT_MAG),
            )
            .unwrap(),
        ),
    ]
}

// 2. MoE Router
pub(super) fn build_moe_router() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_moe_router");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let logit_shape = [SEQ_LEN, NUM_EXPERTS];

    let input = b.add_input("hidden", &hidden_shape);
    let router_weight = b.add_input("router_weight", &[HIDDEN_DIM, NUM_EXPERTS]);
    let router_bias = b.add_input("router_bias", &[NUM_EXPERTS]);

    let logits = b.add_matmul(input, router_weight, false, None, &logit_shape);
    let bias_bc = b.add_broadcast(router_bias, &logit_shape);
    let biased = b.add_binary_add(logits, bias_bc, &logit_shape);
    let probs = b.add_softmax(biased, 1, &logit_shape);

    b.build(probs).expect("valid moe_router sub-graph")
}

pub(super) fn moe_router_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, NUM_EXPERTS]),
                deterministic_weights(HIDDEN_DIM * NUM_EXPERTS, WEIGHT_MAG),
            )
            .unwrap(),
        ),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_EXPERTS]), 0.0f32)),
    ]
}

// 3. SwiGLU Expert
pub(super) fn build_swiglu_expert() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_swiglu_expert");
    let input_shape = [SEQ_LEN, HIDDEN_DIM];
    let fused_dim = 2 * INTERMEDIATE;
    let fused_shape = [SEQ_LEN, fused_dim];
    let inter_shape = [SEQ_LEN, INTERMEDIATE];

    let input = b.add_input("x", &input_shape);
    let gate_up_w = b.add_input("gate_up_weight", &[HIDDEN_DIM, fused_dim]);
    let down_w = b.add_input("down_weight", &[INTERMEDIATE, HIDDEN_DIM]);

    let gate_up = b.add_matmul(input, gate_up_w, false, None, &fused_shape);
    let gate = b.add_narrow(gate_up, 1, 0, INTERMEDIATE, &inter_shape);
    let up = b.add_narrow(gate_up, 1, INTERMEDIATE, INTERMEDIATE, &inter_shape);
    let gate_sig = b.add_sigmoid(gate, &inter_shape);
    let silu_gate = b.add_binary_mul(gate, gate_sig, &inter_shape);
    let hidden = b.add_binary_mul(silu_gate, up, &inter_shape);
    let output = b.add_matmul(hidden, down_w, false, None, &input_shape);

    b.build(output).expect("valid swiglu_expert sub-graph")
}

pub(super) fn swiglu_expert_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, 2 * INTERMEDIATE]),
                deterministic_weights(HIDDEN_DIM * 2 * INTERMEDIATE, WEIGHT_MAG),
            )
            .unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[INTERMEDIATE, HIDDEN_DIM]),
                deterministic_weights(INTERMEDIATE * HIDDEN_DIM, WEIGHT_MAG),
            )
            .unwrap(),
        ),
    ]
}

// 4. Decoder Layer
pub(super) fn build_decoder_layer() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_decoder_layer");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &hidden_shape);
    let eps1 = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("input_layernorm_weight", &[HIDDEN_DIM]);
    let attn_w = b.add_input("attn_proxy_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let eps2 = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("post_attention_layernorm_weight", &[HIDDEN_DIM]);
    let expert_w = b.add_input("expert_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let normed1 = b.add_rms_norm(input, eps1, 1, norm1_w, &hidden_shape);
    let attn_out = b.add_matmul(normed1, attn_w, false, None, &hidden_shape);
    let residual1 = b.add_binary_add(input, attn_out, &hidden_shape);

    let normed2 = b.add_rms_norm(residual1, eps2, 1, norm2_w, &hidden_shape);
    let moe_out = b.add_matmul(normed2, expert_w, false, None, &hidden_shape);
    let output = b.add_binary_add(residual1, moe_out, &hidden_shape);

    b.build(output).expect("valid decoder_layer sub-graph")
}

pub(super) fn decoder_layer_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), RMS_EPS)),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM]), ones_vec(HIDDEN_DIM)).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
                deterministic_weights(HIDDEN_DIM * HIDDEN_DIM, WEIGHT_MAG),
            )
            .unwrap(),
        ),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), RMS_EPS)),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM]), ones_vec(HIDDEN_DIM)).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
                deterministic_weights(HIDDEN_DIM * HIDDEN_DIM, WEIGHT_MAG * 0.5),
            )
            .unwrap(),
        ),
    ]
}

// 5. Sliding Window Attention
pub(super) fn build_sliding_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_sliding_attention");
    let qk_shape = [SEQ_LEN, HEAD_DIM];
    let v_shape = [SEQ_LEN, HEAD_DIM];
    let scores_shape = [SEQ_LEN, SEQ_LEN];

    let q = b.add_input("q", &qk_shape);
    let k = b.add_input("k", &qk_shape);
    let v = b.add_input("v", &v_shape);
    let mask = b.add_input("sliding_mask", &scores_shape);

    let kt = b.add_transpose(k, &[1, 0], &[HEAD_DIM, SEQ_LEN]);
    let scores = b.add_matmul(q, kt, false, None, &scores_shape);
    let masked = b.add_binary_add(scores, mask, &scores_shape);
    let attn_weights = b.add_softmax(masked, 1, &scores_shape);
    let output = b.add_matmul(attn_weights, v, false, None, &v_shape);

    b.build(output).expect("valid sliding_attention sub-graph")
}

pub(super) fn sliding_attention_bindings() -> Vec<TensorParamBinding> {
    let mut mask_data = vec![0.0f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            if j > i || (i > SLIDING_WINDOW && j < i - SLIDING_WINDOW) {
                mask_data[i * SEQ_LEN + j] = -1e9;
            }
        }
    }
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, SEQ_LEN]), mask_data).unwrap(),
        ),
    ]
}

// 6. Full Attention
pub(super) fn build_full_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_full_attention");
    let qk_shape = [SEQ_LEN, HEAD_DIM];
    let v_shape = [SEQ_LEN, HEAD_DIM];
    let scores_shape = [SEQ_LEN, SEQ_LEN];

    let q = b.add_input("q", &qk_shape);
    let k = b.add_input("k", &qk_shape);
    let v = b.add_input("v", &v_shape);
    let mask = b.add_input("causal_mask", &scores_shape);

    let kt = b.add_transpose(k, &[1, 0], &[HEAD_DIM, SEQ_LEN]);
    let scores = b.add_matmul(q, kt, false, None, &scores_shape);
    let masked = b.add_binary_add(scores, mask, &scores_shape);
    let attn_weights = b.add_softmax(masked, 1, &scores_shape);
    let output = b.add_matmul(attn_weights, v, false, None, &v_shape);

    b.build(output).expect("valid full_attention sub-graph")
}

pub(super) fn full_attention_bindings() -> Vec<TensorParamBinding> {
    let mut mask_data = vec![0.0f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            if j > i {
                mask_data[i * SEQ_LEN + j] = -1e9;
            }
        }
    }
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, SEQ_LEN]), mask_data).unwrap(),
        ),
    ]
}

// 7. KV Cache Append + Sliding Window Eviction
pub(super) fn build_kv_cache_sliding() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_kv_cache_sliding");
    let new_len = 2;
    let old_len = 3;
    let total_len = old_len + new_len;
    let kv_shape_new = [new_len, HEAD_DIM];
    let kv_shape_old = [old_len, HEAD_DIM];
    let kv_shape_cat = [total_len, HEAD_DIM];
    let evict_start = total_len.saturating_sub(SLIDING_WINDOW);
    let kv_shape_out = [SLIDING_WINDOW.min(total_len), HEAD_DIM];

    let old_kv = b.add_input("old_kv", &kv_shape_old);
    let new_kv = b.add_input("new_kv", &kv_shape_new);
    let concatenated = b.add_concat(&[old_kv, new_kv], 0, &kv_shape_cat);
    let output = b.add_narrow(concatenated, 0, evict_start, kv_shape_out[0], &kv_shape_out);

    b.build(output).expect("valid kv_cache_sliding sub-graph")
}

pub(super) fn kv_cache_sliding_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable, TensorParamBinding::Variable]
}

// 8. MXFP4 Dequantization
pub(super) fn build_mxfp4_dequant() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_mxfp4_dequant");
    let n = 8;
    let val_shape = [n];

    let fp4_values = b.add_input("fp4_values", &val_shape);
    let scale = b.add_input("scale", &[1]);
    let scale_bc = b.add_broadcast(scale, &val_shape);
    let output = b.add_binary_mul(fp4_values, scale_bc, &val_shape);

    b.build(output).expect("valid mxfp4_dequant sub-graph")
}

pub(super) fn mxfp4_dequant_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.25f32)),
    ]
}

// 9. Residual Add (input + layer_output)
pub(super) fn build_residual_add() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_residual_add");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &shape);
    let layer_output = b.add_input("layer_output", &shape);
    let output = b.add_binary_add(input, layer_output, &shape);

    b.build(output).expect("valid residual_add sub-graph")
}

pub(super) fn residual_add_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable, TensorParamBinding::Variable]
}

// 10. LM Head (RMSNorm + linear projection -> logits)
pub(super) fn build_lm_head() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_lm_head");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    // Small vocab for tractability (production is 201088)
    let vocab_size = 32;
    let logit_shape = [SEQ_LEN, vocab_size];

    let input = b.add_input("norm_output", &hidden_shape);
    let lm_weight = b.add_input("lm_head_weight", &[HIDDEN_DIM, vocab_size]);

    let logits = b.add_matmul(input, lm_weight, false, None, &logit_shape);

    b.build(logits).expect("valid lm_head sub-graph")
}

pub(super) fn lm_head_bindings() -> Vec<TensorParamBinding> {
    let vocab_size = 32;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, vocab_size]),
                deterministic_weights(HIDDEN_DIM * vocab_size, WEIGHT_MAG),
            )
            .unwrap(),
        ),
    ]
}

// 11. Embedding Table Lookup
pub(super) fn build_embed_lookup() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_embed_lookup");
    let vocab_size = 32;
    let indices_shape = [SEQ_LEN];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    let indices = b.add_input("token_ids", &indices_shape);
    let embed_weight = b.add_input("embed_weight", &[vocab_size, HIDDEN_DIM]);

    let output = b.add_embedding(indices, embed_weight, &out_shape);

    b.build(output).expect("valid embed_lookup sub-graph")
}

pub(super) fn embed_lookup_bindings() -> Vec<TensorParamBinding> {
    let vocab_size = 32;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[vocab_size, HIDDEN_DIM]),
                deterministic_weights(vocab_size * HIDDEN_DIM, WEIGHT_MAG),
            )
            .unwrap(),
        ),
    ]
}

// 12. RoPE Cos/Sin Pair Application
//
// Models applying cos/sin rotary embeddings to a query/key vector pair:
//   x_rotated = x * cos_theta + x_paired * sin_theta
// where x_paired is the rotated-partner vector (even<->odd dim swap).
pub(super) fn build_rope_pair() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_rope_pair");
    let vec_shape = [SEQ_LEN, HALF_DIM];

    let x = b.add_input("x", &vec_shape);
    let x_paired = b.add_input("x_paired", &vec_shape);
    let cos_theta = b.add_input("cos_theta", &vec_shape);
    let sin_theta = b.add_input("sin_theta", &vec_shape);

    // x * cos_theta
    let x_cos = b.add_binary_mul(x, cos_theta, &vec_shape);
    // x_paired * sin_theta
    let xp_sin = b.add_binary_mul(x_paired, sin_theta, &vec_shape);
    // x_rotated = x * cos + x_paired * sin
    let output = b.add_binary_add(x_cos, xp_sin, &vec_shape);

    b.build(output).expect("valid rope_pair sub-graph")
}

pub(super) fn rope_pair_bindings() -> Vec<TensorParamBinding> {
    // cos and sin are precomputed constants from RoPE frequencies
    let cos_data: Vec<f32> = (0..SEQ_LEN * HALF_DIM)
        .map(|i| {
            let pos = (i / HALF_DIM) as f32;
            let dim = (i % HALF_DIM) as f32;
            let freq = pos / 10000.0_f32.powf(2.0 * dim / HEAD_DIM as f32);
            freq.cos()
        })
        .collect();
    let sin_data: Vec<f32> = (0..SEQ_LEN * HALF_DIM)
        .map(|i| {
            let pos = (i / HALF_DIM) as f32;
            let dim = (i % HALF_DIM) as f32;
            let freq = pos / 10000.0_f32.powf(2.0 * dim / HEAD_DIM as f32);
            freq.sin()
        })
        .collect();
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HALF_DIM]), cos_data).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HALF_DIM]), sin_data).unwrap(),
        ),
    ]
}

// 13. MoE Weighted Expert Combination
//
// Models the weighted combination of top-k expert outputs:
//   output = sum_k(weight_k * expert_k_output)
// For tractability, we use K=2 experts with softmax-renormalized weights.
pub(super) fn build_moe_weight_combine() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_moe_weight_combine");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let weight_shape = [SEQ_LEN, 1];

    let expert1 = b.add_input("expert1_output", &hidden_shape);
    let expert2 = b.add_input("expert2_output", &hidden_shape);
    let w1 = b.add_input("weight1", &weight_shape);
    let w2 = b.add_input("weight2", &weight_shape);

    // Broadcast weights to hidden_dim
    let w1_bc = b.add_broadcast(w1, &hidden_shape);
    let w2_bc = b.add_broadcast(w2, &hidden_shape);

    // Weighted expert outputs
    let weighted1 = b.add_binary_mul(expert1, w1_bc, &hidden_shape);
    let weighted2 = b.add_binary_mul(expert2, w2_bc, &hidden_shape);

    // Sum: output = w1 * expert1 + w2 * expert2
    let output = b.add_binary_add(weighted1, weighted2, &hidden_shape);

    b.build(output).expect("valid moe_weight_combine sub-graph")
}

pub(super) fn moe_weight_combine_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        // Weights are variables (from softmax routing, bounded [0, 1])
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
    ]
}

// 14. GQA Head Projection (Q projection with single-head slice)
//
// Models: hidden → matmul(q_weight) → [SEQ_LEN, ATTN_DIM] → narrow to single head
pub(super) fn build_gqa_head_proj() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_gqa_head_proj");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let full_proj_shape = [SEQ_LEN, ATTN_DIM];
    let head_shape = [SEQ_LEN, HEAD_DIM];

    let input = b.add_input("hidden", &hidden_shape);
    let q_weight = b.add_input("q_weight", &[HIDDEN_DIM, ATTN_DIM]);

    let projected = b.add_matmul(input, q_weight, false, None, &full_proj_shape);
    // Narrow to first head: [SEQ_LEN, HEAD_DIM]
    let head_slice = b.add_narrow(projected, 1, 0, HEAD_DIM, &head_shape);

    b.build(head_slice).expect("valid gqa_head_proj sub-graph")
}

pub(super) fn gqa_head_proj_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, ATTN_DIM]),
                deterministic_weights(HIDDEN_DIM * ATTN_DIM, WEIGHT_MAG),
            )
            .unwrap(),
        ),
    ]
}

// 15. Attention Score Scaling
//
// Models: Q @ K^T / sqrt(head_dim) with causal mask
pub(super) fn build_attn_score_scale() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_attn_score_scale");
    let qk_shape = [SEQ_LEN, HEAD_DIM];
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scale_shape = [1];

    let q = b.add_input("q", &qk_shape);
    let k = b.add_input("k", &qk_shape);
    let scale = b.add_input("scale", &scale_shape);
    let causal_mask = b.add_input("causal_mask", &scores_shape);

    let kt = b.add_transpose(k, &[1, 0], &[HEAD_DIM, SEQ_LEN]);
    let scores = b.add_matmul(q, kt, false, None, &scores_shape);
    let scale_bc = b.add_broadcast(scale, &scores_shape);
    let scaled = b.add_binary_mul(scores, scale_bc, &scores_shape);
    let masked = b.add_binary_add(scaled, causal_mask, &scores_shape);

    b.build(masked).expect("valid attn_score_scale sub-graph")
}

pub(super) fn attn_score_scale_bindings() -> Vec<TensorParamBinding> {
    let scale_val = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut mask_data = vec![0.0f32; SEQ_LEN * SEQ_LEN];
    for i in 0..SEQ_LEN {
        for j in 0..SEQ_LEN {
            if j > i {
                mask_data[i * SEQ_LEN + j] = -1e9;
            }
        }
    }
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), scale_val)),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, SEQ_LEN]), mask_data).unwrap(),
        ),
    ]
}

// 16. Expert Gate Split (fused gate_up → SiLU gate * up)
//
// Models the expert FFN gate/up split and SiLU activation:
//   input [SEQ_LEN, 2*INTERMEDIATE] → gate = narrow → up = narrow
//   → sigmoid(gate) → silu_gate = gate * sigmoid → output = silu_gate * up
pub(super) fn build_expert_gate_split() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_expert_gate_split");
    let fused_dim = 2 * INTERMEDIATE;
    let fused_shape = [SEQ_LEN, fused_dim];
    let inter_shape = [SEQ_LEN, INTERMEDIATE];

    let input = b.add_input("fused_gate_up", &fused_shape);

    let gate = b.add_narrow(input, 1, 0, INTERMEDIATE, &inter_shape);
    let up = b.add_narrow(input, 1, INTERMEDIATE, INTERMEDIATE, &inter_shape);
    let gate_sig = b.add_sigmoid(gate, &inter_shape);
    let silu_gate = b.add_binary_mul(gate, gate_sig, &inter_shape);
    let output = b.add_binary_mul(silu_gate, up, &inter_shape);

    b.build(output).expect("valid expert_gate_split sub-graph")
}

pub(super) fn expert_gate_split_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

// 17. Two-Layer Residual Stack
//
// Models composition of two decoder layers:
//   input → rms_norm → matmul (attn proxy) → add (residual)
//         → rms_norm → matmul (moe proxy) → add (residual) → [layer 1 done]
//         → rms_norm → matmul (attn proxy) → add (residual)
//         → rms_norm → matmul (moe proxy) → add (residual) → output
pub(super) fn build_two_layer_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_two_layer_residual");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];

    let input = b.add_input("hidden", &hidden_shape);

    // Layer 1 norm/attn weights
    let eps1a = b.add_input("layer1_norm1_eps", &[1]);
    let norm1a_w = b.add_input("layer1_norm1_weight", &[HIDDEN_DIM]);
    let attn1_w = b.add_input("layer1_attn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let eps1b = b.add_input("layer1_norm2_eps", &[1]);
    let norm1b_w = b.add_input("layer1_norm2_weight", &[HIDDEN_DIM]);
    let moe1_w = b.add_input("layer1_moe_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Layer 2 norm/attn weights
    let eps2a = b.add_input("layer2_norm1_eps", &[1]);
    let norm2a_w = b.add_input("layer2_norm1_weight", &[HIDDEN_DIM]);
    let attn2_w = b.add_input("layer2_attn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let eps2b = b.add_input("layer2_norm2_eps", &[1]);
    let norm2b_w = b.add_input("layer2_norm2_weight", &[HIDDEN_DIM]);
    let moe2_w = b.add_input("layer2_moe_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Layer 1
    let normed1a = b.add_rms_norm(input, eps1a, 1, norm1a_w, &hidden_shape);
    let attn1_out = b.add_matmul(normed1a, attn1_w, false, None, &hidden_shape);
    let res1a = b.add_binary_add(input, attn1_out, &hidden_shape);
    let normed1b = b.add_rms_norm(res1a, eps1b, 1, norm1b_w, &hidden_shape);
    let moe1_out = b.add_matmul(normed1b, moe1_w, false, None, &hidden_shape);
    let res1b = b.add_binary_add(res1a, moe1_out, &hidden_shape);

    // Layer 2
    let normed2a = b.add_rms_norm(res1b, eps2a, 1, norm2a_w, &hidden_shape);
    let attn2_out = b.add_matmul(normed2a, attn2_w, false, None, &hidden_shape);
    let res2a = b.add_binary_add(res1b, attn2_out, &hidden_shape);
    let normed2b = b.add_rms_norm(res2a, eps2b, 1, norm2b_w, &hidden_shape);
    let moe2_out = b.add_matmul(normed2b, moe2_w, false, None, &hidden_shape);
    let output = b.add_binary_add(res2a, moe2_out, &hidden_shape);

    b.build(output).expect("valid two_layer_residual sub-graph")
}

pub(super) fn two_layer_residual_bindings() -> Vec<TensorParamBinding> {
    let eps_tensor = || ArrayD::from_elem(IxDyn(&[1]), RMS_EPS);
    let norm_w = || ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM]), ones_vec(HIDDEN_DIM)).unwrap();
    let proj_w = |mag: f32| {
        ArrayD::from_shape_vec(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            deterministic_weights(HIDDEN_DIM * HIDDEN_DIM, mag),
        )
        .unwrap()
    };
    vec![
        TensorParamBinding::Variable,
        // Layer 1
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(norm_w()),
        TensorParamBinding::ConstantTensor(proj_w(WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(norm_w()),
        TensorParamBinding::ConstantTensor(proj_w(WEIGHT_MAG * 0.5)),
        // Layer 2
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(norm_w()),
        TensorParamBinding::ConstantTensor(proj_w(WEIGHT_MAG * 0.8)),
        TensorParamBinding::ConstantTensor(eps_tensor()),
        TensorParamBinding::ConstantTensor(norm_w()),
        TensorParamBinding::ConstantTensor(proj_w(WEIGHT_MAG * 0.3)),
    ]
}

// 18. Output Pipeline (final norm → lm_head → softmax)
//
// Models the final output pipeline for next-token prediction:
//   input [SEQ_LEN, HIDDEN_DIM] → rms_norm → matmul(lm_weight) → softmax
pub(super) fn build_output_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gptoss_output_pipeline");
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let vocab_size = 32;
    let logit_shape = [SEQ_LEN, vocab_size];

    let input = b.add_input("hidden", &hidden_shape);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let lm_weight = b.add_input("lm_weight", &[HIDDEN_DIM, vocab_size]);

    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &hidden_shape);
    let logits = b.add_matmul(normed, lm_weight, false, None, &logit_shape);
    let probs = b.add_softmax(logits, 1, &logit_shape);

    b.build(probs).expect("valid output_pipeline sub-graph")
}

pub(super) fn output_pipeline_bindings() -> Vec<TensorParamBinding> {
    let vocab_size = 32;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), RMS_EPS)),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM]), ones_vec(HIDDEN_DIM)).unwrap(),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(
                IxDyn(&[HIDDEN_DIM, vocab_size]),
                deterministic_weights(HIDDEN_DIM * vocab_size, WEIGHT_MAG),
            )
            .unwrap(),
        ),
    ]
}
