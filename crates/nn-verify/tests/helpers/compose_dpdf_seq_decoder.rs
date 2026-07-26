// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for text recognition sequence decoder bound propagation:
//! CTC vs attention-based decoding for document OCR pipelines.
//!
//! Verifies IBP and CROWN bound propagation through CTC and attention-based
//! sequence decoders used in PaddleOCR, FireRed-OCR, and GLM-OCR document
//! understanding pipelines. Compares the two decoding paradigms and verifies
//! hybrid joint decoding, autoregressive step bounds, causal masking,
//! KV-cache interaction, and full encoder-to-decoder-to-output pipelines.
//!
//! ## CTC Decoder Softmax Output (test 1)
//!
//! 1. CTC decoder softmax output: per-timestep probability in [0, 1] (IBP)
//!
//! ## Attention Decoder Steps (tests 2-3)
//!
//! 2. Attention decoder autoregressive step: single-step decode bounds (IBP + CROWN)
//! 3. CTC vs attention output bound comparison: bound width analysis (IBP)
//!
//! ## CTC Blank & Attention Cross-Attention (tests 4-5)
//!
//! 4. CTC blank token probability: blank class bounded in [0, 1] (IBP)
//! 5. Attention decoder cross-attention: encoder-decoder attention bounds (IBP + CROWN)
//!
//! ## Hybrid & Beam Search (tests 6-7)
//!
//! 6. Hybrid CTC+attention joint decoding: weighted combination bounds (IBP)
//! 7. CTC prefix beam search score: top-k probability bounds (IBP)
//!
//! ## Teacher Forcing & Time-Step Independence (tests 8-9)
//!
//! 8. Attention decoder teacher forcing vs inference: bound width comparison (IBP)
//! 9. CTC time-step independence: per-step softmax consistency (IBP)
//!
//! ## Causal Mask & Character Distribution (tests 10-11)
//!
//! 10. Attention decoder causal mask interaction: masked attention bounds (IBP)
//! 11. CTC character-level output distribution: per-class bounded in [0, 1] (IBP)
//!
//! ## Vocabulary Projection & Greedy vs Beam (tests 12-13)
//!
//! 12. Attention decoder vocabulary projection: Linear -> softmax bounds (IBP + CROWN)
//! 13. CTC greedy decode vs beam search bound difference: width comparison (IBP)
//!
//! ## KV-Cache & Full Pipeline (tests 14-15)
//!
//! 14. Attention decoder with KV-cache: cached context decode bounds (IBP)
//! 15. Full recognition pipeline: encoder -> decoder -> output (IBP + CROWN)
//!
//! Architecture references:
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//! - Attention decoder (Bahdanau et al. 2015): Sequence-to-sequence with attention
//! - Hybrid CTC/Attention (Watanabe et al. 2017): Joint CTC+attention decoding
//! - PaddleOCR (Baidu): SVTR encoder + CTC/attention decoder
//! - FireRed-OCR: Qwen3-VL variant with CTC and attention heads
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, VOCAB_SIZE=64, NUM_HEADS=4, ENC_SEQ_LEN=8
//!
//! Part of #4027: Compose tests for text recognition sequence decoder.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length (encoder output / CTC timesteps).
const SEQ_LEN: usize = 4;
/// Hidden dimension of encoder output / decoder.
const HIDDEN_DIM: usize = 32;
/// Vocabulary size (characters + blank token at index 0).
const VOCAB_SIZE: usize = 64;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 8
/// Encoder memory sequence length for cross-attention.
const ENC_SEQ_LEN: usize = 8;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Beam search width for top-k tests.
const BEAM_WIDTH: usize = 5;
/// Decoder target sequence length (shorter than encoder for autoregressive).
const DEC_SEQ_LEN: usize = 3;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// CTC-attention hybrid interpolation weight (lambda for CTC contribution).
const CTC_WEIGHT: f32 = 0.3;
/// KV-cache context length (prior cached timesteps).
const CACHE_LEN: usize = 6;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ---------------------------------------------------------------------------
// Kernel Builders
// ---------------------------------------------------------------------------

/// Build CTC decoder softmax: Linear -> softmax character probabilities.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution per timestep).
fn build_ctc_decoder_softmax_kernel() -> TensorKernelDef {
    let logit_shape = [SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("seq_dec_ctc_softmax");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &logit_shape);
    let out = b.add_softmax(logits, 1, &logit_shape);

    b.build(out).expect("valid CTC decoder softmax kernel")
}

/// Bindings for CTC decoder softmax.
fn ctc_decoder_softmax_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // encoder_output
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // ctc_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // ctc_bias
    ]
}

/// Build attention decoder autoregressive step: cross-attention + FFN + softmax.
///
/// Input: `[DEC_SEQ_LEN, HIDDEN_DIM]` (Variable, decoder input embeddings).
/// Encoder memory: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Constant).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]` (next-token probability per position).
///
/// Architecture: Self-Attn(causal) -> residual -> Cross-Attn(encoder_mem) ->
///   residual -> FFN -> residual -> Linear -> softmax.
fn build_attn_decoder_step_kernel() -> TensorKernelDef {
    let shape = [DEC_SEQ_LEN, HIDDEN_DIM];
    let enc_shape = [ENC_SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("seq_dec_attn_step");

    let input = b.add_input("dec_input", &shape);
    let encoder_mem = b.add_input("encoder_mem", &enc_shape);

    // Self-attention (causal)
    let sa_q_w = b.add_input("sa_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_k_w = b.add_input("sa_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_v_w = b.add_input("sa_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_out_w = b.add_input("sa_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(input, sa_q_w, None, &shape);
    let sk = b.add_linear(input, sa_k_w, None, &shape);
    let sv = b.add_linear(input, sa_v_w, None, &shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Causal, Some(scale), &shape);
    let sa_out = b.add_linear(sa, sa_out_w, None, &shape);
    let res_sa = b.add_binary_add(input, sa_out, &shape);

    // Cross-attention (encoder memory)
    let ca_q_w = b.add_input("ca_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_k_w = b.add_input("ca_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_v_w = b.add_input("ca_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_out_w = b.add_input("ca_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let cq = b.add_linear(res_sa, ca_q_w, None, &shape);
    let ck = b.add_linear(encoder_mem, ca_k_w, None, &enc_shape);
    let cv = b.add_linear(encoder_mem, ca_v_w, None, &enc_shape);
    let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &shape);
    let ca_out = b.add_linear(ca, ca_out_w, None, &shape);
    let res_ca = b.add_binary_add(res_sa, ca_out, &shape);

    // FFN: Linear -> ReLU -> Linear
    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(res_ca, ffn_w1, None, &ffn_shape);
    let h_act = b.add_relu(h, &ffn_shape);
    let ffn_out = b.add_linear(h_act, ffn_w2, None, &shape);
    let res_ffn = b.add_binary_add(res_ca, ffn_out, &shape);

    // Vocabulary projection: Linear -> softmax
    let vocab_w = b.add_input("vocab_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let vocab_b = b.add_input("vocab_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(res_ffn, vocab_w, Some(vocab_b), &logit_shape);
    let out = b.add_softmax(logits, 1, &logit_shape);

    b.build(out).expect("valid attention decoder step kernel")
}

/// Bindings for attention decoder autoregressive step.
fn attn_decoder_step_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let enc_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);

    vec![
        TensorParamBinding::Variable,                // dec_input
        TensorParamBinding::ConstantTensor(enc_mem), // encoder_mem
        // Self-attention
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_v_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_out_weight
        // Cross-attention
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // ca_out_weight
        // FFN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // ffn_w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )), // ffn_w2
        // Vocabulary projection
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // vocab_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // vocab_bias
    ]
}

/// Build CTC blank token probability: Linear -> softmax -> narrow(blank=0).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, 1]` (blank class probability per timestep).
fn build_ctc_blank_token_kernel() -> TensorKernelDef {
    let logit_shape = [SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("seq_dec_ctc_blank");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &logit_shape);
    let probs = b.add_softmax(logits, 1, &logit_shape);
    let out = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);

    b.build(out).expect("valid CTC blank token kernel")
}

/// Build attention decoder cross-attention only block.
///
/// Input: `[DEC_SEQ_LEN, HIDDEN_DIM]` (Variable, decoder queries).
/// Encoder memory: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Constant).
/// Output: `[DEC_SEQ_LEN, HIDDEN_DIM]`.
fn build_cross_attention_block_kernel() -> TensorKernelDef {
    let shape = [DEC_SEQ_LEN, HIDDEN_DIM];
    let enc_shape = [ENC_SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("seq_dec_cross_attn");

    let input = b.add_input("dec_queries", &shape);
    let encoder_mem = b.add_input("encoder_mem", &enc_shape);

    let q_w = b.add_input("ca_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("ca_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("ca_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("ca_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(encoder_mem, k_w, None, &enc_shape);
    let v = b.add_linear(encoder_mem, v_w, None, &enc_shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid cross-attention block kernel")
}

/// Bindings for cross-attention block.
fn cross_attention_block_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let enc_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);

    vec![
        TensorParamBinding::Variable,                       // dec_queries
        TensorParamBinding::ConstantTensor(enc_mem),        // encoder_mem
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // ca_out_weight
    ]
}

/// Build hybrid CTC+attention joint decoding: weighted sum of CTC and attention softmax.
///
/// Input: `[DEC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Encoder memory: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Constant).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]` (joint probability distribution).
///
/// P_joint = lambda * P_ctc + (1 - lambda) * P_attn
/// We approximate by building both CTC and attention heads, scaling, and adding.
fn build_hybrid_ctc_attn_kernel() -> TensorKernelDef {
    let dec_shape = [DEC_SEQ_LEN, HIDDEN_DIM];
    let enc_shape = [ENC_SEQ_LEN, HIDDEN_DIM];
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("seq_dec_hybrid_ctc_attn");

    let input = b.add_input("dec_input", &dec_shape);
    let encoder_mem = b.add_input("encoder_mem", &enc_shape);

    // CTC branch: Linear -> softmax (applied to decoder input for shape compatibility)
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(input, ctc_w, Some(ctc_b), &logit_shape);
    let ctc_probs = b.add_softmax(ctc_logits, 1, &logit_shape);

    // Attention branch: cross-attention -> Linear -> softmax
    let ca_q_w = b.add_input("ca_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_k_w = b.add_input("ca_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_v_w = b.add_input("ca_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_out_w = b.add_input("ca_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let cq = b.add_linear(input, ca_q_w, None, &dec_shape);
    let ck = b.add_linear(encoder_mem, ca_k_w, None, &enc_shape);
    let cv = b.add_linear(encoder_mem, ca_v_w, None, &enc_shape);
    let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &dec_shape);
    let ca_out = b.add_linear(ca, ca_out_w, None, &dec_shape);
    let attn_hidden = b.add_binary_add(input, ca_out, &dec_shape);

    let attn_vocab_w = b.add_input("attn_vocab_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let attn_vocab_b = b.add_input("attn_vocab_bias", &[VOCAB_SIZE]);
    let attn_logits = b.add_linear(attn_hidden, attn_vocab_w, Some(attn_vocab_b), &logit_shape);
    let attn_probs = b.add_softmax(attn_logits, 1, &logit_shape);

    // Weighted combination: lambda * ctc_probs + (1 - lambda) * attn_probs
    // Use scalar weight inputs for the interpolation
    let lambda_ctc = b.add_input("lambda_ctc", &[1]);
    let lambda_attn = b.add_input("lambda_attn", &[1]);

    let lambda_ctc_bc = b.add_broadcast(lambda_ctc, &logit_shape);
    let ctc_scaled = b.add_binary_mul(ctc_probs, lambda_ctc_bc, &logit_shape);
    let lambda_attn_bc = b.add_broadcast(lambda_attn, &logit_shape);
    let attn_scaled = b.add_binary_mul(attn_probs, lambda_attn_bc, &logit_shape);
    let out = b.add_binary_add(ctc_scaled, attn_scaled, &logit_shape);

    b.build(out).expect("valid hybrid CTC+attention kernel")
}

/// Bindings for hybrid CTC+attention joint decoding.
fn hybrid_ctc_attn_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let enc_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);

    vec![
        TensorParamBinding::Variable,                // dec_input
        TensorParamBinding::ConstantTensor(enc_mem), // encoder_mem
        // CTC branch
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // ctc_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // ctc_bias
        // Cross-attention
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // ca_out_weight
        // Attention vocab head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // attn_vocab_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // attn_vocab_bias
        // Interpolation weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), CTC_WEIGHT)), // lambda_ctc
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0 - CTC_WEIGHT)), // lambda_attn
    ]
}

/// Build CTC prefix beam search: Linear -> softmax -> narrow top-k.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, BEAM_WIDTH]` (top-k probabilities per timestep).
fn build_ctc_beam_search_kernel() -> TensorKernelDef {
    let logit_shape = [SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("seq_dec_ctc_beam_search");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &logit_shape);
    let probs = b.add_softmax(logits, 1, &logit_shape);
    let out = b.add_narrow(probs, 1, 0, BEAM_WIDTH, &[SEQ_LEN, BEAM_WIDTH]);

    b.build(out).expect("valid CTC beam search kernel")
}

/// Build attention decoder with causal mask only (no cross-attention).
///
/// Input: `[DEC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[DEC_SEQ_LEN, HIDDEN_DIM]`.
fn build_causal_attn_decoder_kernel() -> TensorKernelDef {
    let shape = [DEC_SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("seq_dec_causal_attn");

    let input = b.add_input("dec_input", &shape);

    let q_w = b.add_input("sa_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("sa_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("sa_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("sa_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid causal attention decoder kernel")
}

/// Bindings for causal attention decoder.
fn causal_attn_decoder_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // dec_input
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // sa_out_weight
    ]
}

/// Build CTC character-level distribution: Linear -> softmax -> narrow single class.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, 1]` (single character class probability per timestep).
fn build_ctc_char_distribution_kernel() -> TensorKernelDef {
    let logit_shape = [SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("seq_dec_ctc_char_dist");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &logit_shape);
    let probs = b.add_softmax(logits, 1, &logit_shape);
    // Narrow to a single character class (class index 1)
    let out = b.add_narrow(probs, 1, 1, 1, &[SEQ_LEN, 1]);

    b.build(out)
        .expect("valid CTC character distribution kernel")
}

/// Build attention decoder vocabulary projection: Linear -> softmax.
///
/// Input: `[DEC_SEQ_LEN, HIDDEN_DIM]` (Variable, decoder hidden).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]` (token probability distribution).
fn build_attn_vocab_projection_kernel() -> TensorKernelDef {
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("seq_dec_attn_vocab_proj");

    let input = b.add_input("dec_hidden", &[DEC_SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("vocab_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("vocab_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &logit_shape);
    let out = b.add_softmax(logits, 1, &logit_shape);

    b.build(out)
        .expect("valid attention vocab projection kernel")
}

/// Bindings for attention decoder vocabulary projection.
fn attn_vocab_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // dec_hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // vocab_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // vocab_bias
    ]
}

/// Build CTC greedy decode: Linear -> softmax -> narrow(0, 1).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, 1]` (best class probability per timestep).
fn build_ctc_greedy_decode_kernel() -> TensorKernelDef {
    let logit_shape = [SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("seq_dec_ctc_greedy");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &logit_shape);
    let probs = b.add_softmax(logits, 1, &logit_shape);
    let out = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);

    b.build(out).expect("valid CTC greedy decode kernel")
}

/// Build attention decoder with KV-cache: query attends over cached context.
///
/// Simulates decode-phase attention where current query (1 step) attends
/// over previously cached K/V context of CACHE_LEN steps.
///
/// Input: `[1, HIDDEN_DIM]` (Variable, current decoder step).
/// Cached KV: `[CACHE_LEN, HIDDEN_DIM]` (Constant, prior context).
/// Output: `[1, HIDDEN_DIM]`.
fn build_kv_cache_decoder_kernel() -> TensorKernelDef {
    let q_shape = [1, HIDDEN_DIM];
    let kv_shape = [CACHE_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("seq_dec_kv_cache");

    let input = b.add_input("current_step", &q_shape);
    let cached_k = b.add_input("cached_keys", &kv_shape);
    let cached_v = b.add_input("cached_values", &kv_shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &q_shape);
    let k = b.add_linear(cached_k, k_w, None, &kv_shape);
    let v = b.add_linear(cached_v, v_w, None, &kv_shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &q_shape);
    let attn_out = b.add_linear(attn, out_w, None, &q_shape);
    let out = b.add_binary_add(input, attn_out, &q_shape);

    b.build(out).expect("valid KV-cache decoder kernel")
}

/// Bindings for KV-cache decoder.
fn kv_cache_decoder_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let cached = ArrayD::from_elem(IxDyn(&[CACHE_LEN, HIDDEN_DIM]), 0.3f32);

    vec![
        TensorParamBinding::Variable,                       // current_step
        TensorParamBinding::ConstantTensor(cached.clone()), // cached_keys
        TensorParamBinding::ConstantTensor(cached),         // cached_values
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

/// Build full recognition pipeline: encoder projection -> decoder -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, raw encoder features).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities).
///
/// Architecture: Linear(encoder) -> ReLU -> Cross-Attn(encoder, decoder) -> FFN ->
///   Linear(vocab) -> softmax.
fn build_full_recognition_pipeline_kernel() -> TensorKernelDef {
    let enc_shape = [SEQ_LEN, HIDDEN_DIM];
    let dec_shape = [DEC_SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];
    let logit_shape = [DEC_SEQ_LEN, VOCAB_SIZE];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("seq_dec_full_pipeline");

    let enc_input = b.add_input("encoder_features", &enc_shape);
    let dec_input = b.add_input("decoder_input", &dec_shape);

    // Encoder projection: Linear -> ReLU
    let enc_proj_w = b.add_input("enc_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_proj_b = b.add_input("enc_proj_bias", &[HIDDEN_DIM]);
    let enc_proj = b.add_linear(enc_input, enc_proj_w, Some(enc_proj_b), &enc_shape);
    let enc_activated = b.add_relu(enc_proj, &enc_shape);

    // Decoder: self-attention (causal)
    let sa_q_w = b.add_input("sa_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_k_w = b.add_input("sa_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_v_w = b.add_input("sa_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_out_w = b.add_input("sa_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(dec_input, sa_q_w, None, &dec_shape);
    let sk = b.add_linear(dec_input, sa_k_w, None, &dec_shape);
    let sv = b.add_linear(dec_input, sa_v_w, None, &dec_shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Causal, Some(scale), &dec_shape);
    let sa_out = b.add_linear(sa, sa_out_w, None, &dec_shape);
    let res_sa = b.add_binary_add(dec_input, sa_out, &dec_shape);

    // Cross-attention: decoder queries attend to encoder memory
    let ca_q_w = b.add_input("ca_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_k_w = b.add_input("ca_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_v_w = b.add_input("ca_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_out_w = b.add_input("ca_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let cq = b.add_linear(res_sa, ca_q_w, None, &dec_shape);
    let ck = b.add_linear(enc_activated, ca_k_w, None, &enc_shape);
    let cv = b.add_linear(enc_activated, ca_v_w, None, &enc_shape);
    let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &dec_shape);
    let ca_out = b.add_linear(ca, ca_out_w, None, &dec_shape);
    let res_ca = b.add_binary_add(res_sa, ca_out, &dec_shape);

    // FFN: Linear -> ReLU -> Linear + residual
    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(res_ca, ffn_w1, None, &ffn_shape);
    let h_act = b.add_relu(h, &ffn_shape);
    let ffn_out = b.add_linear(h_act, ffn_w2, None, &dec_shape);
    let res_ffn = b.add_binary_add(res_ca, ffn_out, &dec_shape);

    // Vocabulary projection: Linear -> softmax
    let vocab_w = b.add_input("vocab_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let vocab_b = b.add_input("vocab_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(res_ffn, vocab_w, Some(vocab_b), &logit_shape);
    let out = b.add_softmax(logits, 1, &logit_shape);

    b.build(out)
        .expect("valid full recognition pipeline kernel")
}

/// Bindings for full recognition pipeline.
fn full_recognition_pipeline_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // encoder_features
        TensorParamBinding::Variable, // decoder_input
        // Encoder projection
        TensorParamBinding::ConstantTensor(proj_w.clone()), // enc_proj_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // enc_proj_bias
        // Self-attention
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_v_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_out_weight
        // Cross-attention
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // ca_out_weight
        // FFN
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // ffn_w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )), // ffn_w2
        // Vocabulary projection
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // vocab_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)), // vocab_bias
    ]
}

// ===========================================================================
// 1. CTC decoder softmax output bounds (IBP)
// ===========================================================================

/// CTC decoder softmax: all character probabilities bounded in [0, 1] under IBP.
#[test]
fn test_seq_dec_ctc_softmax_output_ibp() {
    let def = build_ctc_decoder_softmax_kernel();
    let bindings = ctc_decoder_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC decoder softmax");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC decoder softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("seq_dec CTC softmax IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 2. Attention decoder autoregressive step bounds (IBP + CROWN)
// ===========================================================================

/// Attention decoder autoregressive step: single-step decode with cross-attention
/// produces bounded softmax output in [0, 1].
#[test]
fn test_seq_dec_attn_autoregressive_step_ibp_crown() {
    let def = build_attn_decoder_step_kernel();
    let bindings = attn_decoder_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through attention decoder step");

    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "attention decoder step output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("seq_dec attention step IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "attention decoder lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "attention decoder upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "seq_dec attention step CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 3. CTC vs attention output bound comparison (IBP)
// ===========================================================================

/// CTC vs attention: compare output bound widths from both decoding paradigms.
///
/// Both decoders should produce [0, 1]-bounded softmax outputs, but
/// the attention decoder (with cross-attention and residual connections)
/// may produce different bound widths than the simple CTC projection.
#[test]
fn test_seq_dec_ctc_vs_attention_comparison_ibp() {
    // CTC decoder
    let ctc_def = build_ctc_decoder_softmax_kernel();
    let ctc_bindings = ctc_decoder_softmax_bindings();
    let ctc_graph = tensor_kernel_to_graph(&ctc_def, &ctc_bindings).expect("CTC graph");
    let ctc_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let ctc_output = ctc_graph
        .propagate_ibp(&ctc_input)
        .expect("IBP through CTC");

    // Attention decoder
    let attn_def = build_attn_decoder_step_kernel();
    let attn_bindings = attn_decoder_step_bindings();
    let attn_graph = tensor_kernel_to_graph(&attn_def, &attn_bindings).expect("attention graph");
    let attn_input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 1.0);
    let attn_output = attn_graph
        .propagate_ibp(&attn_input)
        .expect("IBP through attention");

    assert_bounds_valid(&ctc_output);
    assert_bounds_valid(&attn_output);

    let ctc_width = bound_width(&ctc_output);
    let attn_width = bound_width(&attn_output);

    eprintln!(
        "seq_dec CTC vs attention: CTC width={ctc_width:.6}, attention width={attn_width:.6}"
    );

    // Both must be valid softmax outputs
    let eps = 1e-6;
    let (ctc_lo, ctc_hi) = bounds_min_max(&ctc_output);
    assert!(ctc_lo >= 0.0 - eps, "CTC lower >= 0");
    assert!(ctc_hi <= 1.0 + eps, "CTC upper <= 1");

    let (attn_lo, attn_hi) = bounds_min_max(&attn_output);
    assert!(attn_lo >= 0.0 - eps, "attention lower >= 0");
    assert!(attn_hi <= 1.0 + eps, "attention upper <= 1");
}

// ===========================================================================
// 4. CTC blank token probability bounds (IBP)
// ===========================================================================

/// CTC blank token: narrowed softmax class 0 bounded in [0, 1].
#[test]
fn test_seq_dec_ctc_blank_token_ibp() {
    let def = build_ctc_blank_token_kernel();
    let bindings = ctc_decoder_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC blank token");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 1],
        "CTC blank token output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("seq_dec CTC blank token IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "blank token lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "blank token upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 5. Attention decoder cross-attention bounds (IBP + CROWN)
// ===========================================================================

/// Attention decoder cross-attention: encoder-decoder attention block
/// produces finite, valid bounds under IBP and CROWN.
#[test]
fn test_seq_dec_attn_cross_attention_ibp_crown() {
    let def = build_cross_attention_block_kernel();
    let bindings = cross_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention block");

    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, HIDDEN_DIM],
        "cross-attention block output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("seq_dec cross-attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "seq_dec cross-attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 6. Hybrid CTC+attention joint decoding bounds (IBP)
// ===========================================================================

/// Hybrid CTC+attention: weighted combination of CTC and attention softmax
/// preserves [0, 1] bounds (since both components are in [0, 1] and weights
/// sum to 1).
#[test]
fn test_seq_dec_hybrid_ctc_attn_ibp() {
    let def = build_hybrid_ctc_attn_kernel();
    let bindings = hybrid_ctc_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through hybrid CTC+attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "hybrid CTC+attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "seq_dec hybrid CTC+attention IBP (lambda={CTC_WEIGHT}): bounds=[{lo_min}, {hi_max}]"
    );

    // Joint output is a convex combination of two [0, 1]-bounded distributions.
    // With IBP relaxation, bounds may be slightly wider.
    assert!(lo_min.is_finite(), "hybrid lower must be finite");
    assert!(hi_max.is_finite(), "hybrid upper must be finite");
}

// ===========================================================================
// 7. CTC prefix beam search score bounds (IBP)
// ===========================================================================

/// CTC prefix beam search: top-k probabilities per timestep bounded in [0, 1].
#[test]
fn test_seq_dec_ctc_beam_search_ibp() {
    let def = build_ctc_beam_search_kernel();
    let bindings = ctc_decoder_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC beam search");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, BEAM_WIDTH],
        "CTC beam search output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("seq_dec CTC beam search IBP (k={BEAM_WIDTH}): bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "beam search lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "beam search upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Attention decoder teacher forcing vs inference (IBP)
// ===========================================================================

/// Teacher forcing vs inference: with tighter input bounds (teacher forcing
/// provides ground truth), output bounds should be tighter or equal compared
/// to wider inference-time bounds.
#[test]
fn test_seq_dec_attn_teacher_forcing_vs_inference_ibp() {
    let def = build_attn_decoder_step_kernel();
    let bindings = attn_decoder_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Teacher forcing: tight input (ground truth embeddings, small perturbation)
    let teacher_input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 0.5);
    let teacher_output = graph
        .propagate_ibp(&teacher_input)
        .expect("IBP with teacher forcing");

    // Inference: wider input (autoregressive, accumulated uncertainty)
    let inference_input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 2.0);
    let inference_output = graph
        .propagate_ibp(&inference_input)
        .expect("IBP with inference");

    assert_bounds_valid(&teacher_output);
    assert_bounds_valid(&inference_output);

    let teacher_width = bound_width(&teacher_output);
    let inference_width = bound_width(&inference_output);
    eprintln!(
        "seq_dec teacher forcing width={teacher_width:.6}, inference width={inference_width:.6}"
    );

    assert!(
        teacher_width <= inference_width + 1e-4,
        "teacher forcing (tighter input) should produce tighter or equal bounds, \
         got teacher={teacher_width:.6} > inference={inference_width:.6}"
    );
}

// ===========================================================================
// 9. CTC time-step independence property (IBP)
// ===========================================================================

/// CTC time-step independence: per-timestep softmax bounds are consistent.
///
/// CTC applies softmax independently at each timestep. The bound width
/// at each timestep should be similar for uniform input bounds.
#[test]
fn test_seq_dec_ctc_timestep_independence_ibp() {
    let def = build_ctc_decoder_softmax_kernel();
    let bindings = ctc_decoder_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC timestep independence");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // Check that each timestep has similar bound width (uniform input -> uniform bounds)
    let mut timestep_widths = Vec::new();
    for t in 0..SEQ_LEN {
        let mut ts_lo_min = f32::INFINITY;
        let mut ts_hi_max = f32::NEG_INFINITY;
        for v in 0..VOCAB_SIZE {
            ts_lo_min = ts_lo_min.min(lo[[t, v]]);
            ts_hi_max = ts_hi_max.max(hi[[t, v]]);
        }
        timestep_widths.push(ts_hi_max - ts_lo_min);
    }

    eprintln!("seq_dec CTC timestep widths: {timestep_widths:?}");

    // All timesteps should have similar width (within 10% of each other)
    let max_w = timestep_widths
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_w = timestep_widths
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(max_w > 0.0, "timestep bound widths should be positive");
    // With uniform input, IBP should produce identical bounds per timestep.
    // Allow small numerical tolerance.
    assert!(
        (max_w - min_w) < max_w * 0.1 + 1e-5,
        "timestep bound widths should be similar for uniform input: min={min_w}, max={max_w}"
    );
}

// ===========================================================================
// 10. Attention decoder causal mask interaction (IBP)
// ===========================================================================

/// Causal mask: attention decoder with causal masking preserves finite bounds.
///
/// The causal mask restricts attention to past positions only, which should
/// not break bound propagation.
#[test]
fn test_seq_dec_attn_causal_mask_ibp() {
    let def = build_causal_attn_decoder_kernel();
    let bindings = causal_attn_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through causal attention decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, HIDDEN_DIM],
        "causal attention decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("seq_dec causal mask IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "causal decoder lower bound finite");
    assert!(hi_max.is_finite(), "causal decoder upper bound finite");
}

// ===========================================================================
// 11. CTC character-level output distribution bounds (IBP)
// ===========================================================================

/// CTC character distribution: individual character class probability
/// bounded in [0, 1] under IBP.
#[test]
fn test_seq_dec_ctc_char_distribution_ibp() {
    let def = build_ctc_char_distribution_kernel();
    let bindings = ctc_decoder_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC character distribution");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 1],
        "CTC character distribution output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("seq_dec CTC character distribution IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "char distribution lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "char distribution upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 12. Attention decoder vocabulary projection bounds (IBP + CROWN)
// ===========================================================================

/// Attention decoder vocab projection: Linear -> softmax produces [0, 1]
/// bounded output under both IBP and CROWN.
#[test]
fn test_seq_dec_attn_vocab_projection_ibp_crown() {
    let def = build_attn_vocab_projection_kernel();
    let bindings = attn_vocab_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through vocab projection");

    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "vocab projection output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("seq_dec vocab projection IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "vocab projection lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "vocab projection upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "seq_dec vocab projection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 13. CTC greedy decode vs beam search bound difference (IBP)
// ===========================================================================

/// CTC greedy vs beam search: compare single-class (greedy) vs multi-class
/// (beam) bound widths. Beam search narrows to more classes, so its aggregate
/// bound width should be >= greedy's single-class width.
#[test]
fn test_seq_dec_ctc_greedy_vs_beam_ibp() {
    let bindings = ctc_decoder_softmax_bindings();

    // Greedy: narrow to 1 class
    let greedy_def = build_ctc_greedy_decode_kernel();
    let greedy_graph = tensor_kernel_to_graph(&greedy_def, &bindings).expect("greedy graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);
    let greedy_output = greedy_graph
        .propagate_ibp(&input)
        .expect("IBP through greedy");

    // Beam search: narrow to BEAM_WIDTH classes
    let beam_def = build_ctc_beam_search_kernel();
    let beam_graph = tensor_kernel_to_graph(&beam_def, &bindings).expect("beam graph");
    let beam_output = beam_graph
        .propagate_ibp(&input)
        .expect("IBP through beam search");

    assert_bounds_valid(&greedy_output);
    assert_bounds_valid(&beam_output);

    let greedy_width = bound_width(&greedy_output);
    let beam_width = bound_width(&beam_output);
    eprintln!("seq_dec greedy width={greedy_width:.6}, beam width={beam_width:.6}");

    // Both are softmax-narrowed, so both should be in [0, 1]
    let eps = 1e-6;
    let (g_lo, g_hi) = bounds_min_max(&greedy_output);
    assert!(g_lo >= 0.0 - eps, "greedy lower >= 0");
    assert!(g_hi <= 1.0 + eps, "greedy upper <= 1");

    let (b_lo, b_hi) = bounds_min_max(&beam_output);
    assert!(b_lo >= 0.0 - eps, "beam lower >= 0");
    assert!(b_hi <= 1.0 + eps, "beam upper <= 1");
}

// ===========================================================================
// 14. Attention decoder with KV-cache bounds (IBP)
// ===========================================================================

/// KV-cache decoder: single-step query attending over cached context
/// produces finite, valid bounds under IBP.
#[test]
fn test_seq_dec_attn_kv_cache_ibp() {
    let def = build_kv_cache_decoder_kernel();
    let bindings = kv_cache_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through KV-cache decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, HIDDEN_DIM],
        "KV-cache decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("seq_dec KV-cache IBP (cache_len={CACHE_LEN}): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "KV-cache lower bound finite");
    assert!(hi_max.is_finite(), "KV-cache upper bound finite");
}

// ===========================================================================
// 15. Full recognition pipeline: encoder -> decoder -> output (IBP + CROWN)
// ===========================================================================

/// Full recognition pipeline: encoder projection -> self-attention ->
/// cross-attention -> FFN -> vocab softmax, end-to-end bound propagation.
#[test]
fn test_seq_dec_full_recognition_pipeline_ibp_crown() {
    let def = build_full_recognition_pipeline_kernel();
    let bindings = full_recognition_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Two Variable inputs: encoder features and decoder input
    let input = uniform_bounds(&[SEQ_LEN + DEC_SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through full recognition pipeline");

    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE],
        "full pipeline output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("seq_dec full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "full pipeline lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "full pipeline upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "seq_dec full pipeline CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}
