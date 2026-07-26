// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional model-level verification configs: HTDemucs, Whisper, Qwen3, Kokoro.
//!
//! Each builder creates a structurally faithful but small-scale version of the
//! real model architecture, using tiny dimensions for NY tractability.
//!
//! Part of #1696 AC7: All 5 models have entries in nn_verify_status.json.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use super::ModelConfig;

/// Small weight magnitude for stable NY bounds.
const W: f32 = 0.001;

// =========================================================================
// HTDemucs — temporal encoder + cross-domain transformer + temporal decoder
// =========================================================================

/// Build simplified HTDemucs: encoder(Conv1d+GELU) → transformer(MHA) → decoder(ConvTranspose1d).
fn build_htdemucs() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("htdemucs_full");
    let (ch, t_in) = (4, 8);
    let enc_k = 4;

    // Variable: audio [ch, t_in]
    let audio = b.add_input("audio", &[ch, t_in]);

    // Encoder: Conv1d stride=2 → GELU
    let enc_w = b.add_input("enc_w", &[ch, ch, enc_k]);
    let t_enc = (t_in + 2 - enc_k) / 2 + 1; // padding=1, stride=2
    let enc_out = b.add_conv1d(audio, enc_w, None, 2, 1, &[ch, t_enc]);
    let enc_act = b.add_gelu(enc_out, &[ch, t_enc]);

    // Reshape for transformer: [ch, t_enc] → [t_enc, ch]
    let reshaped = b.add_reshape(enc_act, &[t_enc, ch]);

    // Transformer: LayerNorm → self-attention → residual
    let ln_w = b.add_input("ln_w", &[ch]);
    let ln_b = b.add_input("ln_b", &[ch]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(reshaped, eps, 1, ln_w, ln_b, &[t_enc, ch]);

    let qw = b.add_input("qw", &[ch, ch]);
    let kw = b.add_input("kw", &[ch, ch]);
    let vw = b.add_input("vw", &[ch, ch]);
    let ow = b.add_input("ow", &[ch, ch]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            2,
            AttentionMask::Standard,
            &[t_enc, ch],
        )
        .expect("htdemucs self-attention");
    let residual = b.add_binary_add(reshaped, attn, &[t_enc, ch]);

    // Reshape back: [t_enc, ch] → [ch, t_enc]
    let dec_in = b.add_reshape(residual, &[ch, t_enc]);

    // Decoder: ConvTranspose1d stride=2 back to [ch, t_out]
    let dec_w = b.add_input("dec_w", &[ch, ch, enc_k]);
    let t_out = (t_enc - 1) * 2 + enc_k - 2; // padding=1
    let output = b.add_conv_transpose_1d(dec_in, dec_w, None, 2, 1, 1, 1, 0, &[ch, t_out]);

    b.build(output).expect("valid htdemucs_full graph")
}

#[allow(clippy::vec_init_then_push)]
fn htdemucs_bindings() -> Vec<TensorParamBinding> {
    let ch = 4;
    let enc_k = 4;
    let mut v = Vec::new();
    v.push(TensorParamBinding::Variable); // audio
    v.push(ct(&[ch, ch, enc_k], W)); // enc_w
                                     // Transformer
    v.push(ct(&[ch], 1.0)); // ln_w
    v.push(ct(&[ch], 0.0)); // ln_b
    v.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    for _ in 0..4 {
        v.push(ct(&[ch, ch], W)); // qw, kw, vw, ow
    }
    // Decoder
    v.push(ct(&[ch, ch, enc_k], W)); // dec_w
    v
}

// =========================================================================
// Whisper — encoder stem (2 Conv1d) + decoder (MHA + cross-attn + FFN)
// =========================================================================

/// Build simplified Whisper: decoder with causal self-attn + cross-attn on constant encoder output.
fn build_whisper() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_full");
    let (d, seq, vocab) = (8, 4, 16);

    // Variable: token embedding [seq, d]
    let tok = b.add_input("tok_emb", &[seq, d]);

    // Positional embedding (constant, added to token embedding)
    let pos = b.add_input("pos_emb", &[seq, d]);
    let x = b.add_binary_add(tok, pos, &[seq, d]);

    // Encoder output (constant — cross-attention context)
    let enc_ctx = b.add_input("enc_ctx", &[seq, d]);

    // Decoder block: LN → causal self-attn → residual → + encoder context → LN → FFN → residual
    let eps = b.add_input("eps", &[1]);

    // Self-attention sub-block
    let ln1_w = b.add_input("ln1_w", &[d]);
    let ln1_b = b.add_input("ln1_b", &[d]);
    let n1 = b.add_layer_norm(x, eps, 1, ln1_w, ln1_b, &[seq, d]);
    let sq = b.add_input("sq", &[d, d]);
    let sk = b.add_input("sk", &[d, d]);
    let sv = b.add_input("sv", &[d, d]);
    let so = b.add_input("so", &[d, d]);
    let sa = b
        .add_multi_head_attention(n1, sq, sk, sv, so, 2, AttentionMask::Causal, &[seq, d])
        .expect("whisper self-attn");
    let r1 = b.add_binary_add(x, sa, &[seq, d]);

    // Cross-attention approximation: LN(decoder) + Linear(encoder_ctx) → residual.
    // True cross-attention projects Q from decoder and K/V from encoder, but the
    // builder's add_multi_head_attention is single-input. Instead, project the
    // constant encoder context through a linear layer and add to the residual
    // stream — structurally exercises the same layer types for NY.
    let ln2_w = b.add_input("ln2_w", &[d]);
    let ln2_b = b.add_input("ln2_b", &[d]);
    let n2 = b.add_layer_norm(r1, eps, 1, ln2_w, ln2_b, &[seq, d]);
    let cross_w = b.add_input("cross_w", &[d, d]);
    let cross_proj = b.add_linear(enc_ctx, cross_w, None, &[seq, d]);
    let cross_out = b.add_binary_add(n2, cross_proj, &[seq, d]);
    let r2 = b.add_binary_add(r1, cross_out, &[seq, d]);

    // FFN sub-block: LN → Linear → GELU → Linear
    let ln3_w = b.add_input("ln3_w", &[d]);
    let ln3_b = b.add_input("ln3_b", &[d]);
    let n3 = b.add_layer_norm(r2, eps, 1, ln3_w, ln3_b, &[seq, d]);
    let ffn1_w = b.add_input("ffn1_w", &[d * 2, d]);
    let h = b.add_linear(n3, ffn1_w, None, &[seq, d * 2]);
    let h_act = b.add_gelu(h, &[seq, d * 2]);
    let ffn2_w = b.add_input("ffn2_w", &[d, d * 2]);
    let ffn_out = b.add_linear(h_act, ffn2_w, None, &[seq, d]);
    let r3 = b.add_binary_add(r2, ffn_out, &[seq, d]);

    // Final LN + output projection
    let ln_f_w = b.add_input("ln_f_w", &[d]);
    let ln_f_b = b.add_input("ln_f_b", &[d]);
    let final_norm = b.add_layer_norm(r3, eps, 1, ln_f_w, ln_f_b, &[seq, d]);
    let lm_w = b.add_input("lm_w", &[d, vocab]);
    let logits = b.add_matmul(final_norm, lm_w, false, None, &[seq, vocab]);

    b.build(logits).expect("valid whisper_full graph")
}

#[allow(clippy::vec_init_then_push)]
fn whisper_bindings() -> Vec<TensorParamBinding> {
    let d = 8;
    let seq = 4;
    let vocab = 16;
    let mut v = Vec::new();
    v.push(TensorParamBinding::Variable); // tok_emb
    v.push(ct(&[seq, d], W)); // pos_emb
    v.push(ct(&[seq, d], W)); // enc_ctx
    v.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
                                                      // Self-attn sub-block: ln1_w, ln1_b, sq, sk, sv, so
    v.push(ct(&[d], 1.0));
    v.push(ct(&[d], 0.0));
    for _ in 0..4 {
        v.push(ct(&[d, d], W));
    }
    // Cross-attn approximation: ln2_w, ln2_b, cross_w
    v.push(ct(&[d], 1.0));
    v.push(ct(&[d], 0.0));
    v.push(ct(&[d, d], W));
    // FFN sub-block: ln3_w, ln3_b, ffn1_w, ffn2_w
    v.push(ct(&[d], 1.0));
    v.push(ct(&[d], 0.0));
    v.push(ct(&[d * 2, d], W));
    v.push(ct(&[d, d * 2], W));
    // Final: ln_f_w, ln_f_b, lm_w
    v.push(ct(&[d], 1.0));
    v.push(ct(&[d], 0.0));
    v.push(ct(&[d, vocab], W));
    v
}

// =========================================================================
// Qwen3 — decoder-only with RmsNorm + SwiGLU
// =========================================================================

/// Build simplified Qwen3: RmsNorm + causal MHA + SwiGLU MLP × 1 layer + lm_head.
fn build_qwen3() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_full");
    let (d, seq, vocab, inter) = (8, 4, 16, 16);

    let tok = b.add_input("tok_emb", &[seq, d]);
    let eps = b.add_input("eps", &[1]);

    // Decoder block: RmsNorm → causal MHA → residual → RmsNorm → SwiGLU → residual
    let ln1_w = b.add_input("ln1_w", &[d]);
    let n1 = b.add_rms_norm(tok, eps, 1, ln1_w, &[seq, d]);
    let qw = b.add_input("qw", &[d, d]);
    let kw = b.add_input("kw", &[d, d]);
    let vw = b.add_input("vw", &[d, d]);
    let ow = b.add_input("ow", &[d, d]);
    let sa = b
        .add_multi_head_attention(n1, qw, kw, vw, ow, 2, AttentionMask::Causal, &[seq, d])
        .expect("qwen3 self-attn");
    let r1 = b.add_binary_add(tok, sa, &[seq, d]);

    // SwiGLU MLP
    let ln2_w = b.add_input("ln2_w", &[d]);
    let n2 = b.add_rms_norm(r1, eps, 1, ln2_w, &[seq, d]);
    let gate_w = b.add_input("gate_w", &[inter, d]);
    let up_w = b.add_input("up_w", &[inter, d]);
    let down_w = b.add_input("down_w", &[d, inter]);
    let gate = b.add_linear(n2, gate_w, None, &[seq, inter]);
    let gate_sig = b.add_sigmoid(gate, &[seq, inter]);
    let gate_silu = b.add_binary_mul(gate, gate_sig, &[seq, inter]);
    let up = b.add_linear(n2, up_w, None, &[seq, inter]);
    let gated = b.add_binary_mul(gate_silu, up, &[seq, inter]);
    let mlp_out = b.add_linear(gated, down_w, None, &[seq, d]);
    let r2 = b.add_binary_add(r1, mlp_out, &[seq, d]);

    // Final RmsNorm + lm_head
    let ln_f_w = b.add_input("ln_f_w", &[d]);
    let normed = b.add_rms_norm(r2, eps, 1, ln_f_w, &[seq, d]);
    let lm_w = b.add_input("lm_w", &[d, vocab]);
    let logits = b.add_matmul(normed, lm_w, false, None, &[seq, vocab]);

    b.build(logits).expect("valid qwen3_full graph")
}

fn qwen3_bindings() -> Vec<TensorParamBinding> {
    let (d, seq, vocab, inter) = (8, 4, 16, 16);
    let _ = seq; // used only in shape declarations
    let mut v = Vec::new();
    v.push(TensorParamBinding::Variable); // tok_emb
    v.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
                                                      // Decoder block
    v.push(ct(&[d], 1.0)); // ln1_w
    for _ in 0..4 {
        v.push(ct(&[d, d], W)); // qw, kw, vw, ow
    }
    v.push(ct(&[d], 1.0)); // ln2_w
    v.push(ct(&[inter, d], W)); // gate_w
    v.push(ct(&[inter, d], W)); // up_w
    v.push(ct(&[d, inter], W)); // down_w
                                // Final
    v.push(ct(&[d], 1.0)); // ln_f_w
    v.push(ct(&[d, vocab], W)); // lm_w
    v
}

// =========================================================================
// Kokoro — ISTFTNet vocoder: Conv1d + ConvTranspose1d + Snake + Exp
// =========================================================================

/// Build simplified Kokoro decoder: conv_pre → upsample → InstanceNorm+Snake+Conv1d → conv_post → exp.
fn build_kokoro() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("kokoro_decoder");
    let (ch_in, ch, ch_up, ch_out) = (8, 8, 4, 4);
    let t_in = 4;
    let (us, uk, up) = (2, 4, 1); // upsample stride, kernel, padding
    let t_up = (t_in - 1) * us + uk - 2 * up; // = 6

    let feat = b.add_input("features", &[ch_in, t_in]);
    let eps = b.add_input("eps", &[1]);

    // Conv pre
    let pre_w = b.add_input("pre_w", &[ch, ch_in, 3]);
    let x = b.add_conv1d(feat, pre_w, None, 1, 1, &[ch, t_in]);

    // ConvTranspose1d upsample
    let up_w = b.add_input("up_w", &[ch, ch_up, uk]);
    let x_up = b.add_conv_transpose_1d(x, up_w, None, us, up, 1, 1, 0, &[ch_up, t_up]);

    // ResBlock: InstanceNorm + Snake + Conv1d + residual
    let gamma = b.add_input("gamma", &[ch_up]);
    let beta = b.add_input("beta", &[ch_up]);
    let normed = b.add_instance_norm(x_up, eps, 1, Some(gamma), Some(beta), &[ch_up, t_up]);

    let alpha = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &[ch_up, t_up]);
    let snake_k = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_k, &[normed, alpha_bc], &[ch_up, t_up]);

    let res_w = b.add_input("res_w", &[ch_up, ch_up, 3]);
    let sub_out = b.add_conv1d(snake_out, res_w, None, 1, 1, &[ch_up, t_up]);
    let res = b.add_binary_add(x_up, sub_out, &[ch_up, t_up]);

    // Conv post + Exp
    let post_w = b.add_input("post_w", &[ch_out, ch_up, 3]);
    let x_post = b.add_conv1d(res, post_w, None, 1, 1, &[ch_out, t_up]);
    let output = b.add_exp(x_post, &[ch_out, t_up]);

    b.build(output).expect("valid kokoro_decoder graph")
}

#[allow(clippy::vec_init_then_push)]
fn kokoro_bindings() -> Vec<TensorParamBinding> {
    let (ch_in, ch, ch_up, ch_out, uk) = (8, 8, 4, 4, 4);
    let mut v = Vec::new();
    v.push(TensorParamBinding::Variable); // features
    v.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    v.push(ct(&[ch, ch_in, 3], W)); // pre_w
    v.push(ct(&[ch, ch_up, uk], W)); // up_w
    v.push(ct(&[ch_up], 1.0)); // gamma
    v.push(ct(&[ch_up], 0.0)); // beta
    v.push(TensorParamBinding::ConstantScalar(1.0)); // alpha
                                                     // (alpha_broadcast is internal, not a binding)
    v.push(ct(&[ch_up, ch_up, 3], W)); // res_w
    v.push(ct(&[ch_out, ch_up, 3], W)); // post_w
    v
}

// =========================================================================
// Helper
// =========================================================================

/// Shorthand for ConstantTensor filled with `val`.
fn ct(shape: &[usize], val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), val))
}

fn uniform(shape: &[usize], half_range: f32) -> BoundedTensor {
    let lo = ArrayD::from_elem(IxDyn(shape), -half_range);
    let hi = ArrayD::from_elem(IxDyn(shape), half_range);
    BoundedTensor::new(lo, hi).expect("bounded tensor")
}

fn non_negative(shape: &[usize], upper: f32) -> BoundedTensor {
    let lo = ArrayD::from_elem(IxDyn(shape), 0.0f32);
    let hi = ArrayD::from_elem(IxDyn(shape), upper);
    BoundedTensor::new(lo, hi).expect("bounded tensor")
}

// =========================================================================
// Public API
// =========================================================================

pub(super) fn extra_model_configs() -> Vec<ModelConfig> {
    vec![
        ModelConfig {
            name: "htdemucs_full",
            def: build_htdemucs(),
            bindings: htdemucs_bindings(),
            input_bounds: uniform(&[4, 8], 1.0),
            input_lower: -1.0,
            input_upper: 1.0,
        },
        ModelConfig {
            name: "whisper_full",
            def: build_whisper(),
            bindings: whisper_bindings(),
            input_bounds: uniform(&[4, 8], 1.0),
            input_lower: -1.0,
            input_upper: 1.0,
        },
        ModelConfig {
            name: "qwen3_full",
            def: build_qwen3(),
            bindings: qwen3_bindings(),
            input_bounds: uniform(&[4, 8], 1.0),
            input_lower: -1.0,
            input_upper: 1.0,
        },
        ModelConfig {
            name: "kokoro_decoder",
            def: build_kokoro(),
            bindings: kokoro_bindings(),
            input_bounds: non_negative(&[8, 4], 1.0),
            input_lower: 0.0,
            input_upper: 1.0,
        },
    ]
}
