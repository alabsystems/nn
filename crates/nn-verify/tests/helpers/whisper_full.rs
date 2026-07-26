// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Whisper full-model NY composition tests.
//!
//! Encoder output is a constant binding; the decoder's token embedding input
//! is the single Variable for NY propagation.
//!
//! Decoder architecture: Token Embedding + Positional Embedding →
//! N × (CausalSelfAttn + CrossAttn(encoder_output) + FFN) → LayerNorm →
//! Tied output projection (embed_weight^T).
//!
//! Uses tanh-GELU as proxy for ERF-GELU (sound but slightly loose bounds).
//!
//! Part of #1696 AC4: Whisper full-model NY composition.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_dsl::AttentionMask;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

use super::common::conv1d_out_len;

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Model dimension (production: 1280 for large-v3-turbo).
pub(super) const D_MODEL: usize = 8;

/// Number of attention heads (production: 20).
const N_HEADS: usize = 2;

/// FFN hidden dimension (production: 5120, typically 4x d_model).
const FFN_DIM: usize = D_MODEL * 2;

/// Vocabulary size (production: 51866).
pub(super) const VOCAB_SIZE: usize = 16;

/// Decoder sequence length (number of tokens, production: up to 448).
pub(super) const DEC_SEQ_LEN: usize = 4;

/// Sequence length of mel input frames (production: 3000).
/// Used only to compute encoder output sequence length.
const MEL_SEQ_LEN: usize = 8;

/// Number of decoder transformer layers (production: 4 for large-v3-turbo).
const N_DECODER_LAYERS: usize = 2;

/// Conv1d kernel size for encoder stems.
const CONV_KERNEL: usize = 3;

/// Conv1d padding for encoder stems.
const CONV_PADDING: usize = 1;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.001;

/// Encoder output sequence length after stride-2 convolution.
fn encoder_seq_out() -> usize {
    let after_conv1 = conv1d_out_len(MEL_SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING);
    conv1d_out_len(after_conv1, CONV_KERNEL, 2, CONV_PADDING)
}

// ---------------------------------------------------------------------------
// Decoder block builder (manual decomposition)
// ---------------------------------------------------------------------------

/// Weights for a single Whisper decoder block.
///
/// 3 sub-blocks: causal self-attention, cross-attention, FFN.
/// Each sub-block has its own LayerNorm.
struct DecoderBlockWeights {
    // Self-attention sub-block
    sa_ln_w: TensorNodeId,
    sa_ln_b: TensorNodeId,
    sa_q_w: TensorNodeId,
    sa_k_w: TensorNodeId,
    sa_v_w: TensorNodeId,
    sa_out_w: TensorNodeId,
    // Cross-attention sub-block
    ca_ln_w: TensorNodeId,
    ca_ln_b: TensorNodeId,
    ca_q_w: TensorNodeId,
    ca_k_w: TensorNodeId,
    ca_v_w: TensorNodeId,
    ca_out_w: TensorNodeId,
    // FFN sub-block
    ffn_ln_w: TensorNodeId,
    ffn_ln_b: TensorNodeId,
    ffn1_w: TensorNodeId,
    ffn2_w: TensorNodeId,
    // Shared eps
    eps: TensorNodeId,
}

fn add_decoder_block_weights(
    b: &mut TensorBlockBuilder,
    layer_idx: usize,
    eps: TensorNodeId,
) -> DecoderBlockWeights {
    let pfx = format!("dec{layer_idx}");
    DecoderBlockWeights {
        sa_ln_w: b.add_input(&format!("{pfx}_sa_ln_w"), &[D_MODEL]),
        sa_ln_b: b.add_input(&format!("{pfx}_sa_ln_b"), &[D_MODEL]),
        sa_q_w: b.add_input(&format!("{pfx}_sa_qw"), &[D_MODEL, D_MODEL]),
        sa_k_w: b.add_input(&format!("{pfx}_sa_kw"), &[D_MODEL, D_MODEL]),
        sa_v_w: b.add_input(&format!("{pfx}_sa_vw"), &[D_MODEL, D_MODEL]),
        sa_out_w: b.add_input(&format!("{pfx}_sa_ow"), &[D_MODEL, D_MODEL]),
        ca_ln_w: b.add_input(&format!("{pfx}_ca_ln_w"), &[D_MODEL]),
        ca_ln_b: b.add_input(&format!("{pfx}_ca_ln_b"), &[D_MODEL]),
        ca_q_w: b.add_input(&format!("{pfx}_ca_qw"), &[D_MODEL, D_MODEL]),
        ca_k_w: b.add_input(&format!("{pfx}_ca_kw"), &[D_MODEL, D_MODEL]),
        ca_v_w: b.add_input(&format!("{pfx}_ca_vw"), &[D_MODEL, D_MODEL]),
        ca_out_w: b.add_input(&format!("{pfx}_ca_ow"), &[D_MODEL, D_MODEL]),
        ffn_ln_w: b.add_input(&format!("{pfx}_ffn_ln_w"), &[D_MODEL]),
        ffn_ln_b: b.add_input(&format!("{pfx}_ffn_ln_b"), &[D_MODEL]),
        ffn1_w: b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, D_MODEL]),
        ffn2_w: b.add_input(&format!("{pfx}_ffn2w"), &[D_MODEL, FFN_DIM]),
        eps,
    }
}

/// Build a single Whisper decoder block.
///
/// Pre-norm structure:
/// 1. LayerNorm → MHA(self, causal) → + residual
/// 2. LayerNorm → MHA(cross, encoder_output) → + residual
/// 3. LayerNorm → Linear(d→ffn) → GELU → Linear(ffn→d) → + residual
fn build_decoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    encoder_output: TensorNodeId,
    w: &DecoderBlockWeights,
    enc_seq: usize,
) -> TensorNodeId {
    let shape = [DEC_SEQ_LEN, D_MODEL];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];

    // --- Sub-block 1: Causal self-attention ---
    let sa_normed = b.add_layer_norm(input, w.eps, 1, w.sa_ln_w, w.sa_ln_b, &shape);
    let sa_out = b
        .add_multi_head_attention(
            sa_normed,
            w.sa_q_w,
            w.sa_k_w,
            w.sa_v_w,
            w.sa_out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("decoder self-attention");
    let residual1 = b.add_binary_add(input, sa_out, &shape);

    // --- Sub-block 2: Cross-attention with encoder output ---
    let ca_normed = b.add_layer_norm(residual1, w.eps, 1, w.ca_ln_w, w.ca_ln_b, &shape);
    let enc_shape = [enc_seq, D_MODEL];
    let _ = enc_shape; // used by cross-attention output shape calculation
    let ca_out = b
        .add_multi_head_cross_attention(
            ca_normed,
            encoder_output,
            w.ca_q_w,
            w.ca_k_w,
            w.ca_v_w,
            w.ca_out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape, // output follows Q shape [DEC_SEQ_LEN, D_MODEL]
        )
        .expect("decoder cross-attention");
    let residual2 = b.add_binary_add(residual1, ca_out, &shape);

    // --- Sub-block 3: FFN ---
    let ffn_normed = b.add_layer_norm(residual2, w.eps, 1, w.ffn_ln_w, w.ffn_ln_b, &shape);
    let ffn1 = b.add_linear(ffn_normed, w.ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, w.ffn2_w, None, &shape);
    b.add_binary_add(residual2, ffn2, &shape)
}

// ---------------------------------------------------------------------------
// Full model builder
// ---------------------------------------------------------------------------

/// Build the Whisper full model as a single `TensorKernelDef`.
///
/// Architecture:
/// - Encoder output is a constant binding (certified by AC3 separately)
/// - Token embedding input is the single Variable
/// - Decoder: Embedding + PosEmb → N × DecoderBlock → LayerNorm → Output projection
///
/// Returns `(TensorKernelDef, vocab_size)`.
pub(super) fn build_whisper_full() -> (TensorKernelDef, usize) {
    let enc_seq = encoder_seq_out();
    let mut b = TensorBlockBuilder::new("whisper_full_verify");

    // --- Variable input: token embedding indices (approximated as continuous) ---
    // For NY, we approximate discrete token indices as continuous
    // values in a small range. The embedding lookup is approximated by a
    // matmul: tokens [DEC_SEQ_LEN, VOCAB_SIZE] × embed_weight [VOCAB_SIZE, D_MODEL].
    // This represents a soft embedding (continuous relaxation).
    let token_input = b.add_input("token_emb", &[DEC_SEQ_LEN, D_MODEL]);

    // --- Constant: positional embedding [DEC_SEQ_LEN, D_MODEL] ---
    let pos_emb = b.add_input("pos_emb", &[DEC_SEQ_LEN, D_MODEL]);
    let x = b.add_binary_add(token_input, pos_emb, &[DEC_SEQ_LEN, D_MODEL]);

    // --- Constant: encoder output [enc_seq, D_MODEL] ---
    let encoder_output = b.add_input("encoder_output", &[enc_seq, D_MODEL]);

    // --- Shared epsilon ---
    let eps = b.add_input("eps", &[1]);

    // --- N decoder blocks ---
    let mut current = x;
    let mut block_weights = Vec::new();
    for i in 0..N_DECODER_LAYERS {
        let w = add_decoder_block_weights(&mut b, i, eps);
        block_weights.push(w);
    }
    for (i, w) in block_weights.iter().enumerate() {
        let _ = i;
        current = build_decoder_block(&mut b, current, encoder_output, w, enc_seq);
    }

    // --- Final LayerNorm ---
    let ln_w = b.add_input("ln_final_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_final_b", &[D_MODEL]);
    let normed = b.add_layer_norm(current, eps, 1, ln_w, ln_b, &[DEC_SEQ_LEN, D_MODEL]);

    // --- Output projection: [DEC_SEQ_LEN, D_MODEL] × [D_MODEL, VOCAB_SIZE] ---
    // Tied with embedding weight (transposed). We use a separate weight input
    // since the binding just needs to be consistent.
    let proj_w = b.add_input("output_proj_w", &[D_MODEL, VOCAB_SIZE]);
    let logits = b.add_matmul(
        normed,
        proj_w,
        false, // not transposed — weight is already [D_MODEL, VOCAB_SIZE]
        None,
        &[DEC_SEQ_LEN, VOCAB_SIZE],
    );

    (
        b.build(logits).expect("valid whisper full model graph"),
        VOCAB_SIZE,
    )
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Build parameter bindings for the Whisper full model.
///
/// token_emb = Variable, all other inputs = ConstantTensor or ConstantScalar.
#[allow(clippy::vec_init_then_push)]
pub(super) fn whisper_full_bindings() -> Vec<TensorParamBinding> {
    let enc_seq = encoder_seq_out();
    let mut bindings = Vec::new();

    // token_emb: Variable [DEC_SEQ_LEN, D_MODEL]
    bindings.push(TensorParamBinding::Variable);

    // pos_emb: Constant [DEC_SEQ_LEN, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DEC_SEQ_LEN, D_MODEL]),
        WEIGHT_MAG,
    )));

    // encoder_output: Constant [enc_seq, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[enc_seq, D_MODEL]),
        WEIGHT_MAG,
    )));

    // eps: Constant scalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // N decoder blocks
    for _ in 0..N_DECODER_LAYERS {
        // Self-attention sub-block: ln_w, ln_b, q_w, k_w, v_w, out_w
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
        for _ in 0..4 {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, D_MODEL]),
                WEIGHT_MAG,
            )));
        }

        // Cross-attention sub-block: ln_w, ln_b, q_w, k_w, v_w, out_w
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
        for _ in 0..4 {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, D_MODEL]),
                WEIGHT_MAG,
            )));
        }

        // FFN sub-block: ln_w, ln_b, ffn1_w, ffn2_w
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )));
    }

    // Final LayerNorm: weight, bias (eps is shared, already bound above)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));

    // Output projection weight [D_MODEL, VOCAB_SIZE]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, VOCAB_SIZE]),
        WEIGHT_MAG,
    )));

    bindings
}
