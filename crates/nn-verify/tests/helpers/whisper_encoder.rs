// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Whisper encoder NY composition tests.
//!
//! Conv1d stem + GELU + Transpose + positional embedding + N transformer
//! blocks + final LayerNorm as a single verified `TensorKernelDef`.
//!
//! Uses tanh-GELU as proxy for ERF-GELU (sound but slightly loose bounds).
//!
//! Part of #1696 AC3: Whisper encoder NY composition.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

use super::common::conv1d_out_len;

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Number of mel frequency bins (production: 128).
pub(super) const N_MEL: usize = 4;

/// Model dimension (production: 1280 for large-v3-turbo).
pub(super) const D_MODEL: usize = 8;

/// Number of attention heads (production: 20).
const N_HEADS: usize = 2;

/// FFN hidden dimension (production: 5120, typically 4x d_model).
const FFN_DIM: usize = D_MODEL * 2;

/// Sequence length of mel input frames (production: 3000).
pub(super) const SEQ_LEN: usize = 8;

/// Number of encoder transformer blocks (production: 32).
const N_ENCODER_LAYERS: usize = 2;

/// Conv1d kernel size for both stems.
const CONV_KERNEL: usize = 3;

/// Conv1d padding for both stems.
const CONV_PADDING: usize = 1;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.001;

/// Output sequence length after stride-2 convolution.
fn encoder_seq_out() -> usize {
    // Conv1d #1: stride=1, same padding → SEQ_LEN
    let after_conv1 = conv1d_out_len(SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING);
    // Conv1d #2: stride=2, same padding → ceil(SEQ_LEN/2)
    conv1d_out_len(after_conv1, CONV_KERNEL, 2, CONV_PADDING)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build the Whisper encoder as a single `TensorKernelDef`.
///
/// Architecture: mel → Conv1d(N_MEL→D_MODEL, k=3, s=1, p=1) → GELU →
/// Conv1d(D_MODEL→D_MODEL, k=3, s=2, p=1) → GELU → Transpose(1,2) →
/// + sinusoidal positional embedding → N × TransformerBlock → LayerNorm.
///
/// Returns `(TensorKernelDef, output_seq_len)`.
pub(super) fn build_whisper_encoder() -> (TensorKernelDef, usize) {
    let t_out = encoder_seq_out();
    let mut b = TensorBlockBuilder::new("whisper_encoder_verify");

    // --- Variable input: mel spectrogram [N_MEL, SEQ_LEN] ---
    let mel = b.add_input("mel", &[N_MEL, SEQ_LEN]);

    // --- Conv stem #1: Conv1d(N_MEL → D_MODEL, k=3, s=1, p=1) → GELU ---
    let conv1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_KERNEL]);
    let conv1_b = b.add_input("conv1_b", &[D_MODEL]);
    let conv1_out = b.add_conv1d(
        mel,
        conv1_w,
        Some(conv1_b),
        1,
        CONV_PADDING,
        &[D_MODEL, SEQ_LEN],
    );
    let gelu1 = b.add_gelu(conv1_out, &[D_MODEL, SEQ_LEN]);

    // --- Conv stem #2: Conv1d(D_MODEL → D_MODEL, k=3, s=2, p=1) → GELU ---
    let conv2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let conv2_b = b.add_input("conv2_b", &[D_MODEL]);
    let conv2_out = b.add_conv1d(
        gelu1,
        conv2_w,
        Some(conv2_b),
        2,
        CONV_PADDING,
        &[D_MODEL, t_out],
    );
    let gelu2 = b.add_gelu(conv2_out, &[D_MODEL, t_out]);

    // --- Transpose: [D_MODEL, t_out] → [t_out, D_MODEL] ---
    let transposed = b.add_transpose(gelu2, &[1, 0], &[t_out, D_MODEL]);

    // --- Positional embedding (constant, added element-wise) ---
    let pos_emb = b.add_input("pos_emb", &[t_out, D_MODEL]);
    let x = b.add_binary_add(transposed, pos_emb, &[t_out, D_MODEL]);

    // --- N encoder blocks (pre-norm transformer) ---
    let mut current = x;
    for i in 0..N_ENCODER_LAYERS {
        current = build_encoder_block(&mut b, current, i, t_out);
    }

    // --- Final LayerNorm ---
    let ln_w = b.add_input("ln_final_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_final_b", &[D_MODEL]);
    let eps = b.add_input("ln_final_eps", &[1]);
    // LayerNorm along last axis (axis=1 for [t_out, D_MODEL])
    let output = b.add_layer_norm(current, eps, 1, ln_w, ln_b, &[t_out, D_MODEL]);

    (b.build(output).expect("valid whisper encoder graph"), t_out)
}

/// Build a single pre-norm transformer encoder block.
///
/// LayerNorm → MHA(self) → + residual, then
/// LayerNorm → Linear(d→ffn) → GELU → Linear(ffn→d) → + residual
fn build_encoder_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    layer_idx: usize,
    _seq_len: usize,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let pfx = format!("enc{layer_idx}");

    // TransformerBlockWeights
    let weights = TransformerBlockWeights {
        ln1_weight: b.add_input(&format!("{pfx}_ln1_w"), &[D_MODEL]),
        ln1_bias: b.add_input(&format!("{pfx}_ln1_b"), &[D_MODEL]),
        ln2_weight: b.add_input(&format!("{pfx}_ln2_w"), &[D_MODEL]),
        ln2_bias: b.add_input(&format!("{pfx}_ln2_b"), &[D_MODEL]),
        q_weight: b.add_input(&format!("{pfx}_qw"), &[D_MODEL, D_MODEL]),
        k_weight: b.add_input(&format!("{pfx}_kw"), &[D_MODEL, D_MODEL]),
        v_weight: b.add_input(&format!("{pfx}_vw"), &[D_MODEL, D_MODEL]),
        out_weight: b.add_input(&format!("{pfx}_ow"), &[D_MODEL, D_MODEL]),
        ffn1_weight: b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, D_MODEL]),
        ffn2_weight: b.add_input(&format!("{pfx}_ffn2w"), &[D_MODEL, FFN_DIM]),
        eps: b.add_input(&format!("{pfx}_eps"), &[1]),
    };

    let config = TransformerBlockConfig {
        num_heads: N_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    b.add_transformer_block(input, &weights, &config)
        .expect("valid encoder transformer block")
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Build parameter bindings for the Whisper encoder.
///
/// mel = Variable, all other inputs = ConstantTensor or ConstantScalar.
#[allow(clippy::vec_init_then_push)]
pub(super) fn whisper_encoder_bindings() -> Vec<TensorParamBinding> {
    let t_out = encoder_seq_out();
    let mut bindings = Vec::new();

    // mel: Variable
    bindings.push(TensorParamBinding::Variable);

    // Conv1d #1: weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, N_MEL, CONV_KERNEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));

    // Conv1d #2: weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, CONV_KERNEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));

    // Positional embedding [t_out, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[t_out, D_MODEL]),
        WEIGHT_MAG,
    )));

    // N encoder blocks
    for _ in 0..N_ENCODER_LAYERS {
        // ln1_weight, ln1_bias
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
        // ln2_weight, ln2_bias
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL]),
            0.0f32,
        )));
        // q_weight, k_weight, v_weight, out_weight
        for _ in 0..4 {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, D_MODEL]),
                WEIGHT_MAG,
            )));
        }
        // ffn1_weight [FFN_DIM, D_MODEL], ffn2_weight [D_MODEL, FFN_DIM]
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )));
        // eps
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Final LayerNorm: weight, bias, eps
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    bindings
}
