// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, unreachable_pub, clippy::duplicated_attributes)]

//! Builder helpers for adversarial phoneme stability compose tests.
//!
//! Architecture: Phoneme embedding (Variable) → Linear → ReLU → Linear (output).
//!
//! Models a simplified PlBert-like phoneme encoder for testing that CROWN
//! bounds remain tight under adversarial phoneme perturbations defined by
//! confusion sets from `nn_tts_verify::adversarial`.
//!
//! Key design decisions:
//! - Continuous relaxation: embedding output is a Variable input, not discrete
//!   lookup. The per-dimension bounds come from `embedding_bounds_for_token_set`.
//! - Small dimensions for NY tractability (D_MODEL=8, SEQ_LEN=4).
//! - Weight magnitude 0.01 to keep bounds propagation stable.
//!
//! Part of #1740: Adversarial Robustness of TTS.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Phoneme embedding dimension (production PlBert: 768).
pub(super) const EMBED_DIM: usize = 8;

/// Sequence length (number of phonemes).
pub(super) const SEQ_LEN: usize = 4;

/// Hidden dimension in the encoder MLP.
const HIDDEN_DIM: usize = 16;

/// Output dimension (e.g., duration/energy/pitch prediction per phoneme).
pub(super) const OUTPUT_DIM: usize = 4;

/// Vocabulary size (Kokoro: 178 tokens).
pub(super) const VOCAB_SIZE: usize = 16;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.01;

// ---------------------------------------------------------------------------
// Synthetic embedding weights
// ---------------------------------------------------------------------------

/// Generate synthetic embedding weights for testing.
///
/// Creates a `[VOCAB_SIZE, EMBED_DIM]` matrix where each token's embedding
/// is deterministic based on its ID. Adjacent tokens (which form confusion
/// pairs) get similar but distinct embeddings.
pub(super) fn synthetic_embedding_weights() -> Vec<f64> {
    let mut weights = vec![0.0f64; VOCAB_SIZE * EMBED_DIM];
    for t in 0..VOCAB_SIZE {
        for d in 0..EMBED_DIM {
            // Base pattern: token-specific + dimension-specific
            let base = (t as f64) * 0.1 + (d as f64) * 0.05;
            // Add small variation so confusion pairs are close but not identical
            let variation = ((t * 7 + d * 3) % 11) as f64 * 0.02;
            weights[t * EMBED_DIM + d] = base + variation;
        }
    }
    weights
}

/// Confusion sets for the synthetic vocabulary.
///
/// Maps to the same structure as `english_confusion_sets()` but with
/// VOCAB_SIZE=16 token IDs for testing.
pub(super) fn test_confusion_sets() -> Vec<nn_tts_verify::ConfusionSet> {
    vec![
        // Voicing pair: tokens 0 ↔ 1
        nn_tts_verify::ConfusionSet {
            name: "test_voicing_0_1".into(),
            token_ids: vec![0, 1],
            labels: vec!["p".into(), "b".into()],
            category: nn_tts_verify::ConfusionCategory::VoicingPair,
        },
        // Voicing pair: tokens 2 ↔ 3
        nn_tts_verify::ConfusionSet {
            name: "test_voicing_2_3".into(),
            token_ids: vec![2, 3],
            labels: vec!["t".into(), "d".into()],
            category: nn_tts_verify::ConfusionCategory::VoicingPair,
        },
        // Vowel group: tokens 4,5,6
        nn_tts_verify::ConfusionSet {
            name: "test_vowels_4_5_6".into(),
            token_ids: vec![4, 5, 6],
            labels: vec!["ɪ".into(), "iː".into(), "ɛ".into()],
            category: nn_tts_verify::ConfusionCategory::VowelProximity,
        },
        // Nasal group: tokens 7,8
        nn_tts_verify::ConfusionSet {
            name: "test_nasals_7_8".into(),
            token_ids: vec![7, 8],
            labels: vec!["m".into(), "n".into()],
            category: nn_tts_verify::ConfusionCategory::PlaceConfusion,
        },
    ]
}

// ---------------------------------------------------------------------------
// Phoneme encoder builder
// ---------------------------------------------------------------------------

/// Build a simplified phoneme encoder as a single `TensorKernelDef`.
///
/// Architecture: phoneme_emb (Variable [SEQ_LEN, EMBED_DIM])
///   → Linear([HIDDEN_DIM, EMBED_DIM]) → ReLU
///   → Linear([OUTPUT_DIM, HIDDEN_DIM])
///
/// The Variable input represents the continuous relaxation of the discrete
/// phoneme embedding space. Bounds come from `embedding_bounds_for_token_set`
/// or `sequence_perturbation_bounds`.
pub(super) fn build_phoneme_encoder() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("phoneme_encoder_verify");

    // Variable input: phoneme embedding (continuous relaxation)
    let emb_input = b.add_input("phoneme_emb", &[SEQ_LEN, EMBED_DIM]);

    // Hidden layer weights and bias — weight shape [out, in] per add_linear convention
    let w1 = b.add_input("w1", &[HIDDEN_DIM, EMBED_DIM]);
    let b1 = b.add_input("b1", &[HIDDEN_DIM]);

    // Output layer weights and bias — weight shape [out, in]
    let w2 = b.add_input("w2", &[OUTPUT_DIM, HIDDEN_DIM]);
    let b2 = b.add_input("b2", &[OUTPUT_DIM]);

    // Forward: Linear → ReLU → Linear
    let hidden = b.add_linear(emb_input, w1, Some(b1), &[SEQ_LEN, HIDDEN_DIM]);
    let activated = b.add_relu(hidden, &[SEQ_LEN, HIDDEN_DIM]);
    let output = b.add_linear(activated, w2, Some(b2), &[SEQ_LEN, OUTPUT_DIM]);

    b.build(output).expect("valid phoneme encoder graph")
}

/// Build parameter bindings for the phoneme encoder.
///
/// phoneme_emb = Variable, all other inputs = ConstantTensor.
pub(super) fn phoneme_encoder_bindings() -> Vec<TensorParamBinding> {
    vec![
        // phoneme_emb: Variable [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::Variable,
        // w1: [HIDDEN_DIM, EMBED_DIM] — [out, in] convention
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )),
        // b1: [HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        // w2: [OUTPUT_DIM, HIDDEN_DIM] — [out, in] convention
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUTPUT_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // b2: [OUTPUT_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OUTPUT_DIM]), 0.0f32)),
    ]
}

/// Build a deeper phoneme encoder with residual connection.
///
/// Architecture: phoneme_emb (Variable [SEQ_LEN, EMBED_DIM])
///   → Linear([EMBED_DIM, EMBED_DIM]) → ReLU → + residual
///   → Linear([OUTPUT_DIM, EMBED_DIM])
///
/// Tests that residual connections preserve bound tightness.
pub(super) fn build_phoneme_encoder_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("phoneme_encoder_residual_verify");

    // Variable input
    let emb_input = b.add_input("phoneme_emb", &[SEQ_LEN, EMBED_DIM]);

    // Residual block weights
    let w1 = b.add_input("res_w", &[EMBED_DIM, EMBED_DIM]);
    let b1 = b.add_input("res_b", &[EMBED_DIM]);

    // Output projection — weight shape [out, in]
    let w_out = b.add_input("out_w", &[OUTPUT_DIM, EMBED_DIM]);
    let b_out = b.add_input("out_b", &[OUTPUT_DIM]);

    // Forward: Linear → ReLU → + residual → Linear
    let hidden = b.add_linear(emb_input, w1, Some(b1), &[SEQ_LEN, EMBED_DIM]);
    let activated = b.add_relu(hidden, &[SEQ_LEN, EMBED_DIM]);
    let residual = b.add_binary_add(emb_input, activated, &[SEQ_LEN, EMBED_DIM]);
    let output = b.add_linear(residual, w_out, Some(b_out), &[SEQ_LEN, OUTPUT_DIM]);

    b.build(output)
        .expect("valid residual phoneme encoder graph")
}

/// Build parameter bindings for the residual phoneme encoder.
pub(super) fn phoneme_encoder_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        // phoneme_emb: Variable
        TensorParamBinding::Variable,
        // res_w: [EMBED_DIM, EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )),
        // res_b: [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        // out_w: [OUTPUT_DIM, EMBED_DIM] — [out, in] convention
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUTPUT_DIM, EMBED_DIM]),
            WEIGHT_MAG,
        )),
        // out_b: [OUTPUT_DIM]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OUTPUT_DIM]), 0.0f32)),
    ]
}
