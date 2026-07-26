// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for DiffSinger-style singing pitch control verification.
//!
//! Builds a simplified pitch prediction graph for score-indexed pitch-control
//! bound propagation. The production DiffSinger pitch predictor takes
//! score-conditioned features (MIDI note, duration, phoneme identity) and
//! produces a per-note F0 contour. This simplified version:
//!
//!   score_input [NUM_NOTES * NOTE_FEATURES]
//!   → Reshape to [NUM_NOTES, NOTE_FEATURES]
//!   → Linear (NOTE_FEATURES → HIDDEN_DIM) per note
//!   → ReLU
//!   → Linear (HIDDEN_DIM → HIDDEN_DIM) per note
//!   → ReLU
//!   → Linear (HIDDEN_DIM → 1) per note (pitch prediction)
//!   → Reshape to [NUM_NOTES]
//!
//! Simplifications for NY tractability:
//! - No diffusion process (direct prediction instead of iterative denoising)
//! - No phoneme encoder (score features used directly)
//! - Shared linear weights across notes (via matmul on [NUM_NOTES, D] tensor)
//! - ReLU instead of SiLU/Mish (NY native support)
//!
//! **Single-variable approach:** Score features for all notes are packed into
//! one flat vector `score_input [SCORE_INPUT_SIZE]`. The IR splits and reshapes
//! them into per-note feature vectors.
//!
//! Part of #3516: CROWN for singing voice pitch/vibrato/formant proofs.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Number of notes in the score excerpt.
/// Production DiffSinger handles variable-length scores; we use a fixed
/// short excerpt for tractable bound propagation.
pub(super) const NUM_NOTES: usize = 4;

/// Features per note: [midi_pitch_embed, duration_embed].
/// Production DiffSinger uses richer encodings (phoneme, singer ID, etc.).
/// We use 2 features per note as the minimal score representation.
pub(super) const NOTE_FEATURES: usize = 2;

/// Hidden dimension for the pitch prediction MLP.
pub(super) const HIDDEN_DIM: usize = 8;

/// Total flat input size: all note features packed into one vector.
pub(super) const SCORE_INPUT_SIZE: usize = NUM_NOTES * NOTE_FEATURES;

/// Output size: one pitch prediction per note.
pub(super) const OUTPUT_SIZE: usize = NUM_NOTES;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.1;

// ---------------------------------------------------------------------------
// Network building blocks
// ---------------------------------------------------------------------------

/// Add a linear layer: [NUM_NOTES, in_dim] → matmul(W^T) + bias → [NUM_NOTES, out_dim].
fn add_linear_layer(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    in_dim: usize,
    out_dim: usize,
    prefix: &str,
) -> nn_dsl::TensorNodeId {
    let w = b.add_input(&format!("{prefix}_w"), &[out_dim, in_dim]);
    let bias = b.add_input(&format!("{prefix}_b"), &[out_dim]);

    // matmul: [NUM_NOTES, in_dim] @ [out_dim, in_dim]^T → [NUM_NOTES, out_dim]
    let projected = b.add_matmul(input, w, true, None, &[NUM_NOTES, out_dim]);

    // broadcast bias [out_dim] → [NUM_NOTES, out_dim] and add
    let bias_bc = b.add_broadcast(bias, &[NUM_NOTES, out_dim]);
    b.add_binary_add(projected, bias_bc, &[NUM_NOTES, out_dim])
}

// ---------------------------------------------------------------------------
// Full singing pitch control builder
// ---------------------------------------------------------------------------

/// Build a simplified DiffSinger pitch prediction graph.
///
/// Architecture:
///   score_input [SCORE_INPUT_SIZE]
///   → Reshape [NUM_NOTES, NOTE_FEATURES]
///   → Linear(NOTE_FEATURES, HIDDEN_DIM) → ReLU
///   → Linear(HIDDEN_DIM, HIDDEN_DIM)    → ReLU
///   → Linear(HIDDEN_DIM, 1)
///   → Reshape [NUM_NOTES]
///
/// Returns `(TensorKernelDef, [usize; 1])` following the kokoro helper convention.
pub(super) fn build_singing_pitch_control() -> (TensorKernelDef, [usize; 1]) {
    let mut b = TensorBlockBuilder::new("singing_pitch_control_verify");

    // Input: flat score features
    let score_input = b.add_input("score_input", &[SCORE_INPUT_SIZE]);

    // Reshape to per-note features: [NUM_NOTES, NOTE_FEATURES]
    let notes = b.add_reshape(score_input, &[NUM_NOTES, NOTE_FEATURES]);

    // Layer 1: Linear(NOTE_FEATURES → HIDDEN_DIM) + ReLU
    let h1 = add_linear_layer(&mut b, notes, NOTE_FEATURES, HIDDEN_DIM, "layer1");
    let h1_act = b.add_relu(h1, &[NUM_NOTES, HIDDEN_DIM]);

    // Layer 2: Linear(HIDDEN_DIM → HIDDEN_DIM) + ReLU
    let h2 = add_linear_layer(&mut b, h1_act, HIDDEN_DIM, HIDDEN_DIM, "layer2");
    let h2_act = b.add_relu(h2, &[NUM_NOTES, HIDDEN_DIM]);

    // Output layer: Linear(HIDDEN_DIM → 1) per note
    let pitch_2d = add_linear_layer(&mut b, h2_act, HIDDEN_DIM, 1, "pitch_out");

    // Reshape [NUM_NOTES, 1] → [NUM_NOTES]
    let output = b.add_reshape(pitch_2d, &[NUM_NOTES]);

    (
        b.build(output).expect("valid singing pitch control graph"),
        [OUTPUT_SIZE],
    )
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Push bindings for a linear layer (weight + bias).
fn push_linear_bindings(bindings: &mut Vec<TensorParamBinding>, out_dim: usize, in_dim: usize) {
    // w [out_dim, in_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_dim, in_dim]),
        WEIGHT_MAG,
    )));
    // b [out_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_dim]),
        0.0f32,
    )));
}

/// Build parameter bindings for the singing pitch control graph.
///
/// Input order matches `add_input` call order:
/// 1. score_input [SCORE_INPUT_SIZE] — Variable
/// 2. layer1_w [HIDDEN_DIM, NOTE_FEATURES], layer1_b [HIDDEN_DIM]
/// 3. layer2_w [HIDDEN_DIM, HIDDEN_DIM], layer2_b [HIDDEN_DIM]
/// 4. pitch_out_w [1, HIDDEN_DIM], pitch_out_b [1]
#[allow(clippy::vec_init_then_push)]
pub(super) fn singing_pitch_control_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // 1. score_input — Variable
    bindings.push(TensorParamBinding::Variable);

    // 2. Layer 1: NOTE_FEATURES → HIDDEN_DIM
    push_linear_bindings(&mut bindings, HIDDEN_DIM, NOTE_FEATURES);

    // 3. Layer 2: HIDDEN_DIM → HIDDEN_DIM
    push_linear_bindings(&mut bindings, HIDDEN_DIM, HIDDEN_DIM);

    // 4. Output: HIDDEN_DIM → 1
    push_linear_bindings(&mut bindings, 1, HIDDEN_DIM);

    bindings
}
