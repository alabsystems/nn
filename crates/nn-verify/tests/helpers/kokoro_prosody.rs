// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Builder helpers for Kokoro ProsodyPredictor duration positivity verification.
//!
//! Builds a NY graph for the Kokoro ProsodyPredictor:
//!
//!   Conv1d(h) → Transpose → AdaLayerNorm(·, style) → Concat(·, style) →
//!   LSTM(·, h_0=0, c_0=0) → Linear → Transpose → Residual add
//!
//! After all blocks: Transpose → Linear(dur_proj) → dur_logits [B, T]
//!
//! Phase 1A: 1 block + final projection (~30 nodes).
//! Phase 1B: 3 blocks + final projection (~80 nodes).
//!
//! **Single-variable approach:** Both text_features [D_MODEL, SEQ_LEN] and
//! style [STYLE_DIM] are packed into a single flat Variable input of shape
//! [D_MODEL * SEQ_LEN + STYLE_DIM]. Narrow+Reshape in the IR splits them.
//! This avoids the heterogeneous multi-variable shape constraint (NY
//! requires all Variables to have the same shape for stacking).
//!
//! Part of #1729: Attention Monotonicity Proofs.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Model dimension (production Kokoro: 512).
pub(super) const D_MODEL: usize = 8;

/// Style dimension (production Kokoro: 128).
const STYLE_DIM: usize = 4;

/// LSTM hidden size (production Kokoro: 256 per direction, we use unidirectional).
const LSTM_HIDDEN: usize = 4;

/// Conv1d kernel size in ProsodyBlock (production: 3).
const CONV_KERNEL: usize = 3;

/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

/// Sequence length T=1 for Phase 1A/1B.
pub(super) const SEQ_LEN: usize = 1;

/// Total flat input size: text_features + style packed into one vector.
pub(super) const FLAT_INPUT_SIZE: usize = D_MODEL * SEQ_LEN + STYLE_DIM;

/// Number of blocks in the full ProsodyPredictor (production Kokoro: 3).
pub(super) const N_BLOCKS: usize = 3;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.01;

// ---------------------------------------------------------------------------
// Per-block builder helper
// ---------------------------------------------------------------------------

/// Add one ProsodyBlock to the graph. Returns the output hidden state
/// node `h_after_block` with shape [D_MODEL, SEQ_LEN].
///
/// Each block: Conv1d → Transpose → AdaLayerNorm(·, style) → Concat(·, style) →
///             LSTM(zero-init) → Linear → Transpose → Residual add
///
/// The block adds its own weight input nodes (conv_w, conv_b, adaln_proj_w/b,
/// ones, ln_w/b, lstm_h0/c0/w_ih/w_hh/bias, proj_w/b) prefixed with `block{idx}_`.
fn add_prosody_block(
    b: &mut TensorBlockBuilder,
    text_input: TensorNodeId,
    style_input: TensorNodeId,
    eps: TensorNodeId,
    idx: usize,
) -> TensorNodeId {
    let lstm_dim = D_MODEL + STYLE_DIM;

    // 1. Conv1d: [D_MODEL, SEQ_LEN] → [D_MODEL, SEQ_LEN] (same-padding)
    let conv_w = b.add_input(
        &format!("block{idx}_conv_w"),
        &[D_MODEL, D_MODEL, CONV_KERNEL],
    );
    let conv_b = b.add_input(&format!("block{idx}_conv_b"), &[D_MODEL]);
    let conv_b_bc = b.add_broadcast_left(conv_b, &[D_MODEL, SEQ_LEN]);
    let conv_out = b.add_conv1d(
        text_input,
        conv_w,
        None,
        1,
        CONV_PADDING,
        &[D_MODEL, SEQ_LEN],
    );
    let conv_biased = b.add_binary_add(conv_out, conv_b_bc, &[D_MODEL, SEQ_LEN]);

    // 2. Transpose: [D_MODEL, SEQ_LEN] → [SEQ_LEN, D_MODEL]
    let conv_t = b.add_transpose(conv_biased, &[1, 0], &[SEQ_LEN, D_MODEL]);

    // 3. AdaLayerNorm decomposition:
    //    projected = Linear(style, [2*D_MODEL, STYLE_DIM])
    //    gamma = 1 + projected[:D_MODEL]
    //    beta = projected[D_MODEL:]
    //    output = gamma * LayerNorm(x) + beta

    // 3a. Style projection: [STYLE_DIM] → [2*D_MODEL]
    let style_proj_w = b.add_input(
        &format!("block{idx}_adaln_proj_w"),
        &[2 * D_MODEL, STYLE_DIM],
    );
    let style_proj_b = b.add_input(&format!("block{idx}_adaln_proj_b"), &[2 * D_MODEL]);

    // Reshape style from [STYLE_DIM] to [1, STYLE_DIM] for matmul
    let style_rs = b.add_reshape(style_input, &[1, STYLE_DIM]);
    let style_projected = b.add_matmul(style_rs, style_proj_w, true, None, &[1, 2 * D_MODEL]);
    let style_proj_b_bc = b.add_broadcast(style_proj_b, &[1, 2 * D_MODEL]);
    let style_projected_biased =
        b.add_binary_add(style_projected, style_proj_b_bc, &[1, 2 * D_MODEL]);

    // 3b. Narrow to gamma_raw [1, D_MODEL] and beta [1, D_MODEL]
    let gamma_raw = b.add_narrow(style_projected_biased, 1, 0, D_MODEL, &[1, D_MODEL]);
    let beta_raw = b.add_narrow(style_projected_biased, 1, D_MODEL, D_MODEL, &[1, D_MODEL]);

    // 3c. gamma = 1 + gamma_raw
    let ones = b.add_input(&format!("block{idx}_ones"), &[1, D_MODEL]);
    let gamma = b.add_binary_add(ones, gamma_raw, &[1, D_MODEL]);

    // 3d. LayerNorm on conv_t [SEQ_LEN, D_MODEL]
    let ln_w = b.add_input(&format!("block{idx}_ln_w"), &[D_MODEL]);
    let ln_b = b.add_input(&format!("block{idx}_ln_b"), &[D_MODEL]);
    let normed = b.add_layer_norm(conv_t, eps, 1, ln_w, ln_b, &[SEQ_LEN, D_MODEL]);

    // 3e. Scale and shift: output = gamma * normed + beta
    let scaled = b.add_binary_mul(gamma, normed, &[SEQ_LEN, D_MODEL]);
    let adaln_out = b.add_binary_add(scaled, beta_raw, &[SEQ_LEN, D_MODEL]);

    // 4. Concat with style: [SEQ_LEN, D_MODEL] cat [SEQ_LEN, STYLE_DIM]
    let style_for_cat = b.add_reshape(style_input, &[SEQ_LEN, STYLE_DIM]);
    let lstm_input = b.add_concat(
        &[adaln_out, style_for_cat],
        1, // axis=1 (feature dimension)
        &[SEQ_LEN, lstm_dim],
    );

    // 5. LSTM cell (T=1 → single step, zero-init state)
    let h0 = b.add_input(&format!("block{idx}_lstm_h0"), &[LSTM_HIDDEN]);
    let c0 = b.add_input(&format!("block{idx}_lstm_c0"), &[LSTM_HIDDEN]);
    let lstm_w_ih = b.add_input(
        &format!("block{idx}_lstm_w_ih"),
        &[4 * LSTM_HIDDEN, lstm_dim],
    );
    let lstm_w_hh = b.add_input(
        &format!("block{idx}_lstm_w_hh"),
        &[4 * LSTM_HIDDEN, LSTM_HIDDEN],
    );
    let lstm_bias = b.add_input(&format!("block{idx}_lstm_bias"), &[4 * LSTM_HIDDEN]);

    // Squeeze SEQ_LEN=1 dim for LSTM: [1, lstm_dim] → [lstm_dim]
    let lstm_input_sq = b.add_reshape(lstm_input, &[lstm_dim]);
    let h0_sq = b.add_reshape(h0, &[LSTM_HIDDEN]);
    let c0_sq = b.add_reshape(c0, &[LSTM_HIDDEN]);

    let lstm_out = b.add_lstm(
        lstm_input_sq,
        h0_sq,
        c0_sq,
        lstm_w_ih,
        lstm_w_hh,
        Some(lstm_bias),
        &[LSTM_HIDDEN],
    );

    // 6. Project LSTM output back to D_MODEL: [LSTM_HIDDEN] → [D_MODEL]
    let proj_w = b.add_input(&format!("block{idx}_proj_w"), &[D_MODEL, LSTM_HIDDEN]);
    let proj_b = b.add_input(&format!("block{idx}_proj_b"), &[D_MODEL]);

    let lstm_out_rs = b.add_reshape(lstm_out, &[1, LSTM_HIDDEN]);
    let projected = b.add_matmul(lstm_out_rs, proj_w, true, None, &[1, D_MODEL]);
    let proj_b_bc = b.add_broadcast(proj_b, &[1, D_MODEL]);
    let projected_biased = b.add_binary_add(projected, proj_b_bc, &[1, D_MODEL]);

    // 7. Reshape for residual: [1, D_MODEL] → [D_MODEL, SEQ_LEN] via transpose
    let proj_t = b.add_transpose(projected_biased, &[1, 0], &[D_MODEL, SEQ_LEN]);

    // 8. Residual: h_new = h + proj_t
    b.add_binary_add(text_input, proj_t, &[D_MODEL, SEQ_LEN])
}

// ---------------------------------------------------------------------------
// Duration projection (shared final layer)
// ---------------------------------------------------------------------------

/// Add the final duration projection after all blocks.
/// Input: h [D_MODEL, SEQ_LEN], output: dur_logits [SEQ_LEN].
fn add_duration_projection(
    b: &mut TensorBlockBuilder,
    h_after_blocks: TensorNodeId,
) -> TensorNodeId {
    // Transpose: [D_MODEL, SEQ_LEN] → [SEQ_LEN, D_MODEL]
    let h_final_t = b.add_transpose(h_after_blocks, &[1, 0], &[SEQ_LEN, D_MODEL]);

    // Duration projection: Linear([SEQ_LEN, D_MODEL], [1, D_MODEL]) → [SEQ_LEN, 1]
    let dur_w = b.add_input("dur_proj_w", &[1, D_MODEL]);
    let dur_b = b.add_input("dur_proj_b", &[1]);
    let dur_logits_2d = b.add_matmul(h_final_t, dur_w, true, None, &[SEQ_LEN, 1]);
    let dur_b_bc = b.add_broadcast(dur_b, &[SEQ_LEN, 1]);
    let dur_logits_biased = b.add_binary_add(dur_logits_2d, dur_b_bc, &[SEQ_LEN, 1]);

    // Squeeze: [SEQ_LEN, 1] → [SEQ_LEN]
    b.add_reshape(dur_logits_biased, &[SEQ_LEN])
}

// ---------------------------------------------------------------------------
// Input splitting (shared preamble)
// ---------------------------------------------------------------------------

/// Add input splitting: flat_input → (text_input, style_input, eps).
fn add_input_splitting(b: &mut TensorBlockBuilder) -> (TensorNodeId, TensorNodeId, TensorNodeId) {
    let text_size = D_MODEL * SEQ_LEN;

    // Single flat Variable input
    let flat_input = b.add_input("flat_input", &[FLAT_INPUT_SIZE]);

    // Split into text_features and style via Narrow + Reshape
    let text_flat = b.add_narrow(flat_input, 0, 0, text_size, &[text_size]);
    let text_input = b.add_reshape(text_flat, &[D_MODEL, SEQ_LEN]);

    let style_input = b.add_narrow(flat_input, 0, text_size, STYLE_DIM, &[STYLE_DIM]);

    // Shared eps constant
    let eps = b.add_input("eps", &[1]);

    (text_input, style_input, eps)
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Build a single ProsodyBlock + final duration projection (Phase 1A).
///
/// Returns `(TensorKernelDef, output_shape)` where output is [SEQ_LEN].
pub(super) fn build_kokoro_prosody_single_block() -> (TensorKernelDef, [usize; 1]) {
    let mut b = TensorBlockBuilder::new("kokoro_prosody_duration_verify");

    let (text_input, style_input, eps) = add_input_splitting(&mut b);

    // Single block
    let h = add_prosody_block(&mut b, text_input, style_input, eps, 0);

    // Final duration projection
    let dur_logits = add_duration_projection(&mut b, h);

    (
        b.build(dur_logits).expect("valid kokoro prosody graph"),
        [SEQ_LEN],
    )
}

/// Build 3 ProsodyBlocks with residual connections + final duration projection (Phase 1B).
///
/// Each block's output feeds as input to the next block. Residual connections
/// inside each block add the block's output to its input. The key research
/// question: does CROWN stay tight through 3 sequential blocks?
///
/// Returns `(TensorKernelDef, output_shape)` where output is [SEQ_LEN].
pub(super) fn build_kokoro_prosody_three_blocks() -> (TensorKernelDef, [usize; 1]) {
    let mut b = TensorBlockBuilder::new("kokoro_prosody_duration_3block_verify");

    let (text_input, style_input, eps) = add_input_splitting(&mut b);

    // Chain 3 blocks: each block's output is the next block's text_input
    let h0 = add_prosody_block(&mut b, text_input, style_input, eps, 0);
    let h1 = add_prosody_block(&mut b, h0, style_input, eps, 1);
    let h2 = add_prosody_block(&mut b, h1, style_input, eps, 2);

    // Final duration projection
    let dur_logits = add_duration_projection(&mut b, h2);

    (
        b.build(dur_logits)
            .expect("valid kokoro prosody 3-block graph"),
        [SEQ_LEN],
    )
}

// ---------------------------------------------------------------------------
// Bindings helpers
// ---------------------------------------------------------------------------

/// Push bindings for the Conv1d + AdaLayerNorm portion of a ProsodyBlock.
fn push_conv_adaln_bindings(bindings: &mut Vec<TensorParamBinding>) {
    push_conv_adaln_bindings_with_mag(bindings, WEIGHT_MAG);
}

/// Push Conv1d + AdaLayerNorm bindings with a configurable weight magnitude.
fn push_conv_adaln_bindings_with_mag(bindings: &mut Vec<TensorParamBinding>, weight_mag: f32) {
    // conv_w [D_MODEL, D_MODEL, CONV_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, CONV_KERNEL]),
        weight_mag,
    )));
    // conv_b [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
    // adaln_proj_w [2*D_MODEL, STYLE_DIM]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[2 * D_MODEL, STYLE_DIM]),
        weight_mag,
    )));
    // adaln_proj_b [2*D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[2 * D_MODEL]),
        0.0f32,
    )));
    // ones [1, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, D_MODEL]),
        1.0f32,
    )));
    // ln_w [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        1.0f32,
    )));
    // ln_b [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
}

/// Push bindings for the LSTM + projection portion of a ProsodyBlock.
fn push_lstm_proj_bindings(bindings: &mut Vec<TensorParamBinding>) {
    push_lstm_proj_bindings_with_mag(bindings, WEIGHT_MAG);
}

/// Push LSTM + projection bindings with a configurable weight magnitude.
fn push_lstm_proj_bindings_with_mag(bindings: &mut Vec<TensorParamBinding>, weight_mag: f32) {
    let lstm_dim = D_MODEL + STYLE_DIM;

    // lstm_h0 [LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[LSTM_HIDDEN]),
        0.0f32,
    )));
    // lstm_c0 [LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[LSTM_HIDDEN]),
        0.0f32,
    )));
    // lstm_w_ih [4*LSTM_HIDDEN, lstm_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN, lstm_dim]),
        weight_mag,
    )));
    // lstm_w_hh [4*LSTM_HIDDEN, LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN, LSTM_HIDDEN]),
        weight_mag,
    )));
    // lstm_bias [4*LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN]),
        0.0f32,
    )));
    // proj_w [D_MODEL, LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, LSTM_HIDDEN]),
        weight_mag,
    )));
    // proj_b [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
}

/// Push bindings for a single ProsodyBlock (Conv1d+AdaLN + LSTM+Proj).
fn push_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    push_conv_adaln_bindings(bindings);
    push_lstm_proj_bindings(bindings);
}

/// Push bindings for a single ProsodyBlock with configurable weight magnitude.
fn push_block_bindings_with_mag(bindings: &mut Vec<TensorParamBinding>, weight_mag: f32) {
    push_conv_adaln_bindings_with_mag(bindings, weight_mag);
    push_lstm_proj_bindings_with_mag(bindings, weight_mag);
}

/// Push bindings for the final duration projection.
fn push_duration_proj_bindings(bindings: &mut Vec<TensorParamBinding>) {
    push_duration_proj_bindings_with_mag(bindings, WEIGHT_MAG);
}

/// Push duration projection bindings with configurable weight magnitude.
fn push_duration_proj_bindings_with_mag(bindings: &mut Vec<TensorParamBinding>, weight_mag: f32) {
    // dur_proj_w [1, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, D_MODEL]),
        weight_mag,
    )));
    // dur_proj_b [1]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Build parameter bindings for the single-block Kokoro ProsodyPredictor graph.
///
/// Single Variable input (flat_input [FLAT_INPUT_SIZE]). All other inputs are
/// ConstantTensor or ConstantScalar with small synthetic weights.
pub(super) fn kokoro_prosody_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // 1. flat_input [FLAT_INPUT_SIZE] — Variable
    bindings.push(TensorParamBinding::Variable);
    // (Narrow + Reshape for text_input and style_input are internal)

    // 2. eps — ConstantScalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Block 0
    push_block_bindings(&mut bindings);

    // Final duration projection
    push_duration_proj_bindings(&mut bindings);

    bindings
}

/// Build parameter bindings for the 3-block Kokoro ProsodyPredictor graph (Phase 1B).
///
/// Same structure as single-block but with 3 sets of block weights.
pub(super) fn kokoro_prosody_three_block_bindings() -> Vec<TensorParamBinding> {
    kokoro_prosody_three_block_bindings_with_weight_mag(WEIGHT_MAG)
}

/// Build single-block bindings with configurable weight magnitude.
///
/// Used for weight sensitivity analysis (ICLR Table 2).
pub(super) fn kokoro_prosody_bindings_with_weight_mag(weight_mag: f32) -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    bindings.push(TensorParamBinding::Variable);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    push_block_bindings_with_mag(&mut bindings, weight_mag);
    push_duration_proj_bindings_with_mag(&mut bindings, weight_mag);

    bindings
}

/// Build 3-block bindings with configurable weight magnitude.
///
/// Used for weight sensitivity analysis (ICLR Table 2).
pub(super) fn kokoro_prosody_three_block_bindings_with_weight_mag(
    weight_mag: f32,
) -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    bindings.push(TensorParamBinding::Variable);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    for _ in 0..N_BLOCKS {
        push_block_bindings_with_mag(&mut bindings, weight_mag);
    }
    push_duration_proj_bindings_with_mag(&mut bindings, weight_mag);

    bindings
}
