// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Kokoro ProsodyPredictor T=4 LSTM unrolling verification.
//!
//! Phase 1C: Same 3-block architecture as Phase 1B, but with sequence length T=4.
//! The LSTM in each block is unrolled across 4 timesteps using primitive ops
//! (Linear + Sigmoid + Tanh + BinaryMul + BinaryAdd) so that h and c states
//! chain between timesteps. This tests CROWN bound tightness through
//! 3 blocks × 4 timesteps = 12 LSTM cells ≈ 400-500 NY nodes.
//!
//! The key research question: do CROWN bounds stay tight through temporal
//! unrolling (T=4), or does the additional depth cause significant loosening?
//!
//! Part of #1729: Attention Monotonicity Proofs — ICLR submission target.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (same model scale as Phase 1A/1B)
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

/// Sequence length T=4 for Phase 1C.
pub(super) const SEQ_LEN_T4: usize = 4;

/// Number of blocks in the full ProsodyPredictor (production Kokoro: 3).
pub(super) const N_BLOCKS: usize = 3;

/// Total flat input size: text_features [D_MODEL * T] + style [STYLE_DIM].
pub(super) const FLAT_INPUT_SIZE_T4: usize = D_MODEL * SEQ_LEN_T4 + STYLE_DIM;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.01;

// ---------------------------------------------------------------------------
// LSTM cell decomposition (inline, matching lstm_decomposed.rs:70-107)
// ---------------------------------------------------------------------------

/// Decompose one LSTM cell timestep using primitive ops in the builder.
/// Returns (h_new, c_new) node IDs for chaining to the next timestep.
///
/// Gate decomposition:
///   gates = Linear(x_t, w_ih, bias) + Linear(h_prev, w_hh, None)
///   i = sigmoid(gates[0:H])
///   f = sigmoid(gates[H:2H])
///   g = tanh(gates[2H:3H])
///   o = sigmoid(gates[3H:4H])
///   c_new = f * c_prev + i * g
///   h_new = o * tanh(c_new)
fn add_lstm_cell(
    b: &mut TensorBlockBuilder,
    x_t: TensorNodeId,
    h_prev: TensorNodeId,
    c_prev: TensorNodeId,
    w_ih: TensorNodeId,
    w_hh: TensorNodeId,
    bias: Option<TensorNodeId>,
) -> (TensorNodeId, TensorNodeId) {
    let gate_size = 4 * LSTM_HIDDEN;
    let gate_shape = [1, gate_size];
    let h_shape = [1, LSTM_HIDDEN];

    // gates = Linear(x_t, w_ih, bias) + Linear(h_prev, w_hh, None)
    let ih_out = b.add_linear(x_t, w_ih, bias, &gate_shape);
    let hh_out = b.add_linear(h_prev, w_hh, None, &gate_shape);
    let gates = b.add_binary_add(ih_out, hh_out, &gate_shape);

    // Split gates: [1, 4*H] → 4 × [1, H]
    let i_pre = b.add_narrow(gates, 1, 0, LSTM_HIDDEN, &h_shape);
    let f_pre = b.add_narrow(gates, 1, LSTM_HIDDEN, LSTM_HIDDEN, &h_shape);
    let g_pre = b.add_narrow(gates, 1, 2 * LSTM_HIDDEN, LSTM_HIDDEN, &h_shape);
    let o_pre = b.add_narrow(gates, 1, 3 * LSTM_HIDDEN, LSTM_HIDDEN, &h_shape);

    // Activations
    let i = b.add_sigmoid(i_pre, &h_shape);
    let f = b.add_sigmoid(f_pre, &h_shape);
    let g = b.add_tanh(g_pre, &h_shape);
    let o = b.add_sigmoid(o_pre, &h_shape);

    // State update
    let fc = b.add_binary_mul(f, c_prev, &h_shape);
    let ig = b.add_binary_mul(i, g, &h_shape);
    let c_new = b.add_binary_add(fc, ig, &h_shape);
    let c_new_tanh = b.add_tanh(c_new, &h_shape);
    let h_new = b.add_binary_mul(o, c_new_tanh, &h_shape);

    (h_new, c_new)
}

// ---------------------------------------------------------------------------
// Per-block builder (T=4 unrolled LSTM variant)
// ---------------------------------------------------------------------------

/// Add one ProsodyBlock with T=4 LSTM unrolling.
/// Returns the output hidden state node `h_after_block` with shape [D_MODEL, T].
///
/// Architecture identical to Phase 1A/1B except:
///   - Conv1d operates on [D_MODEL, T] instead of [D_MODEL, 1]
///   - LSTM is manually unrolled across T timesteps with chained h/c state
///   - Only the final timestep's h is used for projection (matching production)
fn add_prosody_block_t4(
    b: &mut TensorBlockBuilder,
    text_input: TensorNodeId,  // [D_MODEL, T]
    style_input: TensorNodeId, // [STYLE_DIM]
    eps: TensorNodeId,
    idx: usize,
) -> TensorNodeId {
    let lstm_dim = D_MODEL + STYLE_DIM;
    let t = SEQ_LEN_T4;

    // 1. Conv1d: [D_MODEL, T] → [D_MODEL, T] (same-padding)
    let conv_w = b.add_input(
        &format!("block{idx}_conv_w"),
        &[D_MODEL, D_MODEL, CONV_KERNEL],
    );
    let conv_b = b.add_input(&format!("block{idx}_conv_b"), &[D_MODEL]);
    let conv_b_bc = b.add_broadcast_left(conv_b, &[D_MODEL, t]);
    let conv_out = b.add_conv1d(text_input, conv_w, None, 1, CONV_PADDING, &[D_MODEL, t]);
    let conv_biased = b.add_binary_add(conv_out, conv_b_bc, &[D_MODEL, t]);

    // 2. Transpose: [D_MODEL, T] → [T, D_MODEL]
    let conv_t = b.add_transpose(conv_biased, &[1, 0], &[t, D_MODEL]);

    // 3. AdaLayerNorm decomposition (same as Phase 1A/1B but on [T, D_MODEL])
    let style_proj_w = b.add_input(
        &format!("block{idx}_adaln_proj_w"),
        &[2 * D_MODEL, STYLE_DIM],
    );
    let style_proj_b = b.add_input(&format!("block{idx}_adaln_proj_b"), &[2 * D_MODEL]);

    let style_rs = b.add_reshape(style_input, &[1, STYLE_DIM]);
    let style_projected = b.add_matmul(style_rs, style_proj_w, true, None, &[1, 2 * D_MODEL]);
    let style_proj_b_bc = b.add_broadcast(style_proj_b, &[1, 2 * D_MODEL]);
    let style_projected_biased =
        b.add_binary_add(style_projected, style_proj_b_bc, &[1, 2 * D_MODEL]);

    let gamma_raw = b.add_narrow(style_projected_biased, 1, 0, D_MODEL, &[1, D_MODEL]);
    let beta_raw = b.add_narrow(style_projected_biased, 1, D_MODEL, D_MODEL, &[1, D_MODEL]);

    let ones = b.add_input(&format!("block{idx}_ones"), &[1, D_MODEL]);
    let gamma = b.add_binary_add(ones, gamma_raw, &[1, D_MODEL]);

    let ln_w = b.add_input(&format!("block{idx}_ln_w"), &[D_MODEL]);
    let ln_b = b.add_input(&format!("block{idx}_ln_b"), &[D_MODEL]);
    let normed = b.add_layer_norm(conv_t, eps, 1, ln_w, ln_b, &[t, D_MODEL]);

    // Broadcast gamma/beta to [T, D_MODEL] for element-wise ops
    let gamma_bc = b.add_broadcast(gamma, &[t, D_MODEL]);
    let beta_bc = b.add_broadcast(beta_raw, &[t, D_MODEL]);
    let scaled = b.add_binary_mul(gamma_bc, normed, &[t, D_MODEL]);
    let adaln_out = b.add_binary_add(scaled, beta_bc, &[t, D_MODEL]);

    // 4. Concat with style: [T, D_MODEL] cat [T, STYLE_DIM] → [T, lstm_dim]
    let style_bc = b.add_broadcast(style_rs, &[t, STYLE_DIM]);
    let lstm_input = b.add_concat(&[adaln_out, style_bc], 1, &[t, lstm_dim]);

    // 5. LSTM unrolling: T=4 timesteps with chained h/c state
    let h0 = b.add_input(&format!("block{idx}_lstm_h0"), &[1, LSTM_HIDDEN]);
    let c0 = b.add_input(&format!("block{idx}_lstm_c0"), &[1, LSTM_HIDDEN]);
    let w_ih = b.add_input(
        &format!("block{idx}_lstm_w_ih"),
        &[4 * LSTM_HIDDEN, lstm_dim],
    );
    let w_hh = b.add_input(
        &format!("block{idx}_lstm_w_hh"),
        &[4 * LSTM_HIDDEN, LSTM_HIDDEN],
    );
    let lstm_bias = b.add_input(&format!("block{idx}_lstm_bias"), &[4 * LSTM_HIDDEN]);

    // Unroll T timesteps: slice input, run LSTM cell, chain states
    let mut h = h0;
    let mut c = c0;
    for step in 0..t {
        // Slice timestep: [T, lstm_dim] → [1, lstm_dim]
        let x_t = b.add_narrow(lstm_input, 0, step, 1, &[1, lstm_dim]);
        let (h_new, c_new) = add_lstm_cell(b, x_t, h, c, w_ih, w_hh, Some(lstm_bias));
        h = h_new;
        c = c_new;
    }

    // 6. Project final hidden state: [1, LSTM_HIDDEN] → [1, D_MODEL]
    let proj_w = b.add_input(&format!("block{idx}_proj_w"), &[D_MODEL, LSTM_HIDDEN]);
    let proj_b = b.add_input(&format!("block{idx}_proj_b"), &[D_MODEL]);

    let projected = b.add_matmul(h, proj_w, true, None, &[1, D_MODEL]);
    let proj_b_bc = b.add_broadcast(proj_b, &[1, D_MODEL]);
    let projected_biased = b.add_binary_add(projected, proj_b_bc, &[1, D_MODEL]);

    // 7. Reshape for residual: [1, D_MODEL] → [D_MODEL, T] via broadcast + transpose
    // Broadcast the single-timestep projection to [T, D_MODEL], then transpose
    let proj_bc = b.add_broadcast(projected_biased, &[t, D_MODEL]);
    let proj_t = b.add_transpose(proj_bc, &[1, 0], &[D_MODEL, t]);

    // 8. Residual: h_new = h_input + proj (broadcast projection across T)
    b.add_binary_add(text_input, proj_t, &[D_MODEL, t])
}

// ---------------------------------------------------------------------------
// Duration projection (T=4 variant)
// ---------------------------------------------------------------------------

/// Add the final duration projection for T=4 output.
/// Input: h [D_MODEL, T], output: dur_logits [T].
fn add_duration_projection_t4(
    b: &mut TensorBlockBuilder,
    h_after_blocks: TensorNodeId,
) -> TensorNodeId {
    let t = SEQ_LEN_T4;
    // Transpose: [D_MODEL, T] → [T, D_MODEL]
    let h_final_t = b.add_transpose(h_after_blocks, &[1, 0], &[t, D_MODEL]);

    // Duration projection: Linear([T, D_MODEL], [1, D_MODEL]) → [T, 1]
    let dur_w = b.add_input("dur_proj_w", &[1, D_MODEL]);
    let dur_b = b.add_input("dur_proj_b", &[1]);
    let dur_logits_2d = b.add_matmul(h_final_t, dur_w, true, None, &[t, 1]);
    let dur_b_bc = b.add_broadcast(dur_b, &[t, 1]);
    let dur_logits_biased = b.add_binary_add(dur_logits_2d, dur_b_bc, &[t, 1]);

    // Squeeze: [T, 1] → [T]
    b.add_reshape(dur_logits_biased, &[t])
}

// ---------------------------------------------------------------------------
// Input splitting (T=4 variant)
// ---------------------------------------------------------------------------

/// Add input splitting for T=4: flat_input → (text_input, style_input, eps).
fn add_input_splitting_t4(
    b: &mut TensorBlockBuilder,
) -> (TensorNodeId, TensorNodeId, TensorNodeId) {
    let text_size = D_MODEL * SEQ_LEN_T4;

    // Single flat Variable input
    let flat_input = b.add_input("flat_input", &[FLAT_INPUT_SIZE_T4]);

    // Split into text_features and style via Narrow + Reshape
    let text_flat = b.add_narrow(flat_input, 0, 0, text_size, &[text_size]);
    let text_input = b.add_reshape(text_flat, &[D_MODEL, SEQ_LEN_T4]);

    let style_input = b.add_narrow(flat_input, 0, text_size, STYLE_DIM, &[STYLE_DIM]);

    // Shared eps constant
    let eps = b.add_input("eps", &[1]);

    (text_input, style_input, eps)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build 3-block ProsodyPredictor with T=4 LSTM unrolling (Phase 1C).
///
/// Returns `(TensorKernelDef, output_shape)` where output is [T].
/// Graph size: ~400-500 nodes (3 blocks × 4 timesteps × ~17 nodes/cell +
/// Conv1d + AdaLayerNorm + projection overhead).
pub(super) fn build_kokoro_prosody_t4() -> (TensorKernelDef, [usize; 1]) {
    let mut b = TensorBlockBuilder::new("kokoro_prosody_duration_t4_verify");

    let (text_input, style_input, eps) = add_input_splitting_t4(&mut b);

    // Chain 3 blocks with T=4 unrolled LSTM
    let h0 = add_prosody_block_t4(&mut b, text_input, style_input, eps, 0);
    let h1 = add_prosody_block_t4(&mut b, h0, style_input, eps, 1);
    let h2 = add_prosody_block_t4(&mut b, h1, style_input, eps, 2);

    // Final duration projection
    let dur_logits = add_duration_projection_t4(&mut b, h2);

    (
        b.build(dur_logits).expect("valid kokoro prosody T=4 graph"),
        [SEQ_LEN_T4],
    )
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Push bindings for the Conv1d + AdaLayerNorm portion (T=4 variant).
fn push_conv_adaln_bindings_t4(bindings: &mut Vec<TensorParamBinding>) {
    let t = SEQ_LEN_T4;
    // conv_w [D_MODEL, D_MODEL, CONV_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, CONV_KERNEL]),
        WEIGHT_MAG,
    )));
    // conv_b [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
    // adaln_proj_w [2*D_MODEL, STYLE_DIM]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[2 * D_MODEL, STYLE_DIM]),
        WEIGHT_MAG,
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
    // Suppress unused variable warning for `t` — it's used to document
    // that these are the T=4 variant bindings.
    let _ = t;
}

/// Push bindings for the LSTM + projection portion (T=4 variant).
/// LSTM weights are shared across all T timesteps (same w_ih, w_hh, bias).
fn push_lstm_proj_bindings_t4(bindings: &mut Vec<TensorParamBinding>) {
    let lstm_dim = D_MODEL + STYLE_DIM;

    // lstm_h0 [1, LSTM_HIDDEN] — zero initial hidden state
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN]),
        0.0f32,
    )));
    // lstm_c0 [1, LSTM_HIDDEN] — zero initial cell state
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN]),
        0.0f32,
    )));
    // lstm_w_ih [4*LSTM_HIDDEN, lstm_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN, lstm_dim]),
        WEIGHT_MAG,
    )));
    // lstm_w_hh [4*LSTM_HIDDEN, LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN, LSTM_HIDDEN]),
        WEIGHT_MAG,
    )));
    // lstm_bias [4*LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN]),
        0.0f32,
    )));
    // proj_w [D_MODEL, LSTM_HIDDEN]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, LSTM_HIDDEN]),
        WEIGHT_MAG,
    )));
    // proj_b [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
}

/// Push bindings for a single T=4 ProsodyBlock.
fn push_block_bindings_t4(bindings: &mut Vec<TensorParamBinding>) {
    push_conv_adaln_bindings_t4(bindings);
    push_lstm_proj_bindings_t4(bindings);
}

/// Push bindings for the final duration projection.
fn push_duration_proj_bindings_t4(bindings: &mut Vec<TensorParamBinding>) {
    // dur_proj_w [1, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, D_MODEL]),
        WEIGHT_MAG,
    )));
    // dur_proj_b [1]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));
}

/// Build parameter bindings for the T=4, 3-block Kokoro ProsodyPredictor graph.
pub(super) fn kokoro_prosody_t4_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // 1. flat_input [FLAT_INPUT_SIZE_T4] — Variable
    bindings.push(TensorParamBinding::Variable);

    // 2. eps — ConstantScalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Block 0, 1, 2
    for _ in 0..N_BLOCKS {
        push_block_bindings_t4(&mut bindings);
    }

    // Final duration projection
    push_duration_proj_bindings_t4(&mut bindings);

    bindings
}
