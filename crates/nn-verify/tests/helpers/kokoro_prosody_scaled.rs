// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Scaled ProsodyPredictor builder for parameterized dimension verification.
//!
//! Extends the D=8 ProsodyPredictor from `kokoro_prosody.rs` with parameterized
//! dimensions using `KokoroDims` from `kokoro_scaled_pipeline.rs`. This enables
//! verifying the duration prediction path at D=16, D=32, D=64 — matching the
//! scaling trajectory of the vocoder decoder in Phases 33-34.
//!
//! Architecture (duration path):
//! ```text
//!   flat_input [d_model * seq_len + style_dim]  (Variable)
//!   → Narrow+Reshape → text [d_model, seq_len] + style [style_dim]
//!   → ProsodyBlock × N_BLOCKS:
//!       Conv1d + AdaLayerNorm(style) + Concat(style) + LSTM + Linear + Residual
//!   → Duration projection: Transpose + Linear → dur_logits [seq_len]
//! ```
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 35.

use super::helpers::KokoroDims;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Scaled ProsodyPredictor dimensions
// ---------------------------------------------------------------------------

/// Prosody-specific dimensions derived from KokoroDims.
#[derive(Debug, Clone, Copy)]
pub(super) struct ProsodyDims {
    /// Model dimension (from KokoroDims::d_model).
    pub(super) d_model: usize,
    /// Style dimension (production: 128, scaled proportionally).
    pub(super) style_dim: usize,
    /// LSTM hidden size (production: 256, scaled proportionally).
    pub(super) lstm_hidden: usize,
    /// Sequence length (phoneme tokens).
    pub(super) seq_len: usize,
    /// Number of ProsodyBlocks (production: 3).
    pub(super) n_blocks: usize,
}

impl ProsodyDims {
    /// Derive prosody dimensions from KokoroDims.
    ///
    /// style_dim = d_model / 2 (production: 128/512 ≈ 1/4, but we use 1/2
    /// for smaller scales to keep the LSTM input non-trivial).
    /// lstm_hidden = d_model / 2 (matching production ratio: 256/512).
    pub(super) fn from_kokoro(dims: &KokoroDims) -> Self {
        Self {
            d_model: dims.d_model,
            style_dim: (dims.d_model / 2).max(2),
            lstm_hidden: (dims.d_model / 2).max(2),
            seq_len: dims.seq_len,
            n_blocks: 3,
        }
    }

    /// Flat input size: text_features + style packed into one vector.
    pub(super) fn flat_input_size(&self) -> usize {
        self.d_model * self.seq_len + self.style_dim
    }

    /// LSTM input dimension: d_model + style_dim (after concat).
    fn lstm_dim(&self) -> usize {
        self.d_model + self.style_dim
    }
}

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.01;

/// Conv1d kernel size in ProsodyBlock (production: 3).
const CONV_KERNEL: usize = 3;

/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

// ---------------------------------------------------------------------------
// Input splitting
// ---------------------------------------------------------------------------

/// Add input splitting: flat_input → (text_input, style_input, eps).
fn add_input_splitting(
    b: &mut TensorBlockBuilder,
    pd: &ProsodyDims,
) -> (TensorNodeId, TensorNodeId, TensorNodeId) {
    let text_size = pd.d_model * pd.seq_len;

    let flat_input = b.add_input("flat_input", &[pd.flat_input_size()]);

    let text_flat = b.add_narrow(flat_input, 0, 0, text_size, &[text_size]);
    let text_input = b.add_reshape(text_flat, &[pd.d_model, pd.seq_len]);

    let style_input = b.add_narrow(flat_input, 0, text_size, pd.style_dim, &[pd.style_dim]);

    let eps = b.add_input("eps", &[1]);

    (text_input, style_input, eps)
}

// ---------------------------------------------------------------------------
// Per-block builder
// ---------------------------------------------------------------------------

/// Add one ProsodyBlock at the given scale.
///
/// Conv1d → Transpose → AdaLayerNorm(style) → Concat(style) → LSTM → Linear → Residual
fn add_prosody_block_scaled(
    b: &mut TensorBlockBuilder,
    text_input: TensorNodeId,
    style_input: TensorNodeId,
    eps: TensorNodeId,
    pd: &ProsodyDims,
    idx: usize,
) -> TensorNodeId {
    let d = pd.d_model;
    let s = pd.seq_len;
    let sd = pd.style_dim;
    let lh = pd.lstm_hidden;
    let lstm_dim = pd.lstm_dim();

    // 1. Conv1d: [d_model, seq_len] → [d_model, seq_len] (same-padding)
    let conv_w = b.add_input(&format!("block{idx}_conv_w"), &[d, d, CONV_KERNEL]);
    let conv_b = b.add_input(&format!("block{idx}_conv_b"), &[d]);
    let conv_b_bc = b.add_broadcast_left(conv_b, &[d, s]);
    let conv_out = b.add_conv1d(text_input, conv_w, None, 1, CONV_PADDING, &[d, s]);
    let conv_biased = b.add_binary_add(conv_out, conv_b_bc, &[d, s]);

    // 2. Transpose: [d_model, seq_len] → [seq_len, d_model]
    let conv_t = b.add_transpose(conv_biased, &[1, 0], &[s, d]);

    // 3. AdaLayerNorm decomposition
    let style_proj_w = b.add_input(&format!("block{idx}_adaln_proj_w"), &[2 * d, sd]);
    let style_proj_b = b.add_input(&format!("block{idx}_adaln_proj_b"), &[2 * d]);

    let style_rs = b.add_reshape(style_input, &[1, sd]);
    let style_projected = b.add_matmul(style_rs, style_proj_w, true, None, &[1, 2 * d]);
    let style_proj_b_bc = b.add_broadcast(style_proj_b, &[1, 2 * d]);
    let style_projected_biased = b.add_binary_add(style_projected, style_proj_b_bc, &[1, 2 * d]);

    let gamma_raw = b.add_narrow(style_projected_biased, 1, 0, d, &[1, d]);
    let beta_raw = b.add_narrow(style_projected_biased, 1, d, d, &[1, d]);

    let ones = b.add_input(&format!("block{idx}_ones"), &[1, d]);
    let gamma = b.add_binary_add(ones, gamma_raw, &[1, d]);

    let ln_w = b.add_input(&format!("block{idx}_ln_w"), &[d]);
    let ln_b = b.add_input(&format!("block{idx}_ln_b"), &[d]);
    let normed = b.add_layer_norm(conv_t, eps, 1, ln_w, ln_b, &[s, d]);

    // Broadcast gamma/beta across sequence
    let gamma_bc = b.add_broadcast(gamma, &[s, d]);
    let beta_bc = b.add_broadcast(beta_raw, &[s, d]);
    let scaled = b.add_binary_mul(gamma_bc, normed, &[s, d]);
    let adaln_out = b.add_binary_add(scaled, beta_bc, &[s, d]);

    // 4. Concat with style: [seq_len, d_model] cat [seq_len, style_dim]
    let style_bc = b.add_broadcast(style_rs, &[s, sd]);
    let lstm_input = b.add_concat(&[adaln_out, style_bc], 1, &[s, lstm_dim]);

    // 5. LSTM (single-step per token — use T=1 per-step for tractability)
    // For scaled verification, we use the built-in LSTM node (not unrolled).
    // We process just the first timestep to keep the graph tractable.
    let h0 = b.add_input(&format!("block{idx}_lstm_h0"), &[lh]);
    let c0 = b.add_input(&format!("block{idx}_lstm_c0"), &[lh]);
    let lstm_w_ih = b.add_input(&format!("block{idx}_lstm_w_ih"), &[4 * lh, lstm_dim]);
    let lstm_w_hh = b.add_input(&format!("block{idx}_lstm_w_hh"), &[4 * lh, lh]);
    let lstm_bias = b.add_input(&format!("block{idx}_lstm_bias"), &[4 * lh]);

    // Take first timestep: [seq_len, lstm_dim] → [lstm_dim]
    let first_step = b.add_narrow(lstm_input, 0, 0, 1, &[1, lstm_dim]);
    let first_step_sq = b.add_reshape(first_step, &[lstm_dim]);

    let lstm_out = b.add_lstm(
        first_step_sq,
        h0,
        c0,
        lstm_w_ih,
        lstm_w_hh,
        Some(lstm_bias),
        &[lh],
    );

    // 6. Project LSTM output: [lstm_hidden] → [d_model]
    let proj_w = b.add_input(&format!("block{idx}_proj_w"), &[d, lh]);
    let proj_b = b.add_input(&format!("block{idx}_proj_b"), &[d]);

    let lstm_out_rs = b.add_reshape(lstm_out, &[1, lh]);
    let projected = b.add_matmul(lstm_out_rs, proj_w, true, None, &[1, d]);
    let proj_b_bc = b.add_broadcast(proj_b, &[1, d]);
    let projected_biased = b.add_binary_add(projected, proj_b_bc, &[1, d]);

    // 7. Broadcast + transpose for residual: [1, d_model] → [d_model, seq_len]
    let proj_bc = b.add_broadcast(projected_biased, &[s, d]);
    let proj_t = b.add_transpose(proj_bc, &[1, 0], &[d, s]);

    // 8. Residual: h_new = text_input + projected
    b.add_binary_add(text_input, proj_t, &[d, s])
}

// ---------------------------------------------------------------------------
// Duration projection
// ---------------------------------------------------------------------------

/// Add the final duration projection.
/// Input: h [d_model, seq_len], output: dur_logits [seq_len].
fn add_duration_projection_scaled(
    b: &mut TensorBlockBuilder,
    h_after_blocks: TensorNodeId,
    pd: &ProsodyDims,
) -> TensorNodeId {
    let d = pd.d_model;
    let s = pd.seq_len;

    let h_final_t = b.add_transpose(h_after_blocks, &[1, 0], &[s, d]);

    let dur_w = b.add_input("dur_proj_w", &[1, d]);
    let dur_b = b.add_input("dur_proj_b", &[1]);
    let dur_logits_2d = b.add_matmul(h_final_t, dur_w, true, None, &[s, 1]);
    let dur_b_bc = b.add_broadcast(dur_b, &[s, 1]);
    let dur_logits_biased = b.add_binary_add(dur_logits_2d, dur_b_bc, &[s, 1]);

    b.add_reshape(dur_logits_biased, &[s])
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Build the full scaled ProsodyPredictor (duration path).
///
/// Returns `(TensorKernelDef, seq_len)`.
pub(super) fn build_scaled_prosody(dims: &KokoroDims) -> (TensorKernelDef, usize) {
    let pd = ProsodyDims::from_kokoro(dims);
    let mut b = TensorBlockBuilder::new("kokoro_prosody_scaled");

    let (text_input, style_input, eps) = add_input_splitting(&mut b, &pd);

    // Chain N_BLOCKS ProsodyBlocks with residual connections
    let mut h = text_input;
    for idx in 0..pd.n_blocks {
        h = add_prosody_block_scaled(&mut b, h, style_input, eps, &pd, idx);
    }

    let dur_logits = add_duration_projection_scaled(&mut b, h, &pd);

    (
        b.build(dur_logits).expect("valid scaled prosody graph"),
        pd.seq_len,
    )
}

/// Build a single-block scaled ProsodyPredictor for simpler analysis.
///
/// Returns `(TensorKernelDef, seq_len)`.
pub(super) fn build_scaled_prosody_single_block(dims: &KokoroDims) -> (TensorKernelDef, usize) {
    let pd = ProsodyDims::from_kokoro(dims);
    let mut b = TensorBlockBuilder::new("kokoro_prosody_scaled_1block");

    let (text_input, style_input, eps) = add_input_splitting(&mut b, &pd);
    let h = add_prosody_block_scaled(&mut b, text_input, style_input, eps, &pd, 0);
    let dur_logits = add_duration_projection_scaled(&mut b, h, &pd);

    (
        b.build(dur_logits)
            .expect("valid scaled prosody single-block graph"),
        pd.seq_len,
    )
}

// ---------------------------------------------------------------------------
// Bindings builders
// ---------------------------------------------------------------------------

/// Push bindings for a ProsodyBlock at the given scale.
fn push_block_bindings_scaled(bindings: &mut Vec<TensorParamBinding>, pd: &ProsodyDims) {
    let d = pd.d_model;
    let sd = pd.style_dim;
    let lh = pd.lstm_hidden;
    let lstm_dim = pd.lstm_dim();

    // conv_w [d_model, d_model, CONV_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d, d, CONV_KERNEL]),
        WEIGHT_MAG,
    )));
    // conv_b [d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d]),
        0.0f32,
    )));
    // adaln_proj_w [2*d_model, style_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[2 * d, sd]),
        WEIGHT_MAG,
    )));
    // adaln_proj_b [2*d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[2 * d]),
        0.0f32,
    )));
    // ones [1, d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, d]),
        1.0f32,
    )));
    // ln_w [d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d]),
        1.0f32,
    )));
    // ln_b [d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d]),
        0.0f32,
    )));
    // lstm_h0 [lstm_hidden]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[lh]),
        0.0f32,
    )));
    // lstm_c0 [lstm_hidden]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[lh]),
        0.0f32,
    )));
    // lstm_w_ih [4*lstm_hidden, lstm_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * lh, lstm_dim]),
        WEIGHT_MAG,
    )));
    // lstm_w_hh [4*lstm_hidden, lstm_hidden]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * lh, lh]),
        WEIGHT_MAG,
    )));
    // lstm_bias [4*lstm_hidden]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * lh]),
        0.0f32,
    )));
    // proj_w [d_model, lstm_hidden]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d, lh]),
        WEIGHT_MAG,
    )));
    // proj_b [d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d]),
        0.0f32,
    )));
}

/// Push duration projection bindings.
fn push_duration_proj_bindings_scaled(bindings: &mut Vec<TensorParamBinding>, pd: &ProsodyDims) {
    let d = pd.d_model;
    // dur_proj_w [1, d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, d]),
        WEIGHT_MAG,
    )));
    // dur_proj_b [1]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));
}

/// Build parameter bindings for the full scaled ProsodyPredictor.
pub(super) fn scaled_prosody_bindings(dims: &KokoroDims) -> Vec<TensorParamBinding> {
    let pd = ProsodyDims::from_kokoro(dims);
    let mut bindings = Vec::new();

    // flat_input — Variable
    bindings.push(TensorParamBinding::Variable);
    // eps — ConstantScalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    for _ in 0..pd.n_blocks {
        push_block_bindings_scaled(&mut bindings, &pd);
    }
    push_duration_proj_bindings_scaled(&mut bindings, &pd);

    bindings
}

/// Build parameter bindings for a single-block scaled ProsodyPredictor.
pub(super) fn scaled_prosody_single_block_bindings(dims: &KokoroDims) -> Vec<TensorParamBinding> {
    let pd = ProsodyDims::from_kokoro(dims);
    let mut bindings = Vec::new();

    bindings.push(TensorParamBinding::Variable);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    push_block_bindings_scaled(&mut bindings, &pd);
    push_duration_proj_bindings_scaled(&mut bindings, &pd);

    bindings
}

/// Analysis result for scaled prosody verification.
#[derive(Debug)]
pub(super) struct ProsodyAnalysis {
    pub(super) d_model: usize,
    pub(super) style_dim: usize,
    pub(super) seq_len: usize,
    pub(super) graph_nodes: usize,
    pub(super) avg_bound_width: f64,
    pub(super) min_output_lo: f64,
    pub(super) max_output_hi: f64,
    pub(super) all_finite: bool,
}

/// Analyze IBP output bounds for a scaled prosody pipeline.
pub(super) fn analyze_scaled_prosody(
    ibp_output: &nn_verify::BoundedTensor,
    dims: &KokoroDims,
    graph_nodes: usize,
) -> ProsodyAnalysis {
    let pd = ProsodyDims::from_kokoro(dims);
    let (lo, hi) = ibp_output.lower_upper();

    let all_finite = lo.iter().chain(hi.iter()).all(|v| v.is_finite());

    let min_lo = lo
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, v| a.min(f64::from(v)));
    let max_hi = hi
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, v| a.max(f64::from(v)));

    let avg_width = if !lo.is_empty() {
        lo.iter()
            .zip(hi.iter())
            .map(|(&l, &h)| f64::from(h - l))
            .sum::<f64>()
            / lo.len() as f64
    } else {
        0.0
    };

    ProsodyAnalysis {
        d_model: pd.d_model,
        style_dim: pd.style_dim,
        seq_len: pd.seq_len,
        graph_nodes,
        avg_bound_width: avg_width,
        min_output_lo: min_lo,
        max_output_hi: max_hi,
        all_finite,
    }
}
