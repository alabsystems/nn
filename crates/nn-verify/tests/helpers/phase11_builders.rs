// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 11 graph builders: N-block stacked ProsodyPredictor + D_MODEL scaling.
//!
//! Extends Phase 10 (2-block) to:
//! - Parameterized N-block stacking (1, 2, 3 blocks)
//! - Parameterized D_MODEL for scaling analysis (8, 12, 16)
//!
//! The real Kokoro ProsodyPredictor uses 3 blocks at d_model=512. Phase 11
//! tests whether crossover bounds scale predictably as depth and width increase.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 11.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// Default sequence length for verification.
const DEFAULT_SEQ_LEN: usize = 4;

/// Style embedding dimension.
const STYLE_DIM: usize = 4;

/// Conv1d kernel size (matches Kokoro ProsodyPredictor: kernel=3).
const CONV_KERNEL: usize = 3;

/// Conv1d padding for same-length output (kernel=3 → padding=1).
const CONV_PADDING: usize = 1;

// ---------------------------------------------------------------------------
// Parameterized graph configuration
// ---------------------------------------------------------------------------

/// Configuration for a parameterized ProsodyPredictor verification graph.
pub(super) struct ProsodyConfig {
    pub(super) seq_len: usize,
    pub(super) d_model: usize,
    pub(super) n_blocks: usize,
}

impl ProsodyConfig {
    pub(super) fn hidden_dim(&self) -> usize {
        self.d_model / 2
    }
}

impl Default for ProsodyConfig {
    fn default() -> Self {
        Self {
            seq_len: DEFAULT_SEQ_LEN,
            d_model: 8,
            n_blocks: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-block input nodes
// ---------------------------------------------------------------------------

struct BlockInputs {
    conv_w: TensorNodeId,
    eps: TensorNodeId,
    ln_w: TensorNodeId,
    ln_b: TensorNodeId,
    ones: TensorNodeId,
    gamma: TensorNodeId,
    beta: TensorNodeId,
    gate_wx: TensorNodeId,
    gate_bias: TensorNodeId,
    val_wx: TensorNodeId,
    val_bias: TensorNodeId,
    out_proj_w: TensorNodeId,
}

fn add_block_inputs(b: &mut TensorBlockBuilder, cfg: &ProsodyConfig, suffix: &str) -> BlockInputs {
    let d = cfg.d_model;
    let h = cfg.hidden_dim();
    let t = cfg.seq_len;
    BlockInputs {
        conv_w: b.add_input(&format!("conv_w{suffix}"), &[d, d, CONV_KERNEL]),
        eps: b.add_input(&format!("eps{suffix}"), &[1]),
        ln_w: b.add_input(&format!("ln_w{suffix}"), &[d]),
        ln_b: b.add_input(&format!("ln_b{suffix}"), &[d]),
        ones: b.add_input(&format!("ones{suffix}"), &[t, d]),
        gamma: b.add_input(&format!("gamma{suffix}"), &[t, d]),
        beta: b.add_input(&format!("beta{suffix}"), &[t, d]),
        gate_wx: b.add_input(&format!("gate_wx{suffix}"), &[d, h]),
        gate_bias: b.add_input(&format!("gate_bias{suffix}"), &[t, h]),
        val_wx: b.add_input(&format!("val_wx{suffix}"), &[d, h]),
        val_bias: b.add_input(&format!("val_bias{suffix}"), &[t, h]),
        out_proj_w: b.add_input(&format!("out_proj_w{suffix}"), &[h, d]),
    }
}

fn build_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    w: &BlockInputs,
    cfg: &ProsodyConfig,
) -> TensorNodeId {
    let d = cfg.d_model;
    let h = cfg.hidden_dim();
    let t = cfg.seq_len;

    // Conv1d: transpose → conv → transpose back
    let cf = b.add_transpose(input, &[1, 0], &[d, t]);
    let c = b.add_conv1d(cf, w.conv_w, None, 1, CONV_PADDING, &[d, t]);
    let cb = b.add_transpose(c, &[1, 0], &[t, d]);

    // LayerNorm
    let n = b.add_layer_norm(cb, w.eps, 1, w.ln_w, w.ln_b, &[t, d]);

    // AdaLayerNorm: (1 + gamma) * normed + beta
    let s = b.add_binary_add(w.ones, w.gamma, &[t, d]);
    let sc = b.add_binary_mul(n, s, &[t, d]);
    let ada = b.add_binary_add(sc, w.beta, &[t, d]);

    // Gate: sigmoid(ada @ W_gate + bias) * tanh(ada @ W_val + bias)
    let gx = b.add_matmul(ada, w.gate_wx, false, None, &[t, h]);
    let gr = b.add_binary_add(gx, w.gate_bias, &[t, h]);
    let g = b.add_sigmoid(gr, &[t, h]);
    let vx = b.add_matmul(ada, w.val_wx, false, None, &[t, h]);
    let vr = b.add_binary_add(vx, w.val_bias, &[t, h]);
    let v = b.add_tanh(vr, &[t, h]);
    let gated = b.add_binary_mul(g, v, &[t, h]);

    // Output projection + residual
    let proj = b.add_matmul(gated, w.out_proj_w, false, None, &[t, d]);
    b.add_binary_add(input, proj, &[t, d])
}

// ---------------------------------------------------------------------------
// N-block stacked ProsodyPredictor
// ---------------------------------------------------------------------------

/// Build an N-block stacked ProsodyPredictor + attention scores.
///
/// Architecture:
/// ```text
/// Block 1..N: Conv1d → AdaLayerNorm(style) → Gate → Proj → Residual
/// Attention: Q = blockN_out + PE, scores = Q @ K^T / √D
/// ```
pub(super) fn build_n_block_prosody_predictor(
    cfg: &ProsodyConfig,
) -> (TensorKernelDef, Vec<usize>) {
    let d = cfg.d_model;
    let t = cfg.seq_len;

    let mut b = TensorBlockBuilder::new(&format!(
        "attn_scores_{n}block_d{d}",
        n = cfg.n_blocks,
        d = cfg.d_model
    ));

    let raw_input = b.add_input("raw_input", &[t, d]);

    // Add block weights
    let block_inputs: Vec<BlockInputs> = (0..cfg.n_blocks)
        .map(|i| add_block_inputs(&mut b, cfg, &format!("{}", i + 1)))
        .collect();

    let pe = b.add_input("pe", &[t, d]);
    let k = b.add_input("key", &[t, d]);

    // Chain blocks: each takes output of previous
    let mut h = raw_input;
    for bi in &block_inputs {
        h = build_block(&mut b, h, bi, cfg);
    }

    // Attention scores
    let q = b.add_binary_add(h, pe, &[t, d]);
    let att_scale = 1.0 / (d as f32).sqrt();
    let scores_shape = [t, t];
    let scores = b.add_matmul(q, k, true, Some(att_scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid N-block prosody predictor graph");
    (def, scores_shape.to_vec())
}

// Weight constructors — delegated to super::common::weights (Part of #1938).
use super::common::weights;

fn build_encoder_weight(rows: usize, cols: usize, scale: f32) -> ArrayD<f32> {
    weights::encoder_weight(rows, cols, scale)
}

fn build_conv_weight(out_ch: usize, in_ch: usize, kernel: usize, scale: f32) -> ArrayD<f32> {
    weights::conv_weight(out_ch, in_ch, kernel, scale)
}

/// Build sinusoidal positional encoding [seq_len, d_model].
fn build_sinusoidal_pe(seq_len: usize, d_model: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for i in 0..d_model / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            data[t * d_model + 2 * i] = freq.sin() as f32;
            data[t * d_model + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE shape")
}

/// Build style embedding expanded to [T, S].
fn build_style_expanded(seq_len: usize, magnitude: f32) -> ArrayD<f32> {
    let row: Vec<f32> = (0..STYLE_DIM)
        .map(|i| magnitude * (1.0 + 0.1 * i as f32))
        .collect();
    let data: Vec<f32> = row
        .iter()
        .cycle()
        .take(seq_len * STYLE_DIM)
        .copied()
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[seq_len, STYLE_DIM]), data).expect("valid style shape")
}

/// Build style projection weight [S, out_dim].
fn build_style_proj_weight(out_dim: usize, scale: f32) -> ArrayD<f32> {
    let total = STYLE_DIM * out_dim;
    let mut data = vec![0.0f32; total];
    for i in 0..STYLE_DIM {
        for j in 0..out_dim {
            data[i * out_dim + j] = scale * 0.01 * ((i * j % 7) as f32 + 0.1);
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[STYLE_DIM, out_dim]), data).expect("valid style proj shape")
}

/// Pre-compute style @ W_s for decomposed concat+matmul.
fn precompute_style_bias(style: &ArrayD<f32>, w_s: &ArrayD<f32>) -> ArrayD<f32> {
    let s = style
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("style 2D");
    let w = w_s
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("w_s 2D");
    s.dot(&w).into_dyn()
}

/// Pre-compute style gamma and beta for one block.
fn precompute_style_gamma_beta(
    style: &ArrayD<f32>,
    style_proj_w: &ArrayD<f32>,
    d_model: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let s = style
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2D");
    let w = style_proj_w
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2D");
    let proj = s.dot(&w);
    let gamma = proj
        .slice(ndarray::s![.., 0..d_model])
        .to_owned()
        .into_dyn();
    let beta = proj.slice(ndarray::s![.., d_model..]).to_owned().into_dyn();
    (gamma, beta)
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Build one block's weight bindings.
fn block_bindings(
    cfg: &ProsodyConfig,
    enc_scale: f32,
    style: &ArrayD<f32>,
) -> Vec<TensorParamBinding> {
    let d = cfg.d_model;
    let h = cfg.hidden_dim();
    let t = cfg.seq_len;

    let conv_w = build_conv_weight(d, d, CONV_KERNEL, enc_scale);
    let style_proj_w = build_style_proj_weight(2 * d, enc_scale);
    let (gamma, beta) = precompute_style_gamma_beta(style, &style_proj_w, d);

    // Decomposed gate weights
    let full_gate_w = build_encoder_weight(d + STYLE_DIM, h, enc_scale * 0.5);
    let full_val_w = build_encoder_weight(d + STYLE_DIM, h, enc_scale * 0.5);
    let gate_wx = full_gate_w
        .slice(ndarray::s![0..d, ..])
        .to_owned()
        .into_dyn();
    let val_wx = full_val_w
        .slice(ndarray::s![0..d, ..])
        .to_owned()
        .into_dyn();
    let gate_ws = full_gate_w
        .slice(ndarray::s![d.., ..])
        .to_owned()
        .into_dyn();
    let val_ws = full_val_w.slice(ndarray::s![d.., ..]).to_owned().into_dyn();
    let gate_bias = precompute_style_bias(style, &gate_ws);
    let val_bias = precompute_style_bias(style, &val_ws);
    let out_proj_w = build_encoder_weight(h, d, enc_scale);

    vec![
        TensorParamBinding::ConstantTensor(conv_w),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[t, d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantTensor(gate_wx),
        TensorParamBinding::ConstantTensor(gate_bias),
        TensorParamBinding::ConstantTensor(val_wx),
        TensorParamBinding::ConstantTensor(val_bias),
        TensorParamBinding::ConstantTensor(out_proj_w),
    ]
}

/// Bindings for N-block stacked ProsodyPredictor.
/// Input order: raw_input, [block1 12 weights], ..., [blockN 12 weights], pe, key
pub(super) fn n_block_bindings(
    cfg: &ProsodyConfig,
    enc_scale: f32,
    pe_scale: f32,
) -> Vec<TensorParamBinding> {
    let style = build_style_expanded(cfg.seq_len, 0.5);
    let mut pe = build_sinusoidal_pe(cfg.seq_len, cfg.d_model);
    pe.mapv_inplace(|v| v * pe_scale);

    let mut bindings = vec![TensorParamBinding::Variable]; // raw_input
    for _ in 0..cfg.n_blocks {
        bindings.extend(block_bindings(cfg, enc_scale, &style));
    }
    bindings.push(TensorParamBinding::ConstantTensor(pe.clone())); // pe
    bindings.push(TensorParamBinding::ConstantTensor(pe)); // key
    bindings
}
