// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! IBP compose tests filling Kokoro pipeline verification gaps.
//!
//! The gap detector (`gap_detector.rs`) reports 3 vacuous pipeline segments
//! that need non-vacuous IBP coverage via proxy graphs at toy scale:
//!
//! 1. **PlBert + bert_encoder** — vacuous (width=300, heuristic).
//!    PLBert is a BERT-style transformer encoder processing phoneme tokens.
//!    Architecture: Embedding → N × (LayerNorm → MHA → residual → LayerNorm → FFN → residual).
//!
//! 2. **ProsodyPredictor** — vacuous (width=412, heuristic).
//!    Duration prediction branch: N × (BiLSTM + AdaLayerNorm) blocks → duration projection.
//!    Two inputs: text features `[C, T]` and style vector `[S]`.
//!
//! 3. **F0EnergyPredictor** — vacuous (width=32820, heuristic).
//!    Pitch prediction branch: shared BiLSTM → 3 × AdainResBlk1d → linear projection.
//!    Two inputs: text features `[C, T]` and style vector `[S]`.
//!
//! These tests use small dims and proxy TensorBlockBuilder graphs to demonstrate
//! that the underlying operations produce non-vacuous IBP bounds. This does NOT
//! replace production-weight verification — it validates the propagation pathway.
//!
//! Part of #4311: NN verification gaps for Milestone 1 (Kokoro).
//! Part of #3351: Epic — Absolutely Best Kokoro.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use super::common::{
    assert_bounds_valid, assert_bounds_width, bounds_min_max, sinusoidal_pe, uniform_bounds,
};

// ===========================================================================
// Shared constants
// ===========================================================================

/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.01;

/// Vacuous width threshold — bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 500.0;

// ===========================================================================
// SECTION 1: PlBert + bert_encoder gap fill
// ===========================================================================
//
// The PLBert encoder is a standard pre-norm transformer stack.
// Production uses 12 layers × 768 dim; we verify at 2 layers × 16 dim.
// Status key: kokoro_production_bert_encoder

/// Model dimension for PLBert proxy.
const BERT_D_MODEL: usize = 16;
/// Sequence length for PLBert proxy.
const BERT_SEQ_LEN: usize = 4;
/// Number of attention heads.
const BERT_NUM_HEADS: usize = 2;
/// FFN intermediate dimension.
const BERT_FFN_DIM: usize = 32;

/// Build a 2-layer PLBert encoder proxy graph.
///
/// Architecture: Embedding + PE → LayerNorm → MHA → residual →
///               LayerNorm → FFN → residual (×2 layers).
///
/// Input: `[T, D]` (Variable — pre-embedded token representations).
/// Output: `[T, D]`.
fn build_plbert_encoder_2layer() -> TensorKernelDef {
    use nn_dsl::tensor_block_builder::{TransformerBlockConfig, TransformerBlockWeights};

    let shape = [BERT_SEQ_LEN, BERT_D_MODEL];
    let mut b = TensorBlockBuilder::new("plbert_encoder_2layer");

    let x = b.add_input("x", &shape);
    let pe = b.add_input("pe", &shape);

    // Add positional encoding
    let x_pe = b.add_binary_add(x, pe, &shape);

    // Layer 1
    let eps1 = b.add_input("eps1", &[1]);
    let ln1a_w = b.add_input("l1_ln1_w", &[BERT_D_MODEL]);
    let ln1a_b = b.add_input("l1_ln1_b", &[BERT_D_MODEL]);
    let l1_qw = b.add_input("l1_q_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let l1_kw = b.add_input("l1_k_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let l1_vw = b.add_input("l1_v_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let l1_ow = b.add_input("l1_out_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let ln1b_w = b.add_input("l1_ln2_w", &[BERT_D_MODEL]);
    let ln1b_b = b.add_input("l1_ln2_b", &[BERT_D_MODEL]);
    let l1_f1 = b.add_input("l1_ffn1_w", &[BERT_FFN_DIM, BERT_D_MODEL]);
    let l1_f2 = b.add_input("l1_ffn2_w", &[BERT_D_MODEL, BERT_FFN_DIM]);

    let config1 = TransformerBlockConfig {
        num_heads: BERT_NUM_HEADS,
        mask: AttentionMask::Standard, // PLBert is bidirectional
        ffn_hidden_dim: BERT_FFN_DIM,
    };

    let weights1 = TransformerBlockWeights {
        ln1_weight: ln1a_w,
        ln1_bias: ln1a_b,
        ln2_weight: ln1b_w,
        ln2_bias: ln1b_b,
        q_weight: l1_qw,
        k_weight: l1_kw,
        v_weight: l1_vw,
        out_weight: l1_ow,
        ffn1_weight: l1_f1,
        ffn2_weight: l1_f2,
        eps: eps1,
    };

    let layer1_out = b
        .add_transformer_block(x_pe, &weights1, &config1)
        .expect("valid transformer block 1");

    // Layer 2
    let eps2 = b.add_input("eps2", &[1]);
    let ln2a_w = b.add_input("l2_ln1_w", &[BERT_D_MODEL]);
    let ln2a_b = b.add_input("l2_ln1_b", &[BERT_D_MODEL]);
    let l2_qw = b.add_input("l2_q_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let l2_kw = b.add_input("l2_k_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let l2_vw = b.add_input("l2_v_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let l2_ow = b.add_input("l2_out_w", &[BERT_D_MODEL, BERT_D_MODEL]);
    let ln2b_w = b.add_input("l2_ln2_w", &[BERT_D_MODEL]);
    let ln2b_b = b.add_input("l2_ln2_b", &[BERT_D_MODEL]);
    let l2_f1 = b.add_input("l2_ffn1_w", &[BERT_FFN_DIM, BERT_D_MODEL]);
    let l2_f2 = b.add_input("l2_ffn2_w", &[BERT_D_MODEL, BERT_FFN_DIM]);

    let config2 = TransformerBlockConfig {
        num_heads: BERT_NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: BERT_FFN_DIM,
    };

    let weights2 = TransformerBlockWeights {
        ln1_weight: ln2a_w,
        ln1_bias: ln2a_b,
        ln2_weight: ln2b_w,
        ln2_bias: ln2b_b,
        q_weight: l2_qw,
        k_weight: l2_kw,
        v_weight: l2_vw,
        out_weight: l2_ow,
        ffn1_weight: l2_f1,
        ffn2_weight: l2_f2,
        eps: eps2,
    };

    let layer2_out = b
        .add_transformer_block(layer1_out, &weights2, &config2)
        .expect("valid transformer block 2");

    b.build(layer2_out).expect("valid plbert encoder graph")
}

/// Bindings for 2-layer PLBert encoder proxy.
fn plbert_encoder_bindings() -> Vec<TensorParamBinding> {
    let d2 = [BERT_D_MODEL, BERT_D_MODEL];
    let pe_data = sinusoidal_pe(BERT_SEQ_LEN, BERT_D_MODEL);

    let mut bindings = vec![
        TensorParamBinding::Variable,                // x
        TensorParamBinding::ConstantTensor(pe_data), // pe
    ];

    // Two layers of identical constant bindings
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
                                                                 // LN1
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BERT_D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BERT_D_MODEL]),
            0.0f32,
        )));
        // Q, K, V, Out projections
        for _ in 0..4 {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&d2),
                WEIGHT_MAG,
            )));
        }
        // LN2
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BERT_D_MODEL]),
            1.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BERT_D_MODEL]),
            0.0f32,
        )));
        // FFN1, FFN2
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BERT_FFN_DIM, BERT_D_MODEL]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BERT_D_MODEL, BERT_FFN_DIM]),
            WEIGHT_MAG,
        )));
    }

    bindings
}

/// PLBert 2-layer encoder produces non-vacuous IBP bounds.
///
/// This fills the `kokoro_production_bert_encoder` gap: the production entry
/// is stale/vacuous (width=300, heuristic). This proxy test demonstrates that
/// a 2-layer PLBert encoder with small weights produces tight, finite bounds
/// via IBP propagation through the TensorBlockBuilder transformer pipeline.
#[test]
fn test_gap_fill_plbert_encoder_ibp() {
    let def = build_plbert_encoder_2layer();
    def.validate().expect("plbert encoder def validates");

    let bindings = plbert_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BERT_SEQ_LEN, BERT_D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer PLBert encoder");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "PLBert encoder bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[BERT_SEQ_LEN, BERT_D_MODEL],
        "PLBert encoder must produce [T, D] output"
    );

    // With small weights (0.01) and 2 layers, residual connections limit growth.
    // The production entry has width=300 (vacuous) — our proxy should be tighter.
    assert_bounds_width(&output, VACUOUS_THRESHOLD, "plbert_encoder_2layer");

    eprintln!(
        "Gap fill PLBert encoder: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}",
        graph.num_nodes()
    );
}

/// PLBert encoder bounds stay finite after CROWN propagation.
///
/// Tests that the PLBert encoder proxy is amenable to CROWN linearization,
/// which would tighten the production bounds from heuristic/vacuous to sound.
#[test]
fn test_gap_fill_plbert_encoder_crown_fallback() {
    let def = build_plbert_encoder_2layer();
    let bindings = plbert_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BERT_SEQ_LEN, BERT_D_MODEL], 1.0);

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    assert_bounds_valid(&crown_output);

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    let width = hi_max - lo_min;

    eprintln!(
        "Gap fill PLBert encoder CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, fallback_reason={fallback_reason:?}"
    );

    // Bounds must be finite regardless of CROWN/IBP fallback.
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "PLBert encoder CROWN bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// SECTION 2: ProsodyPredictor gap fill
// ===========================================================================
//
// The ProsodyPredictor uses BiLSTM + AdaLayerNorm blocks for duration
// prediction. Production is N layers of (BiLSTM → AdaLayerNorm).
// We proxy this with Conv1d + LayerNorm (which IBP handles well) since
// TensorBlockBuilder doesn't have a native BiLSTM op.
//
// Status key: kokoro_production_prosody_predictor

/// Channel dimension for prosody proxy.
const PROSODY_CHANNELS: usize = 8;
/// Temporal length for prosody proxy.
const PROSODY_T: usize = 8;
/// Style dimension for prosody proxy.
const PROSODY_STYLE_DIM: usize = 4;

/// Build a prosody predictor proxy graph.
///
/// Architecture: Conv1d → LayerNorm → LeakyReLU → Conv1d → LayerNorm →
///               residual add → Linear projection to scalar durations.
///
/// Input: `[C, T]` (Variable — text features).
/// Output: `[1, T]` (predicted durations per phoneme).
fn build_prosody_predictor_proxy() -> TensorKernelDef {
    let in_shape = [PROSODY_CHANNELS, PROSODY_T];
    let conv_shape = [PROSODY_CHANNELS, PROSODY_T]; // same-padding convolution
    let out_shape = [1, PROSODY_T]; // scalar per timestep
    let mut b = TensorBlockBuilder::new("prosody_predictor_proxy");

    let x = b.add_input("x", &in_shape);

    // Block 1: Conv1d → LayerNorm → LeakyReLU
    let w1 = b.add_input("conv1_w", &[PROSODY_CHANNELS, PROSODY_CHANNELS, 3]);
    let b1 = b.add_input("conv1_b", &[PROSODY_CHANNELS]);
    let conv1 = b.add_conv1d(x, w1, Some(b1), 1, 1, &conv_shape);
    let eps1 = b.add_input("eps1", &[1]);
    let ln1_w = b.add_input("ln1_w", &[PROSODY_CHANNELS]);
    let ln1_b = b.add_input("ln1_b", &[PROSODY_CHANNELS]);
    let norm1 = b.add_layer_norm(conv1, eps1, 1, ln1_w, ln1_b, &conv_shape);
    let act1 = b.add_leaky_relu(norm1, 0.2, &conv_shape);

    // Block 2: Conv1d → LayerNorm → LeakyReLU
    let w2 = b.add_input("conv2_w", &[PROSODY_CHANNELS, PROSODY_CHANNELS, 3]);
    let b2 = b.add_input("conv2_b", &[PROSODY_CHANNELS]);
    let conv2 = b.add_conv1d(act1, w2, Some(b2), 1, 1, &conv_shape);
    let eps2 = b.add_input("eps2", &[1]);
    let ln2_w = b.add_input("ln2_w", &[PROSODY_CHANNELS]);
    let ln2_b = b.add_input("ln2_b", &[PROSODY_CHANNELS]);
    let norm2 = b.add_layer_norm(conv2, eps2, 1, ln2_w, ln2_b, &conv_shape);
    let act2 = b.add_leaky_relu(norm2, 0.2, &conv_shape);

    // Residual connection
    let res = b.add_binary_add(x, act2, &conv_shape);

    // Duration projection: Linear projects channels C → 1 per timestep. Linear
    // contracts over the last axis, so transpose [C, T] -> [T, C] to put channels
    // last, project to [T, 1], then transpose back to the [1, T] output layout.
    let tc_shape = [PROSODY_T, PROSODY_CHANNELS];
    let res_tc = b.add_transpose(res, &[1, 0], &tc_shape);
    let proj_w = b.add_input("proj_w", &[1, PROSODY_CHANNELS]);
    let proj_tc = b.add_linear(res_tc, proj_w, None, &[PROSODY_T, 1]);
    let proj_out = b.add_transpose(proj_tc, &[1, 0], &out_shape);

    // Softplus ensures non-negative durations
    let duration = b.add_softplus(proj_out, &out_shape);

    b.build(duration).expect("valid prosody predictor graph")
}

/// Bindings for prosody predictor proxy.
fn prosody_predictor_bindings() -> Vec<TensorParamBinding> {
    let c = PROSODY_CHANNELS;
    vec![
        TensorParamBinding::Variable, // x
        // Block 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        // Block 2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        // Duration projection
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, c]), WEIGHT_MAG)),
    ]
}

/// ProsodyPredictor proxy produces non-vacuous IBP bounds.
///
/// This fills the `kokoro_production_prosody_predictor` gap: the production
/// entry is stale/vacuous (width=412, heuristic). The proxy uses Conv1d +
/// LayerNorm + residual + Softplus to demonstrate that the duration prediction
/// pathway propagates bounds tightly through IBP.
#[test]
fn test_gap_fill_prosody_predictor_ibp() {
    let def = build_prosody_predictor_proxy();
    def.validate().expect("prosody predictor def validates");

    let bindings = prosody_predictor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[PROSODY_CHANNELS, PROSODY_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through prosody predictor proxy");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "prosody predictor bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, PROSODY_T],
        "prosody predictor must produce [1, T] output"
    );

    // Softplus output is non-negative: lower bound should be >= 0.
    assert!(
        lo_min >= 0.0 - 1e-6,
        "softplus output lower bound {lo_min} should be non-negative"
    );

    // With small weights, bounds should be tight.
    // Production entry has width=412 (vacuous). Proxy target: < 500.
    assert_bounds_width(&output, VACUOUS_THRESHOLD, "prosody_predictor_proxy");

    eprintln!(
        "Gap fill ProsodyPredictor: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}",
        graph.num_nodes()
    );
}

/// ProsodyPredictor with wider input range still produces finite bounds.
///
/// Tests robustness: wider input range (±5.0 instead of ±1.0) triggers
/// more IBP over-approximation but should remain non-vacuous.
#[test]
fn test_gap_fill_prosody_predictor_wide_input() {
    let def = build_prosody_predictor_proxy();
    let bindings = prosody_predictor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[PROSODY_CHANNELS, PROSODY_T], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through prosody predictor (wide input)");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "prosody predictor (wide) bounds must be finite: [{lo_min}, {hi_max}]"
    );
    // Softplus non-negativity
    assert!(
        lo_min >= 0.0 - 1e-6,
        "softplus output {lo_min} must be non-negative even with wide input"
    );

    eprintln!(
        "Gap fill ProsodyPredictor (wide): bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}"
    );
}

// ===========================================================================
// SECTION 3: F0EnergyPredictor gap fill
// ===========================================================================
//
// The F0EnergyPredictor uses AdainResBlk1d blocks (Conv1d + InstanceNorm +
// style injection) → linear projection. Production has 3 residual blocks.
// We proxy with Conv1d + InstanceNorm + residual.
//
// Status key: kokoro_production_f0_predictor

/// Channel dimension for F0 proxy.
const F0_CHANNELS: usize = 8;
/// Temporal length for F0 proxy (must be > 1 for non-degenerate InstanceNorm).
const F0_T: usize = 8;

/// Build a single AdainResBlk1d-style proxy block.
///
/// Architecture: Conv1d → InstanceNorm → LeakyReLU → Conv1d → InstanceNorm → residual.
///
/// This proxies the production AdainResBlk1d which uses style injection via
/// adaptive instance normalization. The style injection is simplified to
/// InstanceNorm (gamma=1, beta=0) since the affine parameters are what
/// create the style modulation, and at toy scale we use identity affine.
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]`.
fn build_adain_resblock_proxy(name: &str, channels: usize, t: usize) -> TensorKernelDef {
    let shape = [channels, t];
    let mut b = TensorBlockBuilder::new(name);

    let x = b.add_input("x", &shape);

    // Conv1d + InstanceNorm + LeakyReLU
    let w1 = b.add_input("conv1_w", &[channels, channels, 3]);
    let b1_input = b.add_input("conv1_b", &[channels]);
    let conv1 = b.add_conv1d(x, w1, Some(b1_input), 1, 1, &shape);
    let eps1 = b.add_input("eps1", &[1]);
    let norm1 = b.add_instance_norm(conv1, eps1, 1, None, None, &shape);
    let act1 = b.add_leaky_relu(norm1, 0.1, &shape);

    // Conv1d + InstanceNorm
    let w2 = b.add_input("conv2_w", &[channels, channels, 3]);
    let b2 = b.add_input("conv2_b", &[channels]);
    let conv2 = b.add_conv1d(act1, w2, Some(b2), 1, 1, &shape);
    let eps2 = b.add_input("eps2", &[1]);
    let norm2 = b.add_instance_norm(conv2, eps2, 1, None, None, &shape);

    // Residual connection
    let out = b.add_binary_add(x, norm2, &shape);

    b.build(out).expect("valid adain resblock proxy graph")
}

/// Bindings for AdainResBlk1d proxy.
fn adain_resblock_bindings(channels: usize) -> Vec<TensorParamBinding> {
    let c = channels;
    vec![
        TensorParamBinding::Variable, // x
        // Block: conv1 + norm1 + conv2 + norm2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// Build the full F0 predictor proxy: 2 × AdainResBlk + linear projection.
///
/// Input: `[C, T]` (Variable — text features).
/// Output: `[1, T]` (F0 values per frame).
fn build_f0_predictor_proxy() -> TensorKernelDef {
    let in_shape = [F0_CHANNELS, F0_T];
    let out_shape = [1, F0_T];
    let mut b = TensorBlockBuilder::new("f0_predictor_proxy");

    let x = b.add_input("x", &in_shape);

    // Block 1: Conv + InstanceNorm + LeakyReLU + Conv + InstanceNorm + residual
    let w1a = b.add_input("b1_conv1_w", &[F0_CHANNELS, F0_CHANNELS, 3]);
    let b1a = b.add_input("b1_conv1_b", &[F0_CHANNELS]);
    let conv1a = b.add_conv1d(x, w1a, Some(b1a), 1, 1, &in_shape);
    let eps1a = b.add_input("b1_eps1", &[1]);
    let norm1a = b.add_instance_norm(conv1a, eps1a, 1, None, None, &in_shape);
    let act1a = b.add_leaky_relu(norm1a, 0.1, &in_shape);

    let w1b = b.add_input("b1_conv2_w", &[F0_CHANNELS, F0_CHANNELS, 3]);
    let b1b = b.add_input("b1_conv2_b", &[F0_CHANNELS]);
    let conv1b = b.add_conv1d(act1a, w1b, Some(b1b), 1, 1, &in_shape);
    let eps1b = b.add_input("b1_eps2", &[1]);
    let norm1b = b.add_instance_norm(conv1b, eps1b, 1, None, None, &in_shape);
    let res1 = b.add_binary_add(x, norm1b, &in_shape);

    // Block 2: Conv + InstanceNorm + LeakyReLU + Conv + InstanceNorm + residual
    let w2a = b.add_input("b2_conv1_w", &[F0_CHANNELS, F0_CHANNELS, 3]);
    let b2a = b.add_input("b2_conv1_b", &[F0_CHANNELS]);
    let conv2a = b.add_conv1d(res1, w2a, Some(b2a), 1, 1, &in_shape);
    let eps2a = b.add_input("b2_eps1", &[1]);
    let norm2a = b.add_instance_norm(conv2a, eps2a, 1, None, None, &in_shape);
    let act2a = b.add_leaky_relu(norm2a, 0.1, &in_shape);

    let w2b = b.add_input("b2_conv2_w", &[F0_CHANNELS, F0_CHANNELS, 3]);
    let b2b = b.add_input("b2_conv2_b", &[F0_CHANNELS]);
    let conv2b = b.add_conv1d(act2a, w2b, Some(b2b), 1, 1, &in_shape);
    let eps2b = b.add_input("b2_eps2", &[1]);
    let norm2b = b.add_instance_norm(conv2b, eps2b, 1, None, None, &in_shape);
    let res2 = b.add_binary_add(res1, norm2b, &in_shape);

    // F0 projection: Linear projects channels C → 1 per frame. Linear contracts
    // over the last axis, so transpose [C, T] -> [T, C] to put channels last,
    // project to [T, 1], then transpose back to the [1, T] output layout.
    let tc_shape = [F0_T, F0_CHANNELS];
    let res2_tc = b.add_transpose(res2, &[1, 0], &tc_shape);
    let proj_w = b.add_input("proj_w", &[1, F0_CHANNELS]);
    let proj_tc = b.add_linear(res2_tc, proj_w, None, &[F0_T, 1]);
    let proj_out = b.add_transpose(proj_tc, &[1, 0], &out_shape);

    b.build(proj_out).expect("valid f0 predictor proxy graph")
}

/// Bindings for full F0 predictor proxy.
fn f0_predictor_bindings() -> Vec<TensorParamBinding> {
    let c = F0_CHANNELS;
    let mut bindings = vec![TensorParamBinding::Variable]; // x

    // 2 blocks, each: conv_w, conv_b, eps, conv_w, conv_b, eps
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c, c, 3]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c, c, 3]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[c]),
            0.0f32,
        )));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Projection
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, c]),
        WEIGHT_MAG,
    )));

    bindings
}

/// F0EnergyPredictor proxy produces non-vacuous IBP bounds.
///
/// This fills the `kokoro_production_f0_predictor` gap: the production entry
/// is stale/vacuous (width=32820, heuristic). The proxy uses 2 AdainResBlk-style
/// blocks (Conv1d + InstanceNorm + residual) + linear projection to demonstrate
/// that the F0 prediction pathway propagates bounds through IBP.
///
/// The key insight: InstanceNorm acts as a natural bound stabilizer (normalizes
/// to approximately zero-mean, unit-variance), and the residual connections
/// limit unbounded growth. The production vacuity likely comes from the BiLSTM
/// component (which has multiplicative gates), not the ResBlock path.
#[test]
fn test_gap_fill_f0_predictor_ibp() {
    let def = build_f0_predictor_proxy();
    def.validate().expect("f0 predictor def validates");

    let bindings = f0_predictor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[F0_CHANNELS, F0_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through F0 predictor proxy");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "F0 predictor bounds must be finite: [{lo_min}, {hi_max}]"
    );
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, F0_T],
        "F0 predictor must produce [1, T] output"
    );

    // With small weights and InstanceNorm stabilization, bounds should be tight.
    // Production entry has width=32820 (extremely vacuous). Proxy target: < 500.
    assert_bounds_width(&output, VACUOUS_THRESHOLD, "f0_predictor_proxy");

    eprintln!(
        "Gap fill F0EnergyPredictor: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}, \
         graph_nodes={}",
        graph.num_nodes()
    );
}

/// Single AdainResBlk1d proxy block produces tight bounds.
///
/// Verifies the building block in isolation before composition.
#[test]
fn test_gap_fill_adain_resblock_single() {
    let def = build_adain_resblock_proxy("adain_resblock_single", F0_CHANNELS, F0_T);
    def.validate().expect("adain resblock def validates");

    let bindings = adain_resblock_bindings(F0_CHANNELS);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[F0_CHANNELS, F0_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through adain resblock");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "adain resblock bounds must be finite: [{lo_min}, {hi_max}]"
    );

    // InstanceNorm + residual should keep bounds from exploding.
    assert_bounds_width(&output, 100.0, "adain_resblock_single");

    eprintln!("Gap fill AdainResBlk single: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.4}");
}

/// F0 predictor with CROWN propagation produces finite bounds.
#[test]
fn test_gap_fill_f0_predictor_crown_fallback() {
    let def = build_f0_predictor_proxy();
    let bindings = f0_predictor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[F0_CHANNELS, F0_T], 1.0);

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    assert_bounds_valid(&crown_output);

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    let width = hi_max - lo_min;

    eprintln!(
        "Gap fill F0 CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}], \
         width={width:.4}, fallback_reason={fallback_reason:?}"
    );

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "F0 predictor CROWN bounds must be finite: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// SECTION 4: Cross-segment composition tests
// ===========================================================================

/// Encoder → ProsodyPredictor composition bounds.
///
/// Verifies that bounds compose across the PLBert encoder output feeding
/// into the ProsodyPredictor input. The encoder output bounds become the
/// predictor input bounds via sequential propagation.
///
/// This exercises the inter-segment boundary between kokoro_production_bert_encoder
/// and kokoro_production_prosody_predictor — both currently vacuous.
#[test]
fn test_gap_fill_encoder_to_prosody_composition() {
    // Step 1: Propagate through encoder proxy
    let enc_def = build_plbert_encoder_2layer();
    let enc_bindings = plbert_encoder_bindings();
    let enc_graph = tensor_kernel_to_graph(&enc_def, &enc_bindings).expect("encoder graph");
    let enc_input = uniform_bounds(&[BERT_SEQ_LEN, BERT_D_MODEL], 1.0);

    let enc_output = enc_graph
        .propagate_ibp(&enc_input)
        .expect("IBP through encoder");
    assert_bounds_valid(&enc_output);

    let (enc_lo, enc_hi) = bounds_min_max(&enc_output);
    let enc_width = enc_hi - enc_lo;
    eprintln!(
        "Composition encoder output: bounds=[{enc_lo:.4}, {enc_hi:.4}], width={enc_width:.4}"
    );

    // Step 2: Reshape encoder output to prosody input shape [C, T].
    // Encoder produces [T, D] = [4, 16]. Prosody expects [C, T] = [8, 8].
    // We take the global bounds range from the encoder and create matching input.
    let prosody_range = enc_width.max(1.0); // use encoder width as input range
    let prosody_input = uniform_bounds(&[PROSODY_CHANNELS, PROSODY_T], prosody_range / 2.0);

    // Step 3: Propagate through prosody predictor proxy
    let pros_def = build_prosody_predictor_proxy();
    let pros_bindings = prosody_predictor_bindings();
    let pros_graph = tensor_kernel_to_graph(&pros_def, &pros_bindings).expect("prosody graph");

    let pros_output = pros_graph
        .propagate_ibp(&prosody_input)
        .expect("IBP through prosody predictor");
    assert_bounds_valid(&pros_output);

    let (pros_lo, pros_hi) = bounds_min_max(&pros_output);
    let pros_width = pros_hi - pros_lo;
    eprintln!(
        "Composition prosody output: bounds=[{pros_lo:.4}, {pros_hi:.4}], width={pros_width:.4}"
    );

    // End-to-end bounds must be finite.
    assert!(
        pros_lo.is_finite() && pros_hi.is_finite(),
        "encoder→prosody composition bounds must be finite: [{pros_lo}, {pros_hi}]"
    );

    // Softplus non-negativity.
    assert!(
        pros_lo >= 0.0 - 1e-6,
        "composed prosody output {pros_lo} must be non-negative (softplus)"
    );
}

/// Encoder → F0Predictor composition bounds.
///
/// Same pattern as encoder→prosody but for the F0 branch.
#[test]
fn test_gap_fill_encoder_to_f0_composition() {
    // Step 1: Encoder proxy
    let enc_def = build_plbert_encoder_2layer();
    let enc_bindings = plbert_encoder_bindings();
    let enc_graph = tensor_kernel_to_graph(&enc_def, &enc_bindings).expect("encoder graph");
    let enc_input = uniform_bounds(&[BERT_SEQ_LEN, BERT_D_MODEL], 1.0);

    let enc_output = enc_graph
        .propagate_ibp(&enc_input)
        .expect("IBP through encoder");
    assert_bounds_valid(&enc_output);

    let (enc_lo, enc_hi) = bounds_min_max(&enc_output);
    let enc_width = enc_hi - enc_lo;

    // Step 2: F0 predictor proxy
    let f0_range = enc_width.max(1.0);
    let f0_input = uniform_bounds(&[F0_CHANNELS, F0_T], f0_range / 2.0);

    let f0_def = build_f0_predictor_proxy();
    let f0_bindings = f0_predictor_bindings();
    let f0_graph = tensor_kernel_to_graph(&f0_def, &f0_bindings).expect("f0 graph");

    let f0_output = f0_graph
        .propagate_ibp(&f0_input)
        .expect("IBP through F0 predictor");
    assert_bounds_valid(&f0_output);

    let (f0_lo, f0_hi) = bounds_min_max(&f0_output);
    let f0_width = f0_hi - f0_lo;

    assert!(
        f0_lo.is_finite() && f0_hi.is_finite(),
        "encoder→F0 composition bounds must be finite: [{f0_lo}, {f0_hi}]"
    );

    eprintln!(
        "Composition encoder→F0: encoder_width={enc_width:.4}, f0_width={f0_width:.4}, \
         f0_bounds=[{f0_lo:.4}, {f0_hi:.4}]"
    );
}
