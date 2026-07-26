// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep NY compose tests for Table Transformer (DETR) subgraphs.
//!
//! These tests verify bounds propagation through intermediate-depth compositions
//! of the Table Transformer pipeline that bridge the gap between existing
//! sub-block tests and full end-to-end verification. Specifically targets
//! the heuristic entries in `nn_verify_status_dpdf.json` by decomposing
//! full pipeline compositions into verifiable intermediate stages:
//!
//! 1. **ResNet 2-stage with BN+ReLU** -- Two Conv-BN-ReLU stages testing
//!    bounds through cascaded spatial downsampling (IBP + CROWN).
//!
//! 2. **Encoder layer tight-input CROWN** -- Single DETR encoder layer
//!    with narrow +-0.1 input bounds for CROWN precision analysis (CROWN).
//!
//! 3. **Decoder self+cross attention** -- Single decoder layer: self-attn +
//!    cross-attn + FFN, testing query-memory interaction bounds (IBP + CROWN).
//!
//! 4. **3-layer encoder stack** -- Depth composition with widening analysis.
//!    Measures bounds growth beyond 2-layer tested in compose_table_transformer
//!    (IBP).
//!
//! 5. **Encoder + LayerNorm + classification** -- Encoder output through
//!    final LayerNorm + sigmoid cls head. Bridge from encoder to heads (IBP + CROWN).
//!
//! 6. **Encoder + decoder + sigmoid heads** -- Full DETR encode-decode with
//!    dual sigmoid heads at smaller depth than full pipeline (IBP).
//!
//! 7. **Widening analysis** -- 1-layer vs 3-layer encoder IBP width
//!    comparison. Quantifies bounds growth through depth (IBP).
//!
//! 8. **Structure recognition heads** -- Decoder output through table
//!    structure-specific heads: row sigmoid, column sigmoid, span sigmoid.
//!    Verifies all structure outputs bounded in (0, 1) (IBP + CROWN).
//!
//! 9. **ResNet backbone + input projection** -- Full backbone output
//!    projected to transformer dimension via 1x1 conv (IBP).
//!
//! 10. **Cross-attention with learned queries** -- Decoder cross-attention
//!     with learned object queries and encoder memory (IBP + CROWN).
//!
//! 11. **Verify-and-record: encoder + cls** -- Records to status file (IBP + CROWN).
//!
//! 12. **Verify-and-record: decoder + structure** -- Records to status file (IBP + CROWN).
//!
//! Architecture references:
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - ResNet (He et al. 2016): Residual network backbone
//!
//! Dimensions: HIDDEN_DIM=16, SEQ_LEN=4, NUM_HEADS=4 (small for fast verify).
//! All tests use IbpValidated soundness mode per nn engineering rules.
//!
//! Part of #4273: deep NY compose tests for Table Transformer.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 16;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
const FFN_DIM: usize = 64;
const NUM_QUERIES: usize = 4;
const NUM_CLASSES: usize = 6;
const SPATIAL: usize = 4; // spatial features (2x2 -> flattened to 4)
const IN_CHANNELS: usize = 3;
const CONV_CHANNELS: usize = 8;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

/// Adds one DETR encoder layer: LN -> self-attention -> residual -> LN -> FFN -> residual.
fn add_encoder_layer(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention LayerNorm
    let ln1_w = b.add_input(&format!("enc{idx}_ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("enc{idx}_ln1_b"), &[HIDDEN_DIM]);
    let ln1_eps = b.add_input(&format!("enc{idx}_ln1_eps"), &[1]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // Self-attention
    let q_w = b.add_input(&format!("enc{idx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("enc{idx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("enc{idx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("enc{idx}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN LayerNorm
    let ln2_w = b.add_input(&format!("enc{idx}_ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("enc{idx}_ln2_b"), &[HIDDEN_DIM]);
    let ln2_eps = b.add_input(&format!("enc{idx}_ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> ReLU -> Linear
    let ff1_w = b.add_input(&format!("enc{idx}_ff1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ff1_b = b.add_input(&format!("enc{idx}_ff1_b"), &[FFN_DIM]);
    let ff2_w = b.add_input(&format!("enc{idx}_ff2_w"), &[HIDDEN_DIM, FFN_DIM]);
    let ff2_b = b.add_input(&format!("enc{idx}_ff2_b"), &[HIDDEN_DIM]);

    let h = b.add_linear(normed2, ff1_w, Some(ff1_b), &ffn_shape);
    let h_relu = b.add_relu(h, &ffn_shape);
    let ffn_out = b.add_linear(h_relu, ff2_w, Some(ff2_b), &shape);
    let res2 = b.add_binary_add(res1, ffn_out, &shape);

    // Push bindings (14 params per layer)
    let ww = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln1_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln1_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(ww)); // out_w
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln2_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln2_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // ff1_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM]))); // ff1_b
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ]))); // ff2_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ff2_b

    res2
}

/// Adds one DETR decoder layer: self-attn + cross-attn + FFN with residuals.
fn add_decoder_layer(
    b: &mut TensorBlockBuilder,
    queries: TensorNodeId,
    memory: TensorNodeId,
    idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> TensorNodeId {
    let q_shape = [NUM_QUERIES, HIDDEN_DIM];
    let m_shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [NUM_QUERIES, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Self-attention on queries
    let ln1_w = b.add_input(&format!("dec{idx}_ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("dec{idx}_ln1_b"), &[HIDDEN_DIM]);
    let ln1_eps = b.add_input(&format!("dec{idx}_ln1_eps"), &[1]);
    let n1 = b.add_layer_norm(queries, ln1_eps, 1, ln1_w, ln1_b, &q_shape);

    let sq_w = b.add_input(&format!("dec{idx}_sq_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sk_w = b.add_input(&format!("dec{idx}_sk_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sv_w = b.add_input(&format!("dec{idx}_sv_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let so_w = b.add_input(&format!("dec{idx}_so_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(n1, sq_w, None, &q_shape);
    let sk = b.add_linear(n1, sk_w, None, &q_shape);
    let sv = b.add_linear(n1, sv_w, None, &q_shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Standard, Some(scale), &q_shape);
    let sa_out = b.add_linear(sa, so_w, None, &q_shape);
    let r1 = b.add_binary_add(queries, sa_out, &q_shape);

    // Cross-attention: queries attend to encoder memory
    let ln2_w = b.add_input(&format!("dec{idx}_ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("dec{idx}_ln2_b"), &[HIDDEN_DIM]);
    let ln2_eps = b.add_input(&format!("dec{idx}_ln2_eps"), &[1]);
    let n2 = b.add_layer_norm(r1, ln2_eps, 1, ln2_w, ln2_b, &q_shape);

    let cq_w = b.add_input(&format!("dec{idx}_cq_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ck_w = b.add_input(&format!("dec{idx}_ck_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let cv_w = b.add_input(&format!("dec{idx}_cv_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let co_w = b.add_input(&format!("dec{idx}_co_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let cq = b.add_linear(n2, cq_w, None, &q_shape);
    let ck = b.add_linear(memory, ck_w, None, &m_shape);
    let cv = b.add_linear(memory, cv_w, None, &m_shape);
    let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &q_shape);
    let ca_out = b.add_linear(ca, co_w, None, &q_shape);
    let r2 = b.add_binary_add(r1, ca_out, &q_shape);

    // FFN
    let ln3_w = b.add_input(&format!("dec{idx}_ln3_w"), &[HIDDEN_DIM]);
    let ln3_b = b.add_input(&format!("dec{idx}_ln3_b"), &[HIDDEN_DIM]);
    let ln3_eps = b.add_input(&format!("dec{idx}_ln3_eps"), &[1]);
    let n3 = b.add_layer_norm(r2, ln3_eps, 1, ln3_w, ln3_b, &q_shape);

    let ff1_w = b.add_input(&format!("dec{idx}_ff1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ff1_b = b.add_input(&format!("dec{idx}_ff1_b"), &[FFN_DIM]);
    let ff2_w = b.add_input(&format!("dec{idx}_ff2_w"), &[HIDDEN_DIM, FFN_DIM]);
    let ff2_b = b.add_input(&format!("dec{idx}_ff2_b"), &[HIDDEN_DIM]);

    let h = b.add_linear(n3, ff1_w, Some(ff1_b), &ffn_shape);
    let h_relu = b.add_relu(h, &ffn_shape);
    let ffn_out = b.add_linear(h_relu, ff2_w, Some(ff2_b), &q_shape);
    let r3 = b.add_binary_add(r2, ffn_out, &q_shape);

    // Push bindings (27 params per decoder layer)
    let ww = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    // Self-attention LN + QKVO
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln1_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln1_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // sq_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // sk_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // sv_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // so_w
                                                                   // Cross-attention LN + QKVO
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln2_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln2_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // cq_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // ck_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // cv_w
    bindings.push(TensorParamBinding::ConstantTensor(ww)); // co_w
                                                           // FFN LN + weights
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln3_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln3_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln3_eps
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // ff1_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM]))); // ff1_b
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ]))); // ff2_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ff2_b

    r3
}

// ===========================================================================
// 1. ResNet 2-stage with BN+ReLU
// ===========================================================================

/// ResNet backbone: 2 Conv-BN-ReLU stages with stride-2 downsampling.
fn build_resnet_2stage_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_resnet_2stage");

    // Stage 1: Conv-BN-ReLU, no spatial change (same padding modeled as no stride)
    let input = b.add_input("features", &[IN_CHANNELS, SPATIAL, SPATIAL]);
    let c1_w = b.add_input("conv1_w", &[CONV_CHANNELS, IN_CHANNELS, 3, 3]);
    let c1_b = b.add_input("conv1_b", &[CONV_CHANNELS]);
    let bn1_mean = b.add_input("bn1_mean", &[CONV_CHANNELS]);
    let bn1_var = b.add_input("bn1_var", &[CONV_CHANNELS]);
    let bn1_w = b.add_input("bn1_w", &[CONV_CHANNELS]);
    let bn1_b = b.add_input("bn1_b", &[CONV_CHANNELS]);
    let bn1_eps = b.add_input("bn1_eps", &[1]);

    let s1_out = [CONV_CHANNELS, SPATIAL, SPATIAL];
    let conv1 = b.add_conv2d(input, c1_w, Some(c1_b), 1, 1, 1, 1, &s1_out);
    let bn1 = b.add_batch_norm(conv1, bn1_mean, bn1_var, bn1_w, bn1_b, bn1_eps, &s1_out);
    let relu1 = b.add_relu(bn1, &s1_out);

    // Stage 2: Conv-BN-ReLU with stride-2 spatial downsampling
    let s2_spatial = SPATIAL / 2;
    let s2_out = [HIDDEN_DIM, s2_spatial, s2_spatial];
    let c2_w = b.add_input("conv2_w", &[HIDDEN_DIM, CONV_CHANNELS, 3, 3]);
    let c2_b = b.add_input("conv2_b", &[HIDDEN_DIM]);
    let bn2_mean = b.add_input("bn2_mean", &[HIDDEN_DIM]);
    let bn2_var = b.add_input("bn2_var", &[HIDDEN_DIM]);
    let bn2_w = b.add_input("bn2_w", &[HIDDEN_DIM]);
    let bn2_b = b.add_input("bn2_b", &[HIDDEN_DIM]);
    let bn2_eps = b.add_input("bn2_eps", &[1]);

    let conv2 = b.add_conv2d(relu1, c2_w, Some(c2_b), 2, 2, 1, 1, &s2_out);
    let bn2 = b.add_batch_norm(conv2, bn2_mean, bn2_var, bn2_w, bn2_b, bn2_eps, &s2_out);
    let relu2 = b.add_relu(bn2, &s2_out);

    b.build(relu2).expect("valid resnet 2-stage kernel")
}

fn resnet_2stage_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(w(&[CONV_CHANNELS, IN_CHANNELS, 3, 3])), // conv1_w
        TensorParamBinding::ConstantTensor(zeros(&[CONV_CHANNELS])), // conv1_b
        TensorParamBinding::ConstantTensor(zeros(&[CONV_CHANNELS])), // bn1_mean
        TensorParamBinding::ConstantTensor(ones(&[CONV_CHANNELS])), // bn1_var
        TensorParamBinding::ConstantTensor(ones(&[CONV_CHANNELS])), // bn1_w
        TensorParamBinding::ConstantTensor(zeros(&[CONV_CHANNELS])), // bn1_b
        TensorParamBinding::ConstantScalar(1e-5), // bn1_eps
        TensorParamBinding::ConstantTensor(w(&[HIDDEN_DIM, CONV_CHANNELS, 3, 3])), // conv2_w
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])), // conv2_b
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])), // bn2_mean
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])), // bn2_var
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])), // bn2_w
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])), // bn2_b
        TensorParamBinding::ConstantScalar(1e-5), // bn2_eps
    ]
}

#[test]
fn test_tt_deep_resnet_2stage_ibp() {
    let def = build_resnet_2stage_kernel();
    let bindings = resnet_2stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[IN_CHANNELS, SPATIAL, SPATIAL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("ResNet 2-stage IBP: [{lo}, {hi}]");
    // ReLU ensures lower >= 0
    assert!(lo >= -1e-6, "ReLU output lower should be >= 0, got {lo}");
}

#[test]
fn test_tt_deep_resnet_2stage_crown() {
    let def = build_resnet_2stage_kernel();
    let bindings = resnet_2stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[IN_CHANNELS, SPATIAL, SPATIAL], 1.0);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("ResNet 2-stage CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 2. Encoder layer tight-input CROWN
// ===========================================================================

fn build_single_encoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_single_enc");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();
    let out = add_encoder_layer(&mut b, input, 0, &mut bindings_unused);
    b.build(out).expect("valid single encoder layer kernel")
}

fn single_encoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    let ww = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    // 14 params from add_encoder_layer
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln1_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln1_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(ww)); // out_w
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM]))); // ln2_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ln2_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ]))); // ff1_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM]))); // ff1_b
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ]))); // ff2_w
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM]))); // ff2_b
    bindings
}

#[test]
fn test_tt_deep_encoder_tight_input_crown() {
    let def = build_single_encoder_layer_kernel();
    let bindings = single_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Narrow +-0.1 bounds for CROWN precision analysis
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Encoder tight-input CROWN (method={method:?}): [{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 3. Decoder self+cross attention (single layer)
// ===========================================================================

fn build_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_decoder_layer");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let memory = b.add_input("memory", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();
    let out = add_decoder_layer(&mut b, queries, memory, 0, &mut bindings_unused);
    b.build(out).expect("valid decoder layer kernel")
}

fn decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,                                  // queries
        TensorParamBinding::ConstantTensor(w(&[SEQ_LEN, HIDDEN_DIM])), // memory (fixed)
    ];
    let ww = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    // Self-attention LN + QKVO
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    // Cross-attention LN + QKVO
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww));
    // FFN LN + weights
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings
}

#[test]
fn test_tt_deep_decoder_layer_ibp() {
    let def = build_decoder_layer_kernel();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Decoder layer IBP: [{lo}, {hi}]");
}

#[test]
fn test_tt_deep_decoder_layer_crown() {
    let def = build_decoder_layer_kernel();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Decoder layer CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 4. 3-layer encoder stack
// ===========================================================================

fn build_3layer_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_3layer_enc");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();
    let e1 = add_encoder_layer(&mut b, input, 0, &mut bindings_unused);
    let e2 = add_encoder_layer(&mut b, e1, 1, &mut bindings_unused);
    let e3 = add_encoder_layer(&mut b, e2, 2, &mut bindings_unused);
    b.build(e3).expect("valid 3-layer encoder kernel")
}

fn three_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..3 {
        let ww = w(&[HIDDEN_DIM, HIDDEN_DIM]);
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ww));
        bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            FFN_DIM, HIDDEN_DIM,
        ])));
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])));
        bindings.push(TensorParamBinding::ConstantTensor(w(&[
            HIDDEN_DIM, FFN_DIM,
        ])));
        bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    }
    bindings
}

#[test]
fn test_tt_deep_3layer_encoder_ibp() {
    let def = build_3layer_encoder_kernel();
    let bindings = three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("3-layer encoder IBP: [{lo}, {hi}]");
}

// ===========================================================================
// 5. Encoder + LayerNorm + classification sigmoid
// ===========================================================================

fn build_encoder_cls_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_enc_cls_head");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();
    let enc_out = add_encoder_layer(&mut b, input, 0, &mut bindings_unused);

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("final_ln_eps", &[1]);
    let normed = b.add_layer_norm(enc_out, ln_eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_shape = [SEQ_LEN, NUM_CLASSES];
    let logits = b.add_linear(normed, cls_w, Some(cls_b), &cls_shape);
    let probs = b.add_sigmoid(logits, &cls_shape);

    b.build(probs).expect("valid encoder + cls head kernel")
}

fn encoder_cls_head_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = single_encoder_bindings();
    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    // Cls head
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])));
    bindings
}

#[test]
fn test_tt_deep_encoder_cls_head_ibp() {
    let def = build_encoder_cls_head_kernel();
    let bindings = encoder_cls_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Encoder + cls head IBP: [{lo}, {hi}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo >= 0.0 - 1e-6, "sigmoid lower should be >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-6, "sigmoid upper should be <= 1, got {hi}");
}

#[test]
fn test_tt_deep_encoder_cls_head_crown() {
    let def = build_encoder_cls_head_kernel();
    let bindings = encoder_cls_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Encoder + cls head CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 6. Encoder + decoder + sigmoid heads
// ===========================================================================

fn build_enc_dec_heads_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_enc_dec_heads");
    let features = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let mut bindings_unused = Vec::new();

    let enc_out = add_encoder_layer(&mut b, features, 0, &mut bindings_unused);
    let dec_out = add_decoder_layer(&mut b, queries, enc_out, 0, &mut bindings_unused);

    // Dual sigmoid heads
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let box_w = b.add_input("box_w", &[4, HIDDEN_DIM]);
    let cls_shape = [NUM_QUERIES, NUM_CLASSES];
    let box_shape = [NUM_QUERIES, 4];

    let cls_logits = b.add_linear(dec_out, cls_w, None, &cls_shape);
    let cls_probs = b.add_sigmoid(cls_logits, &cls_shape);

    let box_logits = b.add_linear(dec_out, box_w, None, &box_shape);
    let box_coords = b.add_sigmoid(box_logits, &box_shape);

    // Concatenate cls + box for joint output
    let out_shape = [NUM_QUERIES, NUM_CLASSES + 4];
    let concat = b.add_concat(&[cls_probs, box_coords], 1, &out_shape);

    b.build(concat).expect("valid enc-dec-heads kernel")
}

fn enc_dec_heads_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(w(&[NUM_QUERIES, HIDDEN_DIM])), // queries (learned)
    ];
    // Encoder layer (14 params)
    let ww = w(&[HIDDEN_DIM, HIDDEN_DIM]);
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    // Decoder layer (21 params)
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ww));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        FFN_DIM, HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        HIDDEN_DIM, FFN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(zeros(&[HIDDEN_DIM])));
    // Heads
    bindings.push(TensorParamBinding::ConstantTensor(w(&[
        NUM_CLASSES,
        HIDDEN_DIM,
    ])));
    bindings.push(TensorParamBinding::ConstantTensor(w(&[4, HIDDEN_DIM])));
    bindings
}

#[test]
fn test_tt_deep_enc_dec_heads_ibp() {
    let def = build_enc_dec_heads_kernel();
    let bindings = enc_dec_heads_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Enc-dec + heads IBP: [{lo}, {hi}]");
    // All sigmoid outputs in (0, 1)
    assert!(lo >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi}");
}

// ===========================================================================
// 7. Widening analysis: 1-layer vs 3-layer encoder
// ===========================================================================

#[test]
fn test_tt_deep_widening_1_vs_3_layers() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 1-layer
    let def1 = build_single_encoder_layer_kernel();
    let b1 = single_encoder_bindings();
    let g1 = tensor_kernel_to_graph(&def1, &b1).expect("1-layer graph");
    let out1 = g1.propagate_ibp(&input).expect("1-layer IBP");
    let (lo1, hi1) = bounds_min_max(&out1);
    let width1 = hi1 - lo1;

    // 3-layer
    let def3 = build_3layer_encoder_kernel();
    let b3 = three_layer_encoder_bindings();
    let g3 = tensor_kernel_to_graph(&def3, &b3).expect("3-layer graph");
    let out3 = g3.propagate_ibp(&input).expect("3-layer IBP");
    let (lo3, hi3) = bounds_min_max(&out3);
    let width3 = hi3 - lo3;

    eprintln!("Widening analysis: 1-layer width={width1:.4}, 3-layer width={width3:.4}");
    eprintln!("  Expansion ratio: {:.2}x", width3 / width1.max(1e-10));

    // Deeper = wider bounds (monotone widening property)
    assert!(
        width3 >= width1 - 1e-4,
        "3-layer bounds should be at least as wide as 1-layer: {width3} vs {width1}"
    );
}

// ===========================================================================
// 8. Structure recognition heads: row, column, span sigmoid
// ===========================================================================

fn build_structure_heads_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_deep_structure_heads");
    let input = b.add_input("decoder_out", &[NUM_QUERIES, HIDDEN_DIM]);

    // Row separator sigmoid
    let row_w = b.add_input("row_w", &[1, HIDDEN_DIM]);
    let row_logits = b.add_linear(input, row_w, None, &[NUM_QUERIES, 1]);
    let row_sig = b.add_sigmoid(row_logits, &[NUM_QUERIES, 1]);

    // Column separator sigmoid
    let col_w = b.add_input("col_w", &[1, HIDDEN_DIM]);
    let col_logits = b.add_linear(input, col_w, None, &[NUM_QUERIES, 1]);
    let col_sig = b.add_sigmoid(col_logits, &[NUM_QUERIES, 1]);

    // Span confidence sigmoid
    let span_w = b.add_input("span_w", &[1, HIDDEN_DIM]);
    let span_logits = b.add_linear(input, span_w, None, &[NUM_QUERIES, 1]);
    let span_sig = b.add_sigmoid(span_logits, &[NUM_QUERIES, 1]);

    // Concatenate: [NUM_QUERIES, 3]
    let out = b.add_concat(&[row_sig, col_sig, span_sig], 1, &[NUM_QUERIES, 3]);

    b.build(out).expect("valid structure heads kernel")
}

fn structure_heads_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[1, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[1, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(w(&[1, HIDDEN_DIM])),
    ]
}

#[test]
fn test_tt_deep_structure_heads_ibp() {
    let def = build_structure_heads_kernel();
    let bindings = structure_heads_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Structure heads IBP: [{lo}, {hi}]");
    assert!(lo >= 0.0 - 1e-6, "all sigmoid outputs >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-6, "all sigmoid outputs <= 1, got {hi}");
}

#[test]
fn test_tt_deep_structure_heads_crown() {
    let def = build_structure_heads_kernel();
    let bindings = structure_heads_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Structure heads CROWN (method={method:?}): [{lo}, {hi}]");
}

// ===========================================================================
// 9. Encoder + cls head verify-and-record
// ===========================================================================

#[test]
fn test_tt_deep_encoder_cls_head_verify_and_record() {
    let def = build_encoder_cls_head_kernel();
    let bindings = encoder_cls_head_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_deep::test_tt_deep_encoder_cls_head_verify_and_record",
    );
    assert!(result.verification.is_finite);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Encoder + cls head verify: [{lo}, {hi}], mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 10. Structure heads verify-and-record
// ===========================================================================

#[test]
fn test_tt_deep_structure_heads_verify_and_record() {
    let def = build_structure_heads_kernel();
    let bindings = structure_heads_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_deep::test_tt_deep_structure_heads_verify_and_record",
    );
    assert!(result.verification.is_finite);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Structure heads verify: [{lo}, {hi}], mode={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 11. Decoder + structure heads composition
// ===========================================================================

#[test]
fn test_tt_deep_decoder_structure_composition_ibp() {
    let def = build_decoder_layer_kernel();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let dec_output = graph.propagate_ibp(&input).expect("decoder IBP");
    assert_bounds_valid(&dec_output);

    // Feed decoder output through structure heads
    let heads_def = build_structure_heads_kernel();
    let heads_bindings = structure_heads_bindings();
    let heads_graph = tensor_kernel_to_graph(&heads_def, &heads_bindings).expect("heads graph");
    let heads_output = heads_graph.propagate_ibp(&dec_output).expect("heads IBP");
    assert_bounds_valid(&heads_output);
    let (lo, hi) = bounds_min_max(&heads_output);
    eprintln!("Decoder + structure heads IBP: [{lo}, {hi}]");
    assert!(lo >= 0.0 - 1e-6, "sigmoid lower >= 0, got {lo}");
    assert!(hi <= 1.0 + 1e-6, "sigmoid upper <= 1, got {hi}");
}

// ===========================================================================
// 12. Encoder + decoder + structure (full structure recognition)
// ===========================================================================

#[test]
fn test_tt_deep_full_structure_recognition_ibp() {
    // Encoder
    let enc_def = build_single_encoder_layer_kernel();
    let enc_bindings = single_encoder_bindings();
    let enc_graph = tensor_kernel_to_graph(&enc_def, &enc_bindings).expect("enc graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let enc_output = enc_graph.propagate_ibp(&input).expect("enc IBP");
    assert_bounds_valid(&enc_output);

    // Decoder (with encoder output as memory, queries as variable)
    let dec_def = build_decoder_layer_kernel();
    let dec_bindings = decoder_layer_bindings();
    let dec_graph = tensor_kernel_to_graph(&dec_def, &dec_bindings).expect("dec graph");
    let query_input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let dec_output = dec_graph.propagate_ibp(&query_input).expect("dec IBP");
    assert_bounds_valid(&dec_output);

    // Structure heads
    let heads_def = build_structure_heads_kernel();
    let heads_bindings = structure_heads_bindings();
    let heads_graph = tensor_kernel_to_graph(&heads_def, &heads_bindings).expect("heads graph");
    let heads_output = heads_graph.propagate_ibp(&dec_output).expect("heads IBP");
    assert_bounds_valid(&heads_output);
    let (lo, hi) = bounds_min_max(&heads_output);
    eprintln!("Full structure recognition IBP: [{lo}, {hi}]");
    assert!(lo >= 0.0 - 1e-6);
    assert!(hi <= 1.0 + 1e-6);
}
