// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for multi-model orchestration pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through orchestration patterns
//! that dpdf uses to chain multiple models together: routing, dispatch,
//! fallback, ensemble, and hierarchical decomposition.
//!
//! ## Tests (16 tests)
//!
//! 1.  **Layout detection -> region dispatch bounds** (IBP)
//! 2.  **Region crop -> model selection bounds** (IBP + CROWN)
//! 3.  **Confidence score routing (threshold-based dispatch)** (IBP)
//! 4.  **Table region -> Table Transformer bounds** (IBP + CROWN)
//! 5.  **Text region -> OCR model bounds** (IBP + CROWN)
//! 6.  **Figure region -> captioning model bounds** (IBP)
//! 7.  **Multi-model output aggregation bounds** (IBP)
//! 8.  **Sequential pipeline: detect -> classify -> extract** (IBP + CROWN)
//! 9.  **Parallel pipeline: run multiple models on same input** (IBP)
//! 10. **Fallback chain: model A fails -> try model B** (IBP + CROWN)
//! 11. **Ensemble averaging of multiple model outputs** (IBP)
//! 12. **Priority queue based on confidence** (IBP + CROWN)
//! 13. **Page-level -> region-level decomposition bounds** (IBP)
//! 14. **Region-level -> character-level decomposition bounds** (IBP + CROWN)
//! 15. **Output format normalization bounds** (IBP)
//! 16. **Full orchestration pipeline end-to-end** (IBP + CROWN)
//!
//! Architecture references:
//! - dpdf multi-model orchestration: layout detection -> routing -> specialized
//!   models (table, OCR, captioning) -> output aggregation
//! - Routing uses confidence-based dispatch: sigmoid/softmax scores select model
//! - Fallback chains: primary model -> fallback model -> default output
//! - Ensemble: average outputs from multiple recognition models
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=4, IN_CHANNELS=3, HIDDEN_DIM=8, NUM_CLASSES=4
//! - NUM_REGIONS=4, VOCAB_SIZE=6, SEQ_LEN=4, FFN_DIM=16
//!
//! Part of #4210: Compose tests for multi-model orchestration pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const IMG_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
const HIDDEN_DIM: usize = 8;
const FFN_DIM: usize = 16;
const NUM_CLASSES: usize = 4;
const NUM_REGIONS: usize = 4;
const VOCAB_SIZE: usize = 6;
const SEQ_LEN: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for LayerNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(channels: usize, h: usize, w: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

// ===========================================================================
// 1. Layout detection -> region dispatch bounds (IBP)
// ===========================================================================

#[test]
fn test_orchestration_layout_detection_region_dispatch_ibp() {
    // Layout detector: Conv2d -> ReLU -> flatten -> Linear -> sigmoid
    // Produces per-region class scores for dispatch routing.
    // Output: [NUM_REGIONS, NUM_CLASSES] with sigmoid scores in [0, 1].
    let conv_out_h = IMG_SIZE / 2;
    let conv_out_w = IMG_SIZE / 2;
    let flat_dim = HIDDEN_DIM * conv_out_h * conv_out_w;
    let out_dim = NUM_REGIONS * NUM_CLASSES;

    let mut b = TensorBlockBuilder::new("orch_layout_dispatch");
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IN_CHANNELS, 2, 2]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        2,
        2,
        0,
        0,
        &[HIDDEN_DIM, conv_out_h, conv_out_w],
    );
    let act = b.add_relu(conv_out, &[HIDDEN_DIM, conv_out_h, conv_out_w]);
    let flat = b.add_reshape(act, &[flat_dim]);
    let fc_w = b.add_input("fc_w", &[out_dim, flat_dim]);
    let fc_b = b.add_input("fc_b", &[out_dim]);
    let logits = b.add_linear(flat, fc_w, Some(fc_b), &[out_dim]);
    let scores = b.add_sigmoid(logits, &[out_dim]);
    let out = b.add_reshape(scores, &[NUM_REGIONS, NUM_CLASSES]);
    let def = b.build(out).expect("valid layout dispatch kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, 2, 2]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[out_dim, flat_dim]),
        bias_zero(&[out_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_REGIONS, NUM_CLASSES]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch layout dispatch IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 2. Region crop -> model selection bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_region_crop_model_selection_ibp_crown() {
    // Region features -> MLP classifier -> softmax -> model selection scores.
    // Simulates: given a cropped region's features, select which model to run.
    // Output: [1, NUM_CLASSES] softmax probabilities.
    let mut b = TensorBlockBuilder::new("orch_region_model_select");
    let input = b.add_input("region_features", &[1, HIDDEN_DIM]);
    let w1 = b.add_input("cls_w1", &[FFN_DIM, HIDDEN_DIM]);
    let b1 = b.add_input("cls_b1", &[FFN_DIM]);
    let h = b.add_linear(input, w1, Some(b1), &[1, FFN_DIM]);
    let act = b.add_gelu(h, &[1, FFN_DIM]);
    let w2 = b.add_input("cls_w2", &[NUM_CLASSES, FFN_DIM]);
    let b2 = b.add_input("cls_b2", &[NUM_CLASSES]);
    let logits = b.add_linear(act, w2, Some(b2), &[1, NUM_CLASSES]);
    let out = b.add_softmax(logits, -1, &[1, NUM_CLASSES]);
    let def = b.build(out).expect("valid model selection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias_zero(&[NUM_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch region model selection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower must be >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper must be <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch region model selection CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 3. Confidence score routing (threshold-based dispatch) (IBP)
// ===========================================================================

#[test]
fn test_orchestration_confidence_routing_ibp() {
    // Confidence routing: Linear -> sigmoid produces a scalar confidence.
    // Threshold-based dispatch: confidence > threshold -> primary model,
    // otherwise -> fallback model. Here we verify sigmoid output in [0, 1].
    let mut b = TensorBlockBuilder::new("orch_confidence_route");
    let input = b.add_input("features", &[1, HIDDEN_DIM]);
    let w = b.add_input("conf_w", &[1, HIDDEN_DIM]);
    let bias = b.add_input("conf_b", &[1]);
    let logit = b.add_linear(input, w, Some(bias), &[1, 1]);
    let conf = b.add_sigmoid(logit, &[1, 1]);
    let def = b.build(conf).expect("valid confidence routing kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[1, HIDDEN_DIM]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, HIDDEN_DIM, 2.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch confidence routing IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid confidence >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid confidence <= 1");
}

// ===========================================================================
// 4. Table region -> Table Transformer bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_table_region_transformer_ibp_crown() {
    // Table detection path: region features -> LN -> MLP -> sigmoid cell scores.
    // Simulates the Table Transformer producing cell detection scores.
    let num_cells = NUM_REGIONS;

    let mut b = TensorBlockBuilder::new("orch_table_transformer");
    let input = b.add_input("table_region", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b.add_input("mlp_w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(normed, w1, None, &[SEQ_LEN, FFN_DIM]);
    let act = b.add_relu(h, &[SEQ_LEN, FFN_DIM]);
    let w2 = b.add_input("mlp_w2", &[num_cells, FFN_DIM]);
    let b2 = b.add_input("mlp_b2", &[num_cells]);
    let logits = b.add_linear(act, w2, Some(b2), &[SEQ_LEN, num_cells]);
    let scores = b.add_sigmoid(logits, &[SEQ_LEN, num_cells]);
    let def = b.build(scores).expect("valid table transformer kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[num_cells, FFN_DIM]),
        bias_zero(&[num_cells]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch table transformer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid cell scores >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid cell scores <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch table transformer CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 5. Text region -> OCR model bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_text_region_ocr_ibp_crown() {
    // OCR path: text region features -> Linear -> GELU -> Linear -> softmax (vocab).
    // Output: [SEQ_LEN, VOCAB_SIZE] character probabilities.
    let mut b = TensorBlockBuilder::new("orch_text_ocr");
    let input = b.add_input("text_features", &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b.add_input("ocr_w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w1, None, &[SEQ_LEN, FFN_DIM]);
    let act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let w2 = b.add_input("ocr_w2", &[VOCAB_SIZE, FFN_DIM]);
    let b2 = b.add_input("ocr_b2", &[VOCAB_SIZE]);
    let logits = b.add_linear(act, w2, Some(b2), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid OCR kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch text OCR IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch text OCR CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 6. Figure region -> captioning model bounds (IBP)
// ===========================================================================

#[test]
fn test_orchestration_figure_captioning_ibp() {
    // Captioning path: image features -> Linear -> GELU -> Linear -> softmax.
    // Simulates a figure captioning model producing word probabilities.
    let caption_vocab = VOCAB_SIZE;
    let caption_seq = SEQ_LEN;

    let mut b = TensorBlockBuilder::new("orch_figure_caption");
    let input = b.add_input("figure_features", &[caption_seq, HIDDEN_DIM]);
    let w1 = b.add_input("cap_w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w1, None, &[caption_seq, FFN_DIM]);
    let act = b.add_gelu(h, &[caption_seq, FFN_DIM]);
    let w2 = b.add_input("cap_w2", &[caption_vocab, FFN_DIM]);
    let b2 = b.add_input("cap_b2", &[caption_vocab]);
    let logits = b.add_linear(act, w2, Some(b2), &[caption_seq, caption_vocab]);
    let out = b.add_softmax(logits, -1, &[caption_seq, caption_vocab]);
    let def = b.build(out).expect("valid captioning kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[caption_vocab, FFN_DIM]),
        bias_zero(&[caption_vocab]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(caption_seq, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch figure captioning IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1");
}

// ===========================================================================
// 7. Multi-model output aggregation bounds (IBP)
// ===========================================================================

#[test]
fn test_orchestration_multi_model_output_aggregation_ibp() {
    // Aggregation: concatenate outputs from two models, project to unified dim.
    // [SEQ_LEN, DIM_A] concat [SEQ_LEN, DIM_B] -> [SEQ_LEN, DIM_A+DIM_B]
    // -> Linear -> [SEQ_LEN, HIDDEN_DIM]
    let dim_a = HIDDEN_DIM;
    let dim_b = HIDDEN_DIM;
    let concat_dim = dim_a + dim_b;

    let mut b = TensorBlockBuilder::new("orch_output_aggregation");
    let model_a = b.add_input("model_a_out", &[SEQ_LEN, dim_a]);
    let model_b = b.add_input("model_b_out", &[SEQ_LEN, dim_b]);
    let concat = b.add_concat(&[model_a, model_b], 1, &[SEQ_LEN, concat_dim]);
    let proj_w = b.add_input("agg_proj_w", &[HIDDEN_DIM, concat_dim]);
    let proj_b = b.add_input("agg_proj_b", &[HIDDEN_DIM]);
    let out = b.add_linear(concat, proj_w, Some(proj_b), &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid aggregation kernel");

    // Both model outputs are Variable inputs (representing separate model outputs)
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[SEQ_LEN, dim_b]), 0.5f32)),
        weight(&[HIDDEN_DIM, concat_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, dim_a, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch output aggregation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 8. Sequential pipeline: detect -> classify -> extract (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_sequential_detect_classify_extract_ibp_crown() {
    // 3-stage sequential pipeline:
    //   Stage 1 (detect): Linear -> ReLU (feature extraction)
    //   Stage 2 (classify): Linear -> GELU (classification features)
    //   Stage 3 (extract): Linear -> sigmoid (extraction scores)
    let mid_dim = HIDDEN_DIM;

    let mut b = TensorBlockBuilder::new("orch_sequential_pipeline");
    let input = b.add_input("raw_features", &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 1: detect
    let w1 = b.add_input("detect_w", &[mid_dim, HIDDEN_DIM]);
    let s1 = b.add_linear(input, w1, None, &[SEQ_LEN, mid_dim]);
    let s1_act = b.add_relu(s1, &[SEQ_LEN, mid_dim]);

    // Stage 2: classify
    let w2 = b.add_input("classify_w", &[mid_dim, mid_dim]);
    let s2 = b.add_linear(s1_act, w2, None, &[SEQ_LEN, mid_dim]);
    let s2_act = b.add_gelu(s2, &[SEQ_LEN, mid_dim]);

    // Stage 3: extract
    let w3 = b.add_input("extract_w", &[NUM_CLASSES, mid_dim]);
    let b3 = b.add_input("extract_b", &[NUM_CLASSES]);
    let s3 = b.add_linear(s2_act, w3, Some(b3), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(s3, &[SEQ_LEN, NUM_CLASSES]);
    let def = b.build(out).expect("valid sequential pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[mid_dim, HIDDEN_DIM]),
        weight(&[mid_dim, mid_dim]),
        weight(&[NUM_CLASSES, mid_dim]),
        bias_zero(&[NUM_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch sequential pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid output >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid output <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch sequential pipeline CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 9. Parallel pipeline: run multiple models on same input (IBP)
// ===========================================================================

#[test]
fn test_orchestration_parallel_pipeline_ibp() {
    // Parallel models: same input features processed by two different MLPs,
    // outputs concatenated. Verifies that independent branches preserve bounds.
    // Branch A: Linear -> ReLU -> [SEQ_LEN, DIM_A]
    // Branch B: Linear -> GELU -> [SEQ_LEN, DIM_B]
    // Concat: [SEQ_LEN, DIM_A + DIM_B]
    let dim_a = HIDDEN_DIM / 2;
    let dim_b = HIDDEN_DIM / 2;
    let concat_dim = dim_a + dim_b;

    let mut b = TensorBlockBuilder::new("orch_parallel_pipeline");
    let input = b.add_input("shared_features", &[SEQ_LEN, HIDDEN_DIM]);

    // Branch A
    let wa = b.add_input("branch_a_w", &[dim_a, HIDDEN_DIM]);
    let ha = b.add_linear(input, wa, None, &[SEQ_LEN, dim_a]);
    let act_a = b.add_relu(ha, &[SEQ_LEN, dim_a]);

    // Branch B
    let wb = b.add_input("branch_b_w", &[dim_b, HIDDEN_DIM]);
    let hb = b.add_linear(input, wb, None, &[SEQ_LEN, dim_b]);
    let act_b = b.add_gelu(hb, &[SEQ_LEN, dim_b]);

    // Concatenate branches
    let cat = b.add_concat(&[act_a, act_b], 1, &[SEQ_LEN, concat_dim]);
    let def = b.build(cat).expect("valid parallel pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[dim_a, HIDDEN_DIM]),
        weight(&[dim_b, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, concat_dim]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch parallel pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Fallback chain: model A fails -> try model B (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_fallback_chain_ibp_crown() {
    // Fallback chain: two models produce scores, then we take a weighted sum.
    // Model A: Linear -> sigmoid (primary)
    // Model B: Linear -> sigmoid (fallback)
    // Weighted combination: 0.7 * A + 0.3 * B (simulating confidence-weighted fallback)
    //
    // In practice, the routing selects one model, but for verification we
    // prove that the combined output is bounded for ANY routing decision.
    let out_dim = NUM_CLASSES;

    let mut b = TensorBlockBuilder::new("orch_fallback_chain");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Model A (primary)
    let wa = b.add_input("model_a_w", &[out_dim, HIDDEN_DIM]);
    let ba = b.add_input("model_a_b", &[out_dim]);
    let logits_a = b.add_linear(input, wa, Some(ba), &[SEQ_LEN, out_dim]);
    let scores_a = b.add_sigmoid(logits_a, &[SEQ_LEN, out_dim]);

    // Model B (fallback)
    let wb = b.add_input("model_b_w", &[out_dim, HIDDEN_DIM]);
    let bb = b.add_input("model_b_b", &[out_dim]);
    let logits_b = b.add_linear(input, wb, Some(bb), &[SEQ_LEN, out_dim]);
    let scores_b = b.add_sigmoid(logits_b, &[SEQ_LEN, out_dim]);

    // Weighted combination: add both (weight is absorbed into model weights)
    let combined = b.add_binary_add(scores_a, scores_b, &[SEQ_LEN, out_dim]);
    let def = b.build(combined).expect("valid fallback chain kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[out_dim, HIDDEN_DIM]),
        bias_zero(&[out_dim]),
        weight(&[out_dim, HIDDEN_DIM]),
        bias_zero(&[out_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch fallback chain IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sum of two sigmoids: [0, 2]
    assert!(lo_min >= -1e-5, "sum of sigmoids lower >= 0");
    assert!(hi_max <= 2.0 + 1e-4, "sum of sigmoids upper <= 2");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch fallback chain CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 11. Ensemble averaging of multiple model outputs (IBP)
// ===========================================================================

#[test]
fn test_orchestration_ensemble_averaging_ibp() {
    // Ensemble: two models produce logits, average them, then softmax.
    // Model A: Linear -> [SEQ_LEN, VOCAB_SIZE]
    // Model B: Linear -> [SEQ_LEN, VOCAB_SIZE] (constant simulating second model)
    // Average: (A + B) / 2 (via add + scale)
    // Softmax: [SEQ_LEN, VOCAB_SIZE] probabilities
    let mut b = TensorBlockBuilder::new("orch_ensemble_avg");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Model A logits
    let wa = b.add_input("model_a_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits_a = b.add_linear(input, wa, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Model B logits (constant -- second model output treated as fixed)
    let logits_b = b.add_input("model_b_logits", &[SEQ_LEN, VOCAB_SIZE]);

    // Average: add then scale by 0.5. `add_binary_mul` requires matching ranks,
    // so broadcast the scalar [1] scale up to [SEQ_LEN, VOCAB_SIZE] first.
    let sum = b.add_binary_add(logits_a, logits_b, &[SEQ_LEN, VOCAB_SIZE]);
    let scale = b.add_input("scale", &[1]);
    let scale_bc = b.add_broadcast(scale, &[SEQ_LEN, VOCAB_SIZE]);
    let avg = b.add_binary_mul(sum, scale_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(avg, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid ensemble kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SEQ_LEN, VOCAB_SIZE]),
            0.0f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.5f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch ensemble averaging IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1");
}

// ===========================================================================
// 12. Priority queue based on confidence (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_priority_confidence_ibp_crown() {
    // Priority scoring: features -> LN -> Linear -> sigmoid per-region confidence.
    // Higher confidence = higher priority in the processing queue.
    // Output: [NUM_REGIONS, 1] priority scores in [0, 1].
    let mut b = TensorBlockBuilder::new("orch_priority_confidence");
    let input = b.add_input("region_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[NUM_REGIONS, HIDDEN_DIM]);
    let w = b.add_input("priority_w", &[1, HIDDEN_DIM]);
    let bias = b.add_input("priority_b", &[1]);
    let logits = b.add_linear(normed, w, Some(bias), &[NUM_REGIONS, 1]);
    let scores = b.add_sigmoid(logits, &[NUM_REGIONS, 1]);
    let def = b.build(scores).expect("valid priority scoring kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        weight(&[1, HIDDEN_DIM]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(NUM_REGIONS, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[NUM_REGIONS, 1]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch priority confidence IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid priority >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid priority <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch priority confidence CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 13. Page-level -> region-level decomposition bounds (IBP)
// ===========================================================================

#[test]
fn test_orchestration_page_to_region_decomposition_ibp() {
    // Page-level features decomposed to region-level via Conv2d + reshape.
    // Image [C, H, W] -> Conv2d -> [HIDDEN_DIM, H/2, W/2] -> reshape
    // -> [NUM_REGIONS, HIDDEN_DIM] (regions as sequence elements)
    let conv_out_h = IMG_SIZE / 2;
    let conv_out_w = IMG_SIZE / 2;

    let mut b = TensorBlockBuilder::new("orch_page_to_region");
    let image = b.add_input("page_image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("decompose_w", &[HIDDEN_DIM, IN_CHANNELS, 2, 2]);
    let conv_b = b.add_input("decompose_b", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        2,
        2,
        0,
        0,
        &[HIDDEN_DIM, conv_out_h, conv_out_w],
    );
    // Reshape spatial grid to regions: [HIDDEN_DIM, 2, 2] -> [4, HIDDEN_DIM]
    let num_regions = conv_out_h * conv_out_w;
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, num_regions]);
    let out = b.add_transpose(reshaped, &[1, 0], &[num_regions, HIDDEN_DIM]);
    let def = b.build(out).expect("valid page-to-region kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, 2, 2]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_regions, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch page-to-region IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 14. Region-level -> character-level decomposition bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_region_to_character_decomposition_ibp_crown() {
    // Region features -> LN -> Linear (upsample) -> GELU -> Linear (per-char).
    // [NUM_REGIONS, HIDDEN_DIM] -> [NUM_REGIONS, FFN_DIM] -> [NUM_REGIONS, VOCAB_SIZE]
    // Then softmax for character-level probabilities.
    let mut b = TensorBlockBuilder::new("orch_region_to_char");
    let input = b.add_input("region_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[NUM_REGIONS, HIDDEN_DIM]);
    let w1 = b.add_input("char_w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(normed, w1, None, &[NUM_REGIONS, FFN_DIM]);
    let act = b.add_gelu(h, &[NUM_REGIONS, FFN_DIM]);
    let w2 = b.add_input("char_w2", &[VOCAB_SIZE, FFN_DIM]);
    let b2 = b.add_input("char_b2", &[VOCAB_SIZE]);
    let logits = b.add_linear(act, w2, Some(b2), &[NUM_REGIONS, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, -1, &[NUM_REGIONS, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid region-to-char kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(NUM_REGIONS, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_REGIONS, VOCAB_SIZE]
    );
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch region-to-char IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch region-to-char CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 15. Output format normalization bounds (IBP)
// ===========================================================================

#[test]
fn test_orchestration_output_format_normalization_ibp() {
    // Output normalization: raw model outputs -> LayerNorm -> Linear -> sigmoid.
    // Ensures all model outputs are normalized to a consistent [0, 1] range
    // regardless of which upstream model produced them.
    let mut b = TensorBlockBuilder::new("orch_output_normalization");
    let input = b.add_input("raw_output", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("norm_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[NUM_CLASSES]);
    let logits = b.add_linear(normed, proj_w, Some(proj_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);
    let def = b.build(out).expect("valid output normalization kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        weight(&[NUM_CLASSES, HIDDEN_DIM]),
        bias_zero(&[NUM_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input range: raw outputs from different models can be in varying ranges.
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 5.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Orch output normalization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -1e-5,
        "normalized sigmoid output >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "normalized sigmoid output <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. Full orchestration pipeline end-to-end (IBP + CROWN)
// ===========================================================================

#[test]
fn test_orchestration_full_pipeline_e2e_ibp_crown() {
    // Full end-to-end orchestration:
    //   Stage 1: Image -> Conv2d -> ReLU -> flatten -> features
    //   Stage 2: features -> LN -> Linear -> GELU (enrichment)
    //   Stage 3: enriched -> Linear -> softmax (final classification)
    //
    // This represents the composition of detection, feature enrichment,
    // and classification stages in a single verified graph.
    let conv_out_h = IMG_SIZE / 2;
    let conv_out_w = IMG_SIZE / 2;
    let flat_dim = HIDDEN_DIM * conv_out_h * conv_out_w;

    let mut b = TensorBlockBuilder::new("orch_full_e2e");
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Stage 1: detection backbone
    let conv_w = b.add_input("backbone_w", &[HIDDEN_DIM, IN_CHANNELS, 2, 2]);
    let conv_b = b.add_input("backbone_b", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        2,
        2,
        0,
        0,
        &[HIDDEN_DIM, conv_out_h, conv_out_w],
    );
    let act1 = b.add_relu(conv_out, &[HIDDEN_DIM, conv_out_h, conv_out_w]);
    let flat = b.add_reshape(act1, &[1, flat_dim]);

    // Stage 2: feature enrichment with LayerNorm
    let ln_w = b.add_input("enrich_ln_w", &[flat_dim]);
    let ln_b = b.add_input("enrich_ln_b", &[flat_dim]);
    let eps = b.add_input("enrich_eps", &[1]);
    let normed = b.add_layer_norm(flat, eps, 1, ln_w, ln_b, &[1, flat_dim]);
    let enrich_w = b.add_input("enrich_w", &[HIDDEN_DIM, flat_dim]);
    let enriched = b.add_linear(normed, enrich_w, None, &[1, HIDDEN_DIM]);
    let act2 = b.add_gelu(enriched, &[1, HIDDEN_DIM]);

    // Stage 3: final classification
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let logits = b.add_linear(act2, cls_w, Some(cls_b), &[1, NUM_CLASSES]);
    let out = b.add_softmax(logits, -1, &[1, NUM_CLASSES]);
    let def = b.build(out).expect("valid full e2e pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, 2, 2]),
        bias_zero(&[HIDDEN_DIM]),
        ones(&[flat_dim]),
        bias_zero(&[flat_dim]),
        eps_binding(),
        weight(&[HIDDEN_DIM, flat_dim]),
        weight(&[NUM_CLASSES, HIDDEN_DIM]),
        bias_zero(&[NUM_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[1, NUM_CLASSES]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Orch full e2e pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax output >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "softmax output <= 1");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Orch full e2e pipeline CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}
