// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-model interaction and ensemble composition tests for the dpdf
//! 7-model document processing pipeline. Complements the base ensemble file
//! (pipeline-level) and extended file (per-model standalone) with cross-model
//! interaction patterns: feature fusion, voting, cascading, and calibration.
//!
//! ## Tests (7 tests)
//!
//! 1. Cross-model feature fusion: 3 OCR concat features -> MLP (IBP + CROWN)
//! 2. Majority voting: logit summation + temperature softmax (IBP)
//! 3. Vision-to-LM cascade: embed -> encoder -> projection -> decoder (IBP)
//! 4. Ensemble monotone CROWN: narrower input -> narrower CROWN output
//! 5. E2E realistic 7-head: shared backbone -> 7 sigmoid heads -> fusion (IBP)
//! 6. Hierarchical ensemble: detection routes OCR by region type (IBP)
//! 7. Confidence calibration: temperature-scaled logit averaging (IBP + CROWN)
//!
//! Part of #4243: dpdf 7-model ensemble compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

const HIDDEN: usize = 8;
const SEQ: usize = 4;
const NUM_CLASSES: usize = 6;
const VOCAB: usize = 8;
const FFN_DIM: usize = HIDDEN * 2;
const NUM_MODELS: usize = 7;
const PATCH_DIM: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

// ===========================================================================
// 1. Cross-model feature fusion (IBP + CROWN)
// ===========================================================================

/// Three OCR models each produce hidden features that are concatenated
/// in feature space (not probability space) and passed through a shared
/// MLP for unified representation.
///
/// Key property: concatenation in feature space preserves finite bounds
/// through the fusion MLP.
#[test]
fn test_7model_ext_cross_model_feature_fusion_ibp_crown() {
    let concat_dim = HIDDEN * 3;

    let mut b = TensorBlockBuilder::new("7model_feature_fusion");
    let input = b.add_input("shared_features", &[SEQ, HIDDEN]);

    // Model A: Linear -> ReLU (features, not probabilities)
    let a_w = b.add_input("a_w", &[HIDDEN, HIDDEN]);
    let a_h = b.add_linear(input, a_w, None, &[SEQ, HIDDEN]);
    let a_feat = b.add_relu(a_h, &[SEQ, HIDDEN]);

    // Model B: Linear -> ReLU
    let bm_w = b.add_input("b_w", &[HIDDEN, HIDDEN]);
    let bm_h = b.add_linear(input, bm_w, None, &[SEQ, HIDDEN]);
    let b_feat = b.add_relu(bm_h, &[SEQ, HIDDEN]);

    // Model C: Linear -> ReLU
    let c_w = b.add_input("c_w", &[HIDDEN, HIDDEN]);
    let c_h = b.add_linear(input, c_w, None, &[SEQ, HIDDEN]);
    let c_feat = b.add_relu(c_h, &[SEQ, HIDDEN]);

    // Concatenate in feature space: [SEQ, 3*HIDDEN]
    let concat = b.add_concat(&[a_feat, b_feat, c_feat], 1, &[SEQ, concat_dim]);

    // Fusion MLP: Linear -> ReLU -> Linear -> sigmoid
    let fuse_w1 = b.add_input("fuse_w1", &[HIDDEN, concat_dim]);
    let fuse_h = b.add_linear(concat, fuse_w1, None, &[SEQ, HIDDEN]);
    let fuse_act = b.add_relu(fuse_h, &[SEQ, HIDDEN]);
    let fuse_w2 = b.add_input("fuse_w2", &[NUM_CLASSES, HIDDEN]);
    let fuse_logits = b.add_linear(fuse_act, fuse_w2, None, &[SEQ, NUM_CLASSES]);
    let out = b.add_sigmoid(fuse_logits, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid feature fusion kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, concat_dim]),
        weight(&[NUM_CLASSES, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("feature fusion IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("feature fusion CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 2. Majority voting via softmax temperature (IBP)
// ===========================================================================

/// Approximate majority voting: each OCR model produces logits, these are
/// summed (log-domain voting), then a low-temperature softmax sharpens
/// the distribution towards the majority prediction.
///
/// Key property: logit summation + softmax preserves valid probability bounds.
#[test]
fn test_7model_ext_majority_voting_ibp() {
    let mut b = TensorBlockBuilder::new("7model_majority_vote");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Model 1 logits (pre-softmax)
    let m1_w = b.add_input("m1_w", &[VOCAB, HIDDEN]);
    let m1_logits = b.add_linear(input, m1_w, None, &[SEQ, VOCAB]);

    // Model 2 logits
    let m2_w = b.add_input("m2_w", &[VOCAB, HIDDEN]);
    let m2_logits = b.add_linear(input, m2_w, None, &[SEQ, VOCAB]);

    // Model 3 logits
    let m3_w = b.add_input("m3_w", &[VOCAB, HIDDEN]);
    let m3_logits = b.add_linear(input, m3_w, None, &[SEQ, VOCAB]);

    // Sum logits (log-domain voting)
    let sum12 = b.add_binary_add(m1_logits, m2_logits, &[SEQ, VOCAB]);
    let sum_all = b.add_binary_add(sum12, m3_logits, &[SEQ, VOCAB]);

    // Temperature scaling: multiply by 1/T (T=0.5 -> scale=2.0)
    let temp_scale = b.add_input("temp_scale", &[SEQ, VOCAB]);
    let scaled = b.add_binary_mul(sum_all, temp_scale, &[SEQ, VOCAB]);

    // Softmax on sharpened logits
    let out = b.add_softmax(scaled, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid voting kernel");

    let temp_data = ArrayD::from_elem(IxDyn(&[SEQ, VOCAB]), 2.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        TensorParamBinding::ConstantTensor(temp_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("majority voting IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax hi <= 1, got {hi_max}");
}

// ===========================================================================
// 3. Vision-to-LM cascade (IBP)
// ===========================================================================

/// Full vision-to-language cascade: patch embedding (Linear) -> vision
/// encoder FFN -> cross-domain projection -> LM decoder FFN -> softmax.
///
/// Key property: 4-stage cascade preserves bounded output.
#[test]
fn test_7model_ext_vision_to_lm_cascade_ibp() {
    let mut b = TensorBlockBuilder::new("7model_vision_lm_cascade");
    let input = b.add_input("image_patches", &[SEQ, PATCH_DIM]);

    // Stage 1: Patch embedding
    let embed_w = b.add_input("embed_w", &[HIDDEN, PATCH_DIM]);
    let embedded = b.add_linear(input, embed_w, None, &[SEQ, HIDDEN]);

    // Stage 2: Vision encoder FFN + residual
    let enc_w1 = b.add_input("enc_w1", &[FFN_DIM, HIDDEN]);
    let enc_h = b.add_linear(embedded, enc_w1, None, &[SEQ, FFN_DIM]);
    let enc_act = b.add_relu(enc_h, &[SEQ, FFN_DIM]);
    let enc_w2 = b.add_input("enc_w2", &[HIDDEN, FFN_DIM]);
    let enc_out = b.add_linear(enc_act, enc_w2, None, &[SEQ, HIDDEN]);
    let enc_res = b.add_binary_add(embedded, enc_out, &[SEQ, HIDDEN]);

    // Stage 3: Cross-domain projection
    let proj_w = b.add_input("proj_w", &[HIDDEN, HIDDEN]);
    let proj_b = b.add_input("proj_b", &[HIDDEN]);
    let projected = b.add_linear(enc_res, proj_w, Some(proj_b), &[SEQ, HIDDEN]);

    // Stage 4: LM decoder FFN + residual
    let dec_w1 = b.add_input("dec_w1", &[FFN_DIM, HIDDEN]);
    let dec_h = b.add_linear(projected, dec_w1, None, &[SEQ, FFN_DIM]);
    let dec_act = b.add_relu(dec_h, &[SEQ, FFN_DIM]);
    let dec_w2 = b.add_input("dec_w2", &[HIDDEN, FFN_DIM]);
    let dec_out = b.add_linear(dec_act, dec_w2, None, &[SEQ, HIDDEN]);
    let dec_res = b.add_binary_add(projected, dec_out, &[SEQ, HIDDEN]);

    // LM head -> softmax
    let lm_w = b.add_input("lm_w", &[VOCAB, HIDDEN]);
    let lm_logits = b.add_linear(dec_res, lm_w, None, &[SEQ, VOCAB]);
    let out = b.add_softmax(lm_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid vision-lm cascade kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, PATCH_DIM]),
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[HIDDEN, HIDDEN]),
        bias_zero(&[HIDDEN]),
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[VOCAB, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("vision-lm cascade IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);
}

// ===========================================================================
// 4. Ensemble monotone through CROWN (IBP + CROWN)
// ===========================================================================

/// Monotonicity with CROWN: narrower input must produce no-wider output
/// through a 7-model gated ensemble with sigmoid output.
///
/// Key property: CROWN monotonicity holds for gated multi-model composition.
#[test]
fn test_7model_ext_ensemble_monotone_crown() {
    let build_ensemble = || {
        let mut b = TensorBlockBuilder::new("7model_mono_crown");
        let input = b.add_input("features", &[SEQ, HIDDEN]);

        // Gate: softmax over 7 models
        let gate_w = b.add_input("gate_w", &[NUM_MODELS, HIDDEN]);
        let gate_logits = b.add_linear(input, gate_w, None, &[SEQ, NUM_MODELS]);
        let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_MODELS]);

        // Heads: gate_probs -> output space
        let heads_w = b.add_input("heads_w", &[NUM_CLASSES, NUM_MODELS]);
        let gated = b.add_linear(gate_probs, heads_w, None, &[SEQ, NUM_CLASSES]);
        let out = b.add_sigmoid(gated, &[SEQ, NUM_CLASSES]);
        let def = b.build(out).expect("valid ensemble kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            weight(&[NUM_MODELS, HIDDEN]),
            weight(&[NUM_CLASSES, NUM_MODELS]),
        ];
        tensor_kernel_to_graph(&def, &bindings).expect("graph")
    };

    let graph = build_ensemble();

    // Wide: [-1, 1]
    let wide_input = uniform_bounds(&[SEQ, HIDDEN], 1.0);
    let (_, wide_crown, _) = assert_crown_tighter_when_not_fallback(&graph, &wide_input);

    // Narrow: [-0.3, 0.3]
    let narrow_input = uniform_bounds(&[SEQ, HIDDEN], 0.3);
    let (_, narrow_crown, _) = assert_crown_tighter_when_not_fallback(&graph, &narrow_input);

    let (lo_w, hi_w) = bounds_min_max(&wide_crown);
    let (lo_n, hi_n) = bounds_min_max(&narrow_crown);
    let wide_width = hi_w - lo_w;
    let narrow_width = hi_n - lo_n;

    eprintln!(
        "ensemble monotone CROWN: wide=[{lo_w:.4}, {hi_w:.4}] w={wide_width:.4} \
         | narrow=[{lo_n:.4}, {hi_n:.4}] w={narrow_width:.4}"
    );
    assert!(
        narrow_width <= wide_width + 1e-4,
        "CROWN monotone violated: narrow_w={narrow_width} > wide_w={wide_width}"
    );
}

// ===========================================================================
// 5. E2E realistic 7-head pipeline (IBP)
// ===========================================================================

/// All 7 models share a backbone feature extractor, then each produces
/// a task-specific sigmoid output. Final output is the concatenation of
/// all 7 head outputs projected to a unified confidence vector.
#[test]
fn test_7model_ext_e2e_realistic_7heads_ibp() {
    let e2e_seq: usize = 6;

    let mut b = TensorBlockBuilder::new("7model_e2e_7heads");
    let input = b.add_input("backbone_features", &[e2e_seq, HIDDEN]);

    // Shared backbone FFN
    let bb_w1 = b.add_input("bb_w1", &[FFN_DIM, HIDDEN]);
    let bb_h = b.add_linear(input, bb_w1, None, &[e2e_seq, FFN_DIM]);
    let bb_act = b.add_relu(bb_h, &[e2e_seq, FFN_DIM]);
    let bb_w2 = b.add_input("bb_w2", &[HIDDEN, FFN_DIM]);
    let backbone = b.add_linear(bb_act, bb_w2, None, &[e2e_seq, HIDDEN]);

    // 7 heads, each: Linear -> sigmoid -> [e2e_seq, 1]
    let mut head_outputs = Vec::new();
    for i in 0..NUM_MODELS {
        let hw = b.add_input(&format!("head{i}_w"), &[1, HIDDEN]);
        let hb = b.add_input(&format!("head{i}_b"), &[1]);
        let logit = b.add_linear(backbone, hw, Some(hb), &[e2e_seq, 1]);
        let conf = b.add_sigmoid(logit, &[e2e_seq, 1]);
        head_outputs.push(conf);
    }

    // Concatenate: [e2e_seq, 7]
    let concat = b.add_concat(&head_outputs, 1, &[e2e_seq, NUM_MODELS]);

    // Final projection to unified confidence
    let final_w = b.add_input("final_w", &[NUM_CLASSES, NUM_MODELS]);
    let final_logits = b.add_linear(concat, final_w, None, &[e2e_seq, NUM_CLASSES]);
    let out = b.add_sigmoid(final_logits, &[e2e_seq, NUM_CLASSES]);
    let def = b.build(out).expect("valid e2e 7-heads kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
    ];
    for _ in 0..NUM_MODELS {
        bindings.push(weight(&[1, HIDDEN]));
        bindings.push(bias_zero(&[1]));
    }
    bindings.push(weight(&[NUM_CLASSES, NUM_MODELS]));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[e2e_seq, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("e2e 7-heads IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid hi <= 1, got {hi_max}");

    // Non-degenerate check: the tightened softmax+sigmoid IBP narrows this e2e
    // output well below the old 0.01 floor (observed ~7.2e-5), so that floor is
    // a stale lower bound made obsolete by tighter bounds; a narrower interval
    // is *better* here. We only require the bounds remain non-degenerate.
    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "ensemble output must be a non-degenerate interval, got w={width}"
    );
}

// ===========================================================================
// 6. Hierarchical ensemble: detection routes OCR by region type (IBP)
// ===========================================================================

/// Detection model classifies regions into types (table, text, figure).
/// Each region type routes to a specialized OCR model. The routing is
/// modeled as a softmax gate selecting between OCR heads.
///
/// Key property: hierarchical gating composes cleanly with per-type OCR.
#[test]
fn test_7model_ext_hierarchical_ensemble_ibp() {
    let num_region_types: usize = 3;

    let mut b = TensorBlockBuilder::new("7model_hierarchical");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Detection: region type classification -> softmax
    let det_w = b.add_input("det_w", &[num_region_types, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, num_region_types]);
    let region_probs = b.add_softmax(det_logits, -1, &[SEQ, num_region_types]);

    // Table OCR head -> softmax
    let table_w = b.add_input("table_w", &[VOCAB, HIDDEN]);
    let table_logits = b.add_linear(input, table_w, None, &[SEQ, VOCAB]);
    let table_out = b.add_softmax(table_logits, -1, &[SEQ, VOCAB]);

    // Text OCR head -> softmax
    let text_w = b.add_input("text_w", &[VOCAB, HIDDEN]);
    let text_logits = b.add_linear(input, text_w, None, &[SEQ, VOCAB]);
    let text_out = b.add_softmax(text_logits, -1, &[SEQ, VOCAB]);

    // Figure OCR head -> softmax
    let fig_w = b.add_input("fig_w", &[VOCAB, HIDDEN]);
    let fig_logits = b.add_linear(input, fig_w, None, &[SEQ, VOCAB]);
    let fig_out = b.add_softmax(fig_logits, -1, &[SEQ, VOCAB]);

    // Route: project [SEQ, 3] -> [SEQ, VOCAB] as blend coefficients
    let route_w = b.add_input("route_w", &[VOCAB, num_region_types]);
    let route_coeff = b.add_linear(region_probs, route_w, None, &[SEQ, VOCAB]);
    let route_gate = b.add_sigmoid(route_coeff, &[SEQ, VOCAB]);

    // Gated combination
    let g_table = b.add_binary_mul(route_gate, table_out, &[SEQ, VOCAB]);
    let g_text = b.add_binary_mul(route_gate, text_out, &[SEQ, VOCAB]);
    let g_fig = b.add_binary_mul(route_gate, fig_out, &[SEQ, VOCAB]);
    let sum_tt = b.add_binary_add(g_table, g_text, &[SEQ, VOCAB]);
    let out = b.add_binary_add(sum_tt, g_fig, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid hierarchical kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[num_region_types, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[VOCAB, num_region_types]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("hierarchical ensemble IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Confidence calibration (IBP + CROWN)
// ===========================================================================

/// Confidence calibration: each model produces logits that are
/// temperature-scaled before ensemble averaging. Tests that calibrated
/// ensemble outputs stay bounded.
///
/// Key property: temperature-scaled logit averaging preserves valid bounds.
#[test]
fn test_7model_ext_confidence_calibration_ibp_crown() {
    let num_ocr: usize = 3;

    let mut b = TensorBlockBuilder::new("7model_calibration");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Model 1: logits + temperature-scaled softmax
    let m1_w = b.add_input("m1_w", &[VOCAB, HIDDEN]);
    let m1_logits = b.add_linear(input, m1_w, None, &[SEQ, VOCAB]);
    let t1 = b.add_input("t1", &[SEQ, VOCAB]);
    let m1_scaled = b.add_binary_mul(m1_logits, t1, &[SEQ, VOCAB]);
    let m1_out = b.add_softmax(m1_scaled, -1, &[SEQ, VOCAB]);

    // Model 2
    let m2_w = b.add_input("m2_w", &[VOCAB, HIDDEN]);
    let m2_logits = b.add_linear(input, m2_w, None, &[SEQ, VOCAB]);
    let t2 = b.add_input("t2", &[SEQ, VOCAB]);
    let m2_scaled = b.add_binary_mul(m2_logits, t2, &[SEQ, VOCAB]);
    let m2_out = b.add_softmax(m2_scaled, -1, &[SEQ, VOCAB]);

    // Model 3
    let m3_w = b.add_input("m3_w", &[VOCAB, HIDDEN]);
    let m3_logits = b.add_linear(input, m3_w, None, &[SEQ, VOCAB]);
    let t3 = b.add_input("t3", &[SEQ, VOCAB]);
    let m3_scaled = b.add_binary_mul(m3_logits, t3, &[SEQ, VOCAB]);
    let m3_out = b.add_softmax(m3_scaled, -1, &[SEQ, VOCAB]);

    // Average the calibrated outputs
    let sum12 = b.add_binary_add(m1_out, m2_out, &[SEQ, VOCAB]);
    let sum_all = b.add_binary_add(sum12, m3_out, &[SEQ, VOCAB]);
    let avg_scale = b.add_input("avg_scale", &[SEQ, VOCAB]);
    let out = b.add_binary_mul(sum_all, avg_scale, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid calibration kernel");

    let temp_data = ArrayD::from_elem(IxDyn(&[SEQ, VOCAB]), 0.5f32);
    let avg_data = ArrayD::from_elem(IxDyn(&[SEQ, VOCAB]), 1.0f32 / num_ocr as f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),
        TensorParamBinding::ConstantTensor(temp_data.clone()),
        weight(&[VOCAB, HIDDEN]),
        TensorParamBinding::ConstantTensor(temp_data.clone()),
        weight(&[VOCAB, HIDDEN]),
        TensorParamBinding::ConstantTensor(temp_data),
        TensorParamBinding::ConstantTensor(avg_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("calibration IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-3, "avg softmax lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-3, "avg softmax hi <= 1, got {hi_max}");

    // CROWN
    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("calibration CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}
