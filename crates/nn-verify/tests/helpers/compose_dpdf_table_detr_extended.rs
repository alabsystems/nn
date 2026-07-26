// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended compose tests for Table Transformer DETR pipeline.
//!
//! Supplements `compose_dpdf_table_detr_full.rs` with:
//! - CROWN variants for stages that previously had IBP-only coverage
//! - Verification-recording tests (verify_and_assert) for key stages
//! - ResNet residual block with skip connection
//! - Encoder + positional encoding composition
//! - Encoder monotone tightening analysis
//!
//! Part of #4237: Compose tests for Table Transformer full DETR pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, sinusoidal_pe,
    uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// Dimensions — match compose_dpdf_table_detr_full.rs
const NUM_QUERIES: usize = 8;
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const HEAD_DIM: usize = HIDDEN_DIM / 4; // NUM_HEADS=4
const ENC_SEQ_LEN: usize = 16;
const BACKBONE_CHANNELS: usize = 128;
const SPATIAL_H: usize = 4;
const SPATIAL_W: usize = 4;
const NUM_CLASSES: usize = 6;
const BOX_DIM: usize = 4;
const NUM_RC_CLASSES: usize = 3;
const WEIGHT_MAG: f32 = 0.02;

fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds_min_max(bounds);
    hi - lo
}

fn const_tensor(shape: &[usize], val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), val))
}

/// Build encoder layer: self-attn -> LN -> FFN -> LN with residuals.
fn add_encoder_layer(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let s = [ENC_SEQ_LEN, HIDDEN_DIM];
    let f = [ENC_SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let d = HIDDEN_DIM;

    let ln_w = b.add_input(&format!("{prefix}sa_ln_w"), &[d]);
    let ln_b = b.add_input(&format!("{prefix}sa_ln_b"), &[d]);
    let eps = b.add_input(&format!("{prefix}sa_eps"), &[1]);
    let n = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &s);
    let qw = b.add_input(&format!("{prefix}sa_qw"), &[d, d]);
    let kw = b.add_input(&format!("{prefix}sa_kw"), &[d, d]);
    let vw = b.add_input(&format!("{prefix}sa_vw"), &[d, d]);
    let ow = b.add_input(&format!("{prefix}sa_ow"), &[d, d]);
    let q = b.add_linear(n, qw, None, &s);
    let k = b.add_linear(n, kw, None, &s);
    let v = b.add_linear(n, vw, None, &s);
    let sa = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &s);
    let sa_p = b.add_linear(sa, ow, None, &s);
    let r1 = b.add_binary_add(input, sa_p, &s);

    let ln2w = b.add_input(&format!("{prefix}ffn_ln_w"), &[d]);
    let ln2b = b.add_input(&format!("{prefix}ffn_ln_b"), &[d]);
    let eps2 = b.add_input(&format!("{prefix}ffn_eps"), &[1]);
    let n2 = b.add_layer_norm(r1, eps2, 1, ln2w, ln2b, &s);
    let f1w = b.add_input(&format!("{prefix}ffn1_w"), &[FFN_DIM, d]);
    let f2w = b.add_input(&format!("{prefix}ffn2_w"), &[d, FFN_DIM]);
    let h = b.add_linear(n2, f1w, None, &f);
    let a = b.add_relu(h, &f);
    let o = b.add_linear(a, f2w, None, &s);
    b.add_binary_add(r1, o, &s)
}

fn push_encoder_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let d = HIDDEN_DIM;
    bindings.push(const_tensor(&[d], 1.0)); // sa_ln_w
    bindings.push(const_tensor(&[d], 0.0)); // sa_ln_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    for _ in 0..4 {
        bindings.push(const_tensor(&[d, d], WEIGHT_MAG));
    } // Q,K,V,O
    bindings.push(const_tensor(&[d], 1.0)); // ffn_ln_w
    bindings.push(const_tensor(&[d], 0.0)); // ffn_ln_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(const_tensor(&[FFN_DIM, d], WEIGHT_MAG)); // ffn1
    bindings.push(const_tensor(&[d, FFN_DIM], WEIGHT_MAG)); // ffn2
}

// ===========================================================================
// 16. ResNet residual block with skip connection (IBP + CROWN)
// ===========================================================================

fn build_resnet_residual() -> nn_dsl::tensor_ir::TensorKernelDef {
    let sl = SPATIAL_H * SPATIAL_W;
    let inner = HIDDEN_DIM / 2;
    let mut b = TensorBlockBuilder::new("table_detr_resnet_residual");
    let inp = b.add_input("features", &[sl, HIDDEN_DIM]);
    let w1 = b.add_input("conv1_w", &[inner, HIDDEN_DIM]);
    let h = b.add_linear(inp, w1, None, &[sl, inner]);
    let a = b.add_relu(h, &[sl, inner]);
    let w2 = b.add_input("conv2_w", &[HIDDEN_DIM, inner]);
    let o = b.add_linear(a, w2, None, &[sl, HIDDEN_DIM]);
    let out = b.add_binary_add(inp, o, &[sl, HIDDEN_DIM]);
    b.build(out).expect("valid resnet residual block")
}

fn resnet_residual_bindings() -> Vec<TensorParamBinding> {
    let inner = HIDDEN_DIM / 2;
    vec![
        TensorParamBinding::Variable,
        const_tensor(&[inner, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[HIDDEN_DIM, inner], WEIGHT_MAG),
    ]
}

#[test]
fn test_resnet_residual_block_skip_bounds_ibp() {
    let def = build_resnet_residual();
    let graph = tensor_kernel_to_graph(&def, &resnet_residual_bindings()).expect("graph");
    let sl = SPATIAL_H * SPATIAL_W;
    let output = graph
        .propagate_ibp(&uniform_bounds(&[sl, HIDDEN_DIM], 1.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("ResNet residual IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

#[test]
fn test_resnet_residual_block_skip_bounds_crown() {
    let def = build_resnet_residual();
    let graph = tensor_kernel_to_graph(&def, &resnet_residual_bindings()).expect("graph");
    let sl = SPATIAL_H * SPATIAL_W;
    let input = uniform_bounds(&[sl, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!(
        "ResNet residual CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 17. Backbone feature extraction CROWN
// ===========================================================================

fn build_backbone() -> nn_dsl::tensor_ir::TensorKernelDef {
    let sl = SPATIAL_H * SPATIAL_W;
    let mut b = TensorBlockBuilder::new("table_detr_backbone_crown");
    let f = b.add_input("features", &[sl, BACKBONE_CHANNELS]);
    let w = b.add_input("proj_w", &[HIDDEN_DIM, BACKBONE_CHANNELS]);
    let p = b.add_linear(f, w, None, &[sl, HIDDEN_DIM]);
    let out = b.add_relu(p, &[sl, HIDDEN_DIM]);
    b.build(out).expect("valid backbone")
}

fn backbone_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        const_tensor(&[HIDDEN_DIM, BACKBONE_CHANNELS], WEIGHT_MAG),
    ]
}

#[test]
fn test_backbone_feature_extraction_crown() {
    let def = build_backbone();
    let graph = tensor_kernel_to_graph(&def, &backbone_bindings()).expect("graph");
    let sl = SPATIAL_H * SPATIAL_W;
    let input = uniform_bounds(&[sl, BACKBONE_CHANNELS], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = bounds_min_max(&crown);
    assert!(lo >= -0.01, "backbone CROWN ReLU lower >= 0, got {lo}");
    eprintln!(
        "Backbone CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 18. Encoder self-attention CROWN (2-layer)
// ===========================================================================

#[test]
fn test_encoder_self_attention_crown() {
    let mut b = TensorBlockBuilder::new("table_detr_enc_2layer_crown");
    let f = b.add_input("encoder_input", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_layer(&mut b, f, "enc0_");
    let l2 = add_encoder_layer(&mut b, l1, "enc1_");
    let def = b.build(l2).expect("valid encoder");
    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_bindings(&mut bindings);
    push_encoder_bindings(&mut bindings);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!(
        "Encoder 2-layer CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 19. Encoder with positional encoding
// ===========================================================================

#[test]
fn test_encoder_with_position_encoding_bounds() {
    let sl = SPATIAL_H * SPATIAL_W;
    let mut b = TensorBlockBuilder::new("table_detr_enc_with_pe");
    let f = b.add_input("features", &[sl, HIDDEN_DIM]);
    let pe = b.add_input("pe", &[sl, HIDDEN_DIM]);
    let sum = b.add_binary_add(f, pe, &[sl, HIDDEN_DIM]);
    let out = add_encoder_layer(&mut b, sum, "enc0_");
    let def = b.build(out).expect("valid PE+encoder");
    let pe_data = sinusoidal_pe(sl, HIDDEN_DIM);
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    push_encoder_bindings(&mut bindings);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&[sl, HIDDEN_DIM], 1.0))
        .expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Enc+PE IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 20. BBox regression head CROWN
// ===========================================================================

#[test]
fn test_bbox_regression_head_crown() {
    let mut b = TensorBlockBuilder::new("table_detr_bbox_crown");
    let q = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let w1 = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(q, w1, None, &[NUM_QUERIES, FFN_DIM]);
    let a = b.add_relu(h, &[NUM_QUERIES, FFN_DIM]);
    let w2 = b.add_input("fc2_w", &[BOX_DIM, FFN_DIM]);
    let l = b.add_linear(a, w2, None, &[NUM_QUERIES, BOX_DIM]);
    let out = b.add_sigmoid(l, &[NUM_QUERIES, BOX_DIM]);
    let def = b.build(out).expect("valid bbox head");
    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[FFN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[BOX_DIM, FFN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&crown);
    assert!(lo >= -0.01 && hi <= 1.01, "bbox sigmoid [{lo}, {hi}]");
    eprintln!(
        "BBox CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 21. Row/column detection CROWN
// ===========================================================================

#[test]
fn test_row_column_detection_crown() {
    let mut b = TensorBlockBuilder::new("table_detr_rc_crown");
    let q = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let w = b.add_input("rc_w", &[NUM_RC_CLASSES, HIDDEN_DIM]);
    let l = b.add_linear(q, w, None, &[NUM_QUERIES, NUM_RC_CLASSES]);
    let out = b.add_sigmoid(l, &[NUM_QUERIES, NUM_RC_CLASSES]);
    let def = b.build(out).expect("valid rc");
    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[NUM_RC_CLASSES, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&crown);
    assert!(lo >= -0.01 && hi <= 1.01, "rc sigmoid [{lo}, {hi}]");
    eprintln!(
        "Row/col CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 22. NMS confidence CROWN
// ===========================================================================

#[test]
fn test_nms_confidence_crown() {
    let mut b = TensorBlockBuilder::new("table_detr_nms_crown");
    let q = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let w = b.add_input("conf_w", &[1, HIDDEN_DIM]);
    let l = b.add_linear(q, w, None, &[NUM_QUERIES, 1]);
    let out = b.add_sigmoid(l, &[NUM_QUERIES, 1]);
    let def = b.build(out).expect("valid NMS");
    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[1, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&crown);
    assert!(lo >= -0.01 && hi <= 1.01, "NMS sigmoid [{lo}, {hi}]");
    eprintln!(
        "NMS CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 23. Encoder monotone tightening
// ===========================================================================

#[test]
fn test_encoder_monotone_tightening() {
    let mut b = TensorBlockBuilder::new("table_detr_enc_monotone");
    let f = b.add_input("input", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_layer(&mut b, f, "enc0_");
    let def = b.build(out).expect("valid encoder");
    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_bindings(&mut bindings);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let mut prev_width: Option<f32> = None;
    for &range in &[2.0, 1.0, 0.5] {
        let output = graph
            .propagate_ibp(&uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], range))
            .expect("IBP");
        assert_bounds_valid(&output);
        let w = bound_width(&output);
        eprintln!("Encoder monotone: range={range:.2}, width={w:.6}");
        if let Some(prev) = prev_width {
            assert!(
                w <= prev + 1e-3,
                "monotone: range={range} w={w} > prev={prev}"
            );
        }
        prev_width = Some(w);
    }
}

// ===========================================================================
// 24-27. Verification-recording tests
// ===========================================================================

#[test]
fn test_backbone_verify_and_record() {
    let def = build_backbone();
    let sl = SPATIAL_H * SPATIAL_W;
    let input = uniform_bounds(&[sl, BACKBONE_CHANNELS], 1.0);
    let result = verify_and_assert(
        &def,
        &backbone_bindings(),
        &input,
        "table_detr_backbone_projection",
    );
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[sl, HIDDEN_DIM]
    );
}

#[test]
fn test_cls_head_verify_and_record() {
    let mut b = TensorBlockBuilder::new("table_detr_cls_head_record");
    let q = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let w1 = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(q, w1, None, &[NUM_QUERIES, FFN_DIM]);
    let a = b.add_relu(h, &[NUM_QUERIES, FFN_DIM]);
    let w2 = b.add_input("fc2_w", &[NUM_CLASSES, FFN_DIM]);
    let l = b.add_linear(a, w2, None, &[NUM_QUERIES, NUM_CLASSES]);
    let out = b.add_sigmoid(l, &[NUM_QUERIES, NUM_CLASSES]);
    let def = b.build(out).expect("valid cls head");
    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[FFN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[NUM_CLASSES, FFN_DIM], WEIGHT_MAG),
    ];
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "table_detr_classification_head");
    assert_eq!(result.num_variables, 1);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    assert!(lo >= -0.01 && hi <= 1.01, "cls verified [{lo}, {hi}]");
}

#[test]
fn test_bbox_head_verify_and_record() {
    let mut b = TensorBlockBuilder::new("table_detr_bbox_head_record");
    let q = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let w1 = b.add_input("fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(q, w1, None, &[NUM_QUERIES, FFN_DIM]);
    let a = b.add_relu(h, &[NUM_QUERIES, FFN_DIM]);
    let w2 = b.add_input("fc2_w", &[BOX_DIM, FFN_DIM]);
    let l = b.add_linear(a, w2, None, &[NUM_QUERIES, BOX_DIM]);
    let out = b.add_sigmoid(l, &[NUM_QUERIES, BOX_DIM]);
    let def = b.build(out).expect("valid bbox head");
    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[FFN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[BOX_DIM, FFN_DIM], WEIGHT_MAG),
    ];
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "table_detr_bbox_regression_head");
    assert_eq!(result.num_variables, 1);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    assert!(lo >= -0.01 && hi <= 1.01, "bbox verified [{lo}, {hi}]");
}

#[test]
fn test_full_pipeline_verify_and_record() {
    let mut b = TensorBlockBuilder::new("table_detr_full_pipeline_record");
    let enc = b.add_input("encoder_input", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let enc_out = add_encoder_layer(&mut b, enc, "enc0_");
    let lnw = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let lnb = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_eps", &[1]);
    let n = b.add_layer_norm(enc_out, eps, 1, lnw, lnb, &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let cw = b.add_input("cls_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let l = b.add_linear(n, cw, None, &[ENC_SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(l, &[ENC_SEQ_LEN, NUM_CLASSES]);
    let def = b.build(out).expect("valid pipeline");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_bindings(&mut bindings);
    bindings.push(const_tensor(&[HIDDEN_DIM], 1.0));
    bindings.push(const_tensor(&[HIDDEN_DIM], 0.0));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(const_tensor(&[NUM_CLASSES, HIDDEN_DIM], WEIGHT_MAG));

    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "table_detr_full_enc_cls_pipeline");
    assert_eq!(result.num_variables, 1);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!("Full pipeline verified: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -0.01 && hi <= 1.01, "pipeline sigmoid [{lo}, {hi}]");
}

// ===========================================================================
// 28. Bipartite assignment score bounds (IBP + CROWN)
// ===========================================================================

/// Bipartite assignment in Hungarian matching: the cost matrix is formed by
/// combining classification probability (cls_head -> sigmoid) with box
/// regression distance (box_head -> sigmoid -> L1 distance proxy).
/// The assignment score per query-target pair is a weighted sum of the two
/// costs. Both are sigmoid-bounded in [0, 1], so the combined score must
/// remain bounded.
///
/// Part of #4237.
#[test]
fn test_bipartite_assignment_score_bounds_ibp() {
    let num_targets = 6;
    let mut b = TensorBlockBuilder::new("table_detr_bipartite_score");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification cost: Linear -> sigmoid -> [NUM_QUERIES, NUM_CLASSES]
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let cls_probs = b.add_sigmoid(cls_logits, &[NUM_QUERIES, NUM_CLASSES]);

    // Box regression cost: Linear -> sigmoid -> [NUM_QUERIES, BOX_DIM]
    let box_w = b.add_input("box_w", &[BOX_DIM, HIDDEN_DIM]);
    let box_logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let box_preds = b.add_sigmoid(box_logits, &[NUM_QUERIES, BOX_DIM]);

    // Project class probs to per-target scores: [NUM_QUERIES, NUM_CLASSES] -> [NUM_QUERIES, num_targets]
    let target_cls_w = b.add_input("target_cls_w", &[num_targets, NUM_CLASSES]);
    let cls_cost = b.add_linear(cls_probs, target_cls_w, None, &[NUM_QUERIES, num_targets]);

    // Project box preds to per-target distance: [NUM_QUERIES, BOX_DIM] -> [NUM_QUERIES, num_targets]
    let target_box_w = b.add_input("target_box_w", &[num_targets, BOX_DIM]);
    let box_cost = b.add_linear(box_preds, target_box_w, None, &[NUM_QUERIES, num_targets]);

    // Combined assignment score: cls_cost + box_cost
    let out = b.add_binary_add(cls_cost, box_cost, &[NUM_QUERIES, num_targets]);
    let def = b.build(out).expect("valid bipartite score kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[NUM_CLASSES, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[BOX_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[num_targets, NUM_CLASSES], 1.0 / NUM_CLASSES as f32),
        const_tensor(&[num_targets, BOX_DIM], 1.0 / BOX_DIM as f32),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Bipartite assignment IBP: [{lo:.6}, {hi:.6}]");
    assert!(
        lo.is_finite() && hi.is_finite(),
        "assignment score must be finite"
    );
}

#[test]
fn test_bipartite_assignment_score_bounds_crown() {
    let num_targets = 6;
    let mut b = TensorBlockBuilder::new("table_detr_bipartite_score_crown");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let cls_probs = b.add_sigmoid(cls_logits, &[NUM_QUERIES, NUM_CLASSES]);
    let box_w = b.add_input("box_w", &[BOX_DIM, HIDDEN_DIM]);
    let box_logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let box_preds = b.add_sigmoid(box_logits, &[NUM_QUERIES, BOX_DIM]);
    let target_cls_w = b.add_input("target_cls_w", &[num_targets, NUM_CLASSES]);
    let cls_cost = b.add_linear(cls_probs, target_cls_w, None, &[NUM_QUERIES, num_targets]);
    let target_box_w = b.add_input("target_box_w", &[num_targets, BOX_DIM]);
    let box_cost = b.add_linear(box_preds, target_box_w, None, &[NUM_QUERIES, num_targets]);
    let out = b.add_binary_add(cls_cost, box_cost, &[NUM_QUERIES, num_targets]);
    let def = b.build(out).expect("valid bipartite score kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[NUM_CLASSES, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[BOX_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[num_targets, NUM_CLASSES], 1.0 / NUM_CLASSES as f32),
        const_tensor(&[num_targets, BOX_DIM], 1.0 / BOX_DIM as f32),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!(
        "Bipartite assignment CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 29. Table cell spanning prediction bounds (IBP + CROWN)
// ===========================================================================

/// Table cell spanning prediction: for each detected cell, predict whether it
/// spans multiple rows (rowspan) or columns (colspan). Modeled as a 2-head
/// predictor: Linear -> ReLU -> Linear -> sigmoid for each of rowspan and
/// colspan confidence.
///
/// Part of #4237.
#[test]
fn test_cell_spanning_prediction_ibp() {
    let span_dim = 2; // rowspan_conf, colspan_conf
    let mut b = TensorBlockBuilder::new("table_detr_cell_spanning");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // Spanning head: Linear -> ReLU -> Linear -> sigmoid
    let fc1_w = b.add_input("span_fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(queries, fc1_w, None, &[NUM_QUERIES, FFN_DIM]);
    let a = b.add_relu(h, &[NUM_QUERIES, FFN_DIM]);
    let fc2_w = b.add_input("span_fc2_w", &[span_dim, FFN_DIM]);
    let logits = b.add_linear(a, fc2_w, None, &[NUM_QUERIES, span_dim]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, span_dim]);
    let def = b.build(out).expect("valid spanning kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[FFN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[span_dim, FFN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Cell spanning IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo >= -0.01 && hi <= 1.01, "spanning sigmoid [{lo}, {hi}]");
}

#[test]
fn test_cell_spanning_prediction_crown() {
    let span_dim = 2;
    let mut b = TensorBlockBuilder::new("table_detr_cell_spanning_crown");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let fc1_w = b.add_input("span_fc1_w", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(queries, fc1_w, None, &[NUM_QUERIES, FFN_DIM]);
    let a = b.add_relu(h, &[NUM_QUERIES, FFN_DIM]);
    let fc2_w = b.add_input("span_fc2_w", &[span_dim, FFN_DIM]);
    let logits = b.add_linear(a, fc2_w, None, &[NUM_QUERIES, span_dim]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, span_dim]);
    let def = b.build(out).expect("valid spanning kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[FFN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        const_tensor(&[span_dim, FFN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&crown);
    assert!(
        lo >= -0.01 && hi <= 1.01,
        "spanning sigmoid CROWN [{lo}, {hi}]"
    );
    eprintln!(
        "Cell spanning CROWN: IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
}

// ===========================================================================
// 30. Multi-scale feature map bounds (IBP)
// ===========================================================================

/// Multi-scale features: backbone produces features at two spatial resolutions
/// (e.g., layer3 and layer4 of ResNet). Each is projected to HIDDEN_DIM and
/// their flattened representations are concatenated before entering the
/// encoder. Verifies that the concatenated multi-scale features have finite,
/// non-vacuous bounds.
///
/// Part of #4237.
#[test]
fn test_multi_scale_feature_map_ibp() {
    let scale1_spatial = SPATIAL_H * SPATIAL_W; // 16
    let scale2_spatial = (SPATIAL_H / 2) * (SPATIAL_W / 2); // 4
    let scale1_ch = BACKBONE_CHANNELS / 2; // 64
    let scale2_ch = BACKBONE_CHANNELS; // 128
    let combined_seq = scale1_spatial + scale2_spatial; // 20

    let mut b = TensorBlockBuilder::new("table_detr_multiscale_features");

    // Scale 1 (higher resolution, fewer channels): project + flatten
    let feat1 = b.add_input("feat_scale1", &[scale1_spatial, scale1_ch]);
    let proj1_w = b.add_input("proj1_w", &[HIDDEN_DIM, scale1_ch]);
    let proj1 = b.add_linear(feat1, proj1_w, None, &[scale1_spatial, HIDDEN_DIM]);
    let proj1_act = b.add_relu(proj1, &[scale1_spatial, HIDDEN_DIM]);

    // Scale 2 (lower resolution, more channels): project + flatten
    let feat2 = b.add_input("feat_scale2", &[scale2_spatial, scale2_ch]);
    let proj2_w = b.add_input("proj2_w", &[HIDDEN_DIM, scale2_ch]);
    let proj2 = b.add_linear(feat2, proj2_w, None, &[scale2_spatial, HIDDEN_DIM]);
    let proj2_act = b.add_relu(proj2, &[scale2_spatial, HIDDEN_DIM]);

    // Concatenate along sequence dimension: [scale1+scale2, HIDDEN_DIM]
    let concat = b.add_concat(&[proj1_act, proj2_act], 0, &[combined_seq, HIDDEN_DIM]);
    let def = b.build(concat).expect("valid multiscale kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // feat_scale1
        const_tensor(&[HIDDEN_DIM, scale1_ch], WEIGHT_MAG),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[scale2_spatial, scale2_ch]),
            0.5f32,
        )), // feat_scale2 (constant for single-variable)
        const_tensor(&[HIDDEN_DIM, scale2_ch], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[scale1_spatial, scale1_ch], 2.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Multi-scale features IBP: [{lo:.6}, {hi:.6}]");
    assert!(
        lo.is_finite() && hi.is_finite(),
        "multi-scale must be finite"
    );
    // ReLU lower bound should be >= 0 for the projected part
    assert!(lo >= -0.01, "multi-scale with ReLU lower >= 0, got {lo}");
}

// ===========================================================================
// 31. Decoder layer norm bounds (IBP + CROWN)
// ===========================================================================

/// Decoder LayerNorm in isolation: verifies that LayerNorm applied to decoder
/// query representations produces finite, well-bounded output. This tests
/// the normalization stability that is critical for the downstream cross-
/// attention and FFN blocks.
///
/// Part of #4237.
#[test]
fn test_decoder_layer_norm_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("table_detr_decoder_layernorm");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let out = b.add_layer_norm(queries, ln_eps, 1, ln_w, ln_b, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid decoder LN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[HIDDEN_DIM], 1.0),
        const_tensor(&[HIDDEN_DIM], 0.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_arr, _) = output.lower_upper();
    assert_eq!(
        lo_arr.shape(),
        &[NUM_QUERIES, HIDDEN_DIM],
        "LN output shape"
    );
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Decoder LayerNorm IBP: [{lo:.6}, {hi:.6}]");
    assert!(lo.is_finite() && hi.is_finite(), "LN bounds must be finite");
}

#[test]
fn test_decoder_layer_norm_bounds_crown() {
    let mut b = TensorBlockBuilder::new("table_detr_decoder_layernorm_crown");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let out = b.add_layer_norm(queries, ln_eps, 1, ln_w, ln_b, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid decoder LN kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_tensor(&[HIDDEN_DIM], 1.0),
        const_tensor(&[HIDDEN_DIM], 0.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp);
    let (_, crown, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&crown);
    eprintln!(
        "Decoder LayerNorm CROWN: [{lo:.6}, {hi:.6}], IBP w={:.6}, CROWN w={:.6}",
        bound_width(&ibp),
        bound_width(&crown)
    );
    assert!(lo.is_finite() && hi.is_finite(), "LN CROWN must be finite");
}
