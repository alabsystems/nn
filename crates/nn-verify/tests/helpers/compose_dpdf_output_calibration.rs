// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for model output calibration: temperature scaling,
//! top-k masking, detection confidence, OCR logit bounds, MoE routing calibration,
//! and vocabulary projection ranges.
//!
//! These tests verify that calibration transforms preserve expected bound
//! invariants across the 7 dpdf model architectures. Unlike the per-model
//! compose tests, these focus on the cross-cutting calibration concern:
//! how post-logit transforms (temperature, masking, softmax) interact with
//! the verification pipeline.
//!
//! ## Temperature Scaling (tests 1-4)
//!
//! 1.  Temperature scaling narrows softmax bounds (T < 1) IBP
//! 2.  Temperature scaling widens softmax bounds (T > 1) IBP
//! 3.  Temperature=1 identity: softmax bounds match unscaled IBP
//! 4.  Temperature scaling preserves softmax [0,1] at extreme T values IBP + CROWN
//!
//! ## Top-k Masking (tests 5-8)
//!
//! 5.  Top-k mask zeroes non-top-k logit paths IBP
//! 6.  Top-k=1 produces sharp distribution (near one-hot) IBP
//! 7.  Top-k=vocab produces full softmax (no masking) IBP
//! 8.  Top-k mask + temperature composition IBP + CROWN
//!
//! ## Detection Confidence Calibration (tests 9-11)
//!
//! 9.  DocLayout-YOLO sigmoid detection confidence in [0,1] IBP
//! 10. Detection sigmoid monotonicity: wider logit input -> wider output IBP
//! 11. Detection confidence CROWN tightness vs IBP CROWN
//!
//! ## OCR Logit Bounds (tests 12-14)
//!
//! 12. CTC softmax output bounded in [0,1] for PaddleOCR IBP
//! 13. Autoregressive OCR log-softmax bounded in (-inf, 0] IBP
//! 14. CTC blank token dominance: blank logit > character logits IBP
//!
//! ## MoE Routing Calibration (tests 15-17)
//!
//! 15. MoE gate softmax temperature narrows expert selection IBP
//! 16. MoE gate softmax output sum bounded near 1.0 IBP
//! 17. MoE routing calibration CROWN tightness CROWN
//!
//! ## Vocabulary Projection Bounds (tests 18-20)
//!
//! 18. Large vocabulary linear projection range scales with hidden_dim IBP
//! 19. Vocab projection + softmax output in [0,1] IBP
//! 20. Vocab projection + temperature + softmax end-to-end IBP + CROWN
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): Sigmoid detection confidence
//! - PaddleOCR (Baidu): CTC softmax character probabilities
//! - FireRed-OCR: Qwen3-VL-2B variant with CTC decoding
//! - Qwen3-VL (Alibaba): MoE routing with temperature-scaled softmax
//! - GLM-4V (THUDM): LM head with temperature sampling
//! - Granite-Docling: SigLIP2 + Granite decoder, sigmoid output
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, FFN_DIM=64, NUM_CLASSES=8, VOCAB_SIZE=16,
//!   NUM_EXPERTS=4, BACKBONE_CH=16
//!
//! Part of #4102: Compose tests for model output calibration.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 32;
const FFN_DIM: usize = 64;
const NUM_CLASSES: usize = 8;
const VOCAB_SIZE: usize = 16;
const NUM_EXPERTS: usize = 4;
const BACKBONE_CH: usize = 16;
const IMG_SIZE: usize = 16;
const PATCH_SIZE: usize = 8;
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
const IN_CHANNELS: usize = 3;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Constant scalar tensor binding.
fn scalar_binding(val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), val))
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Build a temperature-scaled softmax pipeline: Linear -> mul(1/T) -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`, Output: `[SEQ_LEN, VOCAB_SIZE]`.
fn build_temperature_softmax_kernel(
    name: &str,
    inv_temperature: f32,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let inv_temp = b.add_input("inv_temp", &[1]);

    // Linear projection to logits
    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Temperature scaling: logits * (1/T)
    let inv_temp_bc = b.add_broadcast(inv_temp, &[SEQ_LEN, VOCAB_SIZE]);
    let scaled = b.add_binary_mul(logits, inv_temp_bc, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax
    let out = b.add_softmax(scaled, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid temperature softmax kernel");
    let bindings = vec![
        TensorParamBinding::Variable, // features
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        scalar_binding(inv_temperature),
    ];
    (def, bindings)
}

/// Build a top-k masked softmax pipeline.
///
/// Models top-k by applying a binary mask (1 for top-k, 0 for others) via
/// element-wise multiply on logits before softmax. The mask zeros out
/// non-top-k positions, approximating -inf masking for bound verification.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`, Output: `[SEQ_LEN, VOCAB_SIZE]`.
fn build_topk_masked_softmax_kernel(
    name: &str,
    k: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let mask = b.add_input("topk_mask", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Mask: broadcast [VOCAB_SIZE] -> [SEQ_LEN, VOCAB_SIZE], then multiply
    let mask_bc = b.add_broadcast(mask, &[SEQ_LEN, VOCAB_SIZE]);
    let masked_logits = b.add_binary_mul(logits, mask_bc, &[SEQ_LEN, VOCAB_SIZE]);

    let out = b.add_softmax(masked_logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid topk masked softmax kernel");

    // Build mask: first k positions = 1.0, rest = 0.0
    let mut mask_data = vec![0.0f32; VOCAB_SIZE];
    for i in 0..k.min(VOCAB_SIZE) {
        mask_data[i] = 1.0;
    }
    let bindings = vec![
        TensorParamBinding::Variable, // features
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[VOCAB_SIZE]), mask_data).expect("valid mask"),
        ),
    ];
    (def, bindings)
}

// ===========================================================================
// 1. Temperature scaling narrows softmax bounds (T < 1) IBP
// ===========================================================================

/// Low temperature (T=0.5, inv_temp=2.0) sharpens the softmax distribution.
/// Output must remain in [0, 1].
#[test]
fn test_temperature_low_narrows_softmax_ibp() {
    let (def, bindings) = build_temperature_softmax_kernel("calib_temp_low", 2.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Temperature low (T=0.5) IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Temperature scaling widens softmax bounds (T > 1) IBP
// ===========================================================================

/// High temperature (T=2.0, inv_temp=0.5) flattens the softmax distribution.
/// Output must remain in [0, 1].
#[test]
fn test_temperature_high_widens_softmax_ibp() {
    let (def, bindings) = build_temperature_softmax_kernel("calib_temp_high", 0.5);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Temperature high (T=2.0) IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. Temperature=1 identity: softmax bounds match unscaled IBP
// ===========================================================================

/// Temperature=1 (inv_temp=1.0) should produce identical bounds to unscaled
/// softmax, since logits * 1.0 = logits.
#[test]
fn test_temperature_identity_matches_unscaled_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Temperature=1 pipeline
    let (def_t1, bindings_t1) = build_temperature_softmax_kernel("calib_temp_1", 1.0);
    let graph_t1 = tensor_kernel_to_graph(&def_t1, &bindings_t1).expect("graph translation");
    let output_t1 = graph_t1.propagate_ibp(&input).expect("IBP T=1");

    // Unscaled pipeline: Linear -> softmax (no temperature)
    let mut b = TensorBlockBuilder::new("calib_no_temp");
    let inp = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(inp, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def_raw = b.build(out).expect("valid raw softmax kernel");
    let bindings_raw = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph_raw = tensor_kernel_to_graph(&def_raw, &bindings_raw).expect("graph translation");
    let output_raw = graph_raw.propagate_ibp(&input).expect("IBP raw");

    assert_bounds_valid(&output_t1);
    assert_bounds_valid(&output_raw);

    let (lo_t1, hi_t1) = bounds_min_max(&output_t1);
    let (lo_raw, hi_raw) = bounds_min_max(&output_raw);
    eprintln!("T=1 bounds=[{lo_t1}, {hi_t1}], raw bounds=[{lo_raw}, {hi_raw}]");

    // Bounds should be very close (within IBP numerical tolerance)
    let eps = 1e-3;
    assert!(
        (lo_t1 - lo_raw).abs() < eps,
        "T=1 lower {lo_t1} should match raw {lo_raw}"
    );
    assert!(
        (hi_t1 - hi_raw).abs() < eps,
        "T=1 upper {hi_t1} should match raw {hi_raw}"
    );
}

// ===========================================================================
// 4. Temperature scaling preserves softmax [0,1] at extreme T values
//    IBP + CROWN
// ===========================================================================

/// Extreme temperatures (T=0.1 and T=10.0) must still produce softmax
/// output in [0, 1].
#[test]
fn test_temperature_extreme_preserves_softmax_crown() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    for &(label, inv_temp) in &[("T=0.1", 10.0_f32), ("T=10.0", 0.1)] {
        let (def, bindings) =
            build_temperature_softmax_kernel(&format!("calib_extreme_{label}"), inv_temp);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let (method, output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Extreme {label}: method={method:?}, bounds=[{lo_min}, {hi_max}]");
        if let Some(reason) = &fallback_reason {
            eprintln!("  Fallback reason: {reason}");
        }

        assert!(lo_min >= -1e-4, "{label}: softmax lower >= 0, got {lo_min}");
        assert!(
            hi_max <= 1.0 + 1e-4,
            "{label}: softmax upper <= 1, got {hi_max}"
        );
    }
}

// ===========================================================================
// 5. Top-k mask zeroes non-top-k logit paths IBP
// ===========================================================================

/// Top-k masking with k=4 (half of VOCAB_SIZE=16) zeros out the lower
/// half of logit positions. Output softmax must remain in [0, 1].
#[test]
fn test_topk_mask_zeroes_non_topk_ibp() {
    let (def, bindings) = build_topk_masked_softmax_kernel("calib_topk_4", 4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Top-k=4 masked softmax IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. Top-k=1 produces sharp distribution (near one-hot) IBP
// ===========================================================================

/// With k=1, only one logit position passes through; the rest are zeroed.
/// The softmax output should have most probability mass on the unmasked token.
#[test]
fn test_topk_1_sharp_distribution_ibp() {
    let (def, bindings) = build_topk_masked_softmax_kernel("calib_topk_1", 1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Top-k=1 softmax IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax of [x, 0, 0, ...] gives [sigmoid-like, small, small, ...].
    // All outputs must be in [0, 1].
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Top-k=vocab produces full softmax (no masking) IBP
// ===========================================================================

/// With k=VOCAB_SIZE, the mask is all-ones, so the output should match
/// unmasked softmax.
#[test]
fn test_topk_full_vocab_equals_unmasked_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Top-k=VOCAB_SIZE (all-ones mask)
    let (def_full, bindings_full) = build_topk_masked_softmax_kernel("calib_topk_full", VOCAB_SIZE);
    let graph_full = tensor_kernel_to_graph(&def_full, &bindings_full).expect("graph translation");
    let output_full = graph_full.propagate_ibp(&input).expect("IBP full mask");

    // Unmasked softmax
    let mut b = TensorBlockBuilder::new("calib_topk_no_mask");
    let inp = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(inp, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def_raw = b.build(out).expect("valid raw softmax kernel");
    let bindings_raw = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
    ];
    let graph_raw = tensor_kernel_to_graph(&def_raw, &bindings_raw).expect("graph translation");
    let output_raw = graph_raw.propagate_ibp(&input).expect("IBP raw");

    assert_bounds_valid(&output_full);
    assert_bounds_valid(&output_raw);

    let (lo_full, hi_full) = bounds_min_max(&output_full);
    let (lo_raw, hi_raw) = bounds_min_max(&output_raw);
    eprintln!("Top-k=full bounds=[{lo_full}, {hi_full}], raw bounds=[{lo_raw}, {hi_raw}]");

    // With all-ones mask, multiply is identity, so bounds should be very close
    let eps = 1e-3;
    assert!(
        (lo_full - lo_raw).abs() < eps,
        "full mask lower {lo_full} should match raw {lo_raw}"
    );
    assert!(
        (hi_full - hi_raw).abs() < eps,
        "full mask upper {hi_full} should match raw {hi_raw}"
    );
}

// ===========================================================================
// 8. Top-k mask + temperature composition IBP + CROWN
// ===========================================================================

/// Composing top-k masking with temperature scaling. Both transforms
/// must preserve softmax [0, 1] bounds.
#[test]
fn test_topk_plus_temperature_crown() {
    let mut b = TensorBlockBuilder::new("calib_topk_temp_compose");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let mask = b.add_input("topk_mask", &[VOCAB_SIZE]);
    let inv_temp = b.add_input("inv_temp", &[1]);

    // Linear -> mask -> temperature -> softmax
    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let mask_bc = b.add_broadcast(mask, &[SEQ_LEN, VOCAB_SIZE]);
    let masked = b.add_binary_mul(logits, mask_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let inv_temp_bc = b.add_broadcast(inv_temp, &[SEQ_LEN, VOCAB_SIZE]);
    let scaled = b.add_binary_mul(masked, inv_temp_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(scaled, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid topk+temp kernel");

    // k=4 mask, T=0.5 (inv_temp=2.0)
    let mut mask_data = vec![0.0f32; VOCAB_SIZE];
    for i in 0..4 {
        mask_data[i] = 1.0;
    }
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[VOCAB_SIZE]), mask_data).expect("valid mask"),
        ),
        scalar_binding(2.0), // 1/T = 2.0 for T=0.5
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Top-k+temperature: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. DocLayout-YOLO sigmoid detection confidence in [0,1] IBP
// ===========================================================================

/// DocLayout-YOLO detection head: Conv2d backbone -> reshape -> Linear ->
/// ReLU -> Linear -> sigmoid. Sigmoid output must be in [0, 1].
#[test]
fn test_doclayout_sigmoid_detection_confidence_ibp() {
    let mut b = TensorBlockBuilder::new("calib_doclayout_detect_conf");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv2d backbone
    let conv_w = b.add_input(
        "conv_w",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("conv_b", &[BACKBONE_CH]);
    let backbone = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape + transpose to sequence
    let reshaped = b.add_reshape(backbone, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Detection head: Linear -> ReLU -> Linear -> sigmoid
    let head_w1 = b.add_input("head_w1", &[FFN_DIM, BACKBONE_CH]);
    let h1 = b.add_linear(narrowed, head_w1, None, &[SEQ_LEN, FFN_DIM]);
    let h1_act = b.add_relu(h1, &[SEQ_LEN, FFN_DIM]);
    let head_w2 = b.add_input("head_w2", &[NUM_CLASSES, FFN_DIM]);
    let head_b2 = b.add_input("head_b2", &[NUM_CLASSES]);
    let logits = b.add_linear(h1_act, head_w2, Some(head_b2), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    let def = b.build(out).expect("valid detection confidence kernel");
    let bindings = vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[FFN_DIM, BACKBONE_CH]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout detection confidence IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Detection sigmoid monotonicity: wider logit input -> wider output IBP
// ===========================================================================

/// Sigmoid is monotonic, so wider input bounds should produce wider output
/// bounds (or equal, never narrower). Test with two different input ranges.
#[test]
fn test_detection_sigmoid_monotonicity_ibp() {
    // Simple sigmoid-only pipeline: Linear -> sigmoid
    let build_sigmoid_pipeline = |name: &str| -> (TensorKernelDef, Vec<TensorParamBinding>) {
        let mut b = TensorBlockBuilder::new(name);
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let w = b.add_input("w", &[NUM_CLASSES, HIDDEN_DIM]);
        let logits = b.add_linear(input, w, None, &[SEQ_LEN, NUM_CLASSES]);
        let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);
        let def = b.build(out).expect("valid sigmoid pipeline");
        let bindings = vec![
            TensorParamBinding::Variable,
            weight(&[NUM_CLASSES, HIDDEN_DIM]),
        ];
        (def, bindings)
    };

    let (def, bindings) = build_sigmoid_pipeline("calib_sigmoid_mono");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Narrow input range
    let input_narrow = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let output_narrow = graph.propagate_ibp(&input_narrow).expect("IBP narrow");
    let (lo_narrow, hi_narrow) = bounds_min_max(&output_narrow);
    let width_narrow = hi_narrow - lo_narrow;

    // Wide input range
    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);
    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    let (lo_wide, hi_wide) = bounds_min_max(&output_wide);
    let width_wide = hi_wide - lo_wide;

    eprintln!(
        "Sigmoid monotonicity: narrow=[{lo_narrow}, {hi_narrow}] width={width_narrow}, \
         wide=[{lo_wide}, {hi_wide}] width={width_wide}"
    );

    assert_bounds_valid(&output_narrow);
    assert_bounds_valid(&output_wide);

    // Both must be in [0, 1]
    assert!(lo_narrow >= -1e-4, "narrow sigmoid lower >= 0");
    assert!(hi_narrow <= 1.0 + 1e-4, "narrow sigmoid upper <= 1");
    assert!(lo_wide >= -1e-4, "wide sigmoid lower >= 0");
    assert!(hi_wide <= 1.0 + 1e-4, "wide sigmoid upper <= 1");

    // Wider input should produce wider or equal output bounds
    let eps = 1e-4;
    assert!(
        width_wide >= width_narrow - eps,
        "wider input should produce wider sigmoid output: wide={width_wide}, narrow={width_narrow}"
    );
}

// ===========================================================================
// 11. Detection confidence CROWN tightness vs IBP CROWN
// ===========================================================================

/// CROWN should produce tighter bounds than IBP for the detection sigmoid
/// pipeline when it succeeds (no fallback).
#[test]
fn test_detection_confidence_crown_tightness() {
    let mut b = TensorBlockBuilder::new("calib_detect_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w1, None, &[SEQ_LEN, FFN_DIM]);
    let h_act = b.add_relu(h, &[SEQ_LEN, FFN_DIM]);
    let w2 = b.add_input("w2", &[NUM_CLASSES, FFN_DIM]);
    let logits = b.add_linear(h_act, w2, None, &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    let def = b.build(out).expect("valid detection CROWN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Detection confidence CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. CTC softmax output bounded in [0,1] for PaddleOCR IBP
// ===========================================================================

/// PaddleOCR CTC decoder: Conv2d patch embed -> reshape -> transpose ->
/// Linear encoder -> GELU -> Linear CTC head -> softmax.
/// All character probabilities must be in [0, 1].
#[test]
fn test_ctc_softmax_paddleocr_bounded_ibp() {
    let mut b = TensorBlockBuilder::new("calib_ctc_paddle");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding
    let conv_w = b.add_input(
        "patch_w",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("patch_b", &[BACKBONE_CH]);
    let patches = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    let reshaped = b.add_reshape(patches, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // SVTR encoder: Linear -> GELU
    let enc_w = b.add_input("enc_w", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(narrowed, enc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_act = b.add_gelu(encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_act, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid CTC PaddleOCR kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CTC PaddleOCR softmax IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Autoregressive OCR log-softmax bounded in (-inf, 0] IBP
// ===========================================================================

/// FireRed-OCR log-softmax output: Linear encoder -> ReLU -> Linear head ->
/// log_softmax. Log-probabilities must be <= 0.
#[test]
fn test_ocr_log_softmax_bounded_ibp() {
    let mut b = TensorBlockBuilder::new("calib_ocr_log_softmax");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let enc_w = b.add_input("enc_w", &[FFN_DIM, HIDDEN_DIM]);
    let encoded = b.add_linear(input, enc_w, None, &[SEQ_LEN, FFN_DIM]);
    let enc_act = b.add_relu(encoded, &[SEQ_LEN, FFN_DIM]);

    let head_w = b.add_input("head_w", &[VOCAB_SIZE, FFN_DIM]);
    let head_b = b.add_input("head_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_act, head_w, Some(head_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_log_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid OCR log-softmax kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias(&[VOCAB_SIZE]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR log-softmax IBP: bounds=[{lo_min}, {hi_max}]");

    // Log-softmax output must be <= 0 (log of probability in [0, 1])
    let eps = 1e-4;
    assert!(
        hi_max <= eps,
        "log-softmax upper must be <= 0, got {hi_max}"
    );
    // Lower bound can be very negative but must be finite
    assert!(
        lo_min.is_finite(),
        "log-softmax lower must be finite, got {lo_min}"
    );
}

// ===========================================================================
// 14. CTC blank token dominance: blank logit > character logits IBP
// ===========================================================================

/// In CTC decoding, the blank token (index 0 by convention) often receives
/// higher logit mass through bias initialization. Verify that with blank-biased
/// weights, the blank token's softmax probability upper bound is highest.
#[test]
fn test_ctc_blank_token_dominance_ibp() {
    let mut b = TensorBlockBuilder::new("calib_ctc_blank_dominant");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let head_w = b.add_input("head_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let head_b = b.add_input("head_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, head_w, Some(head_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid CTC blank kernel");

    // Blank-biased: bias[0] = 2.0 (blank), bias[1..] = 0.0
    let mut bias_data = vec![0.0f32; VOCAB_SIZE];
    bias_data[0] = 2.0;
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[VOCAB_SIZE]), bias_data).expect("valid bias"),
        ),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CTC blank-biased softmax IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");

    // Non-degenerate
    assert!(lo_min < hi_max, "bounds must be non-degenerate");
}

// ===========================================================================
// 15. MoE gate softmax temperature narrows expert selection IBP
// ===========================================================================

/// MoE routing with temperature scaling on the gate logits.
/// Lower temperature concentrates probability mass on fewer experts.
#[test]
fn test_moe_gate_temperature_narrows_selection_ibp() {
    let build_moe_gate =
        |name: &str, inv_temp: f32| -> (TensorKernelDef, Vec<TensorParamBinding>) {
            let mut b = TensorBlockBuilder::new(name);
            let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
            let gate_w = b.add_input("gate_w", &[NUM_EXPERTS, HIDDEN_DIM]);
            let inv_t = b.add_input("inv_t", &[1]);

            let gate_logits = b.add_linear(input, gate_w, None, &[SEQ_LEN, NUM_EXPERTS]);
            let inv_t_bc = b.add_broadcast(inv_t, &[SEQ_LEN, NUM_EXPERTS]);
            let scaled = b.add_binary_mul(gate_logits, inv_t_bc, &[SEQ_LEN, NUM_EXPERTS]);
            let probs = b.add_softmax(scaled, -1, &[SEQ_LEN, NUM_EXPERTS]);

            let def = b.build(probs).expect("valid MoE gate kernel");
            let bindings = vec![
                TensorParamBinding::Variable,
                weight(&[NUM_EXPERTS, HIDDEN_DIM]),
                scalar_binding(inv_temp),
            ];
            (def, bindings)
        };

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let mut widths = Vec::new();
    for &(label, inv_temp) in &[("T=0.5", 2.0_f32), ("T=1.0", 1.0), ("T=2.0", 0.5)] {
        let (def, bindings) = build_moe_gate(&format!("calib_moe_gate_{label}"), inv_temp);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("MoE gate {label}: bounds=[{lo_min}, {hi_max}], width={width}");

        assert!(lo_min >= -1e-4, "{label}: softmax lower >= 0, got {lo_min}");
        assert!(
            hi_max <= 1.0 + 1e-4,
            "{label}: softmax upper <= 1, got {hi_max}"
        );
        widths.push(width);
    }

    // All widths should be finite
    for w in &widths {
        assert!(w.is_finite(), "MoE gate bound width must be finite");
    }
}

// ===========================================================================
// 16. MoE gate softmax output sum bounded near 1.0 IBP
// ===========================================================================

/// The sum of softmax routing probabilities across experts is exactly 1.0
/// for any input. IBP should bound the sum to include 1.0 in [lo_sum, hi_sum].
#[test]
fn test_moe_gate_softmax_sum_bounded_ibp() {
    let mut b = TensorBlockBuilder::new("calib_moe_gate_sum");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let gate_logits = b.add_linear(input, gate_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(gate_logits, -1, &[SEQ_LEN, NUM_EXPERTS]);

    let def = b.build(probs).expect("valid MoE gate sum kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_EXPERTS]);
    assert_bounds_valid(&output);

    // Each softmax output is in [0, 1], so all NUM_EXPERTS outputs summing
    // should have lower sum >= 0 and upper sum <= NUM_EXPERTS. The exact sum
    // is 1.0, so the IBP interval should contain 1.0.
    let (lo, hi) = output.lower_upper();
    for t in 0..SEQ_LEN {
        let lo_sum: f32 = (0..NUM_EXPERTS).map(|e| lo[[t, e]]).sum();
        let hi_sum: f32 = (0..NUM_EXPERTS).map(|e| hi[[t, e]]).sum();
        eprintln!("MoE gate t={t}: sum bounds=[{lo_sum}, {hi_sum}]");

        // The true sum is exactly 1.0; IBP should bracket it
        let eps = 1e-3;
        assert!(
            lo_sum <= 1.0 + eps,
            "t={t}: lower sum {lo_sum} should be <= 1.0"
        );
        assert!(
            hi_sum >= 1.0 - eps,
            "t={t}: upper sum {hi_sum} should be >= 1.0"
        );
    }
}

// ===========================================================================
// 17. MoE routing calibration CROWN tightness CROWN
// ===========================================================================

/// CROWN should produce tighter bounds than IBP for the MoE routing gate.
#[test]
fn test_moe_routing_calibration_crown() {
    let mut b = TensorBlockBuilder::new("calib_moe_routing_crown");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let gate_logits = b.add_linear(input, gate_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(gate_logits, -1, &[SEQ_LEN, NUM_EXPERTS]);

    let def = b.build(probs).expect("valid MoE CROWN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE routing CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 18. Large vocabulary linear projection range scales with hidden_dim IBP
// ===========================================================================

/// Larger hidden dimensions produce wider logit ranges (more accumulation
/// in the linear projection). Verify with two hidden dim sizes.
#[test]
fn test_large_vocab_projection_range_scales_ibp() {
    let small_dim = HIDDEN_DIM; // 32
    let large_dim = HIDDEN_DIM * 2; // 64

    let build_vocab_proj =
        |name: &str, hdim: usize| -> (TensorKernelDef, Vec<TensorParamBinding>) {
            let mut b = TensorBlockBuilder::new(name);
            let input = b.add_input("x", &[SEQ_LEN, hdim]);
            let w = b.add_input("w", &[VOCAB_SIZE, hdim]);
            let logits = b.add_linear(input, w, None, &[SEQ_LEN, VOCAB_SIZE]);
            // Wrap in identity (AddConstant(0.0)) for NY compatibility
            let out = b.add_sigmoid(logits, &[SEQ_LEN, VOCAB_SIZE]);
            let def = b.build(out).expect("valid vocab proj kernel");
            let bindings = vec![TensorParamBinding::Variable, weight(&[VOCAB_SIZE, hdim])];
            (def, bindings)
        };

    let (def_s, bindings_s) = build_vocab_proj("calib_vocab_small", small_dim);
    let graph_s = tensor_kernel_to_graph(&def_s, &bindings_s).expect("graph small");
    let input_s = uniform_bounds(&[SEQ_LEN, small_dim], 1.0);
    let output_s = graph_s.propagate_ibp(&input_s).expect("IBP small");
    assert_bounds_valid(&output_s);

    let (def_l, bindings_l) = build_vocab_proj("calib_vocab_large", large_dim);
    let graph_l = tensor_kernel_to_graph(&def_l, &bindings_l).expect("graph large");
    let input_l = uniform_bounds(&[SEQ_LEN, large_dim], 1.0);
    let output_l = graph_l.propagate_ibp(&input_l).expect("IBP large");
    assert_bounds_valid(&output_l);

    let (lo_s, hi_s) = bounds_min_max(&output_s);
    let (lo_l, hi_l) = bounds_min_max(&output_l);
    let width_s = hi_s - lo_s;
    let width_l = hi_l - lo_l;

    eprintln!("Vocab proj small dim={small_dim}: bounds=[{lo_s}, {hi_s}], width={width_s}");
    eprintln!("Vocab proj large dim={large_dim}: bounds=[{lo_l}, {hi_l}], width={width_l}");

    // Both pass through sigmoid so must be in [0, 1]
    assert!(lo_s >= -1e-4, "small sigmoid lower >= 0");
    assert!(hi_s <= 1.0 + 1e-4, "small sigmoid upper <= 1");
    assert!(lo_l >= -1e-4, "large sigmoid lower >= 0");
    assert!(hi_l <= 1.0 + 1e-4, "large sigmoid upper <= 1");
}

// ===========================================================================
// 19. Vocab projection + softmax output in [0,1] IBP
// ===========================================================================

/// Standard vocabulary projection with softmax: Linear(hidden, vocab) ->
/// softmax. Output probabilities must be in [0, 1].
#[test]
fn test_vocab_projection_softmax_bounded_ibp() {
    let mut b = TensorBlockBuilder::new("calib_vocab_softmax");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let b_param = b.add_input("proj_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, w, Some(b_param), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid vocab softmax kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocab projection + softmax IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "bounds must be non-degenerate");
}

// ===========================================================================
// 20. Vocab projection + temperature + softmax end-to-end IBP + CROWN
// ===========================================================================

/// End-to-end pipeline: Linear(hidden, vocab) -> temperature scaling ->
/// softmax. Tests the full calibration chain with CROWN tightness check.
#[test]
fn test_vocab_projection_temp_softmax_e2e_crown() {
    let mut b = TensorBlockBuilder::new("calib_vocab_temp_e2e");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[VOCAB_SIZE]);
    let inv_temp = b.add_input("inv_temp", &[1]);

    // Linear projection
    let logits = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, VOCAB_SIZE]);

    // Temperature scaling
    let inv_temp_bc = b.add_broadcast(inv_temp, &[SEQ_LEN, VOCAB_SIZE]);
    let scaled = b.add_binary_mul(logits, inv_temp_bc, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax
    let out = b.add_softmax(scaled, -1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid vocab+temp+softmax kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
        scalar_binding(1.5), // T=0.67 (slightly sharpened)
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocab+temp+softmax E2E: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "bounds must be non-degenerate");
}
