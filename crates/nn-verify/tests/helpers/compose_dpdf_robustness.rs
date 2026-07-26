// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Robustness certification under input perturbation for dpdf model subgraphs.
//!
//! Verifies that bounded input perturbations (epsilon-ball) produce bounded
//! output shifts across key dpdf sub-blocks. This is the core robustness
//! certification property: given L-inf perturbation radius epsilon on the
//! input, the output deviation is bounded and certifiable.
//!
//! ## Tests (8 compose tests):
//!
//! 1. **Detection head robustness**: Linear -> Sigmoid bbox confidence.
//!    Epsilon-ball input perturbation produces bounded sigmoid shift.
//!    Sigmoid is Lipschitz-1/4, so output width <= input width / 4.
//!
//! 2. **Classification head robustness**: Linear -> Softmax logit perturbation.
//!    L-inf input perturbation bounded through softmax output distribution.
//!
//! 3. **OCR encoder robustness**: Linear embedding + LayerNorm.
//!    Character embedding perturbation bounded through normalization.
//!
//! 4. **ViT patch embedding robustness**: Conv2d patch extraction.
//!    Small pixel perturbations propagate through patch embedding with
//!    bounded amplification.
//!
//! 5. **RoPE positional robustness**: cos/sin rotary encoding.
//!    Position perturbation effect on attention bounded via RoPE structure.
//!
//! 6. **Normalization robustness**: LayerNorm under perturbed input.
//!    LayerNorm re-centers and re-scales, bounding perturbation spread.
//!
//! 7. **MoE routing stability**: Linear -> Softmax expert gate.
//!    Small input changes produce bounded routing weight changes.
//!
//! 8. **Residual connection robustness**: Linear + skip connection.
//!    Perturbation growth through residual blocks is bounded.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, PATCH_SIZE=4, NUM_CLASSES=8
//!
//! Part of #4084: Compose tests for dpdf model robustness certification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
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
const NUM_EXPERTS: usize = 4;
const PATCH_SIZE: usize = 4;
const IMG_SIZE: usize = 16;
const IN_CHANNELS: usize = 3;
const WEIGHT_MAG: f32 = 0.02;
const NORM_EPS: f32 = 1e-5;

/// Small perturbation radius for robustness certification.
const EPSILON: f32 = 0.01;

// ---------------------------------------------------------------------------
// Helpers: create epsilon-ball bounds around zero center
// ---------------------------------------------------------------------------

/// Create BoundedTensor centered at 0 with perturbation radius epsilon.
fn epsilon_ball(shape: &[usize], eps: f32) -> BoundedTensor {
    uniform_bounds(shape, eps)
}

/// Create constant weight tensor filled with magnitude.
fn constant_weights(shape: &[usize], mag: f32) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), mag)
}

/// Create zero bias tensor.
fn zero_bias(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

/// Create unit variance tensor (for normalization running_var).
fn unit_var(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

// ===========================================================================
// 1. Detection head robustness: Linear -> Sigmoid bbox confidence
// ===========================================================================

/// Detection head: Linear(HIDDEN_DIM, 1) -> Sigmoid.
///
/// Models a single-channel bbox confidence output. Sigmoid is Lipschitz-1/4,
/// so output perturbation is bounded by input perturbation / 4.
fn build_detection_head_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_robustness_detection_head");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    // nn.Linear weight is [out_features, in_features].
    let w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("cls_bias", &[NUM_CLASSES]);

    // Linear projection: [SEQ_LEN, HIDDEN_DIM] x [NUM_CLASSES, HIDDEN_DIM]^T -> [SEQ_LEN, NUM_CLASSES]
    let logits = b.add_linear(input, w, Some(bias), &[SEQ_LEN, NUM_CLASSES]);

    // Sigmoid: output in [0, 1]
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid detection head robustness kernel")
}

fn detection_head_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(constant_weights(
            &[NUM_CLASSES, HIDDEN_DIM],
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(zero_bias(&[NUM_CLASSES])),
    ]
}

/// Detection head: graph builds and validates.
#[test]
fn test_detection_head_robustness_def_validates() {
    let def = build_detection_head_robustness_kernel();
    def.validate()
        .expect("detection head robustness kernel should validate");
}

/// Detection head: IBP bounds under epsilon-ball input.
#[test]
fn test_detection_head_robustness_ibp() {
    let def = build_detection_head_robustness_kernel();
    let bindings = detection_head_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Detection head robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");

    // Sigmoid outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid upper bound must be <= 1, got {hi_max}"
    );

    // Output width should be bounded (not vacuously wide)
    let width = hi_max - lo_min;
    assert!(
        width < 1.0,
        "detection head output width {width} should be < 1.0 for small epsilon"
    );
}

/// Detection head: CROWN tightens bounds.
#[test]
fn test_detection_head_robustness_crown() {
    let def = build_detection_head_robustness_kernel();
    let bindings = detection_head_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Detection head robustness CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Detection head: verify and record.
#[test]
fn test_detection_head_robustness_verify_and_record() {
    let def = build_detection_head_robustness_kernel();
    let bindings = detection_head_robustness_bindings();
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_detection_head");
    assert_eq!(result.num_variables, 1, "single Variable input");
}

// ===========================================================================
// 2. Classification head robustness: Linear -> Softmax logit perturbation
// ===========================================================================

/// Classification head: Linear(HIDDEN_DIM, NUM_CLASSES) -> Softmax.
///
/// Softmax output sums to 1 and each element is in [0, 1]. L-inf perturbation
/// on the input features causes bounded logit shift -> bounded softmax output.
fn build_classification_head_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_robustness_classification_head");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    // nn.Linear weight is [out_features, in_features].
    let w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("cls_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, w, Some(bias), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid classification head robustness kernel")
}

fn classification_head_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(constant_weights(
            &[NUM_CLASSES, HIDDEN_DIM],
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(zero_bias(&[NUM_CLASSES])),
    ]
}

/// Classification head: IBP bounds softmax output in [0, 1].
#[test]
fn test_classification_head_robustness_ibp() {
    let def = build_classification_head_robustness_kernel();
    let bindings = classification_head_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Classification head robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");

    // Softmax outputs must be in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

/// Classification head: CROWN propagation.
#[test]
fn test_classification_head_robustness_crown() {
    let def = build_classification_head_robustness_kernel();
    let bindings = classification_head_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Classification head robustness CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Classification head: verify and record.
#[test]
fn test_classification_head_robustness_verify_and_record() {
    let def = build_classification_head_robustness_kernel();
    let bindings = classification_head_robustness_bindings();
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "dpdf_robustness_classification_head",
    );
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// 3. OCR encoder robustness: Linear embedding + LayerNorm
// ===========================================================================

/// OCR encoder: Linear(HIDDEN_DIM, HIDDEN_DIM) -> LayerNorm.
///
/// Character embedding perturbation bounded through normalization. LayerNorm
/// re-centers and re-scales, providing natural robustness properties.
fn build_ocr_encoder_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_robustness_ocr_encoder");

    let input = b.add_input("char_features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);

    // Linear projection
    let proj = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm
    let out = b.add_layer_norm(proj, ln_eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid OCR encoder robustness kernel")
}

fn ocr_encoder_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(constant_weights(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(zero_bias(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(unit_var(&[HIDDEN_DIM])), // ln_weight = 1
        TensorParamBinding::ConstantTensor(zero_bias(&[HIDDEN_DIM])), // ln_bias = 0
        TensorParamBinding::ConstantScalar(NORM_EPS),
    ]
}

/// OCR encoder: IBP bounds through Linear + LayerNorm.
#[test]
fn test_ocr_encoder_robustness_ibp() {
    let def = build_ocr_encoder_robustness_kernel();
    let bindings = ocr_encoder_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR encoder robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// OCR encoder: CROWN propagation.
#[test]
fn test_ocr_encoder_robustness_crown() {
    let def = build_ocr_encoder_robustness_kernel();
    let bindings = ocr_encoder_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR encoder robustness CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// OCR encoder: verify and record.
#[test]
fn test_ocr_encoder_robustness_verify_and_record() {
    let def = build_ocr_encoder_robustness_kernel();
    let bindings = ocr_encoder_robustness_bindings();
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_ocr_encoder");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// 4. ViT patch embedding robustness: Conv2d patch extraction
// ===========================================================================

/// ViT patch embedding: Conv2d(3, HIDDEN_DIM, PATCH_SIZE, stride=PATCH_SIZE).
///
/// Extracts non-overlapping patches from the image and projects to HIDDEN_DIM.
/// Small pixel perturbations propagate through the linear Conv2d with bounded
/// amplification proportional to weight magnitude * kernel area.
fn build_vit_patch_embed_robustness_kernel() -> TensorKernelDef {
    let num_patches = IMG_SIZE / PATCH_SIZE; // 4
    let out_shape = [HIDDEN_DIM, num_patches, num_patches];
    let mut b = TensorBlockBuilder::new("dpdf_robustness_vit_patch_embed");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input(
        "patch_conv_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("patch_conv_bias", &[HIDDEN_DIM]);

    // Conv2d with stride=PATCH_SIZE: non-overlapping patch extraction
    let out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &out_shape,
    );

    b.build(out)
        .expect("valid ViT patch embedding robustness kernel")
}

fn vit_patch_embed_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image [3, 16, 16]
        TensorParamBinding::ConstantTensor(constant_weights(
            &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(zero_bias(&[HIDDEN_DIM])),
    ]
}

/// ViT patch embedding: graph validates.
#[test]
fn test_vit_patch_embed_robustness_def_validates() {
    let def = build_vit_patch_embed_robustness_kernel();
    def.validate()
        .expect("ViT patch embedding robustness kernel should validate");
}

/// ViT patch embedding: IBP bounds under pixel perturbation.
#[test]
fn test_vit_patch_embed_robustness_ibp() {
    let def = build_vit_patch_embed_robustness_kernel();
    let bindings = vit_patch_embed_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Image pixels perturbed within epsilon-ball around 0.5 (mid-range)
    let input_shape = [IN_CHANNELS, IMG_SIZE, IMG_SIZE];
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), 0.5 - EPSILON),
        ArrayD::from_elem(IxDyn(&input_shape), 0.5 + EPSILON),
    )
    .expect("valid pixel perturbation bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let num_patches = IMG_SIZE / PATCH_SIZE;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[HIDDEN_DIM, num_patches, num_patches],
        "patch embedding output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT patch embed robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

/// ViT patch embedding: verify and record.
#[test]
fn test_vit_patch_embed_robustness_verify_and_record() {
    let def = build_vit_patch_embed_robustness_kernel();
    let bindings = vit_patch_embed_robustness_bindings();
    let input_shape = [IN_CHANNELS, IMG_SIZE, IMG_SIZE];
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), 0.5 - EPSILON),
        ArrayD::from_elem(IxDyn(&input_shape), 0.5 + EPSILON),
    )
    .expect("valid pixel perturbation bounds");

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_vit_patch_embed");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// 5. RoPE positional robustness: cos/sin rotary encoding
// ===========================================================================

/// RoPE positional robustness: input features multiplied by cos/sin rotation.
///
/// RoPE applies element-wise rotation: x_rot = x * cos(theta) + rotate(x) * sin(theta).
/// We model this as: binary_mul(input, cos_table) + binary_mul(input_rotated, sin_table).
/// Since cos/sin are bounded in [-1, 1], the output perturbation is bounded by
/// 2 * input perturbation.
fn build_rope_robustness_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_robustness_rope");

    let input = b.add_input("query", &shape);
    let cos_table = b.add_input("cos_cache", &shape);
    let sin_table = b.add_input("sin_cache", &shape);
    // Rotated input (second half negated, first half shifted) — modeled as
    // a separate variable for graph simplicity. In practice this is a
    // deterministic function of `input`, but for bound propagation with
    // same epsilon it is sound to treat as correlated variable.
    let input_rotated = b.add_input("query_rotated", &shape);

    // x * cos(theta)
    let x_cos = b.add_binary_mul(input, cos_table, &shape);
    // rotate(x) * sin(theta)
    let x_sin = b.add_binary_mul(input_rotated, sin_table, &shape);
    // x_rot = x_cos + x_sin
    let out = b.add_binary_add(x_cos, x_sin, &shape);

    b.build(out).expect("valid RoPE robustness kernel")
}

fn rope_robustness_bindings() -> Vec<TensorParamBinding> {
    // cos/sin tables bounded in [-1, 1], modeled as constants for this test.
    // Using cos(0)=1, sin(0)=0 as representative values.
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let n = SEQ_LEN * HIDDEN_DIM;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HIDDEN_DIM {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * d as f64 / HIDDEN_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    vec![
        TensorParamBinding::Variable, // query
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&shape), cos_data).expect("cos table"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&shape), sin_data).expect("sin table"),
        ),
        TensorParamBinding::Variable, // query_rotated (same epsilon ball)
    ]
}

/// RoPE: IBP bounds under query perturbation.
#[test]
fn test_rope_robustness_ibp() {
    let def = build_rope_robustness_kernel();
    let bindings = rope_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Both query and query_rotated get the same epsilon-ball input
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let n: usize = shape.iter().product();
    let lower = ArrayD::from_elem(IxDyn(&[n * 2]), -EPSILON);
    let upper = ArrayD::from_elem(IxDyn(&[n * 2]), EPSILON);
    let input = BoundedTensor::new(lower, upper).expect("valid RoPE input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );

    // Output width should be bounded (cos/sin scale by at most 1 each)
    let width = hi_max - lo_min;
    assert!(
        width < 1.0,
        "RoPE output width {width} should be < 1.0 for small epsilon"
    );
}

/// RoPE: verify and record.
#[test]
fn test_rope_robustness_verify_and_record() {
    let def = build_rope_robustness_kernel();
    let bindings = rope_robustness_bindings();
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let n: usize = shape.iter().product();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[n * 2]), -EPSILON),
        ArrayD::from_elem(IxDyn(&[n * 2]), EPSILON),
    )
    .expect("valid RoPE input bounds");

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_rope");
    assert_eq!(
        result.num_variables, 2,
        "two Variable inputs (query + query_rotated)"
    );
}

// ===========================================================================
// 6. Normalization robustness: LayerNorm under perturbed input
// ===========================================================================

/// LayerNorm robustness: LayerNorm(Linear(x)).
///
/// LayerNorm re-centers to zero mean and re-scales to unit variance,
/// providing natural robustness properties. The key certification property
/// is that the output bounds remain finite and non-vacuous despite
/// normalization's nonlinear mean/variance computation.
fn build_norm_robustness_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_robustness_normalization");

    let input = b.add_input("features", &shape);
    let w = b.add_input("linear_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("linear_bias", &[HIDDEN_DIM]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);

    // Linear: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, HIDDEN_DIM]
    let linear_out = b.add_linear(input, w, Some(bias), &shape);

    // LayerNorm: normalizes along last dimension
    let out = b.add_layer_norm(linear_out, ln_eps, 1, ln_w, ln_b, &shape);

    b.build(out).expect("valid normalization robustness kernel")
}

fn norm_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(constant_weights(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(zero_bias(&[HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(unit_var(&[HIDDEN_DIM])), // ln_weight
        TensorParamBinding::ConstantTensor(zero_bias(&[HIDDEN_DIM])), // ln_bias
        TensorParamBinding::ConstantScalar(NORM_EPS),
    ]
}

/// Normalization: IBP bounds through Linear + LayerNorm.
#[test]
fn test_norm_robustness_ibp() {
    let def = build_norm_robustness_kernel();
    let bindings = norm_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Normalization robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );
}

/// Normalization: CROWN propagation.
#[test]
fn test_norm_robustness_crown() {
    let def = build_norm_robustness_kernel();
    let bindings = norm_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Normalization robustness CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Normalization: verify and record.
#[test]
fn test_norm_robustness_verify_and_record() {
    let def = build_norm_robustness_kernel();
    let bindings = norm_robustness_bindings();
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_normalization");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// 7. MoE routing stability: Linear -> Softmax expert gate
// ===========================================================================

/// MoE routing: Linear(HIDDEN_DIM, NUM_EXPERTS) -> Softmax.
///
/// Expert routing weights are softmax outputs summing to 1. Small input
/// perturbations cause bounded routing weight changes, preventing
/// catastrophic expert switching.
fn build_moe_routing_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_robustness_moe_routing");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    // nn.Linear weight is [out_features, in_features] = [NUM_EXPERTS, HIDDEN_DIM].
    let gate_w = b.add_input("gate_weight", &[NUM_EXPERTS, HIDDEN_DIM]);
    let gate_b = b.add_input("gate_bias", &[NUM_EXPERTS]);

    // Gate linear: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, NUM_EXPERTS]
    let logits = b.add_linear(input, gate_w, Some(gate_b), &[SEQ_LEN, NUM_EXPERTS]);

    // Softmax: routing weights in [0, 1] summing to 1
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, NUM_EXPERTS]);

    b.build(out).expect("valid MoE routing robustness kernel")
}

fn moe_routing_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(constant_weights(
            &[NUM_EXPERTS, HIDDEN_DIM],
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(zero_bias(&[NUM_EXPERTS])),
    ]
}

/// MoE routing: IBP bounds softmax routing weights.
#[test]
fn test_moe_routing_robustness_ibp() {
    let def = build_moe_routing_robustness_kernel();
    let bindings = moe_routing_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE routing robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");

    // Softmax outputs in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1, got {hi_max}");
}

/// MoE routing: CROWN propagation.
#[test]
fn test_moe_routing_robustness_crown() {
    let def = build_moe_routing_robustness_kernel();
    let bindings = moe_routing_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE routing robustness CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// MoE routing: verify and record.
#[test]
fn test_moe_routing_robustness_verify_and_record() {
    let def = build_moe_routing_robustness_kernel();
    let bindings = moe_routing_robustness_bindings();
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_moe_routing");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// 8. Residual connection robustness: Linear + skip connection
// ===========================================================================

/// Residual connection: x + Linear(x).
///
/// The skip connection preserves the input signal and adds a bounded
/// perturbation from the linear transform. Perturbation growth through
/// residual blocks is bounded by (1 + Lipschitz(f)) per block.
fn build_residual_robustness_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_robustness_residual");

    let input = b.add_input("features", &shape);
    let w = b.add_input("ffn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("ffn_bias", &[HIDDEN_DIM]);

    // FFN branch: Linear
    let ffn_out = b.add_linear(input, w, Some(bias), &shape);

    // Residual: input + ffn_out
    let out = b.add_binary_add(input, ffn_out, &shape);

    b.build(out).expect("valid residual robustness kernel")
}

fn residual_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(constant_weights(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(zero_bias(&[HIDDEN_DIM])),
    ]
}

/// Residual: IBP bounds through skip + linear.
#[test]
fn test_residual_robustness_ibp() {
    let def = build_residual_robustness_kernel();
    let bindings = residual_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Residual robustness IBP (eps={EPSILON}): bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "bounds must be finite"
    );

    // Residual output width should be bounded: skip + linear with small weights
    let width = hi_max - lo_min;
    assert!(
        width < 2.0,
        "residual output width {width} should be < 2.0 for small epsilon and weights"
    );
}

/// Residual: CROWN propagation.
#[test]
fn test_residual_robustness_crown() {
    let def = build_residual_robustness_kernel();
    let bindings = residual_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let (method, output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Residual robustness CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Residual: verify and record.
#[test]
fn test_residual_robustness_verify_and_record() {
    let def = build_residual_robustness_kernel();
    let bindings = residual_robustness_bindings();
    let input = epsilon_ball(&[SEQ_LEN, HIDDEN_DIM], EPSILON);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_robustness_residual");
    assert_eq!(result.num_variables, 1);
}
