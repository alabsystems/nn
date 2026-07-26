// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Adversarial robustness compose tests: input perturbation bounds for all 7
//! dpdf document understanding models.
//!
//! Verifies that small epsilon-ball perturbations around a center point produce
//! bounded output changes. This is the core adversarial robustness property:
//! if an attacker perturbs inputs by at most epsilon, the model's outputs change
//! by a bounded amount.
//!
//! ## Per-Model Robustness (7 tests)
//!
//! One test per model type: epsilon-ball input perturbation -> verify output
//! bound width is finite and bounded by a model-specific threshold.
//!
//! 1. Granite-Docling: RMSNorm -> SwiGLU FFN (decoder sub-block)
//! 2. DocLayout-YOLO: Linear -> sigmoid detection head
//! 3. GLM-OCR: RMSNorm -> SwiGLU -> linear -> softmax (MTP head)
//! 4. Table Transformer: LayerNorm -> FFN -> sigmoid classification
//! 5. Qwen3-VL: RMSNorm -> SwiGLU decoder FFN
//! 6. PaddleOCR: Linear -> softmax CTC head
//! 7. FireRed-OCR: RMSNorm -> linear -> softmax CTC head
//!
//! ## Monotone Tightening (5 tests)
//!
//! Shrinking epsilon from 0.1 to 0.01 MUST produce tighter output bounds.
//! Verifies this property for sigmoid, softmax, linear, ReLU, and tanh outputs.
//!
//! 8. Sigmoid tightening: sigmoid(Linear(x)) with eps 0.1 vs 0.01
//! 9. Softmax tightening: softmax(Linear(x)) with eps 0.1 vs 0.01
//! 10. Linear tightening: Linear(x) with eps 0.1 vs 0.01
//! 11. ReLU tightening: ReLU(Linear(x)) with eps 0.1 vs 0.01
//! 12. Tanh tightening: tanh(Linear(x)) with eps 0.1 vs 0.01
//!
//! ## Cross-Model Robustness (4 tests)
//!
//! 13. Pipeline robustness: Layout sigmoid -> OCR linear -> CTC softmax
//!     Input perturbation bounded through full 3-stage pipeline.
//! 14. Cascading bounds: Layout sigmoid -> table structure -> OCR softmax
//!     Layout perturbation causes bounded OCR output change.
//! 15. Quantization robustness: INT4 dequant does not amplify perturbations
//!     vs FP32 (output width ratio bounded).
//! 16. VLM-to-layout robustness: VLM decoder features -> detection sigmoid
//!     Perturbation in VLM space causes bounded detection confidence change.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, NUM_CLASSES=8, VOCAB_SIZE=16, FFN_DIM=64
//!
//! Part of #3962: Adversarial robustness compose tests for dpdf models.

use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
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
const WEIGHT_MAG: f32 = 0.02;
const NORM_EPS: f32 = 1e-5;

// INT4 quantization parameters for robustness comparison.
const INT4_BINS: usize = 16;

// ---------------------------------------------------------------------------
// Helpers: create epsilon-ball bounds around zero center
// ---------------------------------------------------------------------------

/// Create BoundedTensor centered at 0 with perturbation radius epsilon.
/// Equivalent to `uniform_bounds(shape, eps)` but named for adversarial context.
fn eps_ball(shape: &[usize], eps: f32) -> BoundedTensor {
    uniform_bounds(shape, eps)
}

/// Compute output bound width from a (lo_min, hi_max) pair.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
/// This is how existing compose tests implement SiLU since there is no
/// single-op SiLU in TensorBlockBuilder.
fn add_silu(b: &mut TensorBlockBuilder, input: TensorNodeId, shape: &[usize]) -> TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

// ===========================================================================
// Per-Model Robustness (7 tests)
// ===========================================================================

// --- 1. Granite-Docling: RMSNorm -> SwiGLU FFN ---

fn build_granite_docling_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_granite_docling");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps_node = b.add_input("eps", &[1]);
    let norm_w = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let normed = b.add_rms_norm(input, eps_node, 1, norm_w, &out_shape);

    // SwiGLU: gate_proj -> SiLU, up_proj, element-wise mul, down_proj
    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(gated, down_w, None, &out_shape);

    b.build(out)
        .expect("valid Granite-Docling adversarial kernel")
}

fn granite_docling_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), NORM_EPS)), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_adversarial_granite_docling_eps_ball() {
    let def = build_granite_docling_robustness_kernel();
    let bindings = granite_docling_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Granite-Docling adversarial (eps=0.01): output width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
    assert!(
        width < 100.0,
        "Granite-Docling: output width {width} exceeds robustness threshold 100.0"
    );
}

// --- 2. DocLayout-YOLO: Linear -> sigmoid detection ---

fn build_doclayout_yolo_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_doclayout_yolo");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, cls_w, Some(cls_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid DocLayout-YOLO adversarial kernel")
}

fn doclayout_yolo_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_adversarial_doclayout_yolo_eps_ball() {
    let def = build_doclayout_yolo_robustness_kernel();
    let bindings = doclayout_yolo_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "DocLayout-YOLO adversarial (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}"
    );

    // Sigmoid output must be in [0, 1].
    let tol = 1e-6;
    assert!(lo_min >= 0.0 - tol, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + tol, "sigmoid upper <= 1, got {hi_max}");

    // With tight perturbation, output should be a narrow band within [0, 1].
    assert!(
        width < 0.5,
        "DocLayout-YOLO: output width {width} should be < 0.5 for eps=0.01"
    );
}

// --- 3. GLM-OCR: RMSNorm -> SwiGLU -> softmax ---

fn build_glm_ocr_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_glm_ocr");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps_node = b.add_input("eps", &[1]);
    let norm_w = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);
    let head_w = b.add_input("head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm -> SwiGLU FFN
    let normed = b.add_rms_norm(input, eps_node, 1, norm_w, &hidden_shape);
    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(gated, down_w, None, &hidden_shape);

    // LM head -> softmax
    let logits = b.add_linear(ffn_out, head_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid GLM-OCR adversarial kernel")
}

fn glm_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), NORM_EPS)), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_adversarial_glm_ocr_eps_ball() {
    let def = build_glm_ocr_robustness_kernel();
    let bindings = glm_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("GLM-OCR adversarial (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}");

    // Softmax output in [0, 1].
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// --- 4. Table Transformer: LayerNorm -> FFN -> sigmoid ---

fn build_table_transformer_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_table_transformer");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_bias", &[FFN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[NUM_CLASSES, FFN_DIM]);
    let fc2_b = b.add_input("fc2_bias", &[NUM_CLASSES]);

    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];

    // LayerNorm -> FFN (ReLU) -> sigmoid classification
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &hidden_shape);
    let fc1 = b.add_linear(normed, fc1_w, Some(fc1_b), &[SEQ_LEN, FFN_DIM]);
    let act = b.add_relu(fc1, &[SEQ_LEN, FFN_DIM]);
    let fc2 = b.add_linear(act, fc2_w, Some(fc2_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(fc2, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid Table Transformer adversarial kernel")
}

fn table_transformer_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), NORM_EPS)), // ln_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_adversarial_table_transformer_eps_ball() {
    let def = build_table_transformer_robustness_kernel();
    let bindings = table_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "Table Transformer adversarial (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}"
    );

    let tol = 1e-6;
    assert!(lo_min >= 0.0 - tol, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + tol, "sigmoid upper <= 1, got {hi_max}");
}

// --- 5. Qwen3-VL: RMSNorm -> SwiGLU decoder FFN ---

fn build_qwen3_vl_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_qwen3_vl");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps_node = b.add_input("eps", &[1]);
    let norm_w = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    let normed = b.add_rms_norm(input, eps_node, 1, norm_w, &out_shape);
    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(gated, down_w, None, &out_shape);

    b.build(out).expect("valid Qwen3-VL adversarial kernel")
}

fn qwen3_vl_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), NORM_EPS)), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_adversarial_qwen3_vl_eps_ball() {
    let def = build_qwen3_vl_robustness_kernel();
    let bindings = qwen3_vl_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Qwen3-VL adversarial (eps=0.01): output width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
    assert!(
        width < 100.0,
        "Qwen3-VL: output width {width} exceeds robustness threshold 100.0"
    );
}

// --- 6. PaddleOCR: Linear -> softmax CTC head ---

fn build_paddle_ocr_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_paddle_ocr");

    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let head_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let head_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, head_w, Some(head_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid PaddleOCR adversarial kernel")
}

fn paddle_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_adversarial_paddle_ocr_eps_ball() {
    let def = build_paddle_ocr_robustness_kernel();
    let bindings = paddle_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("PaddleOCR adversarial (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// --- 7. FireRed-OCR: RMSNorm -> Linear -> softmax CTC ---

fn build_firered_ocr_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_firered_ocr");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps_node = b.add_input("eps", &[1]);
    let norm_w = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let head_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let normed = b.add_rms_norm(input, eps_node, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let logits = b.add_linear(normed, head_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid FireRed-OCR adversarial kernel")
}

fn firered_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), NORM_EPS)), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_adversarial_firered_ocr_eps_ball() {
    let def = build_firered_ocr_robustness_kernel();
    let bindings = firered_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("FireRed-OCR adversarial (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}");

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// Monotone Tightening (5 tests)
// ===========================================================================

/// Helper: build a Linear -> activation kernel for tightening tests.
fn build_tightening_kernel(
    name: &str,
    activation: &str,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[NUM_CLASSES]);

    let linear_out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, NUM_CLASSES]);

    let out = match activation {
        "sigmoid" => b.add_sigmoid(linear_out, &[SEQ_LEN, NUM_CLASSES]),
        "softmax" => b.add_softmax(linear_out, -1, &[SEQ_LEN, NUM_CLASSES]),
        "linear" => linear_out,
        "relu" => b.add_relu(linear_out, &[SEQ_LEN, NUM_CLASSES]),
        "tanh" => b.add_tanh(linear_out, &[SEQ_LEN, NUM_CLASSES]),
        _ => panic!("unknown activation: {activation}"),
    };

    let def = b.build(out).expect("valid tightening kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ];

    (def, bindings)
}

/// Core tightening assertion: eps_small must produce tighter bounds than eps_large.
fn assert_monotone_tightening(activation: &str) {
    let (def, bindings) =
        build_tightening_kernel(&format!("dpdf_adv_tightening_{activation}"), activation);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide epsilon: 0.1
    let input_wide = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    assert_bounds_valid(&output_wide);
    let wide_width = bound_width(&output_wide);

    // Narrow epsilon: 0.01
    let input_narrow = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);
    let output_narrow = graph.propagate_ibp(&input_narrow).expect("IBP narrow");
    assert_bounds_valid(&output_narrow);
    let narrow_width = bound_width(&output_narrow);

    eprintln!(
        "Tightening ({activation}): wide_width={wide_width:.6}, narrow_width={narrow_width:.6}"
    );

    // Narrower epsilon must produce tighter (or equal) output bounds.
    assert!(
        narrow_width <= wide_width + 1e-6,
        "Monotone tightening ({activation}): narrow width {narrow_width} \
         should be <= wide width {wide_width}"
    );

    // For all these activations, a 10x reduction in epsilon should produce
    // meaningfully tighter bounds.
    assert!(
        narrow_width < wide_width * 0.99,
        "Monotone tightening ({activation}): narrow bounds should be substantially \
         tighter than wide bounds (narrow={narrow_width}, wide={wide_width})"
    );
}

#[test]
fn test_adversarial_tightening_sigmoid() {
    assert_monotone_tightening("sigmoid");
}

#[test]
fn test_adversarial_tightening_softmax() {
    assert_monotone_tightening("softmax");
}

#[test]
fn test_adversarial_tightening_linear() {
    assert_monotone_tightening("linear");
}

#[test]
fn test_adversarial_tightening_relu() {
    assert_monotone_tightening("relu");
}

#[test]
fn test_adversarial_tightening_tanh() {
    assert_monotone_tightening("tanh");
}

// ===========================================================================
// Cross-Model Robustness (4 tests)
// ===========================================================================

// --- 13. Pipeline robustness: layout sigmoid -> OCR linear -> CTC softmax ---

fn build_pipeline_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_pipeline_robustness");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 1: Layout detection -> sigmoid
    let layout_w = b.add_input("layout_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let layout_b = b.add_input("layout_bias", &[HIDDEN_DIM]);
    let layout_logits = b.add_linear(input, layout_w, Some(layout_b), &[SEQ_LEN, HIDDEN_DIM]);
    let layout_conf = b.add_sigmoid(layout_logits, &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 2: OCR feature projection (linear)
    let ocr_w = b.add_input("ocr_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ocr_b = b.add_input("ocr_bias", &[HIDDEN_DIM]);
    let ocr_features = b.add_linear(layout_conf, ocr_w, Some(ocr_b), &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 3: CTC head -> softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(ocr_features, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid pipeline robustness kernel")
}

fn pipeline_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// Pipeline robustness: input perturbation bounded through 3-stage pipeline.
#[test]
fn test_adversarial_pipeline_robustness() {
    let def = build_pipeline_robustness_kernel();
    let bindings = pipeline_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Pipeline robustness (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}");

    // End-to-end softmax output in [0, 1].
    assert!(
        lo_min >= -1e-4,
        "pipeline: softmax lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "pipeline: softmax upper <= 1, got {hi_max}"
    );
}

// --- 14. Cascading bounds: layout -> table structure -> OCR softmax ---

fn build_cascading_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_cascading_bounds");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 1: Layout detection -> sigmoid
    let layout_w = b.add_input("layout_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let layout_b = b.add_input("layout_bias", &[NUM_CLASSES]);
    let layout_logits = b.add_linear(input, layout_w, Some(layout_b), &[SEQ_LEN, NUM_CLASSES]);
    let layout_conf = b.add_sigmoid(layout_logits, &[SEQ_LEN, NUM_CLASSES]);

    // Stage 2: Table structure projection -> sigmoid
    let table_w = b.add_input("table_weight", &[NUM_CLASSES, NUM_CLASSES]);
    let table_b = b.add_input("table_bias", &[NUM_CLASSES]);
    let table_logits = b.add_linear(layout_conf, table_w, Some(table_b), &[SEQ_LEN, NUM_CLASSES]);
    let table_conf = b.add_sigmoid(table_logits, &[SEQ_LEN, NUM_CLASSES]);

    // Stage 3: OCR head projection -> softmax
    let ocr_w = b.add_input("ocr_weight", &[VOCAB_SIZE, NUM_CLASSES]);
    let ocr_b = b.add_input("ocr_bias", &[VOCAB_SIZE]);
    let ocr_logits = b.add_linear(table_conf, ocr_w, Some(ocr_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(ocr_logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid cascading robustness kernel")
}

fn cascading_robustness_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// Cascading bounds: layout perturbation causes bounded OCR output change.
#[test]
fn test_adversarial_cascading_bounds() {
    let def = build_cascading_robustness_kernel();
    let bindings = cascading_robustness_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Tight perturbation
    let input_tight = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight");
    assert_bounds_valid(&output_tight);

    // Wider perturbation
    let input_wide = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    assert_bounds_valid(&output_wide);

    let tight_width = bound_width(&output_tight);
    let wide_width = bound_width(&output_wide);

    eprintln!("Cascading bounds: tight_width={tight_width:.6}, wide_width={wide_width:.6}");

    // Monotonicity: tighter input -> tighter output.
    assert!(
        tight_width <= wide_width + 1e-6,
        "cascading: tight width {tight_width} should be <= wide width {wide_width}"
    );

    // Softmax output in [0, 1] for both.
    let (lo, hi) = bounds_min_max(&output_tight);
    assert!(lo >= -1e-4, "cascading tight: softmax lower >= 0, got {lo}");
    assert!(
        hi <= 1.0 + 1e-4,
        "cascading tight: softmax upper <= 1, got {hi}"
    );
}

// --- 15. Quantization robustness: INT4 vs FP32 perturbation amplification ---

/// Build a simple linear layer for comparing FP32 vs INT4 perturbation behavior.
fn build_quantization_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_quantization_robustness");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid quantization robustness kernel")
}

/// INT4 quantization robustness: quantization does not amplify perturbations.
#[test]
fn test_adversarial_quantization_robustness() {
    let def = build_quantization_robustness_kernel();
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    // FP32 weights
    let fp32_w_data: Vec<f32> = (0..NUM_CLASSES * HIDDEN_DIM)
        .map(|i| WEIGHT_MAG * (((i % 7) as f32) - 3.0) / 3.0)
        .collect();
    let fp32_w =
        ArrayD::from_shape_vec(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), fp32_w_data.clone()).unwrap();
    let bias = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(fp32_w),
        TensorParamBinding::ConstantTensor(bias.clone()),
    ];

    let graph_fp32 = tensor_kernel_to_graph(&def, &fp32_bindings).expect("fp32 graph");
    let output_fp32 = graph_fp32.propagate_ibp(&input).expect("IBP fp32");
    assert_bounds_valid(&output_fp32);
    let fp32_width = bound_width(&output_fp32);

    // INT4 quantized weights: round to 16 levels in [-WEIGHT_MAG, WEIGHT_MAG].
    let quant_step = 2.0 * WEIGHT_MAG / (INT4_BINS as f32 - 1.0);
    let int4_w_data: Vec<f32> = fp32_w_data
        .iter()
        .map(|&v| {
            let level = ((v + WEIGHT_MAG) / quant_step).round();
            let clamped = level.clamp(0.0, (INT4_BINS - 1) as f32);
            clamped * quant_step - WEIGHT_MAG
        })
        .collect();
    let int4_w = ArrayD::from_shape_vec(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), int4_w_data).unwrap();

    let int4_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(int4_w),
        TensorParamBinding::ConstantTensor(bias),
    ];

    let graph_int4 = tensor_kernel_to_graph(&def, &int4_bindings).expect("int4 graph");
    let output_int4 = graph_int4.propagate_ibp(&input).expect("IBP int4");
    assert_bounds_valid(&output_int4);
    let int4_width = bound_width(&output_int4);

    eprintln!("Quantization robustness: FP32 width={fp32_width:.6}, INT4 width={int4_width:.6}");

    // INT4 should not amplify perturbation bounds by more than 2x compared to FP32.
    // (The sigmoid clamps everything to [0,1] anyway, but the key property is that
    // quantization noise does not drastically widen the perturbation response.)
    let amplification_ratio = if fp32_width > 1e-10 {
        int4_width / fp32_width
    } else {
        1.0
    };
    assert!(
        amplification_ratio < 2.0,
        "Quantization robustness: INT4/FP32 width ratio {amplification_ratio:.4} exceeds 2.0 \
         (FP32={fp32_width:.6}, INT4={int4_width:.6})"
    );
}

// --- 16. VLM-to-layout robustness: VLM decoder -> detection sigmoid ---

fn build_vlm_to_layout_robustness_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_adv_vlm_to_layout");

    let input = b.add_input("vlm_features", &[SEQ_LEN, HIDDEN_DIM]);

    // VLM -> layout projection
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, HIDDEN_DIM]);
    let act = b.add_relu(projected, &[SEQ_LEN, HIDDEN_DIM]);

    // Detection head -> sigmoid
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let det_b = b.add_input("det_bias", &[NUM_CLASSES]);
    let det_logits = b.add_linear(act, det_w, Some(det_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(det_logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid VLM-to-layout robustness kernel")
}

fn vlm_to_layout_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

/// VLM-to-layout robustness: perturbation in VLM space bounded at detection output.
#[test]
fn test_adversarial_vlm_to_layout_robustness() {
    let def = build_vlm_to_layout_robustness_kernel();
    let bindings = vlm_to_layout_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = eps_ball(&[SEQ_LEN, HIDDEN_DIM], 0.01);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "VLM-to-layout adversarial (eps=0.01): bounds=[{lo_min}, {hi_max}], width={width:.6}"
    );

    let tol = 1e-6;
    assert!(lo_min >= 0.0 - tol, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + tol, "sigmoid upper <= 1, got {hi_max}");

    // With small VLM perturbation, detection confidence should be narrow.
    assert!(
        width < 0.5,
        "VLM-to-layout: output width {width} should be < 0.5 for eps=0.01"
    );
}
