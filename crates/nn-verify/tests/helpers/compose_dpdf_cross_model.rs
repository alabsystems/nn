// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-model composition verification tests for dpdf document understanding.
//!
//! Verifies bounds propagation across model boundaries: output bounds from one
//! model feed correctly as input bounds to the next. This is the compositional
//! safety property: dpdf chains Layout -> OCR -> Table -> VLM, and bounds must
//! compose across the full pipeline.
//!
//! ## Layout -> OCR (7 tests)
//!
//! 1. **DocLayout-YOLO sigmoid -> PaddleOCR detection backbone IBP**:
//!    Layout confidence [0,1] feeds OCR text detector feature extraction.
//!
//! 2. **DocLayout-YOLO box regression -> crop -> PaddleOCR recognition IBP**:
//!    Box coordinates [0,1] define crop region; recognition runs on crop.
//!
//! 3. **Table Transformer -> table structure -> PaddleOCR cell OCR IBP**:
//!    Table cell sigmoid [0,1] -> cell features -> CTC character recognition.
//!
//! 4. **DocLayout-YOLO -> FireRed-OCR recognition IBP**:
//!    Layout features -> linear projection -> FireRed encoder -> CTC head.
//!
//! 5. **Confidence gating: low-confidence layout filtered before OCR IBP**:
//!    Tighter sigmoid bounds ([0.5, 1.0]) produce tighter downstream OCR bounds.
//!
//! 6. **DocLayout-YOLO sigmoid -> PaddleOCR detection backbone CROWN**:
//!    Same as test 1 with CROWN linearization for tighter cross-model bounds.
//!
//! 7. **DocLayout-YOLO DFL regression -> PaddleOCR recognition CROWN**:
//!    DFL box decode -> crop modeling -> recognition with CROWN bounds.
//!
//! ## VLM -> Layout (4 tests)
//!
//! 8. **Qwen3-VL region proposal -> DocLayout-YOLO refinement IBP**:
//!    VLM decoder features -> linear projection -> YOLO detection head.
//!
//! 9. **Granite-Docling page understanding -> table detection IBP**:
//!    Vision projection features -> linear -> sigmoid table confidence.
//!
//! 10. **GLM-OCR token prediction -> layout region association IBP**:
//!     GLM decoder logits -> softmax -> linear -> sigmoid region score.
//!
//! 11. **Qwen3-VL region proposal -> DocLayout-YOLO refinement CROWN**:
//!     Same as test 8 with CROWN linearization through the projection.
//!
//! ## Full Pipeline (5 tests)
//!
//! 12. **3-stage: layout detection -> table structure -> cell OCR IBP**:
//!     DocLayout sigmoid -> Table Transformer sigmoid -> CTC softmax.
//!
//! 13. **4-stage: page classification -> layout -> table -> OCR IBP**:
//!     Softmax page class -> sigmoid layout -> sigmoid table -> CTC softmax.
//!
//! 14. **End-to-end document: image -> regions -> text -> structured output IBP**:
//!     Image [0,1] -> Conv backbone -> sigmoid regions -> linear projection
//!     -> GELU -> CTC head -> softmax character probabilities.
//!
//! 15. **3-stage pipeline CROWN**: Same as test 12 with CROWN linearization
//!     for tighter end-to-end cross-model bounds.
//!
//! 16. **End-to-end document CROWN**: Same as test 14 with CROWN bounds
//!     through the full image-to-text pipeline.
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - PaddleOCR (Baidu): DB detector + SVTR recognizer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure
//! - FireRed-OCR: Qwen3-VL-2B variant fine-tuned for document OCR
//! - Qwen3-VL (Alibaba): Vision-language model
//! - Granite-Docling: SigLIP2 vision encoder + Granite LLM decoder
//! - GLM-4V (THUDM): Vision-language model with GLM-4 decoder
//!
//! Dimensions (small for fast verification, structurally representative):
//! - FEATURE_DIM=32, SEQ_LEN=4, NUM_CLASSES=8, NUM_ANCHORS=6,
//!   VOCAB_SIZE=16, FFN_DIM=64, IMG_SIZE=32, IN_CHANNELS=3
//!
//! Part of #3956: cross-model compose tests for dpdf pipeline verification.

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

/// Feature dimension at model boundaries.
const FEATURE_DIM: usize = 32;
/// Sequence length (number of detection anchors or text positions).
const SEQ_LEN: usize = 4;
/// Number of layout detection classes.
const NUM_CLASSES: usize = 8;
/// Number of detection anchors.
const NUM_ANCHORS: usize = 6;
/// OCR vocabulary size for CTC head.
const VOCAB_SIZE: usize = 16;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Image spatial size.
const IMG_SIZE: usize = 32;
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Backbone output channels.
const BACKBONE_CH: usize = 16;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of table structure classes (row/column/cell).
const TABLE_CLASSES: usize = 4;

// ===========================================================================
// Helpers: image bounds and common bindings
// ===========================================================================

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sigmoid-domain bounds: output of sigmoid in [0, 1].
fn sigmoid_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid sigmoid bounds [0, 1]")
}

/// High-confidence sigmoid bounds: [0.5, 1.0] (post confidence gating).
fn high_confidence_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.5f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid high-confidence bounds [0.5, 1.0]")
}

// ===========================================================================
// 1. Layout -> OCR: DocLayout-YOLO sigmoid -> PaddleOCR detection backbone
// ===========================================================================

/// Build cross-model: layout sigmoid [0,1] -> conv backbone -> ReLU features.
///
/// Models the boundary where DocLayout-YOLO detection confidence feeds into
/// PaddleOCR's DB text detector. The sigmoid output [0,1] from layout
/// detection is projected via 1x1 conv to backbone channel dimension,
/// then batch-normalized and ReLU-activated.
///
/// Input: `[NUM_CLASSES, SEQ_LEN]` (Variable, layout sigmoid outputs in [0, 1]).
/// Output: `[BACKBONE_CH, SEQ_LEN]` (DB backbone features, ReLU >= 0).
fn build_layout_sigmoid_to_ocr_backbone_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_layout_sigmoid_to_ocr_backbone");

    // Layout sigmoid output (modeled as variable with [0,1] bounds).
    // Batch-major [SEQ_LEN, NUM_CLASSES] so nn.Linear contracts the channel
    // dim (NUM_CLASSES) against weight [out, in] = [BACKBONE_CH, NUM_CLASSES].
    let input = b.add_input("layout_sigmoid", &[SEQ_LEN, NUM_CLASSES]);

    // 1x1 conv projection: [SEQ_LEN, NUM_CLASSES] -> [SEQ_LEN, BACKBONE_CH]
    // Modeled as Linear since spatial dim is 1D
    let proj_w = b.add_input("proj_weight", &[BACKBONE_CH, NUM_CLASSES]);
    let proj_b = b.add_input("proj_bias", &[BACKBONE_CH]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, BACKBONE_CH]);

    // ReLU activation (DB backbone uses ReLU)
    let out = b.add_relu(projected, &[SEQ_LEN, BACKBONE_CH]);

    b.build(out)
        .expect("valid layout sigmoid to OCR backbone kernel")
}

fn layout_sigmoid_to_ocr_backbone_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // layout_sigmoid
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BACKBONE_CH, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32)),
    ]
}

#[test]
fn test_cross_layout_sigmoid_to_ocr_backbone_ibp() {
    let def = build_layout_sigmoid_to_ocr_backbone_kernel();
    let bindings = layout_sigmoid_to_ocr_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = sigmoid_bounds(&[SEQ_LEN, NUM_CLASSES]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layout -> OCR backbone");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, BACKBONE_CH]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Layout sigmoid -> OCR backbone IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU clamps lower to >= 0
    assert!(
        lo_min >= -1e-4,
        "ReLU output should have lower >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 2. Layout -> OCR: box regression -> crop -> PaddleOCR recognition
// ===========================================================================

/// Build cross-model: layout box [0,1] -> projection -> GELU -> CTC head.
///
/// Models the boundary where DocLayout-YOLO box regression outputs (sigmoid
/// [0,1] normalized coordinates) define a crop region, and PaddleOCR
/// recognition runs on the crop. For verification, we model the crop as a
/// linear projection from box coordinates to recognition features.
///
/// Input: `[NUM_ANCHORS, 4]` (Variable, box coordinates in [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC character logits).
fn build_layout_box_to_ocr_recognition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_layout_box_to_ocr_recognition");

    let input = b.add_input("box_coords", &[NUM_ANCHORS, 4]);

    // Crop feature extraction: project box coords to feature space
    let feat_w = b.add_input("feat_weight", &[FEATURE_DIM, 4]);
    let features = b.add_linear(input, feat_w, None, &[NUM_ANCHORS, FEATURE_DIM]);

    // Reshape to sequence: [NUM_ANCHORS, FEATURE_DIM] -> [SEQ_LEN, FFN_DIM]
    // (model spatial pooling as reshape + projection)
    let pool_w = b.add_input("pool_weight", &[FFN_DIM, FEATURE_DIM]);
    let seq_features = b.add_linear(features, pool_w, None, &[NUM_ANCHORS, FFN_DIM]);

    // Narrow to SEQ_LEN positions (select first SEQ_LEN anchors)
    let narrowed = b.add_narrow(seq_features, 0, 0, SEQ_LEN, &[SEQ_LEN, FFN_DIM]);

    // GELU activation
    let activated = b.add_gelu(narrowed, &[SEQ_LEN, FFN_DIM]);

    // CTC projection: [SEQ_LEN, FFN_DIM] -> [SEQ_LEN, VOCAB_SIZE]
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let out = b.add_linear(activated, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid layout box to OCR recognition kernel")
}

fn layout_box_to_ocr_recognition_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // box_coords
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FEATURE_DIM, 4]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_layout_box_to_ocr_recognition_ibp() {
    let def = build_layout_box_to_ocr_recognition_kernel();
    let bindings = layout_box_to_ocr_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Box coordinates in [0, 1] from sigmoid
    let input = sigmoid_bounds(&[NUM_ANCHORS, 4]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layout box -> OCR recognition");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Layout box -> OCR recognition IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Table Transformer -> table structure -> PaddleOCR cell OCR
// ===========================================================================

/// Build cross-model: table structure sigmoid -> cell projection -> CTC softmax.
///
/// Models the Table Transformer output (cell classification sigmoid [0,1])
/// feeding into PaddleOCR for per-cell text recognition.
///
/// Input: `[NUM_ANCHORS, TABLE_CLASSES]` (Variable, table cell sigmoid [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC softmax character probabilities [0, 1]).
fn build_table_structure_to_cell_ocr_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_table_structure_to_cell_ocr");

    let input = b.add_input("table_cells", &[NUM_ANCHORS, TABLE_CLASSES]);

    // Project table cell features to OCR feature space
    let proj_w = b.add_input("proj_weight", &[FEATURE_DIM, TABLE_CLASSES]);
    let projected = b.add_linear(input, proj_w, None, &[NUM_ANCHORS, FEATURE_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(projected, 0, 0, SEQ_LEN, &[SEQ_LEN, FEATURE_DIM]);

    // CTC head: Linear -> Softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid table structure to cell OCR kernel")
}

fn table_structure_to_cell_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // table_cells
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FEATURE_DIM, TABLE_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_table_structure_to_cell_ocr_ibp() {
    let def = build_table_structure_to_cell_ocr_kernel();
    let bindings = table_structure_to_cell_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = sigmoid_bounds(&[NUM_ANCHORS, TABLE_CLASSES]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through table structure -> cell OCR");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table structure -> cell OCR IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. DocLayout-YOLO -> FireRed-OCR recognition
// ===========================================================================

/// Build cross-model: layout features -> projection -> GELU -> CTC softmax.
///
/// Models DocLayout-YOLO backbone features feeding into FireRed-OCR's
/// encoder-based recognition path with CTC decoding.
///
/// Input: `[NUM_ANCHORS, FEATURE_DIM]` (Variable, layout features).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC softmax probabilities [0, 1]).
fn build_layout_to_firered_ocr_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_layout_to_firered_ocr");

    let input = b.add_input("layout_features", &[NUM_ANCHORS, FEATURE_DIM]);

    // FireRed-OCR encoder projection
    let enc_w = b.add_input("enc_weight", &[FFN_DIM, FEATURE_DIM]);
    let encoded = b.add_linear(input, enc_w, None, &[NUM_ANCHORS, FFN_DIM]);
    let activated = b.add_gelu(encoded, &[NUM_ANCHORS, FFN_DIM]);

    // Down-project back
    let down_w = b.add_input("down_weight", &[FEATURE_DIM, FFN_DIM]);
    let down = b.add_linear(activated, down_w, None, &[NUM_ANCHORS, FEATURE_DIM]);

    // Residual connection
    let residual = b.add_binary_add(input, down, &[NUM_ANCHORS, FEATURE_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(residual, 0, 0, SEQ_LEN, &[SEQ_LEN, FEATURE_DIM]);

    // CTC head: Linear -> Softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid layout to FireRed-OCR kernel")
}

fn layout_to_firered_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // layout_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FEATURE_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_layout_to_firered_ocr_ibp() {
    let def = build_layout_to_firered_ocr_kernel();
    let bindings = layout_to_firered_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layout -> FireRed-OCR");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Layout -> FireRed-OCR IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. Confidence gating: tighter layout bounds -> tighter OCR bounds
// ===========================================================================

/// Build cross-model: layout confidence -> projection -> sigmoid OCR score.
///
/// Verifies the monotone confidence property: when layout detection
/// filters low-confidence regions (bounds [0.5, 1.0] instead of [0, 1]),
/// downstream OCR bounds become tighter.
///
/// Input: `[NUM_ANCHORS, NUM_CLASSES]` (Variable, layout confidence).
/// Output: `[NUM_ANCHORS, 1]` (OCR quality score in [0, 1]).
fn build_confidence_gating_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_confidence_gating");

    let input = b.add_input("layout_confidence", &[NUM_ANCHORS, NUM_CLASSES]);

    // Project to scalar OCR quality estimate
    let w = b.add_input("gate_weight", &[1, NUM_CLASSES]);
    let bias = b.add_input("gate_bias", &[1]);
    let logit = b.add_linear(input, w, Some(bias), &[NUM_ANCHORS, 1]);
    let out = b.add_sigmoid(logit, &[NUM_ANCHORS, 1]);

    b.build(out).expect("valid confidence gating kernel")
}

fn confidence_gating_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // layout_confidence
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, NUM_CLASSES]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

#[test]
fn test_cross_confidence_gating_monotone_ibp() {
    let def = build_confidence_gating_kernel();
    let bindings = confidence_gating_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: full sigmoid range [0, 1]
    let wide_input = sigmoid_bounds(&[NUM_ANCHORS, NUM_CLASSES]);
    let wide_output = graph
        .propagate_ibp(&wide_input)
        .expect("IBP with wide input");

    // Tight input: high-confidence [0.5, 1.0]
    let tight_input = high_confidence_bounds(&[NUM_ANCHORS, NUM_CLASSES]);
    let tight_output = graph
        .propagate_ibp(&tight_input)
        .expect("IBP with tight input");

    assert_bounds_valid(&wide_output);
    assert_bounds_valid(&tight_output);

    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let (tight_lo, tight_hi) = bounds_min_max(&tight_output);
    let wide_width = wide_hi - wide_lo;
    let tight_width = tight_hi - tight_lo;

    eprintln!(
        "Confidence gating: wide=[{wide_lo}, {wide_hi}] (w={wide_width}), \
         tight=[{tight_lo}, {tight_hi}] (w={tight_width})"
    );

    // Monotone: tighter input -> tighter or equal output
    assert!(
        tight_width <= wide_width + 1e-4,
        "Tighter input should produce tighter output: tight_width={tight_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 6. Layout sigmoid -> OCR backbone CROWN
// ===========================================================================

#[test]
fn test_cross_layout_sigmoid_to_ocr_backbone_crown() {
    let def = build_layout_sigmoid_to_ocr_backbone_kernel();
    let bindings = layout_sigmoid_to_ocr_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Input must match the kernel's declared `[SEQ_LEN, NUM_CLASSES]` layout
    // (the Linear contracts the channel dim NUM_CLASSES against weight
    // [BACKBONE_CH, NUM_CLASSES]); the dims were transposed here, which made the
    // baseline IBP reject the input. Mirror the `_ibp` test's bounds.
    let input = sigmoid_bounds(&[SEQ_LEN, NUM_CLASSES]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, BACKBONE_CH]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Layout sigmoid -> OCR backbone CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. Layout DFL regression -> OCR recognition CROWN
// ===========================================================================

/// Build cross-model: DFL softmax -> weighted sum -> projection -> CTC logits.
///
/// Models the DFL box decode (softmax -> weighted sum) feeding into
/// recognition features. DFL output is bounded by the softmax [0,1] and
/// weighted sum structure.
///
/// Input: `[NUM_ANCHORS, FEATURE_DIM]` (Variable, DFL logits).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC logits).
fn build_dfl_to_ocr_recognition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_dfl_to_ocr_recognition");

    let input = b.add_input("dfl_logits", &[NUM_ANCHORS, FEATURE_DIM]);

    // Softmax (DFL decode): normalize logits to distribution
    let probs = b.add_softmax(input, 1, &[NUM_ANCHORS, FEATURE_DIM]);

    // Project DFL probabilities to recognition features
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, FEATURE_DIM]);
    let projected = b.add_linear(probs, proj_w, None, &[NUM_ANCHORS, FFN_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(projected, 0, 0, SEQ_LEN, &[SEQ_LEN, FFN_DIM]);

    // CTC head
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let out = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid DFL to OCR recognition kernel")
}

fn dfl_to_ocr_recognition_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // dfl_logits
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_dfl_to_ocr_recognition_crown() {
    let def = build_dfl_to_ocr_recognition_kernel();
    let bindings = dfl_to_ocr_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, FEATURE_DIM], 5.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DFL -> OCR recognition CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. VLM -> Layout: Qwen3-VL region proposal -> DocLayout-YOLO refinement
// ===========================================================================

/// Build cross-model: VLM decoder features -> projection -> sigmoid detection.
///
/// Models Qwen3-VL's decoder features (from region proposal tokens) being
/// projected to DocLayout-YOLO detection head for layout refinement.
///
/// Input: `[SEQ_LEN, FEATURE_DIM]` (Variable, VLM decoder features).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (layout class sigmoid [0, 1]).
fn build_vlm_to_layout_refinement_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_vlm_to_layout_refinement");

    let input = b.add_input("vlm_features", &[SEQ_LEN, FEATURE_DIM]);

    // Project VLM features to detection space
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, FEATURE_DIM]);
    let projected = b.add_linear(input, proj_w, None, &[SEQ_LEN, FFN_DIM]);
    let activated = b.add_relu(projected, &[SEQ_LEN, FFN_DIM]);

    // Detection head: Linear -> Sigmoid
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, FFN_DIM]);
    let det_b = b.add_input("det_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(activated, det_w, Some(det_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid VLM to layout refinement kernel")
}

fn vlm_to_layout_refinement_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // vlm_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_cross_vlm_to_layout_refinement_ibp() {
    let def = build_vlm_to_layout_refinement_kernel();
    let bindings = vlm_to_layout_refinement_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VLM -> layout refinement");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM -> layout refinement IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Granite-Docling page understanding -> table detection
// ===========================================================================

/// Build cross-model: vision projection features -> linear -> sigmoid table.
///
/// Models Granite-Docling's vision-language projection output feeding into
/// a table detection head (sigmoid confidence).
///
/// Input: `[SEQ_LEN, FEATURE_DIM]` (Variable, vision projection features).
/// Output: `[SEQ_LEN, TABLE_CLASSES]` (table class sigmoid [0, 1]).
fn build_granite_to_table_detection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_granite_to_table_detection");

    let input = b.add_input("vision_features", &[SEQ_LEN, FEATURE_DIM]);

    // Table detection head: Linear -> Sigmoid
    let det_w = b.add_input("det_weight", &[TABLE_CLASSES, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[TABLE_CLASSES]);
    let logits = b.add_linear(input, det_w, Some(det_b), &[SEQ_LEN, TABLE_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, TABLE_CLASSES]);

    b.build(out)
        .expect("valid Granite to table detection kernel")
}

fn granite_to_table_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // vision_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[TABLE_CLASSES, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[TABLE_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_cross_granite_to_table_detection_ibp() {
    let def = build_granite_to_table_detection_kernel();
    let bindings = granite_to_table_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite -> table detection");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, TABLE_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite -> table detection IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. GLM-OCR token prediction -> layout region association
// ===========================================================================

/// Build cross-model: GLM decoder logits -> softmax -> linear -> sigmoid.
///
/// Models GLM-OCR's token prediction (softmax distribution) being used
/// to associate text with layout regions (sigmoid region score).
///
/// Input: `[SEQ_LEN, FEATURE_DIM]` (Variable, GLM decoder logits).
/// Output: `[SEQ_LEN, 1]` (region association score in [0, 1]).
fn build_glm_to_layout_association_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_glm_to_layout_association");

    let input = b.add_input("glm_logits", &[SEQ_LEN, FEATURE_DIM]);

    // Token probability distribution
    let probs = b.add_softmax(input, 1, &[SEQ_LEN, FEATURE_DIM]);

    // Region association: Linear -> Sigmoid
    let assoc_w = b.add_input("assoc_weight", &[1, FEATURE_DIM]);
    let assoc_b = b.add_input("assoc_bias", &[1]);
    let logit = b.add_linear(probs, assoc_w, Some(assoc_b), &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(logit, &[SEQ_LEN, 1]);

    b.build(out)
        .expect("valid GLM to layout association kernel")
}

fn glm_to_layout_association_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // glm_logits
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, FEATURE_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

#[test]
fn test_cross_glm_to_layout_association_ibp() {
    let def = build_glm_to_layout_association_kernel();
    let bindings = glm_to_layout_association_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM -> layout association");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM -> layout association IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. VLM -> Layout CROWN
// ===========================================================================

#[test]
fn test_cross_vlm_to_layout_refinement_crown() {
    let def = build_vlm_to_layout_refinement_kernel();
    let bindings = vlm_to_layout_refinement_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM -> layout refinement CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. Full Pipeline: 3-stage layout -> table -> cell OCR
// ===========================================================================

/// Build 3-stage pipeline: layout sigmoid -> table sigmoid -> CTC softmax.
///
/// Chains DocLayout-YOLO (sigmoid class confidence) -> Table Transformer
/// (sigmoid cell classification) -> PaddleOCR (CTC softmax character probs).
///
/// Input: `[NUM_ANCHORS, NUM_CLASSES]` (Variable, layout logits).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities [0, 1]).
fn build_three_stage_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_three_stage_pipeline");

    let input = b.add_input("layout_logits", &[NUM_ANCHORS, NUM_CLASSES]);

    // Stage 1: Layout detection sigmoid
    let layout_conf = b.add_sigmoid(input, &[NUM_ANCHORS, NUM_CLASSES]);

    // Stage 2: Table structure detection
    let table_w = b.add_input("table_weight", &[TABLE_CLASSES, NUM_CLASSES]);
    let table_b = b.add_input("table_bias", &[TABLE_CLASSES]);
    let table_logits = b.add_linear(
        layout_conf,
        table_w,
        Some(table_b),
        &[NUM_ANCHORS, TABLE_CLASSES],
    );
    let table_conf = b.add_sigmoid(table_logits, &[NUM_ANCHORS, TABLE_CLASSES]);

    // Stage 3: Cell OCR recognition
    let ocr_w = b.add_input("ocr_weight", &[FEATURE_DIM, TABLE_CLASSES]);
    let ocr_features = b.add_linear(table_conf, ocr_w, None, &[NUM_ANCHORS, FEATURE_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(ocr_features, 0, 0, SEQ_LEN, &[SEQ_LEN, FEATURE_DIM]);

    // CTC head
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid 3-stage pipeline kernel")
}

fn three_stage_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // layout_logits
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[TABLE_CLASSES, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[TABLE_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FEATURE_DIM, TABLE_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_three_stage_pipeline_ibp() {
    let def = build_three_stage_pipeline_kernel();
    let bindings = three_stage_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-stage pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("3-stage pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    // Final softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Full Pipeline: 4-stage page classify -> layout -> table -> OCR
// ===========================================================================

/// Build 4-stage pipeline: softmax page class -> sigmoid layout -> sigmoid
/// table -> CTC softmax.
///
/// Input: `[SEQ_LEN, FEATURE_DIM]` (Variable, page features).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities [0, 1]).
fn build_four_stage_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_four_stage_pipeline");

    let input = b.add_input("page_features", &[SEQ_LEN, FEATURE_DIM]);

    // Stage 1: Page classification (softmax)
    let page_w = b.add_input("page_weight", &[NUM_CLASSES, FEATURE_DIM]);
    let page_b = b.add_input("page_bias", &[NUM_CLASSES]);
    let page_logits = b.add_linear(input, page_w, Some(page_b), &[SEQ_LEN, NUM_CLASSES]);
    let page_probs = b.add_softmax(page_logits, 1, &[SEQ_LEN, NUM_CLASSES]);

    // Stage 2: Layout detection (sigmoid)
    let layout_w = b.add_input("layout_weight", &[NUM_CLASSES, NUM_CLASSES]);
    let layout_b = b.add_input("layout_bias", &[NUM_CLASSES]);
    let layout_logits = b.add_linear(
        page_probs,
        layout_w,
        Some(layout_b),
        &[SEQ_LEN, NUM_CLASSES],
    );
    let layout_conf = b.add_sigmoid(layout_logits, &[SEQ_LEN, NUM_CLASSES]);

    // Stage 3: Table detection (sigmoid)
    let table_w = b.add_input("table_weight", &[TABLE_CLASSES, NUM_CLASSES]);
    let table_b = b.add_input("table_bias", &[TABLE_CLASSES]);
    let table_logits = b.add_linear(
        layout_conf,
        table_w,
        Some(table_b),
        &[SEQ_LEN, TABLE_CLASSES],
    );
    let table_conf = b.add_sigmoid(table_logits, &[SEQ_LEN, TABLE_CLASSES]);

    // Stage 4: OCR recognition (CTC softmax)
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, TABLE_CLASSES]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(table_conf, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid 4-stage pipeline kernel")
}

fn four_stage_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // page_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, FEATURE_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[TABLE_CLASSES, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[TABLE_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, TABLE_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_four_stage_pipeline_ibp() {
    let def = build_four_stage_pipeline_kernel();
    let bindings = four_stage_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 3.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-stage pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("4-stage pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    // Final softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 14. End-to-end: image -> regions -> text -> structured output
// ===========================================================================

/// Build end-to-end document pipeline: image -> conv backbone -> sigmoid
/// regions -> projection -> GELU -> CTC head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities [0, 1]).
fn build_end_to_end_document_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_end_to_end_document");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv backbone: image -> features
    let conv_w = b.add_input("conv_weight", &[BACKBONE_CH, IN_CHANNELS, 3, 3]);
    let conv_bias = b.add_input("conv_bias", &[BACKBONE_CH]);
    let bn_mean = b.add_input("bn_mean", &[BACKBONE_CH]);
    let bn_var = b.add_input("bn_var", &[BACKBONE_CH]);
    let bn_weight = b.add_input("bn_weight", &[BACKBONE_CH]);
    let bn_bias = b.add_input("bn_bias", &[BACKBONE_CH]);
    let eps = b.add_input("eps", &[1]);

    let conv_out_size = IMG_SIZE / 2; // stride=2
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_bias),
        2,
        2,
        1,
        1,
        &[BACKBONE_CH, conv_out_size, conv_out_size],
    );
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        eps,
        &[BACKBONE_CH, conv_out_size, conv_out_size],
    );
    let features = b.add_relu(bn_out, &[BACKBONE_CH, conv_out_size, conv_out_size]);

    // Region detection: 1x1 conv -> sigmoid
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, BACKBONE_CH, 1, 1]);
    let det_bias = b.add_input("det_bias", &[NUM_CLASSES]);
    let det_out = b.add_conv2d(
        features,
        det_w,
        Some(det_bias),
        1,
        1,
        0,
        0,
        &[NUM_CLASSES, conv_out_size, conv_out_size],
    );
    let regions = b.add_sigmoid(det_out, &[NUM_CLASSES, conv_out_size, conv_out_size]);

    // Flatten spatial: [NUM_CLASSES, H, W] -> [NUM_CLASSES, H*W]
    let flat_len = conv_out_size * conv_out_size;
    let flat = b.add_reshape(regions, &[NUM_CLASSES, flat_len]);

    // Transpose: [NUM_CLASSES, flat_len] -> [flat_len, NUM_CLASSES]
    let transposed = b.add_transpose(flat, &[1, 0], &[flat_len, NUM_CLASSES]);

    // Narrow to SEQ_LEN positions
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, NUM_CLASSES]);

    // OCR projection: Linear -> GELU -> Linear
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, NUM_CLASSES]);
    let projected = b.add_linear(narrowed, proj_w, None, &[SEQ_LEN, FFN_DIM]);
    let activated = b.add_gelu(projected, &[SEQ_LEN, FFN_DIM]);

    // CTC head: Linear -> Softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(activated, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid end-to-end document pipeline kernel")
}

fn end_to_end_document_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BACKBONE_CH, IN_CHANNELS, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[BACKBONE_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, BACKBONE_CH, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, NUM_CLASSES]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, FFN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_cross_end_to_end_document_ibp() {
    let def = build_end_to_end_document_kernel();
    let bindings = end_to_end_document_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through end-to-end document pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("End-to-end document IBP: bounds=[{lo_min}, {hi_max}]");

    // Final softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. 3-stage pipeline CROWN
// ===========================================================================

#[test]
fn test_cross_three_stage_pipeline_crown() {
    let def = build_three_stage_pipeline_kernel();
    let bindings = three_stage_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, NUM_CLASSES], 5.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("3-stage pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 16. End-to-end document CROWN
// ===========================================================================

#[test]
fn test_cross_end_to_end_document_crown() {
    let def = build_end_to_end_document_kernel();
    let bindings = end_to_end_document_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("End-to-end document CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
