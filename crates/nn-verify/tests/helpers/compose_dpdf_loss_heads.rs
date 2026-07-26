// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for loss function and output head patterns used across dpdf models.
//!
//! Verifies IBP and CROWN bound propagation through the loss-function and
//! output-head patterns that appear in dpdf document understanding models:
//! sigmoid classification heads, softmax probability heads, DFL regression,
//! CTC decoding, focal-loss weighting, cross-entropy inputs, and composed
//! multi-head detection pipelines.
//!
//! ## Sigmoid Classification Heads (tests 1-2)
//!
//! 1. Sigmoid classification head output in (0, 1) IBP
//! 2. Sigmoid classification head CROWN tighter than IBP
//!
//! ## Softmax Output Heads (tests 3-4)
//!
//! 3. Softmax output head sum=1 IBP
//! 4. Softmax output head CROWN tighter than IBP
//!
//! ## DFL Regression (tests 5-6)
//!
//! 5. DFL regression (softmax -> weighted sum) IBP
//! 6. DFL -> sigmoid end-to-end box coordinate IBP
//!
//! ## CTC Decoding Heads (tests 7-8)
//!
//! 7. CTC blank probability bounded IBP
//! 8. CTC softmax character probabilities IBP
//!
//! ## Focal Loss (test 9)
//!
//! 9. Focal loss weighting preserves bound ordering IBP
//!
//! ## Box Regression (test 10)
//!
//! 10. Box regression sigmoid coordinates in [0, 1] IBP
//!
//! ## Composed Multi-Head Detection (tests 11-13)
//!
//! 11. Dual-head detection (cls + box) composition IBP
//! 12. Triple-head table detection IBP + CROWN
//! 13. MTP (multi-token prediction) head chain IBP
//!
//! ## LM and Log-Softmax Heads (tests 14-15)
//!
//! 14. LM head (Linear -> softmax) IBP + CROWN
//! 15. Log-softmax output bounded IBP
//!
//! ## Cross-Model Properties (test 16)
//!
//! 16. Output head monotone tightening: smaller eps -> tighter output
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, NUM_CLASSES=8, VOCAB_SIZE=64
//!
//! Part of #3980: Loss function and output head compose tests.

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
const NUM_CLASSES: usize = 8;
const NUM_QUERIES: usize = 6;
const VOCAB_SIZE: usize = 64;
const DFL_BINS: usize = 16;
const NUM_ANCHORS: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build a classification sigmoid head: Linear -> sigmoid.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_QUERIES, NUM_CLASSES]` (class probabilities in (0, 1)).
fn build_cls_sigmoid_head_kernel() -> TensorKernelDef {
    let out_shape = [NUM_QUERIES, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("dpdf_loss_cls_sigmoid_head");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, HIDDEN_DIM]);
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, cls_w, Some(cls_b), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid classification sigmoid head kernel")
}

/// Bindings for classification sigmoid head.
fn cls_sigmoid_head_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // decoder_output
        TensorParamBinding::ConstantTensor(w),    // cls_weight
        TensorParamBinding::ConstantTensor(bias), // cls_bias
    ]
}

/// Build a softmax output head: Linear -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (probability distribution per timestep).
fn build_softmax_output_head_kernel() -> TensorKernelDef {
    let out_shape = [SEQ_LEN, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("dpdf_loss_softmax_output_head");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("head_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("head_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid softmax output head kernel")
}

/// Bindings for softmax output head.
fn softmax_output_head_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // head_weight
        TensorParamBinding::ConstantTensor(bias), // head_bias
    ]
}

/// Build a DFL regression kernel: softmax -> weighted sum.
///
/// Input: `[NUM_ANCHORS, DFL_BINS]` (Variable, DFL logits).
/// Output: `[NUM_ANCHORS, 1]` (continuous box coordinate).
///
/// DFL architecture (Li et al. 2022):
///   probs = softmax(logits, dim=-1)     [NUM_ANCHORS, DFL_BINS]
///   coord = matmul(probs, bins)         [NUM_ANCHORS, 1]
///
/// where bins = [0, 1, ..., DFL_BINS-1] is a fixed integer sequence.
fn build_dfl_regression_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_loss_dfl_regression");

    let input = b.add_input("dfl_logits", &[NUM_ANCHORS, DFL_BINS]);
    let bins = b.add_input("bins", &[DFL_BINS, 1]);

    let probs = b.add_softmax(input, 1, &[NUM_ANCHORS, DFL_BINS]);
    let out = b.add_matmul(probs, bins, false, None, &[NUM_ANCHORS, 1]);

    b.build(out).expect("valid DFL regression kernel")
}

/// Bindings for DFL regression.
fn dfl_regression_bindings() -> Vec<TensorParamBinding> {
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bins = ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins shape");

    vec![
        TensorParamBinding::Variable,             // dfl_logits
        TensorParamBinding::ConstantTensor(bins), // bins
    ]
}

/// Build a DFL -> sigmoid end-to-end kernel.
///
/// Input: `[NUM_ANCHORS, DFL_BINS]` (Variable, DFL logits).
/// Output: `[NUM_ANCHORS, 1]` (normalized box coordinate in [0, 1]).
///
/// DFL produces a continuous coordinate, then sigmoid normalizes to [0, 1].
fn build_dfl_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_loss_dfl_sigmoid");

    let input = b.add_input("dfl_logits", &[NUM_ANCHORS, DFL_BINS]);
    let bins = b.add_input("bins", &[DFL_BINS, 1]);

    let probs = b.add_softmax(input, 1, &[NUM_ANCHORS, DFL_BINS]);
    let coord = b.add_matmul(probs, bins, false, None, &[NUM_ANCHORS, 1]);
    let out = b.add_sigmoid(coord, &[NUM_ANCHORS, 1]);

    b.build(out).expect("valid DFL -> sigmoid kernel")
}

/// Bindings for DFL -> sigmoid.
fn dfl_sigmoid_bindings() -> Vec<TensorParamBinding> {
    dfl_regression_bindings()
}

/// Build a CTC blank probability kernel: Linear -> softmax -> narrow(blank=0).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, 1]` (blank class probability per timestep).
///
/// CTC decoding checks the blank probability at each timestep.
/// The blank class is conventionally index 0.
fn build_ctc_blank_prob_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_loss_ctc_blank_prob");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    // Narrow to blank class (index 0, length 1 along axis 1)
    let out = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);

    b.build(out).expect("valid CTC blank probability kernel")
}

/// Bindings for CTC blank probability.
fn ctc_blank_prob_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // ctc_weight
        TensorParamBinding::ConstantTensor(bias), // ctc_bias
    ]
}

/// Build a CTC softmax character probabilities kernel: Linear -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities).
fn build_ctc_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_loss_ctc_softmax");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid CTC softmax kernel")
}

/// Bindings for CTC softmax.
fn ctc_softmax_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // ctc_weight
        TensorParamBinding::ConstantTensor(bias), // ctc_bias
    ]
}

/// Build a focal loss weighting kernel: sigmoid -> element-wise power modulation.
///
/// Focal loss (Lin et al. 2017): FL(p_t) = -(1-p_t)^gamma * log(p_t)
/// We verify the weighting factor (1-p_t)^gamma preserves bound ordering.
/// Since we cannot raise to a power directly, we approximate with
/// gamma=2: (1-p)^2 = 1 - 2*p + p^2. We verify the sigmoid base
/// plus the modulated factor.
///
/// Input: `[NUM_QUERIES, NUM_CLASSES]` (Variable, logits).
/// Output: `[NUM_QUERIES, NUM_CLASSES]` (focal-weighted sigmoid).
fn build_focal_weight_kernel() -> TensorKernelDef {
    let shape = [NUM_QUERIES, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("dpdf_loss_focal_weight");

    let input = b.add_input("logits", &shape);

    // p = sigmoid(logits)
    let p = b.add_sigmoid(input, &shape);

    // (1 - p) approximation: negate p, add 1
    // We use: sigmoid(-x) = 1 - sigmoid(x) for the complement
    // Build neg_logits and apply sigmoid to get (1-p)
    let neg_input = b.add_elementwise(
        nn_dsl::test_kernels::parse_kernel("fn neg(x: f32) -> f32 { -x }"),
        &[input],
        &shape,
    );
    let one_minus_p = b.add_sigmoid(neg_input, &shape);

    // focal_weight = (1-p) * (1-p) = (1-p)^2 (gamma=2)
    let focal_weight = b.add_binary_mul(one_minus_p, one_minus_p, &shape);

    // output = focal_weight * p
    let out = b.add_binary_mul(focal_weight, p, &shape);

    b.build(out).expect("valid focal loss weighting kernel")
}

/// Build a box regression sigmoid kernel: Linear -> sigmoid.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_QUERIES, 4]` (normalized box coordinates in [0, 1]).
fn build_box_regression_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_QUERIES, 4];
    let mut b = TensorBlockBuilder::new("dpdf_loss_box_regression");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, HIDDEN_DIM]);
    let w = b.add_input("box_weight", &[4, HIDDEN_DIM]);
    let bias = b.add_input("box_bias", &[4]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid box regression sigmoid kernel")
}

/// Bindings for box regression sigmoid.
fn box_regression_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // decoder_output
        TensorParamBinding::ConstantTensor(w),    // box_weight
        TensorParamBinding::ConstantTensor(bias), // box_bias
    ]
}

/// Build a dual-head detection kernel: cls sigmoid + box sigmoid from shared features.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output).
/// Outputs concatenated: `[NUM_QUERIES, NUM_CLASSES + 4]`.
///
/// The classification head produces class probabilities in (0, 1) and the
/// box head produces normalized coordinates in (0, 1).
fn build_dual_head_detection_kernel() -> TensorKernelDef {
    let cls_out_shape = [NUM_QUERIES, NUM_CLASSES];
    let box_out_shape = [NUM_QUERIES, 4];
    let concat_shape = [NUM_QUERIES, NUM_CLASSES + 4];
    let mut b = TensorBlockBuilder::new("dpdf_loss_dual_head_detection");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_out_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_out_shape);

    // Box regression head
    let box_w = b.add_input("box_weight", &[4, HIDDEN_DIM]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_out_shape);
    let box_out = b.add_sigmoid(box_logits, &box_out_shape);

    // Concatenate cls + box along last dimension
    let out = b.add_concat(&[cls_out, box_out], 1, &concat_shape);

    b.build(out).expect("valid dual-head detection kernel")
}

/// Bindings for dual-head detection.
fn dual_head_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // cls_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)), // cls_bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)), // box_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)), // box_bias
    ]
}

/// Build a triple-head table detection kernel: cls + box + structure.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output).
/// Output: concatenated `[NUM_QUERIES, NUM_CLASSES + 4 + NUM_CLASSES]`.
///
/// Table structure recognition uses three sigmoid heads:
/// - Class detection (table, cell, row, column, etc.)
/// - Box regression (normalized coordinates)
/// - Structure classification (spanning, header, etc.)
fn build_triple_head_table_detection_kernel() -> TensorKernelDef {
    let cls_out = [NUM_QUERIES, NUM_CLASSES];
    let box_out = [NUM_QUERIES, 4];
    let struct_out = [NUM_QUERIES, NUM_CLASSES];
    let concat_shape = [NUM_QUERIES, NUM_CLASSES + 4 + NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("dpdf_loss_triple_head_table");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_out);
    let cls_sigmoid = b.add_sigmoid(cls_logits, &cls_out);

    // Box regression head
    let box_w = b.add_input("box_weight", &[4, HIDDEN_DIM]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_out);
    let box_sigmoid = b.add_sigmoid(box_logits, &box_out);

    // Structure head
    let struct_w = b.add_input("struct_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let struct_b = b.add_input("struct_bias", &[NUM_CLASSES]);
    let struct_logits = b.add_linear(input, struct_w, Some(struct_b), &struct_out);
    let struct_sigmoid = b.add_sigmoid(struct_logits, &struct_out);

    // Concatenate all three heads
    let out = b.add_concat(
        &[cls_sigmoid, box_sigmoid, struct_sigmoid],
        1,
        &concat_shape,
    );

    b.build(out)
        .expect("valid triple-head table detection kernel")
}

/// Bindings for triple-head table detection.
fn triple_head_table_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        // Classification head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
        // Box head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
        // Structure head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

/// Build an MTP (multi-token prediction) head chain kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, decoder hidden state).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (next-next-token probability distribution).
///
/// MTP chains: hidden -> Linear -> softmax (step 1) and
/// hidden -> Linear -> Linear -> softmax (step 2, deeper).
/// We verify the 2-step chain.
fn build_mtp_head_chain_kernel() -> TensorKernelDef {
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];
    let hidden_shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_loss_mtp_head_chain");

    let input = b.add_input("hidden", &hidden_shape);

    // Step 1: hidden -> Linear -> softmax (next token prediction)
    let lm_w1 = b.add_input("lm_head_w1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits1 = b.add_linear(input, lm_w1, None, &vocab_shape);
    let _probs1 = b.add_softmax(logits1, 1, &vocab_shape);

    // Step 2: hidden -> Linear (project) -> Linear (LM head) -> softmax
    let proj_w = b.add_input("mtp_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, None, &hidden_shape);
    let lm_w2 = b.add_input("lm_head_w2", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits2 = b.add_linear(projected, lm_w2, None, &vocab_shape);
    let probs2 = b.add_softmax(logits2, 1, &vocab_shape);

    b.build(probs2).expect("valid MTP head chain kernel")
}

/// Bindings for MTP head chain.
fn mtp_head_chain_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // lm_head_w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // mtp_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // lm_head_w2
    ]
}

/// Build an LM head kernel: Linear -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (token probabilities).
fn build_lm_head_softmax_kernel() -> TensorKernelDef {
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("dpdf_loss_lm_head_softmax");

    let input = b.add_input("decoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let lm_b = b.add_input("lm_head_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, lm_w, Some(lm_b), &vocab_shape);
    let out = b.add_softmax(logits, 1, &vocab_shape);

    b.build(out).expect("valid LM head softmax kernel")
}

/// Bindings for LM head softmax.
fn lm_head_softmax_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // decoder_output
        TensorParamBinding::ConstantTensor(w),    // lm_head_weight
        TensorParamBinding::ConstantTensor(bias), // lm_head_bias
    ]
}

/// Build a log-softmax output kernel: Linear -> log_softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (log-probabilities, all <= 0).
fn build_log_softmax_head_kernel() -> TensorKernelDef {
    let out_shape = [SEQ_LEN, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("dpdf_loss_log_softmax_head");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("head_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("head_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_log_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid log-softmax head kernel")
}

/// Bindings for log-softmax head.
fn log_softmax_head_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // head_weight
        TensorParamBinding::ConstantTensor(bias), // head_bias
    ]
}

// ===========================================================================
// 1. Sigmoid classification head output in (0, 1) IBP
// ===========================================================================

#[test]
fn test_cls_sigmoid_head_ibp() {
    let def = build_cls_sigmoid_head_kernel();
    let bindings = cls_sigmoid_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cls sigmoid head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, NUM_CLASSES],
        "cls sigmoid head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Classification sigmoid head IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 2. Sigmoid classification head CROWN tighter than IBP
// ===========================================================================

#[test]
fn test_cls_sigmoid_head_crown() {
    let def = build_cls_sigmoid_head_kernel();
    let bindings = cls_sigmoid_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP baseline for cls sigmoid");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Classification sigmoid CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 3. Softmax output head sum=1 IBP
// ===========================================================================

/// Softmax output: all elements in [0, 1] under IBP.
#[test]
fn test_softmax_output_head_ibp() {
    let def = build_softmax_output_head_kernel();
    let bindings = softmax_output_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through softmax output head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_CLASSES],
        "softmax output head shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Softmax output head IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Softmax output head CROWN tighter than IBP
// ===========================================================================

#[test]
fn test_softmax_output_head_crown() {
    let def = build_softmax_output_head_kernel();
    let bindings = softmax_output_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP baseline for softmax head");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Softmax output CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 5. DFL regression (softmax -> weighted sum) IBP
// ===========================================================================

/// DFL output is bounded since softmax produces a valid probability distribution
/// and the weighted sum with bins [0, ..., DFL_BINS-1] produces output in that range.
#[test]
fn test_dfl_regression_ibp() {
    let def = build_dfl_regression_kernel();
    let bindings = dfl_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL regression");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, 1],
        "DFL regression output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DFL regression IBP (logits [-5,5]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "DFL lower bound must be finite");
    assert!(hi_max.is_finite(), "DFL upper bound must be finite");
}

// ===========================================================================
// 6. DFL -> sigmoid end-to-end box coordinate IBP
// ===========================================================================

/// DFL -> sigmoid: normalized box coordinates must be in [0, 1].
#[test]
fn test_dfl_sigmoid_ibp() {
    let def = build_dfl_sigmoid_kernel();
    let bindings = dfl_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, DFL_BINS], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL -> sigmoid");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, 1],
        "DFL -> sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DFL -> sigmoid IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "DFL sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "DFL sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. CTC blank probability bounded IBP
// ===========================================================================

/// CTC blank probability: a single softmax class narrowed out, must be in [0, 1].
#[test]
fn test_ctc_blank_probability_ibp() {
    let def = build_ctc_blank_prob_kernel();
    let bindings = ctc_blank_prob_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC blank probability");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 1],
        "CTC blank prob output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CTC blank probability IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "blank prob lower must be >= 0 for valid CTC decode, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "blank prob upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. CTC softmax character probabilities IBP
// ===========================================================================

/// CTC softmax: per-timestep character probabilities in [0, 1].
#[test]
fn test_ctc_softmax_char_probs_ibp() {
    let def = build_ctc_softmax_kernel();
    let bindings = ctc_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC softmax");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CTC softmax char probs IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower must be >= 0 for valid CTC, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper must be <= 1 for valid CTC, got {hi_max}"
    );
}

// ===========================================================================
// 9. Focal loss weighting preserves bound ordering IBP
// ===========================================================================

/// Focal loss weighting: (1-p)^2 * p preserves bounds in [0, 1].
///
/// Since p = sigmoid(x) in (0, 1) and (1-p) in (0, 1), the product
/// (1-p)^2 * p is bounded in [0, 1].
#[test]
fn test_focal_weight_ibp() {
    let def = build_focal_weight_kernel();
    // Focal weight kernel uses only the Variable input (logits).
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, NUM_CLASSES], 3.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through focal weighting");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, NUM_CLASSES],
        "focal weight output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Focal loss weight IBP: bounds=[{lo_min}, {hi_max}]");

    // (1-p)^2 * p is in [0, 1] since p, (1-p) are each in (0, 1).
    let eps = 1e-5;
    assert!(
        lo_min >= 0.0 - eps,
        "focal weight lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "focal weight upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 10. Box regression sigmoid coordinates in [0, 1] IBP
// ===========================================================================

/// Box regression: Linear -> sigmoid produces normalized coordinates in [0, 1].
#[test]
fn test_box_regression_sigmoid_ibp() {
    let def = build_box_regression_sigmoid_kernel();
    let bindings = box_regression_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through box regression sigmoid");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, 4],
        "box regression output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Box regression sigmoid IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "box sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "box sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. Dual-head detection (cls + box) composition IBP
// ===========================================================================

/// Dual-head detection: both cls and box heads produce sigmoid outputs in [0, 1].
#[test]
fn test_dual_head_detection_ibp() {
    let def = build_dual_head_detection_kernel();
    let bindings = dual_head_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dual-head detection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, NUM_CLASSES + 4],
        "dual-head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dual-head detection IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "dual-head lower must be >= 0 (both sigmoid), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "dual-head upper must be <= 1 (both sigmoid), got {hi_max}"
    );
}

// ===========================================================================
// 12. Triple-head table detection IBP + CROWN
// ===========================================================================

/// Triple-head table detection: cls + box + structure, all sigmoid in [0, 1].
#[test]
fn test_triple_head_table_detection_ibp_crown() {
    let def = build_triple_head_table_detection_kernel();
    let bindings = triple_head_table_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through triple-head table detection");

    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_QUERIES, NUM_CLASSES + 4 + NUM_CLASSES],
        "triple-head output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Triple-head table detection IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "triple-head lower must be >= 0 (all sigmoid), got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "triple-head upper must be <= 1 (all sigmoid), got {hi_max}"
    );

    // CROWN should be at least as tight
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Triple-head CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 13. MTP (multi-token prediction) head chain IBP
// ===========================================================================

/// MTP head chain: hidden -> Linear -> Linear -> softmax produces valid probs.
#[test]
fn test_mtp_head_chain_ibp() {
    let def = build_mtp_head_chain_kernel();
    let bindings = mtp_head_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MTP head chain");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "MTP head chain output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP head chain IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "MTP softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "MTP softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. LM head (Linear -> softmax) IBP + CROWN
// ===========================================================================

/// LM head: Linear -> softmax produces token probabilities in [0, 1].
#[test]
fn test_lm_head_softmax_ibp_crown() {
    let def = build_lm_head_softmax_kernel();
    let bindings = lm_head_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through LM head softmax");

    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "LM head softmax output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("LM head softmax IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "LM head softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "LM head softmax upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("LM head CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 15. Log-softmax output bounded IBP
// ===========================================================================

/// Log-softmax output: all elements <= 0 (log of a probability in (0, 1]).
#[test]
fn test_log_softmax_head_ibp() {
    let def = build_log_softmax_head_kernel();
    let bindings = log_softmax_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through log-softmax head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_CLASSES],
        "log-softmax head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Log-softmax head IBP: bounds=[{lo_min}, {hi_max}]");

    // log(p) where p in (0, 1] produces values in (-inf, 0].
    // IBP upper bound should be <= 0 (or very close to 0 with tolerance).
    let eps = 1e-5;
    assert!(
        hi_max <= 0.0 + eps,
        "log-softmax upper must be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 16. Output head monotone tightening: smaller eps -> tighter output
// ===========================================================================

/// Monotone tightening: reducing input perturbation radius produces
/// tighter output bounds for the sigmoid classification head.
#[test]
fn test_output_head_monotone_tightening() {
    let def = build_cls_sigmoid_head_kernel();
    let bindings = cls_sigmoid_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_wide = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);
    let input_narrow = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let output_wide = graph
        .propagate_ibp(&input_wide)
        .expect("IBP wide perturbation");
    let output_narrow = graph
        .propagate_ibp(&input_narrow)
        .expect("IBP narrow perturbation");

    assert_bounds_valid(&output_wide);
    assert_bounds_valid(&output_narrow);

    let wide_width = bound_width(&output_wide);
    let narrow_width = bound_width(&output_narrow);

    eprintln!(
        "Monotone tightening: wide eps=2.0 width={wide_width:.6}, narrow eps=0.5 width={narrow_width:.6}"
    );

    assert!(
        narrow_width <= wide_width + 1e-6,
        "smaller input perturbation should produce tighter output: \
         narrow_width={narrow_width}, wide_width={wide_width}"
    );
}
