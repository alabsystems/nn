// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for multi-head prediction (MTP) patterns used in GLM-OCR.
//!
//! Verifies IBP and CROWN bound propagation through Multi-Token Prediction
//! head variants. MTP extends autoregressive LM decoding by predicting
//! multiple future tokens simultaneously through parallel or sequential
//! prediction heads. GLM-OCR uses MTP for high-throughput OCR decoding.
//!
//! 1.  **Single MTP head**: Linear -> softmax output in [0, 1] (IBP)
//! 2.  **MTP head with RMSNorm gating** (IBP + CROWN)
//! 3.  **2-step MTP chain**: head1 -> head2 sequential prediction (IBP)
//! 4.  **3-step MTP chain**: deep prediction (IBP)
//! 5.  **MTP residual**: shared representation + per-head projection (IBP)
//! 6.  **MTP with tied weights across heads** (IBP)
//! 7.  **MTP independent heads**: parallel projections from same hidden (IBP)
//! 8.  **MTP confidence ranking**: per-head softmax max comparison (IBP)
//! 9.  **MTP vocabulary coverage**: all heads project to same vocab (IBP)
//! 10. **MTP with different hidden dimensions per head** (IBP)
//! 11. **CROWN tightness for MTP chains vs single head** (CROWN)
//! 12. **MTP monotone tightening**: smaller eps -> tighter bounds (IBP)
//! 13. **MTP + decoder composition**: decoder -> MTP heads (IBP + CROWN)
//! 14. **MTP head dropout/masking effect on bounds** (IBP)
//! 15. **Full MTP pipeline**: decoder -> RMSNorm -> N MTP heads -> softmax (IBP + CROWN)
//!
//! Architecture references:
//! - Multi-Token Prediction (Gloeckle et al., 2024): parallel prediction heads
//! - GLM-4V (THUDM): vision-language model with GLM-4 decoder + MTP
//! - RMSNorm (Zhang & Sennrich, 2019): normalization in GLM decoder layers
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN used in decoder blocks
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128, VOCAB_SIZE=256, NUM_HEADS=4
//!
//! Part of #4042: Compose tests for multi-head prediction (MTP) patterns.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const VOCAB_SIZE: usize = 256;
const WEIGHT_MAG: f32 = 0.02;
const NUM_HEADS: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a single MTP head: Linear(HIDDEN_DIM -> VOCAB_SIZE) -> softmax.
fn build_mtp_head(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> nn_dsl::TensorNodeId {
    let lm_w = b.add_input(&format!("{prefix}_lm_w"), &[vocab_size, hidden_dim]);
    let logits = b.add_linear(input, lm_w, None, &[seq_len, vocab_size]);
    b.add_softmax(logits, 1, &[seq_len, vocab_size])
}

/// Build a single MTP head returning pre-softmax logits (for chaining).
fn build_mtp_head_logits(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> nn_dsl::TensorNodeId {
    let lm_w = b.add_input(&format!("{prefix}_lm_w"), &[vocab_size, hidden_dim]);
    b.add_linear(input, lm_w, None, &[seq_len, vocab_size])
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Build a SwiGLU FFN block for decoder composition.
fn build_swiglu_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push MTP head weight binding (lm_w).
fn push_mtp_head_binding(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    vocab_size: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[vocab_size, hidden_dim]),
        weight_mag,
    )));
}

/// Push SwiGLU weight bindings (gate_w, up_w, down_w).
fn push_swiglu_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim, ffn_dim]),
        weight_mag,
    )));
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Single MTP head: Linear -> softmax output in [0, 1] (IBP)
// ===========================================================================

fn build_single_mtp_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_single_head");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_mtp_head(&mut b, input, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    b.build(out).expect("valid single MTP head kernel")
}

fn single_mtp_head_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    bindings
}

#[test]
fn test_mtp_single_head_softmax_01_ibp() {
    let def = build_single_mtp_head_kernel();
    let bindings = single_mtp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP single head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

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
// 2. MTP head with RMSNorm gating (IBP + CROWN)
// ===========================================================================

fn build_mtp_rmsnorm_gated_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_rmsnorm_gated");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);

    // RMSNorm gate before MTP head
    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_mtp_head(&mut b, normed, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    b.build(out).expect("valid MTP RMSNorm gated kernel")
}

fn mtp_rmsnorm_gated_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    bindings
}

#[test]
fn test_mtp_rmsnorm_gated_ibp() {
    let def = build_mtp_rmsnorm_gated_kernel();
    let bindings = mtp_rmsnorm_gated_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP RMSNorm gated IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

#[test]
fn test_mtp_rmsnorm_gated_crown() {
    let def = build_mtp_rmsnorm_gated_kernel();
    let bindings = mtp_rmsnorm_gated_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP RMSNorm gated CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 3. 2-step MTP chain: head1 -> head2 sequential prediction (IBP)
// ===========================================================================

fn build_mtp_chain_2step_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_chain_2step");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Step 1: project to vocab logits
    let logits1 = build_mtp_head_logits(&mut b, input, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);

    // Step 2: project logits back to hidden, then to vocab + softmax
    let down_w = b.add_input("down_proj_w", &[HIDDEN_DIM, VOCAB_SIZE]);
    let hidden2 = b.add_linear(logits1, down_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_mtp_head(&mut b, hidden2, "head1", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);

    b.build(out).expect("valid MTP 2-step chain kernel")
}

fn mtp_chain_2step_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // head0 lm_w
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    // down_proj_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]),
        WEIGHT_MAG,
    )));
    // head1 lm_w
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    bindings
}

#[test]
fn test_mtp_chain_2step_ibp() {
    let def = build_mtp_chain_2step_kernel();
    let bindings = mtp_chain_2step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP 2-step chain IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. 3-step MTP chain: deep prediction (IBP)
// ===========================================================================

#[test]
fn test_mtp_chain_3step_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_chain_3step");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Step 1
    let logits1 = build_mtp_head_logits(&mut b, input, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let down_w1 = b.add_input("down_proj_w1", &[HIDDEN_DIM, VOCAB_SIZE]);
    let hidden2 = b.add_linear(logits1, down_w1, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Step 2
    let logits2 = build_mtp_head_logits(&mut b, hidden2, "head1", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let down_w2 = b.add_input("down_proj_w2", &[HIDDEN_DIM, VOCAB_SIZE]);
    let hidden3 = b.add_linear(logits2, down_w2, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Step 3: final softmax
    let out = build_mtp_head(&mut b, hidden3, "head2", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let def = b.build(out).expect("valid MTP 3-step chain kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]),
        WEIGHT_MAG,
    )));
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]),
        WEIGHT_MAG,
    )));
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP 3-step chain IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. MTP residual: shared representation + per-head projection (IBP)
// ===========================================================================

#[test]
fn test_mtp_residual_shared_representation_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_residual");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Shared projection
    let shared_w = b.add_input("shared_proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let shared = b.add_linear(input, shared_w, None, &shape);

    // Residual: input + shared projection
    let combined = b.add_binary_add(input, shared, &shape);

    // Per-head MTP projection from residual
    let out = build_mtp_head(&mut b, combined, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let def = b.build(out).expect("valid MTP residual kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. MTP with tied weights across heads (IBP)
// ===========================================================================

#[test]
fn test_mtp_tied_weights_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_tied_weights");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Shared LM head weight used by both heads (weight tying)
    let shared_lm_w = b.add_input("shared_lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Head 0: direct projection
    let logits0 = b.add_linear(input, shared_lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let _probs0 = b.add_softmax(logits0, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Head 1: same weight, different input path (offset by linear transform)
    let offset_w = b.add_input("offset_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let offset_input = b.add_linear(input, offset_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let logits1 = b.add_linear(offset_input, shared_lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs1 = b.add_softmax(logits1, 1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(probs1).expect("valid MTP tied weights kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP tied weights IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. MTP independent heads: parallel projections from same hidden (IBP)
// ===========================================================================

#[test]
fn test_mtp_independent_heads_parallel_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_independent_heads");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Three independent MTP heads from same hidden state
    let _probs0 = build_mtp_head(&mut b, input, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let _probs1 = build_mtp_head(&mut b, input, "head1", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let probs2 = build_mtp_head(&mut b, input, "head2", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);

    // Build with last head as output (NY verifies the full graph)
    let def = b.build(probs2).expect("valid MTP independent heads kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..3 {
        push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP independent heads IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. MTP confidence ranking: per-head softmax max comparison (IBP)
// ===========================================================================

/// Verify that each MTP head produces valid softmax bounds independently,
/// enabling confidence-based ranking across heads.
#[test]
fn test_mtp_confidence_ranking_ibp() {
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let mut widths = Vec::new();
    for i in 0..3 {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_mtp_confidence_head{i}"));
        let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
        // Different weight magnitudes per head to simulate varying confidence
        let mag = WEIGHT_MAG * (1.0 + 0.5 * i as f32);
        let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
        let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
        let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
        let def = b.build(out).expect("valid confidence head kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
                mag,
            )),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        widths.push(width);
        eprintln!("MTP confidence head{i} (mag={mag:.3}) IBP: width={width:.6}");
    }

    // All heads should produce finite widths
    for (i, w) in widths.iter().enumerate() {
        assert!(w.is_finite(), "head{i} bound width must be finite");
    }
}

// ===========================================================================
// 9. MTP vocabulary coverage: all heads project to same vocab (IBP)
// ===========================================================================

/// All MTP heads project to the same vocabulary size, verifying that
/// independent heads produce consistent output dimensionality.
#[test]
fn test_mtp_vocab_coverage_same_vocab_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_vocab_coverage");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Two heads projecting to same vocab with different weight initializations
    let lm_w0 = b.add_input("lm_w0", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits0 = b.add_linear(input, lm_w0, None, &[SEQ_LEN, VOCAB_SIZE]);
    let _probs0 = b.add_softmax(logits0, 1, &[SEQ_LEN, VOCAB_SIZE]);

    let lm_w1 = b.add_input("lm_w1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits1 = b.add_linear(input, lm_w1, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs1 = b.add_softmax(logits1, 1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(probs1).expect("valid MTP vocab coverage kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG * 1.5,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Verify output shape matches vocab size
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "MTP output shape must match vocab size"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP vocab coverage IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. MTP with different hidden dimensions per head (IBP)
// ===========================================================================

#[test]
fn test_mtp_different_hidden_dims_ibp() {
    // Head with smaller hidden dim (32) via intermediate projection
    let small_hidden = 32;

    let mut b = TensorBlockBuilder::new("dpdf_mtp_diff_hidden_dims");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Down-project to smaller hidden dim
    let down_w = b.add_input("down_proj_w", &[small_hidden, HIDDEN_DIM]);
    let small_hidden_state = b.add_linear(input, down_w, None, &[SEQ_LEN, small_hidden]);

    // MTP head from smaller hidden
    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, small_hidden]);
    let logits = b.add_linear(small_hidden_state, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    let def = b.build(out).expect("valid MTP diff hidden dims kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[small_hidden, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, small_hidden]),
            WEIGHT_MAG,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP diff hidden dims (small={small_hidden}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. CROWN tightness for MTP chains vs single head (CROWN)
// ===========================================================================

#[test]
fn test_mtp_crown_tightness_chain_vs_single() {
    // Single head CROWN
    let single_def = build_single_mtp_head_kernel();
    let single_bindings = single_mtp_head_bindings();
    let single_graph = tensor_kernel_to_graph(&single_def, &single_bindings).expect("single graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (single_method, single_output, single_fallback) =
        assert_crown_tighter_when_not_fallback(&single_graph, &input);
    assert_bounds_valid(&single_output);
    let single_width = bound_width(&single_output);

    // 2-step chain CROWN
    let chain_def = build_mtp_chain_2step_kernel();
    let chain_bindings = mtp_chain_2step_bindings();
    let chain_graph = tensor_kernel_to_graph(&chain_def, &chain_bindings).expect("chain graph");

    let (chain_method, chain_output, chain_fallback) =
        assert_crown_tighter_when_not_fallback(&chain_graph, &input);
    assert_bounds_valid(&chain_output);
    let chain_width = bound_width(&chain_output);

    eprintln!(
        "MTP CROWN tightness: single={single_width:.6} (method={single_method:?}), \
         chain={chain_width:.6} (method={chain_method:?})"
    );
    if let Some(reason) = &single_fallback {
        eprintln!("Single fallback: {reason}");
    }
    if let Some(reason) = &chain_fallback {
        eprintln!("Chain fallback: {reason}");
    }

    // Both must produce finite, valid bounds
    assert!(single_width.is_finite(), "single width must be finite");
    assert!(chain_width.is_finite(), "chain width must be finite");
}

// ===========================================================================
// 12. MTP monotone tightening: smaller eps -> tighter bounds (IBP)
// ===========================================================================

#[test]
fn test_mtp_monotone_tightening_ibp() {
    let def = build_single_mtp_head_kernel();
    let bindings = single_mtp_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "MTP monotone tightening: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tighter input must produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 13. MTP + decoder composition: decoder -> MTP heads (IBP + CROWN)
// ===========================================================================

fn build_decoder_mtp_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_decoder_compose");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Simplified decoder block: MHA -> residual -> RMSNorm -> SwiGLU -> residual
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    let h = b.add_binary_add(input, attn_out, &shape);

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(h, eps, 1, norm_w, &shape);

    // SwiGLU FFN
    let ffn_out = build_swiglu_block(&mut b, normed, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let decoder_out = b.add_binary_add(h, ffn_out, &shape);

    // MTP head from decoder output
    let out = build_mtp_head(
        &mut b,
        decoder_out,
        "mtp_head",
        SEQ_LEN,
        HIDDEN_DIM,
        VOCAB_SIZE,
    );

    b.build(out).expect("valid decoder + MTP kernel")
}

fn decoder_mtp_bindings() -> Vec<TensorParamBinding> {
    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };
    let mut bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    bindings
}

#[test]
fn test_mtp_decoder_compose_ibp() {
    let def = build_decoder_mtp_kernel();
    let bindings = decoder_mtp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder + MTP IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

#[test]
fn test_mtp_decoder_compose_crown() {
    let def = build_decoder_mtp_kernel();
    let bindings = decoder_mtp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder + MTP CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. MTP head dropout/masking effect on bounds (IBP)
// ===========================================================================

/// Model MTP head dropout as: prob * head_output + (1 - prob) * 0.
/// At inference prob=1 (no dropout), but we verify bounds with a scaling
/// factor to model the effect of stochastic head masking.
#[test]
fn test_mtp_head_dropout_masking_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_head_dropout");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];

    // MTP head output (pre-softmax for scaling)
    let logits = build_mtp_head_logits(&mut b, input, "head0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);

    // Scale by survival probability (models dropout masking)
    let alpha = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &vocab_shape);
    let scaled = b.add_binary_mul(logits, alpha_bc, &vocab_shape);

    // Softmax after scaling
    let out = b.add_softmax(scaled, 1, &vocab_shape);
    let def = b.build(out).expect("valid MTP dropout kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    // alpha = 0.9 (90% head survival probability)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.9f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MTP dropout masking IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. Full MTP pipeline: decoder -> RMSNorm -> N MTP heads -> softmax
//     (IBP + CROWN)
// ===========================================================================

fn build_full_mtp_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mtp_full_pipeline");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Simplified decoder: MHA -> residual
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");
    let h = b.add_binary_add(input, attn_out, &shape);

    // RMSNorm before MTP heads
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(h, eps, 1, norm_w, &shape);

    // 3 parallel MTP heads from the normalized decoder output
    let _probs0 = build_mtp_head(&mut b, normed, "mtp0", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let _probs1 = build_mtp_head(&mut b, normed, "mtp1", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let probs2 = build_mtp_head(&mut b, normed, "mtp2", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);

    b.build(probs2).expect("valid full MTP pipeline kernel")
}

fn full_mtp_pipeline_bindings() -> Vec<TensorParamBinding> {
    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };
    let mut bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    // 3 MTP head weights
    for _ in 0..3 {
        push_mtp_head_binding(&mut bindings, HIDDEN_DIM, VOCAB_SIZE, WEIGHT_MAG);
    }
    bindings
}

#[test]
fn test_mtp_full_pipeline_ibp() {
    let def = build_full_mtp_pipeline_kernel();
    let bindings = full_mtp_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full MTP pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

#[test]
fn test_mtp_full_pipeline_crown() {
    let def = build_full_mtp_pipeline_kernel();
    let bindings = full_mtp_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full MTP pipeline CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
