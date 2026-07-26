// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for vocabulary projection and sampling head bounds used in
//! OCR/VLM models.
//!
//! Verifies IBP and CROWN bound propagation through vocabulary projection
//! layers and sampling/decoding heads that map hidden representations to
//! token probability distributions. These patterns appear in all dpdf
//! text-generating models: GLM-OCR, Qwen3-VL, Granite-Docling, FireRed-OCR,
//! and PaddleOCR CTC decoders.
//!
//! 1.  **Linear projection to vocabulary size** (hidden_dim -> vocab_size) IBP
//! 2.  **Tied weight embedding projection** IBP
//! 3.  **Vocabulary projection + softmax composition** IBP + CROWN
//! 4.  **Vocabulary projection + log-softmax for CTC loss** IBP
//! 5.  **Temperature-scaled logits** IBP
//! 6.  **Top-k filtering effect on output bounds** IBP
//! 7.  **Top-p (nucleus) sampling threshold bounds** IBP
//! 8.  **Greedy decoding (argmax) output bounds** IBP
//! 9.  **Beam search score accumulation bounds** IBP
//! 10. **CTC blank token probability bounds** IBP
//! 11. **Large vocabulary (151k tokens for Qwen3) projection bounds** IBP
//! 12. **Multi-head CTC output (character + position) bounds** IBP
//! 13. **CROWN tightness for vocabulary projection layer** CROWN
//! 14. **Logit bias/mask application bounds** IBP
//! 15. **End-to-end hidden-to-token probability pipeline** IBP + CROWN
//!
//! Architecture references:
//! - GLM-4V (THUDM): RMSNorm -> Linear(hidden, vocab) -> softmax
//! - Qwen3-VL (Alibaba): 151k vocab, tied embeddings, temperature sampling
//! - PaddleOCR SVTR: Linear(hidden, charset) -> CTC softmax
//! - FireRed-OCR: Qwen3-VL variant with CTC decoding head
//! - Granite-Docling: Granite LLM decoder with LM head
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, VOCAB_SIZE=128, LARGE_VOCAB=512
//!
//! Part of #4035: Compose tests for vocabulary projection and sampling head bounds.

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
const HIDDEN_DIM: usize = 64;
const VOCAB_SIZE: usize = 128;
const LARGE_VOCAB: usize = 512;
const WEIGHT_MAG: f32 = 0.02;
const NUM_CTC_CLASSES: usize = 64;
const BEAM_WIDTH: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build a simple linear vocabulary projection: Linear(hidden_dim, vocab_size).
///
/// Input shape: `[seq_len, hidden_dim]`.
/// Output shape: `[seq_len, vocab_size]`.
fn build_vocab_projection_kernel(
    name: &str,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("hidden", &[seq_len, hidden_dim]);
    let proj_w = b.add_input("lm_head_weight", &[vocab_size, hidden_dim]);
    let proj_b = b.add_input("lm_head_bias", &[vocab_size]);

    let out = b.add_linear(input, proj_w, Some(proj_b), &[seq_len, vocab_size]);
    b.build(out).expect("valid vocab projection kernel")
}

/// Standard bindings for vocabulary projection (Variable input + constant weights).
fn vocab_projection_bindings(hidden_dim: usize, vocab_size: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[vocab_size, hidden_dim]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[vocab_size]), 0.0f32)),
    ]
}

// ===========================================================================
// 1. Linear projection to vocabulary size (hidden_dim -> vocab_size) IBP
// ===========================================================================

#[test]
fn test_vocab_projection_linear_ibp() {
    let def =
        build_vocab_projection_kernel("dpdf_vocab_proj_linear", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let bindings = vocab_projection_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocab projection linear IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear projection should produce symmetric bounds around zero with zero bias
    let width = hi_max - lo_min;
    assert!(
        width.is_finite() && width > 0.0,
        "projection should produce non-trivial bounds, width={width}"
    );
}

// ===========================================================================
// 2. Tied weight embedding projection bounds (IBP)
// ===========================================================================

/// Tied embedding: reuse embedding matrix W^T as the LM head projection.
/// In practice this is equivalent to Linear(hidden, vocab) with W = E^T
/// where E is the token embedding matrix [vocab, hidden].
#[test]
fn test_vocab_tied_embedding_projection_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_tied_embed");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    // Tied weights: same embedding matrix used for projection
    // Shape is [VOCAB_SIZE, HIDDEN_DIM] for both embed lookup and projection
    let embed_w = b.add_input("embed_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Projection: hidden @ embed_w^T = Linear(hidden, vocab) with no bias
    let out = b.add_linear(input, embed_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid tied embedding kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Tied embedding projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Vocabulary projection + softmax composition (IBP + CROWN)
// ===========================================================================

fn build_vocab_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let proj_b = b.add_input("lm_head_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(out).expect("valid vocab softmax kernel")
}

fn vocab_softmax_bindings() -> Vec<TensorParamBinding> {
    vocab_projection_bindings(HIDDEN_DIM, VOCAB_SIZE)
}

#[test]
fn test_vocab_softmax_ibp() {
    let def = build_vocab_softmax_kernel();
    let bindings = vocab_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Vocab projection + softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

#[test]
fn test_vocab_softmax_crown() {
    let def = build_vocab_softmax_kernel();
    let bindings = vocab_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Vocab projection + softmax CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. Vocabulary projection + log-softmax for CTC loss (IBP)
// ===========================================================================

#[test]
fn test_vocab_log_softmax_ctc_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_log_softmax_ctc");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("ctc_weight", &[NUM_CTC_CLASSES, HIDDEN_DIM]);
    let proj_b = b.add_input("ctc_bias", &[NUM_CTC_CLASSES]);

    let logits = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, NUM_CTC_CLASSES]);
    let out = b.add_log_softmax(logits, 1, &[SEQ_LEN, NUM_CTC_CLASSES]);
    let def = b.build(out).expect("valid CTC log-softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CTC_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CTC_CLASSES]), 0.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocab log-softmax CTC IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // log-softmax outputs are <= 0
    let tol = 1e-4;
    assert!(
        hi_max <= 0.0 + tol,
        "log-softmax upper must be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 5. Temperature-scaled logits bounds (IBP)
// ===========================================================================

/// Temperature scaling: logits * (1/T) before softmax.
/// Lower temperature -> sharper softmax distribution; higher -> flatter.
/// Modeled as multiplication by inverse temperature since TensorBlockBuilder
/// does not have a division primitive.
#[test]
fn test_vocab_temperature_scaled_logits_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_temp_scaled");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let inv_temperature = b.add_input("inv_temperature", &[1]);

    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Multiply by 1/T (inverse temperature): logits * (1/T)
    let inv_temp_bc = b.add_broadcast(inv_temperature, &[SEQ_LEN, VOCAB_SIZE]);
    let scaled_logits = b.add_binary_mul(logits, inv_temp_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(scaled_logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid temperature-scaled kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // Inverse temperature = 1/0.7 ≈ 1.4286 (sharper distribution)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32 / 0.7f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Temperature-scaled logits IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 6. Top-k filtering effect on output bounds (IBP)
// ===========================================================================

/// Top-k filtering: project to vocab, then narrow to top-k classes via
/// a second projection. Models the bound-relevant path where only k logits
/// survive masking.
#[test]
fn test_vocab_topk_filtering_ibp() {
    let top_k = 8;
    let mut b = TensorBlockBuilder::new("dpdf_vocab_topk_filter");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Model top-k as a projection that selects k columns
    let topk_w = b.add_input("topk_select", &[top_k, VOCAB_SIZE]);
    let topk_logits = b.add_linear(logits, topk_w, None, &[SEQ_LEN, top_k]);
    let out = b.add_softmax(topk_logits, 1, &[SEQ_LEN, top_k]);
    let def = b.build(out).expect("valid top-k filtering kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // Selection matrix: sparse selector (modeled as small constant)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[top_k, VOCAB_SIZE]),
            1.0 / VOCAB_SIZE as f32,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Top-k filtering IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. Top-p (nucleus) sampling threshold bounds (IBP)
// ===========================================================================

/// Top-p: softmax probabilities are accumulated; tokens above cumulative
/// threshold p are kept. We verify the softmax output itself is bounded.
#[test]
fn test_vocab_topp_nucleus_threshold_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_topp_nucleus");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Softmax gives per-token probabilities; nucleus sampling selects from these
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    // Cumulative sum approximation: project through a lower-triangular-like
    // matrix to accumulate probabilities. Bounded output verifies cumsum.
    let cumsum_w = b.add_input("cumsum_approx", &[VOCAB_SIZE, VOCAB_SIZE]);
    let out = b.add_linear(probs, cumsum_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid nucleus sampling kernel");

    // Build a lower-triangular approximation matrix (all ones below diagonal)
    let mut cumsum_data = vec![0.0f32; VOCAB_SIZE * VOCAB_SIZE];
    for i in 0..VOCAB_SIZE {
        for j in 0..=i {
            cumsum_data[i * VOCAB_SIZE + j] = 1.0;
        }
    }
    let cumsum_mat = ArrayD::from_shape_vec(IxDyn(&[VOCAB_SIZE, VOCAB_SIZE]), cumsum_data).unwrap();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(cumsum_mat),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Top-p nucleus threshold IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "cumulative probability lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "cumulative probability upper must be finite"
    );
}

// ===========================================================================
// 8. Greedy decoding (argmax) output bounds (IBP)
// ===========================================================================

/// Greedy decoding takes the argmax over softmax output. We verify the
/// softmax probability distribution that feeds argmax is properly bounded.
#[test]
fn test_vocab_greedy_argmax_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_greedy_argmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let proj_b = b.add_input("lm_head_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, VOCAB_SIZE]);
    // Softmax produces the probability distribution argmax operates on
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid greedy argmax kernel");

    let bindings = vocab_projection_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Greedy argmax softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output: each element in [0, 1]
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Beam search score accumulation bounds (IBP)
// ===========================================================================

/// Beam search accumulates log-probabilities across timesteps.
/// We model 2 timesteps of log-softmax -> additive accumulation.
#[test]
fn test_vocab_beam_search_accumulation_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_beam_search");
    let input_t1 = b.add_input("hidden_t1", &[BEAM_WIDTH, HIDDEN_DIM]);
    let input_t2 = b.add_input("hidden_t2", &[BEAM_WIDTH, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Timestep 1: log-softmax score
    let logits_t1 = b.add_linear(input_t1, proj_w, None, &[BEAM_WIDTH, VOCAB_SIZE]);
    let log_probs_t1 = b.add_log_softmax(logits_t1, 1, &[BEAM_WIDTH, VOCAB_SIZE]);

    // Timestep 2: log-softmax score
    let logits_t2 = b.add_linear(input_t2, proj_w, None, &[BEAM_WIDTH, VOCAB_SIZE]);
    let log_probs_t2 = b.add_log_softmax(logits_t2, 1, &[BEAM_WIDTH, VOCAB_SIZE]);

    // Accumulate: score = log_prob_t1 + log_prob_t2
    let out = b.add_binary_add(log_probs_t1, log_probs_t2, &[BEAM_WIDTH, VOCAB_SIZE]);
    let def = b.build(out).expect("valid beam search kernel");

    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable, // hidden_t1
        TensorParamBinding::Variable, // hidden_t2
        TensorParamBinding::ConstantTensor(w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Two Variable inputs (hidden_t1, hidden_t2) are stacked into a single flat
    // tensor by `setup_multi_variable_inputs`, so the IBP input must cover BOTH
    // variables: 2 * BEAM_WIDTH * HIDDEN_DIM elements. Providing only one
    // variable's worth makes the second variable's Slice clamp to an empty range.
    let total_flat = 2 * BEAM_WIDTH * HIDDEN_DIM;
    let input = uniform_bounds(&[total_flat], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-4;
    eprintln!("Beam search accumulation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sum of two log-softmax values: each <= 0, so sum <= 0
    assert!(
        hi_max <= 0.0 + tol,
        "accumulated log-probs upper must be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 10. CTC blank token probability bounds (IBP)
// ===========================================================================

/// CTC blank token is typically index 0. Verify that the softmax probability
/// at the blank position is properly bounded in [0, 1].
#[test]
fn test_vocab_ctc_blank_probability_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_ctc_blank");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_weight", &[NUM_CTC_CLASSES, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[NUM_CTC_CLASSES]);

    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &[SEQ_LEN, NUM_CTC_CLASSES]);
    // Softmax over character classes (including blank at index 0)
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_CTC_CLASSES]);
    let def = b.build(out).expect("valid CTC blank kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CTC_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CTC_CLASSES]), 0.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("CTC blank probability IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "CTC blank lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "CTC blank upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. Large vocabulary (151k tokens for Qwen3) projection bounds (IBP)
// ===========================================================================

/// Qwen3-VL uses ~151k vocabulary. We test with LARGE_VOCAB=512 as a
/// structural representative of large vocab projection scaling behavior.
#[test]
fn test_vocab_large_vocabulary_projection_ibp() {
    let def =
        build_vocab_projection_kernel("dpdf_vocab_large_proj", SEQ_LEN, HIDDEN_DIM, LARGE_VOCAB);
    let bindings = vocab_projection_bindings(HIDDEN_DIM, LARGE_VOCAB);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Compare with standard vocab size
    let small_def =
        build_vocab_projection_kernel("dpdf_vocab_small_proj", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let small_bindings = vocab_projection_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let small_graph = tensor_kernel_to_graph(&small_def, &small_bindings).expect("small graph");
    let small_output = small_graph.propagate_ibp(&input).expect("small IBP");
    assert_bounds_valid(&small_output);

    let large_width = bound_width(&output);
    let small_width = bound_width(&small_output);
    eprintln!("Large vocab IBP: width={large_width:.6}, small vocab: width={small_width:.6}");
    assert!(large_width.is_finite(), "large vocab width must be finite");
    assert!(small_width.is_finite(), "small vocab width must be finite");
}

// ===========================================================================
// 12. Multi-head CTC output (character + position) bounds (IBP)
// ===========================================================================

/// Multi-head CTC: separate character and position classification heads
/// sharing the same encoder output.
#[test]
fn test_vocab_multihead_ctc_ibp() {
    let num_chars = 48;
    let num_positions = 16;
    let mut b = TensorBlockBuilder::new("dpdf_vocab_multihead_ctc");
    let input = b.add_input("encoder_out", &[SEQ_LEN, HIDDEN_DIM]);

    // Character head: Linear -> softmax
    let char_w = b.add_input("char_weight", &[num_chars, HIDDEN_DIM]);
    let char_logits = b.add_linear(input, char_w, None, &[SEQ_LEN, num_chars]);
    let char_probs = b.add_softmax(char_logits, 1, &[SEQ_LEN, num_chars]);

    // Position head: Linear -> softmax
    let pos_w = b.add_input("pos_weight", &[num_positions, HIDDEN_DIM]);
    let pos_logits = b.add_linear(input, pos_w, None, &[SEQ_LEN, num_positions]);
    let _pos_probs = b.add_softmax(pos_logits, 1, &[SEQ_LEN, num_positions]);

    // Output is the character probability head (primary output)
    let def = b.build(char_probs).expect("valid multi-head CTC kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_chars, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_positions, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Multi-head CTC IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "char softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "char softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. CROWN tightness for vocabulary projection layer (CROWN)
// ===========================================================================

#[test]
fn test_vocab_projection_crown_tightness() {
    let def =
        build_vocab_projection_kernel("dpdf_vocab_proj_crown", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let bindings = vocab_projection_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vocab projection CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. Logit bias/mask application bounds (IBP)
// ===========================================================================

/// Logit bias: add a per-token bias to logits before softmax.
/// Used for prompt engineering, forced token generation, vocabulary filtering.
#[test]
fn test_vocab_logit_bias_mask_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_logit_bias");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logit_bias = b.add_input("logit_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, proj_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Apply per-token logit bias
    let bias_bc = b.add_broadcast(logit_bias, &[SEQ_LEN, VOCAB_SIZE]);
    let biased_logits = b.add_binary_add(logits, bias_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(biased_logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid logit bias kernel");

    // Logit bias: small positive/negative biases (models token preferences)
    let bias_data = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.1f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(bias_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Logit bias IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 15. End-to-end hidden-to-token probability pipeline (IBP + CROWN)
// ===========================================================================

/// Full pipeline: RMSNorm -> Linear(hidden, vocab) -> softmax.
/// This is the standard LM head pattern in GLM-OCR, Qwen3-VL, Granite-Docling.
fn build_e2e_hidden_to_token_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vocab_e2e_pipeline");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);

    // RMSNorm
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // Vocabulary projection
    let proj_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let proj_b = b.add_input("lm_head_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed, proj_w, Some(proj_b), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax to token probabilities
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    b.build(out).expect("valid e2e hidden-to-token kernel")
}

fn e2e_hidden_to_token_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

#[test]
fn test_vocab_e2e_hidden_to_token_ibp() {
    let def = build_e2e_hidden_to_token_kernel();
    let bindings = e2e_hidden_to_token_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("E2E hidden-to-token IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

#[test]
fn test_vocab_e2e_hidden_to_token_crown() {
    let def = build_e2e_hidden_to_token_kernel();
    let bindings = e2e_hidden_to_token_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("E2E hidden-to-token CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
