// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for language model head (LM head) bounds in decoder-based
//! OCR/VLM models.
//!
//! Verifies IBP and CROWN bound propagation through LM head patterns used
//! in all dpdf text-generating models: GLM-OCR, Qwen3-VL, Granite-Docling,
//! FireRed-OCR. The LM head is the final stage that maps decoder hidden
//! states to vocabulary logits/probabilities for autoregressive generation.
//!
//! 1.  **RMSNorm before LM head projection** (IBP)
//! 2.  **Linear projection hidden_dim -> vocab_size** (IBP)
//! 3.  **RMSNorm + Linear composition** (IBP + CROWN)
//! 4.  **Softmax output in [0, 1] after LM head** (IBP)
//! 5.  **Log-softmax output <= 0 after LM head** (IBP)
//! 6.  **Temperature scaling effect on output bounds** (IBP)
//! 7.  **Top-k logit masking effect on bounds** (IBP)
//! 8.  **Repetition penalty application bounds** (IBP)
//! 9.  **LM head with tied embeddings (weight sharing)** (IBP)
//! 10. **Multi-token prediction: 2-step LM head chain** (IBP)
//! 11. **LM head numerical stability (large logits)** (IBP)
//! 12. **CROWN tightness for RMSNorm + LM head** (CROWN)
//! 13. **LM head monotone tightening: smaller eps -> tighter bounds** (IBP)
//! 14. **LM head with different vocab sizes (32k, 64k, 151k)** (IBP)
//! 15. **Full decoder -> RMSNorm -> LM head -> softmax pipeline** (IBP + CROWN)
//!
//! Architecture references:
//! - GLM-4V (THUDM): RMSNorm -> Linear(hidden, vocab) -> softmax
//! - Qwen3-VL (Alibaba): 151k vocab, tied embeddings, RMSNorm LM head
//! - Granite-Docling: Granite LLM decoder with RMSNorm LM head
//! - Llama (Touvron et al., 2023): RMSNorm -> Linear LM head
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, VOCAB_SIZE=128
//!
//! Part of #4040: Compose tests for language model head bounds.

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
const VOCAB_SIZE: usize = 128;
const FFN_DIM: usize = 128;
const WEIGHT_MAG: f32 = 0.02;
const NUM_HEADS: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build RMSNorm -> Linear LM head.
///
/// Standard LM head pattern: RMSNorm(x) -> Linear(hidden_dim, vocab_size).
/// Input shape: `[seq_len, hidden_dim]`.
/// Output shape: `[seq_len, vocab_size]`.
fn build_rmsnorm_lm_head_kernel(
    name: &str,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("hidden", &[seq_len, hidden_dim]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[hidden_dim]);

    // RMSNorm
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[seq_len, hidden_dim]);

    // LM head linear projection
    let lm_w = b.add_input("lm_head_weight", &[vocab_size, hidden_dim]);
    let out = b.add_linear(normed, lm_w, None, &[seq_len, vocab_size]);
    b.build(out).expect("valid RMSNorm + LM head kernel")
}

/// Standard bindings for RMSNorm + LM head (Variable input + constant params).
fn rmsnorm_lm_head_bindings(hidden_dim: usize, vocab_size: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[hidden_dim]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[vocab_size, hidden_dim]),
            WEIGHT_MAG,
        )),
    ]
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

/// Build a standard SwiGLU FFN block for decoder layer tests.
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

// ===========================================================================
// 1. RMSNorm before LM head projection (IBP)
// ===========================================================================

#[test]
fn test_lm_head_rmsnorm_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_rmsnorm");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid RMSNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LM head RMSNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Linear projection hidden_dim -> vocab_size (IBP)
// ===========================================================================

#[test]
fn test_lm_head_linear_projection_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_linear_proj");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let out = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid linear projection kernel");

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
    eprintln!("LM head linear projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with zero-mean weights: output should be symmetric around 0
    let width = hi_max - lo_min;
    assert!(
        width.is_finite() && width > 0.0,
        "projection should produce non-trivial bounds, width={width}"
    );
}

// ===========================================================================
// 3. RMSNorm + Linear composition (IBP + CROWN)
// ===========================================================================

#[test]
fn test_lm_head_rmsnorm_linear_ibp() {
    let def = build_rmsnorm_lm_head_kernel(
        "dpdf_lm_head_rmsnorm_linear",
        SEQ_LEN,
        HIDDEN_DIM,
        VOCAB_SIZE,
    );
    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("RMSNorm + LM head linear IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_lm_head_rmsnorm_linear_crown() {
    let def = build_rmsnorm_lm_head_kernel(
        "dpdf_lm_head_rmsnorm_linear_crown",
        SEQ_LEN,
        HIDDEN_DIM,
        VOCAB_SIZE,
    );
    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("RMSNorm + LM head linear CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. Softmax output in [0, 1] after LM head (IBP)
// ===========================================================================

#[test]
fn test_lm_head_softmax_bounded_01_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid LM head softmax kernel");

    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    // rmsnorm_lm_head_bindings already has the lm_w binding
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("LM head softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 5. Log-softmax output <= 0 after LM head (IBP)
// ===========================================================================

#[test]
fn test_lm_head_log_softmax_nonpositive_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_log_softmax");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_log_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid LM head log-softmax kernel");

    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-4;
    eprintln!("LM head log-softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // log-softmax outputs are <= 0 (log of probability in [0,1])
    assert!(
        hi_max <= 0.0 + tol,
        "log-softmax upper must be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 6. Temperature scaling effect on output bounds (IBP)
// ===========================================================================

/// Temperature scaling: logits / T before softmax.
/// Modeled as multiplication by inverse temperature (1/T).
/// Lower T -> sharper distribution; higher T -> flatter.
#[test]
fn test_lm_head_temperature_scaling_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_temp_scale");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let inv_temp = b.add_input("inv_temperature", &[1]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let inv_temp_bc = b.add_broadcast(inv_temp, &[SEQ_LEN, VOCAB_SIZE]);
    let scaled_logits = b.add_binary_mul(logits, inv_temp_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(scaled_logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(out)
        .expect("valid temperature-scaled LM head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // Inverse temperature = 1/0.7 (sharper distribution)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32 / 0.7f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("LM head temperature scaling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 7. Top-k logit masking effect on bounds (IBP)
// ===========================================================================

/// Top-k filtering: project to vocab, then narrow to top-k classes via
/// a selection projection. Models the bound-relevant path.
#[test]
fn test_lm_head_topk_masking_ibp() {
    let top_k = 8;
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_topk_mask");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Model top-k as a selection projection to k columns
    let topk_w = b.add_input("topk_select", &[top_k, VOCAB_SIZE]);
    let topk_logits = b.add_linear(logits, topk_w, None, &[SEQ_LEN, top_k]);
    let out = b.add_softmax(topk_logits, 1, &[SEQ_LEN, top_k]);
    let def = b.build(out).expect("valid top-k LM head kernel");

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
    eprintln!("LM head top-k masking IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 8. Repetition penalty application bounds (IBP)
// ===========================================================================

/// Repetition penalty: scale logits of previously-generated tokens.
/// Modeled as element-wise multiply by a penalty vector before softmax.
/// penalty > 1.0 suppresses repeated tokens.
#[test]
fn test_lm_head_repetition_penalty_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_rep_penalty");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let penalty = b.add_input("rep_penalty", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Apply repetition penalty: logits * penalty_vector
    // penalty[i] = 1.0 for fresh tokens, 1/1.2 for repeated tokens
    let penalty_bc = b.add_broadcast(penalty, &[SEQ_LEN, VOCAB_SIZE]);
    let penalized = b.add_binary_mul(logits, penalty_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(penalized, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid repetition penalty kernel");

    // Penalty vector: mix of 1.0 (no penalty) and 1/1.2 (penalized)
    let mut penalty_data = vec![1.0f32; VOCAB_SIZE];
    // Penalize first 16 tokens (simulate previously generated)
    for p in penalty_data.iter_mut().take(16) {
        *p = 1.0 / 1.2;
    }
    let penalty_arr = ArrayD::from_shape_vec(IxDyn(&[VOCAB_SIZE]), penalty_data).unwrap();

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(penalty_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("LM head repetition penalty IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 9. LM head with tied embeddings (weight sharing) (IBP)
// ===========================================================================

/// Tied embedding: reuse embedding matrix E as the LM head projection.
/// Linear(hidden, vocab) with W = E^T where E is [vocab, hidden].
/// No separate LM head weight -- reduces parameter count.
#[test]
fn test_lm_head_tied_embeddings_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_tied_embed");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);

    // RMSNorm before projection
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &[SEQ_LEN, HIDDEN_DIM]);

    // Tied weight: embedding matrix reused as LM head
    let embed_w = b.add_input("embed_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let out = b.add_linear(normed, embed_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid tied embedding LM head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        // Tied embedding weights (slightly larger magnitude than random init)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            0.03f32,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LM head tied embeddings IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Multi-token prediction: 2-step LM head chain (IBP)
// ===========================================================================

/// Multi-token prediction (MTP): chain two LM heads to predict 2 tokens.
/// Each step: RMSNorm -> Linear -> softmax.
/// The output of step 1 feeds (via a projection) into step 2.
#[test]
fn test_lm_head_multi_token_2step_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_mtp_2step");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Step 1: RMSNorm -> LM head
    let eps1 = b.add_input("norm_eps_1", &[1]);
    let norm_w1 = b.add_input("norm_weight_1", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, eps1, 1, norm_w1, &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w1 = b.add_input("lm_head_weight_1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits1 = b.add_linear(normed1, lm_w1, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Project logits back to hidden space for step 2
    let proj_w = b.add_input("mtp_proj_weight", &[HIDDEN_DIM, VOCAB_SIZE]);
    let hidden2 = b.add_linear(logits1, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Step 2: RMSNorm -> LM head -> softmax
    let eps2 = b.add_input("norm_eps_2", &[1]);
    let norm_w2 = b.add_input("norm_weight_2", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(hidden2, eps2, 1, norm_w2, &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w2 = b.add_input("lm_head_weight_2", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits2 = b.add_linear(normed2, lm_w2, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits2, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid MTP 2-step kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        // Step 1 norm
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        // Step 1 LM head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // MTP projection
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, VOCAB_SIZE]),
            WEIGHT_MAG,
        )),
        // Step 2 norm
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        // Step 2 LM head
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
    let tol = 1e-6;
    eprintln!("LM head MTP 2-step IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "final softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "final softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. LM head numerical stability (large logits) (IBP)
// ===========================================================================

/// Verify LM head produces finite bounds even with large input range.
/// RMSNorm normalizes the input, preventing explosion in the projection.
#[test]
fn test_lm_head_numerical_stability_large_input_ibp() {
    let def =
        build_rmsnorm_lm_head_kernel("dpdf_lm_head_large_input", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Large input range: [-10, 10]
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 10.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LM head large input IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "lower bound must be finite for large inputs"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite for large inputs"
    );

    let width = hi_max - lo_min;
    assert!(
        width.is_finite(),
        "bound width must be finite even for large inputs"
    );
}

// ===========================================================================
// 12. CROWN tightness for RMSNorm + LM head (CROWN)
// ===========================================================================

#[test]
fn test_lm_head_crown_tightness() {
    let def = build_rmsnorm_lm_head_kernel(
        "dpdf_lm_head_crown_tightness",
        SEQ_LEN,
        HIDDEN_DIM,
        VOCAB_SIZE,
    );
    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);
    let ibp_width = bound_width(&ibp_output);

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&crown_output);
    let crown_width = bound_width(&crown_output);

    eprintln!(
        "LM head CROWN tightness: method={method:?}, ibp_width={ibp_width:.6}, crown_width={crown_width:.6}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. LM head monotone tightening: smaller eps -> tighter bounds (IBP)
// ===========================================================================

#[test]
fn test_lm_head_monotone_tightening_ibp() {
    let def =
        build_rmsnorm_lm_head_kernel("dpdf_lm_head_monotone", SEQ_LEN, HIDDEN_DIM, VOCAB_SIZE);
    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, VOCAB_SIZE);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!("LM head monotone: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}");
    assert!(
        tight_width <= wide_width + 1e-6,
        "tight input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 14. LM head with different vocab sizes (32k, 64k, 151k) (IBP)
// ===========================================================================

fn test_lm_head_at_vocab_size(vocab_size: usize, label: &str) {
    let def = build_rmsnorm_lm_head_kernel(
        &format!("dpdf_lm_head_vocab_{label}"),
        SEQ_LEN,
        HIDDEN_DIM,
        vocab_size,
    );
    let bindings = rmsnorm_lm_head_bindings(HIDDEN_DIM, vocab_size);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("LM head vocab_size={vocab_size} ({label}) IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

/// 32k vocabulary (small LLM / OCR model).
/// Uses 256 as structural representative to keep verification fast.
#[test]
fn test_lm_head_vocab_32k() {
    test_lm_head_at_vocab_size(256, "32k");
}

/// 64k vocabulary (medium LLM).
/// Uses 384 as structural representative.
#[test]
fn test_lm_head_vocab_64k() {
    test_lm_head_at_vocab_size(384, "64k");
}

/// 151k vocabulary (Qwen3-VL scale).
/// Uses 512 as structural representative.
#[test]
fn test_lm_head_vocab_151k() {
    test_lm_head_at_vocab_size(512, "151k");
}

// ===========================================================================
// 15. Full decoder -> RMSNorm -> LM head -> softmax pipeline (IBP + CROWN)
// ===========================================================================

/// Full decoder pipeline: MHA -> residual -> RMSNorm -> SwiGLU -> residual
/// -> RMSNorm -> Linear -> softmax.
/// This is the standard decoder-to-output pattern in GLM-OCR, Qwen3-VL.
fn build_full_decoder_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_lm_head_full_decoder");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Attention block: Q, K, V, Out projections
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

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // RMSNorm before FFN
    let ffn_eps = b.add_input("ffn_norm_eps", &[1]);
    let ffn_norm_w = b.add_input("ffn_norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(h, ffn_eps, 1, ffn_norm_w, &shape);

    // SwiGLU FFN
    let ffn_out = build_swiglu_block(&mut b, normed, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual
    let h2 = b.add_binary_add(h, ffn_out, &shape);

    // Final RMSNorm -> LM head -> softmax
    let lm_eps = b.add_input("lm_norm_eps", &[1]);
    let lm_norm_w = b.add_input("lm_norm_w", &[HIDDEN_DIM]);
    let lm_normed = b.add_rms_norm(h2, lm_eps, 1, lm_norm_w, &shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(lm_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid full decoder LM head kernel")
}

fn full_decoder_lm_head_bindings() -> Vec<TensorParamBinding> {
    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };
    let mut bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        // FFN norm
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    // SwiGLU weights
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // LM head norm
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1e-5f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        1.0f32,
    )));
    // LM head weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings
}

#[test]
fn test_lm_head_full_decoder_pipeline_ibp() {
    let def = build_full_decoder_lm_head_kernel();
    let bindings = full_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Full decoder -> LM head pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
fn test_lm_head_full_decoder_pipeline_crown() {
    let def = build_full_decoder_lm_head_kernel();
    let bindings = full_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Full decoder -> LM head pipeline CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
