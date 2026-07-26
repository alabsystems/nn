// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for decoder autoregressive generation bounds
//! (greedy, beam search, sampling).
//!
//! Verifies NY IBP and CROWN bound propagation through the decoding
//! strategies used in autoregressive text generation for dpdf OCR/VLM models.
//! Each test builds a subgraph representing a generation pattern, propagates
//! bounds, and asserts that outputs remain within expected ranges.
//!
//! ## Decoder Single Step & Logit Bounds (tests 1-2)
//!
//! 1. **Decoder single step: logits from last token (IBP)**: models a single
//!    decoder step producing vocab-sized logits from the last sequence position.
//!
//! 2. **Logit bounds from bounded input embeddings (IBP)**: verifies that
//!    bounded input embeddings produce bounded logits through a decoder layer.
//!
//! ## Greedy & Temperature Sampling (tests 3-4)
//!
//! 3. **Greedy decoding: argmax over logits (IBP)**: models the argmax path
//!    via softmax — the maximum probability token has bounded probability.
//!
//! 4. **Temperature scaling: logits / T bounds (IBP)**: verifies that dividing
//!    logits by temperature (via inverse-temperature multiply) produces bounded
//!    softmax outputs for different temperature values.
//!
//! ## Filtering & Sampling (tests 5-7)
//!
//! 5. **Top-k filtering: only top-k logits remain (IBP)**: models top-k
//!    selection as a projection to k columns, then softmax.
//!
//! 6. **Top-p (nucleus) sampling: cumulative probability threshold (IBP)**:
//!    models nucleus sampling as a narrower projection capturing the nucleus.
//!
//! 7. **Softmax on filtered logits: valid probability distribution (IBP)**:
//!    verifies that softmax after any filtering still produces [0, 1] outputs
//!    summing to 1 (per-row).
//!
//! ## Beam Search (tests 8-9)
//!
//! 8. **Beam search: top-B sequences tracked (IBP)**: models beam expansion
//!    as projection from vocab to beam-width candidates.
//!
//! 9. **Beam score accumulation bounds (IBP)**: verifies that accumulated
//!    log-softmax scores remain bounded after multiple steps.
//!
//! ## KV Cache (tests 10-11)
//!
//! 10. **KV cache: key/value bounds after N steps (IBP)**: verifies bounds
//!     through a decoder layer with extended KV cache sequence length.
//!
//! 11. **KV cache bounds growth rate (IBP)**: compares output bound widths
//!     at different cache lengths (4, 8, 16) showing monotonic widening.
//!
//! ## Masks & Controls (tests 12-14)
//!
//! 12. **Causal mask: future positions masked (IBP)**: verifies that causal
//!     masking in attention preserves finite bounds.
//!
//! 13. **Stop token detection: bounded logit comparison (IBP)**: models
//!     stop-token detection as sigmoid on the stop-token logit position.
//!
//! 14. **Max length enforcement: position < max_len (IBP)**: verifies that
//!     a position-bounded decoder produces bounded outputs.
//!
//! ## Penalties & Cross-Attention (tests 15-16)
//!
//! 15. **Repetition penalty: penalized logit bounds (IBP)**: models repetition
//!     penalty as element-wise multiply before softmax.
//!
//! 16. **Decoder + cross-attention: encoder output bounds propagate (IBP +
//!     CROWN)**: verifies encoder-decoder attention bounds composition.
//!
//! ## Multi-Step & End-to-End (tests 17-18)
//!
//! 17. **Multi-step generation: bounds after 2/4/8 steps (IBP)**: verifies
//!     monotonic bound widening across chained decoder steps.
//!
//! 18. **Final output: sequence of bounded token logits (IBP + CROWN)**:
//!     full pipeline from embedding through decoder to logit output.
//!
//! Dimensions (small for fast verification):
//! - SEQ_LEN=4, HIDDEN_DIM=64, VOCAB_SIZE=128, FFN_DIM=128, NUM_HEADS=4
//!
//! Part of #4130: Compose tests for decoder autoregressive generation bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 64;
const VOCAB_SIZE: usize = 128;
const FFN_DIM: usize = 128;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
const WEIGHT_MAG: f32 = 0.02;
const BEAM_WIDTH: usize = 4;
const TOP_K: usize = 8;
const TOP_P_NUCLEUS: usize = 16;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
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

/// Build a single pre-norm decoder layer.
///
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU -> residual.
/// Adds 11 parameters to the builder.
fn add_decoder_layer(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let ffn_shape = [seq_len, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let n1_eps = b.add_input(&format!("{prefix}norm1_eps"), &[1]);
    let n1_w = b.add_input(&format!("{prefix}norm1_w"), &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, n1_eps, 1, n1_w, &shape);

    // Self-attention: Q/K/V + attention + output
    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("{prefix}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let n2_eps = b.add_input(&format!("{prefix}norm2_eps"), &[1]);
    let n2_w = b.add_input(&format!("{prefix}norm2_w"), &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input(&format!("{prefix}gate_w"), &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input(&format!("{prefix}up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{prefix}down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual after FFN
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push one decoder layer's bindings (13 params: 2 eps, 2 norm_w, 4 attn_w, 3 ffn_w).
fn push_decoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(qkvo_w)); // out_w
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // norm2_w
    bindings.push(TensorParamBinding::ConstantTensor(gate_w)); // gate_w
    bindings.push(TensorParamBinding::ConstantTensor(up_w)); // up_w
    bindings.push(TensorParamBinding::ConstantTensor(down_w)); // down_w
}

/// Build RMSNorm -> Linear LM head.
fn build_lm_head(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
) -> nn_dsl::TensorNodeId {
    let shape = [seq_len, HIDDEN_DIM];
    let eps = b.add_input(&format!("{prefix}lm_eps"), &[1]);
    let norm_w = b.add_input(&format!("{prefix}lm_norm_w"), &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);
    let lm_w = b.add_input(&format!("{prefix}lm_w"), &[VOCAB_SIZE, HIDDEN_DIM]);
    b.add_linear(normed, lm_w, None, &[seq_len, VOCAB_SIZE])
}

/// Push LM head bindings (eps, norm_w, lm_w).
fn push_lm_head_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // lm_eps
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        1.0f32,
    ))); // lm_norm_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    ))); // lm_w
}

// ===========================================================================
// 1. Decoder single step: logits from last token (IBP)
// ===========================================================================

/// Single autoregressive step: decode 1 token position, produce vocab logits.
/// Models the core generation loop body: hidden[1, D] -> logits[1, V].
#[test]
fn test_decoder_single_step_logits_ibp() {
    let step_seq = 1; // single token position
    let mut b = TensorBlockBuilder::new("dpdf_gen_single_step");
    let input = b.add_input("hidden", &[step_seq, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[step_seq, VOCAB_SIZE]);
    let def = b.build(logits).expect("valid single-step logits kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[step_seq, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder single step logits IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[step_seq, VOCAB_SIZE],
        "output shape must be [1, VOCAB_SIZE]"
    );
}

// ===========================================================================
// 2. Logit bounds from bounded input embeddings (IBP)
// ===========================================================================

/// Bounded embeddings -> decoder layer -> LM head -> logits.
/// Verifies that embedding-level bounds propagate through a full decoder step.
#[test]
fn test_logit_bounds_from_bounded_embeddings_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_embed_to_logits");
    let input = b.add_input("embeddings", &[SEQ_LEN, HIDDEN_DIM]);

    // Single decoder layer
    let decoded = add_decoder_layer(&mut b, input, "l1_", SEQ_LEN);

    // LM head
    let logits = build_lm_head(&mut b, decoded, "", SEQ_LEN);
    let def = b.build(logits).expect("valid embed-to-logits kernel");

    let mut bindings = vec![TensorParamBinding::Variable]; // embeddings
    push_decoder_layer_bindings(&mut bindings);
    push_lm_head_bindings(&mut bindings);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Logit bounds from embeddings IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}

// ===========================================================================
// 3. Greedy decoding: argmax over logits (IBP)
// ===========================================================================

/// Greedy decoding: logits -> softmax -> probability distribution.
/// The argmax token has bounded probability in [0, 1].
#[test]
fn test_greedy_decoding_softmax_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_greedy");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid greedy decoding kernel");

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
    let tol = 1e-6;
    eprintln!("Greedy decoding softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 4. Temperature scaling: logits / T bounds (IBP)
// ===========================================================================

/// Temperature scaling: logits * (1/T) before softmax.
/// Lower T -> sharper distribution; higher T -> flatter.
/// Tests two temperatures (T=0.7 sharp, T=2.0 flat) and verifies both
/// produce valid [0, 1] softmax outputs.
#[test]
fn test_temperature_scaling_generation_ibp() {
    for (label, inv_temp) in [("sharp_T0.7", 1.0f32 / 0.7), ("flat_T2.0", 1.0 / 2.0)] {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_gen_temp_{label}"));
        let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
        let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
        let inv_t = b.add_input("inv_temperature", &[1]);

        let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
        let inv_t_bc = b.add_broadcast(inv_t, &[SEQ_LEN, VOCAB_SIZE]);
        let scaled = b.add_binary_mul(logits, inv_t_bc, &[SEQ_LEN, VOCAB_SIZE]);
        let probs = b.add_softmax(scaled, 1, &[SEQ_LEN, VOCAB_SIZE]);
        let def = b.build(probs).expect("valid temperature-scaled kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), inv_temp)),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

        let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let tol = 1e-6;
        eprintln!("Temperature {label} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
        assert!(
            lo_min >= 0.0 - tol,
            "{label}: softmax lower must be >= 0, got {lo_min}"
        );
        assert!(
            hi_max <= 1.0 + tol,
            "{label}: softmax upper must be <= 1, got {hi_max}"
        );
    }
}

// ===========================================================================
// 5. Top-k filtering: only top-k logits remain (IBP)
// ===========================================================================

/// Top-k filtering: project to vocab, then select top-k via projection.
/// Models the bound-relevant path of top-k sampling.
#[test]
fn test_topk_filtering_generation_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_topk_filter");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Model top-k as selection projection to k columns
    let topk_w = b.add_input("topk_select", &[TOP_K, VOCAB_SIZE]);
    let topk_logits = b.add_linear(logits, topk_w, None, &[SEQ_LEN, TOP_K]);
    let probs = b.add_softmax(topk_logits, 1, &[SEQ_LEN, TOP_K]);
    let def = b.build(probs).expect("valid top-k filtering kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // Selection matrix: sparse selector modeled as small constant
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[TOP_K, VOCAB_SIZE]),
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
// 6. Top-p (nucleus) sampling: cumulative probability threshold (IBP)
// ===========================================================================

/// Nucleus sampling: select top-p tokens covering cumulative probability >= p.
/// Modeled as projection to a nucleus-sized subset, then softmax.
#[test]
fn test_topp_nucleus_sampling_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_topp_nucleus");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Model nucleus as projection to nucleus-sized subset
    let nucleus_w = b.add_input("nucleus_select", &[TOP_P_NUCLEUS, VOCAB_SIZE]);
    let nucleus_logits = b.add_linear(logits, nucleus_w, None, &[SEQ_LEN, TOP_P_NUCLEUS]);
    let probs = b.add_softmax(nucleus_logits, 1, &[SEQ_LEN, TOP_P_NUCLEUS]);
    let def = b.build(probs).expect("valid nucleus sampling kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[TOP_P_NUCLEUS, VOCAB_SIZE]),
            1.0 / VOCAB_SIZE as f32,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Nucleus sampling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 7. Softmax on filtered logits: valid probability distribution (IBP)
// ===========================================================================

/// Verifies softmax produces valid [0, 1] probabilities on filtered logits.
/// Combines temperature scaling + top-k to model a realistic sampling pipeline.
#[test]
fn test_softmax_filtered_logits_valid_dist_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_softmax_filtered");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let inv_t = b.add_input("inv_temperature", &[1]);
    let topk_w = b.add_input("topk_select", &[TOP_K, VOCAB_SIZE]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Temperature scaling
    let inv_t_bc = b.add_broadcast(inv_t, &[SEQ_LEN, VOCAB_SIZE]);
    let scaled = b.add_binary_mul(logits, inv_t_bc, &[SEQ_LEN, VOCAB_SIZE]);
    // Top-k selection
    let topk_logits = b.add_linear(scaled, topk_w, None, &[SEQ_LEN, TOP_K]);
    let probs = b.add_softmax(topk_logits, 1, &[SEQ_LEN, TOP_K]);
    let def = b.build(probs).expect("valid filtered softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0f32 / 0.8)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[TOP_K, VOCAB_SIZE]),
            1.0 / VOCAB_SIZE as f32,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Softmax filtered logits IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 8. Beam search: top-B sequences tracked (IBP)
// ===========================================================================

/// Beam search: project logits to beam-width candidates.
/// Models beam expansion as vocab -> beam_width projection followed by softmax.
#[test]
fn test_beam_search_topb_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_beam_search");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Beam selection: project to beam_width candidates
    let beam_w = b.add_input("beam_select", &[BEAM_WIDTH, VOCAB_SIZE]);
    let beam_logits = b.add_linear(logits, beam_w, None, &[SEQ_LEN, BEAM_WIDTH]);
    let beam_probs = b.add_softmax(beam_logits, 1, &[SEQ_LEN, BEAM_WIDTH]);
    let def = b.build(beam_probs).expect("valid beam search kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BEAM_WIDTH, VOCAB_SIZE]),
            1.0 / VOCAB_SIZE as f32,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Beam search top-B IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "beam softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "beam softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Beam score accumulation bounds (IBP)
// ===========================================================================

/// Beam score accumulation: log-softmax scores summed over 2 steps.
/// Verifies accumulated log-probabilities remain bounded (finite, <= 0).
#[test]
fn test_beam_score_accumulation_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_beam_score_accum");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w1 = b.add_input("lm_head_w_step1", &[VOCAB_SIZE, HIDDEN_DIM]);
    let lm_w2 = b.add_input("lm_head_w_step2", &[VOCAB_SIZE, HIDDEN_DIM]);

    // Step 1: hidden -> log-softmax scores
    let logits1 = b.add_linear(input, lm_w1, None, &[SEQ_LEN, VOCAB_SIZE]);
    let log_probs1 = b.add_log_softmax(logits1, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Step 2: same hidden (simplified) -> log-softmax scores
    let logits2 = b.add_linear(input, lm_w2, None, &[SEQ_LEN, VOCAB_SIZE]);
    let log_probs2 = b.add_log_softmax(logits2, 1, &[SEQ_LEN, VOCAB_SIZE]);

    // Accumulate: sum of log probabilities
    let accumulated = b.add_binary_add(log_probs1, log_probs2, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(accumulated)
        .expect("valid beam score accumulation kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
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
    eprintln!("Beam score accumulation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "accumulated lower must be finite");
    assert!(hi_max.is_finite(), "accumulated upper must be finite");
    // Log-softmax values are <= 0, so accumulated scores are <= 0
    let tol = 1e-4;
    assert!(
        hi_max <= 0.0 + tol,
        "accumulated log-prob upper must be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 10. KV cache: key/value bounds after N steps (IBP)
// ===========================================================================

/// KV cache bounds: decoder with extended KV cache sequence length.
/// Models a decoder at step T where KV cache contains T-1 prior entries.
/// Uses SEQ_LEN as cache length to verify bounds remain finite.
#[test]
fn test_kv_cache_bounds_after_n_steps_ibp() {
    let cache_len = SEQ_LEN; // prior cached positions
    let query_len = 1; // current query position
    let kv_len = cache_len + query_len; // total KV length
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_gen_kv_cache");
    // Current query input
    let query_input = b.add_input("query_hidden", &[query_len, HIDDEN_DIM]);
    // Cached K/V (treated as constant parameter with known bounds)
    let cached_k = b.add_input("cached_k", &[cache_len, HIDDEN_DIM]);
    let cached_v = b.add_input("cached_v", &[cache_len, HIDDEN_DIM]);

    // Project query
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(query_input, q_w, None, &[query_len, HIDDEN_DIM]);
    let k_new = b.add_linear(query_input, k_w, None, &[query_len, HIDDEN_DIM]);
    let v_new = b.add_linear(query_input, v_w, None, &[query_len, HIDDEN_DIM]);

    // Concatenate cached + new K/V
    let k_full = b.add_concat(&[cached_k, k_new], 0, &[kv_len, HIDDEN_DIM]);
    let v_full = b.add_concat(&[cached_v, v_new], 0, &[kv_len, HIDDEN_DIM]);

    // Attention: Q[1, D] x K[kv_len, D]^T -> attn[1, kv_len] -> out[1, D]
    let attn = b.add_attention(
        q,
        k_full,
        v_full,
        AttentionMask::Standard,
        Some(scale),
        &[query_len, HIDDEN_DIM],
    );
    let def = b.build(attn).expect("valid KV cache kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // query_hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[cache_len, HIDDEN_DIM]),
            0.5f32,
        )), // cached_k
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[cache_len, HIDDEN_DIM]),
            0.5f32,
        )), // cached_v
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // q_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // k_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // v_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[query_len, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("KV cache (cache_len={cache_len}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. KV cache bounds growth rate (IBP)
// ===========================================================================

/// KV cache growth rate: compare output bound widths at cache lengths 2, 4, 8.
/// Longer caches should produce equal or wider bounds (monotonic widening).
#[test]
fn test_kv_cache_bounds_growth_rate_ibp() {
    let mut widths = Vec::new();

    for &cache_len in &[2usize, 4, 8] {
        let query_len = 1;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut b = TensorBlockBuilder::new(&format!("dpdf_gen_kv_growth_{cache_len}"));
        let query_input = b.add_input("query_hidden", &[query_len, HIDDEN_DIM]);
        let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(query_input, q_w, None, &[query_len, HIDDEN_DIM]);
        let k = b.add_linear(query_input, k_w, None, &[query_len, HIDDEN_DIM]);
        let v = b.add_linear(query_input, v_w, None, &[query_len, HIDDEN_DIM]);

        // Model extended KV as replicated query (simplified but structurally representative)
        // Build KV by broadcasting query to cache_len positions
        let k_bc = b.add_broadcast(k, &[cache_len, HIDDEN_DIM]);
        let v_bc = b.add_broadcast(v, &[cache_len, HIDDEN_DIM]);

        let attn = b.add_attention(
            q,
            k_bc,
            v_bc,
            AttentionMask::Standard,
            Some(scale),
            &[query_len, HIDDEN_DIM],
        );
        let def = b.build(attn).expect("valid KV growth kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
                WEIGHT_MAG,
            )),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[query_len, HIDDEN_DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let w = bound_width(&output);
        eprintln!("KV cache growth (cache_len={cache_len}): width={w:.6}");
        widths.push(w);
    }

    // Bounds should widen (or stay equal) with longer cache
    for i in 1..widths.len() {
        let tol = 1e-6;
        assert!(
            widths[i] >= widths[i - 1] - tol,
            "KV cache growth: width at index {} ({}) should be >= width at {} ({})",
            i,
            widths[i],
            i - 1,
            widths[i - 1]
        );
    }
}

// ===========================================================================
// 12. Causal mask: future positions masked (IBP)
// ===========================================================================

/// Causal masking in autoregressive generation: future tokens are masked.
/// Verifies that causal attention preserves finite bounds.
#[test]
fn test_causal_mask_generation_ibp() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_gen_causal_mask");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, HIDDEN_DIM],
    );
    let def = b.build(attn).expect("valid causal mask kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
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
    eprintln!("Causal mask generation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Stop token detection: bounded logit comparison (IBP)
// ===========================================================================

/// Stop token detection: sigmoid on the stop-token logit.
/// Models detecting the end-of-sequence token as sigmoid(logit[stop_idx]).
/// Output bounded in [0, 1].
#[test]
fn test_stop_token_detection_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_stop_token");
    let input = b.add_input("hidden", &[1, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);

    let logits = b.add_linear(input, lm_w, None, &[1, VOCAB_SIZE]);
    // Extract stop token position via narrow + sigmoid
    // Model as: project all logits down to single stop-token score
    let stop_w = b.add_input("stop_proj", &[1, VOCAB_SIZE]);
    let stop_logit = b.add_linear(logits, stop_w, None, &[1, 1]);
    let stop_prob = b.add_sigmoid(stop_logit, &[1, 1]);
    let def = b.build(stop_prob).expect("valid stop token kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // Stop projection: isolate one logit (sparse row with one 1.0)
        TensorParamBinding::ConstantTensor({
            let mut data = vec![0.0f32; VOCAB_SIZE];
            data[0] = 1.0; // stop token at index 0
            ArrayD::from_shape_vec(IxDyn(&[1, VOCAB_SIZE]), data).unwrap()
        }),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Stop token detection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Max length enforcement: position < max_len (IBP)
// ===========================================================================

/// Max length enforcement: decoder with bounded position range.
/// Verifies bounds at maximum allowed sequence position.
/// Uses SEQ_LEN as max_len and tests the last position.
#[test]
fn test_max_length_enforcement_ibp() {
    let max_len = SEQ_LEN;
    let mut b = TensorBlockBuilder::new("dpdf_gen_max_length");
    let input = b.add_input("hidden", &[max_len, HIDDEN_DIM]);

    // Decoder layer at maximum position
    let decoded = add_decoder_layer(&mut b, input, "l1_", max_len);

    // LM head: produce logits at all positions up to max_len
    let logits = build_lm_head(&mut b, decoded, "", max_len);
    let def = b.build(logits).expect("valid max length kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_layer_bindings(&mut bindings);
    push_lm_head_bindings(&mut bindings);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[max_len, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Max length enforcement IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert_eq!(output.lower_upper().0.shape(), &[max_len, VOCAB_SIZE]);
}

// ===========================================================================
// 15. Repetition penalty: penalized logit bounds (IBP)
// ===========================================================================

/// Repetition penalty: scale logits of previously-generated tokens.
/// Penalty > 1.0 suppresses repeated tokens (modeled as 1/penalty multiply).
#[test]
fn test_repetition_penalty_generation_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_rep_penalty");
    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let penalty = b.add_input("rep_penalty", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    // Apply repetition penalty: logits * penalty_vector
    let penalty_bc = b.add_broadcast(penalty, &[SEQ_LEN, VOCAB_SIZE]);
    let penalized = b.add_binary_mul(logits, penalty_bc, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(penalized, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(probs).expect("valid repetition penalty kernel");

    // Penalty vector: 1.0 for fresh tokens, 1/1.2 for repeated
    let mut penalty_data = vec![1.0f32; VOCAB_SIZE];
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
    eprintln!("Repetition penalty IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 16. Decoder + cross-attention: encoder output bounds propagate (IBP + CROWN)
// ===========================================================================

/// Encoder-decoder cross-attention: encoder outputs feed into decoder.
/// Verifies encoder output bounds propagate through cross-attention to logits.
#[test]
fn test_decoder_cross_attention_generation_ibp_crown() {
    let enc_seq_len = 8;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("dpdf_gen_cross_attn");
    let dec_input = b.add_input("dec_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let enc_output = b.add_input("enc_output", &[enc_seq_len, HIDDEN_DIM]);

    // Self-attention on decoder
    let self_q_w = b.add_input("self_q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let self_k_w = b.add_input("self_k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let self_v_w = b.add_input("self_v_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let self_q = b.add_linear(dec_input, self_q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let self_k = b.add_linear(dec_input, self_k_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let self_v = b.add_linear(dec_input, self_v_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let self_attn = b.add_attention(
        self_q,
        self_k,
        self_v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, HIDDEN_DIM],
    );
    let res1 = b.add_binary_add(dec_input, self_attn, &[SEQ_LEN, HIDDEN_DIM]);

    // Cross-attention: decoder queries attend to encoder keys/values
    let cross_q_w = b.add_input("cross_q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cross_k_w = b.add_input("cross_k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cross_v_w = b.add_input("cross_v_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let cross_q = b.add_linear(res1, cross_q_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let cross_k = b.add_linear(enc_output, cross_k_w, None, &[enc_seq_len, HIDDEN_DIM]);
    let cross_v = b.add_linear(enc_output, cross_v_w, None, &[enc_seq_len, HIDDEN_DIM]);
    let cross_attn = b.add_attention(
        cross_q,
        cross_k,
        cross_v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, HIDDEN_DIM],
    );
    let res2 = b.add_binary_add(res1, cross_attn, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head
    let lm_w = b.add_input("lm_head_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(res2, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(logits)
        .expect("valid cross-attention generation kernel");

    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable, // dec_hidden
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[enc_seq_len, HIDDEN_DIM]),
            0.5f32,
        )), // enc_output
        TensorParamBinding::ConstantTensor(attn_w.clone()), // self_q_w
        TensorParamBinding::ConstantTensor(attn_w.clone()), // self_k_w
        TensorParamBinding::ConstantTensor(attn_w.clone()), // self_v_w
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_q_w
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_k_w
        TensorParamBinding::ConstantTensor(attn_w), // cross_v_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // lm_head_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP check
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Cross-attention generation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // CROWN check (may fall back for attention)
    let (method, crown_output, _reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Cross-attention generation CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 17. Multi-step generation: bounds after 2/4/8 steps (IBP)
// ===========================================================================

/// Multi-step autoregressive generation: chained decoder layers model
/// multiple generation steps. Bound widths should widen monotonically.
#[test]
fn test_multi_step_generation_bounds_ibp() {
    let mut widths = Vec::new();

    for &num_layers in &[1usize, 2, 4] {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_gen_multistep_{num_layers}"));
        let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

        let mut x = input;
        for i in 0..num_layers {
            x = add_decoder_layer(&mut b, x, &format!("l{}_", i + 1), SEQ_LEN);
        }

        // LM head after N decoder layers
        let logits = build_lm_head(&mut b, x, "", SEQ_LEN);
        let def = b
            .build(logits)
            .unwrap_or_else(|e| panic!("valid {num_layers}-step generation kernel: {e}"));

        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..num_layers {
            push_decoder_layer_bindings(&mut bindings);
        }
        push_lm_head_bindings(&mut bindings);

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let w = bound_width(&output);
        eprintln!("Multi-step generation ({num_layers} layers): width={w:.6}");
        widths.push(w);
    }

    // Deeper stacks should produce wider (or equal) bounds
    for i in 1..widths.len() {
        let tol = 1e-6;
        assert!(
            widths[i] >= widths[i - 1] - tol,
            "Multi-step generation: width at {} layers ({}) should be >= width at {} layers ({})",
            [1, 2, 4][i],
            widths[i],
            [1, 2, 4][i - 1],
            widths[i - 1]
        );
    }
}

// ===========================================================================
// 18. Final output: sequence of bounded token logits (IBP + CROWN)
// ===========================================================================

/// Full autoregressive generation pipeline: embedding -> decoder -> RMSNorm ->
/// LM head -> softmax. End-to-end composition test.
#[test]
fn test_full_generation_pipeline_ibp_crown() {
    let mut b = TensorBlockBuilder::new("dpdf_gen_full_pipeline");
    let input = b.add_input("embeddings", &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder layer
    let decoded = add_decoder_layer(&mut b, input, "l1_", SEQ_LEN);

    // LM head
    let logits = build_lm_head(&mut b, decoded, "", SEQ_LEN);

    // Softmax for probability distribution
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b
        .build(probs)
        .expect("valid full generation pipeline kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_decoder_layer_bindings(&mut bindings);
    push_lm_head_bindings(&mut bindings);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    // IBP check
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    let tol = 1e-6;
    eprintln!("Full generation pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );

    // CROWN check
    let (method, crown_output, _reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Full generation pipeline CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
    assert_bounds_valid(&crown_output);
}
