// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DETR encoder block NY composition.
//!
//! Verifies bounds propagation through a DETR encoder block, which is
//! structurally identical to a standard ViT/pre-norm transformer encoder
//! block but applied to flattened CNN feature map tokens rather than
//! image patches.
//!
//! Architecture (Carion et al. 2020):
//!   x -> LayerNorm -> MHA(bidirectional) -> + residual
//!     -> LayerNorm -> FFN(Linear -> ReLU -> Linear) -> + residual
//!
//! DETR uses ReLU in FFN (not GELU like ViT), so we build this manually
//! rather than using `add_transformer_block` (which uses GELU).
//!
//! Two configurations: small (d=64, h=4) and medium (d=128, h=8).
//!
//! Part of #3556: DETR object detection compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Small configuration: d=64, heads=4, seq=16 (flattened feature map tokens)
// ===========================================================================

mod small {
    pub(super) const EMBED_DIM: usize = 64;
    pub(super) const NUM_HEADS: usize = 4;
    pub(super) const FFN_DIM: usize = 128;
    /// Flattened spatial tokens from CNN backbone (e.g., 4x4 feature map).
    pub(super) const SEQ_LEN: usize = 16;
}

// ===========================================================================
// Medium configuration: d=128, heads=8, seq=32
// ===========================================================================

mod medium {
    pub(super) const EMBED_DIM: usize = 128;
    pub(super) const NUM_HEADS: usize = 8;
    pub(super) const FFN_DIM: usize = 256;
    pub(super) const SEQ_LEN: usize = 32;
}

/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a DETR encoder block: LN -> MHA -> residual -> LN -> FFN(ReLU) -> residual.
///
/// Input: `[seq_len, embed_dim]` (Variable -- flattened feature map tokens).
/// Output: `[seq_len, embed_dim]`.
///
/// DETR encoder uses ReLU (not GELU) in the FFN sub-block. Bidirectional
/// self-attention (no causal mask) since encoder tokens attend to all spatial
/// positions.
fn build_detr_encoder_block_kernel(
    name: &str,
    seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
    ffn_dim: usize,
) -> TensorKernelDef {
    let d = embed_dim;
    let mut b = TensorBlockBuilder::new(name);

    // Inputs
    let input = b.add_input("x", &[seq_len, d]);
    let eps = b.add_input("eps", &[1]);

    // Self-attention weights
    let ln1_w = b.add_input("ln1_weight", &[d]);
    let ln1_b = b.add_input("ln1_bias", &[d]);
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    // FFN weights
    let ln2_w = b.add_input("ln2_weight", &[d]);
    let ln2_b = b.add_input("ln2_bias", &[d]);
    let ffn1_w = b.add_input("ffn1_weight", &[ffn_dim, d]);
    let ffn2_w = b.add_input("ffn2_weight", &[d, ffn_dim]);

    let shape = [seq_len, d];
    let ffn_shape = [seq_len, ffn_dim];

    // --- Sub-block 1: Self-attention ---
    let normed1 = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);
    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            num_heads,
            AttentionMask::Standard, // bidirectional
            &shape,
        )
        .expect("valid self-attention");
    let residual1 = b.add_binary_add(input, attn, &shape);

    // --- Sub-block 2: FFN with ReLU ---
    let normed2 = b.add_layer_norm(residual1, eps, 1, ln2_w, ln2_b, &shape);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_relu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual1, ffn2, &shape);

    b.build(out).expect("valid DETR encoder block kernel")
}

/// Bindings for a DETR encoder block.
fn encoder_block_bindings(embed_dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    let d = embed_dim;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[ffn_dim, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, ffn_dim]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // x [S, D]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // Self-attention LN + projections
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ln1_weight [D]
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ln1_bias [D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj),       // out_weight [D, D]
        // FFN LN + weights
        TensorParamBinding::ConstantTensor(ln_w), // ln2_weight [D]
        TensorParamBinding::ConstantTensor(ln_b), // ln2_bias [D]
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight [FFN, D]
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight [D, FFN]
    ]
}

// ===========================================================================
// Tests: Small configuration (d=64, heads=4, seq=16)
// ===========================================================================

/// DETR encoder block TensorKernelDef validates (small).
#[test]
fn test_detr_encoder_block_small_def_validates() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_small",
        small::SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    def.validate()
        .expect("DETR encoder block (small) should validate");
}

/// DETR encoder block graph builds with sufficient depth (small).
#[test]
fn test_detr_encoder_block_small_graph_builds() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_small_graph",
        small::SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = encoder_block_bindings(small::EMBED_DIM, small::FFN_DIM);
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("encoder block graph should translate");

    // LN + MHA + residual + LN + FFN(2 linears + ReLU) + residual = many nodes
    assert!(
        graph.num_nodes() >= 10,
        "DETR encoder block should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through DETR encoder block (small).
#[test]
fn test_detr_encoder_block_small_ibp_propagates() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_small_ibp",
        small::SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = encoder_block_bindings(small::EMBED_DIM, small::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::SEQ_LEN, small::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR encoder block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::SEQ_LEN, small::EMBED_DIM],
        "encoder block output shape must be [SEQ_LEN, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR encoder block (small) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through DETR encoder block (small).
///
/// LayerNorm requires heuristic CROWN linearization (IbpValidated mode).
#[test]
fn test_detr_encoder_block_small_crown_propagation() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_small_crown",
        small::SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = encoder_block_bindings(small::EMBED_DIM, small::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::SEQ_LEN, small::EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[small::SEQ_LEN, small::EMBED_DIM],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR encoder block (small): method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record DETR encoder block (small) under status key.
#[test]
fn test_detr_encoder_block_small_verify_and_record() {
    let def = build_detr_encoder_block_kernel(
        "detr_encoder_block_small",
        small::SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = encoder_block_bindings(small::EMBED_DIM, small::FFN_DIM);
    let input = uniform_bounds(&[small::SEQ_LEN, small::EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_encoder_block_small");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[small::SEQ_LEN, small::EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "DETR encoder block with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: Medium configuration (d=128, heads=8, seq=32)
// ===========================================================================

/// DETR encoder block validates (medium: d=128, heads=8).
#[test]
fn test_detr_encoder_block_medium_def_validates() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_medium",
        medium::SEQ_LEN,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
        medium::FFN_DIM,
    );
    def.validate()
        .expect("DETR encoder block (medium) should validate");
}

/// IBP bounds propagate through DETR encoder block (medium).
#[test]
fn test_detr_encoder_block_medium_ibp_propagates() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_medium_ibp",
        medium::SEQ_LEN,
        medium::EMBED_DIM,
        medium::NUM_HEADS,
        medium::FFN_DIM,
    );
    let bindings = encoder_block_bindings(medium::EMBED_DIM, medium::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[medium::SEQ_LEN, medium::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR encoder block (medium)");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[medium::SEQ_LEN, medium::EMBED_DIM],
        "encoder block output shape must be [SEQ_LEN, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR encoder block (medium) IBP: bounds=[{lo_min}, {hi_max}]");
}

/// IBP bounds width stays reasonable for DETR encoder block (small).
///
/// With small weights (0.02) and [-1, 1] input, bounds should not blow up.
///
/// The LayerNorm → FFN Linear L2/Cauchy–Schwarz lever (ny) pulls the plain-IBP max
/// width from ~445 (decorrelated box, `‖w‖₁·√n`) down to ~57.7 (exact CS row bound,
/// `‖w‖₂·√n`), clearing the <200 target in a single sound IBP pass (~0.1s). The
/// lever's nominal is now O(out + in) per Linear (box-midpoint identity), so it is
/// cheap enough to run by default. Intersection only tightens; the threshold is NOT
/// weakened.
#[test]
fn test_detr_encoder_block_small_bounds_width() {
    let def = build_detr_encoder_block_kernel(
        "detr_enc_block_small_width",
        small::SEQ_LEN,
        small::EMBED_DIM,
        small::NUM_HEADS,
        small::FFN_DIM,
    );
    let bindings = encoder_block_bindings(small::EMBED_DIM, small::FFN_DIM);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[small::SEQ_LEN, small::EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR encoder block");
    let (lo, hi) = output.lower_upper();

    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| (u - l).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_width < 200.0,
        "DETR encoder block IBP bounds max width {max_width} should be < 200.0"
    );
}
