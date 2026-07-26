// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ViT encoder block NY composition.
//!
//! Verifies bounds propagation through the full ViT encoder block:
//!   x -> LayerNorm -> MHA -> + residual -> LayerNorm -> FFN (Linear->GELU->Linear) -> + residual
//!
//! Uses `TensorBlockBuilder::add_transformer_block()` which already models this
//! architecture. ViT uses `AttentionMask::Standard` (bidirectional, not causal).
//!
//! Dimensions: embed_dim=32, num_heads=4, seq_len=16, ffn_dim=64.
//! These are small for fast verification but structurally representative.
//!
//! Part of #3527: ViT encoder NY compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 16;
const EMBED_DIM: usize = 32;
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension: 2x embed_dim (smaller than typical 4x for speed).
const FFN_DIM: usize = 64;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a single ViT encoder block kernel using `add_transformer_block`.
///
/// This is structurally identical to a pre-norm transformer encoder block
/// with standard (bidirectional) attention — exactly what VitEncoderBlock
/// implements.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
fn build_vit_encoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_encoder_block");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[EMBED_DIM]);
    let ln1_b = b.add_input("ln1_bias", &[EMBED_DIM]);
    let ln2_w = b.add_input("ln2_weight", &[EMBED_DIM]);
    let ln2_b = b.add_input("ln2_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard, // ViT uses bidirectional attention
        ffn_hidden_dim: FFN_DIM,
    };

    let out = b
        .add_transformer_block(input, &weights, &config)
        .expect("valid ViT encoder block");
    b.build(out).expect("valid kernel")
}

/// Build a 2-block ViT encoder stack: block1 -> block2.
///
/// Tests composition of multiple encoder layers, matching the stacked
/// architecture of VitEncoder with `num_layers >= 2`.
fn build_vit_two_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_two_block_encoder");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Block 1 weights
    let b1_ln1_w = b.add_input("b1_ln1_w", &[EMBED_DIM]);
    let b1_ln1_b = b.add_input("b1_ln1_b", &[EMBED_DIM]);
    let b1_ln2_w = b.add_input("b1_ln2_w", &[EMBED_DIM]);
    let b1_ln2_b = b.add_input("b1_ln2_b", &[EMBED_DIM]);
    let b1_q_w = b.add_input("b1_q_w", &[EMBED_DIM, EMBED_DIM]);
    let b1_k_w = b.add_input("b1_k_w", &[EMBED_DIM, EMBED_DIM]);
    let b1_v_w = b.add_input("b1_v_w", &[EMBED_DIM, EMBED_DIM]);
    let b1_out_w = b.add_input("b1_out_w", &[EMBED_DIM, EMBED_DIM]);
    let b1_ffn1_w = b.add_input("b1_ffn1_w", &[FFN_DIM, EMBED_DIM]);
    let b1_ffn2_w = b.add_input("b1_ffn2_w", &[EMBED_DIM, FFN_DIM]);

    // Block 2 weights
    let b2_ln1_w = b.add_input("b2_ln1_w", &[EMBED_DIM]);
    let b2_ln1_b = b.add_input("b2_ln1_b", &[EMBED_DIM]);
    let b2_ln2_w = b.add_input("b2_ln2_w", &[EMBED_DIM]);
    let b2_ln2_b = b.add_input("b2_ln2_b", &[EMBED_DIM]);
    let b2_q_w = b.add_input("b2_q_w", &[EMBED_DIM, EMBED_DIM]);
    let b2_k_w = b.add_input("b2_k_w", &[EMBED_DIM, EMBED_DIM]);
    let b2_v_w = b.add_input("b2_v_w", &[EMBED_DIM, EMBED_DIM]);
    let b2_out_w = b.add_input("b2_out_w", &[EMBED_DIM, EMBED_DIM]);
    let b2_ffn1_w = b.add_input("b2_ffn1_w", &[FFN_DIM, EMBED_DIM]);
    let b2_ffn2_w = b.add_input("b2_ffn2_w", &[EMBED_DIM, FFN_DIM]);

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

    let weights1 = TransformerBlockWeights {
        ln1_weight: b1_ln1_w,
        ln1_bias: b1_ln1_b,
        ln2_weight: b1_ln2_w,
        ln2_bias: b1_ln2_b,
        q_weight: b1_q_w,
        k_weight: b1_k_w,
        v_weight: b1_v_w,
        out_weight: b1_out_w,
        ffn1_weight: b1_ffn1_w,
        ffn2_weight: b1_ffn2_w,
        eps,
    };

    let block1 = b
        .add_transformer_block(input, &weights1, &config)
        .expect("block 1");

    let weights2 = TransformerBlockWeights {
        ln1_weight: b2_ln1_w,
        ln1_bias: b2_ln1_b,
        ln2_weight: b2_ln2_w,
        ln2_bias: b2_ln2_b,
        q_weight: b2_q_w,
        k_weight: b2_k_w,
        v_weight: b2_v_w,
        out_weight: b2_out_w,
        ffn1_weight: b2_ffn1_w,
        ffn2_weight: b2_ffn2_w,
        eps, // shared eps
    };

    let block2 = b
        .add_transformer_block(block1, &weights2, &config)
        .expect("block 2");

    b.build(block2).expect("valid two-block kernel")
}

/// Bindings for a single ViT encoder block.
///
/// Input is Variable, all 11 weight/bias/eps parameters are ConstantTensor.
fn vit_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // input [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5), // eps [1]
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ln1_weight [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ln1_bias [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_w), // ln2_weight [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_b), // ln2_bias [EMBED_DIM]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj), // out_weight [D, D]
        TensorParamBinding::ConstantTensor(w_ffn1), // ffn1_weight [FFN, D]
        TensorParamBinding::ConstantTensor(w_ffn2), // ffn2_weight [D, FFN]
    ]
}

/// Bindings for a 2-block ViT encoder stack.
///
/// Input is Variable, eps is shared, all 20 weight/bias parameters are ConstantTensor.
fn vit_two_block_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // input
        TensorParamBinding::ConstantScalar(1e-5), // eps (shared)
        // Block 1 weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b1_ln1_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b1_ln1_b
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b1_ln2_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b1_ln2_b
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_q_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_k_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_v_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_out_w
        TensorParamBinding::ConstantTensor(w_ffn1.clone()), // b1_ffn1_w
        TensorParamBinding::ConstantTensor(w_ffn2.clone()), // b1_ffn2_w
        // Block 2 weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b2_ln1_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b2_ln1_b
        TensorParamBinding::ConstantTensor(ln_w),         // b2_ln2_w
        TensorParamBinding::ConstantTensor(ln_b),         // b2_ln2_b
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_q_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_k_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_v_w
        TensorParamBinding::ConstantTensor(w_proj),       // b2_out_w
        TensorParamBinding::ConstantTensor(w_ffn1),       // b2_ffn1_w
        TensorParamBinding::ConstantTensor(w_ffn2),       // b2_ffn2_w
    ]
}

// ---------------------------------------------------------------------------
// Single-block tests
// ---------------------------------------------------------------------------

/// ViT encoder block TensorKernelDef validates.
#[test]
fn test_vit_encoder_block_def_validates() {
    let def = build_vit_encoder_block_kernel();
    def.validate()
        .expect("ViT encoder block kernel should validate");
}

/// ViT encoder block translates to NY GraphNetwork.
#[test]
fn test_vit_encoder_block_graph_builds() {
    let def = build_vit_encoder_block_kernel();
    let bindings = vit_encoder_block_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("ViT encoder block graph should translate");

    // LayerNorm + MHA (Q/K/V proj + attention + out proj) + residual add
    // + LayerNorm + Linear + GELU + Linear + residual add = many nodes.
    assert!(
        graph.num_nodes() >= 10,
        "ViT encoder block graph should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through a single ViT encoder block.
#[test]
fn test_vit_encoder_block_ibp_propagates() {
    let def = build_vit_encoder_block_kernel();
    let bindings = vit_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ViT encoder block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT encoder block IBP: bounds=[{lo_min}, {hi_max}]");
}

/// CROWN bounds propagate through a single ViT encoder block.
///
/// CROWN may fall back to IBP through normalization layers (LayerNorm
/// uses heuristic linearization via IbpValidated mode). When CROWN
/// succeeds, it should produce tighter bounds.
#[test]
fn test_vit_encoder_block_crown_propagation() {
    let def = build_vit_encoder_block_kernel();
    let bindings = vit_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT encoder block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// ViT encoder block verify and record under "vit_encoder_block" key.
#[test]
fn test_vit_encoder_block_verify_and_record() {
    let def = build_vit_encoder_block_kernel();
    let bindings = vit_encoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "vit_encoder_block");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation -> Heuristic mode.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ViT encoder block with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ---------------------------------------------------------------------------
// Two-block composition tests
// ---------------------------------------------------------------------------

/// 2-block ViT encoder stack translates into a valid GraphNetwork.
#[test]
fn test_vit_two_block_graph_builds() {
    let def = build_vit_two_block_kernel();
    let bindings = vit_two_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("2-block ViT encoder graph should translate");

    // Two transformer blocks stacked = at least 20 nodes.
    assert!(
        graph.num_nodes() >= 20,
        "2-block ViT encoder graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through a 2-block ViT encoder stack.
#[test]
fn test_vit_two_block_ibp_propagates() {
    let def = build_vit_two_block_kernel();
    let bindings = vit_two_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("2-block graph");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-block ViT encoder");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT 2-block IBP: bounds=[{lo_min}, {hi_max}]");
}

/// 2-block ViT encoder verify and record under "vit_two_block_encoder" key.
#[test]
fn test_vit_two_block_verify_and_record() {
    let def = build_vit_two_block_kernel();
    let bindings = vit_two_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "vit_two_block_encoder");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);
}
