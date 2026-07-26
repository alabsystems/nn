// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: SigLIP2 vision encoder end-to-end NY composition.
//!
//! Verifies bounds propagation through the full SigLIP2 pipeline:
//!   image -> patch_embed (Linear) -> + pos_embed -> N x encoder_block -> head_proj
//!
//! SigLIP2 (Zhai et al. 2023, "Sigmoid Loss for Language Image Pre-Training")
//! uses a ViT backbone with sigmoid contrastive loss. The architecture is
//! structurally identical to standard ViT.
//!
//! Granite-Docling-258M uses SigLIP2 as its vision encoder.
//!
//! Part of #3540: SigLIP2 end-to-end NY compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of image patches (e.g., 2x2 grid from a 32x32 image with 16x16 patches).
const NUM_PATCHES: usize = 4;
/// Patch dimension after flattening (reduced from 768 for fast verification).
const PATCH_DIM: usize = 48;
/// Embedding dimension (tiny SigLIP2 hidden size).
const EMBED_DIM: usize = 32;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension: 4x embed_dim per SigLIP2/ViT spec.
const FFN_DIM: usize = 128;
/// Head output dimension.
const HEAD_DIM: usize = 16;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build the patch embedding + position embedding pipeline.
fn build_patch_plus_pos_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_patch_pos_embed");

    let patches = b.add_input("patches", &[NUM_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, EMBED_DIM]);

    let embedded = b.add_linear(patches, proj_w, Some(proj_b), &[NUM_PATCHES, EMBED_DIM]);
    let out = b.add_binary_add(embedded, pos_embed, &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid patch+pos embed kernel")
}

/// Build a single SigLIP2 encoder block.
fn build_siglip2_encoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_encoder_block");

    let input = b.add_input("x", &[NUM_PATCHES, EMBED_DIM]);
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

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };

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

    let out = b
        .add_transformer_block(input, &weights, &config)
        .expect("valid SigLIP2 encoder block");
    b.build(out).expect("valid kernel")
}

/// Build the full SigLIP2 pipeline: patch_embed + pos_embed + 2 blocks + head.
fn build_siglip2_full_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_full_encoder");

    let patches = b.add_input("patches", &[NUM_PATCHES, PATCH_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, PATCH_DIM]);
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, EMBED_DIM]);
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

    // Head projection
    let head_w = b.add_input("head_weight", &[HEAD_DIM, EMBED_DIM]);
    let head_b = b.add_input("head_bias", &[HEAD_DIM]);

    // Stage 1: Patch + position embedding
    let embedded = b.add_linear(patches, proj_w, Some(proj_b), &[NUM_PATCHES, EMBED_DIM]);
    let x = b.add_binary_add(embedded, pos_embed, &[NUM_PATCHES, EMBED_DIM]);

    // Stage 2+3: Encoder blocks
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
    let x = b
        .add_transformer_block(x, &weights1, &config)
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
        eps,
    };
    let x = b
        .add_transformer_block(x, &weights2, &config)
        .expect("block 2");

    // Stage 4: Head projection
    let out = b.add_linear(x, head_w, Some(head_b), &[NUM_PATCHES, HEAD_DIM]);

    b.build(out).expect("valid full SigLIP2 encoder kernel")
}

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

fn patch_pos_embed_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, PATCH_DIM]),
            0.02f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_PATCHES, EMBED_DIM]),
            0.01f32,
        )),
    ]
}

fn encoder_block_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj.clone()),
        TensorParamBinding::ConstantTensor(w_proj),
        TensorParamBinding::ConstantTensor(w_ffn1),
        TensorParamBinding::ConstantTensor(w_ffn2),
    ]
}

fn full_encoder_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, PATCH_DIM]), 0.02f32);
    let proj_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, EMBED_DIM]), 0.01f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);
    let head_w = ArrayD::from_elem(IxDyn(&[HEAD_DIM, EMBED_DIM]), 0.02f32);
    let head_b = ArrayD::from_elem(IxDyn(&[HEAD_DIM]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantTensor(proj_b),
        TensorParamBinding::ConstantTensor(pos_embed),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    // Block 1 + Block 2 weights (10 params each)
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    bindings.push(TensorParamBinding::ConstantTensor(head_w));
    bindings.push(TensorParamBinding::ConstantTensor(head_b));

    bindings
}

// ---------------------------------------------------------------------------
// Tests: Patch + Position Embedding
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_patch_pos_embed_ibp() {
    let kernel = build_patch_plus_pos_embed_kernel();
    let bindings = patch_pos_embed_bindings();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(
        &kernel,
        &bindings,
        &input_bounds,
        "siglip2_patch_pos_embed_ibp",
    );
    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_patch_pos_embed IBP: [{lo:.4}, {hi:.4}]");
}

#[test]
fn test_siglip2_patch_pos_embed_crown() {
    let kernel = build_patch_plus_pos_embed_kernel();
    let bindings = patch_pos_embed_bindings();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("graph");
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("siglip2_patch_pos_embed CROWN ({method:?}): [{lo:.4}, {hi:.4}]");
}

// ---------------------------------------------------------------------------
// Tests: Single Encoder Block
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_encoder_block_ibp() {
    let kernel = build_siglip2_encoder_block_kernel();
    let bindings = encoder_block_bindings();
    let input_bounds = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let result = verify_and_assert(
        &kernel,
        &bindings,
        &input_bounds,
        "siglip2_encoder_block_ibp",
    );
    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_encoder_block IBP: [{lo:.4}, {hi:.4}]");
}

#[test]
fn test_siglip2_encoder_block_crown() {
    let kernel = build_siglip2_encoder_block_kernel();
    let bindings = encoder_block_bindings();
    let input_bounds = uniform_bounds(&[NUM_PATCHES, EMBED_DIM], 1.0);

    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("graph");
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("siglip2_encoder_block CROWN ({method:?}): [{lo:.4}, {hi:.4}]");
}

// ---------------------------------------------------------------------------
// Tests: Full End-to-End Pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_full_encoder_ibp() {
    let kernel = build_siglip2_full_encoder_kernel();
    let bindings = full_encoder_bindings();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(
        &kernel,
        &bindings,
        &input_bounds,
        "siglip2_full_encoder_ibp",
    );
    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!("siglip2_full_encoder IBP: [{lo:.4}, {hi:.4}]");
}

#[test]
fn test_siglip2_full_encoder_crown() {
    let kernel = build_siglip2_full_encoder_kernel();
    let bindings = full_encoder_bindings();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let graph = tensor_kernel_to_graph(&kernel, &bindings).expect("graph");
    let (method, output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("siglip2_full_encoder CROWN ({method:?}): [{lo:.4}, {hi:.4}]");
}

// ---------------------------------------------------------------------------
// Tests: Bounds sanity
// ---------------------------------------------------------------------------

#[test]
fn test_siglip2_patch_embed_bounds_reasonable() {
    let kernel = build_patch_plus_pos_embed_kernel();
    let bindings = patch_pos_embed_bindings();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(
        &kernel,
        &bindings,
        &input_bounds,
        "siglip2_patch_embed_bounds_width",
    );

    // Patch embedding is linear (exact in IBP). Bounds width should be reasonable.
    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    let diff = hi_arr.to_owned() - lo_arr.to_owned();
    let max_width = diff.iter().copied().fold(0.0f32, f32::max);
    eprintln!("siglip2_patch_embed max bounds width: {max_width:.4}");
    assert!(
        max_width < 100.0,
        "patch embedding bounds width {max_width} should be < 100 for unit input range"
    );
}

#[test]
fn test_siglip2_full_encoder_bounds_finite() {
    let kernel = build_siglip2_full_encoder_kernel();
    let bindings = full_encoder_bindings();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PATCHES, PATCH_DIM]), 1.0f32),
    )
    .expect("bounds");

    let result = verify_and_assert(
        &kernel,
        &bindings,
        &input_bounds,
        "siglip2_full_encoder_finite",
    );

    // All bounds must be finite (no NaN/Inf from attention or norm layers).
    let (lo_arr, hi_arr) = result.output_bounds.lower_upper();
    for &v in lo_arr.iter() {
        assert!(v.is_finite(), "lower bound is not finite: {v}");
    }
    for &v in hi_arr.iter() {
        assert!(v.is_finite(), "upper bound is not finite: {v}");
    }
    for (lo, hi) in lo_arr.iter().zip(hi_arr.iter()) {
        assert!(lo <= hi, "lower {lo} > upper {hi}");
    }
}
