// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Full ViT encoder pipeline NY composition.
//!
//! Verifies end-to-end bounds propagation through a complete ViT encoder:
//!
//! 1. **Patch embedding**: Conv2d(3, D, P, stride=P) -> reshape -> transpose -> [NUM_PATCHES, D]
//! 2. **Position embedding addition**: patch_tokens + pos_embed (learned constant)
//! 3. **Transformer block 1**: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual
//! 4. **Transformer block 2**: same structure, separate weights
//! 5. **Final LayerNorm**: output normalization
//!
//! Architecture (Dosovitskiy et al. 2020 "An Image is Worth 16x16 Words"):
//! - Image is split into non-overlapping P x P patches via Conv2d with kernel=stride=P
//! - Each patch is linearly projected to D dimensions
//! - Learned position embeddings are added
//! - L transformer blocks with pre-norm (LayerNorm before attention and FFN)
//! - Bidirectional self-attention (Standard mask, no causal masking)
//! - FFN with GELU activation and 4x expansion ratio
//! - Final LayerNorm after the last transformer block
//!
//! Input bounds: image pixels in [0, 1] (normalized RGB).
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=32, PATCH_SIZE=16, GRID=2, NUM_PATCHES=4
//! - EMBED_DIM=64, NUM_HEADS=4, FFN_DIM=256, WEIGHT_MAG=0.02
//!
//! Part of #3553: ViT encoder full pipeline compose verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{
    tensor_kernel_to_graph, BoundedTensor, TensorParamBinding, VerificationSoundnessMode,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Image height and width (square image).
const IMG_SIZE: usize = 32;
/// Patch size (P). IMG_SIZE must be divisible by PATCH_SIZE.
const PATCH_SIZE: usize = 16;
/// Number of patches per spatial dimension.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Total number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Embedding dimension (tiny ViT hidden size).
const EMBED_DIM: usize = 64;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per ViT spec.
const FFN_DIM: usize = 256;
/// Weight magnitude for initialization.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build the full ViT encoder pipeline as a single TensorKernelDef.
///
/// Input: `[3, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, EMBED_DIM]` after final LayerNorm.
///
/// Pipeline:
/// 1. Conv2d(3, D, P, stride=P) -> [D, GRID, GRID]
/// 2. Reshape [D, NUM_PATCHES] -> Transpose [NUM_PATCHES, D]
/// 3. BinaryAdd(patch_tokens, pos_embed)
/// 4. TransformerBlock 1: LN -> MHA -> residual -> LN -> FFN(Linear->GELU->Linear) -> residual
/// 5. TransformerBlock 2: same structure, separate weights
/// 6. Final LayerNorm
fn build_vit_full_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_full_encoder");

    // --- Input ---
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // --- Patch embedding weights ---
    let proj_w = b.add_input(
        "proj_weight",
        &[EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);

    // --- Position embedding (constant) ---
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, EMBED_DIM]);

    // --- Shared epsilon ---
    let eps = b.add_input("eps", &[1]);

    // --- Transformer block 1 weights ---
    let b1_ln1_w = b.add_input("b1_ln1_weight", &[EMBED_DIM]);
    let b1_ln1_b = b.add_input("b1_ln1_bias", &[EMBED_DIM]);
    let b1_ln2_w = b.add_input("b1_ln2_weight", &[EMBED_DIM]);
    let b1_ln2_b = b.add_input("b1_ln2_bias", &[EMBED_DIM]);
    let b1_q_w = b.add_input("b1_q_weight", &[EMBED_DIM, EMBED_DIM]);
    let b1_k_w = b.add_input("b1_k_weight", &[EMBED_DIM, EMBED_DIM]);
    let b1_v_w = b.add_input("b1_v_weight", &[EMBED_DIM, EMBED_DIM]);
    let b1_out_w = b.add_input("b1_out_weight", &[EMBED_DIM, EMBED_DIM]);
    let b1_ffn1_w = b.add_input("b1_ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let b1_ffn2_w = b.add_input("b1_ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    // --- Transformer block 2 weights ---
    let b2_ln1_w = b.add_input("b2_ln1_weight", &[EMBED_DIM]);
    let b2_ln1_b = b.add_input("b2_ln1_bias", &[EMBED_DIM]);
    let b2_ln2_w = b.add_input("b2_ln2_weight", &[EMBED_DIM]);
    let b2_ln2_b = b.add_input("b2_ln2_bias", &[EMBED_DIM]);
    let b2_q_w = b.add_input("b2_q_weight", &[EMBED_DIM, EMBED_DIM]);
    let b2_k_w = b.add_input("b2_k_weight", &[EMBED_DIM, EMBED_DIM]);
    let b2_v_w = b.add_input("b2_v_weight", &[EMBED_DIM, EMBED_DIM]);
    let b2_out_w = b.add_input("b2_out_weight", &[EMBED_DIM, EMBED_DIM]);
    let b2_ffn1_w = b.add_input("b2_ffn1_weight", &[FFN_DIM, EMBED_DIM]);
    let b2_ffn2_w = b.add_input("b2_ffn2_weight", &[EMBED_DIM, FFN_DIM]);

    // --- Final LayerNorm weights ---
    let final_ln_w = b.add_input("final_ln_weight", &[EMBED_DIM]);
    let final_ln_b = b.add_input("final_ln_bias", &[EMBED_DIM]);

    // =====================================================================
    // Pipeline construction
    // =====================================================================

    // 1. Patch embedding: Conv2d -> reshape -> transpose
    let conv_out = b.add_conv2d(
        image,
        proj_w,
        Some(proj_b),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[EMBED_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[EMBED_DIM, NUM_PATCHES]);
    let patch_tokens = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, EMBED_DIM]);

    // 2. Position embedding addition
    let embedded = b.add_binary_add(patch_tokens, pos_embed, &[NUM_PATCHES, EMBED_DIM]);

    // 3. Transformer block 1
    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard, // ViT uses bidirectional attention
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
        .add_transformer_block(embedded, &weights1, &config)
        .expect("block 1");

    // 4. Transformer block 2
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

    // 5. Final LayerNorm
    let out = b.add_layer_norm(
        block2,
        eps,
        1, // normalize over last axis (embed_dim)
        final_ln_w,
        final_ln_b,
        &[NUM_PATCHES, EMBED_DIM],
    );

    b.build(out).expect("valid full ViT encoder kernel")
}

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Bindings for the full ViT encoder pipeline.
///
/// Input is Variable (image), all weights are ConstantTensor.
/// Order must match `add_input` calls in `build_vit_full_encoder_kernel`.
fn vit_full_encoder_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(
        IxDyn(&[EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let proj_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, EMBED_DIM]), 0.01f32);

    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        // image [3, 32, 32]
        TensorParamBinding::Variable,
        // Patch embedding
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
        // Position embedding
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed
        // Shared epsilon
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // Block 1 weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b1_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b1_ln1_bias
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b1_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b1_ln2_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_v_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_out_weight
        TensorParamBinding::ConstantTensor(w_ffn1.clone()), // b1_ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2.clone()), // b1_ffn2_weight
        // Block 2 weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b2_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b2_ln1_bias
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b2_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b2_ln2_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_v_weight
        TensorParamBinding::ConstantTensor(w_proj),       // b2_out_weight
        TensorParamBinding::ConstantTensor(w_ffn1),       // b2_ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2),       // b2_ffn2_weight
        // Final LayerNorm
        TensorParamBinding::ConstantTensor(ln_w), // final_ln_weight
        TensorParamBinding::ConstantTensor(ln_b), // final_ln_bias
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full ViT encoder pipeline TensorKernelDef validates.
#[test]
fn test_vit_full_encoder_def_validates() {
    let def = build_vit_full_encoder_kernel();
    def.validate()
        .expect("full ViT encoder kernel should validate");
}

/// Full ViT encoder pipeline translates to NY GraphNetwork.
#[test]
fn test_vit_full_encoder_graph_builds() {
    let def = build_vit_full_encoder_kernel();
    let bindings = vit_full_encoder_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("full ViT encoder graph should translate");

    // Conv2d + Reshape + Transpose + BinaryAdd (pos_embed)
    // + 2 transformer blocks (each ~10+ nodes)
    // + Final LayerNorm
    // = at least 25 nodes.
    assert!(
        graph.num_nodes() >= 25,
        "full ViT encoder graph should have >= 25 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full ViT encoder pipeline.
///
/// Validates end-to-end: image [0,1] -> patch embedding -> position embedding
/// -> 2 transformer blocks -> final LayerNorm -> bounded output.
#[test]
fn test_vit_full_encoder_ibp_propagates() {
    let def = build_vit_full_encoder_kernel();
    let bindings = vit_full_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full ViT encoder");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape should be [NUM_PATCHES={NUM_PATCHES}, EMBED_DIM={EMBED_DIM}], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "ViT full encoder IBP (image [0,1]): bounds=[{lo_min}, {hi_max}], \
         width={:.4}",
        hi_max - lo_min
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through the full ViT encoder pipeline.
///
/// The pipeline contains multiple LayerNorm layers (4 within the transformer
/// blocks + 1 final), which require heuristic CROWN linearization via
/// IbpValidated mode. GELU activations linearize cleanly. Attention softmax
/// may cause additional approximation.
///
/// When CROWN succeeds (does not fall back to IBP), it should produce
/// tighter bounds than IBP alone.
#[test]
fn test_vit_full_encoder_crown_propagation() {
    let def = build_vit_full_encoder_kernel();
    let bindings = vit_full_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01();

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "ViT full encoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}], \
         width={:.4}",
        hi_max - lo_min
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Compare IBP vs CROWN bounds width through the full ViT encoder pipeline.
///
/// Runs both propagation methods and reports the width difference. CROWN
/// should produce tighter (narrower) bounds when it successfully linearizes
/// through all layers. With normalization layers, CROWN may fall back to
/// IBP or produce vacuously wide bounds (#2715).
#[test]
fn test_vit_full_encoder_ibp_vs_crown_width() {
    let def = build_vit_full_encoder_kernel();
    let bindings = vit_full_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01();

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through full ViT encoder");
    let (ibp_lo_min, ibp_hi_max) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi_max - ibp_lo_min;

    // CROWN with fallback
    let (method, crown_output, _fallback_reason) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");
    let (crown_lo_min, crown_hi_max) = bounds_min_max(&crown_output);
    let crown_width = crown_hi_max - crown_lo_min;

    eprintln!("ViT full encoder IBP vs CROWN comparison:");
    eprintln!("  IBP:   bounds=[{ibp_lo_min}, {ibp_hi_max}], width={ibp_width:.4}");
    eprintln!("  CROWN: bounds=[{crown_lo_min}, {crown_hi_max}], width={crown_width:.4}");
    eprintln!("  Method: {method:?}");

    if matches!(method, nn_verify::PropMethod::Crown) && crown_width < ibp_width {
        let improvement = (ibp_width - crown_width) / ibp_width * 100.0;
        eprintln!("  CROWN tightening: {improvement:.1}% narrower than IBP");
    } else if matches!(method, nn_verify::PropMethod::Crown) {
        eprintln!(
            "  WARNING (#2715): CROWN succeeded but bounds are wider than IBP. \
             Likely normalization layer vacuous bounds."
        );
    } else {
        eprintln!("  CROWN fell back to IBP — widths are expected to match.");
    }

    // Both must be finite and non-degenerate
    assert!(ibp_width.is_finite(), "IBP width must be finite");
    assert!(crown_width.is_finite(), "CROWN width must be finite");
    assert!(ibp_width > 0.0, "IBP bounds must be non-degenerate");
    assert!(crown_width > 0.0, "CROWN bounds must be non-degenerate");
}

/// Full ViT encoder verify and record under "vit_full_encoder" key.
///
/// Records the pipeline result in the per-model verification status file.
/// LayerNorm causes heuristic normalization approximation, so soundness
/// mode should be Heuristic.
#[test]
fn test_vit_full_encoder_verify_and_record() {
    let def = build_vit_full_encoder_kernel();
    let bindings = vit_full_encoder_bindings();
    let input = image_bounds_01();

    let result = verify_and_assert(&def, &bindings, &input, "vit_full_encoder");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape should be [NUM_PATCHES, EMBED_DIM]"
    );

    // Multiple LayerNorm layers use heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "Full ViT encoder with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
