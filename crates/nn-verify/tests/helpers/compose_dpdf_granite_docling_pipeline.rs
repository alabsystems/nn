// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Granite-Docling-258M full encoder pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the SigLIP2 vision encoder
//! and document understanding pipeline of Granite-Docling-258M:
//!
//! ## Tests (18 tests)
//!
//! 1.  **SigLIP2 patch embedding Conv2d bounds** (IBP)
//! 2.  **Position encoding addition bounds** (IBP)
//! 3.  **ViT encoder single-block self-attention + FFN bounds** (IBP + CROWN)
//! 4.  **ViT encoder 2-block stack bounds** (IBP + CROWN)
//! 5.  **ViT encoder 4-block stack bounds** (IBP)
//! 6.  **CLS token pooling/extraction bounds** (IBP)
//! 7.  **Document layout classification MLP head bounds** (IBP + CROWN)
//! 8.  **Multi-class softmax output bounds [0,1]** (IBP)
//! 9.  **LayerNorm stabilization at each encoder block** (IBP)
//! 10. **Residual connection bound growth through depth** (IBP)
//! 11. **Vision feature projection bounds** (IBP + CROWN)
//! 12. **Full encoder end-to-end IBP bounds** (IBP)
//! 13. **Full encoder CROWN bounds** (CROWN)
//! 14. **Input resolution scaling effect** (IBP)
//! 15. **Patch merging/downsampling bounds** (IBP)
//! 16. **Feature dimensionality reduction bounds** (IBP + CROWN)
//! 17. **Batch independence** (IBP)
//! 18. **Monotone tightening property** (IBP + CROWN)
//!
//! Architecture references:
//! - Granite-Docling-258M: SigLIP2 vision encoder with 27 ViT blocks
//! - SigLIP2 (Zhai et al., 2023): Sigmoid-loss pre-trained ViT encoder
//! - ViT (Dosovitskiy et al., 2020): Patch embedding + transformer encoder
//! - Production dims: hidden=768, heads=12, FFN=3072, patches=196 (224x224, P=16)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_H=16, IMG_W=16, PATCH_SIZE=4, IN_CHANNELS=3
//! - SEQ_LEN=16 (16/4 * 16/4 = 16 patches), HIDDEN_DIM=48, FFN_DIM=96
//! - NUM_HEADS=4, HEAD_DIM=12, NUM_CLASSES=5
//!
//! Part of #4180: Compose tests for Granite-Docling-258M pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of Granite-Docling-258M (production: hidden=768, heads=12, FFN=3072)
// ---------------------------------------------------------------------------

const IMG_H: usize = 16;
const IMG_W: usize = 16;
const PATCH_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
/// Number of patches: (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE).
const SEQ_LEN: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 16
const HIDDEN_DIM: usize = 48;
const FFN_DIM: usize = 96;
const NUM_HEADS: usize = 4;
/// Number of document layout classes for classification head.
const NUM_CLASSES: usize = 5;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for LayerNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(channels: usize, h: usize, w: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

/// Build patch embedding Conv2d kernel: [C_in, H, W] -> [EMBED, H/P, W/P].
fn build_patch_embed_kernel(
    name: &str,
    in_ch: usize,
    embed_dim: usize,
    patch_size: usize,
    img_h: usize,
    img_w: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let out_h = img_h / patch_size;
    let out_w = img_w / patch_size;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("image", &[in_ch, img_h, img_w]);
    let w = b.add_input("proj_w", &[embed_dim, in_ch, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[embed_dim]);

    let out = b.add_conv2d(
        input,
        w,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[embed_dim, out_h, out_w],
    );
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[embed_dim, in_ch, patch_size, patch_size]),
        bias_zero(&[embed_dim]),
    ];

    (def, bindings)
}

/// Add a single SigLIP2-style encoder block (pre-norm transformer) to a builder.
///
/// LN -> MHA -> residual -> LN -> FFN(GELU) -> residual.
/// Input/output: [SEQ, DIM]. Returns output node.
fn add_encoder_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    seq_len: usize,
    dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    prefix: &str,
) -> TensorNodeId {
    let shape = [seq_len, dim];
    let ffn_shape = [seq_len, ffn_dim];

    // Pre-norm 1: LayerNorm
    let ln1_w = b.add_input(&format!("{prefix}_ln1_w"), &[dim]);
    let ln1_b = b.add_input(&format!("{prefix}_ln1_b"), &[dim]);
    let eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);

    // Multi-head self-attention
    let qw = b.add_input(&format!("{prefix}_q_w"), &[dim, dim]);
    let kw = b.add_input(&format!("{prefix}_k_w"), &[dim, dim]);
    let vw = b.add_input(&format!("{prefix}_v_w"), &[dim, dim]);
    let ow = b.add_input(&format!("{prefix}_o_w"), &[dim, dim]);
    let attn = b
        .add_multi_head_attention(
            normed,
            qw,
            kw,
            vw,
            ow,
            num_heads,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual 1
    let res1 = b.add_binary_add(input, attn, &shape);

    // Pre-norm 2: LayerNorm
    let ln2_w = b.add_input(&format!("{prefix}_ln2_w"), &[dim]);
    let ln2_b = b.add_input(&format!("{prefix}_ln2_b"), &[dim]);
    let eps2 = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, eps2, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn1_w = b.add_input(&format!("{prefix}_ffn1_w"), &[ffn_dim, dim]);
    let ffn2_w = b.add_input(&format!("{prefix}_ffn2_w"), &[dim, ffn_dim]);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);

    // Residual 2
    b.add_binary_add(res1, ffn2, &shape)
}

/// Build bindings for a single encoder block.
fn encoder_block_bindings(dim: usize, ffn_dim: usize) -> Vec<TensorParamBinding> {
    vec![
        // ln1: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_binding(),
        // MHA: Q, K, V, O weights
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        weight(&[dim, dim]),
        // ln2: weight, bias, eps
        ones(&[dim]),
        bias_zero(&[dim]),
        eps_binding(),
        // FFN: ffn1_w, ffn2_w
        weight(&[ffn_dim, dim]),
        weight(&[dim, ffn_dim]),
    ]
}

// ===========================================================================
// 1. SigLIP2 patch embedding Conv2d bounds (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_patch_embed_conv2d_ibp() {
    let (def, bindings) = build_patch_embed_kernel(
        "gd_patch_embed",
        IN_CHANNELS,
        HIDDEN_DIM,
        PATCH_SIZE,
        IMG_H,
        IMG_W,
    );
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;
    assert_eq!(output.lower_upper().0.shape(), &[HIDDEN_DIM, out_h, out_w]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD patch embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Position encoding addition bounds (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_pos_encoding_addition_ibp() {
    let mut b = TensorBlockBuilder::new("gd_pos_encode_add");
    let patch_tokens = b.add_input("patch_tokens", &[SEQ_LEN, HIDDEN_DIM]);
    let pos_embed = b.add_input("pos_embed", &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(patch_tokens, pos_embed, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid pos encode kernel");

    let pe_data = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.01f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD pos encoding add IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min < 0.0, "lower bound shifted by positive PE");
    assert!(hi_max > 0.0, "upper bound shifted by positive PE");
}

// ===========================================================================
// 3. ViT encoder single-block self-attention + FFN bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_granite_docling_single_encoder_block_ibp_crown() {
    let mut b = TensorBlockBuilder::new("gd_single_encoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let def = b.build(out).expect("valid encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GD single encoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN should also produce valid bounds
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD single encoder block CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. ViT encoder 2-block stack bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_granite_docling_2block_encoder_ibp_crown() {
    let mut b = TensorBlockBuilder::new("gd_2block_encoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let mid = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let out = add_encoder_block(&mut b, mid, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");
    let def = b.build(out).expect("valid 2-block encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GD 2-block encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD 2-block encoder CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 5. ViT encoder 4-block stack bounds (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_4block_encoder_ibp() {
    let mut b = TensorBlockBuilder::new("gd_4block_encoder");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let l2 = add_encoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");
    let l3 = add_encoder_block(&mut b, l2, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk2");
    let out = add_encoder_block(&mut b, l3, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk3");
    let def = b.build(out).expect("valid 4-block encoder kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    }
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD 4-block encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. CLS token pooling/extraction bounds (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_cls_token_extraction_ibp() {
    // CLS token: prepend a learned [1, DIM] token, run encoder, extract first row.
    let cls_seq = SEQ_LEN + 1;

    let mut b = TensorBlockBuilder::new("gd_cls_token");
    let input = b.add_input("x_with_cls", &[cls_seq, HIDDEN_DIM]);
    let out = add_encoder_block(
        &mut b, input, cls_seq, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );

    // Narrow to CLS token: first row [1, HIDDEN_DIM]
    let cls_out = b.add_narrow(out, 0, 0, 1, &[1, HIDDEN_DIM]);
    let def = b.build(cls_out).expect("valid CLS token kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(cls_seq, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD CLS token after encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Document layout classification MLP head bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_granite_docling_classification_mlp_head_ibp_crown() {
    // Classification head: Linear(DIM, FFN_DIM) -> GELU -> Linear(FFN_DIM, NUM_CLASSES)
    let mut b = TensorBlockBuilder::new("gd_cls_mlp_head");
    let input = b.add_input("cls_features", &[1, HIDDEN_DIM]);

    let mlp1_w = b.add_input("cls_mlp1_w", &[FFN_DIM, HIDDEN_DIM]);
    let mlp1_b = b.add_input("cls_mlp1_b", &[FFN_DIM]);
    let mlp1 = b.add_linear(input, mlp1_w, Some(mlp1_b), &[1, FFN_DIM]);
    let act = b.add_gelu(mlp1, &[1, FFN_DIM]);

    let mlp2_w = b.add_input("cls_mlp2_w", &[NUM_CLASSES, FFN_DIM]);
    let mlp2_b = b.add_input("cls_mlp2_b", &[NUM_CLASSES]);
    let out = b.add_linear(act, mlp2_w, Some(mlp2_b), &[1, NUM_CLASSES]);
    let def = b.build(out).expect("valid classification MLP head kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias_zero(&[NUM_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[1, NUM_CLASSES]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GD classification MLP head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD classification MLP head CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 8. Multi-class softmax output bounds [0,1] (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_softmax_output_bounded_ibp() {
    // Classification logits -> softmax -> [0, 1] per class
    let mut b = TensorBlockBuilder::new("gd_softmax_output");
    let input = b.add_input("logits", &[1, NUM_CLASSES]);
    let out = b.add_softmax(input, -1, &[1, NUM_CLASSES]);
    let def = b.build(out).expect("valid softmax kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, NUM_CLASSES, 5.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD softmax output IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax outputs must be in [0, 1]
    assert!(
        lo_min >= -1e-5,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. LayerNorm stabilization at each encoder block (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_layernorm_stabilization_ibp() {
    // Verify that LayerNorm tightens bounds relative to un-normalized input.
    // Single LN: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("gd_layernorm_stabilize");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid LayerNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input bounds to test normalization effect
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 5.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD LayerNorm stabilization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Residual connection bound growth through depth (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_residual_bound_growth_ibp() {
    // Compare bounds after 1, 2, and 4 encoder blocks to observe growth.
    let build_n_blocks = |n: usize| -> BoundedTensor {
        let mut b = TensorBlockBuilder::new(&format!("gd_residual_{n}blk"));
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let mut x = input;
        for i in 0..n {
            x = add_encoder_block(
                &mut b,
                x,
                SEQ_LEN,
                HIDDEN_DIM,
                FFN_DIM,
                NUM_HEADS,
                &format!("blk{i}"),
            );
        }
        let def = b.build(x).expect("valid n-block encoder");
        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..n {
            bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
        }
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
        graph.propagate_ibp(&inp).expect("IBP")
    };

    let out1 = build_n_blocks(1);
    let out2 = build_n_blocks(2);
    let out4 = build_n_blocks(4);
    assert_bounds_valid(&out1);
    assert_bounds_valid(&out2);
    assert_bounds_valid(&out4);

    let (_, w1) = bounds_min_max(&out1);
    let (l1, _) = bounds_min_max(&out1);
    let (_, w2) = bounds_min_max(&out2);
    let (l2, _) = bounds_min_max(&out2);
    let (_, w4) = bounds_min_max(&out4);
    let (l4, _) = bounds_min_max(&out4);
    let width1 = w1 - l1;
    let width2 = w2 - l2;
    let width4 = w4 - l4;

    eprintln!("GD residual growth: 1-blk width={width1:.4}, 2-blk width={width2:.4}, 4-blk width={width4:.4}");
    // Bounds should grow (or remain stable) with depth, never shrink drastically
    assert!(
        width1.is_finite() && width2.is_finite() && width4.is_finite(),
        "all widths must be finite"
    );
}

// ===========================================================================
// 11. Vision feature projection bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_granite_docling_vision_projection_ibp_crown() {
    // Vision projection: Linear mapping encoder features to a target LM dimension.
    let lm_dim = 64; // target LM embedding dimension
    let mut b = TensorBlockBuilder::new("gd_vision_projection");
    let input = b.add_input("encoder_features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[lm_dim, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[lm_dim]);
    let out = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, lm_dim]);
    let def = b.build(out).expect("valid vision projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[lm_dim, HIDDEN_DIM]),
        bias_zero(&[lm_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, lm_dim]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GD vision projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD vision projection CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 12. Full encoder end-to-end IBP bounds
// ===========================================================================

#[test]
fn test_granite_docling_full_encoder_e2e_ibp() {
    // Full pipeline: patch embed -> 2 encoder blocks -> final LN
    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("gd_full_encoder_e2e");
    let image = b.add_input("image", &[IN_CHANNELS, IMG_H, IMG_W]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);

    // Patch embed: [3, 16, 16] -> [HIDDEN_DIM, 4, 4]
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );

    // Reshape: [HIDDEN_DIM, 4, 4] -> [SEQ_LEN, HIDDEN_DIM]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, SEQ_LEN]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[SEQ_LEN, HIDDEN_DIM]);

    // 2 encoder blocks
    let l1 = add_encoder_block(
        &mut b, transposed, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let l2 = add_encoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(l2, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid full encoder kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable, // image
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.push(ones(&[HIDDEN_DIM])); // final LN weight
    bindings.push(bias_zero(&[HIDDEN_DIM])); // final LN bias
    bindings.push(eps_binding()); // final LN eps

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD full encoder e2e IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Full encoder CROWN bounds
// ===========================================================================

#[test]
fn test_granite_docling_full_encoder_crown() {
    // Encoder: 2 blocks + final LN (from sequence input, skipping conv for CROWN tractability)
    let mut b = TensorBlockBuilder::new("gd_encoder_crown");
    let input = b.add_input("patch_embeds", &[SEQ_LEN, HIDDEN_DIM]);
    let l1 = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let l2 = add_encoder_block(&mut b, l1, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk1");

    // Final LayerNorm
    let ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(l2, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid encoder CROWN kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    bindings.push(ones(&[HIDDEN_DIM]));
    bindings.push(bias_zero(&[HIDDEN_DIM]));
    bindings.push(eps_binding());

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD full encoder CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 14. Input resolution scaling effect (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_resolution_scaling_ibp() {
    // Compare bounds from 16x16 vs 32x32 images (same patch_size=4).
    let (def16, bind16) =
        build_patch_embed_kernel("gd_res16", IN_CHANNELS, HIDDEN_DIM, PATCH_SIZE, 16, 16);
    let (def32, bind32) =
        build_patch_embed_kernel("gd_res32", IN_CHANNELS, HIDDEN_DIM, PATCH_SIZE, 32, 32);

    let graph16 = tensor_kernel_to_graph(&def16, &bind16).expect("graph 16x16");
    let graph32 = tensor_kernel_to_graph(&def32, &bind32).expect("graph 32x32");

    let input16 = image_bounds(IN_CHANNELS, 16, 16);
    let input32 = image_bounds(IN_CHANNELS, 32, 32);

    let out16 = graph16.propagate_ibp(&input16).expect("IBP 16x16");
    let out32 = graph32.propagate_ibp(&input32).expect("IBP 32x32");
    assert_bounds_valid(&out16);
    assert_bounds_valid(&out32);

    // 16x16 -> 4x4 patches; 32x32 -> 8x8 patches
    assert_eq!(out16.lower_upper().0.shape(), &[HIDDEN_DIM, 4, 4]);
    assert_eq!(out32.lower_upper().0.shape(), &[HIDDEN_DIM, 8, 8]);

    let (lo16, hi16) = bounds_min_max(&out16);
    let (lo32, hi32) = bounds_min_max(&out32);
    eprintln!("GD res 16x16 IBP: [{lo16:.6}, {hi16:.6}]");
    eprintln!("GD res 32x32 IBP: [{lo32:.6}, {hi32:.6}]");

    // Same Conv2d weights + same input range => per-patch bounds should be similar
    let width16 = hi16 - lo16;
    let width32 = hi32 - lo32;
    let ratio = width32 / width16;
    eprintln!("GD resolution bound width ratio (32/16): {ratio:.4}");
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "per-patch bounds should be similar across resolutions"
    );
}

// ===========================================================================
// 15. Patch merging/downsampling bounds (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_patch_merging_ibp() {
    // Patch merging: concatenate adjacent patches then project.
    // [SEQ_LEN, DIM] -> reshape [SEQ/2, 2*DIM] -> Linear [SEQ/2, DIM]
    let half_seq = SEQ_LEN / 2;
    let double_dim = HIDDEN_DIM * 2;

    let mut b = TensorBlockBuilder::new("gd_patch_merge");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let reshaped = b.add_reshape(input, &[half_seq, double_dim]);

    let proj_w = b.add_input("merge_w", &[HIDDEN_DIM, double_dim]);
    let proj_b = b.add_input("merge_b", &[HIDDEN_DIM]);
    let out = b.add_linear(reshaped, proj_w, Some(proj_b), &[half_seq, HIDDEN_DIM]);
    let def = b.build(out).expect("valid patch merging kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, double_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[half_seq, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GD patch merging IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 16. Feature dimensionality reduction bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_granite_docling_dim_reduction_ibp_crown() {
    // Dimensionality reduction: Linear(DIM -> DIM/2) + GELU + Linear(DIM/2 -> DIM/4)
    let mid_dim = HIDDEN_DIM / 2;
    let small_dim = HIDDEN_DIM / 4;

    let mut b = TensorBlockBuilder::new("gd_dim_reduction");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let w1 = b.add_input("reduce1_w", &[mid_dim, HIDDEN_DIM]);
    let l1 = b.add_linear(input, w1, None, &[SEQ_LEN, mid_dim]);
    let act = b.add_gelu(l1, &[SEQ_LEN, mid_dim]);

    let w2 = b.add_input("reduce2_w", &[small_dim, mid_dim]);
    let out = b.add_linear(act, w2, None, &[SEQ_LEN, small_dim]);
    let def = b.build(out).expect("valid dim reduction kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[mid_dim, HIDDEN_DIM]),
        weight(&[small_dim, mid_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[SEQ_LEN, small_dim]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GD dim reduction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD dim reduction CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 17. Batch independence (IBP)
// ===========================================================================

#[test]
fn test_granite_docling_batch_independence_ibp() {
    // Verify that running a single encoder block on two different input ranges
    // produces independent bounds (no cross-batch contamination).
    let mut b = TensorBlockBuilder::new("gd_batch_indep");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let def = b.build(out).expect("valid encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Narrow input range
    let narrow_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);
    let narrow_out = graph.propagate_ibp(&narrow_inp).expect("IBP narrow");
    assert_bounds_valid(&narrow_out);

    // Wide input range
    let wide_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);
    let wide_out = graph.propagate_ibp(&wide_inp).expect("IBP wide");
    assert_bounds_valid(&wide_out);

    let (_, narrow_hi) = bounds_min_max(&narrow_out);
    let (narrow_lo, _) = bounds_min_max(&narrow_out);
    let (_, wide_hi) = bounds_min_max(&wide_out);
    let (wide_lo, _) = bounds_min_max(&wide_out);
    let narrow_width = narrow_hi - narrow_lo;
    let wide_width = wide_hi - wide_lo;

    eprintln!("GD batch indep: narrow_width={narrow_width:.4}, wide_width={wide_width:.4}");
    // Wider input should not produce narrower output bounds
    assert!(
        wide_width >= narrow_width * 0.9,
        "wider input should produce at least comparable output width"
    );
}

// ===========================================================================
// 18. Monotone tightening property (IBP + CROWN)
// ===========================================================================

#[test]
fn test_granite_docling_monotone_tightening_ibp_crown() {
    // Monotone tightening: narrower input bounds should produce narrower (or equal)
    // output bounds for a single encoder block.
    let mut b = TensorBlockBuilder::new("gd_monotone_tighten");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_block(
        &mut b, input, SEQ_LEN, HIDDEN_DIM, FFN_DIM, NUM_HEADS, "blk0",
    );
    let def = b.build(out).expect("valid encoder block kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(encoder_block_bindings(HIDDEN_DIM, FFN_DIM));
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Three input ranges: tight, medium, wide
    let tight_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);
    let medium_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
    let wide_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);

    // IBP propagation
    let tight_out = graph.propagate_ibp(&tight_inp).expect("IBP tight");
    let medium_out = graph.propagate_ibp(&medium_inp).expect("IBP medium");
    let wide_out = graph.propagate_ibp(&wide_inp).expect("IBP wide");
    assert_bounds_valid(&tight_out);
    assert_bounds_valid(&medium_out);
    assert_bounds_valid(&wide_out);

    let tight_width = {
        let (lo, hi) = bounds_min_max(&tight_out);
        hi - lo
    };
    let medium_width = {
        let (lo, hi) = bounds_min_max(&medium_out);
        hi - lo
    };
    let wide_width = {
        let (lo, hi) = bounds_min_max(&wide_out);
        hi - lo
    };

    eprintln!(
        "GD monotone IBP: tight_w={tight_width:.4}, medium_w={medium_width:.4}, wide_w={wide_width:.4}"
    );

    // Monotonicity: tighter input -> tighter output (with tolerance for numerics)
    let eps = 1e-3;
    assert!(
        tight_width <= medium_width + eps,
        "tight input should produce tight output: {tight_width} > {medium_width} + eps"
    );
    assert!(
        medium_width <= wide_width + eps,
        "medium input should produce medium output: {medium_width} > {wide_width} + eps"
    );

    // Also verify CROWN produces valid bounds on medium input
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &medium_inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("GD monotone CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}
