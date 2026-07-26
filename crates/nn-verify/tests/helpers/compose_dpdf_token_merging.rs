// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for token merging and spatial reduction bounds in dpdf
//! vision-language models.
//!
//! Verifies IBP and CROWN bound propagation through token merging and
//! spatial reduction patterns used across dpdf models (SigLIP2, Qwen3-VL,
//! Granite-Docling, Swin Transformer, PaddleOCR).
//! Token merging reduces the number of tokens in a sequence (e.g., by pooling,
//! attention-weighted aggregation, or strided convolution) before projecting
//! to a text embedding space for vision-language alignment.
//!
//! 1.  **Adaptive average pooling spatial reduction** (IBP)
//! 2.  **Spatial reshape (H*W -> seq_len)** (IBP)
//! 3.  **Token concatenation from multi-scale features** (IBP)
//! 4.  **Vision-to-text linear projection after pooling** (IBP)
//! 5.  **Adaptive pooling CROWN bounds** (CROWN)
//! 6.  **Multi-scale feature concatenation + projection** (IBP)
//! 7.  **Token selection/sampling pattern** (IBP)
//! 8.  **Spatial reduction via strided convolution** (IBP)
//! 9.  **Token merging with attention weights** (IBP)
//! 10. **Vision-text projection after spatial reduction** (CROWN)
//! 11. **Monotone tightening for spatial reduction pipeline** (IBP)
//! 12. **Full pipeline: Conv features -> pool -> reshape -> project -> text embedding** (IBP)
//!
//! Architecture references:
//! - Swin Transformer (Liu et al., 2021): Patch merging via reshape + linear
//! - SigLIP2 (Zhai et al., 2023): Vision-text projection after pooling
//! - Qwen3-VL (Alibaba): Multi-scale spatial reduction for vision tokens
//! - Granite-Docling: Vision encoder spatial reduction before LM projection
//! - ToMe (Bolya et al., 2023): Token merging via bipartite soft matching
//!
//! Dimensions (small for fast verification, structurally representative):
//! - Feature maps: 8x8 input, channels=8/16
//! - Sequence: SEQ_LEN=4/8, HIDDEN_DIM=32/64
//!
//! Part of #4062: Compose tests for token merging and spatial reduction bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{ReduceOp, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SPATIAL: usize = 8;
const CHANNELS: usize = 8;
const CHANNELS_WIDE: usize = 16;
const SEQ_LEN: usize = 4;
const SEQ_LEN_LONG: usize = 8;
const HIDDEN_DIM: usize = 32;
const TEXT_DIM: usize = 64;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Helper to create constant weight bindings.
fn const_weight(shape: &[usize], val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), val))
}

// ===========================================================================
// 1. Adaptive average pooling spatial reduction (IBP)
// ===========================================================================

/// Build adaptive average pooling: input 8x8 -> output 2x2.
/// Models the spatial reduction stage in vision transformers before
/// flattening to a token sequence.
fn build_adaptive_avg_pool_reduction_kernel() -> TensorKernelDef {
    let target_h = 2;
    let target_w = 2;
    let kernel_h = SPATIAL / target_h; // 4
    let kernel_w = SPATIAL / target_w; // 4

    let mut b = TensorBlockBuilder::new("dpdf_token_adaptive_pool");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let out = b.add_avg_pool_2d(
        input,
        kernel_h,
        kernel_w,
        kernel_h,
        kernel_w,
        0,
        0,
        &[CHANNELS, target_h, target_w],
    );
    b.build(out)
        .expect("valid adaptive avg pool reduction kernel")
}

#[test]
fn test_adaptive_avg_pool_spatial_reduction_ibp() {
    let def = build_adaptive_avg_pool_reduction_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Adaptive avg pool reduction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Avg pool of uniform [-1, 1] stays in [-1, 1]
    assert!(
        lo_min >= -1.0 - 1e-4,
        "adaptive pool lower >= -1.0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "adaptive pool upper <= 1.0, got {hi_max}"
    );
}

// ===========================================================================
// 2. Spatial reshape (H*W -> seq_len) (IBP)
// ===========================================================================

/// Build spatial reshape: [C, H, W] -> reshape -> [H*W, C].
/// This is the flatten + transpose step that converts spatial feature maps
/// into a sequence of tokens for transformer processing.
#[test]
fn test_spatial_reshape_to_seq_ibp() {
    let num_tokens = SPATIAL * SPATIAL; // 64

    let mut b = TensorBlockBuilder::new("dpdf_token_spatial_reshape");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    // Reshape: [C, H, W] -> [C, H*W]
    let flat = b.add_reshape(input, &[CHANNELS, num_tokens]);
    // Transpose: [C, H*W] -> [H*W, C]
    let out = b.add_transpose(flat, &[1, 0], &[num_tokens, CHANNELS]);
    let def = b.build(out).expect("valid spatial reshape kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spatial reshape (H*W -> seq) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Reshape/transpose preserve exact bounds
    assert!(
        lo_min >= -1.0 - 1e-6,
        "reshape preserves lower, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "reshape preserves upper, got {hi_max}"
    );
}

// ===========================================================================
// 3. Token concatenation from multi-scale features (IBP)
// ===========================================================================

/// Build token concatenation: tokens from two spatial scales are concatenated
/// along the sequence dimension. Models multi-scale feature fusion in FPN-style
/// vision backbones before vision-language projection.
#[test]
fn test_token_concat_multi_scale_ibp() {
    let scale1_tokens = 4; // from 2x2 feature map
    let scale2_tokens = 4; // from 2x2 feature map (different scale)
    let total_tokens = scale1_tokens + scale2_tokens; // 8

    let mut b = TensorBlockBuilder::new("dpdf_token_concat_multiscale");
    let tokens_s1 = b.add_input("tokens_s1", &[scale1_tokens, HIDDEN_DIM]);
    let tokens_s2 = b.add_input("tokens_s2", &[scale2_tokens, HIDDEN_DIM]);
    // Concat along sequence dim: [4, D] + [4, D] -> [8, D]
    let out = b.add_concat(&[tokens_s1, tokens_s2], 0, &[total_tokens, HIDDEN_DIM]);
    let def = b.build(out).expect("valid token concat kernel");

    // The kernel has two distinct Input nodes (tokens_s1, tokens_s2), so it
    // needs two Variable bindings. The two variables share an identical shape
    // [scale_tokens, HIDDEN_DIM] and are fed as a single network input stacked
    // along a new leading dim 0 (see setup_multi_variable_inputs / the
    // multi-variable convention in graph_translate_tensor_multi_var.rs).
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2, scale1_tokens, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token concat multi-scale IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Vision-to-text linear projection after pooling (IBP)
// ===========================================================================

/// Build vision-to-text projection: pool -> reshape -> linear -> text space.
/// This models the vision-language alignment projection in VLMs like SigLIP2
/// and Granite-Docling that maps pooled visual features to the LM embedding
/// dimension.
fn build_vision_text_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_token_vision_text_proj");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);

    // Global avg pool: [C, H, W] -> [C, 1, 1]
    let pooled = b.add_avg_pool_2d(
        input,
        SPATIAL,
        SPATIAL,
        SPATIAL,
        SPATIAL,
        0,
        0,
        &[CHANNELS, 1, 1],
    );
    // Reshape: [C, 1, 1] -> [1, C]
    let flat = b.add_reshape(pooled, &[1, CHANNELS]);
    // Linear projection: [1, C] -> [1, TEXT_DIM]
    let proj_w = b.add_input("proj_w", &[TEXT_DIM, CHANNELS]);
    let proj_b = b.add_input("proj_b", &[TEXT_DIM]);
    let out = b.add_linear(flat, proj_w, Some(proj_b), &[1, TEXT_DIM]);

    b.build(out).expect("valid vision-text projection kernel")
}

fn vision_text_proj_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        const_weight(&[TEXT_DIM, CHANNELS], WEIGHT_MAG),
        const_weight(&[TEXT_DIM], 0.0),
    ]
}

#[test]
fn test_vision_text_linear_projection_ibp() {
    let def = build_vision_text_projection_kernel();
    let bindings = vision_text_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision-text projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Adaptive pooling CROWN bounds (CROWN)
// ===========================================================================

#[test]
fn test_adaptive_avg_pool_crown() {
    let def = build_adaptive_avg_pool_reduction_kernel();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Adaptive avg pool CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. Multi-scale feature concatenation + projection (IBP)
// ===========================================================================

/// Build multi-scale features -> concat -> projection pipeline.
/// Two feature maps at different scales are pooled, reshaped to tokens,
/// concatenated, and projected to the text embedding dimension.
#[test]
fn test_multiscale_concat_projection_ibp() {
    let scale1_tokens = 4; // from 2x2 pool
    let scale2_tokens = 1; // from 1x1 global pool
    let total_tokens = scale1_tokens + scale2_tokens; // 5

    let mut b = TensorBlockBuilder::new("dpdf_token_multiscale_proj");

    // Scale 1: [C, 8, 8] -> avg pool(4,4) -> [C, 2, 2] -> reshape [4, C]
    let feat1 = b.add_input("feat1", &[CHANNELS, SPATIAL, SPATIAL]);
    let pool1 = b.add_avg_pool_2d(feat1, 4, 4, 4, 4, 0, 0, &[CHANNELS, 2, 2]);
    let flat1 = b.add_reshape(pool1, &[CHANNELS, scale1_tokens]);
    let tok1 = b.add_transpose(flat1, &[1, 0], &[scale1_tokens, CHANNELS]);

    // Scale 2: [C, 8, 8] -> global avg pool -> [C, 1, 1] -> reshape [1, C]
    let pool2 = b.add_avg_pool_2d(
        feat1,
        SPATIAL,
        SPATIAL,
        SPATIAL,
        SPATIAL,
        0,
        0,
        &[CHANNELS, 1, 1],
    );
    let tok2 = b.add_reshape(pool2, &[scale2_tokens, CHANNELS]);

    // Concat: [4, C] + [1, C] -> [5, C]
    let concat = b.add_concat(&[tok1, tok2], 0, &[total_tokens, CHANNELS]);

    // Linear projection: [5, C] -> [5, TEXT_DIM]
    let proj_w = b.add_input("proj_w", &[TEXT_DIM, CHANNELS]);
    let out = b.add_linear(concat, proj_w, None, &[total_tokens, TEXT_DIM]);
    let def = b
        .build(out)
        .expect("valid multiscale concat + projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_weight(&[TEXT_DIM, CHANNELS], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multiscale concat + projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. Token selection/sampling pattern (IBP)
// ===========================================================================

/// Build token selection: narrow a subset of tokens from the sequence.
/// Models the ToMe-style token selection where a subset of "important" tokens
/// is selected for further processing. Uses narrow (slice) to select tokens.
#[test]
fn test_token_selection_sampling_ibp() {
    let selected_tokens = SEQ_LEN_LONG / 2; // select 4 out of 8 tokens

    let mut b = TensorBlockBuilder::new("dpdf_token_selection");
    let input = b.add_input("x", &[SEQ_LEN_LONG, HIDDEN_DIM]);
    // Select first half of tokens: narrow axis=0, start=0, len=4
    let selected = b.add_narrow(input, 0, 0, selected_tokens, &[selected_tokens, HIDDEN_DIM]);
    // Project selected tokens
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(selected, proj_w, None, &[selected_tokens, HIDDEN_DIM]);
    let def = b.build(out).expect("valid token selection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_weight(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN_LONG, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token selection/sampling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Spatial reduction via strided convolution (IBP)
// ===========================================================================

/// Build spatial reduction via strided Conv2d: [C, 8, 8] -> [C_wide, 4, 4].
/// Strided convolution is an alternative to pooling for spatial reduction,
/// used in ResNet-style downsampling and PaddleOCR backbones.
#[test]
fn test_strided_conv_spatial_reduction_ibp() {
    let stride = 2;
    let out_spatial = SPATIAL / stride; // 4

    let mut b = TensorBlockBuilder::new("dpdf_token_strided_conv_reduction");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);
    let conv_w = b.add_input("conv_w", &[CHANNELS_WIDE, CHANNELS, 3, 3]);
    // Conv2d stride=2, pad=1: [C, 8, 8] -> [C_wide, 4, 4]
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        None,
        stride,
        stride,
        1,
        1,
        &[CHANNELS_WIDE, out_spatial, out_spatial],
    );
    // ReLU activation
    let out = b.add_relu(conv_out, &[CHANNELS_WIDE, out_spatial, out_spatial]);
    let def = b.build(out).expect("valid strided conv reduction kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        const_weight(&[CHANNELS_WIDE, CHANNELS, 3, 3], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Strided conv spatial reduction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU ensures lower >= 0
    assert!(
        lo_min >= -1e-4,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 9. Token merging with attention weights (IBP)
// ===========================================================================

/// Build attention-weighted token merging: compute attention scores over
/// tokens, apply softmax to get merge weights, multiply with values, and
/// reduce to produce merged tokens.
///
/// Input: [SEQ_LEN, HIDDEN_DIM] -> Linear -> scores -> softmax -> weighted
/// sum -> [1, HIDDEN_DIM] merged token.
fn build_attention_token_merging_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_token_attn_merge");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let score_w = b.add_input("score_w", &[1, HIDDEN_DIM]);

    // Attention scores: [SEQ_LEN, HIDDEN_DIM] * [1, HIDDEN_DIM]^T -> [SEQ_LEN, 1]
    let scores = b.add_linear(input, score_w, None, &[SEQ_LEN, 1]);
    // Softmax over sequence: [SEQ_LEN, 1]
    let weights = b.add_softmax(scores, 0, &[SEQ_LEN, 1]);
    // Broadcast weights: [SEQ_LEN, 1] -> [SEQ_LEN, HIDDEN_DIM]
    let weights_bc = b.add_broadcast(weights, &[SEQ_LEN, HIDDEN_DIM]);
    // Weighted features: element-wise mul
    let weighted = b.add_binary_mul(input, weights_bc, &[SEQ_LEN, HIDDEN_DIM]);
    // Reduce sum over sequence: [SEQ_LEN, HIDDEN_DIM] -> [HIDDEN_DIM]
    let out = b.add_reduce(weighted, ReduceOp::Sum, 0, false, &[HIDDEN_DIM]);

    b.build(out).expect("valid attention token merging kernel")
}

#[test]
fn test_token_merging_attention_weights_ibp() {
    let def = build_attention_token_merging_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        const_weight(&[1, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token merging (attention weights) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Vision-text projection after spatial reduction (CROWN)
// ===========================================================================

#[test]
fn test_vision_text_projection_crown() {
    let def = build_vision_text_projection_kernel();
    let bindings = vision_text_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision-text projection CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Monotone tightening for spatial reduction pipeline (IBP)
// ===========================================================================

/// Verify that tighter input bounds produce tighter output bounds through
/// the full spatial reduction pipeline (pool -> reshape -> project).
#[test]
fn test_spatial_reduction_monotone_tightening_ibp() {
    let def = build_vision_text_projection_kernel();
    let bindings = vision_text_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: eps = 1.0
    let wide_input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    // Tight input: eps = 0.1
    let tight_input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "Spatial reduction monotone tightening: eps=1.0 width={wide_width:.6}, \
         eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tighter input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 12. Full pipeline: Conv features -> pool -> reshape -> project -> text (IBP)
// ===========================================================================

/// Build a complete vision-to-text token pipeline:
/// Conv2d -> ReLU -> GlobalAvgPool -> Reshape -> Linear(text_dim) -> Sigmoid.
///
/// This is the canonical VLM spatial reduction + projection pipeline:
/// visual features are extracted by convolution, spatially reduced by pooling,
/// and projected to the text embedding space for cross-modal alignment.
fn build_full_token_pipeline_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("dpdf_token_full_pipeline");
    let input = b.add_input("x", &[CHANNELS, SPATIAL, SPATIAL]);

    // Conv2d feature extraction: [C, 8, 8] -> [C_wide, 8, 8]
    let conv_w = b.add_input("conv_w", &[CHANNELS_WIDE, CHANNELS, 3, 3]);
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        None,
        1,
        1,
        1,
        1,
        &[CHANNELS_WIDE, SPATIAL, SPATIAL],
    );

    // ReLU activation
    let relu_out = b.add_relu(conv_out, &[CHANNELS_WIDE, SPATIAL, SPATIAL]);

    // Global avg pool: [C_wide, 8, 8] -> [C_wide, 1, 1]
    let pooled = b.add_avg_pool_2d(
        relu_out,
        SPATIAL,
        SPATIAL,
        SPATIAL,
        SPATIAL,
        0,
        0,
        &[CHANNELS_WIDE, 1, 1],
    );

    // Reshape: [C_wide, 1, 1] -> [1, C_wide]
    let flat = b.add_reshape(pooled, &[1, CHANNELS_WIDE]);

    // Linear projection to text dim: [1, C_wide] -> [1, TEXT_DIM]
    let proj_w = b.add_input("proj_w", &[TEXT_DIM, CHANNELS_WIDE]);
    let proj_b = b.add_input("proj_b", &[TEXT_DIM]);
    let proj_out = b.add_linear(flat, proj_w, Some(proj_b), &[1, TEXT_DIM]);

    // Sigmoid: bounds output to (0, 1)
    let out = b.add_sigmoid(proj_out, &[1, TEXT_DIM]);

    let def = b.build(out).expect("valid full token pipeline kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        const_weight(&[CHANNELS_WIDE, CHANNELS, 3, 3], WEIGHT_MAG),
        const_weight(&[TEXT_DIM, CHANNELS_WIDE], WEIGHT_MAG),
        const_weight(&[TEXT_DIM], 0.0),
    ];
    (def, bindings)
}

#[test]
fn test_full_pipeline_conv_pool_reshape_project_ibp() {
    let (def, bindings) = build_full_token_pipeline_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!(
        "Full pipeline (conv->pool->reshape->project->sigmoid) IBP: \
         bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(
        lo_min >= 0.0 - tol,
        "sigmoid output must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "sigmoid output must be <= 1, got {hi_max}"
    );
}
