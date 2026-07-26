// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ViT patch embedding NY composition.
//!
//! Verifies bounds propagation through the ViT patch embedding pipeline:
//!
//! 1. **Linear projection** (equivalent to Conv2d with kernel=stride=P):
//!    input [num_patches, patch_dim] -> Linear -> [num_patches, embed_dim]
//!
//! 2. **Full Conv2d pipeline** (architectural fidelity):
//!    input [3, H, W] -> Conv2d(3, D, P, stride=P) -> [D, H/P, W/P]
//!    -> reshape [D, num_patches] -> transpose [num_patches, D]
//!
//! Architecture (Dosovitskiy et al. 2020):
//! - Image is split into non-overlapping P x P patches
//! - Each patch (3 * P * P values for RGB) is linearly projected to D dims
//! - Conv2d with kernel_size=stride=P is mathematically equivalent to
//!   flattening patches and applying a linear layer
//!
//! Input bounds: image pixels in [0, 1] (normalized RGB).
//!
//! Part of #3527: ViT encoder NY compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
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
/// Flattened patch dimension: 3 * 16 * 16 = 768.
const PATCH_DIM: usize = IN_CHANNELS * PATCH_SIZE * PATCH_SIZE;
/// Embedding dimension (tiny ViT hidden size).
const EMBED_DIM: usize = 64;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a patch embedding kernel using linear projection on pre-flattened patches.
///
/// Input: `[NUM_PATCHES, PATCH_DIM]` (Variable).
/// Output: `[NUM_PATCHES, EMBED_DIM]`.
///
/// This models the Conv2d-based patch embedding as an equivalent linear
/// projection on flattened patches (mathematically identical).
fn build_patch_embedding_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_patch_embedding_linear");

    let input = b.add_input("patches", &[NUM_PATCHES, PATCH_DIM]);
    let weight = b.add_input("proj_weight", &[EMBED_DIM, PATCH_DIM]);
    let bias = b.add_input("proj_bias", &[EMBED_DIM]);

    let out = b.add_linear(input, weight, Some(bias), &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid linear patch embedding kernel")
}

/// Build a patch embedding kernel using Conv2d -> reshape -> transpose.
///
/// Input: `[3, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, EMBED_DIM]` after reshape and transpose.
///
/// Conv2d(in_channels=3, out_channels=D, kernel=P, stride=P, padding=0)
/// produces `[D, H/P, W/P]` = `[D, GRID_SIZE, GRID_SIZE]`.
/// Reshape to `[D, NUM_PATCHES]`, then transpose to `[NUM_PATCHES, D]`.
///
/// Note: TensorBlockBuilder Conv2d uses 3D input `[C, H, W]` (no batch dim),
/// matching the NY graph convention.
fn build_patch_embedding_conv2d_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_patch_embedding_conv2d");

    // Input: [3, 32, 32] image (no batch dim for TensorBlockBuilder Conv2d)
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    // Conv2d weight: [D, 3, P, P]
    let weight = b.add_input(
        "proj_weight",
        &[EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("proj_bias", &[EMBED_DIM]);

    // Conv2d: [3, 32, 32] -> [D, 2, 2]
    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[EMBED_DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [D, 2, 2] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[EMBED_DIM, NUM_PATCHES]);

    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid Conv2d patch embedding kernel")
}

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Bindings for linear patch embedding.
fn patch_embedding_linear_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, PATCH_DIM]), 0.02f32);
    let bias = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // patches [NUM_PATCHES, PATCH_DIM]
        TensorParamBinding::ConstantTensor(w), // proj_weight [EMBED_DIM, PATCH_DIM]
        TensorParamBinding::ConstantTensor(bias), // proj_bias [EMBED_DIM]
    ]
}

/// Bindings for Conv2d patch embedding.
fn patch_embedding_conv2d_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        0.02f32,
    );
    let bias = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // image [3, 32, 32]
        TensorParamBinding::ConstantTensor(w),    // proj_weight [D, 3, P, P]
        TensorParamBinding::ConstantTensor(bias), // proj_bias [D]
    ]
}

// ---------------------------------------------------------------------------
// Linear projection tests
// ---------------------------------------------------------------------------

/// Linear patch embedding TensorKernelDef validates.
#[test]
fn test_vit_patch_embedding_linear_def_validates() {
    let def = build_patch_embedding_linear_kernel();
    def.validate()
        .expect("linear patch embedding kernel should validate");
}

/// Linear patch embedding translates to NY GraphNetwork.
#[test]
fn test_vit_patch_embedding_linear_graph_builds() {
    let def = build_patch_embedding_linear_kernel();
    let bindings = patch_embedding_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("linear patch embedding graph should translate");

    assert!(
        graph.num_nodes() >= 1,
        "linear patch embedding graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through linear patch embedding with [0, 1] image input.
#[test]
fn test_vit_patch_embedding_linear_ibp_image_bounds() {
    let def = build_patch_embedding_linear_kernel();
    let bindings = patch_embedding_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Image pixels in [0, 1] flattened to patch vectors.
    let input = image_bounds_01(&[NUM_PATCHES, PATCH_DIM]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through linear patch embedding");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT patch embedding linear IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // With [0, 1] input and 0.02 weights: each output ~= sum(0.02 * pixel)
    // over PATCH_DIM=768 values, so max ~= 0.02 * 768 = 15.36 per element.
    // IBP with all-positive weights and [0, 1] input: lower >= 0 (bias=0).
    assert!(
        lo_min >= -1.0,
        "IBP lower with [0,1] input should be >= -1, got {lo_min}"
    );
    assert!(
        hi_max < 20.0,
        "IBP upper should be < 20 with small weights and [0,1] input, got {hi_max}"
    );
}

/// CROWN bounds propagate through linear patch embedding.
#[test]
fn test_vit_patch_embedding_linear_crown_propagation() {
    let def = build_patch_embedding_linear_kernel();
    let bindings = patch_embedding_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[NUM_PATCHES, PATCH_DIM]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT patch embedding linear: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Conv2d pipeline tests (Conv2d -> reshape -> transpose)
// ---------------------------------------------------------------------------

/// Conv2d patch embedding TensorKernelDef validates.
#[test]
fn test_vit_patch_embedding_conv2d_def_validates() {
    let def = build_patch_embedding_conv2d_kernel();
    def.validate()
        .expect("Conv2d patch embedding kernel should validate");
}

/// Conv2d patch embedding translates to NY GraphNetwork.
#[test]
fn test_vit_patch_embedding_conv2d_graph_builds() {
    let def = build_patch_embedding_conv2d_kernel();
    let bindings = patch_embedding_conv2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("Conv2d patch embedding graph should translate");

    // Conv2d + Reshape + Transpose = at least 3 nodes.
    assert!(
        graph.num_nodes() >= 3,
        "Conv2d patch embedding graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through Conv2d patch embedding with [0, 1] image input.
///
/// Validates the full Conv2d -> reshape -> transpose pipeline with
/// image-domain input bounds.
#[test]
fn test_vit_patch_embedding_conv2d_ibp_propagates() {
    let def = build_patch_embedding_conv2d_kernel();
    let bindings = patch_embedding_conv2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Image pixels in [0, 1]: [3, 32, 32]
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv2d patch embedding");

    // Output shape: [NUM_PATCHES, EMBED_DIM] after reshape + transpose
    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape should be [NUM_PATCHES={NUM_PATCHES}, EMBED_DIM={EMBED_DIM}], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT patch embedding Conv2d IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through Conv2d patch embedding.
#[test]
fn test_vit_patch_embedding_conv2d_crown_propagation() {
    let def = build_patch_embedding_conv2d_kernel();
    let bindings = patch_embedding_conv2d_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT patch embedding Conv2d: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Patch embedding verify and record under "vit_patch_embedding" key.
///
/// Uses the Conv2d pipeline (architecturally faithful) for recording.
#[test]
fn test_vit_patch_embedding_verify_and_record() {
    let def = build_patch_embedding_conv2d_kernel();
    let bindings = patch_embedding_conv2d_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "vit_patch_embedding");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);
}
