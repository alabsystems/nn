// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for patch embedding and image tokenization bounds.
//!
//! Verifies IBP and CROWN bound propagation through patch embedding patterns
//! used across dpdf vision models (SigLIP2, Qwen3-VL, Granite-Docling).
//! Patch embedding is the first stage of all vision transformers: a Conv2d
//! projects non-overlapping image patches to embedding vectors, followed by
//! flatten + transpose to produce a sequence of patch tokens.
//!
//! 1.  **Basic Conv2d patch projection (patch_size=14)** (IBP)
//! 2.  **Patch projection with patch_size=16** (IBP)
//! 3.  **Patch projection with patch_size=32** (IBP)
//! 4.  **Patch flatten + transpose for ViT sequence format** (IBP)
//! 5.  **Learnable position embedding addition bounds** (IBP)
//! 6.  **Sinusoidal 2D position embedding bounds** (IBP)
//! 7.  **CLS token prepend bounds propagation** (IBP)
//! 8.  **Conv2d projection with different channels (3->768 vs 3->1024)** (IBP)
//! 9.  **Patch embedding + LayerNorm composition** (IBP + CROWN)
//! 10. **Batch dimension handling in patch projection** (IBP)
//! 11. **CROWN tightness vs IBP for patch embedding** (CROWN)
//! 12. **Overlapping patch embedding (stride < patch_size)** (IBP)
//! 13. **Interpolated position embedding for variable resolution** (IBP)
//! 14. **Patch merging (downsampling via reshape+linear)** (IBP)
//! 15. **End-to-end image-to-tokens pipeline bounds** (IBP + CROWN)
//!
//! Architecture references:
//! - ViT (Dosovitskiy et al., 2020): Vision Transformer with patch embedding
//! - SigLIP2 (Zhai et al., 2023): Sigmoid-loss pre-trained ViT patch embedding
//! - Qwen3-VL (Alibaba): 3D patch embedding for video+image
//! - Swin Transformer (Liu et al., 2021): Patch merging for hierarchical ViT
//! - Granite-Docling: Document understanding with ViT vision encoder
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_H=16, IMG_W=16, IN_CHANNELS=3, EMBED_DIM=32
//!
//! Part of #4034: Compose tests for patch embedding and image tokenization bounds.

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

const IMG_H: usize = 16;
const IMG_W: usize = 16;
const IN_CHANNELS: usize = 3;
const EMBED_DIM: usize = 32;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Create image-domain input bounds: pixels in [0, 1] ± eps.
fn image_input_bounds(channels: usize, h: usize, w: usize, eps: f32) -> BoundedTensor {
    let n = channels * h * w;
    let lo = vec![0.5 - eps; n];
    let hi = vec![0.5 + eps; n];
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[channels, h, w]), lo).expect("valid lower"),
        ArrayD::from_shape_vec(IxDyn(&[channels, h, w]), hi).expect("valid upper"),
    )
    .expect("valid image bounds")
}

/// Build a Conv2d patch projection kernel.
///
/// Input: `[C_in, H, W]`. Weight: `[C_out, C_in, P, P]`.
/// Output: `[C_out, H/P, W/P]`.
fn build_patch_proj_kernel(
    name: &str,
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
    img_h: usize,
    img_w: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let out_h = img_h / patch_size;
    let out_w = img_w / patch_size;

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[in_channels, img_h, img_w]);
    let weight = b.add_input("proj_w", &[embed_dim, in_channels, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[embed_dim]);

    let out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[embed_dim, out_h, out_w],
    );
    let def = b.build(out).expect("valid patch projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[embed_dim, in_channels, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[embed_dim]), 0.0f32)),
    ];

    (def, bindings)
}

// ===========================================================================
// 1. Basic Conv2d patch projection (patch_size=14) (IBP)
// ===========================================================================

#[test]
fn test_patch_embed_conv2d_patch14_ibp() {
    // patch_size=14 requires image divisible by 14; use 28x28
    let img_h = 28;
    let img_w = 28;
    let patch_size = 14;
    let (def, bindings) = build_patch_proj_kernel(
        "dpdf_patch_embed_p14",
        IN_CHANNELS,
        EMBED_DIM,
        patch_size,
        img_h,
        img_w,
    );
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, img_h, img_w, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed p=14 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Patch projection with patch_size=16 (IBP)
// ===========================================================================

#[test]
fn test_patch_embed_conv2d_patch16_ibp() {
    let patch_size = 16;
    let (def, bindings) = build_patch_proj_kernel(
        "dpdf_patch_embed_p16",
        IN_CHANNELS,
        EMBED_DIM,
        patch_size,
        IMG_H,
        IMG_W,
    );
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed p=16 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Patch projection with patch_size=32 (IBP)
// ===========================================================================

#[test]
fn test_patch_embed_conv2d_patch32_ibp() {
    // patch_size=32 requires 32x32 image
    let img_h = 32;
    let img_w = 32;
    let patch_size = 32;
    let (def, bindings) = build_patch_proj_kernel(
        "dpdf_patch_embed_p32",
        IN_CHANNELS,
        EMBED_DIM,
        patch_size,
        img_h,
        img_w,
    );
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, img_h, img_w, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed p=32 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Patch flatten + transpose for ViT sequence format (IBP)
// ===========================================================================

/// Build Conv2d -> reshape [D, H', W'] -> [D, N_patches] -> transpose [N_patches, D].
/// This is the standard ViT patch embedding pipeline that converts spatial
/// feature maps into a sequence of patch tokens.
fn build_patch_flatten_transpose_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let patch_size = 16;
    let out_h = IMG_H / patch_size; // 1
    let out_w = IMG_W / patch_size; // 1
    let num_patches = out_h * out_w; // 1

    let mut b = TensorBlockBuilder::new("dpdf_patch_flatten_transpose");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let weight = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[EMBED_DIM]);

    // Conv2d patch projection: [C_in, H, W] -> [D, out_h, out_w]
    let conv = b.add_conv2d(
        input,
        weight,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );

    // Reshape: [D, out_h, out_w] -> [D, N_patches]
    let flat = b.add_reshape(conv, &[EMBED_DIM, num_patches]);

    // Transpose: [D, N_patches] -> [N_patches, D]
    let out = b.add_transpose(flat, &[1, 0], &[num_patches, EMBED_DIM]);

    let def = b.build(out).expect("valid flatten+transpose kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ];
    (def, bindings)
}

#[test]
fn test_patch_flatten_transpose_ibp() {
    let (def, bindings) = build_patch_flatten_transpose_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch flatten+transpose IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Learnable position embedding addition bounds (IBP)
// ===========================================================================

/// Build patch embed + learnable position embedding addition.
/// Pattern: conv2d -> flatten -> transpose -> add(pos_embed).
#[test]
fn test_learnable_position_embedding_ibp() {
    let patch_size = 8;
    let out_h = IMG_H / patch_size; // 2
    let out_w = IMG_W / patch_size; // 2
    let num_patches = out_h * out_w; // 4

    let mut b = TensorBlockBuilder::new("dpdf_patch_learned_pe");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let weight = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[EMBED_DIM]);

    // Conv2d -> reshape -> transpose
    let conv = b.add_conv2d(
        input,
        weight,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(conv, &[EMBED_DIM, num_patches]);
    let tokens = b.add_transpose(flat, &[1, 0], &[num_patches, EMBED_DIM]);

    // Learnable position embedding: [N_patches, D]
    let pos_embed = b.add_input("pos_embed", &[num_patches, EMBED_DIM]);
    let out = b.add_binary_add(tokens, pos_embed, &[num_patches, EMBED_DIM]);

    let def = b.build(out).expect("valid learned PE kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_patches, EMBED_DIM]),
            0.01f32,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Learned PE addition IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Sinusoidal 2D position embedding bounds (IBP)
// ===========================================================================

/// Build sinusoidal 2D PE tensor for spatial grids.
fn sinusoidal_pe_2d(h: usize, w: usize, d: usize) -> ArrayD<f32> {
    let half = d / 2;
    let n = h * w;
    let mut data = vec![0.0f32; n * d];
    for y in 0..h {
        for x in 0..w {
            for i in 0..half / 2 {
                let freq_y = (y as f64) / 10000.0_f64.powf(4.0 * i as f64 / d as f64);
                let freq_x = (x as f64) / 10000.0_f64.powf(4.0 * i as f64 / d as f64);
                let idx = (y * w + x) * d;
                data[idx + 2 * i] = freq_y.sin() as f32;
                data[idx + 2 * i + 1] = freq_y.cos() as f32;
                data[idx + half + 2 * i] = freq_x.sin() as f32;
                data[idx + half + 2 * i + 1] = freq_x.cos() as f32;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[n, d]), data).expect("valid 2D PE")
}

#[test]
fn test_sinusoidal_2d_pe_bounded_ibp() {
    let patch_size = 8;
    let out_h = IMG_H / patch_size; // 2
    let out_w = IMG_W / patch_size; // 2
    let num_patches = out_h * out_w; // 4

    let mut b = TensorBlockBuilder::new("dpdf_patch_sin2d_pe");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let weight = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[EMBED_DIM]);

    let conv = b.add_conv2d(
        input,
        weight,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(conv, &[EMBED_DIM, num_patches]);
    let tokens = b.add_transpose(flat, &[1, 0], &[num_patches, EMBED_DIM]);

    // Sinusoidal 2D PE: values in [-1, 1]
    let pe = b.add_input("sin2d_pe", &[num_patches, EMBED_DIM]);
    let out = b.add_binary_add(tokens, pe, &[num_patches, EMBED_DIM]);

    let def = b.build(out).expect("valid sinusoidal 2D PE kernel");

    let pe_data = sinusoidal_pe_2d(out_h, out_w, EMBED_DIM);
    // Verify PE values are in [-1, 1]
    for &v in pe_data.iter() {
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&v),
            "sinusoidal PE value {v} should be in [-1, 1]"
        );
    }

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(pe_data),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sinusoidal 2D PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. CLS token prepend bounds propagation (IBP)
// ===========================================================================

/// Build patch embed -> prepend CLS token via concat.
/// CLS token is a learnable [1, D] vector concatenated at position 0.
#[test]
fn test_cls_token_prepend_ibp() {
    let patch_size = 8;
    let out_h = IMG_H / patch_size; // 2
    let out_w = IMG_W / patch_size; // 2
    let num_patches = out_h * out_w; // 4
    let seq_with_cls = num_patches + 1; // 5

    let mut b = TensorBlockBuilder::new("dpdf_patch_cls_prepend");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let weight = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[EMBED_DIM]);

    let conv = b.add_conv2d(
        input,
        weight,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(conv, &[EMBED_DIM, num_patches]);
    let tokens = b.add_transpose(flat, &[1, 0], &[num_patches, EMBED_DIM]);

    // CLS token: [1, D]
    let cls_token = b.add_input("cls_token", &[1, EMBED_DIM]);

    // Concat CLS + patch tokens along sequence axis -> [N_patches+1, D]
    let out = b.add_concat(&[cls_token, tokens], 0, &[seq_with_cls, EMBED_DIM]);

    let def = b.build(out).expect("valid CLS prepend kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, EMBED_DIM]), 0.01f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("CLS token prepend IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Conv2d projection with different channels (3->768 vs 3->1024) (IBP)
// ===========================================================================

#[test]
fn test_patch_embed_different_channels_ibp() {
    let patch_size = 16;
    // Test 3->768 (SigLIP2-style)
    // Use smaller dims for fast verification: 3->64 and 3->128
    let embed_768 = 64;
    let embed_1024 = 128;

    let (def_768, bindings_768) = build_patch_proj_kernel(
        "dpdf_patch_embed_768",
        IN_CHANNELS,
        embed_768,
        patch_size,
        IMG_H,
        IMG_W,
    );
    let (def_1024, bindings_1024) = build_patch_proj_kernel(
        "dpdf_patch_embed_1024",
        IN_CHANNELS,
        embed_1024,
        patch_size,
        IMG_H,
        IMG_W,
    );

    let graph_768 = tensor_kernel_to_graph(&def_768, &bindings_768).expect("graph 768");
    let graph_1024 = tensor_kernel_to_graph(&def_1024, &bindings_1024).expect("graph 1024");

    let input_768 = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);
    let input_1024 = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output_768 = graph_768.propagate_ibp(&input_768).expect("IBP 768");
    let output_1024 = graph_1024.propagate_ibp(&input_1024).expect("IBP 1024");

    assert_bounds_valid(&output_768);
    assert_bounds_valid(&output_1024);

    let width_768 = bound_width(&output_768);
    let width_1024 = bound_width(&output_1024);
    eprintln!(
        "Channel comparison IBP: embed_dim={embed_768} width={width_768:.6}, \
         embed_dim={embed_1024} width={width_1024:.6}"
    );
    assert!(width_768.is_finite(), "768 width must be finite");
    assert!(width_1024.is_finite(), "1024 width must be finite");
}

// ===========================================================================
// 9. Patch embedding + LayerNorm composition (IBP + CROWN)
// ===========================================================================

fn build_patch_embed_layernorm_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let patch_size = 8;
    let out_h = IMG_H / patch_size; // 2
    let out_w = IMG_W / patch_size; // 2
    let num_patches = out_h * out_w; // 4

    let mut b = TensorBlockBuilder::new("dpdf_patch_embed_layernorm");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let proj_w = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let proj_b = b.add_input("proj_b", &[EMBED_DIM]);

    let conv = b.add_conv2d(
        input,
        proj_w,
        Some(proj_b),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );
    let flat = b.add_reshape(conv, &[EMBED_DIM, num_patches]);
    let tokens = b.add_transpose(flat, &[1, 0], &[num_patches, EMBED_DIM]);

    // LayerNorm over embedding dimension (axis=1)
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_b", &[EMBED_DIM]);
    let out = b.add_layer_norm(tokens, ln_eps, 1, ln_w, ln_b, &[num_patches, EMBED_DIM]);

    let def = b.build(out).expect("valid patch embed + LN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ];
    (def, bindings)
}

#[test]
fn test_patch_embed_layernorm_ibp() {
    let (def, bindings) = build_patch_embed_layernorm_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed + LayerNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_patch_embed_layernorm_crown() {
    let (def, bindings) = build_patch_embed_layernorm_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.25);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Patch embed + LayerNorm CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 10. Batch dimension handling in patch projection (IBP)
// ===========================================================================

/// Test patch projection with a batch dimension.
/// Input: [B*C_in, H, W] where B patches are stacked along channel dim.
/// This simulates batched inference through the Conv2d patch embedding.
#[test]
fn test_patch_embed_batch_dim_ibp() {
    let batch = 2;
    let patch_size = 8;
    let out_h = IMG_H / patch_size; // 2
    let out_w = IMG_W / patch_size; // 2
                                    // Model batch as extra channels: batch*C_in
    let batched_channels = batch * IN_CHANNELS;

    let mut b = TensorBlockBuilder::new("dpdf_patch_embed_batch");
    let input = b.add_input("x", &[batched_channels, IMG_H, IMG_W]);
    let weight = b.add_input(
        "proj_w",
        &[EMBED_DIM, batched_channels, patch_size, patch_size],
    );
    let bias = b.add_input("proj_b", &[EMBED_DIM]);

    let out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );
    let def = b.build(out).expect("valid batched patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, batched_channels, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(batched_channels, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Batched patch embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. CROWN tightness vs IBP for patch embedding (CROWN)
// ===========================================================================

#[test]
fn test_patch_embed_crown_tightness() {
    let patch_size = 16;
    let (def, bindings) = build_patch_proj_kernel(
        "dpdf_patch_embed_crown_tight",
        IN_CHANNELS,
        EMBED_DIM,
        patch_size,
        IMG_H,
        IMG_W,
    );
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.25);

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&crown_output);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Patch embed CROWN tightness: method={method:?}, ibp_width={ibp_width:.6}, \
         crown_width={crown_width:.6}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(crown_width.is_finite(), "CROWN width must be finite");
}

// ===========================================================================
// 12. Overlapping patch embedding (stride < patch_size) (IBP)
// ===========================================================================

/// Overlapping patches: stride < patch_size. Used in some dense prediction
/// models where spatial overlap improves locality.
#[test]
fn test_overlapping_patch_embed_ibp() {
    let patch_size = 8;
    let stride = 4; // 50% overlap
    let out_h = (IMG_H - patch_size) / stride + 1; // (16 - 8) / 4 + 1 = 3
    let out_w = (IMG_W - patch_size) / stride + 1; // 3

    let mut b = TensorBlockBuilder::new("dpdf_patch_embed_overlap");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let weight = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let bias = b.add_input("proj_b", &[EMBED_DIM]);

    let out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        stride,
        stride,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );
    let def = b.build(out).expect("valid overlapping patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Overlapping patch embed (stride={stride}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Interpolated position embedding for variable resolution (IBP)
// ===========================================================================

/// Simulates position embedding interpolation: learned PE at base resolution
/// linearly interpolated to a different resolution. Modeled as a Linear
/// projection of the PE weight (downsampling via learned interpolation weights).
#[test]
fn test_interpolated_position_embedding_ibp() {
    let base_patches = 4; // base resolution: 4 patches
    let target_patches = 2; // target: 2 patches (downsampled)

    let mut b = TensorBlockBuilder::new("dpdf_patch_interp_pe");

    // Patch tokens: [target_patches, D]
    let tokens = b.add_input("tokens", &[target_patches, EMBED_DIM]);

    // Interpolation: learned PE [base, D] -> interp weight [target, base] -> matmul -> [target, D].
    // This is a plain matmul `interp_w @ base_pe` (contracting the shared `base`
    // axis), not a Linear (which would contract `base_pe`'s last axis D against
    // interp_w's last axis base — a feature mismatch).
    let base_pe = b.add_input("base_pe", &[base_patches, EMBED_DIM]);
    let interp_w = b.add_input("interp_w", &[target_patches, base_patches]);
    let interp_pe = b.add_matmul(
        interp_w,
        base_pe,
        false,
        None,
        &[target_patches, EMBED_DIM],
    );

    // Add interpolated PE to tokens
    let out = b.add_binary_add(tokens, interp_pe, &[target_patches, EMBED_DIM]);

    let def = b.build(out).expect("valid interpolated PE kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[base_patches, EMBED_DIM]),
            0.01f32,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[target_patches, base_patches]),
            1.0 / base_patches as f32,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[target_patches, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Interpolated PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. Patch merging (downsampling via reshape+linear) (IBP)
// ===========================================================================

/// Swin-Transformer patch merging: concatenate 2x2 neighboring patches,
/// then project with a linear layer to halve the spatial resolution and
/// double the channels.
///
/// Input: [4*D] (4 neighboring patches concatenated) -> Linear -> [2*D]
#[test]
fn test_patch_merging_ibp() {
    let num_patches = 4; // 2x2 patches before merging
    let merged_patches = 1; // After 2x2 merge: 1 patch
    let concat_dim = num_patches * EMBED_DIM; // 4 patches concatenated
    let merged_dim = EMBED_DIM * 2; // Output dimension after merging

    let mut b = TensorBlockBuilder::new("dpdf_patch_merging");

    // Input: 4 patches flattened as [merged_patches, 4*D]
    let input = b.add_input("x", &[merged_patches, concat_dim]);

    // Linear projection: [merged_patches, 4*D] -> [merged_patches, 2*D]
    let merge_w = b.add_input("merge_w", &[merged_dim, concat_dim]);
    let merge_b = b.add_input("merge_b", &[merged_dim]);
    let out = b.add_linear(input, merge_w, Some(merge_b), &[merged_patches, merged_dim]);

    let def = b.build(out).expect("valid patch merging kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[merged_dim, concat_dim]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[merged_dim]), 0.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[merged_patches, concat_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch merging IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. End-to-end image-to-tokens pipeline bounds (IBP + CROWN)
// ===========================================================================

/// Full pipeline: Conv2d patch embed -> flatten -> transpose -> PE addition
/// -> LayerNorm -> output tokens.
fn build_full_image_to_tokens_kernel() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let patch_size = 8;
    let out_h = IMG_H / patch_size; // 2
    let out_w = IMG_W / patch_size; // 2
    let num_patches = out_h * out_w; // 4

    let mut b = TensorBlockBuilder::new("dpdf_patch_e2e_pipeline");
    let input = b.add_input("x", &[IN_CHANNELS, IMG_H, IMG_W]);
    let proj_w = b.add_input("proj_w", &[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]);
    let proj_b = b.add_input("proj_b", &[EMBED_DIM]);

    // Conv2d patch projection
    let conv = b.add_conv2d(
        input,
        proj_w,
        Some(proj_b),
        patch_size,
        patch_size,
        0,
        0,
        &[EMBED_DIM, out_h, out_w],
    );

    // Flatten + transpose: [D, H', W'] -> [D, N] -> [N, D]
    let flat = b.add_reshape(conv, &[EMBED_DIM, num_patches]);
    let tokens = b.add_transpose(flat, &[1, 0], &[num_patches, EMBED_DIM]);

    // Position embedding addition
    let pos_embed = b.add_input("pos_embed", &[num_patches, EMBED_DIM]);
    let with_pe = b.add_binary_add(tokens, pos_embed, &[num_patches, EMBED_DIM]);

    // LayerNorm
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_b", &[EMBED_DIM]);
    let out = b.add_layer_norm(with_pe, ln_eps, 1, ln_w, ln_b, &[num_patches, EMBED_DIM]);

    let def = b.build(out).expect("valid end-to-end pipeline kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        // proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, IN_CHANNELS, patch_size, patch_size]),
            WEIGHT_MAG,
        )),
        // proj_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
        // pos_embed
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_patches, EMBED_DIM]),
            0.01f32,
        )),
        // ln_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        // ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32)),
        // ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ];
    (def, bindings)
}

#[test]
fn test_e2e_image_to_tokens_ibp() {
    let (def, bindings) = build_full_image_to_tokens_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("End-to-end image-to-tokens IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_e2e_image_to_tokens_crown() {
    let (def, bindings) = build_full_image_to_tokens_kernel();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_input_bounds(IN_CHANNELS, IMG_H, IMG_W, 0.25);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "End-to-end image-to-tokens CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
