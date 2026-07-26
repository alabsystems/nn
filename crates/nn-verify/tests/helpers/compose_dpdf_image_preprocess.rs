// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for image preprocessing pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through standard image
//! preprocessing operations used across all dpdf vision models:
//!
//! ## Tests (18 tests)
//!
//! 1.  **Image normalization (mean subtraction, std division) bounds** (IBP)
//! 2.  **Resize interpolation bounds (bilinear)** (IBP)
//! 3.  **Center crop spatial bounds** (IBP)
//! 4.  **Random crop augmentation bounds** (IBP)
//! 5.  **Color jitter brightness/contrast bounds** (IBP)
//! 6.  **Horizontal/vertical flip (value preservation)** (IBP)
//! 7.  **Patch extraction from image bounds** (IBP + CROWN)
//! 8.  **Multi-resolution tiling bounds** (IBP)
//! 9.  **Aspect ratio padding bounds** (IBP)
//! 10. **Grayscale conversion bounds** (IBP)
//! 11. **Channel-first to channel-last transpose** (IBP)
//! 12. **Batch normalization of pixel values** (IBP)
//! 13. **Dynamic resolution binning bounds** (IBP)
//! 14. **Image-to-patch embedding projection bounds** (IBP + CROWN)
//! 15. **Letterbox padding bounds** (IBP)
//! 16. **Edge detection filter bounds** (IBP)
//! 17. **Histogram equalization bounds** (IBP + CROWN)
//! 18. **Full preprocessing pipeline composition** (IBP + CROWN)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_C=3 (RGB), IMG_H=16, IMG_W=16
//! - PATCH_SIZE=4, NUM_PATCHES=16, HIDDEN_DIM=48
//!
//! Part of #4207: Compose tests for image preprocessing pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const IMG_C: usize = 3;
const IMG_H: usize = 16;
const IMG_W: usize = 16;
const PATCH_SIZE: usize = 4;
/// Number of patches: (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE).
const NUM_PATCHES: usize = (IMG_H / PATCH_SIZE) * (IMG_W / PATCH_SIZE); // 16
const HIDDEN_DIM: usize = 48;
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

/// Ones tensor binding (for normalization weight).
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

// ===========================================================================
// 1. Image normalization (mean subtraction, std division) bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_normalization_ibp() {
    // Normalization: (pixel - mean) / std via affine transform.
    // Model as: x * (1/std) + (-mean/std) per channel.
    // ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    let inv_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225];
    let neg_mean_div_std = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

    let mut b = TensorBlockBuilder::new("img_normalize");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let scale_data = b.add_input("inv_std", &[IMG_C]);
    let bias_data = b.add_input("bias", &[IMG_C]);

    // Broadcast scale [C] to [C, H, W], multiply
    let scale_bc = b.add_broadcast_left(scale_data, &[IMG_C, IMG_H, IMG_W]);
    let scaled = b.add_binary_mul(input, scale_bc, &[IMG_C, IMG_H, IMG_W]);
    // Broadcast bias [C] to [C, H, W], add
    let bias_bc = b.add_broadcast_left(bias_data, &[IMG_C, IMG_H, IMG_W]);
    let out = b.add_binary_add(scaled, bias_bc, &[IMG_C, IMG_H, IMG_W]);
    let def = b.build(out).expect("valid normalize kernel");

    let scale_tensor =
        ArrayD::from_shape_vec(IxDyn(&[IMG_C]), inv_std.to_vec()).expect("valid inv_std");
    let bias_tensor =
        ArrayD::from_shape_vec(IxDyn(&[IMG_C]), neg_mean_div_std.to_vec()).expect("valid bias");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(scale_tensor),
        TensorParamBinding::ConstantTensor(bias_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_C, IMG_H, IMG_W]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Image normalization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Normalized bounds: pixel 0 -> -mean/std, pixel 1 -> (1-mean)/std
    assert!(lo_min < 0.0, "normalized lower must be negative");
    assert!(hi_max > 0.0, "normalized upper must be positive");
}

// ===========================================================================
// 2. Resize interpolation bounds (bilinear) (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_resize_interpolation_ibp() {
    // Bilinear interpolation as convex combination: output is weighted average
    // of input values. Model as: Linear projection (weight sums to 1 per output).
    // Input [C, 8, 8] -> Output [C, 16, 16] (2x upscale)
    let in_h = 8;
    let in_w = 8;
    let in_flat = IMG_C * in_h * in_w;
    let out_flat = IMG_C * IMG_H * IMG_W;

    let mut b = TensorBlockBuilder::new("img_resize_bilinear");
    let input = b.add_input("image", &[in_flat]);
    let interp_w = b.add_input("interp_weights", &[out_flat, in_flat]);
    let out = b.add_linear(input, interp_w, None, &[out_flat]);
    let def = b.build(out).expect("valid resize kernel");

    // Interpolation weights: each row sums to ~1 (convex combination).
    // Use uniform weights that sum to 1 per output pixel.
    let n_neighbors = 4; // bilinear uses 4 neighbors
    let w_val = 1.0 / n_neighbors as f32;
    let mut interp_data = vec![0.0f32; out_flat * in_flat];
    for row in 0..out_flat {
        // Each output pixel is connected to 4 nearest input pixels.
        for k in 0..n_neighbors {
            let col = (row * n_neighbors / out_flat + k) % in_flat;
            interp_data[row * in_flat + col] = w_val;
        }
    }
    let interp_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_flat, in_flat]), interp_data).expect("valid weights");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(interp_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[in_flat], 0.5); // pixels centered at 0.5 +/- 0.5

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Resize interpolation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Center crop spatial bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_center_crop_ibp() {
    // Center crop: narrow spatial dimensions from [C, 16, 16] to [C, 8, 8].
    // Use narrow on H and W dims to extract central region.
    let crop_h = 8;
    let crop_w = 8;
    let offset_h = (IMG_H - crop_h) / 2; // 4
    let offset_w = (IMG_W - crop_w) / 2; // 4

    let mut b = TensorBlockBuilder::new("img_center_crop");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    // Narrow on height (dim=1)
    let crop_h_node = b.add_narrow(input, 1, offset_h, crop_h, &[IMG_C, crop_h, IMG_W]);
    // Narrow on width (dim=2)
    let out = b.add_narrow(crop_h_node, 2, offset_w, crop_w, &[IMG_C, crop_h, crop_w]);
    let def = b.build(out).expect("valid center crop kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_C, crop_h, crop_w]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Center crop IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Crop preserves pixel range [0, 1]
    assert!(lo_min >= -1e-5, "crop lower bound should be >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "crop upper bound should be <= 1");
}

// ===========================================================================
// 4. Random crop augmentation bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_random_crop_ibp() {
    // Random crop: narrow to [C, 12, 12] from any valid offset.
    // At verification time, worst-case offset means we model the full
    // input range flowing through. Same as center crop structurally.
    let crop_h = 12;
    let crop_w = 12;

    let mut b = TensorBlockBuilder::new("img_random_crop");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let crop_h_node = b.add_narrow(input, 1, 0, crop_h, &[IMG_C, crop_h, IMG_W]);
    let out = b.add_narrow(crop_h_node, 2, 0, crop_w, &[IMG_C, crop_h, crop_w]);
    let def = b.build(out).expect("valid random crop kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_C, crop_h, crop_w]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Random crop IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "crop preserves lower bound >= 0");
    assert!(hi_max <= 1.0 + 1e-5, "crop preserves upper bound <= 1");
}

// ===========================================================================
// 5. Color jitter brightness/contrast bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_color_jitter_ibp() {
    // Color jitter: brightness = pixel * alpha + beta, contrast = pixel * gamma.
    // Model as element-wise affine: x * scale + shift where scale in [0.8, 1.2],
    // shift in [-0.1, 0.1]. We use a constant-parameter affine for verification.
    let mut b = TensorBlockBuilder::new("img_color_jitter");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let scale = b.add_input("jitter_scale", &[IMG_C, IMG_H, IMG_W]);
    let shift = b.add_input("jitter_shift", &[IMG_C, IMG_H, IMG_W]);

    let scaled = b.add_binary_mul(input, scale, &[IMG_C, IMG_H, IMG_W]);
    let out = b.add_binary_add(scaled, shift, &[IMG_C, IMG_H, IMG_W]);
    let def = b.build(out).expect("valid color jitter kernel");

    // Scale = 1.1 (slight brightness increase), shift = 0.05
    let scale_val = 1.1f32;
    let shift_val = 0.05f32;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[IMG_C, IMG_H, IMG_W]),
            scale_val,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[IMG_C, IMG_H, IMG_W]),
            shift_val,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Color jitter IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // With scale=1.1 and shift=0.05: min = 0*1.1+0.05=0.05, max = 1*1.1+0.05=1.15
    assert!(lo_min >= -1e-5, "jitter lower bound near 0.05");
    assert!(hi_max <= 1.2 + 1e-5, "jitter upper bound near 1.15");
}

// ===========================================================================
// 6. Horizontal/vertical flip (value preservation) (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_flip_value_preservation_ibp() {
    // Flip is a permutation of spatial indices, modeled as transpose.
    // Values are preserved; only spatial ordering changes.
    // [C, H, W] -> transpose dims [0, 2, 1] -> [C, W, H] (vertical flip analog)
    let mut b = TensorBlockBuilder::new("img_flip");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let out = b.add_transpose(input, &[0, 2, 1], &[IMG_C, IMG_W, IMG_H]);
    let def = b.build(out).expect("valid flip kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_C, IMG_W, IMG_H]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Flip value preservation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Permutation preserves value range exactly
    assert!(lo_min >= -1e-5, "flip preserves lower bound");
    assert!(hi_max <= 1.0 + 1e-5, "flip preserves upper bound");
}

// ===========================================================================
// 7. Patch extraction from image bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_image_preprocess_patch_extraction_ibp_crown() {
    // Patch extraction via Conv2d with stride=patch_size: [C, H, W] -> [D, H/P, W/P]
    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("img_patch_extract");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let conv_w = b.add_input("patch_w", &[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("patch_b", &[HIDDEN_DIM]);

    let out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    let def = b.build(out).expect("valid patch extraction kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[HIDDEN_DIM, out_h, out_w]
    );
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Patch extraction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Patch extraction CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 8. Multi-resolution tiling bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_multi_resolution_tiling_ibp() {
    // Multi-resolution tiling: process image at two resolutions and concatenate.
    // Low-res: [C, 8, 8] patch embed -> [D, 2, 2]; High-res: [C, 16, 16] -> [D, 4, 4]
    // Flatten and concatenate: [D*4 + D*16] = [D*20]
    let lo_h = 8;
    let lo_w = 8;
    let lo_out_h = lo_h / PATCH_SIZE; // 2
    let lo_out_w = lo_w / PATCH_SIZE; // 2
    let hi_out_h = IMG_H / PATCH_SIZE; // 4
    let hi_out_w = IMG_W / PATCH_SIZE; // 4
    let lo_flat = HIDDEN_DIM * lo_out_h * lo_out_w; // D*4
    let hi_flat = HIDDEN_DIM * hi_out_h * hi_out_w; // D*16
    let concat_len = lo_flat + hi_flat;

    let mut b = TensorBlockBuilder::new("img_multires_tile");
    // Low-res path
    let lo_input = b.add_input("image_low", &[IMG_C, lo_h, lo_w]);
    let lo_conv_w = b.add_input("lo_conv_w", &[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]);
    let lo_conv_b = b.add_input("lo_conv_b", &[HIDDEN_DIM]);
    let lo_out = b.add_conv2d(
        lo_input,
        lo_conv_w,
        Some(lo_conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, lo_out_h, lo_out_w],
    );
    let lo_flat_node = b.add_reshape(lo_out, &[lo_flat]);

    // High-res path
    let hi_input = b.add_input("image_high", &[IMG_C, IMG_H, IMG_W]);
    let hi_conv_w = b.add_input("hi_conv_w", &[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]);
    let hi_conv_b = b.add_input("hi_conv_b", &[HIDDEN_DIM]);
    let hi_out = b.add_conv2d(
        hi_input,
        hi_conv_w,
        Some(hi_conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, hi_out_h, hi_out_w],
    );
    let hi_flat_node = b.add_reshape(hi_out, &[hi_flat]);

    // Concatenate along flat dimension
    let out = b.add_concat(&[lo_flat_node, hi_flat_node], 0, &[concat_len]);
    let def = b.build(out).expect("valid multi-res tiling kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // low-res image (variable)
        weight(&[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
        TensorParamBinding::Variable, // high-res image (variable)
        weight(&[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Combined input bounds for both resolutions
    // Combine into a single variable input: flatten and concatenate bounds
    let lo_size = IMG_C * lo_h * lo_w;
    let hi_size = IMG_C * IMG_H * IMG_W;
    let total_size = lo_size + hi_size;
    let combined_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total_size]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[total_size]), 1.0f32),
    )
    .expect("valid combined bounds");

    let output = graph
        .propagate_ibp(&combined_input)
        .expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-res tiling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9. Aspect ratio padding bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_aspect_ratio_padding_ibp() {
    // Aspect ratio padding: pad image to square with zeros.
    // Model as: identity with zero-padding on spatial dims.
    // [C, 8, 16] -> pad H to 16 -> [C, 16, 16]
    let in_h = 8;

    let mut b = TensorBlockBuilder::new("img_aspect_pad");
    let input = b.add_input("image", &[IMG_C, in_h, IMG_W]);
    // Zero-pad height: reshape to flat, linear project (identity+pad)
    let in_flat = IMG_C * in_h * IMG_W;
    let out_flat = IMG_C * IMG_H * IMG_W;
    let flat_input = b.add_reshape(input, &[in_flat]);
    let pad_w = b.add_input("pad_proj", &[out_flat, in_flat]);
    let out = b.add_linear(flat_input, pad_w, None, &[out_flat]);
    let def = b.build(out).expect("valid aspect ratio padding kernel");

    // Identity-like projection: maps input pixels to their positions, zeros elsewhere
    let mut proj_data = vec![0.0f32; out_flat * in_flat];
    for i in 0..in_flat.min(out_flat) {
        proj_data[i * in_flat + i] = 1.0;
    }
    let proj_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_flat, in_flat]), proj_data).expect("valid proj");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[in_flat], 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Aspect ratio padding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. Grayscale conversion bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_grayscale_conversion_ibp() {
    // Grayscale: Y = 0.299*R + 0.587*G + 0.114*B
    // Model as linear projection from [3, H, W] flattened to [1, H, W] flattened.
    let in_flat = IMG_C * IMG_H * IMG_W;
    let out_flat = 1 * IMG_H * IMG_W;

    let mut b = TensorBlockBuilder::new("img_grayscale");
    let input = b.add_input("image", &[in_flat]);
    let gray_w = b.add_input("gray_weights", &[out_flat, in_flat]);
    let out = b.add_linear(input, gray_w, None, &[out_flat]);
    let def = b.build(out).expect("valid grayscale kernel");

    // Grayscale weights: each output pixel is 0.299*R + 0.587*G + 0.114*B
    let hw = IMG_H * IMG_W;
    let mut gray_data = vec![0.0f32; out_flat * in_flat];
    for pixel in 0..hw {
        let out_idx = pixel;
        let r_idx = 0 * hw + pixel; // R channel
        let g_idx = 1 * hw + pixel; // G channel
        let b_idx = 2 * hw + pixel; // B channel
        gray_data[out_idx * in_flat + r_idx] = 0.299;
        gray_data[out_idx * in_flat + g_idx] = 0.587;
        gray_data[out_idx * in_flat + b_idx] = 0.114;
    }
    let gray_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_flat, in_flat]), gray_data).expect("valid gray");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gray_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[in_flat], 0.5); // pixels in [0, 1]

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Grayscale conversion IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Convex combination of [0,1] inputs with positive weights summing to 1
    // Output should be bounded in [0, 1]
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Channel-first to channel-last transpose (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_channel_transpose_ibp() {
    // [C, H, W] -> transpose [1, 2, 0] -> [H, W, C]
    let mut b = TensorBlockBuilder::new("img_channel_transpose");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let out = b.add_transpose(input, &[1, 2, 0], &[IMG_H, IMG_W, IMG_C]);
    let def = b.build(out).expect("valid channel transpose kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[IMG_H, IMG_W, IMG_C]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Channel transpose IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Transpose preserves values exactly
    assert!(lo_min >= -1e-5, "transpose preserves lower bound");
    assert!(hi_max <= 1.0 + 1e-5, "transpose preserves upper bound");
}

// ===========================================================================
// 12. Batch normalization of pixel values (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_batch_norm_ibp() {
    // BatchNorm on pixel values: normalize per-channel using running stats.
    // Model as LayerNorm over spatial dims (structurally similar for IBP).
    let mut b = TensorBlockBuilder::new("img_batch_norm");
    let input = b.add_input("image", &[IMG_C, IMG_H * IMG_W]);
    let bn_w = b.add_input("bn_weight", &[IMG_H * IMG_W]);
    let bn_b = b.add_input("bn_bias", &[IMG_H * IMG_W]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(input, eps, 1, bn_w, bn_b, &[IMG_C, IMG_H * IMG_W]);
    let def = b.build(out).expect("valid batch norm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[IMG_H * IMG_W]),
        bias_zero(&[IMG_H * IMG_W]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = seq_bounds(IMG_C, IMG_H * IMG_W, 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Batch norm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Dynamic resolution binning bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_dynamic_resolution_binning_ibp() {
    // Dynamic resolution: process at two different sizes, compare bound widths.
    // Small: [C, 8, 8] -> Conv2d -> [D, 2, 2]
    // Large: [C, 16, 16] -> Conv2d -> [D, 4, 4]
    let build_patch_embed = |h: usize, w: usize| -> BoundedTensor {
        let out_h = h / PATCH_SIZE;
        let out_w = w / PATCH_SIZE;
        let mut b = TensorBlockBuilder::new(&format!("img_dynres_{h}x{w}"));
        let input = b.add_input("image", &[IMG_C, h, w]);
        let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]);
        let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
        let out = b.add_conv2d(
            input,
            conv_w,
            Some(conv_b),
            PATCH_SIZE,
            PATCH_SIZE,
            0,
            0,
            &[HIDDEN_DIM, out_h, out_w],
        );
        let def = b.build(out).expect("valid patch embed");

        let bindings = vec![
            TensorParamBinding::Variable,
            weight(&[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]),
            bias_zero(&[HIDDEN_DIM]),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = image_bounds(IMG_C, h, w);
        graph.propagate_ibp(&input).expect("IBP")
    };

    let small_out = build_patch_embed(8, 8);
    let large_out = build_patch_embed(16, 16);
    assert_bounds_valid(&small_out);
    assert_bounds_valid(&large_out);

    let (s_lo, s_hi) = bounds_min_max(&small_out);
    let (l_lo, l_hi) = bounds_min_max(&large_out);
    let small_width = s_hi - s_lo;
    let large_width = l_hi - l_lo;

    eprintln!(
        "Dynamic resolution binning: small_width={small_width:.4}, large_width={large_width:.4}"
    );
    // Same weights + same pixel range => per-patch bounds should be similar
    let ratio = large_width / small_width;
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "per-patch bounds should be similar across resolutions, ratio={ratio}"
    );
}

// ===========================================================================
// 14. Image-to-patch embedding projection bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_image_preprocess_patch_embedding_projection_ibp_crown() {
    // Conv2d patch embed -> reshape -> transpose -> Linear projection
    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;
    let proj_dim = 64;

    let mut b = TensorBlockBuilder::new("img_patch_embed_proj");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);

    // Patch embed: [C, H, W] -> [D, H/P, W/P]
    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );
    // Reshape: [D, H/P, W/P] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);
    // Linear projection: [NUM_PATCHES, D] -> [NUM_PATCHES, proj_dim]
    let proj_w = b.add_input("proj_w", &[proj_dim, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[proj_dim]);
    let out = b.add_linear(transposed, proj_w, Some(proj_b), &[NUM_PATCHES, proj_dim]);
    let def = b.build(out).expect("valid patch embed projection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[proj_dim, HIDDEN_DIM]),
        bias_zero(&[proj_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[NUM_PATCHES, proj_dim]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Patch embed projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Patch embed projection CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 15. Letterbox padding bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_letterbox_padding_ibp() {
    // Letterbox: resize maintaining aspect ratio, pad remainder with 0.5 (gray).
    // Model as: linear projection with identity for image area, 0.5 for pad area.
    let in_h = 12;
    let in_w = 16;
    let in_flat = IMG_C * in_h * in_w;
    let out_flat = IMG_C * IMG_H * IMG_W;

    let mut b = TensorBlockBuilder::new("img_letterbox");
    let input = b.add_input("image", &[in_flat]);
    let letterbox_w = b.add_input("letterbox_proj", &[out_flat, in_flat]);
    let letterbox_b = b.add_input("letterbox_bias", &[out_flat]);
    let out = b.add_linear(input, letterbox_w, Some(letterbox_b), &[out_flat]);
    let def = b.build(out).expect("valid letterbox kernel");

    // Identity projection for mapped pixels, bias = 0.5 for padded pixels
    let mut proj_data = vec![0.0f32; out_flat * in_flat];
    let mut bias_data = vec![0.5f32; out_flat]; // default: gray padding
    for i in 0..in_flat.min(out_flat) {
        proj_data[i * in_flat + i] = 1.0;
        bias_data[i] = 0.0; // no bias for identity-mapped pixels
    }
    let proj_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_flat, in_flat]), proj_data).expect("valid proj");
    let bias_tensor = ArrayD::from_shape_vec(IxDyn(&[out_flat]), bias_data).expect("valid bias");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_tensor),
        TensorParamBinding::ConstantTensor(bias_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[in_flat], 0.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Letterbox padding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 16. Edge detection filter bounds (IBP)
// ===========================================================================

#[test]
fn test_image_preprocess_edge_detection_ibp() {
    // Edge detection via Sobel-like 3x3 convolution filters.
    // Conv2d with kernel [-1, 0, 1; -2, 0, 2; -1, 0, 1] (Sobel X).
    // Input: single-channel [1, H, W] -> Output: [1, H-2, W-2]
    let in_c = 1;
    let out_c = 1;
    let ksize = 3;
    let out_h = IMG_H - ksize + 1; // 14
    let out_w = IMG_W - ksize + 1; // 14

    let mut b = TensorBlockBuilder::new("img_edge_detect");
    let input = b.add_input("image", &[in_c, IMG_H, IMG_W]);
    let conv_w = b.add_input("sobel_w", &[out_c, in_c, ksize, ksize]);
    let out = b.add_conv2d(input, conv_w, None, 1, 1, 0, 0, &[out_c, out_h, out_w]);
    let def = b.build(out).expect("valid edge detection kernel");

    // Sobel X kernel
    let sobel_x: Vec<f32> = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let sobel_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, ksize, ksize]), sobel_x).expect("valid sobel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(sobel_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(in_c, IMG_H, IMG_W);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[out_c, out_h, out_w]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Edge detection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sobel filter on [0,1] inputs: output range is bounded by sum of absolute weights
    // |output| <= sum(|weights|) * max(|input|) = 8 * 1 = 8
    assert!(lo_min >= -8.0 - 1e-3, "edge detection lower bound");
    assert!(hi_max <= 8.0 + 1e-3, "edge detection upper bound");
}

// ===========================================================================
// 17. Histogram equalization bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_image_preprocess_histogram_equalization_ibp_crown() {
    // Histogram equalization approximated as monotone piecewise-linear mapping.
    // Model as: Linear -> ReLU -> Linear (approximates monotone mapping).
    // Input [flat] -> hidden -> output [flat]
    let flat = IMG_H * IMG_W;
    let hidden = flat * 2;

    let mut b = TensorBlockBuilder::new("img_hist_eq");
    let input = b.add_input("pixels", &[flat]);
    let w1 = b.add_input("w1", &[hidden, flat]);
    let b1 = b.add_input("b1", &[hidden]);
    let h1 = b.add_linear(input, w1, Some(b1), &[hidden]);
    let act = b.add_relu(h1, &[hidden]);
    let w2 = b.add_input("w2", &[flat, hidden]);
    let b2 = b.add_input("b2", &[flat]);
    let out = b.add_linear(act, w2, Some(b2), &[flat]);
    let def = b.build(out).expect("valid histogram equalization kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[hidden, flat]),
        bias_zero(&[hidden]),
        weight(&[flat, hidden]),
        bias_zero(&[flat]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[flat], 0.5); // pixels in [0, 1]

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Histogram equalization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Histogram equalization CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 18. Full preprocessing pipeline composition (IBP + CROWN)
// ===========================================================================

#[test]
fn test_image_preprocess_full_pipeline_ibp_crown() {
    // Full pipeline: normalize -> Conv2d patch embed -> reshape -> transpose ->
    //   LayerNorm -> Linear projection
    // [C, H, W] -> normalize -> [C, H, W] -> Conv2d -> [D, H/P, W/P] ->
    //   reshape [D, N] -> transpose [N, D] -> LayerNorm [N, D] -> Linear [N, proj]
    let out_h = IMG_H / PATCH_SIZE;
    let out_w = IMG_W / PATCH_SIZE;
    let proj_dim = 32;

    let mut b = TensorBlockBuilder::new("img_full_pipeline");
    let input = b.add_input("image", &[IMG_C, IMG_H, IMG_W]);

    // Step 1: Normalization (scale + shift)
    let norm_scale = b.add_input("norm_scale", &[IMG_C]);
    let norm_bias = b.add_input("norm_bias", &[IMG_C]);
    let norm_scale_bc = b.add_broadcast_left(norm_scale, &[IMG_C, IMG_H, IMG_W]);
    let norm_bias_bc = b.add_broadcast_left(norm_bias, &[IMG_C, IMG_H, IMG_W]);
    let normed = b.add_binary_mul(input, norm_scale_bc, &[IMG_C, IMG_H, IMG_W]);
    let normed = b.add_binary_add(normed, norm_bias_bc, &[IMG_C, IMG_H, IMG_W]);

    // Step 2: Conv2d patch embedding
    let conv_w = b.add_input("conv_w", &[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_b", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        normed,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, out_h, out_w],
    );

    // Step 3: Reshape and transpose
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    // Step 4: LayerNorm
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_out = b.add_layer_norm(transposed, eps, 1, ln_w, ln_b, &[NUM_PATCHES, HIDDEN_DIM]);

    // Step 5: Linear projection
    let proj_w = b.add_input("proj_w", &[proj_dim, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[proj_dim]);
    let out = b.add_linear(ln_out, proj_w, Some(proj_b), &[NUM_PATCHES, proj_dim]);
    let def = b.build(out).expect("valid full pipeline kernel");

    // ImageNet normalization constants
    let inv_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225];
    let neg_mean_div_std = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

    let bindings = vec![
        TensorParamBinding::Variable,
        // normalization scale and bias
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_C]), inv_std.to_vec()).expect("inv_std"),
        ),
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[IMG_C]), neg_mean_div_std.to_vec())
                .expect("neg_mean_div_std"),
        ),
        // Conv2d patch embed
        weight(&[HIDDEN_DIM, IMG_C, PATCH_SIZE, PATCH_SIZE]),
        bias_zero(&[HIDDEN_DIM]),
        // LayerNorm
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        // Linear projection
        weight(&[proj_dim, HIDDEN_DIM]),
        bias_zero(&[proj_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IMG_C, IMG_H, IMG_W);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    assert_eq!(ibp_output.lower_upper().0.shape(), &[NUM_PATCHES, proj_dim]);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Full preprocessing pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("Full preprocessing pipeline CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}
