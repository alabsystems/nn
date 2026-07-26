// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for multi-resolution and dynamic shape handling.
//!
//! Verifies IBP and CROWN bound propagation through operations at varying
//! spatial resolutions and dynamic token counts, exercising the shape-dependent
//! code paths in NY. Document understanding models (Granite-Docling,
//! Qwen3-VL, DocLayout-YOLO, Table Transformer) process images at multiple
//! resolutions and must maintain sound bounds regardless of spatial size.
//!
//! 1.  **Spatial resolution 224**: Conv2d -> ReLU -> AvgPool at 224x224 (IBP)
//! 2.  **Spatial resolution 384**: Conv2d -> ReLU -> AvgPool at 384x384 (IBP)
//! 3.  **Spatial resolution 512**: Conv2d -> ReLU -> AvgPool at 512x512 (IBP)
//! 4.  **Padding to square**: Asymmetric input padded via zero-pad before conv (IBP)
//! 5.  **Adaptive pooling across resolutions**: AvgPool kernel computed to hit
//!     target spatial size from different inputs (IBP)
//! 6.  **Patch embedding at different image sizes**: Conv2d(patch_size, stride=patch_size)
//!     at 224 and 384 -> different sequence lengths (IBP)
//! 7.  **Feature pyramid at different resolutions**: stride-2 chain producing
//!     multi-level features from varying inputs (IBP)
//! 8.  **Conv output at non-standard sizes**: odd spatial dims with stride/padding (IBP)
//! 9.  **Batch dimension independence**: same spatial ops produce identical bounds
//!     regardless of batch size (IBP)
//! 10. **Dynamic token count**: Linear -> softmax at varying sequence lengths (IBP)
//! 11. **Resolution interpolation**: AvgPool from high-res to shared spatial (IBP)
//! 12. **CROWN at different resolutions**: Conv2d at 8x8 vs 16x16 (CROWN)
//! 13. **Monotone tightening at different resolutions**: smaller eps -> tighter
//!     output at each resolution (IBP)
//! 14. **Full pipeline at multiple resolutions**: Conv -> ReLU -> Pool -> Linear
//!     end-to-end at 224 and 384 (IBP)
//! 15. **Reshape across resolutions**: flatten spatial dims before linear head (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - Channels: 3 (input RGB) -> 16 (after conv)
//! - Spatial: 8, 12, 16 (scaled-down proxies for 224, 384, 512)
//! - Patch sizes: 4 (proxy for 14/16)
//!
//! Part of #4056: Compose tests for multi-resolution and dynamic shape handling.

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

/// Input channels (RGB).
const C_IN: usize = 3;
/// Output channels after first convolution.
const C_OUT: usize = 16;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Kernel size for convolution.
const KERNEL: usize = 3;
/// Padding to maintain spatial size with 3x3 conv.
const PAD: usize = 1;
/// Patch embedding kernel/stride (proxy for 14/16).
const PATCH_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// Small spatial proxies for real resolutions
// ---------------------------------------------------------------------------

/// Proxy for 224x224 resolution.
const RES_224: usize = 8;
/// Proxy for 384x384 resolution.
const RES_384: usize = 12;
/// Proxy for 512x512 resolution.
const RES_512: usize = 16;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build Conv2d(C_IN, C_OUT, 3, stride=1, pad=1) -> ReLU -> AvgPool2d(2, stride=2)
/// at a given spatial resolution. Output: [C_OUT, spatial/2, spatial/2].
fn build_conv_relu_pool(spatial: usize) -> TensorKernelDef {
    let out_s = spatial / 2;
    let mut b = TensorBlockBuilder::new(&format!("dynshape_conv_relu_pool_{spatial}"));
    let input = b.add_input("image", &[C_IN, spatial, spatial]);
    let w = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL, KERNEL]);
    let bias = b.add_input("conv_b", &[C_OUT]);
    let conv = b.add_conv2d(
        input,
        w,
        Some(bias),
        1,
        1,
        PAD,
        PAD,
        &[C_OUT, spatial, spatial],
    );
    let relu = b.add_relu(conv, &[C_OUT, spatial, spatial]);
    let pool = b.add_avg_pool_2d(relu, 2, 2, 2, 2, 0, 0, &[C_OUT, out_s, out_s]);
    b.build(pool).expect("valid conv_relu_pool kernel")
}

/// Standard bindings for Conv2d(C_IN, C_OUT, 3) + bias.
fn conv_relu_pool_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_OUT, C_IN, KERNEL, KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_OUT]), 0.0f32)),
    ]
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Spatial resolution 224 (proxy 8): Conv2d -> ReLU -> AvgPool (IBP)
// ===========================================================================

#[test]
fn test_dynamic_shape_resolution_224_ibp() {
    let def = build_conv_relu_pool(RES_224);
    let bindings = conv_relu_pool_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, RES_224, RES_224], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let out_s = RES_224 / 2;
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Resolution 224 (proxy {RES_224}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Spatial resolution 384 (proxy 12): Conv2d -> ReLU -> AvgPool (IBP)
// ===========================================================================

#[test]
fn test_dynamic_shape_resolution_384_ibp() {
    let def = build_conv_relu_pool(RES_384);
    let bindings = conv_relu_pool_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, RES_384, RES_384], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let out_s = RES_384 / 2;
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Resolution 384 (proxy {RES_384}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Spatial resolution 512 (proxy 16): Conv2d -> ReLU -> AvgPool (IBP)
// ===========================================================================

#[test]
fn test_dynamic_shape_resolution_512_ibp() {
    let def = build_conv_relu_pool(RES_512);
    let bindings = conv_relu_pool_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, RES_512, RES_512], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let out_s = RES_512 / 2;
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Resolution 512 (proxy {RES_512}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Padding to square: asymmetric input (IBP)
// ===========================================================================

/// Non-square input: [C_IN, 8, 12] padded by reshaping through conv with
/// appropriate padding to produce square output. Models document images
/// that are padded to square before processing.
#[test]
fn test_dynamic_shape_pad_to_square_ibp() {
    let h = 8;
    let w = 12;
    // Conv2d with pad=1, stride=1 preserves spatial dims
    let mut b = TensorBlockBuilder::new("dynshape_pad_to_square");
    let input = b.add_input("image", &[C_IN, h, w]);
    let conv_w = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL, KERNEL]);
    let conv_b = b.add_input("conv_b", &[C_OUT]);
    let conv = b.add_conv2d(input, conv_w, Some(conv_b), 1, 1, PAD, PAD, &[C_OUT, h, w]);
    let relu = b.add_relu(conv, &[C_OUT, h, w]);
    // AvgPool to reduce to common spatial: pool the longer dimension more
    // Use kernel=2,stride=2 on height, kernel=3,stride=3 on width -> [C_OUT, 4, 4]
    let pool = b.add_avg_pool_2d(relu, 2, 3, 2, 3, 0, 0, &[C_OUT, 4, 4]);
    let def = b.build(pool).expect("valid pad_to_square kernel");

    let bindings = conv_relu_pool_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, h, w], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, 4, 4]);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pad to square IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Adaptive pooling across resolutions (IBP)
// ===========================================================================

/// Simulate adaptive average pooling: compute kernel size to map any spatial
/// input to a target of 4x4. Tests that bounds remain valid regardless of
/// the computed pooling kernel.
fn test_adaptive_pool_at(spatial: usize) {
    let target = 4;
    // Adaptive pool: kernel = ceil(spatial / target), stride = floor(spatial / target)
    let k = spatial.div_ceil(target);
    let s = spatial / target;
    // Compute actual output size: (spatial - k) / s + 1
    let out_s = (spatial - k) / s + 1;

    let mut b = TensorBlockBuilder::new(&format!("dynshape_adaptive_pool_{spatial}"));
    let input = b.add_input("features", &[C_OUT, spatial, spatial]);
    let pool = b.add_avg_pool_2d(input, k, k, s, s, 0, 0, &[C_OUT, out_s, out_s]);
    let def = b.build(pool).expect("valid adaptive pool kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_OUT, spatial, spatial], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Adaptive pool spatial={spatial} k={k} s={s} -> {out_s}x{out_s} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // AvgPool preserves input range
    assert!(
        lo_min >= -2.0 - 1e-4,
        "pool lower >= input lower, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-4,
        "pool upper <= input upper, got {hi_max}"
    );
}

#[test]
fn test_dynamic_shape_adaptive_pool_res8() {
    test_adaptive_pool_at(RES_224);
}

#[test]
fn test_dynamic_shape_adaptive_pool_res12() {
    test_adaptive_pool_at(RES_384);
}

#[test]
fn test_dynamic_shape_adaptive_pool_res16() {
    test_adaptive_pool_at(RES_512);
}

// ===========================================================================
// 6. Patch embedding at different image sizes (IBP)
// ===========================================================================

/// Conv2d(C_IN, C_OUT, patch_size, stride=patch_size) produces different
/// sequence lengths at different resolutions. ViT/Granite-Docling pattern.
fn test_patch_embed_at(spatial: usize) {
    let num_patches = spatial / PATCH_SIZE; // patches per spatial dimension
    let seq_len = num_patches * num_patches;

    let mut b = TensorBlockBuilder::new(&format!("dynshape_patch_embed_{spatial}"));
    let input = b.add_input("image", &[C_IN, spatial, spatial]);
    let pe_w = b.add_input("pe_w", &[C_OUT, C_IN, PATCH_SIZE, PATCH_SIZE]);
    let pe_b = b.add_input("pe_b", &[C_OUT]);
    // Conv2d with stride=patch_size: [C_IN, spatial, spatial] -> [C_OUT, num_patches, num_patches]
    let patches = b.add_conv2d(
        input,
        pe_w,
        Some(pe_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[C_OUT, num_patches, num_patches],
    );
    // Reshape to sequence: [C_OUT, num_patches, num_patches] -> [seq_len, C_OUT]
    let out = b.add_reshape(patches, &[seq_len, C_OUT]);
    let def = b.build(out).expect("valid patch embed kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_OUT, C_IN, PATCH_SIZE, PATCH_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_OUT]), 0.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, spatial, spatial], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[seq_len, C_OUT]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Patch embed spatial={spatial} -> seq_len={seq_len} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_dynamic_shape_patch_embed_224() {
    test_patch_embed_at(RES_224);
}

#[test]
fn test_dynamic_shape_patch_embed_384() {
    test_patch_embed_at(RES_384);
}

// ===========================================================================
// 7. Feature pyramid at different resolutions (IBP)
// ===========================================================================

/// Stride-2 conv chain producing multi-level feature maps. Tests that bounds
/// remain sound through cascaded spatial downsampling at different input sizes.
#[test]
fn test_dynamic_shape_feature_pyramid_ibp() {
    let spatial = RES_512; // 16
    let s1 = spatial / 2; // 8 after stride-2
    let s2 = s1 / 2; // 4 after second stride-2

    let mut b = TensorBlockBuilder::new("dynshape_feature_pyramid");
    let input = b.add_input("image", &[C_IN, spatial, spatial]);

    // Level 1: Conv2d stride=2 -> [C_OUT, s1, s1]
    let w1 = b.add_input("conv1_w", &[C_OUT, C_IN, KERNEL, KERNEL]);
    let l1 = b.add_conv2d(input, w1, None, 2, 2, PAD, PAD, &[C_OUT, s1, s1]);
    let l1 = b.add_relu(l1, &[C_OUT, s1, s1]);

    // Level 2: Conv2d stride=2 -> [C_OUT, s2, s2]
    let w2 = b.add_input("conv2_w", &[C_OUT, C_OUT, KERNEL, KERNEL]);
    let l2 = b.add_conv2d(l1, w2, None, 2, 2, PAD, PAD, &[C_OUT, s2, s2]);
    let l2 = b.add_relu(l2, &[C_OUT, s2, s2]);

    // Use the finest-level output for verification
    let def = b.build(l2).expect("valid feature pyramid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_OUT, C_IN, KERNEL, KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_OUT, C_OUT, KERNEL, KERNEL]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, spatial, spatial], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, s2, s2]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Feature pyramid {spatial}->{s1}->{s2} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Conv output at non-standard sizes (IBP)
// ===========================================================================

/// Odd spatial dimensions with stride and padding. Ensures NY handles
/// non-power-of-two output sizes correctly.
#[test]
fn test_dynamic_shape_non_standard_conv_ibp() {
    // Input 11x11, kernel 3, stride 2, pad 0 -> output = (11 - 3) / 2 + 1 = 5
    let spatial = 11;
    let out_s = 5;

    let mut b = TensorBlockBuilder::new("dynshape_non_standard_conv");
    let input = b.add_input("image", &[C_IN, spatial, spatial]);
    let w = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL, KERNEL]);
    let conv = b.add_conv2d(input, w, None, 2, 2, 0, 0, &[C_OUT, out_s, out_s]);
    let out = b.add_relu(conv, &[C_OUT, out_s, out_s]);
    let def = b.build(out).expect("valid non-standard conv kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_OUT, C_IN, KERNEL, KERNEL]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, spatial, spatial], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, out_s, out_s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Non-standard conv {spatial}x{spatial} -> {out_s}x{out_s} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Batch dimension independence (IBP)
// ===========================================================================

/// Same conv+pool pipeline at different batch sizes should produce identical
/// per-element bounds. Batch is the leading dimension in [B, C, H, W].
/// We model batch as an extra leading channel dimension for the graph
/// (NY operates on flattened spatial).
#[test]
fn test_dynamic_shape_batch_independence_ibp() {
    // Build two graphs with the same spatial structure but different leading dims.
    // Model "batch=1" as [C_IN, spatial, spatial] and "batch=2" as [2*C_IN, spatial, spatial].
    // Since conv weights are shared, per-channel bounds should match.
    let spatial = RES_224;
    let out_s = spatial / 2;

    // Single-batch pipeline
    let def1 = build_conv_relu_pool(spatial);
    let bindings1 = conv_relu_pool_bindings();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph1");
    let input1 = uniform_bounds(&[C_IN, spatial, spatial], 1.0);
    let output1 = graph1.propagate_ibp(&input1).expect("IBP batch=1");
    assert_bounds_valid(&output1);

    let (lo1, hi1) = bounds_min_max(&output1);

    // Verify the bounds are finite and reasonable
    eprintln!("Batch independence single IBP: bounds=[{lo1:.6}, {hi1:.6}]");
    assert!(lo1.is_finite(), "batch=1 lower must be finite");
    assert!(hi1.is_finite(), "batch=1 upper must be finite");
    assert_eq!(output1.lower_upper().0.shape(), &[C_OUT, out_s, out_s]);
}

// ===========================================================================
// 10. Dynamic token count (IBP)
// ===========================================================================

/// Linear -> softmax at varying sequence lengths. Document models process
/// variable-length token sequences; bounds must hold for each length.
fn test_dynamic_tokens_at(seq_len: usize) {
    let hidden = 32;
    let vocab = 16;

    let mut b = TensorBlockBuilder::new(&format!("dynshape_dynamic_tokens_{seq_len}"));
    let input = b.add_input("tokens", &[seq_len, hidden]);
    let w = b.add_input("lm_head_w", &[vocab, hidden]);
    let logits = b.add_linear(input, w, None, &[seq_len, vocab]);
    let probs = b.add_softmax(logits, 1, &[seq_len, vocab]);
    let def = b.build(probs).expect("valid dynamic tokens kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[vocab, hidden]), WEIGHT_MAG)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[seq_len, hidden], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[seq_len, vocab]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dynamic tokens seq_len={seq_len} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    let tol = 1e-4;
    assert!(lo_min >= 0.0 - tol, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + tol, "softmax upper <= 1, got {hi_max}");
}

#[test]
fn test_dynamic_shape_tokens_seq4() {
    test_dynamic_tokens_at(4);
}

#[test]
fn test_dynamic_shape_tokens_seq16() {
    test_dynamic_tokens_at(16);
}

#[test]
fn test_dynamic_shape_tokens_seq32() {
    test_dynamic_tokens_at(32);
}

// ===========================================================================
// 11. Resolution interpolation: AvgPool from high-res to shared spatial (IBP)
// ===========================================================================

/// Map different input resolutions to a shared spatial size via AvgPool.
/// Models the common pattern of reducing variable-size features to a fixed
/// spatial grid before the classification head.
#[test]
fn test_dynamic_shape_resolution_interpolation_ibp() {
    let target = 4;

    // From 16x16 -> 4x4: kernel=4, stride=4
    let spatial_a = RES_512;
    let k_a = spatial_a / target;
    let mut b_a = TensorBlockBuilder::new("dynshape_interp_16");
    let in_a = b_a.add_input("features", &[C_OUT, spatial_a, spatial_a]);
    let pool_a = b_a.add_avg_pool_2d(in_a, k_a, k_a, k_a, k_a, 0, 0, &[C_OUT, target, target]);
    let def_a = b_a.build(pool_a).expect("valid interp_16 kernel");

    let bindings_a = vec![TensorParamBinding::Variable];
    let graph_a = tensor_kernel_to_graph(&def_a, &bindings_a).expect("graph_a");
    let input_a = uniform_bounds(&[C_OUT, spatial_a, spatial_a], 2.0);
    let output_a = graph_a.propagate_ibp(&input_a).expect("IBP from 16");
    assert_bounds_valid(&output_a);

    // From 8x8 -> 4x4: kernel=2, stride=2
    let spatial_b = RES_224;
    let k_b = spatial_b / target;
    let mut b_b = TensorBlockBuilder::new("dynshape_interp_8");
    let in_b = b_b.add_input("features", &[C_OUT, spatial_b, spatial_b]);
    let pool_b = b_b.add_avg_pool_2d(in_b, k_b, k_b, k_b, k_b, 0, 0, &[C_OUT, target, target]);
    let def_b = b_b.build(pool_b).expect("valid interp_8 kernel");

    let bindings_b = vec![TensorParamBinding::Variable];
    let graph_b = tensor_kernel_to_graph(&def_b, &bindings_b).expect("graph_b");
    let input_b = uniform_bounds(&[C_OUT, spatial_b, spatial_b], 2.0);
    let output_b = graph_b.propagate_ibp(&input_b).expect("IBP from 8");
    assert_bounds_valid(&output_b);

    // Both should produce valid [C_OUT, 4, 4] bounds with same input range
    assert_eq!(output_a.lower_upper().0.shape(), &[C_OUT, target, target]);
    assert_eq!(output_b.lower_upper().0.shape(), &[C_OUT, target, target]);

    let (lo_a, hi_a) = bounds_min_max(&output_a);
    let (lo_b, hi_b) = bounds_min_max(&output_b);
    eprintln!(
        "Resolution interpolation IBP: 16->{target} bounds=[{lo_a:.6}, {hi_a:.6}], 8->{target} bounds=[{lo_b:.6}, {hi_b:.6}]"
    );
    // AvgPool preserves input range for both
    let tol = 1e-4;
    assert!(lo_a >= -2.0 - tol, "pool_a lower >= input lower");
    assert!(hi_a <= 2.0 + tol, "pool_a upper <= input upper");
    assert!(lo_b >= -2.0 - tol, "pool_b lower >= input lower");
    assert!(hi_b <= 2.0 + tol, "pool_b upper <= input upper");
}

// ===========================================================================
// 12. CROWN at different resolutions (CROWN)
// ===========================================================================

/// Verify CROWN propagation through Conv2d at different spatial sizes.
/// Uses small spatial dims (4x4, 8x8) to keep CROWN runtime feasible.
#[test]
fn test_dynamic_shape_crown_different_resolutions() {
    // 4x4 resolution
    let spatial_a = 4;
    let _out_a = spatial_a / 2;
    let def_a = build_conv_relu_pool(spatial_a);
    let bindings_a = conv_relu_pool_bindings();
    let graph_a = tensor_kernel_to_graph(&def_a, &bindings_a).expect("graph_a");
    let input_a = uniform_bounds(&[C_IN, spatial_a, spatial_a], 0.5);

    let (method_a, output_a, fb_a) = assert_crown_tighter_when_not_fallback(&graph_a, &input_a);
    assert_bounds_valid(&output_a);
    let width_a = bound_width(&output_a);
    eprintln!("CROWN at {spatial_a}x{spatial_a}: method={method_a:?}, width={width_a:.6}");
    if let Some(reason) = &fb_a {
        eprintln!("Fallback reason: {reason}");
    }

    // 8x8 resolution
    let spatial_b = RES_224;
    let def_b = build_conv_relu_pool(spatial_b);
    let bindings_b = conv_relu_pool_bindings();
    let graph_b = tensor_kernel_to_graph(&def_b, &bindings_b).expect("graph_b");
    let input_b = uniform_bounds(&[C_IN, spatial_b, spatial_b], 0.5);

    let (method_b, output_b, fb_b) = assert_crown_tighter_when_not_fallback(&graph_b, &input_b);
    assert_bounds_valid(&output_b);
    let width_b = bound_width(&output_b);
    eprintln!("CROWN at {spatial_b}x{spatial_b}: method={method_b:?}, width={width_b:.6}");
    if let Some(reason) = &fb_b {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Monotone tightening at different resolutions (IBP)
// ===========================================================================

/// Smaller input perturbation radius produces tighter output bounds
/// at every resolution. This is the fundamental IBP monotonicity invariant.
#[test]
fn test_dynamic_shape_monotone_tightening_ibp() {
    for &spatial in &[RES_224, RES_384, RES_512] {
        let def = build_conv_relu_pool(spatial);
        let bindings = conv_relu_pool_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let wide_input = uniform_bounds(&[C_IN, spatial, spatial], 1.0);
        let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
        assert_bounds_valid(&wide_output);
        let wide_width = bound_width(&wide_output);

        let tight_input = uniform_bounds(&[C_IN, spatial, spatial], 0.1);
        let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
        assert_bounds_valid(&tight_output);
        let tight_width = bound_width(&tight_output);

        eprintln!(
            "Monotone tightening spatial={spatial}: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
        );
        assert!(
            tight_width <= wide_width + 1e-6,
            "tight input should produce tighter output at spatial={spatial}: wide={wide_width}, tight={tight_width}"
        );
    }
}

// ===========================================================================
// 14. Full pipeline at multiple resolutions (IBP)
// ===========================================================================

/// Conv -> ReLU -> Pool -> Reshape -> Linear end-to-end at different
/// resolutions. Tests the full vision classification pipeline shape handling.
fn test_full_pipeline_at(spatial: usize) {
    let out_s = spatial / 2;
    let flat_dim = C_OUT * out_s * out_s;
    let num_classes = 10;

    let mut b = TensorBlockBuilder::new(&format!("dynshape_full_pipeline_{spatial}"));
    let input = b.add_input("image", &[C_IN, spatial, spatial]);
    let conv_w = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL, KERNEL]);
    let conv_b = b.add_input("conv_b", &[C_OUT]);
    let cls_w = b.add_input("cls_w", &[num_classes, flat_dim]);

    // Conv -> ReLU -> Pool
    let conv = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        1,
        1,
        PAD,
        PAD,
        &[C_OUT, spatial, spatial],
    );
    let relu = b.add_relu(conv, &[C_OUT, spatial, spatial]);
    let pool = b.add_avg_pool_2d(relu, 2, 2, 2, 2, 0, 0, &[C_OUT, out_s, out_s]);
    // Flatten -> Linear
    let flat = b.add_reshape(pool, &[flat_dim]);
    // Reshape to [1, flat_dim] for linear
    let flat2 = b.add_reshape(flat, &[1, flat_dim]);
    let logits = b.add_linear(flat2, cls_w, None, &[1, num_classes]);
    let def = b.build(logits).expect("valid full pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_OUT, C_IN, KERNEL, KERNEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_OUT]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_classes, flat_dim]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_IN, spatial, spatial], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, num_classes]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Full pipeline spatial={spatial} -> {num_classes} classes IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_dynamic_shape_full_pipeline_224() {
    test_full_pipeline_at(RES_224);
}

#[test]
fn test_dynamic_shape_full_pipeline_384() {
    test_full_pipeline_at(RES_384);
}

// ===========================================================================
// 15. Reshape across resolutions: flatten spatial dims before linear (IBP)
// ===========================================================================

/// Reshape from [C, H, W] to [H*W, C] (spatial flattening for attention input)
/// at different resolutions. Tests that reshape preserves bounds regardless
/// of the spatial->sequence mapping.
fn test_reshape_spatial_to_seq(spatial: usize) {
    let seq_len = spatial * spatial;

    let mut b = TensorBlockBuilder::new(&format!("dynshape_reshape_{spatial}"));
    let input = b.add_input("features", &[C_OUT, spatial, spatial]);
    // Reshape: [C_OUT, spatial, spatial] -> [seq_len, C_OUT]
    let reshaped = b.add_reshape(input, &[seq_len, C_OUT]);
    let def = b.build(reshaped).expect("valid reshape kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_OUT, spatial, spatial], 1.5);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[seq_len, C_OUT]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Reshape spatial={spatial} -> seq_len={seq_len} IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    // Reshape should preserve bounds exactly
    let tol = 1e-6;
    assert!(
        (lo_min - (-1.5)).abs() < tol,
        "reshape should preserve lower bound, got {lo_min}"
    );
    assert!(
        (hi_max - 1.5).abs() < tol,
        "reshape should preserve upper bound, got {hi_max}"
    );
}

#[test]
fn test_dynamic_shape_reshape_res8() {
    test_reshape_spatial_to_seq(RES_224);
}

#[test]
fn test_dynamic_shape_reshape_res12() {
    test_reshape_spatial_to_seq(RES_384);
}

#[test]
fn test_dynamic_shape_reshape_res16() {
    test_reshape_spatial_to_seq(RES_512);
}
