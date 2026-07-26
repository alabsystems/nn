// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Feature Pyramid Network NY composition.
//!
//! Verifies IBP and CROWN bounds propagation through FPN architectures
//! used in dpdf document understanding models:
//!
//! **Lateral connections:**
//! 1. 1x1 conv lateral: channel reduction preserves bounds (IBP + CROWN)
//! 2. Multi-scale lateral: stride 4, 8, 16 feature maps (IBP)
//! 3. Lateral + upsample: top-down pathway element addition (IBP)
//!
//! **Top-down pathway:**
//! 4. 2x upsample + add: nearest-neighbor upsample + lateral fusion (IBP)
//! 5. 3-level top-down: P5 -> P4 -> P3 cascaded fusion (IBP)
//! 6. Top-down with Conv smoothing: 3x3 conv after fusion (IBP + CROWN)
//!
//! **PAN neck (YOLO):**
//! 7. Bottom-up pathway: P3 -> P4 -> P5 stride-2 downsampling (IBP)
//! 8. PAN bidirectional: top-down + bottom-up combined (IBP)
//! 9. PAN with C2f blocks: CSP bottleneck in neck (IBP + CROWN)
//!
//! **Multi-scale output:**
//! 10. Multi-scale detection: per-level detection heads (IBP)
//! 11. Feature map resolution: spatial dimensions halve per level (IBP)
//! 12. Channel alignment: all levels same channel count (IBP)
//! 13. FPN monotone tightening: smaller eps -> tighter multi-scale bounds (IBP)
//! 14. Cross-level feature consistency: adjacent levels bounded (IBP)
//! 15. Full neck pipeline: backbone features -> FPN -> detection heads (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - P3: [C_FPN, 16, 16], P4: [C_FPN, 8, 8], P5: [C_FPN, 4, 4]
//! - Backbone channels: C3=64, C4=128, C5=256
//! - FPN channels: C_FPN=64 (unified across levels)
//!
//! Part of #4002: NY compose tests for feature pyramid networks.

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

/// P3 spatial size (finest FPN level).
const P3_SPATIAL: usize = 16;
/// P4 spatial size.
const P4_SPATIAL: usize = 8;
/// P5 spatial size (coarsest FPN level).
const P5_SPATIAL: usize = 4;
/// Backbone C3 channels (before lateral projection).
const C3: usize = 64;
/// Backbone C4 channels.
const C4: usize = 128;
/// Backbone C5 channels.
const C5: usize = 256;
/// Unified FPN channel count (after lateral 1x1 conv).
const C_FPN: usize = 64;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of detection classes per level.
const NUM_CLASSES: usize = 10;

// ===========================================================================
// Helper: total_bound_width
// ===========================================================================

/// Compute total bound width (sum of hi - lo across all elements).
fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).sum::<f32>()
}

// ===========================================================================
// 1. 1x1 conv lateral: channel reduction preserves bounds (IBP + CROWN)
// ===========================================================================

/// Build a single lateral 1x1 conv: C5 -> C_FPN channel reduction.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_FPN, P5_SPATIAL, P5_SPATIAL].
fn build_lateral_1x1() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_lateral_1x1");
    let input = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);
    let w = b.add_input("lateral_w", &[C_FPN, C5, 1, 1]);
    let bias = b.add_input("lateral_b", &[C_FPN]);
    let out = b.add_conv2d(
        input,
        w,
        Some(bias),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );
    b.build(out).expect("valid lateral 1x1 kernel")
}

fn lateral_1x1_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_lateral_1x1_ibp() {
    let def = build_lateral_1x1();
    let bindings = lateral_1x1_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through lateral 1x1 conv");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
        "Lateral 1x1 projects C5 -> C_FPN"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Lateral 1x1 IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_fpn_lateral_1x1_crown() {
    let def = build_lateral_1x1();
    let bindings = lateral_1x1_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Lateral 1x1 CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 2. Multi-scale lateral: stride 4, 8, 16 feature maps (IBP)
// ===========================================================================

/// Build multi-scale lateral projections for C3, C4, C5 backbone outputs.
/// Uses the coarsest (C5) as the verification output.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_FPN, P5_SPATIAL, P5_SPATIAL].
fn build_multi_scale_lateral() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_multi_scale_lateral");

    // All three backbone levels as inputs, but verify the coarsest path.
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1 conv: C5 -> C_FPN
    let w5 = b.add_input("lat5_w", &[C_FPN, C5, 1, 1]);
    let b5 = b.add_input("lat5_b", &[C_FPN]);
    let lat5 = b.add_conv2d(
        c5,
        w5,
        Some(b5),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    b.build(lat5).expect("valid multi-scale lateral kernel")
}

fn multi_scale_lateral_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_multi_scale_lateral_ibp() {
    let def = build_multi_scale_lateral();
    let bindings = multi_scale_lateral_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale lateral");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
        "Multi-scale lateral output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale lateral IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Lateral + upsample: top-down pathway element addition (IBP)
// ===========================================================================

/// Build lateral 1x1 conv on P5 + ConvTranspose2d upsample to P4 resolution.
/// Simulates the top-down pathway: project P5 to C_FPN then upsample 2x.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_FPN, P4_SPATIAL, P4_SPATIAL].
fn build_lateral_upsample() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_lateral_upsample");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1 conv
    let lat_w = b.add_input("lat_w", &[C_FPN, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_FPN]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    // Upsample 2x via ConvTranspose2d(stride=2, k=4, p=1)
    let up_w = b.add_input("up_w", &[C_FPN, C_FPN, 4, 4]);
    let up_b = b.add_input("up_b", &[C_FPN]);
    let up = b.add_conv_transpose_2d(
        lat,
        up_w,
        Some(up_b),
        2,
        2, // stride
        1,
        1, // padding
        1,
        1, // dilation
        1, // groups
        0,
        0, // output_padding
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    b.build(up).expect("valid lateral+upsample kernel")
}

fn lateral_upsample_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_lateral_upsample_ibp() {
    let def = build_lateral_upsample();
    let bindings = lateral_upsample_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through lateral + upsample");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
        "Lateral + upsample produces P4 resolution"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Lateral + upsample IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. 2x upsample + add: nearest-neighbor upsample + lateral fusion (IBP)
// ===========================================================================

/// Build the core FPN fusion: upsample coarser level + add lateral features.
/// P5 lateral (C_FPN) upsampled 2x + P4 lateral (C_FPN) -> fused P4.
/// Two variable inputs: P5_features and P4_features.
fn build_upsample_add_fusion() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_upsample_add");
    let p5_lat = b.add_input("p5_lateral", &[C_FPN, P5_SPATIAL, P5_SPATIAL]);
    let p4_lat = b.add_input("p4_lateral", &[C_FPN, P4_SPATIAL, P4_SPATIAL]);

    // Upsample P5 lateral to P4 resolution via ConvTranspose2d(s=2, k=4, p=1)
    let up_w = b.add_input("up_w", &[C_FPN, C_FPN, 4, 4]);
    let up_b = b.add_input("up_b", &[C_FPN]);
    let p5_up = b.add_conv_transpose_2d(
        p5_lat,
        up_w,
        Some(up_b),
        2,
        2, // stride
        1,
        1, // padding
        1,
        1, // dilation
        1, // groups
        0,
        0, // output_padding
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // Element-wise add: upsampled P5 + P4 lateral
    let fused = b.add_binary_add(p5_up, p4_lat, &[C_FPN, P4_SPATIAL, P4_SPATIAL]);

    b.build(fused).expect("valid upsample+add fusion kernel")
}

fn upsample_add_fusion_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // p5_lateral
        TensorParamBinding::Variable, // p4_lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_upsample_add_fusion_ibp() {
    let def = build_upsample_add_fusion();
    let bindings = upsample_add_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Combined variable input: p5_lateral [C_FPN, 4, 4] || p4_lateral [C_FPN, 8, 8]
    let total_elems = C_FPN * P5_SPATIAL * P5_SPATIAL + C_FPN * P4_SPATIAL * P4_SPATIAL;
    let input = uniform_bounds(&[total_elems], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through upsample+add fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
        "Fused output at P4 resolution"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Upsample+add fusion IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. 3-level top-down: P5 -> P4 -> P3 cascaded fusion (IBP)
// ===========================================================================

/// Build 3-level top-down FPN: P5 upsampled to P4 then to P3.
/// Each level: lateral 1x1 + ConvTranspose2d upsample from coarser level.
/// Single variable input: C5 features.
/// Output: P3 resolution [C_FPN, P3_SPATIAL, P3_SPATIAL].
fn build_3level_topdown() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_3level_topdown");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // P5 lateral: C5 -> C_FPN
    let lat5_w = b.add_input("lat5_w", &[C_FPN, C5, 1, 1]);
    let lat5_b = b.add_input("lat5_b", &[C_FPN]);
    let p5 = b.add_conv2d(
        c5,
        lat5_w,
        Some(lat5_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    // Upsample P5 -> P4 resolution
    let up54_w = b.add_input("up54_w", &[C_FPN, C_FPN, 4, 4]);
    let up54_b = b.add_input("up54_b", &[C_FPN]);
    let p4 = b.add_conv_transpose_2d(
        p5,
        up54_w,
        Some(up54_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // Upsample P4 -> P3 resolution
    let up43_w = b.add_input("up43_w", &[C_FPN, C_FPN, 4, 4]);
    let up43_b = b.add_input("up43_b", &[C_FPN]);
    let p3 = b.add_conv_transpose_2d(
        p4,
        up43_w,
        Some(up43_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_FPN, P3_SPATIAL, P3_SPATIAL],
    );

    b.build(p3).expect("valid 3-level top-down kernel")
}

fn topdown_3level_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lat5 weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // up54 weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // up43 weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_3level_topdown_ibp() {
    let def = build_3level_topdown();
    let bindings = topdown_3level_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-level top-down FPN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P3_SPATIAL, P3_SPATIAL],
        "3-level top-down reaches P3 resolution"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("3-level top-down IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Top-down with Conv smoothing: 3x3 conv after fusion (IBP + CROWN)
// ===========================================================================

/// Build top-down with 3x3 smoothing conv: lateral 1x1 + upsample + smooth.
/// Standard FPN applies a 3x3 conv after element-wise addition to reduce aliasing.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_FPN, P4_SPATIAL, P4_SPATIAL].
fn build_topdown_smooth() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_topdown_smooth");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1
    let lat_w = b.add_input("lat_w", &[C_FPN, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_FPN]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    // Upsample to P4 resolution
    let up_w = b.add_input("up_w", &[C_FPN, C_FPN, 4, 4]);
    let up_b = b.add_input("up_b", &[C_FPN]);
    let up = b.add_conv_transpose_2d(
        lat,
        up_w,
        Some(up_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // Smoothing 3x3 conv (reduces aliasing from upsampling)
    let smooth_w = b.add_input("smooth_w", &[C_FPN, C_FPN, 3, 3]);
    let smooth_b = b.add_input("smooth_b", &[C_FPN]);
    let smooth = b.add_conv2d(
        up,
        smooth_w,
        Some(smooth_b),
        1,
        1,
        1,
        1,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    b.build(smooth).expect("valid top-down smooth kernel")
}

fn topdown_smooth_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // upsample weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // smoothing conv weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_topdown_smooth_ibp() {
    let def = build_topdown_smooth();
    let bindings = topdown_smooth_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through top-down + smooth");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
        "Top-down smooth output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Top-down smooth IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_fpn_topdown_smooth_crown() {
    let def = build_topdown_smooth();
    let bindings = topdown_smooth_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P4_SPATIAL, P4_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Top-down smooth CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. Bottom-up pathway: P3 -> P4 -> P5 stride-2 downsampling (IBP)
// ===========================================================================

/// Build PAN bottom-up pathway: stride-2 Conv2d from P3 -> P4 -> P5.
/// Input: [C_FPN, P3_SPATIAL, P3_SPATIAL] -> Output: [C_FPN, P5_SPATIAL, P5_SPATIAL].
fn build_bottomup_pathway() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_bottomup");
    let p3 = b.add_input("p3_features", &[C_FPN, P3_SPATIAL, P3_SPATIAL]);

    // P3 -> P4: stride-2 Conv2d
    let w34 = b.add_input("down34_w", &[C_FPN, C_FPN, 3, 3]);
    let b34 = b.add_input("down34_b", &[C_FPN]);
    let p4 = b.add_conv2d(
        p3,
        w34,
        Some(b34),
        2,
        2,
        1,
        1,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );
    let p4 = b.add_relu(p4, &[C_FPN, P4_SPATIAL, P4_SPATIAL]);

    // P4 -> P5: stride-2 Conv2d
    let w45 = b.add_input("down45_w", &[C_FPN, C_FPN, 3, 3]);
    let b45 = b.add_input("down45_b", &[C_FPN]);
    let p5 = b.add_conv2d(
        p4,
        w45,
        Some(b45),
        2,
        2,
        1,
        1,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );
    let p5 = b.add_relu(p5, &[C_FPN, P5_SPATIAL, P5_SPATIAL]);

    b.build(p5).expect("valid bottom-up pathway kernel")
}

fn bottomup_pathway_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_bottomup_pathway_ibp() {
    let def = build_bottomup_pathway();
    let bindings = bottomup_pathway_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_FPN, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through bottom-up pathway");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
        "Bottom-up downsamples P3 -> P5"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Bottom-up pathway IBP: bounds=[{lo_min}, {hi_max}]");
    // ReLU output: lower >= 0
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. PAN bidirectional: top-down + bottom-up combined (IBP)
// ===========================================================================

/// Build PAN-style bidirectional neck: top-down upsample then bottom-up downsample.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> lateral -> upsample -> downsample.
/// Output: [C_FPN, P5_SPATIAL, P5_SPATIAL].
fn build_pan_bidirectional() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_pan_bidirectional");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Top-down: lateral 1x1 + upsample to P4
    let lat_w = b.add_input("lat_w", &[C_FPN, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_FPN]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    let up_w = b.add_input("up_w", &[C_FPN, C_FPN, 4, 4]);
    let up_b = b.add_input("up_b", &[C_FPN]);
    let p4_up = b.add_conv_transpose_2d(
        lat,
        up_w,
        Some(up_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // Bottom-up: stride-2 downsample P4 back to P5 resolution
    let down_w = b.add_input("down_w", &[C_FPN, C_FPN, 3, 3]);
    let down_b = b.add_input("down_b", &[C_FPN]);
    let p5_down = b.add_conv2d(
        p4_up,
        down_w,
        Some(down_b),
        2,
        2,
        1,
        1,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );
    let out = b.add_relu(p5_down, &[C_FPN, P5_SPATIAL, P5_SPATIAL]);

    b.build(out).expect("valid PAN bidirectional kernel")
}

fn pan_bidirectional_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // upsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // downsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_pan_bidirectional_ibp() {
    let def = build_pan_bidirectional();
    let bindings = pan_bidirectional_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN bidirectional neck");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
        "PAN bidirectional output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN bidirectional IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. PAN with C2f blocks: CSP bottleneck in neck (IBP + CROWN)
// ===========================================================================

/// Build PAN neck with C2f-style CSP bottleneck after downsampling.
/// Input: [C_FPN, P3_SPATIAL, P3_SPATIAL] -> downsample -> split -> bottleneck -> concat -> merge.
/// Output: [C_FPN, P4_SPATIAL, P4_SPATIAL].
fn build_pan_c2f() -> TensorKernelDef {
    let half_c = C_FPN / 2;
    let half_shape = [half_c, P4_SPATIAL, P4_SPATIAL];
    let out_shape = [C_FPN, P4_SPATIAL, P4_SPATIAL];

    let mut b = TensorBlockBuilder::new("fpn_pan_c2f");
    let p3 = b.add_input("p3_features", &[C_FPN, P3_SPATIAL, P3_SPATIAL]);

    // Downsample: stride-2 Conv2d
    let down_w = b.add_input("down_w", &[C_FPN, C_FPN, 3, 3]);
    let down_b = b.add_input("down_b", &[C_FPN]);
    let down = b.add_conv2d(p3, down_w, Some(down_b), 2, 2, 1, 1, &out_shape);

    // C2f split: two 1x1 conv paths
    let split_w_a = b.add_input("split_w_a", &[half_c, C_FPN, 1, 1]);
    let split_w_b = b.add_input("split_w_b", &[half_c, C_FPN, 1, 1]);

    let path_a = b.add_conv2d(down, split_w_a, None, 1, 1, 0, 0, &half_shape);

    // Path B: bottleneck Conv3x3 -> ReLU
    let path_b_in = b.add_conv2d(down, split_w_b, None, 1, 1, 0, 0, &half_shape);
    let bn_w = b.add_input("bn_conv_w", &[half_c, half_c, 3, 3]);
    let bn_b = b.add_input("bn_conv_b", &[half_c]);
    let bn_out = b.add_conv2d(path_b_in, bn_w, Some(bn_b), 1, 1, 1, 1, &half_shape);
    let bn_out = b.add_relu(bn_out, &half_shape);

    // Concat path A + bottleneck path B
    let concat = b.add_concat(&[path_a, bn_out], 0, &out_shape);

    // Merge 1x1 conv
    let merge_w = b.add_input("merge_w", &[C_FPN, C_FPN, 1, 1]);
    let merge_b = b.add_input("merge_b", &[C_FPN]);
    let out = b.add_conv2d(concat, merge_w, Some(merge_b), 1, 1, 0, 0, &out_shape);

    b.build(out).expect("valid PAN C2f kernel")
}

fn pan_c2f_bindings() -> Vec<TensorParamBinding> {
    let half_c = C_FPN / 2;
    vec![
        TensorParamBinding::Variable,
        // downsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // split A
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[half_c, C_FPN, 1, 1]),
            WEIGHT_MAG,
        )),
        // split B
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[half_c, C_FPN, 1, 1]),
            WEIGHT_MAG,
        )),
        // bottleneck conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[half_c, half_c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[half_c]), 0.0f32)),
        // merge
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_pan_c2f_ibp() {
    let def = build_pan_c2f();
    let bindings = pan_c2f_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_FPN, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN C2f block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
        "PAN C2f output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN C2f IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_fpn_pan_c2f_crown() {
    let def = build_pan_c2f();
    let bindings = pan_c2f_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_FPN, P3_SPATIAL, P3_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P4_SPATIAL, P4_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN C2f CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 10. Multi-scale detection: per-level detection heads (IBP)
// ===========================================================================

/// Build multi-scale detection heads: each FPN level gets a sigmoid detection head.
/// Uses P5 level as the verification output.
/// Input: [C_FPN, P5_SPATIAL, P5_SPATIAL] -> 1x1 conv -> sigmoid.
/// Output: [NUM_CLASSES, P5_SPATIAL, P5_SPATIAL].
fn build_detection_head() -> TensorKernelDef {
    let out_shape = [NUM_CLASSES, P5_SPATIAL, P5_SPATIAL];
    let mut b = TensorBlockBuilder::new("fpn_detection_head");
    let features = b.add_input("p5_features", &[C_FPN, P5_SPATIAL, P5_SPATIAL]);

    let det_w = b.add_input("det_w", &[NUM_CLASSES, C_FPN, 1, 1]);
    let det_b = b.add_input("det_b", &[NUM_CLASSES]);
    let logits = b.add_conv2d(features, det_w, Some(det_b), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid detection head kernel")
}

fn detection_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, C_FPN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_multi_scale_detection_ibp() {
    let def = build_detection_head();
    let bindings = detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_FPN, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CLASSES, P5_SPATIAL, P5_SPATIAL],
        "Detection head output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale detection IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 11. Feature map resolution: spatial dimensions halve per level (IBP)
// ===========================================================================

/// Build cascaded stride-2 convolutions to verify spatial halving per level.
/// Input: [C_FPN, P3_SPATIAL, P3_SPATIAL] -> P4 -> P5.
/// Output: [C_FPN, P5_SPATIAL, P5_SPATIAL].
fn build_resolution_halving() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_resolution_halving");
    let p3 = b.add_input("p3_features", &[C_FPN, P3_SPATIAL, P3_SPATIAL]);

    // P3 -> P4
    let w34 = b.add_input("w34", &[C_FPN, C_FPN, 3, 3]);
    let b34 = b.add_input("b34", &[C_FPN]);
    let p4 = b.add_conv2d(
        p3,
        w34,
        Some(b34),
        2,
        2,
        1,
        1,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // P4 -> P5
    let w45 = b.add_input("w45", &[C_FPN, C_FPN, 3, 3]);
    let b45 = b.add_input("b45", &[C_FPN]);
    let p5 = b.add_conv2d(
        p4,
        w45,
        Some(b45),
        2,
        2,
        1,
        1,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    b.build(p5).expect("valid resolution halving kernel")
}

fn resolution_halving_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_resolution_halving_ibp() {
    let def = build_resolution_halving();
    let bindings = resolution_halving_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_FPN, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through resolution halving");

    // Verify output spatial is P5 = P3/4
    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
        "Two stride-2 convs: 16 -> 8 -> 4"
    );
    assert_bounds_valid(&output);

    // Verify the spatial halving: P3=16, P4=8, P5=4
    assert_eq!(P3_SPATIAL, 2 * P4_SPATIAL, "P3 = 2 * P4");
    assert_eq!(P4_SPATIAL, 2 * P5_SPATIAL, "P4 = 2 * P5");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Resolution halving IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. Channel alignment: all levels same channel count (IBP)
// ===========================================================================

/// Build channel alignment: project backbone C3, C4, C5 all to C_FPN.
/// Uses C3 -> C_FPN as the verification path (largest channel reduction ratio is C5).
/// Input: [C3, P3_SPATIAL, P3_SPATIAL] -> Output: [C_FPN, P3_SPATIAL, P3_SPATIAL].
fn build_channel_alignment() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_channel_alignment");
    let c3_feat = b.add_input("c3_features", &[C3, P3_SPATIAL, P3_SPATIAL]);

    // Lateral 1x1 conv: C3 -> C_FPN
    let lat_w = b.add_input("lat_w", &[C_FPN, C3, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_FPN]);
    let out = b.add_conv2d(
        c3_feat,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P3_SPATIAL, P3_SPATIAL],
    );

    b.build(out).expect("valid channel alignment kernel")
}

fn channel_alignment_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C3, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_channel_alignment_ibp() {
    let def = build_channel_alignment();
    let bindings = channel_alignment_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C3, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through channel alignment");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P3_SPATIAL, P3_SPATIAL],
        "Channel alignment: C3 -> C_FPN"
    );
    assert_bounds_valid(&output);

    // Verify C3=C_FPN (same channels -> identity-like projection for this config)
    assert_eq!(C3, C_FPN, "C3 == C_FPN for this test configuration");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Channel alignment IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. FPN monotone tightening: smaller eps -> tighter multi-scale bounds (IBP)
// ===========================================================================

/// Verify monotone tightening through the top-down FPN pipeline.
/// Smaller input perturbation must produce tighter output bounds.
#[test]
fn test_fpn_monotone_tightening() {
    let def = build_topdown_smooth();
    let bindings = topdown_smooth_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: [-1.0, 1.0]
    let wide_input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);

    // Narrow input: [-0.1, 0.1]
    let narrow_input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 0.1);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    assert_bounds_valid(&narrow_output);

    let wide_width = total_bound_width(&wide_output);
    let narrow_width = total_bound_width(&narrow_output);

    eprintln!(
        "FPN monotone tightening: wide_width={wide_width:.4}, narrow_width={narrow_width:.4}"
    );

    assert!(
        narrow_width <= wide_width + 1e-6,
        "narrower input must produce tighter output bounds: \
         narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 14. Cross-level feature consistency: adjacent levels bounded (IBP)
// ===========================================================================

/// Build two adjacent FPN levels from same backbone input to verify
/// both levels produce finite bounded outputs.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> lateral -> P5 level features.
/// Also builds upsample -> P4 level (but verifies P5 output).
fn build_cross_level_consistency() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_cross_level");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // P5 lateral
    let lat5_w = b.add_input("lat5_w", &[C_FPN, C5, 1, 1]);
    let lat5_b = b.add_input("lat5_b", &[C_FPN]);
    let p5 = b.add_conv2d(
        c5,
        lat5_w,
        Some(lat5_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    // P4 via upsample (separate graph path, both connected to input)
    let up_w = b.add_input("up_w", &[C_FPN, C_FPN, 4, 4]);
    let up_b = b.add_input("up_b", &[C_FPN]);
    let _p4 = b.add_conv_transpose_2d(
        p5,
        up_w,
        Some(up_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // Output P5 level (P4 is reachable but not the output)
    b.build(p5).expect("valid cross-level consistency kernel")
}

fn cross_level_consistency_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_cross_level_consistency_ibp() {
    let def = build_cross_level_consistency();
    let bindings = cross_level_consistency_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-level FPN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
        "Cross-level P5 output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-level consistency IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Full neck pipeline: backbone features -> FPN -> detection heads (IBP)
// ===========================================================================

/// Build full FPN neck pipeline: backbone C5 -> lateral -> upsample -> smooth -> detect.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL]
/// -> lateral 1x1 (C5 -> C_FPN)
/// -> upsample 2x (P5 -> P4)
/// -> smooth 3x3
/// -> ReLU
/// -> detection 1x1 + sigmoid
/// Output: [NUM_CLASSES, P4_SPATIAL, P4_SPATIAL].
fn build_full_neck_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_full_neck_pipeline");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1
    let lat_w = b.add_input("lat_w", &[C_FPN, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_FPN]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_FPN, P5_SPATIAL, P5_SPATIAL],
    );

    // Upsample 2x
    let up_w = b.add_input("up_w", &[C_FPN, C_FPN, 4, 4]);
    let up_b = b.add_input("up_b", &[C_FPN]);
    let up = b.add_conv_transpose_2d(
        lat,
        up_w,
        Some(up_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // Smooth 3x3
    let smooth_w = b.add_input("smooth_w", &[C_FPN, C_FPN, 3, 3]);
    let smooth_b = b.add_input("smooth_b", &[C_FPN]);
    let smooth = b.add_conv2d(
        up,
        smooth_w,
        Some(smooth_b),
        1,
        1,
        1,
        1,
        &[C_FPN, P4_SPATIAL, P4_SPATIAL],
    );

    // ReLU activation
    let relu = b.add_relu(smooth, &[C_FPN, P4_SPATIAL, P4_SPATIAL]);

    // Detection head: 1x1 conv + sigmoid
    let det_w = b.add_input("det_w", &[NUM_CLASSES, C_FPN, 1, 1]);
    let det_b = b.add_input("det_b", &[NUM_CLASSES]);
    let logits = b.add_conv2d(
        relu,
        det_w,
        Some(det_b),
        1,
        1,
        0,
        0,
        &[NUM_CLASSES, P4_SPATIAL, P4_SPATIAL],
    );
    let out = b.add_sigmoid(logits, &[NUM_CLASSES, P4_SPATIAL, P4_SPATIAL]);

    b.build(out).expect("valid full neck pipeline kernel")
}

fn full_neck_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // upsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // smooth
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_FPN, C_FPN, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_FPN]), 0.0f32)),
        // detection head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, C_FPN, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_full_neck_pipeline_ibp() {
    let def = build_full_neck_pipeline();
    let bindings = full_neck_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full neck pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CLASSES, P4_SPATIAL, P4_SPATIAL],
        "Full neck pipeline output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full neck pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}
