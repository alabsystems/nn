// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for multi-scale feature aggregation (PAN/FPN fusion)
//! bound propagation through detection neck architectures.
//!
//! Verifies NY IBP/CROWN bounds through PAN and FPN multi-scale
//! feature aggregation sub-blocks used in dpdf document understanding models:
//! DocLayout-YOLO, Table Transformer, and PaddleOCR detection.
//!
//! ## PAN Bottom-Up Path (tests 1-2)
//!
//! 1. PAN bottom-up path (P3 -> P4 -> P5): stride-2 conv downsampling (IBP)
//! 2. PAN top-down path (P5 -> P4 -> P3): ConvTranspose2d upsampling (IBP)
//!
//! ## PAN Bidirectional Fusion (tests 3-4)
//!
//! 3. PAN bidirectional fusion: top-down + bottom-up combined (IBP + CROWN)
//! 4. FPN + PAN combined neck: lateral + top-down + bottom-up (IBP)
//!
//! ## C2f and Feature Processing (tests 5-6)
//!
//! 5. C2f (Cross-Stage Partial) within neck: split + bottleneck + concat (IBP + CROWN)
//! 6. Multi-scale feature concatenation: concat P3 + P4 + P5 features (IBP)
//!
//! ## Upsample and Downsample Paths (tests 7-8)
//!
//! 7. Upsample + lateral connection composition: ConvTranspose2d + add (IBP)
//! 8. Downsample (stride-2 conv) + fusion: Conv2d(s=2) + add (IBP + CROWN)
//!
//! ## Multi-Level Comparison (tests 9-10)
//!
//! 9. 3-level vs 4-level feature pyramid: bound width comparison (IBP)
//! 10. Scale-specific bound widths: per-level width tracking (IBP)
//!
//! ## Alignment and Normalization (tests 11-12)
//!
//! 11. Feature dimension alignment across scales: 1x1 conv projection (IBP)
//! 12. PAN with SPPF: multi-scale MaxPool2d cascaded pooling (IBP)
//!
//! ## Output Normalization and Detection (tests 13-15)
//!
//! 13. Neck output normalization: ReLU bounded output (IBP + CROWN)
//! 14. Multi-scale detection head input: per-level sigmoid head (IBP)
//! 15. Full neck pipeline: backbone features -> PAN -> detection heads (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - P2: [C_NECK, 32, 32], P3: [C_NECK, 16, 16], P4: [C_NECK, 8, 8], P5: [C_NECK, 4, 4]
//! - Backbone channels: C3=64, C4=128, C5=256
//! - Neck channels: C_NECK=64 (unified across levels)
//!
//! Part of #4020: NY compose tests for multi-scale feature aggregation.

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

/// P2 spatial size (finest level, used in 4-level pyramid).
const P2_SPATIAL: usize = 32;
/// P3 spatial size.
const P3_SPATIAL: usize = 16;
/// P4 spatial size.
const P4_SPATIAL: usize = 8;
/// P5 spatial size (coarsest level).
const P5_SPATIAL: usize = 4;
/// Backbone C3 channels.
const C3: usize = 64;
/// Backbone C4 channels.
const C4: usize = 128;
/// Backbone C5 channels.
const C5: usize = 256;
/// Unified neck channel count (after lateral 1x1 conv).
const C_NECK: usize = 64;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of detection classes.
const NUM_CLASSES: usize = 10;
/// Half channel count for C2f split.
const HALF_C: usize = C_NECK / 2;
/// SPPF MaxPool2d kernel size (5x5 per YOLO spec).
const SPPF_KERNEL: usize = 5;
/// SPPF padding (keeps spatial dimensions unchanged with k=5).
const SPPF_PAD: usize = 2;

// ===========================================================================
// Helper: total_bound_width
// ===========================================================================

/// Compute total bound width (sum of hi - lo across all elements).
fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).sum::<f32>()
}

// ===========================================================================
// 1. PAN bottom-up path (P3 -> P4 -> P5) bound propagation (IBP)
// ===========================================================================

/// Build PAN bottom-up pathway: stride-2 Conv2d from P3 -> P4 -> P5.
/// Input: [C_NECK, P3_SPATIAL, P3_SPATIAL] -> Output: [C_NECK, P5_SPATIAL, P5_SPATIAL].
fn build_pan_bottomup() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pan_bottomup");
    let p3 = b.add_input("p3_features", &[C_NECK, P3_SPATIAL, P3_SPATIAL]);

    // P3 -> P4: stride-2 Conv2d + ReLU
    let w34 = b.add_input("down34_w", &[C_NECK, C_NECK, 3, 3]);
    let b34 = b.add_input("down34_b", &[C_NECK]);
    let p4 = b.add_conv2d(
        p3,
        w34,
        Some(b34),
        2,
        2,
        1,
        1,
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );
    let p4 = b.add_relu(p4, &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    // P4 -> P5: stride-2 Conv2d + ReLU
    let w45 = b.add_input("down45_w", &[C_NECK, C_NECK, 3, 3]);
    let b45 = b.add_input("down45_b", &[C_NECK]);
    let p5 = b.add_conv2d(
        p4,
        w45,
        Some(b45),
        2,
        2,
        1,
        1,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );
    let p5 = b.add_relu(p5, &[C_NECK, P5_SPATIAL, P5_SPATIAL]);

    b.build(p5).expect("valid PAN bottom-up kernel")
}

fn pan_bottomup_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_pan_bottomup_p3_to_p5_ibp() {
    let def = build_pan_bottomup();
    let bindings = pan_bottomup_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_NECK, P3_SPATIAL, P3_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN bottom-up P3 -> P5");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
        "PAN bottom-up downsamples P3 -> P5"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN bottom-up IBP: bounds=[{lo_min}, {hi_max}]");
    // ReLU output: lower >= 0
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. PAN top-down path (P5 -> P4 -> P3) bound propagation (IBP)
// ===========================================================================

/// Build PAN top-down pathway: ConvTranspose2d upsample from P5 -> P4 -> P3.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> lateral 1x1 -> upsample x2 -> upsample x2.
/// Output: [C_NECK, P3_SPATIAL, P3_SPATIAL].
fn build_pan_topdown() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pan_topdown");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1: C5 -> C_NECK
    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let p5 = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    // P5 -> P4: upsample via ConvTranspose2d(stride=2, k=4, p=1)
    let up54_w = b.add_input("up54_w", &[C_NECK, C_NECK, 4, 4]);
    let up54_b = b.add_input("up54_b", &[C_NECK]);
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // P4 -> P3: upsample via ConvTranspose2d(stride=2, k=4, p=1)
    let up43_w = b.add_input("up43_w", &[C_NECK, C_NECK, 4, 4]);
    let up43_b = b.add_input("up43_b", &[C_NECK]);
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
        &[C_NECK, P3_SPATIAL, P3_SPATIAL],
    );

    b.build(p3).expect("valid PAN top-down kernel")
}

fn pan_topdown_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // up54
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // up43
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_pan_topdown_p5_to_p3_ibp() {
    let def = build_pan_topdown();
    let bindings = pan_topdown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN top-down P5 -> P3");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P3_SPATIAL, P3_SPATIAL],
        "PAN top-down upsamples P5 -> P3"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN top-down IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. PAN bidirectional fusion bounds (IBP + CROWN)
// ===========================================================================

/// Build PAN bidirectional neck: top-down upsample then bottom-up downsample.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> lateral -> upsample -> smooth -> downsample.
/// Output: [C_NECK, P5_SPATIAL, P5_SPATIAL].
fn build_pan_bidirectional_fusion() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pan_bidirectional_fusion");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Top-down: lateral 1x1 + upsample to P4
    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    let up_w = b.add_input("up_w", &[C_NECK, C_NECK, 4, 4]);
    let up_b = b.add_input("up_b", &[C_NECK]);
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // Smoothing 3x3 conv at P4
    let smooth_w = b.add_input("smooth_w", &[C_NECK, C_NECK, 3, 3]);
    let smooth_b = b.add_input("smooth_b", &[C_NECK]);
    let smooth = b.add_conv2d(
        p4_up,
        smooth_w,
        Some(smooth_b),
        1,
        1,
        1,
        1,
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // Bottom-up: stride-2 downsample P4 back to P5
    let down_w = b.add_input("down_w", &[C_NECK, C_NECK, 3, 3]);
    let down_b = b.add_input("down_b", &[C_NECK]);
    let p5 = b.add_conv2d(
        smooth,
        down_w,
        Some(down_b),
        2,
        2,
        1,
        1,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );
    let out = b.add_relu(p5, &[C_NECK, P5_SPATIAL, P5_SPATIAL]);

    b.build(out).expect("valid PAN bidirectional fusion kernel")
}

fn pan_bidirectional_fusion_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // upsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // smooth
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // downsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_pan_bidirectional_fusion_ibp() {
    let def = build_pan_bidirectional_fusion();
    let bindings = pan_bidirectional_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PAN bidirectional fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
        "PAN bidirectional fusion output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN bidirectional fusion IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_pan_bidirectional_fusion_crown() {
    let def = build_pan_bidirectional_fusion();
    let bindings = pan_bidirectional_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN bidirectional fusion CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. FPN + PAN combined neck bounds (IBP)
// ===========================================================================

/// Build FPN + PAN combined: lateral 1x1 + top-down upsample + bottom-up downsample.
/// This models the full YOLO neck: backbone C5 -> FPN lateral -> upsample -> downsample.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_NECK, P4_SPATIAL, P4_SPATIAL].
fn build_fpn_pan_combined() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("fpn_pan_combined");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // FPN lateral: C5 -> C_NECK
    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    // FPN top-down: upsample P5 -> P4
    let up_w = b.add_input("up_w", &[C_NECK, C_NECK, 4, 4]);
    let up_b = b.add_input("up_b", &[C_NECK]);
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // FPN top-down: upsample P4 -> P3
    let up43_w = b.add_input("up43_w", &[C_NECK, C_NECK, 4, 4]);
    let up43_b = b.add_input("up43_b", &[C_NECK]);
    let p3_up = b.add_conv_transpose_2d(
        p4_up,
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
        &[C_NECK, P3_SPATIAL, P3_SPATIAL],
    );

    // PAN bottom-up: P3 -> P4 via stride-2 conv
    let down_w = b.add_input("down_w", &[C_NECK, C_NECK, 3, 3]);
    let down_b = b.add_input("down_b", &[C_NECK]);
    let p4_down = b.add_conv2d(
        p3_up,
        down_w,
        Some(down_b),
        2,
        2,
        1,
        1,
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );
    let out = b.add_relu(p4_down, &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    b.build(out).expect("valid FPN+PAN combined kernel")
}

fn fpn_pan_combined_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // up54
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // up43
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // downsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_fpn_pan_combined_neck_ibp() {
    let def = build_fpn_pan_combined();
    let bindings = fpn_pan_combined_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN+PAN combined neck");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
        "FPN+PAN combined outputs at P4"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FPN+PAN combined IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. C2f (Cross-Stage Partial) within neck bounds (IBP + CROWN)
// ===========================================================================

/// Build C2f block: split -> two 1x1 conv paths -> bottleneck conv + ReLU -> concat -> merge.
/// Input: [C_NECK, P4_SPATIAL, P4_SPATIAL] -> Output: [C_NECK, P4_SPATIAL, P4_SPATIAL].
fn build_c2f_neck() -> TensorKernelDef {
    let half_shape = [HALF_C, P4_SPATIAL, P4_SPATIAL];
    let out_shape = [C_NECK, P4_SPATIAL, P4_SPATIAL];

    let mut b = TensorBlockBuilder::new("c2f_neck");
    let x = b.add_input("neck_features", &out_shape);

    // C2f split: two 1x1 conv paths
    let split_w_a = b.add_input("split_w_a", &[HALF_C, C_NECK, 1, 1]);
    let split_w_b = b.add_input("split_w_b", &[HALF_C, C_NECK, 1, 1]);

    let path_a = b.add_conv2d(x, split_w_a, None, 1, 1, 0, 0, &half_shape);

    // Path B: bottleneck Conv3x3 -> ReLU
    let path_b_in = b.add_conv2d(x, split_w_b, None, 1, 1, 0, 0, &half_shape);
    let bn_w = b.add_input("bn_conv_w", &[HALF_C, HALF_C, 3, 3]);
    let bn_b = b.add_input("bn_conv_b", &[HALF_C]);
    let bn_out = b.add_conv2d(path_b_in, bn_w, Some(bn_b), 1, 1, 1, 1, &half_shape);
    let bn_out = b.add_relu(bn_out, &half_shape);

    // Concat path A + bottleneck path B
    let concat = b.add_concat(&[path_a, bn_out], 0, &out_shape);

    // Merge 1x1 conv
    let merge_w = b.add_input("merge_w", &[C_NECK, C_NECK, 1, 1]);
    let merge_b = b.add_input("merge_b", &[C_NECK]);
    let out = b.add_conv2d(concat, merge_w, Some(merge_b), 1, 1, 0, 0, &out_shape);

    b.build(out).expect("valid C2f neck kernel")
}

fn c2f_neck_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // split A
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HALF_C, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        // split B
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HALF_C, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        // bottleneck conv
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HALF_C, HALF_C, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HALF_C]), 0.0f32)),
        // merge
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_c2f_neck_ibp() {
    let def = build_c2f_neck();
    let bindings = c2f_neck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_NECK, P4_SPATIAL, P4_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through C2f neck block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
        "C2f preserves spatial and channel dims"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("C2f neck IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_c2f_neck_crown() {
    let def = build_c2f_neck();
    let bindings = c2f_neck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_NECK, P4_SPATIAL, P4_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P4_SPATIAL, P4_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("C2f neck CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. Multi-scale feature concatenation bounds (IBP)
// ===========================================================================

/// Build multi-scale feature concatenation: flatten P3, P4, P5 features and concat.
/// All features projected to C_NECK channels then concatenated along channel dim.
/// Uses P4 and P5 projected features as a simplified concat verification.
/// Two variable inputs: P4 [C_NECK, P4_SPATIAL, P4_SPATIAL] and P5 [C_NECK, P5_SPATIAL, P5_SPATIAL].
/// After 1x1 conv on each, concatenate along channel dim at P5 spatial.
fn build_multiscale_concat() -> TensorKernelDef {
    let cat_channels = C_NECK * 2;
    let mut b = TensorBlockBuilder::new("multiscale_concat");
    // Both inputs at P5 resolution (P4 features are downsampled before reaching here)
    let feat_a = b.add_input("feat_a", &[C_NECK, P5_SPATIAL, P5_SPATIAL]);
    let feat_b = b.add_input("feat_b", &[C_NECK, P5_SPATIAL, P5_SPATIAL]);

    // 1x1 conv on each
    let w_a = b.add_input("w_a", &[C_NECK, C_NECK, 1, 1]);
    let b_a = b.add_input("b_a", &[C_NECK]);
    let proj_a = b.add_conv2d(
        feat_a,
        w_a,
        Some(b_a),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    let w_b = b.add_input("w_b", &[C_NECK, C_NECK, 1, 1]);
    let b_b = b.add_input("b_b", &[C_NECK]);
    let proj_b = b.add_conv2d(
        feat_b,
        w_b,
        Some(b_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    // Concat along channel dim
    let cat = b.add_concat(
        &[proj_a, proj_b],
        0,
        &[cat_channels, P5_SPATIAL, P5_SPATIAL],
    );

    b.build(cat).expect("valid multi-scale concat kernel")
}

fn multiscale_concat_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // feat_a
        TensorParamBinding::Variable, // feat_b
        // proj A
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // proj B
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_multiscale_concat_ibp() {
    let cat_channels = C_NECK * 2;
    let def = build_multiscale_concat();
    let bindings = multiscale_concat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Combined variable input: feat_a + feat_b
    let total_elems = 2 * C_NECK * P5_SPATIAL * P5_SPATIAL;
    let input = uniform_bounds(&[total_elems], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale concat");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[cat_channels, P5_SPATIAL, P5_SPATIAL],
        "Concatenated feature channels = 2 * C_NECK"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale concat IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. Upsample + lateral connection composition (IBP)
// ===========================================================================

/// Build upsample + lateral add: ConvTranspose2d upsample P5 + lateral P4.
/// Two variable inputs.
/// Output: [C_NECK, P4_SPATIAL, P4_SPATIAL].
fn build_upsample_lateral() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("upsample_lateral");
    let p5_lat = b.add_input("p5_lateral", &[C_NECK, P5_SPATIAL, P5_SPATIAL]);
    let p4_lat = b.add_input("p4_lateral", &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    // Upsample P5 to P4 resolution
    let up_w = b.add_input("up_w", &[C_NECK, C_NECK, 4, 4]);
    let up_b = b.add_input("up_b", &[C_NECK]);
    let p5_up = b.add_conv_transpose_2d(
        p5_lat,
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // Element-wise add: upsampled P5 + P4 lateral
    let fused = b.add_binary_add(p5_up, p4_lat, &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    b.build(fused).expect("valid upsample+lateral kernel")
}

fn upsample_lateral_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // p5_lateral
        TensorParamBinding::Variable, // p4_lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_upsample_lateral_ibp() {
    let def = build_upsample_lateral();
    let bindings = upsample_lateral_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_elems = C_NECK * P5_SPATIAL * P5_SPATIAL + C_NECK * P4_SPATIAL * P4_SPATIAL;
    let input = uniform_bounds(&[total_elems], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through upsample+lateral");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
        "Fused output at P4 resolution"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Upsample+lateral IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Downsample (stride-2 conv) + fusion bounds (IBP + CROWN)
// ===========================================================================

/// Build downsample + fusion: stride-2 Conv2d on P3 + add P4 features.
/// Two variable inputs.
/// Output: [C_NECK, P4_SPATIAL, P4_SPATIAL].
fn build_downsample_fusion() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("downsample_fusion");
    let p3 = b.add_input("p3_features", &[C_NECK, P3_SPATIAL, P3_SPATIAL]);
    let p4 = b.add_input("p4_features", &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    // Downsample P3 -> P4 via stride-2 conv
    let down_w = b.add_input("down_w", &[C_NECK, C_NECK, 3, 3]);
    let down_b = b.add_input("down_b", &[C_NECK]);
    let p3_down = b.add_conv2d(
        p3,
        down_w,
        Some(down_b),
        2,
        2,
        1,
        1,
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // Fusion: downsample + P4 features
    let fused = b.add_binary_add(p3_down, p4, &[C_NECK, P4_SPATIAL, P4_SPATIAL]);
    let out = b.add_relu(fused, &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    b.build(out).expect("valid downsample+fusion kernel")
}

fn downsample_fusion_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // p3
        TensorParamBinding::Variable, // p4
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_downsample_fusion_ibp() {
    let def = build_downsample_fusion();
    let bindings = downsample_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_elems = C_NECK * P3_SPATIAL * P3_SPATIAL + C_NECK * P4_SPATIAL * P4_SPATIAL;
    let input = uniform_bounds(&[total_elems], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through downsample+fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
        "Fused output at P4 resolution"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Downsample+fusion IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_downsample_fusion_crown() {
    let def = build_downsample_fusion();
    let bindings = downsample_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_elems = C_NECK * P3_SPATIAL * P3_SPATIAL + C_NECK * P4_SPATIAL * P4_SPATIAL;
    let input = uniform_bounds(&[total_elems], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P4_SPATIAL, P4_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Downsample+fusion CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 9. 3-level vs 4-level feature pyramid comparison (IBP)
// ===========================================================================

/// Build 3-level pyramid: C5 -> lateral -> upsample x2 -> P3.
fn build_3level_pyramid() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pyramid_3level");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let p5 = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    let up54_w = b.add_input("up54_w", &[C_NECK, C_NECK, 4, 4]);
    let up54_b = b.add_input("up54_b", &[C_NECK]);
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    let up43_w = b.add_input("up43_w", &[C_NECK, C_NECK, 4, 4]);
    let up43_b = b.add_input("up43_b", &[C_NECK]);
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
        &[C_NECK, P3_SPATIAL, P3_SPATIAL],
    );

    b.build(p3).expect("valid 3-level pyramid kernel")
}

fn pyramid_3level_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

/// Build 4-level pyramid: C5 -> lateral -> upsample x3 -> P2.
fn build_4level_pyramid() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pyramid_4level");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let p5 = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    let up54_w = b.add_input("up54_w", &[C_NECK, C_NECK, 4, 4]);
    let up54_b = b.add_input("up54_b", &[C_NECK]);
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    let up43_w = b.add_input("up43_w", &[C_NECK, C_NECK, 4, 4]);
    let up43_b = b.add_input("up43_b", &[C_NECK]);
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
        &[C_NECK, P3_SPATIAL, P3_SPATIAL],
    );

    let up32_w = b.add_input("up32_w", &[C_NECK, C_NECK, 4, 4]);
    let up32_b = b.add_input("up32_b", &[C_NECK]);
    let p2 = b.add_conv_transpose_2d(
        p3,
        up32_w,
        Some(up32_b),
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        &[C_NECK, P2_SPATIAL, P2_SPATIAL],
    );

    b.build(p2).expect("valid 4-level pyramid kernel")
}

fn pyramid_4level_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_3level_vs_4level_pyramid_ibp() {
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    // 3-level pyramid
    let def3 = build_3level_pyramid();
    let graph3 = tensor_kernel_to_graph(&def3, &pyramid_3level_bindings()).expect("graph 3-level");
    let out3 = graph3.propagate_ibp(&input).expect("IBP 3-level");
    assert_bounds_valid(&out3);
    let w3 = total_bound_width(&out3);

    // 4-level pyramid
    let def4 = build_4level_pyramid();
    let graph4 = tensor_kernel_to_graph(&def4, &pyramid_4level_bindings()).expect("graph 4-level");
    let out4 = graph4.propagate_ibp(&input).expect("IBP 4-level");
    assert_bounds_valid(&out4);
    let w4 = total_bound_width(&out4);

    eprintln!("3-level pyramid width={w3:.4}, 4-level pyramid width={w4:.4}");

    // 4-level should be wider (more upsample stages)
    assert!(
        w4 >= w3 - 1e-6,
        "4-level pyramid should have >= bound width than 3-level: w4={w4} < w3={w3}"
    );
}

// ===========================================================================
// 10. Scale-specific bound widths tracking (IBP)
// ===========================================================================

/// Track bound widths at each pyramid level from the same backbone input.
/// Verifies coarser levels produce tighter bounds than finer levels.
#[test]
fn test_scale_specific_bound_widths_ibp() {
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    // P5 level: just lateral 1x1
    let mut b = TensorBlockBuilder::new("scale_p5");
    let c5 = b.add_input("c5", &[C5, P5_SPATIAL, P5_SPATIAL]);
    let w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let bi = b.add_input("lat_b", &[C_NECK]);
    let out = b.add_conv2d(
        c5,
        w,
        Some(bi),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );
    let def_p5 = b.build(out).expect("valid P5 kernel");

    let bindings_p5 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ];

    let graph_p5 = tensor_kernel_to_graph(&def_p5, &bindings_p5).expect("graph P5");
    let out_p5 = graph_p5.propagate_ibp(&input).expect("IBP P5");
    assert_bounds_valid(&out_p5);
    let w_p5 = total_bound_width(&out_p5);

    // P4 level: lateral + upsample
    let def_topdown = build_pan_topdown();
    let graph_topdown =
        tensor_kernel_to_graph(&def_topdown, &pan_topdown_bindings()).expect("graph top-down");
    let out_p3 = graph_topdown.propagate_ibp(&input).expect("IBP P3");
    assert_bounds_valid(&out_p3);
    let w_p3 = total_bound_width(&out_p3);

    eprintln!("Scale-specific widths: P5={w_p5:.4}, P3={w_p3:.4}");

    // Both must be finite
    assert!(w_p5.is_finite(), "P5 width must be finite");
    assert!(w_p3.is_finite(), "P3 width must be finite");
    // More upsample levels -> wider bounds
    assert!(
        w_p3 >= w_p5 - 1e-6,
        "P3 (more layers) should have >= width than P5: w_p3={w_p3} < w_p5={w_p5}"
    );
}

// ===========================================================================
// 11. Feature dimension alignment across scales (IBP)
// ===========================================================================

/// Build dimension alignment: project C3, C4, C5 backbone features to C_NECK.
/// Verifies all three laterals produce same output channel count.
/// Tests C5 (largest channel reduction: 256 -> 64).
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_NECK, P5_SPATIAL, P5_SPATIAL].
fn build_dim_alignment() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dim_alignment");
    let c5_feat = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1: C5 -> C_NECK (4:1 channel reduction)
    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let out = b.add_conv2d(
        c5_feat,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    b.build(out).expect("valid dim alignment kernel")
}

fn dim_alignment_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_dim_alignment_across_scales_ibp() {
    // Test C5 -> C_NECK (4:1 reduction, hardest case)
    let def = build_dim_alignment();
    let bindings = dim_alignment_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dimension alignment");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
        "Lateral 1x1 projects C5={C5} -> C_NECK={C_NECK}"
    );
    assert_bounds_valid(&output);

    // Verify the channel ratio
    assert_eq!(C5 / C_NECK, 4, "C5/C_NECK = 4:1 reduction");

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dim alignment (C5->C_NECK) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. PAN with SPPF bounds (IBP)
// ===========================================================================

/// Build SPPF (Spatial Pyramid Pooling - Fast): cascaded MaxPool2d + concat.
/// YOLO-v8 applies 3 cascaded 5x5 MaxPool2d with padding=2 (same spatial),
/// then concatenates the original + 3 pooled features.
/// Input: [C_NECK, P5_SPATIAL, P5_SPATIAL] -> 4 * C_NECK channels -> 1x1 merge.
/// Output: [C_NECK, P5_SPATIAL, P5_SPATIAL].
fn build_pan_sppf() -> TensorKernelDef {
    let p5_shape = [C_NECK, P5_SPATIAL, P5_SPATIAL];
    let cat_c = C_NECK * 4;
    let cat_shape = [cat_c, P5_SPATIAL, P5_SPATIAL];

    let mut b = TensorBlockBuilder::new("pan_sppf");
    let x = b.add_input("p5_features", &p5_shape);

    // 3 cascaded MaxPool2d (5x5, stride=1, pad=2 -> same spatial)
    let pool1 = b.add_max_pool_2d(
        x,
        SPPF_KERNEL,
        SPPF_KERNEL,
        1,
        1,
        SPPF_PAD,
        SPPF_PAD,
        &p5_shape,
    );
    let pool2 = b.add_max_pool_2d(
        pool1,
        SPPF_KERNEL,
        SPPF_KERNEL,
        1,
        1,
        SPPF_PAD,
        SPPF_PAD,
        &p5_shape,
    );
    let pool3 = b.add_max_pool_2d(
        pool2,
        SPPF_KERNEL,
        SPPF_KERNEL,
        1,
        1,
        SPPF_PAD,
        SPPF_PAD,
        &p5_shape,
    );

    // Concat original + 3 pooled features
    let cat = b.add_concat(&[x, pool1, pool2, pool3], 0, &cat_shape);

    // 1x1 conv merge: 4*C_NECK -> C_NECK
    let merge_w = b.add_input("merge_w", &[C_NECK, cat_c, 1, 1]);
    let merge_b = b.add_input("merge_b", &[C_NECK]);
    let out = b.add_conv2d(cat, merge_w, Some(merge_b), 1, 1, 0, 0, &p5_shape);

    b.build(out).expect("valid PAN SPPF kernel")
}

fn pan_sppf_bindings() -> Vec<TensorParamBinding> {
    let cat_c = C_NECK * 4;
    vec![
        TensorParamBinding::Variable,
        // merge
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, cat_c, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_pan_sppf_ibp() {
    let def = build_pan_sppf();
    let bindings = pan_sppf_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_NECK, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through PAN SPPF");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
        "SPPF preserves spatial dims and merges channels back to C_NECK"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PAN SPPF IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Neck output normalization bounds (IBP + CROWN)
// ===========================================================================

/// Build neck output with normalization: lateral 1x1 + ReLU activation.
/// ReLU ensures output lower bounds are >= 0 (bounded normalization effect).
/// Input: [C5, P5_SPATIAL, P5_SPATIAL] -> Output: [C_NECK, P5_SPATIAL, P5_SPATIAL].
fn build_neck_output_norm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("neck_output_norm");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1
    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    // 3x3 smoothing conv
    let smooth_w = b.add_input("smooth_w", &[C_NECK, C_NECK, 3, 3]);
    let smooth_b = b.add_input("smooth_b", &[C_NECK]);
    let smooth = b.add_conv2d(
        lat,
        smooth_w,
        Some(smooth_b),
        1,
        1,
        1,
        1,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    // ReLU normalization: bounds clipped at 0
    let out = b.add_relu(smooth, &[C_NECK, P5_SPATIAL, P5_SPATIAL]);

    b.build(out).expect("valid neck output norm kernel")
}

fn neck_output_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // smooth
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
    ]
}

#[test]
fn test_neck_output_norm_ibp() {
    let def = build_neck_output_norm();
    let bindings = neck_output_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through neck output normalization");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
        "Neck output norm shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Neck output norm IBP: bounds=[{lo_min}, {hi_max}]");
    // ReLU output: lower >= 0
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_neck_output_norm_crown() {
    let def = build_neck_output_norm();
    let bindings = neck_output_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C_NECK, P5_SPATIAL, P5_SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Neck output norm CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. Multi-scale detection head input bounds (IBP)
// ===========================================================================

/// Build per-level detection head: 1x1 conv -> sigmoid at each FPN level.
/// Uses P5 level for verification.
/// Input: [C_NECK, P5_SPATIAL, P5_SPATIAL] -> Output: [NUM_CLASSES, P5_SPATIAL, P5_SPATIAL].
fn build_multiscale_detection_head() -> TensorKernelDef {
    let out_shape = [NUM_CLASSES, P5_SPATIAL, P5_SPATIAL];
    let mut b = TensorBlockBuilder::new("multiscale_detection_head");
    let features = b.add_input("neck_features", &[C_NECK, P5_SPATIAL, P5_SPATIAL]);

    let det_w = b.add_input("det_w", &[NUM_CLASSES, C_NECK, 1, 1]);
    let det_b = b.add_input("det_b", &[NUM_CLASSES]);
    let logits = b.add_conv2d(features, det_w, Some(det_b), 1, 1, 0, 0, &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid multi-scale detection head kernel")
}

fn multiscale_detection_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_multiscale_detection_head_ibp() {
    let def = build_multiscale_detection_head();
    let bindings = multiscale_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C_NECK, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale detection head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CLASSES, P5_SPATIAL, P5_SPATIAL],
        "Detection head output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale detection head IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 15. Full neck pipeline: backbone features -> PAN -> detection heads (IBP)
// ===========================================================================

/// Build full neck pipeline:
/// C5 -> lateral 1x1 -> upsample -> smooth -> ReLU -> downsample ->
///   ReLU -> detection 1x1 -> sigmoid.
/// Input: [C5, P5_SPATIAL, P5_SPATIAL]
/// Output: [NUM_CLASSES, P5_SPATIAL, P5_SPATIAL].
fn build_full_pan_pipeline() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("full_pan_pipeline");
    let c5 = b.add_input("c5_features", &[C5, P5_SPATIAL, P5_SPATIAL]);

    // Lateral 1x1: C5 -> C_NECK
    let lat_w = b.add_input("lat_w", &[C_NECK, C5, 1, 1]);
    let lat_b = b.add_input("lat_b", &[C_NECK]);
    let lat = b.add_conv2d(
        c5,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );

    // Top-down: upsample to P4
    let up_w = b.add_input("up_w", &[C_NECK, C_NECK, 4, 4]);
    let up_b = b.add_input("up_b", &[C_NECK]);
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
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );

    // Smoothing 3x3 at P4
    let smooth_w = b.add_input("smooth_w", &[C_NECK, C_NECK, 3, 3]);
    let smooth_b = b.add_input("smooth_b", &[C_NECK]);
    let smooth = b.add_conv2d(
        p4_up,
        smooth_w,
        Some(smooth_b),
        1,
        1,
        1,
        1,
        &[C_NECK, P4_SPATIAL, P4_SPATIAL],
    );
    let smooth = b.add_relu(smooth, &[C_NECK, P4_SPATIAL, P4_SPATIAL]);

    // Bottom-up: downsample P4 -> P5
    let down_w = b.add_input("down_w", &[C_NECK, C_NECK, 3, 3]);
    let down_b = b.add_input("down_b", &[C_NECK]);
    let p5_down = b.add_conv2d(
        smooth,
        down_w,
        Some(down_b),
        2,
        2,
        1,
        1,
        &[C_NECK, P5_SPATIAL, P5_SPATIAL],
    );
    let p5_down = b.add_relu(p5_down, &[C_NECK, P5_SPATIAL, P5_SPATIAL]);

    // Detection head: 1x1 conv + sigmoid
    let det_w = b.add_input("det_w", &[NUM_CLASSES, C_NECK, 1, 1]);
    let det_b = b.add_input("det_b", &[NUM_CLASSES]);
    let logits = b.add_conv2d(
        p5_down,
        det_w,
        Some(det_b),
        1,
        1,
        0,
        0,
        &[NUM_CLASSES, P5_SPATIAL, P5_SPATIAL],
    );
    let out = b.add_sigmoid(logits, &[NUM_CLASSES, P5_SPATIAL, P5_SPATIAL]);

    b.build(out).expect("valid full PAN pipeline kernel")
}

fn full_pan_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // lateral
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C5, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // upsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 4, 4]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // smooth
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // downsample
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[C_NECK, C_NECK, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C_NECK]), 0.0f32)),
        // detection head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, C_NECK, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32)),
    ]
}

#[test]
fn test_full_pan_pipeline_ibp() {
    let def = build_full_pan_pipeline();
    let bindings = full_pan_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C5, P5_SPATIAL, P5_SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full PAN pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CLASSES, P5_SPATIAL, P5_SPATIAL],
        "Full PAN pipeline -> detection head output"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full PAN pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}
