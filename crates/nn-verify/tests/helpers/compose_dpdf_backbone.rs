// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Backbone feature extraction architecture NY composition.
//!
//! Verifies IBP and CROWN bounds propagation through backbone architectures
//! used across dpdf document understanding models:
//!
//! **ResNet backbone (Table Transformer):**
//! 1. BasicBlock: Conv2d -> BN -> ReLU -> Conv2d -> BN + skip -> ReLU (IBP + CROWN)
//! 2. ResNet stage: 2 BasicBlocks with stride-2 downsampling (IBP)
//! 3. 2-stage backbone: cascaded stride-2 stages (IBP)
//! 4. Feature map spatial dimensions: verify halving per stage (IBP)
//!
//! **YOLO backbone (DocLayout-YOLO):**
//! 5. ConvBnAct stack: 3 cascaded Conv-BN-SiLU blocks (IBP)
//! 6. C2f block: cross-stage partial with bottleneck (IBP + CROWN)
//! 7. SPPF integration: multi-scale pooling at backbone output (IBP)
//! 8. Backbone -> neck connection: feature pyramid (IBP)
//!
//! **ViT backbone (Granite-Docling, Qwen3-VL):**
//! 9.  Patch embed -> encoder block composition (IBP)
//! 10. 2-block ViT encoder stack (IBP + CROWN)
//! 11. Window attention ViT: local attention with partition (IBP)
//! 12. Deep stack fusion: multi-level feature combination (IBP)
//!
//! **Cross-architecture:**
//! 13. Backbone output dimension comparison across models (IBP)
//! 14. Backbone monotone tightening: smaller eps -> tighter features (IBP)
//! 15. Backbone -> head projection composition (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - Spatial: 8x8 input, 4x4 / 2x2 after stride-2 stages
//! - Channels: 16 -> 32 -> 64 through stages
//! - ViT: SEQ_LEN=4, HIDDEN_DIM=64
//!
//! Part of #3992: NY compose tests for backbone architectures.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef, TensorNodeId};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Spatial size of backbone input feature maps.
const SPATIAL: usize = 8;
/// Stage 1 channels (ResNet / YOLO).
const C1: usize = 16;
/// Stage 2 channels (after first downsampling).
const C2: usize = 32;
/// Stage 3 channels (after second downsampling).
const C3: usize = 64;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// ViT hidden dimension.
const HIDDEN_DIM: usize = 64;
/// ViT sequence length (number of patches).
const SEQ_LEN: usize = 4;
/// ViT FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Number of attention heads.
const NUM_HEADS: usize = 4;

// ===========================================================================
// Helper: ResNet BasicBlock builder (reusable for stage composition)
// ===========================================================================

/// Build bindings for a ResNet BasicBlock with given channel count.
fn basic_block_bindings(c: usize) -> Vec<TensorParamBinding> {
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    // Two conv-BN paths: conv1_w, conv1_b, bn1_mean, bn1_var, bn1_weight, bn1_bias, bn1_eps,
    //                     conv2_w, conv2_b, bn2_mean, bn2_var, bn2_weight, bn2_bias, bn2_eps
    vec![
        // conv1
        TensorParamBinding::ConstantTensor(conv_w.clone()),
        TensorParamBinding::ConstantTensor(conv_b.clone()),
        // bn1
        TensorParamBinding::ConstantTensor(bn_mean.clone()),
        TensorParamBinding::ConstantTensor(bn_var.clone()),
        TensorParamBinding::ConstantTensor(bn_weight.clone()),
        TensorParamBinding::ConstantTensor(bn_bias.clone()),
        TensorParamBinding::ConstantScalar(1e-5),
        // conv2
        TensorParamBinding::ConstantTensor(conv_w),
        TensorParamBinding::ConstantTensor(conv_b),
        // bn2
        TensorParamBinding::ConstantTensor(bn_mean),
        TensorParamBinding::ConstantTensor(bn_var),
        TensorParamBinding::ConstantTensor(bn_weight),
        TensorParamBinding::ConstantTensor(bn_bias),
        TensorParamBinding::ConstantScalar(1e-5),
    ]
}

/// Append a ResNet BasicBlock to a TensorBlockBuilder graph.
///
/// Conv2d(k=3,s=1,p=1) -> BN -> ReLU -> Conv2d(k=3,s=1,p=1) -> BN + skip -> ReLU.
/// Returns the output node ID.
fn append_basic_block(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    c: usize,
    s: usize,
) -> TensorNodeId {
    let feat_shape = [c, s, s];

    let conv1_w = b.add_input("bb_conv1_w", &[c, c, 3, 3]);
    let conv1_b = b.add_input("bb_conv1_b", &[c]);
    let bn1_mean = b.add_input("bb_bn1_mean", &[c]);
    let bn1_var = b.add_input("bb_bn1_var", &[c]);
    let bn1_weight = b.add_input("bb_bn1_weight", &[c]);
    let bn1_bias = b.add_input("bb_bn1_bias", &[c]);
    let bn1_eps = b.add_input("bb_bn1_eps", &[1]);

    let conv2_w = b.add_input("bb_conv2_w", &[c, c, 3, 3]);
    let conv2_b = b.add_input("bb_conv2_b", &[c]);
    let bn2_mean = b.add_input("bb_bn2_mean", &[c]);
    let bn2_var = b.add_input("bb_bn2_var", &[c]);
    let bn2_weight = b.add_input("bb_bn2_weight", &[c]);
    let bn2_bias = b.add_input("bb_bn2_bias", &[c]);
    let bn2_eps = b.add_input("bb_bn2_eps", &[1]);

    let conv1_out = b.add_conv2d(input, conv1_w, Some(conv1_b), 1, 1, 1, 1, &feat_shape);
    let bn1_out = b.add_batch_norm(
        conv1_out,
        bn1_mean,
        bn1_var,
        bn1_weight,
        bn1_bias,
        bn1_eps,
        &feat_shape,
    );
    let relu1 = b.add_relu(bn1_out, &feat_shape);

    let conv2_out = b.add_conv2d(relu1, conv2_w, Some(conv2_b), 1, 1, 1, 1, &feat_shape);
    let bn2_out = b.add_batch_norm(
        conv2_out,
        bn2_mean,
        bn2_var,
        bn2_weight,
        bn2_bias,
        bn2_eps,
        &feat_shape,
    );

    let residual = b.add_binary_add(bn2_out, input, &feat_shape);
    b.add_relu(residual, &feat_shape)
}

/// Compute total bound width (sum of hi - lo across all elements).
fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    lo.iter().zip(hi.iter()).map(|(&l, &h)| h - l).sum::<f32>()
}

// ===========================================================================
// 1. ResNet BasicBlock: Conv-BN-ReLU-Conv-BN + skip -> ReLU (IBP + CROWN)
// ===========================================================================

/// Build a single ResNet BasicBlock.
///
/// Input: [C1, SPATIAL, SPATIAL] (Variable).
/// Output: [C1, SPATIAL, SPATIAL].
fn build_resnet_basic_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("backbone_resnet_basic_block");
    let input = b.add_input("features", &[C1, SPATIAL, SPATIAL]);
    let out = append_basic_block(&mut b, input, C1, SPATIAL);
    b.build(out).expect("valid ResNet basic block kernel")
}

fn resnet_basic_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(basic_block_bindings(C1));
    bindings
}

#[test]
fn test_backbone_resnet_basic_block_ibp() {
    let def = build_resnet_basic_block();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through BasicBlock");

    assert_eq!(output.lower_upper().0.shape(), &[C1, SPATIAL, SPATIAL]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ResNet BasicBlock IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU output has non-negative lower bound
    assert!(
        lo_min >= -1e-6,
        "BasicBlock output lower >= 0 (ReLU), got {lo_min}"
    );
}

#[test]
fn test_backbone_resnet_basic_block_crown() {
    let def = build_resnet_basic_block();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[C1, SPATIAL, SPATIAL]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ResNet BasicBlock CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 2. ResNet stage: 2 BasicBlocks with stride-2 downsampling (IBP)
// ===========================================================================

/// Build a ResNet stage: Conv2d(stride=2) downsample -> 2 BasicBlocks.
///
/// Input: [C1, SPATIAL, SPATIAL] -> downsample -> [C2, SPATIAL/2, SPATIAL/2]
/// -> BasicBlock -> BasicBlock.
fn build_resnet_stage() -> TensorKernelDef {
    let s_out = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("backbone_resnet_stage");

    let input = b.add_input("features", &[C1, SPATIAL, SPATIAL]);

    // Stride-2 downsample with channel expansion: Conv2d(C1 -> C2, k=3, s=2, p=1)
    let ds_w = b.add_input("ds_weight", &[C2, C1, 3, 3]);
    let ds_b = b.add_input("ds_bias", &[C2]);
    let downsampled = b.add_conv2d(input, ds_w, Some(ds_b), 2, 2, 1, 1, &[C2, s_out, s_out]);
    let downsampled = b.add_relu(downsampled, &[C2, s_out, s_out]);

    // BasicBlock 1
    let block1 = append_basic_block(&mut b, downsampled, C2, s_out);
    // BasicBlock 2
    let block2 = append_basic_block(&mut b, block1, C2, s_out);

    b.build(block2).expect("valid ResNet stage kernel")
}

fn resnet_stage_bindings() -> Vec<TensorParamBinding> {
    let ds_w = ArrayD::from_elem(IxDyn(&[C2, C1, 3, 3]), WEIGHT_MAG);
    let ds_b = ArrayD::from_elem(IxDyn(&[C2]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ds_w),
        TensorParamBinding::ConstantTensor(ds_b),
    ];
    bindings.extend(basic_block_bindings(C2));
    bindings.extend(basic_block_bindings(C2));
    bindings
}

#[test]
fn test_backbone_resnet_stage_ibp() {
    let def = build_resnet_stage();
    let bindings = resnet_stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ResNet stage");

    let s_out = SPATIAL / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[C2, s_out, s_out],
        "ResNet stage output shape: downsampled spatial and expanded channels"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ResNet stage IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. 2-stage backbone: cascaded stride-2 stages (IBP)
// ===========================================================================

/// Build a 2-stage ResNet backbone: stage1 + stage2 with cascaded downsampling.
///
/// Input: [C1, SPATIAL, SPATIAL] -> stage1 -> [C2, SPATIAL/2, SPATIAL/2]
/// -> stage2 -> [C3, SPATIAL/4, SPATIAL/4].
fn build_resnet_2stage_backbone() -> TensorKernelDef {
    let s1 = SPATIAL / 2;
    let s2 = SPATIAL / 4;
    let mut b = TensorBlockBuilder::new("backbone_resnet_2stage");

    let input = b.add_input("features", &[C1, SPATIAL, SPATIAL]);

    // Stage 1: downsample C1 -> C2, spatial SPATIAL -> SPATIAL/2
    let ds1_w = b.add_input("ds1_weight", &[C2, C1, 3, 3]);
    let ds1_b = b.add_input("ds1_bias", &[C2]);
    let stage1 = b.add_conv2d(input, ds1_w, Some(ds1_b), 2, 2, 1, 1, &[C2, s1, s1]);
    let stage1 = b.add_relu(stage1, &[C2, s1, s1]);

    // BasicBlock in stage 1
    let stage1 = append_basic_block(&mut b, stage1, C2, s1);

    // Stage 2: downsample C2 -> C3, spatial SPATIAL/2 -> SPATIAL/4
    let ds2_w = b.add_input("ds2_weight", &[C3, C2, 3, 3]);
    let ds2_b = b.add_input("ds2_bias", &[C3]);
    let stage2 = b.add_conv2d(stage1, ds2_w, Some(ds2_b), 2, 2, 1, 1, &[C3, s2, s2]);
    let stage2 = b.add_relu(stage2, &[C3, s2, s2]);

    // BasicBlock in stage 2
    let stage2 = append_basic_block(&mut b, stage2, C3, s2);

    b.build(stage2)
        .expect("valid 2-stage ResNet backbone kernel")
}

fn resnet_2stage_bindings() -> Vec<TensorParamBinding> {
    let ds1_w = ArrayD::from_elem(IxDyn(&[C2, C1, 3, 3]), WEIGHT_MAG);
    let ds1_b = ArrayD::from_elem(IxDyn(&[C2]), 0.0f32);
    let ds2_w = ArrayD::from_elem(IxDyn(&[C3, C2, 3, 3]), WEIGHT_MAG);
    let ds2_b = ArrayD::from_elem(IxDyn(&[C3]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ds1_w),
        TensorParamBinding::ConstantTensor(ds1_b),
    ];
    bindings.extend(basic_block_bindings(C2));
    bindings.push(TensorParamBinding::ConstantTensor(ds2_w));
    bindings.push(TensorParamBinding::ConstantTensor(ds2_b));
    bindings.extend(basic_block_bindings(C3));
    bindings
}

#[test]
fn test_backbone_resnet_2stage_ibp() {
    let def = build_resnet_2stage_backbone();
    let bindings = resnet_2stage_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-stage ResNet");

    let s2 = SPATIAL / 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[C3, s2, s2],
        "2-stage backbone output: 2x downsampled spatial, C3 channels"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ResNet 2-stage backbone IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Feature map spatial dimensions: verify halving per stage (IBP)
// ===========================================================================

/// Verify that spatial dimensions halve per stage through cascaded stride-2 convs.
///
/// Stage 0: [C1, 8, 8], Stage 1: [C2, 4, 4], Stage 2: [C3, 2, 2].
#[test]
fn test_backbone_spatial_halving_per_stage() {
    // Stage 0 -> Stage 1: 8x8 -> 4x4
    let s0 = SPATIAL;
    let s1 = SPATIAL / 2;
    {
        let mut b = TensorBlockBuilder::new("backbone_halving_s0s1");
        let input = b.add_input("features", &[C1, s0, s0]);
        let w = b.add_input("ds_w", &[C2, C1, 3, 3]);
        let bias = b.add_input("ds_b", &[C2]);
        let out = b.add_conv2d(input, w, Some(bias), 2, 2, 1, 1, &[C2, s1, s1]);
        let def = b.build(out).expect("valid s0->s1 kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[C2, C1, 3, 3]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2]), 0.0f32)),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input_bounds = uniform_bounds(&[C1, s0, s0], 1.0);
        let output = graph.propagate_ibp(&input_bounds).expect("IBP s0->s1");

        assert_eq!(
            output.lower_upper().0.shape(),
            &[C2, s1, s1],
            "Stage 0->1: spatial halves from {s0} to {s1}"
        );
        assert_bounds_valid(&output);
        eprintln!("Stage 0->1: [{C1}, {s0}, {s0}] -> [{C2}, {s1}, {s1}]");
    }

    // Stage 1 -> Stage 2: 4x4 -> 2x2
    let s2 = SPATIAL / 4;
    {
        let mut b = TensorBlockBuilder::new("backbone_halving_s1s2");
        let input = b.add_input("features", &[C2, s1, s1]);
        let w = b.add_input("ds_w", &[C3, C2, 3, 3]);
        let bias = b.add_input("ds_b", &[C3]);
        let out = b.add_conv2d(input, w, Some(bias), 2, 2, 1, 1, &[C3, s2, s2]);
        let def = b.build(out).expect("valid s1->s2 kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[C3, C2, 3, 3]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C3]), 0.0f32)),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input_bounds = uniform_bounds(&[C2, s1, s1], 1.0);
        let output = graph.propagate_ibp(&input_bounds).expect("IBP s1->s2");

        assert_eq!(
            output.lower_upper().0.shape(),
            &[C3, s2, s2],
            "Stage 1->2: spatial halves from {s1} to {s2}"
        );
        assert_bounds_valid(&output);
        eprintln!("Stage 1->2: [{C2}, {s1}, {s1}] -> [{C3}, {s2}, {s2}]");
    }
}

// ===========================================================================
// 5. YOLO ConvBnAct stack: 3 cascaded Conv-BN-SiLU blocks (IBP)
// ===========================================================================

/// Build a YOLO-style ConvBnAct stack: 3 cascaded Conv2d -> BN -> SiLU.
///
/// Input: [C1, SPATIAL, SPATIAL].
/// Block 1: Conv(C1 -> C1, k=3, s=1, p=1) -> BN -> SiLU
/// Block 2: Conv(C1 -> C2, k=3, s=2, p=1) -> BN -> SiLU  (downsample)
/// Block 3: Conv(C2 -> C2, k=3, s=1, p=1) -> BN -> SiLU
/// Output: [C2, SPATIAL/2, SPATIAL/2].
fn build_yolo_conv_bn_act_stack() -> TensorKernelDef {
    let s0 = SPATIAL;
    let s1 = SPATIAL / 2;
    let mut b = TensorBlockBuilder::new("backbone_yolo_conv_bn_act");

    let input = b.add_input("features", &[C1, s0, s0]);

    // Block 1: C1 -> C1, stride=1
    let w1 = b.add_input("conv1_w", &[C1, C1, 3, 3]);
    let b1 = b.add_input("conv1_b", &[C1]);
    let bn1_mean = b.add_input("bn1_mean", &[C1]);
    let bn1_var = b.add_input("bn1_var", &[C1]);
    let bn1_weight = b.add_input("bn1_weight", &[C1]);
    let bn1_bias = b.add_input("bn1_bias", &[C1]);
    let bn1_eps = b.add_input("bn1_eps", &[1]);
    let conv1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 1, 1, &[C1, s0, s0]);
    let bn1 = b.add_batch_norm(
        conv1,
        bn1_mean,
        bn1_var,
        bn1_weight,
        bn1_bias,
        bn1_eps,
        &[C1, s0, s0],
    );
    // SiLU = x * sigmoid(x)
    let sig1 = b.add_sigmoid(bn1, &[C1, s0, s0]);
    let silu1 = b.add_binary_mul(bn1, sig1, &[C1, s0, s0]);

    // Block 2: C1 -> C2, stride=2 (downsample)
    let w2 = b.add_input("conv2_w", &[C2, C1, 3, 3]);
    let b2 = b.add_input("conv2_b", &[C2]);
    let bn2_mean = b.add_input("bn2_mean", &[C2]);
    let bn2_var = b.add_input("bn2_var", &[C2]);
    let bn2_weight = b.add_input("bn2_weight", &[C2]);
    let bn2_bias = b.add_input("bn2_bias", &[C2]);
    let bn2_eps = b.add_input("bn2_eps", &[1]);
    let conv2 = b.add_conv2d(silu1, w2, Some(b2), 2, 2, 1, 1, &[C2, s1, s1]);
    let bn2 = b.add_batch_norm(
        conv2,
        bn2_mean,
        bn2_var,
        bn2_weight,
        bn2_bias,
        bn2_eps,
        &[C2, s1, s1],
    );
    let sig2 = b.add_sigmoid(bn2, &[C2, s1, s1]);
    let silu2 = b.add_binary_mul(bn2, sig2, &[C2, s1, s1]);

    // Block 3: C2 -> C2, stride=1
    let w3 = b.add_input("conv3_w", &[C2, C2, 3, 3]);
    let b3 = b.add_input("conv3_b", &[C2]);
    let bn3_mean = b.add_input("bn3_mean", &[C2]);
    let bn3_var = b.add_input("bn3_var", &[C2]);
    let bn3_weight = b.add_input("bn3_weight", &[C2]);
    let bn3_bias = b.add_input("bn3_bias", &[C2]);
    let bn3_eps = b.add_input("bn3_eps", &[1]);
    let conv3 = b.add_conv2d(silu2, w3, Some(b3), 1, 1, 1, 1, &[C2, s1, s1]);
    let bn3 = b.add_batch_norm(
        conv3,
        bn3_mean,
        bn3_var,
        bn3_weight,
        bn3_bias,
        bn3_eps,
        &[C2, s1, s1],
    );
    let sig3 = b.add_sigmoid(bn3, &[C2, s1, s1]);
    let silu3 = b.add_binary_mul(bn3, sig3, &[C2, s1, s1]);

    b.build(silu3).expect("valid YOLO ConvBnAct stack kernel")
}

fn yolo_conv_bn_act_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];

    // Block 1: C1 -> C1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C1, C1, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C1]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C1]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C1]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C1]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C1]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Block 2: C1 -> C2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2, C1, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Block 3: C2 -> C2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2, C2, 3, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[C2]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    bindings
}

#[test]
fn test_backbone_yolo_conv_bn_act_stack_ibp() {
    let def = build_yolo_conv_bn_act_stack();
    let bindings = yolo_conv_bn_act_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through YOLO ConvBnAct");

    let s1 = SPATIAL / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[C2, s1, s1],
        "YOLO ConvBnAct stack output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("YOLO ConvBnAct stack IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. C2f block: cross-stage partial with bottleneck (IBP + CROWN)
// ===========================================================================

/// Build a simplified C2f (Cross-Stage Partial with 2 convolutions) block.
///
/// Input: [C2, SPATIAL/2, SPATIAL/2].
/// Split into two halves via 1x1 convs, one half through bottleneck, concat.
/// Architecture: 1x1 conv split -> bottleneck(half) -> concat -> 1x1 conv merge.
/// Output: [C2, SPATIAL/2, SPATIAL/2].
fn build_yolo_c2f_block() -> TensorKernelDef {
    let s = SPATIAL / 2;
    let half_c = C2 / 2;
    let shape = [C2, s, s];
    let half_shape = [half_c, s, s];
    let mut b = TensorBlockBuilder::new("backbone_yolo_c2f");

    let input = b.add_input("features", &shape);

    // Split via 1x1 conv: full -> half channels (path A: direct, path B: bottleneck)
    let split_w_a = b.add_input("split_w_a", &[half_c, C2, 1, 1]);
    let split_w_b = b.add_input("split_w_b", &[half_c, C2, 1, 1]);

    let path_a = b.add_conv2d(input, split_w_a, None, 1, 1, 0, 0, &half_shape);

    // Path B: bottleneck (Conv3x3 -> ReLU -> Conv3x3 -> ReLU)
    let path_b_in = b.add_conv2d(input, split_w_b, None, 1, 1, 0, 0, &half_shape);
    let bn_w = b.add_input("bn_conv_w", &[half_c, half_c, 3, 3]);
    let bn_b = b.add_input("bn_conv_b", &[half_c]);
    let bn_out = b.add_conv2d(path_b_in, bn_w, Some(bn_b), 1, 1, 1, 1, &half_shape);
    let bn_out = b.add_relu(bn_out, &half_shape);

    // Concat path A + bottleneck path B
    let concat = b.add_concat(&[path_a, bn_out], 0, &shape);

    // Merge via 1x1 conv
    let merge_w = b.add_input("merge_w", &[C2, C2, 1, 1]);
    let out = b.add_conv2d(concat, merge_w, None, 1, 1, 0, 0, &shape);

    b.build(out).expect("valid C2f block kernel")
}

fn yolo_c2f_bindings() -> Vec<TensorParamBinding> {
    let half_c = C2 / 2;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[half_c, C2, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[half_c, C2, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[half_c, half_c, 3, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[half_c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2, C2, 1, 1]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_backbone_yolo_c2f_ibp() {
    let def = build_yolo_c2f_block();
    let bindings = yolo_c2f_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let s = SPATIAL / 2;
    let input = uniform_bounds(&[C2, s, s], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through C2f block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C2, s, s],
        "C2f output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("YOLO C2f block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_backbone_yolo_c2f_crown() {
    let def = build_yolo_c2f_block();
    let bindings = yolo_c2f_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let s = SPATIAL / 2;
    let input = uniform_bounds(&[C2, s, s], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[C2, s, s]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("YOLO C2f CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. SPPF integration: multi-scale pooling at backbone output (IBP)
// ===========================================================================

/// Build SPPF-style block at backbone output: 3 cascaded MaxPool2d + concat.
///
/// Input: [C2, SPATIAL/2, SPATIAL/2].
/// Output: [C2 * 4, SPATIAL/2, SPATIAL/2] (4x channel expansion from concat).
fn build_sppf_backbone_output() -> TensorKernelDef {
    let s = SPATIAL / 2;
    let shape = [C2, s, s];
    let out_shape = [C2 * 4, s, s];
    let mut b = TensorBlockBuilder::new("backbone_sppf_output");

    let input = b.add_input("features", &shape);

    // 3 cascaded MaxPool2d(k=5, s=1, p=2) — preserves spatial dims
    let pool1 = b.add_max_pool_2d(input, 5, 5, 1, 1, 2, 2, &shape);
    let pool2 = b.add_max_pool_2d(pool1, 5, 5, 1, 1, 2, 2, &shape);
    let pool3 = b.add_max_pool_2d(pool2, 5, 5, 1, 1, 2, 2, &shape);

    // Concat: [input, pool1, pool2, pool3] along channel axis
    let out = b.add_concat(&[input, pool1, pool2, pool3], 0, &out_shape);

    b.build(out).expect("valid SPPF backbone output kernel")
}

fn sppf_backbone_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_backbone_sppf_integration_ibp() {
    let def = build_sppf_backbone_output();
    let bindings = sppf_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let s = SPATIAL / 2;
    let input = uniform_bounds(&[C2, s, s], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SPPF");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C2 * 4, s, s],
        "SPPF output shape: 4x channels from concat"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SPPF backbone output IBP: bounds=[{lo_min}, {hi_max}]");
    // MaxPool chain cannot expand bounds beyond input range
    assert!(lo_min >= -2.0 - 1e-6, "SPPF lower >= -2.0, got {lo_min}");
    assert!(hi_max <= 2.0 + 1e-6, "SPPF upper <= 2.0, got {hi_max}");
}

// ===========================================================================
// 8. Backbone -> neck connection: feature pyramid (IBP)
// ===========================================================================

/// Build a backbone -> FPN neck pattern: two backbone features + lateral projection.
///
/// Feature P3: [C2, SPATIAL/2, SPATIAL/2] (finer)
/// Feature P4: [C3, SPATIAL/4, SPATIAL/4] (coarser)
/// Neck: project P4 to C2 channels via 1x1 conv, verify bounds.
fn build_backbone_neck_fpn() -> TensorKernelDef {
    let _s_fine = SPATIAL / 2;
    let s_coarse = SPATIAL / 4;
    let mut b = TensorBlockBuilder::new("backbone_neck_fpn");

    // Backbone feature: coarse level (input to neck)
    let p4 = b.add_input("p4_features", &[C3, s_coarse, s_coarse]);

    // Lateral 1x1 conv: C3 -> C2 (align channels for FPN add)
    let lat_w = b.add_input("lateral_w", &[C2, C3, 1, 1]);
    let lat_b = b.add_input("lateral_b", &[C2]);
    let lateral = b.add_conv2d(
        p4,
        lat_w,
        Some(lat_b),
        1,
        1,
        0,
        0,
        &[C2, s_coarse, s_coarse],
    );

    // ReLU activation
    let out = b.add_relu(lateral, &[C2, s_coarse, s_coarse]);

    b.build(out).expect("valid backbone-neck FPN kernel")
}

fn backbone_neck_fpn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2, C3, 1, 1]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2]), 0.0f32)),
    ]
}

#[test]
fn test_backbone_neck_fpn_connection_ibp() {
    let def = build_backbone_neck_fpn();
    let bindings = backbone_neck_fpn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let s_coarse = SPATIAL / 4;
    let input = uniform_bounds(&[C3, s_coarse, s_coarse], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through backbone-neck FPN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C2, s_coarse, s_coarse],
        "FPN lateral projects C3 -> C2 channels"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Backbone -> neck FPN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU output: lower >= 0
    assert!(lo_min >= -1e-6, "FPN+ReLU output lower >= 0, got {lo_min}");
}

// ===========================================================================
// 9. ViT: Patch embed -> encoder block composition (IBP)
// ===========================================================================

/// Build a ViT patch embedding + single encoder block.
///
/// Patch embed: Linear(patch_dim -> HIDDEN_DIM), treated as flattened patches.
/// Encoder block: LayerNorm -> Linear (simulating attention) -> residual.
/// Input: [SEQ_LEN, HIDDEN_DIM] (Variable, pre-embedded patches).
/// Output: [SEQ_LEN, HIDDEN_DIM].
fn build_vit_patch_embed_encoder() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("backbone_vit_patch_encoder");

    let input = b.add_input("patches", &shape);

    // Encoder block: LN -> Linear -> residual
    let eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let attn_w = b.add_input("attn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ffn_w = b.add_input("ffn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);
    // Simulated attention: Linear
    let attn_out = b.add_linear(normed, attn_w, None, &shape);
    // Residual
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // FFN sublayer: Linear
    let ffn_out = b.add_linear(residual1, ffn_w, None, &shape);
    let out = b.add_binary_add(residual1, ffn_out, &shape);

    b.build(out)
        .expect("valid ViT patch-embed + encoder kernel")
}

fn vit_patch_embed_encoder_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

#[test]
fn test_backbone_vit_patch_embed_encoder_ibp() {
    let def = build_vit_patch_embed_encoder();
    let bindings = vit_patch_embed_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ViT encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "ViT encoder preserves sequence shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT patch embed + encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. 2-block ViT encoder stack (IBP + CROWN)
// ===========================================================================

/// Build a 2-block ViT encoder stack with LayerNorm + Linear + residual.
///
/// Each block: LN -> Linear -> residual -> LN -> Linear -> residual.
/// Input: [SEQ_LEN, HIDDEN_DIM].
/// Output: [SEQ_LEN, HIDDEN_DIM].
fn build_vit_2block_encoder() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("backbone_vit_2block_encoder");

    let input = b.add_input("patches", &shape);

    // Block 1: LN -> attn -> residual -> LN -> FFN -> residual
    let eps1 = b.add_input("b1_ln1_eps", &[1]);
    let ln1_w = b.add_input("b1_ln1_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("b1_ln1_b", &[HIDDEN_DIM]);
    let attn1_w = b.add_input("b1_attn_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let eps1b = b.add_input("b1_ln2_eps", &[1]);
    let ln1b_w = b.add_input("b1_ln2_w", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("b1_ln2_b", &[HIDDEN_DIM]);
    let ffn1_w = b.add_input("b1_ffn_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let normed1 = b.add_layer_norm(input, eps1, 1, ln1_w, ln1_b, &shape);
    let attn1_out = b.add_linear(normed1, attn1_w, None, &shape);
    let res1 = b.add_binary_add(input, attn1_out, &shape);
    let normed1b = b.add_layer_norm(res1, eps1b, 1, ln1b_w, ln1b_b, &shape);
    let ffn1_out = b.add_linear(normed1b, ffn1_w, None, &shape);
    let block1 = b.add_binary_add(res1, ffn1_out, &shape);

    // Block 2: same structure
    let eps2 = b.add_input("b2_ln1_eps", &[1]);
    let ln2_w = b.add_input("b2_ln1_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("b2_ln1_b", &[HIDDEN_DIM]);
    let attn2_w = b.add_input("b2_attn_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let eps2b = b.add_input("b2_ln2_eps", &[1]);
    let ln2b_w = b.add_input("b2_ln2_w", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("b2_ln2_b", &[HIDDEN_DIM]);
    let ffn2_w = b.add_input("b2_ffn_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let normed2 = b.add_layer_norm(block1, eps2, 1, ln2_w, ln2_b, &shape);
    let attn2_out = b.add_linear(normed2, attn2_w, None, &shape);
    let res2 = b.add_binary_add(block1, attn2_out, &shape);
    let normed2b = b.add_layer_norm(res2, eps2b, 1, ln2b_w, ln2b_b, &shape);
    let ffn2_out = b.add_linear(normed2b, ffn2_w, None, &shape);
    let block2 = b.add_binary_add(res2, ffn2_out, &shape);

    b.build(block2).expect("valid 2-block ViT encoder kernel")
}

fn vit_2block_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable];

    // Block 1: ln1_eps, ln1_w, ln1_b, attn_w, ln2_eps, ln2_w, ln2_b, ffn_w
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    }

    bindings
}

#[test]
fn test_backbone_vit_2block_encoder_ibp() {
    let def = build_vit_2block_encoder();
    let bindings = vit_2block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-block ViT");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT 2-block encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_backbone_vit_2block_encoder_crown() {
    let def = build_vit_2block_encoder();
    let bindings = vit_2block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT 2-block CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Window attention ViT: local attention with partition (IBP)
// ===========================================================================

/// Build a window-attention ViT block (Qwen3-VL pattern).
///
/// Simulates local attention by applying MHA to a fixed-size window.
/// Uses `add_multi_head_attention(input, q_w, k_w, v_w, out_w, ...)` which
/// handles Q/K/V projection internally.
/// Input: [SEQ_LEN, HIDDEN_DIM].
/// Output: [SEQ_LEN, HIDDEN_DIM].
fn build_vit_window_attention() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("backbone_vit_window_attn");

    let input = b.add_input("patches", &shape);

    // Pre-norm
    let eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Window attention: weight matrices for Q, K, V, output projections
    let q_w = b.add_input("q_proj", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_proj", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid window MHA");

    // Residual
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid window attention ViT kernel")
}

fn vit_window_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_backbone_vit_window_attention_ibp() {
    let def = build_vit_window_attention();
    let bindings = vit_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through window attention ViT");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT window attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. Deep stack fusion: multi-level feature combination (IBP)
// ===========================================================================

/// Build a multi-level feature fusion block.
///
/// Fuses features from two spatial scales via 1x1 projection + element-wise add.
/// Coarse: [C3, 2, 2] projected to [C2, 2, 2].
/// Fine: [C2, 2, 2] (already C2 channels).
/// Output: [C2, 2, 2] (fused features).
fn build_deep_stack_fusion() -> TensorKernelDef {
    let s = SPATIAL / 4; // 2
    let mut b = TensorBlockBuilder::new("backbone_deep_stack_fusion");

    // We take two feature maps as separate inputs (Variable for the primary,
    // constant for the secondary to keep the graph single-variable).
    let fine = b.add_input("fine_features", &[C2, s, s]);

    // Coarse features projected to same channels
    let coarse_proj_w = b.add_input("coarse_proj_w", &[C2, C2, 1, 1]);
    let coarse_features = b.add_input("coarse_features", &[C2, s, s]);
    let coarse_projected = b.add_conv2d(
        coarse_features,
        coarse_proj_w,
        None,
        1,
        1,
        0,
        0,
        &[C2, s, s],
    );

    // Fusion: element-wise add
    let fused = b.add_binary_add(fine, coarse_projected, &[C2, s, s]);
    let out = b.add_relu(fused, &[C2, s, s]);

    b.build(out).expect("valid deep stack fusion kernel")
}

fn deep_stack_fusion_bindings() -> Vec<TensorParamBinding> {
    let s = SPATIAL / 4;
    vec![
        TensorParamBinding::Variable, // fine features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2, C2, 1, 1]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2, s, s]), 0.5f32)),
    ]
}

#[test]
fn test_backbone_deep_stack_fusion_ibp() {
    let def = build_deep_stack_fusion();
    let bindings = deep_stack_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let s = SPATIAL / 4;
    let input = uniform_bounds(&[C2, s, s], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through deep stack fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[C2, s, s],
        "Fused feature shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Deep stack fusion IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU output
    assert!(
        lo_min >= -1e-6,
        "fused+ReLU output lower >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 13. Backbone output dimension comparison across models (IBP)
// ===========================================================================

/// Compare backbone output bounds across ResNet and ViT architectures.
///
/// Both produce HIDDEN_DIM-dimensional features; verify bounds are valid
/// for both and that they produce finite, well-ordered bounds.
#[test]
fn test_backbone_output_dimension_comparison() {
    // ResNet-style: Conv2d path
    let s = SPATIAL / 2;
    {
        let mut b = TensorBlockBuilder::new("backbone_dim_cmp_resnet");
        let input = b.add_input("features", &[C1, SPATIAL, SPATIAL]);
        let w = b.add_input("conv_w", &[C2, C1, 3, 3]);
        let bias = b.add_input("conv_b", &[C2]);
        let conv = b.add_conv2d(input, w, Some(bias), 2, 2, 1, 1, &[C2, s, s]);
        let out = b.add_relu(conv, &[C2, s, s]);
        let def = b.build(out).expect("valid ResNet dim cmp kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[C2, C1, 3, 3]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[C2]), 0.0f32)),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input_bounds = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);
        let output = graph.propagate_ibp(&input_bounds).expect("IBP ResNet path");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("ResNet backbone output IBP: bounds=[{lo_min}, {hi_max}]");
    }

    // ViT-style: Linear path
    {
        let shape = [SEQ_LEN, HIDDEN_DIM];
        let mut b = TensorBlockBuilder::new("backbone_dim_cmp_vit");
        let input = b.add_input("patches", &shape);
        let w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let out = b.add_linear(input, w, None, &shape);
        let def = b.build(out).expect("valid ViT dim cmp kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
                WEIGHT_MAG,
            )),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input_bounds = uniform_bounds(&shape, 1.0);
        let output = graph.propagate_ibp(&input_bounds).expect("IBP ViT path");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("ViT backbone output IBP: bounds=[{lo_min}, {hi_max}]");
    }
}

// ===========================================================================
// 14. Backbone monotone tightening: smaller eps -> tighter features (IBP)
// ===========================================================================

/// Verify that smaller input perturbation produces tighter backbone output bounds.
///
/// This is a fundamental property of interval arithmetic: narrower input
/// intervals produce narrower output intervals through any monotone pipeline.
#[test]
fn test_backbone_monotone_tightening() {
    // Use the ResNet basic block as representative backbone
    let def = build_resnet_basic_block();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: [-1.0, 1.0]
    let wide_input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);

    // Narrow input: [-0.1, 0.1]
    let narrow_input = uniform_bounds(&[C1, SPATIAL, SPATIAL], 0.1);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    assert_bounds_valid(&narrow_output);

    let wide_width = total_bound_width(&wide_output);
    let narrow_width = total_bound_width(&narrow_output);

    eprintln!(
        "Backbone monotone tightening: wide_width={wide_width:.4}, narrow_width={narrow_width:.4}"
    );

    assert!(
        narrow_width <= wide_width + 1e-6,
        "narrower input must produce tighter output bounds: \
         narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 15. Backbone -> head projection composition (IBP)
// ===========================================================================

/// Build a backbone -> classification head pipeline.
///
/// Backbone features: [C2, SPATIAL/2, SPATIAL/2] -> AvgPool2d (global) ->
/// reshape [C2] -> Linear(C2, NUM_CLASSES) -> Sigmoid.
/// Output: [NUM_CLASSES] with bounds in (0, 1).
fn build_backbone_head_projection() -> TensorKernelDef {
    let s = SPATIAL / 2;
    let num_classes = 8usize;
    let mut b = TensorBlockBuilder::new("backbone_head_projection");

    let input = b.add_input("features", &[C2, s, s]);

    // Global average pooling: [C2, s, s] -> [C2, 1, 1]
    let pooled = b.add_avg_pool_2d(input, s, s, s, s, 0, 0, &[C2, 1, 1]);

    // Reshape: [C2, 1, 1] -> [C2]
    let flat = b.add_reshape(pooled, &[C2]);

    // Classification head: Linear(C2 -> NUM_CLASSES) -> Sigmoid
    let head_w = b.add_input("head_weight", &[num_classes, C2]);
    let head_b = b.add_input("head_bias", &[num_classes]);
    let logits = b.add_linear(flat, head_w, Some(head_b), &[num_classes]);
    let out = b.add_sigmoid(logits, &[num_classes]);

    b.build(out).expect("valid backbone-head projection kernel")
}

fn backbone_head_projection_bindings() -> Vec<TensorParamBinding> {
    let num_classes = 8usize;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[num_classes, C2]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[num_classes]), 0.0f32)),
    ]
}

#[test]
fn test_backbone_head_projection_ibp() {
    let def = build_backbone_head_projection();
    let bindings = backbone_head_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let s = SPATIAL / 2;
    let input = uniform_bounds(&[C2, s, s], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through backbone -> head");

    let num_classes = 8usize;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[num_classes],
        "Head output shape: [NUM_CLASSES]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Backbone -> head projection IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output: bounds must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid output upper <= 1, got {hi_max}"
    );
}
