// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Residual and skip connection pattern NY composition.
//!
//! Verifies IBP and CROWN bounds propagation through residual and skip
//! connection patterns found across dpdf document understanding models:
//!
//! 1. **Pre-norm residual (IBP)**: x + LayerNorm(sublayer(x)).
//!    Standard pre-norm transformer pattern (Qwen3-VL, FireRed-OCR, GLM-OCR).
//!
//! 2. **Post-norm residual (IBP)**: LayerNorm(x + sublayer(x)).
//!    Classic post-norm pattern (Table Transformer DETR encoder/decoder).
//!
//! 3. **RMSNorm residual (IBP + CROWN)**: x + RMSNorm(sublayer(x)).
//!    Granite/GLM/Qwen3 decoder pattern with lighter normalization.
//!
//! 4. **Dense residual / DenseNet-style concatenation (IBP)**:
//!    Concat([x, sublayer(x)]) along channel axis. Used in feature pyramid
//!    and SPPF-like multi-scale fusion.
//!
//! 5. **ResNet basic block (IBP + CROWN)**: Conv-BN-ReLU-Conv-BN + skip -> ReLU.
//!    Core building block in Table Transformer and DocLayout-YOLO backbones.
//!
//! 6. **ResNet bottleneck (IBP)**: 1x1-3x3-1x1 Conv + skip.
//!    Deeper ResNet-50/101 pattern for efficient channel reduction.
//!
//! 7. **FPN lateral connection (IBP)**: 1x1 conv projection + element-wise add.
//!    Feature Pyramid Network pattern for multi-scale feature fusion.
//!
//! 8. **Transformer residual accumulation through depth (IBP)**:
//!    3-layer stacked residual blocks. Tests bound stability through depth.
//!
//! 9. **Residual monotone tightening (IBP)**: Smaller input epsilon produces
//!    tighter output bounds through the residual connection.
//!
//! 10. **Skip connection preserves bound width ordering (IBP)**:
//!     Bound width at output >= bound width at skip (residual only adds).
//!
//! 11. **Stochastic depth residual / scale factor (IBP)**:
//!     x + alpha * sublayer(x), alpha in (0, 1). Drop-path training pattern.
//!
//! 12. **Cross-attention residual / DETR decoder (IBP + CROWN)**:
//!     q + cross_attn(q, kv). Encoder-decoder attention with residual.
//!
//! 13. **Multi-scale residual fusion (IBP)**: Add features from two spatial
//!     scales after 1x1 projection alignment.
//!
//! 14. **Residual gradient stability (CROWN)**: CROWN linearization through
//!     deep (4-layer) residual stack verifies bounded gradient flow.
//!
//! 15. **Pre-norm vs post-norm bound width comparison (IBP)**:
//!     Compare output bound widths from pre-norm and post-norm residual
//!     patterns on identical inputs and sublayers.
//!
//! Architecture references:
//! - ResNet (He et al. 2016): Residual learning for image recognition
//! - DenseNet (Huang et al. 2017): Densely connected convolutional networks
//! - Pre-LN Transformer (Xiong et al. 2020): On layer normalization in the transformer
//! - Stochastic Depth (Huang et al. 2016): Deep networks with stochastic depth
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - FPN (Lin et al. 2017): Feature Pyramid Networks
//!
//! Dimensions (small for fast verification):
//! - HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=4, CHANNELS=16, SPATIAL=8
//!
//! Part of #3981: Residual and skip connection compose tests across model architectures.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension for transformer-style tests.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Sequence length for 2D [SEQ_LEN, HIDDEN_DIM] inputs.
const SEQ_LEN: usize = 4;
/// Number of channels for CNN-style tests.
const CHANNELS: usize = 16;
/// Spatial dimension for [CHANNELS, SPATIAL, SPATIAL] inputs.
const SPATIAL: usize = 8;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of attention heads.
const NUM_HEADS: usize = 4;

// ===========================================================================
// 1. Pre-norm residual (IBP): x + LayerNorm(Linear(x))
// ===========================================================================

/// Build a pre-norm residual block.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: out = x + Linear(LayerNorm(x))
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_pre_norm_residual_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_pre_norm_residual");

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ffn_weight = b.add_input("ffn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Pre-norm: LayerNorm(x)
    let normed = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &shape);
    // Sublayer: Linear
    let sublayer_out = b.add_linear(normed, ffn_weight, None, &shape);
    // Residual: x + sublayer(LayerNorm(x))
    let out = b.add_binary_add(input, sublayer_out, &shape);

    b.build(out).expect("valid pre-norm residual kernel")
}

/// Bindings for pre-norm residual.
fn pre_norm_residual_bindings() -> Vec<TensorParamBinding> {
    let ln_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let ffn_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                   // hidden
        TensorParamBinding::ConstantScalar(1e-5),       // eps
        TensorParamBinding::ConstantTensor(ln_weight),  // ln_weight
        TensorParamBinding::ConstantTensor(ln_bias),    // ln_bias
        TensorParamBinding::ConstantTensor(ffn_weight), // ffn_weight
    ]
}

/// Pre-norm residual IBP: x + Linear(LayerNorm(x)) bounds propagate finitely.
#[test]
fn test_dpdf_pre_norm_residual_ibp() {
    let def = build_pre_norm_residual_kernel();
    let bindings = pre_norm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pre-norm residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "pre-norm residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf pre-norm residual IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Post-norm residual (IBP): LayerNorm(x + Linear(x))
// ===========================================================================

/// Build a post-norm residual block.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: out = LayerNorm(x + Linear(x))
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_post_norm_residual_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_post_norm_residual");

    let input = b.add_input("hidden", &shape);
    let ffn_weight = b.add_input("ffn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_bias = b.add_input("ln_bias", &[HIDDEN_DIM]);

    // Sublayer: Linear(x)
    let sublayer_out = b.add_linear(input, ffn_weight, None, &shape);
    // Residual: x + Linear(x)
    let residual = b.add_binary_add(input, sublayer_out, &shape);
    // Post-norm: LayerNorm(x + sublayer(x))
    let out = b.add_layer_norm(residual, eps, 1, ln_weight, ln_bias, &shape);

    b.build(out).expect("valid post-norm residual kernel")
}

/// Bindings for post-norm residual.
fn post_norm_residual_bindings() -> Vec<TensorParamBinding> {
    let ffn_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ln_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                   // hidden
        TensorParamBinding::ConstantTensor(ffn_weight), // ffn_weight
        TensorParamBinding::ConstantScalar(1e-5),       // eps
        TensorParamBinding::ConstantTensor(ln_weight),  // ln_weight
        TensorParamBinding::ConstantTensor(ln_bias),    // ln_bias
    ]
}

/// Post-norm residual IBP: LayerNorm(x + Linear(x)) bounds propagate finitely.
#[test]
fn test_dpdf_post_norm_residual_ibp() {
    let def = build_post_norm_residual_kernel();
    let bindings = post_norm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "post-norm residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf post-norm residual IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. RMSNorm residual (IBP): x + Linear(RMSNorm(x))
// ===========================================================================

/// Build an RMSNorm-based pre-norm residual block (Granite/GLM pattern).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: out = x + Linear(RMSNorm(x))
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_rmsnorm_residual_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_rmsnorm_residual");

    let input = b.add_input("hidden", &shape);
    let eps = b.add_input("eps", &[1]);
    let rms_weight = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let ffn_weight = b.add_input("ffn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Pre-norm: RMSNorm(x)
    let normed = b.add_rms_norm(input, eps, 1, rms_weight, &shape);
    // Sublayer: Linear
    let sublayer_out = b.add_linear(normed, ffn_weight, None, &shape);
    // Residual: x + sublayer
    let out = b.add_binary_add(input, sublayer_out, &shape);

    b.build(out).expect("valid RMSNorm residual kernel")
}

/// Bindings for RMSNorm residual.
fn rmsnorm_residual_bindings() -> Vec<TensorParamBinding> {
    let rms_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ffn_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                   // hidden
        TensorParamBinding::ConstantScalar(1e-5),       // eps
        TensorParamBinding::ConstantTensor(rms_weight), // rms_weight
        TensorParamBinding::ConstantTensor(ffn_weight), // ffn_weight
    ]
}

/// RMSNorm residual IBP bounds propagate finitely.
#[test]
fn test_dpdf_rmsnorm_residual_ibp() {
    let def = build_rmsnorm_residual_kernel();
    let bindings = rmsnorm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm residual IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3b. RMSNorm residual CROWN
// ===========================================================================

/// CROWN bounds through RMSNorm residual.
#[test]
fn test_dpdf_rmsnorm_residual_crown() {
    let def = build_rmsnorm_residual_kernel();
    let bindings = rmsnorm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf RMSNorm residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. Dense residual / DenseNet-style concatenation (IBP)
// ===========================================================================

/// Build a DenseNet-style dense connection block.
///
/// Input: `[SPATIAL, CHANNELS]` (Variable, batch-major so nn.Linear contracts
/// the channel dim against weight [out, in] = [CHANNELS, CHANNELS]).
/// Architecture: out = Concat([x, ReLU(Linear(x))], axis=1)
/// Output: `[SPATIAL, CHANNELS * 2]`.
fn build_dense_residual_kernel() -> TensorKernelDef {
    let in_shape = [SPATIAL, CHANNELS];
    let out_shape = [SPATIAL, CHANNELS * 2];
    let mut b = TensorBlockBuilder::new("dpdf_dense_residual");

    let input = b.add_input("features", &in_shape);
    let weight = b.add_input("weight", &[CHANNELS, CHANNELS]);

    // Sublayer: Linear -> ReLU (produces same shape as input)
    let sublayer = b.add_linear(input, weight, None, &in_shape);
    let sublayer = b.add_relu(sublayer, &in_shape);

    // Dense connection: Concat([x, sublayer(x)]) along channel axis (axis 1)
    let out = b.add_concat(&[input, sublayer], 1, &out_shape);

    b.build(out).expect("valid dense residual kernel")
}

/// Bindings for dense residual.
fn dense_residual_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[CHANNELS, CHANNELS]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // features
        TensorParamBinding::ConstantTensor(weight), // weight
    ]
}

/// DenseNet-style concat residual IBP bounds propagate finitely.
#[test]
fn test_dpdf_dense_residual_concat_ibp() {
    let def = build_dense_residual_kernel();
    let bindings = dense_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPATIAL, CHANNELS], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dense residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SPATIAL, CHANNELS * 2],
        "dense residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf dense residual (concat) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. ResNet basic block (IBP + CROWN)
// ===========================================================================

/// Build a ResNet basic block: Conv-BN-ReLU-Conv-BN + skip -> ReLU.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable).
/// Output: `[CHANNELS, SPATIAL, SPATIAL]`.
fn build_resnet_basic_block_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("dpdf_resnet_basic_block");

    let input = b.add_input("features", &feat_shape);

    // First conv path
    let conv1_w = b.add_input("conv1_weight", &[c, c, 3, 3]);
    let conv1_b = b.add_input("conv1_bias", &[c]);
    let bn1_mean = b.add_input("bn1_running_mean", &[c]);
    let bn1_var = b.add_input("bn1_running_var", &[c]);
    let bn1_weight = b.add_input("bn1_weight", &[c]);
    let bn1_bias = b.add_input("bn1_bias", &[c]);
    let bn1_eps = b.add_input("bn1_eps", &[1]);

    // Second conv path
    let conv2_w = b.add_input("conv2_weight", &[c, c, 3, 3]);
    let conv2_b = b.add_input("conv2_bias", &[c]);
    let bn2_mean = b.add_input("bn2_running_mean", &[c]);
    let bn2_var = b.add_input("bn2_running_var", &[c]);
    let bn2_weight = b.add_input("bn2_weight", &[c]);
    let bn2_bias = b.add_input("bn2_bias", &[c]);
    let bn2_eps = b.add_input("bn2_eps", &[1]);

    // Conv2d(C,C,3,s=1,p=1) -> BN -> ReLU
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

    // Conv2d(C,C,3,s=1,p=1) -> BN
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

    // Residual: bn2 + input
    let residual = b.add_binary_add(bn2_out, input, &feat_shape);
    // Final ReLU
    let out = b.add_relu(residual, &feat_shape);

    b.build(out).expect("valid ResNet basic block kernel")
}

/// Bindings for ResNet basic block.
fn resnet_basic_block_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                          // features
        TensorParamBinding::ConstantTensor(conv_w.clone()),    // conv1_weight
        TensorParamBinding::ConstantTensor(conv_b.clone()),    // conv1_bias
        TensorParamBinding::ConstantTensor(bn_mean.clone()),   // bn1_running_mean
        TensorParamBinding::ConstantTensor(bn_var.clone()),    // bn1_running_var
        TensorParamBinding::ConstantTensor(bn_weight.clone()), // bn1_weight
        TensorParamBinding::ConstantTensor(bn_bias.clone()),   // bn1_bias
        TensorParamBinding::ConstantScalar(1e-5),              // bn1_eps
        TensorParamBinding::ConstantTensor(conv_w),            // conv2_weight
        TensorParamBinding::ConstantTensor(conv_b),            // conv2_bias
        TensorParamBinding::ConstantTensor(bn_mean),           // bn2_running_mean
        TensorParamBinding::ConstantTensor(bn_var),            // bn2_running_var
        TensorParamBinding::ConstantTensor(bn_weight),         // bn2_weight
        TensorParamBinding::ConstantTensor(bn_bias),           // bn2_bias
        TensorParamBinding::ConstantScalar(1e-5),              // bn2_eps
    ]
}

/// ResNet basic block IBP: Conv-BN-ReLU-Conv-BN + skip -> ReLU.
#[test]
fn test_dpdf_resnet_basic_block_residual_ibp() {
    let def = build_resnet_basic_block_kernel();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ResNet basic block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "ResNet basic block output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf ResNet basic block residual IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU clamps lower >= 0
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
}

/// ResNet basic block CROWN.
#[test]
fn test_dpdf_resnet_basic_block_residual_crown() {
    let def = build_resnet_basic_block_kernel();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL]
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "dpdf ResNet basic block residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 6. ResNet bottleneck (IBP): 1x1-3x3-1x1 + skip
// ===========================================================================

/// Build a ResNet bottleneck block.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable).
/// Architecture: 1x1(C -> C/4) -> 3x3(C/4 -> C/4) -> 1x1(C/4 -> C) + skip -> ReLU
/// Output: `[CHANNELS, SPATIAL, SPATIAL]`.
fn build_resnet_bottleneck_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let neck = c / 4; // bottleneck channels
    let s = SPATIAL;
    let feat_shape = [c, s, s];
    let neck_shape = [neck, s, s];
    let mut b = TensorBlockBuilder::new("dpdf_resnet_bottleneck");

    let input = b.add_input("features", &feat_shape);

    // 1x1 reduce
    let w1 = b.add_input("conv1_weight", &[neck, c, 1, 1]);
    let b1 = b.add_input("conv1_bias", &[neck]);
    // 3x3
    let w2 = b.add_input("conv2_weight", &[neck, neck, 3, 3]);
    let b2 = b.add_input("conv2_bias", &[neck]);
    // 1x1 expand
    let w3 = b.add_input("conv3_weight", &[c, neck, 1, 1]);
    let b3 = b.add_input("conv3_bias", &[c]);

    // 1x1 reduce -> ReLU
    let out1 = b.add_conv2d(input, w1, Some(b1), 1, 1, 0, 0, &neck_shape);
    let out1 = b.add_relu(out1, &neck_shape);

    // 3x3 -> ReLU
    let out2 = b.add_conv2d(out1, w2, Some(b2), 1, 1, 1, 1, &neck_shape);
    let out2 = b.add_relu(out2, &neck_shape);

    // 1x1 expand
    let out3 = b.add_conv2d(out2, w3, Some(b3), 1, 1, 0, 0, &feat_shape);

    // Residual + ReLU
    let residual = b.add_binary_add(out3, input, &feat_shape);
    let out = b.add_relu(residual, &feat_shape);

    b.build(out).expect("valid ResNet bottleneck kernel")
}

/// Bindings for ResNet bottleneck.
fn resnet_bottleneck_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let neck = c / 4;

    let w1 = ArrayD::from_elem(IxDyn(&[neck, c, 1, 1]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[neck]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[neck, neck, 3, 3]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[neck]), 0.0f32);
    let w3 = ArrayD::from_elem(IxDyn(&[c, neck, 1, 1]), WEIGHT_MAG);
    let b3 = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,           // features
        TensorParamBinding::ConstantTensor(w1), // conv1_weight
        TensorParamBinding::ConstantTensor(b1), // conv1_bias
        TensorParamBinding::ConstantTensor(w2), // conv2_weight
        TensorParamBinding::ConstantTensor(b2), // conv2_bias
        TensorParamBinding::ConstantTensor(w3), // conv3_weight
        TensorParamBinding::ConstantTensor(b3), // conv3_bias
    ]
}

/// ResNet bottleneck IBP: 1x1-3x3-1x1 + skip -> ReLU.
#[test]
fn test_dpdf_resnet_bottleneck_residual_ibp() {
    let def = build_resnet_bottleneck_kernel();
    let bindings = resnet_bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ResNet bottleneck");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "ResNet bottleneck output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf ResNet bottleneck residual IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 7. FPN lateral connection (IBP): 1x1 conv + add
// ===========================================================================

/// Build an FPN lateral connection block.
///
/// Two inputs of same spatial size but different channels. 1x1 conv aligns
/// channels, then element-wise add fuses features.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable).
/// Output: `[CHANNELS, SPATIAL, SPATIAL]`.
fn build_fpn_lateral_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("dpdf_fpn_lateral");

    let input = b.add_input("backbone_features", &feat_shape);
    let lateral_w = b.add_input("lateral_weight", &[c, c, 1, 1]);
    let lateral_b = b.add_input("lateral_bias", &[c]);

    // 1x1 conv to project backbone features
    let projected = b.add_conv2d(input, lateral_w, Some(lateral_b), 1, 1, 0, 0, &feat_shape);
    // FPN lateral: add projected features to skip
    let out = b.add_binary_add(input, projected, &feat_shape);

    b.build(out).expect("valid FPN lateral kernel")
}

/// Bindings for FPN lateral.
fn fpn_lateral_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let w = ArrayD::from_elem(IxDyn(&[c, c, 1, 1]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // backbone_features
        TensorParamBinding::ConstantTensor(w),    // lateral_weight
        TensorParamBinding::ConstantTensor(bias), // lateral_bias
    ]
}

/// FPN lateral connection IBP: 1x1 conv + add.
#[test]
fn test_dpdf_fpn_lateral_connection_ibp() {
    let def = build_fpn_lateral_kernel();
    let bindings = fpn_lateral_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FPN lateral");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "FPN lateral output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf FPN lateral IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Transformer residual accumulation through depth (IBP)
// ===========================================================================

/// Build a 3-layer stacked residual block (Linear + residual per layer).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: 3x (x = x + Linear(LayerNorm(x)))
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_deep_residual_stack_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_deep_residual_stack");

    let input = b.add_input("hidden", &shape);

    // Shared param pattern: each layer has LN + Linear
    let eps1 = b.add_input("eps1", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_bias", &[HIDDEN_DIM]);
    let ffn1_w = b.add_input("ffn1_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let eps2 = b.add_input("eps2", &[1]);
    let ln2_w = b.add_input("ln2_weight", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_bias", &[HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let eps3 = b.add_input("eps3", &[1]);
    let ln3_w = b.add_input("ln3_weight", &[HIDDEN_DIM]);
    let ln3_b = b.add_input("ln3_bias", &[HIDDEN_DIM]);
    let ffn3_w = b.add_input("ffn3_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Layer 1: x = x + Linear(LN(x))
    let normed1 = b.add_layer_norm(input, eps1, 1, ln1_w, ln1_b, &shape);
    let sub1 = b.add_linear(normed1, ffn1_w, None, &shape);
    let h1 = b.add_binary_add(input, sub1, &shape);

    // Layer 2: x = x + Linear(LN(x))
    let normed2 = b.add_layer_norm(h1, eps2, 1, ln2_w, ln2_b, &shape);
    let sub2 = b.add_linear(normed2, ffn2_w, None, &shape);
    let h2 = b.add_binary_add(h1, sub2, &shape);

    // Layer 3: x = x + Linear(LN(x))
    let normed3 = b.add_layer_norm(h2, eps3, 1, ln3_w, ln3_b, &shape);
    let sub3 = b.add_linear(normed3, ffn3_w, None, &shape);
    let out = b.add_binary_add(h2, sub3, &shape);

    b.build(out).expect("valid deep residual stack kernel")
}

/// Bindings for 3-layer deep residual stack.
fn deep_residual_stack_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let ffn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                      // hidden
        TensorParamBinding::ConstantScalar(1e-5),          // eps1
        TensorParamBinding::ConstantTensor(ln_w.clone()),  // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),  // ln1_bias
        TensorParamBinding::ConstantTensor(ffn_w.clone()), // ffn1_weight
        TensorParamBinding::ConstantScalar(1e-5),          // eps2
        TensorParamBinding::ConstantTensor(ln_w.clone()),  // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),  // ln2_bias
        TensorParamBinding::ConstantTensor(ffn_w.clone()), // ffn2_weight
        TensorParamBinding::ConstantScalar(1e-5),          // eps3
        TensorParamBinding::ConstantTensor(ln_w),          // ln3_weight
        TensorParamBinding::ConstantTensor(ln_b),          // ln3_bias
        TensorParamBinding::ConstantTensor(ffn_w),         // ffn3_weight
    ]
}

/// Deep residual stack (3 layers) IBP: bounds remain finite through depth.
#[test]
fn test_dpdf_deep_residual_stack_ibp() {
    let def = build_deep_residual_stack_kernel();
    let bindings = deep_residual_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through deep residual stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "deep residual stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf deep residual stack (3-layer) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Residual monotone tightening (IBP)
// ===========================================================================

/// Smaller input epsilon produces tighter output bounds through residual.
#[test]
fn test_dpdf_residual_monotone_tightening_ibp() {
    let def = build_pre_norm_residual_kernel();
    let bindings = pre_norm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let input_tight = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight");

    let (wide_lo, wide_hi) = bounds_min_max(&output_wide);
    let (tight_lo, tight_hi) = bounds_min_max(&output_tight);
    let wide_width = wide_hi - wide_lo;
    let tight_width = tight_hi - tight_lo;

    eprintln!(
        "dpdf residual monotone tightening: wide_width={wide_width}, tight_width={tight_width}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tighter input must produce tighter output: tight={tight_width} > wide={wide_width}"
    );
}

// ===========================================================================
// 10. Skip connection preserves bound width ordering (IBP)
// ===========================================================================

/// Residual (x + f(x)) output bound width >= input bound width.
///
/// Adding sublayer output to skip can only widen (or preserve) the interval.
#[test]
fn test_dpdf_skip_preserves_bound_width_ibp() {
    let def = build_pre_norm_residual_kernel();
    let bindings = pre_norm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pre-norm residual");

    // Input width is 2.0 (range [-1, 1])
    let input_width = 2.0f32;
    let (lo_min, hi_max) = bounds_min_max(&output);
    let output_width = hi_max - lo_min;

    eprintln!("dpdf skip bound width: input_width={input_width}, output_width={output_width}");
    // Residual addition (x + f(x)) widens or preserves IBP interval
    assert!(
        output_width >= input_width - 1e-4,
        "residual output width ({output_width}) should be >= input width ({input_width})"
    );
}

// ===========================================================================
// 11. Stochastic depth residual / scale factor (IBP)
// ===========================================================================

/// Build a stochastic depth residual: x + alpha * Linear(x).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: out = x + scale * Linear(x), scale < 1.
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_stochastic_depth_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_stochastic_depth_residual");

    let input = b.add_input("hidden", &shape);
    let ffn_weight = b.add_input("ffn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let scale = b.add_input("drop_scale", &[1]);

    // Sublayer: Linear(x)
    let sublayer = b.add_linear(input, ffn_weight, None, &shape);
    // Scale by drop-path survival probability
    let scaled = b.add_layer_scale(sublayer, scale, &shape);
    // Residual: x + alpha * sublayer
    let out = b.add_binary_add(input, scaled, &shape);

    b.build(out)
        .expect("valid stochastic depth residual kernel")
}

/// Bindings for stochastic depth residual with scale=0.8.
fn stochastic_depth_bindings() -> Vec<TensorParamBinding> {
    let ffn_weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                   // hidden
        TensorParamBinding::ConstantTensor(ffn_weight), // ffn_weight
        TensorParamBinding::ConstantScalar(0.8),        // drop_scale
    ]
}

/// Stochastic depth residual IBP: x + 0.8 * Linear(x).
#[test]
fn test_dpdf_stochastic_depth_residual_ibp() {
    let def = build_stochastic_depth_kernel();
    let bindings = stochastic_depth_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through stochastic depth residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "stochastic depth output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf stochastic depth residual (alpha=0.8) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. Cross-attention residual / DETR decoder (IBP + CROWN)
// ===========================================================================

/// Build a cross-attention residual block (DETR decoder pattern).
///
/// Q input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, object queries).
/// Architecture: out = q + CrossAttn(q, kv)
/// We approximate with self-attention on the queries (single-input graph)
/// followed by residual.
///
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_cross_attn_residual_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_cross_attn_residual");

    let input = b.add_input("queries", &shape);
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Self-attention (approximation of cross-attention when q=kv)
    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // Residual: queries + attn_output
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid cross-attn residual kernel")
}

/// Bindings for cross-attention residual.
fn cross_attn_residual_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // queries
        TensorParamBinding::ConstantTensor(w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w),         // out_weight
    ]
}

/// Cross-attention residual IBP: q + Attn(q, kv).
#[test]
fn test_dpdf_cross_attn_residual_ibp() {
    let def = build_cross_attn_residual_kernel();
    let bindings = cross_attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attn residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "cross-attn residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf cross-attn residual IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Cross-attention residual CROWN.
#[test]
fn test_dpdf_cross_attn_residual_crown() {
    let def = build_cross_attn_residual_kernel();
    let bindings = cross_attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf cross-attn residual CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Multi-scale residual fusion (IBP)
// ===========================================================================

/// Build a multi-scale residual fusion block.
///
/// Two branches at same spatial scale fused via 1x1 projection + add.
///
/// Input: `[CHANNELS, SPATIAL, SPATIAL]` (Variable).
/// Architecture: out = input + ReLU(Conv1x1(input))
/// Output: `[CHANNELS, SPATIAL, SPATIAL]`.
fn build_multi_scale_fusion_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = SPATIAL;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("dpdf_multi_scale_fusion");

    let input = b.add_input("features", &feat_shape);
    let proj_w = b.add_input("proj_weight", &[c, c, 1, 1]);
    let proj_b = b.add_input("proj_bias", &[c]);

    // Branch: 1x1 conv projection + ReLU
    let projected = b.add_conv2d(input, proj_w, Some(proj_b), 1, 1, 0, 0, &feat_shape);
    let activated = b.add_relu(projected, &feat_shape);

    // Fusion: add
    let out = b.add_binary_add(input, activated, &feat_shape);

    b.build(out).expect("valid multi-scale fusion kernel")
}

/// Bindings for multi-scale fusion.
fn multi_scale_fusion_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let w = ArrayD::from_elem(IxDyn(&[c, c, 1, 1]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // features
        TensorParamBinding::ConstantTensor(w),    // proj_weight
        TensorParamBinding::ConstantTensor(bias), // proj_bias
    ]
}

/// Multi-scale residual fusion IBP.
#[test]
fn test_dpdf_multi_scale_residual_fusion_ibp() {
    let def = build_multi_scale_fusion_kernel();
    let bindings = multi_scale_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, SPATIAL, SPATIAL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale fusion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, SPATIAL, SPATIAL],
        "multi-scale fusion output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dpdf multi-scale residual fusion IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. Residual gradient stability (CROWN) — deep 4-layer stack
// ===========================================================================

/// Build a 4-layer stacked residual block for CROWN gradient stability test.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: 4x (x = x + Linear(LayerNorm(x)))
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_4layer_residual_crown_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("dpdf_4layer_residual_crown");

    let input = b.add_input("hidden", &shape);

    // 4 layers of pre-norm residual
    let mut eps_nodes = Vec::new();
    let mut ln_w_nodes = Vec::new();
    let mut ln_b_nodes = Vec::new();
    let mut ffn_w_nodes = Vec::new();

    for i in 0..4 {
        eps_nodes.push(b.add_input(&format!("eps{i}"), &[1]));
        ln_w_nodes.push(b.add_input(&format!("ln{i}_weight"), &[HIDDEN_DIM]));
        ln_b_nodes.push(b.add_input(&format!("ln{i}_bias"), &[HIDDEN_DIM]));
        ffn_w_nodes.push(b.add_input(&format!("ffn{i}_weight"), &[HIDDEN_DIM, HIDDEN_DIM]));
    }

    let mut h = input;
    for i in 0..4 {
        let normed = b.add_layer_norm(h, eps_nodes[i], 1, ln_w_nodes[i], ln_b_nodes[i], &shape);
        let sub = b.add_linear(normed, ffn_w_nodes[i], None, &shape);
        h = b.add_binary_add(h, sub, &shape);
    }

    b.build(h).expect("valid 4-layer residual CROWN kernel")
}

/// Bindings for 4-layer residual CROWN test.
fn residual_4layer_crown_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let ffn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn_w.clone()));
    }
    bindings
}

/// CROWN through 4-layer deep residual stack verifies bounded gradient flow.
#[test]
fn test_dpdf_residual_gradient_stability_crown() {
    let def = build_4layer_residual_crown_kernel();
    let bindings = residual_4layer_crown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "dpdf 4-layer residual gradient stability CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Pre-norm vs post-norm bound width comparison (IBP)
// ===========================================================================

/// Compare pre-norm and post-norm residual output bound widths.
///
/// Both patterns use the same sublayer (Linear) and input. This test verifies
/// that both produce finite bounds and logs the width difference for analysis.
#[test]
fn test_dpdf_pre_norm_vs_post_norm_bound_width_ibp() {
    // Pre-norm
    let pre_def = build_pre_norm_residual_kernel();
    let pre_bindings = pre_norm_residual_bindings();
    let pre_graph = tensor_kernel_to_graph(&pre_def, &pre_bindings).expect("pre-norm graph");

    // Post-norm
    let post_def = build_post_norm_residual_kernel();
    let post_bindings = post_norm_residual_bindings();
    let post_graph = tensor_kernel_to_graph(&post_def, &post_bindings).expect("post-norm graph");

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let pre_output = pre_graph.propagate_ibp(&input).expect("IBP pre-norm");
    let post_output = post_graph.propagate_ibp(&input).expect("IBP post-norm");

    assert_bounds_valid(&pre_output);
    assert_bounds_valid(&post_output);

    let (pre_lo, pre_hi) = bounds_min_max(&pre_output);
    let (post_lo, post_hi) = bounds_min_max(&post_output);
    let pre_width = pre_hi - pre_lo;
    let post_width = post_hi - post_lo;

    eprintln!("dpdf pre-norm vs post-norm IBP comparison:");
    eprintln!("  pre-norm:  width={pre_width} bounds=[{pre_lo}, {pre_hi}]");
    eprintln!("  post-norm: width={post_width} bounds=[{post_lo}, {post_hi}]");
    // Both must be finite; relative ordering is architecture-dependent
    assert!(pre_width.is_finite(), "pre-norm width must be finite");
    assert!(post_width.is_finite(), "post-norm width must be finite");
}
