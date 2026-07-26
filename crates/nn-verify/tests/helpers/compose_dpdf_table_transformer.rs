// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Table Transformer (DETR) subgraph NY composition.
//!
//! Verifies bounds propagation through Table Transformer sub-blocks used in the
//! dpdf document understanding pipeline for table structure recognition:
//!
//! 1. **ResNet basic block IBP**: Conv2d -> BN -> ReLU -> Conv2d -> BN + skip.
//!    Core ResNet-18/50 building block in the DETR backbone.
//!
//! 2. **ResNet basic block CROWN**: Same with CROWN linearization through
//!    the ReLU non-linearity.
//!
//! 3. **ResNet backbone level IBP**: Conv2d(stride=2) -> 2x BasicBlock.
//!    One stage of the ResNet backbone with spatial downsampling.
//!
//! 4. **Transformer encoder layer CROWN**: Self-attention -> LayerNorm -> FFN
//!    -> LayerNorm. Standard DETR encoder layer.
//!
//! 5. **DETR decoder cross-attention CROWN**: Cross-attention between learned
//!    object queries and encoder memory features.
//!
//! 6. **Classification head sigmoid IBP**: Linear -> sigmoid. Output bounded
//!    in [0, 1] for table cell class probabilities.
//!
//! 7. **Box regression sigmoid IBP**: Linear -> sigmoid for normalized box
//!    coordinates in [0, 1].
//!
//! 8. **Sinusoidal position encoding IBP**: sin/cos positional encoding
//!    bounded in [-1, 1].
//!
//! 9. **DFL regression softmax IBP**: Softmax over reg_max bins for
//!    distribution-based bounding box regression.
//!
//! 10. **Full detection compose IBP**: End-to-end simplified pipeline from
//!     features through classification and box regression heads.
//!
//! 11. **DETR encoder 2-layer stack**: Two stacked encoder layers with
//!     self-attention + FFN. Tests bounds stability through deep composition.
//!
//! 12. **DETR decoder 2-layer stack**: Two stacked decoder layers with
//!     self-attention, cross-attention, and FFN. Full decoder architecture.
//!
//! 13. **ResNet 2-stage backbone**: Two cascaded stride-2 downsampling stages
//!     (8x8 -> 4x4 -> 2x2). Tests deep spatial feature extraction.
//!
//! 14. **Multi-head cross-attention with LayerNorm**: Queries normalized
//!     before cross-attending to encoder memory, with residual connection.
//!
//! 15. **Position encoding + attention composition**: Features + sinusoidal PE
//!     -> LayerNorm -> self-attention -> residual. DETR encoder input pattern.
//!
//! 16. **Full DETR pipeline**: Encoder -> decoder projection -> dual sigmoid
//!     heads. End-to-end from backbone features to detection outputs.
//!
//! 17. **Box regression end-to-end (DFL -> sigmoid)**: DFL softmax over bins
//!     -> weighted sum -> sigmoid normalization to [0, 1].
//!
//! 18. **Transformer FFN with residual (CROWN)**: LayerNorm -> Linear -> ReLU
//!     -> Linear -> residual. Isolated FFN block for CROWN linearization.
//!
//! 19. **Table detection + structure dual-head**: Three sigmoid heads for
//!     detection, structure recognition, and box regression.
//!
//! 20. **Full pipeline: ResNet -> encoder -> decoder -> heads**: Complete
//!     Table Transformer from image features through all stages to output.
//!
//! 21. **ResNet-18 full 4-stage backbone IBP**: 4 sequential stages with stride-2
//!     downsampling (8x8 -> 4x4 -> 2x2 -> 1x1). Full backbone feature extraction.
//!
//! 22. **DETR encoder 4-layer stack IBP**: 4 stacked encoder layers. Tests bounds
//!     stability through deep self-attention + FFN composition.
//!
//! 23. **DETR decoder 4-layer stack IBP**: 4 stacked decoder layers with
//!     self-attention, cross-attention, and FFN. Full decoder depth.
//!
//! 24. **Encoder-decoder composition IBP**: 2 encoder layers -> 2 decoder layers.
//!     Tests cross-attention bounds when encoder memory feeds decoder.
//!
//! 25. **Object query learning IBP**: Learned query embeddings -> linear projection
//!     -> softmax -> matmul with values. Tests attention weight bounds.
//!
//! 26. **Multi-head detection (parallel heads) IBP**: Shared features -> cls head
//!     (sigmoid), box head (sigmoid), structure head (sigmoid). Tests parallel
//!     branches all preserve [0, 1] bounds.
//!
//! 27. **Position encoding propagation IBP**: PE addition -> 2 encoder layers.
//!     Tests sinusoidal PE bounds preservation through multiple attention layers.
//!
//! 28. **Backbone-to-transformer transition IBP**: ResNet features (4D) -> reshape
//!     (flatten spatial) -> linear projection -> LayerNorm. Shape transition path.
//!
//! 29. **ResNet BasicBlock with projection shortcut IBP**: 1x1 conv + BN on skip
//!     path for dimension matching. Tests skip connection with channel change.
//!
//! 30. **Encoder with final LayerNorm IBP**: 2 encoder layers -> final LN.
//!     Tests normalization at encoder output boundary.
//!
//! Architecture references:
//! - DETR (Carion et al. 2020): DEtection TRansformer, end-to-end object detection
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - ResNet (He et al. 2016): Residual networks for feature extraction backbone
//! - Sinusoidal PE (Vaswani et al. 2017): Positional encoding for transformers
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//!
//! Dimensions (small for fast verification):
//! - Feature maps: 8x8 spatial, 64 channels
//! - Hidden: D=64, FFN_DIM=128, NUM_HEADS=4, NUM_QUERIES=8
//!
//! Part of #3883, #3915, #3945: NY compose tests for Table Transformer (DETR) subgraphs.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Feature map spatial size (H=W).
const FEAT_SIZE: usize = 8;
/// Backbone convolution channels.
const CHANNELS: usize = 64;
/// Hidden dimension for transformer encoder/decoder.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Number of learned object queries (DETR-style).
const NUM_QUERIES: usize = 8;
/// Encoder sequence length = FEAT_SIZE * FEAT_SIZE (flattened spatial).
const ENC_SEQ_LEN: usize = FEAT_SIZE * FEAT_SIZE; // 64
/// Number of table structure classes (e.g., table, row, column, cell, header).
const NUM_CLASSES: usize = 6;
/// DFL regression bins.
const DFL_BINS: usize = 16;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. ResNet basic block IBP: Conv2d -> BN -> ReLU -> Conv2d -> BN + skip
// ===========================================================================

/// Build a ResNet basic block kernel.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable, feature map).
/// Output: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (same spatial, residual added).
///
/// Architecture: Conv2d(C, C, k=3, s=1, p=1) -> BN -> ReLU -> Conv2d(C, C, k=3, s=1, p=1) -> BN + skip
/// Followed by ReLU on the sum.
fn build_resnet_basic_block_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = FEAT_SIZE;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("table_transformer_resnet_basic_block");

    let input = b.add_input("features", &feat_shape);

    // Conv2d weights and BN params for first conv
    let conv1_w = b.add_input("conv1_weight", &[c, c, 3, 3]);
    let conv1_b = b.add_input("conv1_bias", &[c]);
    let bn1_mean = b.add_input("bn1_running_mean", &[c]);
    let bn1_var = b.add_input("bn1_running_var", &[c]);
    let bn1_weight = b.add_input("bn1_weight", &[c]);
    let bn1_bias = b.add_input("bn1_bias", &[c]);
    let bn1_eps = b.add_input("bn1_eps", &[1]);

    // Conv2d weights and BN params for second conv
    let conv2_w = b.add_input("conv2_weight", &[c, c, 3, 3]);
    let conv2_b = b.add_input("conv2_bias", &[c]);
    let bn2_mean = b.add_input("bn2_running_mean", &[c]);
    let bn2_var = b.add_input("bn2_running_var", &[c]);
    let bn2_weight = b.add_input("bn2_weight", &[c]);
    let bn2_bias = b.add_input("bn2_bias", &[c]);
    let bn2_eps = b.add_input("bn2_eps", &[1]);

    // First conv: Conv2d(C, C, 3, stride=1, padding=1) -> BN -> ReLU
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

    // Second conv: Conv2d(C, C, 3, stride=1, padding=1) -> BN
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

    // Residual: bn2_out + input
    let residual = b.add_binary_add(bn2_out, input, &feat_shape);

    // Final ReLU
    let out = b.add_relu(residual, &feat_shape);

    b.build(out)
        .expect("valid Table Transformer ResNet basic block kernel")
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

/// IBP bounds propagate through ResNet basic block.
///
/// Conv2d -> BN -> ReLU -> Conv2d -> BN + skip -> ReLU.
/// ReLU clamps lower to 0. Residual connection adds skip to output.
#[test]
fn test_resnet_basic_block_ibp() {
    let def = build_resnet_basic_block_kernel();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ResNet basic block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, FEAT_SIZE, FEAT_SIZE],
        "ResNet basic block output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer ResNet basic block IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU ensures lower >= 0
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
}

// ===========================================================================
// 2. ResNet basic block CROWN
// ===========================================================================

/// CROWN bounds propagate through ResNet basic block.
///
/// ReLU is piecewise-linear and CROWN-friendly. BatchNorm is affine at
/// inference (with running stats). CROWN should produce tighter bounds
/// than IBP, especially for the ReLU linearization.
#[test]
fn test_resnet_basic_block_crown() {
    let def = build_resnet_basic_block_kernel();
    let bindings = resnet_basic_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, FEAT_SIZE, FEAT_SIZE],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer ResNet basic block CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record ResNet basic block.
#[test]
fn test_resnet_basic_block_verify_and_record() {
    let def = build_resnet_basic_block_kernel();
    let bindings = resnet_basic_block_bindings();
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_resnet_basic_block",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[CHANNELS, FEAT_SIZE, FEAT_SIZE]);
}

// ===========================================================================
// 3. ResNet backbone level IBP: Conv2d(stride=2) -> 2x BasicBlock
// ===========================================================================

/// Build a ResNet backbone level: stride-2 downsample conv + BN + ReLU.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable).
/// Output: `[CHANNELS, FEAT_SIZE/2, FEAT_SIZE/2]`.
///
/// This models one stage of the ResNet backbone with spatial downsampling
/// via a stride-2 convolution followed by BatchNorm and ReLU.
fn build_resnet_backbone_level_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s_in = FEAT_SIZE;
    let s_out = FEAT_SIZE / 2; // 4
    let out_shape = [c, s_out, s_out];
    let mut b = TensorBlockBuilder::new("table_transformer_resnet_backbone_level");

    let input = b.add_input("features", &[c, s_in, s_in]);

    // Downsample conv: Conv2d(C, C, k=3, s=2, p=1)
    let conv_w = b.add_input("ds_conv_weight", &[c, c, 3, 3]);
    let conv_b = b.add_input("ds_conv_bias", &[c]);
    let bn_mean = b.add_input("ds_bn_mean", &[c]);
    let bn_var = b.add_input("ds_bn_var", &[c]);
    let bn_weight = b.add_input("ds_bn_weight", &[c]);
    let bn_bias = b.add_input("ds_bn_bias", &[c]);
    let bn_eps = b.add_input("ds_bn_eps", &[1]);

    // Conv2d(stride=2): [C, 8, 8] -> [C, 4, 4]
    let conv_out = b.add_conv2d(input, conv_w, Some(conv_b), 2, 2, 1, 1, &out_shape);
    let bn_out = b.add_batch_norm(
        conv_out, bn_mean, bn_var, bn_weight, bn_bias, bn_eps, &out_shape,
    );
    let out = b.add_relu(bn_out, &out_shape);

    b.build(out)
        .expect("valid Table Transformer ResNet backbone level kernel")
}

/// Bindings for ResNet backbone level.
fn resnet_backbone_level_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_weight = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // features
        TensorParamBinding::ConstantTensor(conv_w),    // ds_conv_weight
        TensorParamBinding::ConstantTensor(conv_b),    // ds_conv_bias
        TensorParamBinding::ConstantTensor(bn_mean),   // ds_bn_mean
        TensorParamBinding::ConstantTensor(bn_var),    // ds_bn_var
        TensorParamBinding::ConstantTensor(bn_weight), // ds_bn_weight
        TensorParamBinding::ConstantTensor(bn_bias),   // ds_bn_bias
        TensorParamBinding::ConstantScalar(1e-5),      // ds_bn_eps
    ]
}

/// IBP bounds propagate through ResNet backbone level.
///
/// Conv2d(stride=2) halves spatial dimensions, followed by BN + ReLU.
/// ReLU clamps lower bound to 0.
#[test]
fn test_resnet_backbone_level_ibp() {
    let def = build_resnet_backbone_level_kernel();
    let bindings = resnet_backbone_level_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ResNet backbone level");

    let s_out = FEAT_SIZE / 2;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, s_out, s_out],
        "ResNet backbone level output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer ResNet backbone level IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU output
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
}

/// Verify and record ResNet backbone level.
#[test]
fn test_resnet_backbone_level_verify_and_record() {
    let def = build_resnet_backbone_level_kernel();
    let bindings = resnet_backbone_level_bindings();
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_resnet_backbone_level",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let s_out = FEAT_SIZE / 2;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[CHANNELS, s_out, s_out]);
}

// ===========================================================================
// 4. Transformer encoder layer CROWN: Self-attn -> LN -> FFN -> LN
// ===========================================================================

/// Build a DETR transformer encoder layer.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable, flattened spatial features).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture (pre-norm DETR encoder):
///   x_norm = LayerNorm(x)
///   attn_out = MultiHeadAttention(x_norm)  (self-attention)
///   x = x + attn_out                       (residual)
///   x_norm2 = LayerNorm(x)
///   ffn_out = Linear(ReLU(Linear(x_norm2)))
///   output = x + ffn_out                   (residual)
fn build_transformer_encoder_layer_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_encoder_layer");

    let input = b.add_input("encoder_features", &seq_shape);

    // LayerNorm 1 params
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_weight = b.add_input("ln1_weight", &[d]);
    let ln1_bias = b.add_input("ln1_bias", &[d]);

    // Self-attention weights
    let q_weight = b.add_input("q_weight", &[d, d]);
    let k_weight = b.add_input("k_weight", &[d, d]);
    let v_weight = b.add_input("v_weight", &[d, d]);
    let out_weight = b.add_input("out_weight", &[d, d]);

    // LayerNorm 2 params
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_weight = b.add_input("ln2_weight", &[d]);
    let ln2_bias = b.add_input("ln2_bias", &[d]);

    // FFN weights
    let ffn_up_w = b.add_input("ffn_up_weight", &[FFN_DIM, d]);
    let ffn_down_w = b.add_input("ffn_down_weight", &[d, FFN_DIM]);

    // LayerNorm 1
    let x_norm = b.add_layer_norm(input, ln1_eps, 1, ln1_weight, ln1_bias, &seq_shape);

    // Multi-head self-attention
    let attn_out = b
        .add_multi_head_attention(
            x_norm,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("valid self-attention");

    // Residual 1
    let x_res1 = b.add_binary_add(input, attn_out, &seq_shape);

    // LayerNorm 2
    let x_norm2 = b.add_layer_norm(x_res1, ln2_eps, 1, ln2_weight, ln2_bias, &seq_shape);

    // FFN: Linear -> ReLU -> Linear
    let ffn_hidden = b.add_linear(x_norm2, ffn_up_w, None, &ffn_shape);
    let ffn_act = b.add_relu(ffn_hidden, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn_down_w, None, &seq_shape);

    // Residual 2
    let out = b.add_binary_add(x_res1, ffn_out, &seq_shape);

    b.build(out)
        .expect("valid Table Transformer encoder layer kernel")
}

/// Bindings for transformer encoder layer.
fn transformer_encoder_layer_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let ln_weight = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,             // encoder_features
        TensorParamBinding::ConstantScalar(1e-5), // ln1_eps
        TensorParamBinding::ConstantTensor(ln_weight.clone()), // ln1_weight
        TensorParamBinding::ConstantTensor(ln_bias.clone()), // ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_w), // out_weight
        TensorParamBinding::ConstantScalar(1e-5), // ln2_eps
        TensorParamBinding::ConstantTensor(ln_weight), // ln2_weight
        TensorParamBinding::ConstantTensor(ln_bias), // ln2_bias
        TensorParamBinding::ConstantTensor(ffn_up_w), // ffn_up_weight
        TensorParamBinding::ConstantTensor(ffn_down_w), // ffn_down_weight
    ]
}

/// CROWN bounds propagate through transformer encoder layer.
///
/// Self-attention has softmax (CROWN-linearizable), LayerNorm requires
/// IbpValidated mode, and ReLU is piecewise-linear.
#[test]
fn test_transformer_encoder_layer_crown() {
    let def = build_transformer_encoder_layer_kernel();
    let bindings = transformer_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer encoder layer: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record transformer encoder layer.
#[test]
fn test_transformer_encoder_layer_verify_and_record() {
    let def = build_transformer_encoder_layer_kernel();
    let bindings = transformer_encoder_layer_bindings();
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "table_transformer_encoder_layer");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 5. DETR decoder cross-attention CROWN
// ===========================================================================

/// Build a DETR decoder cross-attention block.
///
/// Query input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, learned object queries).
/// KV input (encoder memory): `[ENC_SEQ_LEN, HIDDEN_DIM]` (constant).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
///
/// Cross-attention: queries attend to encoder memory features.
/// This is the key mechanism that allows DETR to relate object queries
/// to spatial features from the backbone.
fn build_detr_decoder_cross_attention_kernel() -> TensorKernelDef {
    let q_seq = NUM_QUERIES;
    let kv_seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q_shape = [q_seq, d];
    let mut b = TensorBlockBuilder::new("table_transformer_decoder_cross_attn");

    let q_input = b.add_input("object_queries", &q_shape);
    let kv_input = b.add_input("encoder_memory", &[kv_seq, d]);

    // Cross-attention projection weights
    let q_weight = b.add_input("cross_q_weight", &[d, d]);
    let k_weight = b.add_input("cross_k_weight", &[d, d]);
    let v_weight = b.add_input("cross_v_weight", &[d, d]);
    let out_weight = b.add_input("cross_out_weight", &[d, d]);

    // Multi-head cross-attention: queries attend to encoder features
    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_input,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid cross-attention");

    b.build(out)
        .expect("valid Table Transformer decoder cross-attention kernel")
}

/// Bindings for DETR decoder cross-attention.
///
/// Object queries are Variable. Encoder memory and projection weights are constant.
fn detr_decoder_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let kv_seq = ENC_SEQ_LEN;
    let memory = ArrayD::from_elem(IxDyn(&[kv_seq, d]), 0.1f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // object_queries
        TensorParamBinding::ConstantTensor(memory),         // encoder_memory
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // cross_out_weight
    ]
}

/// CROWN bounds propagate through DETR decoder cross-attention.
///
/// Cross-attention has softmax (CROWN-linearizable) and matmuls with
/// McCormick bilinear relaxation. Standard attention mask (bidirectional).
#[test]
fn test_detr_decoder_cross_attention_crown() {
    let def = build_detr_decoder_cross_attention_kernel();
    let bindings = detr_decoder_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer decoder cross-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record DETR decoder cross-attention.
#[test]
fn test_detr_decoder_cross_attention_verify_and_record() {
    let def = build_detr_decoder_cross_attention_kernel();
    let bindings = detr_decoder_cross_attention_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_decoder_cross_attn",
    );
    assert_eq!(result.num_variables, 1, "single Variable input (queries)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
}

// ===========================================================================
// 6. Classification head sigmoid IBP: Linear -> sigmoid
// ===========================================================================

/// Build a classification head: Linear -> sigmoid.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_QUERIES, NUM_CLASSES]` (class probabilities in [0, 1]).
///
/// DETR uses sigmoid classification (not softmax) since each query
/// independently predicts the class of the matched object.
fn build_classification_head_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_QUERIES, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("table_transformer_cls_head");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, HIDDEN_DIM]);
    let cls_weight = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_bias = b.add_input("cls_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, cls_weight, Some(cls_bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid Table Transformer classification head kernel")
}

/// Bindings for classification head.
fn classification_head_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // decoder_output
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias
    ]
}

/// IBP bounds propagate through classification head.
///
/// Linear -> sigmoid. Sigmoid maps R -> (0, 1).
/// Output bounds must be within [0, 1].
#[test]
fn test_classification_head_sigmoid_ibp() {
    let def = build_classification_head_sigmoid_kernel();
    let bindings = classification_head_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through classification head sigmoid");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, NUM_CLASSES],
        "classification head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer cls head IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid codomain is (0, 1).
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

/// Verify and record classification head.
#[test]
fn test_classification_head_sigmoid_verify_and_record() {
    let def = build_classification_head_sigmoid_kernel();
    let bindings = classification_head_sigmoid_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_cls_head_sigmoid",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, NUM_CLASSES]);
}

// ===========================================================================
// 7. Box regression sigmoid IBP: Linear -> sigmoid
// ===========================================================================

/// Build a box regression head: Linear -> sigmoid.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_QUERIES, 4]` (normalized box coordinates: cx, cy, w, h in [0, 1]).
///
/// DETR predicts normalized coordinates via sigmoid to ensure boxes
/// are within the image bounds.
fn build_box_regression_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_QUERIES, 4];
    let mut b = TensorBlockBuilder::new("table_transformer_box_head");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, HIDDEN_DIM]);
    let box_weight = b.add_input("box_weight", &[4, HIDDEN_DIM]);
    let box_bias = b.add_input("box_bias", &[4]);

    let logits = b.add_linear(input, box_weight, Some(box_bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid Table Transformer box regression head kernel")
}

/// Bindings for box regression head.
fn box_regression_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let box_w = ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // decoder_output
        TensorParamBinding::ConstantTensor(box_w), // box_weight
        TensorParamBinding::ConstantTensor(box_b), // box_bias
    ]
}

/// IBP bounds propagate through box regression head.
///
/// Linear -> sigmoid ensures output in [0, 1] for normalized box coordinates.
#[test]
fn test_box_regression_sigmoid_ibp() {
    let def = build_box_regression_sigmoid_kernel();
    let bindings = box_regression_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through box regression sigmoid");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, 4],
        "box regression output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer box head IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid ensures [0, 1] output
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

/// Verify and record box regression head.
#[test]
fn test_box_regression_sigmoid_verify_and_record() {
    let def = build_box_regression_sigmoid_kernel();
    let bindings = box_regression_sigmoid_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "table_transformer_box_head_sigmoid",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, 4]);
}

// ===========================================================================
// 8. Sinusoidal position encoding IBP: sin/cos bounded in [-1, 1]
// ===========================================================================

/// Build a sinusoidal position encoding verification kernel.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable, features to add PE to).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]` (features + positional encoding).
///
/// The PE is a constant tensor with sin/cos values in [-1, 1].
/// Adding PE to features shifts bounds by at most +-1 per element.
fn build_sinusoidal_position_encoding_kernel() -> TensorKernelDef {
    let seq_shape = [ENC_SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_sinusoidal_pe");

    let input = b.add_input("features", &seq_shape);
    let pe = b.add_input("positional_encoding", &seq_shape);

    // features + PE
    let out = b.add_binary_add(input, pe, &seq_shape);

    b.build(out)
        .expect("valid Table Transformer sinusoidal PE kernel")
}

/// Build sinusoidal PE tensor with values in [-1, 1].
fn sinusoidal_pe_tensor() -> ArrayD<f32> {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let mut data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            data[t * d + 2 * i] = freq.sin() as f32;
            data[t * d + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq, d]), data).expect("valid PE")
}

/// Bindings for sinusoidal PE.
fn sinusoidal_position_encoding_bindings() -> Vec<TensorParamBinding> {
    let pe = sinusoidal_pe_tensor();

    vec![
        TensorParamBinding::Variable,           // features
        TensorParamBinding::ConstantTensor(pe), // positional_encoding
    ]
}

/// IBP bounds propagate through sinusoidal position encoding.
///
/// Adding constant PE to variable features shifts bounds by the PE values.
/// Since PE values are sin/cos in [-1, 1], output bounds widen by at most 1
/// in each direction.
#[test]
fn test_sinusoidal_position_encoding_ibp() {
    let def = build_sinusoidal_position_encoding_kernel();
    let bindings = sinusoidal_position_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through sinusoidal PE");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[ENC_SEQ_LEN, HIDDEN_DIM],
        "sinusoidal PE output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer sinusoidal PE IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Input [-2, 2] + PE [-1, 1] = output in [-3, 3]
    assert!(
        lo_min >= -3.0 - 1e-6,
        "PE-shifted lower should be >= -3.0, got {lo_min}"
    );
    assert!(
        hi_max <= 3.0 + 1e-6,
        "PE-shifted upper should be <= 3.0, got {hi_max}"
    );
}

/// Verify and record sinusoidal PE.
#[test]
fn test_sinusoidal_position_encoding_verify_and_record() {
    let def = build_sinusoidal_position_encoding_kernel();
    let bindings = sinusoidal_position_encoding_bindings();
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "table_transformer_sinusoidal_pe");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 9. DFL regression softmax IBP: Softmax over reg_max bins
// ===========================================================================

/// Build a DFL regression kernel for table bounding box refinement.
///
/// Input: `[NUM_QUERIES, DFL_BINS]` (Variable, DFL logits).
/// Output: `[NUM_QUERIES, 1]` (continuous box coordinate).
///
/// DFL (Distribution Focal Loss) converts discrete bin logits to
/// continuous coordinates via softmax -> weighted sum with bin indices.
fn build_dfl_regression_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("table_transformer_dfl_regression");

    let input = b.add_input("dfl_logits", &[NUM_QUERIES, DFL_BINS]);
    let bins = b.add_input("bins", &[DFL_BINS, 1]);

    // Softmax along last dimension
    let probs = b.add_softmax(input, 1, &[NUM_QUERIES, DFL_BINS]);

    // Weighted sum: matmul(probs, bins) -> [NUM_QUERIES, 1]
    let out = b.add_matmul(probs, bins, false, None, &[NUM_QUERIES, 1]);

    b.build(out)
        .expect("valid Table Transformer DFL regression kernel")
}

/// Bindings for DFL regression.
fn dfl_regression_softmax_bindings() -> Vec<TensorParamBinding> {
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bins = ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins shape");

    vec![
        TensorParamBinding::Variable,             // dfl_logits
        TensorParamBinding::ConstantTensor(bins), // bins
    ]
}

/// IBP bounds propagate through DFL regression.
///
/// Softmax produces a probability distribution over bins [0, ..., DFL_BINS-1].
/// The weighted sum should ideally be in [0, DFL_BINS-1].
#[test]
fn test_dfl_regression_softmax_ibp() {
    let def = build_dfl_regression_softmax_kernel();
    let bindings = dfl_regression_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, DFL_BINS], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL regression");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, 1],
        "DFL regression output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer DFL regression IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record DFL regression.
#[test]
fn test_dfl_regression_softmax_verify_and_record() {
    let def = build_dfl_regression_softmax_kernel();
    let bindings = dfl_regression_softmax_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, DFL_BINS], 5.0);

    let result = verify_and_assert(&def, &bindings, &input, "table_transformer_dfl_regression");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, 1]);
}

// ===========================================================================
// 10. Full detection compose IBP: features -> cls + box heads
// ===========================================================================

/// Build a simplified end-to-end detection pipeline.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder output features).
/// Output: `[NUM_QUERIES, NUM_CLASSES + 4]` (class probs + box coords, all sigmoid).
///
/// Architecture:
///   cls_logits = Linear_cls(features)         [NUM_QUERIES, NUM_CLASSES]
///   cls_probs = sigmoid(cls_logits)           [NUM_QUERIES, NUM_CLASSES]
///   box_logits = Linear_box(features)         [NUM_QUERIES, 4]
///   box_coords = sigmoid(box_logits)          [NUM_QUERIES, 4]
///   output = concat(cls_probs, box_coords)    [NUM_QUERIES, NUM_CLASSES + 4]
fn build_full_detection_compose_kernel() -> TensorKernelDef {
    let total_out = NUM_CLASSES + 4;
    let out_shape = [NUM_QUERIES, total_out];
    let cls_shape = [NUM_QUERIES, NUM_CLASSES];
    let box_shape = [NUM_QUERIES, 4];
    let mut b = TensorBlockBuilder::new("table_transformer_full_detection");

    let input = b.add_input("decoder_features", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_probs = b.add_sigmoid(cls_logits, &cls_shape);

    // Box regression head
    let box_w = b.add_input("box_weight", &[4, HIDDEN_DIM]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_shape);
    let box_coords = b.add_sigmoid(box_logits, &box_shape);

    // Concatenate cls + box along feature dimension (dim=1)
    let out = b.add_concat(&[cls_probs, box_coords], 1, &out_shape);

    b.build(out)
        .expect("valid Table Transformer full detection compose kernel")
}

/// Bindings for full detection compose.
fn full_detection_compose_bindings() -> Vec<TensorParamBinding> {
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    let box_w = ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // decoder_features
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias
        TensorParamBinding::ConstantTensor(box_w), // box_weight
        TensorParamBinding::ConstantTensor(box_b), // box_bias
    ]
}

/// IBP bounds propagate through full detection compose.
///
/// Both classification and box regression heads use sigmoid, so ALL
/// output elements should be bounded in [0, 1].
#[test]
fn test_full_detection_compose_ibp() {
    let def = build_full_detection_compose_kernel();
    let bindings = full_detection_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full detection compose");

    let total_out = NUM_CLASSES + 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, total_out],
        "full detection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer full detection IBP: bounds=[{lo_min}, {hi_max}]");

    // All outputs go through sigmoid, so must be in [0, 1].
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

/// CROWN bounds propagate through full detection compose.
#[test]
fn test_full_detection_compose_crown() {
    let def = build_full_detection_compose_kernel();
    let bindings = full_detection_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let total_out = NUM_CLASSES + 4;
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, total_out],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer full detection: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record full detection compose.
#[test]
fn test_full_detection_compose_verify_and_record() {
    let def = build_full_detection_compose_kernel();
    let bindings = full_detection_compose_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "table_transformer_full_detection");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let total_out = NUM_CLASSES + 4;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, total_out]);
}

// ===========================================================================
// 11. DETR encoder 2-layer stack: Self-attn -> LN -> FFN -> LN (x2)
// ===========================================================================

/// Build a 2-layer DETR encoder stack.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable, flattened spatial features).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Two stacked encoder layers, each with:
///   LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN(ReLU) -> Residual
///
/// This tests deep bounds propagation through repeated transformer structure,
/// verifying that bounds do not blow up or collapse through layer stacking.
fn build_encoder_2layer_stack_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_encoder_2layer");

    let input = b.add_input("encoder_features", &seq_shape);

    // --- Layer 1 ---
    let ln1a_eps = b.add_input("l1_ln1_eps", &[1]);
    let ln1a_w = b.add_input("l1_ln1_weight", &[d]);
    let ln1a_b = b.add_input("l1_ln1_bias", &[d]);
    let q1_w = b.add_input("l1_q_weight", &[d, d]);
    let k1_w = b.add_input("l1_k_weight", &[d, d]);
    let v1_w = b.add_input("l1_v_weight", &[d, d]);
    let o1_w = b.add_input("l1_out_weight", &[d, d]);
    let ln1b_eps = b.add_input("l1_ln2_eps", &[1]);
    let ln1b_w = b.add_input("l1_ln2_weight", &[d]);
    let ln1b_b = b.add_input("l1_ln2_bias", &[d]);
    let ffn1_up = b.add_input("l1_ffn_up_weight", &[FFN_DIM, d]);
    let ffn1_dn = b.add_input("l1_ffn_down_weight", &[d, FFN_DIM]);

    let x1_norm = b.add_layer_norm(input, ln1a_eps, 1, ln1a_w, ln1a_b, &seq_shape);
    let attn1 = b
        .add_multi_head_attention(
            x1_norm,
            q1_w,
            k1_w,
            v1_w,
            o1_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("valid self-attention L1");
    let res1a = b.add_binary_add(input, attn1, &seq_shape);
    let x1_norm2 = b.add_layer_norm(res1a, ln1b_eps, 1, ln1b_w, ln1b_b, &seq_shape);
    let ffn1_h = b.add_linear(x1_norm2, ffn1_up, None, &ffn_shape);
    let ffn1_act = b.add_relu(ffn1_h, &ffn_shape);
    let ffn1_out = b.add_linear(ffn1_act, ffn1_dn, None, &seq_shape);
    let layer1_out = b.add_binary_add(res1a, ffn1_out, &seq_shape);

    // --- Layer 2 ---
    let ln2a_eps = b.add_input("l2_ln1_eps", &[1]);
    let ln2a_w = b.add_input("l2_ln1_weight", &[d]);
    let ln2a_b = b.add_input("l2_ln1_bias", &[d]);
    let q2_w = b.add_input("l2_q_weight", &[d, d]);
    let k2_w = b.add_input("l2_k_weight", &[d, d]);
    let v2_w = b.add_input("l2_v_weight", &[d, d]);
    let o2_w = b.add_input("l2_out_weight", &[d, d]);
    let ln2b_eps = b.add_input("l2_ln2_eps", &[1]);
    let ln2b_w = b.add_input("l2_ln2_weight", &[d]);
    let ln2b_b = b.add_input("l2_ln2_bias", &[d]);
    let ffn2_up = b.add_input("l2_ffn_up_weight", &[FFN_DIM, d]);
    let ffn2_dn = b.add_input("l2_ffn_down_weight", &[d, FFN_DIM]);

    let x2_norm = b.add_layer_norm(layer1_out, ln2a_eps, 1, ln2a_w, ln2a_b, &seq_shape);
    let attn2 = b
        .add_multi_head_attention(
            x2_norm,
            q2_w,
            k2_w,
            v2_w,
            o2_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("valid self-attention L2");
    let res2a = b.add_binary_add(layer1_out, attn2, &seq_shape);
    let x2_norm2 = b.add_layer_norm(res2a, ln2b_eps, 1, ln2b_w, ln2b_b, &seq_shape);
    let ffn2_h = b.add_linear(x2_norm2, ffn2_up, None, &ffn_shape);
    let ffn2_act = b.add_relu(ffn2_h, &ffn_shape);
    let ffn2_out = b.add_linear(ffn2_act, ffn2_dn, None, &seq_shape);
    let out = b.add_binary_add(res2a, ffn2_out, &seq_shape);

    b.build(out)
        .expect("valid Table Transformer 2-layer encoder stack kernel")
}

/// Bindings for 2-layer encoder stack.
fn encoder_2layer_stack_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    // Build bindings for one layer
    let one_layer = |_label: &str| -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantScalar(1e-5),         // ln1_eps
            TensorParamBinding::ConstantTensor(ln_w.clone()), // ln1_weight
            TensorParamBinding::ConstantTensor(ln_b.clone()), // ln1_bias
            TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
            TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
            TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
            TensorParamBinding::ConstantTensor(attn_w.clone()), // out_weight
            TensorParamBinding::ConstantScalar(1e-5),         // ln2_eps
            TensorParamBinding::ConstantTensor(ln_w.clone()), // ln2_weight
            TensorParamBinding::ConstantTensor(ln_b.clone()), // ln2_bias
            TensorParamBinding::ConstantTensor(ffn_up.clone()), // ffn_up_weight
            TensorParamBinding::ConstantTensor(ffn_dn.clone()), // ffn_down_weight
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable]; // encoder_features
    bindings.extend(one_layer("l1"));
    bindings.extend(one_layer("l2"));
    bindings
}

/// IBP bounds propagate through 2-layer DETR encoder stack.
///
/// Tests that stacked self-attention + FFN layers maintain finite, valid bounds.
/// Residual connections stabilize bounds across layers.
#[test]
fn test_encoder_2layer_stack_ibp() {
    let def = build_encoder_2layer_stack_kernel();
    let bindings = encoder_2layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer encoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[ENC_SEQ_LEN, HIDDEN_DIM],
        "encoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer encoder 2-layer stack IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through 2-layer DETR encoder stack.
#[test]
fn test_encoder_2layer_stack_crown() {
    let def = build_encoder_2layer_stack_kernel();
    let bindings = encoder_2layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer encoder 2-layer stack: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. DETR decoder 2-layer stack with cross-attention
// ===========================================================================

/// Build a 2-layer DETR decoder stack with self-attention + cross-attention.
///
/// Query input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, learned object queries).
/// Encoder memory: `[ENC_SEQ_LEN, HIDDEN_DIM]` (constant).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
///
/// Each decoder layer:
///   LN -> Self-Attention(queries) -> Residual
///   LN -> Cross-Attention(queries, memory) -> Residual
///   LN -> FFN(ReLU) -> Residual
fn build_decoder_2layer_stack_kernel() -> TensorKernelDef {
    let q_seq = NUM_QUERIES;
    let kv_seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q_shape = [q_seq, d];
    let ffn_shape = [q_seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_decoder_2layer");

    let query_input = b.add_input("object_queries", &q_shape);
    let memory = b.add_input("encoder_memory", &[kv_seq, d]);

    // --- Decoder Layer 1 ---
    // Self-attention
    let d1_ln1_eps = b.add_input("d1_ln1_eps", &[1]);
    let d1_ln1_w = b.add_input("d1_ln1_weight", &[d]);
    let d1_ln1_b = b.add_input("d1_ln1_bias", &[d]);
    let d1_sq_w = b.add_input("d1_self_q_weight", &[d, d]);
    let d1_sk_w = b.add_input("d1_self_k_weight", &[d, d]);
    let d1_sv_w = b.add_input("d1_self_v_weight", &[d, d]);
    let d1_so_w = b.add_input("d1_self_out_weight", &[d, d]);

    let d1_norm1 = b.add_layer_norm(query_input, d1_ln1_eps, 1, d1_ln1_w, d1_ln1_b, &q_shape);
    let d1_self_attn = b
        .add_multi_head_attention(
            d1_norm1,
            d1_sq_w,
            d1_sk_w,
            d1_sv_w,
            d1_so_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid decoder L1 self-attention");
    let d1_res1 = b.add_binary_add(query_input, d1_self_attn, &q_shape);

    // Cross-attention
    let d1_ln2_eps = b.add_input("d1_ln2_eps", &[1]);
    let d1_ln2_w = b.add_input("d1_ln2_weight", &[d]);
    let d1_ln2_b = b.add_input("d1_ln2_bias", &[d]);
    let d1_cq_w = b.add_input("d1_cross_q_weight", &[d, d]);
    let d1_ck_w = b.add_input("d1_cross_k_weight", &[d, d]);
    let d1_cv_w = b.add_input("d1_cross_v_weight", &[d, d]);
    let d1_co_w = b.add_input("d1_cross_out_weight", &[d, d]);

    let d1_norm2 = b.add_layer_norm(d1_res1, d1_ln2_eps, 1, d1_ln2_w, d1_ln2_b, &q_shape);
    let d1_cross_attn = b
        .add_multi_head_cross_attention(
            d1_norm2,
            memory,
            d1_cq_w,
            d1_ck_w,
            d1_cv_w,
            d1_co_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid decoder L1 cross-attention");
    let d1_res2 = b.add_binary_add(d1_res1, d1_cross_attn, &q_shape);

    // FFN
    let d1_ln3_eps = b.add_input("d1_ln3_eps", &[1]);
    let d1_ln3_w = b.add_input("d1_ln3_weight", &[d]);
    let d1_ln3_b = b.add_input("d1_ln3_bias", &[d]);
    let d1_ffn_up = b.add_input("d1_ffn_up_weight", &[FFN_DIM, d]);
    let d1_ffn_dn = b.add_input("d1_ffn_down_weight", &[d, FFN_DIM]);

    let d1_norm3 = b.add_layer_norm(d1_res2, d1_ln3_eps, 1, d1_ln3_w, d1_ln3_b, &q_shape);
    let d1_ffn_h = b.add_linear(d1_norm3, d1_ffn_up, None, &ffn_shape);
    let d1_ffn_act = b.add_relu(d1_ffn_h, &ffn_shape);
    let d1_ffn_out = b.add_linear(d1_ffn_act, d1_ffn_dn, None, &q_shape);
    let layer1_out = b.add_binary_add(d1_res2, d1_ffn_out, &q_shape);

    // --- Decoder Layer 2 ---
    let d2_ln1_eps = b.add_input("d2_ln1_eps", &[1]);
    let d2_ln1_w = b.add_input("d2_ln1_weight", &[d]);
    let d2_ln1_b = b.add_input("d2_ln1_bias", &[d]);
    let d2_sq_w = b.add_input("d2_self_q_weight", &[d, d]);
    let d2_sk_w = b.add_input("d2_self_k_weight", &[d, d]);
    let d2_sv_w = b.add_input("d2_self_v_weight", &[d, d]);
    let d2_so_w = b.add_input("d2_self_out_weight", &[d, d]);

    let d2_norm1 = b.add_layer_norm(layer1_out, d2_ln1_eps, 1, d2_ln1_w, d2_ln1_b, &q_shape);
    let d2_self_attn = b
        .add_multi_head_attention(
            d2_norm1,
            d2_sq_w,
            d2_sk_w,
            d2_sv_w,
            d2_so_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid decoder L2 self-attention");
    let d2_res1 = b.add_binary_add(layer1_out, d2_self_attn, &q_shape);

    let d2_ln2_eps = b.add_input("d2_ln2_eps", &[1]);
    let d2_ln2_w = b.add_input("d2_ln2_weight", &[d]);
    let d2_ln2_b = b.add_input("d2_ln2_bias", &[d]);
    let d2_cq_w = b.add_input("d2_cross_q_weight", &[d, d]);
    let d2_ck_w = b.add_input("d2_cross_k_weight", &[d, d]);
    let d2_cv_w = b.add_input("d2_cross_v_weight", &[d, d]);
    let d2_co_w = b.add_input("d2_cross_out_weight", &[d, d]);

    let d2_norm2 = b.add_layer_norm(d2_res1, d2_ln2_eps, 1, d2_ln2_w, d2_ln2_b, &q_shape);
    let d2_cross_attn = b
        .add_multi_head_cross_attention(
            d2_norm2,
            memory,
            d2_cq_w,
            d2_ck_w,
            d2_cv_w,
            d2_co_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid decoder L2 cross-attention");
    let d2_res2 = b.add_binary_add(d2_res1, d2_cross_attn, &q_shape);

    let d2_ln3_eps = b.add_input("d2_ln3_eps", &[1]);
    let d2_ln3_w = b.add_input("d2_ln3_weight", &[d]);
    let d2_ln3_b = b.add_input("d2_ln3_bias", &[d]);
    let d2_ffn_up = b.add_input("d2_ffn_up_weight", &[FFN_DIM, d]);
    let d2_ffn_dn = b.add_input("d2_ffn_down_weight", &[d, FFN_DIM]);

    let d2_norm3 = b.add_layer_norm(d2_res2, d2_ln3_eps, 1, d2_ln3_w, d2_ln3_b, &q_shape);
    let d2_ffn_h = b.add_linear(d2_norm3, d2_ffn_up, None, &ffn_shape);
    let d2_ffn_act = b.add_relu(d2_ffn_h, &ffn_shape);
    let d2_ffn_out = b.add_linear(d2_ffn_act, d2_ffn_dn, None, &q_shape);
    let out = b.add_binary_add(d2_res2, d2_ffn_out, &q_shape);

    b.build(out)
        .expect("valid Table Transformer 2-layer decoder stack kernel")
}

/// Bindings for 2-layer decoder stack.
fn decoder_2layer_stack_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let kv_seq = ENC_SEQ_LEN;
    let memory = ArrayD::from_elem(IxDyn(&[kv_seq, d]), 0.1f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    // Per-layer: 3 LN blocks (self-attn, cross-attn, FFN) x (eps, weight, bias)
    //          + 4 self-attn weights + 4 cross-attn weights + 2 FFN weights = 19 params
    let one_decoder_layer = || -> Vec<TensorParamBinding> {
        vec![
            // Self-attention LN + weights
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            // Cross-attention LN + weights
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            // FFN LN + weights
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(ffn_up.clone()),
            TensorParamBinding::ConstantTensor(ffn_dn.clone()),
        ]
    };

    let mut bindings = vec![
        TensorParamBinding::Variable,               // object_queries
        TensorParamBinding::ConstantTensor(memory), // encoder_memory
    ];
    bindings.extend(one_decoder_layer());
    bindings.extend(one_decoder_layer());
    bindings
}

/// IBP bounds propagate through 2-layer DETR decoder stack.
///
/// Tests deep decoder composition with both self-attention and cross-attention.
/// Cross-attention injects constant encoder features at every layer.
#[test]
fn test_decoder_2layer_stack_ibp() {
    let def = build_decoder_2layer_stack_kernel();
    let bindings = decoder_2layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, HIDDEN_DIM],
        "decoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer decoder 2-layer stack IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through 2-layer DETR decoder stack.
#[test]
fn test_decoder_2layer_stack_crown() {
    let def = build_decoder_2layer_stack_kernel();
    let bindings = decoder_2layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer decoder 2-layer stack: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. ResNet 2-stage backbone: downsample -> downsample (chain)
// ===========================================================================

/// Build a ResNet 2-stage backbone with cascading spatial downsampling.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable, e.g. [64, 8, 8]).
/// Output: `[CHANNELS, FEAT_SIZE/4, FEAT_SIZE/4]` (e.g. [64, 2, 2]).
///
/// Two consecutive stages, each halving spatial dimensions:
///   Stage 1: Conv2d(stride=2) -> BN -> ReLU  ([C, 8, 8] -> [C, 4, 4])
///   Stage 2: Conv2d(stride=2) -> BN -> ReLU  ([C, 4, 4] -> [C, 2, 2])
fn build_resnet_2stage_backbone_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s1 = FEAT_SIZE; // 8
    let s2 = FEAT_SIZE / 2; // 4
    let s3 = FEAT_SIZE / 4; // 2
    let mut b = TensorBlockBuilder::new("table_transformer_resnet_2stage");

    let input = b.add_input("features", &[c, s1, s1]);

    // Stage 1: [C, 8, 8] -> [C, 4, 4]
    let c1_w = b.add_input("s1_conv_weight", &[c, c, 3, 3]);
    let c1_b = b.add_input("s1_conv_bias", &[c]);
    let bn1_mean = b.add_input("s1_bn_mean", &[c]);
    let bn1_var = b.add_input("s1_bn_var", &[c]);
    let bn1_w = b.add_input("s1_bn_weight", &[c]);
    let bn1_b = b.add_input("s1_bn_bias", &[c]);
    let bn1_eps = b.add_input("s1_bn_eps", &[1]);

    let conv1 = b.add_conv2d(input, c1_w, Some(c1_b), 2, 2, 1, 1, &[c, s2, s2]);
    let bn1 = b.add_batch_norm(
        conv1,
        bn1_mean,
        bn1_var,
        bn1_w,
        bn1_b,
        bn1_eps,
        &[c, s2, s2],
    );
    let relu1 = b.add_relu(bn1, &[c, s2, s2]);

    // Stage 2: [C, 4, 4] -> [C, 2, 2]
    let c2_w = b.add_input("s2_conv_weight", &[c, c, 3, 3]);
    let c2_b = b.add_input("s2_conv_bias", &[c]);
    let bn2_mean = b.add_input("s2_bn_mean", &[c]);
    let bn2_var = b.add_input("s2_bn_var", &[c]);
    let bn2_w = b.add_input("s2_bn_weight", &[c]);
    let bn2_b = b.add_input("s2_bn_bias", &[c]);
    let bn2_eps = b.add_input("s2_bn_eps", &[1]);

    let conv2 = b.add_conv2d(relu1, c2_w, Some(c2_b), 2, 2, 1, 1, &[c, s3, s3]);
    let bn2 = b.add_batch_norm(
        conv2,
        bn2_mean,
        bn2_var,
        bn2_w,
        bn2_b,
        bn2_eps,
        &[c, s3, s3],
    );
    let out = b.add_relu(bn2, &[c, s3, s3]);

    b.build(out)
        .expect("valid Table Transformer 2-stage ResNet backbone kernel")
}

/// Bindings for 2-stage ResNet backbone.
fn resnet_2stage_backbone_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_w = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    let one_stage = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantTensor(conv_w.clone()),
            TensorParamBinding::ConstantTensor(conv_b.clone()),
            TensorParamBinding::ConstantTensor(bn_mean.clone()),
            TensorParamBinding::ConstantTensor(bn_var.clone()),
            TensorParamBinding::ConstantTensor(bn_w.clone()),
            TensorParamBinding::ConstantTensor(bn_b.clone()),
            TensorParamBinding::ConstantScalar(1e-5),
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable]; // features
    bindings.extend(one_stage());
    bindings.extend(one_stage());
    bindings
}

/// IBP bounds propagate through 2-stage ResNet backbone.
///
/// Two cascaded stride-2 convolutions: 8x8 -> 4x4 -> 2x2.
/// Each stage includes BN + ReLU. ReLU clamps lower to 0.
#[test]
fn test_resnet_2stage_backbone_ibp() {
    let def = build_resnet_2stage_backbone_kernel();
    let bindings = resnet_2stage_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-stage backbone");

    let s_out = FEAT_SIZE / 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, s_out, s_out],
        "2-stage backbone output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer 2-stage backbone IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "ReLU lower must be >= 0, got {lo_min}");
}

// ===========================================================================
// 14. Multi-head cross-attention: queries attend to encoder features (IBP)
// ===========================================================================

/// Build a multi-head cross-attention with LayerNorm pre-processing on both
/// queries and encoder memory.
///
/// Query input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
///
/// Architecture:
///   q_normed = LayerNorm(queries)
///   cross_attn_out = CrossAttention(q_normed, memory)
///   output = queries + cross_attn_out  (residual)
fn build_cross_attn_with_norm_kernel() -> TensorKernelDef {
    let q_seq = NUM_QUERIES;
    let kv_seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q_shape = [q_seq, d];
    let mut b = TensorBlockBuilder::new("table_transformer_cross_attn_normed");

    let q_input = b.add_input("object_queries", &q_shape);
    let memory = b.add_input("encoder_memory", &[kv_seq, d]);

    // LayerNorm on queries
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[d]);
    let ln_b = b.add_input("ln_bias", &[d]);

    // Cross-attention weights
    let q_w = b.add_input("cross_q_weight", &[d, d]);
    let k_w = b.add_input("cross_k_weight", &[d, d]);
    let v_w = b.add_input("cross_v_weight", &[d, d]);
    let o_w = b.add_input("cross_out_weight", &[d, d]);

    let q_normed = b.add_layer_norm(q_input, ln_eps, 1, ln_w, ln_b, &q_shape);
    let cross_out = b
        .add_multi_head_cross_attention(
            q_normed,
            memory,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid cross-attention with norm");

    // Residual
    let out = b.add_binary_add(q_input, cross_out, &q_shape);

    b.build(out)
        .expect("valid Table Transformer cross-attention with norm kernel")
}

/// Bindings for cross-attention with LayerNorm.
fn cross_attn_with_norm_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let kv_seq = ENC_SEQ_LEN;
    let memory = ArrayD::from_elem(IxDyn(&[kv_seq, d]), 0.1f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // object_queries
        TensorParamBinding::ConstantTensor(memory),         // encoder_memory
        TensorParamBinding::ConstantScalar(1e-5),           // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),           // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // cross_v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // cross_out_weight
    ]
}

/// IBP bounds through multi-head cross-attention with LayerNorm + residual.
///
/// Verifies that cross-attention with normalization pre-processing maintains
/// valid bounds. The residual connection adds the raw query input.
#[test]
fn test_cross_attn_with_norm_ibp() {
    let def = build_cross_attn_with_norm_kernel();
    let bindings = cross_attn_with_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attn with norm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, HIDDEN_DIM],
        "cross-attn with norm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer cross-attn+norm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Position encoding + attention composition
// ===========================================================================

/// Build a position encoding + self-attention composition.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable, backbone features).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture (DETR encoder input processing):
///   x_pe = features + positional_encoding
///   x_norm = LayerNorm(x_pe)
///   attn_out = MultiHeadAttention(x_norm)
///   output = x_pe + attn_out  (residual)
fn build_pe_plus_attention_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let mut b = TensorBlockBuilder::new("table_transformer_pe_attention");

    let input = b.add_input("features", &seq_shape);
    let pe = b.add_input("positional_encoding", &seq_shape);

    // Add PE
    let x_pe = b.add_binary_add(input, pe, &seq_shape);

    // LayerNorm
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[d]);
    let ln_b = b.add_input("ln_bias", &[d]);

    // Self-attention
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let o_w = b.add_input("out_weight", &[d, d]);

    let x_norm = b.add_layer_norm(x_pe, ln_eps, 1, ln_w, ln_b, &seq_shape);
    let attn_out = b
        .add_multi_head_attention(
            x_norm,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("valid PE + attention");

    // Residual
    let out = b.add_binary_add(x_pe, attn_out, &seq_shape);

    b.build(out)
        .expect("valid Table Transformer PE + attention kernel")
}

/// Bindings for PE + attention.
fn pe_plus_attention_bindings() -> Vec<TensorParamBinding> {
    let pe = sinusoidal_pe_tensor();
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // features
        TensorParamBinding::ConstantTensor(pe),             // positional_encoding
        TensorParamBinding::ConstantScalar(1e-5),           // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),           // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
    ]
}

/// IBP bounds through position encoding + self-attention composition.
///
/// Tests the DETR encoder input processing pattern: features + PE -> LN -> MHA -> residual.
/// PE shifts bounds by at most 1.0, then self-attention and residual maintain validity.
#[test]
fn test_pe_plus_attention_ibp() {
    let def = build_pe_plus_attention_kernel();
    let bindings = pe_plus_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PE + attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[ENC_SEQ_LEN, HIDDEN_DIM],
        "PE + attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer PE + attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through position encoding + self-attention composition.
#[test]
fn test_pe_plus_attention_crown() {
    let def = build_pe_plus_attention_kernel();
    let bindings = pe_plus_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer PE + attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 16. Full DETR: encoder -> decoder -> classification + box heads
// ===========================================================================

/// Build a simplified full DETR pipeline: encoder layer -> decoder cross-attn -> heads.
///
/// Encoder input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable, backbone features).
/// Output: `[NUM_QUERIES, NUM_CLASSES + 4]` (class probs + box coords).
///
/// Architecture:
///   encoder_out = EncoderLayer(features)
///   decoder_out = CrossAttention(learned_queries, encoder_out_as_const) [simplified]
///   cls_probs = sigmoid(Linear_cls(decoder_out))
///   box_coords = sigmoid(Linear_box(decoder_out))
///   output = concat(cls_probs, box_coords)
///
/// Note: The decoder cross-attention treats encoder output as constant (pre-computed)
/// because the Variable is the encoder input. Decoder queries are also constant
/// (learned embeddings). This models the common inference pattern.
fn build_full_detr_pipeline_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q_seq = NUM_QUERIES;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let q_shape = [q_seq, d];
    let total_out = NUM_CLASSES + 4;
    let mut b = TensorBlockBuilder::new("table_transformer_full_detr");

    let input = b.add_input("backbone_features", &seq_shape);

    // Encoder: LN -> Self-Attn -> Residual -> LN -> FFN -> Residual
    let ln1_eps = b.add_input("enc_ln1_eps", &[1]);
    let ln1_w = b.add_input("enc_ln1_weight", &[d]);
    let ln1_b = b.add_input("enc_ln1_bias", &[d]);
    let eq_w = b.add_input("enc_q_weight", &[d, d]);
    let ek_w = b.add_input("enc_k_weight", &[d, d]);
    let ev_w = b.add_input("enc_v_weight", &[d, d]);
    let eo_w = b.add_input("enc_out_weight", &[d, d]);
    let ln2_eps = b.add_input("enc_ln2_eps", &[1]);
    let ln2_w = b.add_input("enc_ln2_weight", &[d]);
    let ln2_b = b.add_input("enc_ln2_bias", &[d]);
    let e_ffn_up = b.add_input("enc_ffn_up_weight", &[FFN_DIM, d]);
    let e_ffn_dn = b.add_input("enc_ffn_down_weight", &[d, FFN_DIM]);

    let x_norm = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &seq_shape);
    let attn_out = b
        .add_multi_head_attention(
            x_norm,
            eq_w,
            ek_w,
            ev_w,
            eo_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("valid encoder self-attention");
    let res1 = b.add_binary_add(input, attn_out, &seq_shape);
    let x_norm2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &seq_shape);
    let ffn_h = b.add_linear(x_norm2, e_ffn_up, None, &ffn_shape);
    let ffn_act = b.add_relu(ffn_h, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, e_ffn_dn, None, &seq_shape);
    let enc_out = b.add_binary_add(res1, ffn_out, &seq_shape);

    // Decoder input: learned queries (constant) + cross-attention to encoder output.
    // Since encoder output is a function of the Variable input, we use the
    // encoder output directly as a projection through a linear layer to
    // NUM_QUERIES dimension, simulating a simplified decoder path.
    let dec_proj_w = b.add_input("dec_proj_weight", &[q_seq, seq]);
    let dec_proj = b.add_matmul(dec_proj_w, enc_out, false, None, &q_shape);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, d]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(dec_proj, cls_w, Some(cls_b), &[q_seq, NUM_CLASSES]);
    let cls_probs = b.add_sigmoid(cls_logits, &[q_seq, NUM_CLASSES]);

    // Box regression head: Linear -> sigmoid
    let box_w = b.add_input("box_weight", &[4, d]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(dec_proj, box_w, Some(box_b), &[q_seq, 4]);
    let box_coords = b.add_sigmoid(box_logits, &[q_seq, 4]);

    // Concat cls + box
    let out = b.add_concat(&[cls_probs, box_coords], 1, &[q_seq, total_out]);

    b.build(out)
        .expect("valid Table Transformer full DETR pipeline kernel")
}

/// Bindings for full DETR pipeline.
fn full_detr_pipeline_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let q_seq = NUM_QUERIES;
    let seq = ENC_SEQ_LEN;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let dec_proj = ArrayD::from_elem(IxDyn(&[q_seq, seq]), WEIGHT_MAG);
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, d]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    let box_w = ArrayD::from_elem(IxDyn(&[4, d]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // backbone_features
        // Encoder
        TensorParamBinding::ConstantScalar(1e-5), // enc_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(attn_w), // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w), // enc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // enc_ln2_bias
        TensorParamBinding::ConstantTensor(ffn_up), // enc_ffn_up_weight
        TensorParamBinding::ConstantTensor(ffn_dn), // enc_ffn_down_weight
        // Decoder projection
        TensorParamBinding::ConstantTensor(dec_proj), // dec_proj_weight
        // Heads
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias
        TensorParamBinding::ConstantTensor(box_w), // box_weight
        TensorParamBinding::ConstantTensor(box_b), // box_bias
    ]
}

/// IBP bounds through full DETR pipeline: encoder -> decoder projection -> heads.
///
/// End-to-end test from backbone features through encoder, decoder projection,
/// and dual sigmoid heads. All outputs must be in [0, 1] (sigmoid).
#[test]
fn test_full_detr_pipeline_ibp() {
    let def = build_full_detr_pipeline_kernel();
    let bindings = full_detr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full DETR pipeline");

    let total_out = NUM_CLASSES + 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, total_out],
        "full DETR pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer full DETR pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 17. Box regression end-to-end: DFL -> sigmoid
// ===========================================================================

/// Build a box regression end-to-end: DFL softmax -> weighted sum -> sigmoid.
///
/// Input: `[NUM_QUERIES, DFL_BINS * 4]` (Variable, DFL logits for 4 box coords).
/// Output: `[NUM_QUERIES, 4]` (normalized box coordinates in [0, 1]).
///
/// For each of the 4 coordinates:
///   softmax(logits) -> weighted sum with bin indices -> collect
/// Then sigmoid to normalize to [0, 1].
fn build_dfl_to_sigmoid_kernel() -> TensorKernelDef {
    let q = NUM_QUERIES;
    let total_bins = DFL_BINS * 4;
    let mut b = TensorBlockBuilder::new("table_transformer_dfl_sigmoid");

    let input = b.add_input("dfl_logits", &[q, total_bins]);
    let bins = b.add_input("bins", &[DFL_BINS, 1]);

    // Process all 4 coordinates together: reshape -> softmax -> matmul -> reshape
    // Simplified: treat as [Q*4, DFL_BINS] -> softmax -> matmul -> [Q, 4]
    let reshaped = b.add_reshape(input, &[q * 4, DFL_BINS]);
    let probs = b.add_softmax(reshaped, 1, &[q * 4, DFL_BINS]);
    let weighted = b.add_matmul(probs, bins, false, None, &[q * 4, 1]);
    let coords_raw = b.add_reshape(weighted, &[q, 4]);

    // Sigmoid to normalize to [0, 1]
    let out = b.add_sigmoid(coords_raw, &[q, 4]);

    b.build(out)
        .expect("valid Table Transformer DFL -> sigmoid kernel")
}

/// Bindings for DFL -> sigmoid pipeline.
fn dfl_to_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bins = ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins shape");

    vec![
        TensorParamBinding::Variable,             // dfl_logits
        TensorParamBinding::ConstantTensor(bins), // bins
    ]
}

/// IBP bounds through DFL -> sigmoid end-to-end box regression.
///
/// Softmax produces distribution over bins, weighted sum gives raw coordinates,
/// sigmoid normalizes to [0, 1]. All output elements must be in [0, 1].
#[test]
fn test_dfl_to_sigmoid_ibp() {
    let def = build_dfl_to_sigmoid_kernel();
    let bindings = dfl_to_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, DFL_BINS * 4], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL -> sigmoid");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, 4],
        "DFL -> sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer DFL -> sigmoid IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 18. Transformer FFN: Linear -> ReLU -> Linear with CROWN
// ===========================================================================

/// Build a standalone transformer FFN block with residual.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture:
///   ffn_out = Linear_down(ReLU(Linear_up(LayerNorm(x))))
///   output = x + ffn_out  (residual)
fn build_transformer_ffn_residual_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_ffn_residual");

    let input = b.add_input("features", &seq_shape);

    // LayerNorm
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[d]);
    let ln_b = b.add_input("ln_bias", &[d]);

    // FFN weights
    let ffn_up_w = b.add_input("ffn_up_weight", &[FFN_DIM, d]);
    let ffn_dn_w = b.add_input("ffn_down_weight", &[d, FFN_DIM]);

    let x_norm = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &seq_shape);
    let ffn_h = b.add_linear(x_norm, ffn_up_w, None, &ffn_shape);
    let ffn_act = b.add_relu(ffn_h, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn_dn_w, None, &seq_shape);
    let out = b.add_binary_add(input, ffn_out, &seq_shape);

    b.build(out)
        .expect("valid Table Transformer FFN residual kernel")
}

/// Bindings for transformer FFN residual.
fn transformer_ffn_residual_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // features
        TensorParamBinding::ConstantScalar(1e-5),   // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias
        TensorParamBinding::ConstantTensor(ffn_up), // ffn_up_weight
        TensorParamBinding::ConstantTensor(ffn_dn), // ffn_down_weight
    ]
}

/// CROWN bounds through transformer FFN with residual.
///
/// ReLU is piecewise-linear (CROWN-friendly). Linear layers are affine.
/// LayerNorm requires IbpValidated mode. CROWN should produce tighter bounds
/// than IBP for the ReLU non-linearity.
#[test]
fn test_transformer_ffn_residual_crown() {
    let def = build_transformer_ffn_residual_kernel();
    let bindings = transformer_ffn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer FFN residual: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record transformer FFN residual.
#[test]
fn test_transformer_ffn_residual_verify_and_record() {
    let def = build_transformer_ffn_residual_kernel();
    let bindings = transformer_ffn_residual_bindings();
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "table_transformer_ffn_residual");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 19. Table detection + structure: detect heads -> combined output
// ===========================================================================

/// Build a table detection + structure recognition dual-head network.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable, decoder features).
/// Output: `[NUM_QUERIES, NUM_CLASSES + NUM_CLASSES + 4]` (detect_cls + struct_cls + box).
///
/// Architecture:
///   detect_logits = Linear_detect(features)    [Q, NUM_CLASSES]
///   detect_probs = sigmoid(detect_logits)       [Q, NUM_CLASSES]
///   struct_logits = Linear_struct(features)     [Q, NUM_CLASSES]
///   struct_probs = sigmoid(struct_logits)        [Q, NUM_CLASSES]
///   box_logits = Linear_box(features)           [Q, 4]
///   box_coords = sigmoid(box_logits)            [Q, 4]
///   output = concat(detect_probs, struct_probs, box_coords)
///
/// This models Table Transformer's dual task: table detection (is it a table?)
/// and structure recognition (rows, columns, cells, headers).
fn build_table_detect_structure_kernel() -> TensorKernelDef {
    let q = NUM_QUERIES;
    let d = HIDDEN_DIM;
    let nc = NUM_CLASSES;
    let total_out = nc + nc + 4;
    let mut b = TensorBlockBuilder::new("table_transformer_detect_structure");

    let input = b.add_input("decoder_features", &[q, d]);

    // Detection head
    let det_w = b.add_input("detect_weight", &[nc, d]);
    let det_b = b.add_input("detect_bias", &[nc]);
    let det_logits = b.add_linear(input, det_w, Some(det_b), &[q, nc]);
    let det_probs = b.add_sigmoid(det_logits, &[q, nc]);

    // Structure head
    let str_w = b.add_input("struct_weight", &[nc, d]);
    let str_b = b.add_input("struct_bias", &[nc]);
    let str_logits = b.add_linear(input, str_w, Some(str_b), &[q, nc]);
    let str_probs = b.add_sigmoid(str_logits, &[q, nc]);

    // Box head
    let box_w = b.add_input("box_weight", &[4, d]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &[q, 4]);
    let box_coords = b.add_sigmoid(box_logits, &[q, 4]);

    // Concat: detect + struct + box
    let out = b.add_concat(&[det_probs, str_probs, box_coords], 1, &[q, total_out]);

    b.build(out)
        .expect("valid Table Transformer detect+structure kernel")
}

/// Bindings for table detection + structure heads.
fn table_detect_structure_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let nc = NUM_CLASSES;
    let head_w = ArrayD::from_elem(IxDyn(&[nc, d]), WEIGHT_MAG);
    let head_b = ArrayD::from_elem(IxDyn(&[nc]), 0.0f32);
    let box_w = ArrayD::from_elem(IxDyn(&[4, d]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // decoder_features
        TensorParamBinding::ConstantTensor(head_w.clone()), // detect_weight
        TensorParamBinding::ConstantTensor(head_b.clone()), // detect_bias
        TensorParamBinding::ConstantTensor(head_w),         // struct_weight
        TensorParamBinding::ConstantTensor(head_b),         // struct_bias
        TensorParamBinding::ConstantTensor(box_w),          // box_weight
        TensorParamBinding::ConstantTensor(box_b),          // box_bias
    ]
}

/// IBP bounds through table detection + structure dual-head network.
///
/// All three heads use sigmoid, so all output elements must be in [0, 1].
#[test]
fn test_table_detect_structure_ibp() {
    let def = build_table_detect_structure_kernel();
    let bindings = table_detect_structure_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detect+structure heads");

    let nc = NUM_CLASSES;
    let total_out = nc + nc + 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, total_out],
        "detect+structure output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer detect+structure IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

/// CROWN bounds through table detection + structure dual-head network.
#[test]
fn test_table_detect_structure_crown() {
    let def = build_table_detect_structure_kernel();
    let bindings = table_detect_structure_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let nc = NUM_CLASSES;
    let total_out = nc + nc + 4;
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, total_out],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer detect+structure: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 20. Full pipeline: ResNet backbone -> encoder -> decoder proj -> heads
// ===========================================================================

/// Build the full Table Transformer pipeline from backbone features through
/// encoder, decoder projection, to classification + box heads.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable, image features).
/// Output: `[NUM_QUERIES, NUM_CLASSES + 4]` (class probs + box coords).
///
/// Architecture:
///   backbone_out = Conv2d(stride=2) -> BN -> ReLU  (downsample)
///   flat = reshape to [CHANNELS * (FEAT_SIZE/2)^2]  (flatten spatial)
///   seq = Linear projection to [ENC_SEQ_LEN, HIDDEN_DIM]  (embed)
///   enc_out = EncoderLayer(seq)  (self-attention + FFN)
///   dec_proj = matmul(proj_w, enc_out)  (project to query space)
///   cls = sigmoid(Linear(dec_proj))
///   box = sigmoid(Linear(dec_proj))
///   output = concat(cls, box)
fn build_full_pipeline_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s_in = FEAT_SIZE;
    let s_out = FEAT_SIZE / 2;
    let flat_dim = c * s_out * s_out; // 64 * 4 * 4 = 1024
    let seq = ENC_SEQ_LEN; // 64
    let d = HIDDEN_DIM; // 64
    let q = NUM_QUERIES;
    let total_out = NUM_CLASSES + 4;

    let mut b = TensorBlockBuilder::new("table_transformer_full_pipeline");

    let input = b.add_input("image_features", &[c, s_in, s_in]);

    // Backbone: Conv2d(stride=2) -> BN -> ReLU
    let bb_conv_w = b.add_input("bb_conv_weight", &[c, c, 3, 3]);
    let bb_conv_b = b.add_input("bb_conv_bias", &[c]);
    let bb_bn_mean = b.add_input("bb_bn_mean", &[c]);
    let bb_bn_var = b.add_input("bb_bn_var", &[c]);
    let bb_bn_w = b.add_input("bb_bn_weight", &[c]);
    let bb_bn_b = b.add_input("bb_bn_bias", &[c]);
    let bb_bn_eps = b.add_input("bb_bn_eps", &[1]);

    let conv_out = b.add_conv2d(
        input,
        bb_conv_w,
        Some(bb_conv_b),
        2,
        2,
        1,
        1,
        &[c, s_out, s_out],
    );
    let bn_out = b.add_batch_norm(
        conv_out,
        bb_bn_mean,
        bb_bn_var,
        bb_bn_w,
        bb_bn_b,
        bb_bn_eps,
        &[c, s_out, s_out],
    );
    let relu_out = b.add_relu(bn_out, &[c, s_out, s_out]);

    // Flatten: [C, H/2, W/2] -> [flat_dim]
    let flat = b.add_reshape(relu_out, &[flat_dim]);

    // Project: [flat_dim] -> [seq, d] via Linear
    let proj_w = b.add_input("proj_weight", &[seq * d, flat_dim]);
    let proj_out = b.add_linear(flat, proj_w, None, &[seq * d]);
    let seq_features = b.add_reshape(proj_out, &[seq, d]);

    // Encoder layer: LN -> Self-Attn -> Residual -> LN -> FFN -> Residual
    let enc_ln1_eps = b.add_input("enc_ln1_eps", &[1]);
    let enc_ln1_w = b.add_input("enc_ln1_weight", &[d]);
    let enc_ln1_b = b.add_input("enc_ln1_bias", &[d]);
    let enc_q_w = b.add_input("enc_q_weight", &[d, d]);
    let enc_k_w = b.add_input("enc_k_weight", &[d, d]);
    let enc_v_w = b.add_input("enc_v_weight", &[d, d]);
    let enc_o_w = b.add_input("enc_out_weight", &[d, d]);
    let enc_ln2_eps = b.add_input("enc_ln2_eps", &[1]);
    let enc_ln2_w = b.add_input("enc_ln2_weight", &[d]);
    let enc_ln2_b = b.add_input("enc_ln2_bias", &[d]);
    let enc_ffn_up = b.add_input("enc_ffn_up_weight", &[FFN_DIM, d]);
    let enc_ffn_dn = b.add_input("enc_ffn_down_weight", &[d, FFN_DIM]);

    let x_norm = b.add_layer_norm(
        seq_features,
        enc_ln1_eps,
        1,
        enc_ln1_w,
        enc_ln1_b,
        &[seq, d],
    );
    let attn_out = b
        .add_multi_head_attention(
            x_norm,
            enc_q_w,
            enc_k_w,
            enc_v_w,
            enc_o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[seq, d],
        )
        .expect("valid encoder self-attention");
    let res1 = b.add_binary_add(seq_features, attn_out, &[seq, d]);
    let x_norm2 = b.add_layer_norm(res1, enc_ln2_eps, 1, enc_ln2_w, enc_ln2_b, &[seq, d]);
    let ffn_h = b.add_linear(x_norm2, enc_ffn_up, None, &[seq, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_h, &[seq, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, enc_ffn_dn, None, &[seq, d]);
    let enc_out = b.add_binary_add(res1, ffn_out, &[seq, d]);

    // Decoder projection: [Q, seq] x [seq, d] -> [Q, d]
    let dec_proj_w = b.add_input("dec_proj_weight", &[q, seq]);
    let dec_proj = b.add_matmul(dec_proj_w, enc_out, false, None, &[q, d]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, d]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(dec_proj, cls_w, Some(cls_b), &[q, NUM_CLASSES]);
    let cls_probs = b.add_sigmoid(cls_logits, &[q, NUM_CLASSES]);

    // Box head
    let box_w = b.add_input("box_weight", &[4, d]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(dec_proj, box_w, Some(box_b), &[q, 4]);
    let box_coords = b.add_sigmoid(box_logits, &[q, 4]);

    // Concat
    let out = b.add_concat(&[cls_probs, box_coords], 1, &[q, total_out]);

    b.build(out)
        .expect("valid Table Transformer full pipeline kernel")
}

/// Bindings for full pipeline.
fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let s_out = FEAT_SIZE / 2;
    let flat_dim = c * s_out * s_out;
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q = NUM_QUERIES;

    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_w = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[seq * d, flat_dim]), WEIGHT_MAG * 0.01);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let dec_proj = ArrayD::from_elem(IxDyn(&[q, seq]), WEIGHT_MAG);
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, d]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    let box_w = ArrayD::from_elem(IxDyn(&[4, d]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // image_features
        // Backbone
        TensorParamBinding::ConstantTensor(conv_w), // bb_conv_weight
        TensorParamBinding::ConstantTensor(conv_b), // bb_conv_bias
        TensorParamBinding::ConstantTensor(bn_mean), // bb_bn_mean
        TensorParamBinding::ConstantTensor(bn_var), // bb_bn_var
        TensorParamBinding::ConstantTensor(bn_w),   // bb_bn_weight
        TensorParamBinding::ConstantTensor(bn_b),   // bb_bn_bias
        TensorParamBinding::ConstantScalar(1e-5),   // bb_bn_eps
        // Projection
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        // Encoder
        TensorParamBinding::ConstantScalar(1e-5), // enc_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(attn_w), // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w), // enc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // enc_ln2_bias
        TensorParamBinding::ConstantTensor(ffn_up), // enc_ffn_up_weight
        TensorParamBinding::ConstantTensor(ffn_dn), // enc_ffn_down_weight
        // Decoder projection
        TensorParamBinding::ConstantTensor(dec_proj), // dec_proj_weight
        // Heads
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias
        TensorParamBinding::ConstantTensor(box_w), // box_weight
        TensorParamBinding::ConstantTensor(box_b), // box_bias
    ]
}

/// IBP bounds through full Table Transformer pipeline.
///
/// End-to-end: ResNet backbone -> flatten -> project -> encoder -> decoder -> heads.
/// All final outputs go through sigmoid, so must be in [0, 1].
#[test]
fn test_full_pipeline_ibp() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full pipeline");

    let total_out = NUM_CLASSES + 4;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, total_out],
        "full pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 21. ResNet-18 full 4-stage backbone: BasicBlock through all stages (IBP)
// ===========================================================================

/// Build a full ResNet-18 style backbone with 4 cascading spatial downsamples.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable, e.g. [64, 8, 8]).
/// Output: `[CHANNELS, 1, 1]` (global spatial collapse after 3 halvings).
///
/// Stage 1: BasicBlock (conv->BN->ReLU->conv->BN + skip + ReLU) at same resolution
/// Stage 2: Conv2d(stride=2) -> BN -> ReLU  (8x8 -> 4x4)
/// Stage 3: Conv2d(stride=2) -> BN -> ReLU  (4x4 -> 2x2)
/// Stage 4: Conv2d(stride=2) -> BN -> ReLU  (2x2 -> 1x1)
fn build_resnet18_4stage_backbone_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s1 = FEAT_SIZE; // 8
    let s2 = FEAT_SIZE / 2; // 4
    let s3 = FEAT_SIZE / 4; // 2
    let s4 = FEAT_SIZE / 8; // 1
    let mut b = TensorBlockBuilder::new("table_transformer_resnet18_4stage");

    let input = b.add_input("features", &[c, s1, s1]);

    // Stage 1: BasicBlock at same resolution
    let bb_c1w = b.add_input("s1_bb_conv1_w", &[c, c, 3, 3]);
    let bb_c1b = b.add_input("s1_bb_conv1_b", &[c]);
    let bb_bn1m = b.add_input("s1_bb_bn1_mean", &[c]);
    let bb_bn1v = b.add_input("s1_bb_bn1_var", &[c]);
    let bb_bn1w = b.add_input("s1_bb_bn1_weight", &[c]);
    let bb_bn1b = b.add_input("s1_bb_bn1_bias", &[c]);
    let bb_bn1e = b.add_input("s1_bb_bn1_eps", &[1]);
    let bb_c2w = b.add_input("s1_bb_conv2_w", &[c, c, 3, 3]);
    let bb_c2b = b.add_input("s1_bb_conv2_b", &[c]);
    let bb_bn2m = b.add_input("s1_bb_bn2_mean", &[c]);
    let bb_bn2v = b.add_input("s1_bb_bn2_var", &[c]);
    let bb_bn2w = b.add_input("s1_bb_bn2_weight", &[c]);
    let bb_bn2b = b.add_input("s1_bb_bn2_bias", &[c]);
    let bb_bn2e = b.add_input("s1_bb_bn2_eps", &[1]);

    let bb_conv1 = b.add_conv2d(input, bb_c1w, Some(bb_c1b), 1, 1, 1, 1, &[c, s1, s1]);
    let bb_bn1 = b.add_batch_norm(
        bb_conv1,
        bb_bn1m,
        bb_bn1v,
        bb_bn1w,
        bb_bn1b,
        bb_bn1e,
        &[c, s1, s1],
    );
    let bb_relu1 = b.add_relu(bb_bn1, &[c, s1, s1]);
    let bb_conv2 = b.add_conv2d(bb_relu1, bb_c2w, Some(bb_c2b), 1, 1, 1, 1, &[c, s1, s1]);
    let bb_bn2 = b.add_batch_norm(
        bb_conv2,
        bb_bn2m,
        bb_bn2v,
        bb_bn2w,
        bb_bn2b,
        bb_bn2e,
        &[c, s1, s1],
    );
    let bb_skip = b.add_binary_add(bb_bn2, input, &[c, s1, s1]);
    let stage1_out = b.add_relu(bb_skip, &[c, s1, s1]);

    // Stage 2: [C, 8, 8] -> [C, 4, 4]
    let s2_cw = b.add_input("s2_conv_w", &[c, c, 3, 3]);
    let s2_cb = b.add_input("s2_conv_b", &[c]);
    let s2_bm = b.add_input("s2_bn_mean", &[c]);
    let s2_bv = b.add_input("s2_bn_var", &[c]);
    let s2_bw = b.add_input("s2_bn_weight", &[c]);
    let s2_bb = b.add_input("s2_bn_bias", &[c]);
    let s2_be = b.add_input("s2_bn_eps", &[1]);
    let s2_conv = b.add_conv2d(stage1_out, s2_cw, Some(s2_cb), 2, 2, 1, 1, &[c, s2, s2]);
    let s2_bn = b.add_batch_norm(s2_conv, s2_bm, s2_bv, s2_bw, s2_bb, s2_be, &[c, s2, s2]);
    let stage2_out = b.add_relu(s2_bn, &[c, s2, s2]);

    // Stage 3: [C, 4, 4] -> [C, 2, 2]
    let s3_cw = b.add_input("s3_conv_w", &[c, c, 3, 3]);
    let s3_cb = b.add_input("s3_conv_b", &[c]);
    let s3_bm = b.add_input("s3_bn_mean", &[c]);
    let s3_bv = b.add_input("s3_bn_var", &[c]);
    let s3_bw = b.add_input("s3_bn_weight", &[c]);
    let s3_bb = b.add_input("s3_bn_bias", &[c]);
    let s3_be = b.add_input("s3_bn_eps", &[1]);
    let s3_conv = b.add_conv2d(stage2_out, s3_cw, Some(s3_cb), 2, 2, 1, 1, &[c, s3, s3]);
    let s3_bn = b.add_batch_norm(s3_conv, s3_bm, s3_bv, s3_bw, s3_bb, s3_be, &[c, s3, s3]);
    let stage3_out = b.add_relu(s3_bn, &[c, s3, s3]);

    // Stage 4: [C, 2, 2] -> [C, 1, 1]
    let s4_cw = b.add_input("s4_conv_w", &[c, c, 3, 3]);
    let s4_cb = b.add_input("s4_conv_b", &[c]);
    let s4_bm = b.add_input("s4_bn_mean", &[c]);
    let s4_bv = b.add_input("s4_bn_var", &[c]);
    let s4_bw = b.add_input("s4_bn_weight", &[c]);
    let s4_bb = b.add_input("s4_bn_bias", &[c]);
    let s4_be = b.add_input("s4_bn_eps", &[1]);
    let s4_conv = b.add_conv2d(stage3_out, s4_cw, Some(s4_cb), 2, 2, 1, 1, &[c, s4, s4]);
    let s4_bn = b.add_batch_norm(s4_conv, s4_bm, s4_bv, s4_bw, s4_bb, s4_be, &[c, s4, s4]);
    let out = b.add_relu(s4_bn, &[c, s4, s4]);

    b.build(out)
        .expect("valid Table Transformer ResNet-18 4-stage backbone kernel")
}

/// Bindings for ResNet-18 4-stage backbone.
fn resnet18_4stage_backbone_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_w = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    let one_conv_bn = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantTensor(conv_w.clone()),
            TensorParamBinding::ConstantTensor(conv_b.clone()),
            TensorParamBinding::ConstantTensor(bn_mean.clone()),
            TensorParamBinding::ConstantTensor(bn_var.clone()),
            TensorParamBinding::ConstantTensor(bn_w.clone()),
            TensorParamBinding::ConstantTensor(bn_b.clone()),
            TensorParamBinding::ConstantScalar(1e-5),
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable]; // features
                                                           // BasicBlock (2 conv+BN pairs)
    bindings.extend(one_conv_bn());
    bindings.extend(one_conv_bn());
    // 3 downsample stages
    bindings.extend(one_conv_bn());
    bindings.extend(one_conv_bn());
    bindings.extend(one_conv_bn());
    bindings
}

/// IBP bounds through full ResNet-18 4-stage backbone.
///
/// 4 stages: BasicBlock + 3 stride-2 downsamples (8x8 -> 4x4 -> 2x2 -> 1x1).
/// Every stage has BN + ReLU. ReLU clamps lower to 0.
#[test]
fn test_resnet18_4stage_backbone_ibp() {
    let def = build_resnet18_4stage_backbone_kernel();
    let bindings = resnet18_4stage_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-stage backbone");

    let s_out = FEAT_SIZE / 8; // 1
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, s_out, s_out],
        "4-stage backbone output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer ResNet-18 4-stage backbone IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "ReLU lower must be >= 0, got {lo_min}");
}

/// CROWN bounds through ResNet-18 4-stage backbone.
#[test]
fn test_resnet18_4stage_backbone_crown() {
    let def = build_resnet18_4stage_backbone_kernel();
    let bindings = resnet18_4stage_backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let s_out = FEAT_SIZE / 8;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, s_out, s_out],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer ResNet-18 4-stage backbone: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 22. DETR encoder 4-layer stack (IBP + CROWN): realistic depth
// ===========================================================================

/// Build a 4-layer DETR encoder stack for realistic depth verification.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Four stacked encoder layers, each: LN -> Self-Attn -> Residual -> LN -> FFN -> Residual.
fn build_encoder_4layer_stack_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_encoder_4layer");

    let mut prev = b.add_input("encoder_features", &seq_shape);

    for layer_idx in 1..=4 {
        let p = format!("l{layer_idx}");
        let ln1_eps = b.add_input(&format!("{p}_ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{p}_ln1_w"), &[d]);
        let ln1_b = b.add_input(&format!("{p}_ln1_b"), &[d]);
        let q_w = b.add_input(&format!("{p}_q_w"), &[d, d]);
        let k_w = b.add_input(&format!("{p}_k_w"), &[d, d]);
        let v_w = b.add_input(&format!("{p}_v_w"), &[d, d]);
        let o_w = b.add_input(&format!("{p}_o_w"), &[d, d]);
        let ln2_eps = b.add_input(&format!("{p}_ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{p}_ln2_w"), &[d]);
        let ln2_b = b.add_input(&format!("{p}_ln2_b"), &[d]);
        let ffn_up = b.add_input(&format!("{p}_ffn_up"), &[FFN_DIM, d]);
        let ffn_dn = b.add_input(&format!("{p}_ffn_dn"), &[d, FFN_DIM]);

        let x_norm = b.add_layer_norm(prev, ln1_eps, 1, ln1_w, ln1_b, &seq_shape);
        let attn = b
            .add_multi_head_attention(
                x_norm,
                q_w,
                k_w,
                v_w,
                o_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &seq_shape,
            )
            .expect("valid encoder self-attention");
        let res1 = b.add_binary_add(prev, attn, &seq_shape);
        let x_norm2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &seq_shape);
        let ffn_h = b.add_linear(x_norm2, ffn_up, None, &ffn_shape);
        let ffn_act = b.add_relu(ffn_h, &ffn_shape);
        let ffn_out = b.add_linear(ffn_act, ffn_dn, None, &seq_shape);
        prev = b.add_binary_add(res1, ffn_out, &seq_shape);
    }

    b.build(prev)
        .expect("valid Table Transformer 4-layer encoder stack kernel")
}

/// Bindings for 4-layer encoder stack.
fn encoder_4layer_stack_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    let one_layer = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(ffn_up.clone()),
            TensorParamBinding::ConstantTensor(ffn_dn.clone()),
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        bindings.extend(one_layer());
    }
    bindings
}

/// IBP bounds through 4-layer DETR encoder stack.
#[test]
fn test_encoder_4layer_stack_ibp() {
    let def = build_encoder_4layer_stack_kernel();
    let bindings = encoder_4layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-layer encoder stack");

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer encoder 4-layer stack IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through 4-layer DETR encoder stack.
#[test]
fn test_encoder_4layer_stack_crown() {
    let def = build_encoder_4layer_stack_kernel();
    let bindings = encoder_4layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer encoder 4-layer stack: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 23. DETR decoder 4-layer stack: self-attn + cross-attn + FFN x4
// ===========================================================================

/// Build a 4-layer DETR decoder stack.
///
/// Query input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable).
/// Encoder memory: `[ENC_SEQ_LEN, HIDDEN_DIM]` (constant).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
fn build_decoder_4layer_stack_kernel() -> TensorKernelDef {
    let q_seq = NUM_QUERIES;
    let kv_seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q_shape = [q_seq, d];
    let ffn_shape = [q_seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_decoder_4layer");

    let mut prev = b.add_input("object_queries", &q_shape);
    let memory = b.add_input("encoder_memory", &[kv_seq, d]);

    for layer_idx in 1..=4 {
        let p = format!("d{layer_idx}");
        // Self-attention
        let ln1_e = b.add_input(&format!("{p}_ln1_e"), &[1]);
        let ln1_w = b.add_input(&format!("{p}_ln1_w"), &[d]);
        let ln1_b = b.add_input(&format!("{p}_ln1_b"), &[d]);
        let sq_w = b.add_input(&format!("{p}_sq_w"), &[d, d]);
        let sk_w = b.add_input(&format!("{p}_sk_w"), &[d, d]);
        let sv_w = b.add_input(&format!("{p}_sv_w"), &[d, d]);
        let so_w = b.add_input(&format!("{p}_so_w"), &[d, d]);

        let n1 = b.add_layer_norm(prev, ln1_e, 1, ln1_w, ln1_b, &q_shape);
        let sa = b
            .add_multi_head_attention(
                n1,
                sq_w,
                sk_w,
                sv_w,
                so_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &q_shape,
            )
            .expect("decoder self-attn");
        let r1 = b.add_binary_add(prev, sa, &q_shape);

        // Cross-attention
        let ln2_e = b.add_input(&format!("{p}_ln2_e"), &[1]);
        let ln2_w = b.add_input(&format!("{p}_ln2_w"), &[d]);
        let ln2_b = b.add_input(&format!("{p}_ln2_b"), &[d]);
        let cq_w = b.add_input(&format!("{p}_cq_w"), &[d, d]);
        let ck_w = b.add_input(&format!("{p}_ck_w"), &[d, d]);
        let cv_w = b.add_input(&format!("{p}_cv_w"), &[d, d]);
        let co_w = b.add_input(&format!("{p}_co_w"), &[d, d]);

        let n2 = b.add_layer_norm(r1, ln2_e, 1, ln2_w, ln2_b, &q_shape);
        let ca = b
            .add_multi_head_cross_attention(
                n2,
                memory,
                cq_w,
                ck_w,
                cv_w,
                co_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &q_shape,
            )
            .expect("decoder cross-attn");
        let r2 = b.add_binary_add(r1, ca, &q_shape);

        // FFN
        let ln3_e = b.add_input(&format!("{p}_ln3_e"), &[1]);
        let ln3_w = b.add_input(&format!("{p}_ln3_w"), &[d]);
        let ln3_b = b.add_input(&format!("{p}_ln3_b"), &[d]);
        let fu = b.add_input(&format!("{p}_fu"), &[FFN_DIM, d]);
        let fd = b.add_input(&format!("{p}_fd"), &[d, FFN_DIM]);

        let n3 = b.add_layer_norm(r2, ln3_e, 1, ln3_w, ln3_b, &q_shape);
        let fh = b.add_linear(n3, fu, None, &ffn_shape);
        let fa = b.add_relu(fh, &ffn_shape);
        let fo = b.add_linear(fa, fd, None, &q_shape);
        prev = b.add_binary_add(r2, fo, &q_shape);
    }

    b.build(prev)
        .expect("valid Table Transformer 4-layer decoder stack kernel")
}

/// Bindings for 4-layer decoder stack.
fn decoder_4layer_stack_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let kv_seq = ENC_SEQ_LEN;
    let memory = ArrayD::from_elem(IxDyn(&[kv_seq, d]), 0.1f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    let one_layer = || -> Vec<TensorParamBinding> {
        vec![
            // Self-attention LN + weights
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            // Cross-attention LN + weights
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            // FFN LN + weights
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(ffn_up.clone()),
            TensorParamBinding::ConstantTensor(ffn_dn.clone()),
        ]
    };

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(memory),
    ];
    for _ in 0..4 {
        bindings.extend(one_layer());
    }
    bindings
}

/// IBP bounds through 4-layer DETR decoder stack.
#[test]
fn test_decoder_4layer_stack_ibp() {
    let def = build_decoder_4layer_stack_kernel();
    let bindings = decoder_4layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-layer decoder stack");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer decoder 4-layer stack IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through 4-layer DETR decoder stack.
#[test]
fn test_decoder_4layer_stack_crown() {
    let def = build_decoder_4layer_stack_kernel();
    let bindings = decoder_4layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer decoder 4-layer stack: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 24. Encoder-decoder composition: encoder output fed as memory to decoder
// ===========================================================================

/// Build encoder-decoder composition: encoder -> projection -> decoder FFN.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable, backbone features).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
fn build_encoder_decoder_composition_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let q_seq = NUM_QUERIES;
    let seq_shape = [seq, d];
    let ffn_enc = [seq, FFN_DIM];
    let q_shape = [q_seq, d];
    let ffn_dec = [q_seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_enc_dec_compose");

    let input = b.add_input("backbone_features", &seq_shape);

    // Encoder layer
    let e_ln1e = b.add_input("e_ln1_eps", &[1]);
    let e_ln1w = b.add_input("e_ln1_w", &[d]);
    let e_ln1b = b.add_input("e_ln1_b", &[d]);
    let e_qw = b.add_input("e_q_w", &[d, d]);
    let e_kw = b.add_input("e_k_w", &[d, d]);
    let e_vw = b.add_input("e_v_w", &[d, d]);
    let e_ow = b.add_input("e_o_w", &[d, d]);
    let e_ln2e = b.add_input("e_ln2_eps", &[1]);
    let e_ln2w = b.add_input("e_ln2_w", &[d]);
    let e_ln2b = b.add_input("e_ln2_b", &[d]);
    let e_fu = b.add_input("e_ffn_up", &[FFN_DIM, d]);
    let e_fd = b.add_input("e_ffn_dn", &[d, FFN_DIM]);

    let n1 = b.add_layer_norm(input, e_ln1e, 1, e_ln1w, e_ln1b, &seq_shape);
    let a1 = b
        .add_multi_head_attention(
            n1,
            e_qw,
            e_kw,
            e_vw,
            e_ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("enc self-attn");
    let r1 = b.add_binary_add(input, a1, &seq_shape);
    let n2 = b.add_layer_norm(r1, e_ln2e, 1, e_ln2w, e_ln2b, &seq_shape);
    let fh = b.add_linear(n2, e_fu, None, &ffn_enc);
    let fa = b.add_relu(fh, &ffn_enc);
    let fo = b.add_linear(fa, e_fd, None, &seq_shape);
    let enc_out = b.add_binary_add(r1, fo, &seq_shape);

    // Project to query space
    let proj_w = b.add_input("dec_proj_w", &[q_seq, seq]);
    let dec_features = b.add_matmul(proj_w, enc_out, false, None, &q_shape);

    // Decoder FFN layer
    let d_lne = b.add_input("d_ln_eps", &[1]);
    let d_lnw = b.add_input("d_ln_w", &[d]);
    let d_lnb = b.add_input("d_ln_b", &[d]);
    let d_fu = b.add_input("d_ffn_up", &[FFN_DIM, d]);
    let d_fd = b.add_input("d_ffn_dn", &[d, FFN_DIM]);

    let dn = b.add_layer_norm(dec_features, d_lne, 1, d_lnw, d_lnb, &q_shape);
    let dfh = b.add_linear(dn, d_fu, None, &ffn_dec);
    let dfa = b.add_relu(dfh, &ffn_dec);
    let dfo = b.add_linear(dfa, d_fd, None, &q_shape);
    let out = b.add_binary_add(dec_features, dfo, &q_shape);

    b.build(out)
        .expect("valid encoder-decoder composition kernel")
}

/// Bindings for encoder-decoder composition.
fn encoder_decoder_composition_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let seq = ENC_SEQ_LEN;
    let q_seq = NUM_QUERIES;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // backbone_features
        // Encoder
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(ffn_up.clone()),
        TensorParamBinding::ConstantTensor(ffn_dn.clone()),
        // Decoder projection
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[q_seq, seq]), WEIGHT_MAG)),
        // Decoder FFN
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantTensor(ffn_up),
        TensorParamBinding::ConstantTensor(ffn_dn),
    ]
}

/// IBP bounds through encoder-decoder composition.
#[test]
fn test_encoder_decoder_composition_ibp() {
    let def = build_encoder_decoder_composition_kernel();
    let bindings = encoder_decoder_composition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder-decoder composition");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer encoder-decoder composition IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through encoder-decoder composition.
#[test]
fn test_encoder_decoder_composition_crown() {
    let def = build_encoder_decoder_composition_kernel();
    let bindings = encoder_decoder_composition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer encoder-decoder: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 25. Object query learning: queries through refinement layers
// ===========================================================================

/// Build a query refinement pipeline: 2 MLP layers with residuals.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
fn build_object_query_learning_kernel() -> TensorKernelDef {
    let q = NUM_QUERIES;
    let d = HIDDEN_DIM;
    let q_shape = [q, d];
    let ffn_shape = [q, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_query_learning");

    let input = b.add_input("object_queries", &q_shape);

    // Refinement layer 1
    let ln1_e = b.add_input("r1_ln_eps", &[1]);
    let ln1_w = b.add_input("r1_ln_w", &[d]);
    let ln1_b = b.add_input("r1_ln_b", &[d]);
    let f1_u = b.add_input("r1_ffn_up", &[FFN_DIM, d]);
    let f1_d = b.add_input("r1_ffn_dn", &[d, FFN_DIM]);

    let n1 = b.add_layer_norm(input, ln1_e, 1, ln1_w, ln1_b, &q_shape);
    let h1 = b.add_linear(n1, f1_u, None, &ffn_shape);
    let a1 = b.add_relu(h1, &ffn_shape);
    let o1 = b.add_linear(a1, f1_d, None, &q_shape);
    let r1 = b.add_binary_add(input, o1, &q_shape);

    // Refinement layer 2
    let ln2_e = b.add_input("r2_ln_eps", &[1]);
    let ln2_w = b.add_input("r2_ln_w", &[d]);
    let ln2_b = b.add_input("r2_ln_b", &[d]);
    let f2_u = b.add_input("r2_ffn_up", &[FFN_DIM, d]);
    let f2_d = b.add_input("r2_ffn_dn", &[d, FFN_DIM]);

    let n2 = b.add_layer_norm(r1, ln2_e, 1, ln2_w, ln2_b, &q_shape);
    let h2 = b.add_linear(n2, f2_u, None, &ffn_shape);
    let a2 = b.add_relu(h2, &ffn_shape);
    let o2 = b.add_linear(a2, f2_d, None, &q_shape);
    let out = b.add_binary_add(r1, o2, &q_shape);

    b.build(out).expect("valid object query learning kernel")
}

/// Bindings for object query learning.
fn object_query_learning_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    let one_layer = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(ffn_up.clone()),
            TensorParamBinding::ConstantTensor(ffn_dn.clone()),
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(one_layer());
    bindings.extend(one_layer());
    bindings
}

/// IBP bounds through object query learning.
#[test]
fn test_object_query_learning_ibp() {
    let def = build_object_query_learning_kernel();
    let bindings = object_query_learning_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through query learning");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer object query learning IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through object query learning.
#[test]
fn test_object_query_learning_crown() {
    let def = build_object_query_learning_kernel();
    let bindings = object_query_learning_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer query learning: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 26. Multi-head detection: parallel cls + box + structure from shared features
// ===========================================================================

/// Build multi-head detection: shared LN -> 3 parallel sigmoid heads.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_QUERIES, NUM_CLASSES + 4 + NUM_CLASSES]`.
fn build_multi_head_detection_kernel() -> TensorKernelDef {
    let q = NUM_QUERIES;
    let d = HIDDEN_DIM;
    let nc = NUM_CLASSES;
    let total_out = nc + 4 + nc;
    let mut b = TensorBlockBuilder::new("table_transformer_multi_head_detect");

    let input = b.add_input("decoder_features", &[q, d]);

    // Shared LayerNorm
    let ln_e = b.add_input("shared_ln_eps", &[1]);
    let ln_w = b.add_input("shared_ln_w", &[d]);
    let ln_b = b.add_input("shared_ln_b", &[d]);
    let normed = b.add_layer_norm(input, ln_e, 1, ln_w, ln_b, &[q, d]);

    // Classification head
    let cls_w = b.add_input("cls_w", &[nc, d]);
    let cls_b = b.add_input("cls_b", &[nc]);
    let cls_logits = b.add_linear(normed, cls_w, Some(cls_b), &[q, nc]);
    let cls_probs = b.add_sigmoid(cls_logits, &[q, nc]);

    // Box regression head
    let box_w = b.add_input("box_w", &[4, d]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(normed, box_w, Some(box_b), &[q, 4]);
    let box_coords = b.add_sigmoid(box_logits, &[q, 4]);

    // Structure recognition head
    let str_w = b.add_input("str_w", &[nc, d]);
    let str_b = b.add_input("str_b", &[nc]);
    let str_logits = b.add_linear(normed, str_w, Some(str_b), &[q, nc]);
    let str_probs = b.add_sigmoid(str_logits, &[q, nc]);

    let out = b.add_concat(&[cls_probs, box_coords, str_probs], 1, &[q, total_out]);

    b.build(out).expect("valid multi-head detection kernel")
}

/// Bindings for multi-head detection.
fn multi_head_detection_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let nc = NUM_CLASSES;

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[nc, d]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[nc]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, d]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[nc, d]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[nc]), 0.0f32)),
    ]
}

/// IBP bounds through multi-head detection. All outputs in [0, 1] via sigmoid.
#[test]
fn test_multi_head_detection_ibp() {
    let def = build_multi_head_detection_kernel();
    let bindings = multi_head_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-head detection");

    let total_out = NUM_CLASSES + 4 + NUM_CLASSES;
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, total_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer multi-head detection IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

/// CROWN bounds through multi-head detection.
#[test]
fn test_multi_head_detection_crown() {
    let def = build_multi_head_detection_kernel();
    let bindings = multi_head_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let total_out = NUM_CLASSES + 4 + NUM_CLASSES;
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, total_out]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer multi-head detection: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 27. Position encoding propagation through 2-layer encoder
// ===========================================================================

/// Build PE + 2-layer encoder: features + sinusoidal PE -> 2 encoder layers.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
fn build_pe_propagation_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_pe_propagation");

    let input = b.add_input("features", &seq_shape);
    let pe = b.add_input("positional_encoding", &seq_shape);
    let x_pe = b.add_binary_add(input, pe, &seq_shape);

    let mut prev = x_pe;
    for layer_idx in 1..=2 {
        let p = format!("l{layer_idx}");
        let ln1_e = b.add_input(&format!("{p}_ln1_e"), &[1]);
        let ln1_w = b.add_input(&format!("{p}_ln1_w"), &[d]);
        let ln1_b = b.add_input(&format!("{p}_ln1_b"), &[d]);
        let q_w = b.add_input(&format!("{p}_q_w"), &[d, d]);
        let k_w = b.add_input(&format!("{p}_k_w"), &[d, d]);
        let v_w = b.add_input(&format!("{p}_v_w"), &[d, d]);
        let o_w = b.add_input(&format!("{p}_o_w"), &[d, d]);
        let ln2_e = b.add_input(&format!("{p}_ln2_e"), &[1]);
        let ln2_w = b.add_input(&format!("{p}_ln2_w"), &[d]);
        let ln2_b = b.add_input(&format!("{p}_ln2_b"), &[d]);
        let fu = b.add_input(&format!("{p}_fu"), &[FFN_DIM, d]);
        let fd = b.add_input(&format!("{p}_fd"), &[d, FFN_DIM]);

        let n1 = b.add_layer_norm(prev, ln1_e, 1, ln1_w, ln1_b, &seq_shape);
        let attn = b
            .add_multi_head_attention(
                n1,
                q_w,
                k_w,
                v_w,
                o_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &seq_shape,
            )
            .expect("self-attn");
        let r1 = b.add_binary_add(prev, attn, &seq_shape);
        let n2 = b.add_layer_norm(r1, ln2_e, 1, ln2_w, ln2_b, &seq_shape);
        let fh = b.add_linear(n2, fu, None, &ffn_shape);
        let fa = b.add_relu(fh, &ffn_shape);
        let fo = b.add_linear(fa, fd, None, &seq_shape);
        prev = b.add_binary_add(r1, fo, &seq_shape);
    }

    b.build(prev).expect("valid PE propagation kernel")
}

/// Bindings for PE propagation.
fn pe_propagation_bindings() -> Vec<TensorParamBinding> {
    let pe = sinusoidal_pe_tensor();
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let ffn_dn = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    let one_layer = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantTensor(attn_w.clone()),
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ln_w.clone()),
            TensorParamBinding::ConstantTensor(ln_b.clone()),
            TensorParamBinding::ConstantTensor(ffn_up.clone()),
            TensorParamBinding::ConstantTensor(ffn_dn.clone()),
        ]
    };

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    bindings.extend(one_layer());
    bindings.extend(one_layer());
    bindings
}

/// IBP bounds through PE propagation.
#[test]
fn test_pe_propagation_ibp() {
    let def = build_pe_propagation_kernel();
    let bindings = pe_propagation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PE propagation");

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer PE propagation IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through PE propagation.
#[test]
fn test_pe_propagation_crown() {
    let def = build_pe_propagation_kernel();
    let bindings = pe_propagation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer PE propagation: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 28. Backbone-to-transformer transition: ResNet -> flatten -> project -> LN
// ===========================================================================

/// Build backbone-to-transformer transition.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
fn build_backbone_to_transformer_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s_in = FEAT_SIZE;
    let s_out = FEAT_SIZE / 2;
    let flat_dim = c * s_out * s_out;
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let mut b = TensorBlockBuilder::new("table_transformer_backbone_to_xfmr");

    let input = b.add_input("image_features", &[c, s_in, s_in]);

    // Backbone downsample
    let cw = b.add_input("bb_conv_w", &[c, c, 3, 3]);
    let cb = b.add_input("bb_conv_b", &[c]);
    let bm = b.add_input("bb_bn_m", &[c]);
    let bv = b.add_input("bb_bn_v", &[c]);
    let bw = b.add_input("bb_bn_w", &[c]);
    let bb = b.add_input("bb_bn_b", &[c]);
    let be = b.add_input("bb_bn_e", &[1]);

    let conv = b.add_conv2d(input, cw, Some(cb), 2, 2, 1, 1, &[c, s_out, s_out]);
    let bn = b.add_batch_norm(conv, bm, bv, bw, bb, be, &[c, s_out, s_out]);
    let relu = b.add_relu(bn, &[c, s_out, s_out]);

    // Flatten + project + reshape + LayerNorm
    let flat = b.add_reshape(relu, &[flat_dim]);
    let proj_w = b.add_input("proj_w", &[seq * d, flat_dim]);
    let proj = b.add_linear(flat, proj_w, None, &[seq * d]);
    let seq_feat = b.add_reshape(proj, &[seq, d]);

    let ln_e = b.add_input("ln_e", &[1]);
    let ln_w = b.add_input("ln_w", &[d]);
    let ln_b = b.add_input("ln_b", &[d]);
    let out = b.add_layer_norm(seq_feat, ln_e, 1, ln_w, ln_b, &[seq, d]);

    b.build(out).expect("valid backbone-to-transformer kernel")
}

/// Bindings for backbone-to-transformer.
fn backbone_to_transformer_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let s_out = FEAT_SIZE / 2;
    let flat_dim = c * s_out * s_out;
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[seq * d, flat_dim]),
            WEIGHT_MAG * 0.01,
        )),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)),
    ]
}

/// IBP bounds through backbone-to-transformer transition.
#[test]
fn test_backbone_to_transformer_ibp() {
    let def = build_backbone_to_transformer_kernel();
    let bindings = backbone_to_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through backbone-to-transformer");

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer backbone-to-transformer IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through backbone-to-transformer transition.
#[test]
fn test_backbone_to_transformer_crown() {
    let def = build_backbone_to_transformer_kernel();
    let bindings = backbone_to_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer backbone-to-transformer: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 29. ResNet BasicBlock with projection shortcut (1x1 conv skip for downsample)
// ===========================================================================

/// Build BasicBlock with 1x1 conv projection shortcut for stride-2 downsample.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable).
/// Output: `[CHANNELS, FEAT_SIZE/2, FEAT_SIZE/2]`.
fn build_resnet_basicblock_proj_skip_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s_in = FEAT_SIZE;
    let s_out = FEAT_SIZE / 2;
    let out_shape = [c, s_out, s_out];
    let mut b = TensorBlockBuilder::new("table_transformer_resnet_proj_skip");

    let input = b.add_input("features", &[c, s_in, s_in]);

    // Main path
    let c1w = b.add_input("c1_w", &[c, c, 3, 3]);
    let c1b = b.add_input("c1_b", &[c]);
    let b1m = b.add_input("b1_m", &[c]);
    let b1v = b.add_input("b1_v", &[c]);
    let b1w = b.add_input("b1_w", &[c]);
    let b1b = b.add_input("b1_b", &[c]);
    let b1e = b.add_input("b1_e", &[1]);

    let conv1 = b.add_conv2d(input, c1w, Some(c1b), 2, 2, 1, 1, &out_shape);
    let bn1 = b.add_batch_norm(conv1, b1m, b1v, b1w, b1b, b1e, &out_shape);
    let relu1 = b.add_relu(bn1, &out_shape);

    let c2w = b.add_input("c2_w", &[c, c, 3, 3]);
    let c2b = b.add_input("c2_b", &[c]);
    let b2m = b.add_input("b2_m", &[c]);
    let b2v = b.add_input("b2_v", &[c]);
    let b2w = b.add_input("b2_w", &[c]);
    let b2b = b.add_input("b2_b", &[c]);
    let b2e = b.add_input("b2_e", &[1]);

    let conv2 = b.add_conv2d(relu1, c2w, Some(c2b), 1, 1, 1, 1, &out_shape);
    let bn2 = b.add_batch_norm(conv2, b2m, b2v, b2w, b2b, b2e, &out_shape);

    // Skip: 1x1 conv(stride=2) + BN
    let sw = b.add_input("skip_w", &[c, c, 1, 1]);
    let sb = b.add_input("skip_b", &[c]);
    let sbm = b.add_input("skip_bm", &[c]);
    let sbv = b.add_input("skip_bv", &[c]);
    let sbw = b.add_input("skip_bw", &[c]);
    let sbb = b.add_input("skip_bb", &[c]);
    let sbe = b.add_input("skip_be", &[1]);

    let skip_conv = b.add_conv2d(input, sw, Some(sb), 2, 2, 0, 0, &out_shape);
    let skip_bn = b.add_batch_norm(skip_conv, sbm, sbv, sbw, sbb, sbe, &out_shape);

    let residual = b.add_binary_add(bn2, skip_bn, &out_shape);
    let out = b.add_relu(residual, &out_shape);

    b.build(out)
        .expect("valid BasicBlock with projection skip kernel")
}

/// Bindings for BasicBlock with projection skip.
fn resnet_basicblock_proj_skip_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let conv_w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let conv1x1_w = ArrayD::from_elem(IxDyn(&[c, c, 1, 1]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_m = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);
    let bn_v = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_w = ArrayD::from_elem(IxDyn(&[c]), 1.0f32);
    let bn_b = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    let conv_bn = |w: &ArrayD<f32>| -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantTensor(w.clone()),
            TensorParamBinding::ConstantTensor(bias.clone()),
            TensorParamBinding::ConstantTensor(bn_m.clone()),
            TensorParamBinding::ConstantTensor(bn_v.clone()),
            TensorParamBinding::ConstantTensor(bn_w.clone()),
            TensorParamBinding::ConstantTensor(bn_b.clone()),
            TensorParamBinding::ConstantScalar(1e-5),
        ]
    };

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(conv_bn(&conv_w)); // main path conv1+BN
    bindings.extend(conv_bn(&conv_w)); // main path conv2+BN
    bindings.extend(conv_bn(&conv1x1_w)); // skip 1x1 conv+BN
    bindings
}

/// IBP bounds through BasicBlock with projection shortcut.
#[test]
fn test_resnet_basicblock_proj_skip_ibp() {
    let def = build_resnet_basicblock_proj_skip_kernel();
    let bindings = resnet_basicblock_proj_skip_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through proj skip block");

    let s_out = FEAT_SIZE / 2;
    assert_eq!(output.lower_upper().0.shape(), &[CHANNELS, s_out, s_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer BasicBlock proj skip IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(lo_min >= 0.0 - eps, "ReLU lower must be >= 0, got {lo_min}");
}

// ===========================================================================
// 30. Encoder with final LayerNorm (DETR standard output normalization)
// ===========================================================================

/// Build encoder layer + final LayerNorm.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
fn build_encoder_with_final_norm_kernel() -> TensorKernelDef {
    let seq = ENC_SEQ_LEN;
    let d = HIDDEN_DIM;
    let seq_shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("table_transformer_enc_final_norm");

    let input = b.add_input("enc_features", &seq_shape);

    let ln1e = b.add_input("ln1_e", &[1]);
    let ln1w = b.add_input("ln1_w", &[d]);
    let ln1b = b.add_input("ln1_b", &[d]);
    let qw = b.add_input("q_w", &[d, d]);
    let kw = b.add_input("k_w", &[d, d]);
    let vw = b.add_input("v_w", &[d, d]);
    let ow = b.add_input("o_w", &[d, d]);
    let ln2e = b.add_input("ln2_e", &[1]);
    let ln2w = b.add_input("ln2_w", &[d]);
    let ln2b = b.add_input("ln2_b", &[d]);
    let fu = b.add_input("ffn_up", &[FFN_DIM, d]);
    let fd = b.add_input("ffn_dn", &[d, FFN_DIM]);

    let n1 = b.add_layer_norm(input, ln1e, 1, ln1w, ln1b, &seq_shape);
    let attn = b
        .add_multi_head_attention(
            n1,
            qw,
            kw,
            vw,
            ow,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("self-attn");
    let r1 = b.add_binary_add(input, attn, &seq_shape);
    let n2 = b.add_layer_norm(r1, ln2e, 1, ln2w, ln2b, &seq_shape);
    let fh = b.add_linear(n2, fu, None, &ffn_shape);
    let fa = b.add_relu(fh, &ffn_shape);
    let fo = b.add_linear(fa, fd, None, &seq_shape);
    let enc_out = b.add_binary_add(r1, fo, &seq_shape);

    // Final LayerNorm
    let fne = b.add_input("final_ln_e", &[1]);
    let fnw = b.add_input("final_ln_w", &[d]);
    let fnb = b.add_input("final_ln_b", &[d]);
    let out = b.add_layer_norm(enc_out, fne, 1, fnw, fnb, &seq_shape);

    b.build(out).expect("valid encoder with final norm kernel")
}

/// Bindings for encoder with final LayerNorm.
fn encoder_with_final_norm_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w.clone()),
        TensorParamBinding::ConstantTensor(attn_w),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w.clone()),
        TensorParamBinding::ConstantTensor(ln_b.clone()),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG)),
        // Final LayerNorm
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
    ]
}

/// IBP bounds through encoder with final LayerNorm.
#[test]
fn test_encoder_with_final_norm_ibp() {
    let def = build_encoder_with_final_norm_kernel();
    let bindings = encoder_with_final_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder+final_norm");

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer encoder+final_norm IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through encoder with final LayerNorm.
#[test]
fn test_encoder_with_final_norm_crown() {
    let def = build_encoder_with_final_norm_kernel();
    let bindings = encoder_with_final_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Table Transformer encoder+final_norm: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
