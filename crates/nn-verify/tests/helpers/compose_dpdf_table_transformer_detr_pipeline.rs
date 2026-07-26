// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the Table Transformer (DETR-based) pipeline.
//!
//! Verifies IBP and CROWN bound propagation through the full DETR pipeline used
//! for table structure recognition in the dpdf document understanding stack.
//!
//! **Sub-blocks tested:**
//!
//! 1. **DETR backbone (Conv-ReLU-Conv-ReLU)**: ResNet-style feature extraction
//!    with stride-2 downsampling. Verifies bounds through convolution + ReLU chains.
//!
//! 2. **Positional encoding (Linear-Sigmoid)**: Learned 2D position encoding
//!    projected through linear + sigmoid. Output bounded in [0, 1].
//!
//! 3. **Encoder self-attention block**: LayerNorm -> Linear Q/K/V -> MHA ->
//!    residual -> LayerNorm -> FFN(Linear-ReLU-Linear) -> residual. Full
//!    pre-norm DETR encoder layer.
//!
//! 4. **Decoder cross-attention**: LayerNorm -> Linear Q/K/V -> cross-MHA ->
//!    residual -> LayerNorm -> FFN -> residual. Object queries attend to
//!    encoder memory.
//!
//! 5. **Table detection head (Linear-Sigmoid)**: Bounding box + class prediction
//!    heads with sigmoid output bounded in [0, 1].
//!
//! 6. **Full E2E mini pipeline**: backbone -> encoder -> decoder -> heads.
//!    End-to-end bounds propagation through the complete DETR pipeline.
//!
//! Architecture references:
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - ResNet (He et al. 2016): Backbone feature extraction
//!
//! Dimensions (small for fast verification):
//! - Feature maps: 4x4 spatial, 32 channels
//! - Hidden: D=32, FFN_DIM=64, NUM_HEADS=4, NUM_QUERIES=4
//!
//! Part of #4237: Compose tests for Table Transformer full DETR pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Feature map spatial size (H=W) after backbone.
const FEAT_SIZE: usize = 4;
/// Backbone convolution channels.
const CHANNELS: usize = 32;
/// Hidden dimension for transformer encoder/decoder.
const HIDDEN_DIM: usize = 32;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Number of learned object queries (DETR-style).
const NUM_QUERIES: usize = 4;
/// Encoder sequence length = FEAT_SIZE * FEAT_SIZE (flattened spatial).
const ENC_SEQ_LEN: usize = FEAT_SIZE * FEAT_SIZE; // 16
/// Number of table structure classes.
const NUM_CLASSES: usize = 6;
/// Box coordinate dimensions (cx, cy, w, h).
const BOX_DIM: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. DETR backbone: Conv -> ReLU -> Conv -> ReLU (ResNet-style)
// ===========================================================================

/// Build a ResNet-style backbone feature extractor.
///
/// Input: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]` (Variable).
/// Output: `[CHANNELS, FEAT_SIZE, FEAT_SIZE]`.
///
/// Architecture: Conv2d(C,C,3,s=1,p=1) -> ReLU -> Conv2d(C,C,3,s=1,p=1) -> ReLU
fn build_backbone_kernel() -> TensorKernelDef {
    let c = CHANNELS;
    let s = FEAT_SIZE;
    let shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("detr_pipeline_backbone");

    let input = b.add_input("features", &shape);
    let conv1_w = b.add_input("conv1_weight", &[c, c, 3, 3]);
    let conv1_b = b.add_input("conv1_bias", &[c]);
    let conv2_w = b.add_input("conv2_weight", &[c, c, 3, 3]);
    let conv2_b = b.add_input("conv2_bias", &[c]);

    // Conv -> ReLU -> Conv -> ReLU
    let h1 = b.add_conv2d(input, conv1_w, Some(conv1_b), 1, 1, 1, 1, &shape);
    let h1_act = b.add_relu(h1, &shape);
    let h2 = b.add_conv2d(h1_act, conv2_w, Some(conv2_b), 1, 1, 1, 1, &shape);
    let out = b.add_relu(h2, &shape);

    b.build(out).expect("valid backbone kernel")
}

/// Bindings for the backbone kernel.
fn backbone_bindings() -> Vec<TensorParamBinding> {
    let c = CHANNELS;
    let w = ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[c]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                     // features [C, S, S]
        TensorParamBinding::ConstantTensor(w.clone()),    // conv1_weight
        TensorParamBinding::ConstantTensor(bias.clone()), // conv1_bias
        TensorParamBinding::ConstantTensor(w),            // conv2_weight
        TensorParamBinding::ConstantTensor(bias),         // conv2_bias
    ]
}

#[test]
fn test_detr_pipeline_backbone_def_validates() {
    let def = build_backbone_kernel();
    def.validate().expect("backbone should validate");
}

#[test]
fn test_detr_pipeline_backbone_ibp_propagates() {
    let def = build_backbone_kernel();
    let bindings = backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through backbone");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, FEAT_SIZE, FEAT_SIZE],
        "backbone output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR backbone IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // ReLU clamps lower bound to >= 0
    assert!(
        lo_min >= 0.0,
        "ReLU output lower bound should be >= 0, got {lo_min}"
    );
}

#[test]
fn test_detr_pipeline_backbone_crown_propagation() {
    let def = build_backbone_kernel();
    let bindings = backbone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[CHANNELS, FEAT_SIZE, FEAT_SIZE], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[CHANNELS, FEAT_SIZE, FEAT_SIZE],
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR backbone CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 2. Positional encoding: Linear -> Sigmoid (learned 2D position encoding)
// ===========================================================================

/// Build learned positional encoding: features -> Linear -> Sigmoid.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable -- flattened spatial tokens).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Sigmoid ensures output is bounded in [0, 1], modeling learned 2D position
/// encoding that maps spatial coordinates to attention-compatible representations.
fn build_positional_encoding_kernel() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let seq = ENC_SEQ_LEN;
    let shape = [seq, d];
    let mut b = TensorBlockBuilder::new("detr_pipeline_pos_enc");

    let input = b.add_input("spatial_features", &shape);
    let proj_w = b.add_input("pos_proj_weight", &[d, d]);
    let proj_b = b.add_input("pos_proj_bias", &[d]);

    let projected = b.add_linear(input, proj_w, Some(proj_b), &shape);
    let out = b.add_sigmoid(projected, &shape);

    b.build(out).expect("valid positional encoding kernel")
}

/// Bindings for the positional encoding kernel.
fn positional_encoding_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // spatial_features [S, D]
        TensorParamBinding::ConstantTensor(w),    // pos_proj_weight [D, D]
        TensorParamBinding::ConstantTensor(bias), // pos_proj_bias [D]
    ]
}

#[test]
fn test_detr_pipeline_pos_enc_def_validates() {
    let def = build_positional_encoding_kernel();
    def.validate().expect("positional encoding should validate");
}

#[test]
fn test_detr_pipeline_pos_enc_ibp_sigmoid_bounds() {
    let def = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through pos enc");

    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR pos_enc IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= 0.0, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_detr_pipeline_pos_enc_crown_propagation() {
    let def = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR pos_enc CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    // Sigmoid invariant holds under CROWN too
    assert!(lo_min >= 0.0, "sigmoid lower >= 0 under CROWN");
    assert!(hi_max <= 1.0, "sigmoid upper <= 1 under CROWN");
}

#[test]
fn test_detr_pipeline_pos_enc_verify_and_record() {
    let def = build_positional_encoding_kernel();
    let bindings = positional_encoding_bindings();
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_pipeline_pos_enc");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 3. Encoder self-attention block: LN -> MHA -> res -> LN -> FFN(ReLU) -> res
// ===========================================================================

/// Build a DETR encoder layer (pre-norm with ReLU FFN).
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[ENC_SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture:
///   LN -> MHA(bidirectional) -> + residual
///   -> LN -> Linear -> ReLU -> Linear -> + residual
fn build_encoder_self_attention_kernel() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let seq = ENC_SEQ_LEN;
    let shape = [seq, d];
    let ffn_shape = [seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("detr_pipeline_encoder_sa");

    let input = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    // Self-attention sub-block
    let ln1_w = b.add_input("ln1_weight", &[d]);
    let ln1_b = b.add_input("ln1_bias", &[d]);
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    // FFN sub-block
    let ln2_w = b.add_input("ln2_weight", &[d]);
    let ln2_b = b.add_input("ln2_bias", &[d]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, d]);
    let ffn2_w = b.add_input("ffn2_weight", &[d, FFN_DIM]);

    // Sub-block 1: Self-attention
    let normed1 = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &shape);
    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard, // bidirectional
            &shape,
        )
        .expect("valid self-attention");
    let residual1 = b.add_binary_add(input, attn, &shape);

    // Sub-block 2: FFN with ReLU
    let normed2 = b.add_layer_norm(residual1, eps, 1, ln2_w, ln2_b, &shape);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_relu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(residual1, ffn2, &shape);

    b.build(out).expect("valid encoder self-attention kernel")
}

/// Bindings for the encoder self-attention kernel.
fn encoder_self_attention_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // x [S, D]
        TensorParamBinding::ConstantScalar(1e-5),           // eps
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // ln1_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj),         // out_weight
        TensorParamBinding::ConstantTensor(ln_w),           // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln2_bias
        TensorParamBinding::ConstantTensor(w_ffn1),         // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2),         // ffn2_weight
    ]
}

#[test]
fn test_detr_pipeline_encoder_sa_def_validates() {
    let def = build_encoder_self_attention_kernel();
    def.validate()
        .expect("encoder self-attention should validate");
}

#[test]
fn test_detr_pipeline_encoder_sa_ibp_propagates() {
    let def = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through encoder SA");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[ENC_SEQ_LEN, HIDDEN_DIM],
        "encoder output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR encoder SA IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_detr_pipeline_encoder_sa_crown_propagation() {
    let def = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR encoder SA CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_detr_pipeline_encoder_sa_verify_and_record() {
    let def = build_encoder_self_attention_kernel();
    let bindings = encoder_self_attention_bindings();
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_pipeline_encoder_sa");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[ENC_SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm produces heuristic soundness mode
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "encoder with LayerNorm should be Heuristic"
    );
}

// ===========================================================================
// 4. Decoder cross-attention: LN -> Cross-MHA -> res -> LN -> FFN -> res
// ===========================================================================

/// Build a DETR decoder layer with cross-attention.
///
/// Q input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable -- object queries).
/// KV input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Constant -- encoder memory).
/// Output: `[NUM_QUERIES, HIDDEN_DIM]`.
///
/// Architecture:
///   LN(queries) -> Cross-MHA(Q=queries, KV=encoder_memory) -> + residual
///   -> LN -> Linear -> ReLU -> Linear -> + residual
fn build_decoder_cross_attention_kernel() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let q_seq = NUM_QUERIES;
    let q_shape = [q_seq, d];
    let ffn_shape = [q_seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("detr_pipeline_decoder_ca");

    let q_input = b.add_input("object_queries", &q_shape);
    let kv_input = b.add_input("encoder_memory", &[ENC_SEQ_LEN, d]);
    let eps = b.add_input("eps", &[1]);

    // Cross-attention sub-block
    let ln1_w = b.add_input("ln1_weight", &[d]);
    let ln1_b = b.add_input("ln1_bias", &[d]);
    let q_w = b.add_input("q_weight", &[d, d]);
    let k_w = b.add_input("k_weight", &[d, d]);
    let v_w = b.add_input("v_weight", &[d, d]);
    let out_w = b.add_input("out_weight", &[d, d]);

    // FFN sub-block
    let ln2_w = b.add_input("ln2_weight", &[d]);
    let ln2_b = b.add_input("ln2_bias", &[d]);
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, d]);
    let ffn2_w = b.add_input("ffn2_weight", &[d, FFN_DIM]);

    // Sub-block 1: Cross-attention (Q from decoder, KV from encoder)
    let normed_q = b.add_layer_norm(q_input, eps, 1, ln1_w, ln1_b, &q_shape);
    let attn = b
        .add_multi_head_cross_attention(
            normed_q,
            kv_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid cross-attention");
    let residual1 = b.add_binary_add(q_input, attn, &q_shape);

    // Sub-block 2: FFN with ReLU
    let normed2 = b.add_layer_norm(residual1, eps, 1, ln2_w, ln2_b, &q_shape);
    let ffn1 = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let act = b.add_relu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ffn2_w, None, &q_shape);
    let out = b.add_binary_add(residual1, ffn2, &q_shape);

    b.build(out).expect("valid decoder cross-attention kernel")
}

/// Bindings for the decoder cross-attention kernel.
///
/// Object queries are Variable; encoder memory is constant.
fn decoder_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let encoder_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.5f32);

    vec![
        TensorParamBinding::Variable,                    // object_queries [Q, D]
        TensorParamBinding::ConstantTensor(encoder_mem), // encoder_memory [S, D]
        TensorParamBinding::ConstantScalar(1e-5),        // eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ln1_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj),      // out_weight
        TensorParamBinding::ConstantTensor(ln_w),        // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),        // ln2_bias
        TensorParamBinding::ConstantTensor(w_ffn1),      // ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2),      // ffn2_weight
    ]
}

#[test]
fn test_detr_pipeline_decoder_ca_def_validates() {
    let def = build_decoder_cross_attention_kernel();
    def.validate()
        .expect("decoder cross-attention should validate");
}

#[test]
fn test_detr_pipeline_decoder_ca_ibp_propagates() {
    let def = build_decoder_cross_attention_kernel();
    let bindings = decoder_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through decoder CA");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, HIDDEN_DIM],
        "decoder output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder CA IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_detr_pipeline_decoder_ca_crown_propagation() {
    let def = build_decoder_cross_attention_kernel();
    let bindings = decoder_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder CA CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_detr_pipeline_decoder_ca_verify_and_record() {
    let def = build_decoder_cross_attention_kernel();
    let bindings = decoder_cross_attention_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_pipeline_decoder_ca");
    assert_eq!(result.num_variables, 1, "single Variable input (queries)");
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, HIDDEN_DIM]);

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "decoder with LayerNorm should be Heuristic"
    );
}

// ===========================================================================
// 5. Table detection head: Linear -> Sigmoid (bbox + class prediction)
// ===========================================================================

/// Build a dual detection head: class sigmoid + box sigmoid, concatenated.
///
/// Input: `[NUM_QUERIES, HIDDEN_DIM]` (Variable -- decoder output).
/// Output: `[NUM_QUERIES, NUM_CLASSES + BOX_DIM]`.
///
/// Architecture:
///   cls_head: Linear(D, C) -> Sigmoid
///   box_head: Linear(D, 4) -> Sigmoid
///   output: concat(cls, box) along dim=1
fn build_detection_head_kernel() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let q = NUM_QUERIES;
    let mut b = TensorBlockBuilder::new("detr_pipeline_detection_head");

    let input = b.add_input("decoder_output", &[q, d]);
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, d]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let box_w = b.add_input("box_weight", &[BOX_DIM, d]);
    let box_b = b.add_input("box_bias", &[BOX_DIM]);

    // Class head: Linear -> Sigmoid
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &[q, NUM_CLASSES]);
    let cls_out = b.add_sigmoid(cls_logits, &[q, NUM_CLASSES]);

    // Box head: Linear -> Sigmoid
    let box_logits = b.add_linear(input, box_w, Some(box_b), &[q, BOX_DIM]);
    let box_out = b.add_sigmoid(box_logits, &[q, BOX_DIM]);

    // Concatenate
    let out_dim = NUM_CLASSES + BOX_DIM;
    let out = b.add_concat(&[cls_out, box_out], 1, &[q, out_dim]);

    b.build(out).expect("valid detection head kernel")
}

/// Bindings for the detection head kernel.
fn detection_head_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, d]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    let box_w = ArrayD::from_elem(IxDyn(&[BOX_DIM, d]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[BOX_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // decoder_output [Q, D]
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight [C, D]
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias [C]
        TensorParamBinding::ConstantTensor(box_w), // box_weight [4, D]
        TensorParamBinding::ConstantTensor(box_b), // box_bias [4]
    ]
}

#[test]
fn test_detr_pipeline_detection_head_def_validates() {
    let def = build_detection_head_kernel();
    def.validate().expect("detection head should validate");
}

#[test]
fn test_detr_pipeline_detection_head_ibp_sigmoid_bounds() {
    let def = build_detection_head_kernel();
    let bindings = detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection head");
    let out_dim = NUM_CLASSES + BOX_DIM;

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, out_dim],
        "detection head output shape"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR detection head IBP: bounds=[{lo_min}, {hi_max}]");
    // Both branches use sigmoid -- output bounded in [0, 1]
    assert!(lo_min >= 0.0, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_detr_pipeline_detection_head_crown_propagation() {
    let def = build_detection_head_kernel();
    let bindings = detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let out_dim = NUM_CLASSES + BOX_DIM;
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, out_dim]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR detection head CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    // Sigmoid bounds preserved under CROWN
    assert!(lo_min >= 0.0, "sigmoid lower >= 0 under CROWN");
    assert!(hi_max <= 1.0, "sigmoid upper <= 1 under CROWN");
}

#[test]
fn test_detr_pipeline_detection_head_verify_and_record() {
    let def = build_detection_head_kernel();
    let bindings = detection_head_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_pipeline_detection_head");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let out_dim = NUM_CLASSES + BOX_DIM;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, out_dim]);
}

// ===========================================================================
// 6. Full E2E mini pipeline: backbone -> encoder -> decoder -> heads
// ===========================================================================

/// Build a full DETR mini pipeline end-to-end.
///
/// Input: `[ENC_SEQ_LEN, HIDDEN_DIM]` (Variable -- flattened backbone features).
/// Output: `[NUM_QUERIES, NUM_CLASSES + BOX_DIM]`.
///
/// Architecture:
///   features -> LN -> MHA(self-attn) -> res -> LN -> FFN(ReLU) -> res  (encoder)
///   -> decoder cross-attn with constant queries -> LN -> FFN -> res    (decoder)
///   -> cls sigmoid + box sigmoid concat                                (heads)
///
/// NOTE: The backbone Conv2d output is modeled as already flattened to [S, D]
/// to keep the pipeline in 2D tensor space (conv2d outputs are 3D).
/// The decoder uses constant queries attending to encoder output.
fn build_full_e2e_pipeline_kernel() -> TensorKernelDef {
    let d = HIDDEN_DIM;
    let enc_seq = ENC_SEQ_LEN;
    let q_seq = NUM_QUERIES;
    let enc_shape = [enc_seq, d];
    let enc_ffn_shape = [enc_seq, FFN_DIM];
    let q_shape = [q_seq, d];
    let q_ffn_shape = [q_seq, FFN_DIM];
    let mut b = TensorBlockBuilder::new("detr_pipeline_full_e2e");

    // Encoder input (flattened backbone features)
    let enc_input = b.add_input("features", &enc_shape);
    let eps = b.add_input("eps", &[1]);

    // Encoder self-attention weights
    let enc_ln1_w = b.add_input("enc_ln1_weight", &[d]);
    let enc_ln1_b = b.add_input("enc_ln1_bias", &[d]);
    let enc_q_w = b.add_input("enc_q_weight", &[d, d]);
    let enc_k_w = b.add_input("enc_k_weight", &[d, d]);
    let enc_v_w = b.add_input("enc_v_weight", &[d, d]);
    let enc_out_w = b.add_input("enc_out_weight", &[d, d]);

    // Encoder FFN weights
    let enc_ln2_w = b.add_input("enc_ln2_weight", &[d]);
    let enc_ln2_b = b.add_input("enc_ln2_bias", &[d]);
    let enc_ffn1_w = b.add_input("enc_ffn1_weight", &[FFN_DIM, d]);
    let enc_ffn2_w = b.add_input("enc_ffn2_weight", &[d, FFN_DIM]);

    // Decoder: constant object queries, cross-attention to encoder output
    let dec_queries = b.add_input("dec_queries", &q_shape);
    let dec_ln1_w = b.add_input("dec_ln1_weight", &[d]);
    let dec_ln1_b = b.add_input("dec_ln1_bias", &[d]);
    let dec_q_w = b.add_input("dec_q_weight", &[d, d]);
    let dec_k_w = b.add_input("dec_k_weight", &[d, d]);
    let dec_v_w = b.add_input("dec_v_weight", &[d, d]);
    let dec_out_w = b.add_input("dec_out_weight", &[d, d]);

    // Decoder FFN
    let dec_ln2_w = b.add_input("dec_ln2_weight", &[d]);
    let dec_ln2_b = b.add_input("dec_ln2_bias", &[d]);
    let dec_ffn1_w = b.add_input("dec_ffn1_weight", &[FFN_DIM, d]);
    let dec_ffn2_w = b.add_input("dec_ffn2_weight", &[d, FFN_DIM]);

    // Detection head weights
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, d]);
    let cls_bias = b.add_input("cls_bias", &[NUM_CLASSES]);
    let box_w = b.add_input("box_weight", &[BOX_DIM, d]);
    let box_bias = b.add_input("box_bias", &[BOX_DIM]);

    // ---- Encoder ----
    let enc_normed1 = b.add_layer_norm(enc_input, eps, 1, enc_ln1_w, enc_ln1_b, &enc_shape);
    let enc_attn = b
        .add_multi_head_attention(
            enc_normed1,
            enc_q_w,
            enc_k_w,
            enc_v_w,
            enc_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &enc_shape,
        )
        .expect("valid encoder self-attention");
    let enc_res1 = b.add_binary_add(enc_input, enc_attn, &enc_shape);

    let enc_normed2 = b.add_layer_norm(enc_res1, eps, 1, enc_ln2_w, enc_ln2_b, &enc_shape);
    let enc_ffn1 = b.add_linear(enc_normed2, enc_ffn1_w, None, &enc_ffn_shape);
    let enc_act = b.add_relu(enc_ffn1, &enc_ffn_shape);
    let enc_ffn2 = b.add_linear(enc_act, enc_ffn2_w, None, &enc_shape);
    let encoder_out = b.add_binary_add(enc_res1, enc_ffn2, &enc_shape);

    // ---- Decoder (cross-attention: queries attend to encoder output) ----
    let dec_normed1 = b.add_layer_norm(dec_queries, eps, 1, dec_ln1_w, dec_ln1_b, &q_shape);
    let dec_attn = b
        .add_multi_head_cross_attention(
            dec_normed1,
            encoder_out,
            dec_q_w,
            dec_k_w,
            dec_v_w,
            dec_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &q_shape,
        )
        .expect("valid decoder cross-attention");
    let dec_res1 = b.add_binary_add(dec_queries, dec_attn, &q_shape);

    let dec_normed2 = b.add_layer_norm(dec_res1, eps, 1, dec_ln2_w, dec_ln2_b, &q_shape);
    let dec_ffn1 = b.add_linear(dec_normed2, dec_ffn1_w, None, &q_ffn_shape);
    let dec_act = b.add_relu(dec_ffn1, &q_ffn_shape);
    let dec_ffn2 = b.add_linear(dec_act, dec_ffn2_w, None, &q_shape);
    let decoder_out = b.add_binary_add(dec_res1, dec_ffn2, &q_shape);

    // ---- Detection heads ----
    let cls_logits = b.add_linear(decoder_out, cls_w, Some(cls_bias), &[q_seq, NUM_CLASSES]);
    let cls_out = b.add_sigmoid(cls_logits, &[q_seq, NUM_CLASSES]);

    let box_logits = b.add_linear(decoder_out, box_w, Some(box_bias), &[q_seq, BOX_DIM]);
    let box_out = b.add_sigmoid(box_logits, &[q_seq, BOX_DIM]);

    let out_dim = NUM_CLASSES + BOX_DIM;
    let out = b.add_concat(&[cls_out, box_out], 1, &[q_seq, out_dim]);

    b.build(out).expect("valid full E2E pipeline kernel")
}

/// Bindings for the full E2E pipeline kernel.
fn full_e2e_pipeline_bindings() -> Vec<TensorParamBinding> {
    let d = HIDDEN_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let queries = ArrayD::from_elem(IxDyn(&[NUM_QUERIES, d]), 0.1f32);
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, d]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    let box_w = ArrayD::from_elem(IxDyn(&[BOX_DIM, d]), WEIGHT_MAG);
    let box_b = ArrayD::from_elem(IxDyn(&[BOX_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // features [S, D]
        TensorParamBinding::ConstantScalar(1e-5), // eps
        // Encoder self-attention
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc_ln1_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // enc_out_weight
        // Encoder FFN
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc_ln2_bias
        TensorParamBinding::ConstantTensor(w_ffn1.clone()), // enc_ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2.clone()), // enc_ffn2_weight
        // Decoder cross-attention
        TensorParamBinding::ConstantTensor(queries), // dec_queries [Q, D]
        TensorParamBinding::ConstantTensor(ln_w.clone()), // dec_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // dec_ln1_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // dec_q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // dec_k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // dec_v_weight
        TensorParamBinding::ConstantTensor(w_proj),  // dec_out_weight
        // Decoder FFN
        TensorParamBinding::ConstantTensor(ln_w), // dec_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // dec_ln2_bias
        TensorParamBinding::ConstantTensor(w_ffn1), // dec_ffn1_weight
        TensorParamBinding::ConstantTensor(w_ffn2), // dec_ffn2_weight
        // Detection heads
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias
        TensorParamBinding::ConstantTensor(box_w), // box_weight
        TensorParamBinding::ConstantTensor(box_b), // box_bias
    ]
}

#[test]
fn test_detr_pipeline_full_e2e_def_validates() {
    let def = build_full_e2e_pipeline_kernel();
    def.validate().expect("full E2E pipeline should validate");
}

#[test]
fn test_detr_pipeline_full_e2e_graph_builds() {
    let def = build_full_e2e_pipeline_kernel();
    let bindings = full_e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Full pipeline: encoder (LN + MHA + res + LN + FFN + res) + decoder (same) + heads
    assert!(
        graph.num_nodes() >= 20,
        "full E2E pipeline should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_detr_pipeline_full_e2e_ibp_propagates() {
    let def = build_full_e2e_pipeline_kernel();
    let bindings = full_e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full E2E pipeline");
    let out_dim = NUM_CLASSES + BOX_DIM;

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, out_dim],
        "full pipeline output shape [Q, C+4]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR full E2E IBP: bounds=[{lo_min}, {hi_max}]");
    // Sigmoid heads clamp output to [0, 1]
    assert!(lo_min >= 0.0, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0, "sigmoid upper <= 1, got {hi_max}");
}

#[test]
fn test_detr_pipeline_full_e2e_bounds_width() {
    let def = build_full_e2e_pipeline_kernel();
    let bindings = full_e2e_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through full E2E");
    let (lo, hi) = output.lower_upper();

    let max_width = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, u)| (u - l).abs())
        .fold(0.0f32, f32::max);

    // Sigmoid output bounds width <= 1.0 by construction
    assert!(
        max_width <= 1.0,
        "sigmoid output bounds max width {max_width} should be <= 1.0"
    );
    eprintln!("DETR full E2E IBP max width: {max_width}");
}

#[test]
fn test_detr_pipeline_full_e2e_verify_and_record() {
    let def = build_full_e2e_pipeline_kernel();
    let bindings = full_e2e_pipeline_bindings();
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_pipeline_full_e2e");
    assert_eq!(result.num_variables, 1, "single Variable input");
    let out_dim = NUM_CLASSES + BOX_DIM;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, out_dim]);

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "full pipeline with LayerNorm should be Heuristic"
    );
}
