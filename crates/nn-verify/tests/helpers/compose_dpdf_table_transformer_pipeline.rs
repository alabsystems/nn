// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Table Transformer DETR full pipeline bounds.
//!
//! Verifies IBP and CROWN bound propagation through the complete Table
//! Transformer encoder-decoder pipeline at production-representative
//! dimensions (hidden=256, heads=8, FFN=2048).
//!
//! 1.  **resnet18_backbone_feature_extraction**: ResNet18 Conv-BN-ReLU
//!     backbone feature extraction bounds (IBP).
//! 2.  **position_encoding_sinusoidal**: Sinusoidal 2D PE bounded in [-1, 1]
//!     with additive projection (IBP).
//! 3.  **transformer_encoder_self_attention**: Self-attention + LN + FFN +
//!     residual encoder layer (IBP + CROWN).
//! 4.  **detr_decoder_cross_attention**: Q from decoder queries, KV from
//!     encoder memory cross-attention (IBP + CROWN).
//! 5.  **object_query_init_refinement**: Learned query embeddings projected
//!     through linear + LayerNorm (IBP).
//! 6.  **table_cell_classification_softmax**: Linear -> softmax class
//!     probabilities in [0, 1] (IBP).
//! 7.  **table_row_col_regression_sigmoid**: Linear -> sigmoid for normalized
//!     row/column coordinates in [0, 1] (IBP).
//! 8.  **hungarian_matching_cost_computation**: Dual-head cls + box cost
//!     matrix entries bounded (IBP).
//! 9.  **full_encoder_pipeline_e2e**: Features + PE -> 2 encoder layers
//!     -> LayerNorm end-to-end (IBP + CROWN).
//! 10. **full_decoder_pipeline_e2e**: Queries + encoder memory -> 2 decoder
//!     layers -> LayerNorm -> dual sigmoid heads end-to-end (IBP + CROWN).
//!
//! Architecture references:
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - ResNet (He et al. 2016): Backbone feature extraction
//!
//! Dimensions (production-representative Table Transformer):
//! - HIDDEN_DIM=256, NUM_HEADS=8, FFN_DIM=2048, NUM_QUERIES=100
//! - Feature spatial: 8x8 (small for fast verification)
//!
//! Part of #4177: Compose tests for Table Transformer DETR full pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- production Table Transformer configuration
// ---------------------------------------------------------------------------

/// Hidden dimension (Table Transformer default).
const HIDDEN_DIM: usize = 256;
/// Number of attention heads (Table Transformer default).
const NUM_HEADS: usize = 8;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 32
/// FFN intermediate dimension (Table Transformer default).
const FFN_DIM: usize = 2048;
/// Number of learned object queries (Table Transformer default).
const NUM_QUERIES: usize = 100;
/// Feature map spatial size (H=W), small for fast verification.
const FEAT_SIZE: usize = 8;
/// Backbone convolution channels (ResNet18 layer4 output).
const BACKBONE_CHANNELS: usize = 512;
/// Encoder sequence length = FEAT_SIZE * FEAT_SIZE (flattened spatial).
const ENC_SEQ_LEN: usize = FEAT_SIZE * FEAT_SIZE; // 64
/// Number of table structure classes (table, row, column, cell, header, spanning, background).
const NUM_CLASSES: usize = 7;
/// Box coordinate dimensions (x_center, y_center, width, height).
const BOX_DIM: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// 2D sinusoidal positional encoding for spatial grid (H x W) -> (H*W, D).
fn sinusoidal_pe_2d(h: usize, w: usize, d_model: usize) -> ArrayD<f32> {
    let seq_len = h * w;
    let mut data = vec![0.0f32; seq_len * d_model];
    let half_d = d_model / 2;
    for row in 0..h {
        for col in 0..w {
            let pos = row * w + col;
            for i in 0..half_d / 2 {
                let freq_h = (row as f64) / 10000.0_f64.powf(4.0 * i as f64 / d_model as f64);
                let freq_w = (col as f64) / 10000.0_f64.powf(4.0 * i as f64 / d_model as f64);
                data[pos * d_model + 4 * i] = freq_h.sin() as f32;
                data[pos * d_model + 4 * i + 1] = freq_h.cos() as f32;
                data[pos * d_model + 4 * i + 2] = freq_w.sin() as f32;
                data[pos * d_model + 4 * i + 3] = freq_w.cos() as f32;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid 2D PE")
}

/// Build a DETR encoder layer: LN -> self-attn -> residual -> LN -> FFN -> residual.
fn add_encoder_layer(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let seq_shape = [ENC_SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [ENC_SEQ_LEN, FFN_DIM];

    // LayerNorm 1
    let ln1_w = b.add_input(&format!("{prefix}ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("{prefix}ln1_b"), &[HIDDEN_DIM]);
    let ln1_eps = b.add_input(&format!("{prefix}ln1_eps"), &[1]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &seq_shape);

    // Self-attention
    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input(&format!("{prefix}o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &seq_shape,
        )
        .expect("valid self-attention");
    let res1 = b.add_binary_add(input, attn, &seq_shape);

    // LayerNorm 2
    let ln2_w = b.add_input(&format!("{prefix}ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("{prefix}ln2_b"), &[HIDDEN_DIM]);
    let ln2_eps = b.add_input(&format!("{prefix}ln2_eps"), &[1]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &seq_shape);

    // FFN: Linear -> ReLU -> Linear
    let ffn1_w = b.add_input(&format!("{prefix}ffn1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input(&format!("{prefix}ffn2_w"), &[HIDDEN_DIM, FFN_DIM]);
    let ffn_hidden = b.add_linear(normed2, ffn1_w, None, &ffn_shape);
    let ffn_act = b.add_relu(ffn_hidden, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn2_w, None, &seq_shape);

    b.add_binary_add(res1, ffn_out, &seq_shape)
}

/// Push bindings for one encoder layer (13 params).
fn push_encoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln1_w
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln1_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(attn_w)); // o_w
    bindings.push(TensorParamBinding::ConstantTensor(ln_w)); // ln2_w
    bindings.push(TensorParamBinding::ConstantTensor(ln_b)); // ln2_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(ffn1_w)); // ffn1_w
    bindings.push(TensorParamBinding::ConstantTensor(ffn2_w)); // ffn2_w
}

/// Build a DETR decoder layer: self-attn -> cross-attn(queries, encoder_mem) -> FFN.
fn add_decoder_layer(
    b: &mut TensorBlockBuilder,
    queries: nn_dsl::tensor_ir::TensorNodeId,
    encoder_mem: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let q_shape = [NUM_QUERIES, HIDDEN_DIM];
    let enc_shape = [ENC_SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [NUM_QUERIES, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-norm: LayerNorm -> Self-attention -> residual
    let sa_ln_w = b.add_input(&format!("{prefix}sa_ln_w"), &[HIDDEN_DIM]);
    let sa_ln_b = b.add_input(&format!("{prefix}sa_ln_b"), &[HIDDEN_DIM]);
    let sa_eps = b.add_input(&format!("{prefix}sa_eps"), &[1]);
    let normed_sa = b.add_layer_norm(queries, sa_eps, 1, sa_ln_w, sa_ln_b, &q_shape);

    let sa_qw = b.add_input(&format!("{prefix}sa_qw"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_kw = b.add_input(&format!("{prefix}sa_kw"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_vw = b.add_input(&format!("{prefix}sa_vw"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_ow = b.add_input(&format!("{prefix}sa_ow"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(normed_sa, sa_qw, None, &q_shape);
    let sk = b.add_linear(normed_sa, sa_kw, None, &q_shape);
    let sv = b.add_linear(normed_sa, sa_vw, None, &q_shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Standard, Some(scale), &q_shape);
    let sa_proj = b.add_linear(sa, sa_ow, None, &q_shape);
    let res_sa = b.add_binary_add(queries, sa_proj, &q_shape);

    // Cross-attention: LayerNorm -> cross-attn(query, encoder_mem) -> residual
    let ca_ln_w = b.add_input(&format!("{prefix}ca_ln_w"), &[HIDDEN_DIM]);
    let ca_ln_b = b.add_input(&format!("{prefix}ca_ln_b"), &[HIDDEN_DIM]);
    let ca_eps = b.add_input(&format!("{prefix}ca_eps"), &[1]);
    let normed_ca = b.add_layer_norm(res_sa, ca_eps, 1, ca_ln_w, ca_ln_b, &q_shape);

    let ca_qw = b.add_input(&format!("{prefix}ca_qw"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_kw = b.add_input(&format!("{prefix}ca_kw"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_vw = b.add_input(&format!("{prefix}ca_vw"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_ow = b.add_input(&format!("{prefix}ca_ow"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let cq = b.add_linear(normed_ca, ca_qw, None, &q_shape);
    let ck = b.add_linear(encoder_mem, ca_kw, None, &enc_shape);
    let cv = b.add_linear(encoder_mem, ca_vw, None, &enc_shape);
    let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &q_shape);
    let ca_proj = b.add_linear(ca, ca_ow, None, &q_shape);
    let res_ca = b.add_binary_add(res_sa, ca_proj, &q_shape);

    // FFN: LayerNorm -> Linear -> ReLU -> Linear -> residual
    let ffn_ln_w = b.add_input(&format!("{prefix}ffn_ln_w"), &[HIDDEN_DIM]);
    let ffn_ln_b = b.add_input(&format!("{prefix}ffn_ln_b"), &[HIDDEN_DIM]);
    let ffn_eps = b.add_input(&format!("{prefix}ffn_eps"), &[1]);
    let normed_ffn = b.add_layer_norm(res_ca, ffn_eps, 1, ffn_ln_w, ffn_ln_b, &q_shape);

    let ffn1_w = b.add_input(&format!("{prefix}ffn1_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input(&format!("{prefix}ffn2_w"), &[HIDDEN_DIM, FFN_DIM]);

    let ffn_hidden = b.add_linear(normed_ffn, ffn1_w, None, &ffn_shape);
    let ffn_act = b.add_relu(ffn_hidden, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn2_w, None, &q_shape);
    b.add_binary_add(res_ca, ffn_out, &q_shape)
}

/// Push one DETR decoder layer's bindings (19 params) onto the vec.
fn push_decoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // Self-attention norm + projections
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // sa_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // sa_ln_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // sa_eps
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_qw
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_kw
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_vw
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_ow
                                                                       // Cross-attention norm + projections
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ca_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ca_ln_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ca_eps
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // ca_qw
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // ca_kw
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // ca_vw
    bindings.push(TensorParamBinding::ConstantTensor(proj_w)); // ca_ow
                                                               // FFN norm + projections
    bindings.push(TensorParamBinding::ConstantTensor(ln_w)); // ffn_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ln_b)); // ffn_ln_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ffn_eps
    bindings.push(TensorParamBinding::ConstantTensor(ffn1_w)); // ffn1_w
    bindings.push(TensorParamBinding::ConstantTensor(ffn2_w)); // ffn2_w
}

// ===========================================================================
// 1. ResNet18 backbone feature extraction bounds (IBP)
// ===========================================================================

/// ResNet18 backbone produces feature maps via Conv2d -> BN -> ReLU.
/// Verifies that backbone features at production channel width (512)
/// produce finite, non-negative bounds through ReLU.
#[test]
fn test_resnet18_backbone_feature_extraction_ibp() {
    let c = BACKBONE_CHANNELS;
    let s = FEAT_SIZE;
    let feat_shape = [c, s, s];
    let mut b = TensorBlockBuilder::new("tt_pipe_resnet18_backbone");

    let input = b.add_input("features", &feat_shape);

    // Conv2d(512, 512, 3, stride=1, padding=1) -> BN -> ReLU
    let conv_w = b.add_input("conv_weight", &[c, c, 3, 3]);
    let conv_b = b.add_input("conv_bias", &[c]);
    let bn_mean = b.add_input("bn_mean", &[c]);
    let bn_var = b.add_input("bn_var", &[c]);
    let bn_weight = b.add_input("bn_weight", &[c]);
    let bn_bias = b.add_input("bn_bias", &[c]);
    let bn_eps = b.add_input("bn_eps", &[1]);

    let conv_out = b.add_conv2d(input, conv_w, Some(conv_b), 1, 1, 1, 1, &feat_shape);
    let bn_out = b.add_batch_norm(
        conv_out,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        bn_eps,
        &feat_shape,
    );
    let relu_out = b.add_relu(bn_out, &feat_shape);

    // Project to HIDDEN_DIM: flatten spatial -> Linear
    // For verification we model the channel projection as a simple linear
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, c]);
    let flat_shape = [s * s, c];
    let proj_shape = [s * s, HIDDEN_DIM];
    let reshaped = b.add_reshape(relu_out, &flat_shape);
    let out = b.add_linear(reshaped, proj_w, None, &proj_shape);
    let def = b.build(out).expect("valid backbone kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c, c, 3, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[c]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, c]), WEIGHT_MAG)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[c, s, s], 2.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through backbone");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline ResNet18 backbone IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "backbone lower must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "backbone upper must be finite, got {hi_max}"
    );
}

// ===========================================================================
// 2. Position encoding sinusoidal bounds (IBP)
// ===========================================================================

/// 2D sinusoidal positional encoding is bounded in [-1, 1]. Adding PE to
/// encoder features and projecting produces bounded output.
#[test]
fn test_position_encoding_sinusoidal_ibp() {
    let mut b = TensorBlockBuilder::new("tt_pipe_sinusoidal_pe");
    let features = b.add_input("features", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let pe = b.add_input("pos_enc", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let sum = b.add_binary_add(features, pe, &[ENC_SEQ_LEN, HIDDEN_DIM]);

    // Project the combined features
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(sum, proj_w, None, &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let pe_data = sinusoidal_pe_2d(FEAT_SIZE, FEAT_SIZE, HIDDEN_DIM);
    // Verify PE is bounded in [-1, 1]
    for &val in pe_data.iter() {
        assert!(
            (-1.01..=1.01).contains(&val),
            "2D PE component must be in [-1, 1], got {val}"
        );
    }

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline sinusoidal PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "PE lower must be finite");
    assert!(hi_max.is_finite(), "PE upper must be finite");
}

// ===========================================================================
// 3. Transformer encoder self-attention bounds (IBP + CROWN)
// ===========================================================================

/// Encoder layer at production dimensions: self-attention + LN + FFN + residual.
fn build_encoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_pipe_encoder_layer");
    let input = b.add_input("encoder_features", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let out = add_encoder_layer(&mut b, input, "enc_");
    b.build(out).expect("valid encoder layer kernel")
}

fn encoder_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_encoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_transformer_encoder_self_attention_ibp() {
    let def = build_encoder_layer_kernel();
    let bindings = encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder layer");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[ENC_SEQ_LEN, HIDDEN_DIM],
        "encoder output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline encoder self-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "encoder IBP lower must be finite");
    assert!(hi_max.is_finite(), "encoder IBP upper must be finite");
}

#[test]
fn test_transformer_encoder_self_attention_crown() {
    let def = build_encoder_layer_kernel();
    let bindings = encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "TT pipeline encoder self-attn CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. DETR decoder cross-attention bounds (IBP + CROWN)
// ===========================================================================

/// Cross-attention where Q comes from decoder queries and K/V from encoder memory.
fn build_decoder_cross_attn_kernel() -> TensorKernelDef {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("tt_pipe_decoder_cross_attn");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm on queries
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(queries, ln_eps, 1, ln_w, ln_b, &[NUM_QUERIES, HIDDEN_DIM]);

    // Cross-attention projections
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let k = b.add_linear(enc_mem, k_w, None, &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let v = b.add_linear(enc_mem, v_w, None, &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_QUERIES, HIDDEN_DIM],
    );
    let proj = b.add_linear(attn, o_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    // Residual
    let out = b.add_binary_add(queries, proj, &[NUM_QUERIES, HIDDEN_DIM]);
    b.build(out).expect("valid cross-attn kernel")
}

fn decoder_cross_attn_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]),
            0.5f32,
        )), // encoder_mem (constant)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_detr_decoder_cross_attention_ibp() {
    let def = build_decoder_cross_attn_kernel();
    let bindings = decoder_cross_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder cross-attn");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline decoder cross-attn IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "cross-attn IBP lower must be finite");
    assert!(hi_max.is_finite(), "cross-attn IBP upper must be finite");
}

#[test]
fn test_detr_decoder_cross_attention_crown() {
    let def = build_decoder_cross_attn_kernel();
    let bindings = decoder_cross_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "TT pipeline decoder cross-attn CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 5. Object query initialization and refinement (IBP)
// ===========================================================================

/// Learned object query embeddings are projected through Linear + LayerNorm
/// to produce initial decoder input. Bounds must remain finite and tight.
#[test]
fn test_object_query_init_refinement_ibp() {
    let mut b = TensorBlockBuilder::new("tt_pipe_query_init_refine");
    let queries = b.add_input("learned_queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // Linear projection for refinement
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(queries, proj_w, None, &[NUM_QUERIES, HIDDEN_DIM]);

    // LayerNorm for stabilization
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let out = b.add_layer_norm(projected, ln_eps, 1, ln_w, ln_b, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_QUERIES, HIDDEN_DIM],
        "query refinement output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline query init+refine IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "query init lower must be finite");
    assert!(hi_max.is_finite(), "query init upper must be finite");

    // Monotone tightening: tighter input -> tighter output
    let narrow_input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    let wide_width = bound_width(&output);
    let narrow_width = bound_width(&narrow_output);
    assert!(
        narrow_width <= wide_width + 1e-6,
        "monotone tightening: narrow_width={narrow_width} > wide_width={wide_width}"
    );
}

// ===========================================================================
// 6. Table cell classification head softmax bounds (IBP)
// ===========================================================================

/// Linear -> softmax produces class probabilities in [0, 1] for table cell
/// classification (table, row, column, cell, header, spanning, background).
#[test]
fn test_table_cell_classification_softmax_ibp() {
    let mut b = TensorBlockBuilder::new("tt_pipe_cls_softmax");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let out = b.add_softmax(logits, 1, &[NUM_QUERIES, NUM_CLASSES]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline cls softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -0.01,
        "softmax lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max <= 1.01, "softmax upper should be <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Table row/column regression sigmoid bounds (IBP)
// ===========================================================================

/// Linear -> sigmoid for normalized row/column coordinates in [0, 1].
/// This is the table structure regression head producing bounding boxes.
#[test]
fn test_table_row_col_regression_sigmoid_ibp() {
    let mut b = TensorBlockBuilder::new("tt_pipe_reg_sigmoid");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, BOX_DIM]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BOX_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline box sigmoid IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -0.01,
        "sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max <= 1.01, "sigmoid upper should be <= 1, got {hi_max}");
}

// ===========================================================================
// 8. Hungarian matching cost computation (IBP)
// ===========================================================================

/// Hungarian matching cost matrix entries are bounded. The matching cost
/// combines classification probability (sigmoid) and box regression (sigmoid).
/// Both heads produce [0, 1]-bounded outputs, so cost differences are finite.
#[test]
fn test_hungarian_matching_cost_computation_ibp() {
    let mut b = TensorBlockBuilder::new("tt_pipe_hungarian_cost");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let _cls_probs = b.add_sigmoid(cls_logits, &[NUM_QUERIES, NUM_CLASSES]);

    // Box regression head: Linear -> sigmoid
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let box_logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let _box_out = b.add_sigmoid(box_logits, &[NUM_QUERIES, BOX_DIM]);

    // Structure head: Linear -> sigmoid (table-specific)
    let struct_w = b.add_input("struct_weight", &[BOX_DIM, HIDDEN_DIM]);
    let struct_logits = b.add_linear(queries, struct_w, None, &[NUM_QUERIES, BOX_DIM]);
    let struct_out = b.add_sigmoid(struct_logits, &[NUM_QUERIES, BOX_DIM]);

    // Build through structure head (all share sigmoid -> [0, 1])
    let def = b.build(struct_out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BOX_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[BOX_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline hungarian cost IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // All three heads go through sigmoid -> [0, 1]
    assert!(
        lo_min >= -0.01,
        "cost sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "cost sigmoid upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Full encoder pipeline end-to-end (IBP + CROWN)
// ===========================================================================

/// Features + sinusoidal PE -> 2 encoder layers -> final LayerNorm.
/// Tests full encoder pipeline at production dimensions.
fn build_full_encoder_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_pipe_full_encoder");
    let features = b.add_input("features", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let pe = b.add_input("pos_enc", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let x = b.add_binary_add(features, pe, &[ENC_SEQ_LEN, HIDDEN_DIM]);

    // 2 encoder layers
    let x = add_encoder_layer(&mut b, x, "enc0_");
    let x = add_encoder_layer(&mut b, x, "enc1_");

    // Final LayerNorm
    let fn_ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let fn_ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let fn_eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(x, fn_eps, 1, fn_ln_w, fn_ln_b, &[ENC_SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid full encoder kernel")
}

fn full_encoder_pipeline_bindings() -> Vec<TensorParamBinding> {
    let pe_data = sinusoidal_pe_2d(FEAT_SIZE, FEAT_SIZE, HIDDEN_DIM);
    let mut bindings = vec![
        TensorParamBinding::Variable,                // features
        TensorParamBinding::ConstantTensor(pe_data), // pos_enc
    ];
    push_encoder_layer_bindings(&mut bindings); // enc0
    push_encoder_layer_bindings(&mut bindings); // enc1
                                                // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings
}

#[test]
fn test_full_encoder_pipeline_e2e_ibp() {
    let def = build_full_encoder_pipeline_kernel();
    let bindings = full_encoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full encoder");
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[ENC_SEQ_LEN, HIDDEN_DIM],
        "encoder pipeline output shape"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline full encoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "full encoder IBP lower must be finite");
    assert!(hi_max.is_finite(), "full encoder IBP upper must be finite");
}

#[test]
fn test_full_encoder_pipeline_e2e_crown() {
    let def = build_full_encoder_pipeline_kernel();
    let bindings = full_encoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "TT pipeline full encoder CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 10. Full decoder pipeline end-to-end (IBP + CROWN)
// ===========================================================================

/// Queries + encoder memory -> 2 decoder layers -> LayerNorm -> dual sigmoid heads.
/// Tests full decoder pipeline at production dimensions.
fn build_full_decoder_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("tt_pipe_full_decoder");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    // 2 decoder layers
    let x = add_decoder_layer(&mut b, queries, enc_mem, "dec0_");
    let x = add_decoder_layer(&mut b, x, enc_mem, "dec1_");

    // Final LayerNorm
    let fn_ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let fn_ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let fn_eps = b.add_input("final_eps", &[1]);
    let normed = b.add_layer_norm(x, fn_eps, 1, fn_ln_w, fn_ln_b, &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_logits = b.add_linear(normed, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let _cls_out = b.add_sigmoid(cls_logits, &[NUM_QUERIES, NUM_CLASSES]);

    // Box regression head: Linear -> sigmoid
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let box_logits = b.add_linear(normed, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let box_out = b.add_sigmoid(box_logits, &[NUM_QUERIES, BOX_DIM]);

    b.build(box_out).expect("valid full decoder kernel")
}

fn full_decoder_pipeline_bindings() -> Vec<TensorParamBinding> {
    let enc_mem_data = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,                     // queries
        TensorParamBinding::ConstantTensor(enc_mem_data), // encoder_mem
    ];
    push_decoder_layer_bindings(&mut bindings); // dec0
    push_decoder_layer_bindings(&mut bindings); // dec1
                                                // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    // Classification head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[NUM_CLASSES, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    // Box regression head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BOX_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings
}

#[test]
fn test_full_decoder_pipeline_e2e_ibp() {
    let def = build_full_decoder_pipeline_kernel();
    let bindings = full_decoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full decoder");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("TT pipeline full decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Output goes through sigmoid -> [0, 1]
    assert!(
        lo_min >= -0.01,
        "decoder pipeline sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "decoder pipeline sigmoid upper should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_full_decoder_pipeline_e2e_crown() {
    let def = build_full_decoder_pipeline_kernel();
    let bindings = full_decoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "TT pipeline full decoder CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
    // Sigmoid clamps to [0, 1] regardless of method
    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    assert!(
        lo_min >= -0.01,
        "CROWN decoder sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "CROWN decoder sigmoid upper should be <= 1, got {hi_max}"
    );
}
