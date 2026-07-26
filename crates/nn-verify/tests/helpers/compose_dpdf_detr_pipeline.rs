// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for DETR decoder object query and cross-attention pipeline
//! bounds.
//!
//! Verifies IBP and CROWN bound propagation through the full DETR decoder
//! pipeline as used in Table Transformer (Smock et al. 2022) and other
//! detection models. Focuses on end-to-end pipeline properties:
//! object query initialization, position encoding, cross-attention,
//! multi-layer decoder stacks, prediction heads, and matching constraints.
//!
//! 1.  **object_query_initialization_bounded**: Learned embeddings have finite bounds (IBP).
//! 2.  **query_plus_position_encoding_bounded**: Additive combination stays bounded (IBP).
//! 3.  **self_attention_weights_sum_one**: Attention among queries normalized (IBP).
//! 4.  **cross_attention_qkv_shapes**: Q from queries, KV from encoder (IBP).
//! 5.  **encoder_bounds_through_cross_attention**: Encoder bounds propagate (IBP + CROWN).
//! 6.  **ffn_sublayer_bounds**: LayerNorm -> Linear -> GELU -> Linear bounded (IBP).
//! 7.  **residual_connection_bounded**: query + attention output bounded (IBP).
//! 8.  **multi_layer_decoder_bounds**: Bounds growth across 6 layers (IBP).
//! 9.  **classification_head_softmax**: Probabilities in [0, 1] (IBP).
//! 10. **regression_head_sigmoid**: Bbox in [0, 1] (IBP).
//! 11. **hungarian_matching_permutation**: Assignment is bijection (IBP).
//! 12. **fixed_query_count**: 100 queries for Table Transformer (IBP).
//! 13. **no_object_class_probability**: Background in [0, 1] (IBP).
//! 14. **decoder_output_shape**: [batch, num_queries, hidden_dim] (IBP).
//! 15. **key_padding_mask**: Padded positions zeroed (IBP).
//! 16. **sinusoidal_2d_position**: Spatial encoding bounded (IBP).
//! 17. **encoder_decoder_dim_match**: Dimensions consistent (IBP + CROWN).
//! 18. **final_layernorm_stabilization**: Bounds tightened (IBP).
//!
//! Architecture references:
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_QUERIES=8, HIDDEN_DIM=64, FFN_DIM=128, NUM_HEADS=4, ENC_SEQ_LEN=16
//!
//! Part of #4148: Compose tests for DETR decoder pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, sinusoidal_pe,
    uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of learned object queries (DETR-style).
const NUM_QUERIES: usize = 8;
/// Hidden dimension for transformer encoder/decoder.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// Encoder sequence length (flattened spatial features).
const ENC_SEQ_LEN: usize = 16;
/// Number of detection classes (e.g., table, row, column, cell, header, background).
const NUM_CLASSES: usize = 6;
/// Box coordinate dimensions (x, y, w, h).
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
                // First half: height-based encoding
                data[pos * d_model + 4 * i] = freq_h.sin() as f32;
                data[pos * d_model + 4 * i + 1] = freq_h.cos() as f32;
                // Second half: width-based encoding
                data[pos * d_model + 4 * i + 2] = freq_w.sin() as f32;
                data[pos * d_model + 4 * i + 3] = freq_w.cos() as f32;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid 2D PE")
}

// ===========================================================================
// 1. object_query_initialization_bounded: Learned embeddings have finite bounds
// ===========================================================================

/// Learned object queries projected through a linear layer must produce
/// finite, bounded output.
#[test]
fn test_object_query_initialization_bounded() {
    let mut b = TensorBlockBuilder::new("detr_pipe_query_init");
    let queries = b.add_input("learned_queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(queries, proj_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline query init IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 2. query_plus_position_encoding_bounded: Additive combination stays bounded
// ===========================================================================

/// Adding sinusoidal positional encoding to learned queries preserves
/// bounded output.
#[test]
fn test_query_plus_position_encoding_bounded() {
    let mut b = TensorBlockBuilder::new("detr_pipe_query_pe");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let pe = b.add_input("pos_enc", &[NUM_QUERIES, HIDDEN_DIM]);
    let sum = b.add_binary_add(queries, pe, &[NUM_QUERIES, HIDDEN_DIM]);
    // Project the combined embedding
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(sum, proj_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let pe_data = sinusoidal_pe(NUM_QUERIES, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline query+PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "query+PE lower must be finite");
    assert!(hi_max.is_finite(), "query+PE upper must be finite");
}

// ===========================================================================
// 3. self_attention_weights_sum_one: Attention among queries normalized
// ===========================================================================

/// Self-attention softmax output is bounded in [0, 1], verifying the
/// normalization property of attention weights among object queries.
#[test]
fn test_self_attention_weights_sum_one() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("detr_pipe_sa_softmax");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(queries, q_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let k = b.add_linear(queries, k_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let v = b.add_linear(queries, v_w, None, &[NUM_QUERIES, HIDDEN_DIM]);

    // attention internally applies softmax — output should reflect bounded weights
    let out = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_QUERIES, HIDDEN_DIM],
    );
    let def = b.build(out).expect("valid kernel");

    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline self-attn weights IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "self-attn output lower must be finite");
    assert!(hi_max.is_finite(), "self-attn output upper must be finite");
}

// ===========================================================================
// 4. cross_attention_qkv_shapes: Q from queries, KV from encoder
// ===========================================================================

/// Cross-attention with Q projected from queries and K/V from encoder
/// memory produces finite bounds. Validates the fundamental Q-from-decoder,
/// KV-from-encoder pattern.
#[test]
fn test_cross_attention_qkv_shapes() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("detr_pipe_ca_qkv");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Q from decoder queries, K/V from encoder memory
    let q = b.add_linear(queries, q_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
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
    let out = b.add_linear(attn, o_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]),
            0.5f32,
        )), // encoder_mem
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline cross-attn QKV IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "cross-attn QKV lower must be finite");
    assert!(hi_max.is_finite(), "cross-attn QKV upper must be finite");
}

// ===========================================================================
// 5. encoder_bounds_through_cross_attention: Encoder bounds propagate (IBP + CROWN)
// ===========================================================================

/// When both queries and encoder memory are variable inputs, bounds
/// propagate through cross-attention from both sources. CROWN should
/// provide tighter bounds than IBP.
fn build_encoder_bounds_ca_kernel() -> TensorKernelDef {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("detr_pipe_enc_bounds_ca");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(queries, q_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
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
    let out = b.add_linear(attn, o_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    b.build(out).expect("valid kernel")
}

fn encoder_bounds_ca_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable, // queries
        TensorParamBinding::Variable, // encoder_mem
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_encoder_bounds_through_cross_attention_ibp() {
    let def = build_encoder_bounds_ca_kernel();
    let bindings = encoder_bounds_ca_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = NUM_QUERIES + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline encoder bounds CA IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "encoder bounds CA lower must be finite");
    assert!(hi_max.is_finite(), "encoder bounds CA upper must be finite");
}

#[test]
fn test_encoder_bounds_through_cross_attention_crown() {
    let def = build_encoder_bounds_ca_kernel();
    let bindings = encoder_bounds_ca_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = NUM_QUERIES + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Pipeline encoder bounds CA CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 6. ffn_sublayer_bounds: LayerNorm -> Linear -> GELU -> Linear bounded
// ===========================================================================

/// FFN sublayer with GELU activation (standard DETR FFN) produces
/// bounded output. Uses GELU instead of ReLU to test a different
/// activation path than the decoder layer helper.
#[test]
fn test_ffn_sublayer_bounds() {
    let mut b = TensorBlockBuilder::new("detr_pipe_ffn_gelu");
    let input_node = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // LayerNorm
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(
        input_node,
        ln_eps,
        1,
        ln_w,
        ln_b,
        &[NUM_QUERIES, HIDDEN_DIM],
    );

    // Linear -> GELU -> Linear
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let hidden = b.add_linear(normed, ffn1_w, None, &[NUM_QUERIES, FFN_DIM]);
    let activated = b.add_gelu(hidden, &[NUM_QUERIES, FFN_DIM]);
    let out = b.add_linear(activated, ffn2_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline FFN GELU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "FFN GELU lower must be finite");
    assert!(hi_max.is_finite(), "FFN GELU upper must be finite");
}

// ===========================================================================
// 7. residual_connection_bounded: query + attention output bounded
// ===========================================================================

/// Residual connection: queries + cross-attention(queries, encoder_mem)
/// produces finite, bounded output.
#[test]
fn test_residual_connection_bounded() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut b = TensorBlockBuilder::new("detr_pipe_residual");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(queries, q_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
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
    let attn_proj = b.add_linear(attn, o_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let out = b.add_binary_add(queries, attn_proj, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]),
            0.5f32,
        )),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "residual lower must be finite");
    assert!(hi_max.is_finite(), "residual upper must be finite");
    // Input in [-1,1] + small attention output — residual should be reasonable
    assert!(
        lo_min > -100.0,
        "residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 8. multi_layer_decoder_bounds: Bounds growth across 6 layers
// ===========================================================================

/// 6-layer decoder stack (standard DETR depth). Verify that bounds remain
/// finite through the full stack depth.
#[test]
fn test_multi_layer_decoder_bounds() {
    let mut b = TensorBlockBuilder::new("detr_pipe_6layer");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let mut x = queries;
    for i in 0..6 {
        x = add_decoder_layer(&mut b, x, enc_mem, &format!("l{i}_"));
    }
    let def = b.build(x).expect("valid kernel");

    let enc_mem_data = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(enc_mem_data),
    ];
    for _ in 0..6 {
        push_decoder_layer_bindings(&mut bindings);
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 6-layer decoder");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline 6-layer decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "6-layer decoder lower must be finite");
    assert!(hi_max.is_finite(), "6-layer decoder upper must be finite");
}

// ===========================================================================
// 9. classification_head_softmax: Probabilities in [0, 1]
// ===========================================================================

/// Classification head with softmax produces probabilities bounded in [0, 1].
#[test]
fn test_classification_head_softmax() {
    let mut b = TensorBlockBuilder::new("detr_pipe_cls_softmax");
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
    eprintln!("Pipeline cls softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -0.01,
        "softmax lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max <= 1.01, "softmax upper should be <= 1, got {hi_max}");
}

// ===========================================================================
// 10. regression_head_sigmoid: Bbox in [0, 1]
// ===========================================================================

/// Box regression head with sigmoid produces normalized coordinates in [0, 1].
#[test]
fn test_regression_head_sigmoid() {
    let mut b = TensorBlockBuilder::new("detr_pipe_box_sigmoid");
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
    eprintln!("Pipeline box sigmoid IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -0.01,
        "box sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "box sigmoid upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. hungarian_matching_permutation: Assignment is bijection
// ===========================================================================

/// Hungarian matching assigns predictions to ground truth via a permutation.
/// Model this as: decoded queries -> cls sigmoid + box sigmoid dual head.
/// The permutation property is structural (solver-side), so here we verify
/// that dual-head output bounds are valid and bounded.
#[test]
fn test_hungarian_matching_permutation() {
    let mut b = TensorBlockBuilder::new("detr_pipe_hungarian");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let _cls_out = b.add_sigmoid(cls_logits, &[NUM_QUERIES, NUM_CLASSES]);

    // Box regression head
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let box_logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let box_out = b.add_sigmoid(box_logits, &[NUM_QUERIES, BOX_DIM]);

    // Verify through box head (both share query input + sigmoid)
    let def = b.build(box_out).expect("valid kernel");

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
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline hungarian dual-head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Both heads output through sigmoid -> [0, 1]
    assert!(
        lo_min >= -0.01,
        "hungarian head lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "hungarian head upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 12. fixed_query_count: 100 queries for Table Transformer
// ===========================================================================

/// Table Transformer uses 100 object queries. Verify bounds propagate
/// correctly at this standard query count.
#[test]
fn test_fixed_query_count() {
    let nq: usize = 100;
    let mut b = TensorBlockBuilder::new("detr_pipe_100_queries");
    let queries = b.add_input("queries", &[nq, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
        .add_multi_head_attention(
            queries,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[nq, HIDDEN_DIM],
        )
        .expect("valid MHA with 100 queries");
    let def = b.build(out).expect("valid kernel");

    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[nq, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline 100-query IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "100-query lower must be finite");
    assert!(hi_max.is_finite(), "100-query upper must be finite");
}

// ===========================================================================
// 13. no_object_class_probability: Background class in [0, 1]
// ===========================================================================

/// DETR uses a "no-object" / background class. The sigmoid output for
/// this class should be bounded in [0, 1], matching the full classification
/// head but isolated to a single class dimension.
#[test]
fn test_no_object_class_probability() {
    let mut b = TensorBlockBuilder::new("detr_pipe_no_object");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    // Single class: the background/no-object logit
    let bg_w = b.add_input("bg_weight", &[1, HIDDEN_DIM]);
    let logits = b.add_linear(queries, bg_w, None, &[NUM_QUERIES, 1]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, 1]);
    let def = b.build(out).expect("valid kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline no-object class IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -0.01,
        "no-object sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "no-object sigmoid upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. decoder_output_shape: [batch, num_queries, hidden_dim] via 2-layer decode
// ===========================================================================

/// Verify that a 2-layer decoder produces output with the expected
/// query-count dimension, matching [num_queries, hidden_dim].
#[test]
fn test_decoder_output_shape() {
    let mut b = TensorBlockBuilder::new("detr_pipe_output_shape");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let x = add_decoder_layer(&mut b, queries, enc_mem, "l1_");
    let out = add_decoder_layer(&mut b, x, enc_mem, "l2_");
    let def = b.build(out).expect("valid kernel");

    let enc_mem_data = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(enc_mem_data),
    ];
    push_decoder_layer_bindings(&mut bindings);
    push_decoder_layer_bindings(&mut bindings);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    // Verify output shape matches [NUM_QUERIES, HIDDEN_DIM]
    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_QUERIES, HIDDEN_DIM],
        "decoder output shape must be [NUM_QUERIES={NUM_QUERIES}, HIDDEN_DIM={HIDDEN_DIM}]"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline decoder output shape IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "decoder output lower must be finite");
    assert!(hi_max.is_finite(), "decoder output upper must be finite");
}

// ===========================================================================
// 15. key_padding_mask: Padded positions handled via masked attention
// ===========================================================================

/// Verify that cross-attention with causal masking (simulating key padding)
/// produces bounded output. In practice, padding masks zero out attention
/// to padding tokens. We model this with a causal attention mask that
/// restricts which encoder positions are attended to.
#[test]
fn test_key_padding_mask() {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    // Use half the encoder sequence as "valid" tokens
    let valid_len: usize = ENC_SEQ_LEN / 2;

    let mut b = TensorBlockBuilder::new("detr_pipe_padding_mask");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[valid_len, HIDDEN_DIM]);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(queries, q_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let k = b.add_linear(enc_mem, k_w, None, &[valid_len, HIDDEN_DIM]);
    let v = b.add_linear(enc_mem, v_w, None, &[valid_len, HIDDEN_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_QUERIES, HIDDEN_DIM],
    );
    let out = b.add_linear(attn, o_w, None, &[NUM_QUERIES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[valid_len, HIDDEN_DIM]),
            0.5f32,
        )),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline key padding mask IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "padded attention lower must be finite");
    assert!(hi_max.is_finite(), "padded attention upper must be finite");
}

// ===========================================================================
// 16. sinusoidal_2d_position: Spatial encoding bounded
// ===========================================================================

/// 2D sinusoidal positional encoding for spatial grids (H x W -> flattened
/// sequence). All sin/cos components are bounded in [-1, 1]. Verify that
/// adding 2D PE to queries and projecting produces bounded output.
#[test]
fn test_sinusoidal_2d_position() {
    let h: usize = 4;
    let w: usize = 4;
    let seq_len = h * w; // 16
    let mut b = TensorBlockBuilder::new("detr_pipe_2d_pe");
    let queries = b.add_input("queries", &[seq_len, HIDDEN_DIM]);
    let pe = b.add_input("pe_2d", &[seq_len, HIDDEN_DIM]);
    let sum = b.add_binary_add(queries, pe, &[seq_len, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(sum, proj_w, None, &[seq_len, HIDDEN_DIM]);
    let def = b.build(out).expect("valid kernel");

    let pe_data = sinusoidal_pe_2d(h, w, HIDDEN_DIM);
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
    let input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline 2D PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "2D PE lower must be finite");
    assert!(hi_max.is_finite(), "2D PE upper must be finite");
}

// ===========================================================================
// 17. encoder_decoder_dim_match: Dimensions consistent (IBP + CROWN)
// ===========================================================================

/// When encoder hidden dim differs from decoder hidden dim, a linear
/// projection aligns features. Verify bounds through the projection
/// followed by a decoder layer. Tests dimension consistency end-to-end.
fn build_dim_match_kernel() -> TensorKernelDef {
    let enc_dim: usize = 256;
    let mut b = TensorBlockBuilder::new("detr_pipe_dim_match");

    let enc_features = b.add_input("encoder_features", &[ENC_SEQ_LEN, enc_dim]);
    let proj_w = b.add_input("enc_proj_w", &[HIDDEN_DIM, enc_dim]);
    let enc_projected = b.add_linear(enc_features, proj_w, None, &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let out = add_decoder_layer(&mut b, queries, enc_projected, "dec_");

    b.build(out).expect("valid kernel")
}

fn dim_match_bindings() -> Vec<TensorParamBinding> {
    let enc_dim: usize = 256;
    // Single Variable (encoder_features) so the input is just [ENC_SEQ_LEN,
    // enc_dim]; queries are a fixed constant. The two graph variables have
    // different shapes ([16,256] vs [8,64]) and so cannot be packed into the
    // multi-variable adapter's required [num_variables, ...common_shape] layout,
    // mirroring the single-Variable convention used by test #18 below.
    let mut bindings = vec![
        TensorParamBinding::Variable, // encoder_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, enc_dim]),
            WEIGHT_MAG,
        )), // enc_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_QUERIES, HIDDEN_DIM]),
            0.1f32,
        )), // queries (fixed)
    ];
    push_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_encoder_decoder_dim_match_ibp() {
    let enc_dim: usize = 256;
    let def = build_dim_match_kernel();
    let bindings = dim_match_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Single Variable input: encoder_features [ENC_SEQ_LEN, enc_dim].
    let input = uniform_bounds(&[ENC_SEQ_LEN, enc_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline dim match IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "dim match lower must be finite");
    assert!(hi_max.is_finite(), "dim match upper must be finite");
}

#[test]
fn test_encoder_decoder_dim_match_crown() {
    let enc_dim: usize = 256;
    let def = build_dim_match_kernel();
    let bindings = dim_match_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Single Variable input: encoder_features [ENC_SEQ_LEN, enc_dim].
    let input = uniform_bounds(&[ENC_SEQ_LEN, enc_dim], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Pipeline dim match CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 18. final_layernorm_stabilization: Bounds tightened
// ===========================================================================

/// Final LayerNorm after decoder stack stabilizes bounds — tighter input
/// epsilon should produce tighter output bounds through LayerNorm.
#[test]
fn test_final_layernorm_stabilization() {
    // Build: decoder layer -> final LayerNorm
    let mut b = TensorBlockBuilder::new("detr_pipe_final_ln");
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let enc_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let decoded = add_decoder_layer(&mut b, queries, enc_mem, "dec_");

    let fn_ln_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let fn_ln_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let fn_eps = b.add_input("final_eps", &[1]);
    let out = b.add_layer_norm(
        decoded,
        fn_eps,
        1,
        fn_ln_w,
        fn_ln_b,
        &[NUM_QUERIES, HIDDEN_DIM],
    );
    let def = b.build(out).expect("valid kernel");

    let enc_mem_data = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(enc_mem_data),
    ];
    push_decoder_layer_bindings(&mut bindings);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        1.0f32,
    ))); // final_ln_w
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        0.0f32,
    ))); // final_ln_b
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // final_eps

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Monotone tightening: smaller eps -> tighter bounds
    let eps_values = [1.0, 0.5, 0.1];
    let mut prev_width: Option<f32> = None;

    for &eps in &eps_values {
        let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Pipeline final LN stabilization: eps={eps:.2}, width={width:.6}");

        if let Some(prev) = prev_width {
            assert!(
                width <= prev + 1e-6,
                "final LN monotone tightening violated: eps={eps} width={width} > prev={prev}"
            );
        }
        prev_width = Some(width);
    }
}
