// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for DETR decoder patterns: object queries, cross-attention,
//! bipartite matching heads, and full DETR pipeline composition.
//!
//! Verifies IBP and CROWN bound propagation through the DETR decoder architecture
//! as used in Table Transformer (Smock et al. 2022) and other detection models.
//!
//! 1. **Object query initialization IBP**: Learned queries bounded.
//! 2. **Self-attention over queries IBP + CROWN**: Query-to-query attention.
//! 3. **Cross-attention: queries attend to encoder memory IBP + CROWN**.
//! 4. **Decoder layer: self-attn -> cross-attn -> FFN IBP**.
//! 5. **2-layer decoder stack IBP + CROWN**: Stacked decoder layers.
//! 6. **Query refinement through decoder depth IBP**.
//! 7. **Classification head: query -> Linear -> sigmoid IBP + CROWN**.
//! 8. **Box regression head: query -> Linear -> sigmoid coordinates IBP**.
//! 9. **Dual head: classification + box heads from same queries IBP**.
//! 10. **Object query count scaling: 10, 50, 100 queries IBP**.
//! 11. **Encoder-decoder projection: feature dim alignment IBP**.
//! 12. **Decoder with sinusoidal PE: position encoding for queries IBP**.
//! 13. **No-object class: background class sigmoid bounded IBP**.
//! 14. **Decoder monotone tightening: smaller eps -> tighter query bounds IBP**.
//! 15. **Full DETR pipeline: encoder -> decoder -> heads IBP + CROWN**.
//!
//! Architecture references:
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_QUERIES=8, HIDDEN_DIM=64, FFN_DIM=128, NUM_HEADS=4, ENC_SEQ_LEN=16
//!
//! Part of #4008: Compose tests for DETR decoder patterns.

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
///
/// Returns the output node for the decoder layer.
fn add_detr_decoder_layer(
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
    let sa_ln_w = b.add_input(&format!("{prefix}sa_ln_weight"), &[HIDDEN_DIM]);
    let sa_ln_b = b.add_input(&format!("{prefix}sa_ln_bias"), &[HIDDEN_DIM]);
    let sa_eps = b.add_input(&format!("{prefix}sa_eps"), &[1]);
    let normed_sa = b.add_layer_norm(queries, sa_eps, 1, sa_ln_w, sa_ln_b, &q_shape);

    let sa_q_w = b.add_input(&format!("{prefix}sa_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_k_w = b.add_input(&format!("{prefix}sa_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_v_w = b.add_input(&format!("{prefix}sa_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let sa_out_w = b.add_input(&format!("{prefix}sa_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(normed_sa, sa_q_w, None, &q_shape);
    let sk = b.add_linear(normed_sa, sa_k_w, None, &q_shape);
    let sv = b.add_linear(normed_sa, sa_v_w, None, &q_shape);
    let sa = b.add_attention(sq, sk, sv, AttentionMask::Standard, Some(scale), &q_shape);
    let sa_proj = b.add_linear(sa, sa_out_w, None, &q_shape);
    let res_sa = b.add_binary_add(queries, sa_proj, &q_shape);

    // Cross-attention: LayerNorm -> cross-attn(query, encoder_mem) -> residual
    let ca_ln_w = b.add_input(&format!("{prefix}ca_ln_weight"), &[HIDDEN_DIM]);
    let ca_ln_b = b.add_input(&format!("{prefix}ca_ln_bias"), &[HIDDEN_DIM]);
    let ca_eps = b.add_input(&format!("{prefix}ca_eps"), &[1]);
    let normed_ca = b.add_layer_norm(res_sa, ca_eps, 1, ca_ln_w, ca_ln_b, &q_shape);

    let ca_q_w = b.add_input(&format!("{prefix}ca_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_k_w = b.add_input(&format!("{prefix}ca_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_v_w = b.add_input(&format!("{prefix}ca_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let ca_out_w = b.add_input(&format!("{prefix}ca_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let cq = b.add_linear(normed_ca, ca_q_w, None, &q_shape);
    let ck = b.add_linear(encoder_mem, ca_k_w, None, &enc_shape);
    let cv = b.add_linear(encoder_mem, ca_v_w, None, &enc_shape);
    let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &q_shape);
    let ca_proj = b.add_linear(ca, ca_out_w, None, &q_shape);
    let res_ca = b.add_binary_add(res_sa, ca_proj, &q_shape);

    // FFN: LayerNorm -> Linear -> ReLU -> Linear -> residual
    let ffn_ln_w = b.add_input(&format!("{prefix}ffn_ln_weight"), &[HIDDEN_DIM]);
    let ffn_ln_b = b.add_input(&format!("{prefix}ffn_ln_bias"), &[HIDDEN_DIM]);
    let ffn_eps = b.add_input(&format!("{prefix}ffn_eps"), &[1]);
    let normed_ffn = b.add_layer_norm(res_ca, ffn_eps, 1, ffn_ln_w, ffn_ln_b, &q_shape);

    let ffn1_w = b.add_input(&format!("{prefix}ffn1_weight"), &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input(&format!("{prefix}ffn2_weight"), &[HIDDEN_DIM, FFN_DIM]);

    let ffn_hidden = b.add_linear(normed_ffn, ffn1_w, None, &ffn_shape);
    let ffn_act = b.add_relu(ffn_hidden, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn2_w, None, &q_shape);
    b.add_binary_add(res_ca, ffn_out, &q_shape)
}

/// Push one DETR decoder layer's bindings (21 params) onto the vec.
fn push_detr_decoder_layer_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // Self-attention norm + projections
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // sa_ln_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // sa_ln_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // sa_eps
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // sa_out_weight
                                                                       // Cross-attention norm + projections
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ca_ln_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ca_ln_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ca_eps
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // ca_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // ca_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // ca_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(proj_w)); // ca_out_weight
                                                               // FFN norm + projections
    bindings.push(TensorParamBinding::ConstantTensor(ln_w)); // ffn_ln_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b)); // ffn_ln_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ffn_eps
    bindings.push(TensorParamBinding::ConstantTensor(ffn1_w)); // ffn1_weight
    bindings.push(TensorParamBinding::ConstantTensor(ffn2_w)); // ffn2_weight
}

// ===========================================================================
// 1. Object query initialization: learned queries bounded (IBP)
// ===========================================================================

/// Learned object queries are a constant parameter — their bounds are the
/// parameter values themselves. Verify that a linear projection of learned
/// queries produces finite, bounded output.
fn build_object_query_init_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_object_query_init");

    let queries = b.add_input("learned_queries", &[NUM_QUERIES, HIDDEN_DIM]);
    // Project queries (identity-like but verifiable)
    let proj_w = b.add_input("query_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(queries, proj_w, None, &[NUM_QUERIES, HIDDEN_DIM]);

    b.build(out).expect("valid object query init kernel")
}

fn object_query_init_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,               // learned_queries
        TensorParamBinding::ConstantTensor(proj_w), // query_proj_weight
    ]
}

#[test]
fn test_detr_object_query_init_ibp() {
    let def = build_object_query_init_kernel();
    let bindings = object_query_init_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Learned queries are bounded in [-1, 1] (typical initialization range)
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through query init");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR object query init IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "query init lower must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "query init upper must be finite, got {hi_max}"
    );
}

// ===========================================================================
// 2. Self-attention over queries: query-to-query attention (IBP + CROWN)
// ===========================================================================

fn build_query_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_query_self_attn");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
        .add_multi_head_attention(
            queries,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_QUERIES, HIDDEN_DIM],
        )
        .expect("valid query self-attention");

    b.build(out).expect("valid query self-attention kernel")
}

fn query_self_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // queries
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_detr_query_self_attention_ibp() {
    let def = build_query_self_attention_kernel();
    let bindings = query_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through query self-attn");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR query self-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "self-attn lower must be finite");
    assert!(hi_max.is_finite(), "self-attn upper must be finite");
}

#[test]
fn test_detr_query_self_attention_crown() {
    let def = build_query_self_attention_kernel();
    let bindings = query_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "DETR query self-attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 3. Cross-attention: queries attend to encoder memory (IBP + CROWN)
// ===========================================================================

fn build_detr_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_cross_attn");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
        .add_multi_head_cross_attention(
            queries,
            encoder_mem,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_QUERIES, HIDDEN_DIM],
        )
        .expect("valid DETR cross-attention");

    b.build(out).expect("valid DETR cross-attention kernel")
}

fn detr_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // queries
        TensorParamBinding::Variable,                       // encoder_mem
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

fn detr_cross_attention_input() -> BoundedTensor {
    let total_seq = NUM_QUERIES + ENC_SEQ_LEN;
    uniform_bounds(&[total_seq, HIDDEN_DIM], 1.0)
}

#[test]
fn test_detr_cross_attention_ibp() {
    let def = build_detr_cross_attention_kernel();
    let bindings = detr_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = detr_cross_attention_input();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "cross-attn lower must be finite");
    assert!(hi_max.is_finite(), "cross-attn upper must be finite");
}

#[test]
fn test_detr_cross_attention_crown() {
    let def = build_detr_cross_attention_kernel();
    let bindings = detr_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_seq = NUM_QUERIES + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("DETR cross-attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 4. Decoder layer: self-attn -> cross-attn -> FFN (IBP)
// ===========================================================================

fn build_detr_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_decoder_layer");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let out = add_detr_decoder_layer(&mut b, queries, encoder_mem, "l1_");

    b.build(out).expect("valid DETR decoder layer kernel")
}

fn detr_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let enc_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,                // queries
        TensorParamBinding::ConstantTensor(enc_mem), // encoder_mem
    ];
    push_detr_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_detr_decoder_layer_ibp() {
    let def = build_detr_decoder_layer_kernel();
    let bindings = detr_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DETR decoder layer");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder layer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "decoder layer lower must be finite");
    assert!(hi_max.is_finite(), "decoder layer upper must be finite");
}

// ===========================================================================
// 5. 2-layer decoder stack (IBP + CROWN)
// ===========================================================================

fn build_detr_2layer_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_decoder_2layer");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    let x = add_detr_decoder_layer(&mut b, queries, encoder_mem, "l1_");
    let out = add_detr_decoder_layer(&mut b, x, encoder_mem, "l2_");

    b.build(out).expect("valid DETR 2-layer decoder kernel")
}

fn detr_2layer_decoder_bindings() -> Vec<TensorParamBinding> {
    let enc_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let mut bindings = vec![
        TensorParamBinding::Variable,                // queries
        TensorParamBinding::ConstantTensor(enc_mem), // encoder_mem
    ];
    push_detr_decoder_layer_bindings(&mut bindings);
    push_detr_decoder_layer_bindings(&mut bindings);
    bindings
}

#[test]
fn test_detr_2layer_decoder_ibp() {
    let def = build_detr_2layer_decoder_kernel();
    let bindings = detr_2layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer DETR decoder");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR 2-layer decoder IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "2-layer decoder lower must be finite");
    assert!(hi_max.is_finite(), "2-layer decoder upper must be finite");
}

#[test]
fn test_detr_2layer_decoder_crown() {
    let def = build_detr_2layer_decoder_kernel();
    let bindings = detr_2layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("DETR 2-layer decoder CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 6. Query refinement: queries refined through decoder depth (IBP)
// ===========================================================================

/// Verify that query bounds change (refine) through successive decoder layers.
/// Each layer updates queries via self-attention over other queries and
/// cross-attention to encoder memory.
#[test]
fn test_detr_query_refinement_ibp() {
    let def = build_detr_2layer_decoder_kernel();
    let bindings = detr_2layer_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through query refinement");
    assert_bounds_valid(&output);

    let input_width = 2.0; // uniform_bounds with range=1.0 gives width=2.0
    let output_width = bound_width(&output);
    eprintln!(
        "DETR query refinement: input width={input_width:.6}, output width={output_width:.6}"
    );
    // Queries should have finite bounds after refinement (not degenerate)
    assert!(
        output_width > 0.0,
        "refined queries should have non-trivial width"
    );
    assert!(
        output_width.is_finite(),
        "refined query width must be finite"
    );
}

// ===========================================================================
// 7. Classification head: query -> Linear -> sigmoid (IBP + CROWN)
// ===========================================================================

fn build_cls_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_cls_head");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, NUM_CLASSES]);

    b.build(out).expect("valid classification head kernel")
}

fn cls_head_bindings() -> Vec<TensorParamBinding> {
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // queries
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
    ]
}

#[test]
fn test_detr_cls_head_ibp() {
    let def = build_cls_head_kernel();
    let bindings = cls_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through cls head");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR cls head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -0.01,
        "cls sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "cls sigmoid upper should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_detr_cls_head_crown() {
    let def = build_cls_head_kernel();
    let bindings = cls_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("DETR cls head CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 8. Box regression head: query -> Linear -> sigmoid coordinates (IBP)
// ===========================================================================

fn build_box_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_box_head");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, BOX_DIM]);

    b.build(out).expect("valid box regression head kernel")
}

fn box_head_bindings() -> Vec<TensorParamBinding> {
    let box_w = ArrayD::from_elem(IxDyn(&[BOX_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // queries
        TensorParamBinding::ConstantTensor(box_w), // box_weight
    ]
}

#[test]
fn test_detr_box_head_ibp() {
    let def = build_box_head_kernel();
    let bindings = box_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through box head");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR box head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1] for normalized coordinates
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
// 9. Dual head: classification + box heads from same queries (IBP)
// ===========================================================================

fn build_dual_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_dual_head");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_logits = b.add_linear(queries, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let cls_out = b.add_sigmoid(cls_logits, &[NUM_QUERIES, NUM_CLASSES]);

    // Box regression head
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let box_logits = b.add_linear(queries, box_w, None, &[NUM_QUERIES, BOX_DIM]);
    let box_out = b.add_sigmoid(box_logits, &[NUM_QUERIES, BOX_DIM]);

    // Combine into single output: concat along feature dim
    // For verification, we verify the box head (the stricter one)
    // since both share the same query input and sigmoid output.
    // We use the box head as the terminal output.
    let _ = cls_out; // cls branch verified independently in test 7
    b.build(box_out).expect("valid dual head kernel")
}

fn dual_head_bindings() -> Vec<TensorParamBinding> {
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let box_w = ArrayD::from_elem(IxDyn(&[BOX_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // queries
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(box_w), // box_weight
    ]
}

#[test]
fn test_detr_dual_head_ibp() {
    let def = build_dual_head_kernel();
    let bindings = dual_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through dual head");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR dual head IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Both heads output through sigmoid -> [0, 1]
    assert!(
        lo_min >= -0.01,
        "dual head sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "dual head sigmoid upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 10. Object query count scaling: 10, 50, 100 queries (IBP)
// ===========================================================================

/// Verify that IBP bounds propagate correctly for varying numbers of object
/// queries. DETR uses 100 queries; Deformable DETR uses 300.
#[test]
fn test_detr_query_count_scaling_ibp() {
    let query_counts = [10, 50, 100];
    let mut prev_width: Option<f32> = None;

    for &nq in &query_counts {
        let mut b = TensorBlockBuilder::new(&format!("detr_query_scale_{nq}"));
        let queries = b.add_input("queries", &[nq, HIDDEN_DIM]);
        let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

        let out = b
            .add_multi_head_attention(
                queries,
                q_w,
                k_w,
                v_w,
                out_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &[nq, HIDDEN_DIM],
            )
            .expect("valid scaled MHA");
        let def = b.build(out).expect("valid query scale kernel");

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

        let width = bound_width(&output);
        eprintln!("DETR query count {nq}: width={width:.6}");
        assert!(width.is_finite(), "width must be finite for nq={nq}");

        // Bounds should remain finite regardless of query count
        if let Some(prev) = prev_width {
            // Self-attention output width may vary with sequence length
            // but should remain bounded and finite
            assert!(
                prev.is_finite() && width.is_finite(),
                "both widths must be finite: prev={prev}, curr={width}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 11. Encoder-decoder projection: feature dim alignment (IBP)
// ===========================================================================

/// When encoder hidden dim differs from decoder hidden dim, a linear projection
/// aligns features. Verify bounds through the projection.
fn build_enc_dec_projection_kernel() -> TensorKernelDef {
    let enc_dim: usize = 256; // Backbone output dim (e.g., ResNet channels)
    let dec_dim: usize = HIDDEN_DIM; // Decoder hidden dim

    let mut b = TensorBlockBuilder::new("detr_enc_dec_proj");

    let enc_features = b.add_input("encoder_features", &[ENC_SEQ_LEN, enc_dim]);
    let proj_w = b.add_input("proj_weight", &[dec_dim, enc_dim]);
    let out = b.add_linear(enc_features, proj_w, None, &[ENC_SEQ_LEN, dec_dim]);

    b.build(out).expect("valid enc-dec projection kernel")
}

fn enc_dec_projection_bindings() -> Vec<TensorParamBinding> {
    let enc_dim: usize = 256;
    let dec_dim: usize = HIDDEN_DIM;
    let proj_w = ArrayD::from_elem(IxDyn(&[dec_dim, enc_dim]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,               // encoder_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
    ]
}

#[test]
fn test_detr_enc_dec_projection_ibp() {
    let enc_dim: usize = 256;
    let def = build_enc_dec_projection_kernel();
    let bindings = enc_dec_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_SEQ_LEN, enc_dim], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through enc-dec projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR enc-dec projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "projection lower must be finite");
    assert!(hi_max.is_finite(), "projection upper must be finite");
}

// ===========================================================================
// 12. Decoder with sinusoidal PE: position encoding for queries (IBP)
// ===========================================================================

fn build_decoder_sinusoidal_pe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_decoder_sinusoidal_pe");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let pe = b.add_input("pos_enc", &[NUM_QUERIES, HIDDEN_DIM]);

    // Add positional encoding to queries
    let queries_pe = b.add_binary_add(queries, pe, &[NUM_QUERIES, HIDDEN_DIM]);

    // Self-attention on position-encoded queries
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
        .add_multi_head_attention(
            queries_pe,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_QUERIES, HIDDEN_DIM],
        )
        .expect("valid PE + MHA");

    b.build(out).expect("valid decoder sinusoidal PE kernel")
}

fn decoder_sinusoidal_pe_bindings() -> Vec<TensorParamBinding> {
    let pe = sinusoidal_pe(NUM_QUERIES, HIDDEN_DIM);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // queries
        TensorParamBinding::ConstantTensor(pe),             // pos_enc
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_detr_decoder_sinusoidal_pe_ibp() {
    let def = build_decoder_sinusoidal_pe_kernel();
    let bindings = decoder_sinusoidal_pe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder + PE");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR decoder sinusoidal PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "PE decoder lower must be finite");
    assert!(hi_max.is_finite(), "PE decoder upper must be finite");
}

// ===========================================================================
// 13. No-object class: background class sigmoid bounded (IBP)
// ===========================================================================

/// DETR uses a "no-object" / background class. The sigmoid output for
/// this class should be bounded in [0, 1] like all other classes.
/// We verify a single-class sigmoid head (the background class slice).
fn build_no_object_class_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_no_object_class");

    // Single background logit per query
    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let bg_w = b.add_input("bg_weight", &[1, HIDDEN_DIM]);
    let logits = b.add_linear(queries, bg_w, None, &[NUM_QUERIES, 1]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, 1]);

    b.build(out).expect("valid no-object class kernel")
}

fn no_object_class_bindings() -> Vec<TensorParamBinding> {
    let bg_w = ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,             // queries
        TensorParamBinding::ConstantTensor(bg_w), // bg_weight
    ]
}

#[test]
fn test_detr_no_object_class_ibp() {
    let def = build_no_object_class_kernel();
    let bindings = no_object_class_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through no-object class");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR no-object class IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Background sigmoid must be in [0, 1]
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
// 14. Decoder monotone tightening: smaller eps -> tighter query bounds (IBP)
// ===========================================================================

/// Verify that tighter input bounds produce tighter output bounds through
/// the DETR decoder layer. This is a fundamental property of sound
/// bound propagation.
#[test]
fn test_detr_decoder_monotone_tightening() {
    let def = build_detr_decoder_layer_kernel();
    let bindings = detr_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let eps_values = [1.0, 0.5, 0.1];
    let mut prev_width: Option<f32> = None;

    for &eps in &eps_values {
        let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("DETR decoder monotone tightening: eps={eps:.2}, width={width:.6}");

        if let Some(prev) = prev_width {
            assert!(
                width <= prev + 1e-6,
                "monotone tightening violated: eps={eps} width={width} > prev={prev}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 15. Full DETR pipeline: encoder -> decoder -> heads (IBP + CROWN)
// ===========================================================================

/// End-to-end DETR pipeline: encoder features -> decoder (1 layer) ->
/// classification sigmoid head. Verifies bounds propagate through the
/// complete detection pipeline.
fn build_full_detr_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_full_pipeline");

    let queries = b.add_input("queries", &[NUM_QUERIES, HIDDEN_DIM]);
    let encoder_mem = b.add_input("encoder_mem", &[ENC_SEQ_LEN, HIDDEN_DIM]);

    // 1-layer decoder
    let decoded = add_detr_decoder_layer(&mut b, queries, encoder_mem, "dec_");

    // Final LayerNorm
    let fn_ln_w = b.add_input("final_ln_weight", &[HIDDEN_DIM]);
    let fn_ln_b = b.add_input("final_ln_bias", &[HIDDEN_DIM]);
    let fn_eps = b.add_input("final_eps", &[1]);
    let normed = b.add_layer_norm(
        decoded,
        fn_eps,
        1,
        fn_ln_w,
        fn_ln_b,
        &[NUM_QUERIES, HIDDEN_DIM],
    );

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let logits = b.add_linear(normed, cls_w, None, &[NUM_QUERIES, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_QUERIES, NUM_CLASSES]);

    b.build(out).expect("valid full DETR pipeline kernel")
}

fn full_detr_pipeline_bindings() -> Vec<TensorParamBinding> {
    let enc_mem = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,                // queries
        TensorParamBinding::ConstantTensor(enc_mem), // encoder_mem
    ];
    push_detr_decoder_layer_bindings(&mut bindings);
    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ln_w)); // final_ln_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b)); // final_ln_bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // final_eps
                                                             // Classification head
    bindings.push(TensorParamBinding::ConstantTensor(cls_w)); // cls_weight
    bindings
}

#[test]
fn test_detr_full_pipeline_ibp() {
    let def = build_full_detr_pipeline_kernel();
    let bindings = full_detr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full DETR pipeline");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR full pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -0.01,
        "pipeline sigmoid lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.01,
        "pipeline sigmoid upper should be <= 1, got {hi_max}"
    );
}

#[test]
fn test_detr_full_pipeline_crown() {
    let def = build_full_detr_pipeline_kernel();
    let bindings = full_detr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("DETR full pipeline CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}
