// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated gpt-oss-20b composition tests.
//!
//! Verifies NY bounds propagation through 8 gpt-oss MoE sub-graphs.
//! Part of #4271: gpt-oss NY compose verification.

#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/gptoss_compose.rs"]
mod gptoss_helpers;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use gptoss_helpers::{
    attn_score_scale_bindings, attn_sink_bias_bindings, build_attn_score_scale,
    build_attn_sink_bias, build_clamped_swiglu, build_decoder_layer, build_embed_lookup,
    build_embed_rmsnorm_proj, build_expert_gate_split, build_full_attention, build_gqa_head_proj,
    build_gqa_kv_repeat, build_kv_cache_sliding, build_lm_head, build_moe_router,
    build_moe_weight_combine, build_mxfp4_dequant, build_output_pipeline, build_residual_add,
    build_rope_pair, build_sliding_attention, build_swiglu_expert, build_topk_expert_select,
    build_two_layer_residual, build_yarn_freq_mod, clamped_swiglu_bindings, decoder_layer_bindings,
    embed_lookup_bindings, embed_rmsnorm_proj_bindings, expert_gate_split_bindings,
    full_attention_bindings, gqa_head_proj_bindings, gqa_kv_repeat_bindings,
    kv_cache_sliding_bindings, lm_head_bindings, moe_router_bindings, moe_weight_combine_bindings,
    mxfp4_dequant_bindings, output_pipeline_bindings, residual_add_bindings, rope_pair_bindings,
    sliding_attention_bindings, swiglu_expert_bindings, topk_expert_select_bindings,
    two_layer_residual_bindings, yarn_freq_mod_bindings, ATTN_DIM, HALF_DIM, HEAD_DIM, HIDDEN_DIM,
    INTERMEDIATE, NUM_EXPERTS, SEQ_LEN, SLIDING_WINDOW, TOP_K,
};
use nn_verify::tensor_kernel_to_graph;

// ============================================================================
// 1. Embedding + RMSNorm + Q projection
// ============================================================================

#[test]
fn test_gptoss_embed_rmsnorm_proj_def_validates() {
    let def = build_embed_rmsnorm_proj();
    def.validate().expect("embed_rmsnorm_proj should validate");
}

#[test]
fn test_gptoss_embed_rmsnorm_proj_graph_builds() {
    let def = build_embed_rmsnorm_proj();
    let bindings = embed_rmsnorm_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // rms_norm + matmul = 2 ops; the constant norm/proj weights are folded into
    // their consumers rather than becoming graph nodes.
    assert!(
        graph.num_nodes() >= 2,
        "graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_embed_rmsnorm_proj_ibp_propagates() {
    let def = build_embed_rmsnorm_proj();
    let bindings = embed_rmsnorm_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through embed_rmsnorm_proj");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, ATTN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss embed_rmsnorm_proj IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min >= -100.0,
        "IBP lower should be >= -100, got {lo_min}"
    );
    assert!(hi_max <= 100.0, "IBP upper should be <= 100, got {hi_max}");
}

#[test]
fn test_gptoss_embed_rmsnorm_proj_crown_propagation() {
    let def = build_embed_rmsnorm_proj();
    let bindings = embed_rmsnorm_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss embed_rmsnorm_proj: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_embed_rmsnorm_proj_verify_and_record() {
    let def = build_embed_rmsnorm_proj();
    let bindings = embed_rmsnorm_proj_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_embed_rmsnorm_proj");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, ATTN_DIM]
    );
}

// ============================================================================
// 2. MoE Router
// ============================================================================

#[test]
fn test_gptoss_moe_router_def_validates() {
    let def = build_moe_router();
    def.validate().expect("moe_router should validate");
}

#[test]
fn test_gptoss_moe_router_graph_builds() {
    let def = build_moe_router();
    let bindings = moe_router_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // matmul + (const-bias broadcast folded into) add + softmax = 3 nodes after
    // constant folding of the broadcast bias.
    assert!(
        graph.num_nodes() >= 3,
        "graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_moe_router_ibp_propagates() {
    let def = build_moe_router();
    let bindings = moe_router_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through moe_router");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_EXPERTS]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss moe_router IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -0.01, "softmax lower >= -0.01, got {lo_min}");
    assert!(hi_max <= 1.01, "softmax upper <= 1.01, got {hi_max}");
}

#[test]
fn test_gptoss_moe_router_crown_propagation() {
    let def = build_moe_router();
    let bindings = moe_router_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss moe_router: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_moe_router_verify_and_record() {
    let def = build_moe_router();
    let bindings = moe_router_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_moe_router");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_EXPERTS]
    );
}

// ============================================================================
// 3. SwiGLU Expert
// ============================================================================

#[test]
fn test_gptoss_swiglu_expert_def_validates() {
    let def = build_swiglu_expert();
    def.validate().expect("swiglu_expert should validate");
}

#[test]
fn test_gptoss_swiglu_expert_graph_builds() {
    let def = build_swiglu_expert();
    let bindings = swiglu_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 7,
        "graph should have >= 7 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_swiglu_expert_ibp_propagates() {
    let def = build_swiglu_expert();
    let bindings = swiglu_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through swiglu_expert");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss swiglu_expert IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1000.0, "IBP lower >= -1000, got {lo_min}");
    assert!(hi_max <= 1000.0, "IBP upper <= 1000, got {hi_max}");
}

#[test]
fn test_gptoss_swiglu_expert_verify_and_record() {
    let def = build_swiglu_expert();
    let bindings = swiglu_expert_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_swiglu_expert");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ============================================================================
// 4. Decoder Layer
// ============================================================================

#[test]
fn test_gptoss_decoder_layer_def_validates() {
    let def = build_decoder_layer();
    def.validate().expect("decoder_layer should validate");
}

#[test]
fn test_gptoss_decoder_layer_graph_builds() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 6,
        "graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_decoder_layer_ibp_propagates() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder_layer");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss decoder_layer IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -500.0, "IBP lower >= -500, got {lo_min}");
    assert!(hi_max <= 500.0, "IBP upper <= 500, got {hi_max}");
}

#[test]
fn test_gptoss_decoder_layer_verify_and_record() {
    let def = build_decoder_layer();
    let bindings = decoder_layer_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_decoder_layer");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ============================================================================
// 5. Sliding Window Attention
// ============================================================================

#[test]
fn test_gptoss_sliding_attention_def_validates() {
    let def = build_sliding_attention();
    def.validate().expect("sliding_attention should validate");
}

#[test]
fn test_gptoss_sliding_attention_graph_builds() {
    let def = build_sliding_attention();
    let bindings = sliding_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_sliding_attention_ibp_propagates() {
    let def = build_sliding_attention();
    let bindings = sliding_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3 * SEQ_LEN * HEAD_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through sliding_attention");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HEAD_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss sliding_attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -50.0, "IBP lower >= -50, got {lo_min}");
    assert!(hi_max <= 50.0, "IBP upper <= 50, got {hi_max}");
}

#[test]
fn test_gptoss_sliding_attention_verify_and_record() {
    let def = build_sliding_attention();
    let bindings = sliding_attention_bindings();
    let input = uniform_bounds(&[3 * SEQ_LEN * HEAD_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_sliding_attention");
    assert_eq!(result.num_variables, 3);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM]
    );
}

// ============================================================================
// 6. Full Attention
// ============================================================================

#[test]
fn test_gptoss_full_attention_def_validates() {
    let def = build_full_attention();
    def.validate().expect("full_attention should validate");
}

#[test]
fn test_gptoss_full_attention_graph_builds() {
    let def = build_full_attention();
    let bindings = full_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_full_attention_ibp_propagates() {
    let def = build_full_attention();
    let bindings = full_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3 * SEQ_LEN * HEAD_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full_attention");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HEAD_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss full_attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -50.0, "IBP lower >= -50, got {lo_min}");
    assert!(hi_max <= 50.0, "IBP upper <= 50, got {hi_max}");
}

#[test]
fn test_gptoss_full_attention_verify_and_record() {
    let def = build_full_attention();
    let bindings = full_attention_bindings();
    let input = uniform_bounds(&[3 * SEQ_LEN * HEAD_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_full_attention");
    assert_eq!(result.num_variables, 3);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM]
    );
}

// ============================================================================
// 7. KV Cache Append + Sliding Window Eviction
// ============================================================================

#[test]
fn test_gptoss_kv_cache_sliding_def_validates() {
    let def = build_kv_cache_sliding();
    def.validate().expect("kv_cache_sliding should validate");
}

#[test]
fn test_gptoss_kv_cache_sliding_graph_builds() {
    let def = build_kv_cache_sliding();
    let bindings = kv_cache_sliding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 2,
        "graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_kv_cache_sliding_ibp_propagates() {
    let def = build_kv_cache_sliding();
    let bindings = kv_cache_sliding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[5 * HEAD_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through kv_cache_sliding");
    assert_eq!(output.lower_upper().0.shape(), &[SLIDING_WINDOW, HEAD_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss kv_cache_sliding IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(
        lo_min >= -1.0 - 1e-6,
        "should preserve lower bound, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "should preserve upper bound, got {hi_max}"
    );
}

#[test]
fn test_gptoss_kv_cache_sliding_verify_and_record() {
    let def = build_kv_cache_sliding();
    let bindings = kv_cache_sliding_bindings();
    let input = uniform_bounds(&[5 * HEAD_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_kv_cache_sliding");
    assert_eq!(result.num_variables, 2);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SLIDING_WINDOW, HEAD_DIM]
    );
}

// ============================================================================
// 8. MXFP4 Dequantization Error Bounds
// ============================================================================

#[test]
fn test_gptoss_mxfp4_dequant_def_validates() {
    let def = build_mxfp4_dequant();
    def.validate().expect("mxfp4_dequant should validate");
}

#[test]
fn test_gptoss_mxfp4_dequant_graph_builds() {
    let def = build_mxfp4_dequant();
    let bindings = mxfp4_dequant_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // After constant folding only the single variable-dependent op remains.
    assert!(
        graph.num_nodes() >= 1,
        "graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_mxfp4_dequant_ibp_propagates() {
    let def = build_mxfp4_dequant();
    let bindings = mxfp4_dequant_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[8], 6.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mxfp4_dequant");
    assert_eq!(output.lower_upper().0.shape(), &[8]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss mxfp4_dequant IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1.5 - 1e-4, "MXFP4 lower >= -1.5, got {lo_min}");
    assert!(hi_max <= 1.5 + 1e-4, "MXFP4 upper <= 1.5, got {hi_max}");
}

#[test]
fn test_gptoss_mxfp4_dequant_verify_and_record() {
    let def = build_mxfp4_dequant();
    let bindings = mxfp4_dequant_bindings();
    let input = uniform_bounds(&[8], 6.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_mxfp4_dequant");
    assert_eq!(result.num_variables, 1);
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[8]);
}

// ============================================================================
// 9. Residual Add
// ============================================================================

#[test]
fn test_gptoss_residual_add_def_validates() {
    let def = build_residual_add();
    def.validate().expect("residual_add should validate");
}

#[test]
fn test_gptoss_residual_add_graph_builds() {
    let def = build_residual_add();
    let bindings = residual_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 2,
        "graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_residual_add_ibp_propagates() {
    let def = build_residual_add();
    let bindings = residual_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Two equally-shaped [SEQ_LEN, HIDDEN_DIM] Variable inputs. The residual add
    // has no constant operand to infer a per-variable shape from, so the input
    // must use the explicit multi-variable layout [num_vars, SEQ_LEN, HIDDEN_DIM]
    // (sliced along the leading axis), not a flat [2*SEQ_LEN*HIDDEN_DIM] vector.
    let input = uniform_bounds(&[2, SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through residual_add");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss residual_add IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -2.0 - 1e-4, "IBP lower >= -2.0, got {lo_min}");
    assert!(hi_max <= 2.0 + 1e-4, "IBP upper <= 2.0, got {hi_max}");
}

#[test]
fn test_gptoss_residual_add_crown_propagation() {
    let def = build_residual_add();
    let bindings = residual_add_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2 * SEQ_LEN * HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss residual_add: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_residual_add_verify_and_record() {
    let def = build_residual_add();
    let bindings = residual_add_bindings();
    // Multi-variable layout [num_vars, SEQ_LEN, HIDDEN_DIM] (see the IBP test):
    // the residual add lacks a constant operand to infer per-variable shapes.
    let input = uniform_bounds(&[2, SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_residual_add");
    assert_eq!(result.num_variables, 2);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ============================================================================
// 10. LM Head (final linear projection)
// ============================================================================

#[test]
fn test_gptoss_lm_head_def_validates() {
    let def = build_lm_head();
    def.validate().expect("lm_head should validate");
}

#[test]
fn test_gptoss_lm_head_graph_builds() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // Single matmul; the constant lm-head weight is folded into the op.
    assert!(
        graph.num_nodes() >= 1,
        "graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_lm_head_ibp_propagates() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through lm_head");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 32]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss lm_head IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -100.0, "IBP lower >= -100, got {lo_min}");
    assert!(hi_max <= 100.0, "IBP upper <= 100, got {hi_max}");
}

#[test]
fn test_gptoss_lm_head_crown_propagation() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss lm_head: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_lm_head_verify_and_record() {
    let def = build_lm_head();
    let bindings = lm_head_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_lm_head");
    assert_eq!(result.num_variables, 1);
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ_LEN, 32]);
}

// ============================================================================
// 11. Embedding Table Lookup
// ============================================================================

#[test]
fn test_gptoss_embed_lookup_def_validates() {
    let def = build_embed_lookup();
    def.validate().expect("embed_lookup should validate");
}

#[test]
fn test_gptoss_embed_lookup_graph_builds() {
    let def = build_embed_lookup();
    let bindings = embed_lookup_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // Single embedding op; the constant embedding table is folded into the op.
    assert!(
        graph.num_nodes() >= 1,
        "graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_embed_lookup_ibp_propagates() {
    let def = build_embed_lookup();
    let bindings = embed_lookup_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Embedding indices are integers in [0, vocab_size). Use range [0, 31] for 32-vocab.
    let input = uniform_bounds(&[SEQ_LEN], 16.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through embed_lookup");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss embed_lookup IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -1.0, "embed lower >= -1.0, got {lo_min}");
    assert!(hi_max <= 1.0, "embed upper <= 1.0, got {hi_max}");
}

#[test]
fn test_gptoss_embed_lookup_crown_propagation() {
    let def = build_embed_lookup();
    let bindings = embed_lookup_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 16.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss embed_lookup: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_embed_lookup_verify_and_record() {
    let def = build_embed_lookup();
    let bindings = embed_lookup_bindings();
    let input = uniform_bounds(&[SEQ_LEN], 16.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_embed_lookup");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ============================================================================
// 12. RoPE Cos/Sin Pair Application
// ============================================================================

#[test]
fn test_gptoss_rope_pair_def_validates() {
    let def = build_rope_pair();
    def.validate().expect("rope_pair should validate");
}

#[test]
fn test_gptoss_rope_pair_graph_builds() {
    let def = build_rope_pair();
    let bindings = rope_pair_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 4,
        "graph should have >= 4 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_rope_pair_ibp_propagates() {
    let def = build_rope_pair();
    let bindings = rope_pair_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Two variable inputs: x and x_paired, each [SEQ_LEN, HALF_DIM]
    let input = uniform_bounds(&[2 * SEQ_LEN * HALF_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through rope_pair");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HALF_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss rope_pair IBP: bounds=[{lo_min}, {hi_max}]");
    // cos/sin bounded by [-1, 1], so output bounded by 2 * input range
    assert!(lo_min >= -3.0, "RoPE lower >= -3.0, got {lo_min}");
    assert!(hi_max <= 3.0, "RoPE upper <= 3.0, got {hi_max}");
}

#[test]
fn test_gptoss_rope_pair_crown_propagation() {
    let def = build_rope_pair();
    let bindings = rope_pair_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2 * SEQ_LEN * HALF_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss rope_pair: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_rope_pair_verify_and_record() {
    let def = build_rope_pair();
    let bindings = rope_pair_bindings();
    let input = uniform_bounds(&[2 * SEQ_LEN * HALF_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_rope_pair");
    assert_eq!(result.num_variables, 2);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HALF_DIM]
    );
}

// ============================================================================
// 13. MoE Weighted Expert Combination
// ============================================================================

#[test]
fn test_gptoss_moe_weight_combine_def_validates() {
    let def = build_moe_weight_combine();
    def.validate().expect("moe_weight_combine should validate");
}

#[test]
fn test_gptoss_moe_weight_combine_graph_builds() {
    let def = build_moe_weight_combine();
    let bindings = moe_weight_combine_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_moe_weight_combine_ibp_propagates() {
    let def = build_moe_weight_combine();
    let bindings = moe_weight_combine_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // 4 variable inputs: expert1[S,H], expert2[S,H], w1[S,1], w2[S,1]
    let total_elems = 2 * SEQ_LEN * HIDDEN_DIM + 2 * SEQ_LEN;
    let input = uniform_bounds(&[total_elems], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through moe_weight_combine");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss moe_weight_combine IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -10.0, "IBP lower >= -10, got {lo_min}");
    assert!(hi_max <= 10.0, "IBP upper <= 10, got {hi_max}");
}

#[test]
fn test_gptoss_moe_weight_combine_crown_propagation() {
    let def = build_moe_weight_combine();
    let bindings = moe_weight_combine_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let total_elems = 2 * SEQ_LEN * HIDDEN_DIM + 2 * SEQ_LEN;
    let input = uniform_bounds(&[total_elems], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss moe_weight_combine: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_moe_weight_combine_verify_and_record() {
    let def = build_moe_weight_combine();
    let bindings = moe_weight_combine_bindings();
    let total_elems = 2 * SEQ_LEN * HIDDEN_DIM + 2 * SEQ_LEN;
    let input = uniform_bounds(&[total_elems], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_moe_weight_combine");
    assert_eq!(result.num_variables, 4);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ============================================================================
// 14. GQA Head Projection
// ============================================================================

#[test]
fn test_gptoss_gqa_head_proj_def_validates() {
    let def = build_gqa_head_proj();
    def.validate().expect("gqa_head_proj should validate");
}

#[test]
fn test_gptoss_gqa_head_proj_graph_builds() {
    let def = build_gqa_head_proj();
    let bindings = gqa_head_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // matmul (const weight folded) + narrow = 2 nodes.
    assert!(
        graph.num_nodes() >= 2,
        "graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_gqa_head_proj_ibp_propagates() {
    let def = build_gqa_head_proj();
    let bindings = gqa_head_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through gqa_head_proj");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HEAD_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss gqa_head_proj IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min >= -100.0, "IBP lower >= -100, got {lo_min}");
    assert!(hi_max <= 100.0, "IBP upper <= 100, got {hi_max}");
}

#[test]
fn test_gptoss_gqa_head_proj_crown_propagation() {
    let def = build_gqa_head_proj();
    let bindings = gqa_head_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss gqa_head_proj: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_gqa_head_proj_verify_and_record() {
    let def = build_gqa_head_proj();
    let bindings = gqa_head_proj_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_gqa_head_proj");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM]
    );
}

// ============================================================================
// 15. Attention Score Scaling
// ============================================================================

#[test]
fn test_gptoss_attn_score_scale_def_validates() {
    let def = build_attn_score_scale();
    def.validate().expect("attn_score_scale should validate");
}

#[test]
fn test_gptoss_attn_score_scale_graph_builds() {
    let def = build_attn_score_scale();
    let bindings = attn_score_scale_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_attn_score_scale_ibp_propagates() {
    let def = build_attn_score_scale();
    let bindings = attn_score_scale_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Two variable inputs: q and k, each [SEQ_LEN, HEAD_DIM]
    let input = uniform_bounds(&[2 * SEQ_LEN * HEAD_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attn_score_scale");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, SEQ_LEN]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss attn_score_scale IBP: bounds=[{lo_min}, {hi_max}]");
    // Scores can be large negative due to causal mask (-1e9)
    assert!(lo_min >= -1.1e9, "IBP lower >= -1.1e9, got {lo_min}");
    assert!(hi_max <= 100.0, "IBP upper <= 100, got {hi_max}");
}

#[test]
fn test_gptoss_attn_score_scale_crown_propagation() {
    let def = build_attn_score_scale();
    let bindings = attn_score_scale_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2 * SEQ_LEN * HEAD_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss attn_score_scale: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_attn_score_scale_verify_and_record() {
    let def = build_attn_score_scale();
    let bindings = attn_score_scale_bindings();
    let input = uniform_bounds(&[2 * SEQ_LEN * HEAD_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_attn_score_scale");
    assert_eq!(result.num_variables, 2);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, SEQ_LEN]
    );
}

// ============================================================================
// 16. Expert Gate Split
// ============================================================================

#[test]
fn test_gptoss_expert_gate_split_def_validates() {
    let def = build_expert_gate_split();
    def.validate().expect("expert_gate_split should validate");
}

#[test]
fn test_gptoss_expert_gate_split_graph_builds() {
    let def = build_expert_gate_split();
    let bindings = expert_gate_split_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_expert_gate_split_ibp_propagates() {
    let def = build_expert_gate_split();
    let bindings = expert_gate_split_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, 2 * INTERMEDIATE], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through expert_gate_split");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, INTERMEDIATE]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss expert_gate_split IBP: bounds=[{lo_min}, {hi_max}]");
    // SiLU * up: bounded by product of ranges
    assert!(lo_min >= -10.0, "IBP lower >= -10, got {lo_min}");
    assert!(hi_max <= 10.0, "IBP upper <= 10, got {hi_max}");
}

#[test]
fn test_gptoss_expert_gate_split_crown_propagation() {
    let def = build_expert_gate_split();
    let bindings = expert_gate_split_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, 2 * INTERMEDIATE], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss expert_gate_split: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_expert_gate_split_verify_and_record() {
    let def = build_expert_gate_split();
    let bindings = expert_gate_split_bindings();
    let input = uniform_bounds(&[SEQ_LEN, 2 * INTERMEDIATE], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_expert_gate_split");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, INTERMEDIATE]
    );
}

// ============================================================================
// 17. Two-Layer Residual Stack
// ============================================================================

#[test]
fn test_gptoss_two_layer_residual_def_validates() {
    let def = build_two_layer_residual();
    def.validate().expect("two_layer_residual should validate");
}

#[test]
fn test_gptoss_two_layer_residual_graph_builds() {
    let def = build_two_layer_residual();
    let bindings = two_layer_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 12,
        "graph should have >= 12 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_two_layer_residual_ibp_propagates() {
    let def = build_two_layer_residual();
    let bindings = two_layer_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through two_layer_residual");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss two_layer_residual IBP: bounds=[{lo_min}, {hi_max}]");
    // Two residual layers can amplify bounds; use generous threshold
    assert!(lo_min >= -5000.0, "IBP lower >= -5000, got {lo_min}");
    assert!(hi_max <= 5000.0, "IBP upper <= 5000, got {hi_max}");
}

#[test]
fn test_gptoss_two_layer_residual_crown_propagation() {
    let def = build_two_layer_residual();
    let bindings = two_layer_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss two_layer_residual: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_two_layer_residual_verify_and_record() {
    let def = build_two_layer_residual();
    let bindings = two_layer_residual_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_two_layer_residual");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM]
    );
}

// ============================================================================
// 18. Output Pipeline (final norm + lm_head + softmax)
// ============================================================================

#[test]
fn test_gptoss_output_pipeline_def_validates() {
    let def = build_output_pipeline();
    def.validate().expect("output_pipeline should validate");
}

#[test]
fn test_gptoss_output_pipeline_graph_builds() {
    let def = build_output_pipeline();
    let bindings = output_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // rms_norm + matmul + softmax = 3 nodes (const norm/lm weights folded).
    assert!(
        graph.num_nodes() >= 3,
        "graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_output_pipeline_ibp_propagates() {
    let def = build_output_pipeline();
    let bindings = output_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through output_pipeline");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 32]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss output_pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    // Softmax output bounded in [0, 1]
    assert!(lo_min >= -0.01, "softmax lower >= -0.01, got {lo_min}");
    assert!(hi_max <= 1.01, "softmax upper <= 1.01, got {hi_max}");
}

#[test]
fn test_gptoss_output_pipeline_crown_propagation() {
    let def = build_output_pipeline();
    let bindings = output_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss output_pipeline: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_output_pipeline_verify_and_record() {
    let def = build_output_pipeline();
    let bindings = output_pipeline_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_output_pipeline");
    assert_eq!(result.num_variables, 1);
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ_LEN, 32]);
}

// ============================================================================
// 19. Attention Sink Bias
// ============================================================================

#[test]
fn test_gptoss_attn_sink_bias_def_validates() {
    let def = build_attn_sink_bias();
    def.validate().expect("attn_sink_bias should validate");
}

#[test]
fn test_gptoss_attn_sink_bias_graph_builds() {
    let def = build_attn_sink_bias();
    let bindings = attn_sink_bias_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // (const sink-bias broadcast folded into) add + softmax = 2 nodes.
    assert!(
        graph.num_nodes() >= 2,
        "graph should have >= 2 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_attn_sink_bias_ibp_propagates() {
    let def = build_attn_sink_bias();
    let bindings = attn_sink_bias_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, SEQ_LEN], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through attn_sink_bias");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, SEQ_LEN]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss attn_sink_bias IBP: bounds=[{lo_min}, {hi_max}]");
    // Softmax output bounded in [0, 1]
    assert!(lo_min >= -0.01, "softmax lower >= -0.01, got {lo_min}");
    assert!(hi_max <= 1.01, "softmax upper <= 1.01, got {hi_max}");
}

#[test]
fn test_gptoss_attn_sink_bias_crown_propagation() {
    let def = build_attn_sink_bias();
    let bindings = attn_sink_bias_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, SEQ_LEN], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss attn_sink_bias: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_attn_sink_bias_verify_and_record() {
    let def = build_attn_sink_bias();
    let bindings = attn_sink_bias_bindings();
    let input = uniform_bounds(&[SEQ_LEN, SEQ_LEN], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_attn_sink_bias");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, SEQ_LEN]
    );
}

// ============================================================================
// 20. YaRN Frequency Modulation
// ============================================================================

#[test]
fn test_gptoss_yarn_freq_mod_def_validates() {
    let def = build_yarn_freq_mod();
    def.validate().expect("yarn_freq_mod should validate");
}

#[test]
fn test_gptoss_yarn_freq_mod_graph_builds() {
    let def = build_yarn_freq_mod();
    let bindings = yarn_freq_mod_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // Single mul; the constant scale-factor broadcast is folded into the op.
    assert!(
        graph.num_nodes() >= 1,
        "graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_yarn_freq_mod_ibp_propagates() {
    let def = build_yarn_freq_mod();
    let bindings = yarn_freq_mod_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HALF_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through yarn_freq_mod");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HALF_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss yarn_freq_mod IBP: bounds=[{lo_min}, {hi_max}]");
    // Input [-1, 1] scaled by [1.0, 1.5] → output in [-1.5, 1.5]
    assert!(lo_min >= -2.0, "IBP lower >= -2.0, got {lo_min}");
    assert!(hi_max <= 2.0, "IBP upper <= 2.0, got {hi_max}");
}

#[test]
fn test_gptoss_yarn_freq_mod_crown_propagation() {
    let def = build_yarn_freq_mod();
    let bindings = yarn_freq_mod_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HALF_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss yarn_freq_mod: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_yarn_freq_mod_verify_and_record() {
    let def = build_yarn_freq_mod();
    let bindings = yarn_freq_mod_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HALF_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_yarn_freq_mod");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, HALF_DIM]
    );
}

// ============================================================================
// 21. Top-k Expert Selection
// ============================================================================

#[test]
fn test_gptoss_topk_expert_select_def_validates() {
    let def = build_topk_expert_select();
    def.validate().expect("topk_expert_select should validate");
}

#[test]
fn test_gptoss_topk_expert_select_graph_builds() {
    let def = build_topk_expert_select();
    let bindings = topk_expert_select_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 5,
        "graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_topk_expert_select_ibp_propagates() {
    let def = build_topk_expert_select();
    let bindings = topk_expert_select_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, NUM_EXPERTS], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through topk_expert_select");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, TOP_K]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss topk_expert_select IBP: bounds=[{lo_min}, {hi_max}]");
    // Product of softmax probs and their sum: bounded
    assert!(lo_min >= -1.0, "IBP lower >= -1.0, got {lo_min}");
    assert!(hi_max <= 2.0, "IBP upper <= 2.0, got {hi_max}");
}

#[test]
fn test_gptoss_topk_expert_select_crown_propagation() {
    let def = build_topk_expert_select();
    let bindings = topk_expert_select_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, NUM_EXPERTS], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss topk_expert_select: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_topk_expert_select_verify_and_record() {
    let def = build_topk_expert_select();
    let bindings = topk_expert_select_bindings();
    let input = uniform_bounds(&[SEQ_LEN, NUM_EXPERTS], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_topk_expert_select");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, TOP_K]
    );
}

// ============================================================================
// 22. Clamped SwiGLU Activation
// ============================================================================

#[test]
fn test_gptoss_clamped_swiglu_def_validates() {
    let def = build_clamped_swiglu();
    def.validate().expect("clamped_swiglu should validate");
}

#[test]
fn test_gptoss_clamped_swiglu_graph_builds() {
    let def = build_clamped_swiglu();
    let bindings = clamped_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 4,
        "graph should have >= 4 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_clamped_swiglu_ibp_propagates() {
    let def = build_clamped_swiglu();
    let bindings = clamped_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Two equally-shaped [SEQ_LEN, INTERMEDIATE] Variable inputs (gate, up). The
    // first op is a unary sigmoid(gate) with no constant operand to infer a
    // per-variable shape from, so the input must use the explicit multi-variable
    // layout [num_vars, SEQ_LEN, INTERMEDIATE], not a flat vector.
    let input = uniform_bounds(&[2, SEQ_LEN, INTERMEDIATE], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through clamped_swiglu");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, INTERMEDIATE]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss clamped_swiglu IBP: bounds=[{lo_min}, {hi_max}]");
    // SiLU(gate) * up: sigmoid in [0, 1], gate in [-1, 1], up in [-1, 1]
    assert!(lo_min >= -10.0, "IBP lower >= -10, got {lo_min}");
    assert!(hi_max <= 10.0, "IBP upper <= 10, got {hi_max}");
}

#[test]
fn test_gptoss_clamped_swiglu_crown_propagation() {
    let def = build_clamped_swiglu();
    let bindings = clamped_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[2 * SEQ_LEN * INTERMEDIATE], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss clamped_swiglu: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_clamped_swiglu_verify_and_record() {
    let def = build_clamped_swiglu();
    let bindings = clamped_swiglu_bindings();
    // Multi-variable layout [num_vars, SEQ_LEN, INTERMEDIATE] (see the IBP test):
    // the unary-first SwiGLU has no constant operand to infer per-variable shapes.
    let input = uniform_bounds(&[2, SEQ_LEN, INTERMEDIATE], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_clamped_swiglu");
    assert_eq!(result.num_variables, 2);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, INTERMEDIATE]
    );
}

// ============================================================================
// 23. GQA KV Repeat
// ============================================================================

#[test]
fn test_gptoss_gqa_kv_repeat_def_validates() {
    let def = build_gqa_kv_repeat();
    def.validate().expect("gqa_kv_repeat should validate");
}

#[test]
fn test_gptoss_gqa_kv_repeat_graph_builds() {
    let def = build_gqa_kv_repeat();
    let bindings = gqa_kv_repeat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    // Single concat op over the one variable input.
    assert!(
        graph.num_nodes() >= 1,
        "graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_gptoss_gqa_kv_repeat_ibp_propagates() {
    let def = build_gqa_kv_repeat();
    let bindings = gqa_kv_repeat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through gqa_kv_repeat");
    // repeat_kv duplicates along the head/feature axis: [SEQ_LEN, 2*HEAD_DIM].
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 2 * HEAD_DIM]);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("gptoss gqa_kv_repeat IBP: bounds=[{lo_min}, {hi_max}]");
    // Concat preserves input bounds
    assert!(
        lo_min >= -1.0 - 1e-6,
        "should preserve lower bound, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "should preserve upper bound, got {hi_max}"
    );
}

#[test]
fn test_gptoss_gqa_kv_repeat_crown_propagation() {
    let def = build_gqa_kv_repeat();
    let bindings = gqa_kv_repeat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);
    let (method, _output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("gptoss gqa_kv_repeat: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("fallback: {reason}");
    }
}

#[test]
fn test_gptoss_gqa_kv_repeat_verify_and_record() {
    let def = build_gqa_kv_repeat();
    let bindings = gqa_kv_repeat_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);
    let result = verify_and_assert(&def, &bindings, &input, "gptoss_gqa_kv_repeat");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ_LEN, 2 * HEAD_DIM]
    );
}
