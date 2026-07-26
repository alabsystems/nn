// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for multi-modal fusion patterns in vision-language models.
//!
//! Verifies IBP and CROWN bound propagation through multi-modal fusion
//! sub-blocks that combine vision and language features. These patterns
//! appear in VLMs like Granite-Docling (SigLIP2 vision + Granite LLM),
//! Qwen3-VL (vision encoder + language decoder), and similar architectures.
//!
//! ## Vision Feature Projection (tests 1-3)
//!
//! 1. Vision feature projection (Linear) IBP
//! 2. Vision feature projection CROWN
//! 3. Projection + LayerNorm composition IBP + CROWN
//!
//! ## Cross-Modal Attention (tests 4-6)
//!
//! 4. Cross-modal attention (image queries, text memory) IBP
//! 5. Cross-modal attention CROWN
//! 6. Cross-modal residual connection IBP
//!
//! ## Fusion Patterns (tests 7-10)
//!
//! 7. Vision-language concatenation IBP
//! 8. Gated fusion (sigmoid gate * vision + (1-sigmoid) * text) IBP
//! 9. Multi-scale vision feature fusion before projection IBP
//! 10. Interleaved vision-text token sequence IBP
//!
//! ## Composed Pipelines (tests 11-15)
//!
//! 11. Vision encoder -> projection -> decoder attention pipeline IBP
//! 12. Vision-language alignment bounds IBP
//! 13. Multi-modal monotone tightening IBP
//! 14. Full VLM path: vision encode -> project -> decode IBP
//! 15. Full VLM path CROWN
//!
//! Dimensions (small for fast verification, structurally representative):
//! - VIS_SEQ=4, TXT_SEQ=6, VIS_DIM=16, TXT_DIM=16, PROJ_DIM=16
//! - NUM_HEADS=4, HEAD_DIM=4, FFN_DIM=32
//!
//! Part of #3991: Multi-modal fusion compose tests for vision-language models.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Vision token sequence length (e.g. number of patches).
const VIS_SEQ: usize = 4;
/// Text token sequence length.
const TXT_SEQ: usize = 6;
/// Vision feature dimension.
const VIS_DIM: usize = 16;
/// Text / language model dimension.
const TXT_DIM: usize = 16;
/// Projected dimension for cross-modal alignment.
const PROJ_DIM: usize = 16;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = DIM / NUM_HEADS.
const HEAD_DIM: usize = PROJ_DIM / NUM_HEADS; // 4
/// FFN hidden dimension.
const FFN_DIM: usize = 32;
/// Weight magnitude for constant tensors.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Build a SwiGLU FFN block: gate_proj -> SiLU -> mul(up_proj) -> down_proj.
fn add_swiglu_ffn(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    seq: usize,
    dim: usize,
    ffn_dim: usize,
    gate_w: nn_dsl::TensorNodeId,
    up_w: nn_dsl::TensorNodeId,
    down_w: nn_dsl::TensorNodeId,
) -> nn_dsl::TensorNodeId {
    let gate = b.add_linear(input, gate_w, None, &[seq, ffn_dim]);
    let gate_act = add_silu(b, gate, &[seq, ffn_dim]);
    let up = b.add_linear(input, up_w, None, &[seq, ffn_dim]);
    let gated = b.add_binary_mul(gate_act, up, &[seq, ffn_dim]);
    b.add_linear(gated, down_w, None, &[seq, dim])
}

// ===========================================================================
// 1. Vision feature projection (Linear) IBP
// ===========================================================================

fn build_vision_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_vis_proj");

    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[PROJ_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[PROJ_DIM]);

    let out = b.add_linear(vis_feat, proj_w, Some(proj_b), &[VIS_SEQ, PROJ_DIM]);

    b.build(out).expect("valid vision projection kernel")
}

fn vision_projection_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[PROJ_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

#[test]
fn test_vision_projection_ibp() {
    let def = build_vision_projection_kernel();
    let bindings = vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 2. Vision feature projection CROWN
// ===========================================================================

#[test]
fn test_vision_projection_crown() {
    let def = build_vision_projection_kernel();
    let bindings = vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Vision projection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 3. Projection + LayerNorm composition IBP + CROWN
// ===========================================================================

fn build_proj_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_proj_ln");

    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[PROJ_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[PROJ_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[PROJ_DIM]);
    let ln_b = b.add_input("ln_bias", &[PROJ_DIM]);

    let projected = b.add_linear(vis_feat, proj_w, Some(proj_b), &[VIS_SEQ, PROJ_DIM]);
    let out = b.add_layer_norm(projected, ln_eps, 1, ln_w, ln_b, &[VIS_SEQ, PROJ_DIM]);

    b.build(out).expect("valid proj+LN kernel")
}

fn proj_layernorm_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[PROJ_DIM]), 0.0f32);
    let ln_eps = ArrayD::from_elem(IxDyn(&[1]), 1e-5f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[PROJ_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
        TensorParamBinding::ConstantTensor(ln_eps), // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias
    ]
}

#[test]
fn test_proj_layernorm_ibp() {
    let def = build_proj_layernorm_kernel();
    let bindings = proj_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through proj+LN");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Proj+LN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

#[test]
fn test_proj_layernorm_crown() {
    let def = build_proj_layernorm_kernel();
    let bindings = proj_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Proj+LN CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 4. Cross-modal attention (image queries, text memory) IBP
// ===========================================================================

fn build_cross_modal_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_cross_attn");

    // Vision tokens as queries, text tokens as key/value memory
    let vis_input = b.add_input("vis_tokens", &[VIS_SEQ, PROJ_DIM]);
    let txt_input = b.add_input("txt_tokens", &[TXT_SEQ, TXT_DIM]);
    let q_w = b.add_input("q_weight", &[PROJ_DIM, PROJ_DIM]);
    let k_w = b.add_input("k_weight", &[PROJ_DIM, TXT_DIM]);
    let v_w = b.add_input("v_weight", &[PROJ_DIM, TXT_DIM]);
    let out_w = b.add_input("out_weight", &[PROJ_DIM, PROJ_DIM]);

    let out = b
        .add_multi_head_cross_attention(
            vis_input,
            txt_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[VIS_SEQ, PROJ_DIM],
        )
        .expect("valid cross-attention");

    b.build(out).expect("valid cross-modal attention kernel")
}

fn cross_modal_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, PROJ_DIM]), WEIGHT_MAG);
    let kv_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, TXT_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // vis_tokens
        TensorParamBinding::Variable,                       // txt_tokens
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(kv_w.clone()),   // k_weight
        TensorParamBinding::ConstantTensor(kv_w),           // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

/// Build input bounds for cross-modal tests: vision and text tokens concatenated.
fn cross_modal_input_bounds(vis_range: f32, txt_range: f32) -> BoundedTensor {
    // Multi-variable kernels use a single concatenated input with slicing.
    // Total input = VIS_SEQ*PROJ_DIM + TXT_SEQ*TXT_DIM
    let vis_count = VIS_SEQ * PROJ_DIM;
    let txt_count = TXT_SEQ * TXT_DIM;
    let total = vis_count + txt_count;
    let mut lo = vec![0.0f32; total];
    let mut hi = vec![0.0f32; total];
    for i in 0..vis_count {
        lo[i] = -vis_range;
        hi[i] = vis_range;
    }
    for i in 0..txt_count {
        lo[vis_count + i] = -txt_range;
        hi[vis_count + i] = txt_range;
    }
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lo).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), hi).unwrap(),
    )
    .expect("valid cross-modal bounds")
}

#[test]
fn test_cross_modal_attention_ibp() {
    let def = build_cross_modal_attention_kernel();
    let bindings = cross_modal_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = cross_modal_input_bounds(1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-modal attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-modal attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 5. Cross-modal attention CROWN
// ===========================================================================

#[test]
fn test_cross_modal_attention_crown() {
    let def = build_cross_modal_attention_kernel();
    let bindings = cross_modal_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = cross_modal_input_bounds(0.5, 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Cross-modal attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 6. Cross-modal residual connection IBP
// ===========================================================================

fn build_cross_modal_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_cross_residual");

    let vis_input = b.add_input("vis_tokens", &[VIS_SEQ, PROJ_DIM]);
    let txt_input = b.add_input("txt_tokens", &[TXT_SEQ, TXT_DIM]);
    let q_w = b.add_input("q_weight", &[PROJ_DIM, PROJ_DIM]);
    let k_w = b.add_input("k_weight", &[PROJ_DIM, TXT_DIM]);
    let v_w = b.add_input("v_weight", &[PROJ_DIM, TXT_DIM]);
    let out_w = b.add_input("out_weight", &[PROJ_DIM, PROJ_DIM]);

    let attn = b
        .add_multi_head_cross_attention(
            vis_input,
            txt_input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[VIS_SEQ, PROJ_DIM],
        )
        .expect("valid cross-attention");

    // Residual: vis_input + cross_attn(vis, txt)
    let out = b.add_binary_add(vis_input, attn, &[VIS_SEQ, PROJ_DIM]);

    b.build(out).expect("valid cross-modal residual kernel")
}

fn cross_modal_residual_bindings() -> Vec<TensorParamBinding> {
    // Same as cross_modal_attention_bindings
    cross_modal_attention_bindings()
}

#[test]
fn test_cross_modal_residual_ibp() {
    let def = build_cross_modal_residual_kernel();
    let bindings = cross_modal_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = cross_modal_input_bounds(1.0, 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-modal residual");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-modal residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 7. Vision-language concatenation IBP
// ===========================================================================

fn build_vl_concat_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_vl_concat");

    // Project vision features to text dim, then concatenate along sequence axis
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[TXT_DIM, VIS_DIM]);

    let vis_projected = b.add_linear(vis_feat, proj_w, None, &[VIS_SEQ, TXT_DIM]);

    let txt_feat = b.add_input("txt_features", &[TXT_SEQ, TXT_DIM]);

    // Concatenate: [VIS_SEQ, TXT_DIM] ++ [TXT_SEQ, TXT_DIM] -> [VIS_SEQ+TXT_SEQ, TXT_DIM]
    let total_seq = VIS_SEQ + TXT_SEQ;
    let out = b.add_concat(&[vis_projected, txt_feat], 0, &[total_seq, TXT_DIM]);

    b.build(out).expect("valid VL concat kernel")
}

fn vl_concat_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[TXT_DIM, VIS_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::Variable,               // txt_features
    ]
}

fn vl_concat_input_bounds(range: f32) -> BoundedTensor {
    let vis_count = VIS_SEQ * VIS_DIM;
    let txt_count = TXT_SEQ * TXT_DIM;
    let total = vis_count + txt_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid concat bounds")
}

#[test]
fn test_vl_concat_ibp() {
    let def = build_vl_concat_kernel();
    let bindings = vl_concat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = vl_concat_input_bounds(1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through VL concat");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VL concat IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 8. Gated fusion (sigmoid gate * vision + (1-sigmoid) * text) IBP
// ===========================================================================

fn build_gated_fusion_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_gated_fusion");

    // Both inputs at same shape: [VIS_SEQ, PROJ_DIM]
    // Gate is learned from concatenation of both features
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, PROJ_DIM]);
    let txt_feat = b.add_input("txt_features", &[VIS_SEQ, PROJ_DIM]);
    let gate_w = b.add_input("gate_weight", &[PROJ_DIM, PROJ_DIM]);

    // Gate = sigmoid(Linear(vis_features))
    let gate_proj = b.add_linear(vis_feat, gate_w, None, &[VIS_SEQ, PROJ_DIM]);
    let gate = b.add_sigmoid(gate_proj, &[VIS_SEQ, PROJ_DIM]);

    // fused = gate * vis + (1 - gate) * txt
    // Approximate as: gate * vis + txt (simplified gating for tractable
    // verification bounds -- the sigmoid gate constrains the vision
    // contribution while the text path adds directly).
    let gated_vis = b.add_binary_mul(gate, vis_feat, &[VIS_SEQ, PROJ_DIM]);
    let out = b.add_binary_add(gated_vis, txt_feat, &[VIS_SEQ, PROJ_DIM]);

    b.build(out).expect("valid gated fusion kernel")
}

fn gated_fusion_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, PROJ_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::Variable,               // txt_features
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
    ]
}

fn gated_fusion_input_bounds(range: f32) -> BoundedTensor {
    let vis_count = VIS_SEQ * PROJ_DIM;
    let txt_count = VIS_SEQ * PROJ_DIM;
    let total = vis_count + txt_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid gated fusion bounds")
}

#[test]
fn test_gated_fusion_ibp() {
    let def = build_gated_fusion_kernel();
    let bindings = gated_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = gated_fusion_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through gated fusion");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Gated fusion IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
    // Sigmoid gate bounds output between some reasonable range
    assert!(
        hi_max - lo_min < 200.0,
        "gated fusion should produce bounded output, width={}",
        hi_max - lo_min
    );
}

// ===========================================================================
// 9. Multi-scale vision feature fusion before projection IBP
// ===========================================================================

fn build_multiscale_fusion_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_multiscale");

    // Two vision scales: coarse (half seq) and fine (full seq)
    // Both at PROJ_DIM, combine via 1x1 linear + add
    let fine_feat = b.add_input("fine_features", &[VIS_SEQ, VIS_DIM]);
    let coarse_feat = b.add_input("coarse_features", &[VIS_SEQ, VIS_DIM]);
    let merge_w = b.add_input("merge_weight", &[PROJ_DIM, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[PROJ_DIM, VIS_DIM]);

    // Project coarse to same space
    let coarse_proj = b.add_linear(coarse_feat, merge_w, None, &[VIS_SEQ, PROJ_DIM]);
    // Project fine
    let fine_proj = b.add_linear(fine_feat, proj_w, None, &[VIS_SEQ, PROJ_DIM]);
    // Add multi-scale features
    let out = b.add_binary_add(fine_proj, coarse_proj, &[VIS_SEQ, PROJ_DIM]);

    b.build(out).expect("valid multiscale fusion kernel")
}

fn multiscale_fusion_bindings() -> Vec<TensorParamBinding> {
    let merge_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                // fine_features
        TensorParamBinding::Variable,                // coarse_features
        TensorParamBinding::ConstantTensor(merge_w), // merge_weight
        TensorParamBinding::ConstantTensor(proj_w),  // proj_weight
    ]
}

fn multiscale_input_bounds(range: f32) -> BoundedTensor {
    let count = VIS_SEQ * VIS_DIM;
    let total = count * 2; // fine + coarse
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid multiscale bounds")
}

#[test]
fn test_multiscale_fusion_ibp() {
    let def = build_multiscale_fusion_kernel();
    let bindings = multiscale_fusion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = multiscale_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multiscale fusion");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multiscale fusion IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 10. Interleaved vision-text token sequence IBP
// ===========================================================================

fn build_interleaved_vl_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_interleaved");

    // Vision and text tokens projected to same dim, then concatenated and
    // processed by self-attention (simulating interleaved VL sequence).
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let txt_feat = b.add_input("txt_features", &[TXT_SEQ, TXT_DIM]);
    let vis_proj_w = b.add_input("vis_proj_weight", &[PROJ_DIM, VIS_DIM]);
    let txt_proj_w = b.add_input("txt_proj_weight", &[PROJ_DIM, TXT_DIM]);

    // Project both modalities to common dimension
    let vis_proj = b.add_linear(vis_feat, vis_proj_w, None, &[VIS_SEQ, PROJ_DIM]);
    let txt_proj = b.add_linear(txt_feat, txt_proj_w, None, &[TXT_SEQ, PROJ_DIM]);

    // Concatenate along sequence axis: [VIS_SEQ + TXT_SEQ, PROJ_DIM]
    let total_seq = VIS_SEQ + TXT_SEQ;
    let interleaved = b.add_concat(&[vis_proj, txt_proj], 0, &[total_seq, PROJ_DIM]);

    // Self-attention over the interleaved sequence
    let q_w = b.add_input("q_weight", &[PROJ_DIM, PROJ_DIM]);
    let k_w = b.add_input("k_weight", &[PROJ_DIM, PROJ_DIM]);
    let v_w = b.add_input("v_weight", &[PROJ_DIM, PROJ_DIM]);
    let out_w = b.add_input("out_weight", &[PROJ_DIM, PROJ_DIM]);

    let out = b
        .add_multi_head_attention(
            interleaved,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[total_seq, PROJ_DIM],
        )
        .expect("valid MHA");

    b.build(out).expect("valid interleaved VL kernel")
}

fn interleaved_vl_bindings() -> Vec<TensorParamBinding> {
    let vis_proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let txt_proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, TXT_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, PROJ_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // vis_features
        TensorParamBinding::Variable,                       // txt_features
        TensorParamBinding::ConstantTensor(vis_proj_w),     // vis_proj_weight
        TensorParamBinding::ConstantTensor(txt_proj_w),     // txt_proj_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

fn interleaved_input_bounds(range: f32) -> BoundedTensor {
    let vis_count = VIS_SEQ * VIS_DIM;
    let txt_count = TXT_SEQ * TXT_DIM;
    let total = vis_count + txt_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid interleaved bounds")
}

#[test]
fn test_interleaved_vl_ibp() {
    let def = build_interleaved_vl_kernel();
    let bindings = interleaved_vl_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = interleaved_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through interleaved VL");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Interleaved VL IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 11. Vision encoder -> projection -> decoder attention pipeline IBP
// ===========================================================================

fn build_vis_enc_proj_dec_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_enc_proj_dec");

    // Vision encoder: self-attention + FFN
    let vis_input = b.add_input("vis_input", &[VIS_SEQ, VIS_DIM]);
    let enc_q_w = b.add_input("enc_q_weight", &[VIS_DIM, VIS_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[VIS_DIM, VIS_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[VIS_DIM, VIS_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[VIS_DIM, VIS_DIM]);

    let enc_attn = b
        .add_multi_head_attention(
            vis_input,
            enc_q_w,
            enc_k_w,
            enc_v_w,
            enc_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[VIS_SEQ, VIS_DIM],
        )
        .expect("valid encoder MHA");

    // Residual + encoder FFN (simplified: Linear -> ReLU -> Linear)
    let enc_residual = b.add_binary_add(vis_input, enc_attn, &[VIS_SEQ, VIS_DIM]);
    let ffn_up_w = b.add_input("ffn_up_weight", &[FFN_DIM, VIS_DIM]);
    let ffn_down_w = b.add_input("ffn_down_weight", &[VIS_DIM, FFN_DIM]);
    let ffn_up = b.add_linear(enc_residual, ffn_up_w, None, &[VIS_SEQ, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_up, &[VIS_SEQ, FFN_DIM]);
    let ffn_down = b.add_linear(ffn_act, ffn_down_w, None, &[VIS_SEQ, VIS_DIM]);
    let encoder_out = b.add_binary_add(enc_residual, ffn_down, &[VIS_SEQ, VIS_DIM]);

    // Vision-language projection
    let proj_w = b.add_input("proj_weight", &[PROJ_DIM, VIS_DIM]);
    let projected = b.add_linear(encoder_out, proj_w, None, &[VIS_SEQ, PROJ_DIM]);

    // Decoder cross-attention: text queries attend to projected vision features
    let txt_input = b.add_input("txt_input", &[TXT_SEQ, TXT_DIM]);
    let dec_q_w = b.add_input("dec_q_weight", &[PROJ_DIM, TXT_DIM]);
    let dec_k_w = b.add_input("dec_k_weight", &[PROJ_DIM, PROJ_DIM]);
    let dec_v_w = b.add_input("dec_v_weight", &[PROJ_DIM, PROJ_DIM]);
    let dec_out_w = b.add_input("dec_out_weight", &[TXT_DIM, PROJ_DIM]);

    let dec_attn = b
        .add_multi_head_cross_attention(
            txt_input,
            projected,
            dec_q_w,
            dec_k_w,
            dec_v_w,
            dec_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[TXT_SEQ, TXT_DIM],
        )
        .expect("valid decoder cross-attention");

    // Decoder residual
    let out = b.add_binary_add(txt_input, dec_attn, &[TXT_SEQ, TXT_DIM]);

    b.build(out).expect("valid enc-proj-dec kernel")
}

fn vis_enc_proj_dec_bindings() -> Vec<TensorParamBinding> {
    let dim_w = ArrayD::from_elem(IxDyn(&[VIS_DIM, VIS_DIM]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, VIS_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[VIS_DIM, FFN_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let dec_q_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, TXT_DIM]), WEIGHT_MAG);
    let dec_kv_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, PROJ_DIM]), WEIGHT_MAG);
    let dec_out_w = ArrayD::from_elem(IxDyn(&[TXT_DIM, PROJ_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                         // vis_input
        TensorParamBinding::ConstantTensor(dim_w.clone()),    // enc_q_weight
        TensorParamBinding::ConstantTensor(dim_w.clone()),    // enc_k_weight
        TensorParamBinding::ConstantTensor(dim_w.clone()),    // enc_v_weight
        TensorParamBinding::ConstantTensor(dim_w),            // enc_out_weight
        TensorParamBinding::ConstantTensor(ffn_up_w),         // ffn_up_weight
        TensorParamBinding::ConstantTensor(ffn_down_w),       // ffn_down_weight
        TensorParamBinding::ConstantTensor(proj_w),           // proj_weight
        TensorParamBinding::Variable,                         // txt_input
        TensorParamBinding::ConstantTensor(dec_q_w),          // dec_q_weight
        TensorParamBinding::ConstantTensor(dec_kv_w.clone()), // dec_k_weight
        TensorParamBinding::ConstantTensor(dec_kv_w),         // dec_v_weight
        TensorParamBinding::ConstantTensor(dec_out_w),        // dec_out_weight
    ]
}

fn enc_proj_dec_input_bounds(range: f32) -> BoundedTensor {
    let vis_count = VIS_SEQ * VIS_DIM;
    let txt_count = TXT_SEQ * TXT_DIM;
    let total = vis_count + txt_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid enc-proj-dec bounds")
}

#[test]
fn test_vis_enc_proj_dec_ibp() {
    let def = build_vis_enc_proj_dec_kernel();
    let bindings = vis_enc_proj_dec_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = enc_proj_dec_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through enc->proj->dec");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Enc->Proj->Dec IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 12. Vision-language alignment bounds IBP
// ===========================================================================

fn build_vl_alignment_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_vl_alignment");

    // Project both modalities to shared space, compute alignment via matmul
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let txt_feat = b.add_input("txt_features", &[TXT_SEQ, TXT_DIM]);
    let vis_proj_w = b.add_input("vis_proj_weight", &[PROJ_DIM, VIS_DIM]);
    let txt_proj_w = b.add_input("txt_proj_weight", &[PROJ_DIM, TXT_DIM]);

    let vis_proj = b.add_linear(vis_feat, vis_proj_w, None, &[VIS_SEQ, PROJ_DIM]);
    let txt_proj = b.add_linear(txt_feat, txt_proj_w, None, &[TXT_SEQ, PROJ_DIM]);

    // Alignment scores: [VIS_SEQ, PROJ_DIM] @ [TXT_SEQ, PROJ_DIM]^T -> [VIS_SEQ, TXT_SEQ]
    // Use matmul with transpose_right=true for alignment matrix
    let alignment = b.add_matmul(vis_proj, txt_proj, true, None, &[VIS_SEQ, TXT_SEQ]);

    // Softmax over text dim to get attention weights in [0, 1]
    let out = b.add_softmax(alignment, 1, &[VIS_SEQ, TXT_SEQ]);

    b.build(out).expect("valid VL alignment kernel")
}

fn vl_alignment_bindings() -> Vec<TensorParamBinding> {
    let vis_proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let txt_proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, TXT_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                   // vis_features
        TensorParamBinding::Variable,                   // txt_features
        TensorParamBinding::ConstantTensor(vis_proj_w), // vis_proj_weight
        TensorParamBinding::ConstantTensor(txt_proj_w), // txt_proj_weight
    ]
}

fn vl_alignment_input_bounds(range: f32) -> BoundedTensor {
    let vis_count = VIS_SEQ * VIS_DIM;
    let txt_count = TXT_SEQ * TXT_DIM;
    let total = vis_count + txt_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid alignment bounds")
}

#[test]
fn test_vl_alignment_ibp() {
    let def = build_vl_alignment_kernel();
    let bindings = vl_alignment_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = vl_alignment_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VL alignment");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VL alignment IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-5, "softmax lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-5,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. Multi-modal monotone tightening IBP
// ===========================================================================

#[test]
fn test_multimodal_monotone_tightening() {
    let def = build_vision_projection_kernel();
    let bindings = vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide = uniform_bounds(&[VIS_SEQ, VIS_DIM], 2.0);
    let narrow = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let wide_out = graph.propagate_ibp(&wide).expect("IBP wide");
    let narrow_out = graph.propagate_ibp(&narrow).expect("IBP narrow");

    let wide_width = bound_width(&wide_out);
    let narrow_width = bound_width(&narrow_out);

    eprintln!(
        "Monotone tightening: wide eps width={wide_width:.6}, narrow eps width={narrow_width:.6}"
    );
    assert!(
        narrow_width <= wide_width + 1e-5,
        "tighter input must produce tighter output: narrow={narrow_width}, wide={wide_width}"
    );
}

// ===========================================================================
// 14. Full VLM path: vision encode -> project -> decode IBP
// ===========================================================================

fn build_full_vlm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mm_full_vlm");

    // Vision encoder: self-attention
    let vis_input = b.add_input("vis_input", &[VIS_SEQ, VIS_DIM]);
    let enc_q_w = b.add_input("enc_q_weight", &[VIS_DIM, VIS_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[VIS_DIM, VIS_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[VIS_DIM, VIS_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[VIS_DIM, VIS_DIM]);

    let enc_attn = b
        .add_multi_head_attention(
            vis_input,
            enc_q_w,
            enc_k_w,
            enc_v_w,
            enc_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[VIS_SEQ, VIS_DIM],
        )
        .expect("valid encoder MHA");
    let enc_out = b.add_binary_add(vis_input, enc_attn, &[VIS_SEQ, VIS_DIM]);

    // Vision-to-language projection with LayerNorm
    let proj_w = b.add_input("proj_weight", &[PROJ_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[PROJ_DIM]);
    let projected = b.add_linear(enc_out, proj_w, Some(proj_b), &[VIS_SEQ, PROJ_DIM]);

    // Decoder: text self-attention + cross-attention over projected vision
    let txt_input = b.add_input("txt_input", &[TXT_SEQ, TXT_DIM]);

    // Text self-attention
    let txt_q_w = b.add_input("txt_q_weight", &[TXT_DIM, TXT_DIM]);
    let txt_k_w = b.add_input("txt_k_weight", &[TXT_DIM, TXT_DIM]);
    let txt_v_w = b.add_input("txt_v_weight", &[TXT_DIM, TXT_DIM]);
    let txt_out_w = b.add_input("txt_out_weight", &[TXT_DIM, TXT_DIM]);

    let txt_self_attn = b
        .add_multi_head_attention(
            txt_input,
            txt_q_w,
            txt_k_w,
            txt_v_w,
            txt_out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[TXT_SEQ, TXT_DIM],
        )
        .expect("valid text self-attn");
    let txt_residual = b.add_binary_add(txt_input, txt_self_attn, &[TXT_SEQ, TXT_DIM]);

    // Cross-attention: text attends to vision
    let dec_q_w = b.add_input("dec_q_weight", &[PROJ_DIM, TXT_DIM]);
    let dec_k_w = b.add_input("dec_k_weight", &[PROJ_DIM, PROJ_DIM]);
    let dec_v_w = b.add_input("dec_v_weight", &[PROJ_DIM, PROJ_DIM]);
    let dec_out_w = b.add_input("dec_out_weight", &[TXT_DIM, PROJ_DIM]);

    let cross_attn = b
        .add_multi_head_cross_attention(
            txt_residual,
            projected,
            dec_q_w,
            dec_k_w,
            dec_v_w,
            dec_out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[TXT_SEQ, TXT_DIM],
        )
        .expect("valid decoder cross-attention");

    // Decoder SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, TXT_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, TXT_DIM]);
    let down_w = b.add_input("down_weight", &[TXT_DIM, FFN_DIM]);

    let dec_residual = b.add_binary_add(txt_residual, cross_attn, &[TXT_SEQ, TXT_DIM]);
    let ffn_out = add_swiglu_ffn(
        &mut b,
        dec_residual,
        TXT_SEQ,
        TXT_DIM,
        FFN_DIM,
        gate_w,
        up_w,
        down_w,
    );
    let out = b.add_binary_add(dec_residual, ffn_out, &[TXT_SEQ, TXT_DIM]);

    b.build(out).expect("valid full VLM kernel")
}

fn full_vlm_bindings() -> Vec<TensorParamBinding> {
    let vis_dim_w = ArrayD::from_elem(IxDyn(&[VIS_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[PROJ_DIM]), 0.0f32);
    let txt_dim_w = ArrayD::from_elem(IxDyn(&[TXT_DIM, TXT_DIM]), WEIGHT_MAG);
    let dec_q_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, TXT_DIM]), WEIGHT_MAG);
    let dec_kv_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, PROJ_DIM]), WEIGHT_MAG);
    let dec_out_w = ArrayD::from_elem(IxDyn(&[TXT_DIM, PROJ_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, TXT_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, TXT_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[TXT_DIM, FFN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                          // vis_input
        TensorParamBinding::ConstantTensor(vis_dim_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(vis_dim_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(vis_dim_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(vis_dim_w),         // enc_out_weight
        TensorParamBinding::ConstantTensor(proj_w),            // proj_weight
        TensorParamBinding::ConstantTensor(proj_b),            // proj_bias
        TensorParamBinding::Variable,                          // txt_input
        TensorParamBinding::ConstantTensor(txt_dim_w.clone()), // txt_q_weight
        TensorParamBinding::ConstantTensor(txt_dim_w.clone()), // txt_k_weight
        TensorParamBinding::ConstantTensor(txt_dim_w.clone()), // txt_v_weight
        TensorParamBinding::ConstantTensor(txt_dim_w),         // txt_out_weight
        TensorParamBinding::ConstantTensor(dec_q_w),           // dec_q_weight
        TensorParamBinding::ConstantTensor(dec_kv_w.clone()),  // dec_k_weight
        TensorParamBinding::ConstantTensor(dec_kv_w),          // dec_v_weight
        TensorParamBinding::ConstantTensor(dec_out_w),         // dec_out_weight
        TensorParamBinding::ConstantTensor(gate_w),            // gate_weight
        TensorParamBinding::ConstantTensor(up_w),              // up_weight
        TensorParamBinding::ConstantTensor(down_w),            // down_weight
    ]
}

fn full_vlm_input_bounds(range: f32) -> BoundedTensor {
    let vis_count = VIS_SEQ * VIS_DIM;
    let txt_count = TXT_SEQ * TXT_DIM;
    let total = vis_count + txt_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid full VLM bounds")
}

#[test]
fn test_full_vlm_ibp() {
    let def = build_full_vlm_kernel();
    let bindings = full_vlm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = full_vlm_input_bounds(1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through full VLM");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full VLM IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 15. Full VLM path CROWN
// ===========================================================================

#[test]
fn test_full_vlm_crown() {
    let def = build_full_vlm_kernel();
    let bindings = full_vlm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = full_vlm_input_bounds(0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Full VLM CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}
