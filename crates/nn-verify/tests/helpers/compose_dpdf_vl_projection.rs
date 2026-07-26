// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for vision-language projection and token merging patterns.
//!
//! Verifies IBP and CROWN bound propagation through vision-language projection
//! sub-blocks that map vision encoder features to LLM input space. These patterns
//! appear in VLMs like Granite-Docling (SigLIP2 -> Granite), Qwen3-VL (ViT ->
//! Qwen3 decoder), LLaVA (CLIP -> Vicuna), and similar architectures.
//!
//! ## Vision-to-Language Projection (tests 1-3)
//!
//! 1. Vision-to-language linear projection IBP bounds
//! 2. MLP projection (2-layer with GELU) IBP + CROWN bounds
//! 3. Token merging/pooling (spatial -> sequence) IBP bounds
//!
//! ## Normalization & Residual (tests 4-5)
//!
//! 4. Cross-modal LayerNorm before projection IBP + CROWN bounds
//! 5. Projection with residual connection IBP bounds
//!
//! ## Token Count & Spatial (tests 6-8)
//!
//! 6. Dynamic resolution token count IBP bounds
//! 7. Spatial token flattening (H*W -> seq_len) IBP bounds
//! 8. Perceiver resampler (fixed-length output) IBP bounds
//!
//! ## Compression & Fusion (tests 9-10)
//!
//! 9. Vision token compression ratio IBP bounds
//! 10. Multi-scale vision token fusion before projection IBP bounds
//!
//! ## Alignment & Embedding (tests 11-13)
//!
//! 11. Projection dimension alignment (vision_dim -> llm_dim) IBP + CROWN bounds
//! 12. Token type embedding addition IBP bounds
//! 13. Position embedding for projected tokens IBP bounds
//!
//! ## Composition (tests 14-15)
//!
//! 14. Projection + RoPE composition IBP bounds
//! 15. Full VL projection: vision encoder -> merge -> project -> LLM input IBP + CROWN
//!
//! Dimensions (small for fast verification, structurally representative):
//! - VIS_SEQ=8, VIS_DIM=16, LLM_DIM=32, PROJ_HIDDEN=24
//! - NUM_HEADS=4, HEAD_DIM=8, MERGED_SEQ=4
//!
//! Part of #4021: Compose tests for vision-language projection and token merging.

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

/// Vision token sequence length (e.g. number of patches from ViT encoder).
const VIS_SEQ: usize = 8;
/// Merged/compressed token sequence length (after token merging).
const MERGED_SEQ: usize = 4;
/// Vision encoder feature dimension.
const VIS_DIM: usize = 16;
/// LLM embedding dimension (target projection space).
const LLM_DIM: usize = 32;
/// MLP projection hidden dimension.
const PROJ_HIDDEN: usize = 24;
/// Number of attention heads (for perceiver resampler).
const NUM_HEADS: usize = 4;
/// Head dimension = LLM_DIM / NUM_HEADS.
const HEAD_DIM: usize = LLM_DIM / NUM_HEADS; // 8
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
// 1. Vision-to-language linear projection IBP bounds
// ===========================================================================

fn build_vl_linear_proj_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_linear_proj");

    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[LLM_DIM]);

    let out = b.add_linear(vis_feat, proj_w, Some(proj_b), &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid VL linear projection kernel")
}

fn vl_linear_proj_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

#[test]
fn test_vl_linear_proj_ibp() {
    let def = build_vl_linear_proj_kernel();
    let bindings = vl_linear_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VL linear projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 2. MLP projection (2-layer with GELU) IBP + CROWN bounds
// ===========================================================================

fn build_mlp_proj_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_mlp_proj");

    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let w1 = b.add_input("mlp_w1", &[PROJ_HIDDEN, VIS_DIM]);
    let b1 = b.add_input("mlp_b1", &[PROJ_HIDDEN]);
    let w2 = b.add_input("mlp_w2", &[LLM_DIM, PROJ_HIDDEN]);
    let b2 = b.add_input("mlp_b2", &[LLM_DIM]);

    // Linear -> GELU -> Linear
    let hidden = b.add_linear(vis_feat, w1, Some(b1), &[VIS_SEQ, PROJ_HIDDEN]);
    let activated = b.add_gelu(hidden, &[VIS_SEQ, PROJ_HIDDEN]);
    let out = b.add_linear(activated, w2, Some(b2), &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid MLP projection kernel")
}

fn mlp_proj_bindings() -> Vec<TensorParamBinding> {
    let w1 = ArrayD::from_elem(IxDyn(&[PROJ_HIDDEN, VIS_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[PROJ_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[LLM_DIM, PROJ_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,           // vis_features
        TensorParamBinding::ConstantTensor(w1), // mlp_w1
        TensorParamBinding::ConstantTensor(b1), // mlp_b1
        TensorParamBinding::ConstantTensor(w2), // mlp_w2
        TensorParamBinding::ConstantTensor(b2), // mlp_b2
    ]
}

#[test]
fn test_mlp_proj_ibp() {
    let def = build_mlp_proj_kernel();
    let bindings = mlp_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MLP projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MLP projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

#[test]
fn test_mlp_proj_crown() {
    let def = build_mlp_proj_kernel();
    let bindings = mlp_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("MLP projection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 3. Token merging/pooling (spatial -> sequence) IBP bounds
// ===========================================================================

fn build_token_merge_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_token_merge");

    // Simulate token merging by projecting VIS_SEQ tokens down to MERGED_SEQ
    // via a learned merge matrix: [VIS_SEQ, VIS_DIM] -> [MERGED_SEQ, VIS_DIM]
    // This is done as matmul: merge_w @ vis_feat where merge_w is [MERGED_SEQ, VIS_SEQ]
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let merge_w = b.add_input("merge_weight", &[MERGED_SEQ, VIS_SEQ]);

    let merged = b.add_matmul(merge_w, vis_feat, false, None, &[MERGED_SEQ, VIS_DIM]);

    // Project merged tokens to LLM dim
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let out = b.add_linear(merged, proj_w, None, &[MERGED_SEQ, LLM_DIM]);

    b.build(out).expect("valid token merge kernel")
}

fn token_merge_bindings() -> Vec<TensorParamBinding> {
    // Merge weights: softmax-like normalized rows (each row sums ~1)
    let merge_w = ArrayD::from_elem(IxDyn(&[MERGED_SEQ, VIS_SEQ]), 1.0 / VIS_SEQ as f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                // vis_features
        TensorParamBinding::ConstantTensor(merge_w), // merge_weight
        TensorParamBinding::ConstantTensor(proj_w),  // proj_weight
    ]
}

#[test]
fn test_token_merge_ibp() {
    let def = build_token_merge_kernel();
    let bindings = token_merge_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through token merge");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token merge IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 4. Cross-modal LayerNorm before projection IBP + CROWN bounds
// ===========================================================================

fn build_ln_proj_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_ln_proj");

    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[VIS_DIM]);
    let ln_b = b.add_input("ln_bias", &[VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[LLM_DIM]);

    // LayerNorm -> Linear projection
    let normed = b.add_layer_norm(vis_feat, ln_eps, 1, ln_w, ln_b, &[VIS_SEQ, VIS_DIM]);
    let out = b.add_linear(normed, proj_w, Some(proj_b), &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid LN+proj kernel")
}

fn ln_proj_bindings() -> Vec<TensorParamBinding> {
    let ln_eps = ArrayD::from_elem(IxDyn(&[1]), 1e-5f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[VIS_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[VIS_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(ln_eps), // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

#[test]
fn test_ln_proj_ibp() {
    let def = build_ln_proj_kernel();
    let bindings = ln_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through LN+proj");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LN+proj IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

#[test]
fn test_ln_proj_crown() {
    let def = build_ln_proj_kernel();
    let bindings = ln_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("LN+proj CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 5. Projection with residual connection IBP bounds
// ===========================================================================

fn build_proj_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_proj_residual");

    // Vision features projected to LLM dim with residual (requires same dim).
    // Use VIS_DIM as both input and output to enable residual addition.
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[VIS_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[VIS_DIM]);

    let projected = b.add_linear(vis_feat, proj_w, Some(proj_b), &[VIS_SEQ, VIS_DIM]);
    // Residual: vis_feat + projection(vis_feat)
    let out = b.add_binary_add(vis_feat, projected, &[VIS_SEQ, VIS_DIM]);

    b.build(out).expect("valid projection + residual kernel")
}

fn proj_residual_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[VIS_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[VIS_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

#[test]
fn test_proj_residual_ibp() {
    let def = build_proj_residual_kernel();
    let bindings = proj_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through proj+residual");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Proj+residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 6. Dynamic resolution token count IBP bounds
// ===========================================================================

fn build_dynamic_resolution_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_dynamic_res");

    // Model two different resolution inputs (small: VIS_SEQ/2, large: VIS_SEQ)
    // projected through the same linear layer. Test with the larger resolution.
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[LLM_DIM]);

    // Project to LLM dim (works for any sequence length)
    let projected = b.add_linear(vis_feat, proj_w, Some(proj_b), &[VIS_SEQ, LLM_DIM]);

    // Add a learned scale factor (simulating resolution-dependent normalization)
    let scale_w = b.add_input("scale_weight", &[LLM_DIM, LLM_DIM]);
    let out = b.add_linear(projected, scale_w, None, &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid dynamic resolution kernel")
}

fn dynamic_resolution_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    // Scale close to identity
    let scale_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, LLM_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                // vis_features
        TensorParamBinding::ConstantTensor(proj_w),  // proj_weight
        TensorParamBinding::ConstantTensor(proj_b),  // proj_bias
        TensorParamBinding::ConstantTensor(scale_w), // scale_weight
    ]
}

#[test]
fn test_dynamic_resolution_ibp() {
    let def = build_dynamic_resolution_kernel();
    let bindings = dynamic_resolution_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dynamic resolution");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dynamic resolution IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 7. Spatial token flattening (H*W -> seq_len) IBP bounds
// ===========================================================================

/// Spatial grid: 2x4 = 8 tokens (= VIS_SEQ).
const SPATIAL_H: usize = 2;
const SPATIAL_W: usize = 4;

fn build_spatial_flatten_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_spatial_flatten");

    // Input as 2D spatial grid [H, W, VIS_DIM]
    let vis_feat = b.add_input("vis_features", &[SPATIAL_H * SPATIAL_W, VIS_DIM]);

    // Reshape from [H*W, VIS_DIM] to [VIS_SEQ, VIS_DIM] (identity reshape)
    let flattened = b.add_reshape(vis_feat, &[VIS_SEQ, VIS_DIM]);

    // Project flattened tokens to LLM dim
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let proj_b = b.add_input("proj_bias", &[LLM_DIM]);
    let out = b.add_linear(flattened, proj_w, Some(proj_b), &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid spatial flatten kernel")
}

fn spatial_flatten_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // vis_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

#[test]
fn test_spatial_flatten_ibp() {
    let def = build_spatial_flatten_kernel();
    let bindings = spatial_flatten_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SPATIAL_H * SPATIAL_W, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through spatial flatten");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spatial flatten IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 8. Perceiver resampler (fixed-length output) IBP bounds
// ===========================================================================

fn build_perceiver_resampler_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_perceiver");

    // Learned query tokens cross-attend to vision features
    // Produces fixed MERGED_SEQ tokens regardless of input VIS_SEQ.
    //
    // `add_multi_head_cross_attention` requires the Q and KV inputs to share the
    // same feature dim (its reshape/head-split logic assumes a single model_dim).
    // A perceiver resampler maps VIS_DIM vision features into the LLM_DIM query
    // space, so project vision features VIS_DIM -> LLM_DIM up front; the cross
    // attention then operates entirely in LLM_DIM.
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let vis_proj_w = b.add_input("vis_proj_weight", &[LLM_DIM, VIS_DIM]);
    let vis_in_llm = b.add_linear(vis_feat, vis_proj_w, None, &[VIS_SEQ, LLM_DIM]);

    let query_tokens = b.add_input("query_tokens", &[MERGED_SEQ, LLM_DIM]);
    let q_w = b.add_input("q_weight", &[LLM_DIM, LLM_DIM]);
    let k_w = b.add_input("k_weight", &[LLM_DIM, LLM_DIM]);
    let v_w = b.add_input("v_weight", &[LLM_DIM, LLM_DIM]);
    let out_w = b.add_input("out_weight", &[LLM_DIM, LLM_DIM]);

    let out = b
        .add_multi_head_cross_attention(
            query_tokens,
            vis_in_llm,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[MERGED_SEQ, LLM_DIM],
        )
        .expect("valid perceiver cross-attention");

    b.build(out).expect("valid perceiver resampler kernel")
}

fn perceiver_resampler_bindings() -> Vec<TensorParamBinding> {
    let vis_proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    let query_tokens = ArrayD::from_elem(IxDyn(&[MERGED_SEQ, LLM_DIM]), WEIGHT_MAG);
    let q_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, LLM_DIM]), WEIGHT_MAG);
    let kv_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, LLM_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, LLM_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                     // vis_features
        TensorParamBinding::ConstantTensor(vis_proj_w),   // vis_proj_weight
        TensorParamBinding::ConstantTensor(query_tokens), // query_tokens
        TensorParamBinding::ConstantTensor(q_w),          // q_weight
        TensorParamBinding::ConstantTensor(kv_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(kv_w),         // v_weight
        TensorParamBinding::ConstantTensor(out_w),        // out_weight
    ]
}

fn perceiver_input_bounds(range: f32) -> BoundedTensor {
    // Vision features are variable, query tokens are constant
    // Total variable input = VIS_SEQ * VIS_DIM
    uniform_bounds(&[VIS_SEQ, VIS_DIM], range)
}

#[test]
fn test_perceiver_resampler_ibp() {
    let def = build_perceiver_resampler_kernel();
    let bindings = perceiver_resampler_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = perceiver_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through perceiver resampler");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Perceiver resampler IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 9. Vision token compression ratio IBP bounds
// ===========================================================================

fn build_compression_ratio_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_compression");

    // Two-stage compression: VIS_SEQ -> MERGED_SEQ via learned merge + MLP
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let merge_w = b.add_input("merge_weight", &[MERGED_SEQ, VIS_SEQ]);

    // Stage 1: merge tokens
    let merged = b.add_matmul(merge_w, vis_feat, false, None, &[MERGED_SEQ, VIS_DIM]);

    // Stage 2: MLP refinement on merged tokens
    let w1 = b.add_input("refine_w1", &[PROJ_HIDDEN, VIS_DIM]);
    let w2 = b.add_input("refine_w2", &[VIS_DIM, PROJ_HIDDEN]);

    let hidden = b.add_linear(merged, w1, None, &[MERGED_SEQ, PROJ_HIDDEN]);
    let activated = b.add_gelu(hidden, &[MERGED_SEQ, PROJ_HIDDEN]);
    let refined = b.add_linear(activated, w2, None, &[MERGED_SEQ, VIS_DIM]);

    // Residual: merged + refined
    let out = b.add_binary_add(merged, refined, &[MERGED_SEQ, VIS_DIM]);

    b.build(out).expect("valid compression ratio kernel")
}

fn compression_ratio_bindings() -> Vec<TensorParamBinding> {
    let merge_w = ArrayD::from_elem(IxDyn(&[MERGED_SEQ, VIS_SEQ]), 1.0 / VIS_SEQ as f32);
    let w1 = ArrayD::from_elem(IxDyn(&[PROJ_HIDDEN, VIS_DIM]), WEIGHT_MAG);
    let w2 = ArrayD::from_elem(IxDyn(&[VIS_DIM, PROJ_HIDDEN]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                // vis_features
        TensorParamBinding::ConstantTensor(merge_w), // merge_weight
        TensorParamBinding::ConstantTensor(w1),      // refine_w1
        TensorParamBinding::ConstantTensor(w2),      // refine_w2
    ]
}

#[test]
fn test_compression_ratio_ibp() {
    let def = build_compression_ratio_kernel();
    let bindings = compression_ratio_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through compression");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Compression ratio IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 10. Multi-scale vision token fusion before projection IBP bounds
// ===========================================================================

fn build_multiscale_proj_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_multiscale_proj");

    // Two feature scales: fine (VIS_SEQ tokens) and coarse (MERGED_SEQ tokens)
    // Coarse upsampled via linear to match fine, then fused + projected
    let fine_feat = b.add_input("fine_features", &[VIS_SEQ, VIS_DIM]);
    let coarse_feat = b.add_input("coarse_features", &[MERGED_SEQ, VIS_DIM]);

    // Upsample coarse: [MERGED_SEQ, VIS_DIM] -> [VIS_SEQ, VIS_DIM] via learned linear
    let upsample_w = b.add_input("upsample_weight", &[VIS_SEQ, MERGED_SEQ]);
    let coarse_up = b.add_matmul(upsample_w, coarse_feat, false, None, &[VIS_SEQ, VIS_DIM]);

    // Fuse: fine + upsampled_coarse
    let fused = b.add_binary_add(fine_feat, coarse_up, &[VIS_SEQ, VIS_DIM]);

    // Project fused to LLM dim
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let out = b.add_linear(fused, proj_w, None, &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid multiscale projection kernel")
}

fn multiscale_proj_bindings() -> Vec<TensorParamBinding> {
    let upsample_w = ArrayD::from_elem(IxDyn(&[VIS_SEQ, MERGED_SEQ]), 1.0 / MERGED_SEQ as f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                   // fine_features
        TensorParamBinding::Variable,                   // coarse_features
        TensorParamBinding::ConstantTensor(upsample_w), // upsample_weight
        TensorParamBinding::ConstantTensor(proj_w),     // proj_weight
    ]
}

fn multiscale_proj_input_bounds(range: f32) -> BoundedTensor {
    let fine_count = VIS_SEQ * VIS_DIM;
    let coarse_count = MERGED_SEQ * VIS_DIM;
    let total = fine_count + coarse_count;
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[total]), -range),
        ArrayD::from_elem(IxDyn(&[total]), range),
    )
    .expect("valid multiscale projection bounds")
}

#[test]
fn test_multiscale_proj_ibp() {
    let def = build_multiscale_proj_kernel();
    let bindings = multiscale_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = multiscale_proj_input_bounds(1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multiscale projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multiscale projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 11. Projection dimension alignment (vision_dim -> llm_dim) IBP + CROWN
// ===========================================================================

fn build_dim_alignment_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_dim_align");

    // Two-stage projection: VIS_DIM -> PROJ_HIDDEN -> LLM_DIM
    // Ensures dimension mismatch is bridged cleanly
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let w1 = b.add_input("align_w1", &[PROJ_HIDDEN, VIS_DIM]);
    let w2 = b.add_input("align_w2", &[LLM_DIM, PROJ_HIDDEN]);

    let intermediate = b.add_linear(vis_feat, w1, None, &[VIS_SEQ, PROJ_HIDDEN]);
    let activated = b.add_gelu(intermediate, &[VIS_SEQ, PROJ_HIDDEN]);
    let out = b.add_linear(activated, w2, None, &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid dimension alignment kernel")
}

fn dim_alignment_bindings() -> Vec<TensorParamBinding> {
    let w1 = ArrayD::from_elem(IxDyn(&[PROJ_HIDDEN, VIS_DIM]), WEIGHT_MAG);
    let w2 = ArrayD::from_elem(IxDyn(&[LLM_DIM, PROJ_HIDDEN]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,           // vis_features
        TensorParamBinding::ConstantTensor(w1), // align_w1
        TensorParamBinding::ConstantTensor(w2), // align_w2
    ]
}

#[test]
fn test_dim_alignment_ibp() {
    let def = build_dim_alignment_kernel();
    let bindings = dim_alignment_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dim alignment");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dim alignment IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

#[test]
fn test_dim_alignment_crown() {
    let def = build_dim_alignment_kernel();
    let bindings = dim_alignment_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Dim alignment CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 12. Token type embedding addition IBP bounds
// ===========================================================================

fn build_token_type_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_token_type");

    // Projected vision tokens + learned token type embedding
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let token_type_embed = b.add_input("token_type_embed", &[LLM_DIM]);

    let projected = b.add_linear(vis_feat, proj_w, None, &[VIS_SEQ, LLM_DIM]);

    // Add token type embedding (broadcast over sequence dimension).
    // `add_binary_add` requires matching ranks, so explicitly broadcast the
    // [LLM_DIM] embedding to [VIS_SEQ, LLM_DIM] (right-aligned, NumPy-style)
    // before the add.
    let token_type_bc = b.add_broadcast(token_type_embed, &[VIS_SEQ, LLM_DIM]);
    let out = b.add_binary_add(projected, token_type_bc, &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid token type embedding kernel")
}

fn token_type_embed_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    let token_type_embed = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.01f32);
    vec![
        TensorParamBinding::Variable,                         // vis_features
        TensorParamBinding::ConstantTensor(proj_w),           // proj_weight
        TensorParamBinding::ConstantTensor(token_type_embed), // token_type_embed
    ]
}

#[test]
fn test_token_type_embed_ibp() {
    let def = build_token_type_embed_kernel();
    let bindings = token_type_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through token type embed");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token type embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 13. Position embedding for projected tokens IBP bounds
// ===========================================================================

fn build_position_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_pos_embed");

    // Projected vision features + sinusoidal position embedding (constant)
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let pos_embed = b.add_input("pos_embed", &[VIS_SEQ, LLM_DIM]);

    let projected = b.add_linear(vis_feat, proj_w, None, &[VIS_SEQ, LLM_DIM]);
    // Add position embedding to projected tokens
    let out = b.add_binary_add(projected, pos_embed, &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid position embedding kernel")
}

fn position_embed_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    // Sinusoidal PE bounded in [-1, 1]
    let mut pe_data = vec![0.0f32; VIS_SEQ * LLM_DIM];
    for t in 0..VIS_SEQ {
        for i in 0..LLM_DIM / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / LLM_DIM as f64);
            pe_data[t * LLM_DIM + 2 * i] = freq.sin() as f32;
            pe_data[t * LLM_DIM + 2 * i + 1] = freq.cos() as f32;
        }
    }
    let pos_embed = ArrayD::from_shape_vec(IxDyn(&[VIS_SEQ, LLM_DIM]), pe_data).expect("valid PE");
    vec![
        TensorParamBinding::Variable,                  // vis_features
        TensorParamBinding::ConstantTensor(proj_w),    // proj_weight
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed
    ]
}

#[test]
fn test_position_embed_ibp() {
    let def = build_position_embed_kernel();
    let bindings = position_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through position embed");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Position embed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 14. Projection + RoPE composition IBP bounds
// ===========================================================================

fn build_proj_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_proj_rope");

    // Project vision features then apply RoPE-style rotation
    // RoPE: x_rot = x * cos(theta) + rotate_half(x) * sin(theta)
    // Approximate via: projected + cos_embed * projected + sin_embed * projected_rotated
    let vis_feat = b.add_input("vis_features", &[VIS_SEQ, VIS_DIM]);
    let proj_w = b.add_input("proj_weight", &[LLM_DIM, VIS_DIM]);
    let cos_embed = b.add_input("cos_embed", &[VIS_SEQ, LLM_DIM]);
    let sin_embed = b.add_input("sin_embed", &[VIS_SEQ, LLM_DIM]);

    let projected = b.add_linear(vis_feat, proj_w, None, &[VIS_SEQ, LLM_DIM]);

    // cos_part = projected * cos_embed
    let cos_part = b.add_binary_mul(projected, cos_embed, &[VIS_SEQ, LLM_DIM]);
    // sin_part = projected * sin_embed (simplified: real RoPE rotates half dims)
    let sin_part = b.add_binary_mul(projected, sin_embed, &[VIS_SEQ, LLM_DIM]);
    // result = cos_part + sin_part
    let out = b.add_binary_add(cos_part, sin_part, &[VIS_SEQ, LLM_DIM]);

    b.build(out).expect("valid proj+RoPE kernel")
}

fn proj_rope_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LLM_DIM, VIS_DIM]), WEIGHT_MAG);
    // RoPE cos/sin embeddings bounded in [-1, 1]
    let mut cos_data = vec![0.0f32; VIS_SEQ * LLM_DIM];
    let mut sin_data = vec![0.0f32; VIS_SEQ * LLM_DIM];
    for t in 0..VIS_SEQ {
        for i in 0..LLM_DIM / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / LLM_DIM as f64);
            cos_data[t * LLM_DIM + 2 * i] = freq.cos() as f32;
            cos_data[t * LLM_DIM + 2 * i + 1] = freq.cos() as f32;
            sin_data[t * LLM_DIM + 2 * i] = freq.sin() as f32;
            sin_data[t * LLM_DIM + 2 * i + 1] = (-freq.sin()) as f32;
        }
    }
    let cos_embed =
        ArrayD::from_shape_vec(IxDyn(&[VIS_SEQ, LLM_DIM]), cos_data).expect("valid cos");
    let sin_embed =
        ArrayD::from_shape_vec(IxDyn(&[VIS_SEQ, LLM_DIM]), sin_data).expect("valid sin");
    vec![
        TensorParamBinding::Variable,                  // vis_features
        TensorParamBinding::ConstantTensor(proj_w),    // proj_weight
        TensorParamBinding::ConstantTensor(cos_embed), // cos_embed
        TensorParamBinding::ConstantTensor(sin_embed), // sin_embed
    ]
}

#[test]
fn test_proj_rope_ibp() {
    let def = build_proj_rope_kernel();
    let bindings = proj_rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through proj+RoPE");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Proj+RoPE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

// ===========================================================================
// 15. Full VL projection: vision encoder -> merge -> project -> LLM input
// ===========================================================================

fn build_full_vl_proj_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_vlp_full_pipeline");

    // Stage 1: Vision encoder self-attention
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

    // Stage 2: Token merging (VIS_SEQ -> MERGED_SEQ)
    let merge_w = b.add_input("merge_weight", &[MERGED_SEQ, VIS_SEQ]);
    let merged = b.add_matmul(merge_w, enc_out, false, None, &[MERGED_SEQ, VIS_DIM]);

    // Stage 3: MLP projection to LLM dim with GELU
    let proj_w1 = b.add_input("proj_w1", &[PROJ_HIDDEN, VIS_DIM]);
    let proj_w2 = b.add_input("proj_w2", &[LLM_DIM, PROJ_HIDDEN]);
    let proj_b2 = b.add_input("proj_b2", &[LLM_DIM]);

    let hidden = b.add_linear(merged, proj_w1, None, &[MERGED_SEQ, PROJ_HIDDEN]);
    let activated = b.add_gelu(hidden, &[MERGED_SEQ, PROJ_HIDDEN]);
    let projected = b.add_linear(activated, proj_w2, Some(proj_b2), &[MERGED_SEQ, LLM_DIM]);

    // Stage 4: LayerNorm to stabilize before LLM input
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[LLM_DIM]);
    let ln_b = b.add_input("ln_bias", &[LLM_DIM]);
    let out = b.add_layer_norm(projected, ln_eps, 1, ln_w, ln_b, &[MERGED_SEQ, LLM_DIM]);

    b.build(out).expect("valid full VL projection kernel")
}

fn full_vl_proj_bindings() -> Vec<TensorParamBinding> {
    let vis_dim_w = ArrayD::from_elem(IxDyn(&[VIS_DIM, VIS_DIM]), WEIGHT_MAG);
    let merge_w = ArrayD::from_elem(IxDyn(&[MERGED_SEQ, VIS_SEQ]), 1.0 / VIS_SEQ as f32);
    let proj_w1 = ArrayD::from_elem(IxDyn(&[PROJ_HIDDEN, VIS_DIM]), WEIGHT_MAG);
    let proj_w2 = ArrayD::from_elem(IxDyn(&[LLM_DIM, PROJ_HIDDEN]), WEIGHT_MAG);
    let proj_b2 = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    let ln_eps = ArrayD::from_elem(IxDyn(&[1]), 1e-5f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[LLM_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                          // vis_input
        TensorParamBinding::ConstantTensor(vis_dim_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(vis_dim_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(vis_dim_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(vis_dim_w),         // enc_out_weight
        TensorParamBinding::ConstantTensor(merge_w),           // merge_weight
        TensorParamBinding::ConstantTensor(proj_w1),           // proj_w1
        TensorParamBinding::ConstantTensor(proj_w2),           // proj_w2
        TensorParamBinding::ConstantTensor(proj_b2),           // proj_b2
        TensorParamBinding::ConstantTensor(ln_eps),            // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),              // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),              // ln_bias
    ]
}

#[test]
fn test_full_vl_proj_ibp() {
    let def = build_full_vl_proj_kernel();
    let bindings = full_vl_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full VL projection");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full VL projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite, got {lo_min}");
    assert!(hi_max.is_finite(), "upper must be finite, got {hi_max}");
}

#[test]
fn test_full_vl_proj_crown() {
    let def = build_full_vl_proj_kernel();
    let bindings = full_vl_proj_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[VIS_SEQ, VIS_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Full VL projection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}
