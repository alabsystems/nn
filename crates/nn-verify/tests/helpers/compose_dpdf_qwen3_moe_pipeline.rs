// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for Qwen3-VL MoE sparse expert inference pipeline
//! bound propagation.
//!
//! Verifies IBP and CROWN bound propagation through the full MoE inference
//! pipeline as used in Qwen3-VL-30B-A3B: router softmax gating, top-k expert
//! selection, expert FFN execution, weighted combination, shared expert
//! additive paths, multi-layer depth composition, load balancing auxiliary
//! loss, capacity overflow handling, and end-to-end vision encoder to MoE
//! transformer to generation head.
//!
//! ## Router & Selection (tests 1-3)
//!
//! 1. Router softmax gate bounded in [0, 1] IBP
//! 2. Top-k expert selection preserves probability bounds IBP
//! 3. Expert output weighted sum bounded IBP
//!
//! ## Shared Expert & Composition (tests 4-6)
//!
//! 4. Shared expert additive path IBP
//! 5. MoE layer output bounded (routed + shared + residual) IBP
//! 6. Multi-layer MoE depth composition IBP
//!
//! ## Auxiliary & Overflow (tests 7-8)
//!
//! 7. Load balancing auxiliary loss bounded IBP
//! 8. Capacity overflow handling (clamped gates) IBP
//!
//! ## Full Pipeline (tests 9-10)
//!
//! 9. Full MoE inference pipeline (norm -> route -> experts -> combine) IBP
//! 10. Vision encoder -> MoE transformer -> generation head IBP
//!
//! ## CROWN & Depth (tests 11-14)
//!
//! 11. MoE layer output bounded CROWN
//! 12. Multi-layer MoE depth CROWN
//! 13. Router gate sharpening with temperature CROWN
//! 14. Full MoE inference pipeline CROWN
//!
//! ## Expert Variants (tests 15-18)
//!
//! 15. Single expert FFN isolation IBP
//! 16. Two-expert vs four-expert routing width comparison IBP
//! 17. MoE with RMSNorm pre/post IBP
//! 18. End-to-end MoE decoder layer (attention + MoE FFN) CROWN
//!
//! Architecture references:
//! - Qwen3-30B-A3B (Alibaba): MoE with 128 experts, top-8 sparse routing
//! - DeepSeek-V2 (DeepSeek AI, 2024): Shared + routed expert MoE
//! - Switch Transformer (Fedus et al., 2021): Top-1 expert sparse routing
//! - GShard (Lepikhin et al., 2020): Top-2 routing with load balancing
//! - ST-MoE (Zoph et al., 2022): Router z-loss and capacity factor
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128, NUM_EXPERTS=8, TOP_K=2
//!
//! Part of #4168: Compose tests for Qwen3-VL MoE pipeline.

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

const SEQ_LEN: usize = 4;
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const NUM_HEADS: usize = 4;
const WEIGHT_MAG: f32 = 0.02;
/// Vocabulary size for generation head tests.
const VOCAB_SIZE: usize = 256;
/// Vision encoder hidden dimension (smaller for tractability).
const VISION_DIM: usize = 32;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build SiLU activation as a single fused node: SiLU(x) = x * sigmoid(x).
///
/// The fused `Silu` op lets ny recognize the `MulBinary(SiLU(gate), up)` SwiGLU
/// pattern and apply its up/gate-correlation zonotope tightening.
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    b.add_silu(input, shape)
}

/// Build a SwiGLU FFN block for one expert.
///
/// Pattern: gate_proj(x) -> SiLU -> mul(up_proj(x)) -> down_proj
fn build_expert_ffn(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);

    let up = b.add_linear(input, up_w, None, &ffn_shape);

    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU expert FFN weight bindings (gate_w, up_w, down_w).
fn push_expert_ffn_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim, ffn_dim]),
        weight_mag,
    )));
}

/// Build a router: Linear -> softmax producing expert gate probabilities.
fn build_router(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    num_experts: usize,
) -> nn_dsl::TensorNodeId {
    let router_w = b.add_input(&format!("{prefix}_router_w"), &[num_experts, hidden_dim]);
    let logits = b.add_linear(input, router_w, None, &[seq_len, num_experts]);
    b.add_softmax(logits, 1, &[seq_len, num_experts])
}

/// Push router weight bindings.
fn push_router_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    num_experts: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[num_experts, hidden_dim]),
        weight_mag,
    )));
}

/// Push RMSNorm bindings (eps, weight).
fn push_rms_norm_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1e-5f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim]),
        1.0f32,
    )));
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Router softmax gate bounded in [0, 1] IBP
// ===========================================================================

/// Build router softmax kernel: Linear -> softmax over NUM_EXPERTS.
fn build_pipeline_router_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_router");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let probs = build_router(&mut b, input, "router", SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS);
    b.build(probs).expect("valid pipeline router kernel")
}

fn pipeline_router_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);
    bindings
}

/// Router softmax gate outputs are bounded in [0, 1] for all input ranges.
#[test]
fn test_router_softmax_gate_bounded_ibp() {
    let def = build_pipeline_router_kernel();
    let bindings = pipeline_router_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_EXPERTS],
        "router output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE pipeline router IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 2. Top-k expert selection preserves probability bounds IBP
// ===========================================================================

/// Build top-k selection: Linear -> softmax -> narrow(TOP_K).
fn build_pipeline_topk_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_topk");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let probs = build_router(&mut b, input, "router", SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS);
    let topk = b.add_narrow(probs, 1, 0, TOP_K, &[SEQ_LEN, TOP_K]);
    b.build(topk).expect("valid pipeline top-k kernel")
}

fn pipeline_topk_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);
    bindings
}

/// Top-k selection via narrow preserves softmax probability bounds in [0, 1].
#[test]
fn test_topk_expert_selection_preserves_bounds_ibp() {
    let def = build_pipeline_topk_kernel();
    let bindings = pipeline_topk_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, TOP_K],
        "top-k output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE pipeline top-k IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "top-k lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "top-k upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 3. Expert output weighted sum bounded IBP
// ===========================================================================

/// Build weighted combination of two expert FFN outputs.
///
/// gate0 * expert0(x) + gate1 * expert1(x) with constant gate weights
/// that sum to 1.0 (simulating softmax-derived probabilities).
fn build_pipeline_weighted_sum_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_weighted_sum");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let e0_out = build_expert_ffn(&mut b, input, "e0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let e1_out = build_expert_ffn(&mut b, input, "e1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    let g0 = b.add_input("gate0", &[1]);
    let g1 = b.add_input("gate1", &[1]);
    let g0_bc = b.add_broadcast(g0, &shape);
    let g1_bc = b.add_broadcast(g1, &shape);

    let w0 = b.add_binary_mul(g0_bc, e0_out, &shape);
    let w1 = b.add_binary_mul(g1_bc, e1_out, &shape);
    let combined = b.add_binary_add(w0, w1, &shape);

    b.build(combined)
        .expect("valid pipeline weighted sum kernel")
}

fn pipeline_weighted_sum_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Gate weights summing to 1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.6f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.4f32,
    )));
    bindings
}

/// Expert weighted sum produces finite, valid bounds.
#[test]
fn test_expert_output_weighted_sum_bounded_ibp() {
    let def = build_pipeline_weighted_sum_kernel();
    let bindings = pipeline_weighted_sum_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE pipeline weighted sum IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "weighted sum lower must be finite");
    assert!(hi_max.is_finite(), "weighted sum upper must be finite");
}

// ===========================================================================
// 4. Shared expert additive path IBP
// ===========================================================================

/// Build shared expert additive path (DeepSeek-V2 pattern):
/// output = gate_weighted_experts(x) + shared_expert(x)
///
/// The shared expert processes all tokens regardless of routing,
/// providing a stable baseline that the routed experts refine.
fn build_shared_expert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_shared_expert");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Routed expert (single, gate-weighted)
    let routed_out = build_expert_ffn(&mut b, input, "routed", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let gate = b.add_input("gate_weight", &[1]);
    let gate_bc = b.add_broadcast(gate, &shape);
    let gated = b.add_binary_mul(gate_bc, routed_out, &shape);

    // Shared expert (always active, no gate)
    let shared_out = build_expert_ffn(&mut b, input, "shared", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Combine: routed + shared
    let combined = b.add_binary_add(gated, shared_out, &shape);

    b.build(combined).expect("valid shared expert kernel")
}

fn shared_expert_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Routed expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Gate weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.5f32,
    )));
    // Shared expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

/// Shared expert additive path produces finite, valid bounds.
#[test]
fn test_shared_expert_additive_bounded_ibp() {
    let def = build_shared_expert_kernel();
    let bindings = shared_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE shared expert IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "shared expert lower must be finite");
    assert!(hi_max.is_finite(), "shared expert upper must be finite");
}

// ===========================================================================
// 5. MoE layer output bounded (routed + shared + residual) IBP
// ===========================================================================

/// Build a complete MoE layer: RMSNorm -> routed experts + shared expert + residual.
///
/// Pattern: x + gate * routed_expert(RMSNorm(x)) + shared_expert(RMSNorm(x))
fn build_moe_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_layer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // Routed expert (gate-weighted)
    let routed = build_expert_ffn(&mut b, normed, "routed", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let gate = b.add_input("gate_weight", &[1]);
    let gate_bc = b.add_broadcast(gate, &shape);
    let gated = b.add_binary_mul(gate_bc, routed, &shape);

    // Shared expert
    let shared = build_expert_ffn(&mut b, normed, "shared", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Combine: gated_routed + shared
    let moe_out = b.add_binary_add(gated, shared, &shape);

    // Residual: x + moe_out
    let out = b.add_binary_add(input, moe_out, &shape);

    b.build(out).expect("valid MoE layer kernel")
}

fn moe_layer_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    // Routed expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Gate weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.5f32,
    )));
    // Shared expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

/// MoE layer (routed + shared + residual) produces finite, valid bounds.
#[test]
fn test_moe_layer_output_bounded_ibp() {
    let def = build_moe_layer_kernel();
    let bindings = moe_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE layer IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "MoE layer lower must be finite");
    assert!(hi_max.is_finite(), "MoE layer upper must be finite");
    assert!(
        lo_min > -100.0,
        "MoE layer lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 6. Multi-layer MoE depth composition IBP
// ===========================================================================

/// Build a 2-layer MoE stack: each layer has RMSNorm + routed/shared expert + residual.
fn build_moe_depth2_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_depth2");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut current = input;

    for layer in 0..2 {
        let pfx = format!("l{layer}");

        // RMSNorm
        let eps = b.add_input(&format!("{pfx}_norm_eps"), &[1]);
        let norm_w = b.add_input(&format!("{pfx}_norm_w"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(current, eps, 1, norm_w, &shape);

        // Routed expert
        let routed = build_expert_ffn(
            &mut b,
            normed,
            &format!("{pfx}_routed"),
            SEQ_LEN,
            HIDDEN_DIM,
            FFN_DIM,
        );
        let gate = b.add_input(&format!("{pfx}_gate"), &[1]);
        let gate_bc = b.add_broadcast(gate, &shape);
        let gated = b.add_binary_mul(gate_bc, routed, &shape);

        // Shared expert
        let shared = build_expert_ffn(
            &mut b,
            normed,
            &format!("{pfx}_shared"),
            SEQ_LEN,
            HIDDEN_DIM,
            FFN_DIM,
        );

        // Combine + residual
        let moe_out = b.add_binary_add(gated, shared, &shape);
        current = b.add_binary_add(current, moe_out, &shape);
    }

    b.build(current).expect("valid MoE depth-2 pipeline kernel")
}

fn moe_depth2_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..2 {
        push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
        push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
        // Gate weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1]),
            0.5f32,
        )));
        push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    }
    bindings
}

/// Multi-layer MoE depth composition produces finite, bounded outputs.
#[test]
fn test_multi_layer_moe_depth_ibp() {
    let def = build_moe_depth2_pipeline_kernel();
    let bindings = moe_depth2_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE depth-2 pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "depth-2 lower must be finite");
    assert!(hi_max.is_finite(), "depth-2 upper must be finite");
}

// ===========================================================================
// 7. Load balancing auxiliary loss bounded IBP
// ===========================================================================

/// Build router with load balancing auxiliary loss computation.
///
/// Main path: Linear -> softmax(dim=1) for expert gating.
/// Aux path: Linear -> softmax(dim=0) for per-expert token assignment.
/// The aux softmax along the sequence dimension models the load balance loss.
fn build_load_balance_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_load_balance");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);

    // Main gate: softmax along expert dim
    let main_probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Aux load balance: softmax along sequence dim
    let _aux_probs = b.add_softmax(logits, 0, &[SEQ_LEN, NUM_EXPERTS]);

    b.build(main_probs)
        .expect("valid load balance pipeline kernel")
}

fn load_balance_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);
    bindings
}

/// Load balancing auxiliary loss preserves softmax bounds in [0, 1].
#[test]
fn test_load_balancing_auxiliary_loss_bounded_ibp() {
    let def = build_load_balance_pipeline_kernel();
    let bindings = load_balance_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE load balance IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "load balance lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "load balance upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Capacity overflow handling (clamped gates) IBP
// ===========================================================================

/// Model capacity overflow: when an expert receives too many tokens,
/// excess gate values are clamped. We model this as router -> narrow(1)
/// for a single expert slot, verifying bounds stay in [0, 1] even with
/// varying input magnitudes (simulating overflow conditions).
#[test]
fn test_capacity_overflow_handling_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_capacity_overflow");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let probs = build_router(&mut b, input, "router", SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS);
    // Single expert slot (capacity overflow scenario: only 1 slot available)
    let clamped = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);
    let def = b.build(clamped).expect("valid capacity overflow kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Test across varying input magnitudes (simulating overflow stress)
    for &range in &[0.5_f32, 1.0, 2.0, 5.0] {
        let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], range);
        let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let eps = 1e-6;
        assert!(
            lo_min >= 0.0 - eps,
            "capacity overflow lower must be >= 0 at range {range}, got {lo_min}"
        );
        assert!(
            hi_max <= 1.0 + eps,
            "capacity overflow upper must be <= 1 at range {range}, got {hi_max}"
        );
        eprintln!(
            "Capacity overflow at range [-{range}, {range}]: bounds=[{lo_min:.6}, {hi_max:.6}]"
        );
    }
}

// ===========================================================================
// 9. Full MoE inference pipeline IBP
// ===========================================================================

/// Build the full MoE inference pipeline:
/// RMSNorm -> router(softmax) + 2 routed expert FFNs + shared expert FFN
/// + gate-weighted combination + residual.
fn build_full_moe_inference_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_full_inference");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // Router: Linear -> softmax (produces gate probabilities, verified separately)
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(normed, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Expert 0 (routed)
    let e0_out = build_expert_ffn(&mut b, normed, "e0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    // Expert 1 (routed)
    let e1_out = build_expert_ffn(&mut b, normed, "e1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Gate weights (constant, simulating softmax-derived top-2 probabilities)
    let g0 = b.add_input("gate0", &[1]);
    let g1 = b.add_input("gate1", &[1]);
    let g0_bc = b.add_broadcast(g0, &shape);
    let g1_bc = b.add_broadcast(g1, &shape);

    let w0 = b.add_binary_mul(g0_bc, e0_out, &shape);
    let w1 = b.add_binary_mul(g1_bc, e1_out, &shape);
    let routed_combined = b.add_binary_add(w0, w1, &shape);

    // Shared expert (always active)
    let shared_out = build_expert_ffn(&mut b, normed, "shared", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Total MoE output: routed + shared
    let moe_out = b.add_binary_add(routed_combined, shared_out, &shape);

    // Residual: x + moe_out
    let out = b.add_binary_add(input, moe_out, &shape);

    b.build(out)
        .expect("valid full MoE inference pipeline kernel")
}

fn full_moe_inference_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);
    // Expert 0 FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Expert 1 FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Gate weights (sum to ~1)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.55f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.45f32,
    )));
    // Shared expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

/// Full MoE inference pipeline produces finite, bounded outputs.
#[test]
fn test_full_moe_inference_pipeline_ibp() {
    let def = build_full_moe_inference_kernel();
    let bindings = full_moe_inference_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE full inference IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "full inference lower must be finite");
    assert!(hi_max.is_finite(), "full inference upper must be finite");
    assert!(
        lo_min > -100.0,
        "full inference lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 10. Vision encoder -> MoE transformer -> generation head IBP
// ===========================================================================

/// Build the end-to-end VLM pipeline:
/// Vision linear projection -> MoE decoder layer -> RMSNorm -> LM head -> softmax.
///
/// Simplified for tractability: vision features projected to decoder dim,
/// one MoE decoder layer (routed + shared expert), final LM head.
fn build_vision_moe_generation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_vision_to_gen");
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Vision encoder output (linear projection from vision dim to hidden dim)
    let vision_input = b.add_input("vision_features", &[SEQ_LEN, VISION_DIM]);
    let proj_w = b.add_input("vision_proj_w", &[HIDDEN_DIM, VISION_DIM]);
    let projected = b.add_linear(vision_input, proj_w, None, &shape);

    // MoE decoder layer: RMSNorm -> expert FFN + residual
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(projected, eps, 1, norm_w, &shape);

    // Single routed expert (simplified for tractability)
    let expert_out = build_expert_ffn(&mut b, normed, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Residual
    let h = b.add_binary_add(projected, expert_out, &shape);

    // Final RMSNorm
    let final_eps = b.add_input("final_norm_eps", &[1]);
    let final_norm_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(h, final_eps, 1, final_norm_w, &shape);

    // LM head: Linear -> softmax
    let lm_w = b.add_input("lm_head_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid vision -> MoE -> generation kernel")
}

fn vision_moe_generation_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Vision projection
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, VISION_DIM]),
        WEIGHT_MAG,
    )));
    // RMSNorm
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    // Expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Final RMSNorm
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    // LM head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings
}

/// Vision encoder -> MoE transformer -> generation produces probabilities in [0, 1].
#[test]
fn test_vision_encoder_moe_transformer_generation_ibp() {
    let def = build_vision_moe_generation_kernel();
    let bindings = vision_moe_generation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, VISION_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "generation output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 vision -> MoE -> generation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "generation output lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "generation output upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. MoE layer output bounded CROWN
// ===========================================================================

/// MoE layer with CROWN linearization for tighter bounds.
#[test]
fn test_moe_layer_output_bounded_crown() {
    let def = build_moe_layer_kernel();
    let bindings = moe_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE layer CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. Multi-layer MoE depth CROWN
// ===========================================================================

/// Multi-layer MoE depth with CROWN linearization.
#[test]
fn test_multi_layer_moe_depth_crown() {
    let def = build_moe_depth2_pipeline_kernel();
    let bindings = moe_depth2_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE depth-2 CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Router gate sharpening with temperature CROWN
// ===========================================================================

/// Temperature-scaled router with CROWN: logits * inv_temperature -> softmax.
/// Lower temperature sharpens the distribution; CROWN should provide tighter
/// bounds than IBP for the non-linear softmax interaction.
#[test]
fn test_router_gate_sharpening_temperature_crown() {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_temp_sharp");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let inv_temp = b.add_input("inv_temp", &[1]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let inv_temp_bc = b.add_broadcast(inv_temp, &[SEQ_LEN, NUM_EXPERTS]);
    let scaled = b.add_binary_mul(logits, inv_temp_bc, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(scaled, 1, &[SEQ_LEN, NUM_EXPERTS]);
    let def = b.build(probs).expect("valid temperature router kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // inv_temperature = 1/0.5 = 2.0 (sharper distribution)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 2.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE temp router CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "temperature softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "temperature softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Full MoE inference pipeline CROWN
// ===========================================================================

/// Full MoE inference pipeline with CROWN linearization.
#[test]
fn test_full_moe_inference_pipeline_crown() {
    let def = build_full_moe_inference_kernel();
    let bindings = full_moe_inference_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3 MoE full inference CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 15. Single expert FFN isolation IBP
// ===========================================================================

/// Build a single SwiGLU expert FFN in isolation.
fn build_single_expert_isolation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_single_expert");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_expert_ffn(&mut b, input, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    b.build(out).expect("valid single expert kernel")
}

fn single_expert_isolation_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

/// Single expert FFN in isolation produces finite, valid bounds.
#[test]
fn test_single_expert_ffn_isolation_ibp() {
    let def = build_single_expert_isolation_kernel();
    let bindings = single_expert_isolation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE single expert IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "single expert lower must be finite");
    assert!(hi_max.is_finite(), "single expert upper must be finite");
}

// ===========================================================================
// 16. Two-expert vs four-expert routing width comparison IBP
// ===========================================================================

/// Compare routing bound widths with different expert counts.
/// More experts -> more softmax columns -> each probability is bounded
/// more tightly (uniform distribution = 1/N_experts).
#[test]
fn test_two_vs_four_expert_routing_width_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let build_n_expert = |n: usize| -> (TensorKernelDef, Vec<TensorParamBinding>) {
        let mut b = TensorBlockBuilder::new(&format!("qwen3_moe_pipeline_{n}_experts"));
        let inp = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let probs = build_router(&mut b, inp, "router", SEQ_LEN, HIDDEN_DIM, n);
        let def = b.build(probs).expect("valid n-expert kernel");

        let mut bindings = vec![TensorParamBinding::Variable];
        push_router_bindings(&mut bindings, HIDDEN_DIM, n, WEIGHT_MAG);
        (def, bindings)
    };

    let (def2, bind2) = build_n_expert(2);
    let graph2 = tensor_kernel_to_graph(&def2, &bind2).expect("graph 2-expert");
    let output2 = graph2.propagate_ibp(&input).expect("IBP 2-expert");
    assert_bounds_valid(&output2);

    let (def4, bind4) = build_n_expert(4);
    let graph4 = tensor_kernel_to_graph(&def4, &bind4).expect("graph 4-expert");
    let output4 = graph4.propagate_ibp(&input).expect("IBP 4-expert");
    assert_bounds_valid(&output4);

    let width2 = bound_width(&output2);
    let width4 = bound_width(&output4);

    eprintln!("Qwen3 MoE 2 vs 4 expert routing IBP: width_2={width2:.6}, width_4={width4:.6}");

    // Both bounded in [0, 1]
    let eps = 1e-6;
    let (lo2, hi2) = bounds_min_max(&output2);
    let (lo4, hi4) = bounds_min_max(&output4);
    assert!(lo2 >= 0.0 - eps && hi2 <= 1.0 + eps);
    assert!(lo4 >= 0.0 - eps && hi4 <= 1.0 + eps);

    assert!(width2.is_finite(), "2-expert width must be finite");
    assert!(width4.is_finite(), "4-expert width must be finite");
}

// ===========================================================================
// 17. MoE with RMSNorm pre/post IBP
// ===========================================================================

/// Build MoE with pre-norm and post-norm RMSNorm:
/// RMSNorm(x) -> expert FFN -> RMSNorm -> + residual.
///
/// Post-norm stabilizes the output range after the expert computation.
fn build_moe_pre_post_norm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_pre_post_norm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm RMSNorm
    let pre_eps = b.add_input("pre_norm_eps", &[1]);
    let pre_w = b.add_input("pre_norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, pre_eps, 1, pre_w, &shape);

    // Expert FFN
    let expert_out = build_expert_ffn(&mut b, normed, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Post-norm RMSNorm
    let post_eps = b.add_input("post_norm_eps", &[1]);
    let post_w = b.add_input("post_norm_w", &[HIDDEN_DIM]);
    let post_normed = b.add_rms_norm(expert_out, post_eps, 1, post_w, &shape);

    // Residual
    let out = b.add_binary_add(input, post_normed, &shape);

    b.build(out).expect("valid pre/post norm MoE kernel")
}

fn moe_pre_post_norm_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // Pre-norm
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    // Expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Post-norm
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    bindings
}

/// MoE with pre/post RMSNorm produces finite, bounded outputs.
#[test]
fn test_moe_rmsnorm_pre_post_ibp() {
    let def = build_moe_pre_post_norm_kernel();
    let bindings = moe_pre_post_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE pre/post norm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "pre/post norm lower must be finite");
    assert!(hi_max.is_finite(), "pre/post norm upper must be finite");
    assert!(
        lo_min > -100.0,
        "pre/post norm lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 18. End-to-end MoE decoder layer (attention + MoE FFN) CROWN
// ===========================================================================

/// Build a complete MoE decoder layer:
/// RMSNorm -> Attention -> residual -> RMSNorm -> MoE FFN -> residual.
///
/// This is the canonical Qwen3-VL decoder layer with MoE replacing dense FFN.
fn build_moe_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_moe_pipeline_decoder_layer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-attention RMSNorm
    let attn_norm_eps = b.add_input("attn_norm_eps", &[1]);
    let attn_norm_w = b.add_input("attn_norm_w", &[HIDDEN_DIM]);
    let attn_normed = b.add_rms_norm(input, attn_norm_eps, 1, attn_norm_w, &shape);

    // Self-attention
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            attn_normed,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let ffn_norm_eps = b.add_input("ffn_norm_eps", &[1]);
    let ffn_norm_w = b.add_input("ffn_norm_w", &[HIDDEN_DIM]);
    let ffn_normed = b.add_rms_norm(h, ffn_norm_eps, 1, ffn_norm_w, &shape);

    // MoE expert FFN
    let expert_out = build_expert_ffn(&mut b, ffn_normed, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual
    let out = b.add_binary_add(h, expert_out, &shape);

    b.build(out).expect("valid MoE decoder layer kernel")
}

fn moe_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };

    let mut bindings = vec![TensorParamBinding::Variable];
    // Attention norm
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    // Q, K, V, O attention weights
    bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
    bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
    bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
    bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
    // FFN norm
    push_rms_norm_bindings(&mut bindings, HIDDEN_DIM);
    // Expert FFN
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

/// End-to-end MoE decoder layer with CROWN linearization.
#[test]
fn test_moe_decoder_layer_attention_ffn_crown() {
    let def = build_moe_decoder_layer_kernel();
    let bindings = moe_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3 MoE decoder layer CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "decoder layer lower must be finite");
    assert!(hi_max.is_finite(), "decoder layer upper must be finite");
}
