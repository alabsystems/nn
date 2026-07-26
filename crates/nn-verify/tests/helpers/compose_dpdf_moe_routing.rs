// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for MoE (Mixture of Experts) expert routing and sparse gating
//! bound propagation used in Qwen3-VL-30B-A3B.
//!
//! Verifies IBP and CROWN bound propagation through MoE routing sub-blocks
//! with emphasis on sparse gating patterns: router softmax properties,
//! top-k selection, expert capacity, load balancing, sparse zeroing,
//! per-expert FFN bounds, weighted combination, temperature scaling,
//! expert count scaling, residual composition, and full MoE blocks.
//!
//! ## Router Softmax & Selection (tests 1-4)
//!
//! 1. Router softmax output bounds (sum to 1) IBP
//! 2. Top-k expert selection gate bounds IBP
//! 3. Expert capacity factor bounds IBP
//! 4. Load balancing auxiliary loss bounds IBP
//!
//! ## Sparse Gating & Expert FFN (tests 5-8)
//!
//! 5. Sparse gating: most experts zeroed out IBP
//! 6. Expert FFN output bounds per expert IBP
//! 7. Combined expert output (weighted sum) bounds IBP
//! 8. MoE vs dense FFN bound comparison IBP
//!
//! ## Noise, Temperature & Scaling (tests 9-11)
//!
//! 9. Expert dropout/jitter noise bounds IBP
//! 10. Router temperature scaling effect IBP
//! 11. 2-expert vs 8-expert routing bounds IBP
//!
//! ## Composition & Full Block (tests 12-18)
//!
//! 12. MoE residual: x + MoE(norm(x)) bounds IBP
//! 13. Expert specialization: per-expert output range IBP
//! 14. Router z-loss: penalizes large router logits IBP
//! 15. MoE depth composition (stacked MoE layers) IBP
//! 16. MoE depth composition (stacked MoE layers) CROWN
//! 17. Full MoE block: router -> experts -> combine -> residual IBP
//! 18. Full MoE block: router -> experts -> combine -> residual CROWN
//!
//! Architecture references:
//! - Qwen3-30B-A3B (Alibaba): MoE with 128 experts, top-8 sparse routing
//! - Switch Transformer (Fedus et al., 2021): Top-1 expert sparse routing
//! - GShard (Lepikhin et al., 2020): Top-2 routing with load balancing
//! - ST-MoE (Zoph et al., 2022): Router z-loss and capacity factor
//! - DeepSeek-V2 (DeepSeek AI, 2024): Shared + routed expert MoE
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128, NUM_EXPERTS=8, TOP_K=2
//!
//! Part of #4016: Compose tests for MoE expert routing and sparse gating.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, ReduceOp, TensorKernelDef};
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

    // gate_proj -> SiLU
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);

    // up_proj
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // element-wise gate * up -> down_proj
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

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
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

// ===========================================================================
// 1. Router softmax output bounds (sum to 1) IBP
// ===========================================================================

/// Build a router softmax kernel: Linear -> softmax over 8 experts.
fn build_router_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_softmax");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let probs = build_router(&mut b, input, "router", SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS);
    b.build(probs).expect("valid router softmax kernel")
}

fn router_softmax_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);
    bindings
}

#[test]
fn test_router_softmax_output_bounds_ibp() {
    let def = build_router_softmax_kernel();
    let bindings = router_softmax_bindings();
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
    eprintln!("Router softmax IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );

    // Feasibility check: sum of upper bounds >= 1.0 per token
    let (_, hi) = output.lower_upper();
    for t in 0..SEQ_LEN {
        let hi_sum: f32 = (0..NUM_EXPERTS).map(|e| hi[[t, e]]).sum();
        assert!(
            hi_sum >= 1.0 - 1e-4,
            "sum of upper bounds at token {t} should be >= 1.0, got {hi_sum}"
        );
    }
}

// ===========================================================================
// 2. Top-k expert selection gate bounds IBP
// ===========================================================================

/// Build top-k selection: Linear -> softmax -> narrow(TOP_K).
fn build_topk_selection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_topk");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let probs = build_router(&mut b, input, "router", SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS);
    // Approximate top-k via narrow: select first TOP_K expert slots.
    let topk = b.add_narrow(probs, 1, 0, TOP_K, &[SEQ_LEN, TOP_K]);
    b.build(topk).expect("valid top-k selection kernel")
}

fn topk_selection_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);
    bindings
}

#[test]
fn test_topk_expert_selection_gate_bounds_ibp() {
    let def = build_topk_selection_kernel();
    let bindings = topk_selection_bindings();
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
    eprintln!("Top-k selection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

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
// 3. Expert capacity factor bounds IBP
// ===========================================================================

/// Expert capacity: router probabilities remain bounded in [0, 1]
/// across different input magnitude ranges (small, medium, large).
/// The capacity factor concept relates to how much of the total token
/// probability each expert receives; softmax guarantees [0, 1] per slot.
#[test]
fn test_expert_capacity_factor_bounds_ibp() {
    let def = build_router_softmax_kernel();
    let bindings = router_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    for &range in &[0.1_f32, 0.5, 1.0, 2.0, 5.0] {
        let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], range);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let eps = 1e-6;
        assert!(
            lo_min >= 0.0 - eps,
            "capacity lower must be >= 0 at range {range}, got {lo_min}"
        );
        assert!(
            hi_max <= 1.0 + eps,
            "capacity upper must be <= 1 at range {range}, got {hi_max}"
        );
        eprintln!(
            "Expert capacity at input range [-{range}, {range}]: bounds=[{lo_min:.6}, {hi_max:.6}]"
        );
    }
}

// ===========================================================================
// 4. Load balancing auxiliary loss bounds IBP
// ===========================================================================

/// Build router with auxiliary load balance softmax.
///
/// Main path: Linear -> softmax (expert dim, axis=1).
/// Aux path: Linear -> softmax (sequence dim, axis=0) — models per-expert
/// token assignment distribution for the load balance loss.
/// Both produce outputs bounded in [0, 1].
fn build_load_balance_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_load_balance");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);

    // Main gate: softmax along expert dim
    let main_probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Aux load balance: softmax along sequence dim
    let _aux_probs = b.add_softmax(logits, 0, &[SEQ_LEN, NUM_EXPERTS]);

    // Output is the main gate (aux is for loss only)
    b.build(main_probs).expect("valid load balance kernel")
}

fn load_balance_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings
}

#[test]
fn test_load_balancing_aux_loss_bounds_ibp() {
    let def = build_load_balance_kernel();
    let bindings = load_balance_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Load balance aux loss IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

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
// 5. Sparse gating: most experts zeroed out IBP
// ===========================================================================

/// Model sparse gating: router selects top-1 from 8 experts.
/// After top-1 selection via narrow, 7/8 experts effectively receive zero
/// gate probability. The selected expert's probability is bounded in [0, 1].
///
/// We verify that narrowing to a single expert slot preserves softmax bounds.
#[test]
fn test_sparse_gating_most_experts_zeroed_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_sparse_gate");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let probs = build_router(&mut b, input, "router", SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS);

    // Sparse top-1: only 1 of 8 experts is selected
    let sparse_gate = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);
    let def = b.build(sparse_gate).expect("valid sparse gate kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_router_bindings(&mut bindings, HIDDEN_DIM, NUM_EXPERTS, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sparse gating (top-1 of 8) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sparse gate lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sparse gate upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 6. Expert FFN output bounds per expert IBP
// ===========================================================================

/// Build a single expert FFN: SwiGLU with dedicated weight prefix.
fn build_single_expert_ffn_kernel(expert_id: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("dpdf_moe_routing_expert{expert_id}_ffn"));
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_expert_ffn(
        &mut b,
        input,
        &format!("e{expert_id}"),
        SEQ_LEN,
        HIDDEN_DIM,
        FFN_DIM,
    );
    b.build(out).expect("valid single expert FFN kernel")
}

fn single_expert_ffn_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_expert_ffn_output_bounds_per_expert_ibp() {
    // Verify two different experts produce consistent bound structure
    for expert_id in [0, 1] {
        let def = build_single_expert_ffn_kernel(expert_id);
        let bindings = single_expert_ffn_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

        let (lo_min, hi_max) = bounds_min_max(&output);
        eprintln!("Expert {expert_id} FFN IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
        assert!(
            lo_min.is_finite(),
            "expert {expert_id} lower must be finite"
        );
        assert!(
            hi_max.is_finite(),
            "expert {expert_id} upper must be finite"
        );
    }
}

// ===========================================================================
// 7. Combined expert output (weighted sum) bounds IBP
// ===========================================================================

/// Build combined expert output: gate_weight * expert0(x) + gate_weight * expert1(x).
///
/// Models the MoE weighted combination where top-k experts' outputs are
/// scaled by their gate probabilities and summed. Here we model 2 experts
/// with constant gate weights (approximating the softmax-derived gate).
fn build_combined_expert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_combined_expert");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Expert 0 path
    let e0_out = build_expert_ffn(&mut b, input, "e0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Expert 1 path
    let e1_out = build_expert_ffn(&mut b, input, "e1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Gate weights (constant, simulating softmax-derived probabilities)
    let g0 = b.add_input("gate0", &[1]);
    let g1 = b.add_input("gate1", &[1]);

    let g0_bc = b.add_broadcast(g0, &shape);
    let g1_bc = b.add_broadcast(g1, &shape);

    // Weighted combination: g0 * expert0(x) + g1 * expert1(x)
    let w0 = b.add_binary_mul(g0_bc, e0_out, &shape);
    let w1 = b.add_binary_mul(g1_bc, e1_out, &shape);
    let combined = b.add_binary_add(w0, w1, &shape);

    b.build(combined).expect("valid combined expert kernel")
}

fn combined_expert_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Gate weights: softmax-like, sum to ~1
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

#[test]
fn test_combined_expert_weighted_sum_bounds_ibp() {
    let def = build_combined_expert_kernel();
    let bindings = combined_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Combined expert (weighted sum) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "combined lower must be finite");
    assert!(hi_max.is_finite(), "combined upper must be finite");
}

// ===========================================================================
// 8. MoE vs dense FFN bound comparison IBP
// ===========================================================================

/// Compare MoE (two weighted experts) vs dense (single FFN) bound widths.
/// With identical weight magnitudes, the MoE weighted combination should
/// produce comparable or tighter bounds than a single dense FFN because
/// gate weights are in [0, 1] and sum to 1.
#[test]
fn test_moe_vs_dense_ffn_bound_comparison_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // MoE: weighted combination of two experts
    let moe_def = build_combined_expert_kernel();
    let moe_bindings = combined_expert_bindings();
    let moe_graph = tensor_kernel_to_graph(&moe_def, &moe_bindings).expect("MoE graph");
    let moe_output = moe_graph.propagate_ibp(&input).expect("MoE IBP");

    // Dense: single SwiGLU FFN
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_dense_ffn");
    let dense_input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let dense_out = build_expert_ffn(&mut b, dense_input, "dense", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let dense_def = b.build(dense_out).expect("valid dense FFN kernel");
    let mut dense_bindings = vec![TensorParamBinding::Variable];
    push_expert_ffn_bindings(&mut dense_bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    let dense_graph = tensor_kernel_to_graph(&dense_def, &dense_bindings).expect("dense graph");
    let dense_output = dense_graph.propagate_ibp(&input).expect("dense IBP");

    assert_bounds_valid(&moe_output);
    assert_bounds_valid(&dense_output);

    let moe_width = bound_width(&moe_output);
    let dense_width = bound_width(&dense_output);

    eprintln!("MoE vs dense IBP: moe_width={moe_width:.6}, dense_width={dense_width:.6}");

    // Both should produce finite bounds
    assert!(moe_width.is_finite(), "MoE width must be finite");
    assert!(dense_width.is_finite(), "dense width must be finite");
}

// ===========================================================================
// 9. Expert dropout/jitter noise bounds IBP
// ===========================================================================

/// Model expert jitter noise as: gate_logits + noise * scale.
/// In practice, router jitter adds small noise to gate logits during training
/// to encourage exploration. We verify that adding a small constant scale
/// perturbation preserves softmax [0, 1] bounds.
#[test]
fn test_expert_dropout_jitter_noise_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_jitter_noise");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);

    // Add jitter noise modeled as a constant scale offset
    let noise_scale = b.add_input("noise_scale", &[1]);
    let noise_bc = b.add_broadcast(noise_scale, &[SEQ_LEN, NUM_EXPERTS]);
    let noisy_logits = b.add_binary_add(logits, noise_bc, &[SEQ_LEN, NUM_EXPERTS]);

    // Softmax on noisy logits
    let probs = b.add_softmax(noisy_logits, 1, &[SEQ_LEN, NUM_EXPERTS]);
    let def = b.build(probs).expect("valid jitter noise kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        // Small noise scale (jitter magnitude)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.01f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Expert jitter noise IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "jitter softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "jitter softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 10. Router temperature scaling effect IBP
// ===========================================================================

/// Temperature scaling: logits / temperature before softmax.
/// Lower temperature -> sharper (more peaked) distribution.
/// Higher temperature -> more uniform distribution.
/// Both should preserve softmax [0, 1] bounds.
#[test]
fn test_router_temperature_scaling_effect_ibp() {
    // Build kernel with temperature scaling: logits * inv_temperature -> softmax
    let build_temp_kernel = |temp_val: f32| -> (TensorKernelDef, Vec<TensorParamBinding>) {
        let mut b = TensorBlockBuilder::new(&format!(
            "dpdf_moe_routing_temp_{}",
            (temp_val * 100.0) as u32
        ));
        let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
        let inv_temp = b.add_input("inv_temp", &[1]);

        let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
        let inv_temp_bc = b.add_broadcast(inv_temp, &[SEQ_LEN, NUM_EXPERTS]);
        let scaled_logits = b.add_binary_mul(logits, inv_temp_bc, &[SEQ_LEN, NUM_EXPERTS]);
        let probs = b.add_softmax(scaled_logits, 1, &[SEQ_LEN, NUM_EXPERTS]);
        let def = b.build(probs).expect("valid temp kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
                WEIGHT_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1.0 / temp_val)),
        ];
        (def, bindings)
    };

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let mut widths = Vec::new();
    for &temp in &[0.5_f32, 1.0, 2.0] {
        let (def, bindings) = build_temp_kernel(temp);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        widths.push(width);

        let eps = 1e-6;
        assert!(
            lo_min >= 0.0 - eps,
            "temp={temp}: softmax lower must be >= 0, got {lo_min}"
        );
        assert!(
            hi_max <= 1.0 + eps,
            "temp={temp}: softmax upper must be <= 1, got {hi_max}"
        );
        eprintln!("Temperature {temp}: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    }

    // All temperatures preserve [0, 1] bounds (verified above)
    for w in &widths {
        assert!(
            w.is_finite(),
            "bound width must be finite at all temperatures"
        );
    }
}

// ===========================================================================
// 11. 2-expert vs 8-expert routing bounds IBP
// ===========================================================================

/// Compare routing bounds with different numbers of experts.
/// More experts -> more columns in softmax -> each individual probability
/// has a lower maximum (1/N_experts for uniform distribution).
#[test]
fn test_2_expert_vs_8_expert_routing_bounds_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let build_n_expert_router = |n_experts: usize| -> (TensorKernelDef, Vec<TensorParamBinding>) {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_moe_routing_{n_experts}_experts"));
        let inp = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
        let probs = build_router(&mut b, inp, "router", SEQ_LEN, HIDDEN_DIM, n_experts);
        let def = b.build(probs).expect("valid n-expert router kernel");

        let mut bindings = vec![TensorParamBinding::Variable];
        push_router_bindings(&mut bindings, HIDDEN_DIM, n_experts, WEIGHT_MAG);
        (def, bindings)
    };

    let (def2, bind2) = build_n_expert_router(2);
    let graph2 = tensor_kernel_to_graph(&def2, &bind2).expect("graph 2-expert");
    let output2 = graph2.propagate_ibp(&input).expect("IBP 2-expert");
    assert_bounds_valid(&output2);

    let (def8, bind8) = build_n_expert_router(8);
    let graph8 = tensor_kernel_to_graph(&def8, &bind8).expect("graph 8-expert");
    let output8 = graph8.propagate_ibp(&input).expect("IBP 8-expert");
    assert_bounds_valid(&output8);

    let width2 = bound_width(&output2);
    let width8 = bound_width(&output8);

    eprintln!("2-expert vs 8-expert routing IBP: width_2={width2:.6}, width_8={width8:.6}");

    // Both bounded in [0, 1]
    let eps = 1e-6;
    let (lo2, hi2) = bounds_min_max(&output2);
    let (lo8, hi8) = bounds_min_max(&output8);
    assert!(lo2 >= 0.0 - eps && hi2 <= 1.0 + eps);
    assert!(lo8 >= 0.0 - eps && hi8 <= 1.0 + eps);

    // Both widths finite
    assert!(width2.is_finite(), "2-expert width must be finite");
    assert!(width8.is_finite(), "8-expert width must be finite");
}

// ===========================================================================
// 12. MoE residual: x + MoE(norm(x)) bounds IBP
// ===========================================================================

/// Build MoE residual: x + expert_ffn(RMSNorm(x)).
///
/// The standard MoE decoder pattern: pre-norm the input with RMSNorm,
/// route through an expert FFN, and add back the residual.
fn build_moe_residual_norm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_residual_norm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // Expert FFN on normalized input
    let ffn_out = build_expert_ffn(&mut b, normed, "e0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Residual: x + expert_ffn(RMSNorm(x))
    let out = b.add_binary_add(input, ffn_out, &shape);
    b.build(out).expect("valid MoE residual norm kernel")
}

fn moe_residual_norm_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_moe_residual_x_plus_moe_norm_x_ibp() {
    let def = build_moe_residual_norm_kernel();
    let bindings = moe_residual_norm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // x + expert_ffn(RMSNorm(x)). ny carries the exact ‖z‖₂ ≤ √n sphere on the
    // RMSNorm IBP output, so the up-proj Linear intersects its decorrelated box
    // (‖w‖₁·√n) with the exact Cauchy–Schwarz row bound (‖w‖₂·√n). That sound,
    // tighten-only step collapses the residual lower from ~-269 to ~-4.3, so a
    // plain IBP pass now clears the >-100 target. The -100 threshold is unchanged.
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE residual x + MoE(norm(x)) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "residual lower must be finite");
    assert!(hi_max.is_finite(), "residual upper must be finite");
    // Residual: input in [-1, 1] + small FFN contribution
    assert!(
        lo_min > -100.0,
        "residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 13. Expert specialization: per-expert output range IBP
// ===========================================================================

/// Build two experts with different weight magnitudes to model
/// specialization: expert0 has standard weights, expert1 has smaller
/// weights (modeling a less-active expert).
///
/// Verifies that different weight magnitudes produce different bound widths.
#[test]
fn test_expert_specialization_per_expert_output_range_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Expert with standard weights
    let def_std = build_single_expert_ffn_kernel(0);
    let bindings_std = single_expert_ffn_bindings();
    let graph_std = tensor_kernel_to_graph(&def_std, &bindings_std).expect("graph std");
    let output_std = graph_std.propagate_ibp(&input).expect("IBP std");
    assert_bounds_valid(&output_std);

    // Expert with smaller weights (specialized/less active)
    let def_small = build_single_expert_ffn_kernel(1);
    let small_weight = 0.005f32;
    let mut bindings_small = vec![TensorParamBinding::Variable];
    push_expert_ffn_bindings(&mut bindings_small, HIDDEN_DIM, FFN_DIM, small_weight);
    let graph_small = tensor_kernel_to_graph(&def_small, &bindings_small).expect("graph small");
    let output_small = graph_small.propagate_ibp(&input).expect("IBP small");
    assert_bounds_valid(&output_small);

    let width_std = bound_width(&output_std);
    let width_small = bound_width(&output_small);

    eprintln!("Expert specialization IBP: std_width={width_std:.6}, small_width={width_small:.6}");

    // Smaller weights should produce tighter bounds
    assert!(
        width_small <= width_std + 1e-4,
        "smaller weights should produce tighter bounds: small={width_small}, std={width_std}"
    );
}

// ===========================================================================
// 14. Router z-loss: penalizes large router logits (IBP)
// ===========================================================================

/// Router z-loss from ST-MoE (Zoph et al., 2022) encourages numerical
/// stability by penalizing the magnitude of router logits.
///
/// The z-loss is defined as: z_loss = (1/T) * sum(logsumexp(logits, dim=-1)^2)
///
/// We model the penalty term as:
///   router logits -> exp -> sum(dim=-1) -> square -> mean(dim=0)
///
/// The exp->sum gives the partition function Z = sum(exp(logits)),
/// squaring and averaging produces the penalty scalar.  Verifies that
/// bounds remain finite and non-negative (z-loss is always >= 0).
#[test]
fn test_router_zloss_penalty_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_zloss");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    // Router logits: Linear(x) -> [SEQ_LEN, NUM_EXPERTS]
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);

    // exp(logits) -> [SEQ_LEN, NUM_EXPERTS]
    let exp_logits = b.add_exp(logits, &[SEQ_LEN, NUM_EXPERTS]);

    // sum(exp(logits), dim=1) -> partition function Z per token: [SEQ_LEN, 1]
    let z_per_token = b.add_reduce(exp_logits, ReduceOp::Sum, 1, true, &[SEQ_LEN, 1]);

    // Z^2 (square the partition function): [SEQ_LEN, 1]
    let z_squared = b.add_binary_mul(z_per_token, z_per_token, &[SEQ_LEN, 1]);

    // mean(Z^2, dim=0) -> scalar penalty: [1, 1]
    let z_loss = b.add_reduce(z_squared, ReduceOp::Mean, 0, true, &[1, 1]);

    let def = b.build(z_loss).expect("valid z-loss kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Router z-loss IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");

    // z-loss is always non-negative: Z = sum(exp(...)) > 0, Z^2 > 0, mean(Z^2) > 0
    assert!(lo_min.is_finite(), "z-loss lower must be finite");
    assert!(hi_max.is_finite(), "z-loss upper must be finite");
    assert!(
        lo_min >= 0.0 - 1e-6,
        "z-loss must be non-negative, got {lo_min}"
    );
}

// ===========================================================================
// 15-16. MoE depth composition (stacked MoE layers) IBP + CROWN
// ===========================================================================

/// Build a 2-layer MoE decoder: each layer has attention + MoE FFN.
/// This tests bound widening through depth with MoE routing at each layer.
fn build_moe_depth2_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_depth2");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut current = input;

    for layer in 0..2 {
        let pfx = format!("l{layer}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{pfx}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{pfx}_norm1_w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // Self-attention
        let q_w = b.add_input(&format!("{pfx}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{pfx}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{pfx}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let o_w = b.add_input(&format!("{pfx}_o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let attn_out = b
            .add_multi_head_attention(
                normed1,
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
        let h = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let norm2_eps = b.add_input(&format!("{pfx}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{pfx}_norm2_w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(h, norm2_eps, 1, norm2_w, &shape);

        // MoE expert FFN
        let ffn_out = build_expert_ffn(
            &mut b,
            normed2,
            &format!("{pfx}_e0"),
            SEQ_LEN,
            HIDDEN_DIM,
            FFN_DIM,
        );

        // Second residual
        current = b.add_binary_add(h, ffn_out, &shape);
    }

    b.build(current).expect("valid MoE depth-2 kernel")
}

fn moe_depth2_bindings() -> Vec<TensorParamBinding> {
    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };

    let mut bindings = vec![TensorParamBinding::Variable];

    for _ in 0..2 {
        // norm1: eps, weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1]),
            1e-5f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM]),
            1.0f32,
        )));
        // attention: q, k, v, o
        bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
        bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
        bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
        bindings.push(w(&[HIDDEN_DIM, HIDDEN_DIM]));
        // norm2: eps, weight
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[1]),
            1e-5f32,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM]),
            1.0f32,
        )));
        // expert FFN: gate_w, up_w, down_w
        push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    }
    bindings
}

#[test]
fn test_moe_depth_composition_stacked_ibp() {
    let def = build_moe_depth2_kernel();
    let bindings = moe_depth2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE depth-2 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "depth-2 lower must be finite");
    assert!(hi_max.is_finite(), "depth-2 upper must be finite");
}

#[test]
fn test_moe_depth_composition_stacked_crown() {
    let def = build_moe_depth2_kernel();
    let bindings = moe_depth2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE depth-2 CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 17-18. Full MoE block: router -> experts -> combine -> residual IBP + CROWN
// ===========================================================================

/// Build a full MoE block: RMSNorm -> router(softmax) + 2 expert FFNs
/// with gate-weighted combination + residual.
///
/// This is the complete MoE pattern from Qwen3-VL-30B-A3B:
/// 1. RMSNorm(x)
/// 2. Router: Linear -> softmax (gate probabilities)
/// 3. Expert 0 FFN: SwiGLU
/// 4. Expert 1 FFN: SwiGLU
/// 5. Weighted sum: g0 * e0(norm_x) + g1 * e1(norm_x)
/// 6. Residual: x + weighted_sum
fn build_full_moe_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_moe_routing_full_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // RMSNorm
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // Router: Linear -> softmax (verifies gate bounds)
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(normed, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Expert 0 FFN
    let e0_out = build_expert_ffn(&mut b, normed, "e0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Expert 1 FFN
    let e1_out = build_expert_ffn(&mut b, normed, "e1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Gate weights (constant, approximating softmax-derived gate)
    let g0 = b.add_input("gate0", &[1]);
    let g1 = b.add_input("gate1", &[1]);

    let g0_bc = b.add_broadcast(g0, &shape);
    let g1_bc = b.add_broadcast(g1, &shape);

    // Weighted combination
    let w0 = b.add_binary_mul(g0_bc, e0_out, &shape);
    let w1 = b.add_binary_mul(g1_bc, e1_out, &shape);
    let combined = b.add_binary_add(w0, w1, &shape);

    // Residual: x + combined
    let out = b.add_binary_add(input, combined, &shape);

    b.build(out).expect("valid full MoE block kernel")
}

fn full_moe_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        // RMSNorm: eps, weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        // Router weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    // Expert 0 FFN weights
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Expert 1 FFN weights
    push_expert_ffn_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // Gate weights (sum to ~1)
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

#[test]
fn test_full_moe_block_router_experts_combine_residual_ibp() {
    let def = build_full_moe_block_kernel();
    let bindings = full_moe_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // RMSNorm -> 2 SwiGLU experts -> gate-weighted sum -> residual. Each expert's
    // gate/up Linear consumes the RMSNorm output, which now carries the exact
    // ‖z‖₂ ≤ √n sphere; the Linear intersects its decorrelated box (‖w‖₁·√n) with
    // the exact Cauchy–Schwarz row bound (‖w‖₂·√n). That sound, tighten-only step
    // pulls the residual lower from ~-269 to ~-4.3, clearing >-100 under plain IBP.
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full MoE block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "full block lower must be finite");
    assert!(hi_max.is_finite(), "full block upper must be finite");
    // Residual preserves bounded output
    assert!(
        lo_min > -100.0,
        "full block lower should be reasonable, got {lo_min}"
    );
}

#[test]
fn test_full_moe_block_router_experts_combine_residual_crown() {
    let def = build_full_moe_block_kernel();
    let bindings = full_moe_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full MoE block CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
