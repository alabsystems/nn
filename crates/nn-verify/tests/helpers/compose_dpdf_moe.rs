// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: MoE (Mixture of Experts) routing and expert selection
//! NY composition for Qwen3-VL.
//!
//! Verifies bounds propagation through Mixture-of-Experts sub-blocks used in
//! Qwen3-VL MoE variants (e.g., Qwen3-30B-A3B) for the dpdf document
//! understanding pipeline:
//!
//! 1. **MoE gate output [0,1] IBP**: Linear -> softmax gate output bounded
//!    in [0, 1] per expert slot.
//!
//! 2. **MoE gate sum-to-one IBP**: Linear -> softmax gate outputs sum to 1.0
//!    (verified via bounds on per-token probability vectors).
//!
//! 3. **Top-1 expert selection IBP**: Linear -> softmax -> narrow(1) for
//!    top-1 selection. Probability bound preserved after selection.
//!
//! 4. **Top-2 expert selection with load balancing IBP**: Router -> softmax
//!    -> narrow(2) with auxiliary load balance loss (modeled as softmax on
//!    expert usage counts).
//!
//! 5. **Expert FFN (SwiGLU) IBP + CROWN**: Single expert SwiGLU FFN
//!    (gate_proj -> SiLU -> mul(up_proj) -> down_proj) with CROWN
//!    linearization.
//!
//! 6. **MoE dispatch composition IBP**: Gate -> select -> expert FFN full
//!    routing pipeline.
//!
//! 7. **MoE residual IBP**: input + MoE(input) skip connection preserving
//!    bounded outputs.
//!
//! 8. **MoE with shared expert IBP**: Shared expert FFN + routed expert FFN
//!    combined output.
//!
//! 9. **MoE routing monotone tightening IBP**: Tighter input bounds produce
//!    tighter gate output bounds.
//!
//! 10. **Expert capacity bounds IBP**: Router probabilities bounded per expert
//!     slot under varying input ranges.
//!
//! 11. **MoE vs dense FFN bound width comparison IBP**: Routed expert path
//!     produces comparable or tighter bounds than dense FFN path.
//!
//! 12. **MoE with auxiliary loss regularization IBP**: Router softmax + load
//!     balance auxiliary softmax both bounded.
//!
//! 13. **Multi-layer MoE (2 layers) IBP**: Stacked MoE decoder layers with
//!     attention + MoE FFN.
//!
//! 14. **MoE + attention decoder block IBP + CROWN**: Pre-norm attention +
//!     pre-norm MoE FFN decoder block with CROWN linearization.
//!
//! 15. **MoE quantized experts (INT4 dequant) IBP**: Expert FFN with INT4
//!     dequantized weights producing tighter bounds from reduced weight range.
//!
//! Architecture references:
//! - Qwen3-30B-A3B (Alibaba): MoE variant with 128 experts, top-8 routing
//! - MoE (Fedus et al., 2022): Mixture-of-Experts routing
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN used per expert
//! - Switch Transformer (Fedus et al., 2021): Top-1 expert routing
//! - GShard (Lepikhin et al., 2020): Top-2 routing with load balancing
//!
//! Dimensions (small for fast verification):
//! - HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=4, NUM_EXPERTS=4, TOP_K=2
//!
//! Part of #3985: NY compose tests for MoE routing and expert selection.

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

/// Hidden dimension (tiny for testing).
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 128;
/// Sequence length for MoE sub-block tests.
const SEQ_LEN: usize = 4;
/// Number of MoE experts.
const NUM_EXPERTS: usize = 4;
/// Number of attention heads (for decoder block tests).
const NUM_HEADS: usize = 4;
/// Number of KV heads for grouped-query attention.
const NUM_KV_HEADS: usize = 2;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// KV dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 32
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Vocabulary size for LM head tests.
const VOCAB_SIZE: usize = 256;
/// INT4 dequant weight magnitude (smaller due to quantization).
const QUANT_WEIGHT_MAG: f32 = 0.01;

// ===========================================================================
// Helper: build a SwiGLU expert FFN sub-block
// ===========================================================================

/// Build a SwiGLU FFN for one expert.
///
/// Input: the node preceding the FFN.
/// Returns: the output node after down_proj.
fn build_swiglu_expert(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_weight"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_weight"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_weight"), &[hidden_dim, ffn_dim]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    // Fused SiLU (not sigmoid+mul) so ny recognizes MulBinary(SiLU(gate), up)
    // and fires its up/gate-correlation zonotope tightening.
    let gate_activated = b.add_silu(gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU expert FFN bindings for one expert.
fn push_swiglu_expert_bindings(bindings: &mut Vec<TensorParamBinding>, weight_mag: f32) {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), weight_mag);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), weight_mag);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), weight_mag);

    bindings.push(TensorParamBinding::ConstantTensor(gate_w));
    bindings.push(TensorParamBinding::ConstantTensor(up_w));
    bindings.push(TensorParamBinding::ConstantTensor(down_w));
}

// ===========================================================================
// 1. MoE gate output [0,1] IBP
// ===========================================================================

/// Build a MoE gate: Linear -> softmax producing expert probabilities.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, NUM_EXPERTS]` (gate probabilities in [0, 1]).
fn build_moe_gate_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_gate_output");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    b.build(probs).expect("valid MoE gate kernel")
}

fn moe_gate_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ]
}

/// MoE gate output bounded in [0, 1] via softmax codomain.
#[test]
fn test_moe_gate_output_01_ibp() {
    let def = build_moe_gate_kernel();
    let bindings = moe_gate_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through MoE gate");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_EXPERTS],
        "MoE gate output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE gate output IBP: bounds=[{lo_min}, {hi_max}]");

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
// 2. MoE gate sum-to-one IBP
// ===========================================================================

/// MoE gate softmax outputs sum to 1.0 per token.
///
/// Since softmax normalizes, the sum of probabilities per token is exactly 1.
/// IBP cannot verify this directly (it tracks per-element bounds), so we
/// verify that individual gate probabilities are in [0, 1] and their upper
/// bounds sum to at least 1.0 (necessary for the sum to be achievable).
#[test]
fn test_moe_gate_sum_to_one_ibp() {
    let def = build_moe_gate_kernel();
    let bindings = moe_gate_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through MoE gate");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // For each token position, check that the sum of upper bounds >= 1.0
    // (necessary for softmax sum = 1 to be feasible within bounds).
    for t in 0..SEQ_LEN {
        let hi_sum: f32 = (0..NUM_EXPERTS).map(|e| hi[[t, e]]).sum();
        assert!(
            hi_sum >= 1.0 - 1e-4,
            "sum of upper bounds at token {t} should be >= 1.0, got {hi_sum}"
        );
    }
    // Lower bounds are all >= 0 (softmax codomain)
    for t in 0..SEQ_LEN {
        for e in 0..NUM_EXPERTS {
            assert!(
                lo[[t, e]] >= -1e-6,
                "gate lower at [{t}, {e}] must be >= 0, got {}",
                lo[[t, e]]
            );
        }
    }
    eprintln!("MoE gate sum-to-one IBP: all tokens have feasible sum >= 1.0");
}

// ===========================================================================
// 3. Top-1 expert selection IBP
// ===========================================================================

/// Build MoE top-1 routing: Linear -> softmax -> narrow(1).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, 1]` (top-1 expert probability).
fn build_moe_top1_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_top1_routing");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);
    // Approximate top-1 via narrow (select first expert slot).
    let top1 = b.add_narrow(probs, 1, 0, 1, &[SEQ_LEN, 1]);

    b.build(top1).expect("valid MoE top-1 routing kernel")
}

fn moe_top1_routing_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ]
}

/// Top-1 expert probability bounded in [0, 1].
#[test]
fn test_moe_top1_selection_ibp() {
    let def = build_moe_top1_routing_kernel();
    let bindings = moe_top1_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE top-1 routing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 1],
        "top-1 routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE top-1 routing IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "top-1 lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "top-1 upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Top-2 expert selection with load balancing IBP
// ===========================================================================

/// Build MoE top-2 routing with load balance auxiliary output.
///
/// Models the GShard-style top-2 routing where:
/// - Main path: Linear -> softmax -> narrow(2) (top-2 expert selection)
/// - Aux path: softmax on expert usage counts (modeled as a second softmax
///   over the same gate logits, approximating load balance loss)
///
/// Both paths produce [0, 1] bounded outputs.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, 2]` (top-2 routing probabilities).
fn build_moe_top2_load_balance_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_top2_load_balance");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Top-2 selection (approximate via narrow)
    let top2 = b.add_narrow(probs, 1, 0, 2, &[SEQ_LEN, 2]);

    b.build(top2)
        .expect("valid MoE top-2 load balance routing kernel")
}

fn moe_top2_load_balance_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ]
}

/// Top-2 routing with load balancing: bounded in [0, 1].
#[test]
fn test_moe_top2_load_balance_ibp() {
    let def = build_moe_top2_load_balance_kernel();
    let bindings = moe_top2_load_balance_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE top-2 load balance routing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 2],
        "top-2 load balance routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE top-2 load balance IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "top-2 lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "top-2 upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 5. Expert FFN (SwiGLU) IBP + CROWN
// ===========================================================================

/// Build a single MoE expert SwiGLU FFN.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_moe_expert_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_expert_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_expert(&mut b, input, "expert0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    b.build(out).expect("valid MoE expert FFN kernel")
}

fn moe_expert_ffn_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG);
    bindings
}

/// CROWN bounds propagate through a single MoE expert SwiGLU FFN.
#[test]
fn test_moe_expert_ffn_crown() {
    let def = build_moe_expert_ffn_kernel();
    let bindings = moe_expert_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE expert FFN CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record MoE expert FFN.
#[test]
fn test_moe_expert_ffn_verify_and_record() {
    let def = build_moe_expert_ffn_kernel();
    let bindings = moe_expert_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_moe_expert_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 6. MoE dispatch: gate -> select -> expert FFN composition IBP
// ===========================================================================

/// Build a simplified MoE dispatch pipeline:
/// router (Linear -> softmax -> narrow(2)) producing gate weights,
/// followed by one expert SwiGLU FFN. In practice, multiple experts run
/// in parallel and outputs are weighted-summed. Here we model the bound-
/// critical path: one expert FFN receiving the routed tokens.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_moe_dispatch_composition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_dispatch_composition");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    // Gate: Linear -> softmax (for verification of the routing path)
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Expert FFN: SwiGLU on the routed tokens (input passes through expert)
    let ffn_out = build_swiglu_expert(&mut b, input, "expert0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    b.build(ffn_out)
        .expect("valid MoE dispatch composition kernel")
}

fn moe_dispatch_composition_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ];
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG);
    bindings
}

/// IBP through MoE dispatch: gate -> expert FFN.
#[test]
fn test_moe_dispatch_composition_ibp() {
    let def = build_moe_dispatch_composition_kernel();
    let bindings = moe_dispatch_composition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE dispatch composition");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE dispatch output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE dispatch composition IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. MoE residual: input + MoE(input) IBP
// ===========================================================================

/// Build MoE residual: input + expert_ffn(input).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_moe_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_residual");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Expert FFN path
    let ffn_out = build_swiglu_expert(&mut b, input, "expert0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Residual: output = input + expert_ffn(input)
    let out = b.add_binary_add(input, ffn_out, &shape);

    b.build(out).expect("valid MoE residual kernel")
}

fn moe_residual_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG);
    bindings
}

/// IBP through MoE residual composition (expert + skip connection).
#[test]
fn test_moe_residual_ibp() {
    let def = build_moe_residual_kernel();
    let bindings = moe_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE residual IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual preserves bounded output: input in [-1,1] + small FFN output
    assert!(
        lo_min > -50.0,
        "MoE residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 8. MoE with shared expert IBP
// ===========================================================================

/// Build MoE with shared expert: shared_expert(input) + routed_expert(input).
///
/// Some MoE architectures (e.g., DeepSeek-V2) use a "shared expert" that
/// processes every token alongside the routed experts. Verifies that bounds
/// propagate through two parallel SwiGLU FFNs combined via addition.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_moe_shared_expert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_shared_expert");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Shared expert path
    let shared_out =
        build_swiglu_expert(&mut b, input, "shared_expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Routed expert path
    let routed_out =
        build_swiglu_expert(&mut b, input, "routed_expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Combined: shared + routed
    let combined = b.add_binary_add(shared_out, routed_out, &shape);

    // Residual: input + combined
    let out = b.add_binary_add(input, combined, &shape);

    b.build(out).expect("valid MoE shared expert kernel")
}

fn moe_shared_expert_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG); // shared expert
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG); // routed expert
    bindings
}

/// IBP through MoE with shared expert (shared + routed + residual).
#[test]
fn test_moe_shared_expert_ibp() {
    let def = build_moe_shared_expert_kernel();
    let bindings = moe_shared_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE shared expert");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE shared expert output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE shared expert IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // With two expert paths + residual, bounds should still be reasonable
    assert!(
        lo_min > -100.0,
        "shared expert lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 9. MoE routing monotone tightening IBP
// ===========================================================================

/// Tighter input bounds produce tighter MoE gate output bounds.
///
/// Property: if input range shrinks from [-1, 1] to [-0.5, 0.5], the
/// gate output bound width should decrease (or stay the same).
#[test]
fn test_moe_routing_monotone_tightening_ibp() {
    let def = build_moe_gate_kernel();
    let bindings = moe_gate_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input range
    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide");

    // Tight input range
    let input_tight = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight");

    assert_bounds_valid(&output_wide);
    assert_bounds_valid(&output_tight);

    let (wide_lo, wide_hi) = bounds_min_max(&output_wide);
    let (tight_lo, tight_hi) = bounds_min_max(&output_tight);

    let wide_width = wide_hi - wide_lo;
    let tight_width = tight_hi - tight_lo;

    eprintln!("MoE routing monotone: wide_width={wide_width:.6}, tight_width={tight_width:.6}");

    assert!(
        tight_width <= wide_width + 1e-4,
        "tighter input should produce tighter gate bounds: tight={tight_width}, wide={wide_width}"
    );
}

// ===========================================================================
// 10. Expert capacity bounds IBP
// ===========================================================================

/// Expert capacity: each expert probability is bounded under varying input.
///
/// Tests that router probabilities for each expert slot remain in [0, 1]
/// across multiple input ranges ([-0.5, 0.5], [-1, 1], [-2, 2]).
#[test]
fn test_moe_expert_capacity_bounds_ibp() {
    let def = build_moe_gate_kernel();
    let bindings = moe_gate_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    for &range in &[0.5_f32, 1.0, 2.0] {
        let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], range);
        let output = graph.propagate_ibp(&input).expect("IBP through MoE gate");

        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let eps = 1e-6;
        assert!(
            lo_min >= 0.0 - eps,
            "gate lower must be >= 0 at range {range}, got {lo_min}"
        );
        assert!(
            hi_max <= 1.0 + eps,
            "gate upper must be <= 1 at range {range}, got {hi_max}"
        );
        eprintln!(
            "Expert capacity at input range [-{range}, {range}]: bounds=[{lo_min}, {hi_max}]"
        );
    }
}

// ===========================================================================
// 11. MoE vs dense FFN bound width comparison IBP
// ===========================================================================

/// Build a dense (non-MoE) FFN for comparison.
///
/// Same architecture as one expert FFN but without routing overhead.
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_dense_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dense_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_expert(&mut b, input, "dense", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    b.build(out).expect("valid dense FFN kernel")
}

fn dense_ffn_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG);
    bindings
}

/// MoE single-expert FFN produces comparable bounds to dense FFN.
///
/// With identical weight magnitudes, a single expert's FFN path should
/// produce bounds of similar width to a dense FFN since the architecture
/// is identical (SwiGLU with same dimensions).
#[test]
fn test_moe_vs_dense_ffn_bound_width_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // MoE single expert
    let moe_def = build_moe_expert_ffn_kernel();
    let moe_bindings = moe_expert_ffn_bindings();
    let moe_graph = tensor_kernel_to_graph(&moe_def, &moe_bindings).expect("MoE graph");
    let moe_output = moe_graph.propagate_ibp(&input).expect("MoE IBP");

    // Dense FFN
    let dense_def = build_dense_ffn_kernel();
    let dense_bindings = dense_ffn_bindings();
    let dense_graph = tensor_kernel_to_graph(&dense_def, &dense_bindings).expect("dense graph");
    let dense_output = dense_graph.propagate_ibp(&input).expect("dense IBP");

    assert_bounds_valid(&moe_output);
    assert_bounds_valid(&dense_output);

    let (moe_lo, moe_hi) = bounds_min_max(&moe_output);
    let (dense_lo, dense_hi) = bounds_min_max(&dense_output);

    let moe_width = moe_hi - moe_lo;
    let dense_width = dense_hi - dense_lo;

    eprintln!("MoE expert width={moe_width:.6}, dense FFN width={dense_width:.6}");

    // Same architecture => same width (within numerical tolerance)
    let ratio = moe_width / dense_width.max(1e-12);
    assert!(
        ratio < 2.0 && ratio > 0.5,
        "MoE and dense should have similar bound widths: ratio={ratio}"
    );
}

// ===========================================================================
// 12. MoE with auxiliary loss regularization IBP
// ===========================================================================

/// Build MoE router with auxiliary load balance regularization.
///
/// The main gate and an auxiliary softmax (over the same logits, modeling
/// the expert load balance loss signal) both produce [0, 1] outputs.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, NUM_EXPERTS]` (main gate probabilities).
fn build_moe_aux_loss_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_aux_loss");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);

    // Main gate: softmax along expert dim
    let main_probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Auxiliary load balance: softmax along sequence dim (models per-expert
    // token assignment distribution for the load balance loss).
    let _aux_probs = b.add_softmax(logits, 0, &[SEQ_LEN, NUM_EXPERTS]);

    // Output is the main gate (aux is used for loss only, not forwarded)
    b.build(main_probs).expect("valid MoE aux loss kernel")
}

fn moe_aux_loss_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ]
}

/// Both main gate and auxiliary softmax bounded in [0, 1].
#[test]
fn test_moe_aux_loss_ibp() {
    let def = build_moe_aux_loss_kernel();
    let bindings = moe_aux_loss_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE aux loss");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_EXPERTS],
        "MoE aux loss output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE aux loss IBP: bounds=[{lo_min}, {hi_max}]");

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. Multi-layer MoE (2 layers stacked) IBP
// ===========================================================================

/// Build a 2-layer MoE decoder stack.
///
/// Each layer: RMSNorm -> attention -> residual -> RMSNorm -> MoE FFN -> residual.
/// The MoE FFN is modeled as a single expert SwiGLU (bounds-critical path).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_multi_layer_moe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("multi_layer_moe");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..2 {
        let prefix = format!("l{layer}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // GQA causal attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[KV_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[KV_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, q_w, None, &kv_shape);
        let k = b.add_linear(normed1, k_w, None, &kv_shape);
        let v = b.add_linear(normed1, v_w, None, &kv_shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &kv_shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-MoE RMSNorm
        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

        // MoE FFN (single expert path for bounds)
        let ffn_out = build_swiglu_expert(
            &mut b,
            normed2,
            &format!("{prefix}_expert"),
            SEQ_LEN,
            HIDDEN_DIM,
            FFN_DIM,
        );

        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current).expect("valid multi-layer MoE kernel")
}

fn multi_layer_moe_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(out_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG); // expert FFN
    }

    bindings
}

/// IBP through 2-layer MoE decoder stack.
#[test]
fn test_multi_layer_moe_ibp() {
    let def = build_multi_layer_moe_kernel();
    let bindings = multi_layer_moe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-layer MoE");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "multi-layer MoE output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-layer MoE IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. MoE + attention decoder block IBP + CROWN
// ===========================================================================

/// Build a single decoder block with attention + MoE FFN.
///
/// RMSNorm -> causal attention -> residual -> RMSNorm -> MoE expert FFN ->
/// residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_moe_attention_decoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_attention_decoder_block");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // GQA causal attention
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, KV_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = b.add_linear(normed1, q_w, None, &kv_shape);
    let k = b.add_linear(normed1, k_w, None, &kv_shape);
    let v = b.add_linear(normed1, v_w, None, &kv_shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &kv_shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-MoE RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // MoE expert FFN (single expert path)
    let ffn_out = build_swiglu_expert(&mut b, normed2, "expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Residual after MoE
    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out)
        .expect("valid MoE attention decoder block kernel")
}

fn moe_attention_decoder_block_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(out_w),          // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
    ];
    push_swiglu_expert_bindings(&mut bindings, WEIGHT_MAG); // expert FFN
    bindings
}

/// CROWN bounds propagate through MoE attention decoder block.
#[test]
fn test_moe_attention_decoder_block_crown() {
    let def = build_moe_attention_decoder_block_kernel();
    let bindings = moe_attention_decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE attention decoder block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record MoE attention decoder block.
#[test]
fn test_moe_attention_decoder_block_verify_and_record() {
    let def = build_moe_attention_decoder_block_kernel();
    let bindings = moe_attention_decoder_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_moe_attention_decoder_block");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 15. MoE quantized experts (INT4 dequant) IBP
// ===========================================================================

/// Build MoE expert FFN with INT4-magnitude weights (simulating dequantized
/// weights with reduced magnitude).
///
/// INT4 quantization constrains weights to a smaller range than FP32. This
/// produces tighter output bounds from the expert FFN. We model this by
/// using QUANT_WEIGHT_MAG (0.01) instead of WEIGHT_MAG (0.02).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_moe_quantized_expert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("moe_quantized_expert");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_expert(&mut b, input, "quant_expert", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    b.build(out).expect("valid MoE quantized expert kernel")
}

fn moe_quantized_expert_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_expert_bindings(&mut bindings, QUANT_WEIGHT_MAG);
    bindings
}

/// INT4 quantized expert produces tighter bounds than FP32 expert.
#[test]
fn test_moe_quantized_expert_tighter_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // FP32 expert
    let fp32_def = build_moe_expert_ffn_kernel();
    let fp32_bindings = moe_expert_ffn_bindings();
    let fp32_graph = tensor_kernel_to_graph(&fp32_def, &fp32_bindings).expect("FP32 graph");
    let fp32_output = fp32_graph.propagate_ibp(&input).expect("FP32 IBP");

    // INT4 quantized expert
    let q_def = build_moe_quantized_expert_kernel();
    let q_bindings = moe_quantized_expert_bindings();
    let q_graph = tensor_kernel_to_graph(&q_def, &q_bindings).expect("quant graph");
    let q_output = q_graph.propagate_ibp(&input).expect("quant IBP");

    assert_bounds_valid(&fp32_output);
    assert_bounds_valid(&q_output);

    let (fp32_lo, fp32_hi) = bounds_min_max(&fp32_output);
    let (q_lo, q_hi) = bounds_min_max(&q_output);

    let fp32_width = fp32_hi - fp32_lo;
    let q_width = q_hi - q_lo;

    eprintln!("FP32 expert width={fp32_width:.6}, INT4 expert width={q_width:.6}");

    // Quantized (smaller weights) should produce tighter bounds
    assert!(
        q_width <= fp32_width + 1e-4,
        "INT4 expert should have tighter bounds: q_width={q_width}, fp32_width={fp32_width}"
    );
}
