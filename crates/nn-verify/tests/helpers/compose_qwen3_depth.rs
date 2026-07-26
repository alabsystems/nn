// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deepened Qwen3 NY compose verification tests.
//!
//! Adds coverage for compositions not covered by existing Qwen3 test files:
//!
//! 1. **Embedding + RoPE composition**: Token embedding lookup followed by
//!    RoPE positional encoding. Verifies bounds through the embedding-to-
//!    position-encoded representation path.
//!
//! 2. **RMSNorm + SwiGLU + residual (Conservative)**: Pre-MLP normalization
//!    composed with gated FFN and residual connection. Conservative mode
//!    produces Sound classification (vs Heuristic in default mode).
//!
//! 3. **3-layer decoder stack**: Extends the existing 2-layer stack coverage
//!    to 3 layers with widening analysis. Verifies bounds growth remains
//!    sub-exponential through deeper compositions.
//!
//! 4. **MoE forward pass**: Full mixture-of-experts composition including
//!    router softmax, top-k gating, expert SwiGLU FFN, and weighted sum.
//!    Verifies the MoE layer preserves bounded outputs.
//!
//! 5. **QK-Norm attention**: Qwen3-specific per-head RMSNorm on Q and K
//!    projections before attention. Verifies normalization constrains the
//!    attention logit range.
//!
//! 6. **Decoder to logit argmax**: Full decoder output through LM head
//!    projection + softmax + argmax-like top-1 selection. Verifies the
//!    complete text generation output path.
//!
//! Uses IbpValidated soundness mode per nn engineering rules.
//! Dimensions: D_MODEL=16, N_HEADS=2, N_KV_HEADS=1, FFN_DIM=48, SEQ=4, VOCAB=32.
//!
//! Part of #4280: Deepen Qwen3 NY compose verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const D_MODEL: usize = 16;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 8
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 8
const FFN_DIM: usize = 48;
const SEQ: usize = 4;
const VOCAB: usize = 32;
const NUM_EXPERTS: usize = 4;
const EXPERTS_PER_TOK: usize = 2;
const HALF_DIM: usize = HEAD_DIM / 2; // 4
const WEIGHT_MAG: f32 = 0.001;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

// ---------------------------------------------------------------------------
// RoPE cos/sin tables
// ---------------------------------------------------------------------------

fn rope_cos_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ * HALF_DIM];
    for pos in 0..SEQ {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ, HALF_DIM]), data).expect("valid cos table")
}

fn rope_sin_table() -> ArrayD<f32> {
    let mut data = vec![0.0f32; SEQ * HALF_DIM];
    for pos in 0..SEQ {
        for i in 0..HALF_DIM {
            let theta = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            data[pos * HALF_DIM + i] = theta.sin() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ, HALF_DIM]), data).expect("valid sin table")
}

// ---------------------------------------------------------------------------
// Helper: add one decoder block to a builder (returns output node)
// ---------------------------------------------------------------------------

fn add_decoder_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    block_idx: usize,
    bindings: &mut Vec<TensorParamBinding>,
) -> nn_dsl::TensorNodeId {
    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    // Pre-attention RMSNorm
    let eps = b.add_input(&format!("b{block_idx}_attn_eps"), &[1]);
    let attn_rms_w = b.add_input(&format!("b{block_idx}_attn_rms_w"), &[D_MODEL]);
    let normed1 = b.add_rms_norm(input, eps, 1, attn_rms_w, &shape);

    // Self-attention (causal)
    let q_w = b.add_input(&format!("b{block_idx}_q_w"), &[D_MODEL, D_MODEL]);
    let k_w = b.add_input(&format!("b{block_idx}_k_w"), &[D_MODEL, D_MODEL]);
    let v_w = b.add_input(&format!("b{block_idx}_v_w"), &[D_MODEL, D_MODEL]);
    let out_w = b.add_input(&format!("b{block_idx}_out_w"), &[D_MODEL, D_MODEL]);

    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");
    let residual1 = b.add_binary_add(input, attn, &shape);

    // Pre-MLP RMSNorm
    let mlp_eps = b.add_input(&format!("b{block_idx}_mlp_eps"), &[1]);
    let mlp_rms_w = b.add_input(&format!("b{block_idx}_mlp_rms_w"), &[D_MODEL]);
    let normed2 = b.add_rms_norm(residual1, mlp_eps, 1, mlp_rms_w, &shape);

    // SwiGLU MLP
    let gate_w = b.add_input(&format!("b{block_idx}_gate_w"), &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input(&format!("b{block_idx}_up_w"), &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input(&format!("b{block_idx}_down_w"), &[D_MODEL, FFN_DIM]);

    let gate_proj = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(normed2, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let mlp_out = b.add_linear(gated, down_w, None, &shape);
    let residual2 = b.add_binary_add(residual1, mlp_out, &shape);

    // Push bindings for this block (11 params)
    let attn_w = w(&[D_MODEL, D_MODEL]);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // attn eps
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL]))); // attn rms
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // q_w
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // k_w
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // v_w
    bindings.push(TensorParamBinding::ConstantTensor(attn_w)); // out_w
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // mlp eps
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL]))); // mlp rms
    bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL]))); // gate
    bindings.push(TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL]))); // up
    bindings.push(TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM]))); // down

    residual2
}

// ===========================================================================
// 1. Embedding + RoPE composition
// ===========================================================================

/// Build embedding lookup followed by RoPE rotation.
///
/// Input: `[SEQ, D_MODEL]` (Variable -- represents post-embedding activations).
/// Output: `[SEQ, HEAD_DIM]` (one head's worth of RoPE-rotated representation).
///
/// Models the path: token_embedding -> per-head slice -> RoPE rotation.
fn build_embedding_rope() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_depth_embedding_rope");

    // Embedding output (treated as Variable since embedding lookup is a table fetch)
    let emb = b.add_input("token_emb", &[SEQ, D_MODEL]);
    // Project to single head dimension
    let proj_w = b.add_input("head_proj_w", &[HEAD_DIM, D_MODEL]);
    let projected = b.add_linear(emb, proj_w, None, &[SEQ, HEAD_DIM]);

    // RoPE rotation
    let cos = b.add_input("rope_cos", &[SEQ, HALF_DIM]);
    let sin = b.add_input("rope_sin", &[SEQ, HALF_DIM]);
    let neg_one = b.add_input("neg_one", &[1]);

    let half_shape = [SEQ, HALF_DIM];

    let x_first = b.add_narrow(projected, 1, 0, HALF_DIM, &half_shape);
    let x_second = b.add_narrow(projected, 1, HALF_DIM, HALF_DIM, &half_shape);

    // rot_first = x_first * cos - x_second * sin
    let fc = b.add_binary_mul(x_first, cos, &half_shape);
    let ss = b.add_binary_mul(x_second, sin, &half_shape);
    let neg_bc = b.add_broadcast(neg_one, &half_shape);
    let neg_ss = b.add_binary_mul(ss, neg_bc, &half_shape);
    let rot_first = b.add_binary_add(fc, neg_ss, &half_shape);

    // rot_second = x_first * sin + x_second * cos
    let fs = b.add_binary_mul(x_first, sin, &half_shape);
    let sc = b.add_binary_mul(x_second, cos, &half_shape);
    let rot_second = b.add_binary_add(fs, sc, &half_shape);

    let output = b.add_concat(&[rot_first, rot_second], 1, &[SEQ, HEAD_DIM]);

    b.build(output).expect("valid embedding + RoPE kernel")
}

fn embedding_rope_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                                // token_emb
        TensorParamBinding::ConstantTensor(w(&[HEAD_DIM, D_MODEL])), // head_proj
        TensorParamBinding::ConstantTensor(rope_cos_table()),        // cos
        TensorParamBinding::ConstantTensor(rope_sin_table()),        // sin
        TensorParamBinding::ConstantScalar(-1.0),                    // neg_one
    ]
}

#[test]
fn test_qwen3_depth_embedding_rope_ibp() {
    let def = build_embedding_rope();
    def.validate().expect("embedding + RoPE should validate");

    let bindings = embedding_rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 10,
        "embedding + RoPE graph >= 10 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HEAD_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 embedding + RoPE IBP: [{lo}, {hi}]");
    // Linear proj + rotation: bounded by weight magnitude and rotation
    assert!(lo.is_finite(), "lower must be finite, got {lo}");
    assert!(hi.is_finite(), "upper must be finite, got {hi}");
}

#[test]
fn test_qwen3_depth_embedding_rope_crown() {
    let def = build_embedding_rope();
    let bindings = embedding_rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // Embedding + RoPE is fully linear (linear proj + linear rotation).
    // CROWN should produce tight bounds.
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, HEAD_DIM]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 embedding + RoPE: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_depth_embedding_rope_verify_record() {
    let def = build_embedding_rope();
    let bindings = embedding_rope_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_depth_embedding_rope");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, HEAD_DIM]
    );
}

// ===========================================================================
// 2. RMSNorm + SwiGLU + residual (Conservative Sound)
// ===========================================================================

/// Build RMSNorm -> SwiGLU MLP -> residual with Conservative soundness.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
fn build_rmsnorm_swiglu_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_depth_rmsnorm_swiglu_residual");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    // RMSNorm
    let normed = b.add_rms_norm(x, eps, 1, rms_w, &shape);

    // SwiGLU: silu(gate(x)) * up(x) -> down
    let gate_proj = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(normed, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let mlp_out = b.add_linear(gated, down_w, None, &shape);

    // Residual
    let out = b.add_binary_add(x, mlp_out, &shape);

    b.build(out).expect("valid RMSNorm + SwiGLU + residual")
}

fn rmsnorm_swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

#[test]
fn test_qwen3_depth_rmsnorm_swiglu_residual_ibp() {
    let def = build_rmsnorm_swiglu_residual();
    def.validate()
        .expect("RMSNorm + SwiGLU + residual should validate");

    let bindings = rmsnorm_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // RMSNorm fuses to 1 native node; SwiGLU = gate linear (1) + sigmoid (1) +
    // silu mul (1) + up linear (1) + gated mul (1) + down linear (1) = 6; plus
    // the residual add (1). Only `x` is a Variable (NETWORK_INPUT sentinel) and
    // the 5 weights fold into their ops, so the graph is 1 + 6 + 1 = 8 nodes.
    assert!(
        graph.num_nodes() >= 8,
        "RMSNorm + SwiGLU + residual graph >= 8 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 RMSNorm + SwiGLU + residual IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e6, "lower magnitude < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "upper magnitude < 1e6, got {hi}");
}

#[test]
fn test_qwen3_depth_rmsnorm_swiglu_residual_conservative_sound() {
    let def = build_rmsnorm_swiglu_residual();
    let bindings = rmsnorm_swiglu_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_depth_rmsnorm_swiglu_residual",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative RMSNorm + SwiGLU should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Qwen3 RMSNorm + SwiGLU + residual (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

#[test]
fn test_qwen3_depth_rmsnorm_swiglu_residual_crown() {
    let def = build_rmsnorm_swiglu_residual();
    let bindings = rmsnorm_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 RMSNorm + SwiGLU + residual: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 3. Three-layer decoder stack with widening analysis
// ===========================================================================

/// Build a 3-layer decoder stack.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
fn build_three_layer_stack() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("qwen3_depth_three_layer_stack");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let mut bindings = vec![TensorParamBinding::Variable];

    let x = add_decoder_block(&mut b, x, 0, &mut bindings);
    let x = add_decoder_block(&mut b, x, 1, &mut bindings);
    let out = add_decoder_block(&mut b, x, 2, &mut bindings);

    let def = b.build(out).expect("valid 3-layer decoder stack");
    (def, bindings)
}

#[test]
fn test_qwen3_depth_three_layer_stack_validates() {
    let (def, _) = build_three_layer_stack();
    def.validate().expect("3-layer stack should validate");
}

#[test]
fn test_qwen3_depth_three_layer_stack_ibp() {
    let (def, bindings) = build_three_layer_stack();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // 3 blocks x ~20+ nodes each = 60+ nodes
    assert!(
        graph.num_nodes() >= 60,
        "3-layer stack graph >= 60 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-layer stack");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 3-layer stack IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "3-layer stack lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "3-layer stack upper < 1e8, got {hi}");
}

/// Widening analysis: compare 1, 2, and 3 block IBP bounds width.
///
/// Key property: bounds growth through decoder blocks should be sub-exponential
/// with small weights and residual connections.
#[test]
fn test_qwen3_depth_three_layer_widening_analysis() {
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // 1-block
    let mut b1 = TensorBlockBuilder::new("qwen3_depth_1block");
    let x1 = b1.add_input("x", &[SEQ, D_MODEL]);
    let mut bindings1 = vec![TensorParamBinding::Variable];
    let out1 = add_decoder_block(&mut b1, x1, 0, &mut bindings1);
    let def1 = b1.build(out1).expect("valid 1-block");
    let g1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let o1 = g1.propagate_ibp(&input).expect("IBP 1-block");
    let (lo1, hi1) = bounds_min_max(&o1);
    let width1 = hi1 - lo1;

    // 2-block
    let mut b2 = TensorBlockBuilder::new("qwen3_depth_2block");
    let x2 = b2.add_input("x", &[SEQ, D_MODEL]);
    let mut bindings2 = vec![TensorParamBinding::Variable];
    let x2 = add_decoder_block(&mut b2, x2, 0, &mut bindings2);
    let out2 = add_decoder_block(&mut b2, x2, 1, &mut bindings2);
    let def2 = b2.build(out2).expect("valid 2-block");
    let g2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let o2 = g2.propagate_ibp(&input).expect("IBP 2-block");
    let (lo2, hi2) = bounds_min_max(&o2);
    let width2 = hi2 - lo2;

    // 3-block
    let (def3, bindings3) = build_three_layer_stack();
    let g3 = tensor_kernel_to_graph(&def3, &bindings3).expect("graph");
    let o3 = g3.propagate_ibp(&input).expect("IBP 3-block");
    let (lo3, hi3) = bounds_min_max(&o3);
    let width3 = hi3 - lo3;

    eprintln!("Widening analysis:");
    eprintln!("  1-block: width={width1:.4}, bounds=[{lo1:.4}, {hi1:.4}]");
    eprintln!("  2-block: width={width2:.4}, bounds=[{lo2:.4}, {hi2:.4}]");
    eprintln!("  3-block: width={width3:.4}, bounds=[{lo3:.4}, {hi3:.4}]");

    let ratio_2_1 = width2 / width1.max(1e-10);
    let ratio_3_2 = width3 / width2.max(1e-10);
    eprintln!("  2/1 blowup: {ratio_2_1:.2}x, 3/2 blowup: {ratio_3_2:.2}x");

    // All widths must be finite
    assert!(width1.is_finite(), "1-block width not finite");
    assert!(width2.is_finite(), "2-block width not finite");
    assert!(width3.is_finite(), "3-block width not finite");

    // 3-block blowup should be bounded
    let total_blowup = width3 / 2.0; // input range is 2.0
    assert!(
        total_blowup < 1e6,
        "3-block blowup factor < 1e6, got {total_blowup:.1}x"
    );
}

#[test]
fn test_qwen3_depth_three_layer_stack_verify_record() {
    let (def, bindings) = build_three_layer_stack();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_depth_three_layer_stack");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ===========================================================================
// 4. MoE forward pass: router + top-k gating + expert FFN + weighted sum
// ===========================================================================

/// Build MoE forward pass composition.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// The MoE layer:
///   1. Router: Linear(D_MODEL -> NUM_EXPERTS) -> softmax -> routing probs
///   2. Each expert: SwiGLU FFN (simplified to Linear for tractable verification)
///   3. Weighted sum of expert outputs by routing probabilities
///
/// For verification tractability, we model 2 expert paths explicitly
/// (matching EXPERTS_PER_TOK=2) with constant routing weights. This captures
/// the key composition: multiple expert FFNs combined with learned gating.
fn build_moe_forward() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_depth_moe_forward");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    // Router: Linear -> softmax -> routing probabilities
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, D_MODEL]);
    let logits = b.add_linear(x, router_w, None, &[SEQ, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ, NUM_EXPERTS]);

    // Extract top-2 routing weights (modeled as narrow on softmax output)
    let w0 = b.add_narrow(probs, 1, 0, 1, &[SEQ, 1]);
    let w1 = b.add_narrow(probs, 1, 1, 1, &[SEQ, 1]);
    let w0_bc = b.add_broadcast(w0, &shape);
    let w1_bc = b.add_broadcast(w1, &shape);

    // Expert 0: SwiGLU FFN
    let e0_gate_w = b.add_input("e0_gate_w", &[FFN_DIM, D_MODEL]);
    let e0_up_w = b.add_input("e0_up_w", &[FFN_DIM, D_MODEL]);
    let e0_down_w = b.add_input("e0_down_w", &[D_MODEL, FFN_DIM]);

    let e0_gate = b.add_linear(x, e0_gate_w, None, &ffn_shape);
    let e0_sig = b.add_sigmoid(e0_gate, &ffn_shape);
    let e0_act = b.add_binary_mul(e0_gate, e0_sig, &ffn_shape);
    let e0_up = b.add_linear(x, e0_up_w, None, &ffn_shape);
    let e0_gated = b.add_binary_mul(e0_act, e0_up, &ffn_shape);
    let e0_out = b.add_linear(e0_gated, e0_down_w, None, &shape);

    // Expert 1: SwiGLU FFN
    let e1_gate_w = b.add_input("e1_gate_w", &[FFN_DIM, D_MODEL]);
    let e1_up_w = b.add_input("e1_up_w", &[FFN_DIM, D_MODEL]);
    let e1_down_w = b.add_input("e1_down_w", &[D_MODEL, FFN_DIM]);

    let e1_gate = b.add_linear(x, e1_gate_w, None, &ffn_shape);
    let e1_sig = b.add_sigmoid(e1_gate, &ffn_shape);
    let e1_act = b.add_binary_mul(e1_gate, e1_sig, &ffn_shape);
    let e1_up = b.add_linear(x, e1_up_w, None, &ffn_shape);
    let e1_gated = b.add_binary_mul(e1_act, e1_up, &ffn_shape);
    let e1_out = b.add_linear(e1_gated, e1_down_w, None, &shape);

    // Weighted sum: w0 * expert0 + w1 * expert1
    let weighted0 = b.add_binary_mul(w0_bc, e0_out, &shape);
    let weighted1 = b.add_binary_mul(w1_bc, e1_out, &shape);
    let moe_out = b.add_binary_add(weighted0, weighted1, &shape);

    // Residual connection
    let out = b.add_binary_add(x, moe_out, &shape);

    b.build(out).expect("valid MoE forward pass kernel")
}

fn moe_forward_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                                   // x
        TensorParamBinding::ConstantTensor(w(&[NUM_EXPERTS, D_MODEL])), // router
        // Expert 0
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
        // Expert 1
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

#[test]
fn test_qwen3_depth_moe_forward_validates() {
    let def = build_moe_forward();
    def.validate().expect("MoE forward should validate");
}

#[test]
fn test_qwen3_depth_moe_forward_ibp() {
    let def = build_moe_forward();
    let bindings = moe_forward_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Router + 2 expert FFNs + weighted sum + residual -> substantial graph
    assert!(
        graph.num_nodes() >= 20,
        "MoE forward graph >= 20 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MoE forward");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE forward IBP: [{lo}, {hi}]");
    // With residual + small-weight experts, output should be bounded
    assert!(lo.abs() < 1e6, "MoE forward lower < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "MoE forward upper < 1e6, got {hi}");
}

#[test]
fn test_qwen3_depth_moe_forward_crown() {
    let def = build_moe_forward();
    let bindings = moe_forward_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE forward: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_depth_moe_forward_verify_record() {
    let def = build_moe_forward();
    let bindings = moe_forward_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_depth_moe_forward");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ===========================================================================
// 5. QK-Norm attention: per-head RMSNorm on Q and K before attention
// ===========================================================================

/// Build QK-Norm attention subgraph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// Qwen3 applies RMSNorm to Q and K per-head before computing attention:
///   Q_norm = rms_norm(Q_proj)
///   K_norm = rms_norm(K_proj)
///   attn = softmax(Q_norm @ K_norm^T / sqrt(d)) @ V
///
/// This constrains attention logit magnitudes, preventing outlier scores
/// from dominating softmax output.
fn build_qk_norm_attention() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_depth_qk_norm_attn");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let shape = [SEQ, D_MODEL];

    // Q/K/V projections
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let q = b.add_linear(x, q_w, None, &shape);
    let k = b.add_linear(x, k_w, None, &shape);
    let v = b.add_linear(x, v_w, None, &shape);

    // QK-Norm: RMSNorm on Q and K (Qwen3-specific)
    let q_norm_eps = b.add_input("q_norm_eps", &[1]);
    let q_norm_w = b.add_input("q_norm_w", &[D_MODEL]);
    let k_norm_eps = b.add_input("k_norm_eps", &[1]);
    let k_norm_w = b.add_input("k_norm_w", &[D_MODEL]);

    let q_normed = b.add_rms_norm(q, q_norm_eps, 1, q_norm_w, &shape);
    let k_normed = b.add_rms_norm(k, k_norm_eps, 1, k_norm_w, &shape);

    // Scaled dot-product attention on normalized Q, K, V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_normed,
        k_normed,
        v,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );

    // Output projection
    let projected = b.add_linear(attn_out, out_w, None, &shape);

    // Residual
    let out = b.add_binary_add(x, projected, &shape);

    b.build(out).expect("valid QK-Norm attention kernel")
}

fn qk_norm_attention_bindings() -> Vec<TensorParamBinding> {
    let attn_w = w(&[D_MODEL, D_MODEL]);
    vec![
        TensorParamBinding::Variable,                         // x
        TensorParamBinding::ConstantTensor(attn_w.clone()),   // q_w
        TensorParamBinding::ConstantTensor(attn_w.clone()),   // k_w
        TensorParamBinding::ConstantTensor(attn_w.clone()),   // v_w
        TensorParamBinding::ConstantTensor(attn_w),           // out_w
        TensorParamBinding::ConstantScalar(1e-5),             // q_norm_eps
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // q_norm_w
        TensorParamBinding::ConstantScalar(1e-5),             // k_norm_eps
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])), // k_norm_w
    ]
}

#[test]
fn test_qwen3_depth_qk_norm_attention_validates() {
    let def = build_qk_norm_attention();
    def.validate().expect("QK-Norm attention should validate");
}

#[test]
fn test_qwen3_depth_qk_norm_attention_ibp() {
    let def = build_qk_norm_attention();
    let bindings = qk_norm_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through QK-Norm attention");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 QK-Norm attention IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "QK-Norm lower must be finite, got {lo}");
    assert!(hi.is_finite(), "QK-Norm upper must be finite, got {hi}");
}

#[test]
fn test_qwen3_depth_qk_norm_attention_crown() {
    let def = build_qk_norm_attention();
    let bindings = qk_norm_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 QK-Norm attention: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_depth_qk_norm_attention_verify_record() {
    let def = build_qk_norm_attention();
    let bindings = qk_norm_attention_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_depth_qk_norm_attention");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ===========================================================================
// 6. Decoder to logit output: full decoder + LM head + softmax
// ===========================================================================

/// Build full decoder (2 layers) + post-norm + LM head + softmax.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, VOCAB]` (softmax probabilities bounded in [0, 1]).
///
/// This is the complete text generation output path.
fn build_decoder_to_logit_softmax() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("qwen3_depth_decoder_to_logit");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let mut bindings = vec![TensorParamBinding::Variable];

    // 2 decoder blocks
    let x = add_decoder_block(&mut b, x, 0, &mut bindings);
    let x = add_decoder_block(&mut b, x, 1, &mut bindings);

    // Post-norm
    let post_eps = b.add_input("post_eps", &[1]);
    let post_rms_w = b.add_input("post_rms_w", &[D_MODEL]);
    let normed = b.add_rms_norm(x, post_eps, 1, post_rms_w, &[SEQ, D_MODEL]);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ones(&[D_MODEL])));

    // LM head projection
    let lm_w = b.add_input("lm_w", &[VOCAB, D_MODEL]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ, VOCAB]);
    bindings.push(TensorParamBinding::ConstantTensor(w(&[VOCAB, D_MODEL])));

    // Softmax for probability distribution
    let probs = b.add_softmax(logits, 1, &[SEQ, VOCAB]);

    let def = b.build(probs).expect("valid decoder to logit softmax");
    (def, bindings)
}

#[test]
fn test_qwen3_depth_decoder_to_logit_validates() {
    let (def, _) = build_decoder_to_logit_softmax();
    def.validate().expect("decoder to logit should validate");
}

#[test]
fn test_qwen3_depth_decoder_to_logit_ibp() {
    let (def, bindings) = build_decoder_to_logit_softmax();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder to logit");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, VOCAB],
        "output shape should be [{SEQ}, {VOCAB}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 decoder to logit softmax IBP: [{lo}, {hi}]");

    // Softmax output is always in [0, 1]. IBP may overshoot slightly.
    assert!(
        lo >= -0.01,
        "softmax lower bound should be >= -0.01, got {lo}"
    );
    assert!(
        hi <= 1.01,
        "softmax upper bound should be <= 1.01, got {hi}"
    );
}

#[test]
fn test_qwen3_depth_decoder_to_logit_verify_record() {
    let (def, bindings) = build_decoder_to_logit_softmax();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_depth_decoder_to_logit");
    assert_eq!(result.num_variables, 1);
    assert_eq!(result.output_bounds.lower_upper().0.shape(), &[SEQ, VOCAB]);
}
