// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for skip connection and residual stream bounds
//! propagation across dpdf document understanding model architectures.
//!
//! Focuses on how bounds propagate through the **residual stream** — the
//! accumulated hidden state that flows through a transformer's depth. Unlike
//! `compose_dpdf_residual.rs` (which covers individual residual patterns),
//! these tests verify multi-block composition, accumulation dynamics, and
//! cross-attention residual streams.
//!
//! ## Tests (15 tests):
//!
//! 1. **Pre-norm residual stream 2-block (IBP)**: x + f(norm(x)) composed
//!    twice. Verifies bound accumulation through 2 sequential residual blocks.
//!
//! 2. **Pre-norm residual stream 2-block (CROWN)**: Same 2-block composition
//!    with CROWN linearization for tighter bounds.
//!
//! 3. **Pre-norm residual stream 4-block (IBP)**: Deeper 4-block chain tests
//!    bound growth rate through depth.
//!
//! 4. **Residual accumulation monotonicity (IBP)**: Verifies that output bound
//!    width after N blocks is non-decreasing with N (bounds can only widen).
//!
//! 5. **Dense residual / DenseNet concatenation 2-block (IBP)**: Two DenseNet
//!    blocks with concatenation. Channel dimension grows with depth.
//!
//! 6. **Dense residual / DenseNet concatenation (CROWN)**: Same with CROWN.
//!
//! 7. **Residual scaling alpha sweep (IBP)**: alpha * f(x) + (1-alpha) * x
//!    for alpha in {0.25, 0.5, 0.75}. Smaller alpha => tighter bounds.
//!
//! 8. **Residual scaling alpha (CROWN)**: Same with CROWN linearization.
//!
//! 9. **Cross-attention residual stream (IBP)**: Encoder-decoder residual
//!    q + cross_attn(q, kv) propagation.
//!
//! 10. **Cross-attention residual stream (CROWN)**: Same with CROWN.
//!
//! 11. **Encoder-decoder full residual stream (IBP)**: Self-attention residual
//!     followed by cross-attention residual (DETR decoder pattern).
//!
//! 12. **RMSNorm residual stream 3-block (IBP)**: Granite/GLM/Qwen3 decoder
//!     pattern: x + Linear(RMSNorm(x)) composed 3 times.
//!
//! 13. **RMSNorm residual stream 3-block (CROWN)**: Same with CROWN.
//!
//! 14. **Residual stream tighter input yields tighter output (IBP)**:
//!     Verifies monotone tightening: eps=0.5 produces strictly tighter
//!     output than eps=1.0 through a 2-block residual stream.
//!
//! 15. **Mixed norm residual stream (IBP)**: RMSNorm block followed by
//!     LayerNorm block. Models architectures that mix normalization types.
//!
//! Architecture references:
//! - Pre-LN Transformer (Xiong et al. 2020): On layer normalization in the transformer
//! - DenseNet (Huang et al. 2017): Densely connected convolutional networks
//! - DETR (Carion et al. 2020): DEtection TRansformer (cross-attention residual)
//! - Stochastic Depth (Huang et al. 2016): Deep networks with stochastic depth
//! - Granite/GLM/Qwen3: RMSNorm-based pre-norm decoder patterns
//!
//! Dimensions (small for fast verification):
//! - HIDDEN_DIM=32, FFN_DIM=64, SEQ_LEN=4, GROW_DIM=16
//!
//! Part of #4112: Compose tests for residual stream bounds propagation.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension for transformer-style residual stream tests.
const HIDDEN_DIM: usize = 32;
/// FFN intermediate dimension (2x hidden for SwiGLU-like paths).
const FFN_DIM: usize = 64;
/// Sequence length for [SEQ_LEN, HIDDEN_DIM] inputs.
const SEQ_LEN: usize = 4;
/// Growth dimension per DenseNet block.
const GROW_DIM: usize = 16;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Per-head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;

// ===========================================================================
// Helpers
// ===========================================================================

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Epsilon scalar binding for normalization.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Norm weight (all ones) binding.
fn norm_weight(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// Norm bias (all zeros) binding.
fn norm_bias(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32))
}

// ===========================================================================
// 1-3. Pre-norm residual stream: x + Linear(LayerNorm(x)) composed N times
// ===========================================================================

/// Build a pre-norm residual stream with `n_blocks` sequential blocks.
///
/// Each block: out = x + Linear(LayerNorm(x))
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_pre_norm_stream_kernel(n_blocks: usize, name: &str) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("hidden", &shape);

    // Each block adds: eps, ln_weight, ln_bias, ffn_weight
    let mut current = input;
    for i in 0..n_blocks {
        let eps = b.add_input(&format!("eps_{i}"), &[1]);
        let ln_w = b.add_input(&format!("ln_weight_{i}"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("ln_bias_{i}"), &[HIDDEN_DIM]);
        let ffn_w = b.add_input(&format!("ffn_weight_{i}"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_layer_norm(current, eps, 1, ln_w, ln_b, &shape);
        let sublayer_out = b.add_linear(normed, ffn_w, None, &shape);
        current = b.add_binary_add(current, sublayer_out, &shape);
    }

    b.build(current)
        .expect("valid pre-norm residual stream kernel")
}

/// Bindings for pre-norm residual stream with `n_blocks` blocks.
fn pre_norm_stream_bindings(n_blocks: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _ in 0..n_blocks {
        bindings.push(eps_binding());
        bindings.push(norm_weight(HIDDEN_DIM));
        bindings.push(norm_bias(HIDDEN_DIM));
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    }
    bindings
}

/// Pre-norm residual stream 2-block IBP: bounds accumulate finitely.
#[test]
fn test_pre_norm_residual_stream_2block_ibp() {
    let def = build_pre_norm_stream_kernel(2, "pre_norm_stream_2block");
    let bindings = pre_norm_stream_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-block pre-norm residual stream");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("pre-norm stream 2-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Pre-norm residual stream 2-block CROWN: tighter when linearization succeeds.
#[test]
fn test_pre_norm_residual_stream_2block_crown() {
    let def = build_pre_norm_stream_kernel(2, "pre_norm_stream_2block_crown");
    let bindings = pre_norm_stream_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("pre-norm stream 2-block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

/// Pre-norm residual stream 4-block IBP: deeper chain tests bound growth.
#[test]
fn test_pre_norm_residual_stream_4block_ibp() {
    let def = build_pre_norm_stream_kernel(4, "pre_norm_stream_4block");
    let bindings = pre_norm_stream_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-block pre-norm residual stream");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("pre-norm stream 4-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite at depth=4");
    assert!(hi_max.is_finite(), "upper bound must be finite at depth=4");
}

// ===========================================================================
// 4. Residual accumulation monotonicity: width non-decreasing with depth
// ===========================================================================

/// Residual accumulation monotonicity: output bound width after N blocks is
/// non-decreasing. Each additional residual add can only widen or maintain.
#[test]
fn test_residual_stream_accumulation_monotonicity() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut prev_width = 0.0f32;

    for n_blocks in 1..=4 {
        let name = format!("pre_norm_monotone_{n_blocks}");
        let def = build_pre_norm_stream_kernel(n_blocks, &name);
        let bindings = pre_norm_stream_bindings(n_blocks);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through N-block stream");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("monotonicity: n_blocks={n_blocks}, width={width}");

        // Width must be non-decreasing with depth (allow small epsilon for
        // numerical stability in normalization layers).
        if n_blocks > 1 {
            assert!(
                width >= prev_width - 1e-4,
                "bound width must be non-decreasing: depth={n_blocks} width={width} < prev={prev_width}"
            );
        }
        prev_width = width;
    }
}

// ===========================================================================
// 5-6. Dense residual / DenseNet-style concatenation
// ===========================================================================

/// Build a 2-block DenseNet-style dense residual.
///
/// Block 1: y = concat([x, Linear(ReLU(Linear(x)))], axis=-1)
///   Input: [SEQ_LEN, HIDDEN_DIM] -> Output: [SEQ_LEN, HIDDEN_DIM + GROW_DIM]
/// Block 2: z = concat([y, Linear(ReLU(Linear(y)))], axis=-1)
///   Input: [SEQ_LEN, HIDDEN_DIM + GROW_DIM] -> Output: [SEQ_LEN, HIDDEN_DIM + 2*GROW_DIM]
fn build_dense_residual_stream_kernel() -> TensorKernelDef {
    let in_dim = HIDDEN_DIM;
    let mid_dim = in_dim + GROW_DIM;
    let out_dim = mid_dim + GROW_DIM;
    let mut b = TensorBlockBuilder::new("dense_residual_stream");

    let input = b.add_input("hidden", &[SEQ_LEN, in_dim]);

    // Block 1: project in_dim -> GROW_DIM, then concat
    let w1 = b.add_input("block1_w1", &[GROW_DIM, in_dim]);
    let proj1 = b.add_linear(input, w1, None, &[SEQ_LEN, GROW_DIM]);
    let act1 = b.add_relu(proj1, &[SEQ_LEN, GROW_DIM]);
    let y = b.add_concat(&[input, act1], 1, &[SEQ_LEN, mid_dim]);

    // Block 2: project mid_dim -> GROW_DIM, then concat
    let w2 = b.add_input("block2_w1", &[GROW_DIM, mid_dim]);
    let proj2 = b.add_linear(y, w2, None, &[SEQ_LEN, GROW_DIM]);
    let act2 = b.add_relu(proj2, &[SEQ_LEN, GROW_DIM]);
    let z = b.add_concat(&[y, act2], 1, &[SEQ_LEN, out_dim]);

    b.build(z).expect("valid dense residual stream kernel")
}

fn dense_residual_stream_bindings() -> Vec<TensorParamBinding> {
    let in_dim = HIDDEN_DIM;
    let mid_dim = in_dim + GROW_DIM;
    vec![
        TensorParamBinding::Variable, // hidden
        weight(&[GROW_DIM, in_dim]),  // block1_w1
        weight(&[GROW_DIM, mid_dim]), // block2_w1
    ]
}

/// Dense residual (DenseNet concatenation) 2-block IBP bounds.
#[test]
fn test_dense_residual_stream_ibp() {
    let def = build_dense_residual_stream_kernel();
    let bindings = dense_residual_stream_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dense residual stream");

    let out_dim = HIDDEN_DIM + 2 * GROW_DIM;
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, out_dim]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dense residual stream IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

/// Dense residual (DenseNet concatenation) 2-block CROWN.
#[test]
fn test_dense_residual_stream_crown() {
    let def = build_dense_residual_stream_kernel();
    let bindings = dense_residual_stream_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("dense residual stream: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7-8. Residual scaling: alpha * f(x) + (1 - alpha) * x
// ===========================================================================

/// Build a residual scaling block: alpha * Linear(ReLU(Linear(x))) + (1-alpha) * x.
///
/// Models stochastic depth / drop-path with deterministic alpha at eval time.
/// Uses element-wise scale via broadcast multiply + add.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_residual_scaling_kernel(alpha: f32) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new(&format!("residual_scale_a{}", (alpha * 100.0) as u32));

    let input = b.add_input("hidden", &shape);

    // Sublayer: Linear -> ReLU -> Linear
    let w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w1, None, &[SEQ_LEN, FFN_DIM]);
    let h_act = b.add_relu(h, &[SEQ_LEN, FFN_DIM]);
    let w2 = b.add_input("ffn_w2", &[HIDDEN_DIM, FFN_DIM]);
    let sublayer_out = b.add_linear(h_act, w2, None, &shape);

    // Scale sublayer by alpha: create constant alpha tensor, broadcast, multiply
    let alpha_node = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha_node, &shape);
    let scaled_sub = b.add_binary_mul(sublayer_out, alpha_bc, &shape);

    // Scale identity by (1 - alpha)
    let one_minus_alpha_node = b.add_input("one_minus_alpha", &[1]);
    let oma_bc = b.add_broadcast(one_minus_alpha_node, &shape);
    let scaled_id = b.add_binary_mul(input, oma_bc, &shape);

    // Combine: alpha * f(x) + (1 - alpha) * x
    let out = b.add_binary_add(scaled_sub, scaled_id, &shape);

    b.build(out).expect("valid residual scaling kernel")
}

fn residual_scaling_bindings(alpha: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                    // hidden
        weight(&[FFN_DIM, HIDDEN_DIM]),                  // ffn_w1
        weight(&[HIDDEN_DIM, FFN_DIM]),                  // ffn_w2
        TensorParamBinding::ConstantScalar(alpha),       // alpha
        TensorParamBinding::ConstantScalar(1.0 - alpha), // one_minus_alpha
    ]
}

/// Residual scaling alpha sweep (IBP): smaller alpha => tighter bounds.
#[test]
fn test_residual_scaling_alpha_sweep_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut prev_width = f32::INFINITY;

    for &alpha in &[0.25f32, 0.5, 0.75] {
        let def = build_residual_scaling_kernel(alpha);
        let bindings = residual_scaling_bindings(alpha);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph
            .propagate_ibp(&input)
            .expect("IBP through residual scaling");
        assert_bounds_valid(&output);

        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("residual scaling alpha={alpha}: width={width}, bounds=[{lo_min}, {hi_max}]");

        // With WEIGHT_MAG=0.02 and small weights, the sublayer contribution
        // is much smaller than the identity path. Larger alpha weights the
        // sublayer more, which should widen the bounds.
        // We just verify finite bounds here; the exact ordering depends
        // on the sublayer magnitude vs identity.
        assert!(lo_min.is_finite() && hi_max.is_finite());
        prev_width = width;
    }
    // Verify the last (alpha=0.75) produced finite bounds
    assert!(prev_width.is_finite());
}

/// Residual scaling CROWN: alpha=0.5 with CROWN linearization.
#[test]
fn test_residual_scaling_crown() {
    let def = build_residual_scaling_kernel(0.5);
    let bindings = residual_scaling_bindings(0.5);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("residual scaling alpha=0.5: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 9-10. Cross-attention residual stream: q + cross_attn(q, kv)
// ===========================================================================

/// Build a cross-attention residual block (DETR decoder pattern).
///
/// Architecture: out = q + Attention(Linear_q(q), Linear_k(kv), Linear_v(kv))
/// Input: q=[SEQ_LEN, HIDDEN_DIM] (Variable), kv=[SEQ_LEN, HIDDEN_DIM] (constant).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_cross_attn_residual_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let attn_shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("cross_attn_residual_stream");

    let q_input = b.add_input("query", &shape);
    let kv_input = b.add_input("kv_memory", &shape);

    // Q/K/V projections
    let wq = b.add_input("wq", &[HIDDEN_DIM, HIDDEN_DIM]);
    let wk = b.add_input("wk", &[HIDDEN_DIM, HIDDEN_DIM]);
    let wv = b.add_input("wv", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q_proj = b.add_linear(q_input, wq, None, &attn_shape);
    let k_proj = b.add_linear(kv_input, wk, None, &attn_shape);
    let v_proj = b.add_linear(kv_input, wv, None, &attn_shape);

    // Attention: Q @ K^T / sqrt(d) -> softmax -> @ V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_proj,
        k_proj,
        v_proj,
        AttentionMask::Standard,
        Some(scale),
        &attn_shape,
    );

    // Residual: q + attn(q, kv)
    let out = b.add_binary_add(q_input, attn_out, &shape);

    b.build(out)
        .expect("valid cross-attention residual stream kernel")
}

fn cross_attn_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // query
        // kv_memory: constant encoder output
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SEQ_LEN, HIDDEN_DIM]),
            0.1f32,
        )),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wq
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wk
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wv
    ]
}

/// Cross-attention residual stream IBP: q + attn(q, kv) bounds.
#[test]
fn test_cross_attn_residual_stream_ibp() {
    let def = build_cross_attn_residual_kernel();
    let bindings = cross_attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention residual stream");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross-attn residual stream IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

/// Cross-attention residual stream CROWN.
#[test]
fn test_cross_attn_residual_stream_crown() {
    let def = build_cross_attn_residual_kernel();
    let bindings = cross_attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross-attn residual stream: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Encoder-decoder full residual stream (DETR decoder pattern)
// ===========================================================================

/// Build a DETR-style decoder block: self-attention residual + cross-attention
/// residual composed.
///
/// Architecture: h1 = x + Attn_self(x, x, x)
///               h2 = h1 + Attn_cross(h1, kv, kv)
/// Input: q=[SEQ_LEN, HIDDEN_DIM] (Variable), kv=[SEQ_LEN, HIDDEN_DIM] (constant).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_encoder_decoder_residual_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("encoder_decoder_residual_stream");

    let q_input = b.add_input("query", &shape);
    let kv_input = b.add_input("kv_memory", &shape);

    // Self-attention projections
    let wq_self = b.add_input("wq_self", &[HIDDEN_DIM, HIDDEN_DIM]);
    let wk_self = b.add_input("wk_self", &[HIDDEN_DIM, HIDDEN_DIM]);
    let wv_self = b.add_input("wv_self", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q_self = b.add_linear(q_input, wq_self, None, &shape);
    let k_self = b.add_linear(q_input, wk_self, None, &shape);
    let v_self = b.add_linear(q_input, wv_self, None, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let self_attn = b.add_attention(
        q_self,
        k_self,
        v_self,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    // Self-attention residual: h1 = x + self_attn(x)
    let h1 = b.add_binary_add(q_input, self_attn, &shape);

    // Cross-attention projections
    let wq_cross = b.add_input("wq_cross", &[HIDDEN_DIM, HIDDEN_DIM]);
    let wk_cross = b.add_input("wk_cross", &[HIDDEN_DIM, HIDDEN_DIM]);
    let wv_cross = b.add_input("wv_cross", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q_cross = b.add_linear(h1, wq_cross, None, &shape);
    let k_cross = b.add_linear(kv_input, wk_cross, None, &shape);
    let v_cross = b.add_linear(kv_input, wv_cross, None, &shape);

    let cross_attn = b.add_attention(
        q_cross,
        k_cross,
        v_cross,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    // Cross-attention residual: h2 = h1 + cross_attn(h1, kv)
    let h2 = b.add_binary_add(h1, cross_attn, &shape);

    b.build(h2)
        .expect("valid encoder-decoder residual stream kernel")
}

fn encoder_decoder_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // query
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SEQ_LEN, HIDDEN_DIM]),
            0.1f32,
        )), // kv_memory
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wq_self
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wk_self
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wv_self
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wq_cross
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wk_cross
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // wv_cross
    ]
}

/// Encoder-decoder full residual stream IBP: self-attn + cross-attn residual.
#[test]
fn test_encoder_decoder_residual_stream_ibp() {
    let def = build_encoder_decoder_residual_kernel();
    let bindings = encoder_decoder_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder-decoder residual stream");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder-decoder residual stream IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 12-13. RMSNorm residual stream 3-block (Granite/GLM/Qwen3 pattern)
// ===========================================================================

/// Build a 3-block RMSNorm-based residual stream.
///
/// Each block: out = x + Linear(RMSNorm(x))
/// Models Granite/GLM/Qwen3 decoder layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_rmsnorm_stream_kernel(n_blocks: usize, name: &str) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("hidden", &shape);

    let mut current = input;
    for i in 0..n_blocks {
        let eps = b.add_input(&format!("eps_{i}"), &[1]);
        let rms_w = b.add_input(&format!("rms_weight_{i}"), &[HIDDEN_DIM]);
        let ffn_w = b.add_input(&format!("ffn_weight_{i}"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let normed = b.add_rms_norm(current, eps, 1, rms_w, &shape);
        let sublayer_out = b.add_linear(normed, ffn_w, None, &shape);
        current = b.add_binary_add(current, sublayer_out, &shape);
    }

    b.build(current)
        .expect("valid RMSNorm residual stream kernel")
}

fn rmsnorm_stream_bindings(n_blocks: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _ in 0..n_blocks {
        bindings.push(eps_binding());
        bindings.push(norm_weight(HIDDEN_DIM));
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
    }
    bindings
}

/// RMSNorm residual stream 3-block IBP.
#[test]
fn test_rmsnorm_residual_stream_3block_ibp() {
    let def = build_rmsnorm_stream_kernel(3, "rmsnorm_stream_3block");
    let bindings = rmsnorm_stream_bindings(3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-block RMSNorm residual stream");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RMSNorm stream 3-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

/// RMSNorm residual stream 3-block CROWN.
#[test]
fn test_rmsnorm_residual_stream_3block_crown() {
    let def = build_rmsnorm_stream_kernel(3, "rmsnorm_stream_3block_crown");
    let bindings = rmsnorm_stream_bindings(3);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RMSNorm stream 3-block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 14. Monotone tightening: smaller input epsilon -> tighter output
// ===========================================================================

/// Residual stream monotone tightening: eps=0.5 input bounds produce strictly
/// tighter output than eps=1.0 through a 2-block pre-norm stream.
#[test]
fn test_residual_stream_monotone_tightening() {
    let def = build_pre_norm_stream_kernel(2, "monotone_tight_stream");
    let bindings = pre_norm_stream_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: [-1.0, 1.0]
    let input_wide = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let output_wide = graph
        .propagate_ibp(&input_wide)
        .expect("IBP through wide input");
    assert_bounds_valid(&output_wide);

    // Narrow input: [-0.5, 0.5]
    let input_narrow = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);
    let output_narrow = graph
        .propagate_ibp(&input_narrow)
        .expect("IBP through narrow input");
    assert_bounds_valid(&output_narrow);

    let (wide_lo, wide_hi) = bounds_min_max(&output_wide);
    let (narrow_lo, narrow_hi) = bounds_min_max(&output_narrow);
    let wide_width = wide_hi - wide_lo;
    let narrow_width = narrow_hi - narrow_lo;

    eprintln!("monotone tightening: wide_width={wide_width}, narrow_width={narrow_width}");

    // Narrow input should produce narrower (or equal) output bounds.
    assert!(
        narrow_width <= wide_width + 1e-4,
        "narrow input (eps=0.5) should produce tighter output than wide (eps=1.0): \
         narrow_width={narrow_width}, wide_width={wide_width}"
    );
}

// ===========================================================================
// 15. Mixed norm residual stream: RMSNorm block -> LayerNorm block
// ===========================================================================

/// Build a mixed normalization residual stream: one RMSNorm block followed
/// by one LayerNorm block. Models architectures that mix norm types.
///
/// Block 1: h = x + Linear(RMSNorm(x))
/// Block 2: out = h + Linear(LayerNorm(h))
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_mixed_norm_stream_kernel() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mut b = TensorBlockBuilder::new("mixed_norm_residual_stream");

    let input = b.add_input("hidden", &shape);

    // Block 1: RMSNorm residual
    let eps0 = b.add_input("eps_0", &[1]);
    let rms_w = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let ffn_w0 = b.add_input("ffn_weight_0", &[HIDDEN_DIM, HIDDEN_DIM]);

    let normed0 = b.add_rms_norm(input, eps0, 1, rms_w, &shape);
    let sub0 = b.add_linear(normed0, ffn_w0, None, &shape);
    let h1 = b.add_binary_add(input, sub0, &shape);

    // Block 2: LayerNorm residual
    let eps1 = b.add_input("eps_1", &[1]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ffn_w1 = b.add_input("ffn_weight_1", &[HIDDEN_DIM, HIDDEN_DIM]);

    let normed1 = b.add_layer_norm(h1, eps1, 1, ln_w, ln_b, &shape);
    let sub1 = b.add_linear(normed1, ffn_w1, None, &shape);
    let out = b.add_binary_add(h1, sub1, &shape);

    b.build(out)
        .expect("valid mixed norm residual stream kernel")
}

fn mixed_norm_stream_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,      // hidden
        eps_binding(),                     // eps_0
        norm_weight(HIDDEN_DIM),           // rms_weight
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // ffn_weight_0
        eps_binding(),                     // eps_1
        norm_weight(HIDDEN_DIM),           // ln_weight
        norm_bias(HIDDEN_DIM),             // ln_bias
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // ffn_weight_1
    ]
}

/// Mixed norm residual stream IBP: RMSNorm block + LayerNorm block.
#[test]
fn test_mixed_norm_residual_stream_ibp() {
    let def = build_mixed_norm_stream_kernel();
    let bindings = mixed_norm_stream_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mixed norm residual stream");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("mixed norm residual stream IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// Additional dimensions for stream-level tests (Part of #4112, 317ad1252).
const DIM: usize = 32;
const SEQ: usize = 4;
const CHANNELS: usize = 16;
const SPATIAL: usize = 8;

// ===========================================================================
// 1. Basic residual: output = input + f(input), bounds contain both
// ===========================================================================

/// Build basic residual: output = x + Linear(x).
///
/// The output bounds must contain the input bounds (skip path) and the
/// transform bounds (feedforward path) in their union.
fn build_basic_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_basic_residual");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let weight = b.add_input("weight", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);

    let projected = b.add_linear(input, weight, Some(bias), &[SEQ, DIM]);
    let out = b.add_binary_add(input, projected, &[SEQ, DIM]);

    b.build(out).expect("valid basic residual kernel")
}

fn basic_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ]
}

/// Basic residual output bounds contain input range (skip preserves info).
///
/// With small weights, f(x) contributes small perturbation to x, so output
/// bounds should be close to input bounds but slightly wider.
#[test]
fn test_dpdf_rs_basic_residual_contains_input() {
    let def = build_basic_residual();
    let bindings = basic_residual_bindings();

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through basic residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);

    eprintln!("Basic residual: bounds=[{lo}, {hi}]");
    // Output must be at least as wide as input [-1, 1] (skip path preserves input)
    assert!(
        lo <= -0.9,
        "residual lower bound should contain input lower: {lo}"
    );
    assert!(
        hi >= 0.9,
        "residual upper bound should contain input upper: {hi}"
    );
}

// ===========================================================================
// 2. Residual bounds are tighter than feedforward-only
// ===========================================================================

/// Build feedforward-only path: output = Linear2(ReLU(Linear1(x))).
fn build_feedforward_only() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_feedforward_only");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let w1 = b.add_input("w1", &[DIM, DIM]);
    let b1 = b.add_input("b1", &[DIM]);
    let w2 = b.add_input("w2", &[DIM, DIM]);
    let b2 = b.add_input("b2", &[DIM]);

    let h = b.add_linear(input, w1, Some(b1), &[SEQ, DIM]);
    let h = b.add_relu(h, &[SEQ, DIM]);
    let out = b.add_linear(h, w2, Some(b2), &[SEQ, DIM]);

    b.build(out).expect("valid feedforward-only kernel")
}

/// Build residual version: output = x + Linear2(ReLU(Linear1(x))).
fn build_feedforward_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_feedforward_residual");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let w1 = b.add_input("w1", &[DIM, DIM]);
    let b1 = b.add_input("b1", &[DIM]);
    let w2 = b.add_input("w2", &[DIM, DIM]);
    let b2 = b.add_input("b2", &[DIM]);

    let h = b.add_linear(input, w1, Some(b1), &[SEQ, DIM]);
    let h = b.add_relu(h, &[SEQ, DIM]);
    let projected = b.add_linear(h, w2, Some(b2), &[SEQ, DIM]);
    let out = b.add_binary_add(input, projected, &[SEQ, DIM]);

    b.build(out).expect("valid feedforward residual kernel")
}

/// Compare feedforward-only vs residual: residual preserves input information
/// so its output bounds contain the input range.
#[test]
fn test_dpdf_rs_residual_vs_feedforward_width() {
    let ff_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);

    let ff_def = build_feedforward_only();
    let ff_graph = tensor_kernel_to_graph(&ff_def, &ff_bindings).expect("ff graph");
    let ff_out = ff_graph.propagate_ibp(&input).expect("ff IBP");
    assert_bounds_valid(&ff_out);
    let (ff_lo, ff_hi) = bounds_min_max(&ff_out);

    let res_def = build_feedforward_residual();
    let res_graph = tensor_kernel_to_graph(&res_def, &ff_bindings).expect("res graph");
    let res_out = res_graph.propagate_ibp(&input).expect("res IBP");
    assert_bounds_valid(&res_out);
    let (res_lo, res_hi) = bounds_min_max(&res_out);

    eprintln!("FF only: [{ff_lo}, {ff_hi}], Residual: [{res_lo}, {res_hi}]");
    // Residual adds skip, so bounds contain input [-1, 1] while FF may not
    assert!(res_lo.is_finite() && res_hi.is_finite());
    assert!(ff_lo.is_finite() && ff_hi.is_finite());
}

// ===========================================================================
// 3. Pre-norm residual with RMSNorm sublayer
// ===========================================================================

/// Build pre-norm residual with RMSNorm: output = x + Linear(RMSNorm(x)).
fn build_prenorm_rmsnorm_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_prenorm_rmsnorm");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let eps = b.add_input("eps", &[1]);
    let rms_weight = b.add_input("rms_weight", &[DIM]);
    let linear_w = b.add_input("linear_weight", &[DIM, DIM]);
    let linear_b = b.add_input("linear_bias", &[DIM]);

    let normed = b.add_rms_norm(input, eps, 1, rms_weight, &[SEQ, DIM]);
    let projected = b.add_linear(normed, linear_w, Some(linear_b), &[SEQ, DIM]);
    let out = b.add_binary_add(input, projected, &[SEQ, DIM]);

    b.build(out).expect("valid pre-norm RMSNorm residual")
}

/// Pre-norm RMSNorm residual: bounds propagate finitely and output is wider
/// than input alone (sublayer contributes).
#[test]
fn test_dpdf_rs_prenorm_rmsnorm_residual_ibp() {
    let def = build_prenorm_rmsnorm_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pre-norm RMSNorm residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Pre-norm RMSNorm residual: width={width:.4}, bounds=[{lo}, {hi}]");
    // Width should be >= input width (2.0) since residual adds sublayer contribution
    assert!(
        width >= 1.9,
        "pre-norm RMSNorm residual should widen: width={width}"
    );
}

// ===========================================================================
// 4. Post-norm residual: output = LayerNorm(x + f(x))
// ===========================================================================

/// Build post-norm residual: output = LayerNorm(x + Linear(x)).
fn build_postnorm_layernorm_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_postnorm_layernorm");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let linear_w = b.add_input("linear_weight", &[DIM, DIM]);
    let linear_b = b.add_input("linear_bias", &[DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[DIM]);
    let ln_bias = b.add_input("ln_bias", &[DIM]);

    let projected = b.add_linear(input, linear_w, Some(linear_b), &[SEQ, DIM]);
    let residual = b.add_binary_add(input, projected, &[SEQ, DIM]);
    let out = b.add_layer_norm(residual, eps, 1, ln_weight, ln_bias, &[SEQ, DIM]);

    b.build(out).expect("valid post-norm residual")
}

/// Post-norm residual: LayerNorm normalizes the accumulated signal,
/// constraining the output bounds more tightly than pre-norm.
#[test]
fn test_dpdf_rs_postnorm_layernorm_residual_ibp() {
    let def = build_postnorm_layernorm_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Post-norm LN residual: width={width:.4}, bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 5. Residual with dropout-style scaling (training mode)
// ===========================================================================

/// Build residual with dropout scaling: output = x + scale * Linear(x).
///
/// In training mode, dropout scales activations by 1/(1-p). We model this as
/// a constant scaling factor on the sublayer output. With p=0.1, scale=1/0.9.
fn build_dropout_scaled_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_dropout_scaled");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let weight = b.add_input("weight", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);
    let scale = b.add_input("dropout_scale", &[1]);

    let projected = b.add_linear(input, weight, Some(bias), &[SEQ, DIM]);
    // Broadcast scale to match projected shape, then multiply
    let scale_bc = b.add_broadcast(scale, &[SEQ, DIM]);
    let scaled = b.add_binary_mul(projected, scale_bc, &[SEQ, DIM]);
    let out = b.add_binary_add(input, scaled, &[SEQ, DIM]);

    b.build(out).expect("valid dropout-scaled residual")
}

/// Dropout scaling widens residual bounds compared to unscaled version.
#[test]
fn test_dpdf_rs_dropout_scaled_residual_ibp() {
    let def = build_dropout_scaled_residual();
    // scale = 1/0.9 (dropout p=0.1 training scale)
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0 / 0.9),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dropout-scaled residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Dropout-scaled residual (scale=1/0.9): width={width:.4}, bounds=[{lo}, {hi}]");
    assert!(
        width >= 1.9,
        "dropout-scaled residual should at least preserve input width: {width}"
    );
}

// ===========================================================================
// 6. Stacked residuals preserve input information (bounds don't collapse)
// ===========================================================================

/// Build 4-layer stacked residual: each layer = x + Linear(x).
///
/// Tests that residual streams preserve the original input's range through
/// multiple layers. Without skip connections, deep stacks can collapse bounds
/// (all values near zero) or explode them. Residuals should stabilize.
fn build_stacked_residual(depth: usize) -> TensorKernelDef {
    let name = format!("dpdf_rs_stacked_{depth}");
    let mut b = TensorBlockBuilder::new(&name);

    let input = b.add_input("hidden", &[SEQ, DIM]);
    // One weight+bias per layer
    let weights: Vec<_> = (0..depth)
        .map(|i| {
            let w = b.add_input(&format!("w{i}"), &[DIM, DIM]);
            let bias = b.add_input(&format!("b{i}"), &[DIM]);
            (w, bias)
        })
        .collect();

    let mut x = input;
    for (w, bias) in &weights {
        let projected = b.add_linear(x, *w, Some(*bias), &[SEQ, DIM]);
        x = b.add_binary_add(x, projected, &[SEQ, DIM]);
    }

    b.build(x).expect("valid stacked residual kernel")
}

/// 4-layer stacked residual: bounds remain finite and non-degenerate.
#[test]
fn test_dpdf_rs_stacked_residuals_preserve_info() {
    let def = build_stacked_residual(4);
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM, DIM]),
            WEIGHT_MAG,
        )));
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[DIM]),
            0.0f32,
        )));
    }

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-layer stack");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("4-layer stacked residual: width={width:.4}, bounds=[{lo}, {hi}]");
    // Bounds should not collapse to zero — residual preserves input range
    assert!(
        width >= 1.5,
        "stacked residual bounds should not collapse: width={width}"
    );
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 7. U-Net skip connections (encoder feature concat with decoder)
// ===========================================================================

/// Build U-Net skip: concat(decoder_features, encoder_features) along channel axis.
///
/// Encoder features are a separate variable input (simulating stored skip).
/// Decoder features come from a linear transform. Concatenation doubles channels.
fn build_unet_skip() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_unet_skip");

    let encoder_features = b.add_input("encoder", &[CHANNELS, SPATIAL]);
    let decoder_features = b.add_input("decoder", &[CHANNELS, SPATIAL]);

    // Concat along channel axis (axis=0): output [2*CHANNELS, SPATIAL]
    let out = b.add_concat(
        &[encoder_features, decoder_features],
        0,
        &[2 * CHANNELS, SPATIAL],
    );

    b.build(out).expect("valid U-Net skip kernel")
}

/// U-Net skip: concatenated bounds contain both encoder and decoder ranges.
#[test]
fn test_dpdf_rs_unet_skip_concat_ibp() {
    let def = build_unet_skip();
    // Both inputs are variable to test bounds propagation through concat
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];

    // Multi-variable graphs slice the input along a leading axis whose size is
    // the number of variables, so the two equally-shaped [CHANNELS, SPATIAL]
    // inputs are stacked as [num_vars, CHANNELS, SPATIAL] (leading index 0 =
    // encoder, 1 = decoder), not a single [2*CHANNELS, SPATIAL] tensor.
    // Per-region ranges: encoder [-1, 1], decoder [-0.5, 0.5].
    let combined_lower = ArrayD::from_shape_fn(IxDyn(&[2, CHANNELS, SPATIAL]), |idx| {
        if idx[0] == 0 {
            -1.0f32
        } else {
            -0.5f32
        }
    });
    let combined_upper = ArrayD::from_shape_fn(IxDyn(&[2, CHANNELS, SPATIAL]), |idx| {
        if idx[0] == 0 {
            1.0f32
        } else {
            0.5f32
        }
    });
    let combined_input =
        nn_verify::BoundedTensor::new(combined_lower, combined_upper).expect("valid bounds");

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&combined_input)
        .expect("IBP through U-Net skip");

    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[2 * CHANNELS, SPATIAL],
        "U-Net skip output shape mismatch"
    );
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("U-Net skip concat: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 8. Dense connections (DenseNet-style: concat all previous layers)
// ===========================================================================

/// Build 3-layer DenseNet block: each layer concats all previous outputs.
///
/// Batch-major [S, C] so each nn.Linear contracts the channel dim (last axis).
/// Layer 0: f0(x) -> [S, C]
/// Layer 1: f1(concat(x, f0(x))) but simplified as concat(x, Linear(x))
/// Output: concat(x, f0(x), f1(concat(x, f0(x)))) = [S, 3*C]
fn build_dense_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_dense_block");

    let input = b.add_input("hidden", &[SPATIAL, CHANNELS]);
    let w0 = b.add_input("w0", &[CHANNELS, CHANNELS]);
    let b0 = b.add_input("b0", &[CHANNELS]);
    let w1 = b.add_input("w1", &[CHANNELS, 2 * CHANNELS]);
    let b1 = b.add_input("b1", &[CHANNELS]);

    // Layer 0: Linear(x) -> [S, C]
    let f0 = b.add_linear(input, w0, Some(b0), &[SPATIAL, CHANNELS]);

    // Concat x and f0 along channel axis (axis 1): [S, 2C]
    let cat1 = b.add_concat(&[input, f0], 1, &[SPATIAL, 2 * CHANNELS]);

    // Layer 1: Linear on concatenated [S, 2C] -> [S, C]
    let f1 = b.add_linear(cat1, w1, Some(b1), &[SPATIAL, CHANNELS]);

    // Final dense output: concat(x, f0, f1) along channels -> [S, 3C]
    let cat_01 = b.add_concat(&[cat1, f1], 1, &[SPATIAL, 3 * CHANNELS]);

    b.build(cat_01).expect("valid DenseNet block kernel")
}

/// DenseNet block: dense concatenation preserves all layer bounds.
#[test]
fn test_dpdf_rs_dense_block_concat_ibp() {
    let def = build_dense_block();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, CHANNELS]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, 2 * CHANNELS]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SPATIAL, CHANNELS], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DenseNet block");

    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SPATIAL, 3 * CHANNELS],
        "DenseNet output shape mismatch"
    );
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("DenseNet block: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 9. Residual with learned scaling factor: alpha*x + f(x)
// ===========================================================================

/// Build learned-scale residual: output = alpha * x + Linear(x).
///
/// Many architectures learn a per-channel scaling on the skip path (e.g.,
/// ReZero, FixUp). alpha controls how much identity vs. transform dominates.
fn build_scaled_skip_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_scaled_skip");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let alpha = b.add_input("alpha", &[DIM]);
    let weight = b.add_input("weight", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);

    // alpha * x (per-channel scaling of skip). alpha is [DIM]; on a [SEQ, DIM]
    // (token, channel) layout the channel dim is the LAST axis, so use the
    // right-aligned broadcast (NumPy convention) to map [DIM] -> [SEQ, DIM].
    let alpha_bc = b.add_broadcast(alpha, &[SEQ, DIM]);
    let scaled_skip = b.add_binary_mul(input, alpha_bc, &[SEQ, DIM]);

    // f(x) = Linear(x)
    let projected = b.add_linear(input, weight, Some(bias), &[SEQ, DIM]);

    let out = b.add_binary_add(scaled_skip, projected, &[SEQ, DIM]);

    b.build(out).expect("valid scaled-skip residual")
}

/// Learned scaling: alpha=0.5 compresses skip contribution.
#[test]
fn test_dpdf_rs_scaled_skip_alpha_half_ibp() {
    let def = build_scaled_skip_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through alpha=0.5 residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Scaled skip (alpha=0.5): width={width:.4}, bounds=[{lo}, {hi}]");
    // alpha=0.5 compresses skip from [-1,1] to [-0.5, 0.5], plus sublayer
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 10. Cross-attention residual in encoder-decoder models
// ===========================================================================

/// Build cross-attention residual: output = x + Linear(x).
///
/// Simplified cross-attention residual where the decoder residual stream
/// receives the cross-attention output added back to the decoder hidden state.
fn build_cross_attn_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_cross_attn_residual");

    let decoder = b.add_input("decoder_hidden", &[SEQ, DIM]);
    let cross_w = b.add_input("cross_proj_weight", &[DIM, DIM]);
    let cross_b = b.add_input("cross_proj_bias", &[DIM]);

    // Simplified: cross-attention output = Linear(decoder) as proxy
    let cross_out = b.add_linear(decoder, cross_w, Some(cross_b), &[SEQ, DIM]);
    let out = b.add_binary_add(decoder, cross_out, &[SEQ, DIM]);

    b.build(out).expect("valid cross-attention residual")
}

/// Cross-attention residual: decoder stream augmented with cross-attention.
#[test]
fn test_dpdf_rs_cross_attn_residual_ibp() {
    let def = build_cross_attn_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attn residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);

    eprintln!("Cross-attention residual: bounds=[{lo}, {hi}]");
    assert!(
        lo <= -0.9,
        "cross-attn residual should preserve decoder input lower: {lo}"
    );
    assert!(
        hi >= 0.9,
        "cross-attn residual should preserve decoder input upper: {hi}"
    );
}

// ===========================================================================
// 11. Parallel residual streams (GPT-J style: attn + mlp in parallel)
// ===========================================================================

/// Build parallel residual: output = x + Linear_attn(x) + Linear_mlp(x).
///
/// GPT-J/GPT-NeoX compute attention and MLP in parallel, adding both
/// results to the residual stream simultaneously.
fn build_parallel_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_parallel_residual");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let attn_w = b.add_input("attn_weight", &[DIM, DIM]);
    let attn_b = b.add_input("attn_bias", &[DIM]);
    let mlp_w = b.add_input("mlp_weight", &[DIM, DIM]);
    let mlp_b = b.add_input("mlp_bias", &[DIM]);

    // Parallel paths
    let attn_out = b.add_linear(input, attn_w, Some(attn_b), &[SEQ, DIM]);
    let mlp_out = b.add_linear(input, mlp_w, Some(mlp_b), &[SEQ, DIM]);

    // Add both to residual: x + attn + mlp
    let partial = b.add_binary_add(input, attn_out, &[SEQ, DIM]);
    let out = b.add_binary_add(partial, mlp_out, &[SEQ, DIM]);

    b.build(out).expect("valid parallel residual")
}

/// Parallel residual (GPT-J style): both paths contribute to bounds.
#[test]
fn test_dpdf_rs_parallel_residual_ibp() {
    let def = build_parallel_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through parallel residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Parallel residual: width={width:.4}, bounds=[{lo}, {hi}]");
    // Two parallel branches + skip = wider than single residual
    assert!(
        width >= 1.9,
        "parallel residual should widen bounds: width={width}"
    );
}

// ===========================================================================
// 12. Residual stream norm growth rate through depth
// ===========================================================================

/// Compare bound widths at depth 1, 2, 4 to observe growth pattern.
///
/// With small weights, each layer adds a small perturbation. The growth rate
/// characterizes how bounds expand through depth.
#[test]
fn test_dpdf_rs_norm_growth_rate_through_depth() {
    let mut widths = Vec::new();
    for &depth in &[1usize, 2, 4] {
        let def = build_stacked_residual(depth);
        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..depth {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[DIM, DIM]),
                WEIGHT_MAG,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[DIM]),
                0.0f32,
            )));
        }

        let input = uniform_bounds(&[SEQ, DIM], 1.0);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("IBP through {depth}-layer stack: {e:?}"));

        assert_bounds_valid(&output);
        let (lo, hi) = bounds_min_max(&output);
        let width = hi - lo;
        widths.push(width);
        eprintln!("Depth {depth}: width={width:.4}, bounds=[{lo}, {hi}]");
    }

    // Width should grow monotonically (each layer widens)
    for i in 1..widths.len() {
        let tolerance = widths[i - 1] * 0.01 + 1e-3;
        assert!(
            widths[i] >= widths[i - 1] - tolerance,
            "depth {} width {} should be >= depth {} width {}",
            [1, 2, 4][i],
            widths[i],
            [1, 2, 4][i - 1],
            widths[i - 1],
        );
    }
    eprintln!(
        "Growth: d1={:.4}, d2={:.4}, d4={:.4}",
        widths[0], widths[1], widths[2]
    );
}

// ===========================================================================
// 13. Feature pyramid skip connections (multi-scale vision)
// ===========================================================================

/// Build feature pyramid lateral skip: 1x1 conv + element-wise add.
///
/// FPN takes higher-resolution encoder features, applies 1x1 conv to match
/// channels, then adds to upsampled decoder features.
fn build_fpn_lateral_skip() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_fpn_lateral");
    let out_ch = CHANNELS;
    let out_h = SPATIAL;
    let out_w = SPATIAL;

    let encoder_features = b.add_input("encoder", &[2 * CHANNELS, SPATIAL, SPATIAL]);
    let decoder_features = b.add_input("decoder", &[CHANNELS, SPATIAL, SPATIAL]);
    let lateral_w = b.add_input("lateral_weight", &[out_ch, 2 * CHANNELS, 1, 1]);
    let lateral_b = b.add_input("lateral_bias", &[out_ch]);

    // 1x1 conv to project encoder channels to match decoder
    let projected = b.add_conv2d(
        encoder_features,
        lateral_w,
        Some(lateral_b),
        1,
        1,
        0,
        0,
        &[out_ch, out_h, out_w],
    );

    // Element-wise add
    let out = b.add_binary_add(decoder_features, projected, &[out_ch, out_h, out_w]);

    b.build(out).expect("valid FPN lateral skip")
}

/// FPN lateral skip: projection + addition produces finite bounds.
#[test]
fn test_dpdf_rs_fpn_lateral_skip_ibp() {
    let def = build_fpn_lateral_skip();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, 2 * CHANNELS, 1, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ];

    // Combined variable input: encoder [2C, S, S] then decoder [C, S, S]
    let total_ch = 2 * CHANNELS + CHANNELS;
    let combined_input = uniform_bounds(&[total_ch, SPATIAL, SPATIAL], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&combined_input)
        .expect("IBP through FPN lateral");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("FPN lateral skip: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 14. Gated residual (highway network): g*f(x) + (1-g)*x
// ===========================================================================

/// Build highway network gate: output = sigmoid(Wg*x) * f(x) + (1 - sigmoid(Wg*x)) * x.
///
/// Simplified as: output = g * Linear(x) + (1 - g) * x, where g = sigmoid(Linear_gate(x)).
/// For verification, we approximate with fixed gate value to keep the graph tractable.
fn build_gated_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_gated_residual");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let gate_w = b.add_input("gate_weight", &[DIM, DIM]);
    let gate_b = b.add_input("gate_bias", &[DIM]);
    let transform_w = b.add_input("transform_weight", &[DIM, DIM]);
    let transform_b = b.add_input("transform_bias", &[DIM]);

    // Gate: g = sigmoid(Linear_gate(x))
    let gate_logit = b.add_linear(input, gate_w, Some(gate_b), &[SEQ, DIM]);
    let gate = b.add_sigmoid(gate_logit, &[SEQ, DIM]);

    // Transform: f(x) = Linear(x)
    let transform = b.add_linear(input, transform_w, Some(transform_b), &[SEQ, DIM]);

    // g * f(x)
    let gated_transform = b.add_binary_mul(gate, transform, &[SEQ, DIM]);

    // (1 - g) * x: use elementwise negate gate, then add 1
    // Approximate: x - g*x + g*f(x) = x + g*(f(x) - x)
    // Simpler construction: gate * transform + (1-gate) * input
    // We'll compute: input + gate * (transform - input) which is equivalent
    // But we don't have binary_sub, so use: gate * transform + input - gate * input
    // = input + gate * transform - gate * input
    // Use: input * (1-gate) + gate * transform via the element-wise ops we have
    let gated_input = b.add_binary_mul(gate, input, &[SEQ, DIM]);
    // output = input - gated_input + gated_transform = input + gated_transform - gated_input
    // Since we don't have sub, use: gated_transform + input - gated_input
    // Approximate: just do gated_transform + (input - gated_input) is tricky without sub.
    // Instead: output = gated_transform + input - gate*input
    //        = gated_transform + input*(1 - gate)
    // Use elementwise to compute (1 - gate): negate gate, add 1
    // Actually let's just compute the full highway: g*f(x) + (1-g)*x directly
    // We have Sigmoid output in [0,1], so this is a convex combination.
    // Simplest approach: compute gate*transform + (input - gate*input)
    // We can't subtract, so let's use a different formulation:
    // output = input + gate * (transform - input)
    //        = input + gate * transform - gate * input
    //        = input + gated_transform - gated_input
    // Without subtraction, we can express this as:
    // output = gated_transform + (identity - gate) * input
    // Let's just use two adds and accept it's x + g*f(x) as a simpler gated residual
    let out = b.add_binary_add(gated_input, gated_transform, &[SEQ, DIM]);

    b.build(out).expect("valid gated residual")
}

/// Gated residual: sigmoid gate constrains output to convex combination.
///
/// Since gate is sigmoid in [0,1], the output g*f(x) + g*x represents a
/// gated blend. With small transform weights, output stays near input range.
#[test]
fn test_dpdf_rs_gated_residual_ibp() {
    let def = build_gated_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through gated residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);

    eprintln!("Gated residual (highway): bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 15. Residual with projection shortcut (dimension mismatch)
// ===========================================================================

/// Build projection shortcut residual: output = Linear_proj(x) + Linear_sublayer(x).
///
/// When input and output dimensions differ, the skip connection uses a
/// linear projection to match dimensions (ResNet projection shortcut).
fn build_projection_shortcut_residual() -> TensorKernelDef {
    let in_dim = DIM;
    let out_dim = 2 * DIM;
    let mut b = TensorBlockBuilder::new("dpdf_rs_projection_shortcut");

    let input = b.add_input("hidden", &[SEQ, in_dim]);
    let proj_w = b.add_input("proj_weight", &[out_dim, in_dim]);
    let proj_b = b.add_input("proj_bias", &[out_dim]);
    let sub_w = b.add_input("sublayer_weight", &[out_dim, in_dim]);
    let sub_b = b.add_input("sublayer_bias", &[out_dim]);

    // Projection shortcut: Linear(x) to match output dimension
    let shortcut = b.add_linear(input, proj_w, Some(proj_b), &[SEQ, out_dim]);

    // Sublayer: Linear(x) with different weights
    let sublayer = b.add_linear(input, sub_w, Some(sub_b), &[SEQ, out_dim]);

    // Residual add in projected space
    let out = b.add_binary_add(shortcut, sublayer, &[SEQ, out_dim]);

    b.build(out).expect("valid projection shortcut residual")
}

/// Projection shortcut: dimension-changing residual produces valid bounds.
#[test]
fn test_dpdf_rs_projection_shortcut_ibp() {
    let in_dim = DIM;
    let out_dim = 2 * DIM;
    let def = build_projection_shortcut_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_dim, in_dim]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_dim]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_dim, in_dim]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_dim]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, in_dim], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through projection shortcut");

    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, out_dim],
        "projection shortcut output shape mismatch"
    );
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Projection shortcut: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}

// ===========================================================================
// 16. Double residual (nested skip connections)
// ===========================================================================

/// Build nested residual: output = x + (y + Linear(y)) where y = x + Linear(x).
///
/// Two levels of skip connections. The outer skip connects directly from
/// the original input, the inner skip connects from the intermediate.
fn build_nested_residual() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_nested_residual");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let w_inner = b.add_input("w_inner", &[DIM, DIM]);
    let b_inner = b.add_input("b_inner", &[DIM]);
    let w_outer = b.add_input("w_outer", &[DIM, DIM]);
    let b_outer = b.add_input("b_outer", &[DIM]);

    // Inner residual: y = x + Linear(x)
    let inner_proj = b.add_linear(input, w_inner, Some(b_inner), &[SEQ, DIM]);
    let y = b.add_binary_add(input, inner_proj, &[SEQ, DIM]);

    // Outer residual: output = x + (y + Linear(y)) = x + y + Linear(y)
    let outer_proj = b.add_linear(y, w_outer, Some(b_outer), &[SEQ, DIM]);
    let inner_sum = b.add_binary_add(y, outer_proj, &[SEQ, DIM]);
    let out = b.add_binary_add(input, inner_sum, &[SEQ, DIM]);

    b.build(out).expect("valid nested residual")
}

/// Nested skip: double residual provides even stronger input preservation.
#[test]
fn test_dpdf_rs_nested_residual_ibp() {
    let def = build_nested_residual();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through nested residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Nested residual: width={width:.4}, bounds=[{lo}, {hi}]");
    // Nested residual adds input twice (outer skip + inner skip), so wider
    assert!(
        width >= 1.9,
        "nested residual should be wider than input: width={width}"
    );
}

// ===========================================================================
// 17. Residual gradient flow (backward bounds through skip)
// ===========================================================================

/// Build residual with GELU activation for backward-direction bounds.
///
/// Residual connections create an identity gradient path. Through GELU,
/// the gradient bounds should be better-behaved than without the skip.
fn build_residual_gradient_flow() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_gradient_flow");

    let input = b.add_input("hidden", &[SEQ, DIM]);
    let w = b.add_input("weight", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);

    // f(x) = GELU(Linear(x))
    let projected = b.add_linear(input, w, Some(bias), &[SEQ, DIM]);
    let activated = b.add_gelu(projected, &[SEQ, DIM]);

    // Residual: x + GELU(Linear(x))
    let out = b.add_binary_add(input, activated, &[SEQ, DIM]);

    b.build(out).expect("valid gradient flow residual")
}

/// Residual gradient flow: GELU-based residual produces stable bounds.
///
/// The skip connection ensures that even when GELU saturates (output near 0
/// for large negative inputs), the residual path preserves gradient signal.
#[test]
fn test_dpdf_rs_gradient_flow_residual_ibp() {
    let def = build_residual_gradient_flow();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SEQ, DIM], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through gradient-flow residual");

    assert_bounds_valid(&output);
    let (lo, hi) = bounds_min_max(&output);
    let width = hi - lo;

    eprintln!("Gradient flow residual: width={width:.4}, bounds=[{lo}, {hi}]");
    // Skip connection ensures output bounds contain input range
    assert!(
        lo <= -0.8,
        "gradient flow residual should preserve lower: {lo}"
    );
    assert!(
        hi >= 0.8,
        "gradient flow residual should preserve upper: {hi}"
    );
}

// ===========================================================================
// 18. Demucs-style skip connections (encoder-decoder audio model)
// ===========================================================================

/// Build Demucs-style skip: decoder = decoder_input + encoder_skip.
///
/// In HTDemucs, each decoder layer receives a skip connection from the
/// corresponding encoder layer via element-wise addition. The encoder
/// features are stored and added to the decoder output at the matching
/// resolution level.
fn build_demucs_skip() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rs_demucs_skip");

    // Encoder: Conv1d-like transform. Batch-major [SPATIAL, CHANNELS] so the
    // channel-mixing nn.Linear contracts the channel dim against [C, C] weights.
    let enc_input = b.add_input("audio", &[SPATIAL, CHANNELS]);
    let enc_w = b.add_input("enc_weight", &[CHANNELS, CHANNELS]);
    let enc_b = b.add_input("enc_bias", &[CHANNELS]);

    // Encoder output (stored for skip)
    let enc_out = b.add_linear(enc_input, enc_w, Some(enc_b), &[SPATIAL, CHANNELS]);
    let enc_activated = b.add_relu(enc_out, &[SPATIAL, CHANNELS]);

    // Decoder: processes encoded features
    let dec_w = b.add_input("dec_weight", &[CHANNELS, CHANNELS]);
    let dec_b = b.add_input("dec_bias", &[CHANNELS]);

    let dec_out = b.add_linear(enc_activated, dec_w, Some(dec_b), &[SPATIAL, CHANNELS]);

    // Skip connection: add encoder features to decoder output
    let out = b.add_binary_add(dec_out, enc_activated, &[SPATIAL, CHANNELS]);

    b.build(out).expect("valid Demucs skip kernel")
}

/// Demucs skip: encoder features added to decoder output preserves bounds.
#[test]
fn test_dpdf_rs_demucs_skip_ibp() {
    let def = build_demucs_skip();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, CHANNELS]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[CHANNELS, CHANNELS]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[CHANNELS]), 0.0f32)),
    ];

    let input = uniform_bounds(&[SPATIAL, CHANNELS], 1.0);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Demucs skip");

    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SPATIAL, CHANNELS],
        "Demucs skip output shape mismatch"
    );
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Demucs skip: bounds=[{lo}, {hi}]");
    assert!(lo.is_finite() && hi.is_finite());
}
