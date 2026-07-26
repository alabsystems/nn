// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4063: Compose tests for KV-cache update and autoregressive generation bounds.
//!
//! Verifies IBP and CROWN bound propagation through KV-cache update patterns
//! during autoregressive generation in dpdf text decoders (Qwen3-VL, GLM-OCR,
//! Granite-Docling). These patterns are fundamental to efficient LLM inference:
//! cache append, prefill vs decode, multi-step generation, RoPE position
//! updates, cross-attention with encoder cache, generation length effects,
//! and full autoregressive steps.
//!
//! ## Cache Append & Cross-Attention (tests 1-4)
//!
//! 1. Cache append single token IBP bounds
//! 2. Cache-attended cross-attention IBP
//! 3. Prefill phase (long sequence) IBP bounds
//! 4. Decode phase (single token + cached K/V) IBP bounds
//!
//! ## Phase Comparison & Multi-Step (tests 5-8)
//!
//! 5. Prefill vs decode bound width comparison
//! 6. Multi-step generation cache growth IBP
//! 7. Cache + RoPE position update IBP
//! 8. Cache state propagation across 3 steps
//!
//! ## CROWN & Generation Length (tests 9-12)
//!
//! 9. Cross-attention with encoder cache CROWN
//! 10. Generation length effect on bound width
//! 11. Monotone tightening for cache-attended attention
//! 12. Full autoregressive step: embed -> cache_attn -> FFN -> project
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=16, NUM_HEADS=4, HEAD_DIM=4, FFN_DIM=32, CACHE_LEN=8

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
const DIM: usize = 16;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = DIM / NUM_HEADS; // 4
const FFN_DIM: usize = 32;
/// Cache length for KV-cache tests (previously generated tokens).
const CACHE_LEN: usize = 8;
/// Encoder sequence length for cross-attention tests.
const ENC_SEQ_LEN: usize = 6;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Vocabulary size for output projection.
const VOCAB_SIZE: usize = 32;

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

/// Build a cache-attended attention block: Q from current tokens, K/V from
/// cached context. This is the core decode-phase pattern.
fn build_cache_attended_attention(
    b: &mut TensorBlockBuilder,
    q_input: nn_dsl::TensorNodeId,
    kv_input: nn_dsl::TensorNodeId,
    q_len: usize,
    _kv_len: usize,
    prefix: &str,
) -> nn_dsl::TensorNodeId {
    let q_w = b.add_input(&format!("{prefix}q_weight"), &[DIM, DIM]);
    let k_w = b.add_input(&format!("{prefix}k_weight"), &[DIM, DIM]);
    let v_w = b.add_input(&format!("{prefix}v_weight"), &[DIM, DIM]);
    let out_w = b.add_input(&format!("{prefix}out_weight"), &[DIM, DIM]);

    b.add_multi_head_cross_attention(
        q_input,
        kv_input,
        q_w,
        k_w,
        v_w,
        out_w,
        NUM_HEADS,
        AttentionMask::Standard,
        &[q_len, DIM],
    )
    .expect("valid cache-attended attention")
}

/// Standard projection weight bindings for cache-attended attention.
fn cache_attn_weight_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

// ===========================================================================
// 1. Cache append single token IBP bounds
// ===========================================================================

/// Cache append: a single new token's KV is appended to existing cache,
/// then the new token attends over the full (cached + new) KV context.
fn build_cache_append_single_token_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens;

    let mut b = TensorBlockBuilder::new("dpdf_kv_cache_append_single");

    let q_input = b.add_input("query", &[current_tokens, DIM]);
    let kv_input = b.add_input("kv_context", &[total_kv, DIM]);

    let out =
        build_cache_attended_attention(&mut b, q_input, kv_input, current_tokens, total_kv, "");

    b.build(out)
        .expect("valid cache append single token kernel")
}

fn cache_append_single_token_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable, // query
        TensorParamBinding::Variable, // kv_context
    ];
    bindings.extend(cache_attn_weight_bindings());
    bindings
}

#[test]
fn test_cache_append_single_token_ibp() {
    let def = build_cache_append_single_token_kernel();
    let bindings = cache_append_single_token_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens;
    let total_seq = current_tokens + total_kv;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cache append");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cache append single token IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "cache append lower must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "cache append upper must be finite, got {hi_max}"
    );
}

// ===========================================================================
// 2. Cache-attended cross-attention IBP
// ===========================================================================

/// Cross-attention where decoder queries attend to a frozen encoder cache.
/// The encoder features are pre-computed and remain constant across all
/// decoder steps, modeled here as a fixed KV-cache from the encoder.
fn build_cache_attended_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_kv_cache_cross_attn");

    // Decoder queries from current step: [SEQ_LEN, DIM]
    let q_input = b.add_input("decoder_query", &[SEQ_LEN, DIM]);
    // Encoder cache (frozen): [ENC_SEQ_LEN, DIM]
    let enc_cache = b.add_input("encoder_cache", &[ENC_SEQ_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            enc_cache,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid cache cross-attention");

    b.build(out)
        .expect("valid cache-attended cross-attention kernel")
}

fn cache_attended_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // decoder_query
        TensorParamBinding::Variable,                       // encoder_cache
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_cache_attended_cross_attention_ibp() {
    let def = build_cache_attended_cross_attention_kernel();
    let bindings = cache_attended_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cache cross-attention");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cache-attended cross-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "cache cross-attn lower must be finite");
    assert!(hi_max.is_finite(), "cache cross-attn upper must be finite");
}

// ===========================================================================
// 3. Prefill phase (long sequence) IBP bounds
// ===========================================================================

/// Prefill: full prompt is processed in a single forward pass.
/// All tokens attend to all preceding tokens via causal attention.
fn build_prefill_phase_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_kv_cache_prefill");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[SEQ_LEN, DIM],
        )
        .expect("valid prefill MHA");

    // Residual connection
    let result = b.add_binary_add(input, out, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid prefill phase kernel")
}

fn prefill_phase_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_prefill_phase_ibp() {
    let def = build_prefill_phase_kernel();
    let bindings = prefill_phase_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through prefill");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Prefill phase IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "prefill lower must be finite");
    assert!(hi_max.is_finite(), "prefill upper must be finite");
}

// ===========================================================================
// 4. Decode phase (single token + cached K/V) IBP bounds
// ===========================================================================

/// Decode phase: a single new token attends over cached K/V from previous
/// steps. This is the steady-state pattern during autoregressive generation.
fn build_decode_phase_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;

    let mut b = TensorBlockBuilder::new("dpdf_kv_cache_decode");

    let q_input = b.add_input("query", &[current_tokens, DIM]);
    let kv_cache = b.add_input("kv_cache", &[CACHE_LEN, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_cross_attention(
            q_input,
            kv_cache,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[current_tokens, DIM],
        )
        .expect("valid decode MHA");

    b.build(out).expect("valid decode phase kernel")
}

fn decode_phase_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // query
        TensorParamBinding::Variable,                       // kv_cache
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_decode_phase_ibp() {
    let def = build_decode_phase_kernel();
    let bindings = decode_phase_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_seq = current_tokens + CACHE_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decode phase");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decode phase IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "decode lower must be finite");
    assert!(hi_max.is_finite(), "decode upper must be finite");
}

// ===========================================================================
// 5. Prefill vs decode bound width comparison
// ===========================================================================

/// Compare bound widths between prefill (full causal self-attention) and
/// decode (single token attending cached KV). Both should produce finite
/// bounds; their relative widths characterize the verification overhead
/// of each phase.
fn build_phase_comparison_kernel(q_len: usize, kv_len: usize) -> TensorKernelDef {
    let name = format!("dpdf_kv_phase_q{q_len}_kv{kv_len}");
    let mut b = TensorBlockBuilder::new(&name);

    let q_input = b.add_input("query", &[q_len, DIM]);
    let kv_input = b.add_input("kv", &[kv_len, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let q = b.add_linear(q_input, q_w, None, &[q_len, DIM]);
    let k = b.add_linear(kv_input, k_w, None, &[kv_len, DIM]);
    let v = b.add_linear(kv_input, v_w, None, &[kv_len, DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &[q_len, DIM]);

    let out = b.add_linear(attn, out_w, None, &[q_len, DIM]);

    b.build(out).expect("valid phase comparison kernel")
}

fn phase_comparison_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_prefill_vs_decode_bound_width() {
    let bindings = phase_comparison_bindings();

    // Prefill: full sequence Q over full sequence KV
    let prefill_def = build_phase_comparison_kernel(SEQ_LEN, SEQ_LEN);
    let prefill_graph = tensor_kernel_to_graph(&prefill_def, &bindings).expect("prefill graph");
    let prefill_total = SEQ_LEN + SEQ_LEN;
    let prefill_input = uniform_bounds(&[prefill_total, DIM], 1.0);
    let prefill_output = prefill_graph
        .propagate_ibp(&prefill_input)
        .expect("prefill IBP");
    assert_bounds_valid(&prefill_output);

    // Decode: single token Q over cached KV
    let decode_def = build_phase_comparison_kernel(1, CACHE_LEN);
    let decode_graph = tensor_kernel_to_graph(&decode_def, &bindings).expect("decode graph");
    let decode_total = 1 + CACHE_LEN;
    let decode_input = uniform_bounds(&[decode_total, DIM], 1.0);
    let decode_output = decode_graph
        .propagate_ibp(&decode_input)
        .expect("decode IBP");
    assert_bounds_valid(&decode_output);

    let prefill_width = bound_width(&prefill_output);
    let decode_width = bound_width(&decode_output);
    eprintln!(
        "Prefill vs decode: prefill width={prefill_width:.6}, decode width={decode_width:.6}"
    );

    assert!(prefill_width.is_finite(), "prefill width must be finite");
    assert!(decode_width.is_finite(), "decode width must be finite");
}

// ===========================================================================
// 6. Multi-step generation cache growth IBP
// ===========================================================================

/// Multi-step generation: simulate 3 decode steps where the KV-cache grows
/// by one entry per step. At each step, the new token attends over the full
/// cache (original + previously generated tokens).
#[test]
fn test_multi_step_generation_cache_growth_ibp() {
    let bindings = phase_comparison_bindings();

    let mut widths = Vec::new();
    for step in 0..3usize {
        let cache_size = CACHE_LEN + step;
        let def = build_phase_comparison_kernel(1, cache_size);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let total_seq = 1 + cache_size;
        let input = uniform_bounds(&[total_seq, DIM], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Multi-step generation step {step}: cache_size={cache_size}, width={width:.6}");
        assert!(width.is_finite(), "step {step} width must be finite");
        widths.push(width);
    }

    // All widths should be finite
    for (i, w) in widths.iter().enumerate() {
        assert!(w.is_finite(), "step {i} width must be finite, got {w}");
    }
}

// ===========================================================================
// 7. Cache + RoPE position update IBP
// ===========================================================================

/// Cache-attended attention with RoPE positional encoding. During generation,
/// each new token gets a position offset equal to the cache length. Simulated
/// by adding positional encoding before Q/K projections.
fn build_cache_rope_update_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens;

    let mut b = TensorBlockBuilder::new("dpdf_kv_cache_rope_update");

    let q_input = b.add_input("query", &[current_tokens, DIM]);
    let kv_input = b.add_input("kv_context", &[total_kv, DIM]);

    // RoPE for current token (position = CACHE_LEN)
    let q_pe = b.add_input("q_pos_enc", &[current_tokens, DIM]);
    // RoPE for all KV positions (0..total_kv)
    let kv_pe = b.add_input("kv_pos_enc", &[total_kv, DIM]);

    let q_with_pe = b.add_binary_add(q_input, q_pe, &[current_tokens, DIM]);
    let kv_with_pe = b.add_binary_add(kv_input, kv_pe, &[total_kv, DIM]);

    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let q = b.add_linear(q_with_pe, q_w, None, &[current_tokens, DIM]);
    let k = b.add_linear(kv_with_pe, k_w, None, &[total_kv, DIM]);
    let v = b.add_linear(kv_with_pe, v_w, None, &[total_kv, DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[current_tokens, DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[current_tokens, DIM]);

    b.build(out).expect("valid cache + RoPE update kernel")
}

fn cache_rope_update_bindings() -> Vec<TensorParamBinding> {
    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens;

    // Sinusoidal PE for the current token position
    let q_pe = super::common::sinusoidal_pe(current_tokens, DIM);
    // Sinusoidal PE for all KV positions
    let kv_pe = super::common::sinusoidal_pe(total_kv, DIM);

    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // query
        TensorParamBinding::Variable,                       // kv_context
        TensorParamBinding::ConstantTensor(q_pe),           // q_pos_enc
        TensorParamBinding::ConstantTensor(kv_pe),          // kv_pos_enc
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
    ]
}

#[test]
fn test_cache_rope_position_update_ibp() {
    let def = build_cache_rope_update_kernel();
    let bindings = cache_rope_update_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_kv = CACHE_LEN + current_tokens;
    let total_seq = current_tokens + total_kv;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cache + RoPE");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cache + RoPE update IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "cache+RoPE lower must be finite");
    assert!(hi_max.is_finite(), "cache+RoPE upper must be finite");
}

// ===========================================================================
// 8. Cache state propagation across 3 steps
// ===========================================================================

/// Verify that KV-cache attention produces valid bounds at each of 3
/// consecutive generation steps with increasing cache lengths.
/// Each step builds an independent graph (as each step has different shapes).
#[test]
fn test_cache_state_propagation_3_steps() {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);

    let mut prev_width: Option<f32> = None;

    for step in 0..3usize {
        let cache_at_step = CACHE_LEN + step;
        let current_tokens: usize = 1;

        let name = format!("dpdf_kv_state_step{step}");
        let mut b = TensorBlockBuilder::new(&name);

        let q_input = b.add_input("query", &[current_tokens, DIM]);
        let kv_input = b.add_input("kv_cache", &[cache_at_step, DIM]);

        let q_w = b.add_input("q_weight", &[DIM, DIM]);
        let k_w = b.add_input("k_weight", &[DIM, DIM]);
        let v_w = b.add_input("v_weight", &[DIM, DIM]);
        let out_w = b.add_input("out_weight", &[DIM, DIM]);

        let out = b
            .add_multi_head_cross_attention(
                q_input,
                kv_input,
                q_w,
                k_w,
                v_w,
                out_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &[current_tokens, DIM],
            )
            .expect("valid cache step attention");

        let def = b.build(out).expect("valid cache state propagation kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(proj_w.clone()),
            TensorParamBinding::ConstantTensor(proj_w.clone()),
            TensorParamBinding::ConstantTensor(proj_w.clone()),
            TensorParamBinding::ConstantTensor(proj_w.clone()),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let total_seq = current_tokens + cache_at_step;
        let input = uniform_bounds(&[total_seq, DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Cache propagation step {step}: cache={cache_at_step}, width={width:.6}");
        assert!(width.is_finite(), "step {step} width must be finite");

        prev_width = Some(width);
    }

    assert!(
        prev_width.is_some(),
        "should have computed at least one width"
    );
}

// ===========================================================================
// 9. Cross-attention with encoder cache CROWN
// ===========================================================================

/// Cross-attention with frozen encoder cache verified via CROWN.
/// CROWN should produce tighter bounds than IBP when it succeeds.
#[test]
fn test_cross_attention_encoder_cache_crown() {
    let def = build_cache_attended_cross_attention_kernel();
    let bindings = cache_attended_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let total_seq = SEQ_LEN + ENC_SEQ_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Encoder cache CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 10. Generation length effect on bound width
// ===========================================================================

/// Test how increasing generation length (cache size) affects output bound
/// width. Longer cache means more KV entries for the query to attend over.
#[test]
fn test_generation_length_effect_on_bound_width() {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let cache_lengths = [2usize, 4, 8, 12];
    let mut widths = Vec::new();

    for &cache_len in &cache_lengths {
        let def = build_phase_comparison_kernel(1, cache_len);

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(proj_w.clone()),
            TensorParamBinding::ConstantTensor(proj_w.clone()),
            TensorParamBinding::ConstantTensor(proj_w.clone()),
            TensorParamBinding::ConstantTensor(proj_w.clone()),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let total_seq = 1 + cache_len;
        let input = uniform_bounds(&[total_seq, DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Generation length effect: cache_len={cache_len}, width={width:.6}");
        assert!(
            width.is_finite(),
            "cache_len={cache_len} width must be finite"
        );
        widths.push(width);
    }

    // All widths should be finite
    for (i, w) in widths.iter().enumerate() {
        assert!(
            w.is_finite(),
            "cache_len={} width must be finite, got {w}",
            cache_lengths[i]
        );
    }
}

// ===========================================================================
// 11. Monotone tightening for cache-attended attention
// ===========================================================================

/// Verify that tighter input bounds produce tighter output bounds for
/// cache-attended attention. This is a fundamental soundness property
/// of bound propagation.
#[test]
fn test_cache_attended_monotone_tightening() {
    let def = build_decode_phase_kernel();
    let bindings = decode_phase_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_seq = current_tokens + CACHE_LEN;

    let eps_values = [1.0, 0.5, 0.1];
    let mut prev_width: Option<f32> = None;

    for &eps in &eps_values {
        let input = uniform_bounds(&[total_seq, DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Cache-attended monotone tightening: eps={eps:.2}, width={width:.6}");

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
// 12. Full autoregressive step: embed -> cache_attn -> FFN -> project
// ===========================================================================

/// Full autoregressive decode step: token embedding lookup (linear from
/// one-hot), cache-attended attention with residual, LayerNorm, SwiGLU FFN
/// with residual, and vocabulary projection.
fn build_full_autoregressive_step_kernel() -> TensorKernelDef {
    let current_tokens: usize = 1;

    let mut b = TensorBlockBuilder::new("dpdf_kv_full_autoregressive_step");

    // Token embedding input (already embedded): [1, DIM]
    let embed_input = b.add_input("embed", &[current_tokens, DIM]);
    // KV cache: [CACHE_LEN, DIM]
    let kv_cache = b.add_input("kv_cache", &[CACHE_LEN, DIM]);

    // --- Cache-attended attention with residual ---
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let attn_out = b
        .add_multi_head_cross_attention(
            embed_input,
            kv_cache,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[current_tokens, DIM],
        )
        .expect("valid cache attention");

    // Residual after attention
    let post_attn = b.add_binary_add(embed_input, attn_out, &[current_tokens, DIM]);

    // --- LayerNorm ---
    let ln_w = b.add_input("ln_weight", &[DIM]);
    let ln_b = b.add_input("ln_bias", &[DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(post_attn, eps, 1, ln_w, ln_b, &[current_tokens, DIM]);

    // --- SwiGLU FFN ---
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_weight", &[DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &[current_tokens, FFN_DIM]);
    let gate_act = add_silu(&mut b, gate, &[current_tokens, FFN_DIM]);
    let up = b.add_linear(normed, up_w, None, &[current_tokens, FFN_DIM]);
    let gated = b.add_binary_mul(gate_act, up, &[current_tokens, FFN_DIM]);
    let ffn_out = b.add_linear(gated, down_w, None, &[current_tokens, DIM]);

    // Residual after FFN
    let post_ffn = b.add_binary_add(normed, ffn_out, &[current_tokens, DIM]);

    // --- Vocabulary projection ---
    let vocab_w = b.add_input("vocab_weight", &[VOCAB_SIZE, DIM]);
    let logits = b.add_linear(post_ffn, vocab_w, None, &[current_tokens, VOCAB_SIZE]);

    b.build(logits)
        .expect("valid full autoregressive step kernel")
}

fn full_autoregressive_step_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[DIM, FFN_DIM]), WEIGHT_MAG);
    let vocab_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // embed
        TensorParamBinding::Variable,                       // kv_cache
        TensorParamBinding::ConstantTensor(proj_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(proj_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(proj_w),         // out_weight
        TensorParamBinding::ConstantTensor(ln_w),           // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),           // eps
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(vocab_w),        // vocab_weight
    ]
}

#[test]
fn test_full_autoregressive_step_ibp() {
    let def = build_full_autoregressive_step_kernel();
    let bindings = full_autoregressive_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_seq = current_tokens + CACHE_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full autoregressive step");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full autoregressive step IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "full step lower must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "full step upper must be finite, got {hi_max}"
    );
}

#[test]
fn test_full_autoregressive_step_crown() {
    let def = build_full_autoregressive_step_kernel();
    let bindings = full_autoregressive_step_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let current_tokens: usize = 1;
    let total_seq = current_tokens + CACHE_LEN;
    let input = uniform_bounds(&[total_seq, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Full autoregressive step CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}
