// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! QK-Norm attention verification infrastructure for Qwen3.
//!
//! Qwen3 applies per-head RMSNorm to Q and K projections before attention scoring.
//! This constrains attention logit magnitudes and prevents outlier scores from
//! dominating softmax output. The real Qwen3 attention path is:
//!
//!   x -> Q_proj -> reshape [B, seq, n_heads, head_dim] -> transpose [B, n_heads, seq, head_dim]
//!     -> per-head RMSNorm (weight shape [head_dim], norm axis=last) -> RoPE -> attention
//!
//! This file models that path structurally for NY verification:
//!
//! 1. **Per-head QK-Norm (flat 2D)**: Models Q projection -> reshape to per-head
//!    segments -> RMSNorm over HEAD_DIM -> flatten back. Uses IbpValidated mode.
//!
//! 2. **QK-Norm + attention composition**: Full path from hidden state through
//!    QK-Norm Q/K, V projection, attention scoring, and output projection.
//!
//! 3. **Bounds tightening comparison**: Compares bounds with and without QK-Norm
//!    to quantify how normalization constrains the attention logit range.
//!
//! ## Known gap: NY#3172
//!
//! NY does not yet propagate tight CROWN bounds through RMSNorm applied
//! per-head (i.e., RMSNorm on reshaped sub-tensors). IBP propagation works but
//! produces conservative (wider) bounds. Once NY#3172 lands, CROWN
//! through per-head QK-Norm will tighten bounds significantly.
//!
//! Until then, the tests here verify:
//! - IBP propagation produces finite, valid bounds
//! - The QK-Norm subgraph structure is correct for verification
//! - CROWN fallback behavior is documented and exercised
//!
//! Uses IbpValidated soundness mode per nn engineering rules (Source: #3356).
//! Dimensions: D_MODEL=16, N_HEADS=2, N_KV_HEADS=1, HEAD_DIM=8, SEQ=4.
//!
//! Part of #2951: QK-Norm attention verification for Qwen3.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback,
    bounds_min_max, uniform_bounds, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding,
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
const HALF_DIM: usize = HEAD_DIM / 2; // 4
const SEQ: usize = 4;
const WEIGHT_MAG: f32 = 0.001;

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

fn ibp_validated_config() -> VerifyConfig {
    // IbpValidated, not Sound, per engineering rules: Sound refuses
    // linearization for normalization layers. (Source: #3356)
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// RoPE cos/sin tables for HEAD_DIM
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

// ===========================================================================
// 1. Per-head QK-Norm subgraph (flat 2D representation)
// ===========================================================================

/// Build the per-head QK-Norm subgraph for a single projection (Q or K).
///
/// Models the actual Qwen3 path:
///   hidden [SEQ, D_MODEL] -> Linear projection [SEQ, D_MODEL]
///     -> reshape [SEQ * N_HEADS, HEAD_DIM]  (flatten batch/seq with heads)
///     -> RMSNorm over HEAD_DIM (weight shape [HEAD_DIM])
///     -> reshape back [SEQ, D_MODEL]
///
/// The reshape-RMSNorm-reshape pattern captures the per-head normalization
/// that Qwen3 applies. In the real model, the reshape is to [B, seq, n_heads,
/// head_dim] then transpose; here we flatten to 2D for NY compatibility
/// while preserving the essential structure: RMSNorm normalizes each HEAD_DIM
/// slice independently.
///
/// NOTE: NY#3172 blocks tight CROWN through this reshape-norm-reshape
/// pattern. IBP propagation works but produces conservative bounds.
fn build_per_head_qk_norm_single() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_qk_norm_per_head_single");

    let x = b.add_input("x", &[SEQ, D_MODEL]);

    // Linear projection (Q or K)
    let proj_w = b.add_input("proj_w", &[D_MODEL, D_MODEL]);
    let projected = b.add_linear(x, proj_w, None, &[SEQ, D_MODEL]);

    // Reshape to expose per-head slices: [SEQ, D_MODEL] -> [SEQ * N_HEADS, HEAD_DIM]
    // This models: [B, seq, n_heads, head_dim].reshape(-1, head_dim)
    let flat_heads = b.add_reshape(projected, &[SEQ * N_HEADS, HEAD_DIM]);

    // Per-head RMSNorm: weight shape [HEAD_DIM], normalizing the last axis
    let eps = b.add_input("qk_norm_eps", &[1]);
    let norm_w = b.add_input("qk_norm_w", &[HEAD_DIM]);
    let normed = b.add_rms_norm(flat_heads, eps, 1, norm_w, &[SEQ * N_HEADS, HEAD_DIM]);

    // Reshape back: [SEQ * N_HEADS, HEAD_DIM] -> [SEQ, D_MODEL]
    let out = b.add_reshape(normed, &[SEQ, D_MODEL]);

    b.build(out).expect("valid per-head QK-Norm kernel")
}

fn per_head_qk_norm_single_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                               // x
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])), // proj_w
        TensorParamBinding::ConstantScalar(1e-5),                   // eps
        TensorParamBinding::ConstantTensor(ones(&[HEAD_DIM])),      // norm_w
    ]
}

/// Validates the per-head QK-Norm subgraph structure.
#[test]
fn test_qwen3_qk_norm_per_head_single_validates() {
    let def = build_per_head_qk_norm_single();
    def.validate()
        .expect("per-head QK-Norm single should validate");
}

/// IBP through per-head QK-Norm: finite bounds, valid structure.
#[test]
fn test_qwen3_qk_norm_per_head_single_ibp() {
    let def = build_per_head_qk_norm_single();
    let bindings = per_head_qk_norm_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through per-head QK-Norm");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 per-head QK-Norm (single) IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower must be finite, got {lo}");
    assert!(hi.is_finite(), "upper must be finite, got {hi}");
}

/// CROWN through per-head QK-Norm.
///
/// TODO(NY#3172): CROWN currently falls back to IBP through
/// the reshape-RMSNorm-reshape pattern. When NY#3172 lands,
/// this test should produce CROWN bounds tighter than IBP.
#[test]
fn test_qwen3_qk_norm_per_head_single_crown() {
    let def = build_per_head_qk_norm_single();
    let bindings = per_head_qk_norm_single_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 per-head QK-Norm (single) CROWN: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        // Expected until NY#3172 lands
        eprintln!("CROWN fallback (expected — blocked on NY#3172): {r}");
    }
}

/// Verification record for per-head QK-Norm with IbpValidated soundness.
#[test]
fn test_qwen3_qk_norm_per_head_single_verify_record() {
    let def = build_per_head_qk_norm_single();
    let bindings = per_head_qk_norm_single_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_qk_norm_per_head",
        &ibp_validated_config(),
    );
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
    eprintln!(
        "Qwen3 per-head QK-Norm verify: soundness={:?}, method={:?}",
        result.verification.soundness_mode, result.verification.method,
    );
}

// ===========================================================================
// 2. Full QK-Norm attention: Q_proj -> per-head norm -> attention -> O_proj
// ===========================================================================

/// Build the complete QK-Norm attention subgraph matching real Qwen3.
///
/// Models the actual Qwen3Attention::forward path:
///   x [SEQ, D_MODEL]
///     -> Q_proj, K_proj, V_proj (linear projections)
///     -> Q,K: reshape [SEQ*N_HEADS, HEAD_DIM] -> RMSNorm -> reshape [SEQ, D_MODEL]
///     -> scaled dot-product attention with causal mask
///     -> O_proj (output projection)
///     -> residual connection
///
/// Key difference from the existing depth test (compose_qwen3_depth.rs):
/// - RMSNorm is applied per-head (HEAD_DIM) not per-model (D_MODEL)
/// - The reshape-norm-reshape pattern matches the actual model structure
/// - K projection uses KV_DIM (for GQA) but is broadcast to D_MODEL for attention
///
/// TODO(NY#3172): Tight CROWN through per-head RMSNorm requires
/// NY support for bounds propagation through reshape-norm-reshape.
fn build_qk_norm_attention_full() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_qk_norm_attention_full");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let shape = [SEQ, D_MODEL];

    // Q projection: [SEQ, D_MODEL] -> [SEQ, D_MODEL]
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let q_proj = b.add_linear(x, q_w, None, &shape);

    // K projection: [SEQ, D_MODEL] -> [SEQ, KV_DIM]
    let k_w = b.add_input("k_w", &[KV_DIM, D_MODEL]);
    let k_proj = b.add_linear(x, k_w, None, &[SEQ, KV_DIM]);

    // V projection: [SEQ, D_MODEL] -> [SEQ, KV_DIM]
    let v_w = b.add_input("v_w", &[KV_DIM, D_MODEL]);
    let v_proj = b.add_linear(x, v_w, None, &[SEQ, KV_DIM]);

    // QK-Norm: per-head RMSNorm on Q
    // Q: [SEQ, D_MODEL] -> reshape [SEQ*N_HEADS, HEAD_DIM] -> RMSNorm -> reshape [SEQ, D_MODEL]
    let q_norm_eps = b.add_input("q_norm_eps", &[1]);
    let q_norm_w = b.add_input("q_norm_w", &[HEAD_DIM]);
    let q_flat = b.add_reshape(q_proj, &[SEQ * N_HEADS, HEAD_DIM]);
    let q_normed = b.add_rms_norm(q_flat, q_norm_eps, 1, q_norm_w, &[SEQ * N_HEADS, HEAD_DIM]);
    let q_out = b.add_reshape(q_normed, &shape);

    // QK-Norm: per-head RMSNorm on K
    // K: [SEQ, KV_DIM] -> reshape [SEQ*N_KV_HEADS, HEAD_DIM] -> RMSNorm -> reshape [SEQ, KV_DIM]
    let k_norm_eps = b.add_input("k_norm_eps", &[1]);
    let k_norm_w = b.add_input("k_norm_w", &[HEAD_DIM]);
    let k_flat = b.add_reshape(k_proj, &[SEQ * N_KV_HEADS, HEAD_DIM]);
    let k_normed = b.add_rms_norm(
        k_flat,
        k_norm_eps,
        1,
        k_norm_w,
        &[SEQ * N_KV_HEADS, HEAD_DIM],
    );
    let k_out = b.add_reshape(k_normed, &[SEQ, KV_DIM]);

    // GQA repeat_kv: tile K/V along the feature axis (axis 1) from KV_DIM to
    // D_MODEL. This is a genuine repeat, not a size-1 broadcast.
    let kv_repeat = D_MODEL / KV_DIM;
    let k_reps = vec![k_out; kv_repeat];
    let v_reps = vec![v_proj; kv_repeat];
    let k_expanded = b.add_concat(&k_reps, 1, &shape);
    let v_expanded = b.add_concat(&v_reps, 1, &shape);

    // Scaled dot-product attention (causal)
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_out,
        k_expanded,
        v_expanded,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );

    // Output projection
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);
    let projected = b.add_linear(attn_out, out_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(x, projected, &shape);

    b.build(out).expect("valid QK-Norm attention full kernel")
}

fn qk_norm_attention_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                               // x
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])), // q_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, D_MODEL])),  // k_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, D_MODEL])),  // v_w
        TensorParamBinding::ConstantScalar(1e-5),                   // q_norm_eps
        TensorParamBinding::ConstantTensor(ones(&[HEAD_DIM])),      // q_norm_w
        TensorParamBinding::ConstantScalar(1e-5),                   // k_norm_eps
        TensorParamBinding::ConstantTensor(ones(&[HEAD_DIM])),      // k_norm_w
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])), // out_w
    ]
}

/// Validates the full QK-Norm attention subgraph.
#[test]
fn test_qwen3_qk_norm_attention_full_validates() {
    let def = build_qk_norm_attention_full();
    def.validate()
        .expect("full QK-Norm attention should validate");
}

/// IBP through the full QK-Norm attention path.
///
/// With per-head RMSNorm constraining Q/K magnitudes, the attention logit
/// range should be bounded even for wide input bounds.
#[test]
fn test_qwen3_qk_norm_attention_full_ibp() {
    let def = build_qk_norm_attention_full();
    let bindings = qk_norm_attention_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Graph should have: linear projections (3) + reshape-norm-reshape (Q,K: 6)
    // + broadcast (2) + attention (1) + output proj (1) + residual (1) = 14+ nodes
    assert!(
        graph.num_nodes() >= 14,
        "QK-Norm attention graph >= 14 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full QK-Norm attention");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 QK-Norm attention (full) IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower must be finite, got {lo}");
    assert!(hi.is_finite(), "upper must be finite, got {hi}");

    // With residual + small weights, output should be close to input range
    assert!(lo.abs() < 1e4, "QK-Norm attention lower < 1e4, got {lo}");
    assert!(hi.abs() < 1e4, "QK-Norm attention upper < 1e4, got {hi}");
}

/// CROWN through the full QK-Norm attention.
///
/// TODO(NY#3172): CROWN through reshape-RMSNorm-reshape is expected
/// to fall back to IBP until NY supports per-head normalization
/// bounds propagation. This test documents the current behavior and will
/// serve as a regression test when the blocker is resolved.
#[test]
fn test_qwen3_qk_norm_attention_full_crown() {
    let def = build_qk_norm_attention_full();
    let bindings = qk_norm_attention_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 QK-Norm attention (full) CROWN: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("CROWN fallback (expected — blocked on NY#3172): {r}");
    }
}

/// Verification record for the full QK-Norm attention path.
#[test]
fn test_qwen3_qk_norm_attention_full_verify_record() {
    let def = build_qk_norm_attention_full();
    let bindings = qk_norm_attention_full_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_qk_norm_attention_full",
        &ibp_validated_config(),
    );
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
    eprintln!(
        "Qwen3 QK-Norm attention (full) verify: soundness={:?}",
        result.verification.soundness_mode,
    );
}

// ===========================================================================
// 3. Bounds tightening comparison: with vs without QK-Norm
// ===========================================================================

/// Build attention WITHOUT QK-Norm for comparison.
///
/// Same structure as build_qk_norm_attention_full but without the
/// reshape-RMSNorm-reshape on Q and K. This lets us measure how much
/// QK-Norm tightens the attention output bounds.
fn build_attention_without_qk_norm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_attention_no_qk_norm");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let shape = [SEQ, D_MODEL];

    // Q/K/V projections (same as QK-Norm version)
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let q = b.add_linear(x, q_w, None, &shape);

    let k_w = b.add_input("k_w", &[KV_DIM, D_MODEL]);
    let k = b.add_linear(x, k_w, None, &[SEQ, KV_DIM]);

    let v_w = b.add_input("v_w", &[KV_DIM, D_MODEL]);
    let v = b.add_linear(x, v_w, None, &[SEQ, KV_DIM]);

    // No QK-Norm: use Q and K directly. GQA repeat_kv tiles K/V along the
    // feature axis (axis 1) from KV_DIM to D_MODEL (genuine repeat, not broadcast).
    let kv_repeat = D_MODEL / KV_DIM;
    let k_reps = vec![k; kv_repeat];
    let v_reps = vec![v; kv_repeat];
    let k_expanded = b.add_concat(&k_reps, 1, &shape);
    let v_expanded = b.add_concat(&v_reps, 1, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q,
        k_expanded,
        v_expanded,
        AttentionMask::Causal,
        Some(scale),
        &shape,
    );

    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);
    let projected = b.add_linear(attn_out, out_w, None, &shape);
    let out = b.add_binary_add(x, projected, &shape);

    b.build(out).expect("valid attention without QK-Norm")
}

fn attention_without_qk_norm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                               // x
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])), // q_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, D_MODEL])),  // k_w
        TensorParamBinding::ConstantTensor(w(&[KV_DIM, D_MODEL])),  // v_w
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, D_MODEL])), // out_w
    ]
}

/// Compare IBP bounds width with and without QK-Norm.
///
/// QK-Norm normalizes Q/K per-head, which should constrain the dot-product
/// range and produce tighter attention output bounds. With small weights and
/// unit norm, the effect is subtle; with larger weights, the difference
/// becomes more pronounced.
///
/// This test documents the current bounds widths and serves as a regression
/// test for future NY improvements.
#[test]
fn test_qwen3_qk_norm_bounds_comparison() {
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // With QK-Norm
    let def_with = build_qk_norm_attention_full();
    let bindings_with = qk_norm_attention_full_bindings();
    let graph_with = tensor_kernel_to_graph(&def_with, &bindings_with).expect("graph with QK-Norm");
    let out_with = graph_with.propagate_ibp(&input).expect("IBP with QK-Norm");
    let (lo_with, hi_with) = bounds_min_max(&out_with);
    let width_with = hi_with - lo_with;

    // Without QK-Norm
    let def_without = build_attention_without_qk_norm();
    let bindings_without = attention_without_qk_norm_bindings();
    let graph_without =
        tensor_kernel_to_graph(&def_without, &bindings_without).expect("graph without QK-Norm");
    let out_without = graph_without
        .propagate_ibp(&input)
        .expect("IBP without QK-Norm");
    let (lo_without, hi_without) = bounds_min_max(&out_without);
    let width_without = hi_without - lo_without;

    eprintln!("QK-Norm bounds comparison (IBP):");
    eprintln!("  With QK-Norm:    width={width_with:.4}, bounds=[{lo_with:.4}, {hi_with:.4}]");
    eprintln!(
        "  Without QK-Norm: width={width_without:.4}, bounds=[{lo_without:.4}, {hi_without:.4}]"
    );

    if width_with < width_without {
        let tightening = 1.0 - width_with / width_without;
        eprintln!("  QK-Norm tightening: {:.1}%", tightening * 100.0);
    } else {
        // With small weights (0.001), the norm effect may be negligible in IBP.
        // This is expected — the real benefit appears with realistic weight
        // magnitudes and CROWN propagation (blocked on NY#3172).
        eprintln!(
            "  QK-Norm IBP bounds are not tighter (expected with small weights). \
             Real tightening requires CROWN support (NY#3172)."
        );
    }

    // Both must be finite and valid
    assert!(width_with.is_finite(), "QK-Norm width not finite");
    assert!(width_without.is_finite(), "no-QK-Norm width not finite");
}
