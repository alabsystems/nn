// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: TTS cross-attention monotonicity via NY + certificates.
//!
//! Verifies #1729 AC3: for autoregressive TTS cross-attention (codec tokens
//! attending to text embeddings), CROWN bounds on the pre-softmax score matrix
//! can prove diagonal dominance — a sufficient condition for monotonic attention.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 19.

#![allow(dead_code)]

use super::common;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Constants — TTS-scale parameters
// ---------------------------------------------------------------------------

/// Sequence length (must be equal for Q and KV due to NY constraint).
const SEQ_LEN: usize = 6;

/// Model dimension (embedding size).
const D_MODEL: usize = 8;

/// Number of attention heads.
const NUM_HEADS: usize = 2;

/// Per-head dimension.
const HEAD_DIM: usize = D_MODEL / NUM_HEADS;

/// Weight scale — small weights keep bounds tractable.
const W_SCALE: f32 = 0.05;

// ---------------------------------------------------------------------------
// Builder: cross-attention pre-softmax scores with projections
// ---------------------------------------------------------------------------

/// Build a cross-attention score graph that outputs pre-softmax S = Q_proj @ K_proj^T / sqrt(d_k).
fn build_tts_cross_attention_scores() -> nn_dsl::tensor_ir::TensorKernelDef {
    let d = D_MODEL;
    let h = NUM_HEADS;
    let dk = HEAD_DIM;
    let t = SEQ_LEN;

    let mut b = TensorBlockBuilder::new("tts_cross_attn_scores");

    // Inputs
    let q_input = b.add_input("decoder_hidden", &[t, d]); // codec token states
    let k_input = b.add_input("encoder_text", &[t, d]); // text embeddings
    let w_q = b.add_input("w_q", &[d, d]);
    let w_k = b.add_input("w_k", &[d, d]);

    // Project Q and K: [T, D] @ [D, D] → [T, D]
    let q_proj = b.add_matmul(q_input, w_q, false, None, &[t, d]);
    let k_proj = b.add_matmul(k_input, w_k, false, None, &[t, d]);

    // Reshape to multi-head: [T, D] → [T, H, dk]
    let q_mh = b.add_reshape(q_proj, &[t, h, dk]);
    let k_mh = b.add_reshape(k_proj, &[t, h, dk]);

    // Transpose to [H, T, dk]: permute [T, H, dk] → [H, T, dk] via axes [1, 0, 2]
    let q_t = b.add_transpose(q_mh, &[1, 0, 2], &[h, t, dk]);
    let k_t = b.add_transpose(k_mh, &[1, 0, 2], &[h, t, dk]);

    // Attention scores: Q @ K^T / sqrt(dk) → [H, T, T]
    let scale = 1.0 / (dk as f32).sqrt();
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &[h, t, t]);

    b.build(scores).expect("valid cross-attention score graph")
}

/// Build bindings for cross-attention: Q=Variable (decoder), K=ConstantTensor (encoder).
fn tts_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let t = SEQ_LEN;
    let d = D_MODEL;

    // Encoder text embeddings: identity-like structure.
    let mut k_data = vec![0.0f32; t * d];
    let cols_per_pos = d / t;
    if cols_per_pos > 0 {
        for pos in 0..t {
            for c in 0..cols_per_pos {
                let col = pos * cols_per_pos + c;
                if col < d {
                    k_data[pos * d + col] = 1.0;
                }
            }
        }
    } else {
        // Fallback for T > D: use one-hot-ish with modular wrapping.
        for pos in 0..t {
            k_data[pos * d + (pos % d)] = 1.0;
        }
    }
    let k_tensor = ArrayD::from_shape_vec(IxDyn(&[t, d]), k_data).expect("valid K shape");

    // Projection weights: small uniform for tractable bounds.
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), W_SCALE);

    vec![
        TensorParamBinding::Variable, // decoder_hidden (Variable)
        TensorParamBinding::ConstantTensor(k_tensor), // encoder_text (Constant)
        TensorParamBinding::ConstantTensor(w_proj.clone()), // w_q
        TensorParamBinding::ConstantTensor(w_proj), // w_k
    ]
}

/// Input bounds: decoder hidden states in [-1, 1].
fn tts_input_bounds() -> BoundedTensor {
    common::uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0)
}

/// Propagate through graph (IBP).
fn graph_propagate(
    def: &nn_dsl::tensor_ir::TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
) -> BoundedTensor {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph");
    graph.propagate_ibp(input).expect("IBP propagation")
}

// ---------------------------------------------------------------------------
// Test: graph builds
// ---------------------------------------------------------------------------

#[test]
fn test_tts_cross_attn_scores_graph_builds() {
    let def = build_tts_cross_attention_scores();
    let bindings = tts_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("cross-attn score graph");

    assert!(
        graph.num_nodes() >= 5,
        "cross-attn score graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test: IBP propagation produces valid bounds
// ---------------------------------------------------------------------------

#[test]
fn test_tts_cross_attn_scores_ibp() {
    let def = build_tts_cross_attention_scores();
    let bindings = tts_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = tts_input_bounds();
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, _hi) = output.lower_upper();

    // Output shape: [H, T, T]
    assert_eq!(lo.shape(), &[NUM_HEADS, SEQ_LEN, SEQ_LEN], "output shape");
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test: CROWN propagation
// ---------------------------------------------------------------------------

#[test]
fn test_tts_cross_attn_scores_crown() {
    let def = build_tts_cross_attention_scores();
    let bindings = tts_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = tts_input_bounds();
    let (method, output, fallback_reason) =
        common::assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("TTS cross-attn scores: method={method:?}, fallback={fallback_reason:?}");
    common::assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Test: extract score bounds → monotonicity certificate
// ---------------------------------------------------------------------------

#[test]
fn test_tts_cross_attn_monotonicity_certificate() {
    let def = build_tts_cross_attention_scores();
    let bindings = tts_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = tts_input_bounds();
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    let t = SEQ_LEN;

    for head in 0..NUM_HEADS {
        let score_lower: Vec<f32> = (0..t * t).map(|i| lo[[head, i / t, i % t]]).collect();
        let score_upper: Vec<f32> = (0..t * t).map(|i| hi[[head, i / t, i % t]]).collect();

        let cert = nn_tts_verify::monotonicity::interpret_attention_monotonicity(
            &score_lower,
            &score_upper,
            t, // decoder_steps
            t, // encoder_positions (same for now)
            1.0,
            "IBP",
        )
        .expect("valid certificate");

        eprintln!(
            "Head {head}: min_margin={:.4}, proven={}, margins={:?}",
            cert.min_margin, cert.is_proven, cert.row_margins
        );

        assert_eq!(cert.decoder_steps, t);
        assert_eq!(cert.encoder_positions, t);
        assert_eq!(cert.row_margins.len(), t);

        for (row, margin) in cert.row_margins.iter().enumerate() {
            assert!(margin.is_finite(), "row {row} margin must be finite");
        }
    }
}

// ---------------------------------------------------------------------------
// Test: diagonal-boosted K → proven monotonicity
// ---------------------------------------------------------------------------

#[test]
fn test_tts_cross_attn_monotonicity_tight_bounds() {
    let t = 4; // Smaller for tighter bounds
    let d = 8;

    let mut b = TensorBlockBuilder::new("tts_mono_tight");
    let hidden = b.add_input("hidden", &[t, d]);
    let pe = b.add_input("pe", &[t, d]);
    let k = b.add_input("key", &[t, d]);

    // Q = hidden + PE
    let q = b.add_binary_add(hidden, pe, &[t, d]);

    // Scores = Q @ K^T / sqrt(d) → [T, T]
    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[t, t]);
    let def = b.build(scores).expect("valid");

    // Build identity-like K: each position has large value in distinct columns.
    let k_scale = 3.0;
    let cols_per = d / t;
    let mut k_data = vec![0.0f32; t * d];
    for pos in 0..t {
        for c in 0..cols_per {
            k_data[pos * d + pos * cols_per + c] = k_scale;
        }
    }
    let k_tensor = ArrayD::from_shape_vec(IxDyn(&[t, d]), k_data).expect("K");

    let bindings = vec![
        TensorParamBinding::Variable, // hidden (Variable)
        TensorParamBinding::ConstantTensor(k_tensor.clone()), // pe (Constant = K)
        TensorParamBinding::ConstantTensor(k_tensor), // key (Constant = K)
    ];

    // Tiny hidden perturbation: [-0.01, 0.01]
    let input = common::uniform_bounds(&[t, d], 0.01);
    let output = graph_propagate(&def, &bindings, &input);

    let (lo, hi) = output.lower_upper();
    let score_lower: Vec<f32> = lo.iter().copied().collect();
    let score_upper: Vec<f32> = hi.iter().copied().collect();

    let cert = nn_tts_verify::monotonicity::interpret_attention_monotonicity(
        &score_lower,
        &score_upper,
        t,
        t,
        0.01,
        "IBP",
    )
    .expect("valid certificate");

    eprintln!(
        "PE-aware: min_margin={:.6}, proven={}, margins={:?}",
        cert.min_margin, cert.is_proven, cert.row_margins
    );

    assert!(
        cert.is_proven,
        "diagonal dominance should be provable with PE-aware Q: min_margin={}",
        cert.min_margin
    );
    assert!(cert.min_margin > 0.0, "margin must be positive");
}

// ---------------------------------------------------------------------------
// Test: moonshot certificate integration (Property 3)
// ---------------------------------------------------------------------------

#[test]
fn test_tts_monotonicity_certificate_maps_to_property3() {
    let t = 4;
    let d = 8;

    let mut b = TensorBlockBuilder::new("mono_prop3");
    let hidden = b.add_input("hidden", &[t, d]);
    let pe = b.add_input("pe", &[t, d]);
    let k = b.add_input("key", &[t, d]);
    let q = b.add_binary_add(hidden, pe, &[t, d]);
    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[t, t]);
    let def = b.build(scores).expect("valid");

    let k_scale = 3.0;
    let cols_per = d / t;
    let mut k_data = vec![0.0f32; t * d];
    for pos in 0..t {
        for c in 0..cols_per {
            k_data[pos * d + pos * cols_per + c] = k_scale;
        }
    }
    let k_tensor = ArrayD::from_shape_vec(IxDyn(&[t, d]), k_data).expect("K");
    let bindings = vec![
        TensorParamBinding::Variable, // hidden (Variable)
        TensorParamBinding::ConstantTensor(k_tensor.clone()), // pe = K
        TensorParamBinding::ConstantTensor(k_tensor), // key = K
    ];

    let input = common::uniform_bounds(&[t, d], 0.01);
    let output = graph_propagate(&def, &bindings, &input);
    let (lo, hi) = output.lower_upper();

    let score_lower: Vec<f32> = lo.iter().copied().collect();
    let score_upper: Vec<f32> = hi.iter().copied().collect();

    let cert = nn_tts_verify::monotonicity::interpret_attention_monotonicity(
        &score_lower,
        &score_upper,
        t,
        t,
        0.01,
        "IBP",
    )
    .expect("valid certificate");

    assert!(cert.is_proven, "monotonicity must be provable");
    assert!(
        cert.min_margin > 0.0,
        "positive margin proves diagonal dominance"
    );

    assert_eq!(cert.decoder_steps, t);
    assert_eq!(cert.encoder_positions, t);
    assert_eq!(cert.row_margins.len(), t);
    assert!(
        cert.row_margins.iter().all(|m| *m > 0.0),
        "all rows must have positive margin for proven monotonicity"
    );
    assert_eq!(cert.propagation_mode, "IBP");

    // Verify MoonshotStatus knows Property 3 is "Intelligible (attention monotonic)"
    let status = nn_tts_verify::moonshot::MoonshotStatus::from_repo();
    assert_eq!(
        status.properties[2].name, "Intelligible (attention monotonic)",
        "Property 3 (index 2) is the monotonicity property"
    );
}
