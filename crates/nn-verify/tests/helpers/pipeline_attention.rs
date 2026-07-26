// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attention certificate builder and synthetic evidence helpers for the
//! 5-stage moonshot pipeline integration tests.
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

use crate::common::uniform_bounds;

// ---------------------------------------------------------------------------
// Attention certificate builder
// ---------------------------------------------------------------------------

/// Build a PE-aware cross-attention score graph and interpret the resulting
/// bounds as an `AttentionMonotonicityCertificate`.
///
/// Graph: Q = hidden + PE, Scores = (Q @ K^T) / sqrt(d).
///   - `hidden` is Variable with `±input_bound`.
///   - PE and K are diagonally-dominant constants: each position has large
///     values in distinct columns so that S[t,t] > S[t,j].
///
/// Follows the same construction as `build_attention_certificate()` in
/// `compose_moonshot_attention_integration.rs` but is self-contained for
/// the 5-stage pipeline test.
pub(crate) fn build_attention_certificate_for_pipeline(
    input_bound: f32,
) -> nn_tts_verify::monotonicity::AttentionMonotonicityCertificate {
    let t = 4; // decoder/encoder positions
    let d = 8; // embedding dimension

    let mut b = TensorBlockBuilder::new("full_pipeline_attn");
    let hidden = b.add_input("hidden", &[t, d]);
    let pe = b.add_input("pe", &[t, d]);
    let k = b.add_input("key", &[t, d]);

    // Q = hidden + PE
    let q = b.add_binary_add(hidden, pe, &[t, d]);

    // Scores = Q @ K^T / sqrt(d) → [T, T]
    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[t, t]);
    let def = b.build(scores).expect("valid attention graph");

    // Identity-like K: each position has large value in distinct columns.
    let k_scale = 3.0;
    let cols_per = d / t;
    let mut k_data = vec![0.0f32; t * d];
    for pos in 0..t {
        for c in 0..cols_per {
            k_data[pos * d + pos * cols_per + c] = k_scale;
        }
    }
    let k_tensor = ArrayD::from_shape_vec(IxDyn(&[t, d]), k_data).expect("K");

    // PE = K (same structure) → diagonally dominant constant component.
    let bindings = vec![
        TensorParamBinding::Variable,                         // hidden
        TensorParamBinding::ConstantTensor(k_tensor.clone()), // pe = K
        TensorParamBinding::ConstantTensor(k_tensor),         // key = K
    ];

    let input = uniform_bounds(&[t, d], input_bound);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    let (lo, hi) = output.lower_upper();
    let score_lower: Vec<f32> = lo.iter().copied().collect();
    let score_upper: Vec<f32> = hi.iter().copied().collect();

    let mode = "IBP".to_string();
    nn_tts_verify::monotonicity::interpret_attention_monotonicity(
        &score_lower,
        &score_upper,
        t,
        t,
        f64::from(input_bound),
        &mode,
    )
    .expect("valid monotonicity certificate")
}

// ---------------------------------------------------------------------------
// Synthetic evidence builders for P4/P5
// ---------------------------------------------------------------------------

/// Build a synthetic timing certificate for the 5-stage pipeline.
pub(crate) fn build_synthetic_timing(
    bounds_cert: &nn_tts_verify::pipeline::PipelineCertificate,
    dim: usize,
) -> nn_tts_verify::pipeline::TimingCertificate {
    nn_tts_verify::pipeline::TimingCertificate::new(
        bounds_cert.clone(),
        vec![
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "f0_energy_predictor",
                2_000_000,
                4 * dim as u64,
                8_000.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "f0_to_prosody_adapter",
                50_000,
                dim as u64,
                200.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "prosody_predictor",
                3_000_000,
                4 * dim as u64,
                10_000.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "duration_to_decoder_adapter",
                50_000,
                dim as u64,
                200.0,
                None,
            ),
            nn_tts_verify::cost_model::LayerCostProfile::new(
                "kokoro_decoder",
                20_000_000,
                16 * dim as u64,
                25_000.0,
                None,
            ),
        ],
        43_400.0,
        25_100_000,
        26 * dim as u64,
        "M4 Max (synthetic)",
        100_000.0,
        true,
        true,
        None,
    )
}

/// Build synthetic speaker consistency evidence.
pub(crate) fn build_synthetic_speaker() -> nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence
{
    let embed_dim = 32;
    nn_tts_verify::moonshot_crown::SpeakerConsistencyEvidence::new(
        embed_dim,
        vec![-0.05; embed_dim],
        vec![0.05; embed_dim],
        vec![0.0; embed_dim],
        0.5,
        true,
    )
}
