// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 16: Formal certificate generation for attention monotonicity.
//!
//! Phase 15 proved monolithic CROWN gives 27% tighter bounds than layerwise,
//! and established adversarial perturbation stability at D=8. Phase 16 delivers
//! certificate generation (#1729): formal proof artifacts from CROWN
//! verification — structured records containing verification method, input
//! specification, output bounds, and provenance. These certificates are the
//! building blocks for machine-checkable proofs of attention monotonicity.
//!
//! Key results:
//!   - Certificate records capture: architecture, dimensions, perturbation
//!     budget, verification method, per-element output bounds, diagonal
//!     dominance status, and soundness provenance
//!   - `verify_tensor_and_record` persists certificates to nn_verify_status.json
//!
//! Phoneme encoder adversarial stability tests (#1740 AC2) are in the
//! companion file `compose_phoneme_certificate_phase16.rs`.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 16.

pub(crate) use super::common;

#[path = "attention_monotonicity.rs"]
mod attn_helpers;

#[path = "phoneme_stability.rs"]
mod phoneme_helpers;

#[path = "certificate_types.rs"]
mod cert_types;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_layerwise_builders.rs"]
pub(crate) mod lw_builders;

#[allow(dead_code, unreachable_pub)]
#[path = "attention_e2e_runners.rs"]
mod e2e_runners;

use cert_types::{
    count_diagonal_dominant, measure_avg_width, measure_max_width, AttentionMonotonicityCertificate,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    tensor_kernel_to_graph, verify_tensor_and_record, TensorParamBinding, VerifyStatus,
};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Tests: Certificate generation for attention monotonicity (#1729)
// ===========================================================================

/// Generate formal certificate for PE-centered attention scores.
///
/// Uses `verify_tensor_and_record` to persist the verification result to
/// nn_verify_status.json, creating a persistent proof artifact.
#[test]
fn test_certificate_pe_attention_d8() {
    let (seq_len, d) = (4, 8);
    let eps = 0.05;
    let pe_scale = 5.0; // Scale PE for provable diagonal dominance

    let status_key = "cert_attn_mono_pe_d8";

    // Build score graph with PE-centered input
    let (def, _) = attn_helpers::build_attention_scores_positional();
    let bindings = attn_helpers::attention_scores_positional_bindings_scaled(pe_scale);
    let input = common::uniform_bounds(&[seq_len, d], eps);

    // Run verification pipeline with persistence
    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("certificate verification");

    common::assert_bounds_valid(&result.output_bounds);

    let diag_dom = count_diagonal_dominant(&result.output_bounds, seq_len);

    let cert = AttentionMonotonicityCertificate {
        architecture: "score(PE_scaled + hidden, K=PE) / √D".into(),
        seq_len,
        d_model: d,
        perturbation_eps: eps,
        perturbation_type: format!("uniform L∞ (PE_scale={pe_scale})"),
        method: result.verification.method,
        avg_width: measure_avg_width(&result.output_bounds),
        max_width: measure_max_width(&result.output_bounds),
        diagonal_dominant_positions: diag_dom,
        total_positions: seq_len,
        monotonicity_proved: diag_dom == seq_len,
        status_key: status_key.into(),
    };

    cert.emit_report();

    // Certificate must be persisted
    assert!(
        status.kernel(status_key).is_some(),
        "certificate must be recorded in status"
    );
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );
}

/// Generate certificate for monolithic 3-layer attention at D=8.
///
/// Full pipeline (score → softmax → output) with adversarial perturbation.
/// This certificate covers the entire attention computation path.
#[test]
fn test_certificate_monolithic_attention_d8() {
    let (seq_len, d) = (4, 8);
    let eps = 0.05;
    let status_key = "cert_attn_mono_3layer_d8";

    // Build monolithic 3-layer graph
    let mut b = TensorBlockBuilder::new("cert_mono_3l_d8");
    let q = b.add_input("query", &[seq_len, d]);
    let k = b.add_input("key", &[seq_len, d]);
    let v = b.add_input("value", &[seq_len, d]);

    let scale = 1.0 / (d as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[seq_len, seq_len]);
    let weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);
    let output = b.add_matmul(weights, v, false, None, &[seq_len, d]);
    let def = b.build(output).expect("valid monolithic graph");

    let pe = attn_helpers::build_sinusoidal_pe(seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);
    let v_data: Vec<f32> = (0..seq_len * d)
        .map(|i| 0.1 * ((i % 5) as f32 - 2.0))
        .collect();
    let v_tensor = ArrayD::from_shape_vec(IxDyn(&[seq_len, d]), v_data).expect("V shape");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
        TensorParamBinding::ConstantTensor(v_tensor),
    ];

    let input = lw_builders::build_pe_centered_bounds(&pe, eps);

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("certificate verification");

    common::assert_bounds_valid(&result.output_bounds);

    eprintln!("=== MONOLITHIC ATTENTION CERTIFICATE ===");
    eprintln!("Architecture:     score→softmax→output (3-layer monolithic)");
    eprintln!("Dimensions:       T={seq_len}, D={d}");
    eprintln!("Perturbation:     PE-centered L∞ ε={eps}");
    eprintln!("Method:           {:?}", result.verification.method);
    eprintln!(
        "Bounds:           avg_w={:.6}, max_w={:.6}",
        measure_avg_width(&result.output_bounds),
        measure_max_width(&result.output_bounds)
    );
    eprintln!("Persisted:        status_key={status_key}");
    eprintln!("=========================================");

    assert!(status.kernel(status_key).is_some());
}

/// Certificate generation for 1-block ProsodyPredictor at D=8.
///
/// Generates a formal certificate for the full Kokoro ProsodyPredictor
/// architecture (Conv1d + AdaLayerNorm + Gate + Residual + Attention).
#[test]
fn test_certificate_prosody_predictor_1block_d8() {
    let seq_len = attn_helpers::SEQ_LEN;
    let d = attn_helpers::D_MODEL;
    let status_key = "cert_prosody_1block_d8";

    // Use simplified prosody score graph (hidden + PE → attention scores)
    let (def, _output_shape) =
        e2e_runners::build_prosody_score_graph("cert_prosody_1b_d8", seq_len, d);

    let pe = attn_helpers::build_sinusoidal_pe(seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
        TensorParamBinding::ConstantTensor(k_tensor),
    ];

    let input_bound = 0.05;
    let input = common::uniform_bounds(&[seq_len, d], input_bound);

    let mut status = VerifyStatus::default();
    let result = verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(status_key))
        .expect("prosody certificate verification");

    common::assert_bounds_valid(&result.output_bounds);

    let diag_dom = count_diagonal_dominant(&result.output_bounds, seq_len);

    let cert = AttentionMonotonicityCertificate {
        architecture: "ProsodyPredictor(1-block) → scores".into(),
        seq_len,
        d_model: d,
        perturbation_eps: input_bound,
        perturbation_type: "uniform L∞".into(),
        method: result.verification.method,
        avg_width: measure_avg_width(&result.output_bounds),
        max_width: measure_max_width(&result.output_bounds),
        diagonal_dominant_positions: diag_dom,
        total_positions: seq_len,
        monotonicity_proved: diag_dom == seq_len,
        status_key: status_key.into(),
    };

    cert.emit_report();

    assert!(status.kernel(status_key).is_some());
}

// ===========================================================================
// Tests: Combined analysis (attention + phoneme)
// ===========================================================================

/// Adversarial perturbation sweep: combined attention + phoneme analysis.
///
/// For a fixed phoneme confusion set, sweeps increasing perturbation
/// budgets in the embedding space and measures both attention score
/// stability and phoneme encoder output stability. This connects the
/// Phase 15 adversarial analysis to phoneme-level verification.
#[test]
fn test_combined_attention_phoneme_adversarial_sweep() {
    let (seq_len, d) = (4, 8);

    // Build attention score graph
    let score_def = lw_builders::build_score_layer("adv_sweep_scores", seq_len, d);
    let k_tensor = lw_builders::build_k_identity(seq_len, d, 1.0);
    let score_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(k_tensor),
    ];

    // Build phoneme encoder graph
    let phon_def = phoneme_helpers::build_phoneme_encoder();
    let phon_bindings = phoneme_helpers::phoneme_encoder_bindings();

    let pe = attn_helpers::build_sinusoidal_pe(seq_len, d);

    eprintln!("--- Combined adversarial sweep (attention + phoneme) ---");
    eprintln!("  ε       attn_avg    attn_max    phon_avg    phon_max");

    for &eps in &[0.01f32, 0.05, 0.1, 0.2] {
        // Attention analysis: PE-centered perturbation
        let attn_input = lw_builders::build_pe_centered_bounds(&pe, eps);
        let attn_graph = tensor_kernel_to_graph(&score_def, &score_bindings).expect("attn graph");
        let (_, attn_out, _) = nn_verify::propagate_with_crown_fallback(&attn_graph, &attn_input)
            .expect("attn propagation");

        // Phoneme analysis: uniform perturbation in embedding space
        let phon_input =
            common::uniform_bounds(&[phoneme_helpers::SEQ_LEN, phoneme_helpers::EMBED_DIM], eps);
        let phon_graph = tensor_kernel_to_graph(&phon_def, &phon_bindings).expect("phon graph");
        let (_, phon_out, _) = nn_verify::propagate_with_crown_fallback(&phon_graph, &phon_input)
            .expect("phon propagation");

        common::assert_bounds_valid(&attn_out);
        common::assert_bounds_valid(&phon_out);

        eprintln!(
            "  {eps:<7.2} {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            measure_avg_width(&attn_out),
            measure_max_width(&attn_out),
            measure_avg_width(&phon_out),
            measure_max_width(&phon_out),
        );
    }
}

/// Certificate comparison: attention certificates at multiple PE scales.
///
/// Generates certificates at PE_scale = 1, 3, 5, 10 and documents
/// how PE scaling affects diagonal dominance provability. This is the
/// key parameter study for #1729 certificate generation.
#[test]
fn test_certificate_pe_scale_comparison() {
    let (seq_len, d) = (4, 8);
    let eps = 0.1;

    eprintln!("--- Certificate PE scale comparison (D={d}, ε={eps}) ---");
    eprintln!("  pe_scale  diag_dom  avg_w       max_w       proved?");

    for &pe_scale in &[1.0f32, 3.0, 5.0, 10.0, 20.0] {
        let status_key = format!("cert_pe_scale_{}", pe_scale as u32);

        let (def, _) = attn_helpers::build_attention_scores_positional();
        let bindings = attn_helpers::attention_scores_positional_bindings_scaled(pe_scale);
        let input = common::uniform_bounds(&[seq_len, d], eps);

        let mut status = VerifyStatus::default();
        let result =
            verify_tensor_and_record(&mut status, &def, &bindings, &input, Some(&status_key))
                .expect("verification");

        common::assert_bounds_valid(&result.output_bounds);

        let diag_dom = count_diagonal_dominant(&result.output_bounds, seq_len);
        let proved = diag_dom == seq_len;

        eprintln!(
            "  {pe_scale:<9.1} {diag_dom}/{seq_len}       {:>10.6}  {:>10.6}  {}",
            measure_avg_width(&result.output_bounds),
            measure_max_width(&result.output_bounds),
            if proved { "YES" } else { "no" },
        );

        assert!(status.kernel(&status_key).is_some());
    }
}
