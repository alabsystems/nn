// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Real NY pipeline certificates for TTS.
//!
//! Builds simplified TTS stages as `TensorKernelDef` graphs, runs actual
//! CROWN/IBP propagation, converts results via `stage_from_propagation()`,
//! and composes into `PipelineCertificate` objects.
//!
//! This is the first test producing pipeline certificates from real
//! NY verification output — not synthetic uniform bounds.
//!
//! Run with: `cargo test -p nn-tts-verify --test pipeline_crown --features NY`
//!
//! Part of #1725: Formally Verified End-to-End TTS Pipeline.

#![cfg(feature = "ny")]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::TensorKernelDef;
use nn_tts_verify::{stage_from_propagation, verify_pipeline};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Feature dimension from text encoder (production: 512).
const FEATURE_DIM: usize = 8;

/// Sequence length (production: variable).
const SEQ_LEN: usize = 4;

/// Hidden channels in decoder (production: 256).
const HIDDEN_CHANNELS: usize = 4;

/// Output channels (production: 2 * n_fft_bins).
const OUT_CHANNELS: usize = 4;

/// Embedding dimension from input phonemes (production: 256).
const EMBED_DIM: usize = 4;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.01;

// ---------------------------------------------------------------------------
// Stage builders
// ---------------------------------------------------------------------------

/// Build a simplified prosody predictor stage.
///
/// Architecture: input [FEATURE_DIM, SEQ_LEN] → Linear → ReLU → Linear
///   → output [HIDDEN_CHANNELS, SEQ_LEN]
///
/// This models the prosody predictor mapping text features to a latent
/// representation consumed by the decoder.
fn build_prosody_stage() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("prosody_predictor");

    // Variable input: text features [FEATURE_DIM, SEQ_LEN]
    let input = b.add_input("text_features", &[FEATURE_DIM, SEQ_LEN]);

    // Linear projection: Conv1d with kernel=1 (equivalent to per-timestep linear)
    // [FEATURE_DIM, SEQ_LEN] → [HIDDEN_CHANNELS, SEQ_LEN]
    let w1 = b.add_input("proj_w", &[HIDDEN_CHANNELS, FEATURE_DIM, 1]);
    let h = b.add_conv1d(input, w1, None, 1, 0, &[HIDDEN_CHANNELS, SEQ_LEN]);

    // ReLU activation
    let h_relu = b.add_relu(h, &[HIDDEN_CHANNELS, SEQ_LEN]);

    // Second linear projection (identity-scale for tractability)
    let w2 = b.add_input("out_proj_w", &[HIDDEN_CHANNELS, HIDDEN_CHANNELS, 1]);
    let output = b.add_conv1d(h_relu, w2, None, 1, 0, &[HIDDEN_CHANNELS, SEQ_LEN]);

    let def = b.build(output).expect("valid prosody graph");

    let bindings = vec![
        // text_features: Variable
        TensorParamBinding::Variable,
        // proj_w [HIDDEN_CHANNELS, FEATURE_DIM, 1]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_CHANNELS, FEATURE_DIM, 1]),
            WEIGHT_MAG,
        )),
        // out_proj_w [HIDDEN_CHANNELS, HIDDEN_CHANNELS, 1]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_CHANNELS, HIDDEN_CHANNELS, 1]),
            WEIGHT_MAG,
        )),
    ];

    (def, bindings)
}

/// Build a simplified decoder stage.
///
/// Architecture: input [HIDDEN_CHANNELS, SEQ_LEN] → Conv1d → Exp
///   → output [OUT_CHANNELS, SEQ_LEN]
///
/// This models the vocoder decoder that maps latent features to
/// audio spectral magnitudes (exp converts log-magnitude to magnitude).
fn build_decoder_stage() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("decoder");

    // Variable input: latent features [HIDDEN_CHANNELS, SEQ_LEN]
    let input = b.add_input("latent_features", &[HIDDEN_CHANNELS, SEQ_LEN]);

    // Conv1d projection to output channels
    let w = b.add_input("conv_w", &[OUT_CHANNELS, HIDDEN_CHANNELS, 3]);
    let h = b.add_conv1d(input, w, None, 1, 1, &[OUT_CHANNELS, SEQ_LEN]);

    // Exp activation: log-magnitude → magnitude (always positive)
    let output = b.add_exp(h, &[OUT_CHANNELS, SEQ_LEN]);

    let def = b.build(output).expect("valid decoder graph");

    let bindings = vec![
        // latent_features: Variable
        TensorParamBinding::Variable,
        // conv_w [OUT_CHANNELS, HIDDEN_CHANNELS, 3]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CHANNELS, HIDDEN_CHANNELS, 3]),
            WEIGHT_MAG,
        )),
    ];

    (def, bindings)
}

/// Build a simplified text encoder stage.
///
/// Architecture: input [EMBED_DIM, SEQ_LEN] → Conv1d → ReLU
///   → output [FEATURE_DIM, SEQ_LEN]
///
/// This models the phoneme embedding → text feature transformation that
/// feeds into the prosody predictor.
fn build_text_encoder_stage() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("text_encoder");

    // Variable input: phoneme embeddings [EMBED_DIM, SEQ_LEN]
    let input = b.add_input("phoneme_embeddings", &[EMBED_DIM, SEQ_LEN]);

    // Conv1d projection: [EMBED_DIM, SEQ_LEN] → [FEATURE_DIM, SEQ_LEN]
    let w = b.add_input("enc_w", &[FEATURE_DIM, EMBED_DIM, 1]);
    let h = b.add_conv1d(input, w, None, 1, 0, &[FEATURE_DIM, SEQ_LEN]);

    // ReLU activation
    let output = b.add_relu(h, &[FEATURE_DIM, SEQ_LEN]);

    let def = b.build(output).expect("valid text encoder graph");

    let bindings = vec![
        // phoneme_embeddings: Variable
        TensorParamBinding::Variable,
        // enc_w [FEATURE_DIM, EMBED_DIM, 1]
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FEATURE_DIM, EMBED_DIM, 1]),
            WEIGHT_MAG,
        )),
    ];

    (def, bindings)
}

/// Create uniform BoundedTensor with [-range, +range] bounds.
fn uniform_bounds(shape: &[usize], range: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), -range);
    let upper = ArrayD::from_elem(IxDyn(shape), range);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

// ---------------------------------------------------------------------------
// Helpers for CROWN propagation
// ---------------------------------------------------------------------------

/// Build a stage graph, run CROWN propagation, and return a `VerifiedStage`.
fn propagate_stage(
    name: &str,
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
) -> (nn_tts_verify::VerifiedStage, BoundedTensor) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph translation");
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, input).expect("CROWN propagation");
    let stage = stage_from_propagation(name, input, &output, &method);
    (stage, output)
}

/// Compute a uniform input range that encompasses the given bounds with margin.
fn range_with_margin(bounds: &BoundedTensor, min_range: f32) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi_max - lo_min).abs();
    (span + span * 0.5).max(min_range)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Individual prosody stage produces valid IBP bounds.
#[test]
fn test_prosody_stage_ibp_propagation() {
    let (def, bindings) = build_prosody_stage();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FEATURE_DIM, SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[HIDDEN_CHANNELS, SEQ_LEN]);

    // Check finiteness and ordering (second Conv1d can produce negatives).
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(lo_min.is_finite(), "lower bounds should be finite");
    assert!(hi_max.is_finite(), "upper bounds should be finite");
    assert!(lo_min <= hi_max, "lower <= upper");
}

/// Individual decoder stage produces valid IBP bounds.
#[test]
fn test_decoder_stage_ibp_propagation() {
    let (def, bindings) = build_decoder_stage();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[HIDDEN_CHANNELS, SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, SEQ_LEN]);

    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Exp output should always be positive.
    assert!(
        lo_min > 0.0,
        "exp output must be positive, got lo_min={lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bounds should be finite");
}

/// Two-stage TTS pipeline certificate from real NY CROWN propagation.
///
/// Pipeline: prosody_predictor (Linear→ReLU→Linear) → decoder (Conv1d→Exp).
/// Junction check: prosody output bounds ⊆ decoder verified input range.
#[test]
fn test_two_stage_pipeline_real_crown() {
    // Stage 1: Prosody predictor with CROWN.
    let (pros_def, pros_bindings) = build_prosody_stage();
    let pros_input = uniform_bounds(&[FEATURE_DIM, SEQ_LEN], 1.0);
    let (pros_stage, pros_output) =
        propagate_stage("prosody_predictor", &pros_def, &pros_bindings, &pros_input);

    // Stage 2: Decoder with input range encompassing prosody output + margin.
    let (dec_def, dec_bindings) = build_decoder_stage();
    let dec_range = range_with_margin(&pros_output, 1.0);
    let dec_input = uniform_bounds(&[HIDDEN_CHANNELS, SEQ_LEN], dec_range);
    let (dec_stage, _) = propagate_stage("decoder", &dec_def, &dec_bindings, &dec_input);

    // Compose pipeline and verify.
    let cert = verify_pipeline(&[pros_stage, dec_stage]).expect("pipeline verification");
    eprintln!("{}", cert.report());

    assert_eq!(cert.stages.len(), 2);
    assert_eq!(cert.junctions.len(), 1);
    assert_eq!(cert.stages[0].name, "prosody_predictor");
    assert_eq!(cert.stages[1].name, "decoder");
    assert!(
        cert.junctions[0].shape_compatible,
        "shapes should be compatible"
    );
    assert!(!cert.e2e_input_lower.is_empty());
    assert!(!cert.e2e_output_lower.is_empty());

    // Decoder output (after exp) should have positive lower bounds.
    let output_lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        output_lo_min > 0.0,
        "exp output must be positive, got {output_lo_min}"
    );

    assert!(
        cert.is_valid,
        "Two-stage pipeline must be valid — prosody output ⊆ decoder input (by construction via range_with_margin)"
    );
}

/// Pipeline certificate soundness reflects CROWN vs IBP propagation.
#[test]
fn test_pipeline_soundness_from_real_propagation() {
    // IBP (not sound) for prosody.
    let (pros_def, pros_bindings) = build_prosody_stage();
    let pros_graph = tensor_kernel_to_graph(&pros_def, &pros_bindings).expect("prosody graph");
    let pros_input = uniform_bounds(&[FEATURE_DIM, SEQ_LEN], 1.0);
    let pros_ibp_output = pros_graph
        .propagate_ibp(&pros_input)
        .expect("IBP propagation");
    let pros_stage = stage_from_propagation(
        "prosody_ibp",
        &pros_input,
        &pros_ibp_output,
        &nn_verify::PropMethod::Ibp,
    );
    assert!(!pros_stage.is_sound, "IBP stage should not be sound");

    // CROWN (sound) for decoder.
    let (dec_def, dec_bindings) = build_decoder_stage();
    let dec_input = uniform_bounds(&[HIDDEN_CHANNELS, SEQ_LEN], 5.0);
    let (dec_stage, _) = propagate_stage("decoder_crown", &dec_def, &dec_bindings, &dec_input);

    let cert = verify_pipeline(&[pros_stage, dec_stage]).expect("pipeline verification");
    assert!(
        !cert.is_sound,
        "pipeline with IBP stage must not be marked sound"
    );
}

/// Pipeline report includes real bound values from NY.
#[test]
fn test_pipeline_report_has_real_bounds() {
    let (pros_def, pros_bindings) = build_prosody_stage();
    let pros_input = uniform_bounds(&[FEATURE_DIM, SEQ_LEN], 1.0);
    let (pros_stage, _) =
        propagate_stage("prosody_predictor", &pros_def, &pros_bindings, &pros_input);

    let (dec_def, dec_bindings) = build_decoder_stage();
    let dec_input = uniform_bounds(&[HIDDEN_CHANNELS, SEQ_LEN], 5.0);
    let (dec_stage, _) = propagate_stage("decoder", &dec_def, &dec_bindings, &dec_input);

    let cert = verify_pipeline(&[pros_stage, dec_stage]).expect("pipeline");
    let report = cert.report();

    assert!(
        report.contains("prosody_predictor"),
        "report should contain stage name"
    );
    assert!(
        report.contains("decoder"),
        "report should contain stage name"
    );
    assert!(
        report.contains("Pipeline Verification Report"),
        "report should have header"
    );
    assert!(
        report.contains("End-to-end bounds"),
        "report should have e2e section"
    );
    // CROWN/IBP propagation produces non-trivial bounds for non-trivial graphs.
    // The report must NOT contain all-zero bounds (which would indicate a
    // degenerate graph or propagation failure).
    assert!(
        !report.contains("bounds: [0.0000, 0.0000]"),
        "report should have non-zero bounds from NY, got all-zero"
    );
    eprintln!("{report}");
}

/// Individual text encoder stage produces valid IBP bounds.
#[test]
fn test_text_encoder_stage_ibp_propagation() {
    let (def, bindings) = build_text_encoder_stage();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[EMBED_DIM, SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[FEATURE_DIM, SEQ_LEN]);

    // ReLU output: lower bounds should be >= 0.
    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        lo_min >= 0.0,
        "ReLU output must be non-negative, got lo_min={lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bounds should be finite");
}

/// Three-stage TTS pipeline certificate from real NY CROWN propagation.
///
/// Pipeline: text_encoder (Conv1d→ReLU) → prosody_predictor (Linear→ReLU→Linear)
///   → decoder (Conv1d→Exp).
///
/// Junction checks:
///   J0: text_encoder output bounds ⊆ prosody verified input range.
///   J1: prosody output bounds ⊆ decoder verified input range.
#[test]
fn test_three_stage_tts_pipeline_real_crown() {
    // Stage 1: Text encoder with CROWN.
    let (enc_def, enc_bindings) = build_text_encoder_stage();
    let enc_input = uniform_bounds(&[EMBED_DIM, SEQ_LEN], 1.0);
    let (enc_stage, enc_output) =
        propagate_stage("text_encoder", &enc_def, &enc_bindings, &enc_input);

    // Stage 2: Prosody predictor with input range encompassing encoder output + margin.
    let (pros_def, pros_bindings) = build_prosody_stage();
    let pros_range = range_with_margin(&enc_output, 1.0);
    let pros_input = uniform_bounds(&[FEATURE_DIM, SEQ_LEN], pros_range);
    let (pros_stage, pros_output) =
        propagate_stage("prosody_predictor", &pros_def, &pros_bindings, &pros_input);

    // Stage 3: Decoder with input range encompassing prosody output + margin.
    let (dec_def, dec_bindings) = build_decoder_stage();
    let dec_range = range_with_margin(&pros_output, 1.0);
    let dec_input = uniform_bounds(&[HIDDEN_CHANNELS, SEQ_LEN], dec_range);
    let (dec_stage, _) = propagate_stage("decoder", &dec_def, &dec_bindings, &dec_input);

    // Compose full 3-stage pipeline and verify.
    let cert = verify_pipeline(&[enc_stage, pros_stage, dec_stage]).expect("pipeline verification");
    eprintln!("{}", cert.report());

    // Structure assertions.
    assert_eq!(cert.stages.len(), 3);
    assert_eq!(cert.junctions.len(), 2);
    assert_eq!(cert.stages[0].name, "text_encoder");
    assert_eq!(cert.stages[1].name, "prosody_predictor");
    assert_eq!(cert.stages[2].name, "decoder");

    // Both junctions should have compatible shapes.
    assert!(
        cert.junctions[0].shape_compatible,
        "J0 shapes should be compatible (encoder→prosody)"
    );
    assert!(
        cert.junctions[1].shape_compatible,
        "J1 shapes should be compatible (prosody→decoder)"
    );

    // End-to-end bounds should be non-empty.
    assert!(!cert.e2e_input_lower.is_empty());
    assert!(!cert.e2e_output_lower.is_empty());

    // Decoder output (after exp) should have positive lower bounds.
    let output_lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        output_lo_min > 0.0,
        "exp output must be positive, got {output_lo_min}"
    );

    assert!(
        cert.is_valid,
        "Three-stage pipeline must be valid — each stage's output ⊆ next stage's input (by construction via range_with_margin)"
    );
}

/// Three-stage pipeline with mixed CROWN/IBP retains soundness tracking.
#[test]
fn test_three_stage_mixed_soundness() {
    // IBP (not sound) for text encoder.
    let (enc_def, enc_bindings) = build_text_encoder_stage();
    let enc_graph = tensor_kernel_to_graph(&enc_def, &enc_bindings).expect("encoder graph");
    let enc_input = uniform_bounds(&[EMBED_DIM, SEQ_LEN], 1.0);
    let enc_ibp_output = enc_graph
        .propagate_ibp(&enc_input)
        .expect("IBP propagation");
    let enc_stage = stage_from_propagation(
        "text_encoder_ibp",
        &enc_input,
        &enc_ibp_output,
        &nn_verify::PropMethod::Ibp,
    );
    assert!(!enc_stage.is_sound, "IBP stage should not be sound");

    // CROWN (sound) for prosody.
    let (pros_def, pros_bindings) = build_prosody_stage();
    let pros_input = uniform_bounds(&[FEATURE_DIM, SEQ_LEN], 5.0);
    let (pros_stage, _) = propagate_stage("prosody_crown", &pros_def, &pros_bindings, &pros_input);

    // CROWN (sound) for decoder.
    let (dec_def, dec_bindings) = build_decoder_stage();
    let dec_input = uniform_bounds(&[HIDDEN_CHANNELS, SEQ_LEN], 5.0);
    let (dec_stage, _) = propagate_stage("decoder_crown", &dec_def, &dec_bindings, &dec_input);

    let cert = verify_pipeline(&[enc_stage, pros_stage, dec_stage]).expect("pipeline verification");
    assert!(
        !cert.is_sound,
        "pipeline with any IBP stage must not be marked sound"
    );
    assert_eq!(cert.stages.len(), 3);
}
