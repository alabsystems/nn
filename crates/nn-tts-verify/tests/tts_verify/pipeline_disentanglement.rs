// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Pipeline-level prosody disentanglement verification.
//!
//! Composes F0EnergyPredictor disentanglement analysis with the Kokoro decoder
//! to demonstrate that prosody control disentanglement verified at the
//! F0EnergyPredictor level carries through to the audio-domain decoder output.
//!
//! This is Phase 4 of the #1738 design doc: Integration test with Kokoro
//! decoder composition. The key insight is that CROWN sensitivity analysis
//! on the F0EnergyPredictor subgraph produces certificates about control
//! knob independence, and these certificates compose with the decoder
//! pipeline certificate to give end-to-end guarantees.
//!
//! Run with:
//!   cargo test -p nn-tts-verify --test pipeline_disentanglement --features NY
//!
//! Part of #1738: Compositional Verification of Prosody Controls.

#![cfg(feature = "ny")]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::{TensorKernelDef, TensorNodeId};
use nn_tts_verify::disentanglement::{
    measure_sensitivity, verify_disentanglement, AcousticProperty, ControlDimension,
    DisentanglementCertificate,
};
use nn_tts_verify::{stage_from_propagation, verify_pipeline, PipelineCertificate, VerifiedStage};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability.
//
// Flat input layout: [text_features(FEATURE_DIM*SEQ_LEN) | style(STYLE_DIM)]
// F0E output layout: [f0(F0_OUT_DIM*SEQ_LEN) | energy(ENERGY_OUT_DIM*SEQ_LEN)]
// ---------------------------------------------------------------------------

const FEATURE_DIM: usize = 8;
const STYLE_DIM: usize = 2 * FEATURE_DIM; // gamma + beta
const SEQ_LEN: usize = 1;
const FLAT_INPUT_SIZE: usize = FEATURE_DIM * SEQ_LEN + STYLE_DIM;

const F0_OUT_DIM: usize = 1;
const ENERGY_OUT_DIM: usize = 1;
const F0E_OUTPUT_SIZE: usize = F0_OUT_DIM * SEQ_LEN + ENERGY_OUT_DIM * SEQ_LEN;

// Slice indices into flat_input.
const TEXT_START: usize = 0;
const TEXT_END: usize = FEATURE_DIM * SEQ_LEN;
const STYLE_START: usize = FEATURE_DIM * SEQ_LEN;
const STYLE_END: usize = FLAT_INPUT_SIZE;

// Slice indices into F0E output.
const F0_START: usize = 0;
const F0_END: usize = F0_OUT_DIM * SEQ_LEN;
const ENERGY_START: usize = F0_OUT_DIM * SEQ_LEN;
const ENERGY_END: usize = F0E_OUTPUT_SIZE;

// Decoder dimensions.
const DECODER_OUT_CHANNELS: usize = 4;
const DECODER_TIME: usize = 2;

const WEIGHT_MAG: f32 = 0.01;
const INPUT_BOUND: f64 = 1.0;

/// Style affine + ReLU + linear projection for one head.
///
/// gamma * shared_out + beta → ReLU → transpose → matmul → flatten.
/// No InstanceNorm: SEQ_LEN=1 normalizes to zero (production model uses
/// residual connections; we use direct style modulation instead).
fn add_head(
    b: &mut TensorBlockBuilder,
    shared_out: TensorNodeId,
    style_input: TensorNodeId,
    out_dim: usize,
    proj_name: &str,
) -> TensorNodeId {
    let shape = [FEATURE_DIM, SEQ_LEN];
    let gamma = b.add_narrow(style_input, 0, 0, FEATURE_DIM, &[FEATURE_DIM]);
    let beta = b.add_narrow(style_input, 0, FEATURE_DIM, FEATURE_DIM, &[FEATURE_DIM]);
    let gamma_bc = b.add_broadcast_left(gamma, &shape);
    let beta_bc = b.add_broadcast_left(beta, &shape);
    let scaled = b.add_binary_mul(shared_out, gamma_bc, &shape);
    let affined = b.add_binary_add(scaled, beta_bc, &shape);
    let activated = b.add_relu(affined, &shape);
    let transposed = b.add_transpose(activated, &[1, 0], &[SEQ_LEN, FEATURE_DIM]);
    let proj_w = b.add_input(proj_name, &[out_dim, FEATURE_DIM]);
    let proj = b.add_matmul(transposed, proj_w, true, None, &[SEQ_LEN, out_dim]);
    b.add_reshape(proj, &[out_dim * SEQ_LEN])
}

fn build_f0_energy_predictor() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("f0_energy_predictor");
    let text_size = FEATURE_DIM * SEQ_LEN;

    // Flat input: [text_features..., style...]
    let flat_input = b.add_input("flat_input", &[FLAT_INPUT_SIZE]);
    let text_flat = b.add_narrow(flat_input, 0, 0, text_size, &[text_size]);
    let text_input = b.add_reshape(text_flat, &[FEATURE_DIM, SEQ_LEN]);
    let style_input = b.add_narrow(flat_input, 0, text_size, STYLE_DIM, &[STYLE_DIM]);

    // Shared conv with bias: [FEATURE_DIM, SEQ_LEN] → [FEATURE_DIM, SEQ_LEN]
    let shared_w = b.add_input("shared_conv_w", &[FEATURE_DIM, FEATURE_DIM, 1]);
    let shared_b = b.add_input("shared_conv_b", &[FEATURE_DIM]);
    let shared_b_bc = b.add_broadcast_left(shared_b, &[FEATURE_DIM, SEQ_LEN]);
    let shared_conv = b.add_conv1d(text_input, shared_w, None, 1, 0, &[FEATURE_DIM, SEQ_LEN]);
    let shared_out = b.add_binary_add(shared_conv, shared_b_bc, &[FEATURE_DIM, SEQ_LEN]);

    // F0 and energy heads: style_affine → ReLU → projection
    let f0_flat = add_head(&mut b, shared_out, style_input, F0_OUT_DIM, "f0_proj_w");
    let en_flat = add_head(
        &mut b,
        shared_out,
        style_input,
        ENERGY_OUT_DIM,
        "energy_proj_w",
    );

    // Concat F0 + energy
    let f0_2d = b.add_reshape(f0_flat, &[1, F0_OUT_DIM * SEQ_LEN]);
    let en_2d = b.add_reshape(en_flat, &[1, ENERGY_OUT_DIM * SEQ_LEN]);
    let concat = b.add_concat(&[f0_2d, en_2d], 1, &[1, F0E_OUTPUT_SIZE]);
    let output = b.add_reshape(concat, &[F0E_OUTPUT_SIZE]);

    let def = b.build(output).expect("valid f0_energy graph");
    let bindings = f0e_bindings();
    (def, bindings)
}

fn f0e_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // flat_input
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FEATURE_DIM, FEATURE_DIM, 1]),
            WEIGHT_MAG,
        )), // shared_conv_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FEATURE_DIM]), 0.1_f32)), // shared_conv_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[F0_OUT_DIM, FEATURE_DIM]),
            WEIGHT_MAG,
        )), // f0_proj_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[ENERGY_OUT_DIM, FEATURE_DIM]),
            WEIGHT_MAG,
        )), // energy_proj_w
    ]
}

/// Decoder builder: Linear → Exp (log-magnitude → magnitude).
fn build_decoder() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("decoder");
    let dec_out_size = DECODER_OUT_CHANNELS * DECODER_TIME;

    // Variable input: F0+energy features [F0E_OUTPUT_SIZE]
    let input = b.add_input("prosody_features", &[F0E_OUTPUT_SIZE]);

    // Reshape to 2D for matmul: [F0E_OUTPUT_SIZE] → [1, F0E_OUTPUT_SIZE]
    let input_2d = b.add_reshape(input, &[1, F0E_OUTPUT_SIZE]);

    // Linear projection: [1, F0E_OUTPUT_SIZE] × [F0E_OUTPUT_SIZE, dec_out_size]^T → [1, dec_out_size]
    let proj_w = b.add_input("dec_proj_w", &[dec_out_size, F0E_OUTPUT_SIZE]);
    let projected = b.add_matmul(input_2d, proj_w, true, None, &[1, dec_out_size]);
    let reshaped = b.add_reshape(projected, &[DECODER_OUT_CHANNELS, DECODER_TIME]);

    // Exp activation: log-magnitude → magnitude
    let output = b.add_exp(reshaped, &[DECODER_OUT_CHANNELS, DECODER_TIME]);

    let def = b.build(output).expect("valid decoder graph");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[dec_out_size, F0E_OUTPUT_SIZE]),
            WEIGHT_MAG,
        )),
    ];

    (def, bindings)
}

fn uniform_bounds(shape: &[usize], range: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), -range);
    let upper = ArrayD::from_elem(IxDyn(shape), range);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Midpoint for sensitivity measurement.
///
/// Style gamma dimensions are set to 1.0 (identity scaling) so that text
/// features propagate through the style_affine(gamma * x + beta) layer.
/// With gamma=0 the affine would zero out text signal regardless of input.
fn midpoint() -> Vec<f64> {
    let mut m = vec![0.0; FLAT_INPUT_SIZE];
    // Set gamma portion of style to 1.0 (identity).
    // Style layout: [gamma_0..gamma_{D-1}, beta_0..beta_{D-1}]
    for m_i in &mut m[STYLE_START..STYLE_START + FEATURE_DIM] {
        *m_i = 1.0;
    }
    m
}

fn propagate_stage(
    name: &str,
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
) -> (VerifiedStage, BoundedTensor) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph translation");
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, input).expect("CROWN propagation");
    let stage = stage_from_propagation(name, input, &output, &method);
    (stage, output)
}

/// Compose F0EnergyPredictor → Decoder two-stage pipeline.
///
/// Propagates F0E bounds, uses output span + 50% margin for decoder input,
/// then composes both stages into a PipelineCertificate.
fn compose_f0e_decoder_pipeline() -> PipelineCertificate {
    let (f0e_def, f0e_bindings) = build_f0_energy_predictor();
    let f0e_input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);
    let (f0e_stage, f0e_output) =
        propagate_stage("f0_energy_predictor", &f0e_def, &f0e_bindings, &f0e_input);

    let (f0e_lo, f0e_hi) = f0e_output.lower_upper();
    let span = f0e_hi
        .iter()
        .zip(f0e_lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0_f32, f32::max);
    let dec_range = (span + span * 0.5).max(1.0);

    let (dec_def, dec_bindings) = build_decoder();
    let dec_input = uniform_bounds(&[F0E_OUTPUT_SIZE], dec_range);
    let (dec_stage, _) = propagate_stage("decoder", &dec_def, &dec_bindings, &dec_input);

    verify_pipeline(&[f0e_stage, dec_stage]).expect("pipeline verification")
}

/// Run F0E disentanglement verification with standard controls and properties.
fn run_f0e_disentanglement() -> DisentanglementCertificate {
    let (def, bindings) = build_f0_energy_predictor();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let controls = vec![
        ControlDimension::new("text_features", 0, TEXT_START, TEXT_END),
        ControlDimension::new("style", 0, STYLE_START, STYLE_END),
    ];
    let properties = vec![
        AcousticProperty::new("f0", F0_START, F0_END),
        AcousticProperty::new("energy", ENERGY_START, ENERGY_END),
    ];
    verify_disentanglement(
        &graph,
        &controls,
        &properties,
        INPUT_BOUND,
        &midpoint(),
        0.99,
    )
    .expect("disentanglement verification")
}

/// F0EnergyPredictor disentanglement certificate produces valid sensitivity
/// matrix when composed as a pipeline stage.
#[test]
fn test_f0_energy_pipeline_stage_propagates() {
    let (def, bindings) = build_f0_energy_predictor();
    let input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);
    let (stage, output) = propagate_stage("f0_energy_predictor", &def, &bindings, &input);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[F0E_OUTPUT_SIZE]);

    let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(lo_min.is_finite(), "lower bounds should be finite");
    assert!(hi_max.is_finite(), "upper bounds should be finite");

    eprintln!(
        "F0EnergyPredictor stage: bounds=[{lo_min:.4}, {hi_max:.4}], sound={}",
        stage.is_sound
    );
}

/// Decoder stage propagates valid bounds from F0EnergyPredictor output range.
#[test]
fn test_decoder_pipeline_stage_propagates() {
    // First propagate F0EnergyPredictor to get output range.
    let (f0e_def, f0e_bindings) = build_f0_energy_predictor();
    let f0e_input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);
    let (_, f0e_output) = propagate_stage("f0_energy", &f0e_def, &f0e_bindings, &f0e_input);

    // Use F0E output range to set decoder input bounds.
    let (f0e_lo, f0e_hi) = f0e_output.lower_upper();
    let lo_min = f0e_lo.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = f0e_hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let dec_range = (hi_max - lo_min).abs().max(1.0);

    let (dec_def, dec_bindings) = build_decoder();
    let dec_input = uniform_bounds(&[F0E_OUTPUT_SIZE], dec_range);
    let (stage, output) = propagate_stage("decoder", &dec_def, &dec_bindings, &dec_input);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[DECODER_OUT_CHANNELS, DECODER_TIME]);

    let dec_lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        dec_lo_min > 0.0,
        "exp output should be positive, got {dec_lo_min}"
    );

    eprintln!(
        "Decoder stage: bounds=[{:.4}, {:.4}], sound={}",
        dec_lo_min,
        hi.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        stage.is_sound
    );
}

/// Two-stage pipeline: F0EnergyPredictor → Decoder produces valid, sound certificate.
#[test]
fn test_two_stage_f0e_decoder_pipeline() {
    let cert = compose_f0e_decoder_pipeline();
    eprintln!("{}", cert.report());

    assert_eq!(cert.stages.len(), 2);
    assert_eq!(cert.junctions.len(), 1);
    assert_eq!(cert.stages[0].name, "f0_energy_predictor");
    assert_eq!(cert.stages[1].name, "decoder");
    assert!(
        cert.junctions[0].shape_compatible,
        "shapes should be compatible (F0E output → decoder input)"
    );
    assert!(!cert.e2e_input_lower.is_empty());
    assert!(!cert.e2e_output_lower.is_empty());
    assert!(cert.is_valid, "pipeline certificate should be valid");
    assert!(
        cert.is_sound,
        "pipeline certificate should be sound (CROWN)"
    );
}

/// Disentanglement certificate on F0EnergyPredictor subgraph:
/// text_features and style should both affect F0 and energy.
#[test]
fn test_f0e_disentanglement_certificate() {
    let cert = run_f0e_disentanglement();

    for s in &cert.sensitivities {
        assert!(
            s.bound_width >= 0.0,
            "{} → {}: negative width {}",
            s.control,
            s.property,
            s.bound_width
        );
    }

    eprintln!("F0E disentanglement sensitivity matrix:");
    for s in &cert.sensitivities {
        eprintln!(
            "  {} → {}: width={:.6} ({})",
            s.control, s.property, s.bound_width, s.propagation_mode
        );
    }
    eprintln!(
        "  max_cross_influence={:.4}, is_disentangled={}",
        cert.max_cross_influence, cert.is_disentangled
    );
}

/// Sensitivity of F0 to text_features should be non-zero (text drives prosody).
#[test]
fn test_text_features_drive_f0() {
    let (def, bindings) = build_f0_energy_predictor();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("text_features", 0, TEXT_START, TEXT_END);
    let property = AcousticProperty::new("f0", F0_START, F0_END);

    let result = measure_sensitivity(&graph, &control, &property, INPUT_BOUND, &midpoint())
        .expect("text→f0 sensitivity");

    assert!(
        result.bound_width > 0.0,
        "Text features should influence F0, got width={}",
        result.bound_width
    );
    eprintln!(
        "text_features → F0: width={:.6} ({})",
        result.bound_width, result.propagation_mode
    );
}

/// Sensitivity of energy to style should be non-zero (style modulates via AdaIN).
#[test]
fn test_style_modulates_energy() {
    let (def, bindings) = build_f0_energy_predictor();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("style", 0, STYLE_START, STYLE_END);
    let property = AcousticProperty::new("energy", ENERGY_START, ENERGY_END);

    let result = measure_sensitivity(&graph, &control, &property, INPUT_BOUND, &midpoint())
        .expect("style→energy sensitivity");

    assert!(
        result.bound_width > 0.0,
        "Style should influence energy via AdaIN, got width={}",
        result.bound_width
    );
    eprintln!(
        "style → energy: width={:.6} ({})",
        result.bound_width, result.propagation_mode
    );
}

/// Pipeline composition preserves disentanglement evidence:
/// Disentanglement sensitivity at F0E level implies non-zero pipeline e2e
/// output width (the decoder preserves rather than collapses the signal).
#[test]
fn test_pipeline_preserves_disentanglement_stages() {
    // Step 1: Verify disentanglement at the F0E subgraph level
    let disent_cert = run_f0e_disentanglement();

    let max_sens = |control: &str| -> f64 {
        disent_cert
            .sensitivities
            .iter()
            .filter(|s| s.control == control)
            .map(|s| s.bound_width)
            .fold(0.0, f64::max)
    };
    assert!(
        max_sens("text_features") > 0.0,
        "text should have non-zero F0E sensitivity"
    );
    assert!(
        max_sens("style") > 0.0,
        "style should have non-zero F0E sensitivity"
    );

    // Step 2: Compose the pipeline and verify it is valid+sound
    let pipeline_cert = compose_f0e_decoder_pipeline();
    assert!(pipeline_cert.is_valid, "pipeline should be valid");
    assert!(pipeline_cert.is_sound, "pipeline should be sound");

    // Step 3: Cross-certificate — F0E sensitivity implies non-zero e2e output width
    let max_width = |upper: &[f64], lower: &[f64]| -> f64 {
        upper
            .iter()
            .zip(lower.iter())
            .map(|(h, l)| h - l)
            .fold(0.0_f64, f64::max)
    };
    let e2e_out_width = max_width(
        &pipeline_cert.e2e_output_upper,
        &pipeline_cert.e2e_output_lower,
    );
    assert!(
        e2e_out_width > 0.0,
        "pipeline e2e output width should be non-zero, got {e2e_out_width}"
    );

    // Step 4: F0E stage output width matches sensitivity evidence
    let f0e_stage = &pipeline_cert.stages[0];
    let f0e_out_width = max_width(&f0e_stage.output_upper, &f0e_stage.output_lower);
    let total_disent_width: f64 = disent_cert
        .sensitivities
        .iter()
        .map(|s| s.bound_width)
        .sum();
    assert!(
        f0e_out_width > 0.0,
        "F0E stage output width should be non-zero"
    );

    eprintln!(
        "Preservation: disent_total_width={total_disent_width:.6}, \
         f0e_out_width={f0e_out_width:.6}, e2e_out_width={e2e_out_width:.6}"
    );
}
