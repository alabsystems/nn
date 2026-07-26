// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NY BoundedTensor bridge tests for pipeline composition.

use super::*;

/// Create a BoundedTensor with uniform bounds.
fn uniform_bt(shape: &[usize], lo: f32, hi: f32) -> nn_verify::BoundedTensor {
    use ndarray::{ArrayD, IxDyn};
    let lower = ArrayD::from_elem(IxDyn(shape), lo);
    let upper = ArrayD::from_elem(IxDyn(shape), hi);
    nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds")
}

#[test]
fn test_stage_from_bounds_extracts_correctly() {
    let input_bt = uniform_bt(&[8, 4], -1.0, 1.0);
    let output_bt = uniform_bt(&[8, 4], -0.5, 0.5);

    let stage = stage_from_bounds("test_encoder", &input_bt, &output_bt, "CROWN", true);

    assert_eq!(stage.name, "test_encoder");
    assert_eq!(stage.input_shape, vec![8, 4]);
    assert_eq!(stage.output_shape, vec![8, 4]);
    assert_eq!(stage.input_lower.len(), 32);
    assert_eq!(stage.output_upper.len(), 32);
    assert!((stage.input_lower[0] - (-1.0)).abs() < 1e-6);
    assert!((stage.input_upper[0] - 1.0).abs() < 1e-6);
    assert!((stage.output_lower[0] - (-0.5)).abs() < 1e-6);
    assert!((stage.output_upper[0] - 0.5).abs() < 1e-6);
    assert_eq!(stage.method, "CROWN");
    assert!(stage.is_sound);
}

#[test]
fn test_stage_from_bounds_f32_to_f64_precision() {
    // Verify that f32 → f64 conversion is exact for representable values.
    let input_bt = uniform_bt(&[2], -0.25, 0.75);
    let output_bt = uniform_bt(&[2], 0.0, 1.0);

    let stage = stage_from_bounds("precision_test", &input_bt, &output_bt, "IBP", false);

    assert_eq!(stage.input_lower[0], f64::from(-0.25_f32));
    assert_eq!(stage.input_upper[0], f64::from(0.75_f32));
    assert_eq!(stage.output_lower[0], 0.0);
    assert_eq!(stage.output_upper[0], 1.0);
    assert!(!stage.is_sound); // IBP is not sound.
}

#[test]
fn test_stage_from_propagation_crown() {
    let input_bt = uniform_bt(&[4], -1.0, 1.0);
    let output_bt = uniform_bt(&[4], -0.8, 0.8);

    let stage = stage_from_propagation(
        "crown_stage",
        &input_bt,
        &output_bt,
        &nn_verify::PropMethod::Crown,
    );

    assert_eq!(stage.method, "CROWN");
    assert!(stage.is_sound);
}

#[test]
fn test_stage_from_propagation_ibp() {
    let input_bt = uniform_bt(&[4], -1.0, 1.0);
    let output_bt = uniform_bt(&[4], -2.0, 2.0);

    let stage = stage_from_propagation(
        "ibp_stage",
        &input_bt,
        &output_bt,
        &nn_verify::PropMethod::Ibp,
    );

    assert_eq!(stage.method, "IBP");
    assert!(!stage.is_sound);
}

#[test]
fn test_stage_from_propagation_alpha_crown() {
    let input_bt = uniform_bt(&[4], -1.0, 1.0);
    let output_bt = uniform_bt(&[4], -0.5, 0.5);

    let stage = stage_from_propagation(
        "alpha_stage",
        &input_bt,
        &output_bt,
        &nn_verify::PropMethod::AlphaCrown,
    );

    assert_eq!(stage.method, "AlphaCrown");
    assert!(stage.is_sound);
}

#[test]
fn test_stage_from_propagation_beta_crown() {
    let input_bt = uniform_bt(&[4], -1.0, 1.0);
    let output_bt = uniform_bt(&[4], -0.25, 0.25);

    let stage = stage_from_propagation(
        "beta_stage",
        &input_bt,
        &output_bt,
        &nn_verify::PropMethod::BetaCrown,
    );

    assert_eq!(stage.method, "BetaCrown");
    assert!(stage.is_sound);
}

#[test]
fn test_pipeline_from_bounded_tensors() {
    // Create a 2-stage pipeline using BoundedTensor bridge.
    // Stage A: encoder with output [-1, 1]
    // Stage B: decoder with input [-2, 2] (compatible: [-1,1] ⊆ [-2,2])
    let enc_in = uniform_bt(&[64], -1.0, 1.0);
    let enc_out = uniform_bt(&[128], -1.0, 1.0);
    let dec_in = uniform_bt(&[128], -2.0, 2.0);
    let dec_out = uniform_bt(&[256], -1.0, 1.0);

    let encoder = stage_from_bounds("encoder", &enc_in, &enc_out, "CROWN", true);
    let decoder = stage_from_bounds("decoder", &dec_in, &dec_out, "CROWN", true);

    let cert = verify_pipeline(&[encoder, decoder]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.stages.len(), 2);
    assert_eq!(cert.junctions[0].from_stage, "encoder");
    assert_eq!(cert.junctions[0].to_stage, "decoder");
    assert!(cert.junctions[0].bounds_contained);
}

#[test]
fn test_pipeline_from_bounded_tensors_incompatible() {
    // Encoder output [0, 5] does NOT fit in decoder input [-1, 3].
    // Upper violation: 5 > 3 → violation = 2.0.
    let enc_in = uniform_bt(&[4], -1.0, 1.0);
    let enc_out = uniform_bt(&[4], 0.0, 5.0);
    let dec_in = uniform_bt(&[4], -1.0, 3.0);
    let dec_out = uniform_bt(&[4], -1.0, 1.0);

    let encoder = stage_from_bounds("encoder", &enc_in, &enc_out, "CROWN", true);
    let decoder = stage_from_bounds("decoder", &dec_in, &dec_out, "CROWN", true);

    let cert = verify_pipeline(&[encoder, decoder]).expect("valid pipeline");
    assert!(!cert.is_valid);
    assert!((cert.junctions[0].max_violation - 2.0).abs() < 1e-6);
}

#[test]
fn test_three_stage_tts_pipeline_from_bounded_tensors() {
    // Simulate Kokoro TTS pipeline: text_encoder → prosody → decoder.
    let txt_in = uniform_bt(&[1, 64], -1.0, 1.0);
    let txt_out = uniform_bt(&[1, 512, 8], -5.0, 5.0);
    let pros_in = uniform_bt(&[1, 512, 8], -10.0, 10.0);
    let pros_out = uniform_bt(&[1, 256, 16], -3.0, 3.0);
    let dec_in = uniform_bt(&[1, 256, 16], -5.0, 5.0);
    let dec_out = uniform_bt(&[1, 1, 4096], -1.0, 1.0);

    let text_encoder = stage_from_propagation(
        "kokoro_text_encoder",
        &txt_in,
        &txt_out,
        &nn_verify::PropMethod::Crown,
    );
    let prosody = stage_from_propagation(
        "kokoro_prosody",
        &pros_in,
        &pros_out,
        &nn_verify::PropMethod::Crown,
    );
    let decoder = stage_from_propagation(
        "kokoro_decoder",
        &dec_in,
        &dec_out,
        &nn_verify::PropMethod::Crown,
    );

    let cert = verify_pipeline(&[text_encoder, prosody, decoder]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.stages.len(), 3);
    assert_eq!(cert.junctions.len(), 2);

    // E2E: input is text embeddings, output is PCM audio.
    assert_eq!(cert.e2e_input_lower.len(), 64);
    assert_eq!(cert.e2e_output_lower.len(), 4096);
    assert!((cert.e2e_output_lower[0] - (-1.0)).abs() < 1e-6);
    assert!((cert.e2e_output_upper[0] - 1.0).abs() < 1e-6);

    // Report should include all stage names.
    let report = cert.report();
    assert!(report.contains("kokoro_text_encoder"));
    assert!(report.contains("kokoro_prosody"));
    assert!(report.contains("kokoro_decoder"));
}

#[test]
fn test_pipeline_report_preserves_alpha_and_beta_crown_methods() {
    let alpha_in = uniform_bt(&[4], -1.0, 1.0);
    let alpha_out = uniform_bt(&[4], -0.5, 0.5);
    let beta_in = uniform_bt(&[4], -0.75, 0.75);
    let beta_out = uniform_bt(&[4], -0.25, 0.25);

    let alpha = stage_from_propagation(
        "alpha_stage",
        &alpha_in,
        &alpha_out,
        &nn_verify::PropMethod::AlphaCrown,
    );
    let beta = stage_from_propagation(
        "beta_stage",
        &beta_in,
        &beta_out,
        &nn_verify::PropMethod::BetaCrown,
    );

    let cert = verify_pipeline(&[alpha, beta]).expect("valid pipeline");
    let report = cert.report();

    assert!(report.contains("method=AlphaCrown"));
    assert!(report.contains("method=BetaCrown"));
}
