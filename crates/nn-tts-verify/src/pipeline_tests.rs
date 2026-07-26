// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Helper: create a verified stage with uniform bounds across all elements.
pub(super) fn make_stage(
    name: &str,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    input_range: (f64, f64),
    output_range: (f64, f64),
    method: &str,
    is_sound: bool,
) -> VerifiedStage {
    let in_elements: usize = input_shape.iter().product();
    let out_elements: usize = output_shape.iter().product();

    VerifiedStage {
        name: name.to_string(),
        input_lower: vec![input_range.0; in_elements],
        input_upper: vec![input_range.1; in_elements],
        output_lower: vec![output_range.0; out_elements],
        output_upper: vec![output_range.1; out_elements],
        input_shape,
        output_shape,
        method: method.to_string(),
        is_sound,
    }
}

#[test]
fn test_two_stage_pipeline_compatible() {
    // Stage A output [-1, 1] ⊆ Stage B input [-2, 2] → compatible.
    let stage_a = make_stage(
        "encoder",
        vec![512, 8],
        vec![512, 8],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "decoder",
        vec![512, 8],
        vec![256, 16],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.junctions.len(), 1);
    assert!(cert.junctions[0].bounds_contained);
    assert!(cert.junctions[0].shape_compatible);
    assert_eq!(cert.junctions[0].max_violation, 0.0);
    assert_eq!(cert.junctions[0].violation_count, 0);
}

#[test]
fn test_two_stage_pipeline_incompatible_bounds() {
    // Stage A output [-1, 3] but Stage B input expects [-2, 2].
    // Upper bound violated: 3 > 2, violation = 1.0.
    let stage_a = make_stage(
        "encoder",
        vec![4],
        vec![4],
        (-1.0, 1.0),
        (-1.0, 3.0),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "decoder",
        vec![4],
        vec![4],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    assert!(!cert.is_valid);
    assert!(!cert.is_sound); // Invalid → not sound.
    let j = &cert.junctions[0];
    assert!(!j.bounds_contained);
    assert!((j.max_violation - 1.0).abs() < 1e-10);
    assert_eq!(j.violation_count, 4); // All 4 elements have upper violation.
}

#[test]
fn test_two_stage_pipeline_incompatible_shape() {
    // Stage A output shape [512, 8] = 4096 elements.
    // Stage B input shape [256, 16] = 4096 elements → compatible.
    // But [512, 8] vs [256, 4] = 4096 vs 1024 → incompatible.
    let stage_a = make_stage(
        "encoder",
        vec![512, 8],
        vec![512, 8],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "decoder",
        vec![256, 4],
        vec![128, 2],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    assert!(!cert.is_valid);
    assert!(!cert.junctions[0].shape_compatible);
}

#[test]
fn test_three_stage_pipeline_end_to_end() {
    // Three compatible stages: encoder → predictor → decoder.
    let encoder = make_stage(
        "text_encoder",
        vec![64],
        vec![128],
        (-1.0, 1.0),
        (-5.0, 5.0),
        "CROWN",
        true,
    );
    let predictor = make_stage(
        "prosody_predictor",
        vec![128],
        vec![256],
        (-10.0, 10.0),
        (-3.0, 3.0),
        "CROWN",
        true,
    );
    let decoder = make_stage(
        "kokoro_decoder",
        vec![256],
        vec![512],
        (-5.0, 5.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[encoder, predictor, decoder]).expect("valid pipeline");
    assert!(cert.is_valid);
    assert!(cert.is_sound);
    assert_eq!(cert.stages.len(), 3);
    assert_eq!(cert.junctions.len(), 2);

    // E2e bounds = first stage input, last stage output.
    assert_eq!(cert.e2e_input_lower.len(), 64);
    assert_eq!(cert.e2e_output_lower.len(), 512);
    assert!((cert.e2e_input_lower[0] - (-1.0)).abs() < 1e-10);
    assert!((cert.e2e_output_upper[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_pipeline_soundness_propagation() {
    // One stage is IBP (not sound) — pipeline should be not sound.
    let stage_a = make_stage(
        "encoder",
        vec![8],
        vec![8],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "decoder",
        vec![8],
        vec![8],
        (-2.0, 2.0),
        (-1.0, 1.0),
        "IBP",
        false,
    );

    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    assert!(cert.is_valid); // Bounds are compatible.
    assert!(!cert.is_sound); // IBP stage is not sound.
}

#[test]
fn test_pipeline_single_stage_error() {
    let stage = make_stage(
        "only_one",
        vec![8],
        vec![8],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "CROWN",
        true,
    );

    let result = verify_pipeline(&[stage]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::InsufficientStages { count: 1 }),
        "expected InsufficientStages, got: {err}",
    );
}

#[test]
fn test_pipeline_empty_error() {
    let result = verify_pipeline(&[]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TtsVerifyError::InsufficientStages { count: 0 }
    ));
}

#[test]
fn test_pipeline_report_format() {
    let stage_a = make_stage(
        "encoder",
        vec![4],
        vec![4],
        (-1.0, 1.0),
        (-0.5, 0.5),
        "CROWN",
        true,
    );
    let stage_b = make_stage(
        "decoder",
        vec![4],
        vec![4],
        (-1.0, 1.0),
        (-1.0, 1.0),
        "alpha-CROWN",
        true,
    );

    let cert = verify_pipeline(&[stage_a, stage_b]).expect("valid pipeline");
    let report = cert.report();

    assert!(report.contains("Pipeline Verification Report"));
    assert!(report.contains("Stages: 2"));
    assert!(report.contains("Valid: true"));
    assert!(report.contains("Sound: true"));
    assert!(report.contains("encoder"));
    assert!(report.contains("decoder"));
    assert!(report.contains("Junction 0"));
    assert!(report.contains("Bounds contained: true"));
    assert!(report.contains("End-to-end bounds"));
}

// --- junction and edge-case tests (extracted to pipeline_tests_junction.rs) ---

#[path = "pipeline_tests_junction.rs"]
mod junction;

// --- NY bridge tests (extracted to pipeline_tests_bridge.rs) ---

#[cfg(feature = "ny")]
#[path = "pipeline_tests_bridge.rs"]
mod bridge;

// --- verify_layerwise tests (extracted to pipeline_tests_layerwise.rs) ---

#[cfg(feature = "ny")]
#[path = "pipeline_tests_layerwise.rs"]
mod layerwise;
