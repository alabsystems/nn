// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for moonshot CROWN property verification.
//!
//! Organized by property group:
//! - `silence`: P1 (non-silence) and P2 (non-clipping)
//! - `streaming`: P6 (streaming safety), bundle tests, D=192
//! - `temporal`: P5 (temporal boundedness), bundle with timing, D=192
//! - `speaker`: P4 (speaker consistency), unified 6-property bundle, D=192

use super::*;
use crate::pipeline::{PipelineCertificate, VerifiedStage};

// -- Shared test helper ---------------------------------------------------

fn bounded_pipeline(
    out_lower: Vec<f64>,
    out_upper: Vec<f64>,
    is_sound: bool,
) -> PipelineCertificate {
    let dim = out_lower.len();
    let stages = vec![
        VerifiedStage {
            name: "encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.5; dim],
            output_upper: vec![0.5; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound,
        },
        VerifiedStage {
            name: "decoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: out_lower.clone(),
            output_upper: out_upper.clone(),
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound,
        },
    ];

    // Manually construct certificate with valid junctions.
    PipelineCertificate {
        e2e_input_lower: vec![-1.0; dim],
        e2e_input_upper: vec![1.0; dim],
        e2e_output_lower: out_lower,
        e2e_output_upper: out_upper,
        junctions: vec![crate::pipeline::JunctionResult {
            junction_index: 0,
            from_stage: "encoder".to_string(),
            to_stage: "decoder".to_string(),
            shape_compatible: true,
            bounds_contained: true,
            max_violation: 0.0,
            violation_count: 0,
        }],
        stages,
        is_valid: true,
        is_sound,
    }
}

// -- P3 + bundle basics + verify_moonshot + display -----------------------

#[test]
fn test_intelligibility_proxy_finite_bounds() {
    let cert = bounded_pipeline(vec![-0.3; 8], vec![0.3; 8], true);
    // Tight threshold: output range = 0.6, input range = 2.0, ratio = 0.3 < 1.0
    let result = check_intelligibility_proxy(&cert, 1.0);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(result.bound_value < 1.0);
}

#[test]
fn test_intelligibility_proxy_vacuous_bounds() {
    // Extremely wide bounds indicate IBP blowup.
    let cert = bounded_pipeline(vec![-1e6; 8], vec![1e6; 8], true);
    // Even generous threshold rejects 1e6 ratio.
    let result = check_intelligibility_proxy(&cert, 10.0);
    // Output range = 2e6, input range = 2.0, ratio = 1e6 > 10.0
    assert!(!result.proven);
}

#[test]
fn test_bundle_all_proven() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    assert_eq!(bundle.results.len(), 4);
    assert!(bundle.all_proven);
    assert_eq!(bundle.verification_dim, 64);
}

#[test]
fn test_bundle_partial_failure() {
    // Clipping bounds exceed [-1,1] but non-silence passes.
    let cert = bounded_pipeline(vec![-1.5; 8], vec![1.5; 8], true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    assert!(!bundle.all_proven);
    // Non-silence should pass, non-clipping should fail.
    assert!(bundle.results[0].proven); // non-silence
    assert!(!bundle.results[1].proven); // non-clipping
                                        // Streaming: range=3.0, step=1/239≈0.00418, bound≈0.013 < 0.3 → passes.
    assert!(bundle.results[3].proven); // streaming
}

#[test]
fn test_bundle_display() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let bundle = verify_properties_from_pipeline(&cert, 64);
    let s = format!("{bundle}");
    assert!(s.contains("Moonshot CROWN Bundle (D=64)"));
    assert!(s.contains("P1:"));
    assert!(s.contains("P2:"));
    assert!(s.contains("P3:"));
    assert!(s.contains("P6:"));
    assert!(s.contains("4/4 proven"));
}

#[test]
fn test_verify_moonshot_from_stages() {
    let dim = 8;
    let stages = vec![
        VerifiedStage {
            name: "layer_0".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.5; dim],
            output_upper: vec![0.5; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "layer_1".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.3; dim],
            output_upper: vec![0.3; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ];

    let bundle = verify_moonshot_from_stages(&stages, dim).unwrap();
    assert!(bundle.all_proven);
    assert_eq!(bundle.verification_dim, 8);
}

#[test]
fn test_verify_moonshot_insufficient_stages() {
    let stages = vec![VerifiedStage {
        name: "only_one".to_string(),
        input_lower: vec![-1.0; 4],
        input_upper: vec![1.0; 4],
        output_lower: vec![-0.5; 4],
        output_upper: vec![0.5; 4],
        input_shape: vec![1, 4],
        output_shape: vec![1, 4],
        method: "CROWN".to_string(),
        is_sound: true,
    }];

    let result = verify_moonshot_from_stages(&stages, 4);
    assert!(result.is_err());
}

// -- Display tests --------------------------------------------------------

#[test]
fn test_property_result_display() {
    let cert = bounded_pipeline(vec![-0.5; 8], vec![0.5; 8], true);
    let result = check_non_silence(&cert, 0.01);
    let s = format!("{result}");
    assert!(s.contains("P1:"));
    assert!(s.contains("CROWN_PROVEN"));
    assert!(s.contains("Non-silent"));
}

#[test]
fn test_temporal_result_display() {
    let (_cert, timing_cert) =
        temporal::timing_certificate(8, vec![-0.5; 8], vec![0.5; 8], true, 50_000.0, 100_000.0);
    let result = check_temporal_boundedness(&timing_cert);
    let s = format!("{result}");
    assert!(s.contains("P5:"));
    assert!(s.contains("CROWN_PROVEN"));
    assert!(s.contains("Temporally bounded"));
    assert!(s.contains("50000.0"));
    assert!(s.contains("100000.0"));
}

// -- NaN propagation tests (#1911) ----------------------------------------

#[test]
fn test_check_non_silence_nan_in_lower_bounds() {
    // AC2: NaN in e2e_output_lower must propagate through max_abs combining.
    let mut lower = vec![0.1; 8];
    lower[3] = f64::NAN;
    let cert = bounded_pipeline(lower, vec![0.5; 8], true);
    let result = check_non_silence(&cert, 0.01);
    // NaN propagates → max_abs is NaN → NaN > threshold is false → not proven.
    assert!(!result.proven, "NaN in lower bounds must prevent proof");
    assert!(result.bound_value.is_nan(), "bound_value must be NaN");
}

#[test]
fn test_check_non_silence_nan_in_upper_bounds() {
    let mut upper = vec![0.5; 8];
    upper[5] = f64::NAN;
    let cert = bounded_pipeline(vec![-0.3; 8], upper, true);
    let result = check_non_silence(&cert, 0.01);
    assert!(!result.proven, "NaN in upper bounds must prevent proof");
    assert!(result.bound_value.is_nan(), "bound_value must be NaN");
}

#[test]
fn test_check_non_clipping_nan_propagates() {
    // NaN in output bounds must propagate through fold_max/fold_min.
    let mut upper = vec![0.5; 8];
    upper[2] = f64::NAN;
    let cert = bounded_pipeline(vec![-0.3; 8], upper, true);
    let result = check_non_clipping(&cert);
    // finite_bounds check catches NaN → not proven.
    assert!(!result.proven, "NaN in bounds must prevent clipping proof");
}

// -- Submodules -----------------------------------------------------------

#[path = "moonshot_crown_tests_silence.rs"]
mod silence;

#[path = "moonshot_crown_tests_streaming.rs"]
mod streaming;

#[path = "moonshot_crown_tests_temporal.rs"]
mod temporal;

#[path = "moonshot_crown_tests_speaker.rs"]
mod speaker;

#[path = "moonshot_crown_tests_speaker_composed.rs"]
mod speaker_composed;

#[path = "moonshot_crown_tests_temporal_composed.rs"]
mod temporal_composed;

#[path = "moonshot_crown_tests_full_certificate.rs"]
mod full_certificate;

#[path = "moonshot_crown_tests_kokoro_helpers.rs"]
mod kokoro_helpers;

#[path = "moonshot_crown_tests_kokoro_timing.rs"]
mod kokoro_timing;

#[path = "moonshot_crown_tests_vad_helpers.rs"]
mod vad_helpers;

#[path = "moonshot_crown_tests_vad_timing.rs"]
mod vad_timing;

#[path = "moonshot_crown_tests_memory.rs"]
mod memory;

#[path = "moonshot_crown_tests_implementation.rs"]
mod implementation;

#[path = "moonshot_crown_tests_intelligibility.rs"]
mod intelligibility;

#[path = "moonshot_crown_tests_intelligibility_weight.rs"]
mod intelligibility_weight;

#[path = "moonshot_crown_tests_production_dispatch.rs"]
mod production_dispatch;
