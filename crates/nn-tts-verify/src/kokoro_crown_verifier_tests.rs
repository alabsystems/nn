// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `KokoroCrownVerifier`.
//!
//! Part of #3874.

use std::collections::HashMap;
use std::path::Path;

use super::status::{
    extract_best_bounds, StatusEntry, StatusFile, StatusInputBounds, StatusOutputBounds,
};
use super::*;

// ============================================================================
// Helper constructors
// ============================================================================

/// Build a `SegmentBounds` with uniform scalar bounds.
fn make_segment(
    seg: SegmentId,
    lower: f64,
    upper: f64,
    shape: Vec<usize>,
    is_sound: bool,
) -> SegmentBounds {
    let elements: usize = shape.iter().product();
    let in_shape = shape.clone();
    SegmentBounds {
        segment: seg,
        status_key: format!("test_{}", seg.name()),
        method: if is_sound {
            "CROWN".to_string()
        } else {
            "IBP".to_string()
        },
        is_sound,
        proof_strength: if is_sound {
            "sound_crown".to_string()
        } else {
            "vacuous".to_string()
        },
        output_shape: shape,
        output_lower: vec![lower; elements],
        output_upper: vec![upper; elements],
        output_width: upper - lower,
        input_lower: vec![-1.0; elements],
        input_upper: vec![1.0; elements],
        input_shape: in_shape,
    }
}

/// Build a composable segment with explicit input bounds.
fn make_composable_segment(
    seg: SegmentId,
    in_lower: f64,
    in_upper: f64,
    out_lower: f64,
    out_upper: f64,
    shape: Vec<usize>,
    is_sound: bool,
) -> SegmentBounds {
    let elements: usize = shape.iter().product();
    SegmentBounds {
        segment: seg,
        status_key: format!("test_{}", seg.name()),
        method: if is_sound {
            "CROWN".to_string()
        } else {
            "IBP".to_string()
        },
        is_sound,
        proof_strength: if is_sound {
            "sound_crown".to_string()
        } else {
            "vacuous".to_string()
        },
        output_shape: shape.clone(),
        output_lower: vec![out_lower; elements],
        output_upper: vec![out_upper; elements],
        output_width: out_upper - out_lower,
        input_lower: vec![in_lower; elements],
        input_upper: vec![in_upper; elements],
        input_shape: shape,
    }
}

/// Build a full 5-segment verifier with sound, composable bounds.
///
/// Each stage's output is contained within the next stage's input bounds.
fn make_sound_verifier() -> KokoroCrownVerifier {
    // Pipeline: each stage's output fits within the next stage's input.
    // BertEncoder: in [-1,1] -> out [-3,3]
    // TextEncoder: in [-3,3] -> out [-2,2]
    // ProsodyPredictor: in [-2,2] -> out [-1,1]
    // F0EnergyPredictor: in [-1,1] -> out [-1,1] (8 elements)
    // Generator: in [-1,1] -> out [-1,1] (16 elements)
    //
    // Note: shapes must match at junctions. We use the same shape (16) for all.
    let segments = vec![
        make_composable_segment(SegmentId::BertEncoder, -1.0, 1.0, -3.0, 3.0, vec![16], true),
        make_composable_segment(SegmentId::TextEncoder, -3.0, 3.0, -2.0, 2.0, vec![16], true),
        make_composable_segment(
            SegmentId::ProsodyPredictor,
            -2.0,
            2.0,
            -1.0,
            1.0,
            vec![16],
            true,
        ),
        make_composable_segment(
            SegmentId::F0EnergyPredictor,
            -1.0,
            1.0,
            -1.0,
            1.0,
            vec![16],
            true,
        ),
        make_composable_segment(SegmentId::Generator, -1.0, 1.0, -1.0, 1.0, vec![16], true),
    ];
    KokoroCrownVerifier::from_segments(segments, "test-kokoro")
}

fn make_status_entry(
    method: &str,
    soundness_mode: &str,
    output_width: f64,
    lower: f64,
    upper: f64,
    shape: &[usize],
) -> StatusEntry {
    let proof_strength = match soundness_mode {
        "sound" if matches!(method, "CROWN" | "AlphaCrown" | "BetaCrown") => "sound_crown",
        "sound" => "sound_ibp",
        _ => "heuristic",
    };

    StatusEntry {
        status: Some("verified".to_string()),
        method: Some(method.to_string()),
        soundness_mode: Some(soundness_mode.to_string()),
        proof_strength: Some(proof_strength.to_string()),
        output_width: Some(output_width),
        output_bounds: Some(StatusOutputBounds {
            lower: Some(lower),
            upper: Some(upper),
            tensor_lower: None,
            tensor_upper: None,
            shape: Some(shape.to_vec()),
        }),
        input_bounds: Some(StatusInputBounds {
            input_shape: Some(shape.to_vec()),
            input_range: Some(vec![-1.0, 1.0]),
        }),
        stale: false,
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[test]
fn test_segment_id_all_has_5() {
    assert_eq!(SegmentId::ALL.len(), 5);
}

#[test]
fn test_segment_id_names() {
    assert_eq!(SegmentId::BertEncoder.name(), "BertEncoder");
    assert_eq!(SegmentId::Generator.name(), "Generator");
}

#[test]
fn test_segment_bounds_proves_pcm_range() {
    let seg = make_segment(SegmentId::Generator, -1.0, 1.0, vec![16], true);
    assert!(seg.proves_pcm_range());

    let seg_wide = make_segment(SegmentId::Generator, -2.0, 2.0, vec![16], true);
    assert!(!seg_wide.proves_pcm_range());
}

#[test]
fn test_segment_bounds_proves_f0_range() {
    let seg = make_segment(SegmentId::F0EnergyPredictor, -5.0, 800.0, vec![8], true);
    assert!(seg.proves_f0_range());

    let seg_wide = make_segment(SegmentId::F0EnergyPredictor, -100.0, 800.0, vec![8], true);
    assert!(!seg_wide.proves_f0_range());
}

#[test]
fn test_verifier_from_segments() {
    let verifier = make_sound_verifier();
    assert_eq!(verifier.segment_count(), 5);
    assert!(verifier.all_sound());
}

#[test]
fn test_verify_segment_generator_pcm() {
    let verifier = make_sound_verifier();
    let result = verifier.verify_segment(SegmentId::Generator).unwrap();
    assert!(result.property_proven);
    assert!(result.explanation.contains("PCM in [-1.0, 1.0]: true"));
}

#[test]
fn test_verify_segment_f0_range() {
    // Use a verifier where F0 segment output is within J2 contract bounds.
    let segments = vec![
        make_composable_segment(SegmentId::BertEncoder, -1.0, 1.0, -3.0, 3.0, vec![16], true),
        make_composable_segment(SegmentId::TextEncoder, -3.0, 3.0, -2.0, 2.0, vec![16], true),
        make_composable_segment(
            SegmentId::ProsodyPredictor,
            -2.0,
            2.0,
            -1.0,
            1.0,
            vec![16],
            true,
        ),
        make_composable_segment(
            SegmentId::F0EnergyPredictor,
            -1.0,
            1.0,
            -5.0,
            800.0,
            vec![16],
            true,
        ),
        make_composable_segment(
            SegmentId::Generator,
            -800.0,
            800.0,
            -1.0,
            1.0,
            vec![16],
            true,
        ),
    ];
    let verifier = KokoroCrownVerifier::from_segments(segments, "test-kokoro");
    let result = verifier
        .verify_segment(SegmentId::F0EnergyPredictor)
        .unwrap();
    assert!(result.property_proven);
    assert!(result.explanation.contains("F0"));
}

#[test]
fn test_extract_best_bounds_prefers_sound_alpha_crown_family_over_sound_ibp() {
    let mut kernels = HashMap::new();
    kernels.insert(
        "kokoro_production_text_encoder_ibp".to_string(),
        make_status_entry("IBP", "sound", 0.1, -0.05, 0.05, &[4]),
    );
    kernels.insert(
        "kokoro_production_text_encoder_alpha".to_string(),
        make_status_entry("AlphaCrown", "sound", 0.4, -0.2, 0.2, &[4]),
    );

    let status = StatusFile { kernels };
    let bounds =
        extract_best_bounds(&status, SegmentId::TextEncoder).expect("best bounds should extract");

    assert_eq!(
        bounds.status_key, "kokoro_production_text_encoder_alpha",
        "sound AlphaCrown should outrank a narrower sound IBP sibling"
    );
    assert_eq!(bounds.method, "AlphaCrown");
    assert!(bounds.is_sound);
}

#[test]
fn test_extract_best_bounds_preserves_beta_crown_method_into_verified_stage() {
    let mut kernels = HashMap::new();
    kernels.insert(
        "kokoro_production_generator_ibp".to_string(),
        make_status_entry("IBP", "sound", 0.2, -0.1, 0.1, &[8]),
    );
    kernels.insert(
        "kokoro_production_generator_beta".to_string(),
        make_status_entry("BetaCrown", "sound", 1.2, -0.6, 0.6, &[8]),
    );

    let status = StatusFile { kernels };
    let bounds =
        extract_best_bounds(&status, SegmentId::Generator).expect("best bounds should extract");
    let stage = bounds.to_verified_stage();

    assert_eq!(
        bounds.method, "BetaCrown",
        "sound BetaCrown should survive segment extraction instead of degrading to IBP"
    );
    assert_eq!(
        stage.method, "BetaCrown",
        "the verified-stage bridge should preserve the selected BetaCrown provenance"
    );
    assert!(bounds.is_sound);
}

#[test]
fn test_verify_all_produces_certificate() {
    let verifier = make_sound_verifier();
    let result = verifier.verify_all().unwrap();

    // Certificate should be structurally valid.
    assert!(result.certificate.validate().is_ok());

    // Pipeline should be sound since all stages are sound.
    assert!(result.is_sound());

    // Should have segment results for all 5 segments.
    assert_eq!(result.segment_results.len(), 5);
}

#[test]
fn test_verify_all_pcm_property_from_pipeline() {
    let verifier = make_sound_verifier();
    let result = verifier.verify_all().unwrap();

    // The pipeline's end-to-end output bounds come from the Generator stage.
    // Since Generator bounds are [-1, 1], the non-clipping property should
    // be proven (P2, index 1).
    let p2 = result
        .crown_bundle
        .results
        .iter()
        .find(|r| r.property_index == 1);

    // P2 should exist and be proven (bounds are within [-1, 1]).
    assert!(p2.is_some(), "P2 (non-clipping) should be in results");
    assert!(p2.unwrap().proven, "P2 should be proven for [-1, 1] bounds");
}

#[test]
fn test_verify_all_certificate_summary() {
    let verifier = make_sound_verifier();
    let result = verifier.verify_all().unwrap();
    let summary = result.summary();
    assert!(summary.contains("Kokoro CROWN Certificate"));
    assert!(summary.contains("test-kokoro"));
}

#[test]
fn test_unsound_segment_marks_pipeline_unsound() {
    let segments = vec![
        make_composable_segment(SegmentId::BertEncoder, -1.0, 1.0, -3.0, 3.0, vec![16], true),
        make_composable_segment(
            SegmentId::TextEncoder,
            -3.0,
            3.0,
            -2.0,
            2.0,
            vec![16],
            false,
        ), // unsound
        make_composable_segment(
            SegmentId::ProsodyPredictor,
            -2.0,
            2.0,
            -1.0,
            1.0,
            vec![16],
            true,
        ),
        make_composable_segment(
            SegmentId::F0EnergyPredictor,
            -1.0,
            1.0,
            -1.0,
            1.0,
            vec![16],
            true,
        ),
        make_composable_segment(SegmentId::Generator, -1.0, 1.0, -1.0, 1.0, vec![16], true),
    ];
    let verifier = KokoroCrownVerifier::from_segments(segments, "test");
    assert!(!verifier.all_sound());

    let result = verifier.verify_all().unwrap();
    assert!(!result.is_sound());
}

#[test]
fn test_input_in_pcm_domain() {
    let verifier = make_sound_verifier();
    assert!(verifier.input_in_pcm_domain());

    // Verifier with wide Generator bounds should fail.
    let segments = vec![
        make_composable_segment(SegmentId::BertEncoder, -1.0, 1.0, -3.0, 3.0, vec![16], true),
        make_composable_segment(SegmentId::TextEncoder, -3.0, 3.0, -2.0, 2.0, vec![16], true),
        make_composable_segment(
            SegmentId::ProsodyPredictor,
            -2.0,
            2.0,
            -2.0,
            2.0,
            vec![16],
            true,
        ),
        make_composable_segment(
            SegmentId::F0EnergyPredictor,
            -2.0,
            2.0,
            -2.0,
            2.0,
            vec![16],
            true,
        ),
        make_composable_segment(SegmentId::Generator, -2.0, 2.0, -2.0, 2.0, vec![16], true),
    ];
    let wide = KokoroCrownVerifier::from_segments(segments, "test");
    assert!(!wide.input_in_pcm_domain());
}

#[test]
fn test_save_and_load_roundtrip() {
    let verifier = make_sound_verifier();
    let result = verifier.verify_all().unwrap();

    let dir = std::env::temp_dir().join("nn_test_kokoro_crown_verifier");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_cert.json");

    // Save.
    KokoroCrownVerifier::save(&result, &path).unwrap();
    assert!(path.exists());

    // Load.
    let loaded = KokoroCrownVerifier::load(&path).unwrap();
    assert_eq!(loaded.model_name, "test-kokoro");
    assert_eq!(loaded.version, result.certificate.version);
    assert_eq!(loaded.proven_count, result.certificate.proven_count);

    // Clean up.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_from_status_file_real() {
    // This test loads the actual status file if it exists.
    let repo_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../nn_verify_status_kokoro.json");
    if !repo_path.exists() {
        // Skip if the status file is not available.
        return;
    }

    let verifier = KokoroCrownVerifier::from_status_file(&repo_path);
    match verifier {
        Ok(v) => {
            assert_eq!(v.segment_count(), 5);
            let result = v.verify_all().unwrap();
            assert!(result.certificate.validate().is_ok());
            // Verify we get a non-empty summary.
            let summary = result.summary();
            assert!(summary.contains("Kokoro CROWN Certificate"));
        }
        Err(VerifierError::MissingSegment { segment, prefix }) => {
            // Some segments may have only stale entries in the status file.
            // This is expected until all production entries are refreshed.
            eprintln!(
                "Segment '{segment}' not found (tried {prefix}) — \
                 expected when production entries are stale"
            );
        }
        Err(e) => panic!("unexpected error loading status file: {e}"),
    }
}

#[test]
fn test_check_segment_property_unsound_not_proven() {
    let seg = make_segment(SegmentId::Generator, -1.0, 1.0, vec![16], false);
    let (proven, explanation) = check_segment_property(&seg);
    // PCM range is satisfied but unsound, so not proven.
    assert!(!proven);
    assert!(explanation.contains("sound=false"));
}

#[test]
fn test_junction_contracts_with_tight_bounds() {
    let segments = vec![
        make_segment(SegmentId::BertEncoder, -3.0, 3.0, vec![16], true),
        make_segment(SegmentId::TextEncoder, -2.0, 2.0, vec![16], true),
        make_segment(SegmentId::ProsodyPredictor, -10.0, 10.0, vec![16], true),
        make_segment(SegmentId::F0EnergyPredictor, -5.0, 800.0, vec![16], true),
        make_segment(SegmentId::Generator, -1.0, 1.0, vec![16], true),
    ];
    let junctions = check_junction_contracts(&segments);

    // J5_AUDIO should be verified since Generator is [-1, 1].
    let j5 = junctions.iter().find(|j| j.contract.name == "J5_AUDIO");
    assert!(j5.is_some());
    assert!(j5.unwrap().bounds_verified);
}
