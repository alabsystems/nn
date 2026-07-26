// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `dpdf_certify` module.

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal valid dpdf status JSON with one sound entry per model family.
fn minimal_status_json() -> &'static str {
    r#"{
        "model": "dpdf",
        "last_updated": "2026-03-28",
        "kernels": {
            "doclayout_yolo::test_detection_sigmoid_ibp": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": false,
                "compose_test": "compose_dpdf_models_all::doclayout_yolo::test_detection_sigmoid_ibp"
            },
            "glm_ocr::test_rms_norm_ibp": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": false,
                "compose_test": "compose_dpdf_models_all::glm_ocr::test_rms_norm_ibp"
            },
            "table_transformer::test_resnet_basic_block_ibp": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": false,
                "compose_test": "compose_dpdf_models_all::table_transformer::test_resnet_basic_block_ibp"
            },
            "paddle_ocr::test_db_sigmoid_output_ibp": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": false,
                "compose_test": "compose_dpdf_models_all::paddle_ocr::test_db_sigmoid_output_ibp"
            },
            "qwen3_vl::test_conv3d_patch_embed_ibp": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": false,
                "compose_test": "compose_dpdf_models_all::qwen3_vl::test_conv3d_patch_embed_ibp"
            }
        }
    }"#
}

/// Status JSON with heuristic-only entries.
fn heuristic_only_json() -> &'static str {
    r#"{
        "model": "dpdf",
        "last_updated": "2026-03-28",
        "kernels": {
            "doclayout_yolo::test_full_compose_ibp": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "heuristic",
                "stale": false,
                "compose_test": "compose_dpdf_models_all::doclayout_yolo::test_full_compose_ibp"
            }
        }
    }"#
}

/// Status JSON with no kernels.
fn empty_status_json() -> &'static str {
    r#"{
        "model": "dpdf",
        "last_updated": "2026-03-28",
        "kernels": {}
    }"#
}

/// Status JSON with stale entries that should be excluded.
fn stale_entries_json() -> &'static str {
    r#"{
        "model": "dpdf",
        "last_updated": "2026-03-28",
        "kernels": {
            "doclayout_yolo::test_old_entry": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": true,
                "compose_test": "old_test"
            },
            "glm_ocr::test_fresh": {
                "soundness_mode": "IbpValidated",
                "proof_strength": "sound",
                "stale": false,
                "compose_test": "fresh_test"
            }
        }
    }"#
}

// ---------------------------------------------------------------------------
// DpdfProperty tests
// ---------------------------------------------------------------------------

#[test]
fn test_dpdf_property_all_has_eight_entries() {
    assert_eq!(DpdfProperty::ALL.len(), 8);
}

#[test]
fn test_dpdf_property_numbers_are_one_through_eight() {
    for (i, prop) in DpdfProperty::ALL.iter().enumerate() {
        assert_eq!(prop.number(), i + 1);
    }
}

#[test]
fn test_dpdf_property_display_includes_number_and_name() {
    let p = DpdfProperty::P1LayoutSigmoidBounds;
    let s = format!("{p}");
    assert!(
        s.starts_with("P1:"),
        "display should start with P1: got {s}"
    );
    assert!(
        s.contains("sigmoid"),
        "display should contain sigmoid, got {s}"
    );
}

#[test]
fn test_dpdf_property_name_is_nonempty() {
    for prop in &DpdfProperty::ALL {
        assert!(!prop.name().is_empty(), "property {prop:?} has empty name");
    }
}

// ---------------------------------------------------------------------------
// PropertyStatus tests
// ---------------------------------------------------------------------------

#[test]
fn test_property_status_display() {
    assert_eq!(format!("{}", PropertyStatus::Proven), "PROVEN");
    assert_eq!(format!("{}", PropertyStatus::Heuristic), "HEURISTIC");
    assert_eq!(format!("{}", PropertyStatus::Unverified), "UNVERIFIED");
    assert_eq!(format!("{}", PropertyStatus::NotApplicable), "N/A");
}

// ---------------------------------------------------------------------------
// Certificate generation from JSON
// ---------------------------------------------------------------------------

#[test]
fn test_generate_from_minimal_json() {
    let cert = DpdfCertificate::generate_from_json(minimal_status_json())
        .expect("should parse minimal json");
    assert_eq!(cert.properties.len(), 8);
    assert_eq!(cert.compose_test_count, 5); // 5 sound, 0 heuristic
    assert!(!cert.models_covered.is_empty());
}

#[test]
fn test_generate_from_empty_json() {
    let cert =
        DpdfCertificate::generate_from_json(empty_status_json()).expect("should parse empty json");
    assert_eq!(cert.compose_test_count, 0);
    // With no models at all, P5/P6/P7 should be unverified, P8 unverified
    let (proven, _, unverified, _) = cert.status_counts();
    assert_eq!(proven, 0);
    assert!(unverified > 0, "some properties should be unverified");
}

#[test]
fn test_generate_stale_entries_excluded() {
    let cert =
        DpdfCertificate::generate_from_json(stale_entries_json()).expect("should parse stale json");
    // Only 1 non-stale entry (glm_ocr)
    assert_eq!(cert.compose_test_count, 1);
}

#[test]
fn test_generate_heuristic_entries() {
    let cert = DpdfCertificate::generate_from_json(heuristic_only_json())
        .expect("should parse heuristic json");
    // 0 sound, 1 heuristic
    assert_eq!(cert.compose_test_count, 1);
    // P1 should be heuristic (doclayout_yolo present, but 0 sound < threshold)
    let p1 = cert
        .properties
        .iter()
        .find(|(p, _, _)| *p == DpdfProperty::P1LayoutSigmoidBounds)
        .expect("P1 should exist");
    assert_eq!(p1.1, PropertyStatus::Heuristic);
}

#[test]
fn test_generate_invalid_json_returns_error() {
    let result = DpdfCertificate::generate_from_json("not json at all");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

#[test]
fn test_report_contains_header() {
    let cert = DpdfCertificate::generate_from_json(minimal_status_json()).unwrap();
    let report = cert.to_report();
    assert!(
        report.contains("# dpdf Certification Report"),
        "report missing header"
    );
}

#[test]
fn test_report_contains_property_table() {
    let cert = DpdfCertificate::generate_from_json(minimal_status_json()).unwrap();
    let report = cert.to_report();
    assert!(report.contains("| P1 |"), "report missing P1 row");
    assert!(report.contains("| P8 |"), "report missing P8 row");
}

#[test]
fn test_report_contains_summary_section() {
    let cert = DpdfCertificate::generate_from_json(minimal_status_json()).unwrap();
    let report = cert.to_report();
    assert!(
        report.contains("Compose tests:"),
        "report missing compose test count"
    );
    assert!(
        report.contains("Deployment ready:"),
        "report missing deployment ready"
    );
}

// ---------------------------------------------------------------------------
// Deployment readiness
// ---------------------------------------------------------------------------

#[test]
fn test_deployment_ready_when_all_p1_p7_proven_or_heuristic() {
    let props: Vec<_> = DpdfProperty::ALL
        .iter()
        .map(|&p| {
            if p.number() <= 7 {
                (p, PropertyStatus::Proven, "evidence".to_string())
            } else {
                // P8 can be unverified and still be deployment-ready
                (p, PropertyStatus::Unverified, "none".to_string())
            }
        })
        .collect();
    let cert = DpdfCertificate::new(
        props,
        10,
        0,
        0,
        vec!["test".to_string()],
        "2026-03-28".to_string(),
    );
    assert!(cert.is_deployment_ready());
}

#[test]
fn test_not_deployment_ready_when_p1_unverified() {
    let props: Vec<_> = DpdfProperty::ALL
        .iter()
        .map(|&p| {
            if p == DpdfProperty::P1LayoutSigmoidBounds {
                (p, PropertyStatus::Unverified, "none".to_string())
            } else {
                (p, PropertyStatus::Proven, "evidence".to_string())
            }
        })
        .collect();
    let cert = DpdfCertificate::new(
        props,
        10,
        0,
        0,
        vec!["test".to_string()],
        "2026-03-28".to_string(),
    );
    assert!(!cert.is_deployment_ready());
}

#[test]
fn test_deployment_ready_with_not_applicable() {
    let props: Vec<_> = DpdfProperty::ALL
        .iter()
        .map(|&p| {
            if p.number() <= 7 {
                (p, PropertyStatus::NotApplicable, "n/a".to_string())
            } else {
                (p, PropertyStatus::Unverified, "none".to_string())
            }
        })
        .collect();
    let cert = DpdfCertificate::new(props, 0, 0, 0, vec![], "2026-03-28".to_string());
    assert!(cert.is_deployment_ready());
}

// ---------------------------------------------------------------------------
// Status counts
// ---------------------------------------------------------------------------

#[test]
fn test_status_counts_correct() {
    let props = vec![
        (
            DpdfProperty::P1LayoutSigmoidBounds,
            PropertyStatus::Proven,
            String::new(),
        ),
        (
            DpdfProperty::P2OcrSoftmaxDistribution,
            PropertyStatus::Proven,
            String::new(),
        ),
        (
            DpdfProperty::P3TableBoxNormalized,
            PropertyStatus::Heuristic,
            String::new(),
        ),
        (
            DpdfProperty::P4DflRegressionValid,
            PropertyStatus::Heuristic,
            String::new(),
        ),
        (
            DpdfProperty::P5NmsPreservesTopConfidence,
            PropertyStatus::Unverified,
            String::new(),
        ),
        (
            DpdfProperty::P6IoUBounded,
            PropertyStatus::NotApplicable,
            String::new(),
        ),
        (
            DpdfProperty::P7ConfidenceFilterMonotone,
            PropertyStatus::Proven,
            String::new(),
        ),
        (
            DpdfProperty::P8QuantizedEpsilonBound,
            PropertyStatus::Unverified,
            String::new(),
        ),
    ];
    let cert = DpdfCertificate::new(props, 0, 0, 0, vec![], "2026-03-28".to_string());
    let (proven, heuristic, unverified, na) = cert.status_counts();
    assert_eq!(proven, 3);
    assert_eq!(heuristic, 2);
    assert_eq!(unverified, 2);
    assert_eq!(na, 1);
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_display_contains_deployment_ready() {
    let cert = DpdfCertificate::generate_from_json(minimal_status_json()).unwrap();
    let s = format!("{cert}");
    assert!(
        s.contains("Deployment ready:"),
        "Display should show deployment readiness"
    );
}

// ---------------------------------------------------------------------------
// Serialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_serde_roundtrip() {
    let cert = DpdfCertificate::generate_from_json(minimal_status_json()).unwrap();
    let json = serde_json::to_string(&cert).expect("serialize");
    let deser: DpdfCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deser.properties.len(), cert.properties.len());
    assert_eq!(deser.compose_test_count, cert.compose_test_count);
    assert_eq!(deser.models_covered, cert.models_covered);
}

// ---------------------------------------------------------------------------
// Date helper
// ---------------------------------------------------------------------------

#[test]
fn test_current_date_format() {
    let d = current_date();
    // Should match YYYY-MM-DD
    assert_eq!(d.len(), 10, "date length should be 10: {d}");
    assert_eq!(&d[4..5], "-", "expected dash at pos 4: {d}");
    assert_eq!(&d[7..8], "-", "expected dash at pos 7: {d}");
}
