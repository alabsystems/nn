// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `KokoroCertificateBundle` aggregation and deployment gating.
//!
//! Part of #4254.

use crate::kokoro_certificate_bundle::{BundleConfig, EntrySoundness, KokoroCertificateBundle};
use crate::status::VerifyStatus;

/// Build a VerifyStatus from a JSON string. This is the test-friendly way
/// to construct status objects with private `kernels` field access.
fn status_from_json(json: &str) -> VerifyStatus {
    serde_json::from_str(json).expect("valid test JSON")
}

/// Build a VerifyStatus with the given entries.
/// Each entry is (name, method_json, soundness_mode, output_width).
/// method_json uses the serde UPPERCASE format: "IBP", "CROWN", "ALPHACROWN", etc.
/// soundness uses serde snake_case: "sound", "heuristic".
fn build_status(entries: &[(&str, &str, &str, f32)]) -> VerifyStatus {
    let mut kernel_json_parts = Vec::new();
    for (name, method, soundness, width) in entries {
        let lower = -width / 2.0;
        let upper = width / 2.0;
        kernel_json_parts.push(format!(
            r#""{name}": {{
                "status": "verified",
                "method": "{method}",
                "input_bounds": {{
                    "variable_inputs": [{{"param_index": 0, "lower": -1.0, "upper": 1.0}}],
                    "constant_params": [],
                    "input_range": [-1.0, 1.0]
                }},
                "output_bounds": {{
                    "lower": {lower},
                    "upper": {upper}
                }},
                "output_width": {width},
                "soundness_mode": "{soundness}"
            }}"#
        ));
    }
    let json = format!(r#"{{"kernels": {{{}}}}}"#, kernel_json_parts.join(","));
    status_from_json(&json)
}

/// Default test config with relaxed thresholds.
fn test_config(min_sound_ratio: f64, max_vacuous: usize) -> BundleConfig {
    BundleConfig {
        status_path: std::path::PathBuf::from("/dev/null"),
        model_hash: "test_hash".to_string(),
        gamma_crown_rev: "test_rev".to_string(),
        min_sound_ratio,
        max_vacuous,
        max_heuristic: None,
        min_crown_stages: 0,
        max_gaps: 100, // Relaxed — no real pipeline stages in unit tests
    }
}

#[test]
fn test_all_sound_entries() {
    let status = build_status(&[
        ("entry_a", "CROWN", "sound", 2.0),
        ("entry_b", "ALPHACROWN", "sound", 1.5),
        ("entry_c", "IBP", "sound", 3.0),
    ]);

    let config = test_config(1.0, 0);
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert_eq!(bundle.sound_count(), 3);
    assert_eq!(bundle.heuristic_count(), 0);
    assert_eq!(bundle.vacuous_count(), 0);
    assert_eq!(bundle.total_entries(), 3);
    assert!((bundle.sound_ratio() - 1.0).abs() < 1e-9);
    assert!(bundle.soundness.all_sound());
}

#[test]
fn test_mixed_soundness_breakdown() {
    let status = build_status(&[
        ("sound_crown", "CROWN", "sound", 2.0),
        ("sound_ibp", "IBP", "sound", 5.0),
        ("heuristic", "IBP", "heuristic", 10.0),
        ("vacuous", "IBP", "sound", 200.0),
    ]);

    let config = test_config(0.5, 1);
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    // sound_crown -> SoundCrown, sound_ibp -> SoundIbp
    // heuristic -> Heuristic, vacuous -> Vacuous (width 200 > 100 threshold)
    assert_eq!(bundle.sound_count(), 2);
    assert_eq!(bundle.heuristic_count(), 1);
    assert_eq!(bundle.vacuous_count(), 1);
    assert_eq!(bundle.total_entries(), 4);
    assert!((bundle.sound_ratio() - 0.5).abs() < 1e-9);
    assert!(!bundle.soundness.all_sound());
}

#[test]
fn test_deployment_blocked_by_vacuous() {
    let status = build_status(&[
        ("good", "CROWN", "sound", 2.0),
        ("bad", "IBP", "sound", 200.0),
    ]);

    let config = test_config(0.5, 0); // Zero tolerance for vacuous
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert_eq!(bundle.vacuous_count(), 1);
    assert!(!bundle.is_deployment_ready());
}

#[test]
fn test_deployment_blocked_by_heuristic_limit() {
    let status = build_status(&[
        ("s1", "CROWN", "sound", 2.0),
        ("s2", "CROWN", "sound", 2.0),
        ("h1", "IBP", "heuristic", 5.0),
        ("h2", "IBP", "heuristic", 5.0),
    ]);

    let mut config = test_config(0.5, 0);
    config.max_heuristic = Some(1); // Only allow 1 heuristic

    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert_eq!(bundle.heuristic_count(), 2);
    assert!(!bundle.is_deployment_ready());
}

#[test]
fn test_deployment_blocked_by_low_sound_ratio() {
    let status = build_status(&[
        ("s1", "CROWN", "sound", 2.0),
        ("h1", "IBP", "heuristic", 5.0),
        ("h2", "IBP", "heuristic", 5.0),
        ("h3", "IBP", "heuristic", 5.0),
    ]);

    let config = test_config(0.90, 0); // Need 90% but only have 25%
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert!((bundle.sound_ratio() - 0.25).abs() < 1e-9);
    assert!(!bundle.is_deployment_ready());
}

#[test]
fn test_empty_status_not_deployment_ready() {
    let status = VerifyStatus::default();
    let config = test_config(0.0, 100);

    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert_eq!(bundle.total_entries(), 0);
    assert!(!bundle.is_deployment_ready());
}

#[test]
fn test_soundness_for_lookup() {
    let status = build_status(&[
        ("crown_entry", "CROWN", "sound", 2.0),
        ("ibp_entry", "IBP", "sound", 5.0),
    ]);

    let config = test_config(0.0, 100);
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert_eq!(
        bundle.soundness_for("crown_entry"),
        Some(EntrySoundness::SoundCrown)
    );
    assert_eq!(
        bundle.soundness_for("ibp_entry"),
        Some(EntrySoundness::SoundIbp)
    );
    assert_eq!(bundle.soundness_for("nonexistent"), None);
}

#[test]
fn test_method_counts_populated() {
    let status = build_status(&[
        ("a", "CROWN", "sound", 2.0),
        ("b", "CROWN", "sound", 2.0),
        ("c", "IBP", "sound", 5.0),
        ("d", "ALPHACROWN", "sound", 1.0),
    ]);

    let config = test_config(0.0, 100);
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert_eq!(bundle.soundness.method_counts.get("CROWN"), Some(&2));
    assert_eq!(bundle.soundness.method_counts.get("IBP"), Some(&1));
    assert_eq!(bundle.soundness.method_counts.get("AlphaCrown"), Some(&1));
}

#[test]
fn test_stale_entries_excluded() {
    // Build JSON with one active and one stale entry.
    let json = r#"{
        "kernels": {
            "active": {
                "status": "verified",
                "method": "CROWN",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [],
                    "input_range": [-1.0, 1.0]
                },
                "output_bounds": {"lower": -1.0, "upper": 1.0},
                "output_width": 2.0,
                "soundness_mode": "sound"
            },
            "stale_one": {
                "status": "verified",
                "method": "IBP",
                "input_bounds": {
                    "variable_inputs": [{"param_index": 0, "lower": -1.0, "upper": 1.0}],
                    "constant_params": [],
                    "input_range": [-1.0, 1.0]
                },
                "output_bounds": {"lower": -25.0, "upper": 25.0},
                "output_width": 50.0,
                "soundness_mode": "heuristic",
                "stale": true,
                "stale_reason": "outdated model"
            }
        }
    }"#;
    let status = status_from_json(json);
    let config = test_config(1.0, 0);

    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    // Only the active entry should be counted.
    assert_eq!(bundle.total_entries(), 1);
    assert_eq!(bundle.sound_count(), 1);
    assert_eq!(bundle.heuristic_count(), 0);
}

#[test]
fn test_json_roundtrip() {
    let status = build_status(&[("a", "CROWN", "sound", 2.0), ("b", "IBP", "heuristic", 5.0)]);

    let mut config = test_config(0.5, 0);
    config.model_hash = "deadbeef".to_string();
    config.gamma_crown_rev = "abc123".to_string();

    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    let json = bundle.to_json().expect("serialization should succeed");
    let roundtripped =
        KokoroCertificateBundle::from_json(&json).expect("deserialization should succeed");

    assert_eq!(bundle.sound_count(), roundtripped.sound_count());
    assert_eq!(bundle.heuristic_count(), roundtripped.heuristic_count());
    assert_eq!(bundle.vacuous_count(), roundtripped.vacuous_count());
    assert_eq!(bundle.total_entries(), roundtripped.total_entries());
    assert_eq!(bundle.content_hash, roundtripped.content_hash);
}

#[test]
fn test_content_hash_integrity() {
    let status = build_status(&[("a", "CROWN", "sound", 2.0)]);

    let config = test_config(0.0, 100);
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert!(bundle.content_hash.is_some());
    assert!(bundle.verify_integrity());

    // Tamper with the bundle and verify integrity fails.
    let mut tampered = bundle;
    tampered.soundness.sound = 999;
    assert!(!tampered.verify_integrity());
}

#[test]
fn test_entry_soundness_is_sound() {
    assert!(EntrySoundness::SoundCrown.is_sound());
    assert!(EntrySoundness::SoundIbp.is_sound());
    assert!(EntrySoundness::SoundMixed.is_sound());
    assert!(!EntrySoundness::Heuristic.is_sound());
    assert!(!EntrySoundness::Vacuous.is_sound());
}

#[test]
fn test_thresholds_recorded_in_bundle() {
    let status = build_status(&[("a", "CROWN", "sound", 2.0)]);

    let config = BundleConfig {
        status_path: std::path::PathBuf::from("/dev/null"),
        model_hash: String::new(),
        gamma_crown_rev: String::new(),
        min_sound_ratio: 0.85,
        max_vacuous: 2,
        max_heuristic: Some(5),
        min_crown_stages: 4,
        max_gaps: 1,
    };

    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    assert!((bundle.thresholds.min_sound_ratio - 0.85).abs() < 1e-9);
    assert_eq!(bundle.thresholds.max_vacuous, 2);
    assert_eq!(bundle.thresholds.max_heuristic, Some(5));
    assert_eq!(bundle.thresholds.min_crown_stages, 4);
    assert_eq!(bundle.thresholds.max_gaps, 1);
}

#[test]
fn test_entry_records_sorted_by_name() {
    let status = build_status(&[
        ("z_entry", "IBP", "sound", 5.0),
        ("a_entry", "CROWN", "sound", 2.0),
        ("m_entry", "IBP", "heuristic", 10.0),
    ]);

    let config = test_config(0.0, 100);
    let bundle = KokoroCertificateBundle::from_status(&status, &config)
        .expect("bundle generation should succeed");

    let names: Vec<&str> = bundle
        .entry_records()
        .iter()
        .map(|e| e.kernel_name.as_str())
        .collect();
    assert_eq!(names, vec!["a_entry", "m_entry", "z_entry"]);
}
