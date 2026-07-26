// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bundle, display, and performance tests for certificate checker.
//!
//! Extracted from `certificate_checker_tests.rs` — covers CertificateBundle
//! checking, CheckIssue Display formatting, file round-trip, and graph-aware
//! trace performance regression.
//!
//! Part of #1678.

use super::checker_test_shared::{consistent_layer_bounds, sample_input_spec, sample_verification};
use super::*;
use crate::certificate::{CertificateBundle, ProofCertificate};
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;
use crate::VerificationSoundnessMode;

// ---------------------------------------------------------------------------
// Bundle checking
// ---------------------------------------------------------------------------

#[test]
fn test_check_bundle_all_valid() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    let results = check_bundle(&bundle, None, None);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_valid());
}

#[test]
fn test_check_bundle_mixed() {
    let result = sample_verification();
    let good_cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));

    let mut bad_result = sample_verification();
    bad_result.kernel_name = String::new();
    let bad_cert = ProofCertificate::from_verification(&bad_result, sample_input_spec());

    let bundle = CertificateBundle::new("test_model")
        .with_certificate(good_cert)
        .with_certificate(bad_cert);

    let results = check_bundle(&bundle, None, None);
    assert_eq!(results.len(), 2);
    assert!(results[0].is_valid());
    assert!(!results[1].is_valid());
}

// ---------------------------------------------------------------------------
// Display formatting
// ---------------------------------------------------------------------------

#[test]
fn test_check_issue_display() {
    let issue = CheckIssue::LayerTraceGap {
        layer_index: 2,
        output_bounds: vec![(-1.0, 1.0)],
        next_input_bounds: vec![(-2.0, 2.0)],
    };
    let msg = format!("{issue}");
    assert!(msg.contains("layer trace gap at layer 2"));

    let issue = CheckIssue::OutputMismatch {
        certificate_lower: -5.0,
        certificate_upper: 5.0,
        trace_lower: -3.0,
        trace_upper: 3.0,
    };
    let msg = format!("{issue}");
    assert!(msg.contains("output mismatch"));
}

#[test]
fn test_check_bundle_file_roundtrip() {
    let dir = std::env::temp_dir().join(format!("nn_checker_bundle_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("check_test.proof.json");

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    let bundle = CertificateBundle::new("test_model").with_certificate(cert);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(path.with_extension("proof.json.tmp"));

    bundle.save(&path).expect("save");

    let results = check_bundle_file(&path, None, None).expect("check");
    assert_eq!(results.len(), 1);
    assert!(results[0].is_valid());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// Graph-aware trace: large chain verifies in O(n) (performance regression)
// ---------------------------------------------------------------------------

#[test]
fn test_graph_aware_trace_large_chain_linear_time() {
    // Build a 1000-layer chain. With the O(n²) linear-scan approach,
    // this would do ~500K comparisons. With the HashMap index, it is O(n).
    let n = 1000;
    let mut bounds = Vec::with_capacity(n);
    for i in 0..n {
        bounds.push(LayerBoundRecord {
            layer_index: i,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-1.0, 1.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: if i == 0 {
                Some(vec![]) // network input
            } else {
                Some(vec![i - 1])
            },
        });
    }

    let result = sample_verification();
    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);

    let start = std::time::Instant::now();
    let check = check_certificate(&cert, None, None);
    let elapsed = start.elapsed();

    // A consistent chain should have no gaps (all bounds are identical).
    let gap_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .count();
    assert_eq!(gap_count, 0, "consistent chain should have no gaps");

    // With O(n) implementation, 1000 layers should complete in <50ms even
    // on slow machines. An O(n²) implementation would take measurably longer.
    assert!(
        elapsed.as_millis() < 500,
        "1000-layer trace check took {}ms — possible O(n²) regression",
        elapsed.as_millis()
    );
}

/// Companion to the linear-time test above: verifies that gap detection
/// actually catches a real gap in a chain. The linear-time test's gap_count==0
/// assertion is trivially true (all bounds identical). This test introduces
/// a deliberate mismatch so we know the checker would catch real gaps.
#[test]
fn test_graph_aware_trace_detects_gap_in_chain() {
    let n = 10;
    let mut bounds = Vec::with_capacity(n);
    for i in 0..n {
        bounds.push(LayerBoundRecord {
            layer_index: i,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-1.0, 1.0)],
            output_bounds: vec![(-1.0, 1.0)],
            method: PropMethod::Ibp,
            node_name: None,
            input_sources: if i == 0 {
                Some(vec![])
            } else {
                Some(vec![i - 1])
            },
        });
    }
    // Layer 5 claims output [-1, 1], but layer 6's input is [-10, 10] — a gap.
    bounds[6].input_bounds = vec![(-10.0, 10.0)];

    let result = sample_verification();
    let cert =
        ProofCertificate::from_verification(&result, sample_input_spec()).with_layer_bounds(bounds);

    let check = check_certificate(&cert, None, None);
    let gap_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::LayerTraceGap { .. }))
        .count();
    assert!(
        gap_count >= 1,
        "chain with mismatched bounds at layer 6 must produce at least 1 LayerTraceGap, got issues: {:?}",
        check.issues
    );
}

// ---------------------------------------------------------------------------
// Bundle filtering (Part of #1680)
// ---------------------------------------------------------------------------

/// Build a certificate with the given kernel name.
fn make_cert(name: &str) -> ProofCertificate {
    let mut v = sample_verification();
    v.kernel_name = name.to_string();
    let mut cert = ProofCertificate::from_verification(&v, sample_input_spec());
    cert.kernel_name = name.to_string();
    cert
}

/// Build a certificate with the given kernel name and source hash.
fn make_cert_with_hash(name: &str, hash: &str) -> ProofCertificate {
    let mut cert = make_cert(name);
    cert.source_hash = Some(hash.to_string());
    cert
}

#[test]
fn test_filter_by_names_basic() {
    let bundle = CertificateBundle::new("full")
        .with_certificate(make_cert("relu"))
        .with_certificate(make_cert("sigmoid"))
        .with_certificate(make_cert("tanh_act"))
        .with_certificate(make_cert("snake"));

    let filtered = bundle.filter_by_names("vad_verified", &["relu", "sigmoid"]);
    assert_eq!(filtered.model_name, "vad_verified");
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered.certificates[0].kernel_name, "relu");
    assert_eq!(filtered.certificates[1].kernel_name, "sigmoid");
}

#[test]
fn test_filter_by_names_empty_result() {
    let bundle = CertificateBundle::new("full")
        .with_certificate(make_cert("relu"))
        .with_certificate(make_cert("sigmoid"));

    let filtered = bundle.filter_by_names("empty", &["nonexistent"]);
    assert_eq!(filtered.len(), 0);
    assert!(filtered.is_empty());
}

#[test]
fn test_filter_preserves_certificate_data() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    let bundle = CertificateBundle::new("full").with_certificate(cert);

    let filtered = bundle.filter_by_names("model", &["snake"]);
    assert_eq!(filtered.len(), 1);
    let c = &filtered.certificates[0];
    assert_eq!(c.kernel_name, "snake");
    assert!(c.is_finite);
    assert_eq!(c.soundness_mode, VerificationSoundnessMode::Sound);
    assert_eq!(c.source_hash.as_deref(), Some(&"b".repeat(64) as &str));
    assert_eq!(c.weight_hash.as_deref(), Some(&"a".repeat(64) as &str));
    assert!(c.layer_bounds.is_some());
}

#[test]
fn test_all_have_source_hash() {
    let bundle = CertificateBundle::new("test")
        .with_certificate(make_cert_with_hash("relu", &"a".repeat(64)))
        .with_certificate(make_cert_with_hash("sigmoid", &"b".repeat(64)));
    assert!(bundle.all_have_source_hash());

    let bundle_missing = CertificateBundle::new("test")
        .with_certificate(make_cert_with_hash("relu", &"a".repeat(64)))
        .with_certificate(make_cert("sigmoid"));
    assert!(!bundle_missing.all_have_source_hash());

    let bundle_empty =
        CertificateBundle::new("test").with_certificate(make_cert_with_hash("relu", ""));
    assert!(!bundle_empty.all_have_source_hash());
}

#[test]
fn test_all_sound() {
    let bundle = CertificateBundle::new("test")
        .with_certificate(make_cert("relu"))
        .with_certificate(make_cert("sigmoid"));
    // sample_verification sets soundness_mode = Sound
    assert!(bundle.all_sound());

    let mut non_sound = sample_verification();
    non_sound.kernel_name = "tanh_act".to_string();
    non_sound.soundness_mode = VerificationSoundnessMode::Heuristic;
    let mut cert = ProofCertificate::from_verification(&non_sound, sample_input_spec());
    cert.kernel_name = "tanh_act".to_string();
    let bundle_mixed = CertificateBundle::new("test")
        .with_certificate(make_cert("relu"))
        .with_certificate(cert);
    assert!(!bundle_mixed.all_sound());
}

#[test]
fn test_filter_and_validate_roundtrip() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    let bundle = CertificateBundle::new("full").with_certificate(cert);

    let filtered = bundle.filter_by_names("vad", &["snake"]);
    assert!(filtered.validate_all().is_ok());

    let dir = std::env::temp_dir().join(format!("nn_filter_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("filtered.proof.json");
    filtered.save(&path).expect("save");

    let loaded = CertificateBundle::load(&path).expect("load");
    assert_eq!(loaded.model_name, "vad");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.certificates[0].kernel_name, "snake");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// Bundle hash caching: check_bundle produces identical results to per-cert
// checking, and hash mismatches are still detected through the cache.
// ---------------------------------------------------------------------------

#[test]
fn test_check_bundle_cached_hash_matches_individual() {
    let dir = std::env::temp_dir().join(format!("nn_bundle_cache_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);

    // Create a source file with known content.
    let source_path = dir.join("model.rs");
    std::fs::write(&source_path, b"fn main() {}").expect("write source");
    let source_hash = compute_file_hash(&source_path).expect("hash source");

    // Build 3 certificates referencing the same source hash.
    let result = sample_verification();
    let certs: Vec<_> = (0..3)
        .map(|i| {
            let mut v = result.clone();
            v.kernel_name = format!("kernel_{i}");
            let mut cert = ProofCertificate::from_verification(&v, sample_input_spec());
            cert.kernel_name = format!("kernel_{i}");
            cert.source_hash = Some(source_hash.clone());
            cert.with_layer_bounds(consistent_layer_bounds())
                .with_weight_hash("a".repeat(64))
        })
        .collect();

    let bundle = certs.into_iter().fold(
        CertificateBundle::new("cached_test"),
        crate::certificate::bundle::CertificateBundle::with_certificate,
    );

    // Bundle check (uses cache internally).
    let bundle_results = check_bundle(&bundle, None, Some(&source_path));

    // Individual checks (each computes hash independently).
    let individual_results: Vec<_> = bundle
        .certificates
        .iter()
        .map(|cert| check_certificate(cert, None, Some(&source_path)))
        .collect();

    // Results must be identical.
    assert_eq!(bundle_results.len(), individual_results.len());
    for (b, i) in bundle_results.iter().zip(individual_results.iter()) {
        assert_eq!(b.kernel_name, i.kernel_name);
        assert_eq!(b.issues.len(), i.issues.len());
        assert_eq!(b.is_valid(), i.is_valid());
    }

    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_check_bundle_cached_hash_detects_mismatch() {
    let dir = std::env::temp_dir().join(format!("nn_bundle_mismatch_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);

    let source_path = dir.join("model.rs");
    std::fs::write(&source_path, b"fn main() {}").expect("write source");

    // Certificate claims a wrong source hash.
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("wrong_hash_that_will_not_match".to_string());

    let bundle = CertificateBundle::new("mismatch_test")
        .with_certificate(cert.clone())
        .with_certificate(cert);

    let results = check_bundle(&bundle, None, Some(&source_path));

    // Both certificates should detect the source hash mismatch.
    for (idx, r) in results.iter().enumerate() {
        let has_mismatch = r
            .issues
            .iter()
            .any(|i| matches!(i, CheckIssue::SourceHashMismatch { .. }));
        assert!(
            has_mismatch,
            "certificate {idx} should have SourceHashMismatch, got: {:?}",
            r.issues
        );
    }

    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// check_bundle_file_with_key — keyed file round-trip (#3325)
// ---------------------------------------------------------------------------

#[test]
fn test_check_bundle_file_with_key_valid_signature() {
    let dir = std::env::temp_dir().join(format!("nn_checker_bundle_keyed_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("keyed_test.proof.json");

    let key = b"test-key-for-bundle-file-keyed";
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    crate::certificate::integrity::sign_certificate(&mut cert, key).unwrap();

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);
    bundle.save(&path).expect("save");

    let results = check_bundle_file_with_key(&path, None, None, Some(key)).expect("check");
    assert_eq!(results.len(), 1, "bundle has one certificate");
    assert!(
        results[0].is_valid(),
        "signed cert with correct key should be valid: {:?}",
        results[0].issues
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_check_bundle_file_with_key_wrong_key_detected() {
    let dir = std::env::temp_dir().join(format!(
        "nn_checker_bundle_wrongkey_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("wrongkey_test.proof.json");

    let correct_key = b"correct-signing-key";
    let wrong_key = b"attacker-key";
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    crate::certificate::integrity::sign_certificate(&mut cert, correct_key).unwrap();

    let bundle = CertificateBundle::new("test_model").with_certificate(cert);
    bundle.save(&path).expect("save");

    let results = check_bundle_file_with_key(&path, None, None, Some(wrong_key)).expect("check");
    assert_eq!(results.len(), 1);
    let has_sig_issue = results[0].issues.iter().any(|i| {
        matches!(
            i,
            CheckIssue::SignatureInvalid { .. } | CheckIssue::SignatureKeyError { .. }
        )
    });
    assert!(
        has_sig_issue,
        "wrong key must produce signature issue: {:?}",
        results[0].issues
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_check_bundle_file_with_key_none_key_skips_hmac() {
    let dir = std::env::temp_dir().join(format!("nn_checker_bundle_nokey_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("nokey_test.proof.json");

    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(consistent_layer_bounds())
        .with_weight_hash("a".repeat(64))
        .with_source_hash("b".repeat(64));
    let bundle = CertificateBundle::new("test_model").with_certificate(cert);
    bundle.save(&path).expect("save");

    let results = check_bundle_file_with_key(&path, None, None, None).expect("check");
    assert_eq!(results.len(), 1);
    assert!(
        results[0].is_valid(),
        "None key should skip HMAC and pass: {:?}",
        results[0].issues
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
