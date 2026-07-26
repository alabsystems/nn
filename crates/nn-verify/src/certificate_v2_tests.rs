// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for v2 proof certificate features: builder methods, v2 validation,
//! v1 backward compatibility, SHA-256 fingerprinting, and serde roundtrips.
//!
//! Enriched pipeline and v2 bundle tests are in `certificate_v2_tests_enriched.rs`.

use std::path::Path;

use super::certificate_test_helpers::*;
use super::*;

// ---------------------------------------------------------------------------
// v2 tests: builder methods
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_with_layer_bounds() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(sample_layer_bounds());

    assert!(cert.layer_bounds.is_some());
    let bounds = cert.layer_bounds.as_ref().unwrap();
    assert_eq!(bounds.len(), 2);
    assert_eq!(bounds[0].layer_type, "Linear");
    assert_eq!(bounds[1].layer_type, "ReLU");
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_with_kani_status() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_kani_status(sample_kani_record());

    let kani = cert.kani_status.as_ref().unwrap();
    assert_eq!(kani.harness_count, 3);
    assert_eq!(kani.status, KaniOutcome::Passed);
    assert_eq!(kani.properties.len(), 3);
    assert_eq!(kani.cbmc_version.as_deref(), Some("6.0.0"));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_with_hashes() {
    let result = sample_verification();
    let valid_hash = "a".repeat(64); // 64 hex chars
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_weight_hash(valid_hash.clone())
        .with_source_hash(valid_hash.clone());

    assert_eq!(cert.weight_hash.as_deref(), Some(valid_hash.as_str()));
    assert_eq!(cert.source_hash.as_deref(), Some(valid_hash.as_str()));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_with_verifier_version() {
    let result = sample_verification();
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_verifier_version("NY 0.3.0".to_string());

    assert_eq!(cert.verifier_version.as_deref(), Some("NY 0.3.0"));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_v2_full_roundtrip() {
    let result = sample_verification();
    let valid_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(sample_layer_bounds())
        .with_kani_status(sample_kani_record())
        .with_weight_hash(valid_hash.to_string())
        .with_source_hash(valid_hash.to_string())
        .with_verifier_version("NY 0.3.0".to_string())
        .with_smt_outcome("proven");

    assert!(cert.validate().is_ok());

    // JSON roundtrip preserves all v2 fields
    let json = cert.to_json().expect("serialize");
    let parsed: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cert, parsed);
    assert!(parsed.layer_bounds.is_some());
    assert!(parsed.kani_status.is_some());
    assert!(parsed.weight_hash.is_some());
    assert!(parsed.source_hash.is_some());
    assert!(parsed.verifier_version.is_some());
}

// ---------------------------------------------------------------------------
// v2 tests: validation
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_validate_v1_version_accepted() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.version = 1; // v1 still valid
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_validate_empty_layer_bounds() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.layer_bounds = Some(vec![]);
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::EmptyLayerBounds));
}

#[test]
fn test_certificate_validate_layer_index_mismatch() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    let mut bounds = sample_layer_bounds();
    bounds[1].layer_index = 5; // Should be 1
    cert.layer_bounds = Some(bounds);
    let err = cert.validate().unwrap_err();
    assert!(matches!(
        err,
        CertificateError::LayerIndexMismatch {
            expected: 1,
            actual: 5
        }
    ));
}

#[test]
fn test_certificate_validate_invalid_weight_hash() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.weight_hash = Some("not-a-valid-hash".to_string());
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::InvalidHash { .. }));
}

#[test]
fn test_certificate_validate_invalid_source_hash_wrong_length() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    cert.source_hash = Some("abc123".to_string()); // Too short
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::InvalidHash { .. }));
}

#[test]
fn test_certificate_validate_invalid_hash_non_hex() {
    let result = sample_verification();
    let mut cert = ProofCertificate::from_verification(&result, sample_input_spec());
    // 64 chars but contains 'g' which is not hex
    cert.weight_hash = Some("g".repeat(64));
    let err = cert.validate().unwrap_err();
    assert!(matches!(err, CertificateError::InvalidHash { .. }));
}

// ---------------------------------------------------------------------------
// v1 backward compatibility: v1 JSON → v2 struct
// ---------------------------------------------------------------------------

#[test]
fn test_v1_json_deserializes_into_v2_struct() {
    // Simulate a v1 certificate JSON (no v2 fields present).
    let v1_json = r#"{
        "version": 1,
        "kernel_name": "snake",
        "input_spec": {
            "variable_inputs": [{"param_index": 0, "lower": -10.0, "upper": 10.0}],
            "constant_params": [1.0],
            "input_shape": [1],
            "input_range": [-10.0, 10.0]
        },
        "output_bounds": {"lower": -9.704, "upper": 10.296},
        "output_width": 20.0,
        "is_finite": true,
        "method": "IBP",
        "soundness_mode": "sound",
        "generated_at": "1234567890Z"
    }"#;

    let cert: ProofCertificate = serde_json::from_str(v1_json).expect("deserialize v1");
    assert_eq!(cert.version, 1);
    assert_eq!(cert.kernel_name, "snake");
    // v2 fields should all be None (via #[serde(default)])
    assert!(cert.layer_bounds.is_none());
    assert!(cert.kani_status.is_none());
    assert!(cert.weight_hash.is_none());
    assert!(cert.source_hash.is_none());
    assert!(cert.verifier_version.is_none());
    // v1 certificate should pass validation
    assert!(cert.validate().is_ok());
}

#[test]
fn test_v1_bundle_json_deserializes_into_v2() {
    let v1_bundle_json = r#"{
        "version": 1,
        "model_name": "legacy_model",
        "certificates": [{
            "version": 1,
            "kernel_name": "silu_mul",
            "input_spec": {
                "variable_inputs": [{"param_index": 0, "lower": -5.0, "upper": 5.0}],
                "constant_params": [],
                "input_shape": [1],
                "input_range": [-5.0, 5.0]
            },
            "output_bounds": {"lower": -1.5, "upper": 3.5},
            "output_width": 5.0,
            "is_finite": true,
            "method": "CROWN",
            "soundness_mode": "heuristic",
            "generated_at": "1234567890Z"
        }],
        "generated_at": "1234567890Z"
    }"#;

    let bundle: CertificateBundle =
        serde_json::from_str(v1_bundle_json).expect("deserialize v1 bundle");
    assert_eq!(bundle.version, 1);
    assert_eq!(bundle.model_name, "legacy_model");
    assert_eq!(bundle.len(), 1);
    assert!(bundle.validate_all().is_ok());
    // v2 fields absent in v1 JSON
    assert!(bundle.certificates[0].layer_bounds.is_none());
    assert!(bundle.certificates[0].kani_status.is_none());
}

// ---------------------------------------------------------------------------
// SHA-256 fingerprinting tests
// ---------------------------------------------------------------------------

#[test]
fn test_compute_bytes_hash_known_value() {
    // SHA-256("hello") is well-known.
    let hash = compute_bytes_hash(b"hello");
    assert_eq!(
        hash,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn test_compute_bytes_hash_empty() {
    // SHA-256("") is also well-known.
    let hash = compute_bytes_hash(b"");
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_compute_bytes_hash_deterministic() {
    let data = b"nn proof certificate test data";
    let h1 = compute_bytes_hash(data);
    let h2 = compute_bytes_hash(data);
    assert_eq!(h1, h2);
}

#[test]
fn test_compute_file_hash_matches_bytes_hash() {
    let dir = std::env::temp_dir().join(format!("nn_hash_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_data.bin");
    let data = b"some test content for hashing";
    std::fs::write(&path, data).expect("write test file");

    let file_hash = compute_file_hash(&path).expect("compute file hash");
    let bytes_hash = compute_bytes_hash(data);
    assert_eq!(file_hash, bytes_hash);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_compute_file_hash_nonexistent() {
    let result = compute_file_hash(Path::new("/nonexistent/path/to/file"));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// KaniOutcome / KaniProofRecord serde
// ---------------------------------------------------------------------------

#[test]
fn test_kani_outcome_roundtrip() {
    for outcome in [
        KaniOutcome::Passed,
        KaniOutcome::Failed,
        KaniOutcome::NotRun,
        KaniOutcome::Timeout,
    ] {
        let json = serde_json::to_string(&outcome).expect("serialize");
        let parsed: KaniOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(outcome, parsed);
    }
}

#[test]
fn test_kani_proof_record_roundtrip() {
    let record = sample_kani_record();
    let json = serde_json::to_string_pretty(&record).expect("serialize");
    let parsed: KaniProofRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(record, parsed);
}

// ---------------------------------------------------------------------------
// LayerBoundRecord serde
// ---------------------------------------------------------------------------

#[test]
fn test_layer_bound_record_roundtrip() {
    let bounds = sample_layer_bounds();
    let json = serde_json::to_string_pretty(&bounds).expect("serialize");
    let parsed: Vec<LayerBoundRecord> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(bounds, parsed);
}

// ---------------------------------------------------------------------------
// v5: PrecisionModel (#3023)
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_precision_model_default_none() {
    let cert = ProofCertificate::from_verification(&sample_verification(), sample_input_spec());
    assert!(cert.precision_model.is_none());
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_with_precision_model_f32_only() {
    let cert = ProofCertificate::from_verification(&sample_verification(), sample_input_spec())
        .with_precision_model(PrecisionModel::F32Only);
    assert_eq!(cert.precision_model, Some(PrecisionModel::F32Only));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_certificate_with_precision_model_f16_aware() {
    let model = PrecisionModel::F16Aware {
        cast_count: 4,
        total_epsilon: 0.004,
    };
    let cert = ProofCertificate::from_verification(&sample_verification(), sample_input_spec())
        .with_precision_model(model.clone());
    assert_eq!(cert.precision_model, Some(model));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_precision_model_serde_roundtrip_f32_only() {
    let model = PrecisionModel::F32Only;
    let json = serde_json::to_string(&model).expect("serialize");
    let parsed: PrecisionModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(model, parsed);
}

#[test]
fn test_precision_model_serde_roundtrip_f16_aware() {
    let model = PrecisionModel::F16Aware {
        cast_count: 6,
        total_epsilon: 0.006,
    };
    let json = serde_json::to_string(&model).expect("serialize");
    let parsed: PrecisionModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(model, parsed);
}

#[test]
fn test_certificate_precision_model_absent_in_v4_json() {
    // Pre-v5 JSON without precision_model should deserialize with None.
    let cert = ProofCertificate::from_verification(&sample_verification(), sample_input_spec());
    let json = cert.to_json().expect("serialize");
    // Remove precision_model if present (it won't be since it's None + skip_serializing)
    assert!(!json.contains("precision_model"));
    let parsed: ProofCertificate = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.precision_model.is_none());
}
