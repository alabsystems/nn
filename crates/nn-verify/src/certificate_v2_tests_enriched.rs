// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Enriched pipeline and v2 bundle tests for proof certificates.
//!
//! Core v2 builder/validation/hash/serde tests live in `certificate_v2_tests.rs`.

use std::path::PathBuf;

use super::certificate_test_helpers::*;
use super::*;

// ---------------------------------------------------------------------------
// certificate_from_pipeline_enriched tests
// ---------------------------------------------------------------------------

#[test]
fn test_enriched_pipeline_none_equals_basic() {
    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -10.0,
        upper: 10.0,
    }];

    let basic = certificate_from_pipeline(&result, &variable_inputs, &[1.0], None);
    let enriched =
        certificate_from_pipeline_enriched(&result, &variable_inputs, &[1.0], None, None);

    assert_eq!(basic.kernel_name, enriched.kernel_name);
    assert_eq!(basic.method, enriched.method);
    assert_eq!(basic.input_spec, enriched.input_spec);
    assert!(enriched.source_hash.is_none());
    assert!(enriched.weight_hash.is_none());
    assert!(enriched.kani_status.is_none());
    assert!(enriched.layer_bounds.is_none());
    assert!(enriched.verifier_version.is_none());
}

#[test]
fn test_enriched_pipeline_with_source_hash() {
    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -10.0,
        upper: 10.0,
    }];

    // Write a temp file to hash
    let dir = std::env::temp_dir().join(format!("nn_enriched_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let source_path = dir.join("kernel.rs");
    std::fs::write(&source_path, b"fn snake(x: f32) -> f32 { x }").expect("write");

    let enrichment = CertificateEnrichment {
        source_path: Some(source_path.clone()),
        ..Default::default()
    };
    let cert = certificate_from_pipeline_enriched(
        &result,
        &variable_inputs,
        &[1.0],
        None,
        Some(&enrichment),
    );

    assert!(cert.source_hash.is_some());
    let hash = cert.source_hash.as_ref().unwrap();
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    // Hash should match direct computation
    assert_eq!(hash.as_str(), &compute_file_hash(&source_path).unwrap());

    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_enriched_pipeline_with_layer_bounds_and_version() {
    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -10.0,
        upper: 10.0,
    }];

    let enrichment = CertificateEnrichment {
        layer_bounds: Some(sample_layer_bounds()),
        verifier_version: Some("NY 0.4.0".to_string()),
        ..Default::default()
    };
    let cert = certificate_from_pipeline_enriched(
        &result,
        &variable_inputs,
        &[1.0],
        Some("proven"),
        Some(&enrichment),
    );

    assert!(cert.layer_bounds.is_some());
    assert_eq!(cert.layer_bounds.as_ref().unwrap().len(), 2);
    assert_eq!(cert.verifier_version.as_deref(), Some("NY 0.4.0"));
    assert_eq!(cert.smt_outcome.as_deref(), Some("proven"));
    assert!(cert.validate().is_ok());
}

#[test]
fn test_enriched_pipeline_nonexistent_paths_ignored() {
    let result = sample_verification();
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: -10.0,
        upper: 10.0,
    }];

    let enrichment = CertificateEnrichment {
        source_path: Some(PathBuf::from("/nonexistent/kernel.rs")),
        weight_path: Some(PathBuf::from("/nonexistent/weights.safetensors")),
        kani_status_path: Some(PathBuf::from("/nonexistent/kani_status.json")),
        ..Default::default()
    };
    let cert = certificate_from_pipeline_enriched(
        &result,
        &variable_inputs,
        &[1.0],
        None,
        Some(&enrichment),
    );

    // Nonexistent paths should be silently ignored
    assert!(cert.source_hash.is_none());
    assert!(cert.weight_hash.is_none());
    assert!(cert.kani_status.is_none());
    assert!(cert.validate().is_ok());
}

// ---------------------------------------------------------------------------
// v2 bundle save/load roundtrip with all fields
// ---------------------------------------------------------------------------

#[test]
fn test_bundle_v2_save_load_roundtrip() {
    let result = sample_verification();
    let valid_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    let cert = ProofCertificate::from_verification(&result, sample_input_spec())
        .with_layer_bounds(sample_layer_bounds())
        .with_kani_status(sample_kani_record())
        .with_weight_hash(valid_hash.to_string())
        .with_source_hash(valid_hash.to_string())
        .with_verifier_version("NY 0.3.0".to_string());

    let bundle = CertificateBundle::new("v2_model").with_certificate(cert);
    assert!(bundle.validate_all().is_ok());

    let dir = std::env::temp_dir().join(format!("nn_cert_v2_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("v2_test.proof.json");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(path.with_extension("proof.json.tmp"));

    bundle.save(&path).expect("save v2 bundle");
    let loaded = CertificateBundle::load(&path).expect("load v2 bundle");
    assert_eq!(bundle, loaded);

    // Verify v2 fields survived roundtrip
    let loaded_cert = &loaded.certificates[0];
    assert!(loaded_cert.layer_bounds.is_some());
    assert_eq!(loaded_cert.layer_bounds.as_ref().unwrap().len(), 2);
    assert!(loaded_cert.kani_status.is_some());
    assert_eq!(
        loaded_cert.kani_status.as_ref().unwrap().status,
        KaniOutcome::Passed
    );
    assert_eq!(loaded_cert.weight_hash.as_deref(), Some(valid_hash));
    assert_eq!(loaded_cert.source_hash.as_deref(), Some(valid_hash));
    assert_eq!(
        loaded_cert.verifier_version.as_deref(),
        Some("NY 0.3.0")
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
