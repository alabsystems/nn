// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certificate generation test for full Silero VAD model.
//!
//! Extracted from `compose_silero_vad_full.rs` (#1437) to prevent cargo
//! test timeout when running the full suite. This test runs CROWN propagation
//! with `collect_layer_bounds=true` which takes ~485s — well within the
//! per-binary timeout but causes suite-level timeout when combined with
//! the other 7 VAD full tests.

#[path = "silero_vad_test_helpers.rs"]
mod silero_vad_test_helpers;

use crate::silero_vad_test_helpers::{
    build_full_silero_vad, full_model_bindings, stft_input_bounds,
};
use nn_verify::{
    certificate_from_pipeline_enriched, check_certificate, verify_tensor_and_record_with_config,
    CertificateEnrichment, ParamInputRecord, VerifyConfig, VerifyStatus,
};

/// Generate a proof certificate from the full VAD model tensor pipeline and
/// validate it with the independent checker.
///
/// Exercises the end-to-end path: tensor pipeline with `collect_layer_bounds`
/// → `certificate_from_pipeline_enriched` → `check_certificate` validation.
/// This is the tensor-pipeline analog of verify_all's scalar certificate path.
#[test]
fn test_full_vad_certificate_from_tensor_pipeline() {
    let def = build_full_silero_vad();
    let bindings = full_model_bindings();
    let input = stft_input_bounds();

    let mut status = VerifyStatus::default();
    let config = VerifyConfig::default().with_collect_layer_bounds(true);
    let result = verify_tensor_and_record_with_config(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("silero_vad_full_cert"),
        &config,
    )
    .expect("tensor pipeline with layer bounds");

    // Layer bounds should be populated.
    assert!(
        result.layer_bounds.is_some(),
        "layer_bounds must be Some when collect_layer_bounds is true"
    );
    let layer_bounds = result.layer_bounds.as_ref().unwrap();
    assert!(
        !layer_bounds.is_empty(),
        "layer_bounds must not be empty for a multi-layer model"
    );

    // Build a certificate from the tensor pipeline result.
    let variable_inputs = vec![ParamInputRecord::new(0, 0.0, 10.0)];
    let enrichment = CertificateEnrichment {
        layer_bounds: result.layer_bounds.clone(),
        verifier_version: Some("test".to_string()),
        ..CertificateEnrichment::default()
    };
    let mut cert = certificate_from_pipeline_enriched(
        &result.verification,
        &variable_inputs,
        &[],
        None,
        Some(&enrichment),
    );
    cert.kernel_name = "silero_vad_full_cert".to_string();
    // Provide a test source_hash — no real source file exists in tests, but the
    // independent checker requires it (tightened in #1683 soundness gap fixes).
    cert.source_hash =
        Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string());

    // Validate the certificate structurally.
    cert.validate()
        .expect("certificate must be structurally valid");

    // Run the independent checker — should pass all checks.
    let check_result = check_certificate(&cert, None, None);
    assert!(
        check_result.is_valid(),
        "certificate checker found issues: {:?}",
        check_result.issues
    );
}
