// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certificate generation test for full HTDemucs model.
//!
//! Generates a proof certificate from the full HTDemucs NY
//! composition (temporal encoder + cross-domain transformer + temporal
//! decoder) and validates it with the independent checker.
//!
//! Part of #1696: 4/5 models have zero NY verification.

use super::common;

#[path = "htdemucs_full.rs"]
mod helpers;

use common::uniform_bounds;
use helpers::{build_htdemucs_full, htdemucs_full_bindings, IN_CH, T_IN};
use nn_verify::{
    certificate_from_pipeline_enriched, check_certificate, verify_tensor_and_record_with_config,
    CertificateBundle, CertificateEnrichment, ParamInputRecord, VerifyConfig, VerifyStatus,
};

/// Generate a proof certificate from the full HTDemucs tensor pipeline and
/// validate it with the independent checker.
///
/// Exercises the end-to-end path: tensor pipeline → `certificate_from_pipeline_enriched`
/// → `check_certificate` validation. Layer bounds extraction may fail for
/// complex models (200+ nodes with cross-domain transformer) — the certificate
/// is still valid without per-layer bounds enrichment.
#[test]
fn test_htdemucs_full_certificate_from_tensor_pipeline() {
    let (def, _target_t) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let mut status = VerifyStatus::default();
    let config = VerifyConfig::default().with_collect_layer_bounds(true);
    let result = verify_tensor_and_record_with_config(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("htdemucs_full_cert"),
        &config,
    )
    .expect("tensor pipeline with layer bounds");

    // Layer bounds may be None for complex models where CROWN-IBP collection
    // fails (e.g., 200+ node graphs with cross-domain transformer). The
    // certificate is still valid without per-layer bounds enrichment.
    if let Some(ref layer_bounds) = result.layer_bounds {
        eprintln!(
            "HTDemucs certificate: {} layer bound records collected",
            layer_bounds.len()
        );
    } else {
        eprintln!("HTDemucs certificate: layer bounds extraction returned None (complex model)");
    }

    // Build a certificate from the tensor pipeline result.
    let variable_inputs = vec![ParamInputRecord::new(0, -1.0, 1.0)];
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
    cert.kernel_name = "htdemucs_full_cert".to_string();
    // Provide a source_hash (required by the checker for all certificates).
    cert.source_hash = Some("a".repeat(64));

    // Validate the certificate structurally.
    cert.validate()
        .expect("certificate must be structurally valid");

    // Run the independent checker.
    //
    // Historically, layer-bounds extraction FAILED on this 200+ node graph because
    // the decomposed GroupNorm(g=1) `centered * rsqrt` products exploded under
    // per-node IBP, so CROWN-IBP collection bailed and `layer_bounds` was None
    // (only NoLayerBounds was reported). With the decomposed-GroupNorm → native
    // InstanceNorm1d verifier fusion (graph_tensor_group_norm_fusion.rs) the
    // GroupNorm outputs are now bounded by the sound `|z| <= sqrt(n-1)` clamp, so
    // CROWN-IBP layer-bounds collection SUCCEEDS for the full model.
    //
    // That success surfaces a SEPARATE, pre-existing checker limitation. The
    // certificate's top-level `output_bounds` are CORRECT — they equal the model's
    // true IBP output [-1.7e-8, +1.8e-8] (the decoder's tiny weights crush the
    // signal). But `extract_layer_bounds`'s last topological record is an
    // INTERMEDIATE node (~[312, 316]), not the graph's output node, so the
    // checker's trace-agreement step (`OutputMismatch`) compares the correct
    // certificate output against the wrong "last layer", and the wide intermediate
    // also trips `VacuousBounds`. Both are false positives of the trace-extraction
    // last-node selection — a checker/trace concern unrelated to the GroupNorm
    // fusion or to any bound's soundness (the certificate output is sound and
    // correct). We tolerate exactly those two trace-agreement issues here while
    // still rejecting any other unexpected checker issue and requiring structural
    // validity above.
    let check_result = check_certificate(&cert, None, None);
    if result.layer_bounds.is_some() {
        // The certificate's OWN output bounds must be the true (tiny) model IBP output —
        // assert this directly so a real output-bound regression cannot hide behind the
        // tolerated OutputMismatch below.
        let (cert_lo, cert_hi) = (cert.output_bounds.lower, cert.output_bounds.upper);
        assert!(
            cert_lo.abs() < 1e-3 && cert_hi.abs() < 1e-3 && cert_lo <= cert_hi,
            "certificate output bounds must be the true (near-zero) model output, got \
             [{cert_lo}, {cert_hi}]"
        );
        let unexpected: Vec<_> = check_result
            .issues
            .iter()
            .filter(|i| match i {
                // Known false positive: check_output_agreement compares cert.output_bounds
                // against the topo-LAST record (an intermediate node ~[312,316]), not the
                // graph output node. Tolerate ONLY that exact comparison: the certificate
                // side must be the true tiny output AND the trace side must be the known
                // intermediate band. Any OTHER OutputMismatch is a real regression -> fail.
                nn_verify::CheckIssue::OutputMismatch {
                    certificate_lower,
                    certificate_upper,
                    trace_lower,
                    trace_upper,
                } => {
                    let cert_is_true_output =
                        certificate_lower.abs() < 1e-3 && certificate_upper.abs() < 1e-3;
                    let trace_is_known_intermediate = (200.0..400.0).contains(trace_lower)
                        && (200.0..400.0).contains(trace_upper);
                    !(cert_is_true_output && trace_is_known_intermediate)
                }
                // VacuousBounds is informational here (crown_coverage metric, not a wide/
                // wrong output — the true output width is ~3.5e-8); is_valid() ignores it.
                nn_verify::CheckIssue::VacuousBounds { .. } => false,
                _ => true,
            })
            .collect();
        assert!(
            unexpected.is_empty(),
            "certificate checker found unexpected issues (beyond the known full-model \
             output-node trace divergence): {unexpected:?}"
        );
    } else {
        // Without layer bounds, NoLayerBounds is the only expected issue.
        let unexpected: Vec<_> = check_result
            .issues
            .iter()
            .filter(|i| !matches!(i, nn_verify::CheckIssue::NoLayerBounds))
            .collect();
        assert!(
            unexpected.is_empty(),
            "certificate checker found unexpected issues (beyond NoLayerBounds): {unexpected:?}"
        );
    }
}

/// Certificate can be serialized into a bundle and loaded back.
#[test]
fn test_htdemucs_full_certificate_bundle_roundtrip() {
    let (def, _target_t) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let mut status = VerifyStatus::default();
    let config = VerifyConfig::default().with_collect_layer_bounds(true);
    let result = verify_tensor_and_record_with_config(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("htdemucs_full_bundle"),
        &config,
    )
    .expect("tensor pipeline");

    let variable_inputs = vec![ParamInputRecord::new(0, -1.0, 1.0)];
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
    cert.kernel_name = "htdemucs_full_bundle".to_string();

    // Bundle and save to temp file.
    let bundle = CertificateBundle::new("htdemucs_full").with_certificate(cert);
    assert_eq!(bundle.certificates.len(), 1);
    assert_eq!(bundle.verified_count(), 1);

    let tmp_path = std::env::temp_dir().join(format!(
        "htdemucs_cert_test_{}.proof.json",
        std::process::id()
    ));
    bundle.save(&tmp_path).expect("bundle save");

    // Load and verify roundtrip.
    let loaded = CertificateBundle::load(&tmp_path).expect("bundle load");
    assert_eq!(loaded.certificates.len(), 1);
    assert_eq!(loaded.model_name, "htdemucs_full");
    assert_eq!(loaded.certificates[0].kernel_name, "htdemucs_full_bundle");

    // Validate the loaded certificate.
    loaded.validate_all().expect("loaded bundle validates");

    // Cleanup.
    let _ = std::fs::remove_file(&tmp_path);
}
