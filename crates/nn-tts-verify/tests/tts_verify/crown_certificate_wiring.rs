// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for CROWN certificate production wiring.
//!
//! Verifies that:
//! 1. Certificate is populated during synthesis with correct bound checks.
//! 2. Junction contracts are evaluated at stage boundaries.
//! 3. CROWN bounds propagation produces non-trivial (non-vacuous) bounds.
//! 4. `verify_synthesis_crown_full()` correctly wires moonshot + junctions.
//!
//! Part of #4254.

use std::collections::HashMap;

use nn_tts_verify::certificate::Certificate;
use nn_tts_verify::crown_junction::{
    check_all_junction_contracts, check_junction_bound, contract_bounds_map,
    verify_crown_with_junction_checks,
};
use nn_tts_verify::crown_synthesis::{
    verify_synthesis_crown, verify_synthesis_crown_full, CrownCertificateConfig,
};
use nn_tts_verify::error::TtsVerifyError;
use nn_tts_verify::moonshot::{MoonshotCertificate, MoonshotStatus, VerificationLevel};
use nn_tts_verify::TtsVerifier;

// ---------------------------------------------------------------------------
// Helper: generate a 24 kHz multi-frequency signal
// ---------------------------------------------------------------------------

fn test_audio(duration_sec: f64) -> Vec<f32> {
    let sample_rate = 24000;
    let n = (f64::from(sample_rate) * duration_sec) as usize;
    // Multi-frequency signal: 440 + 880 + 2000 + 5000 Hz
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            let pi2 = 2.0 * std::f64::consts::PI;
            let s = 0.15 * (pi2 * 440.0 * t).sin()
                + 0.10 * (pi2 * 880.0 * t).sin()
                + 0.08 * (pi2 * 2000.0 * t).sin()
                + 0.05 * (pi2 * 5000.0 * t).sin();
            s as f32
        })
        .collect()
}

/// Extract the Certificate from a verify() result regardless of pass/fail.
///
/// The verifier returns `Err(VerificationRejected { cert })` when hard bounds
/// fail under the default `Reject` policy. We extract the cert either way
/// since CROWN enrichment works on both passing and failing certificates.
fn extract_certificate(result: Result<Certificate, TtsVerifyError>) -> Certificate {
    match result {
        Ok(cert) => cert,
        Err(TtsVerifyError::VerificationRejected { cert }) => *cert,
        Err(e) => panic!("unexpected verification error: {e:?}"),
    }
}

/// Create a verifier at 24 kHz with default hard bounds.
fn default_verifier() -> TtsVerifier {
    TtsVerifier::builder()
        .build()
        .expect("valid verifier config")
}

/// Build intermediates that are within all junction contract bounds.
fn within_bounds_intermediates() -> HashMap<String, (f32, f32)> {
    let mut intermediates = HashMap::new();
    intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
    intermediates.insert("J2_ENERGY".to_string(), (-10.0_f32, 10.0_f32));
    intermediates.insert("J3_MAGNITUDE".to_string(), (-40.0_f32, 40.0_f32));
    intermediates.insert("J3B_PHASE".to_string(), (-3000.0_f32, 3000.0_f32));
    intermediates.insert("J4_BF16".to_string(), (-64.0_f32, 64.0_f32));
    intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));
    intermediates
}

/// Build intermediates where J5_AUDIO violates its upper bound.
fn violating_intermediates() -> HashMap<String, (f32, f32)> {
    let mut intermediates = within_bounds_intermediates();
    // J5_AUDIO upper bound is 1.0; set actual to 1.5 to violate.
    intermediates.insert("J5_AUDIO".to_string(), (-0.5_f32, 1.5_f32));
    intermediates
}

// ---------------------------------------------------------------------------
// Test: Certificate populated with CROWN evidence
// ---------------------------------------------------------------------------

#[test]
fn test_crown_evidence_populated_on_synthesis_certificate() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let config = CrownCertificateConfig::default();
    let moonshot = verify_synthesis_crown(&cert, &config);

    // Enrichment from hard bounds should map P1, P2, P6.
    assert!(
        moonshot.properties[0].level >= VerificationLevel::Empirical,
        "P1 (non-silence) should be at least Empirical"
    );
    assert!(
        moonshot.properties[1].level >= VerificationLevel::Empirical,
        "P2 (non-clipping) should be at least Empirical"
    );
    assert!(
        moonshot.properties[5].level >= VerificationLevel::Empirical,
        "P6 (streaming-safe proxy) should be at least Empirical"
    );

    // Attach and verify round-trip.
    let enriched = cert.with_crown_evidence(moonshot);
    assert!(enriched.has_crown_evidence());
    let evidence = enriched.crown_evidence.as_ref().unwrap();
    assert_eq!(evidence.properties.len(), 8, "moonshot has 8 properties");
}

// ---------------------------------------------------------------------------
// Test: Junction contracts evaluated at stage boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_junction_contracts_all_pass_within_bounds() {
    let intermediates = within_bounds_intermediates();
    let checks = check_all_junction_contracts(&intermediates);

    assert_eq!(checks.len(), 6, "all 6 contracts should be checked");
    for check in &checks {
        assert!(
            check.passed,
            "junction {} should pass: actual [{}, {}] vs expected [{}, {}]",
            check.junction_name,
            check.actual_lower,
            check.actual_upper,
            check.expected_lower,
            check.expected_upper
        );
    }
}

#[test]
fn test_junction_contracts_detect_violation() {
    let intermediates = violating_intermediates();
    let checks = check_all_junction_contracts(&intermediates);

    let audio_check = checks
        .iter()
        .find(|c| c.junction_name == "J5_AUDIO")
        .expect("J5_AUDIO should be checked");
    assert!(
        !audio_check.passed,
        "J5_AUDIO should fail when actual_upper=1.5 > expected_upper=1.0"
    );

    // Other checks should still pass.
    let other_checks: Vec<_> = checks
        .iter()
        .filter(|c| c.junction_name != "J5_AUDIO")
        .collect();
    assert!(
        other_checks.iter().all(|c| c.passed),
        "all non-J5_AUDIO checks should pass"
    );
}

// ---------------------------------------------------------------------------
// Test: CROWN bounds are non-trivial (non-vacuous)
// ---------------------------------------------------------------------------

#[test]
fn test_crown_bounds_non_vacuous() {
    // Contract bounds should not be vacuously wide (e.g., [-inf, inf]).
    let bounds_map = contract_bounds_map();

    assert!(
        !bounds_map.is_empty(),
        "contract bounds map should not be empty"
    );

    for (name, (lower, upper)) in &bounds_map {
        assert!(
            lower.is_finite(),
            "contract {name} lower bound should be finite, got {lower}"
        );
        assert!(
            upper.is_finite(),
            "contract {name} upper bound should be finite, got {upper}"
        );
        assert!(
            lower < upper,
            "contract {name} should have lower < upper: {lower} < {upper}"
        );

        // Non-trivial: bounds should not span the entire f32 range.
        let span = upper - lower;
        assert!(
            span < 1e10,
            "contract {name} span {span} is vacuously wide"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: verify_synthesis_crown_full wires both moonshot + junctions
// ---------------------------------------------------------------------------

#[test]
fn test_verify_synthesis_crown_full_without_junction_checking() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let config = CrownCertificateConfig::default();
    // check_junction_contracts defaults to false.
    assert!(!config.check_junction_contracts);

    let result = verify_synthesis_crown_full(&cert, &config, None);
    assert!(
        result.junction_summary.is_none(),
        "junction summary should be None when check_junction_contracts=false"
    );
    // Moonshot should still be valid.
    assert_eq!(result.moonshot.properties.len(), 8);
}

#[test]
fn test_verify_synthesis_crown_full_with_junction_checking_pass() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::default()
    };
    let intermediates = within_bounds_intermediates();

    let result = verify_synthesis_crown_full(&cert, &config, Some(&intermediates));

    // Junction summary should be present and all pass.
    let summary = result
        .junction_summary
        .expect("junction summary should be present");
    assert_eq!(summary.total_failed, 0, "all junctions should pass");
    assert_eq!(summary.total_passed, 6, "all 6 junctions should pass");
    assert_eq!(summary.checks.len(), 6);
}

#[test]
fn test_verify_synthesis_crown_full_with_junction_checking_fail() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::default()
    };
    let intermediates = violating_intermediates();

    let result = verify_synthesis_crown_full(&cert, &config, Some(&intermediates));

    let summary = result
        .junction_summary
        .expect("junction summary should be present");
    assert_eq!(
        summary.total_failed, 1,
        "exactly one junction (J5_AUDIO) should fail"
    );
    assert_eq!(summary.total_passed, 5, "5 junctions should pass");
}

#[test]
fn test_verify_synthesis_crown_full_junction_enabled_but_no_intermediates() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::default()
    };

    // When intermediates is None, junction_summary should be None.
    let result = verify_synthesis_crown_full(&cert, &config, None);
    assert!(
        result.junction_summary.is_none(),
        "junction summary should be None when no intermediates provided"
    );
}

// ---------------------------------------------------------------------------
// Test: Certificate.with_junction_summary roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_certificate_with_junction_summary_roundtrip() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let intermediates = within_bounds_intermediates();
    let status = MoonshotStatus::from_repo();
    let moonshot =
        MoonshotCertificate::from_status(&status, "kokoro-v1", "English text", "test-hash");
    let summary = verify_crown_with_junction_checks(&moonshot, &intermediates);

    let enriched = cert.with_junction_summary(summary);
    assert!(enriched.has_junction_summary());
    assert!(enriched.passes_junction_contracts());

    let report = enriched.report();
    assert!(
        report.contains("Junction Contract Checks"),
        "report should contain junction section"
    );
    assert!(
        report.contains("6/6 contracts passed"),
        "report should show all 6 contracts passed"
    );
}

#[test]
fn test_certificate_junction_failure_in_report() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let intermediates = violating_intermediates();
    let status = MoonshotStatus::from_repo();
    let moonshot =
        MoonshotCertificate::from_status(&status, "kokoro-v1", "English text", "test-hash");
    let summary = verify_crown_with_junction_checks(&moonshot, &intermediates);

    let enriched = cert.with_junction_summary(summary);
    assert!(enriched.has_junction_summary());
    assert!(
        !enriched.passes_junction_contracts(),
        "should fail when J5_AUDIO violated"
    );

    let report = enriched.report();
    assert!(
        report.contains("[FAIL] J5_AUDIO"),
        "report should contain FAIL for J5_AUDIO"
    );
}

// ---------------------------------------------------------------------------
// Test: Full enrichment chain (crown_evidence + junction_summary)
// ---------------------------------------------------------------------------

#[test]
fn test_full_enrichment_chain() {
    let audio = test_audio(0.5);
    let verifier = default_verifier();
    let cert = extract_certificate(verifier.verify(&audio));

    let config = CrownCertificateConfig {
        check_junction_contracts: true,
        ..CrownCertificateConfig::default()
    };
    let intermediates = within_bounds_intermediates();

    let result = verify_synthesis_crown_full(&cert, &config, Some(&intermediates));

    // Build the enriched certificate the same way compiled_kokoro_steps does.
    let mut enriched = cert.with_crown_evidence(result.moonshot);
    if let Some(summary) = result.junction_summary {
        enriched = enriched.with_junction_summary(summary);
    }

    assert!(enriched.has_crown_evidence());
    assert!(enriched.has_junction_summary());
    assert!(enriched.passes_junction_contracts());

    let report = enriched.report();
    // Report should contain both sections.
    assert!(
        report.contains("CROWN Verification Evidence"),
        "report should contain CROWN section"
    );
    assert!(
        report.contains("Junction Contract Checks"),
        "report should contain junction section"
    );
    assert!(
        report.contains("6/6 contracts passed"),
        "all 6 junction contracts should pass"
    );
}

// ---------------------------------------------------------------------------
// Test: Individual junction bound edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_junction_bound_exact_boundary() {
    // Exact boundary: actual == expected should pass.
    let check = check_junction_bound("J5_AUDIO", -1.0, 1.0, -1.0, 1.0);
    assert!(check.passed, "exact boundary values should pass");
}

#[test]
fn test_junction_bound_just_outside() {
    // Just outside: actual_upper slightly above expected_upper.
    let check = check_junction_bound("J5_AUDIO", -1.0, 1.0, -0.5, 1.0001);
    assert!(!check.passed, "slightly over upper bound should fail");
}

#[test]
fn test_junction_bound_nan_in_actual() {
    let check = check_junction_bound("J2_F0", -5.0, 800.0, f32::NAN, 500.0);
    assert!(!check.passed, "NaN in actual_lower should fail");
}

#[test]
fn test_junction_bound_infinity_in_expected() {
    let check = check_junction_bound("J2_F0", f32::NEG_INFINITY, 800.0, 0.0, 500.0);
    assert!(!check.passed, "infinity in expected_lower should fail");
}
