// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::moonshot::*;

// --- P7 (Kani) and P8 (SMT) certificate enrichment tests ---

#[test]
fn test_certificate_with_kani_results_all_passed() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let evidence = KaniVerificationEvidence {
        harnesses_passed: 475,
        harnesses_total: 475,
        harness_files: vec![
            "crates/nn-core/src/kani_bounds.rs".to_string(),
            "crates/nn-autodiff/src/kani_backward_proofs.rs".to_string(),
        ],
        all_passed: true,
    };

    let enriched = cert.with_kani_results(&evidence);
    assert_eq!(enriched.properties[6].level, VerificationLevel::KaniProven);
    assert_eq!(enriched.properties[6].bound_value, Some(475.0));
    assert_eq!(enriched.properties[6].threshold, Some(475.0));
    assert_eq!(enriched.properties[6].proof_artifacts.len(), 2);
    assert!(enriched.properties[6].assumptions[0].contains("475 Kani harnesses pass"));
}

#[test]
fn test_certificate_with_kani_results_partial() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let evidence = KaniVerificationEvidence {
        harnesses_passed: 400,
        harnesses_total: 475,
        harness_files: vec!["crates/nn-core/src/kani_bounds.rs".to_string()],
        all_passed: false,
    };

    let enriched = cert.with_kani_results(&evidence);
    assert_eq!(enriched.properties[6].level, VerificationLevel::Empirical);
    assert!(enriched.properties[6].assumptions[0].contains("400/475"));
}

#[test]
fn test_certificate_with_kani_results_zero_harnesses() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let evidence = KaniVerificationEvidence {
        harnesses_passed: 0,
        harnesses_total: 0,
        harness_files: vec![],
        all_passed: true, // vacuously true
    };

    // all_passed=true but harnesses_total=0 → None (no evidence).
    let enriched = cert.with_kani_results(&evidence);
    assert_eq!(enriched.properties[6].level, VerificationLevel::None);
}

#[test]
fn test_certificate_with_smt_results_all_proven() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let evidence = SmtVerificationEvidence {
        kernels_proven: 15,
        kernels_total: 15,
        proven_kernel_names: vec!["snake".to_string(), "silu_mul".to_string()],
        all_proven: true,
    };

    let enriched = cert.with_smt_results(&evidence);
    assert_eq!(enriched.properties[7].level, VerificationLevel::SmtProven);
    assert_eq!(enriched.properties[7].bound_value, Some(15.0));
    assert_eq!(enriched.properties[7].threshold, Some(15.0));
    assert!(enriched.properties[7].assumptions[0].contains("15 kernel proofs"));
    // Artifact paths should be ay paths.
    assert!(enriched.properties[7].proof_artifacts[0].contains("ay/snake"));
}

#[test]
fn test_certificate_with_smt_results_partial() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let evidence = SmtVerificationEvidence {
        kernels_proven: 10,
        kernels_total: 15,
        proven_kernel_names: vec!["snake".to_string()],
        all_proven: false,
    };

    let enriched = cert.with_smt_results(&evidence);
    assert_eq!(enriched.properties[7].level, VerificationLevel::Empirical);
    assert!(enriched.properties[7].assumptions[0].contains("10/15"));
}

#[test]
fn test_certificate_with_smt_results_zero_kernels() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    let evidence = SmtVerificationEvidence {
        kernels_proven: 0,
        kernels_total: 0,
        proven_kernel_names: vec![],
        all_proven: true, // vacuously true
    };

    // all_proven=true but kernels_total=0 → None (no evidence).
    // Mirrors the Kani zero-harness test for symmetric coverage.
    let enriched = cert.with_smt_results(&evidence);
    assert_eq!(enriched.properties[7].level, VerificationLevel::None);
}

#[test]
fn test_certificate_all_proven_with_p7_p8() {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(&status, "test", "test", "hash");

    // Start with all P1-P6 at CrownProven by manually setting levels.
    let mut cert = cert;
    for i in 0..6 {
        cert.properties[i].level = VerificationLevel::CrownProven;
    }

    let kani = KaniVerificationEvidence {
        harnesses_passed: 475,
        harnesses_total: 475,
        harness_files: vec![],
        all_passed: true,
    };
    let smt = SmtVerificationEvidence {
        kernels_proven: 15,
        kernels_total: 15,
        proven_kernel_names: vec![],
        all_proven: true,
    };

    let enriched = cert.with_kani_results(&kani).with_smt_results(&smt);
    assert!(enriched.all_proven, "All 8 properties should be proven");

    // Verify JSON includes P7 and P8 entries.
    let json = enriched.to_json();
    assert!(json.contains("Memory-safe"));
    assert!(json.contains("Correct implementation"));
}
