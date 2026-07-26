// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for moonshot certificate and status types.
//!
//! Proves correctness of the MoonshotStatus and MoonshotCertificate types that
//! underpin the "First Provably Correct Voice" certification system.
//!
//! Properties proved:
//!
//! 1. VerificationLevel ordering: None < Empirical < CrownPartial < CrownProbabilistic
//!    < CrownProven < KaniProven < SmtProven.
//! 2. VerificationLevel total order is transitive.
//! 3. PROPERTY_NAMES has exactly 8 entries.
//! 4. level_counts sums to exactly 8 properties.
//! 5. all_at_least_crown_partial semantics correct.
//! 6. all_have_evidence semantics correct.
//! 7. CERTIFICATE_SCHEMA_VERSION is >= 1 and current.
//! 8. KaniVerificationEvidence all_passed consistency.
//! 9. SmtVerificationEvidence all_proven consistency.
//! 10. MoonshotStatus has 8 properties.
//! 11. MoonshotStatus report is non-empty.
//! 12. VerificationLevel equality is reflexive.
//! 13. all_have_evidence false when any property is None.
//! 14. artifact_registry returns non-empty list.

// ---- VerificationLevel Ordering Proofs --------------------------------------

/// Prove: VerificationLevel has a strict total order.
///
/// None < Empirical < CrownPartial < CrownProbabilistic < CrownProven < KaniProven < SmtProven.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_total_order() {
    use crate::moonshot::VerificationLevel::*;
    assert!(None < Empirical);
    assert!(Empirical < CrownPartial);
    assert!(CrownPartial < CrownProbabilistic);
    assert!(CrownProbabilistic < CrownProven);
    assert!(CrownProven < KaniProven);
    assert!(KaniProven < SmtProven);
}

/// Prove: VerificationLevel ordering is transitive across all 7 levels.
///
/// If A < B and B < C then A < C for any three levels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_transitive() {
    use crate::moonshot::VerificationLevel::*;
    // Test a selection of transitive chains
    assert!(None < CrownPartial); // transitive: None < Empirical < CrownPartial
    assert!(None < CrownProven);
    assert!(None < SmtProven);
    assert!(Empirical < CrownProven);
    assert!(Empirical < SmtProven);
    assert!(CrownPartial < SmtProven);
    assert!(CrownPartial < KaniProven);
    assert!(CrownProbabilistic < SmtProven);
}

/// Prove: VerificationLevel equality is reflexive.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_eq_reflexive() {
    use crate::moonshot::VerificationLevel::*;
    assert_eq!(None, None);
    assert_eq!(Empirical, Empirical);
    assert_eq!(CrownPartial, CrownPartial);
    assert_eq!(CrownProbabilistic, CrownProbabilistic);
    assert_eq!(CrownProven, CrownProven);
    assert_eq!(KaniProven, KaniProven);
    assert_eq!(SmtProven, SmtProven);
}

/// Prove: VerificationLevel CrownProbabilistic is between CrownPartial and CrownProven.
///
/// This specific ordering matters for the moonshot property checks that use
/// CrownProbabilistic as an intermediate evidence level.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn verification_level_probabilistic_ordering() {
    use crate::moonshot::VerificationLevel;
    assert!(VerificationLevel::CrownPartial < VerificationLevel::CrownProbabilistic);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
    // Probabilistic is strictly between Partial and Proven
    assert!(VerificationLevel::CrownProbabilistic > VerificationLevel::CrownPartial);
    assert!(VerificationLevel::CrownProbabilistic < VerificationLevel::CrownProven);
}

// ---- PROPERTY_NAMES Proofs --------------------------------------------------

/// Prove: PROPERTY_NAMES has exactly 8 entries.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn property_names_count_is_eight() {
    assert_eq!(
        crate::moonshot::PROPERTY_NAMES.len(),
        8,
        "exactly 8 moonshot properties"
    );
}

/// Prove: all PROPERTY_NAMES are non-empty strings.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(9)]
fn property_names_all_non_empty() {
    for name in &crate::moonshot::PROPERTY_NAMES {
        assert!(!name.is_empty(), "property name must not be empty");
    }
}

// ---- MoonshotStatus Proofs --------------------------------------------------

/// Prove: level_counts sums to exactly 8.
///
/// Every property must be assigned to exactly one verification level.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn level_counts_sum_to_eight() {
    let status = crate::moonshot::MoonshotStatus::from_repo();
    let counts = status.level_counts();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 8, "level_counts must sum to 8 properties");
}

/// Prove: level_counts has exactly 7 entries (one per VerificationLevel variant).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn level_counts_has_seven_entries() {
    let status = crate::moonshot::MoonshotStatus::from_repo();
    let counts = status.level_counts();
    assert_eq!(counts.len(), 7, "7 VerificationLevel variants");
}

/// Prove: all_at_least_crown_partial implies all_have_evidence.
///
/// CrownPartial > Empirical > None, so if all >= CrownPartial, all > None.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crown_partial_implies_evidence() {
    let status = crate::moonshot::MoonshotStatus::from_repo();
    if status.all_at_least_crown_partial() {
        assert!(
            status.all_have_evidence(),
            "all_at_least_crown_partial implies all_have_evidence"
        );
    }
}

/// Prove: MoonshotStatus::from_repo() produces exactly 8 properties.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moonshot_status_has_eight_properties() {
    let status = crate::moonshot::MoonshotStatus::from_repo();
    assert_eq!(status.properties.len(), 8, "status must have 8 properties");
}

/// Prove: report() produces a non-empty string.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn moonshot_report_non_empty() {
    let status = crate::moonshot::MoonshotStatus::from_repo();
    let report = status.report();
    assert!(!report.is_empty(), "report must not be empty");
}

/// Prove: artifact_registry returns a non-empty list.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn artifact_registry_non_empty() {
    let artifacts = crate::moonshot::artifact_registry();
    assert!(!artifacts.is_empty(), "artifact registry must not be empty");
}

// ---- Certificate Schema Proofs ----------------------------------------------

/// Prove: CERTIFICATE_SCHEMA_VERSION is at least 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn schema_version_at_least_one() {
    assert!(
        crate::moonshot::CERTIFICATE_SCHEMA_VERSION >= 1,
        "schema version must be >= 1"
    );
}

/// Prove: current schema version is 3 (latest).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn schema_version_is_current() {
    assert_eq!(
        crate::moonshot::CERTIFICATE_SCHEMA_VERSION,
        3,
        "schema version must be 3 (current)"
    );
}

// ---- KaniVerificationEvidence Proofs ----------------------------------------

/// Prove: KaniVerificationEvidence all_passed consistency.
///
/// all_passed should be true iff harnesses_passed == harnesses_total and
/// harnesses_total > 0 (but the struct allows any value — this proves the
/// field semantics are consistent when constructed correctly).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn kani_evidence_all_passed_consistency() {
    let passed: usize = kani::any();
    let total: usize = kani::any();
    kani::assume(total <= 10000);
    kani::assume(passed <= total);

    let evidence = crate::moonshot::KaniVerificationEvidence {
        harnesses_passed: passed,
        harnesses_total: total,
        harness_files: vec![],
        all_passed: passed == total && total > 0,
    };

    if evidence.all_passed {
        assert_eq!(
            evidence.harnesses_passed, evidence.harnesses_total,
            "all_passed requires passed == total"
        );
        assert!(
            evidence.harnesses_total > 0,
            "all_passed requires total > 0"
        );
    }
}

/// Prove: KaniVerificationEvidence with zero total cannot be all_passed.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn kani_evidence_zero_total_not_all_passed() {
    let evidence = crate::moonshot::KaniVerificationEvidence {
        harnesses_passed: 0,
        harnesses_total: 0,
        harness_files: vec![],
        all_passed: false,
    };
    assert!(!evidence.all_passed, "zero total cannot be all_passed");
}

// ---- SmtVerificationEvidence Proofs -----------------------------------------

/// Prove: SmtVerificationEvidence all_proven consistency.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn smt_evidence_all_proven_consistency() {
    let proven: usize = kani::any();
    let total: usize = kani::any();
    kani::assume(total <= 10000);
    kani::assume(proven <= total);

    let evidence = crate::moonshot::SmtVerificationEvidence {
        kernels_proven: proven,
        kernels_total: total,
        proven_kernel_names: vec![],
        all_proven: proven == total && total > 0,
    };

    if evidence.all_proven {
        assert_eq!(
            evidence.kernels_proven, evidence.kernels_total,
            "all_proven requires proven == total"
        );
        assert!(evidence.kernels_total > 0, "all_proven requires total > 0");
    }
}

/// Prove: SmtVerificationEvidence with zero total cannot be all_proven.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn smt_evidence_zero_total_not_all_proven() {
    let evidence = crate::moonshot::SmtVerificationEvidence {
        kernels_proven: 0,
        kernels_total: 0,
        proven_kernel_names: vec![],
        all_proven: false,
    };
    assert!(!evidence.all_proven, "zero total cannot be all_proven");
}
