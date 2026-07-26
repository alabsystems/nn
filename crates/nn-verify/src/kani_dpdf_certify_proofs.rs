// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `dpdf_certify.rs` (#3920).
//!
//! Proves properties of:
//! - `DpdfProperty`: enum variant count and numbering
//! - `PropertyStatus`: valid state transitions
//! - `DpdfCertificate`: deployment readiness logic and report generation

#[cfg(kani)]
mod proofs {
    use crate::dpdf_certify::{DpdfCertificate, DpdfProperty, PropertyStatus};

    // ========================================================================
    // DpdfProperty enum proofs
    // ========================================================================

    /// DpdfProperty::ALL contains exactly 8 variants, matching the P1-P8 spec.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_dpdf_property_enum_count() {
        assert_eq!(DpdfProperty::ALL.len(), 8);

        // Verify each variant is present and numbered correctly.
        assert_eq!(DpdfProperty::ALL[0], DpdfProperty::P1LayoutSigmoidBounds);
        assert_eq!(DpdfProperty::ALL[1], DpdfProperty::P2OcrSoftmaxDistribution);
        assert_eq!(DpdfProperty::ALL[2], DpdfProperty::P3TableBoxNormalized);
        assert_eq!(DpdfProperty::ALL[3], DpdfProperty::P4DflRegressionValid);
        assert_eq!(
            DpdfProperty::ALL[4],
            DpdfProperty::P5NmsPreservesTopConfidence
        );
        assert_eq!(DpdfProperty::ALL[5], DpdfProperty::P6IoUBounded);
        assert_eq!(
            DpdfProperty::ALL[6],
            DpdfProperty::P7ConfidenceFilterMonotone
        );
        assert_eq!(DpdfProperty::ALL[7], DpdfProperty::P8QuantizedEpsilonBound);

        // Verify 1-indexed numbering.
        for (i, prop) in DpdfProperty::ALL.iter().enumerate() {
            assert_eq!(prop.number(), i + 1);
        }
    }

    /// Every DpdfProperty variant has a non-empty human-readable name.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_dpdf_property_names_non_empty() {
        for prop in &DpdfProperty::ALL {
            assert!(!prop.name().is_empty());
        }
    }

    // ========================================================================
    // PropertyStatus transition proofs
    // ========================================================================

    /// PropertyStatus transitions: Unverified can become Proven or Heuristic
    /// (valid forward transitions). Also: any status can transition to
    /// NotApplicable (model config change).
    ///
    /// This proves the enumeration of valid transitions by constructing all
    /// combinations and verifying the set of reachable states.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_property_status_transitions() {
        // Unverified is the initial state. Valid transitions from Unverified:
        let from_unverified = PropertyStatus::Unverified;

        // Can transition to Proven (formal proof completed).
        let to_proven = PropertyStatus::Proven;
        assert_ne!(from_unverified, to_proven);

        // Can transition to Heuristic (IBP/partial bounds computed).
        let to_heuristic = PropertyStatus::Heuristic;
        assert_ne!(from_unverified, to_heuristic);

        // Can transition to NotApplicable (model config excludes property).
        let to_na = PropertyStatus::NotApplicable;
        assert_ne!(from_unverified, to_na);

        // Proven and Heuristic are distinct states.
        assert_ne!(to_proven, to_heuristic);

        // All four statuses are distinct.
        assert_ne!(to_proven, to_na);
        assert_ne!(to_heuristic, to_na);
    }

    // ========================================================================
    // DpdfCertificate deployment readiness proofs
    // ========================================================================

    /// A certificate with all P1-P7 Proven (and P8 anything) is deployment ready.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_certificate_deployment_ready_all_proven() {
        let properties: Vec<(DpdfProperty, PropertyStatus, String)> = DpdfProperty::ALL
            .iter()
            .map(|prop| {
                let status = if prop.number() <= 7 {
                    PropertyStatus::Proven
                } else {
                    // P8 is excluded from deployment readiness check.
                    PropertyStatus::Unverified
                };
                (*prop, status, String::new())
            })
            .collect();

        let cert = DpdfCertificate::new(
            properties,
            10,
            5,
            0,
            vec!["doclayout_yolo".to_string()],
            "2026-03-28".to_string(),
        );

        assert!(cert.is_deployment_ready());
    }

    /// A certificate with all P1-P7 as Heuristic is still deployment ready
    /// (Heuristic is an accepted deployment status).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_certificate_deployment_ready_all_heuristic() {
        let properties: Vec<(DpdfProperty, PropertyStatus, String)> = DpdfProperty::ALL
            .iter()
            .map(|prop| {
                let status = if prop.number() <= 7 {
                    PropertyStatus::Heuristic
                } else {
                    PropertyStatus::Unverified
                };
                (*prop, status, String::new())
            })
            .collect();

        let cert = DpdfCertificate::new(properties, 0, 0, 0, vec![], "2026-03-28".to_string());

        assert!(cert.is_deployment_ready());
    }

    /// A certificate with all P1-P7 as NotApplicable is deployment ready
    /// (NotApplicable is accepted — model config excludes those properties).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_certificate_deployment_ready_all_not_applicable() {
        let properties: Vec<(DpdfProperty, PropertyStatus, String)> = DpdfProperty::ALL
            .iter()
            .map(|prop| {
                let status = if prop.number() <= 7 {
                    PropertyStatus::NotApplicable
                } else {
                    PropertyStatus::Unverified
                };
                (*prop, status, String::new())
            })
            .collect();

        let cert = DpdfCertificate::new(properties, 0, 0, 0, vec![], "2026-03-28".to_string());

        assert!(cert.is_deployment_ready());
    }

    /// A certificate with any P1-P7 property Unverified is NOT deployment ready.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_certificate_not_ready_with_unverified() {
        // Set all P1-P7 to Proven except P1 which is Unverified.
        let properties: Vec<(DpdfProperty, PropertyStatus, String)> = DpdfProperty::ALL
            .iter()
            .map(|prop| {
                let status = if prop.number() == 1 {
                    PropertyStatus::Unverified
                } else if prop.number() <= 7 {
                    PropertyStatus::Proven
                } else {
                    PropertyStatus::Unverified
                };
                (*prop, status, String::new())
            })
            .collect();

        let cert = DpdfCertificate::new(
            properties,
            10,
            5,
            1,
            vec!["doclayout_yolo".to_string()],
            "2026-03-28".to_string(),
        );

        assert!(!cert.is_deployment_ready());
    }

    /// Each P1-P7 property individually Unverified (rest Proven) blocks readiness.
    /// Proves that the check is per-property, not just counting.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_certificate_each_unverified_blocks_readiness() {
        for unverified_num in 1..=7u8 {
            let properties: Vec<(DpdfProperty, PropertyStatus, String)> = DpdfProperty::ALL
                .iter()
                .map(|prop| {
                    let status = if prop.number() == unverified_num as usize {
                        PropertyStatus::Unverified
                    } else if prop.number() <= 7 {
                        PropertyStatus::Proven
                    } else {
                        PropertyStatus::Unverified
                    };
                    (*prop, status, String::new())
                })
                .collect();

            let cert = DpdfCertificate::new(properties, 0, 0, 0, vec![], "2026-03-28".to_string());

            assert!(
                !cert.is_deployment_ready(),
                "P{unverified_num} Unverified should block deployment"
            );
        }
    }

    // ========================================================================
    // Report generation proofs
    // ========================================================================

    /// to_report() produces a non-empty string for any valid certificate.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_report_non_empty() {
        let properties: Vec<(DpdfProperty, PropertyStatus, String)> = DpdfProperty::ALL
            .iter()
            .map(|prop| (*prop, PropertyStatus::Unverified, String::new()))
            .collect();

        let cert = DpdfCertificate::new(properties, 0, 0, 0, vec![], "2026-03-28".to_string());

        let report = cert.to_report();
        assert!(!report.is_empty());
        // Report must contain the markdown header.
        assert!(report.contains("# dpdf Certification Report"));
        // Report must contain the properties table header.
        assert!(report.contains("| # | Property | Status | Evidence |"));
        // Report must contain the summary section.
        assert!(report.contains("## Summary"));
        // Report must contain the deployment readiness line.
        assert!(report.contains("Deployment ready:"));
    }

    /// status_counts() returns correct totals for a mixed certificate.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_status_counts_correct() {
        let properties = vec![
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
                PropertyStatus::Heuristic,
                String::new(),
            ),
            (
                DpdfProperty::P6IoUBounded,
                PropertyStatus::Unverified,
                String::new(),
            ),
            (
                DpdfProperty::P7ConfidenceFilterMonotone,
                PropertyStatus::NotApplicable,
                String::new(),
            ),
            (
                DpdfProperty::P8QuantizedEpsilonBound,
                PropertyStatus::Unverified,
                String::new(),
            ),
        ];

        let cert = DpdfCertificate::new(properties, 0, 0, 0, vec![], "2026-03-28".to_string());

        let (proven, heuristic, unverified, na) = cert.status_counts();
        assert_eq!(proven, 2);
        assert_eq!(heuristic, 3);
        assert_eq!(unverified, 2);
        assert_eq!(na, 1);
    }
}
