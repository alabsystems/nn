// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Junction contract checking for CROWN certificate production.
//!
//! Checks intermediate tensor values against the junction contracts (J2-J5)
//! defined in [`kokoro_contracts`]. Each junction represents a zone crossing
//! in the Kokoro TTS pipeline where verified bounds must hold.
//!
//! # Architecture
//!
//! The junction checker receives a map of named intermediate tensors (with
//! observed min/max bounds) and validates each against the corresponding
//! junction contract. This bridges runtime observations with the formal
//! contract system.
//!
//! # Defense-in-depth
//!
//! All bound comparisons check `is_finite()` before relational operators
//! to prevent IEEE 754 NaN bypass (Source: #3356).
//!
//! Part of #4254.

use std::collections::HashMap;
use std::fmt;

use crate::kokoro_contracts;
use crate::moonshot::MoonshotCertificate;

/// Result of checking a single junction contract bound.
#[derive(Debug, Clone)]
pub struct StageBoundCheck {
    /// Junction contract name (e.g., "J2_F0").
    pub junction_name: String,
    /// Expected lower bound from the contract.
    pub expected_lower: f32,
    /// Expected upper bound from the contract.
    pub expected_upper: f32,
    /// Actual observed lower bound.
    pub actual_lower: f32,
    /// Actual observed upper bound.
    pub actual_upper: f32,
    /// Whether the actual bounds are within the expected contract bounds.
    pub passed: bool,
}

impl fmt::Display for StageBoundCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "[{}] {}: actual [{:.4}, {:.4}] vs expected [{:.4}, {:.4}]",
            status,
            self.junction_name,
            self.actual_lower,
            self.actual_upper,
            self.expected_lower,
            self.expected_upper,
        )
    }
}

/// Aggregated pass/fail summary of junction contract checks.
#[derive(Debug, Clone)]
pub struct JunctionCheckSummary {
    /// Individual check results.
    pub checks: Vec<StageBoundCheck>,
    /// Number of checks that passed.
    pub total_passed: usize,
    /// Number of checks that failed.
    pub total_failed: usize,
}

impl fmt::Display for JunctionCheckSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Junction Contract Summary: {}/{} passed",
            self.total_passed,
            self.total_passed + self.total_failed,
        )?;
        for check in &self.checks {
            writeln!(f, "  {check}")?;
        }
        Ok(())
    }
}

/// Check a single junction contract bound.
///
/// Defense-in-depth: checks `!val.is_finite()` for NaN/Inf before
/// relational comparisons (IEEE 754 NaN bypass, Source: #3356).
#[must_use]
pub fn check_junction_bound(
    junction_name: &str,
    expected_lower: f32,
    expected_upper: f32,
    actual_lower: f32,
    actual_upper: f32,
) -> StageBoundCheck {
    // IEEE 754 defense: NaN/Inf always fail.
    let passed = actual_lower.is_finite()
        && actual_upper.is_finite()
        && expected_lower.is_finite()
        && expected_upper.is_finite()
        && actual_lower >= expected_lower
        && actual_upper <= expected_upper;

    StageBoundCheck {
        junction_name: junction_name.to_string(),
        expected_lower,
        expected_upper,
        actual_lower,
        actual_upper,
        passed,
    }
}

/// Check all Kokoro junction contracts (J2-J5) against named intermediates.
///
/// `intermediates` maps junction contract names (e.g., "J2_F0") to observed
/// `(lower, upper)` bound pairs. Contracts without a matching intermediate
/// entry are skipped.
///
/// # Returns
///
/// A vector of [`StageBoundCheck`] results, one per matched contract.
#[must_use]
pub fn check_all_junction_contracts(
    intermediates: &HashMap<String, (f32, f32)>,
) -> Vec<StageBoundCheck> {
    let contracts = kokoro_contracts::all_contracts();

    contracts
        .iter()
        .filter_map(|contract| {
            intermediates
                .get(contract.name)
                .map(|&(actual_lower, actual_upper)| {
                    check_junction_bound(
                        contract.name,
                        contract.lower as f32,
                        contract.upper as f32,
                        actual_lower,
                        actual_upper,
                    )
                })
        })
        .collect()
}

/// Combine a moonshot certificate with junction contract checks into a summary.
///
/// This is the enriched verification step: it takes the existing
/// [`MoonshotCertificate`] (which tracks the 8 formal properties) and
/// supplements it with junction contract validation on observed intermediates.
///
/// The `_certificate` parameter is accepted for API completeness and future
/// enrichment (e.g., attaching junction results to the certificate). Currently
/// the summary is independent.
#[must_use]
pub fn verify_crown_with_junction_checks(
    _certificate: &MoonshotCertificate,
    intermediates: &HashMap<String, (f32, f32)>,
) -> JunctionCheckSummary {
    let checks = check_all_junction_contracts(intermediates);
    let total_passed = checks.iter().filter(|c| c.passed).count();
    let total_failed = checks.len() - total_passed;

    JunctionCheckSummary {
        checks,
        total_passed,
        total_failed,
    }
}

/// Build a map of junction contract names to their expected bounds.
///
/// Convenience function for constructing the intermediates map from
/// the contract definitions themselves (useful for testing).
#[must_use]
pub fn contract_bounds_map() -> HashMap<String, (f32, f32)> {
    let contracts = kokoro_contracts::all_contracts();
    contracts
        .iter()
        .map(|c| (c.name.to_string(), (c.lower as f32, c.upper as f32)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_junction_bound_within_range() {
        let check = check_junction_bound("J2_F0", -5.0, 800.0, 0.0, 500.0);
        assert!(check.passed, "bounds within range should pass");
        assert_eq!(check.junction_name, "J2_F0");
        assert_eq!(check.expected_lower, -5.0);
        assert_eq!(check.expected_upper, 800.0);
        assert_eq!(check.actual_lower, 0.0);
        assert_eq!(check.actual_upper, 500.0);
    }

    #[test]
    fn test_check_junction_bound_violation() {
        // actual_upper exceeds expected_upper
        let check = check_junction_bound("J5_AUDIO", -1.0, 1.0, -0.5, 1.5);
        assert!(!check.passed, "out-of-range upper should fail");

        // actual_lower below expected_lower
        let check = check_junction_bound("J5_AUDIO", -1.0, 1.0, -2.0, 0.5);
        assert!(!check.passed, "out-of-range lower should fail");
    }

    #[test]
    fn test_check_all_junction_contracts_empty_intermediates() {
        let intermediates = HashMap::new();
        let checks = check_all_junction_contracts(&intermediates);
        assert!(
            checks.is_empty(),
            "no intermediates should produce no checks"
        );
    }

    #[test]
    fn test_check_all_junction_contracts_all_present() {
        // Build intermediates that are all within contract bounds.
        let mut intermediates = HashMap::new();
        intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
        intermediates.insert("J2_ENERGY".to_string(), (-10.0_f32, 10.0_f32));
        intermediates.insert("J3_MAGNITUDE".to_string(), (-40.0_f32, 40.0_f32));
        intermediates.insert("J3B_PHASE".to_string(), (-3000.0_f32, 3000.0_f32));
        intermediates.insert("J4_BF16".to_string(), (-64.0_f32, 64.0_f32));
        intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));

        let checks = check_all_junction_contracts(&intermediates);
        assert_eq!(checks.len(), 6, "all 6 contracts should be checked");
        assert!(
            checks.iter().all(|c| c.passed),
            "all checks should pass for within-bounds intermediates"
        );
    }

    #[test]
    fn test_junction_check_summary_formatting() {
        let summary = JunctionCheckSummary {
            checks: vec![
                StageBoundCheck {
                    junction_name: "J2_F0".to_string(),
                    expected_lower: -5.0,
                    expected_upper: 800.0,
                    actual_lower: 0.0,
                    actual_upper: 500.0,
                    passed: true,
                },
                StageBoundCheck {
                    junction_name: "J5_AUDIO".to_string(),
                    expected_lower: -1.0,
                    expected_upper: 1.0,
                    actual_lower: -0.5,
                    actual_upper: 1.5,
                    passed: false,
                },
            ],
            total_passed: 1,
            total_failed: 1,
        };

        let report = format!("{summary}");
        assert!(
            report.contains("1/2 passed"),
            "summary should show pass/total ratio"
        );
        assert!(
            report.contains("[PASS] J2_F0"),
            "should show PASS for J2_F0"
        );
        assert!(
            report.contains("[FAIL] J5_AUDIO"),
            "should show FAIL for J5_AUDIO"
        );
    }

    #[test]
    fn test_verify_crown_with_junction_checks_integration() {
        use crate::moonshot::MoonshotStatus;

        let status = MoonshotStatus::from_repo();
        let certificate =
            MoonshotCertificate::from_status(&status, "kokoro-v1", "English text", "test-hash");

        let mut intermediates = HashMap::new();
        intermediates.insert("J2_F0".to_string(), (0.0_f32, 400.0_f32));
        intermediates.insert("J5_AUDIO".to_string(), (-0.9_f32, 0.9_f32));

        let summary = verify_crown_with_junction_checks(&certificate, &intermediates);
        assert_eq!(summary.checks.len(), 2);
        assert_eq!(summary.total_passed, 2);
        assert_eq!(summary.total_failed, 0);
    }

    #[test]
    fn test_check_junction_bound_nan_handling() {
        let check = check_junction_bound("J2_F0", -5.0, 800.0, f32::NAN, 500.0);
        assert!(!check.passed, "NaN actual_lower should fail");

        let check = check_junction_bound("J2_F0", -5.0, 800.0, 0.0, f32::NAN);
        assert!(!check.passed, "NaN actual_upper should fail");

        let check = check_junction_bound("J2_F0", f32::NAN, 800.0, 0.0, 500.0);
        assert!(!check.passed, "NaN expected_lower should fail");

        let check = check_junction_bound("J2_F0", -5.0, f32::NAN, 0.0, 500.0);
        assert!(!check.passed, "NaN expected_upper should fail");
    }

    #[test]
    fn test_check_junction_bound_infinity_handling() {
        let check = check_junction_bound("J2_F0", -5.0, 800.0, f32::NEG_INFINITY, 500.0);
        assert!(!check.passed, "negative infinity actual_lower should fail");

        let check = check_junction_bound("J2_F0", -5.0, 800.0, 0.0, f32::INFINITY);
        assert!(!check.passed, "positive infinity actual_upper should fail");
    }
}
