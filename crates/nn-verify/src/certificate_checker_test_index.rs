// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Test index for certificate_checker — consolidates 15 test modules.
//! See designs/2026-03-22-code-structure-wave16-test-index-modules.md.

// Re-export parent items so child test modules keep `use super::*` working.
#[allow(unused_imports)]
pub(crate) use super::*;

// Shared test helpers (was: checker_test_shared in parent).
#[path = "certificate_checker_test_shared.rs"]
pub(crate) mod checker_test_shared;

#[path = "certificate_checker_tests_algorithm_audit.rs"]
mod algorithm_audit;
#[path = "certificate_checker_tests.rs"]
mod basic;
#[path = "certificate_checker_tests_bundle.rs"]
mod bundle;
#[path = "certificate_checker_tests_coverage.rs"]
mod coverage;
#[path = "certificate_checker_tests_keyed.rs"]
mod keyed;
#[path = "certificate_checker_tests_performance.rs"]
mod performance;
#[path = "certificate_checker_tests_smt_proof.rs"]
mod smt_proof;
#[path = "certificate_checker_tests_soundness.rs"]
mod soundness;
#[path = "certificate_checker_tests_soundness_1692.rs"]
mod soundness_1692;
#[path = "certificate_checker_tests_soundness_3153.rs"]
mod soundness_3153;
#[path = "certificate_checker_tests_soundness_3200.rs"]
mod soundness_3200;
#[path = "certificate_checker_tests_soundness_3325.rs"]
mod soundness_3325;
#[path = "certificate_checker_tests_soundness_trace.rs"]
mod soundness_trace;
#[path = "certificate_checker_tests_vacuity.rs"]
mod vacuity;
