// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated miscellaneous verification tests.
//!
//! Combines 5 verification test files into a single test binary
//! to reduce compilation overhead (5 NY link steps → 1).
//!
//! - CROWN piecewise (2): verify_crown_piecewise, verify_crown_piecewise_decomp
//! - Other (3): verify_request, verify_select_soundness, verify_spec_provenance
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

// ── CROWN piecewise ───────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_crown_piecewise.rs"]
mod crown_piecewise;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_crown_piecewise_decomp.rs"]
mod crown_piecewise_decomp;

// ── Other verification ────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_request.rs"]
mod verify_request;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_select_soundness.rs"]
mod verify_select_soundness;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_spec_provenance.rs"]
mod verify_spec_provenance;

// ── Misc standalone ─────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/proof_coverage_dashboard.rs"]
mod proof_coverage_dashboard;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/regression_missing_output_policy.rs"]
mod regression_missing_output_policy;
