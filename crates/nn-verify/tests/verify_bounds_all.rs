// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated verify_bounds tests — all categories in one binary.
//!
//! Merges 5 individual `verify_bounds*.rs` test files into a single test
//! binary to eliminate 4 redundant NY link steps (5 → 1).
//!
//! Categories (5 modules total):
//!   Core (1): verify_bounds — scalar kernel bounds verification
//!   Escalation (1): verify_bounds_escalation — escalation/soundness mode tests
//!   Multi (1): verify_bounds_multi — multi-variable kernel bounds
//!   Soundness (1): verify_bounds_soundness — soundness mode propagation
//!   Validation (1): verify_bounds_validation — input validation + error paths
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

// ── Core ────────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_bounds.rs"]
mod bounds_core;

// ── Escalation ──────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_bounds_escalation.rs"]
mod bounds_escalation;

// ── Multi ───────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_bounds_multi.rs"]
mod bounds_multi;

// ── Soundness ───────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_bounds_soundness.rs"]
mod bounds_soundness;

// ── Validation ──────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/verify_bounds_validation.rs"]
mod bounds_validation;
