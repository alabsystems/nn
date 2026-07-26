// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated fusion verification tests.
//!
//! Combines fusion test files into a single test binary to reduce
//! compilation overhead (NY link steps → 1).
//!
//! Categories:
//!   Recording (1): fusion_equivalence_recording
//!   Validation (1): fusion_equivalence_validation
//!   CROWN/IBP (1): fusion_crown_ibp_fallback
//!   Equivalence (1): fusion_equivalence — diamond DAG diff tests
//!   Pairs (1): fusion_equivalence_pairs — RMSNorm+SiLU-Mul, LayerNorm+GELU
//!   AdaLN (1): fusion_equivalence_adaln — AdaLayerNorm fusion (#2714)
//!   Certificates (1): fusion_certificate_integration — D=512 certificates + Monte Carlo
//!   Auto (1): fusion_auto_verify — auto-generated chain → spec → CROWN proof
//!   Pipeline (1): fusion_pipeline_integration — single-call graph → verify pipeline
//!
//! Part of #1982, #2462, #2127, #2714.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_equivalence_recording.rs"]
mod recording;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_equivalence_validation.rs"]
mod validation;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_crown_ibp_fallback.rs"]
mod crown_ibp_fallback;

// ── Equivalence ───────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_equivalence.rs"]
mod equivalence;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_equivalence_pairs.rs"]
mod equivalence_pairs;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_equivalence_adaln.rs"]
mod equivalence_adaln;

// ── Certificates ────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_certificate_integration.rs"]
mod certificate_integration;

// ── Auto-generated fusion verification ──────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_auto_verify.rs"]
mod auto_verify;

// ── Pipeline integration (graph → verify single-call) ────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_pipeline_integration.rs"]
mod pipeline_integration;

// ── Pipeline certificates (graph → verify → certificates) ────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_pipeline_certificates.rs"]
mod pipeline_certificates;

// ── NormActivConv1d per-tap fusion (#2218 F13) ──────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/fusion_norm_activ_conv.rs"]
mod norm_activ_conv;
