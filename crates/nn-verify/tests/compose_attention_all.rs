// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated attention composition test binary.
//!
//! Combines 8 compose_attention_* test files into a single test binary
//! to reduce compilation overhead (8 → 1 NY link step).
//!
//! Categories:
//!   Core (1): compose_attention_block — 3-op scaled dot-product attention
//!   Layerwise (1): compose_attention_layerwise — Phases 12-14 layerwise
//!   End-to-end (1): compose_attention_e2e_phase15 — monolithic composition
//!   Certificate (1): compose_attention_certificate_phase16 — formal certs
//!   Quality (1): compose_attention_quality_phase17 — audio quality bounds
//!   FFN (1): compose_attention_ffn_phase24 — attention + FFN composition
//!   Monotonicity (2): parametric + non-parametric attention monotonicity
//!
//! Part of #1982.

// Shared helper files (attention_monotonicity.rs, attention_layerwise_builders.rs)
// are #[path]-included by multiple child submodules independently.
// This is inherent to the consolidation pattern — suppress the lint.
#![allow(clippy::duplicate_mod)]

mod common;

// ── Core ────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_block.rs"]
mod compose_attention_block;

// ── Layerwise ───────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_layerwise.rs"]
mod compose_attention_layerwise;

// ── End-to-end ──────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_e2e_phase15.rs"]
mod compose_attention_e2e_phase15;

// ── Certificate ─────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_certificate_phase16.rs"]
mod compose_attention_certificate_phase16;

// ── Quality ─────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_quality_phase17.rs"]
mod compose_attention_quality_phase17;

// ── FFN ─────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_ffn_phase24.rs"]
mod compose_attention_ffn_phase24;

// ── Monotonicity ────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_monotonicity.rs"]
mod compose_attention_monotonicity;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_attention_monotonicity_parametric.rs"]
mod compose_attention_monotonicity_parametric;
