// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated moonshot composition tests — all categories in one binary.
//!
//! Merges 6 individual `compose_moonshot*.rs` test files into a single test
//! binary to eliminate 5 redundant NY link steps (6 → 1).
//!
//! Categories (6 modules total):
//!   Attention (1): compose_moonshot_attention_integration — attention sub-graph
//!   Certificate (2): compose_moonshot_certificate (core),
//!     compose_moonshot_certificate_pipeline (pipeline)
//!   CROWN (2): compose_moonshot_crown_full_pipeline (full pipeline),
//!     compose_moonshot_crown_prosody (prosody)
//!   Coupled cost (1): compose_moonshot_coupled_cost — coupled cost propagation
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

// ── Attention ───────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_moonshot_attention_integration.rs"]
mod moonshot_attention;

// ── Certificate ─────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_moonshot_certificate.rs"]
mod moonshot_certificate;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_moonshot_certificate_pipeline.rs"]
mod moonshot_certificate_pipeline;

// ── CROWN ───────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_moonshot_crown_full_pipeline.rs"]
mod moonshot_crown_full;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_moonshot_crown_prosody.rs"]
mod moonshot_crown_prosody;

// ── Coupled cost ────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_moonshot_coupled_cost.rs"]
mod moonshot_coupled_cost;
