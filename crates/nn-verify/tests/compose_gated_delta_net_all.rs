// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Gated DeltaNet composition tests — all categories in one binary.
//!
//! Merges 6 individual `compose_gated_delta_net*.rs` test files into a single
//! test binary to eliminate 5 redundant NY link steps (6 → 1).
//!
//! Categories (6 modules total):
//!   Monolithic (1): compose_gated_delta_net — single-timestep translation
//!   Gate (2): compose_gated_delta_net_gate (D2 sub-graph),
//!     compose_gated_delta_net_gate_d3 (D3 full GDN with gate)
//!   Composition (2): compose_gated_delta_net_mixed (mixed-binding),
//!     compose_gated_delta_net_recurrent (two-timestep + single-variable)
//!   State update (1): compose_gated_delta_net_state_update (state evolution
//!     bounds, contractivity, multi-step growth verification)
//!
//! Part of #1982, #3578.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

// ── Monolithic ────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gated_delta_net.rs"]
mod gdn_monolithic;

// ── Gate sub-graph ────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gated_delta_net_gate.rs"]
mod gdn_gate;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gated_delta_net_gate_d3.rs"]
mod gdn_gate_d3;

// ── Composition ──────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gated_delta_net_mixed.rs"]
mod gdn_mixed;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gated_delta_net_recurrent.rs"]
mod gdn_recurrent;

// ── State update verification (#3578) ────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gated_delta_net_state_update.rs"]
mod gdn_state_update;
