// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated trace-to-graph tests — all categories in one binary.
//!
//! Merges 15 individual `trace_to_graph_*.rs` test files into a single test
//! binary to eliminate 14 redundant NY link steps (15 → 1).
//!
//! Categories (15 modules total):
//!   Model core (1): model_tests
//!   Activations (2): model_activations, model_param_activations
//!   Binary/Reduction (1): model_binary_ops
//!   Normalization (1): model_norm_ops
//!   Shape/Softmax (1): model_shape_softmax
//!   Conv/Embedding (1): model_embedding_conv_transpose
//!   Pool (1): model_pool
//!   Misc ops (1): model_misc_ops
//!   LSTM/VAD (1): model_silero_vad_synthetic
//!   Boundary (1): model_boundary_tests
//!   Error paths (1): error_tests
//!   Topology gaps (1): topology_gap_tests
//!   Multi-input (1): multi_input
//!   Decomposed ops (1): model_decompose_ops

#![allow(clippy::duplicate_mod)]

mod common;

// ── Model core ──────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_tests.rs"]
mod model_tests;

// ── Activations ─────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_activations.rs"]
mod model_activations;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_param_activations.rs"]
mod model_param_activations;

// ── Binary / Reduction ──────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_binary_ops.rs"]
mod model_binary_ops;

// ── Normalization ───────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_norm_ops.rs"]
mod model_norm_ops;

// ── Shape / Softmax ─────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_shape_softmax.rs"]
mod model_shape_softmax;

// ── Conv / Embedding ────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_embedding_conv_transpose.rs"]
mod model_embedding_conv_transpose;

// ── Pool ────────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_pool.rs"]
mod model_pool;

// ── Misc ops ────────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_misc_ops.rs"]
mod model_misc_ops;

// ── LSTM / Silero VAD ───────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_silero_vad_synthetic.rs"]
mod model_silero_vad_synthetic;

// ── Boundary regression ─────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_boundary_tests.rs"]
mod model_boundary_tests;

// ── Error paths ─────────────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_error_tests.rs"]
mod error_tests;

// ── Topology gap proofs ─────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_topology_gap_tests.rs"]
mod topology_gap_tests;

// ── Multi-input (#2377) ────────────────────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_multi_input.rs"]
mod multi_input;

// ── Decomposed op round-trip (#2329) ──────────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_decompose_ops.rs"]
mod model_decompose_ops;

// ── ResizeBilinear + MoeGating (#3545) ───────────────────────────────────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_resize_moe.rs"]
mod model_resize_moe;

// ── Gap-fill ops: Powf, SwiGlu, ScatterAdd, IndexAdd, GridSample (#3557) ─

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_model_gap_fill.rs"]
mod model_gap_fill;

// ── Trace infrastructure: TraceOp coverage, translation round-trips ───────

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/trace_to_graph_trace_infrastructure.rs"]
mod trace_infrastructure;
