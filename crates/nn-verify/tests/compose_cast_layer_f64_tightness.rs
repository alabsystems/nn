// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CastLayer compose tests with f64 tightness validation.
//!
//! Tests that CastLayer (for upcasts) and Clamp (for downcasts) work correctly
//! in the trace_to_graph pipeline with actual IBP/CROWN bound propagation.
//! Also validates f64 tightness for CastLayer-containing pipelines.
//!
//! Part of #4316: CastLayer for ToDtype verification + f64 evaluation.

#![allow(clippy::duplicate_mod)]

mod common;

// IBP/CROWN propagation tests through CastLayer and Clamp.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_cast_layer_f64_tightness.rs"]
mod compose_cast_layer_ibp_crown;

// f64 tightness comparison and CastLayer zero-impact validation.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_cast_layer_f64_comparison.rs"]
mod compose_cast_layer_f64_comparison;
