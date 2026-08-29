// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Kokoro verification tests: analytical bridges, gap detection,
//! and quantization verification.

#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_production_weights.rs"]
mod kokoro_production_weights;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_analytical_bridges.rs"]
mod kokoro_analytical_bridges;

// The canonical gap-detector gate (also its own standalone test target).
// A helpers/ copy of this file used to drift from the canonical one — same
// test name, older assertions; one source of truth now runs in both places.
#[allow(dead_code, unreachable_pub)]
#[path = "kokoro_gap_detector.rs"]
mod kokoro_gap_detector;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_quantization.rs"]
mod kokoro_quantization;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/harmonic_source_segmented.rs"]
mod harmonic_source_segmented;
