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

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_gap_detector.rs"]
mod kokoro_gap_detector;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/kokoro_quantization.rs"]
mod kokoro_quantization;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/harmonic_source_segmented.rs"]
mod harmonic_source_segmented;
