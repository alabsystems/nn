// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Silero VAD composition and verification tests.
//!
//! Combines 7 Silero VAD test files into a single test binary to reduce
//! compilation overhead (7 NY link steps → 1).
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_silero_vad_certificate.rs"]
mod silero_vad_certificate;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_silero_vad_encoder.rs"]
mod silero_vad_encoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_silero_vad_full.rs"]
mod silero_vad_full;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/silero_vad_test_helpers.rs"]
mod silero_vad_test_helpers;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_silero_vad_deep.rs"]
mod silero_vad_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_silero_vad_pipeline.rs"]
mod silero_vad_pipeline;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/validate_silero_vad_proof_bundle.rs"]
mod validate_proof_bundle;
