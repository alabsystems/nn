#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated nn integration tests: kernel syntax, Metal BF16, facade,
//! feature gates, nn reexport, and TTS verify integration.

#![allow(dead_code, unreachable_pub)]

#[cfg(feature = "dsl")]
#[path = "integration/kernel_syntax.rs"]
mod kernel_syntax;

#[path = "integration/metal_bf16_tests.rs"]
mod metal_bf16_tests;

#[path = "integration/metal_facade.rs"]
mod metal_facade;

#[path = "integration/model_feature_gates.rs"]
mod model_feature_gates;

#[path = "integration/nn_reexport_sync.rs"]
mod nn_reexport_sync;

#[cfg(feature = "tts-verify")]
#[path = "integration/tts_verify_integration.rs"]
mod tts_verify_integration;
