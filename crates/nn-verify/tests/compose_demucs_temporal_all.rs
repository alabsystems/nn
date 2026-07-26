// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Demucs temporal composition tests.
//!
//! Combines 3 temporal test families into a single test binary to reduce
//! compilation overhead (4 NY link steps → 1).
//!
//! - `temporal_encoder`: Encoder block composition (Conv1d + GELU + DConv + GLU)
//! - `temporal_decoder`: Decoder block composition (Rewrite + DConv + ConvTranspose)
//! - `temporal_pipeline`: Branch (Enc→Dec) and Full (Enc→Transformer→Dec)
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_temporal_encoder.rs"]
mod temporal_encoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_temporal_decoder.rs"]
mod temporal_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_temporal_pipeline.rs"]
mod temporal_pipeline;
