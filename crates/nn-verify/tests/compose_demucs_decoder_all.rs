// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Demucs decoder composition tests.
//!
//! Combines 6 decoder/encoder block test families into a single test binary
//! to reduce compilation overhead (6 NY link steps → 1).
//!
//! - `decoder_pipeline`: Decoder block + encoder→decoder composition
//! - `decoder_production`: Production builder temporal/spectral decoder tests
//! - `decoder_chain`: Chained decoder composition
//! - `decoder_conv_transpose`: ConvTranspose1d decoder composition
//! - `four_block_decoder`: 4-block decoder composition
//! - `four_block_encoder`: 4-block encoder composition
//!
//! Part of #1982.

// Child helper files use #![allow(dead_code)] internally; suppress the
// duplicated-attribute lint that fires when the outer #[allow(dead_code)]
// on the `mod` declaration overlaps.
#![allow(clippy::duplicated_attributes)]
// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_decoder_pipeline.rs"]
mod decoder_pipeline;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_decoder_production.rs"]
mod decoder_production;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_decoder_chain.rs"]
mod decoder_chain;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_decoder_conv_transpose.rs"]
mod decoder_conv_transpose;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_four_block_decoder.rs"]
mod four_block_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_four_block_encoder.rs"]
mod four_block_encoder;
