// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Demucs misc composition tests.
//!
//! Combines 4 test families into a single test binary to reduce
//! compilation overhead (4 NY link steps → 1).
//!
//! - `encoder_block`: Production + parametric encoder block composition
//! - `cross_domain`: Cross-domain transformer bottleneck (cross-attention)
//! - `demucs_enc_dec_helpers`: Demucs encoder/decoder helper functions
//! - `demucs_enc_dec_helpers_dconv`: Demucs encoder/decoder DConv helpers
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_encoder_block.rs"]
mod encoder_block;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_cross_domain.rs"]
mod cross_domain;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/demucs_enc_dec_helpers.rs"]
mod enc_dec_helpers;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/demucs_enc_dec_helpers_dconv.rs"]
mod enc_dec_helpers_dconv;
