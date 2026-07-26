// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Demucs spectral composition tests.
//!
//! Combines 6 spectral test families into a single test binary to reduce
//! compilation overhead (6 NY link steps → 1).
//!
//! - `spectral_encoder`: Simplified spectral encoder composition
//! - `spectral_full`: Full spectral processing pipeline
//! - `spectral_decoder`: Production builder spectral decoder (IBP + CROWN)
//! - `spectral_decoder_advanced`: CROWN, verify-and-record, last-block, sequential
//! - `spectral_encoder_prod`: Production builder spectral encoder
//! - `spectral_decoder_subdefs`: ConvTranspose1d+trim, DConv residual, Conv2d→GLU decoder blocks
//!
//! Part of #1982.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_spectral_encoder.rs"]
mod spectral_encoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_spectral_full.rs"]
mod spectral_full;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_spectral_decoder.rs"]
mod spectral_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_spectral_decoder_advanced.rs"]
mod spectral_decoder_advanced;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_spectral_encoder_prod.rs"]
mod spectral_encoder_prod;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_spectral_decoder_subdefs.rs"]
mod spectral_decoder_subdefs;
