// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Demucs compose verification tests for vacuous entry promotion.
//!
//! Re-verifies 7 vacuous Demucs entries with NormBoundsMode::Conservative
//! to promote them from heuristic/vacuous to IbpValidated/sound.
//!
//! Targets:
//!   - `demucs_spectral_encoder_block`
//!   - `demucs_spectral_encoder_prod_dconv`
//!   - `demucs_spectral_decoder_dconv`
//!   - `demucs_temporal_encoder_block`
//!   - `demucs_temporal_encoder_prod_block0`
//!   - `demucs_temporal_decoder_block`
//!   - `demucs_cross_domain_bottleneck`
//!
//! Part of verification gap closure for dvoice production models.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_demucs_deep.rs"]
mod demucs_deep;
