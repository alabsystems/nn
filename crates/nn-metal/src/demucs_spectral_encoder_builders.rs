// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thin wrapper re-exporting spectral encoder builders from nn-models.
//!
//! The backend-agnostic builder logic lives in
//! `nn_models::demucs_spectral_encoder_builders`. This module re-exports
//! those items with the original local names so existing call sites in
//! `demucs_spectral_encoder.rs` remain unchanged.
//!
//! Part of #860 — nn-metal extraction.

pub(crate) use nn_models::demucs_spectral_encoder_builders::{
    build_encoder_block_sub_defs as build_block_sub_defs,
    build_encoder_block_weight_maps as build_block_weight_maps,
    SpectralEncoderBlockSubDefs as BlockSubDefs,
    SpectralEncoderBlockWeightMaps as SpectralBlockWeightMaps,
};

pub(super) use nn_models::conv1d_output_len;

use super::{DemucsSpectralEncoderError, DemucsSpectralEncoderWeights};

/// Validate all weight tensors, mapping `DemucsBuilderError` to the module error type.
pub(super) fn validate_all_weights(
    weights: &DemucsSpectralEncoderWeights,
) -> Result<(), DemucsSpectralEncoderError> {
    Ok(nn_models::demucs_spectral_encoder_builders::validate_all_encoder_weights(weights)?)
}
