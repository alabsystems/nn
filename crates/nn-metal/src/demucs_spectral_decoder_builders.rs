// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thin wrapper re-exporting spectral decoder builders from nn-models.
//!
//! The backend-agnostic builder logic lives in
//! `nn_models::demucs_spectral_decoder_builders`. This module re-exports
//! those items with the original local names so existing call sites in
//! `demucs_spectral_decoder.rs` remain unchanged.
//!
//! Part of #860 — nn-metal extraction.

pub(crate) use nn_models::demucs_spectral_decoder_builders::{
    build_decoder_block_sub_defs as build_block_sub_defs,
    build_decoder_block_weight_maps as build_block_weight_maps,
    SpectralDecoderBlockSubDefs as BlockSubDefs,
    SpectralDecoderBlockWeightMaps as SpectralBlockWeightMaps,
};

pub(super) use nn_models::conv1d_output_len;

use super::{DemucsSpectralDecoderError, DemucsSpectralDecoderWeights};

/// Conv2d output length, mapping `DemucsBuilderError` to the module error type.
pub(crate) fn conv2d_output_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, DemucsSpectralDecoderError> {
    Ok(
        nn_models::demucs_spectral_decoder_builders::conv2d_output_len(
            in_len,
            kernel_size,
            stride,
            padding,
        )?,
    )
}

/// Validate all weight tensors, mapping `DemucsBuilderError` to the module error type.
pub(super) fn validate_all_weights(
    weights: &DemucsSpectralDecoderWeights,
) -> Result<(), DemucsSpectralDecoderError> {
    Ok(nn_models::demucs_spectral_decoder_builders::validate_all_decoder_weights(weights)?)
}
