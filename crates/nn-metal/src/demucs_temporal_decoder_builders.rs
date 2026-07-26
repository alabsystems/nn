// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs temporal decoder `TensorKernelDef`s.
//!
//! TensorKernelDef builder and weight map functions are defined in nn-models
//! (backend-agnostic). This module re-exports them and adds Metal-specific
//! weight validation (depends on `DemucsTemporalDecoderError`).
//!
//! Part of #779 Phase A, extracted to nn-models as part of #860.

use std::borrow::Cow;

use super::{
    channels_at_depth, DemucsTemporalDecoderError, DemucsTemporalDecoderWeights, DEPTH,
    KERNEL_SIZE, OUTPUT_CHANNELS, REWRITE_KERNEL,
};
use crate::demucs_shared::{validate_weight_size, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL};

// Re-export builder functions from nn-models.
pub(crate) use nn_models::demucs_temporal_decoder_builders::{
    build_decoder_block_def, build_decoder_weight_map,
};
// Re-export for parent module and tests that use `builders::conv1d_output_len`.
pub(crate) use nn_models::conv1d_output_len;

// ---------------------------------------------------------------------------
// Weight validation (Metal-specific error type)
// ---------------------------------------------------------------------------

fn validate_weight(
    data: &[f32],
    name: &str,
    expected: usize,
) -> Result<(), DemucsTemporalDecoderError> {
    Ok(validate_weight_size(data, name, expected)?)
}

/// Validate all weight tensors for the full decoder.
pub(super) fn validate_all_weights(
    weights: &DemucsTemporalDecoderWeights,
) -> Result<(), DemucsTemporalDecoderError> {
    if weights.blocks.len() != DEPTH {
        return Err(DemucsTemporalDecoderError::WeightSize {
            name: Cow::Borrowed("blocks.len"),
            expected: DEPTH,
            actual: weights.blocks.len(),
        });
    }

    for (block_idx, block) in weights.blocks.iter().enumerate() {
        let encoder_depth = DEPTH - 1 - block_idx;
        let in_ch = channels_at_depth(encoder_depth);
        let out_ch = if encoder_depth == 0 {
            OUTPUT_CHANNELS
        } else {
            channels_at_depth(encoder_depth - 1)
        };
        let compressed = in_ch / DCONV_COMPRESS;
        let prefix = format!("block{block_idx}");

        // Rewrite Conv1d
        validate_weight(
            &block.rewrite_weight,
            &format!("{prefix}.rw_weight"),
            in_ch * 2 * in_ch * REWRITE_KERNEL,
        )?;
        validate_weight(&block.rewrite_bias, &format!("{prefix}.rw_bias"), in_ch * 2)?;

        // DConv sub-layers
        if block.dconv.len() != DCONV_DEPTH {
            return Err(DemucsTemporalDecoderError::WeightSize {
                name: Cow::Owned(format!("{prefix}.dconv.len")),
                expected: DCONV_DEPTH,
                actual: block.dconv.len(),
            });
        }
        for (k, sub) in block.dconv.iter().enumerate() {
            let sp = format!("{prefix}.dc{k}");
            validate_weight(
                &sub.conv_compress_weight,
                &format!("{sp}_cw"),
                compressed * in_ch * DCONV_KERNEL,
            )?;
            validate_weight(&sub.conv_compress_bias, &format!("{sp}_cb"), compressed)?;
            validate_weight(&sub.norm_compress_gamma, &format!("{sp}_ng"), compressed)?;
            validate_weight(&sub.norm_compress_beta, &format!("{sp}_nb"), compressed)?;
            validate_weight(
                &sub.conv_expand_weight,
                &format!("{sp}_ew"),
                in_ch * 2 * compressed,
            )?;
            validate_weight(&sub.conv_expand_bias, &format!("{sp}_eb"), in_ch * 2)?;
            validate_weight(&sub.norm_expand_gamma, &format!("{sp}_eng"), in_ch * 2)?;
            validate_weight(&sub.norm_expand_beta, &format!("{sp}_enb"), in_ch * 2)?;
            validate_weight(&sub.layer_scale, &format!("{sp}_ls"), in_ch)?;
        }

        // ConvTranspose1d
        validate_weight(
            &block.conv_tr_weight,
            &format!("{prefix}.ct_weight"),
            in_ch * out_ch * KERNEL_SIZE,
        )?;
        validate_weight(&block.conv_tr_bias, &format!("{prefix}.ct_bias"), out_ch)?;
    }

    Ok(())
}
