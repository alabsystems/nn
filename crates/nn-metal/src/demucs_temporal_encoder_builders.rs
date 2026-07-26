// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder functions for Demucs temporal encoder `TensorKernelDef`s.
//!
//! TensorKernelDef builder and weight map functions are defined in nn-models
//! (backend-agnostic). This module re-exports them and adds Metal-specific
//! weight validation (depends on `DemucsTemporalEncoderError`).
//!
//! Part of #779 Phase E, extracted to nn-models as part of #860.

use std::borrow::Cow;

use super::{channels_at_depth, DemucsTemporalEncoderError, DemucsTemporalEncoderWeights};
use crate::demucs_shared::{
    validate_weight_size, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL,
    TEMPORAL_BASIC_DEPTH as TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE,
};

// Re-export builder functions from nn-models.
pub(crate) use nn_models::demucs_temporal_encoder_builders::{
    build_encoder_block_def, build_encoder_weight_map, conv1d_out_len,
};

// ---------------------------------------------------------------------------
// Weight validation (Metal-specific error type)
// ---------------------------------------------------------------------------

fn validate_weight(
    data: &[f32],
    name: &str,
    expected: usize,
) -> Result<(), DemucsTemporalEncoderError> {
    Ok(validate_weight_size(data, name, expected)?)
}

/// Validate all weight tensors for the full encoder.
pub(super) fn validate_all_weights(
    weights: &DemucsTemporalEncoderWeights,
) -> Result<(), DemucsTemporalEncoderError> {
    if weights.blocks.len() != TEMPORAL_DEPTH {
        return Err(DemucsTemporalEncoderError::WeightSize {
            name: Cow::Borrowed("blocks.len"),
            expected: TEMPORAL_DEPTH,
            actual: weights.blocks.len(),
        });
    }

    for (block_idx, block) in weights.blocks.iter().enumerate() {
        let in_ch = if block_idx == 0 {
            super::AUDIO_CHANNELS
        } else {
            channels_at_depth(block_idx - 1)
        };
        let out_ch = channels_at_depth(block_idx);
        let compressed = out_ch / DCONV_COMPRESS;
        let prefix = format!("block{block_idx}");

        // Conv1d (downsample)
        validate_weight(
            &block.conv_weight,
            &format!("{prefix}.conv_weight"),
            out_ch * in_ch * TEMPORAL_KERNEL_SIZE,
        )?;
        validate_weight(&block.conv_bias, &format!("{prefix}.conv_bias"), out_ch)?;

        // DConv sub-layers
        if block.dconv.len() != DCONV_DEPTH {
            return Err(DemucsTemporalEncoderError::WeightSize {
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
                compressed * out_ch * DCONV_KERNEL,
            )?;
            validate_weight(&sub.conv_compress_bias, &format!("{sp}_cb"), compressed)?;
            validate_weight(&sub.norm_compress_gamma, &format!("{sp}_ng"), compressed)?;
            validate_weight(&sub.norm_compress_beta, &format!("{sp}_nb"), compressed)?;
            validate_weight(
                &sub.conv_expand_weight,
                &format!("{sp}_ew"),
                out_ch * 2 * compressed,
            )?;
            validate_weight(&sub.conv_expand_bias, &format!("{sp}_eb"), out_ch * 2)?;
            validate_weight(&sub.norm_expand_gamma, &format!("{sp}_eng"), out_ch * 2)?;
            validate_weight(&sub.norm_expand_beta, &format!("{sp}_enb"), out_ch * 2)?;
            validate_weight(&sub.layer_scale, &format!("{sp}_ls"), out_ch)?;
        }

        // Rewrite Conv1d (k=1)
        validate_weight(
            &block.rewrite_weight,
            &format!("{prefix}.rw_weight"),
            out_ch * 2 * out_ch,
        )?;
        validate_weight(
            &block.rewrite_bias,
            &format!("{prefix}.rw_bias"),
            out_ch * 2,
        )?;
    }

    Ok(())
}
