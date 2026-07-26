// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder and validation helpers for the Silero VAD model.
//!
//! Backend-agnostic builders (`build_output_def`, `build_encoder_block_def`,
//! `EncoderBlock`, `ENCODER_BLOCKS`) are re-exported from nn-models.
//! Weight validation (Metal-specific `SileroVadWeights`) stays here.

use nn_dsl::TensorKernelDef;

// Re-export backend-agnostic builders and constants from nn-models.
pub(super) use nn_models::silero_vad_builders::{
    build_encoder_block_def as build_encoder_block_def_inner,
    build_output_def as build_output_def_inner, EncoderBlock, ENCODER_BLOCKS, LSTM_HIDDEN_SIZE,
};

use super::{SileroVadError, SileroVadWeights};

// Re-export from shared module — was previously defined locally.
pub(super) use crate::demucs_shared::conv1d_output_len;

/// Build the output stage, wrapping TensorIRError into SileroVadError.
pub(super) fn build_output_def() -> Result<TensorKernelDef, SileroVadError> {
    Ok(build_output_def_inner()?)
}

/// Build encoder block def, wrapping TensorIRError into SileroVadError.
pub(super) fn build_encoder_block_def(
    block: &EncoderBlock,
    t_in: usize,
    t_out: usize,
) -> Result<TensorKernelDef, SileroVadError> {
    Ok(build_encoder_block_def_inner(block, t_in, t_out)?)
}

/// Validate all weight tensors have the expected element counts.
pub(super) fn validate_all_weights(weights: &SileroVadWeights) -> Result<(), SileroVadError> {
    validate_weight(&weights.stft_basis, "stft_basis", 258 * 256)?;
    for (i, (blk, (w, b))) in ENCODER_BLOCKS
        .iter()
        .zip(weights.enc_weights.iter().zip(weights.enc_biases.iter()))
        .enumerate()
    {
        let w_name: &'static str = match i {
            0 => "encoder_0_weight",
            1 => "encoder_1_weight",
            2 => "encoder_2_weight",
            3 => "encoder_3_weight",
            // ENCODER_BLOCKS has exactly 4 entries. Return error instead of
            // unreachable!() per #1424 policy — prevents panic if array grows.
            _ => {
                return Err(SileroVadError::OutputLength {
                    stage: "encoder_weight_validation",
                    expected: 4,
                    actual: i + 1,
                });
            }
        };
        let b_name: &'static str = match i {
            0 => "encoder_0_bias",
            1 => "encoder_1_bias",
            2 => "encoder_2_bias",
            3 => "encoder_3_bias",
            _ => {
                return Err(SileroVadError::OutputLength {
                    stage: "encoder_bias_validation",
                    expected: 4,
                    actual: i + 1,
                });
            }
        };
        validate_weight(
            w,
            w_name,
            blk.out_channels * blk.in_channels * blk.kernel_size,
        )?;
        validate_weight(b, b_name, blk.out_channels)?;
    }
    // LSTM has 4 gates (i, f, g, o), so weight shape is [4*hidden, hidden].
    let lstm_gate_size = 4 * LSTM_HIDDEN_SIZE;
    validate_weight(
        &weights.lstm_weight_ih,
        "lstm_weight_ih",
        lstm_gate_size * LSTM_HIDDEN_SIZE,
    )?;
    validate_weight(
        &weights.lstm_weight_hh,
        "lstm_weight_hh",
        lstm_gate_size * LSTM_HIDDEN_SIZE,
    )?;
    validate_weight(&weights.lstm_bias_ih, "lstm_bias_ih", lstm_gate_size)?;
    validate_weight(&weights.lstm_bias_hh, "lstm_bias_hh", lstm_gate_size)?;
    validate_weight(&weights.output_weight, "output_weight", LSTM_HIDDEN_SIZE)?;
    validate_weight(&weights.output_bias, "output_bias", 1)?;
    Ok(())
}

fn validate_weight(
    data: &[f32],
    name: &'static str,
    expected: usize,
) -> Result<(), SileroVadError> {
    if data.len() != expected {
        return Err(SileroVadError::WeightSize {
            name,
            expected,
            actual: data.len(),
        });
    }
    let count = data.iter().filter(|v| !v.is_finite()).count();
    if count > 0 {
        return Err(SileroVadError::NonFiniteWeight {
            name: name.to_string(),
            count,
        });
    }
    Ok(())
}
