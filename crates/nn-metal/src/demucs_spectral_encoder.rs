// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Demucs spectral encoder: 4-block encoder branch for HTDemucs source separation.
//!
//! Implements the spectral (frequency-domain) encoder of HTDemucs. Each
//! encoder block operates on 3D tensors `[C, F, T]` (batch=1) and uses:
//!
//!   axis_switch → Conv1d(k=8, s=4) → GELU → axis_switch → DConv → axis_switch
//!   → Rewrite(Conv1d k=1) → GLU → axis_switch
//!
//! The axis-switch pattern collapses one spatial dimension into the batch
//! dimension so existing 1D operations (Conv1d, DConv) apply along the
//! other spatial axis. This is handled CPU-side in Phase 1.
//!
//! Each block is split into 3 sub-`TensorKernelDef`s:
//! 1. Conv1d + GELU: operates on `[C_in, F]` per time step (downsample freq)
//! 2. DConv: operates on `[C_out, T]` per frequency bin
//! 3. Rewrite + GLU: operates on `[C_out, F']` per time step
//!
//! Part of #831.

use crate::PipelineCache;

// Import shared constants with local aliases for minimal diff.
use crate::demucs_shared::{
    SPECTRAL_BASIC_DEPTH as DEPTH, SPECTRAL_CONV_PADDING as CONV_PADDING,
    SPECTRAL_FREQ_EMB_DIM as FREQ_EMB_DIM, SPECTRAL_FREQ_EMB_FEATURES as FREQ_EMB_FEATURES,
    SPECTRAL_INPUT_CHANNELS as INPUT_CHANNELS, SPECTRAL_KERNEL_SIZE as KERNEL_SIZE,
    SPECTRAL_STRIDE,
};

/// Frequency embedding scale (scale * freq_emb_scale = 10 * 0.2 = 2.0).
const FREQ_EMB_SCALE: f32 = 2.0;

#[path = "demucs_spectral_encoder_types.rs"]
mod types;
pub use types::{DemucsSpectralEncoderError, DemucsSpectralEncoderWeights};

#[path = "demucs_spectral_encoder_builders.rs"]
pub(crate) mod builders;
pub(crate) use builders::{BlockSubDefs, SpectralBlockWeightMaps};

#[path = "demucs_spectral_encoder_dispatch.rs"]
mod dispatch_helpers;

#[cfg(test)]
#[path = "demucs_spectral_encoder_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Demucs spectral encoder: 4-block downsampling encoder for STFT magnitudes.
///
/// Pre-builds all `TensorKernelDef`s at construction time. The `forward()`
/// method dispatches sub-defs with CPU-side axis-switch between stages.
/// Returns the bottleneck tensor and skip connections for the spectral decoder.
#[must_use = "DemucsSpectralEncoder is constructed once and reused; call .forward() to run inference"]
pub struct DemucsSpectralEncoder {
    /// Sub-defs per encoder block (3 sub-defs each, 4 blocks).
    block_sub_defs: Vec<BlockSubDefs>,
    /// Pre-built weight maps per block.
    block_weights: Vec<SpectralBlockWeightMaps>,
    /// Output channel count per block.
    block_out_ch: Vec<usize>,
    /// Input channel count per block.
    block_in_ch: Vec<usize>,
    /// Input frequency dimension per block.
    block_f_in: Vec<usize>,
    /// Output frequency dimension per block (after Conv1d downsampling).
    block_f_out: Vec<usize>,
    /// Time dimension (preserved through all blocks).
    time_len: usize,
    /// Frequency embedding weight (row-major [FREQ_EMB_FEATURES, FREQ_EMB_DIM]).
    freq_emb_weight: Option<Vec<f32>>,
}

impl std::fmt::Debug for DemucsSpectralEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemucsSpectralEncoder")
            .field("blocks", &self.block_sub_defs.len())
            .field("block_out_ch", &self.block_out_ch)
            .field("block_f_out", &self.block_f_out)
            .field("time_len", &self.time_len)
            .field("has_freq_emb", &self.freq_emb_weight.is_some())
            .finish_non_exhaustive()
    }
}

/// Output of `DemucsSpectralEncoder::forward()`.
#[derive(Debug)]
#[must_use]
pub struct SpectralEncoderOutput {
    /// Bottleneck tensor: flattened `[channels_at_depth(DEPTH-1), F_bottleneck, T]`.
    pub bottleneck: Vec<f32>,
    /// Skip connections in encoder order (depth 0..3).
    /// Each is flattened `[channels_at_depth(d), F_d, T]`.
    pub skips: Vec<Vec<f32>>,
    /// Frequency dimensions at each encoder depth output.
    /// Used by the spectral decoder for skip trim targets.
    /// (Read in tests to verify encoder geometry; production decoder
    /// recomputes from construction parameters.)
    #[allow(dead_code)]
    pub freq_dims: Vec<usize>,
    /// Time dimension (same at every depth).
    #[allow(dead_code)]
    pub time_dim: usize,
}

impl DemucsSpectralEncoder {
    /// Construct a new spectral encoder, validating weights and building defs.
    ///
    /// `initial_f`: frequency dimension of the STFT magnitude input.
    /// `time_len`: time dimension of the STFT magnitude input.
    pub fn new(
        weights: DemucsSpectralEncoderWeights,
        initial_f: usize,
        time_len: usize,
    ) -> Result<Self, DemucsSpectralEncoderError> {
        builders::validate_all_weights(&weights)?;

        let mut block_sub_defs = Vec::with_capacity(DEPTH);
        let mut block_weight_maps = Vec::with_capacity(DEPTH);
        let mut block_out_ch = Vec::with_capacity(DEPTH);
        let mut block_in_ch = Vec::with_capacity(DEPTH);
        let mut block_f_in = Vec::with_capacity(DEPTH);
        let mut block_f_out = Vec::with_capacity(DEPTH);

        let mut f_in = initial_f;

        for block_idx in 0..DEPTH {
            let in_ch = if block_idx == 0 {
                INPUT_CHANNELS
            } else {
                channels_at_depth(block_idx - 1)
            };
            let out_ch = channels_at_depth(block_idx);

            // Conv1d output freq after stride downsampling.
            let f_out =
                builders::conv1d_output_len(f_in, KERNEL_SIZE, SPECTRAL_STRIDE, CONV_PADDING)?;

            let sub_defs =
                builders::build_block_sub_defs(block_idx, in_ch, out_ch, f_in, f_out, time_len)?;
            block_sub_defs.push(sub_defs);

            let weight_maps = builders::build_block_weight_maps(&weights.blocks[block_idx]);
            block_weight_maps.push(weight_maps);

            block_in_ch.push(in_ch);
            block_out_ch.push(out_ch);
            block_f_in.push(f_in);
            block_f_out.push(f_out);

            f_in = f_out;
        }

        Ok(Self {
            block_sub_defs,
            block_weights: block_weight_maps,
            block_out_ch,
            block_in_ch,
            block_f_in,
            block_f_out,
            time_len,
            freq_emb_weight: weights.freq_emb_weight,
        })
    }

    /// Run the spectral encoder forward pass.
    ///
    /// `stft_mag`: flattened `[INPUT_CHANNELS, F, T]` STFT magnitude input.
    ///
    /// Returns `SpectralEncoderOutput` containing the bottleneck, skip
    /// connections (in encoder depth order), and frequency dimensions.
    pub fn forward(
        &self,
        cache: &PipelineCache,
        stft_mag: &[f32],
    ) -> Result<SpectralEncoderOutput, DemucsSpectralEncoderError> {
        let expected_len = INPUT_CHANNELS * self.block_f_in[0] * self.time_len;
        if stft_mag.len() != expected_len {
            return Err(DemucsSpectralEncoderError::DimMismatch {
                stage: "stft_input".to_string(),
                expected: expected_len,
                actual: stft_mag.len(),
            });
        }

        let mut data = stft_mag.to_vec();
        let mut skips = Vec::with_capacity(DEPTH);
        let mut freq_dims = Vec::with_capacity(DEPTH);

        for block_idx in 0..DEPTH {
            let in_ch = self.block_in_ch[block_idx];
            let out_ch = self.block_out_ch[block_idx];
            let f_in = self.block_f_in[block_idx];
            let f_out = self.block_f_out[block_idx];
            let t = self.time_len;

            // Validate no NaN/Inf in data (single-pass count).
            crate::check_non_finite_err(&data, |count| {
                DemucsSpectralEncoderError::NonFiniteInput {
                    block: block_idx,
                    count,
                }
            })?;

            // Sub-def 1: Conv1d + GELU along freq axis.
            // Dispatch per time step: [C_in, F_in] → [C_out, F_out].
            let conv_out = dispatch_helpers::dispatch_per_time_step(
                cache,
                &self.block_sub_defs[block_idx].conv_gelu_def,
                &data,
                &self.block_weights[block_idx].conv_gelu,
                in_ch,
                out_ch,
                f_in,
                f_out,
                t,
            )?;

            // Sub-def 2: DConv along time axis.
            // Dispatch per freq bin: [C_out, T] → [C_out, T].
            let dconv_out = dispatch_helpers::dispatch_per_freq_bin(
                cache,
                &self.block_sub_defs[block_idx].dconv_def,
                &conv_out,
                &self.block_weights[block_idx].dconv,
                out_ch,
                f_out,
                t,
            )?;

            // Sub-def 3: Rewrite Conv1d(k=1) + GLU along freq axis.
            // Dispatch per time step: [C_out, F_out] → [C_out, F_out].
            let rewrite_out = dispatch_helpers::dispatch_per_time_step(
                cache,
                &self.block_sub_defs[block_idx].rewrite_def,
                &dconv_out,
                &self.block_weights[block_idx].rewrite,
                out_ch,
                out_ch,
                f_out,
                f_out,
                t,
            )?;

            // Apply frequency embedding after depth-0 output.
            data = if block_idx == 0 {
                self.apply_freq_emb(&rewrite_out, out_ch, f_out, t)?
            } else {
                rewrite_out
            };

            // Save skip connection: output of this block.
            skips.push(data.clone());
            freq_dims.push(f_out);
        }

        Ok(SpectralEncoderOutput {
            bottleneck: data,
            skips,
            freq_dims,
            time_dim: self.time_len,
        })
    }

    /// Apply frequency embedding to the depth-0 output.
    ///
    /// The frequency embedding is a learned `[FREQ_EMB_FEATURES, FREQ_EMB_DIM]`
    /// lookup table. For each freq bin f in 0..F', we look up row f of the
    /// embedding, scale by `FREQ_EMB_SCALE`, and broadcast-add to `[C, F', T]`.
    fn apply_freq_emb(
        &self,
        data: &[f32],
        channels: usize,
        freq_out: usize,
        time_len: usize,
    ) -> Result<Vec<f32>, DemucsSpectralEncoderError> {
        let emb_weight = match &self.freq_emb_weight {
            Some(w) => w,
            None => return Ok(data.to_vec()),
        };

        if channels != FREQ_EMB_DIM {
            return Err(DemucsSpectralEncoderError::FreqEmbMismatch {
                field: "channels",
                expected: FREQ_EMB_DIM,
                actual: channels,
            });
        }
        if freq_out > FREQ_EMB_FEATURES {
            return Err(DemucsSpectralEncoderError::FreqEmbMismatch {
                field: "freq_bins",
                expected: FREQ_EMB_FEATURES,
                actual: freq_out,
            });
        }

        let mut result = data.to_vec();

        // emb_weight is [FREQ_EMB_FEATURES, FREQ_EMB_DIM] row-major.
        // For each freq bin f, emb[f] = emb_weight[f * FREQ_EMB_DIM .. (f+1) * FREQ_EMB_DIM].
        // Broadcast-add: data[c, f, t] += emb[f][c] * FREQ_EMB_SCALE.
        for f in 0..freq_out {
            let emb_row_start = f * FREQ_EMB_DIM;
            for c in 0..channels {
                let emb_val = emb_weight[emb_row_start + c] * FREQ_EMB_SCALE;
                for t in 0..time_len {
                    let idx = c * freq_out * time_len + f * time_len + t;
                    result[idx] += emb_val;
                }
            }
        }

        Ok(result)
    }
}

pub(crate) use crate::demucs_shared::channels_at_depth;
