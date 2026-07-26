// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Demucs spectral decoder: 4-block decoder branch for HTDemucs source separation.
//!
//! Implements the spectral (frequency-domain) decoder of HTDemucs. Each
//! decoder block operates on 4D tensors `[C, F, T]` (batch=1) and uses:
//!
//!   skip_add → Rewrite(Conv2d 3×3) → GLU → axis_switch → DConv → axis_switch
//!   → axis_switch → ConvTranspose1d → axis_switch → [GELU]
//!
//! The axis-switch pattern collapses one spatial dimension into the batch
//! dimension so existing 1D operations (DConv, ConvTranspose1d) apply along
//! the other spatial axis. This is handled CPU-side in Phase 1 (no permute
//! op in tensor IR). Each block is split into 3 sub-`TensorKernelDef`s:
//! 1. Rewrite (Conv2d → GLU): operates on `[C, F*T]` (flattened 2D)
//! 2. DConv: operates on `[C, T]` per frequency bin (axis-switch: fold F into batch)
//! 3. ConvTranspose1d: operates on `[C, F]` per time step (axis-switch: fold T into batch)
//!
//! Part of #779 Phase B.

use std::collections::HashMap;

use nn_dsl::ScalarType;

use crate::tensor_dispatch::execute_tensor_dispatch;
use crate::PipelineCache;

// Import shared constants with local aliases for minimal diff.
use crate::demucs_shared::{
    SPECTRAL_BASIC_DEPTH as DEPTH, SPECTRAL_CONV_TR_PADDING as CONV_TR_PADDING,
    SPECTRAL_KERNEL_SIZE as KERNEL_SIZE, SPECTRAL_OUTPUT_CHANNELS as OUTPUT_CHANNELS,
    SPECTRAL_REWRITE_KERNEL as REWRITE_KERNEL, SPECTRAL_REWRITE_PADDING as REWRITE_PADDING,
    SPECTRAL_STRIDE,
};

#[path = "demucs_spectral_decoder_types.rs"]
mod types;
pub use types::{DemucsSpectralDecoderError, DemucsSpectralDecoderWeights};

#[path = "demucs_spectral_decoder_builders.rs"]
pub(crate) mod builders;
pub(crate) use builders::{BlockSubDefs, SpectralBlockWeightMaps};

#[path = "demucs_spectral_decoder_dispatch.rs"]
mod dispatch_helpers;

#[cfg(test)]
#[path = "demucs_spectral_decoder_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Demucs spectral decoder: 4-block skip-connected upsampling decoder.
///
/// Pre-builds all `TensorKernelDef`s at construction time. The `forward()`
/// method dispatches sub-defs with CPU-side axis-switch between stages.
#[must_use = "DemucsSpectralDecoder is constructed once and reused; call .forward() to run inference"]
pub struct DemucsSpectralDecoder {
    /// Sub-defs per decoder block (3 sub-defs each, 4 blocks).
    block_sub_defs: Vec<BlockSubDefs>,
    /// Pre-built weight maps per block.
    block_weights: Vec<SpectralBlockWeightMaps>,
    /// Frequency dimensions at each encoder depth input (for skip trim targets).
    encoder_freqs: Vec<usize>,
    /// Time dimensions at each encoder depth input.
    encoder_times: Vec<usize>,
    /// Input frequency dimension per block.
    block_f_in: Vec<usize>,
    /// Input time dimension per block.
    block_t_in: Vec<usize>,
    /// Channel counts at each block's input.
    block_in_ch: Vec<usize>,
    /// Output frequency dimension per block (after ConvTranspose).
    block_f_out: Vec<usize>,
}

impl std::fmt::Debug for DemucsSpectralDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemucsSpectralDecoder")
            .field("blocks", &self.block_sub_defs.len())
            .field("encoder_freqs", &self.encoder_freqs)
            .field("encoder_times", &self.encoder_times)
            .finish_non_exhaustive()
    }
}

impl DemucsSpectralDecoder {
    /// Construct a new spectral decoder, validating weights and building defs.
    ///
    /// `encoder_freqs`: frequency dim at each encoder depth input (depth 0..3).
    /// `encoder_times`: time dim at each encoder depth input (depth 0..3).
    pub fn new(
        weights: DemucsSpectralDecoderWeights,
        encoder_freqs: &[usize],
        encoder_times: &[usize],
    ) -> Result<Self, DemucsSpectralDecoderError> {
        if encoder_freqs.len() != DEPTH {
            return Err(DemucsSpectralDecoderError::DimMismatch {
                stage: "encoder_freqs".to_string(),
                expected: DEPTH,
                actual: encoder_freqs.len(),
            });
        }
        if encoder_times.len() != DEPTH {
            return Err(DemucsSpectralDecoderError::DimMismatch {
                stage: "encoder_times".to_string(),
                expected: DEPTH,
                actual: encoder_times.len(),
            });
        }

        builders::validate_all_weights(&weights)?;

        let mut block_sub_defs = Vec::with_capacity(DEPTH);
        let mut block_weight_maps = Vec::with_capacity(DEPTH);
        let mut block_f_in = Vec::with_capacity(DEPTH);
        let mut block_t_in = Vec::with_capacity(DEPTH);
        let mut block_in_ch = Vec::with_capacity(DEPTH);
        let mut block_f_out = Vec::with_capacity(DEPTH);

        // Track frequency/time dims flowing through decoder.
        // After deepest encoder, Conv1d reduces freq by stride.
        let mut prev_f = builders::conv1d_output_len(
            encoder_freqs[DEPTH - 1],
            KERNEL_SIZE,
            SPECTRAL_STRIDE,
            KERNEL_SIZE / 4,
        )?;
        // Time is preserved through the spectral branch (no temporal downsampling).
        let mut prev_t = encoder_times[DEPTH - 1];

        for block_idx in 0..DEPTH {
            let encoder_depth = DEPTH - 1 - block_idx;
            let in_ch = channels_at_depth(encoder_depth);
            let out_ch = if encoder_depth == 0 {
                OUTPUT_CHANNELS
            } else {
                channels_at_depth(encoder_depth - 1)
            };
            let is_last = encoder_depth == 0;
            let f_in = prev_f;
            let t_in = prev_t;

            // Rewrite Conv2d(3x3, s=1, p=1) preserves both F and T.
            let rw_f_out = builders::conv2d_output_len(f_in, REWRITE_KERNEL, 1, REWRITE_PADDING)?;
            let rw_t_out = builders::conv2d_output_len(t_in, REWRITE_KERNEL, 1, REWRITE_PADDING)?;

            // ConvTranspose1d on freq axis: upsample by SPECTRAL_STRIDE.
            let ct_f_out = (rw_f_out - 1) * SPECTRAL_STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING;

            // Trim to encoder freq at this depth.
            let target_f = ct_f_out.min(encoder_freqs[encoder_depth]);

            let sub_defs = builders::build_block_sub_defs(
                block_idx, in_ch, out_ch, f_in, t_in, rw_f_out, rw_t_out, target_f, is_last,
            )?;
            block_sub_defs.push(sub_defs);

            let weight_maps = builders::build_block_weight_maps(&weights.blocks[block_idx]);
            block_weight_maps.push(weight_maps);

            block_f_in.push(f_in);
            block_t_in.push(t_in);
            block_in_ch.push(in_ch);
            block_f_out.push(target_f);

            prev_f = target_f;
            prev_t = rw_t_out; // Time preserved through Conv2d rewrite.
        }

        Ok(Self {
            block_sub_defs,
            block_weights: block_weight_maps,
            encoder_freqs: encoder_freqs.to_vec(),
            encoder_times: encoder_times.to_vec(),
            block_f_in,
            block_t_in,
            block_in_ch,
            block_f_out,
        })
    }

    /// Run the spectral decoder forward pass.
    ///
    /// `bottleneck`: flattened `[C, F, T]` where C = channels_at_depth(DEPTH-1).
    /// `skips`: encoder skip connections in encoder order (depth 0..3),
    /// each flattened `[C, F, T]`.
    ///
    /// Returns flattened `[OUTPUT_CHANNELS, F_original, T]`.
    pub fn forward(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
    ) -> Result<Vec<f32>, DemucsSpectralDecoderError> {
        if skips.len() != DEPTH {
            return Err(DemucsSpectralDecoderError::DimMismatch {
                stage: "skips.len".to_string(),
                expected: DEPTH,
                actual: skips.len(),
            });
        }

        // Validate bottleneck length.
        let expected_bottleneck = self.block_in_ch[0] * self.block_f_in[0] * self.block_t_in[0];
        if bottleneck.len() != expected_bottleneck {
            return Err(DemucsSpectralDecoderError::DimMismatch {
                stage: "bottleneck".to_string(),
                expected: expected_bottleneck,
                actual: bottleneck.len(),
            });
        }

        let mut data = bottleneck.to_vec();

        for block_idx in 0..DEPTH {
            let encoder_depth = DEPTH - 1 - block_idx;
            let in_ch = self.block_in_ch[block_idx];
            let f_in = self.block_f_in[block_idx];
            let t_in = self.block_t_in[block_idx];
            let target_f = self.block_f_out[block_idx];
            let out_ch = if encoder_depth == 0 {
                OUTPUT_CHANNELS
            } else {
                channels_at_depth(encoder_depth - 1)
            };

            // Validate no NaN/Inf in data (single-pass count).
            crate::check_non_finite_err(&data, |count| {
                DemucsSpectralDecoderError::NonFiniteInput {
                    block: block_idx,
                    count,
                }
            })?;

            let skip_raw = &skips[encoder_depth];

            // Center-trim skip to match current [C, F, T].
            // Skip is [in_ch, skip_f, skip_t] — we trim F and T independently.
            let skip_spatial = skip_raw.len() / in_ch;
            let skip_f = skip_spatial / t_in;
            let skip_t = t_in; // Time is preserved through spectral branch.
            if skip_f < f_in {
                return Err(DemucsSpectralDecoderError::SkipTooShort {
                    depth: encoder_depth,
                    required_f: f_in,
                    required_t: t_in,
                    actual_f: skip_f,
                    actual_t: skip_t,
                });
            }
            let trimmed_skip =
                dispatch_helpers::center_trim_2d(skip_raw, in_ch, skip_f, skip_t, f_in, t_in)?;

            // Sub-def 1: Rewrite Conv2d(3×3) → GLU.
            // Input: [C, F*T] flattened. Conv2d operates on [C, F, T].
            let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
            inputs.insert(nn_dsl::input_names::DATA, &data);
            inputs.insert(nn_dsl::input_names::SKIP, &trimmed_skip);
            for (name, w) in &self.block_weights[block_idx].rewrite {
                inputs.insert(name.as_str(), w.as_slice());
            }
            let rewrite_out = execute_tensor_dispatch(
                cache,
                &self.block_sub_defs[block_idx].rewrite_def,
                ScalarType::F32,
                &inputs,
            )?;

            // After rewrite: [in_ch, rw_f, rw_t] — same spatial dims
            // (Conv2d k=3, s=1, p=1 preserves).
            let rw_f = f_in; // preserved
            let rw_t = t_in; // preserved

            // Sub-def 2: DConv along time axis.
            // Axis-switch: dispatch [C, T] per freq bin (matching Python's
            // B*F batch-folding pattern). See dispatch_helpers module.
            let dconv_out = dispatch_helpers::dispatch_per_freq_bin(
                cache,
                &self.block_sub_defs[block_idx].dconv_def,
                &rewrite_out,
                &self.block_weights[block_idx].dconv,
                in_ch,
                rw_f,
                rw_t,
            )?;

            // Sub-def 3: ConvTranspose1d along freq axis.
            // Dispatch T times with [C, F] input each time (axis-switch).
            let conv_tr_out = dispatch_helpers::dispatch_per_time_step(
                cache,
                &self.block_sub_defs[block_idx].conv_tr_def,
                &dconv_out,
                &self.block_weights[block_idx].conv_tr,
                in_ch,
                out_ch,
                rw_f,
                rw_t,
                target_f,
            )?;

            // After ConvTranspose: [out_ch, target_f, T] flattened.
            let is_last = encoder_depth == 0;
            if !is_last {
                data = conv_tr_out
                    .iter()
                    .map(|&v| dispatch_helpers::gelu_f32(v))
                    .collect();
            } else {
                data = conv_tr_out;
            }
        }

        Ok(data)
    }
}

pub(crate) use crate::demucs_shared::channels_at_depth;
