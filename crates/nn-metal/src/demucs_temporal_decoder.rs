// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Demucs temporal decoder: 4-block decoder branch for HTDemucs source separation.
//!
//! Implements the temporal (waveform) decoder of HTDemucs following the same
//! CPU round-trip dispatch pattern as [`SileroVad`]. Each decoder block:
//!
//!   skip_add → Rewrite(Conv1d) → GLU → DConv → ConvTranspose1d → trim → [GELU]
//!
//! All operations use existing nn ops (no new TensorOpKind variants).
//! Weight data is pre-validated at construction; `TensorKernelDef`s are built
//! once and reused on every `forward()` call.
//!
//! Part of #779 Phase A.

use crate::GpuWeightCache;
use std::borrow::Cow;
use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;

use crate::buffer::MetalBuffer;
use crate::element::MetalElement;
use crate::{PipelineCache, TensorDispatchError};

#[path = "demucs_temporal_decoder_builders.rs"]
pub(crate) mod builders;

#[path = "demucs_temporal_decoder_forward.rs"]
mod forward;

#[cfg(test)]
use forward::center_trim_1d;

#[cfg(test)]
#[path = "demucs_temporal_decoder_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Configuration constants (from nn-models demucs_shared)
// ---------------------------------------------------------------------------

use crate::demucs_shared::{
    channels_at_depth, DECODER_OUTPUT_CHANNELS as OUTPUT_CHANNELS,
    DECODER_REWRITE_KERNEL as REWRITE_KERNEL, DECODER_REWRITE_PADDING as REWRITE_PADDING,
    TEMPORAL_BASIC_DEPTH as DEPTH, TEMPORAL_CONV_TR_PADDING as CONV_TR_PADDING,
    TEMPORAL_KERNEL_SIZE as KERNEL_SIZE, TEMPORAL_STRIDE as STRIDE,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from Demucs temporal decoder construction or forward pass.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DemucsTemporalDecoderError {
    /// Weight tensor has wrong element count.
    #[error("weight '{name}' expected {expected} elements, got {actual}")]
    WeightSize {
        name: Cow<'static, str>,
        expected: usize,
        actual: usize,
    },

    /// GPU dispatch failed.
    #[error("dispatch error: {0}")]
    Dispatch(#[from] TensorDispatchError),

    /// Dimension mismatch (construction or dispatch output).
    #[error("stage '{stage}' expected {expected} elements, got {actual}")]
    DimMismatch {
        stage: String,
        expected: usize,
        actual: usize,
    },

    /// Non-finite values in input data.
    #[error("non-finite input at block {block}: {count} NaN/Inf values")]
    NonFiniteInput { block: usize, count: usize },

    /// Skip connection too short for center-trim.
    #[error("skip[{depth}] temporal dim {actual} < required {required}")]
    SkipTooShort {
        depth: usize,
        required: usize,
        actual: usize,
    },

    /// Tensor IR construction error (e.g., GLU odd dimension).
    #[error("tensor IR error: {0}")]
    TensorIr(#[from] nn_dsl::TensorIRError),

    /// GPU buffer allocation failed during weight upload.
    #[error("GPU buffer allocation failed: {0}")]
    GpuBufferAlloc(String),

    /// Kernel build error (e.g., GPU center-trim narrow def).
    #[error("kernel build: {0}")]
    KernelBuild(#[from] nn_core::TensorError),
}

impl From<nn_models::DemucsBuilderError> for DemucsTemporalDecoderError {
    fn from(e: nn_models::DemucsBuilderError) -> Self {
        match e {
            nn_models::DemucsBuilderError::WeightSize {
                name,
                expected,
                actual,
            } => Self::WeightSize {
                name,
                expected,
                actual,
            },
            nn_models::DemucsBuilderError::BlockCountMismatch {
                context,
                expected,
                actual,
            } => Self::WeightSize {
                name: context,
                expected,
                actual,
            },
            other => Self::DimMismatch {
                stage: other.to_string(),
                expected: 0,
                actual: 0,
            },
        }
    }
}

impl From<DemucsTemporalDecoderError> for nn_core::TensorError {
    fn from(e: DemucsTemporalDecoderError) -> Self {
        let msg = e.to_string();
        Self::backend_failure_with_source(
            nn_core::BackendDomain::Metal,
            nn_core::BackendErrorKind::Other,
            msg,
            e,
        )
    }
}

// ---------------------------------------------------------------------------
// Weight types
// ---------------------------------------------------------------------------

/// Weight types re-exported from nn-models (backend-agnostic).
pub use nn_models::demucs_temporal_weights::DemucsTemporalDecoderWeights;
pub(crate) use nn_models::demucs_temporal_weights::{DConvSubLayerWeights, DecoderBlockWeights};

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Demucs temporal decoder: 4-block skip-connected upsampling decoder.
///
/// Pre-builds all `TensorKernelDef`s at construction time. The `forward()`
/// method dispatches 4 blocks sequentially using CPU round-trip (Phase 1).
#[must_use = "DemucsTemporalDecoder is constructed once and reused; call .forward() to run inference"]
pub struct DemucsTemporalDecoder {
    /// One `TensorKernelDef` per decoder block (4 total).
    block_defs: Vec<TensorKernelDef>,
    /// Pre-built weight maps per block, matching def input names.
    /// Used by the CPU dispatch path.
    block_weights: Vec<HashMap<String, Vec<f32>>>,
    /// Encoder input time lengths per depth (for trim targets).
    encoder_lengths: Vec<usize>,
    /// Input temporal lengths per block (for skip center-trim and validation).
    block_t_in: Vec<usize>,
    /// Channel counts at each block's input.
    block_in_ch: Vec<usize>,
    /// Lazily-initialized GPU weight buffers for buffer-to-buffer dispatch.
    /// Populated on first `forward_gpu()` call from `block_weights`.
    /// Eliminates per-dispatch CPU→GPU weight re-upload.
    block_gpu_weights: GpuWeightCache<Vec<HashMap<String, MetalBuffer>>>,
}

impl std::fmt::Debug for DemucsTemporalDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemucsTemporalDecoder")
            .field("blocks", &self.block_defs.len())
            .field("encoder_lengths", &self.encoder_lengths)
            .field("block_t_in", &self.block_t_in)
            .finish_non_exhaustive()
    }
}

impl DemucsTemporalDecoder {
    /// Construct a new temporal decoder, validating weights and building
    /// all `TensorKernelDef`s.
    ///
    /// `encoder_lengths`: time dimension at each encoder depth input (depth 0..3).
    /// Used as trim targets for ConvTranspose1d outputs.
    pub fn new(
        weights: DemucsTemporalDecoderWeights,
        encoder_lengths: &[usize],
    ) -> Result<Self, DemucsTemporalDecoderError> {
        if encoder_lengths.len() != DEPTH {
            return Err(DemucsTemporalDecoderError::DimMismatch {
                stage: "encoder_lengths".to_string(),
                expected: DEPTH,
                actual: encoder_lengths.len(),
            });
        }

        builders::validate_all_weights(&weights)?;

        let mut block_defs = Vec::with_capacity(DEPTH);
        let mut block_weight_maps = Vec::with_capacity(DEPTH);
        let mut block_t_in = Vec::with_capacity(DEPTH);
        let mut block_in_ch = Vec::with_capacity(DEPTH);

        // Track the actual temporal length flowing through decoder blocks.
        // The ConvTranspose output may not exactly match the encoder input at
        // each depth (encoder Conv1d with padding truncates asymmetrically).
        // We use the actual ConvTranspose output as the next block's input,
        // NOT the encoder length — skip connections handle the mismatch via
        // center_trim (matching Python HTDemucs behavior).
        let mut prev_t_out = builders::conv1d_output_len(
            encoder_lengths[DEPTH - 1],
            KERNEL_SIZE,
            STRIDE,
            KERNEL_SIZE / 4,
        )?;

        for block_idx in 0..DEPTH {
            let encoder_depth = DEPTH - 1 - block_idx;
            let in_ch = channels_at_depth(encoder_depth);
            let out_ch = if encoder_depth == 0 {
                OUTPUT_CHANNELS
            } else {
                channels_at_depth(encoder_depth - 1)
            };
            let is_last = encoder_depth == 0;
            let t_in = prev_t_out;

            // ConvTranspose1d output length (before any trim).
            // Rewrite Conv1d preserves T (k=3, s=1, p=1), so rw_t_out == t_in.
            let rw_t_out = builders::conv1d_output_len(t_in, REWRITE_KERNEL, 1, REWRITE_PADDING)?;
            let ct_t_out = (rw_t_out - 1) * STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING;

            // Target: trim to encoder input at this depth, or use full output
            // if ConvTranspose doesn't reach that length (happens when input
            // isn't valid_length-padded).
            let target_len = ct_t_out.min(encoder_lengths[encoder_depth]);

            let def = builders::build_decoder_block_def(
                block_idx, in_ch, out_ch, t_in, target_len, is_last,
            )?;
            block_defs.push(def);

            let weight_map = builders::build_decoder_weight_map(&weights.blocks[block_idx]);
            block_weight_maps.push(weight_map);
            block_t_in.push(t_in);
            block_in_ch.push(in_ch);

            // Next block's input is this block's output (after trim).
            prev_t_out = target_len;
        }

        Ok(Self {
            block_defs,
            block_weights: block_weight_maps,
            encoder_lengths: encoder_lengths.to_vec(),
            block_t_in,
            block_in_ch,
            block_gpu_weights: GpuWeightCache::new(),
        })
    }

    /// Lazily upload all CPU weight data to persistent GPU buffers.
    ///
    /// Called on the first `forward_gpu()` invocation. Subsequent calls return
    /// the cached buffers (via `OnceLock`). Each weight `Vec<f32>` is uploaded
    /// once via `f32::create_buffer`, and the resulting `MetalBuffer` is reused
    /// for all future dispatches as `DispatchInput::Gpu` (zero-copy alias).
    fn ensure_gpu_weights(
        &self,
        cache: &PipelineCache,
    ) -> Result<
        crate::GpuWeightRef<'_, Vec<HashMap<String, MetalBuffer>>>,
        DemucsTemporalDecoderError,
    > {
        self.block_gpu_weights.get_or_init_with(
            || {
                let ctx = cache.context();
                self.block_weights
                    .iter()
                    .map(|weight_map| {
                        weight_map
                            .iter()
                            .map(|(name, data)| {
                                let buf = f32::create_buffer(ctx, data)
                                    .map_err(|e| format!("{name}: {e}"))?;
                                Ok((name.clone(), buf))
                            })
                            .collect::<Result<HashMap<String, MetalBuffer>, String>>()
                    })
                    .collect()
            },
            DemucsTemporalDecoderError::GpuBufferAlloc,
        )
    }
}
