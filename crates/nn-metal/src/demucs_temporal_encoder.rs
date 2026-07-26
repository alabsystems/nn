// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Demucs temporal encoder: 4-block encoder branch for HTDemucs source separation.
//!
//! Implements the temporal (waveform) encoder of HTDemucs following the same
//! CPU round-trip dispatch pattern as [`DemucsTemporalDecoder`]. Each encoder block:
//!
//!   [stride_pad] → Conv1d → GELU → DConv → Rewrite(Conv1d k=1) → GLU
//!
//! All operations use existing nn ops (no new TensorOpKind variants).
//! Weight data is pre-validated at construction; `TensorKernelDef`s are built
//! once and reused on every `forward()` call.
//!
//! **Key differences from decoder:**
//! - Conv1d (downsample) instead of ConvTranspose1d (upsample)
//! - Rewrite kernel=1 instead of 3
//! - GELU between Conv1d and DConv (not after ConvTranspose1d)
//! - Produces skip connections (decoder consumes them)
//! - Input padded to stride multiple before Conv1d
//!
//! Part of #779 Phase E.

use crate::GpuWeightCache;
use std::borrow::Cow;
use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;

use crate::buffer::MetalBuffer;
use crate::element::MetalElement;
use crate::{PipelineCache, TensorDispatchError};

#[path = "demucs_temporal_encoder_builders.rs"]
pub(crate) mod builders;

#[path = "demucs_temporal_encoder_forward.rs"]
mod forward;

#[cfg(test)]
#[path = "demucs_temporal_encoder_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Configuration constants (from nn-models demucs_shared)
// ---------------------------------------------------------------------------

use crate::demucs_shared::{
    AUDIO_CHANNELS, TEMPORAL_BASIC_DEPTH as DEPTH, TEMPORAL_STRIDE as STRIDE,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from Demucs temporal encoder construction or forward pass.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DemucsTemporalEncoderError {
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

    /// Dispatch output length is not aligned to expected channel count.
    #[error("block {block} output length {actual} not divisible by {channels} channels")]
    OutputAlignment {
        block: usize,
        actual: usize,
        channels: usize,
    },

    /// Tensor IR construction error.
    #[error("tensor IR error: {0}")]
    TensorIr(#[from] nn_dsl::TensorIRError),

    /// GPU buffer allocation failed during weight upload.
    #[error("GPU buffer allocation failed: {0}")]
    GpuBufferAlloc(String),
}

impl From<nn_models::DemucsBuilderError> for DemucsTemporalEncoderError {
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

impl From<DemucsTemporalEncoderError> for nn_core::TensorError {
    fn from(e: DemucsTemporalEncoderError) -> Self {
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
pub use nn_models::demucs_temporal_weights::DemucsTemporalEncoderWeights;
pub(crate) use nn_models::demucs_temporal_weights::EncoderBlockWeights;

pub(crate) use crate::demucs_shared::channels_at_depth;

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Demucs temporal encoder: 4-block downsampling encoder.
///
/// Pre-builds all `TensorKernelDef`s at construction time. The `forward()`
/// method dispatches 4 blocks sequentially using CPU round-trip (Phase 1)
/// and returns the bottleneck, skip connections, and input lengths.
#[must_use = "DemucsTemporalEncoder is constructed once and reused; call .forward() to run inference"]
pub struct DemucsTemporalEncoder {
    /// One `TensorKernelDef` per encoder block (4 total).
    block_defs: Vec<TensorKernelDef>,
    /// Pre-built weight maps per block, matching def input names.
    /// Used by the CPU dispatch path.
    block_weights: Vec<HashMap<String, Vec<f32>>>,
    /// Channel counts at each block's output (for skip connection shapes).
    block_out_ch: Vec<usize>,
    /// Input temporal dim per block (before stride padding), pre-computed at
    /// construction. Used by `forward_gpu()` for audio length validation.
    block_t_in: Vec<usize>,
    /// Lazily-initialized GPU weight buffers for buffer-to-buffer dispatch.
    /// Populated on first `forward_gpu()` call from `block_weights`.
    /// Eliminates per-dispatch CPU→GPU weight re-upload (~80 memcpys/forward).
    block_gpu_weights: GpuWeightCache<Vec<HashMap<String, MetalBuffer>>>,
}

impl std::fmt::Debug for DemucsTemporalEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemucsTemporalEncoder")
            .field("blocks", &self.block_defs.len())
            .field("block_out_ch", &self.block_out_ch)
            .finish_non_exhaustive()
    }
}

/// Output of `DemucsTemporalEncoder::forward()`.
#[derive(Debug)]
#[must_use]
pub struct EncoderOutput {
    /// Bottleneck tensor: flattened `[channels_at_depth(DEPTH-1), T_bottleneck]`.
    pub bottleneck: Vec<f32>,
    /// Skip connections in encoder order (depth 0..3).
    /// Each is flattened `[channels_at_depth(d), T_d]`.
    pub skips: Vec<Vec<f32>>,
    /// GPU-resident skip connection buffers (encoder depth order, same as `skips`).
    ///
    /// Populated by `forward_gpu()`. When present, the decoder can use
    /// `DispatchInput::Gpu` for skip connections instead of re-uploading from CPU.
    /// `None` when forward was executed on CPU path.
    pub skips_gpu: Option<Vec<MetalBuffer>>,
    /// Input temporal length at each depth (before stride padding).
    /// Used by the decoder for center-trim targets.
    /// (Read in tests to verify encoder geometry; production decoder
    /// recomputes via `compute_encoder_input_lengths`.)
    #[allow(dead_code)]
    pub input_lengths: Vec<usize>,
}

impl DemucsTemporalEncoder {
    /// Construct a new temporal encoder, validating weights and building
    /// all `TensorKernelDef`s.
    ///
    /// `initial_t`: temporal length of the input audio waveform.
    pub fn new(
        weights: DemucsTemporalEncoderWeights,
        initial_t: usize,
    ) -> Result<Self, DemucsTemporalEncoderError> {
        builders::validate_all_weights(&weights)?;

        let mut block_defs = Vec::with_capacity(DEPTH);
        let mut block_weight_maps = Vec::with_capacity(DEPTH);
        let mut block_out_ch = Vec::with_capacity(DEPTH);
        let mut block_t_in_vec = Vec::with_capacity(DEPTH);

        let mut t_in = initial_t;

        for block_idx in 0..DEPTH {
            let in_ch = if block_idx == 0 {
                AUDIO_CHANNELS
            } else {
                channels_at_depth(block_idx - 1)
            };
            let out_ch = channels_at_depth(block_idx);

            block_t_in_vec.push(t_in);

            // Pad to stride multiple (matching Python HEncLayer).
            let padded_t = if !t_in.is_multiple_of(STRIDE) {
                t_in + (STRIDE - t_in % STRIDE)
            } else {
                t_in
            };

            let def = builders::build_encoder_block_def(block_idx, in_ch, out_ch, padded_t)?;
            block_defs.push(def);

            let weight_map = builders::build_encoder_weight_map(&weights.blocks[block_idx]);
            block_weight_maps.push(weight_map);
            block_out_ch.push(out_ch);

            // Output time after Conv1d stride downsampling.
            t_in = builders::conv1d_out_len(padded_t)?;
        }

        Ok(Self {
            block_defs,
            block_weights: block_weight_maps,
            block_out_ch,
            block_t_in: block_t_in_vec,
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
        DemucsTemporalEncoderError,
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
            DemucsTemporalEncoderError::GpuBufferAlloc,
        )
    }
}
