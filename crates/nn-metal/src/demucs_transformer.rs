// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HTDemucs cross-domain transformer bottleneck.
//!
//! Implements the 5-layer alternating self/cross-attention transformer that
//! connects the temporal and spectral encoder branches of HTDemucs. Architecture:
//!
//! 1. Channel upsample: Conv1d (384 → 512) per branch
//! 2. Flatten to sequence + LayerNorm + positional embedding
//! 3. 5 transformer layers (alternating self/cross-attention)
//! 4. Unflatten + Channel downsample: Conv1d (512 → 384) per branch
//!
//! Each self-attention layer: `x += gamma_1 * MHA(LN1(x)); x += gamma_2 * FFN(LN2(x)); x = LN_out(x)`
//! Each cross-attention layer: `x += gamma_1 * CrossMHA(Q=LN1(x), KV=LN2(cross)); x += gamma_2 * FFN(LN3(x)); x = LN_out(x)`
//!
//! Forward pass and dispatch helpers extracted to `demucs_transformer_forward.rs`
//! in #833 to keep files under the 500-line limit. GPU weight management and
//! builder helpers extracted to `demucs_transformer_gpu.rs` in #1410 D4.
//!
//! Part of #779 Phase D.

use crate::GpuWeightCache;
use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_models::TransformerBuildError;

use crate::TensorDispatchError;

/// Weight types — thin re-export wrapper delegating to nn-models.
#[path = "demucs_transformer_weights.rs"]
mod weights;
pub use weights::*;

/// Architecture constants re-exported from nn-models (backend-agnostic).
use nn_models::demucs_transformer_constants::{BOTTLENECK_DIM, NUM_LAYERS, TRANSFORMER_DIM};

/// Builders — thin re-export wrapper delegating to nn-models.
#[path = "demucs_transformer_builders.rs"]
mod builders;

/// Validation — thin re-export wrapper delegating to nn-models.
#[path = "demucs_transformer_validate.rs"]
pub(crate) mod validate;

#[path = "demucs_transformer_helpers.rs"]
mod helpers;

#[cfg(test)]
#[path = "demucs_transformer_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "demucs_transformer_kani_tests.rs"]
mod kani_proofs;

#[path = "demucs_transformer_forward.rs"]
mod forward;

#[path = "demucs_transformer_gpu.rs"]
mod gpu;
use gpu::TransformerGpuWeights;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from Demucs transformer construction or forward pass.
///
/// Wraps backend-agnostic [`TransformerBuildError`] (from nn-models) for
/// construction/validation errors, and adds Metal-specific dispatch errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DemucsTransformerError {
    /// Backend-agnostic build/validation error (weight size, dim mismatch, IR).
    #[error(transparent)]
    Build(#[from] TransformerBuildError),

    /// GPU dispatch failed.
    #[error("dispatch error: {0}")]
    Dispatch(#[from] TensorDispatchError),

    /// Non-finite values in input data (runtime, not construction).
    #[error("non-finite input at layer {layer}: {count} NaN/Inf values")]
    NonFiniteInput { layer: usize, count: usize },

    /// GPU buffer allocation failed during weight upload.
    #[error("GPU buffer allocation failed: {0}")]
    GpuBufferAlloc(String),
}

/// Chain `TensorIRError` → `TransformerBuildError` → `DemucsTransformerError`.
/// Enables `?` in the constructor where builders return `TensorIRError`.
impl From<nn_dsl::TensorIRError> for DemucsTransformerError {
    fn from(e: nn_dsl::TensorIRError) -> Self {
        Self::Build(TransformerBuildError::from(e))
    }
}

impl From<DemucsTransformerError> for nn_core::TensorError {
    fn from(e: DemucsTransformerError) -> Self {
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
// Model struct
// ---------------------------------------------------------------------------

/// HTDemucs cross-domain transformer bottleneck.
///
/// Pre-builds `TensorKernelDef`s for all components at construction. The
/// `forward()` method alternates self/cross-attention across 5 layers, processing
/// temporal and spectral branches simultaneously. Uses CPU round-trip dispatch
/// (Phase 1 pattern).
///
/// Layer pattern: even indices (0, 2, 4) = self-attention, odd (1, 3) = cross-attention.
/// Cross-attention: spectral attends to temporal, temporal attends to spectral.
/// Both read from the previous iteration's outputs (no cascade within a single layer).
#[must_use = "DemucsTransformer is constructed once and reused; call .forward() to run inference"]
pub struct DemucsTransformer {
    /// Channel upsample defs (temporal, spectral).
    upsample_t_def: TensorKernelDef,
    upsample_s_def: TensorKernelDef,
    /// Input LayerNorm defs (temporal, spectral).
    norm_in_t_def: TensorKernelDef,
    norm_in_s_def: TensorKernelDef,
    /// Temporal transformer layer defs (5).
    temporal_layer_defs: Vec<TensorKernelDef>,
    /// Spectral transformer layer defs (5).
    spectral_layer_defs: Vec<TensorKernelDef>,
    /// Channel downsample defs (temporal, spectral).
    downsample_t_def: TensorKernelDef,
    downsample_s_def: TensorKernelDef,
    /// Pre-built weight maps for each component.
    upsample_t_weights: HashMap<String, Vec<f32>>,
    upsample_s_weights: HashMap<String, Vec<f32>>,
    norm_in_t_weights: HashMap<String, Vec<f32>>,
    norm_in_s_weights: HashMap<String, Vec<f32>>,
    temporal_layer_weights: Vec<HashMap<String, Vec<f32>>>,
    spectral_layer_weights: Vec<HashMap<String, Vec<f32>>>,
    downsample_t_weights: HashMap<String, Vec<f32>>,
    downsample_s_weights: HashMap<String, Vec<f32>>,
    /// GPU transpose defs: [C, T] → [T, C] and [T, C] → [C, T].
    transpose_ct_tc_t_def: TensorKernelDef,
    transpose_ct_tc_s_def: TensorKernelDef,
    transpose_tc_ct_t_def: TensorKernelDef,
    transpose_tc_ct_s_def: TensorKernelDef,
    /// GPU sinusoidal positional embedding defs (element-wise add of precomputed table).
    sinusoidal_t_def: TensorKernelDef,
    sinusoidal_s_def: TensorKernelDef,
    sinusoidal_t_weights: HashMap<String, Vec<f32>>,
    sinusoidal_s_weights: HashMap<String, Vec<f32>>,
    /// Sequence lengths for temporal and spectral branches.
    temporal_seq_len: usize,
    spectral_seq_len: usize,
    /// Lazily-initialized GPU weight buffers for buffer-to-buffer dispatch.
    /// Populated on first `forward_gpu()` call. Eliminates per-dispatch
    /// CPU→GPU weight re-upload for all 18 weight maps.
    gpu_weights: GpuWeightCache<TransformerGpuWeights>,
}

impl std::fmt::Debug for DemucsTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemucsTransformer")
            .field("layers", &self.temporal_layer_defs.len())
            .field("temporal_seq_len", &self.temporal_seq_len)
            .field("spectral_seq_len", &self.spectral_seq_len)
            .finish_non_exhaustive()
    }
}

impl DemucsTransformer {
    /// Construct a new transformer bottleneck, validating weights and
    /// building all `TensorKernelDef`s.
    ///
    /// `temporal_seq_len`: T dimension of temporal bottleneck `[384, T]`.
    /// `spectral_seq_len`: F*T dimension of spectral bottleneck (after flattening `[384, F, T]`).
    pub fn new(
        weights: DemucsTransformerWeights,
        temporal_seq_len: usize,
        spectral_seq_len: usize,
    ) -> Result<Self, DemucsTransformerError> {
        builders::validate_all_weights(&weights)?;

        if weights.temporal_layers.len() != NUM_LAYERS {
            return Err(TransformerBuildError::DimMismatch {
                stage: "temporal_layers.len".to_string(),
                expected: NUM_LAYERS,
                actual: weights.temporal_layers.len(),
            }
            .into());
        }
        if weights.spectral_layers.len() != NUM_LAYERS {
            return Err(TransformerBuildError::DimMismatch {
                stage: "spectral_layers.len".to_string(),
                expected: NUM_LAYERS,
                actual: weights.spectral_layers.len(),
            }
            .into());
        }

        // Build channel upsample defs (Conv1d, kernel=1, stride=1, padding=0).
        let (upsample_t_def, _) = builders::build_channel_bridge_def(
            "upsample_t",
            BOTTLENECK_DIM,
            TRANSFORMER_DIM,
            temporal_seq_len,
        )?;
        let (upsample_s_def, _) = builders::build_channel_bridge_def(
            "upsample_s",
            BOTTLENECK_DIM,
            TRANSFORMER_DIM,
            spectral_seq_len,
        )?;

        // Build input LayerNorm defs (operate on [T, D] after flatten+transpose).
        let (norm_in_t_def, norm_in_t_weights) =
            builders::build_layer_norm_def("norm_in_t", temporal_seq_len, &weights.norm_in_t)?;
        let (norm_in_s_def, norm_in_s_weights) =
            builders::build_layer_norm_def("norm_in_s", spectral_seq_len, &weights.norm_in_s)?;

        // Build transformer layer defs.
        let mut temporal_layer_defs = Vec::with_capacity(NUM_LAYERS);
        let mut spectral_layer_defs = Vec::with_capacity(NUM_LAYERS);
        let mut temporal_layer_weight_maps = Vec::with_capacity(NUM_LAYERS);
        let mut spectral_layer_weight_maps = Vec::with_capacity(NUM_LAYERS);

        for i in 0..NUM_LAYERS {
            let is_cross = i % 2 == 1;

            // Temporal layers: self-attn uses temporal_seq_len;
            // cross-attn has Q from temporal, KV from spectral.
            let (t_def, t_wmap) = if is_cross {
                builders::build_cross_attention_layer_def(
                    &format!("temporal_{i}"),
                    temporal_seq_len,
                    spectral_seq_len,
                    &weights.temporal_layers[i],
                )?
            } else {
                builders::build_self_attention_layer_def(
                    &format!("temporal_{i}"),
                    temporal_seq_len,
                    &weights.temporal_layers[i],
                )?
            };
            temporal_layer_defs.push(t_def);
            temporal_layer_weight_maps.push(t_wmap);

            // Spectral layers: self-attn uses spectral_seq_len;
            // cross-attn has Q from spectral, KV from temporal.
            let (s_def, s_wmap) = if is_cross {
                builders::build_cross_attention_layer_def(
                    &format!("spectral_{i}"),
                    spectral_seq_len,
                    temporal_seq_len,
                    &weights.spectral_layers[i],
                )?
            } else {
                builders::build_self_attention_layer_def(
                    &format!("spectral_{i}"),
                    spectral_seq_len,
                    &weights.spectral_layers[i],
                )?
            };
            spectral_layer_defs.push(s_def);
            spectral_layer_weight_maps.push(s_wmap);
        }

        // Build channel downsample defs.
        let (downsample_t_def, _) = builders::build_channel_bridge_def(
            "downsample_t",
            TRANSFORMER_DIM,
            BOTTLENECK_DIM,
            temporal_seq_len,
        )?;
        let (downsample_s_def, _) = builders::build_channel_bridge_def(
            "downsample_s",
            TRANSFORMER_DIM,
            BOTTLENECK_DIM,
            spectral_seq_len,
        )?;

        // Build GPU transpose defs: [C, T] → [T, C] (post-upsample) and
        // [T, C] → [C, T] (pre-downsample). No weights — pure index remapping.
        let transpose_ct_tc_t_def =
            Self::build_transpose_def("transpose_ct_tc_t", TRANSFORMER_DIM, temporal_seq_len)?;
        let transpose_ct_tc_s_def =
            Self::build_transpose_def("transpose_ct_tc_s", TRANSFORMER_DIM, spectral_seq_len)?;
        let transpose_tc_ct_t_def =
            Self::build_transpose_def("transpose_tc_ct_t", temporal_seq_len, TRANSFORMER_DIM)?;
        let transpose_tc_ct_s_def =
            Self::build_transpose_def("transpose_tc_ct_s", spectral_seq_len, TRANSFORMER_DIM)?;

        // Build GPU sinusoidal positional embedding defs. Precompute the
        // embedding table at construction and store as a weight tensor.
        let sinusoidal_t_table = helpers::build_sinusoidal_table(temporal_seq_len, TRANSFORMER_DIM);
        let sinusoidal_s_table = helpers::build_sinusoidal_table(spectral_seq_len, TRANSFORMER_DIM);
        let sinusoidal_t_def =
            Self::build_sinusoidal_add_def("sinusoidal_t", temporal_seq_len, TRANSFORMER_DIM)?;
        let sinusoidal_s_def =
            Self::build_sinusoidal_add_def("sinusoidal_s", spectral_seq_len, TRANSFORMER_DIM)?;

        Ok(Self {
            upsample_t_def,
            upsample_s_def,
            norm_in_t_def,
            norm_in_s_def,
            temporal_layer_defs,
            spectral_layer_defs,
            downsample_t_def,
            downsample_s_def,
            upsample_t_weights: builders::build_conv1d_weight_map(
                &weights.channel_upsampler_t_weight,
                &weights.channel_upsampler_t_bias,
            ),
            upsample_s_weights: builders::build_conv1d_weight_map(
                &weights.channel_upsampler_s_weight,
                &weights.channel_upsampler_s_bias,
            ),
            norm_in_t_weights,
            norm_in_s_weights,
            temporal_layer_weights: temporal_layer_weight_maps,
            spectral_layer_weights: spectral_layer_weight_maps,
            downsample_t_weights: builders::build_conv1d_weight_map(
                &weights.channel_downsampler_t_weight,
                &weights.channel_downsampler_t_bias,
            ),
            downsample_s_weights: builders::build_conv1d_weight_map(
                &weights.channel_downsampler_s_weight,
                &weights.channel_downsampler_s_bias,
            ),
            transpose_ct_tc_t_def,
            transpose_ct_tc_s_def,
            transpose_tc_ct_t_def,
            transpose_tc_ct_s_def,
            sinusoidal_t_def,
            sinusoidal_s_def,
            sinusoidal_t_weights: HashMap::from([("sinusoidal".to_string(), sinusoidal_t_table)]),
            sinusoidal_s_weights: HashMap::from([("sinusoidal".to_string(), sinusoidal_s_table)]),
            temporal_seq_len,
            spectral_seq_len,
            gpu_weights: GpuWeightCache::new(),
        })
    }
}
