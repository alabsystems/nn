// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight types for the Demucs transformer bottleneck.
//!
//! Extracted from `demucs_transformer.rs` to stay under the 500-line limit.
//! Part of #779 — Phase D.

/// Weights for a single LayerNorm.
#[derive(Debug, Clone)]
#[must_use]
pub struct LayerNormWeights {
    /// Gamma: [dim].
    pub weight: Vec<f32>,
    /// Beta: [dim].
    pub bias: Vec<f32>,
}

/// Weights for a single self-attention transformer layer.
///
/// Contains: 3 LayerNorms (norm1, norm2, norm_out) + MHA (Q/K/V/out projections)
/// + FFN (linear1/linear2) + 2 LayerScales (gamma_1, gamma_2).
#[derive(Debug, Clone)]
#[must_use]
pub struct SelfAttentionLayerWeights {
    /// Pre-norm before attention.
    pub norm1: LayerNormWeights,
    /// Pre-norm before FFN.
    pub norm2: LayerNormWeights,
    /// Post-norm after both residuals.
    pub norm_out: LayerNormWeights,
    /// Q projection: [D, D].
    pub q_weight: Vec<f32>,
    /// K projection: [D, D].
    pub k_weight: Vec<f32>,
    /// V projection: [D, D].
    pub v_weight: Vec<f32>,
    /// Output projection: [D, D].
    pub out_weight: Vec<f32>,
    /// FFN linear1: [FFN_DIM, D].
    pub ffn_linear1_weight: Vec<f32>,
    /// FFN linear1 bias: [FFN_DIM].
    pub ffn_linear1_bias: Vec<f32>,
    /// FFN linear2: [D, FFN_DIM].
    pub ffn_linear2_weight: Vec<f32>,
    /// FFN linear2 bias: [D].
    pub ffn_linear2_bias: Vec<f32>,
    /// LayerScale gamma_1: [D].
    pub gamma_1: Vec<f32>,
    /// LayerScale gamma_2: [D].
    pub gamma_2: Vec<f32>,
}

/// Weights for a single cross-attention transformer layer.
///
/// Contains: 4 LayerNorms (norm1, norm2, norm3, norm_out) + cross-MHA
/// + FFN (linear1/linear2) + 2 LayerScales (gamma_1, gamma_2).
#[derive(Debug, Clone)]
#[must_use]
pub struct CrossAttentionLayerWeights {
    /// Pre-norm for query (own branch).
    pub norm1: LayerNormWeights,
    /// Pre-norm for key/value (cross branch).
    pub norm2: LayerNormWeights,
    /// Pre-norm before FFN.
    pub norm3: LayerNormWeights,
    /// Post-norm after both residuals.
    pub norm_out: LayerNormWeights,
    /// Q projection: [D, D].
    pub q_weight: Vec<f32>,
    /// K projection: [D, D].
    pub k_weight: Vec<f32>,
    /// V projection: [D, D].
    pub v_weight: Vec<f32>,
    /// Output projection: [D, D].
    pub out_weight: Vec<f32>,
    /// FFN linear1: [FFN_DIM, D].
    pub ffn_linear1_weight: Vec<f32>,
    /// FFN linear1 bias: [FFN_DIM].
    pub ffn_linear1_bias: Vec<f32>,
    /// FFN linear2: [D, FFN_DIM].
    pub ffn_linear2_weight: Vec<f32>,
    /// FFN linear2 bias: [D].
    pub ffn_linear2_bias: Vec<f32>,
    /// LayerScale gamma_1: [D].
    pub gamma_1: Vec<f32>,
    /// LayerScale gamma_2: [D].
    pub gamma_2: Vec<f32>,
}

/// Weights for a single transformer layer (either self- or cross-attention).
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub enum TransformerLayerWeights {
    SelfAttention(SelfAttentionLayerWeights),
    CrossAttention(CrossAttentionLayerWeights),
}

/// All weights for the Demucs transformer bottleneck.
#[derive(Debug, Clone)]
#[must_use = "DemucsTransformerWeights is a data transfer type; pass to DemucsTransformer::new()"]
pub struct DemucsTransformerWeights {
    /// Channel upsample Conv1d: temporal branch [512, 384, 1].
    pub channel_upsampler_t_weight: Vec<f32>,
    /// Channel upsample Conv1d bias: temporal [512].
    pub channel_upsampler_t_bias: Vec<f32>,
    /// Channel upsample Conv1d: spectral branch [512, 384, 1].
    pub channel_upsampler_s_weight: Vec<f32>,
    /// Channel upsample Conv1d bias: spectral [512].
    pub channel_upsampler_s_bias: Vec<f32>,
    /// Input LayerNorm: temporal branch.
    pub norm_in_t: LayerNormWeights,
    /// Input LayerNorm: spectral branch.
    pub norm_in_s: LayerNormWeights,
    /// Temporal transformer layers (5).
    pub temporal_layers: Vec<TransformerLayerWeights>,
    /// Spectral (shared) transformer layers (5).
    pub spectral_layers: Vec<TransformerLayerWeights>,
    /// Channel downsample Conv1d: temporal branch [384, 512, 1].
    pub channel_downsampler_t_weight: Vec<f32>,
    /// Channel downsample Conv1d bias: temporal [384].
    pub channel_downsampler_t_bias: Vec<f32>,
    /// Channel downsample Conv1d: spectral branch [384, 512, 1].
    pub channel_downsampler_s_weight: Vec<f32>,
    /// Channel downsample Conv1d bias: spectral [384].
    pub channel_downsampler_s_bias: Vec<f32>,
}
