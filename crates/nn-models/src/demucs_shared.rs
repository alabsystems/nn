// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Demucs decoder infrastructure.
//!
//! Contains functions and types duplicated across the temporal, spectral, and
//! transformer decoder modules. Extracting them here removes ~140 lines of
//! character-for-character duplicates (Part of #826).
//!
//! - `conv1d_output_len` — was in temporal, spectral, and silero_vad builders
//! - `channels_at_depth` — was in temporal and spectral decoders
//! - `DConvSubLayerInputs` / `build_dconv_sublayer` — was in temporal and spectral builders
//! - `validate_weight_size` — generic version of 4 per-module `validate_weight` functions

use std::borrow::Cow;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorIRError, TensorNodeId};

use crate::DemucsBuilderError;

// ---------------------------------------------------------------------------
// Demucs architecture constants
// ---------------------------------------------------------------------------

/// Base channel count for Demucs HTDemucs architecture.
pub const BASE_CHANNELS: usize = 48;

/// Channel growth factor per encoder depth.
pub const GROWTH: f64 = 2.0;

/// DConv bottleneck compression ratio.
///
/// Real HTDemucs weights use a ratio of 4 (e.g., 48 channels / 4 = 12 compressed).
/// Verified against production `htdemucs.safetensors` (397 tensors, 83.6M params).
pub const DCONV_COMPRESS: usize = 4;

/// Number of DConv residual sub-layers per decoder block.
pub const DCONV_DEPTH: usize = 2;

/// DConv dilated convolution kernel size.
pub const DCONV_KERNEL: usize = 3;

/// GroupNorm epsilon for DConv normalization layers.
pub const GROUP_NORM_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Temporal encoder/decoder constants
// ---------------------------------------------------------------------------

/// Number of temporal encoder basic blocks (depths 0..3, each with Conv1d + DConv + Rewrite).
///
/// The temporal encoder also has a final block at depth 4 (Conv1d + norm, no DConv/rewrite),
/// giving `TEMPORAL_DEPTH` (5) total encoder depths. The decoder mirrors with 5 blocks (0..4).
pub const TEMPORAL_BASIC_DEPTH: usize = 4;

/// Total temporal encoder/decoder depth including the final norm block.
///
/// Real HTDemucs has time_encoder depths 0..4 and time_decoder depths 0..4.
/// Depths 0-3: basic blocks (Conv1d + GELU + DConv + Rewrite + GLU).
/// Depth 4: final block (Conv1d + GroupNorm, no DConv/rewrite).
pub const TEMPORAL_DEPTH: usize = 5;

/// Temporal convolution kernel size.
pub const TEMPORAL_KERNEL_SIZE: usize = 8;

/// Temporal convolution stride.
pub const TEMPORAL_STRIDE: usize = 4;

/// Conv1d padding for temporal encoder (kernel_size / 4, matching Python htdemucs).
pub const TEMPORAL_CONV_PADDING: usize = TEMPORAL_KERNEL_SIZE / 4;

/// ConvTranspose1d padding for temporal decoder (kernel_size / 4).
pub const TEMPORAL_CONV_TR_PADDING: usize = TEMPORAL_KERNEL_SIZE / 4;

/// Input audio channels (stereo).
pub const AUDIO_CHANNELS: usize = 2;

/// Decoder rewrite Conv1d kernel size.
pub const DECODER_REWRITE_KERNEL: usize = 3;

/// Decoder rewrite Conv1d padding.
pub const DECODER_REWRITE_PADDING: usize = DECODER_REWRITE_KERNEL / 2;

/// Decoder final output channels (4 sources x 2 stereo channels).
pub const DECODER_OUTPUT_CHANNELS: usize = 8;

// ---------------------------------------------------------------------------
// Spectral encoder/decoder constants
// ---------------------------------------------------------------------------

/// Number of spectral encoder basic blocks (depths 0..3, each with Conv2d + DConv + Rewrite).
///
/// Depths 4-5 are "deep" blocks with BiLSTM + local attention in the DConv sub-layers,
/// plus norm layers and a different rewrite structure.
pub const SPECTRAL_BASIC_DEPTH: usize = 4;

/// Total spectral encoder/decoder depth including deep (LSTM + attention) blocks.
///
/// Real HTDemucs has freq_encoder depths 0..5 and freq_decoder depths 0..5.
/// Depths 0-3: basic blocks (Conv + DConv + Rewrite).
/// Depths 4-5: deep blocks (Conv + DConv with BiLSTM + LocalAttention + Norms + Rewrite).
pub const SPECTRAL_DEPTH: usize = 6;

/// Spectral convolution stride (always 4 at every depth).
pub const SPECTRAL_STRIDE: usize = 4;

/// Spectral convolution kernel size.
pub const SPECTRAL_KERNEL_SIZE: usize = 8;

/// Conv1d padding for spectral encoder (kernel_size / 4).
pub const SPECTRAL_CONV_PADDING: usize = SPECTRAL_KERNEL_SIZE / 4;

/// ConvTranspose1d padding for spectral decoder (kernel_size / 4).
pub const SPECTRAL_CONV_TR_PADDING: usize = SPECTRAL_KERNEL_SIZE / 4;

/// Spectral input channels: 2 (stereo) x 2 (real + imaginary) = 4.
pub const SPECTRAL_INPUT_CHANNELS: usize = 4;

/// Spectral output channels: 4 sources x 2 stereo x 2 (real+imag) = 16.
pub const SPECTRAL_OUTPUT_CHANNELS: usize = 16;

/// Spectral decoder rewrite Conv2d kernel size (3x3).
pub const SPECTRAL_REWRITE_KERNEL: usize = 3;

/// Spectral decoder rewrite Conv2d padding (kernel / 2 = 1).
pub const SPECTRAL_REWRITE_PADDING: usize = SPECTRAL_REWRITE_KERNEL / 2;

/// Frequency embedding feature count (max freq bins that can be embedded).
pub const SPECTRAL_FREQ_EMB_FEATURES: usize = 512;

/// Frequency embedding dimension (must equal channels_at_depth(0) = 48).
pub const SPECTRAL_FREQ_EMB_DIM: usize = 48;

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

/// Conv1d output length.
///
/// Delegates to canonical `nn_core::conv1d_out_len` (dilation=1).
///
/// Note: parameter order here is `(in_len, kernel_size, stride, padding)` which
/// differs from canonical `(input_len, kernel_size, padding, stride, dilation)`.
/// This wrapper preserves the model-side convention.
pub fn conv1d_output_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, DemucsBuilderError> {
    Ok(nn_core::conv1d_out_len(
        in_len,
        kernel_size,
        padding,
        stride,
        1,
    )?)
}

/// Maximum encoder depth to prevent overflow in `channels_at_depth`.
///
/// At depth 30 with GROWTH=2.0: 48 x 2^30 ~ 5.2e10, which fits in usize on
/// 64-bit but is unreasonably large. Real architectures use depth <= 6.
const MAX_ENCODER_DEPTH: usize = 30;

/// Compute channel count at a given encoder depth.
///
/// # Panics
///
/// Panics if `depth > MAX_ENCODER_DEPTH` (30). This guards against `f64 -> usize`
/// overflow from `GROWTH.powi(depth)` producing values exceeding `usize::MAX`.
pub fn channels_at_depth(depth: usize) -> usize {
    assert!(
        depth <= MAX_ENCODER_DEPTH,
        "channels_at_depth: depth {depth} exceeds maximum {MAX_ENCODER_DEPTH}"
    );
    (BASE_CHANNELS as f64 * GROWTH.powi(depth as i32)) as usize
}

// ---------------------------------------------------------------------------
// Weight validation
// ---------------------------------------------------------------------------

/// Validate that a weight tensor has the expected number of elements.
///
/// Returns `Ok(())` if `data.len() == expected`, otherwise returns `Err(msg)`
/// with a descriptive message. Callers map this to their module-specific error
/// type via `.map_err()`.
pub fn validate_weight_size(
    data: &[f32],
    name: &str,
    expected: usize,
) -> Result<(), DemucsBuilderError> {
    if data.len() != expected {
        Err(DemucsBuilderError::WeightSize {
            name: Cow::Owned(name.to_string()),
            expected,
            actual: data.len(),
        })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DConv sub-layer (shared between temporal and spectral decoders)
// ---------------------------------------------------------------------------

/// Collected input node IDs for a single DConv sub-layer.
pub struct DConvSubLayerInputs {
    pub conv_compress_weight: TensorNodeId,
    pub conv_compress_bias: TensorNodeId,
    pub norm_compress_gamma: TensorNodeId,
    pub norm_compress_beta: TensorNodeId,
    pub conv_expand_weight: TensorNodeId,
    pub conv_expand_bias: TensorNodeId,
    pub norm_expand_gamma: TensorNodeId,
    pub norm_expand_beta: TensorNodeId,
    pub layer_scale: TensorNodeId,
    pub eps1: TensorNodeId,
    pub eps2: TensorNodeId,
    pub dilation: usize,
}

impl DConvSubLayerInputs {
    /// Add all DConv sub-layer inputs to the builder and return the collected IDs.
    pub fn add_to_builder(
        b: &mut TensorBlockBuilder,
        k: usize,
        channels: usize,
        compressed: usize,
    ) -> Self {
        let doubled = channels * 2;
        Self {
            conv_compress_weight: b
                .add_input(&format!("dc{k}_cw"), &[compressed, channels, DCONV_KERNEL]),
            conv_compress_bias: b.add_input(&format!("dc{k}_cb"), &[compressed]),
            norm_compress_gamma: b.add_input(&format!("dc{k}_ng"), &[compressed]),
            norm_compress_beta: b.add_input(&format!("dc{k}_nb"), &[compressed]),
            conv_expand_weight: b.add_input(&format!("dc{k}_ew"), &[doubled, compressed, 1]),
            conv_expand_bias: b.add_input(&format!("dc{k}_eb"), &[doubled]),
            norm_expand_gamma: b.add_input(&format!("dc{k}_eng"), &[doubled]),
            norm_expand_beta: b.add_input(&format!("dc{k}_enb"), &[doubled]),
            layer_scale: b.add_input(&format!("dc{k}_ls"), &[channels]),
            eps1: b.add_input(&format!("dc{k}_eps"), &[1]),
            eps2: b.add_input(&format!("dc{k}_eps2"), &[1]),
            dilation: 1 << k,
        }
    }
}

/// Build a DConv sub-layer inline within the block's builder.
///
/// Conv1d(dilated) -> GroupNorm -> GELU -> Conv1d(1x1) -> GroupNorm -> GLU ->
/// LayerScale -> residual_add.
pub fn build_dconv_sublayer(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    dc: &DConvSubLayerInputs,
    channels: usize,
    compressed: usize,
    t_len: usize,
) -> Result<TensorNodeId, TensorIRError> {
    let doubled = channels * 2;

    // Causal Conv1d: ZeroPad1d(left=(K-1)*D, right=0) -> Conv1d(padding=0).
    // Matches Python: F.pad(x, ((K-1)*D, 0)) + F.conv1d(h, w, dilation=D).
    let causal_pad_left = (DCONV_KERNEL - 1) * dc.dilation;
    let padded_t = t_len + causal_pad_left;
    let padded = b.add_zero_pad_1d(input, causal_pad_left, 0, &[channels, padded_t]);

    // Dilated Conv1d: [channels, padded_T] -> [compressed, T] (preserves time)
    let c1 = b.add_conv1d_full(
        padded,
        dc.conv_compress_weight,
        Some(dc.conv_compress_bias),
        1,
        0,
        dc.dilation,
        1,
        &[compressed, t_len],
    );

    // GroupNorm(1) on compressed channels
    let n1 = b.add_group_norm_g1(
        c1,
        dc.eps1,
        Some(dc.norm_compress_gamma),
        Some(dc.norm_compress_beta),
        compressed,
        t_len,
    );

    // GELU
    let g1 = b.add_gelu(n1, &[compressed, t_len]);

    // Conv1d expand: [compressed, T] -> [channels*2, T]
    let c2 = b.add_conv1d(
        g1,
        dc.conv_expand_weight,
        Some(dc.conv_expand_bias),
        1,
        0,
        &[doubled, t_len],
    );

    // GroupNorm(1) on expanded channels
    let n2 = b.add_group_norm_g1(
        c2,
        dc.eps2,
        Some(dc.norm_expand_gamma),
        Some(dc.norm_expand_beta),
        doubled,
        t_len,
    );

    // GLU: [channels*2, T] -> [channels, T]
    let glu = b.add_glu(n2, 0, &[doubled, t_len])?;

    // LayerScale: broadcast [channels] -> [channels, T], multiply
    let ls = b.add_layer_scale(glu, dc.layer_scale, &[channels, t_len]);

    // Residual: input + scaled
    Ok(b.add_binary_add(input, ls, &[channels, t_len]))
}

#[cfg(test)]
#[path = "demucs_shared_tests.rs"]
mod tests;
