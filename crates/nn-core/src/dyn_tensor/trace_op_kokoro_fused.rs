// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro-specific fused activation and normalization operations.
//!
//! These variants combine multiple ops (AdaIN, activation, conv) into single
//! kernel dispatches for the Kokoro TTS pipeline. Extracted from `TraceOp`
//! to keep trace_types.rs under the 450-line file limit.

use super::ResBlockActivation;
use super::WeightRef;

/// Kokoro-specific fused activation and normalization operations.
///
/// See individual variants for documentation. Extracted from [`super::TraceOp`]
/// for file-size management — these variants are semantically cohesive and
/// always handled adjacently in dispatchers.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)] // FusedAdainResBlock carries 8 WeightRefs by design
pub enum KokoroFusedOp {
    /// Per-channel Snake activation: `x + (1/alpha) * sin²(alpha * x)`.
    ///
    /// Alpha is a per-channel weight tensor (typically shape `[1, C, 1]`).
    /// Used by Kokoro TTS ISTFTNet decoder (36 invocations per forward).
    SnakeTensor { alpha: WeightRef },
    /// Fused AdaIN + Snake: InstanceNorm → affine(gamma, beta) → Snake(alpha).
    ///
    /// Combines `(1 + gamma) * InstanceNorm(x) + beta` with per-channel Snake
    /// activation in a single kernel dispatch. Eliminates intermediate buffers
    /// between AdaIN and Snake in Kokoro ResBlocks (36 invocations per forward).
    ///
    /// Tensor inputs: `[x, gamma, beta]` (3 inputs).
    /// - `x`: `[B, C, T]` input tensor.
    /// - `gamma`: `[B, C, 1]` style-conditioned scale (from Linear projection).
    /// - `beta`: `[B, C, 1]` style-conditioned shift (from Linear projection).
    AdainSnake { alpha: WeightRef, eps: f64 },
    /// Fused AdaIN + LeakyRelu: InstanceNorm → affine(gamma, beta) → LeakyRelu(slope).
    ///
    /// Combines `(1 + gamma) * InstanceNorm(x) + beta` with LeakyRelu
    /// activation in a single kernel dispatch. Eliminates intermediate buffers
    /// between AdaIN and LeakyRelu in Kokoro F0EnergyPredictor AdainResBlk1d blocks.
    ///
    /// Tensor inputs: `[x, gamma, beta]` (3 inputs).
    /// - `x`: `[B, C, T]` input tensor.
    /// - `gamma`: `[B, C, 1]` style-conditioned scale (from Linear projection).
    /// - `beta`: `[B, C, 1]` style-conditioned shift (from Linear projection).
    AdainLeakyRelu { eps: f64, slope: f64 },
    /// Fused Adaptive LayerNorm: `(1+gamma) * LayerNorm(x, w, b) + beta`.
    ///
    /// Tensor inputs: `[x, gamma, beta]` — x: `[B, T, C]`, gamma/beta: `[B, 1, C]`.
    /// Kokoro ProsodyPredictor (3 per forward). Part of #2482.
    AdaLayerNorm {
        norm_weight: WeightRef,
        norm_bias: WeightRef,
        eps: f64,
    },
    /// Fused AdaIN residual block: two (AdaIN → activation → Conv1d) pairs + residual.
    ///
    /// Captures an entire Generator ResBlock (Snake variant) or F0 AdainResBlk1d
    /// (LeakyRelu variant) as a single trace op, compiling to 1 GPU dispatch
    /// instead of ~11-14 decomposed dispatches.
    ///
    /// Tensor inputs: `[x, style]` (2 inputs).
    /// - `x`: `[B, C_in, T]` input tensor.
    /// - `style`: `[B, S]` style embedding.
    ///
    /// All weights (style projection, convolution, activation parameters)
    /// are captured as [`WeightRef`] fields. See `designs/2026-03-15-dilated-path-fusion.md`.
    FusedAdainResBlock {
        /// Snake (Generator) or LeakyRelu (F0) activation.
        activation: ResBlockActivation,
        /// AdaIN 1: style projection weight `[2*C_in, S]`.
        adain1_weight: WeightRef,
        /// AdaIN 1: style projection bias `[2*C_in]`.
        adain1_bias: WeightRef,
        /// AdaIN 2: style projection weight `[2*C_out, S]`.
        adain2_weight: WeightRef,
        /// AdaIN 2: style projection bias `[2*C_out]`.
        adain2_bias: WeightRef,
        /// Conv1 weight `[C_out, C_in, K]`.
        conv1_weight: WeightRef,
        /// Conv1 bias `[C_out]`.
        conv1_bias: WeightRef,
        /// Conv1 dilation factor.
        conv1_dilation: usize,
        /// Conv1 padding.
        conv1_padding: usize,
        /// Conv2 weight `[C_out, C_out, K]`.
        conv2_weight: WeightRef,
        /// Conv2 bias `[C_out]`.
        conv2_bias: WeightRef,
        /// Conv2 padding.
        conv2_padding: usize,
        /// InstanceNorm epsilon.
        eps: f64,
        /// Residual scale factor (1.0 for Generator, 1/√2 for F0).
        residual_scale: f64,
    },
}
