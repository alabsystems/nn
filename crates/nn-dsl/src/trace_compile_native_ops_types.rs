// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper types for [`NativeOpKind`](super::NativeOpKind): struct parameters,
//! activation enums, and projection descriptors.
//!
//! Extracted from `trace_compile_native_ops.rs` for 450-line compliance.

/// Absorbed style projection parameters for [`FusedResBlock`](super::NativeOpKind::FusedResBlock).
///
/// When present, the FusedResBlock executor takes `[x, style_embed]` as
/// `input_steps` (2 entries instead of 5) and runs two linear projections
/// to produce gamma/beta pairs. Weights are in `weight_data` as
/// `style1_weight`, `style1_bias`, `style2_weight`, `style2_bias`.
/// Part of #2780.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct StyleProjectionParams {
    /// Number of channels for phase 1 (proj1 maps style → \[B, 2*channels1\]).
    pub channels1: usize,
    /// Number of channels for phase 2 (proj2 maps style → \[B, 2*channels2\]).
    pub channels2: usize,
    /// Style embedding dimension (input size of both projection Linears).
    pub style_dim: usize,
}

impl StyleProjectionParams {
    #[must_use]
    pub fn new(channels1: usize, channels2: usize, style_dim: usize) -> Self {
        Self {
            channels1,
            channels2,
            style_dim,
        }
    }
}

/// Per-block offset into a batched style projection output tensor.
///
/// Each FusedResBlock uses `narrow(1, offset, 2*(channels1+channels2))` to
/// extract its gamma/beta pairs from the concatenated projection output.
/// Layout: `[gamma1(C1), beta1(C1), gamma2(C2), beta2(C2)]`.
/// Part of #1815 Tier 1.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct StyleBatchOffset {
    /// Offset into dim 1 of the batched output where this block's projection starts.
    pub offset: usize,
    /// Phase 1 channels.
    pub channels1: usize,
    /// Phase 2 channels.
    pub channels2: usize,
}

impl StyleBatchOffset {
    #[must_use]
    pub fn new(offset: usize, channels1: usize, channels2: usize) -> Self {
        Self {
            offset,
            channels1,
            channels2,
        }
    }
}

/// Shared parameters for a single NormActivConv1d phase within a
/// [`FusedResBlock`](super::NativeOpKind::FusedResBlock).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct NormActivConv1dParams {
    /// Which activation to apply after InstanceNorm + style affine.
    pub activation: NormActivation,
    /// Epsilon for InstanceNorm numerical stability.
    pub eps: f32,
    /// Conv1d dilation factor.
    pub conv_dilation: usize,
    /// Conv1d padding (symmetric).
    pub conv_padding: usize,
    /// Input x shape `[B, C_in, T]`.
    pub input_shape: Vec<usize>,
    /// Number of output channels (Conv1d weight shape\[0\]).
    pub output_channels: usize,
    /// Convolution kernel size (Conv1d weight shape\[2\]).
    pub kernel_size: usize,
}

impl NormActivConv1dParams {
    #[must_use]
    pub fn new(
        activation: NormActivation,
        eps: f32,
        conv_dilation: usize,
        conv_padding: usize,
        input_shape: Vec<usize>,
        output_channels: usize,
        kernel_size: usize,
    ) -> Self {
        Self {
            activation,
            eps,
            conv_dilation,
            conv_padding,
            input_shape,
            output_channels,
            kernel_size,
        }
    }
}

/// Activation applied inside the GEMM write-back for
/// [`LinearActivation`](super::NativeOpKind::LinearActivation) fusion.
///
/// Covers all Linear+Activation patterns in Kokoro (PlBert Gelu), Whisper
/// (fc1→GeluErf), Qwen3 (gate_proj→Silu), and GLM5. Part of #2256.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum GemmActivation {
    /// `max(x, 0)`
    Relu,
    /// Tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
    Gelu,
    /// Erf polynomial (A&S 7.1.26): `0.5 * x * (1 + erf(x / sqrt(2)))`
    GeluErf,
    /// `1 / (1 + exp(-x))`
    Sigmoid,
    /// `x * sigmoid(x)`
    Silu,
    /// `tanh(x)`
    Tanh,
}

/// Activation variant for the fused [`NormActivConv1d`](super::NativeOpKind::NormActivConv1d)
/// NativeOp.
///
/// Determines whether LeakyRelu or Snake activation is applied between
/// the InstanceNorm+affine and Conv1d stages.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum NormActivation {
    /// LeakyRelu with given negative slope. Used in F0/energy predictor blocks.
    LeakyRelu {
        /// Negative slope (typically 0.2).
        slope: f32,
    },
    /// Per-channel Snake activation. Used in Generator ResBlocks.
    /// The `alpha` weight `[C_in]` is in `weight_data["alpha"]`.
    Snake,
}

/// Activation variant for the fused [`FusedConv1dActivation`](super::NativeOpKind::FusedConv1dActivation)
/// NativeOp.
///
/// Covers Conv1d + activation patterns in Kokoro (Snake, LeakyReLU) and
/// general-purpose models (ReLU, SiLU, GELU, Tanh). Part of #4264.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ConvActivation {
    /// Per-channel Snake activation: `x + (1/alpha) * sin(alpha * x)^2`.
    /// The `alpha` weight `[C_out]` is in `weight_data["alpha"]`.
    Snake,
    /// `max(x, 0)`
    Relu,
    /// `max(slope * x, x)` with given negative slope (typically 0.01 or 0.2).
    LeakyRelu {
        /// Negative slope.
        slope: f32,
    },
    /// `x * sigmoid(x)` (SiLU / Swish).
    Silu,
    /// Tanh approximation GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`.
    Gelu,
    /// Erf polynomial GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    GeluErf,
    /// `tanh(x)`.
    Tanh,
}

/// Memory layout for FlashAttention Q/K/V/O tensors. Part of #1815 Tier 5 D1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum AttentionLayout {
    /// Standard layout `[B, H, S, D]` — requires pre-transposed Q/K/V.
    #[default]
    HeadsFirst,
    /// SeqFirst layout `[B, S, H, D]` — avoids Transpose dispatches.
    SeqFirst,
}

/// Parameters for a single block within a [`FusedResBlockChain`](super::NativeOpKind::FusedResBlockChain).
///
/// Each entry describes one complete ResBlock: 2x (InstanceNorm + affine +
/// activation + Conv1d) + residual add. The chain executor processes these
/// sequentially, passing the output of one block as the input to the next.
/// Part of #4264.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ResBlockChainEntry {
    /// First NormActivConv1d phase (norm + activation + dilated conv).
    pub phase1: NormActivConv1dParams,
    /// Second NormActivConv1d phase (norm + activation + stride-1 conv).
    pub phase2: NormActivConv1dParams,
    /// Post-add residual scale factor. 1.0 for Generator (no-op),
    /// `1/sqrt(2)` for F0EnergyPredictor.
    pub residual_scale: f32,
}

impl ResBlockChainEntry {
    #[must_use]
    pub fn new(
        phase1: NormActivConv1dParams,
        phase2: NormActivConv1dParams,
        residual_scale: f32,
    ) -> Self {
        Self {
            phase1,
            phase2,
            residual_scale,
        }
    }
}

/// Which normalization variant to use in a fused NormLinear kernel.
///
/// Part of #3089. Discriminates between LayerNorm (mean+var reduction,
/// weight+bias affine) and RmsNorm (x² mean reduction, weight-only scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum FusedNormKind {
    /// Full LayerNorm: `(x - mean) / sqrt(var + eps) * weight + bias`.
    /// Requires `norm_weight` and `norm_bias` in weight_data.
    LayerNorm,
    /// RmsNorm: `x * rsqrt(mean(x²) + eps) * weight`.
    /// Requires `norm_weight` only. No norm_bias.
    RmsNorm,
}
