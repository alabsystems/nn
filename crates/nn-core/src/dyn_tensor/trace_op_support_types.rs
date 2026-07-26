// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Support types for `TraceOp`: `TraceActivation` and `TraceUpsampleMode`.

/// Named activation function for the generic `TraceOp::Activation` variant.
///
/// Most activations have dedicated `TraceOp` variants (e.g. `TraceOp::Relu`,
/// `TraceOp::Elu { alpha }`). This enum covers the generic path used by
/// import and tracing when a dedicated variant is not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TraceActivation {
    Relu,
    Gelu,
    GeluErf,
    Silu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Elu,
    LeakyRelu,
    Mish,
}

impl TraceActivation {
    /// Lowercase name for display and compilation dispatch.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::GeluErf => "gelu_erf",
            Self::Silu => "silu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Elu => "elu",
            Self::LeakyRelu => "leaky_relu",
            Self::Mish => "mish",
        }
    }
}

/// Upsampling mode for `TraceOp::Upsample2d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TraceUpsampleMode {
    Nearest,
    Bilinear,
    /// Bicubic interpolation — used by some vision backbones and FPNs.
    Bicubic,
}

impl TraceUpsampleMode {
    /// Lowercase name for display and compilation dispatch.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
        }
    }
}
