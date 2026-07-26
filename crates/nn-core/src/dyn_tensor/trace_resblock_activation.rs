// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Activation variant for fused AdaIN residual blocks.
//!
//! Extracted from `trace_types.rs` to stay under the 500-line limit.
//! Part of #2484.

use super::WeightRef;

/// Activation variant for fused AdaIN residual blocks.
///
/// Used by [`super::TraceOp::FusedAdainResBlock`] to distinguish Generator-style
/// (Snake) from F0-style (LeakyRelu) residual blocks.
#[derive(Debug, Clone)]
pub enum ResBlockActivation {
    /// Snake activation: `x + (1/alpha) * sin²(alpha * x)`.
    ///
    /// Used in Kokoro Generator ResBlocks (AdaINResBlock1).
    Snake {
        /// Per-channel alpha for the first activation, shape `[1, C, 1]`.
        alpha1: WeightRef,
        /// Per-channel alpha for the second activation, shape `[1, C, 1]`.
        alpha2: WeightRef,
    },
    /// LeakyRelu activation with given negative slope.
    ///
    /// Used in Kokoro F0/energy predictor (AdainResBlk1d).
    LeakyRelu {
        /// Negative slope (typically 0.2).
        slope: f64,
    },
}
