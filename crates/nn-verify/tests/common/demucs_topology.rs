// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared Demucs/Kokoro decoder topology configuration.
//!
//! Consolidates `channels_at_stage()` and `time_after_stages()` functions
//! that were duplicated across 6 helper files.
//!
//! Part of #1938.

/// Decoder topology configuration.
///
/// Parameterizes the channel-halving and temporal-upsampling calculations
/// common to Demucs decoder compose tests.
#[allow(dead_code)]
pub(crate) struct DemucsTopology {
    /// Initial channel count at stage 0 (default: 8 for test scale).
    pub(crate) init_channels: usize,
    /// Starting temporal length before upsample stages.
    pub(crate) t_dec: usize,
    /// ConvTranspose1d stride per upsample stage.
    pub(crate) stride: usize,
    /// ConvTranspose1d kernel size per upsample stage.
    pub(crate) kernel: usize,
    /// ConvTranspose1d padding per upsample stage.
    pub(crate) padding: usize,
}

impl DemucsTopology {
    /// Standard test topology matching Kokoro decoder test scale.
    ///
    /// INIT_CHANNELS=8, T_DEC=4, stride=2, kernel=4, padding=1.
    #[allow(dead_code)]
    pub(crate) fn default_test() -> Self {
        Self {
            init_channels: 8,
            t_dec: 4,
            stride: 2,
            kernel: 4,
            padding: 1,
        }
    }

    /// Channels at a given decoder stage (halving per stage).
    ///
    /// Stage 0: `init_channels`, Stage 1: `init_channels/2`, etc.
    ///
    /// Replaces: `channels_at_stage` in attention_decoder_multi_stage,
    /// attention_decoder_multi_kernel, attention_decoder_output,
    /// attention_decoder_dilated, attention_decoder_noise (5 free-fn copies)
    /// and attention_decoder_scaled (1 method copy).
    #[allow(dead_code)]
    pub(crate) fn channels_at_stage(&self, stage: usize) -> usize {
        self.init_channels >> stage
    }

    /// Temporal length after `num_stages` ConvTranspose1d upsample stages.
    ///
    /// Each stage: `out = (in - 1) * stride + kernel - 2 * padding`.
    ///
    /// Replaces: `time_after_stages` in attention_decoder_multi_stage,
    /// attention_decoder_multi_kernel, attention_decoder_output,
    /// attention_decoder_dilated, attention_decoder_noise (5 free-fn copies)
    /// and attention_decoder_scaled (1 method copy).
    #[allow(dead_code)]
    pub(crate) fn time_after_stages(&self, num_stages: usize) -> usize {
        let mut t = self.t_dec;
        for _ in 0..num_stages {
            t = (t - 1) * self.stride + self.kernel - 2 * self.padding;
        }
        t
    }
}
