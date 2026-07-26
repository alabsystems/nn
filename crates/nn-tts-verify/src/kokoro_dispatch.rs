// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS vocoder dispatch plan for cost model profiling.
//!
//! Converts the Kokoro-82M Generator (ISTFTNet) architecture into a
//! `Vec<DispatchStep>` suitable for roofline profiling, timing certification,
//! and calibration against measured GPU times.
//!
//! Architecture modeled (from `nn_models::kokoro_decoder::Generator`):
//! 1. **conv_pre**: Conv1d(512, 512, k=7, pad=3)
//! 2. **Per upsample stage** (2 stages): LeakyReLU → ConvTranspose1d →
//!    noise injection (Conv1d + ResBlock) → 3 ResBlocks (averaged)
//! 3. **conv_post**: Conv1d(128, 22, k=7, pad=3) → split → exp + sin
//!
//! ResBlock structure (from `nn_models::kokoro_decoder::ResBlock`):
//! Per dilation: AdaIN1 → Snake → Conv1d(dilated) → AdaIN2 → Snake → Conv1d(d=1) → add
//!
//! Production parameters from `KokoroConfig::default()`:
//! - `upsample_rates = [10, 6]`, `upsample_kernel_sizes = [20, 12]`
//! - `resblock_kernel_sizes = [3, 7, 11]`, `resblock_dilations = [[1,3,5]×3]`
//! - `gen_initial_channels = 512`, `n_fft = 20`
//!
//! Part of #1739 AC3 and #1741 P5.

use nn_dsl::DispatchStep;

use crate::dispatch_builder::DispatchBuilder;

#[path = "kokoro_dispatch_builders.rs"]
mod builders;

use builders::{build_conv_pre, build_output_stage, build_upsample_stage};

// ---------------------------------------------------------------------------
// Architecture constants from KokoroConfig::default()
// ---------------------------------------------------------------------------

/// Generator initial channels (d_en).
const INITIAL_CHANNELS: usize = 512;

/// Upsample rates per stage.
const UPSAMPLE_RATES: [usize; 2] = [10, 6];

/// Upsample ConvTranspose1d kernel sizes per stage.
const UPSAMPLE_KERNELS: [usize; 2] = [20, 12];

/// ResBlock kernel sizes (3 ResBlocks per stage).
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];

/// ResBlock dilations (same for all 3 ResBlocks).
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];

/// FFT size for iSTFT output.
const N_FFT: usize = 20;

/// Output bins = n_fft / 2 + 1.
const N_BINS: usize = N_FFT / 2 + 1; // = 11

/// Style embedding dimension.
const STYLE_DIM: usize = 128;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the full Kokoro-82M vocoder dispatch plan for `seq_len` input tokens.
///
/// Architecture (from `nn_models::kokoro_decoder::Generator`):
/// 1. conv_pre: Conv1d(512→512, k=7, pad=3)
/// 2. Stage 0: LeakyReLU + ConvTranspose1d(512→256, k=20, s=10) + noise +
///    3 ResBlocks (k=3,7,11 × dilations 1,3,5) averaged
/// 3. Stage 1: LeakyReLU + ConvTranspose1d(256→128, k=12, s=6) + noise +
///    3 ResBlocks (k=3,7,11 × dilations 1,3,5) averaged
/// 4. Output: LeakyReLU + conv_post(128→22, k=7) + exp + sin
///
/// Returns `(dispatch_plan, final_temporal_dim)`.
///
/// For 100 tokens: T_final = 100 × 10 × 6 = 6000 audio frames.
pub fn build_kokoro_dispatch_plan(seq_len: usize) -> (Vec<DispatchStep>, usize) {
    let mut b = DispatchBuilder::with_capacity(256);

    let mut t_len = seq_len;
    let mut channels = INITIAL_CHANNELS;

    build_conv_pre(&mut b, t_len);
    for stage in 0..UPSAMPLE_RATES.len() {
        let (new_t, new_ch) = build_upsample_stage(&mut b, stage, channels, t_len);
        t_len = new_t;
        channels = new_ch;
    }
    build_output_stage(&mut b, channels, t_len);

    (b.into_steps(), t_len)
}

/// Build the standard Kokoro dispatch plan with 100 input tokens.
pub fn build_kokoro_dispatch_plan_default() -> (Vec<DispatchStep>, usize) {
    build_kokoro_dispatch_plan(100)
}

// Step count constants (validated by tests)

/// 2 AdaIN(Linear) + 2 Snake(Sigmoid) + 2 Conv1d + 1 BinaryAdd = 7.
pub const STEPS_PER_DILATION: usize = 7;
/// Number of dilation layers per ResBlock (3).
pub const DILATIONS_PER_RESBLOCK: usize = RESBLOCK_DILATIONS.len();
/// Steps per ResBlock: 7 × 3 = 21.
pub const STEPS_PER_RESBLOCK: usize = STEPS_PER_DILATION * DILATIONS_PER_RESBLOCK;
/// Number of ResBlocks per upsample stage (3, one per kernel size).
pub const RESBLOCKS_PER_STAGE: usize = RESBLOCK_KERNELS.len();
/// Noise injection: 1 Conv1d + 1 ResBlock(21) + 1 Add = 23.
pub const NOISE_STEPS_PER_STAGE: usize = 1 + STEPS_PER_RESBLOCK + 1;
/// Per stage: 1 LeakyReLU + 1 ConvTranspose1d + 23 noise + 63 ResBlocks = 88.
pub const STEPS_PER_STAGE: usize =
    1 + 1 + NOISE_STEPS_PER_STAGE + RESBLOCKS_PER_STAGE * STEPS_PER_RESBLOCK;
/// conv_pre: 1 Conv1d.
pub const CONV_PRE_STEPS: usize = 1;
/// Output: 1 LeakyReLU + 1 Conv1d + 1 exp(Tanh) + 1 sin(Tanh) = 4.
pub const OUTPUT_STAGE_STEPS: usize = 4;
/// Total: 1 + 2×88 + 4 = 181.
pub const TOTAL_EXPECTED_STEPS: usize =
    CONV_PRE_STEPS + UPSAMPLE_RATES.len() * STEPS_PER_STAGE + OUTPUT_STAGE_STEPS;
/// Number of upsample stages (2).
pub const NUM_UPSAMPLE_STAGES: usize = UPSAMPLE_RATES.len();
/// Total upsampling factor: 10 × 6 = 60.
pub const TOTAL_UPSAMPLE_FACTOR: usize = 10 * 6;

#[cfg(test)]
#[path = "kokoro_dispatch_tests.rs"]
mod tests;
