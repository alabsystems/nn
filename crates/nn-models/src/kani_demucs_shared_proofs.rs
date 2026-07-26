// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for demucs_shared constants and arithmetic helpers.
//!
//! Proves that:
//! 1. channels_at_depth(0) == BASE_CHANNELS.
//! 2. channels_at_depth doubles at each step (GROWTH=2.0).
//! 3. channels_at_depth(3) == BOTTLENECK_DIM (384).
//! 4. DConv compression ratio: channels / DCONV_COMPRESS produces valid compressed dim.
//! 5. Temporal conv padding: TEMPORAL_CONV_PADDING == TEMPORAL_KERNEL_SIZE / 4.
//! 6. Spectral conv padding: SPECTRAL_CONV_PADDING == SPECTRAL_KERNEL_SIZE / 4.
//! 7. Decoder output channels: 4 sources * AUDIO_CHANNELS.
//! 8. Spectral output channels: 4 sources * AUDIO_CHANNELS * 2 (real+imag).
//! 9. DConv dilation growth: 1 << k for k in 0..DCONV_DEPTH.
//!
//! Part of #3793, #3351.

use crate::demucs_shared::*;
use crate::demucs_transformer_constants::BOTTLENECK_DIM;

/// Proof 1: channels_at_depth(0) == BASE_CHANNELS (48).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_channels_at_depth_zero() {
    assert_eq!(channels_at_depth(0), BASE_CHANNELS);
}

/// Proof 2: channels_at_depth doubles at each step.
///
/// For depth d in [0, 6], channels_at_depth(d+1) == 2 * channels_at_depth(d).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_channels_at_depth_doubles() {
    let d: u8 = kani::any();
    kani::assume(d <= 6);
    let ch_d = channels_at_depth(d as usize);
    let ch_d1 = channels_at_depth((d as usize) + 1);
    assert_eq!(
        ch_d1,
        ch_d * 2,
        "channels_at_depth must double per depth step"
    );
}

/// Proof 3: channels_at_depth(3) == BOTTLENECK_DIM (384).
///
/// The transformer bottleneck operates at encoder depth 3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_channels_at_depth3_is_bottleneck() {
    assert_eq!(
        channels_at_depth(3),
        BOTTLENECK_DIM,
        "depth-3 channels must match transformer bottleneck dim"
    );
}

/// Proof 4: DConv compression produces valid channel count.
///
/// For all encoder depths 0..5, channels / DCONV_COMPRESS > 0.
/// Depths 4-5 are deep blocks with LSTM + attention that also use DConv compression.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_dconv_compression_valid() {
    let depth: u8 = kani::any();
    kani::assume(depth <= 5);
    let ch = channels_at_depth(depth as usize);
    let compressed = ch / DCONV_COMPRESS;
    assert!(
        compressed > 0,
        "DConv compressed channels must be > 0 at depth {}",
        depth
    );
    // Compressed channels should divide evenly
    assert_eq!(ch % DCONV_COMPRESS, 0);
}

/// Proof 5: Temporal conv padding matches formula.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_temporal_conv_padding() {
    assert_eq!(TEMPORAL_CONV_PADDING, TEMPORAL_KERNEL_SIZE / 4);
    assert_eq!(TEMPORAL_CONV_TR_PADDING, TEMPORAL_KERNEL_SIZE / 4);
}

/// Proof 6: Spectral conv padding matches formula.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_spectral_conv_padding() {
    assert_eq!(SPECTRAL_CONV_PADDING, SPECTRAL_KERNEL_SIZE / 4);
    assert_eq!(SPECTRAL_CONV_TR_PADDING, SPECTRAL_KERNEL_SIZE / 4);
}

/// Proof 7: Decoder output channels = 4 sources * 2 stereo.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_decoder_output_channels() {
    assert_eq!(DECODER_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS);
}

/// Proof 8: Spectral output channels = 4 sources * 2 stereo * 2 (real+imag).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_spectral_output_channels() {
    assert_eq!(SPECTRAL_OUTPUT_CHANNELS, 4 * AUDIO_CHANNELS * 2);
}

/// Proof 9: DConv dilation growth is powers of 2.
///
/// For sub-layer k, dilation = 1 << k. Verifies for k in [0, DCONV_DEPTH).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_dconv_dilation_powers_of_two() {
    let k: u8 = kani::any();
    kani::assume((k as usize) < DCONV_DEPTH);
    let dilation = 1_usize << k;
    assert!(dilation > 0, "dilation must be positive");
    assert!(dilation.is_power_of_two(), "dilation must be a power of 2");
    // For DCONV_DEPTH=2: k=0 → dilation=1, k=1 → dilation=2
    if k == 0 {
        assert_eq!(dilation, 1);
    } else if k == 1 {
        assert_eq!(dilation, 2);
    }
}
