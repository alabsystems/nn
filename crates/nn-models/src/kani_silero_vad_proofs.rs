// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for silero_vad_builders constants and block configs.
//!
//! Proves that:
//! 1. LSTM_HIDDEN_SIZE is a power of 2.
//! 2. All 4 encoder blocks have valid channel dimensions.
//! 3. Encoder block chain: output of block i == input of block i+1.
//! 4. First encoder block input matches STFT n_freqs (129).
//! 5. Last encoder block output matches LSTM hidden size (128).
//! 6. All encoder blocks have kernel_size >= stride (no undersampling).
//! 7. Padding is always <= kernel_size / 2.
//!
//! Part of #3793, #3351.

use crate::silero_vad_builders::{ENCODER_BLOCKS, LSTM_HIDDEN_SIZE};

/// Proof 1: LSTM_HIDDEN_SIZE is a power of 2.
///
/// Powers of 2 enable efficient SIMD vectorization.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_lstm_hidden_power_of_two() {
    assert!(LSTM_HIDDEN_SIZE > 0);
    assert!(LSTM_HIDDEN_SIZE.is_power_of_two());
    assert_eq!(LSTM_HIDDEN_SIZE, 128);
}

/// Proof 2: All encoder blocks have non-zero channel dimensions.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_encoder_blocks_nonzero_channels() {
    for block in &ENCODER_BLOCKS {
        assert!(block.in_channels > 0, "in_channels must be > 0");
        assert!(block.out_channels > 0, "out_channels must be > 0");
        assert!(block.kernel_size > 0, "kernel_size must be > 0");
        assert!(block.stride > 0, "stride must be > 0");
    }
}

/// Proof 3: Encoder block chain connectivity.
///
/// Each block's out_channels must match the next block's in_channels.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_encoder_block_chain() {
    assert!(ENCODER_BLOCKS.len() >= 2);
    for i in 0..ENCODER_BLOCKS.len() - 1 {
        assert_eq!(
            ENCODER_BLOCKS[i].out_channels,
            ENCODER_BLOCKS[i + 1].in_channels,
            "block {} out_channels must match block {} in_channels",
            i,
            i + 1
        );
    }
}

/// Proof 4: First encoder block input matches STFT frequency bins (129).
///
/// The STFT produces n_freqs = 256/2 + 1 = 129 frequency bins.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_first_block_matches_stft() {
    assert_eq!(
        ENCODER_BLOCKS[0].in_channels, 129,
        "first block must accept STFT n_freqs=129"
    );
}

/// Proof 5: Last encoder block output matches LSTM hidden size.
///
/// The LSTM takes the final encoder output as input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_last_block_matches_lstm() {
    let last = &ENCODER_BLOCKS[ENCODER_BLOCKS.len() - 1];
    assert_eq!(
        last.out_channels, LSTM_HIDDEN_SIZE,
        "last block out_channels must match LSTM hidden size"
    );
}

/// Proof 6: All encoder blocks have kernel_size >= stride.
///
/// kernel_size < stride would create gaps in the receptive field.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_kernel_ge_stride() {
    for block in &ENCODER_BLOCKS {
        assert!(
            block.kernel_size >= block.stride,
            "kernel_size ({}) must be >= stride ({})",
            block.kernel_size,
            block.stride
        );
    }
}

/// Proof 7: Padding does not exceed half the kernel size.
///
/// Padding > kernel_size/2 would mean the output depends more on
/// padding than on actual input samples.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn proof_padding_bounded() {
    for block in &ENCODER_BLOCKS {
        assert!(
            block.padding <= block.kernel_size / 2,
            "padding ({}) must be <= kernel_size/2 ({})",
            block.padding,
            block.kernel_size / 2
        );
    }
}
