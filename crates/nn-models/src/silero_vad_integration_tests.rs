// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD integration tests covering VAD frame size, output shape,
//! LSTM state dimensions, batch processing shapes, encoder chain shapes,
//! and STFT parameter compatibility (#4186).

use crate::silero_vad_builders::{
    build_encoder_block_def, build_output_def, ENCODER_BLOCKS, LSTM_HIDDEN_SIZE,
};
use crate::stft::StftParams;

// ===========================================================================
// 1. VAD frame size (512 samples @ 16kHz = 32ms)
// ===========================================================================

#[test]
fn test_vad_frame_size_512_samples() {
    // Silero VAD operates on 512-sample chunks at 16kHz = 32ms per frame.
    let sample_rate = 16000usize;
    let frame_samples = 512usize;
    let frame_ms = (frame_samples as f64 / sample_rate as f64) * 1000.0;
    assert!(
        (frame_ms - 32.0).abs() < 0.01,
        "VAD frame should be 32ms at 16kHz, got {frame_ms:.2}ms"
    );
}

#[test]
fn test_vad_stft_params_default() {
    // Silero VAD uses n_fft=256, hop_length=128
    let params = StftParams::default();
    assert_eq!(params.n_fft, 256);
    assert_eq!(params.hop_length, 128);
    assert_eq!(params.n_freqs, 129);
}

#[test]
fn test_vad_stft_frames_from_576_samples() {
    // Silero VAD processes 576 samples (512 new + 64 context)
    // Padded: 576 + 64 = 640. Frames: (640 - 256) / 128 + 1 = 4
    let params = StftParams::default();
    let audio_len = 576usize;
    let padded_len = audio_len + params.pad_right;
    let n_frames = (padded_len - params.n_fft) / params.hop_length + 1;
    assert_eq!(n_frames, 4, "576-sample chunk should produce 4 STFT frames");
}

// ===========================================================================
// 2. Output shape
// ===========================================================================

#[test]
fn test_output_def_shape_is_1x1() {
    // Output: ReLU -> Linear(128->1) -> Sigmoid -> [1, 1]
    let def = build_output_def().expect("output def should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(
        output_shape,
        &[1, 1],
        "VAD output is scalar probability [1, 1]"
    );
}

#[test]
fn test_output_def_input_dim_matches_lstm_hidden() {
    // Output stage input is [1, LSTM_HIDDEN_SIZE=128]
    let def = build_output_def().expect("output def should build");
    // First node should be an input with shape [1, 128]
    let first_input = &def.nodes[0];
    assert_eq!(
        first_input.shape,
        vec![1, LSTM_HIDDEN_SIZE],
        "output stage input should be [1, 128]"
    );
}

// ===========================================================================
// 3. LSTM state dimensions
// ===========================================================================

#[test]
fn test_lstm_hidden_size_128() {
    assert_eq!(LSTM_HIDDEN_SIZE, 128, "Silero VAD LSTM hidden size");
}

#[test]
fn test_lstm_state_matches_encoder_output() {
    // The last encoder block outputs 128 channels, which feeds directly
    // into the LSTM with hidden_size=128.
    let last_block = &ENCODER_BLOCKS[3];
    assert_eq!(
        last_block.out_channels, LSTM_HIDDEN_SIZE,
        "last encoder block output must match LSTM hidden size"
    );
}

// ===========================================================================
// 4. Encoder block configurations
// ===========================================================================

#[test]
fn test_encoder_block_count_is_4() {
    assert_eq!(ENCODER_BLOCKS.len(), 4);
}

#[test]
fn test_encoder_first_block_stft_to_features() {
    // First block: 129 STFT bins -> 128 features
    let block = &ENCODER_BLOCKS[0];
    assert_eq!(block.in_channels, 129, "input is n_freqs=129 STFT bins");
    assert_eq!(block.out_channels, 128);
    assert_eq!(block.stride, 1, "first block preserves temporal dim");
}

#[test]
fn test_encoder_stride2_blocks_halve_time() {
    // Blocks 1 and 2 have stride=2, halving temporal dimension
    assert_eq!(ENCODER_BLOCKS[1].stride, 2, "block 1 stride");
    assert_eq!(ENCODER_BLOCKS[2].stride, 2, "block 2 stride");
    // Blocks 0 and 3 have stride=1
    assert_eq!(ENCODER_BLOCKS[0].stride, 1, "block 0 stride");
    assert_eq!(ENCODER_BLOCKS[3].stride, 1, "block 3 stride");
}

#[test]
fn test_encoder_channel_bottleneck_pattern() {
    // 129 -> 128 -> 64 -> 64 -> 128
    // Matches Silero VAD's hourglass: compress then expand back to LSTM size
    let expected_channels: [(usize, usize); 4] = [(129, 128), (128, 64), (64, 64), (64, 128)];
    for (i, (in_ch, out_ch)) in expected_channels.iter().enumerate() {
        assert_eq!(
            ENCODER_BLOCKS[i].in_channels, *in_ch,
            "block {i} in_channels"
        );
        assert_eq!(
            ENCODER_BLOCKS[i].out_channels, *out_ch,
            "block {i} out_channels"
        );
    }
}

#[test]
fn test_encoder_chain_output_connectivity() {
    // Each block's out_channels must match next block's in_channels
    for i in 0..ENCODER_BLOCKS.len() - 1 {
        assert_eq!(
            ENCODER_BLOCKS[i].out_channels,
            ENCODER_BLOCKS[i + 1].in_channels,
            "block {i} output channels must match block {} input channels",
            i + 1
        );
    }
}

// ===========================================================================
// 5. Temporal dimension through encoder chain
// ===========================================================================

#[test]
fn test_encoder_chain_temporal_dimensions() {
    // Starting from STFT output: n_freqs=129, n_frames=4 (for 576 samples).
    // Feed n_frames=4 through all 4 encoder blocks.
    let _params = StftParams::default(); // validates default is available
    let n_frames = 4usize; // (640 - 256) / 128 + 1

    let mut t = n_frames;
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let t_out = (t + 2 * block.padding - block.kernel_size) / block.stride + 1;
        assert!(
            t_out > 0,
            "encoder block {i}: temporal dim went to zero (t_in={t})"
        );
        t = t_out;
    }
    // After 4 blocks (stride 1, 2, 2, 1): 4 -> 4 -> 2 -> 1 -> 1
    assert_eq!(t, 1, "encoder chain should reduce 4 frames to 1");
}

#[test]
fn test_encoder_chain_temporal_with_longer_input() {
    // With 129 STFT frames (longer audio chunk)
    let mut t = 129usize;
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let t_out = (t + 2 * block.padding - block.kernel_size) / block.stride + 1;
        assert!(
            t_out > 0,
            "encoder block {i}: temporal dim went to zero (t_in={t})"
        );
        t = t_out;
    }
    // 129 -> 129 -> 65 -> 33 -> 33
    assert_eq!(
        t, 33,
        "129 STFT frames should become 33 after encoder chain"
    );
}

// ===========================================================================
// 6. STFT parameter compatibility
// ===========================================================================

#[test]
fn test_stft_n_freqs_matches_first_encoder_input() {
    let params = StftParams::default();
    assert_eq!(
        params.n_freqs, ENCODER_BLOCKS[0].in_channels,
        "STFT n_freqs must match first encoder block input channels"
    );
}

#[test]
fn test_stft_pad_right_is_quarter_nfft() {
    let params = StftParams::default();
    assert_eq!(
        params.pad_right,
        params.n_fft / 4,
        "pad_right should be n_fft/4"
    );
}

// ===========================================================================
// 7. All encoder blocks build successfully
// ===========================================================================

#[test]
fn test_all_encoder_blocks_build_with_stft_input() {
    // Build the full chain from STFT output dimensions
    let mut t = 4usize; // 4 STFT frames from 576-sample chunk

    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let t_out = (t + 2 * block.padding - block.kernel_size) / block.stride + 1;
        let def = build_encoder_block_def(block, t, t_out)
            .unwrap_or_else(|e| panic!("encoder block {i} should build: {e}"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(
            output_shape[1], block.out_channels,
            "block {i} out channels"
        );
        assert_eq!(output_shape[2], t_out, "block {i} temporal dim");
        t = t_out;
    }
}
