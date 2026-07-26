// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::silero_vad_builders`].

use super::*;

// -- Constants ----------------------------------------------------------------

#[test]
fn test_lstm_hidden_size() {
    assert_eq!(LSTM_HIDDEN_SIZE, 128);
}

#[test]
fn test_encoder_block_count() {
    assert_eq!(ENCODER_BLOCKS.len(), 4);
}

#[test]
fn test_encoder_block_channel_progression() {
    // First block: 129→128 (STFT bins to features)
    assert_eq!(ENCODER_BLOCKS[0].in_channels, 129);
    assert_eq!(ENCODER_BLOCKS[0].out_channels, 128);
    // Last block: 64→128 (back to LSTM hidden size)
    assert_eq!(ENCODER_BLOCKS[3].in_channels, 64);
    assert_eq!(ENCODER_BLOCKS[3].out_channels, 128);
}

#[test]
fn test_encoder_blocks_valid_padding() {
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        assert!(
            block.padding <= block.kernel_size / 2,
            "block {i}: padding {} > kernel/2 {}",
            block.padding,
            block.kernel_size / 2
        );
    }
}

// -- build_output_def ---------------------------------------------------------

#[test]
fn test_build_output_def_succeeds() {
    let def = build_output_def().expect("output def should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    // ReLU → Linear(128→1) → Sigmoid → output shape [1, 1]
    assert_eq!(output_shape, &[1, 1]);
}

#[test]
fn test_build_output_def_has_sigmoid() {
    let def = build_output_def().expect("output def should build");
    let has_sigmoid = def
        .nodes
        .iter()
        .any(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Sigmoid { .. }));
    assert!(has_sigmoid, "output def should contain Sigmoid node");
}

#[test]
fn test_build_output_def_has_relu() {
    let def = build_output_def().expect("output def should build");
    let has_relu = def
        .nodes
        .iter()
        .any(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Relu { .. }));
    assert!(has_relu, "output def should contain ReLU node");
}

// -- build_encoder_block_def --------------------------------------------------

#[test]
fn test_build_encoder_block_first() {
    let block = &ENCODER_BLOCKS[0];
    let t_in = 129; // STFT frames
    let t_out = (t_in + 2 * block.padding - block.kernel_size) / block.stride + 1;
    let def = build_encoder_block_def(block, t_in, t_out).expect("encoder block 0 should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[1], block.out_channels);
    assert_eq!(output_shape[2], t_out);
}

#[test]
fn test_build_encoder_block_stride2_halves_time() {
    let block = &ENCODER_BLOCKS[1]; // stride=2
    let t_in = 64;
    let t_out = (t_in + 2 * block.padding - block.kernel_size) / block.stride + 1;
    assert_eq!(t_out, 32, "stride=2 should halve temporal dim");
    let def = build_encoder_block_def(block, t_in, t_out).expect("encoder block should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[2], 32);
}

#[test]
fn test_build_encoder_all_blocks_chain() {
    let mut t = 129; // STFT frame count for 16kHz 512-sample chunks
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let t_out = (t + 2 * block.padding - block.kernel_size) / block.stride + 1;
        let def = build_encoder_block_def(block, t, t_out)
            .unwrap_or_else(|_| panic!("encoder block {i} should build"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(output_shape[1], block.out_channels, "block {i} channels");
        t = t_out;
    }
}
