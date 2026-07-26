// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic builder and constants for the Silero VAD model.
//!
//! Contains encoder block definitions and `TensorKernelDef` construction.
//! Weight validation remains in the backend crate (nn-metal) because it
//! depends on backend-specific weight types.

use nn_dsl::tensor_ir::TensorIRError;
use nn_dsl::TensorBlockBuilder;
use nn_dsl::TensorKernelDef;

/// LSTM hidden size used by Silero VAD 16kHz.
pub const LSTM_HIDDEN_SIZE: usize = 128;

/// Silero VAD encoder block configuration.
pub struct EncoderBlock {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

/// Encoder block configurations for Silero VAD 16kHz.
pub const ENCODER_BLOCKS: [EncoderBlock; 4] = [
    EncoderBlock {
        in_channels: 129,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    },
    EncoderBlock {
        in_channels: 128,
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
    },
    EncoderBlock {
        in_channels: 64,
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
    },
    EncoderBlock {
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    },
];

/// Build the output stage: ReLU + Linear(128→1) + Sigmoid.
pub fn build_output_def() -> Result<TensorKernelDef, TensorIRError> {
    let mut b = TensorBlockBuilder::new("output_stage");
    let input = b.add_input(nn_dsl::input_names::DATA, &[1, LSTM_HIDDEN_SIZE]);
    let weight = b.add_input(nn_dsl::input_names::WEIGHT, &[1, LSTM_HIDDEN_SIZE]);
    let bias = b.add_input(nn_dsl::input_names::BIAS, &[1]);
    let relu_out = b.add_relu(input, &[1, LSTM_HIDDEN_SIZE]);
    let linear_out = b.add_linear(relu_out, weight, Some(bias), &[1, 1]);
    let sig_out = b.add_sigmoid(linear_out, &[1, 1]);
    b.build(sig_out)
}

/// Build encoder block def with explicit input and output temporal dimensions.
pub fn build_encoder_block_def(
    block: &EncoderBlock,
    t_in: usize,
    t_out: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    let mut b = TensorBlockBuilder::new("encoder_block");
    let input = b.add_input(nn_dsl::input_names::DATA, &[1, block.in_channels, t_in]);
    let weight = b.add_input(
        nn_dsl::input_names::WEIGHT,
        &[block.out_channels, block.in_channels, block.kernel_size],
    );
    let bias = b.add_input(nn_dsl::input_names::BIAS, &[block.out_channels]);
    let conv_out = b.add_conv1d(
        input,
        weight,
        Some(bias),
        block.stride,
        block.padding,
        &[1, block.out_channels, t_out],
    );
    let relu_out = b.add_relu(conv_out, &[1, block.out_channels, t_out]);
    b.build(relu_out)
}

#[cfg(test)]
#[path = "silero_vad_builders_tests.rs"]
mod tests;
