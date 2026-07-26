// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test infrastructure for Silero VAD integration tests.
//!
//! Provides model configuration, builder helpers, and parameter bindings
//! for the full Silero VAD post-STFT pipeline. Used by
//! `compose_silero_vad_full.rs` (and future composition tests).

// Note: dead_code and unreachable_pub are suppressed by the parent aggregator's
// #[allow(dead_code, unreachable_pub)] on the mod declaration.

use ndarray::{ArrayD, IxDyn};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{BoundedTensor, TensorParamBinding};

// ---------------------------------------------------------------------------
// Silero VAD full model configuration
// ---------------------------------------------------------------------------

/// Silero VAD encoder block parameters.
pub(crate) struct VadEncoderBlock {
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
}

/// Hidden size for Silero VAD LSTM.
pub(crate) const LSTM_HIDDEN_SIZE: usize = 128;

/// STFT frequency bins (n_freqs = n_fft/2 + 1 = 129 for n_fft=256).
pub(crate) const STFT_N_FREQS: usize = 129;

/// STFT temporal frames for 576-sample input with hop_length=128, n_fft=256.
pub(crate) const STFT_N_FRAMES: usize = 4;

/// Silero VAD 16kHz encoder blocks (matching silero_vad.rs ENCODER_BLOCKS).
pub(crate) const VAD_BLOCKS: [VadEncoderBlock; 4] = [
    VadEncoderBlock {
        in_channels: 129,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    },
    VadEncoderBlock {
        in_channels: 128,
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
    },
    VadEncoderBlock {
        in_channels: 64,
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
    },
    VadEncoderBlock {
        in_channels: 64,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    },
];

// Delegates to canonical nn_core::conv1d_out_len (dilation=1).
// Note: parameter order (stride, padding) differs from canonical (padding, stride).
fn conv1d_out_len(in_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    nn_core::conv1d_out_len(in_len, kernel_size, padding, stride, 1)
        .expect("conv1d_out_len: invalid parameters")
}

/// Add 4 encoder blocks (Conv1d + ReLU each) to the builder.
///
/// Returns `(encoder_output, enc_weights, enc_biases)` where encoder_output
/// is the final ReLU output node with shape `[128, 1]`.
pub(crate) fn add_encoder_blocks(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
) -> (TensorNodeId, Vec<TensorNodeId>, Vec<TensorNodeId>) {
    let enc_weights: Vec<_> = VAD_BLOCKS
        .iter()
        .enumerate()
        .map(|(i, blk)| {
            b.add_input(
                &format!("enc_weight_{i}"),
                &[blk.out_channels, blk.in_channels, blk.kernel_size],
            )
        })
        .collect();

    let enc_biases: Vec<_> = VAD_BLOCKS
        .iter()
        .enumerate()
        .map(|(i, blk)| b.add_input(&format!("enc_bias_{i}"), &[blk.out_channels]))
        .collect();

    let mut prev = input;
    let mut t = STFT_N_FRAMES;
    for (i, blk) in VAD_BLOCKS.iter().enumerate() {
        t = conv1d_out_len(t, blk.kernel_size, blk.stride, blk.padding);
        let out_shape = [blk.out_channels, t];
        let conv = b.add_conv1d(
            prev,
            enc_weights[i],
            Some(enc_biases[i]),
            blk.stride,
            blk.padding,
            &out_shape,
        );
        prev = b.add_relu(conv, &out_shape);
    }
    (prev, enc_weights, enc_biases)
}

/// Build the full Silero VAD post-STFT model as a single `TensorKernelDef`.
///
/// Variable input: stft_mag only.
/// Constant inputs: hidden_state (zero), cell_state (zero), all weights/biases.
pub(crate) fn build_full_silero_vad() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("silero_vad_full");

    // stft_mag is the only Variable input.
    let stft_mag = b.add_input("stft_mag", &[STFT_N_FREQS, STFT_N_FRAMES]);
    // LSTM states: zero-initialized (SileroVadState::zero()).
    let hidden_state = b.add_input("hidden_state", &[1, LSTM_HIDDEN_SIZE]);
    let cell_state = b.add_input("cell_state", &[1, LSTM_HIDDEN_SIZE]);

    // Encoder: 4 Conv1d + ReLU blocks.
    let (enc_out, _enc_w, _enc_b) = add_encoder_blocks(&mut b, stft_mag);

    // LSTM weights.
    let lstm_wih = b.add_input("lstm_weight_ih", &[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]);
    let lstm_whh = b.add_input("lstm_weight_hh", &[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]);
    let lstm_bias = b.add_input("lstm_bias", &[4 * LSTM_HIDDEN_SIZE]);

    // Output weights.
    let output_weight = b.add_input("output_weight", &[1, LSTM_HIDDEN_SIZE]);
    let output_bias = b.add_input("output_bias", &[1]);

    // Reshape encoder [128,1] → [1,128] for LSTM input.
    let lstm_input = b.add_reshape(enc_out, &[1, LSTM_HIDDEN_SIZE]);

    // LSTM cell.
    let lstm_out = b.add_lstm(
        lstm_input,
        hidden_state,
        cell_state,
        lstm_wih,
        lstm_whh,
        Some(lstm_bias),
        &[1, LSTM_HIDDEN_SIZE],
    );

    // Output: ReLU → Linear(128→1) → Sigmoid.
    let relu = b.add_relu(lstm_out, &[1, LSTM_HIDDEN_SIZE]);
    let linear = b.add_linear(relu, output_weight, Some(output_bias), &[1, 1]);
    let prob = b.add_sigmoid(linear, &[1, 1]);

    b.build(prob).expect("valid graph")
}

/// Build parameter bindings for the full model.
///
/// stft_mag is Variable. hidden_state and cell_state are ConstantTensor(zeros).
/// All weights are ConstantTensor (small uniform values for testing).
pub(crate) fn full_model_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // stft_mag: Variable (the input we verify over)
    bindings.push(TensorParamBinding::Variable);

    // LSTM states: ConstantTensor (zero = SileroVadState::zero())
    let h = ArrayD::from_elem(IxDyn(&[1, LSTM_HIDDEN_SIZE]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(h));
    let c = ArrayD::from_elem(IxDyn(&[1, LSTM_HIDDEN_SIZE]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(c));

    // Encoder weights + biases (4 each)
    for blk in &VAD_BLOCKS {
        let w = ArrayD::from_elem(
            IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
            0.01f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(w));
    }
    for blk in &VAD_BLOCKS {
        let bias = ArrayD::from_elem(IxDyn(&[blk.out_channels]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(bias));
    }

    // LSTM weights and bias
    let w_ih = ArrayD::from_elem(IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]), 0.01f32);
    bindings.push(TensorParamBinding::ConstantTensor(w_ih));
    let w_hh = ArrayD::from_elem(IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]), 0.01f32);
    bindings.push(TensorParamBinding::ConstantTensor(w_hh));
    let bias = ArrayD::from_elem(IxDyn(&[4 * LSTM_HIDDEN_SIZE]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(bias));

    // Output weight and bias
    let out_w = ArrayD::from_elem(IxDyn(&[1, LSTM_HIDDEN_SIZE]), 0.01f32);
    bindings.push(TensorParamBinding::ConstantTensor(out_w));
    let out_b = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(out_b));

    bindings
}

/// Build input bounds for the single Variable input (stft_mag).
///
/// STFT magnitude is non-negative. Typical range for 16kHz audio: [0.0, 10.0].
pub(crate) fn stft_input_bounds() -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 10.0f32);
    BoundedTensor::new(lower, upper).expect("input bounds")
}
