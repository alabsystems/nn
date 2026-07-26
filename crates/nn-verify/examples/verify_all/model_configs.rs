// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model-level verification configurations for `verify_all`.
//!
//! Produces composed NY graph networks for full model verification
//! (#839 AC4). The first model is Silero VAD (post-STFT pipeline).

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Silero VAD configuration (matching silero_vad_test_helpers.rs)
// ---------------------------------------------------------------------------

/// LSTM hidden size for Silero VAD.
const LSTM_HIDDEN_SIZE: usize = 128;

/// STFT frequency bins (n_fft/2 + 1 = 129 for n_fft=256, 16kHz).
const STFT_N_FREQS: usize = 129;

/// STFT temporal frames for 576-sample input with hop_length=128, n_fft=256.
const STFT_N_FRAMES: usize = 4;

struct EncoderBlock {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

const ENCODER_BLOCKS: [EncoderBlock; 4] = [
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

// Delegates to canonical nn_core::conv1d_out_len (dilation=1).
fn conv1d_out_len(in_len: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    nn_core::conv1d_out_len(in_len, kernel_size, padding, stride, 1)
        .expect("conv1d_out_len: invalid parameters")
}

fn add_encoder_blocks(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
) -> (TensorNodeId, Vec<TensorNodeId>, Vec<TensorNodeId>) {
    let enc_weights: Vec<_> = ENCODER_BLOCKS
        .iter()
        .enumerate()
        .map(|(i, blk)| {
            b.add_input(
                &format!("enc_weight_{i}"),
                &[blk.out_channels, blk.in_channels, blk.kernel_size],
            )
        })
        .collect();

    let enc_biases: Vec<_> = ENCODER_BLOCKS
        .iter()
        .enumerate()
        .map(|(i, blk)| b.add_input(&format!("enc_bias_{i}"), &[blk.out_channels]))
        .collect();

    let mut prev = input;
    let mut t = STFT_N_FRAMES;
    for (i, blk) in ENCODER_BLOCKS.iter().enumerate() {
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

/// Build the full Silero VAD post-STFT model as a `TensorKernelDef`.
fn build_silero_vad_model() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("silero_vad_full");

    let stft_mag = b.add_input("stft_mag", &[STFT_N_FREQS, STFT_N_FRAMES]);
    let hidden_state = b.add_input("hidden_state", &[1, LSTM_HIDDEN_SIZE]);
    let cell_state = b.add_input("cell_state", &[1, LSTM_HIDDEN_SIZE]);

    let (enc_out, _enc_w, _enc_b) = add_encoder_blocks(&mut b, stft_mag);

    let lstm_wih = b.add_input("lstm_weight_ih", &[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]);
    let lstm_whh = b.add_input("lstm_weight_hh", &[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]);
    let lstm_bias = b.add_input("lstm_bias", &[4 * LSTM_HIDDEN_SIZE]);
    let output_weight = b.add_input("output_weight", &[1, LSTM_HIDDEN_SIZE]);
    let output_bias = b.add_input("output_bias", &[1]);

    let lstm_input = b.add_reshape(enc_out, &[1, LSTM_HIDDEN_SIZE]);
    let lstm_out = b.add_lstm(
        lstm_input,
        hidden_state,
        cell_state,
        lstm_wih,
        lstm_whh,
        Some(lstm_bias),
        &[1, LSTM_HIDDEN_SIZE],
    );
    let relu = b.add_relu(lstm_out, &[1, LSTM_HIDDEN_SIZE]);
    let linear = b.add_linear(relu, output_weight, Some(output_bias), &[1, 1]);
    let prob = b.add_sigmoid(linear, &[1, 1]);

    b.build(prob).expect("valid silero_vad_full graph")
}

/// Build parameter bindings for the Silero VAD model.
fn silero_vad_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // stft_mag: Variable (the input we verify over)
    bindings.push(TensorParamBinding::Variable);

    // LSTM states: zero-initialized
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN_SIZE]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN_SIZE]),
        0.0f32,
    )));

    // Encoder weights + biases
    for blk in &ENCODER_BLOCKS {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[blk.out_channels, blk.in_channels, blk.kernel_size]),
            0.01f32,
        )));
    }
    for blk in &ENCODER_BLOCKS {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[blk.out_channels]),
            0.0f32,
        )));
    }

    // LSTM weights and bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[4 * LSTM_HIDDEN_SIZE]),
        0.0f32,
    )));

    // Output weight and bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, LSTM_HIDDEN_SIZE]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));

    bindings
}

/// STFT magnitude input bounds: non-negative, typical range [0, 10].
fn stft_input_bounds() -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[STFT_N_FREQS, STFT_N_FRAMES]), 10.0f32);
    BoundedTensor::new(lower, upper).expect("STFT input bounds")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A model-level verification configuration.
pub(crate) struct ModelConfig {
    pub name: &'static str,
    pub def: TensorKernelDef,
    pub bindings: Vec<TensorParamBinding>,
    pub input_bounds: BoundedTensor,
    /// Uniform scalar bounds for the variable input.
    pub input_lower: f32,
    pub input_upper: f32,
}

#[path = "model_configs_extra.rs"]
mod extra;

/// Build all model-level verification configurations.
///
/// All 5 dvoice models: Silero VAD (#770/#787), HTDemucs, Whisper,
/// Qwen3, and Kokoro decoder (#1696 AC7).
pub(crate) fn build_model_configs() -> Vec<ModelConfig> {
    let mut configs = vec![ModelConfig {
        name: "silero_vad_full",
        def: build_silero_vad_model(),
        bindings: silero_vad_bindings(),
        input_bounds: stft_input_bounds(),
        input_lower: 0.0,
        input_upper: 10.0,
    }];
    configs.extend(extra::extra_model_configs());
    configs
}
