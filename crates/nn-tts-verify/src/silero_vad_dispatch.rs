// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD dispatch plan for cost model calibration.
//!
//! Converts the Silero VAD architecture (4 Conv1d+ReLU encoder blocks,
//! LSTM cell, ReLU+Linear+Sigmoid output stage) into a `Vec<DispatchStep>`
//! suitable for roofline profiling and calibration against measured GPU times.
//!
//! Part of #1739 AC4.

use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::ReduceOp;
use nn_dsl::{Conv1dParams, DispatchStep};

use crate::dispatch_builder::DispatchBuilder;

/// STFT frame count for Silero VAD 16kHz (512-sample chunks → 33 STFT frames).
const STFT_FRAMES: usize = 33;

/// LSTM hidden size (from `nn_models::silero_vad_builders::LSTM_HIDDEN_SIZE`).
const LSTM_HIDDEN: usize = 128;

/// Encoder block definition matching `nn_models::silero_vad_builders::ENCODER_BLOCKS`.
struct EncBlock {
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

/// The 4 Silero VAD encoder blocks.
const ENC_BLOCKS: [EncBlock; 4] = [
    EncBlock {
        in_ch: 129,
        out_ch: 128,
        kernel: 3,
        stride: 1,
        padding: 1,
    },
    EncBlock {
        in_ch: 128,
        out_ch: 64,
        kernel: 3,
        stride: 2,
        padding: 1,
    },
    EncBlock {
        in_ch: 64,
        out_ch: 64,
        kernel: 3,
        stride: 2,
        padding: 1,
    },
    EncBlock {
        in_ch: 64,
        out_ch: 128,
        kernel: 3,
        stride: 1,
        padding: 1,
    },
];

/// Compute conv1d output length using the canonical validated implementation.
fn conv1d_out_len(in_len: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    nn_core::conv1d_out_len(in_len, kernel, padding, stride, 1)
        .expect("invariant: ENC_BLOCKS constants are valid conv1d parameters")
}

/// Build encoder dispatch steps: 4 × (Conv1d + ReLU) + temporal pool.
///
/// Returns the final temporal dimension after downsampling.
fn build_encoder_steps(b: &mut DispatchBuilder, stft_frames: usize) -> usize {
    let mut t = stft_frames;
    for (i, blk) in ENC_BLOCKS.iter().enumerate() {
        let t_out = conv1d_out_len(t, blk.kernel, blk.stride, blk.padding);

        // Manual Conv1d: encoder blocks have varying strides so
        // total_elements = out_ch * t_out (not out_ch * t_in).
        let (inp, wt, bias, out) = (
            b.alloc_node(),
            b.alloc_node(),
            b.alloc_node(),
            b.alloc_node(),
        );
        b.push_step(DispatchStep::Conv1d(Conv1dParams::new(
            format!("enc_{i}_conv1d"),
            ScalarType::F32,
            inp,
            wt,
            Some(bias),
            out,
            blk.in_ch,
            blk.out_ch,
            blk.kernel,
            t,
            blk.out_ch * t_out,
            blk.stride,
            blk.padding,
            1,
            1,
        )));

        b.relu(format!("enc_{i}_relu"), blk.out_ch * t_out);
        t = t_out;
    }

    // Temporal pool: [1, 128, T_enc] → [1, 128] via mean reduction.
    b.reduce("enc_temporal_pool", ReduceOp::Mean, t, 128);
    t
}

// LSTM cell extracted to silero_vad_dispatch_lstm.rs via #[path] submodule.
use lstm::build_lstm_cell;

#[path = "silero_vad_dispatch_lstm.rs"]
mod lstm;

/// Build output stage steps: ReLU + Linear(128→1) + Sigmoid.
fn build_output_steps(b: &mut DispatchBuilder) {
    b.relu("output_relu", LSTM_HIDDEN);
    b.linear("output_linear", LSTM_HIDDEN, 1, 1);
    b.sigmoid("output_sigmoid", 1);
}

/// Build the full Silero VAD dispatch plan.
///
/// Architecture:
/// 1. **STFT** (not modeled — input arrives as magnitude spectrogram `[1, 129, T]`)
/// 2. **Encoder blocks** (4 × Conv1d + ReLU)
/// 3. **LSTM cell** (decomposed: 12 primitive steps)
/// 4. **Output stage** (ReLU + Linear(128→1) + Sigmoid)
pub fn build_silero_vad_dispatch_plan(stft_frames: usize) -> Vec<DispatchStep> {
    let mut b = DispatchBuilder::with_capacity(32);

    build_encoder_steps(&mut b, stft_frames);
    build_lstm_cell(&mut b);
    build_output_steps(&mut b);

    b.into_steps()
}

/// Build the standard Silero VAD dispatch plan with default 33 STFT frames.
pub fn build_silero_vad_dispatch_plan_default() -> Vec<DispatchStep> {
    build_silero_vad_dispatch_plan(STFT_FRAMES)
}

/// Number of encoder Conv1d+ReLU dispatch step pairs.
pub const ENCODER_STEP_PAIRS: usize = 4;

/// Number of LSTM decomposed dispatch steps.
///
/// 2 Linear (ih + hh) + 1 BinaryAdd (gates) + 3 Sigmoid (i, f, o) +
/// 1 Tanh (g) + 2 BinaryMul (f*c, i*g) + 1 BinaryAdd (c_new) +
/// 1 Tanh (tanh_c) + 1 BinaryMul (h_new) = 12 steps.
pub const LSTM_DECOMPOSED_STEPS: usize = 12;

/// Number of output stage dispatch steps (ReLU + Linear + Sigmoid).
pub const OUTPUT_STAGE_STEPS: usize = 3;

/// Total expected dispatch steps for the full Silero VAD model.
///
/// 4 encoder pairs (Conv1d+ReLU = 8) + 1 temporal pool (Reduce) +
/// 12 LSTM decomposed + 3 output stage = 24.
pub const TOTAL_EXPECTED_STEPS: usize =
    ENCODER_STEP_PAIRS * 2 + 1 + LSTM_DECOMPOSED_STEPS + OUTPUT_STAGE_STEPS;

#[cfg(test)]
#[path = "silero_vad_dispatch_tests.rs"]
mod tests;
