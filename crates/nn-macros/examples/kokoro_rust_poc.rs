// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro-shaped Rust model prototype for issue #7.
//!
//! This is intentionally lightweight: it demonstrates the Rust-first model flow
//! with `#[nn_macros::model]` and `#[nn_macros::kernel]` while documenting
//! concrete API gaps that still block full Kokoro expression.

#![allow(unexpected_cfgs)]

#[nn_macros::kernel(bounds(alpha = "0.1..1e6"))]
fn snake(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * (alpha * x).sin().powi(2)
}

fn text_encoder_stub(phoneme_energy: f32) -> f32 {
    snake(phoneme_energy, 1.5)
}

fn style_encoder_stub(reference_mel: f32) -> f32 {
    reference_mel * 0.25 + 0.1
}

fn duration_predictor_stub(text_hidden: f32, style: f32) -> f32 {
    (text_hidden + style).max(0.0)
}

fn pitch_predictor_stub(text_hidden: f32, style: f32) -> f32 {
    text_hidden - style
}

fn decoder_stub(text_hidden: f32, durations: f32, pitch: f32, style: f32) -> f32 {
    text_hidden + durations * 0.5 + pitch * 0.25 + style
}

fn istft_vocoder_stub(mel: f32) -> f32 {
    mel.tanh()
}

#[nn_macros::model]
fn kokoro_forward(phoneme_energy: f32, reference_mel: f32) -> f32 {
    let text_hidden = text_encoder_stub(phoneme_energy);
    let style = style_encoder_stub(reference_mel);
    let durations = duration_predictor_stub(text_hidden, style);
    let pitch = pitch_predictor_stub(text_hidden, style);
    let mel = decoder_stub(text_hidden, durations, pitch, style);
    istft_vocoder_stub(mel)
}

const API_GAPS: &[&str] = &[
    "crates/nn-core/src/tensor.rs: Tensor has storage + bounds primitives but no model-layer ops (Embedding/Conv/Transformer) needed for Kokoro architecture expression.",
    "Runtime graph execution/planning handled by nn-metal dispatch infrastructure (execute_tensor_dispatch, PipelineCache).",
];

fn main() {
    let output = kokoro_forward(0.3, 0.7);
    println!("kokoro_forward(0.3, 0.7) = {output}");
    println!(
        "model metadata: name={}, inputs={:?}, output={}",
        __kokoro_forward_model_meta::MODEL_NAME,
        __kokoro_forward_model_meta::INPUT_NAMES,
        __kokoro_forward_model_meta::OUTPUT_TYPE,
    );
    println!(
        "model IR: {} steps, callees={:?}",
        __kokoro_forward_model_meta::STEP_COUNT,
        __kokoro_forward_model_meta::CALLEE_NAMES,
    );
    println!("model IR debug:\n{}", __kokoro_forward_model_meta::IR_DEBUG);
    println!("remaining API gaps:");
    for gap in API_GAPS {
        println!("  - {gap}");
    }
}
