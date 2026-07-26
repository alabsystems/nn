// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Builder helpers for the Kokoro TTS → Speaker Encoder composition.
//!
//! **Property 4 (Speaker consistency):** The TTS output, when passed through
//! a simplified ECAPA-TDNN speaker encoder, produces an embedding whose
//! worst-case L2 distance from a reference embedding is bounded.
//!
//! Architecture:
//!
//! ```text
//!   audio [OUT_CHANNELS, TIME_UP] (Variable — from vocoder output)
//!   → SpeakerEncoder: Conv1d + ReLU + Reduce(Mean, axis=1) + Linear
//!   → embedding [EMBED_DIM]
//! ```
//!
//! This is a simplified ECAPA-TDNN: the real model has SE-Res2Net blocks
//! and attentive statistics pooling, but the composition structure is the
//! same: conv feature extraction → temporal pooling → linear projection.
//!
//! Part of #1741: THE MOONSHOT — Property 4 composition proofs.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{ReduceOp, TensorNodeId};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Audio channels from vocoder (production Kokoro: 2 * n_bins).
pub(super) const AUDIO_CHANNELS: usize = 4;

/// Audio time steps from vocoder.
pub(super) const AUDIO_TIME: usize = 4;

/// Internal feature channels in speaker encoder.
const FEAT_CHANNELS: usize = 4;

/// Speaker embedding dimension (production ECAPA-TDNN: 192).
pub(super) const EMBED_DIM: usize = 4;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Speaker encoder stage
// ---------------------------------------------------------------------------

/// Add a simplified ECAPA-TDNN speaker encoder:
/// Conv1d → ReLU → Reduce(Mean, axis=1) → Linear.
///
/// Input: `audio [AUDIO_CHANNELS, AUDIO_TIME]`
/// Output: `embedding [EMBED_DIM]`
fn add_speaker_encoder(b: &mut TensorBlockBuilder, audio: TensorNodeId) -> TensorNodeId {
    // Conv1d: [AUDIO_CHANNELS, AUDIO_TIME] → [FEAT_CHANNELS, AUDIO_TIME]
    let conv_w = b.add_input("spk_conv_w", &[FEAT_CHANNELS, AUDIO_CHANNELS, 3]);
    let conv_out = b.add_conv1d(audio, conv_w, None, 1, 1, &[FEAT_CHANNELS, AUDIO_TIME]);

    // ReLU activation
    let relu_out = b.add_relu(conv_out, &[FEAT_CHANNELS, AUDIO_TIME]);

    // Temporal pooling: Reduce(Mean, axis=1) → [FEAT_CHANNELS]
    // This is a simplified statistics pooling (production uses attentive pooling).
    let pooled = b.add_reduce(
        relu_out,
        ReduceOp::Mean,
        1, // axis=1 (time dimension)
        false,
        &[FEAT_CHANNELS],
    );

    // Linear projection: [FEAT_CHANNELS] → [EMBED_DIM]
    // Reshape [FEAT_CHANNELS] → [1, FEAT_CHANNELS] for matmul
    let pooled_2d = b.add_reshape(pooled, &[1, FEAT_CHANNELS]);
    let proj_w = b.add_input("spk_proj_w", &[EMBED_DIM, FEAT_CHANNELS]);
    let proj_b = b.add_input("spk_proj_b", &[EMBED_DIM]);
    let projected = b.add_matmul(pooled_2d, proj_w, true, None, &[1, EMBED_DIM]);
    let proj_b_bc = b.add_broadcast(proj_b, &[1, EMBED_DIM]);
    let biased = b.add_binary_add(projected, proj_b_bc, &[1, EMBED_DIM]);

    // Reshape back to [EMBED_DIM]
    b.add_reshape(biased, &[EMBED_DIM])
}

// ---------------------------------------------------------------------------
// Pipeline builder: audio → speaker embedding
// ---------------------------------------------------------------------------

/// Build the speaker encoder pipeline.
///
/// Architecture:
///   audio [AUDIO_CHANNELS, AUDIO_TIME] (Variable)
///   → SpeakerEncoder(Conv1d + ReLU + Mean-pool + Linear)
///   → embedding [EMBED_DIM]
///
/// **Property 4 proof:** If the CROWN/IBP bounds on the embedding are tight
/// enough that worst-case L2 distance to the reference embedding is < ε,
/// then speaker consistency is proven for all inputs in the bounded region.
///
/// Returns `(TensorKernelDef, output_shape)`.
pub(super) fn build_speaker_encoder_pipeline() -> (TensorKernelDef, [usize; 1]) {
    let mut b = TensorBlockBuilder::new("speaker_encoder_verify");

    // Variable input: audio from vocoder
    let audio = b.add_input("audio", &[AUDIO_CHANNELS, AUDIO_TIME]);

    // Speaker encoder
    let embedding = add_speaker_encoder(&mut b, audio);

    let out_shape = [EMBED_DIM];
    (
        b.build(embedding)
            .expect("valid speaker encoder pipeline graph"),
        out_shape,
    )
}

/// Build parameter bindings for the speaker encoder pipeline.
///
/// `audio` = Variable. All weights = ConstantTensor with small magnitude.
#[allow(clippy::vec_init_then_push)]
pub(super) fn speaker_encoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // audio: Variable [AUDIO_CHANNELS, AUDIO_TIME]
    bindings.push(TensorParamBinding::Variable);

    push_speaker_encoder_bindings(&mut bindings);

    bindings
}

/// Push speaker encoder weight bindings.
fn push_speaker_encoder_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // spk_conv_w [FEAT_CHANNELS, AUDIO_CHANNELS, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[FEAT_CHANNELS, AUDIO_CHANNELS, 3]),
        WEIGHT_MAG,
    )));

    // spk_proj_w [EMBED_DIM, FEAT_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[EMBED_DIM, FEAT_CHANNELS]),
        WEIGHT_MAG,
    )));

    // spk_proj_b [EMBED_DIM]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[EMBED_DIM]),
        0.0f32,
    )));
}

// ---------------------------------------------------------------------------
// Full TTS+Speaker pipeline: text → audio → embedding
// ---------------------------------------------------------------------------

/// Build the full TTS → Speaker pipeline as a single graph.
///
/// Architecture:
///   text_features [D_MODEL, SEQ_LEN] (Variable)
///   → TextEncoder(Conv1d + ReLU + Linear)
///   → Vocoder(Conv1d → LeakyReLU → ConvTranspose1d → ResBlock → LeakyReLU → Conv1d → Exp)
///   → SpeakerEncoder(Conv1d + ReLU + Mean-pool + Linear)
///   → embedding [EMBED_DIM]
///
/// This graph chains the full Kokoro TTS pipeline with a simplified
/// ECAPA-TDNN speaker encoder, allowing NY to propagate bounds
/// end-to-end from text to speaker embedding.
///
/// Returns `(TensorKernelDef, output_shape)`.
pub(super) fn build_tts_speaker_pipeline() -> (TensorKernelDef, [usize; 1]) {
    let mut b = TensorBlockBuilder::new("kokoro_tts_speaker_verify");

    // --- Stage 1: Text encoder ---
    let text_input = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);
    let encoded = add_text_encoder(&mut b, text_input);

    // --- Stage 2: Vocoder decoder ---
    let audio = add_vocoder_decoder(&mut b, encoded);

    // --- Stage 3: Speaker encoder ---
    let embedding = add_speaker_encoder(&mut b, audio);

    let out_shape = [EMBED_DIM];
    (
        b.build(embedding)
            .expect("valid TTS+speaker pipeline graph"),
        out_shape,
    )
}

/// Build parameter bindings for the full TTS → Speaker pipeline.
#[allow(clippy::vec_init_then_push)]
pub(super) fn tts_speaker_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // text_features: Variable [D_MODEL, SEQ_LEN]
    bindings.push(TensorParamBinding::Variable);

    // Text encoder bindings
    push_text_encoder_bindings(&mut bindings);

    // Vocoder bindings
    push_vocoder_bindings(&mut bindings);

    // Speaker encoder bindings
    push_speaker_encoder_bindings(&mut bindings);

    bindings
}

// ---------------------------------------------------------------------------
// Shared builders (reused from kokoro_full_pipeline.rs constants)
// ---------------------------------------------------------------------------

/// Model dimension (must match kokoro_full_pipeline.rs).
const D_MODEL: usize = 8;
const ENC_DIM: usize = 8;
const VOC_CHANNELS: usize = 4;
const VOC_UP_CHANNELS: usize = 4;
const OUT_CHANNELS: usize = 4;
const SEQ_LEN: usize = 2;
const VOC_UPSAMPLE_STRIDE: usize = 2;
const VOC_UPSAMPLE_KERNEL: usize = 4;
const VOC_UPSAMPLE_PADDING: usize = 1;
const TIME_UP: usize =
    (SEQ_LEN - 1) * VOC_UPSAMPLE_STRIDE + VOC_UPSAMPLE_KERNEL - 2 * VOC_UPSAMPLE_PADDING;

/// Add text encoder (same architecture as kokoro_full_pipeline.rs).
fn add_text_encoder(b: &mut TensorBlockBuilder, text_input: TensorNodeId) -> TensorNodeId {
    let enc_conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let enc_conv_out = b.add_conv1d(text_input, enc_conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);
    let enc_relu = b.add_relu(enc_conv_out, &[D_MODEL, SEQ_LEN]);
    let enc_t = b.add_transpose(enc_relu, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let enc_proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let enc_proj_b = b.add_input("enc_proj_b", &[ENC_DIM]);
    let enc_projected = b.add_matmul(enc_t, enc_proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let enc_proj_b_bc = b.add_broadcast(enc_proj_b, &[SEQ_LEN, ENC_DIM]);
    let enc_biased = b.add_binary_add(enc_projected, enc_proj_b_bc, &[SEQ_LEN, ENC_DIM]);
    b.add_transpose(enc_biased, &[1, 0], &[ENC_DIM, SEQ_LEN])
}

/// Add vocoder decoder (same architecture as kokoro_full_pipeline.rs).
fn add_vocoder_decoder(b: &mut TensorBlockBuilder, encoded: TensorNodeId) -> TensorNodeId {
    use nn_dsl::build_snake_scalar_kernel;

    let up_shape = [VOC_UP_CHANNELS, TIME_UP];
    let eps = b.add_input("voc_eps", &[1]);
    let conv_pre_w = b.add_input("voc_conv_pre_w", &[VOC_CHANNELS, ENC_DIM, 3]);
    let x = b.add_conv1d(encoded, conv_pre_w, None, 1, 1, &[VOC_CHANNELS, SEQ_LEN]);
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CHANNELS, SEQ_LEN]);
    let upsample_w = b.add_input(
        "voc_upsample_w",
        &[VOC_CHANNELS, VOC_UP_CHANNELS, VOC_UPSAMPLE_KERNEL],
    );
    let x_up = b.add_conv_transpose_1d(
        x_act,
        upsample_w,
        None,
        VOC_UPSAMPLE_STRIDE,
        VOC_UPSAMPLE_PADDING,
        1,
        1,
        0, // output_padding
        &up_shape,
    );
    let style_gamma = b.add_input("voc_style_gamma", &[VOC_UP_CHANNELS]);
    let style_beta = b.add_input("voc_style_beta", &[VOC_UP_CHANNELS]);
    let normed = b.add_instance_norm(x_up, eps, 1, Some(style_gamma), Some(style_beta), &up_shape);
    let alpha = b.add_input("voc_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);
    let res_conv_w = b.add_input("voc_res_conv_w", &[VOC_UP_CHANNELS, VOC_UP_CHANNELS, 3]);
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &up_shape);
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);
    let conv_post_w = b.add_input("voc_conv_post_w", &[OUT_CHANNELS, VOC_UP_CHANNELS, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &[OUT_CHANNELS, TIME_UP]);
    b.add_exp(x_post, &[OUT_CHANNELS, TIME_UP])
}

/// Push text encoder weight bindings (same as kokoro_full_pipeline.rs).
fn push_text_encoder_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ENC_DIM, D_MODEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ENC_DIM]),
        0.0f32,
    )));
}

/// Push vocoder decoder weight bindings (same as kokoro_full_pipeline.rs).
fn push_vocoder_bindings(bindings: &mut Vec<TensorParamBinding>) {
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_CHANNELS, ENC_DIM, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_CHANNELS, VOC_UP_CHANNELS, VOC_UPSAMPLE_KERNEL]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_UP_CHANNELS]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_UP_CHANNELS]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1.0));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_UP_CHANNELS, VOC_UP_CHANNELS, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[OUT_CHANNELS, VOC_UP_CHANNELS, 3]),
        WEIGHT_MAG,
    )));
}
