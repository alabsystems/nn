// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Builder helpers for the full Kokoro pipeline composition:
//! text encoder → prosody predictor → vocoder decoder.
//!
//! This builds a single `TensorKernelDef` graph that proves end-to-end
//! properties of the Kokoro TTS pipeline:
//!
//! **Property 1 (Non-silence):** The vocoder's exp() output is always positive
//! (exp(x) > 0 for all finite x), so NY lower bounds > 0 prove
//! the output is never silent.
//!
//! **Property 2 (Non-clipping):** NY upper bounds on the vocoder
//! output prove all samples stay within a bounded range.
//!
//! Architecture (simplified for NY tractability):
//!
//! ```text
//!   text_features [D_MODEL, SEQ_LEN] (Variable)
//!   → TextEncoder: Conv1d + ReLU + Linear → encoded [ENC_DIM, SEQ_LEN]
//!   → ProsodyPath: Linear → duration_logits [SEQ_LEN] (branch output)
//!   → VocoderPath: Conv1d(conv_pre) → LeakyReLU → ConvTranspose1d(upsample)
//!     → ResBlock(InstanceNorm + Snake + Conv1d) + residual
//!     → LeakyReLU → Conv1d(conv_post) → Exp → audio [OUT_CH, TIME_UP]
//! ```
//!
//! **Single-variable approach:** text_features is the sole Variable input.
//! All weights/biases are ConstantTensor. Style parameters for InstanceNorm
//! are constant gamma/beta.
//!
//! Part of #1741: THE MOONSHOT — end-to-end Kokoro pipeline verification.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::{GraphNetwork, NormBoundsMode, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Model dimension (production Kokoro: 512).
pub(super) const D_MODEL: usize = 8;

/// Encoder output dimension fed to vocoder (production: 512).
const ENC_DIM: usize = 8;

/// Vocoder internal channels (production: 512).
const VOC_CHANNELS: usize = 4;

/// Vocoder upsampled channels (production: 256).
const VOC_UP_CHANNELS: usize = 4;

/// Vocoder output channels (production: 2 * n_bins = 22).
///
/// This pipeline ends at `exp()` — the spectral magnitude output (pre-iSTFT).
/// Full audio-domain verification through iSTFT is in `compose_kokoro_istft.rs`
/// (Part of #2916), which extends CROWN from spectral bounds to audio [-1,1].
pub(super) const OUT_CHANNELS: usize = 4;

/// Sequence length (T, number of phoneme tokens).
pub(super) const SEQ_LEN: usize = 2;

/// ConvTranspose1d stride for vocoder upsampling.
const VOC_UPSAMPLE_STRIDE: usize = 2;

/// ConvTranspose1d kernel for vocoder upsampling.
const VOC_UPSAMPLE_KERNEL: usize = 4;

/// Padding for ConvTranspose1d: (kernel - stride) / 2.
const VOC_UPSAMPLE_PADDING: usize = 1;

/// Output time after vocoder upsampling.
/// conv_transpose1d: (in-1)*stride + kernel - 2*padding
pub(super) const TIME_UP: usize =
    (SEQ_LEN - 1) * VOC_UPSAMPLE_STRIDE + VOC_UPSAMPLE_KERNEL - 2 * VOC_UPSAMPLE_PADDING;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Text encoder stage
// ---------------------------------------------------------------------------

/// Add a simplified text encoder: Conv1d + ReLU + Linear projection.
///
/// Input: `text_features [D_MODEL, SEQ_LEN]`
/// Output: `encoded [ENC_DIM, SEQ_LEN]`
fn add_text_encoder(b: &mut TensorBlockBuilder, text_input: TensorNodeId) -> TensorNodeId {
    // Conv1d: [D_MODEL, SEQ_LEN] → [D_MODEL, SEQ_LEN] (same-padding)
    let enc_conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let enc_conv_out = b.add_conv1d(text_input, enc_conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);

    // ReLU activation
    let enc_relu = b.add_relu(enc_conv_out, &[D_MODEL, SEQ_LEN]);

    // Linear projection: [D_MODEL, SEQ_LEN] → [ENC_DIM, SEQ_LEN]
    // Transpose → MatMul → Transpose back
    let enc_t = b.add_transpose(enc_relu, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let enc_proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let enc_proj_b = b.add_input("enc_proj_b", &[ENC_DIM]);
    let enc_projected = b.add_matmul(enc_t, enc_proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let enc_proj_b_bc = b.add_broadcast(enc_proj_b, &[SEQ_LEN, ENC_DIM]);
    let enc_biased = b.add_binary_add(enc_projected, enc_proj_b_bc, &[SEQ_LEN, ENC_DIM]);
    b.add_transpose(enc_biased, &[1, 0], &[ENC_DIM, SEQ_LEN])
}

// ---------------------------------------------------------------------------
// Vocoder (decoder) stage
// ---------------------------------------------------------------------------

/// Add the vocoder decoder: Conv1d → LeakyReLU → ConvTranspose1d → ResBlock → LeakyReLU → Conv1d → Exp.
///
/// Input: `encoded [ENC_DIM, SEQ_LEN]`
/// Output: `audio [OUT_CHANNELS, TIME_UP]`
///
/// This matches the architecture in `kokoro_decoder.rs` but takes the encoder
/// output directly, closing the pipeline loop.
fn add_vocoder_decoder(b: &mut TensorBlockBuilder, encoded: TensorNodeId) -> TensorNodeId {
    let up_shape = [VOC_UP_CHANNELS, TIME_UP];

    // Shared epsilon for InstanceNorm
    let eps = b.add_input("voc_eps", &[1]);

    // Conv pre: [ENC_DIM, SEQ_LEN] → [VOC_CHANNELS, SEQ_LEN]
    let conv_pre_w = b.add_input("voc_conv_pre_w", &[VOC_CHANNELS, ENC_DIM, 3]);
    let x = b.add_conv1d(encoded, conv_pre_w, None, 1, 1, &[VOC_CHANNELS, SEQ_LEN]);

    // LeakyReLU(0.1) before upsample
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CHANNELS, SEQ_LEN]);

    // ConvTranspose1d upsample: [VOC_CHANNELS, SEQ_LEN] → [VOC_UP_CHANNELS, TIME_UP]
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
        1, // dilation
        1, // groups
        0, // output_padding
        &up_shape,
    );

    // ResBlock: InstanceNorm + Snake + Conv1d + residual
    let style_gamma = b.add_input("voc_style_gamma", &[VOC_UP_CHANNELS]);
    let style_beta = b.add_input("voc_style_beta", &[VOC_UP_CHANNELS]);
    let normed = b.add_instance_norm(
        x_up,
        eps,
        1, // axis=1 (time)
        Some(style_gamma),
        Some(style_beta),
        &up_shape,
    );

    // Snake activation
    let alpha = b.add_input("voc_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    // Conv1d in ResBlock
    let res_conv_w = b.add_input("voc_res_conv_w", &[VOC_UP_CHANNELS, VOC_UP_CHANNELS, 3]);
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &up_shape);

    // Residual connection
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);

    // LeakyReLU(0.01) before conv_post
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    // Conv post: [VOC_UP_CHANNELS, TIME_UP] → [OUT_CHANNELS, TIME_UP]
    let conv_post_w = b.add_input("voc_conv_post_w", &[OUT_CHANNELS, VOC_UP_CHANNELS, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &[OUT_CHANNELS, TIME_UP]);

    // Exp activation: log-magnitude → magnitude (always positive)
    b.add_exp(x_post, &[OUT_CHANNELS, TIME_UP])
}

// ---------------------------------------------------------------------------
// Duration predictor branch
// ---------------------------------------------------------------------------

/// Add a simplified duration predictor: Linear → dur_logits.
///
/// Input: `encoded [ENC_DIM, SEQ_LEN]`
/// Output: `dur_logits [SEQ_LEN]`
///
/// In production Kokoro, the ProsodyPredictor is more complex (multiple blocks
/// with Conv1d + AdaLayerNorm + LSTM), but this simplified branch captures the
/// core property: text encoder output → linear projection → duration logits.
///
/// Property 3 (Duration positivity): if `dur_logits` has a finite lower bound,
/// then `exp(dur_logits) > 0`, meaning no phoneme gets zero duration.
fn add_duration_predictor(b: &mut TensorBlockBuilder, encoded: TensorNodeId) -> TensorNodeId {
    // Transpose: [ENC_DIM, SEQ_LEN] → [SEQ_LEN, ENC_DIM]
    let enc_t = b.add_transpose(encoded, &[1, 0], &[SEQ_LEN, ENC_DIM]);

    // Linear: [SEQ_LEN, ENC_DIM] → [SEQ_LEN, 1] via matmul
    let dur_proj_w = b.add_input("dur_proj_w", &[1, ENC_DIM]);
    let dur_proj_b = b.add_input("dur_proj_b", &[1]);
    let dur_projected = b.add_matmul(enc_t, dur_proj_w, true, None, &[SEQ_LEN, 1]);
    let dur_proj_b_bc = b.add_broadcast(dur_proj_b, &[SEQ_LEN, 1]);
    let dur_biased = b.add_binary_add(dur_projected, dur_proj_b_bc, &[SEQ_LEN, 1]);

    // Reshape to [SEQ_LEN] — flat duration logits per phoneme
    b.add_reshape(dur_biased, &[SEQ_LEN])
}

// ---------------------------------------------------------------------------
// Full pipeline builder
// ---------------------------------------------------------------------------

/// Build the full Kokoro pipeline as a single `TensorKernelDef`.
///
/// Architecture:
///   text_features [D_MODEL, SEQ_LEN] (Variable)
///   → TextEncoder(Conv1d + ReLU + Linear) → [ENC_DIM, SEQ_LEN]
///   → Vocoder(Conv1d → LeakyReLU → ConvTranspose1d → ResBlock → LeakyReLU → Conv1d → Exp)
///   → audio [OUT_CHANNELS, TIME_UP]
///
/// Properties proven by this graph:
/// - **P1 (Non-silence):** exp() output lower bound > 0
/// - **P2 (Non-clipping):** exp() output upper bound < threshold
///
/// Returns `(TensorKernelDef, output_shape)`.
pub(super) fn build_kokoro_full_pipeline() -> (TensorKernelDef, [usize; 2]) {
    // Compile-time guard: InstanceNorm spatial dim must be > 1 (#2637).
    const _: () = assert!(TIME_UP > 1);
    let mut b = TensorBlockBuilder::new("kokoro_full_pipeline_verify");

    // Variable input: text features
    let text_input = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);

    // Stage 1: Text encoder
    let encoded = add_text_encoder(&mut b, text_input);

    // Stage 2: Vocoder decoder (includes exp output)
    let audio = add_vocoder_decoder(&mut b, encoded);

    let out_shape = [OUT_CHANNELS, TIME_UP];
    (
        b.build(audio).expect("valid kokoro full pipeline graph"),
        out_shape,
    )
}

/// Build a minimal pipeline (no text encoder) for faster verification.
///
/// Architecture:
///   features [ENC_DIM, SEQ_LEN] (Variable)
///   → Vocoder(full decoder with LeakyReLU + Snake + exp)
///   → audio [OUT_CHANNELS, TIME_UP]
///
/// Same as `build_kokoro_full_pipeline` but skips the text encoder stage,
/// using encoder output directly as the Variable input. Useful for targeted
/// vocoder-only verification with smaller graph.
pub(super) fn build_kokoro_vocoder_only_pipeline() -> (TensorKernelDef, [usize; 2]) {
    let mut b = TensorBlockBuilder::new("kokoro_vocoder_pipeline_verify");

    // Variable input: encoder features (post text-encoder)
    let encoded = b.add_input("encoder_features", &[ENC_DIM, SEQ_LEN]);

    // Vocoder decoder
    let audio = add_vocoder_decoder(&mut b, encoded);

    let out_shape = [OUT_CHANNELS, TIME_UP];
    (
        b.build(audio).expect("valid kokoro vocoder pipeline graph"),
        out_shape,
    )
}

/// Build the duration branch pipeline: text encoder → duration predictor.
///
/// Architecture:
///   text_features [D_MODEL, SEQ_LEN] (Variable)
///   → TextEncoder(Conv1d + ReLU + Linear) → [ENC_DIM, SEQ_LEN]
///   → DurationPredictor(Linear) → dur_logits [SEQ_LEN]
///
/// Property 3 (Duration positivity):
///   If IBP lower bound on dur_logits is finite, then `exp(dur_logits) > 0`,
///   proving no phoneme receives zero duration.
///
/// Returns `(TensorKernelDef, output_len)`.
pub(super) fn build_kokoro_duration_branch() -> (TensorKernelDef, usize) {
    let mut b = TensorBlockBuilder::new("kokoro_duration_branch_verify");

    // Variable input: text features
    let text_input = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);

    // Stage 1: Text encoder (shared with audio path)
    let encoded = add_text_encoder(&mut b, text_input);

    // Stage 2: Duration predictor (linear branch)
    let dur_logits = add_duration_predictor(&mut b, encoded);

    (
        b.build(dur_logits)
            .expect("valid kokoro duration branch graph"),
        SEQ_LEN,
    )
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Push text encoder weight bindings.
fn push_text_encoder_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // enc_conv_w [D_MODEL, D_MODEL, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, 3]),
        WEIGHT_MAG,
    )));

    // enc_proj_w [ENC_DIM, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ENC_DIM, D_MODEL]),
        WEIGHT_MAG,
    )));

    // enc_proj_b [ENC_DIM]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ENC_DIM]),
        0.0f32,
    )));
}

/// Push duration predictor weight bindings.
fn push_duration_predictor_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // dur_proj_w [1, ENC_DIM]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, ENC_DIM]),
        WEIGHT_MAG,
    )));

    // dur_proj_b [1]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));
}

/// Build parameter bindings for the full Kokoro pipeline.
///
/// `text_features` = Variable. All weights = ConstantTensor with small magnitude.
#[allow(clippy::vec_init_then_push)]
pub(super) fn kokoro_full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // text_features: Variable [D_MODEL, SEQ_LEN]
    bindings.push(TensorParamBinding::Variable);

    // --- Text encoder bindings ---
    push_text_encoder_bindings(&mut bindings);

    // --- Vocoder bindings ---
    push_vocoder_bindings(&mut bindings);

    bindings
}

/// Build parameter bindings for the duration branch pipeline.
///
/// `text_features` = Variable. Text encoder + duration predictor weights = ConstantTensor.
#[allow(clippy::vec_init_then_push)]
pub(super) fn kokoro_duration_branch_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // text_features: Variable [D_MODEL, SEQ_LEN]
    bindings.push(TensorParamBinding::Variable);

    // --- Text encoder bindings ---
    push_text_encoder_bindings(&mut bindings);

    // --- Duration predictor bindings ---
    push_duration_predictor_bindings(&mut bindings);

    bindings
}

/// Build parameter bindings for the vocoder-only pipeline.
///
/// `encoder_features` = Variable. All weights = ConstantTensor.
#[allow(clippy::vec_init_then_push)]
pub(super) fn kokoro_vocoder_only_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // encoder_features: Variable [ENC_DIM, SEQ_LEN]
    bindings.push(TensorParamBinding::Variable);

    // Vocoder bindings
    push_vocoder_bindings(&mut bindings);

    bindings
}

/// Push vocoder decoder weight bindings.
fn push_vocoder_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // voc_eps: ConstantScalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // voc_conv_pre_w [VOC_CHANNELS, ENC_DIM, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_CHANNELS, ENC_DIM, 3]),
        WEIGHT_MAG,
    )));

    // voc_upsample_w [VOC_CHANNELS, VOC_UP_CHANNELS, VOC_UPSAMPLE_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_CHANNELS, VOC_UP_CHANNELS, VOC_UPSAMPLE_KERNEL]),
        WEIGHT_MAG,
    )));

    // voc_style_gamma [VOC_UP_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_UP_CHANNELS]),
        1.0f32,
    )));

    // voc_style_beta [VOC_UP_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_UP_CHANNELS]),
        0.0f32,
    )));

    // voc_alpha (scalar for Snake activation)
    bindings.push(TensorParamBinding::ConstantScalar(1.0));

    // (alpha_broadcast is internal — no binding)

    // voc_res_conv_w [VOC_UP_CHANNELS, VOC_UP_CHANNELS, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOC_UP_CHANNELS, VOC_UP_CHANNELS, 3]),
        WEIGHT_MAG,
    )));

    // voc_conv_post_w [OUT_CHANNELS, VOC_UP_CHANNELS, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[OUT_CHANNELS, VOC_UP_CHANNELS, 3]),
        WEIGHT_MAG,
    )));
}

/// Build the full Kokoro pipeline translated to a NY graph with the
/// specified `NormBoundsMode`.
///
/// Convenience wrapper: builds the kernel def, translates to graph with
/// `tensor_kernel_to_graph_with_norm_mode`, and returns the graph + bindings +
/// output shape. Satisfies AC: `build_kokoro_full_pipeline_with_norm_mode(ForwardMode)`.
pub(super) fn build_kokoro_full_pipeline_with_norm_mode(
    mode: NormBoundsMode,
) -> (GraphNetwork, Vec<TensorParamBinding>, [usize; 2]) {
    let (def, out_shape) = build_kokoro_full_pipeline();
    let bindings = kokoro_full_pipeline_bindings();
    let graph = nn_verify::tensor_kernel_to_graph_with_norm_mode(&def, &bindings, mode)
        .expect("graph translation with norm mode");
    (graph, bindings, out_shape)
}
