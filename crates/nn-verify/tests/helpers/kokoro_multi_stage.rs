// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Builder helpers for Kokoro multi-stage pipeline composition tests.
//!
//! Constructs TensorBlockBuilder graphs for individual Kokoro pipeline stages
//! and chained multi-stage combinations at reduced dimensions (hidden=8, seq=4)
//! for NY tractability.
//!
//! Stage builders:
//! - `build_text_encoder`: Conv1d + ReLU + Linear projection
//! - `build_style_projector`: Linear + Tanh + Linear
//! - `build_decoder_block`: Conv1d + LeakyReLU + ConvTranspose1d + ResBlock + Exp
//! - `build_encoder_style_chain`: text encoder + style projector (chained)
//! - `build_full_four_stage_pipeline`: encoder + decoder (end-to-end)
//! - `build_multi_resblock_decoder`: decoder with 2 sequential ResBlocks
//!
//! Part of #3617: Compose verification tests for Kokoro full pipeline.
//! Part of #3351: Epic — Absolutely Best Kokoro.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Model hidden dimension (production Kokoro: 512).
pub(super) const D_MODEL: usize = 8;

/// Encoder output dim fed to style projector and vocoder.
pub(super) const ENC_DIM: usize = 8;

/// Style embedding dimension (production: 128).
pub(super) const STYLE_DIM: usize = 4;

/// Vocoder internal channels.
pub(super) const VOC_CH: usize = 4;

/// Vocoder upsampled channels.
pub(super) const VOC_UP_CH: usize = 4;

/// Output spectral channels (production: 2 * n_bins).
pub(super) const OUT_CH: usize = 4;

/// Sequence length (phoneme tokens).
pub(super) const SEQ_LEN: usize = 4;

/// Upsample stride for ConvTranspose1d.
const UP_STRIDE: usize = 2;

/// Upsample kernel for ConvTranspose1d.
const UP_KERNEL: usize = 4;

/// Upsample padding: (kernel - stride) / 2.
const UP_PADDING: usize = 1;

/// Output time after upsample: (in-1)*stride + kernel - 2*padding.
pub(super) const TIME_UP: usize = (SEQ_LEN - 1) * UP_STRIDE + UP_KERNEL - 2 * UP_PADDING;

/// Weight magnitude for small-scale test weights.
const W_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Stage builders
// ---------------------------------------------------------------------------

/// Build text encoder: Conv1d(k=3, same-pad) + ReLU + Linear projection.
///
/// Input: `text_features [D_MODEL, SEQ_LEN]` (Variable)
/// Output: `encoded [ENC_DIM, SEQ_LEN]`
pub(super) fn build_text_encoder() -> (TensorKernelDef, Vec<TensorParamBinding>, [usize; 2]) {
    let mut b = TensorBlockBuilder::new("kokoro_ms_text_encoder");

    let text = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);
    let conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let conv_out = b.add_conv1d(text, conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);
    let relu_out = b.add_relu(conv_out, &[D_MODEL, SEQ_LEN]);

    let t1 = b.add_transpose(relu_out, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let proj_b = b.add_input("enc_proj_b", &[ENC_DIM]);
    let mm = b.add_matmul(t1, proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let proj_b_bc = b.add_broadcast(proj_b, &[SEQ_LEN, ENC_DIM]);
    let biased = b.add_binary_add(mm, proj_b_bc, &[SEQ_LEN, ENC_DIM]);
    let output = b.add_transpose(biased, &[1, 0], &[ENC_DIM, SEQ_LEN]);

    let def = b.build(output).expect("text encoder graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM]), 0.0f32)),
    ];
    (def, bindings, [ENC_DIM, SEQ_LEN])
}

/// Build style projector: Linear + Tanh + Linear.
///
/// Input: `encoded [ENC_DIM, SEQ_LEN]` (Variable)
/// Output: `style [STYLE_DIM, SEQ_LEN]`
pub(super) fn build_style_projector() -> (TensorKernelDef, Vec<TensorParamBinding>, [usize; 2]) {
    let mut b = TensorBlockBuilder::new("kokoro_ms_style_projector");

    let encoded = b.add_input("encoded", &[ENC_DIM, SEQ_LEN]);
    let t1 = b.add_transpose(encoded, &[1, 0], &[SEQ_LEN, ENC_DIM]);

    let w1 = b.add_input("style_w1", &[STYLE_DIM, ENC_DIM]);
    let b1 = b.add_input("style_b1", &[STYLE_DIM]);
    let mm1 = b.add_matmul(t1, w1, true, None, &[SEQ_LEN, STYLE_DIM]);
    let b1_bc = b.add_broadcast(b1, &[SEQ_LEN, STYLE_DIM]);
    let h1 = b.add_binary_add(mm1, b1_bc, &[SEQ_LEN, STYLE_DIM]);
    let h1_act = b.add_tanh(h1, &[SEQ_LEN, STYLE_DIM]);

    let w2 = b.add_input("style_w2", &[STYLE_DIM, STYLE_DIM]);
    let b2 = b.add_input("style_b2", &[STYLE_DIM]);
    let mm2 = b.add_matmul(h1_act, w2, true, None, &[SEQ_LEN, STYLE_DIM]);
    let b2_bc = b.add_broadcast(b2, &[SEQ_LEN, STYLE_DIM]);
    let h2 = b.add_binary_add(mm2, b2_bc, &[SEQ_LEN, STYLE_DIM]);
    let output = b.add_transpose(h2, &[1, 0], &[STYLE_DIM, SEQ_LEN]);

    let def = b.build(output).expect("style projector graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[STYLE_DIM, ENC_DIM]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[STYLE_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[STYLE_DIM, STYLE_DIM]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[STYLE_DIM]), 0.0f32)),
    ];
    (def, bindings, [STYLE_DIM, SEQ_LEN])
}

/// Build decoder block: Conv1d + LeakyReLU + ConvTranspose1d + ResBlock + Exp.
///
/// Input: `features [VOC_CH, SEQ_LEN]` (Variable)
/// Output: `spectral [OUT_CH, TIME_UP]`
pub(super) fn build_decoder_block() -> (TensorKernelDef, Vec<TensorParamBinding>, [usize; 2]) {
    const _: () = assert!(TIME_UP > 1);
    let up_shape = [VOC_UP_CH, TIME_UP];
    let mut b = TensorBlockBuilder::new("kokoro_ms_decoder");

    let input = b.add_input("features", &[VOC_CH, SEQ_LEN]);
    let eps = b.add_input("dec_eps", &[1]);

    let conv_pre_w = b.add_input("dec_conv_pre_w", &[VOC_CH, VOC_CH, 3]);
    let x = b.add_conv1d(input, conv_pre_w, None, 1, 1, &[VOC_CH, SEQ_LEN]);
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CH, SEQ_LEN]);

    let up_w = b.add_input("dec_up_w", &[VOC_CH, VOC_UP_CH, UP_KERNEL]);
    let x_up =
        b.add_conv_transpose_1d(x_act, up_w, None, UP_STRIDE, UP_PADDING, 1, 1, 0, &up_shape);

    let gamma = b.add_input("dec_gamma", &[VOC_UP_CH]);
    let beta = b.add_input("dec_beta", &[VOC_UP_CH]);
    let normed = b.add_instance_norm(x_up, eps, 1, Some(gamma), Some(beta), &up_shape);

    let alpha = b.add_input("dec_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    let res_conv_w = b.add_input("dec_res_conv_w", &[VOC_UP_CH, VOC_UP_CH, 3]);
    let sublayer = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &up_shape);
    let res_out = b.add_binary_add(x_up, sublayer, &up_shape);
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    let conv_post_w = b.add_input("dec_conv_post_w", &[OUT_CH, VOC_UP_CH, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &[OUT_CH, TIME_UP]);
    let output = b.add_exp(x_post, &[OUT_CH, TIME_UP]);

    let def = b.build(output).expect("decoder block graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, VOC_CH, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_CH, VOC_UP_CH, UP_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_UP_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
    ];
    (def, bindings, [OUT_CH, TIME_UP])
}

/// Build text encoder + style projector chained pipeline.
///
/// Input: `text_features [D_MODEL, SEQ_LEN]` (Variable)
/// Output: `style [STYLE_DIM, SEQ_LEN]`
pub(super) fn build_encoder_style_chain() -> (TensorKernelDef, Vec<TensorParamBinding>, [usize; 2])
{
    let mut b = TensorBlockBuilder::new("kokoro_ms_encoder_style_chain");

    let text = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);

    // Text encoder stage
    let conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let conv_out = b.add_conv1d(text, conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);
    let relu_out = b.add_relu(conv_out, &[D_MODEL, SEQ_LEN]);
    let t1 = b.add_transpose(relu_out, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let proj_b = b.add_input("enc_proj_b", &[ENC_DIM]);
    let mm = b.add_matmul(t1, proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let proj_b_bc = b.add_broadcast(proj_b, &[SEQ_LEN, ENC_DIM]);
    let enc_out = b.add_binary_add(mm, proj_b_bc, &[SEQ_LEN, ENC_DIM]);

    // Style projector stage (stays in [SEQ_LEN, *] layout)
    let w1 = b.add_input("style_w1", &[STYLE_DIM, ENC_DIM]);
    let b1 = b.add_input("style_b1", &[STYLE_DIM]);
    let mm1 = b.add_matmul(enc_out, w1, true, None, &[SEQ_LEN, STYLE_DIM]);
    let b1_bc = b.add_broadcast(b1, &[SEQ_LEN, STYLE_DIM]);
    let h1 = b.add_binary_add(mm1, b1_bc, &[SEQ_LEN, STYLE_DIM]);
    let h1_act = b.add_tanh(h1, &[SEQ_LEN, STYLE_DIM]);
    let w2 = b.add_input("style_w2", &[STYLE_DIM, STYLE_DIM]);
    let b2 = b.add_input("style_b2", &[STYLE_DIM]);
    let mm2 = b.add_matmul(h1_act, w2, true, None, &[SEQ_LEN, STYLE_DIM]);
    let b2_bc = b.add_broadcast(b2, &[SEQ_LEN, STYLE_DIM]);
    let h2 = b.add_binary_add(mm2, b2_bc, &[SEQ_LEN, STYLE_DIM]);
    let output = b.add_transpose(h2, &[1, 0], &[STYLE_DIM, SEQ_LEN]);

    let def = b.build(output).expect("encoder-style chain graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[STYLE_DIM, ENC_DIM]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[STYLE_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[STYLE_DIM, STYLE_DIM]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[STYLE_DIM]), 0.0f32)),
    ];
    (def, bindings, [STYLE_DIM, SEQ_LEN])
}

/// Build full 4-stage pipeline: encoder + decoder.
///
/// Input: `text_features [D_MODEL, SEQ_LEN]` (Variable)
/// Output: `spectral [OUT_CH, TIME_UP]`
pub(super) fn build_full_four_stage_pipeline(
) -> (TensorKernelDef, Vec<TensorParamBinding>, [usize; 2]) {
    const _: () = assert!(TIME_UP > 1);
    let up_shape = [VOC_UP_CH, TIME_UP];
    let mut b = TensorBlockBuilder::new("kokoro_ms_full_four_stage");

    let text = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);

    // Stage 1: Text encoder
    let conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let conv_out = b.add_conv1d(text, conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);
    let relu_out = b.add_relu(conv_out, &[D_MODEL, SEQ_LEN]);
    let t1 = b.add_transpose(relu_out, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let proj_b_node = b.add_input("enc_proj_b", &[ENC_DIM]);
    let mm = b.add_matmul(t1, proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let proj_b_bc = b.add_broadcast(proj_b_node, &[SEQ_LEN, ENC_DIM]);
    let enc_biased = b.add_binary_add(mm, proj_b_bc, &[SEQ_LEN, ENC_DIM]);
    let encoded = b.add_transpose(enc_biased, &[1, 0], &[ENC_DIM, SEQ_LEN]);

    // Stage 2: Decoder (fed from encoder output)
    let eps = b.add_input("dec_eps", &[1]);
    let conv_pre_w = b.add_input("dec_conv_pre_w", &[VOC_CH, ENC_DIM, 3]);
    let x = b.add_conv1d(encoded, conv_pre_w, None, 1, 1, &[VOC_CH, SEQ_LEN]);
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CH, SEQ_LEN]);

    let up_w = b.add_input("dec_up_w", &[VOC_CH, VOC_UP_CH, UP_KERNEL]);
    let x_up =
        b.add_conv_transpose_1d(x_act, up_w, None, UP_STRIDE, UP_PADDING, 1, 1, 0, &up_shape);

    let gamma = b.add_input("dec_gamma", &[VOC_UP_CH]);
    let beta_node = b.add_input("dec_beta", &[VOC_UP_CH]);
    let normed = b.add_instance_norm(x_up, eps, 1, Some(gamma), Some(beta_node), &up_shape);

    let alpha = b.add_input("dec_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    let res_conv_w = b.add_input("dec_res_conv_w", &[VOC_UP_CH, VOC_UP_CH, 3]);
    let sublayer = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &up_shape);
    let res_out = b.add_binary_add(x_up, sublayer, &up_shape);
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    let conv_post_w = b.add_input("dec_conv_post_w", &[OUT_CH, VOC_UP_CH, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &[OUT_CH, TIME_UP]);
    let output = b.add_exp(x_post, &[OUT_CH, TIME_UP]);

    let def = b.build(output).expect("full four-stage pipeline graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, ENC_DIM, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_CH, VOC_UP_CH, UP_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_UP_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
    ];
    (def, bindings, [OUT_CH, TIME_UP])
}

/// Build decoder with 2 sequential ResBlocks.
///
/// Input: `features [VOC_CH, SEQ_LEN]` (Variable)
/// Output: `spectral [OUT_CH, TIME_UP]`
pub(super) fn build_multi_resblock_decoder(
) -> (TensorKernelDef, Vec<TensorParamBinding>, [usize; 2]) {
    const _: () = assert!(TIME_UP > 1);
    let up_shape = [VOC_UP_CH, TIME_UP];
    let mut b = TensorBlockBuilder::new("kokoro_ms_multi_resblock_decoder");

    let input = b.add_input("features", &[VOC_CH, SEQ_LEN]);
    let eps = b.add_input("dec_eps", &[1]);

    let conv_pre_w = b.add_input("dec_conv_pre_w", &[VOC_CH, VOC_CH, 3]);
    let x = b.add_conv1d(input, conv_pre_w, None, 1, 1, &[VOC_CH, SEQ_LEN]);
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CH, SEQ_LEN]);

    let up_w = b.add_input("dec_up_w", &[VOC_CH, VOC_UP_CH, UP_KERNEL]);
    let x_up =
        b.add_conv_transpose_1d(x_act, up_w, None, UP_STRIDE, UP_PADDING, 1, 1, 0, &up_shape);

    // ResBlock 1
    let gamma1 = b.add_input("dec_gamma1", &[VOC_UP_CH]);
    let beta1 = b.add_input("dec_beta1", &[VOC_UP_CH]);
    let normed1 = b.add_instance_norm(x_up, eps, 1, Some(gamma1), Some(beta1), &up_shape);
    let alpha1 = b.add_input("dec_alpha1", &[1]);
    let alpha1_bc = b.add_broadcast(alpha1, &up_shape);
    let snake1 = build_snake_scalar_kernel().expect("snake kernel 1");
    let snake1_out = b.add_elementwise(snake1, &[normed1, alpha1_bc], &up_shape);
    let res1_conv_w = b.add_input("dec_res1_conv_w", &[VOC_UP_CH, VOC_UP_CH, 3]);
    let sub1 = b.add_conv1d(snake1_out, res1_conv_w, None, 1, 1, &up_shape);
    let res1_out = b.add_binary_add(x_up, sub1, &up_shape);

    // ResBlock 2
    let gamma2 = b.add_input("dec_gamma2", &[VOC_UP_CH]);
    let beta2 = b.add_input("dec_beta2", &[VOC_UP_CH]);
    let normed2 = b.add_instance_norm(res1_out, eps, 1, Some(gamma2), Some(beta2), &up_shape);
    let alpha2 = b.add_input("dec_alpha2", &[1]);
    let alpha2_bc = b.add_broadcast(alpha2, &up_shape);
    let snake2 = build_snake_scalar_kernel().expect("snake kernel 2");
    let snake2_out = b.add_elementwise(snake2, &[normed2, alpha2_bc], &up_shape);
    let res2_conv_w = b.add_input("dec_res2_conv_w", &[VOC_UP_CH, VOC_UP_CH, 3]);
    let sub2 = b.add_conv1d(snake2_out, res2_conv_w, None, 1, 1, &up_shape);
    let res2_out = b.add_binary_add(res1_out, sub2, &up_shape);

    let res_act = b.add_leaky_relu(res2_out, 0.01, &up_shape);
    let conv_post_w = b.add_input("dec_conv_post_w", &[OUT_CH, VOC_UP_CH, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &[OUT_CH, TIME_UP]);
    let output = b.add_exp(x_post, &[OUT_CH, TIME_UP]);

    let def = b.build(output).expect("multi-resblock decoder graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, VOC_CH, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_CH, VOC_UP_CH, UP_KERNEL]),
            W_MAG,
        )),
        // ResBlock 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_UP_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
        // ResBlock 2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_UP_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_UP_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
        // conv_post
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CH, VOC_UP_CH, 3]),
            W_MAG,
        )),
    ];
    (def, bindings, [OUT_CH, TIME_UP])
}
