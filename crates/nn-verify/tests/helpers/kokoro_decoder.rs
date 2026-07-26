// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Kokoro decoder (ISTFTNet generator) NY
//! composition tests.
//!
//! The Kokoro decoder is an ISTFTNet-based vocoder with this architecture:
//!   Conv1d (conv_pre) → N × (ConvTranspose1d upsample + ResBlock) →
//!   Conv1d (conv_post) → split → exp(log_mag) + sin(phase)
//!
//! ResBlock sub-layers: AdaIN(InstanceNorm + style affine) → Snake → Conv1d
//!
//! Simplifications for NY tractability:
//! - AdaIN decomposed as InstanceNorm + affine (uses native InstanceNorm op)
//! - LeakyReLU now supported via `build_kokoro_decoder_with_leaky_relu()` (#1741)
//! - Single ResBlock sub-layer instead of 3 dilated sub-layers
//! - Noise injection omitted (additive noise is constant in NY)
//! - Sin omitted (no TensorOpKind variant); exp retained for log-magnitude
//! - Snake alpha is scalar (not per-channel) — elementwise translator limitation
//! - 1 upsampling stage instead of 2
//!
//! Part of #1696 AC6: Kokoro decoder NY composition.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale dimensions for NY tractability
// ---------------------------------------------------------------------------

/// Input channels (production: 512).
pub(super) const IN_CHANNELS: usize = 8;

/// Channels after conv_pre (production: 512).
const CHANNELS: usize = 8;

/// Channels after one upsample stage (production: 256).
const UPSAMPLED_CHANNELS: usize = 4;

/// Output channels (production: 2 * n_bins where n_bins=10).
pub(super) const OUT_CHANNELS: usize = 4;

/// Time length of input (production: varies).
pub(super) const TIME_IN: usize = 4;

/// ConvTranspose1d stride for upsampling (production: 10).
const UPSAMPLE_STRIDE: usize = 2;

/// ConvTranspose1d kernel for upsampling (production: 20, typically 2*stride).
const UPSAMPLE_KERNEL: usize = 4;

/// Padding for ConvTranspose1d: (kernel - stride) / 2.
const UPSAMPLE_PADDING: usize = 1;

/// Output time after upsampling.
/// conv_transpose1d: (in-1)*stride + kernel - 2*padding
pub(super) const TIME_UP: usize =
    (TIME_IN - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;

/// Conv1d kernel for ResBlock convolutions (production: 3).
const RESBLOCK_KERNEL: usize = 3;

/// Padding for ResBlock Conv1d (same-padding).
const RESBLOCK_PADDING: usize = 1;

/// Weight magnitude for small-scale test weights.
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Full decoder builder
// ---------------------------------------------------------------------------

/// Build a simplified Kokoro decoder as a single `TensorKernelDef`.
///
/// Architecture:
///   features [IN_CHANNELS, TIME_IN] (Variable)
///   → Conv1d conv_pre [CHANNELS, TIME_IN]
///   → ConvTranspose1d upsample [UPSAMPLED_CHANNELS, TIME_UP]
///   → ResBlock(InstanceNorm + Snake + Conv1d) + residual
///   → Conv1d conv_post [OUT_CHANNELS, TIME_UP]
///   → Exp (log-magnitude → magnitude)
///
/// Returns `(TensorKernelDef, output_shape)`.
pub(super) fn build_kokoro_decoder() -> (TensorKernelDef, [usize; 2]) {
    // Compile-time guard: InstanceNorm spatial dim must be > 1 (#2637).
    const _: () = assert!(TIME_UP > 1);
    let mut b = TensorBlockBuilder::new("kokoro_decoder_verify");
    let up_shape = [UPSAMPLED_CHANNELS, TIME_UP];

    // --- Variable input: encoder features ---
    let input = b.add_input("features", &[IN_CHANNELS, TIME_IN]);

    // --- Shared epsilon ---
    let eps = b.add_input("eps", &[1]);

    // --- Conv pre: [IN_CHANNELS, TIME_IN] → [CHANNELS, TIME_IN] ---
    let conv_pre_w = b.add_input("conv_pre_w", &[CHANNELS, IN_CHANNELS, 7]);
    let x = b.add_conv1d(input, conv_pre_w, None, 1, 3, &[CHANNELS, TIME_IN]);

    // --- ConvTranspose1d upsample: [CHANNELS, TIME_IN] → [UPSAMPLED_CHANNELS, TIME_UP] ---
    let upsample_w = b.add_input(
        "upsample_w",
        &[CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL],
    );
    let x_up = b.add_conv_transpose_1d(
        x,
        upsample_w,
        None,
        UPSAMPLE_STRIDE,
        UPSAMPLE_PADDING,
        1, // dilation
        1, // groups
        0, // output_padding
        &up_shape,
    );

    // --- ResBlock sub-layer: InstanceNorm + Snake + Conv1d ---

    // InstanceNorm with style affine (core of AdaIN)
    let style_gamma = b.add_input("res_style_gamma", &[UPSAMPLED_CHANNELS]);
    let style_beta = b.add_input("res_style_beta", &[UPSAMPLED_CHANNELS]);
    let normed = b.add_instance_norm(
        x_up,
        eps,
        1, // axis=1 (time dimension)
        Some(style_gamma),
        Some(style_beta),
        &up_shape,
    );

    // Snake activation: x + (1/alpha) * sin^2(alpha * x)
    // Alpha is scalar (all channels share the same alpha for NY tractability).
    // Per-channel alpha would require ConstantTensor, which the elementwise translator
    // rejects as WeightTensor. Scalar ConstantScalar maps to Constant, hitting the
    // native SnakeLayer fast path.
    let alpha = b.add_input("res_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    // Conv1d (same-padding, stride=1)
    let res_conv_w = b.add_input(
        "res_conv_w",
        &[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL],
    );
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, RESBLOCK_PADDING, &up_shape);

    // Residual connection
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);

    // --- Conv post: [UPSAMPLED_CHANNELS, TIME_UP] → [OUT_CHANNELS, TIME_UP] ---
    let conv_post_w = b.add_input("conv_post_w", &[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]);
    let x_post = b.add_conv1d(res_out, conv_post_w, None, 1, 3, &[OUT_CHANNELS, TIME_UP]);

    // --- Exp activation (log-magnitude → magnitude) ---
    let output = b.add_exp(x_post, &[OUT_CHANNELS, TIME_UP]);

    let out_shape = [OUT_CHANNELS, TIME_UP];
    (
        b.build(output).expect("valid kokoro decoder graph"),
        out_shape,
    )
}

// ---------------------------------------------------------------------------
// Expanded decoder with LeakyReLU (closer to real Kokoro architecture)
// ---------------------------------------------------------------------------

/// Build a Kokoro decoder with LeakyReLU activations matching the real
/// ISTFTNet architecture.
///
/// Architecture:
///   features [IN_CHANNELS, TIME_IN] (Variable)
///   → Conv1d conv_pre [CHANNELS, TIME_IN]
///   → LeakyReLU(0.1)
///   → ConvTranspose1d upsample [UPSAMPLED_CHANNELS, TIME_UP]
///   → ResBlock(InstanceNorm + Snake + Conv1d) + residual
///   → LeakyReLU(0.01)
///   → Conv1d conv_post [OUT_CHANNELS, TIME_UP]
///   → Exp (log-magnitude → magnitude)
///
/// Compared to `build_kokoro_decoder()`:
/// - Adds `LeakyReLU(0.1)` before upsample (real architecture)
/// - Adds `LeakyReLU(0.01)` before conv_post (real architecture)
///
/// Returns `(TensorKernelDef, output_shape)`.
#[allow(dead_code)]
pub(super) fn build_kokoro_decoder_with_leaky_relu() -> (TensorKernelDef, [usize; 2]) {
    // Compile-time guard: InstanceNorm spatial dim must be > 1 (#2637).
    const _: () = assert!(TIME_UP > 1);
    let mut b = TensorBlockBuilder::new("kokoro_decoder_leaky_relu_verify");
    let up_shape = [UPSAMPLED_CHANNELS, TIME_UP];

    // --- Variable input: encoder features ---
    let input = b.add_input("features", &[IN_CHANNELS, TIME_IN]);

    // --- Shared epsilon ---
    let eps = b.add_input("eps", &[1]);

    // --- Conv pre: [IN_CHANNELS, TIME_IN] → [CHANNELS, TIME_IN] ---
    let conv_pre_w = b.add_input("conv_pre_w", &[CHANNELS, IN_CHANNELS, 7]);
    let x = b.add_conv1d(input, conv_pre_w, None, 1, 3, &[CHANNELS, TIME_IN]);

    // --- LeakyReLU(0.1) before upsample (matches real Kokoro per-stage activation) ---
    let x_act = b.add_leaky_relu(x, 0.1, &[CHANNELS, TIME_IN]);

    // --- ConvTranspose1d upsample: [CHANNELS, TIME_IN] → [UPSAMPLED_CHANNELS, TIME_UP] ---
    let upsample_w = b.add_input(
        "upsample_w",
        &[CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL],
    );
    let x_up = b.add_conv_transpose_1d(
        x_act,
        upsample_w,
        None,
        UPSAMPLE_STRIDE,
        UPSAMPLE_PADDING,
        1, // dilation
        1, // groups
        0, // output_padding
        &up_shape,
    );

    // --- ResBlock sub-layer: InstanceNorm + Snake + Conv1d ---

    // InstanceNorm with style affine (core of AdaIN)
    let style_gamma = b.add_input("res_style_gamma", &[UPSAMPLED_CHANNELS]);
    let style_beta = b.add_input("res_style_beta", &[UPSAMPLED_CHANNELS]);
    let normed = b.add_instance_norm(
        x_up,
        eps,
        1, // axis=1 (time dimension)
        Some(style_gamma),
        Some(style_beta),
        &up_shape,
    );

    // Snake activation
    let alpha = b.add_input("res_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    // Conv1d (same-padding, stride=1)
    let res_conv_w = b.add_input(
        "res_conv_w",
        &[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL],
    );
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, RESBLOCK_PADDING, &up_shape);

    // Residual connection
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);

    // --- LeakyReLU(0.01) before conv_post (matches real Kokoro final activation) ---
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    // --- Conv post: [UPSAMPLED_CHANNELS, TIME_UP] → [OUT_CHANNELS, TIME_UP] ---
    let conv_post_w = b.add_input("conv_post_w", &[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 3, &[OUT_CHANNELS, TIME_UP]);

    // --- Exp activation (log-magnitude → magnitude) ---
    let output = b.add_exp(x_post, &[OUT_CHANNELS, TIME_UP]);

    let out_shape = [OUT_CHANNELS, TIME_UP];
    (
        b.build(output).expect("valid kokoro decoder graph"),
        out_shape,
    )
}

/// Build parameter bindings for the expanded Kokoro decoder with LeakyReLU.
///
/// Same binding order as `kokoro_decoder_bindings()` — LeakyReLU nodes have
/// no additional inputs (slope is a field on the TensorOpKind, not an input).
#[allow(dead_code)]
#[allow(clippy::vec_init_then_push)]
pub(super) fn kokoro_decoder_leaky_relu_bindings() -> Vec<TensorParamBinding> {
    // LeakyReLU adds zero new inputs — it's the same binding list.
    kokoro_decoder_bindings()
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Build parameter bindings for the Kokoro decoder.
///
/// features = Variable, all other inputs = ConstantTensor or ConstantScalar.
/// Broadcast nodes are internal (not bindings).
#[allow(clippy::vec_init_then_push)]
pub(super) fn kokoro_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // features: Variable [IN_CHANNELS, TIME_IN]
    bindings.push(TensorParamBinding::Variable);

    // eps: Constant scalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // conv_pre weight [CHANNELS, IN_CHANNELS, 7]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[CHANNELS, IN_CHANNELS, 7]),
        WEIGHT_MAG,
    )));

    // upsample weight [CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL]),
        WEIGHT_MAG,
    )));

    // ResBlock: style_gamma [UPSAMPLED_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[UPSAMPLED_CHANNELS]),
        1.0f32,
    )));

    // ResBlock: style_beta [UPSAMPLED_CHANNELS]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[UPSAMPLED_CHANNELS]),
        0.0f32,
    )));

    // ResBlock: alpha (scalar, shared across channels for NY tractability)
    bindings.push(TensorParamBinding::ConstantScalar(1.0));

    // (alpha_broadcast is an internal node, not a binding — skip)

    // ResBlock: conv weight [UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL]),
        WEIGHT_MAG,
    )));

    // conv_post weight [OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]),
        WEIGHT_MAG,
    )));

    bindings
}
