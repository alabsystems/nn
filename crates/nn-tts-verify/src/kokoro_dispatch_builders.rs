// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Internal builder functions for Kokoro dispatch plan construction.
//!
//! Extracted from `kokoro_dispatch.rs` to stay within the 500-line file limit.
//! These functions build `DispatchStep` sequences for each architectural component
//! of the Kokoro-82M ISTFTNet vocoder.

use nn_dsl::ir::ScalarType;
use nn_dsl::{Conv1dParams, DispatchStep};

use crate::dispatch_builder::DispatchBuilder;

use super::{
    INITIAL_CHANNELS, N_BINS, RESBLOCK_DILATIONS, RESBLOCK_KERNELS, STYLE_DIM, UPSAMPLE_KERNELS,
    UPSAMPLE_RATES,
};

// ---------------------------------------------------------------------------
// Step builders
// ---------------------------------------------------------------------------

/// conv_pre: Conv1d(512, 512, k=7, pad=3).
pub(super) fn build_conv_pre(b: &mut DispatchBuilder, t_len: usize) {
    b.conv1d(
        "conv_pre",
        INITIAL_CHANNELS,
        INITIAL_CHANNELS,
        7,
        t_len,
        1,
        3,
        1,
    );
}

/// Single ResBlock dilation layer:
/// AdaIN1(Linear) → Snake → Conv1d(dilated) → AdaIN2(Linear) → Snake → Conv1d(d=1) → Add
pub(super) fn build_resblock_dilation(
    b: &mut DispatchBuilder,
    stage: usize,
    rb: usize,
    dil_idx: usize,
    channels: usize,
    t_len: usize,
    kernel_size: usize,
    dilation: usize,
) {
    let prefix = format!("rb_{stage}_{rb}_d{dil_idx}");
    let pad = dilation * (kernel_size - 1) / 2;
    let pad2 = (kernel_size - 1) / 2;

    b.linear(format!("{prefix}_adain1"), STYLE_DIM, 2 * channels, 1);
    b.sigmoid(format!("{prefix}_snake1"), channels * t_len);
    b.conv1d(
        format!("{prefix}_conv1"),
        channels,
        channels,
        kernel_size,
        t_len,
        1,
        pad,
        dilation,
    );
    b.linear(format!("{prefix}_adain2"), STYLE_DIM, 2 * channels, 1);
    b.sigmoid(format!("{prefix}_snake2"), channels * t_len);
    b.conv1d(
        format!("{prefix}_conv2"),
        channels,
        channels,
        kernel_size,
        t_len,
        1,
        pad2,
        1,
    );
    b.binary_add(format!("{prefix}_residual"), channels * t_len);
}

/// Noise injection for one upsample stage:
/// Conv1d(2*n_bins → channels, k=1, stride=cumulative_remaining_stride) + ResBlock.
pub(super) fn build_noise_injection(
    b: &mut DispatchBuilder,
    stage: usize,
    channels: usize,
    t_len: usize,
    cumulative_stride: usize,
) {
    // Noise Conv1d: downsample harmonic source (stride > 1, manual construction)
    let noise_in_len = t_len * cumulative_stride;
    let (ni, nw, nb, no) = (
        b.alloc_node(),
        b.alloc_node(),
        b.alloc_node(),
        b.alloc_node(),
    );
    b.push_step(DispatchStep::Conv1d(Conv1dParams::new(
        format!("noise_conv_{stage}"),
        ScalarType::F32,
        ni,
        nw,
        Some(nb),
        no,
        2 * N_BINS,
        channels,
        1,
        noise_in_len,
        channels * t_len,
        cumulative_stride,
        0,
        1,
        1,
    )));

    // Noise ResBlock
    for (dil_idx, &dilation) in RESBLOCK_DILATIONS.iter().enumerate() {
        build_resblock_dilation(
            b,
            stage,
            0,
            dil_idx,
            channels,
            t_len,
            RESBLOCK_KERNELS[0],
            dilation,
        );
    }

    b.binary_add(format!("noise_add_{stage}"), channels * t_len);
}

/// Output stage: LeakyReLU → conv_post(channels→2*n_bins, k=7) → split → exp + sin.
pub(super) fn build_output_stage(b: &mut DispatchBuilder, channels: usize, t_len: usize) {
    b.sigmoid("output_leaky_relu", channels * t_len);
    b.conv1d("conv_post", channels, 2 * N_BINS, 7, t_len, 1, 3, 1);
    b.tanh("exp_magnitude", N_BINS * t_len);
    b.tanh("sin_phase", N_BINS * t_len);
}

/// Build one upsample stage: LeakyReLU → ConvTranspose1d → noise → 3 ResBlocks.
///
/// Returns `(new_t_len, new_channels)`.
pub(super) fn build_upsample_stage(
    b: &mut DispatchBuilder,
    stage: usize,
    channels: usize,
    t_len: usize,
) -> (usize, usize) {
    let rate = UPSAMPLE_RATES[stage];
    let kern = UPSAMPLE_KERNELS[stage];
    let next_channels = channels / 2;
    let t_out = t_len * rate;
    let padding = (kern - rate) / 2;

    b.sigmoid(format!("stage_{stage}_leaky_relu"), channels * t_len);
    b.conv_transpose1d(
        format!("upsample_{stage}"),
        channels,
        next_channels,
        kern,
        t_len,
        rate,
        padding,
    );

    // Noise injection
    let remaining_stride: usize = UPSAMPLE_RATES[stage + 1..].iter().product();
    let cum_stride = if remaining_stride > 0 {
        remaining_stride
    } else {
        1
    };
    build_noise_injection(b, stage, next_channels, t_out, cum_stride);

    // 3 ResBlocks
    for (rb_idx, &rb_kernel) in RESBLOCK_KERNELS.iter().enumerate() {
        for (dil_idx, &dilation) in RESBLOCK_DILATIONS.iter().enumerate() {
            build_resblock_dilation(
                b,
                stage,
                rb_idx + 1,
                dil_idx,
                next_channels,
                t_out,
                rb_kernel,
                dilation,
            );
        }
    }

    (t_out, next_channels)
}
