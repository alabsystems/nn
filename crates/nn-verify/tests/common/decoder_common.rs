// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for `attention_decoder_*` test builders (Phases 28–34).
//!
//! Consolidates duplicated constants, weight wrappers, topology functions,
//! struct types, and utility functions across 7 attention decoder helper
//! files (~700 lines total, ~100 lines per file).
//!
//! Part of #1970.

use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

use super::{bounds_min_max, weights};

// -------------------------------------------------------------------------
// Shared constants
// -------------------------------------------------------------------------

pub(crate) const T_DEC: usize = 4;
pub(crate) const T_ENC: usize = 4;
pub(crate) const D_MODEL: usize = 8;
pub(crate) const NUM_HEADS: usize = 2;
pub(crate) const D_K: usize = D_MODEL / NUM_HEADS;
pub(crate) const FFN_DIM: usize = 16;
pub(crate) const INIT_CHANNELS: usize = 8;

pub(crate) const UPSAMPLE_STRIDE: usize = 2;
pub(crate) const UPSAMPLE_KERNEL: usize = 4;
pub(crate) const UPSAMPLE_PADDING: usize = 1;
pub(crate) const MASK_VALUE: f32 = -1e9;
pub(crate) const WEIGHT_MAG: f32 = 0.001;
pub(crate) const OUT_KERNEL: usize = 7;
pub(crate) const OUT_PADDING: usize = 3;

/// Default Kokoro kernel sizes: [3, 7, 11].
pub(crate) const KOKORO_KERNELS: &[usize] = &[3, 7, 11];

/// Default Kokoro dilations: [1, 3, 5].
pub(crate) const KOKORO_DILATIONS: &[usize] = &[1, 3, 5];

/// Output mono channel count.
pub(crate) const OUTPUT_CHANNELS: usize = 1;

/// Noise source channels (scaled down for verification: 4 channels).
pub(crate) const NOISE_CHANNELS: usize = 4;

/// Full noise source temporal length (pre-downsampled).
pub(crate) const NOISE_T_FULL: usize = 32;

// -------------------------------------------------------------------------
// Shared structs
// -------------------------------------------------------------------------

/// IDs for a single attention layer's parameters in the TensorBlockBuilder.
pub(crate) struct AttnLayerIds {
    pub(crate) w_q: TensorNodeId,
    pub(crate) w_k: TensorNodeId,
    pub(crate) w_v: TensorNodeId,
    pub(crate) w_o: TensorNodeId,
    pub(crate) mask: TensorNodeId,
    pub(crate) ln_w: TensorNodeId,
    pub(crate) ln_b: TensorNodeId,
    pub(crate) ln_eps: TensorNodeId,
    pub(crate) ffn_up: TensorNodeId,
    pub(crate) ffn_down: TensorNodeId,
}

/// IDs for a single dilated conv sub-layer within a ResBlock.
pub(crate) struct DilatedSubLayerIds {
    pub(crate) gamma1: TensorNodeId,
    pub(crate) beta1: TensorNodeId,
    pub(crate) alpha1: TensorNodeId,
    pub(crate) conv_w: TensorNodeId,
    pub(crate) gamma2: TensorNodeId,
    pub(crate) beta2: TensorNodeId,
    pub(crate) alpha2: TensorNodeId,
    pub(crate) conv_unit_w: TensorNodeId,
}

/// IDs for a single ResBlock (one kernel size, multiple dilations).
pub(crate) struct ResBlockIds {
    pub(crate) sublayers: Vec<DilatedSubLayerIds>,
}

/// Whether the channel projection to mono happens before or after exp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ProjectionOrder {
    /// Conv1d(ch→1) AFTER exp: project in magnitude domain (default Kokoro).
    AfterExp,
    /// Conv1d(ch→1) BEFORE exp: project in log domain, then exp.
    BeforeExp,
}

// -------------------------------------------------------------------------
// Shared functions
// -------------------------------------------------------------------------

/// Compute same-padding for a dilated conv with stride=1.
/// padding = dilation * (kernel_size - 1) / 2
pub(crate) fn dilated_same_padding(kernel_size: usize, dilation: usize) -> usize {
    dilation * (kernel_size - 1) / 2
}

/// Channel count at a given upsample stage (halves each stage).
pub(crate) fn channels_at_stage(stage: usize) -> usize {
    INIT_CHANNELS >> stage
}

/// Temporal length after `num_stages` upsample stages starting from `T_DEC`.
pub(crate) fn time_after_stages(num_stages: usize) -> usize {
    let mut t = T_DEC;
    for _ in 0..num_stages {
        t = (t - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;
    }
    t
}

/// Cumulative stride for noise conv at a given stage.
/// With UPSAMPLE_STRIDE=2 and ns total stages, stride = 2^(ns-si-1).
pub(crate) fn noise_conv_stride(stage_idx: usize, num_stages: usize) -> usize {
    let remaining = num_stages - stage_idx - 1;
    1 << remaining
}

// -------------------------------------------------------------------------
// Weight wrappers — delegate to common::weights (Part of #1938)
// -------------------------------------------------------------------------

pub(crate) fn near_identity(d: usize, p: f32) -> ArrayD<f32> {
    weights::near_identity(d, p)
}

pub(crate) fn scaled_diag(out_d: usize, in_d: usize, s: f32) -> ArrayD<f32> {
    weights::ffn_weight(out_d, in_d, s)
}

pub(crate) fn encoder_k(t: usize, d: usize) -> ArrayD<f32> {
    weights::encoder_k(t, d)
}

// -------------------------------------------------------------------------
// PE, mask, and utility — delegate to common (Part of #1970)
// -------------------------------------------------------------------------

/// Sinusoidal PE with head-interleaved frequencies.
/// Alias for `common::sinusoidal_pe_interleaved`.
pub(crate) fn sin_pe(seq: usize, dm: usize, nh: usize) -> ArrayD<f32> {
    super::sinusoidal_pe_interleaved(seq, dm, nh)
}

/// Strict causal mask.
/// Alias for `common::build_strict_causal_mask`.
pub(crate) fn causal_mask(td: usize, te: usize) -> ArrayD<f32> {
    super::build_strict_causal_mask(td, te)
}

/// Uniform tensor filled with a single value.
pub(crate) fn uniform(shape: &[usize], val: f32) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), val)
}

/// Pre-compute noise signal for additive injection at a decoder stage.
pub(crate) fn precompute_noise_signal(
    out_ch: usize,
    out_t: usize,
    stage_idx: usize,
    num_stages: usize,
    noise_magnitude: f32,
) -> ArrayD<f32> {
    let stride = noise_conv_stride(stage_idx, num_stages);
    let data: Vec<f32> = (0..out_ch * out_t)
        .map(|i| {
            let ch = i / out_t;
            let t = i % out_t;
            noise_magnitude * ((ch as f32 + 1.0) * (t as f32 * 0.3 / stride as f32)).sin() * 0.1
        })
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_ch, out_t]), data).unwrap()
}

// -------------------------------------------------------------------------
// Analysis helpers — shared across scaled/output/pipeline verification
// -------------------------------------------------------------------------

/// Result of analyzing a verified decoder pipeline's output bounds.
#[derive(Debug)]
pub(crate) struct ScaledPipelineResult {
    pub(crate) d_model: usize,
    pub(crate) num_attn_layers: usize,
    pub(crate) num_stages: usize,
    pub(crate) kernel_sizes: Vec<usize>,
    pub(crate) dilations: Vec<usize>,
    pub(crate) proj_order: ProjectionOrder,
    pub(crate) graph_nodes: usize,
    pub(crate) output_channels: usize,
    pub(crate) output_time: usize,
    pub(crate) min_output_lo: f32,
    pub(crate) max_output_hi: f32,
    pub(crate) avg_bound_width: f32,
    pub(crate) all_positive: bool,
    pub(crate) all_finite: bool,
}

/// Analyze output bounds from a verified scaled pipeline.
pub(crate) fn analyze_scaled_bounds(
    output: &BoundedTensor,
    d_model: usize,
    na: usize,
    ns: usize,
    kernel_sizes: &[usize],
    dilations: &[usize],
    proj_order: ProjectionOrder,
    nodes: usize,
    output_channels: usize,
    output_time: usize,
) -> ScaledPipelineResult {
    let (lo, hi) = output.lower_upper();
    let fl: Vec<f32> = lo.iter().copied().collect();
    let fh: Vec<f32> = hi.iter().copied().collect();

    let (min_lo, max_hi) = bounds_min_max(output);
    let n = fl.len().max(1) as f32;
    let avg_w: f32 = fl.iter().zip(fh.iter()).map(|(&l, &h)| h - l).sum::<f32>() / n;

    ScaledPipelineResult {
        d_model,
        num_attn_layers: na,
        num_stages: ns,
        kernel_sizes: kernel_sizes.to_vec(),
        dilations: dilations.to_vec(),
        proj_order,
        graph_nodes: nodes,
        output_channels,
        output_time,
        min_output_lo: min_lo,
        max_output_hi: max_hi,
        avg_bound_width: avg_w,
        all_positive: fl.iter().all(|&v| v >= 0.0),
        all_finite: fl.iter().chain(fh.iter()).all(|v| v.is_finite()),
    }
}
