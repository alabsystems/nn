// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared builder helpers for Demucs encoder block composition tests.
//!
//! Parametric over channel counts and spatial dimension, supporting both
//! temporal encoder (8→16 ch, T=16) and spectral encoder (4→8 ch, F=16).
//!
//! Replaces `helpers/temporal_encoder.rs` and `helpers/spectral_encoder.rs`
//! which were ~95% identical.
//!
//! Part of #1982: nn-verify test binary consolidation.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

use super::common::conv1d_out_len;

// ---------------------------------------------------------------------------
// Parametric encoder block configuration
// ---------------------------------------------------------------------------

/// Configuration for a Demucs encoder block.
///
/// The temporal and spectral encoders share identical topology:
///   Conv1d(stride) → GELU → DConv(×depth) → Rewrite(Conv1d k=1) → GLU
///
/// They differ only in channel counts and the semantic meaning of the
/// spatial dimension (temporal length vs frequency bins).
pub(super) struct EncoderBlockConfig {
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) spatial_in: usize,
    pub(super) block_name: &'static str,
    pub(super) conv_kernel: usize,
    pub(super) conv_stride: usize,
    pub(super) conv_padding: usize,
    pub(super) dconv_compress_ratio: usize,
    pub(super) dconv_kernel: usize,
    pub(super) dconv_depth: usize,
}

/// Temporal encoder block configuration (8→16 ch, T=16).
pub(super) const TEMPORAL_CONFIG: EncoderBlockConfig = EncoderBlockConfig {
    in_channels: 8,
    out_channels: 16,
    spatial_in: 16,
    block_name: "demucs_enc_block_verify",
    conv_kernel: 8,
    conv_stride: 4,
    conv_padding: 2,
    dconv_compress_ratio: 4,
    dconv_kernel: 3,
    dconv_depth: 2,
};

/// Spectral encoder block configuration (4→8 ch, F=16).
pub(super) const SPECTRAL_CONFIG: EncoderBlockConfig = EncoderBlockConfig {
    in_channels: 4,
    out_channels: 8,
    spatial_in: 16,
    block_name: "demucs_spec_enc_block_verify",
    conv_kernel: 8,
    conv_stride: 4,
    conv_padding: 2,
    dconv_compress_ratio: 4,
    dconv_kernel: 3,
    dconv_depth: 2,
};

// ---------------------------------------------------------------------------
// Topology builder helpers
// ---------------------------------------------------------------------------

/// Collected input node IDs for a single DConv sub-layer.
struct DConvInputs {
    conv_compress_weight: TensorNodeId,
    conv_compress_bias: TensorNodeId,
    norm_compress_gamma: TensorNodeId,
    norm_compress_beta: TensorNodeId,
    conv_expand_weight: TensorNodeId,
    conv_expand_bias: TensorNodeId,
    norm_expand_gamma: TensorNodeId,
    norm_expand_beta: TensorNodeId,
    layer_scale: TensorNodeId,
    eps1: TensorNodeId,
    eps2: TensorNodeId,
    dilation: usize,
}

impl DConvInputs {
    fn add_to_builder(
        b: &mut TensorBlockBuilder,
        k: usize,
        channels: usize,
        compressed: usize,
        dconv_kernel: usize,
    ) -> Self {
        let doubled = channels * 2;
        Self {
            conv_compress_weight: b
                .add_input(&format!("dc{k}_cw"), &[compressed, channels, dconv_kernel]),
            conv_compress_bias: b.add_input(&format!("dc{k}_cb"), &[compressed]),
            norm_compress_gamma: b.add_input(&format!("dc{k}_ng"), &[compressed]),
            norm_compress_beta: b.add_input(&format!("dc{k}_nb"), &[compressed]),
            conv_expand_weight: b.add_input(&format!("dc{k}_ew"), &[doubled, compressed, 1]),
            conv_expand_bias: b.add_input(&format!("dc{k}_eb"), &[doubled]),
            norm_expand_gamma: b.add_input(&format!("dc{k}_eng"), &[doubled]),
            norm_expand_beta: b.add_input(&format!("dc{k}_enb"), &[doubled]),
            layer_scale: b.add_input(&format!("dc{k}_ls"), &[channels]),
            eps1: b.add_input(&format!("dc{k}_eps"), &[1]),
            eps2: b.add_input(&format!("dc{k}_eps2"), &[1]),
            dilation: 1 << k,
        }
    }
}

/// Build a DConv sub-layer inline within the block builder.
///
/// Conv1d(dilated) → GroupNorm(G=1) → GELU → Conv1d(1×1) → GroupNorm(G=1)
/// → GLU → LayerScale → residual_add
fn build_dconv_sublayer(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    dc: &DConvInputs,
    channels: usize,
    compressed: usize,
    spatial_len: usize,
    dconv_kernel: usize,
) -> TensorNodeId {
    let doubled = channels * 2;
    let dc_padding = dc.dilation * (dconv_kernel - 1) / 2;

    // Dilated Conv1d: [channels, S] → [compressed, S]
    let c1 = b.add_conv1d_full(
        input,
        dc.conv_compress_weight,
        Some(dc.conv_compress_bias),
        1,
        dc_padding,
        dc.dilation,
        1,
        &[compressed, spatial_len],
    );

    // GroupNorm(G=1) on compressed channels
    let n1 = b.add_group_norm_g1(
        c1,
        dc.eps1,
        Some(dc.norm_compress_gamma),
        Some(dc.norm_compress_beta),
        compressed,
        spatial_len,
    );

    // GELU
    let g1 = b.add_gelu(n1, &[compressed, spatial_len]);

    // Conv1d expand: [compressed, S] → [channels*2, S]
    let c2 = b.add_conv1d(
        g1,
        dc.conv_expand_weight,
        Some(dc.conv_expand_bias),
        1,
        0,
        &[doubled, spatial_len],
    );

    // GroupNorm(G=1) on expanded channels
    let n2 = b.add_group_norm_g1(
        c2,
        dc.eps2,
        Some(dc.norm_expand_gamma),
        Some(dc.norm_expand_beta),
        doubled,
        spatial_len,
    );

    // GLU: [channels*2, S] → [channels, S]
    let glu = b.add_glu(n2, 0, &[doubled, spatial_len]).expect("even dim");

    // LayerScale: broadcast [channels] → [channels, S], multiply
    let ls = b.add_layer_scale(glu, dc.layer_scale, &[channels, spatial_len]);

    // Residual: input + scaled
    b.add_binary_add(input, ls, &[channels, spatial_len])
}

/// Build a Demucs encoder block using TensorBlockBuilder with the given config.
///
/// Input layout:
///   - "data" [in_channels, spatial_in] — Variable (verified input)
///   - Remaining: weight/bias constants for Conv1d, DConv, Rewrite
///
/// Returns (TensorKernelDef, output spatial length, output channels).
pub(super) fn build_encoder_block(
    cfg: &EncoderBlockConfig,
) -> (nn_dsl::tensor_ir::TensorKernelDef, usize, usize) {
    let compressed = cfg.out_channels / cfg.dconv_compress_ratio;
    let doubled = cfg.out_channels * 2;

    let mut b = TensorBlockBuilder::new(cfg.block_name);

    // --- Variable input ---
    let data = b.add_input("data", &[cfg.in_channels, cfg.spatial_in]);

    // --- Conv1d inputs ---
    let conv_weight = b.add_input(
        "conv_weight",
        &[cfg.out_channels, cfg.in_channels, cfg.conv_kernel],
    );
    let conv_bias = b.add_input("conv_bias", &[cfg.out_channels]);

    // --- DConv inputs ---
    let mut dconv_inputs = Vec::with_capacity(cfg.dconv_depth);
    for k in 0..cfg.dconv_depth {
        let di =
            DConvInputs::add_to_builder(&mut b, k, cfg.out_channels, compressed, cfg.dconv_kernel);
        dconv_inputs.push(di);
    }

    // --- Rewrite inputs ---
    let rw_weight = b.add_input("rw_weight", &[doubled, cfg.out_channels, 1]);
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // --- Step 1: Conv1d (stride downsample) ---
    let conv_out_len = conv1d_out_len(
        cfg.spatial_in,
        cfg.conv_kernel,
        cfg.conv_stride,
        cfg.conv_padding,
    );
    let conv_out = b.add_conv1d(
        data,
        conv_weight,
        Some(conv_bias),
        cfg.conv_stride,
        cfg.conv_padding,
        &[cfg.out_channels, conv_out_len],
    );

    // --- Step 2: GELU ---
    let gelu_out = b.add_gelu(conv_out, &[cfg.out_channels, conv_out_len]);

    // --- Step 3: DConv residual sub-layers ---
    let mut dconv_out = gelu_out;
    for di in &dconv_inputs {
        dconv_out = build_dconv_sublayer(
            &mut b,
            dconv_out,
            di,
            cfg.out_channels,
            compressed,
            conv_out_len,
            cfg.dconv_kernel,
        );
    }

    // --- Step 4: Rewrite Conv1d(k=1) → GLU ---
    let rw_out = b.add_conv1d(
        dconv_out,
        rw_weight,
        Some(rw_bias),
        1,
        0,
        &[doubled, conv_out_len],
    );
    let output = b
        .add_glu(rw_out, 0, &[doubled, conv_out_len])
        .expect("even dim for GLU");

    (
        b.build(output).expect("valid encoder block graph"),
        conv_out_len,
        cfg.out_channels,
    )
}

/// Build parameter bindings for the encoder block.
///
/// data = Variable, all other inputs = ConstantTensor or ConstantScalar.
pub(super) fn encoder_block_bindings(cfg: &EncoderBlockConfig) -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    let compressed = cfg.out_channels / cfg.dconv_compress_ratio;
    let doubled = cfg.out_channels * 2;

    // data: Variable (the input we verify bounds over)
    bindings.push(TensorParamBinding::Variable);

    // Conv1d weight + bias (stride downsample)
    let conv_w = ArrayD::from_elem(
        IxDyn(&[cfg.out_channels, cfg.in_channels, cfg.conv_kernel]),
        0.01f32,
    );
    bindings.push(TensorParamBinding::ConstantTensor(conv_w));
    let conv_b = ArrayD::from_elem(IxDyn(&[cfg.out_channels]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(conv_b));

    // DConv sub-layers
    for _k in 0..cfg.dconv_depth {
        let cw = ArrayD::from_elem(
            IxDyn(&[compressed, cfg.out_channels, cfg.dconv_kernel]),
            0.01f32,
        );
        bindings.push(TensorParamBinding::ConstantTensor(cw));
        let cb = ArrayD::from_elem(IxDyn(&[compressed]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(cb));
        let ng = ArrayD::from_elem(IxDyn(&[compressed]), 1.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(ng));
        let nb = ArrayD::from_elem(IxDyn(&[compressed]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(nb));
        let ew = ArrayD::from_elem(IxDyn(&[doubled, compressed, 1]), 0.01f32);
        bindings.push(TensorParamBinding::ConstantTensor(ew));
        let eb = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(eb));
        let eng = ArrayD::from_elem(IxDyn(&[doubled]), 1.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(eng));
        let enb = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(enb));
        let ls = ArrayD::from_elem(IxDyn(&[cfg.out_channels]), 0.1f32);
        bindings.push(TensorParamBinding::ConstantTensor(ls));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Rewrite Conv1d weight + bias (k=1)
    let rw_w = ArrayD::from_elem(IxDyn(&[doubled, cfg.out_channels, 1]), 0.01f32);
    bindings.push(TensorParamBinding::ConstantTensor(rw_w));
    let rw_b = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(rw_b));

    bindings
}
