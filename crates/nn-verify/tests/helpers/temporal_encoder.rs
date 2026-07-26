// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Demucs temporal encoder block composition tests.
//!
//! Extracted from `compose_demucs_temporal_encoder.rs` to keep the parent
//! under 500 lines (#1669).

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

use super::common::conv1d_out_len;

// ---------------------------------------------------------------------------
// Small-scale encoder block parameters
// ---------------------------------------------------------------------------

/// Input channels (small-scale, matching AUDIO_CHANNELS=2 scaled down).
pub(super) const IN_CHANNELS: usize = 8;

/// Output channels (after Conv1d downsample).
pub(super) const OUT_CHANNELS: usize = 16;

/// Temporal input length (must be multiple of stride for clean Conv1d output).
pub(super) const T_IN: usize = 16;

/// Conv1d kernel size (matching htdemucs KERNEL_SIZE=8).
const CONV_KERNEL: usize = 8;

/// Conv1d stride (matching htdemucs STRIDE=4).
const CONV_STRIDE: usize = 4;

/// Conv1d padding (kernel_size / 4, matching htdemucs).
const CONV_PADDING: usize = CONV_KERNEL / 4;

/// DConv compressed channels (out_channels / compress_ratio).
const DCONV_COMPRESS_RATIO: usize = 4;

/// DConv kernel size.
const DCONV_KERNEL: usize = 3;

/// Number of DConv sub-layers per block.
const DCONV_DEPTH: usize = 2;

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
    ) -> Self {
        let doubled = channels * 2;
        Self {
            conv_compress_weight: b
                .add_input(&format!("dc{k}_cw"), &[compressed, channels, DCONV_KERNEL]),
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
    t_len: usize,
) -> TensorNodeId {
    let doubled = channels * 2;
    let dc_padding = dc.dilation * (DCONV_KERNEL - 1) / 2;

    // Dilated Conv1d: [channels, T] → [compressed, T]
    let c1 = b.add_conv1d_full(
        input,
        dc.conv_compress_weight,
        Some(dc.conv_compress_bias),
        1,
        dc_padding,
        dc.dilation,
        1,
        &[compressed, t_len],
    );

    // GroupNorm(G=1) on compressed channels
    let n1 = b.add_group_norm_g1(
        c1,
        dc.eps1,
        Some(dc.norm_compress_gamma),
        Some(dc.norm_compress_beta),
        compressed,
        t_len,
    );

    // GELU
    let g1 = b.add_gelu(n1, &[compressed, t_len]);

    // Conv1d expand: [compressed, T] → [channels*2, T]
    let c2 = b.add_conv1d(
        g1,
        dc.conv_expand_weight,
        Some(dc.conv_expand_bias),
        1,
        0,
        &[doubled, t_len],
    );

    // GroupNorm(G=1) on expanded channels
    let n2 = b.add_group_norm_g1(
        c2,
        dc.eps2,
        Some(dc.norm_expand_gamma),
        Some(dc.norm_expand_beta),
        doubled,
        t_len,
    );

    // GLU: [channels*2, T] → [channels, T]
    let glu = b.add_glu(n2, 0, &[doubled, t_len]).expect("even dim");

    // LayerScale: broadcast [channels] → [channels, T], multiply
    let ls = b.add_layer_scale(glu, dc.layer_scale, &[channels, t_len]);

    // Residual: input + scaled
    b.add_binary_add(input, ls, &[channels, t_len])
}

/// Build a single Demucs encoder block using TensorBlockBuilder.
///
/// Input layout:
///   - "data" [IN_CHANNELS, T_IN] — Variable (verified input)
///   - Remaining: weight/bias constants for Conv1d, DConv, Rewrite
///
/// Returns (TensorKernelDef, output temporal length, output channels).
pub(super) fn build_encoder_block() -> (nn_dsl::tensor_ir::TensorKernelDef, usize, usize) {
    let compressed = OUT_CHANNELS / DCONV_COMPRESS_RATIO;
    let doubled = OUT_CHANNELS * 2;

    let mut b = TensorBlockBuilder::new("demucs_enc_block_verify");

    // --- Variable input ---
    let data = b.add_input("data", &[IN_CHANNELS, T_IN]);

    // --- Conv1d inputs ---
    let conv_weight = b.add_input("conv_weight", &[OUT_CHANNELS, IN_CHANNELS, CONV_KERNEL]);
    let conv_bias = b.add_input("conv_bias", &[OUT_CHANNELS]);

    // --- DConv inputs (2 sub-layers) ---
    let mut dconv_inputs = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        let di = DConvInputs::add_to_builder(&mut b, k, OUT_CHANNELS, compressed);
        dconv_inputs.push(di);
    }

    // --- Rewrite inputs ---
    let rw_weight = b.add_input("rw_weight", &[doubled, OUT_CHANNELS, 1]);
    let rw_bias = b.add_input("rw_bias", &[doubled]);

    // --- Step 1: Conv1d (stride downsample) ---
    let conv_t_out = conv1d_out_len(T_IN, CONV_KERNEL, CONV_STRIDE, CONV_PADDING);
    let conv_out = b.add_conv1d(
        data,
        conv_weight,
        Some(conv_bias),
        CONV_STRIDE,
        CONV_PADDING,
        &[OUT_CHANNELS, conv_t_out],
    );

    // --- Step 2: GELU (between Conv1d and DConv, matching Python) ---
    let gelu_out = b.add_gelu(conv_out, &[OUT_CHANNELS, conv_t_out]);

    // --- Step 3: DConv (2 residual sub-layers) ---
    let mut dconv_out = gelu_out;
    for di in &dconv_inputs {
        dconv_out =
            build_dconv_sublayer(&mut b, dconv_out, di, OUT_CHANNELS, compressed, conv_t_out);
    }

    // --- Step 4: Rewrite Conv1d(k=1) → GLU ---
    let rw_out = b.add_conv1d(
        dconv_out,
        rw_weight,
        Some(rw_bias),
        1,
        0,
        &[doubled, conv_t_out],
    );
    let output = b
        .add_glu(rw_out, 0, &[doubled, conv_t_out])
        .expect("even dim for GLU");

    (
        b.build(output).expect("valid encoder block graph"),
        conv_t_out,
        OUT_CHANNELS,
    )
}

/// Build parameter bindings for the encoder block.
///
/// data = Variable, all other inputs = ConstantTensor or ConstantScalar.
pub(super) fn encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    let compressed = OUT_CHANNELS / DCONV_COMPRESS_RATIO;
    let doubled = OUT_CHANNELS * 2;

    // data: Variable (the input we verify bounds over)
    bindings.push(TensorParamBinding::Variable);

    // Conv1d weight + bias (stride downsample)
    let conv_w = ArrayD::from_elem(IxDyn(&[OUT_CHANNELS, IN_CHANNELS, CONV_KERNEL]), 0.01f32);
    bindings.push(TensorParamBinding::ConstantTensor(conv_w));
    let conv_b = ArrayD::from_elem(IxDyn(&[OUT_CHANNELS]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(conv_b));

    // DConv sub-layers (2)
    for _k in 0..DCONV_DEPTH {
        // conv_compress_weight
        let cw = ArrayD::from_elem(IxDyn(&[compressed, OUT_CHANNELS, DCONV_KERNEL]), 0.01f32);
        bindings.push(TensorParamBinding::ConstantTensor(cw));
        // conv_compress_bias
        let cb = ArrayD::from_elem(IxDyn(&[compressed]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(cb));
        // norm_compress_gamma
        let ng = ArrayD::from_elem(IxDyn(&[compressed]), 1.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(ng));
        // norm_compress_beta
        let nb = ArrayD::from_elem(IxDyn(&[compressed]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(nb));
        // conv_expand_weight
        let ew = ArrayD::from_elem(IxDyn(&[doubled, compressed, 1]), 0.01f32);
        bindings.push(TensorParamBinding::ConstantTensor(ew));
        // conv_expand_bias
        let eb = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(eb));
        // norm_expand_gamma
        let eng = ArrayD::from_elem(IxDyn(&[doubled]), 1.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(eng));
        // norm_expand_beta
        let enb = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
        bindings.push(TensorParamBinding::ConstantTensor(enb));
        // layer_scale
        let ls = ArrayD::from_elem(IxDyn(&[OUT_CHANNELS]), 0.1f32);
        bindings.push(TensorParamBinding::ConstantTensor(ls));
        // eps1
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        // eps2
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    }

    // Rewrite Conv1d weight + bias (k=1)
    let rw_w = ArrayD::from_elem(IxDyn(&[doubled, OUT_CHANNELS, 1]), 0.01f32);
    bindings.push(TensorParamBinding::ConstantTensor(rw_w));
    let rw_b = ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32);
    bindings.push(TensorParamBinding::ConstantTensor(rw_b));

    bindings
}
