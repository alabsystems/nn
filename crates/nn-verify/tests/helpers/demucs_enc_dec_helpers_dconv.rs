// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DConv sub-layer builder and bindings helpers for Demucs encoder/decoder
//! composition tests.
//!
//! Extracted from `demucs_enc_dec_helpers.rs` for file-size compliance (#1402).

// This file is included as a submodule by demucs_enc_dec_helpers.rs.
// The parent aggregator's #[allow(dead_code, unreachable_pub)] on the mod
// declaration suppresses warnings.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// DConv kernel size (must match parent module's `DCONV_KERNEL`).
const DCONV_KERNEL: usize = 3;

// ---------------------------------------------------------------------------
// DConv sub-layer builder
// ---------------------------------------------------------------------------

pub(crate) struct DConvInputs {
    pub(crate) conv_compress_weight: TensorNodeId,
    pub(crate) conv_compress_bias: TensorNodeId,
    pub(crate) norm_compress_gamma: TensorNodeId,
    pub(crate) norm_compress_beta: TensorNodeId,
    pub(crate) conv_expand_weight: TensorNodeId,
    pub(crate) conv_expand_bias: TensorNodeId,
    pub(crate) norm_expand_gamma: TensorNodeId,
    pub(crate) norm_expand_beta: TensorNodeId,
    pub(crate) layer_scale: TensorNodeId,
    pub(crate) eps1: TensorNodeId,
    pub(crate) eps2: TensorNodeId,
    pub(crate) dilation: usize,
}

impl DConvInputs {
    pub(crate) fn add_to_builder(
        b: &mut TensorBlockBuilder,
        prefix: &str,
        k: usize,
        channels: usize,
        compressed: usize,
    ) -> Self {
        let doubled = channels * 2;
        Self {
            conv_compress_weight: b.add_input(
                &format!("{prefix}_dc{k}_cw"),
                &[compressed, channels, DCONV_KERNEL],
            ),
            conv_compress_bias: b.add_input(&format!("{prefix}_dc{k}_cb"), &[compressed]),
            norm_compress_gamma: b.add_input(&format!("{prefix}_dc{k}_ng"), &[compressed]),
            norm_compress_beta: b.add_input(&format!("{prefix}_dc{k}_nb"), &[compressed]),
            conv_expand_weight: b
                .add_input(&format!("{prefix}_dc{k}_ew"), &[doubled, compressed, 1]),
            conv_expand_bias: b.add_input(&format!("{prefix}_dc{k}_eb"), &[doubled]),
            norm_expand_gamma: b.add_input(&format!("{prefix}_dc{k}_eng"), &[doubled]),
            norm_expand_beta: b.add_input(&format!("{prefix}_dc{k}_enb"), &[doubled]),
            layer_scale: b.add_input(&format!("{prefix}_dc{k}_ls"), &[channels]),
            eps1: b.add_input(&format!("{prefix}_dc{k}_eps1"), &[1]),
            eps2: b.add_input(&format!("{prefix}_dc{k}_eps2"), &[1]),
            dilation: 1 << k,
        }
    }
}

pub(crate) fn build_dconv_sublayer(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    dc: &DConvInputs,
    channels: usize,
    compressed: usize,
    t_len: usize,
) -> TensorNodeId {
    let doubled = channels * 2;
    let dc_padding = dc.dilation * (DCONV_KERNEL - 1) / 2;

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
    let n1 = b.add_group_norm_g1(
        c1,
        dc.eps1,
        Some(dc.norm_compress_gamma),
        Some(dc.norm_compress_beta),
        compressed,
        t_len,
    );
    let g1 = b.add_gelu(n1, &[compressed, t_len]);
    let c2 = b.add_conv1d(
        g1,
        dc.conv_expand_weight,
        Some(dc.conv_expand_bias),
        1,
        0,
        &[doubled, t_len],
    );
    let n2 = b.add_group_norm_g1(
        c2,
        dc.eps2,
        Some(dc.norm_expand_gamma),
        Some(dc.norm_expand_beta),
        doubled,
        t_len,
    );
    let glu = b.add_glu(n2, 0, &[doubled, t_len]).expect("even dim");
    let ls = b.add_layer_scale(glu, dc.layer_scale, &[channels, t_len]);
    b.add_binary_add(input, ls, &[channels, t_len])
}

// ---------------------------------------------------------------------------
// DConv bindings helper
// ---------------------------------------------------------------------------

pub(crate) fn push_dconv_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    channels: usize,
    compressed: usize,
) {
    let doubled = channels * 2;
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed, channels, DCONV_KERNEL]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, compressed, 1]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[channels]),
        0.1f32,
    )));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
}
