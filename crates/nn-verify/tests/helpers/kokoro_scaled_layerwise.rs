// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Per-layer decomposition builders for Kokoro pipeline.
//!
//! Unlike the monolithic pipeline in `kokoro_scaled_pipeline.rs`, these builders
//! decompose the Kokoro architecture into individual layers, each with its own
//! `TensorKernelDef`. This enables `verify_layerwise` (#1762) to apply per-layer
//! CROWN propagation, achieving tighter bounds than monolithic IBP at D=128+.
//!
//! Architecture decomposition:
//! ```text
//!   Layer 0: TextEncoder — Conv1d + ReLU + Linear    [d_model, seq_len] → [enc_dim, seq_len]
//!   Layer 1: VocoderPre — Conv1d + LeakyReLU         [enc_dim, seq_len] → [voc_ch, seq_len]
//!   Layer 2: VocoderUpsample — ConvTranspose1d       [voc_ch, seq_len]  → [vup_ch, time_up]
//!   Layer 3: VocoderResBlock — InstNorm+Snake+Conv1d  [vup_ch, time_up]  → [vup_ch, time_up]
//!   Layer 4: VocoderOutput — LeakyReLU+Conv1d+Clamp+Exp [vup_ch, time_up] → [out_ch, time_up]
//! ```
//!
//! Part of #1741: THE MOONSHOT — per-layer CROWN scaling to D=128+.

use super::helpers::KokoroDims;
use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

/// A single layer in the decomposed Kokoro pipeline.
pub(super) type LayerSpec = (TensorKernelDef, Vec<TensorParamBinding>);

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.001;

/// Build text encoder as a single layer: Conv1d + ReLU + Linear.
///
/// Input: `[d_model, seq_len]` (Variable) → Output: `[enc_dim, seq_len]`
pub(super) fn build_layer_text_encoder(dims: &KokoroDims) -> LayerSpec {
    let d = dims.d_model;
    let enc = dims.enc_dim;
    let s = dims.seq_len;

    let mut b = TensorBlockBuilder::new("layer_text_encoder");
    let input = b.add_input("text_features", &[d, s]);

    // Conv1d: [d_model, seq_len] → [d_model, seq_len]
    let conv_w = b.add_input("enc_conv_w", &[d, d, 3]);
    let conv_out = b.add_conv1d(input, conv_w, None, 1, 1, &[d, s]);

    // ReLU
    let relu_out = b.add_relu(conv_out, &[d, s]);

    // Linear: [d_model, seq_len] → [enc_dim, seq_len]
    let transposed = b.add_transpose(relu_out, &[1, 0], &[s, d]);
    let proj_w = b.add_input("enc_proj_w", &[enc, d]);
    let proj_b = b.add_input("enc_proj_b", &[enc]);
    let projected = b.add_matmul(transposed, proj_w, true, None, &[s, enc]);
    let proj_b_bc = b.add_broadcast(proj_b, &[s, enc]);
    let biased = b.add_binary_add(projected, proj_b_bc, &[s, enc]);
    let output = b.add_transpose(biased, &[1, 0], &[enc, s]);
    let def = b.build(output).expect("valid text encoder layer");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, d, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[enc, d]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[enc]), 0.0f32)),
    ];
    (def, bindings)
}

/// Build vocoder pre-conv + LeakyReLU as a single layer.
///
/// Input: `[enc_dim, seq_len]` (Variable) → Output: `[voc_channels, seq_len]`
pub(super) fn build_layer_vocoder_pre(dims: &KokoroDims) -> LayerSpec {
    let enc = dims.enc_dim;
    let vc = dims.voc_channels;
    let s = dims.seq_len;

    let mut b = TensorBlockBuilder::new("layer_vocoder_pre");
    let input = b.add_input("encoded", &[enc, s]);
    let conv_w = b.add_input("voc_conv_pre_w", &[vc, enc, 3]);
    let x = b.add_conv1d(input, conv_w, None, 1, 1, &[vc, s]);
    let output = b.add_leaky_relu(x, 0.1, &[vc, s]);
    let def = b.build(output).expect("valid vocoder pre layer");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[vc, enc, 3]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

/// Build vocoder upsample (ConvTranspose1d) as a single layer.
///
/// Input: `[voc_channels, seq_len]` (Variable) → Output: `[voc_up_channels, time_up]`
pub(super) fn build_layer_vocoder_upsample(dims: &KokoroDims) -> LayerSpec {
    let vc = dims.voc_channels;
    let vup = dims.voc_up_channels;
    let s = dims.seq_len;
    let t = dims.time_up();

    let mut b = TensorBlockBuilder::new("layer_vocoder_upsample");
    let input = b.add_input("pre_output", &[vc, s]);
    let upsample_w = b.add_input("voc_upsample_w", &[vc, vup, dims.upsample_kernel]);
    let output = b.add_conv_transpose_1d(
        input,
        upsample_w,
        None,
        dims.upsample_stride,
        dims.upsample_padding(),
        1,
        1,
        0, // output_padding
        &[vup, t],
    );
    let def = b.build(output).expect("valid vocoder upsample layer");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[vc, vup, dims.upsample_kernel]),
            WEIGHT_MAG,
        )),
    ];
    (def, bindings)
}

/// Build vocoder ResBlock (InstanceNorm + Snake + Conv1d + residual) as a single layer.
///
/// Input: `[voc_up_channels, time_up]` (Variable) → Output: `[voc_up_channels, time_up]`
pub(super) fn build_layer_vocoder_resblock(dims: &KokoroDims) -> LayerSpec {
    build_layer_vocoder_resblock_n(dims, 0)
}

/// Build the `n`-th vocoder ResBlock with unique naming for multi-block pipelines.
///
/// Each block has 1 InstanceNorm + 1 Snake + 1 Conv1d + residual.
/// Input: `[voc_up_channels, time_up]` (Variable) → Output: `[voc_up_channels, time_up]`
pub(super) fn build_layer_vocoder_resblock_n(dims: &KokoroDims, n: usize) -> LayerSpec {
    let vup = dims.voc_up_channels;
    let t = dims.time_up();
    let shape = [vup, t];

    let mut b = TensorBlockBuilder::new(&format!("layer_vocoder_resblock_{n}"));
    let input = b.add_input(&format!("resblock_{n}_input"), &shape);

    // InstanceNorm
    let eps = b.add_input(&format!("voc_eps_{n}"), &[1]);
    let style_gamma = b.add_input(&format!("voc_style_gamma_{n}"), &[vup]);
    let style_beta = b.add_input(&format!("voc_style_beta_{n}"), &[vup]);
    let normed = b.add_instance_norm(input, eps, 1, Some(style_gamma), Some(style_beta), &shape);

    // Snake activation
    let alpha = b.add_input(&format!("voc_alpha_{n}"), &[1]);
    let alpha_bc = b.add_broadcast(alpha, &shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &shape);

    // Conv1d in ResBlock
    let res_conv_w = b.add_input(&format!("voc_res_conv_w_{n}"), &[vup, vup, 3]);
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &shape);

    // Residual connection
    let output = b.add_binary_add(input, sublayer_out, &shape);
    let def = b.build(output).expect("valid vocoder resblock layer");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[vup]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[vup]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[vup, vup, 3]), WEIGHT_MAG)),
    ];
    (def, bindings)
}

/// Build a 2-input scalar max kernel: `fn max(a, b) -> a.max(b)`.
fn scalar_max_kernel() -> KernelDef {
    KernelDef::new(
        "scalar_max",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Build a 2-input scalar min kernel: `fn min(a, b) -> a.min(b)`.
fn scalar_min_kernel() -> KernelDef {
    KernelDef::new(
        "scalar_min",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Build vocoder output (LeakyReLU + Conv1d + Clamp + Exp) as a single layer.
///
/// Input: `[voc_up_channels, time_up]` (Variable) → Output: `[out_channels, time_up]`
///
/// The clamp to [-88, 88] matches production `kokoro_decoder.rs:279` where
/// `log_mag_clamped = log_mag.clamp(-LOG_MAG_CLAMP_MAX, LOG_MAG_CLAMP_MAX)`.
/// Without this, ForwardMode bounds through deep ResBlock chains can exceed the
/// Exp overflow threshold, causing verification failures (#2625).
pub(super) fn build_layer_vocoder_output(dims: &KokoroDims) -> LayerSpec {
    let vup = dims.voc_up_channels;
    let out = dims.out_channels;
    let t = dims.time_up();

    let mut b = TensorBlockBuilder::new("layer_vocoder_output");
    let input = b.add_input("resblock_output", &[vup, t]);
    let x_act = b.add_leaky_relu(input, 0.01, &[vup, t]);
    let conv_post_w = b.add_input("voc_conv_post_w", &[out, vup, 3]);
    let x_post = b.add_conv1d(x_act, conv_post_w, None, 1, 1, &[out, t]);

    // Clamp to [-88, 88] before exp (matches production kokoro_decoder.rs:279).
    let clamp_lo = b.add_input("clamp_lo", &[1]);
    let clamp_lo_bc = b.add_broadcast(clamp_lo, &[out, t]);
    let clamped_lo = b.add_elementwise(scalar_max_kernel(), &[x_post, clamp_lo_bc], &[out, t]);
    let clamp_hi = b.add_input("clamp_hi", &[1]);
    let clamp_hi_bc = b.add_broadcast(clamp_hi, &[out, t]);
    let clamped = b.add_elementwise(scalar_min_kernel(), &[clamped_lo, clamp_hi_bc], &[out, t]);

    let output = b.add_exp(clamped, &[out, t]);
    let def = b.build(output).expect("valid vocoder output layer");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out, vup, 3]), WEIGHT_MAG)),
        TensorParamBinding::ConstantScalar(-88.0),
        TensorParamBinding::ConstantScalar(88.0),
    ];
    (def, bindings)
}

/// Build the full Kokoro pipeline as a sequence of layers for verify_layerwise.
///
/// Returns 5 layers: text_encoder → vocoder_pre → vocoder_upsample → resblock → output.
pub(super) fn build_kokoro_layerwise(dims: &KokoroDims) -> Vec<LayerSpec> {
    vec![
        build_layer_text_encoder(dims),
        build_layer_vocoder_pre(dims),
        build_layer_vocoder_upsample(dims),
        build_layer_vocoder_resblock(dims),
        build_layer_vocoder_output(dims),
    ]
}

/// Build a production-depth Kokoro pipeline with `num_resblocks` ResBlock layers.
///
/// Each ResBlock contains 1 InstanceNorm, giving `num_resblocks` normalization
/// layers total. Production Kokoro has ~48 InstanceNorm in the vocoder alone
/// (2 upsample stages × 4 ResBlocks × 3 dilations × 2 norms). Using
/// `num_resblocks=12` gives a representative normalization depth for testing
/// how CROWN bounds compound through many normalization layers.
///
/// Pipeline: text_encoder → vocoder_pre → vocoder_upsample → N×resblock → output.
///
/// Part of #2573: production-representative normalization depth.
pub(super) fn build_kokoro_layerwise_deep(
    dims: &KokoroDims,
    num_resblocks: usize,
) -> Vec<LayerSpec> {
    let mut layers = Vec::with_capacity(4 + num_resblocks);
    layers.push(build_layer_text_encoder(dims));
    layers.push(build_layer_vocoder_pre(dims));
    layers.push(build_layer_vocoder_upsample(dims));
    for i in 0..num_resblocks {
        layers.push(build_layer_vocoder_resblock_n(dims, i));
    }
    layers.push(build_layer_vocoder_output(dims));
    layers
}
