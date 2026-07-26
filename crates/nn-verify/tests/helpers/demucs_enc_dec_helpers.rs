// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// This file is a test helper module included by compose_demucs_encoder_decoder.rs
// via `mod demucs_enc_dec_helpers;`. The parent aggregator's #[allow(dead_code, unreachable_pub)]
// on the mod declaration suppresses warnings.

//! Helpers for Demucs temporal encoder → decoder composition tests.
//!
//! Builds a single `TensorKernelDef` containing both encoder and decoder stages.
//! The skip connection is constant-zero (single-variable mode).
//!
//! DConv sub-layer builder and bindings helpers extracted to
//! `demucs_enc_dec_helpers_dconv.rs` (#1402).
//!
//! Architecture (small-scale for tractability):
//! ```text
//! Input [8, 16]
//!   → Encoder: Conv1d(8→16, k=8, s=4, p=2) → GELU → DConv(×1) → Conv1d(k=1) → GLU → [16, 4]
//!   → Decoder: skip_add(0) → Conv1d(k=3) → GLU → DConv(×1) → ConvTranspose1d → GELU → [8, 8]
//! ```

// DConv sub-layer builder and bindings helpers extracted to separate file (#1402).
#[path = "demucs_enc_dec_helpers_dconv.rs"]
mod dconv;

use dconv::{build_dconv_sublayer, push_dconv_bindings, DConvInputs};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Encoder input channels.
pub(crate) const ENC_IN_CH: usize = 8;

/// Encoder output / decoder input channels (bottleneck).
pub(crate) const BOTTLENECK_CH: usize = 16;

/// Decoder output channels.
pub(crate) const DEC_OUT_CH: usize = 8;

/// Encoder temporal input length.
pub(crate) const T_IN: usize = 16;

/// Encoder Conv1d kernel size.
const ENC_CONV_K: usize = 8;

/// Encoder Conv1d stride.
const ENC_CONV_S: usize = 4;

/// Encoder Conv1d padding (kernel / 4).
const ENC_CONV_P: usize = ENC_CONV_K / 4;

/// DConv depth (number of residual sub-layers per block).
/// Reduced to 1: 2 encoder + 2 decoder DConv sub-layers with decomposed
/// GroupNorm + GLU causes IBP bounds to overflow (NaN/Inf). This is the known
/// IBP amplification issue with multi-op chains (design doc: decomposed norms).
/// Depth=1 exercises the full topology while keeping bounds tractable.
const DCONV_DEPTH: usize = 1;

/// Decoder rewrite Conv1d kernel size.
const DEC_RW_K: usize = 3;

/// Decoder rewrite Conv1d padding.
const DEC_RW_P: usize = DEC_RW_K / 2;

/// ConvTranspose1d kernel/stride/padding for decoder upsample.
const CT_K: usize = 4;
const CT_S: usize = 2;
const CT_P: usize = 1;

/// Compress ratio for DConv sub-layers.
const COMPRESS_RATIO: usize = 4;

// ---------------------------------------------------------------------------
// Shape arithmetic
// ---------------------------------------------------------------------------

use super::common::{conv1d_out_len, conv_transpose_out_len};

/// Bottleneck temporal length: encoder output T.
pub(crate) fn bottleneck_t() -> usize {
    conv1d_out_len(T_IN, ENC_CONV_K, ENC_CONV_S, ENC_CONV_P)
}

/// Decoder rewrite temporal length (same as bottleneck_t for k=3, p=1, s=1).
pub(crate) fn dec_rw_t() -> usize {
    conv1d_out_len(bottleneck_t(), DEC_RW_K, 1, DEC_RW_P)
}

/// Final output temporal length after ConvTranspose1d.
pub(crate) fn output_t() -> usize {
    conv_transpose_out_len(dec_rw_t(), CT_S, CT_K, CT_P)
}

// ---------------------------------------------------------------------------
// Encoder/decoder stage builders
// ---------------------------------------------------------------------------

struct EncoderNodes {
    data: TensorNodeId,
    conv_w: TensorNodeId,
    conv_b: TensorNodeId,
    dconv: Vec<DConvInputs>,
    rw_w: TensorNodeId,
    rw_b: TensorNodeId,
}

struct DecoderNodes {
    skip: TensorNodeId,
    rw_w: TensorNodeId,
    rw_b: TensorNodeId,
    dconv: Vec<DConvInputs>,
    ct_w: TensorNodeId,
    ct_b: TensorNodeId,
}

fn add_encoder_inputs(b: &mut TensorBlockBuilder) -> EncoderNodes {
    let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
    let doubled = BOTTLENECK_CH * 2;
    let data = b.add_input("data", &[ENC_IN_CH, T_IN]);
    let conv_w = b.add_input("enc_conv_w", &[BOTTLENECK_CH, ENC_IN_CH, ENC_CONV_K]);
    let conv_b = b.add_input("enc_conv_b", &[BOTTLENECK_CH]);
    let mut dconv = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        dconv.push(DConvInputs::add_to_builder(
            b,
            "enc",
            k,
            BOTTLENECK_CH,
            compressed,
        ));
    }
    let rw_w = b.add_input("enc_rw_w", &[doubled, BOTTLENECK_CH, 1]);
    let rw_b = b.add_input("enc_rw_b", &[doubled]);
    EncoderNodes {
        data,
        conv_w,
        conv_b,
        dconv,
        rw_w,
        rw_b,
    }
}

fn add_decoder_inputs(b: &mut TensorBlockBuilder, t_mid: usize) -> DecoderNodes {
    let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
    let doubled = BOTTLENECK_CH * 2;
    let skip = b.add_input("dec_skip", &[BOTTLENECK_CH, t_mid]);
    let rw_w = b.add_input("dec_rw_w", &[doubled, BOTTLENECK_CH, DEC_RW_K]);
    let rw_b = b.add_input("dec_rw_b", &[doubled]);
    let mut dconv = Vec::with_capacity(DCONV_DEPTH);
    for k in 0..DCONV_DEPTH {
        dconv.push(DConvInputs::add_to_builder(
            b,
            "dec",
            k,
            BOTTLENECK_CH,
            compressed,
        ));
    }
    let ct_w = b.add_input("dec_ct_w", &[BOTTLENECK_CH, DEC_OUT_CH, CT_K]);
    let ct_b = b.add_input("dec_ct_b", &[DEC_OUT_CH]);
    DecoderNodes {
        skip,
        rw_w,
        rw_b,
        dconv,
        ct_w,
        ct_b,
    }
}

/// Wire encoder: Conv1d → GELU → DConv → Rewrite(GLU). Returns encoder output.
fn wire_encoder(b: &mut TensorBlockBuilder, enc: &EncoderNodes) -> TensorNodeId {
    let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
    let doubled = BOTTLENECK_CH * 2;
    let t_mid = bottleneck_t();
    let conv = b.add_conv1d(
        enc.data,
        enc.conv_w,
        Some(enc.conv_b),
        ENC_CONV_S,
        ENC_CONV_P,
        &[BOTTLENECK_CH, t_mid],
    );
    let gelu = b.add_gelu(conv, &[BOTTLENECK_CH, t_mid]);
    let mut x = gelu;
    for di in &enc.dconv {
        x = build_dconv_sublayer(b, x, di, BOTTLENECK_CH, compressed, t_mid);
    }
    let rw = b.add_conv1d(x, enc.rw_w, Some(enc.rw_b), 1, 0, &[doubled, t_mid]);
    b.add_glu(rw, 0, &[doubled, t_mid])
        .expect("even dim for encoder GLU")
}

/// Wire decoder: skip_add → Rewrite(GLU) → DConv → ConvTranspose1d → GELU.
fn wire_decoder(
    b: &mut TensorBlockBuilder,
    dec: &DecoderNodes,
    enc_out: TensorNodeId,
) -> TensorNodeId {
    let compressed = BOTTLENECK_CH / COMPRESS_RATIO;
    let doubled = BOTTLENECK_CH * 2;
    let t_mid = bottleneck_t();
    let rw_t = dec_rw_t();
    let ct_t = output_t();
    let x = b.add_binary_add(enc_out, dec.skip, &[BOTTLENECK_CH, t_mid]);
    let rw = b.add_conv1d(x, dec.rw_w, Some(dec.rw_b), 1, DEC_RW_P, &[doubled, rw_t]);
    let glu = b
        .add_glu(rw, 0, &[doubled, rw_t])
        .expect("even dim for decoder GLU");
    let mut dc = glu;
    for di in &dec.dconv {
        dc = build_dconv_sublayer(b, dc, di, BOTTLENECK_CH, compressed, rw_t);
    }
    let ct = b.add_conv_transpose_1d(
        dc,
        dec.ct_w,
        Some(dec.ct_b),
        CT_S,
        CT_P,
        1, // dilation
        1, // groups
        0, // output_padding
        &[DEC_OUT_CH, ct_t],
    );
    b.add_gelu(ct, &[DEC_OUT_CH, ct_t])
}

// ---------------------------------------------------------------------------
// Public builders
// ---------------------------------------------------------------------------

/// Build the full encoder → decoder pipeline as a single TensorKernelDef.
///
/// Returns (def, final_output_t, final_output_ch).
pub(crate) fn build_encoder_decoder() -> (nn_dsl::tensor_ir::TensorKernelDef, usize, usize) {
    let t_mid = bottleneck_t();
    let mut b = TensorBlockBuilder::new("demucs_enc_dec_verify");
    let enc = add_encoder_inputs(&mut b);
    let dec = add_decoder_inputs(&mut b, t_mid);
    let enc_out = wire_encoder(&mut b, &enc);
    let output = wire_decoder(&mut b, &dec, enc_out);
    let ct_t = output_t();
    (
        b.build(output).expect("valid encoder-decoder graph"),
        ct_t,
        DEC_OUT_CH,
    )
}

/// Build parameter bindings for the encoder → decoder pipeline.
pub(crate) fn encoder_decoder_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    let enc_compressed = BOTTLENECK_CH / COMPRESS_RATIO;
    let enc_doubled = BOTTLENECK_CH * 2;
    let t_mid = bottleneck_t();

    // data: Variable
    bindings.push(TensorParamBinding::Variable);

    // Encoder Conv1d weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BOTTLENECK_CH, ENC_IN_CH, ENC_CONV_K]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BOTTLENECK_CH]),
        0.0f32,
    )));

    for _k in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut bindings, BOTTLENECK_CH, enc_compressed);
    }

    // Encoder rewrite weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[enc_doubled, BOTTLENECK_CH, 1]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[enc_doubled]),
        0.0f32,
    )));

    // Decoder skip: ConstantTensor(zeros)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BOTTLENECK_CH, t_mid]),
        0.0f32,
    )));

    // Decoder rewrite weight + bias
    let dec_doubled = BOTTLENECK_CH * 2;
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dec_doubled, BOTTLENECK_CH, DEC_RW_K]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dec_doubled]),
        0.0f32,
    )));

    // Decoder DConv
    let dec_compressed = BOTTLENECK_CH / COMPRESS_RATIO;
    for _k in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut bindings, BOTTLENECK_CH, dec_compressed);
    }

    // Decoder ConvTranspose1d weight + bias
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[BOTTLENECK_CH, DEC_OUT_CH, CT_K]),
        0.01f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DEC_OUT_CH]),
        0.0f32,
    )));

    bindings
}

/// Input bounds for the encoder input [ENC_IN_CH, T_IN].
pub(crate) fn input_bounds() -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(&[ENC_IN_CH, T_IN]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[ENC_IN_CH, T_IN]), 1.0f32);
    BoundedTensor::new(lower, upper).expect("valid input bounds")
}
