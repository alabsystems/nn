// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Demucs spectral full decoder block composition tests.
//!
//! Extracted from `compose_demucs_spectral_full.rs` for 500-line compliance.
//! Contains constants, struct definitions, and pipeline builder functions
//! used by the spectral full decoder integration tests.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

use super::common::conv1d_out_len;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Input channels to the spectral encoder.
pub(super) const IN_CH: usize = 4;
/// Encoder output / decoder input channels.
const ENC_CH: usize = 8;
/// Frequency dimension.
const FREQ: usize = 4;
/// Time dimension.
const TIME: usize = 4;
/// Flattened spatial dimension.
pub(super) const FT: usize = FREQ * TIME;
/// Encoder Conv1d kernel (stride along F*T flattened dimension).
const ENC_KERNEL: usize = 8;
/// Encoder stride.
const ENC_STRIDE: usize = 4;
/// Encoder padding.
const ENC_PADDING: usize = ENC_KERNEL / 4;
/// Conv2d kernel for spectral rewrite (3×3).
const REWRITE_KERNEL: usize = 3;
/// Conv2d padding (kernel/2).
const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;
/// DConv compression ratio.
const DCONV_COMPRESS_RATIO: usize = 4;
/// DConv kernel size.
const DCONV_KERNEL: usize = 3;
/// DConv depth (1 sub-layer for tractability).
const DCONV_DEPTH: usize = 1;
/// ConvTranspose kernel.
const CT_KERNEL: usize = 8;
/// ConvTranspose stride.
const CT_STRIDE: usize = 4;
/// ConvTranspose padding.
const CT_PADDING: usize = ENC_PADDING;
/// Weight magnitude (small to keep IBP tractable).
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// DConv sub-layer (same structure as temporal)
// ---------------------------------------------------------------------------

struct DConvInputs {
    cw: TensorNodeId,
    cb: TensorNodeId,
    ng: TensorNodeId,
    nb: TensorNodeId,
    ew: TensorNodeId,
    eb: TensorNodeId,
    eng: TensorNodeId,
    enb: TensorNodeId,
    ls: TensorNodeId,
    eps1: TensorNodeId,
    eps2: TensorNodeId,
    dilation: usize,
}

impl DConvInputs {
    fn add(b: &mut TensorBlockBuilder, pfx: &str, k: usize, ch: usize, comp: usize) -> Self {
        let d = ch * 2;
        Self {
            cw: b.add_input(&format!("{pfx}_dc{k}_cw"), &[comp, ch, DCONV_KERNEL]),
            cb: b.add_input(&format!("{pfx}_dc{k}_cb"), &[comp]),
            ng: b.add_input(&format!("{pfx}_dc{k}_ng"), &[comp]),
            nb: b.add_input(&format!("{pfx}_dc{k}_nb"), &[comp]),
            ew: b.add_input(&format!("{pfx}_dc{k}_ew"), &[d, comp, 1]),
            eb: b.add_input(&format!("{pfx}_dc{k}_eb"), &[d]),
            eng: b.add_input(&format!("{pfx}_dc{k}_eng"), &[d]),
            enb: b.add_input(&format!("{pfx}_dc{k}_enb"), &[d]),
            ls: b.add_input(&format!("{pfx}_dc{k}_ls"), &[ch]),
            eps1: b.add_input(&format!("{pfx}_dc{k}_eps"), &[1]),
            eps2: b.add_input(&format!("{pfx}_dc{k}_eps2"), &[1]),
            dilation: 1 << k,
        }
    }
}

/// Conv1d(dilated) → GN(G=1) → GELU → Conv1d(1×1) → GN(G=1) → GLU → LS → residual.
fn build_dconv(
    b: &mut TensorBlockBuilder,
    x: TensorNodeId,
    dc: &DConvInputs,
    ch: usize,
    comp: usize,
    t: usize,
) -> TensorNodeId {
    let d = ch * 2;
    let pad = dc.dilation * (DCONV_KERNEL - 1) / 2;
    let c1 = b.add_conv1d_full(x, dc.cw, Some(dc.cb), 1, pad, dc.dilation, 1, &[comp, t]);
    let n1 = b.add_group_norm_g1(c1, dc.eps1, Some(dc.ng), Some(dc.nb), comp, t);
    let g1 = b.add_gelu(n1, &[comp, t]);
    let c2 = b.add_conv1d(g1, dc.ew, Some(dc.eb), 1, 0, &[d, t]);
    let n2 = b.add_group_norm_g1(c2, dc.eps2, Some(dc.eng), Some(dc.enb), d, t);
    let glu = b.add_glu(n2, 0, &[d, t]).expect("even dim");
    let ls = b.add_layer_scale(glu, dc.ls, &[ch, t]);
    b.add_binary_add(x, ls, &[ch, t])
}

// ---------------------------------------------------------------------------
// Full spectral decoder pipeline builder
// ---------------------------------------------------------------------------

/// All input node IDs for the spectral decoder pipeline.
struct SpectralPipelineInputs {
    /// Spectral data input [IN_CH, F*T].
    data: TensorNodeId,
    /// Encoder Conv1d weight [ENC_CH, IN_CH, ENC_KERNEL].
    ecw: TensorNodeId,
    /// Encoder Conv1d bias [ENC_CH].
    ecb: TensorNodeId,
    /// Rewrite Conv2d weight [ENC_CH*2, ENC_CH, 3, 3].
    rw_w: TensorNodeId,
    /// Rewrite Conv2d bias [ENC_CH*2].
    rw_b: TensorNodeId,
    /// DConv sub-layer inputs.
    dconv: Vec<DConvInputs>,
    /// ConvTranspose weight [ENC_CH, IN_CH, CT_KERNEL].
    ct_w: TensorNodeId,
    /// ConvTranspose bias [IN_CH].
    ct_b: TensorNodeId,
}

fn add_spectral_inputs(b: &mut TensorBlockBuilder) -> SpectralPipelineInputs {
    let dbl = ENC_CH * 2;
    let comp = ENC_CH / DCONV_COMPRESS_RATIO;

    let data = b.add_input("data", &[IN_CH, FT]);
    let ecw = b.add_input("enc_conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let ecb = b.add_input("enc_conv_b", &[ENC_CH]);
    let rw_w = b.add_input("rw_weight", &[dbl, ENC_CH, REWRITE_KERNEL, REWRITE_KERNEL]);
    let rw_b = b.add_input("rw_bias", &[dbl]);
    let dconv: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(b, "spec_dec", k, ENC_CH, comp))
        .collect();
    let ct_w = b.add_input("ct_weight", &[ENC_CH, IN_CH, CT_KERNEL]);
    let ct_b = b.add_input("ct_bias", &[IN_CH]);

    SpectralPipelineInputs {
        data,
        ecw,
        ecb,
        rw_w,
        rw_b,
        dconv,
        ct_w,
        ct_b,
    }
}

/// Wire the spectral rewrite stage: skip_add → Reshape → Conv2d → Reshape → GLU.
///
/// Input: encoder output [ENC_CH, t_enc]. Uses enc_out twice (data + skip).
/// Returns (rewrite_out node, flattened output length).
fn wire_rewrite(
    b: &mut TensorBlockBuilder,
    enc_out: TensorNodeId,
    t_enc: usize,
    rw_w: TensorNodeId,
    rw_b: TensorNodeId,
) -> (TensorNodeId, usize) {
    let dbl = ENC_CH * 2;
    // Spatial dims for Conv2d: height=t_enc, width=1 (degenerate).
    // Conv2d(3×3, p=1) on [C, t_enc, 1]: out_h = t_enc, out_w = 1.
    let rw_h = t_enc;
    let rw_w_dim = 1;
    let rw_out_h = rw_h + 2 * REWRITE_PADDING - REWRITE_KERNEL + 1;
    let rw_out_w = rw_w_dim;

    let enc_3d = b.add_reshape(enc_out, &[ENC_CH, rw_h, rw_w_dim]);
    let skip_3d = b.add_reshape(enc_out, &[ENC_CH, rw_h, rw_w_dim]);
    let x = b.add_binary_add(enc_3d, skip_3d, &[ENC_CH, rw_h, rw_w_dim]);

    let conv_out = b.add_conv2d(
        x,
        rw_w,
        Some(rw_b),
        1,
        1,
        REWRITE_PADDING,
        REWRITE_PADDING,
        &[dbl, rw_out_h, rw_out_w],
    );

    let rw_flat = rw_out_h * rw_out_w;
    let conv_flat = b.add_reshape(conv_out, &[dbl, rw_flat]);
    let out = b
        .add_glu(conv_flat, 0, &[dbl, rw_flat])
        .expect("even channels for GLU");
    (out, rw_flat)
}

/// Wire ConvTranspose1d → Narrow(trim) → GELU. Returns (output node, target_len).
fn wire_conv_transpose(
    b: &mut TensorBlockBuilder,
    x: TensorNodeId,
    ct_w: TensorNodeId,
    ct_b: TensorNodeId,
    in_len: usize,
) -> (TensorNodeId, usize) {
    let ct_out_len =
        super::common::conv_transpose_out_len(in_len, CT_STRIDE, CT_KERNEL, CT_PADDING);
    let ct_out = b.add_conv_transpose_1d(
        x,
        ct_w,
        Some(ct_b),
        CT_STRIDE,
        CT_PADDING,
        1, // dilation
        1, // groups
        0, // output_padding
        &[IN_CH, ct_out_len],
    );
    let target_len = FT.min(ct_out_len);
    let trimmed = if ct_out_len > target_len {
        b.add_narrow(ct_out, 1, 0, target_len, &[IN_CH, target_len])
    } else {
        ct_out
    };
    let out = b.add_gelu(trimmed, &[IN_CH, target_len]);
    (out, target_len)
}

/// Build the full spectral decoder pipeline.
///
/// Returns (TensorKernelDef, output_len) where output_len is the final
/// trailing dimension after ConvTranspose + trim.
pub(super) fn build_spectral_full() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
    let comp = ENC_CH / DCONV_COMPRESS_RATIO;
    let mut b = TensorBlockBuilder::new("demucs_spectral_full_verify");
    let inp = add_spectral_inputs(&mut b);

    // Encoder: Conv1d(stride) → GELU on flattened [IN_CH, F*T].
    let t_enc = conv1d_out_len(FT, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
    let x = b.add_conv1d(
        inp.data,
        inp.ecw,
        Some(inp.ecb),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t_enc],
    );
    let enc_out = b.add_gelu(x, &[ENC_CH, t_enc]);

    // Rewrite: skip_add → Reshape → Conv2d(3×3) → Reshape → GLU.
    let (rewrite_out, rw_flat) = wire_rewrite(&mut b, enc_out, t_enc, inp.rw_w, inp.rw_b);

    // DConv: operates on [ENC_CH, rw_flat].
    let mut x = rewrite_out;
    for di in &inp.dconv {
        x = build_dconv(&mut b, x, di, ENC_CH, comp, rw_flat);
    }

    // ConvTranspose1d → Narrow(trim) → GELU.
    let (out, target_len) = wire_conv_transpose(&mut b, x, inp.ct_w, inp.ct_b, rw_flat);

    (b.build(out).expect("valid spectral full graph"), target_len)
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

fn add_dconv_bindings(b: &mut Vec<TensorParamBinding>, ch: usize, comp: usize) {
    let d = ch * 2;
    push_weight(b, &[comp, ch, DCONV_KERNEL], WEIGHT_MAG);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[comp], 1.0);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[d, comp, 1], WEIGHT_MAG);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[ch], 0.1);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b.push(TensorParamBinding::ConstantScalar(1e-5));
}

pub(super) fn spectral_full_bindings() -> Vec<TensorParamBinding> {
    let dbl = ENC_CH * 2;
    let comp = ENC_CH / DCONV_COMPRESS_RATIO;
    let mut b = Vec::new();

    // data: Variable
    b.push(TensorParamBinding::Variable);
    // Encoder Conv1d weight + bias
    push_weight(&mut b, &[ENC_CH, IN_CH, ENC_KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[ENC_CH], 0.0);
    // Rewrite Conv2d weight [2C, C, 3, 3] + bias [2C]
    push_weight(
        &mut b,
        &[dbl, ENC_CH, REWRITE_KERNEL, REWRITE_KERNEL],
        WEIGHT_MAG,
    );
    push_weight(&mut b, &[dbl], 0.0);
    // DConv sub-layers
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut b, ENC_CH, comp);
    }
    // ConvTranspose weight + bias
    push_weight(&mut b, &[ENC_CH, IN_CH, CT_KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[IN_CH], 0.0);

    b
}
