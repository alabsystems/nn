// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for HTDemucs full-model NY composition.
//!
//! Temporal encoder + cross-domain transformer + temporal decoder as a
//! single `TensorKernelDef`. Spectral encoder output is constant for
//! single-variable NY tractability.
//!
//! Parameter binding helpers extracted to `htdemucs_full_bindings.rs`.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

use super::common::{conv1d_out_len, conv_transpose_out_len};

// Small dims for NY tractability.
pub(super) const IN_CH: usize = 4;
pub(super) const ENC_CH: usize = 8;
pub(super) const T_IN: usize = 16;
pub(super) const ENC_KERNEL: usize = 8;
const ENC_STRIDE: usize = 4;
const ENC_PADDING: usize = ENC_KERNEL / 4;
pub(super) const DCONV_COMPRESS_RATIO: usize = 4;
pub(super) const DCONV_KERNEL: usize = 3;
pub(super) const DCONV_DEPTH: usize = 1;
pub(super) const DEC_REWRITE_KERNEL: usize = 3;
const DEC_REWRITE_PADDING: usize = DEC_REWRITE_KERNEL / 2;
pub(super) const CT_KERNEL: usize = 8;
const CT_STRIDE: usize = 4;
const CT_PADDING: usize = ENC_PADDING;
pub(super) const WEIGHT_MAG: f32 = 0.001;
pub(super) const MODEL_DIM: usize = ENC_CH;
const NUM_HEADS: usize = 2;
pub(super) const FFN_HIDDEN: usize = MODEL_DIM * 2;
pub(super) const F_SEQ: usize = 4;

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
// Cross-attention weights (manual decomposition)
// ---------------------------------------------------------------------------

struct CrossAttnWeights {
    ln1_weight: TensorNodeId,
    ln1_bias: TensorNodeId,
    ln3_weight: TensorNodeId,
    ln3_bias: TensorNodeId,
    ln_out_weight: TensorNodeId,
    ln_out_bias: TensorNodeId,
    q_weight: TensorNodeId,
    k_weight: TensorNodeId,
    v_weight: TensorNodeId,
    out_weight: TensorNodeId,
    ffn1_weight: TensorNodeId,
    ffn2_weight: TensorNodeId,
    eps: TensorNodeId,
}

// ---------------------------------------------------------------------------
// Full pipeline inputs
// ---------------------------------------------------------------------------

struct FullHTDemucsInputs {
    audio: TensorNodeId,
    spectral_kv: TensorNodeId,
    ecw: TensorNodeId,
    ecb: TensorNodeId,
    enc_dc: Vec<DConvInputs>,
    erw: TensorNodeId,
    erb: TensorNodeId,
    // Cross-domain transformer
    t_up_w: TensorNodeId,
    t_up_b: TensorNodeId,
    t_down_w: TensorNodeId,
    t_down_b: TensorNodeId,
    t_self_attn: TransformerBlockWeights,
    t_cross_attn: CrossAttnWeights,
    // Decoder
    drw: TensorNodeId,
    drb: TensorNodeId,
    dec_dc: Vec<DConvInputs>,
    dctw: TensorNodeId,
    dctb: TensorNodeId,
}

fn add_cross_attn_weights(
    b: &mut TensorBlockBuilder,
    prefix: &str,
    eps: TensorNodeId,
) -> CrossAttnWeights {
    let d = MODEL_DIM;
    CrossAttnWeights {
        ln1_weight: b.add_input(&format!("{prefix}_ca_ln1_w"), &[d]),
        ln1_bias: b.add_input(&format!("{prefix}_ca_ln1_b"), &[d]),
        ln3_weight: b.add_input(&format!("{prefix}_ca_ln3_w"), &[d]),
        ln3_bias: b.add_input(&format!("{prefix}_ca_ln3_b"), &[d]),
        ln_out_weight: b.add_input(&format!("{prefix}_ca_lnout_w"), &[d]),
        ln_out_bias: b.add_input(&format!("{prefix}_ca_lnout_b"), &[d]),
        q_weight: b.add_input(&format!("{prefix}_ca_q_w"), &[d, d]),
        k_weight: b.add_input(&format!("{prefix}_ca_k_w"), &[d, d]),
        v_weight: b.add_input(&format!("{prefix}_ca_v_w"), &[d, d]),
        out_weight: b.add_input(&format!("{prefix}_ca_out_w"), &[d, d]),
        ffn1_weight: b.add_input(&format!("{prefix}_ca_ffn1_w"), &[FFN_HIDDEN, d]),
        ffn2_weight: b.add_input(&format!("{prefix}_ca_ffn2_w"), &[d, FFN_HIDDEN]),
        eps,
    }
}

fn add_all_inputs(b: &mut TensorBlockBuilder) -> FullHTDemucsInputs {
    let comp = ENC_CH / DCONV_COMPRESS_RATIO;
    let dbl = ENC_CH * 2;
    let d = MODEL_DIM;

    let audio = b.add_input("audio", &[IN_CH, T_IN]);
    let spectral_kv = b.add_input("spectral_kv", &[F_SEQ, d]);

    // Encoder weights
    let ecw = b.add_input("enc_conv_w", &[ENC_CH, IN_CH, ENC_KERNEL]);
    let ecb = b.add_input("enc_conv_b", &[ENC_CH]);
    let enc_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(b, "enc", k, ENC_CH, comp))
        .collect();
    let erw = b.add_input("enc_rw_w", &[dbl, ENC_CH, 1]);
    let erb = b.add_input("enc_rw_b", &[dbl]);

    // Cross-domain transformer weights
    let t_up_w = b.add_input("t_up_w", &[d, ENC_CH, 1]);
    let t_up_b = b.add_input("t_up_b", &[d]);
    let t_down_w = b.add_input("t_down_w", &[ENC_CH, d, 1]);
    let t_down_b = b.add_input("t_down_b", &[ENC_CH]);
    let eps = b.add_input("eps", &[1]);
    let t_self_attn = {
        TransformerBlockWeights {
            ln1_weight: b.add_input("tf_sa_ln1_w", &[d]),
            ln1_bias: b.add_input("tf_sa_ln1_b", &[d]),
            ln2_weight: b.add_input("tf_sa_ln2_w", &[d]),
            ln2_bias: b.add_input("tf_sa_ln2_b", &[d]),
            q_weight: b.add_input("tf_sa_q_w", &[d, d]),
            k_weight: b.add_input("tf_sa_k_w", &[d, d]),
            v_weight: b.add_input("tf_sa_v_w", &[d, d]),
            out_weight: b.add_input("tf_sa_out_w", &[d, d]),
            ffn1_weight: b.add_input("tf_sa_ffn1_w", &[FFN_HIDDEN, d]),
            ffn2_weight: b.add_input("tf_sa_ffn2_w", &[d, FFN_HIDDEN]),
            eps,
        }
    };
    let t_cross_attn = add_cross_attn_weights(b, "tf", eps);

    // Decoder weights
    let drw = b.add_input("dec_rw_w", &[dbl, ENC_CH, DEC_REWRITE_KERNEL]);
    let drb = b.add_input("dec_rw_b", &[dbl]);
    let dec_dc: Vec<_> = (0..DCONV_DEPTH)
        .map(|k| DConvInputs::add(b, "dec", k, ENC_CH, comp))
        .collect();
    let dctw = b.add_input("dec_ct_w", &[ENC_CH, IN_CH, CT_KERNEL]);
    let dctb = b.add_input("dec_ct_b", &[IN_CH]);

    FullHTDemucsInputs {
        audio,
        spectral_kv,
        ecw,
        ecb,
        enc_dc,
        erw,
        erb,
        t_up_w,
        t_up_b,
        t_down_w,
        t_down_b,
        t_self_attn,
        t_cross_attn,
        drw,
        drb,
        dec_dc,
        dctw,
        dctb,
    }
}

// ---------------------------------------------------------------------------
// Build full model graph
// ---------------------------------------------------------------------------

fn build_cross_attention(
    b: &mut TensorBlockBuilder,
    temporal: TensorNodeId,
    spectral_kv: TensorNodeId,
    ca: &CrossAttnWeights,
    t_seq: usize,
) -> TensorNodeId {
    let d = MODEL_DIM;
    let shape = [t_seq, d];
    let ffn_shape = [t_seq, FFN_HIDDEN];

    // LN1 → Q from temporal, K/V from spectral constant
    let normed_q = b.add_layer_norm(temporal, ca.eps, 1, ca.ln1_weight, ca.ln1_bias, &shape);
    let attn = b
        .add_multi_head_cross_attention(
            normed_q,
            spectral_kv,
            ca.q_weight,
            ca.k_weight,
            ca.v_weight,
            ca.out_weight,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("cross-MHA");
    let residual1 = b.add_binary_add(temporal, attn, &shape);

    // LN3 → FFN → Residual
    let normed3 = b.add_layer_norm(residual1, ca.eps, 1, ca.ln3_weight, ca.ln3_bias, &shape);
    let ffn1 = b.add_linear(normed3, ca.ffn1_weight, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ca.ffn2_weight, None, &shape);
    let residual2 = b.add_binary_add(residual1, ffn2, &shape);

    // LN_out
    b.add_layer_norm(
        residual2,
        ca.eps,
        1,
        ca.ln_out_weight,
        ca.ln_out_bias,
        &shape,
    )
}

/// Build the full HTDemucs model as a single `TensorKernelDef`.
///
/// Returns `(def, output_temporal_length)`.
pub(super) fn build_htdemucs_full() -> (TensorKernelDef, usize) {
    let comp = ENC_CH / DCONV_COMPRESS_RATIO;
    let dbl = ENC_CH * 2;
    let d = MODEL_DIM;
    let mut b = TensorBlockBuilder::new("htdemucs_full_verify");
    let inp = add_all_inputs(&mut b);

    // === Temporal Encoder ===
    let t_enc = conv1d_out_len(T_IN, ENC_KERNEL, ENC_STRIDE, ENC_PADDING);
    let x = b.add_conv1d(
        inp.audio,
        inp.ecw,
        Some(inp.ecb),
        ENC_STRIDE,
        ENC_PADDING,
        &[ENC_CH, t_enc],
    );
    let x = b.add_gelu(x, &[ENC_CH, t_enc]);
    let mut x = x;
    for di in &inp.enc_dc {
        x = build_dconv(&mut b, x, di, ENC_CH, comp, t_enc);
    }
    let x = b.add_conv1d(x, inp.erw, Some(inp.erb), 1, 0, &[dbl, t_enc]);
    let enc_out = b.add_glu(x, 0, &[dbl, t_enc]).expect("encoder GLU");

    // === Cross-domain Transformer Bottleneck ===
    // Channel bridge up: [ENC_CH, T] → Conv1d(1×1) → [D, T]
    let t_up = b.add_conv1d(enc_out, inp.t_up_w, Some(inp.t_up_b), 1, 0, &[d, t_enc]);

    // Transpose [D, T] → [T, D]
    let t_td = b.add_transpose(t_up, &[1, 0], &[t_enc, d]);

    // Self-attention
    let tc = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_HIDDEN,
    };
    let t_self = b
        .add_transformer_block(t_td, &inp.t_self_attn, &tc)
        .expect("temporal self-attention");

    // Cross-attention: temporal queries spectral (constant KV)
    let t_cross = build_cross_attention(&mut b, t_self, inp.spectral_kv, &inp.t_cross_attn, t_enc);

    // Channel bridge down: Transpose [T, D] → [D, T] → Conv1d(D→C) → [C, T]
    let t_dt = b.add_transpose(t_cross, &[1, 0], &[d, t_enc]);
    let x = b.add_conv1d(
        t_dt,
        inp.t_down_w,
        Some(inp.t_down_b),
        1,
        0,
        &[ENC_CH, t_enc],
    );

    // === Temporal Decoder ===
    // Skip connection from encoder
    let x = b.add_binary_add(x, enc_out, &[ENC_CH, t_enc]);

    // Rewrite Conv1d
    let rw_t = conv1d_out_len(t_enc, DEC_REWRITE_KERNEL, 1, DEC_REWRITE_PADDING);
    let x = b.add_conv1d(
        x,
        inp.drw,
        Some(inp.drb),
        1,
        DEC_REWRITE_PADDING,
        &[dbl, rw_t],
    );
    let x = b.add_glu(x, 0, &[dbl, rw_t]).expect("decoder GLU");

    // DConv
    let mut x = x;
    for di in &inp.dec_dc {
        x = build_dconv(&mut b, x, di, ENC_CH, comp, rw_t);
    }

    // ConvTranspose1d upsample
    let ct_t = conv_transpose_out_len(rw_t, CT_STRIDE, CT_KERNEL, CT_PADDING);
    let x = b.add_conv_transpose_1d(
        x,
        inp.dctw,
        Some(inp.dctb),
        CT_STRIDE,
        CT_PADDING,
        1, // dilation
        1, // groups
        0, // output_padding
        &[IN_CH, ct_t],
    );

    // Trim to match input length
    let target_t = T_IN.min(ct_t);
    let x = if ct_t > target_t {
        b.add_narrow(x, 1, 0, target_t, &[IN_CH, target_t])
    } else {
        x
    };
    let out = b.add_gelu(x, &[IN_CH, target_t]);

    (b.build(out).expect("valid htdemucs full graph"), target_t)
}

// ---------------------------------------------------------------------------
// Bindings (extracted to htdemucs_full_bindings.rs for 500-line compliance)
// ---------------------------------------------------------------------------

#[path = "htdemucs_full_bindings.rs"]
mod bindings;
pub(super) use bindings::htdemucs_full_bindings;
