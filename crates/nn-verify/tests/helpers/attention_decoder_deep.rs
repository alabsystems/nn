// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 28 helpers: attention → deep decoder pipeline with variable ResBlocks.
//!
//! Extends Phase 27 with configurable ResBlock count in the decoder.
//! Part of #1729: Attention Monotonicity Proofs — Phase 28.

#![allow(dead_code, clippy::duplicated_attributes)]

use super::common::bounds_min_max;
use super::common::decoder_common::{
    self, AttnLayerIds, D_K, FFN_DIM, UPSAMPLE_KERNEL, UPSAMPLE_PADDING, UPSAMPLE_STRIDE,
    WEIGHT_MAG,
};
use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};

// Re-export items used by test files via `attn_decoder_deep::`.
pub(super) use super::common::decoder_common::{D_MODEL, NUM_HEADS, T_DEC, T_ENC};

// Unique constants for deep decoder (not shared across all 7 files).
pub(super) const DECODER_CHANNELS: usize = 4;
pub(super) const UPSAMPLED_CHANNELS: usize = 4;
pub(super) const OUT_CHANNELS: usize = 4;
pub(super) const TIME_UP: usize =
    (T_DEC - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;
const RESBLOCK_KERNEL: usize = 3;
const RESBLOCK_PADDING: usize = 1;

// Convenience aliases for shared functions.
use decoder_common::{causal_mask, encoder_k, near_identity, scaled_diag, sin_pe, uniform};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct ResBlockIds {
    gamma: TensorNodeId,
    beta: TensorNodeId,
    alpha: TensorNodeId,
    conv_w: TensorNodeId,
}

// ---------------------------------------------------------------------------
// Pipeline builder
// ---------------------------------------------------------------------------

/// Build attention → bridge → deep-decoder pipeline.
///
/// `na`: attention layers (≥2). `nr`: decoder ResBlocks (≥1).
pub(super) fn build_deep_decoder_pipeline(na: usize, nr: usize) -> (TensorKernelDef, [usize; 2]) {
    assert!(na >= 2 && nr >= 1);

    let scale = 1.0 / (D_K as f32).sqrt();
    let ss = [NUM_HEADS, T_DEC, T_ENC];
    let cs = [NUM_HEADS, T_DEC, D_K];
    let up = [UPSAMPLED_CHANNELS, TIME_UP];

    let mut b = TensorBlockBuilder::new("deep_decoder_pipeline");

    let hidden = b.add_input("hidden", &[T_DEC, D_MODEL]);
    let dec_pe = b.add_input("dec_pe", &[T_DEC, D_MODEL]);
    let enc_k = b.add_input("enc_k", &[T_ENC, D_MODEL]);
    let enc_v = b.add_input("enc_v", &[T_ENC, D_MODEL]);

    let mut attn_ids = Vec::with_capacity(na);
    for i in 0..na {
        let s = format!("_L{i}");
        attn_ids.push(AttnLayerIds {
            w_q: b.add_input(&format!("w_q{s}"), &[D_MODEL, D_MODEL]),
            w_k: b.add_input(&format!("w_k{s}"), &[D_MODEL, D_MODEL]),
            w_v: b.add_input(&format!("w_v{s}"), &[D_MODEL, D_MODEL]),
            w_o: b.add_input(&format!("w_o{s}"), &[D_MODEL, D_MODEL]),
            mask: b.add_input(&format!("mask{s}"), &[T_DEC, T_ENC]),
            ln_w: b.add_input(&format!("ln_w{s}"), &[D_MODEL]),
            ln_b: b.add_input(&format!("ln_b{s}"), &[D_MODEL]),
            ln_eps: b.add_input(&format!("ln_eps{s}"), &[1]),
            ffn_up: b.add_input(&format!("ffn_up{s}"), &[FFN_DIM, D_MODEL]),
            ffn_down: b.add_input(&format!("ffn_down{s}"), &[D_MODEL, FFN_DIM]),
        });
    }

    let bridge_w = b.add_input("bridge_w", &[D_MODEL, DECODER_CHANNELS]);
    let dec_eps = b.add_input("dec_eps", &[1]);
    let cpre_w = b.add_input("cpre_w", &[DECODER_CHANNELS, DECODER_CHANNELS, 7]);
    let ups_w = b.add_input(
        "ups_w",
        &[DECODER_CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL],
    );

    let mut rb_ids = Vec::with_capacity(nr);
    for r in 0..nr {
        let s = format!("_R{r}");
        rb_ids.push(ResBlockIds {
            gamma: b.add_input(&format!("g{s}"), &[UPSAMPLED_CHANNELS]),
            beta: b.add_input(&format!("b{s}"), &[UPSAMPLED_CHANNELS]),
            alpha: b.add_input(&format!("a{s}"), &[1]),
            conv_w: b.add_input(
                &format!("c{s}"),
                &[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL],
            ),
        });
    }

    let cpost_w = b.add_input("cpost_w", &[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7]);

    // === Attention Stack ===
    let mut prev = b.add_binary_add(hidden, dec_pe, &[T_DEC, D_MODEL]);
    let mut final_ctx: Option<TensorNodeId> = None;

    for (li, a) in attn_ids.iter().enumerate() {
        let last = li == na - 1;
        let dm = [T_DEC, D_MODEL];
        let em = [T_ENC, D_MODEL];

        let q = b.add_matmul(prev, a.w_q, false, None, &dm);
        let k = b.add_matmul(enc_k, a.w_k, false, None, &em);
        let qr = b.add_reshape(q, &[T_DEC, NUM_HEADS, D_K]);
        let kr = b.add_reshape(k, &[T_ENC, NUM_HEADS, D_K]);
        let qt = b.add_transpose(qr, &[1, 0, 2], &[NUM_HEADS, T_DEC, D_K]);
        let kt = b.add_transpose(kr, &[1, 0, 2], &[NUM_HEADS, T_ENC, D_K]);

        let sc = b.add_matmul(qt, kt, true, Some(scale), &ss);
        let mbc = b.add_broadcast(a.mask, &ss);
        let ma = b.add_binary_add(sc, mbc, &ss);
        let w = b.add_softmax(ma, -1, &ss);

        let v = b.add_matmul(enc_v, a.w_v, false, None, &em);
        let vr = b.add_reshape(v, &[T_ENC, NUM_HEADS, D_K]);
        let vt = b.add_transpose(vr, &[1, 0, 2], &[NUM_HEADS, T_ENC, D_K]);

        let ctx = b.add_matmul(w, vt, false, None, &cs);
        let ct = b.add_transpose(ctx, &[1, 0, 2], &[T_DEC, NUM_HEADS, D_K]);
        let cf = b.add_reshape(ct, &dm);

        if last {
            final_ctx = Some(cf);
            break;
        }

        let ao = b.add_matmul(cf, a.w_o, false, None, &dm);
        let r = b.add_binary_add(prev, ao, &dm);
        let n = b.add_layer_norm(r, a.ln_eps, 1, a.ln_w, a.ln_b, &dm);
        let f1 = b.add_linear(n, a.ffn_up, None, &[T_DEC, FFN_DIM]);
        let fa = b.add_gelu(f1, &[T_DEC, FFN_DIM]);
        let f2 = b.add_linear(fa, a.ffn_down, None, &dm);
        prev = b.add_binary_add(r, f2, &dm);
    }

    let cf = final_ctx.expect("at least 2 layers");

    // === Context Bridge ===
    let br = b.add_matmul(cf, bridge_w, false, None, &[T_DEC, DECODER_CHANNELS]);
    let di = b.add_transpose(br, &[1, 0], &[DECODER_CHANNELS, T_DEC]);

    // === Deep Decoder ===
    let x = b.add_conv1d(di, cpre_w, None, 1, 3, &[DECODER_CHANNELS, T_DEC]);
    let xa = b.add_leaky_relu(x, 0.1, &[DECODER_CHANNELS, T_DEC]);
    let mut xu = b.add_conv_transpose_1d(
        xa,
        ups_w,
        None,
        UPSAMPLE_STRIDE,
        UPSAMPLE_PADDING,
        1,
        1,
        0, // output_padding
        &up,
    );

    let sk = build_snake_scalar_kernel().expect("snake kernel");
    for ri in &rb_ids {
        let n = b.add_instance_norm(xu, dec_eps, 1, Some(ri.gamma), Some(ri.beta), &up);
        let abc = b.add_broadcast(ri.alpha, &up);
        let sn = b.add_elementwise(sk.clone(), &[n, abc], &up);
        let cv = b.add_conv1d(sn, ri.conv_w, None, 1, RESBLOCK_PADDING, &up);
        xu = b.add_binary_add(xu, cv, &up);
    }

    let ra = b.add_leaky_relu(xu, 0.01, &up);
    let xp = b.add_conv1d(ra, cpost_w, None, 1, 3, &[OUT_CHANNELS, TIME_UP]);
    let out = b.add_exp(xp, &[OUT_CHANNELS, TIME_UP]);

    (
        b.build(out).expect("valid deep decoder pipeline"),
        [OUT_CHANNELS, TIME_UP],
    )
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Build bindings for the deep decoder pipeline.
pub(super) fn deep_decoder_bindings(
    na: usize,
    nr: usize,
    pe_scale: f32,
    w_pert: f32,
) -> Vec<TensorParamBinding> {
    let mut pe = sin_pe(T_DEC, D_MODEL, NUM_HEADS);
    pe.mapv_inplace(|v| v * pe_scale);
    let ek = encoder_k(T_ENC, D_MODEL);
    let ev = encoder_k(T_ENC, D_MODEL);
    let wp = near_identity(D_MODEL, w_pert);
    let mk = causal_mask(T_DEC, T_ENC);
    let lw = uniform(&[D_MODEL], 1.0);
    let lb = uniform(&[D_MODEL], 0.0);
    let fu = scaled_diag(FFN_DIM, D_MODEL, 0.1);
    let fd = scaled_diag(D_MODEL, FFN_DIM, 0.1);

    let mut b = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
        TensorParamBinding::ConstantTensor(ek),
        TensorParamBinding::ConstantTensor(ev),
    ];

    for _ in 0..na {
        b.push(TensorParamBinding::ConstantTensor(wp.clone()));
        b.push(TensorParamBinding::ConstantTensor(wp.clone()));
        b.push(TensorParamBinding::ConstantTensor(wp.clone()));
        b.push(TensorParamBinding::ConstantTensor(wp.clone()));
        b.push(TensorParamBinding::ConstantTensor(mk.clone()));
        b.push(TensorParamBinding::ConstantTensor(lw.clone()));
        b.push(TensorParamBinding::ConstantTensor(lb.clone()));
        b.push(TensorParamBinding::ConstantScalar(1e-5));
        b.push(TensorParamBinding::ConstantTensor(fu.clone()));
        b.push(TensorParamBinding::ConstantTensor(fd.clone()));
    }

    // Bridge
    b.push(TensorParamBinding::ConstantTensor(scaled_diag(
        D_MODEL,
        DECODER_CHANNELS,
        0.1,
    )));
    // dec_eps
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    // conv_pre
    b.push(TensorParamBinding::ConstantTensor(uniform(
        &[DECODER_CHANNELS, DECODER_CHANNELS, 7],
        WEIGHT_MAG,
    )));
    // upsample
    b.push(TensorParamBinding::ConstantTensor(uniform(
        &[DECODER_CHANNELS, UPSAMPLED_CHANNELS, UPSAMPLE_KERNEL],
        WEIGHT_MAG,
    )));

    for _ in 0..nr {
        b.push(TensorParamBinding::ConstantTensor(uniform(
            &[UPSAMPLED_CHANNELS],
            1.0,
        )));
        b.push(TensorParamBinding::ConstantTensor(uniform(
            &[UPSAMPLED_CHANNELS],
            0.0,
        )));
        b.push(TensorParamBinding::ConstantScalar(1.0));
        b.push(TensorParamBinding::ConstantTensor(uniform(
            &[UPSAMPLED_CHANNELS, UPSAMPLED_CHANNELS, RESBLOCK_KERNEL],
            WEIGHT_MAG,
        )));
    }

    // conv_post
    b.push(TensorParamBinding::ConstantTensor(uniform(
        &[OUT_CHANNELS, UPSAMPLED_CHANNELS, 7],
        WEIGHT_MAG,
    )));

    b
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct DeepDecoderResult {
    pub(super) num_attn_layers: usize,
    pub(super) num_resblocks: usize,
    pub(super) graph_nodes: usize,
    pub(super) min_output_lo: f32,
    pub(super) max_output_hi: f32,
    pub(super) avg_bound_width: f32,
    pub(super) all_positive: bool,
    pub(super) all_finite: bool,
}

pub(super) fn analyze_deep_decoder_output(
    output: &BoundedTensor,
    na: usize,
    nr: usize,
    nodes: usize,
) -> DeepDecoderResult {
    let (lo, hi) = output.lower_upper();
    let fl: Vec<f32> = lo.iter().copied().collect();
    let fh: Vec<f32> = hi.iter().copied().collect();

    let (min_lo, max_hi) = bounds_min_max(output);
    let avg_w: f32 = fl.iter().zip(fh.iter()).map(|(&l, &h)| h - l).sum::<f32>() / fl.len() as f32;

    DeepDecoderResult {
        num_attn_layers: na,
        num_resblocks: nr,
        graph_nodes: nodes,
        min_output_lo: min_lo,
        max_output_hi: max_hi,
        avg_bound_width: avg_w,
        all_positive: fl.iter().all(|&v| v >= 0.0),
        all_finite: fl.iter().chain(fh.iter()).all(|v| v.is_finite()),
    }
}
