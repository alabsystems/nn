// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 29 helpers: multi-stage upsample decoder composition.
//!
//! Extends Phase 28's single-stage decoder to multiple upsample stages,
//! each with ConvTranspose1d + N ResBlocks. Real Kokoro ISTFTNet uses
//! 3 stages (each 2x upsample), producing 8x temporal resolution.
//!
//! Architecture: attention_stack → bridge → [upsample + N×ResBlock] × S → output
//!
//! Key difference from Phase 28:
//! - Phase 28: 1 upsample + variable ResBlocks
//! - Phase 29: S upsample stages, each with R ResBlocks (S×R total)
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 29.

#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};

use super::common::bounds_min_max;

// Shared decoder dimensions and helpers — delegated to super::common (Part of #1970).
pub(super) use super::common::decoder_common::{
    causal_mask, channels_at_stage, encoder_k, near_identity, scaled_diag, sin_pe,
    time_after_stages, uniform, AttnLayerIds, D_K, D_MODEL, FFN_DIM, NUM_HEADS, OUT_KERNEL,
    OUT_PADDING, T_DEC, T_ENC, UPSAMPLE_KERNEL, UPSAMPLE_PADDING, UPSAMPLE_STRIDE, WEIGHT_MAG,
};

const RESBLOCK_KERNEL: usize = 3;
const RESBLOCK_PADDING: usize = 1;

struct StageIds {
    ups_w: TensorNodeId,
    resblocks: Vec<ResBlockIds>,
}

struct ResBlockIds {
    gamma: TensorNodeId,
    beta: TensorNodeId,
    alpha: TensorNodeId,
    conv_w: TensorNodeId,
}

// -------------------------------------------------------------------------
// Multi-stage pipeline builder
// -------------------------------------------------------------------------

/// Build attention → bridge → multi-stage decoder pipeline.
///
/// `na`: attention layers (≥2).
/// `ns`: upsample stages (1-3).
/// `nr`: ResBlocks per stage (1-3).
///
/// Returns (kernel_def, output_shape).
pub(super) fn build_multi_stage_pipeline(
    na: usize,
    ns: usize,
    nr: usize,
) -> (TensorKernelDef, [usize; 2]) {
    assert!(na >= 2 && (1..=3).contains(&ns) && nr >= 1);

    let scale = 1.0 / (D_K as f32).sqrt();
    let ss = [NUM_HEADS, T_DEC, T_ENC];
    let cs = [NUM_HEADS, T_DEC, D_K];

    let mut b = TensorBlockBuilder::new("multi_stage_pipeline");

    // --- Inputs ---
    let hidden = b.add_input("hidden", &[T_DEC, D_MODEL]);
    let dec_pe = b.add_input("dec_pe", &[T_DEC, D_MODEL]);
    let enc_k = b.add_input("enc_k", &[T_ENC, D_MODEL]);
    let enc_v = b.add_input("enc_v", &[T_ENC, D_MODEL]);

    // --- Attention layer parameters ---
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

    // --- Bridge ---
    let bridge_ch = channels_at_stage(0);
    let bridge_w = b.add_input("bridge_w", &[D_MODEL, bridge_ch]);

    // --- Per-stage decoder parameters ---
    let dec_eps = b.add_input("dec_eps", &[1]);

    let mut stage_ids = Vec::with_capacity(ns);
    for si in 0..ns {
        let in_ch = channels_at_stage(si);
        let out_ch = channels_at_stage(si + 1);
        let s = format!("_S{si}");

        let ups_w = b.add_input(&format!("ups_w{s}"), &[in_ch, out_ch, UPSAMPLE_KERNEL]);

        let mut resblocks = Vec::with_capacity(nr);
        for ri in 0..nr {
            let rs = format!("{s}_R{ri}");
            resblocks.push(ResBlockIds {
                gamma: b.add_input(&format!("g{rs}"), &[out_ch]),
                beta: b.add_input(&format!("b{rs}"), &[out_ch]),
                alpha: b.add_input(&format!("a{rs}"), &[1]),
                conv_w: b.add_input(&format!("c{rs}"), &[out_ch, out_ch, RESBLOCK_KERNEL]),
            });
        }

        stage_ids.push(StageIds { ups_w, resblocks });
    }

    // --- Output conv ---
    let final_ch = channels_at_stage(ns);
    let final_t = time_after_stages(ns);
    let out_w = b.add_input("out_w", &[final_ch, final_ch, OUT_KERNEL]);

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
    let br = b.add_matmul(cf, bridge_w, false, None, &[T_DEC, bridge_ch]);
    let di = b.add_transpose(br, &[1, 0], &[bridge_ch, T_DEC]);

    // === Multi-Stage Decoder ===
    let sk = build_snake_scalar_kernel().expect("snake kernel");
    let mut cur = di;
    let mut cur_ch = bridge_ch;
    let mut cur_t = T_DEC;

    for (si, stage) in stage_ids.iter().enumerate() {
        let out_ch = channels_at_stage(si + 1);
        let out_t = (cur_t - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;
        let up_shape = [out_ch, out_t];

        // Upsample via ConvTranspose1d
        let activated = b.add_leaky_relu(cur, 0.1, &[cur_ch, cur_t]);
        cur = b.add_conv_transpose_1d(
            activated,
            stage.ups_w,
            None,
            UPSAMPLE_STRIDE,
            UPSAMPLE_PADDING,
            1, // dilation
            1, // groups
            0, // output_padding
            &up_shape,
        );

        // ResBlocks
        for rb in &stage.resblocks {
            let n = b.add_instance_norm(cur, dec_eps, 1, Some(rb.gamma), Some(rb.beta), &up_shape);
            let abc = b.add_broadcast(rb.alpha, &up_shape);
            let sn = b.add_elementwise(sk.clone(), &[n, abc], &up_shape);
            let cv = b.add_conv1d(sn, rb.conv_w, None, 1, RESBLOCK_PADDING, &up_shape);
            cur = b.add_binary_add(cur, cv, &up_shape);
        }

        cur_ch = out_ch;
        cur_t = out_t;
    }

    // === Output ===
    let ra = b.add_leaky_relu(cur, 0.01, &[final_ch, final_t]);
    let xp = b.add_conv1d(ra, out_w, None, 1, OUT_PADDING, &[final_ch, final_t]);
    let out = b.add_exp(xp, &[final_ch, final_t]);

    (
        b.build(out).expect("valid multi-stage pipeline"),
        [final_ch, final_t],
    )
}

// -------------------------------------------------------------------------
// Bindings
// -------------------------------------------------------------------------

/// Build bindings for the multi-stage pipeline.
pub(super) fn multi_stage_bindings(
    na: usize,
    ns: usize,
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

    let mut bindings = vec![
        TensorParamBinding::Variable,           // hidden
        TensorParamBinding::ConstantTensor(pe), // dec_pe
        TensorParamBinding::ConstantTensor(ek), // enc_k
        TensorParamBinding::ConstantTensor(ev), // enc_v
    ];

    // Attention layers
    for _ in 0..na {
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone())); // w_q
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone())); // w_k
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone())); // w_v
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone())); // w_o
        bindings.push(TensorParamBinding::ConstantTensor(mk.clone())); // mask
        bindings.push(TensorParamBinding::ConstantTensor(lw.clone())); // ln_w
        bindings.push(TensorParamBinding::ConstantTensor(lb.clone())); // ln_b
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln_eps
        bindings.push(TensorParamBinding::ConstantTensor(fu.clone())); // ffn_up
        bindings.push(TensorParamBinding::ConstantTensor(fd.clone())); // ffn_down
    }

    // Bridge
    let bridge_ch = channels_at_stage(0);
    bindings.push(TensorParamBinding::ConstantTensor(scaled_diag(
        D_MODEL, bridge_ch, 0.1,
    )));

    // dec_eps
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // Per-stage decoder parameters
    for si in 0..ns {
        let in_ch = channels_at_stage(si);
        let out_ch = channels_at_stage(si + 1);

        // upsample weight
        bindings.push(TensorParamBinding::ConstantTensor(uniform(
            &[in_ch, out_ch, UPSAMPLE_KERNEL],
            WEIGHT_MAG,
        )));

        // ResBlock parameters
        for _ in 0..nr {
            bindings.push(TensorParamBinding::ConstantTensor(uniform(&[out_ch], 1.0))); // gamma
            bindings.push(TensorParamBinding::ConstantTensor(uniform(&[out_ch], 0.0))); // beta
            bindings.push(TensorParamBinding::ConstantScalar(1.0)); // alpha
            bindings.push(TensorParamBinding::ConstantTensor(uniform(
                &[out_ch, out_ch, RESBLOCK_KERNEL],
                WEIGHT_MAG,
            ))); // conv
        }
    }

    // Output conv
    let final_ch = channels_at_stage(ns);
    bindings.push(TensorParamBinding::ConstantTensor(uniform(
        &[final_ch, final_ch, OUT_KERNEL],
        WEIGHT_MAG,
    )));

    bindings
}

// -------------------------------------------------------------------------
// Analysis
// -------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct MultiStageResult {
    pub(super) num_attn_layers: usize,
    pub(super) num_stages: usize,
    pub(super) resblocks_per_stage: usize,
    pub(super) graph_nodes: usize,
    pub(super) output_channels: usize,
    pub(super) output_time: usize,
    pub(super) min_output_lo: f32,
    pub(super) max_output_hi: f32,
    pub(super) avg_bound_width: f32,
    pub(super) all_positive: bool,
    pub(super) all_finite: bool,
}

pub(super) fn analyze_multi_stage_output(
    output: &BoundedTensor,
    na: usize,
    ns: usize,
    nr: usize,
    nodes: usize,
) -> MultiStageResult {
    let (lo, hi) = output.lower_upper();
    let fl: Vec<f32> = lo.iter().copied().collect();
    let fh: Vec<f32> = hi.iter().copied().collect();

    let (min_lo, max_hi) = bounds_min_max(output);
    let n = fl.len().max(1) as f32;
    let avg_w: f32 = fl.iter().zip(fh.iter()).map(|(&l, &h)| h - l).sum::<f32>() / n;

    let final_ch = channels_at_stage(ns);
    let final_t = time_after_stages(ns);

    MultiStageResult {
        num_attn_layers: na,
        num_stages: ns,
        resblocks_per_stage: nr,
        graph_nodes: nodes,
        output_channels: final_ch,
        output_time: final_t,
        min_output_lo: min_lo,
        max_output_hi: max_hi,
        avg_bound_width: avg_w,
        all_positive: fl.iter().all(|&v| v >= 0.0),
        all_finite: fl.iter().chain(fh.iter()).all(|v| v.is_finite()),
    }
}
