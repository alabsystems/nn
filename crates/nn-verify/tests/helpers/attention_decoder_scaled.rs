// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 34 helpers: Scaled composition proofs for the Kokoro ISTFTNet decoder.
//!
//! Extends Phase 33's output projection pipeline with parameterized dimensions
//! (`ScaledPipelineConfig`) so the full pipeline can be verified at D=16, D=32,
//! and D=64 — approaching production Kokoro dimensions.
//!
//! Production Kokoro dimensions:
//!   D_MODEL=512, T_DEC=256+, INIT_CHANNELS=512, NUM_HEADS=2
//!
//! This module verifies the pipeline topology is sound at intermediate scales,
//! proving that NY bound propagation works as dimensions grow.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 34.

#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::TensorParamBinding;

// -------------------------------------------------------------------------
// Configurable pipeline dimensions
// -------------------------------------------------------------------------

/// Parameterized pipeline configuration for scaled verification.
///
/// All dimensions derived from `d_model`:
///   - `num_heads` must evenly divide `d_model`
///   - `init_channels` is typically `d_model`
///   - `ffn_dim` is typically `2 * d_model`
#[derive(Debug, Clone)]
pub(super) struct ScaledPipelineConfig {
    pub(super) d_model: usize,
    pub(super) t_dec: usize,
    pub(super) t_enc: usize,
    pub(super) num_heads: usize,
    pub(super) init_channels: usize,
    pub(super) ffn_dim: usize,
    pub(super) noise_channels: usize,
}

impl ScaledPipelineConfig {
    /// Create a config for the given `d_model`, with derived defaults.
    ///
    /// `d_model` must be divisible by `num_heads`.
    pub(crate) fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model must be divisible by num_heads"
        );
        Self {
            d_model,
            t_dec: 4,
            t_enc: 4,
            num_heads,
            init_channels: d_model,
            ffn_dim: d_model * 2,
            noise_channels: d_model / 2,
        }
    }

    fn d_k(&self) -> usize {
        self.d_model / self.num_heads
    }

    pub(crate) fn channels_at_stage(&self, stage: usize) -> usize {
        self.init_channels >> stage
    }

    pub(crate) fn time_after_stages(&self, num_stages: usize) -> usize {
        let mut t = self.t_dec;
        for _ in 0..num_stages {
            t = (t - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;
        }
        t
    }
}

// Shared decoder constants, structs, and helpers (Part of #1970: dedup).
use super::common::decoder_common::{
    causal_mask, dilated_same_padding, encoder_k, near_identity, precompute_noise_signal,
    scaled_diag, sin_pe, uniform, AttnLayerIds, DilatedSubLayerIds, ResBlockIds, OUTPUT_CHANNELS,
    OUT_KERNEL, OUT_PADDING, UPSAMPLE_KERNEL, UPSAMPLE_PADDING, UPSAMPLE_STRIDE, WEIGHT_MAG,
};

// Re-export items used by test files via `attn_decoder_scaled::`.
pub(super) use super::common::decoder_common::{ProjectionOrder, KOKORO_DILATIONS, KOKORO_KERNELS};

struct OutputStageIds {
    ups_w: TensorNodeId,
    noise_signal: TensorNodeId,
    resblocks: Vec<ResBlockIds>,
    avg_scale: TensorNodeId,
}

// -------------------------------------------------------------------------
// Scaled pipeline builder
// -------------------------------------------------------------------------

/// Build the full Kokoro pipeline with configurable dimensions.
///
/// `cfg`: dimension configuration.
/// `na`: attention layers (≥2).
/// `ns`: upsample stages (1-2).
/// `kernel_sizes`: kernel sizes per ResBlock.
/// `dilations`: dilation pattern per ResBlock.
/// `proj_order`: whether channel projection is before or after exp.
///
/// Returns `(TensorKernelDef, output_shape=[1, T])`.
pub(super) fn build_scaled_pipeline(
    cfg: &ScaledPipelineConfig,
    na: usize,
    ns: usize,
    kernel_sizes: &[usize],
    dilations: &[usize],
    proj_order: ProjectionOrder,
) -> (TensorKernelDef, [usize; 2]) {
    assert!(na >= 2 && (1..=2).contains(&ns));
    assert!(!kernel_sizes.is_empty() && !dilations.is_empty());

    let dm = cfg.d_model;
    let td = cfg.t_dec;
    let te = cfg.t_enc;
    let nh = cfg.num_heads;
    let dk = cfg.d_k();
    let ffn = cfg.ffn_dim;
    let scale = 1.0 / (dk as f32).sqrt();
    let ss = [nh, td, te];
    let cs = [nh, td, dk];

    let mut b = TensorBlockBuilder::new("scaled_output_pipeline");

    // --- Inputs ---
    let hidden = b.add_input("hidden", &[td, dm]);
    let dec_pe = b.add_input("dec_pe", &[td, dm]);
    let enc_k = b.add_input("enc_k", &[te, dm]);
    let enc_v = b.add_input("enc_v", &[te, dm]);

    // --- Attention layer parameters ---
    let mut attn_ids = Vec::with_capacity(na);
    for i in 0..na {
        let s = format!("_L{i}");
        attn_ids.push(AttnLayerIds {
            w_q: b.add_input(&format!("w_q{s}"), &[dm, dm]),
            w_k: b.add_input(&format!("w_k{s}"), &[dm, dm]),
            w_v: b.add_input(&format!("w_v{s}"), &[dm, dm]),
            w_o: b.add_input(&format!("w_o{s}"), &[dm, dm]),
            mask: b.add_input(&format!("mask{s}"), &[td, te]),
            ln_w: b.add_input(&format!("ln_w{s}"), &[dm]),
            ln_b: b.add_input(&format!("ln_b{s}"), &[dm]),
            ln_eps: b.add_input(&format!("ln_eps{s}"), &[1]),
            ffn_up: b.add_input(&format!("ffn_up{s}"), &[ffn, dm]),
            ffn_down: b.add_input(&format!("ffn_down{s}"), &[dm, ffn]),
        });
    }

    // --- Bridge ---
    let bridge_ch = cfg.channels_at_stage(0);
    let bridge_w = b.add_input("bridge_w", &[dm, bridge_ch]);

    // --- Per-stage decoder parameters ---
    let dec_eps = b.add_input("dec_eps", &[1]);

    let mut stage_ids = Vec::with_capacity(ns);
    for si in 0..ns {
        let in_ch = cfg.channels_at_stage(si);
        let out_ch = cfg.channels_at_stage(si + 1);
        let out_t = cfg.time_after_stages(si + 1);
        let sp = format!("_S{si}");

        let ups_w = b.add_input(&format!("ups_w{sp}"), &[in_ch, out_ch, UPSAMPLE_KERNEL]);
        let noise_signal = b.add_input(&format!("noise{sp}"), &[out_ch, out_t]);

        let mut resblocks = Vec::with_capacity(kernel_sizes.len());
        for (ki, _) in kernel_sizes.iter().enumerate() {
            let ks = kernel_sizes[ki];
            let mut sublayers = Vec::with_capacity(dilations.len());
            for (di, _) in dilations.iter().enumerate() {
                let ds = format!("{sp}_K{ki}_D{di}");
                sublayers.push(DilatedSubLayerIds {
                    gamma1: b.add_input(&format!("g1{ds}"), &[out_ch]),
                    beta1: b.add_input(&format!("b1{ds}"), &[out_ch]),
                    alpha1: b.add_input(&format!("a1{ds}"), &[1]),
                    conv_w: b.add_input(&format!("cd{ds}"), &[out_ch, out_ch, ks]),
                    gamma2: b.add_input(&format!("g2{ds}"), &[out_ch]),
                    beta2: b.add_input(&format!("b2{ds}"), &[out_ch]),
                    alpha2: b.add_input(&format!("a2{ds}"), &[1]),
                    conv_unit_w: b.add_input(&format!("cu{ds}"), &[out_ch, out_ch, ks]),
                });
            }
            resblocks.push(ResBlockIds { sublayers });
        }

        let avg_scale = b.add_input(&format!("avg_scale{sp}"), &[1]);

        stage_ids.push(OutputStageIds {
            ups_w,
            noise_signal,
            resblocks,
            avg_scale,
        });
    }

    // --- Output projection parameters ---
    let final_ch = cfg.channels_at_stage(ns);
    let final_t = cfg.time_after_stages(ns);

    let act_w = b.add_input("act_w", &[final_ch, final_ch, OUT_KERNEL]);
    // PyTorch convention: [out_channels, in_channels, kernel_size]
    let proj_w = b.add_input("proj_w", &[OUTPUT_CHANNELS, final_ch, OUT_KERNEL]);

    // === Attention Stack ===
    let sk = build_snake_scalar_kernel().expect("snake kernel");
    let mut prev = b.add_binary_add(hidden, dec_pe, &[td, dm]);
    let mut final_ctx: Option<TensorNodeId> = None;

    for (li, a) in attn_ids.iter().enumerate() {
        let last = li == na - 1;
        let dm_shape = [td, dm];
        let em_shape = [te, dm];

        let q = b.add_matmul(prev, a.w_q, false, None, &dm_shape);
        let k = b.add_matmul(enc_k, a.w_k, false, None, &em_shape);
        let qr = b.add_reshape(q, &[td, nh, dk]);
        let kr = b.add_reshape(k, &[te, nh, dk]);
        let qt = b.add_transpose(qr, &[1, 0, 2], &[nh, td, dk]);
        let kt = b.add_transpose(kr, &[1, 0, 2], &[nh, te, dk]);

        let sc = b.add_matmul(qt, kt, true, Some(scale), &ss);
        let mbc = b.add_broadcast(a.mask, &ss);
        let ma = b.add_binary_add(sc, mbc, &ss);
        let w = b.add_softmax(ma, -1, &ss);

        let v = b.add_matmul(enc_v, a.w_v, false, None, &em_shape);
        let vr = b.add_reshape(v, &[te, nh, dk]);
        let vt = b.add_transpose(vr, &[1, 0, 2], &[nh, te, dk]);

        let ctx = b.add_matmul(w, vt, false, None, &cs);
        let ct = b.add_transpose(ctx, &[1, 0, 2], &[td, nh, dk]);
        let cf = b.add_reshape(ct, &dm_shape);

        if last {
            final_ctx = Some(cf);
            break;
        }

        let ao = b.add_matmul(cf, a.w_o, false, None, &dm_shape);
        let r = b.add_binary_add(prev, ao, &dm_shape);
        let n = b.add_layer_norm(r, a.ln_eps, 1, a.ln_w, a.ln_b, &dm_shape);
        let f1 = b.add_linear(n, a.ffn_up, None, &[td, ffn]);
        let fa = b.add_gelu(f1, &[td, ffn]);
        let f2 = b.add_linear(fa, a.ffn_down, None, &dm_shape);
        prev = b.add_binary_add(r, f2, &dm_shape);
    }

    let cf = final_ctx.expect("at least 2 attention layers");

    // === Context Bridge ===
    let br = b.add_matmul(cf, bridge_w, false, None, &[td, bridge_ch]);
    let di = b.add_transpose(br, &[1, 0], &[bridge_ch, td]);

    // === Multi-Stage Decoder with Noise Injection + Multi-Kernel ResBlocks ===
    let mut cur = di;
    let mut cur_ch = bridge_ch;
    let mut cur_t = td;

    for (si, stage) in stage_ids.iter().enumerate() {
        let out_ch = cfg.channels_at_stage(si + 1);
        let out_t = (cur_t - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;
        let up_shape = [out_ch, out_t];

        let activated = b.add_leaky_relu(cur, 0.1, &[cur_ch, cur_t]);
        let upsampled = b.add_conv_transpose_1d(
            activated,
            stage.ups_w,
            None,
            UPSAMPLE_STRIDE,
            UPSAMPLE_PADDING,
            1,
            1,
            0, // output_padding
            &up_shape,
        );

        let with_noise = b.add_binary_add(upsampled, stage.noise_signal, &up_shape);

        let mut rb_outputs: Vec<TensorNodeId> = Vec::with_capacity(kernel_sizes.len());

        for (ki, rb) in stage.resblocks.iter().enumerate() {
            let ks = kernel_sizes[ki];
            let mut rb_cur = with_noise;

            for (di_idx, sl) in rb.sublayers.iter().enumerate() {
                let dilation = dilations[di_idx];

                let n1 = b.add_instance_norm(
                    rb_cur,
                    dec_eps,
                    1,
                    Some(sl.gamma1),
                    Some(sl.beta1),
                    &up_shape,
                );
                let a1bc = b.add_broadcast(sl.alpha1, &up_shape);
                let sn1 = b.add_elementwise(sk.clone(), &[n1, a1bc], &up_shape);
                let dil_pad = dilated_same_padding(ks, dilation);
                let cv1 =
                    b.add_conv1d_full(sn1, sl.conv_w, None, 1, dil_pad, dilation, 1, &up_shape);

                let n2 = b.add_instance_norm(
                    cv1,
                    dec_eps,
                    1,
                    Some(sl.gamma2),
                    Some(sl.beta2),
                    &up_shape,
                );
                let a2bc = b.add_broadcast(sl.alpha2, &up_shape);
                let sn2 = b.add_elementwise(sk.clone(), &[n2, a2bc], &up_shape);
                let unit_pad = dilated_same_padding(ks, 1);
                let cv2 =
                    b.add_conv1d_full(sn2, sl.conv_unit_w, None, 1, unit_pad, 1, 1, &up_shape);

                rb_cur = b.add_binary_add(rb_cur, cv2, &up_shape);
            }

            rb_outputs.push(rb_cur);
        }

        let mut sum = rb_outputs[0];
        for &rb_out in &rb_outputs[1..] {
            sum = b.add_binary_add(sum, rb_out, &up_shape);
        }
        let scale_bc = b.add_broadcast(stage.avg_scale, &up_shape);
        cur = b.add_binary_mul(sum, scale_bc, &up_shape);

        cur_ch = out_ch;
        cur_t = out_t;
    }

    // === Output Path (final activation + projection) ===
    let dec_shape = [final_ch, final_t];
    let mono_shape = [OUTPUT_CHANNELS, final_t];

    let out = match proj_order {
        ProjectionOrder::AfterExp => {
            let ra = b.add_leaky_relu(cur, 0.01, &dec_shape);
            let xp = b.add_conv1d(ra, act_w, None, 1, OUT_PADDING, &dec_shape);
            let ex = b.add_exp(xp, &dec_shape);
            b.add_conv1d(ex, proj_w, None, 1, OUT_PADDING, &mono_shape)
        }
        ProjectionOrder::BeforeExp => {
            let ra = b.add_leaky_relu(cur, 0.01, &dec_shape);
            let xp = b.add_conv1d(ra, act_w, None, 1, OUT_PADDING, &dec_shape);
            let proj = b.add_conv1d(xp, proj_w, None, 1, OUT_PADDING, &mono_shape);
            b.add_exp(proj, &mono_shape)
        }
    };

    (b.build(out).expect("valid scaled pipeline"), mono_shape)
}

// -------------------------------------------------------------------------
// Bindings (parameterized)
// -------------------------------------------------------------------------

/// Build parameter bindings for the scaled pipeline.
pub(super) fn scaled_pipeline_bindings(
    cfg: &ScaledPipelineConfig,
    na: usize,
    ns: usize,
    kernel_sizes: &[usize],
    dilations: &[usize],
    pe_scale: f32,
    w_pert: f32,
) -> Vec<TensorParamBinding> {
    let dm = cfg.d_model;
    let td = cfg.t_dec;
    let te = cfg.t_enc;
    let nh = cfg.num_heads;
    let ffn = cfg.ffn_dim;

    let mut pe = sin_pe(td, dm, nh);
    pe.mapv_inplace(|v| v * pe_scale);
    let ek = encoder_k(te, dm);
    let ev = encoder_k(te, dm);
    let wp = near_identity(dm, w_pert);
    let mk = causal_mask(td, te);
    let lw = uniform(&[dm], 1.0);
    let lb = uniform(&[dm], 0.0);
    let fu = scaled_diag(ffn, dm, 0.1);
    let fd = scaled_diag(dm, ffn, 0.1);

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
        TensorParamBinding::ConstantTensor(ek),
        TensorParamBinding::ConstantTensor(ev),
    ];

    for _ in 0..na {
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(wp.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(mk.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(lw.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(lb.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(fu.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fd.clone()));
    }

    // Bridge
    let bridge_ch = cfg.channels_at_stage(0);
    bindings.push(TensorParamBinding::ConstantTensor(scaled_diag(
        dm, bridge_ch, 0.1,
    )));

    // dec_eps
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    let nk = kernel_sizes.len();
    let avg_val = 1.0 / nk as f32;

    for si in 0..ns {
        let in_ch = cfg.channels_at_stage(si);
        let out_ch = cfg.channels_at_stage(si + 1);
        let out_t = cfg.time_after_stages(si + 1);

        bindings.push(TensorParamBinding::ConstantTensor(uniform(
            &[in_ch, out_ch, UPSAMPLE_KERNEL],
            WEIGHT_MAG,
        )));

        bindings.push(TensorParamBinding::ConstantTensor(precompute_noise_signal(
            out_ch, out_t, si, ns, 0.01,
        )));

        for &ks in kernel_sizes {
            for _ in dilations {
                bindings.push(TensorParamBinding::ConstantTensor(uniform(&[out_ch], 1.0)));
                bindings.push(TensorParamBinding::ConstantTensor(uniform(&[out_ch], 0.0)));
                bindings.push(TensorParamBinding::ConstantScalar(1.0));
                bindings.push(TensorParamBinding::ConstantTensor(uniform(
                    &[out_ch, out_ch, ks],
                    WEIGHT_MAG,
                )));
                bindings.push(TensorParamBinding::ConstantTensor(uniform(&[out_ch], 1.0)));
                bindings.push(TensorParamBinding::ConstantTensor(uniform(&[out_ch], 0.0)));
                bindings.push(TensorParamBinding::ConstantScalar(1.0));
                bindings.push(TensorParamBinding::ConstantTensor(uniform(
                    &[out_ch, out_ch, ks],
                    WEIGHT_MAG,
                )));
            }
        }

        bindings.push(TensorParamBinding::ConstantScalar(avg_val));
    }

    // Internal activation conv: [final_ch, final_ch, OUT_KERNEL]
    let final_ch = cfg.channels_at_stage(ns);
    bindings.push(TensorParamBinding::ConstantTensor(uniform(
        &[final_ch, final_ch, OUT_KERNEL],
        WEIGHT_MAG,
    )));

    // Output projection conv: [1, final_ch, OUT_KERNEL]
    bindings.push(TensorParamBinding::ConstantTensor(uniform(
        &[OUTPUT_CHANNELS, final_ch, OUT_KERNEL],
        WEIGHT_MAG,
    )));

    bindings
}
