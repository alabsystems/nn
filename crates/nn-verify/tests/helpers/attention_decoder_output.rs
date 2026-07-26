// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phase 33 helpers: Output projection + full pipeline composition for
//! Kokoro ISTFTNet decoder.
//!
//! Extends Phase 32's noise-injection pipeline with the production
//! output path: Conv1d projection reducing decoder channels to 1 (mono
//! waveform output). This closes the gap between the verified topology
//! and the actual Kokoro Generator output stage.
//!
//! Production Kokoro output path:
//!   h = LeakyReLU(h)                  // [ch, T]
//!   h = Conv1d(h, kernel=7, pad=3)    // [ch, T] → [ch, T] (internal)
//!   h = exp(h)                        // [ch, T] → waveform magnitude
//!   out = Conv1d(h, kernel=7, pad=3)  // [ch, T] → [1, T] (output projection)
//!
//! The output projection reduces ch → 1 for mono audio. In production
//! this is done BEFORE exp in some variants (channel reduction in log
//! domain) or AFTER exp (magnitude domain). We verify both orderings
//! to ensure bounds hold regardless.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 33.

#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{TensorKernelDef, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};

// Shared decoder constants, structs, and helpers (Part of #1971: dedup).
use super::common::bounds_min_max;
use super::common::decoder_common::{
    causal_mask, channels_at_stage, dilated_same_padding, encoder_k, near_identity,
    precompute_noise_signal, scaled_diag, sin_pe, time_after_stages, uniform, AttnLayerIds,
    DilatedSubLayerIds, ResBlockIds, D_K, FFN_DIM, NUM_HEADS, OUTPUT_CHANNELS, OUT_KERNEL,
    OUT_PADDING, T_ENC, UPSAMPLE_KERNEL, UPSAMPLE_PADDING, UPSAMPLE_STRIDE, WEIGHT_MAG,
};

// Re-export items used by monotonicity_groups_fj.rs via `attn_decoder_output::`.
pub(super) use super::common::decoder_common::{
    ProjectionOrder, D_MODEL, KOKORO_DILATIONS, KOKORO_KERNELS, T_DEC,
};

// -------------------------------------------------------------------------
// File-specific type
// -------------------------------------------------------------------------

struct OutputStageIds {
    ups_w: TensorNodeId,
    noise_signal: TensorNodeId,
    resblocks: Vec<ResBlockIds>,
    avg_scale: TensorNodeId,
}

// -------------------------------------------------------------------------
// Full pipeline builder with output projection
// -------------------------------------------------------------------------

/// Build attention → bridge → multi-stage decoder → output projection → mono.
///
/// `na`: attention layers (≥2).
/// `ns`: upsample stages (1-2).
/// `kernel_sizes`: kernel sizes per ResBlock.
/// `dilations`: dilation pattern per ResBlock.
/// `proj_order`: whether channel projection is before or after exp.
///
/// Returns `(TensorKernelDef, output_shape=[1, T])`.
pub(super) fn build_output_pipeline(
    na: usize,
    ns: usize,
    kernel_sizes: &[usize],
    dilations: &[usize],
    proj_order: ProjectionOrder,
) -> (TensorKernelDef, [usize; 2]) {
    assert!(na >= 2 && (1..=2).contains(&ns));
    assert!(!kernel_sizes.is_empty() && !dilations.is_empty());

    let scale = 1.0 / (D_K as f32).sqrt();
    let ss = [NUM_HEADS, T_DEC, T_ENC];
    let cs = [NUM_HEADS, T_DEC, D_K];

    let mut b = TensorBlockBuilder::new("output_projection_pipeline");

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
        let out_t = time_after_stages(si + 1);
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
    let final_ch = channels_at_stage(ns);
    let final_t = time_after_stages(ns);

    // Internal activation conv: [ch, ch, K]
    let act_w = b.add_input("act_w", &[final_ch, final_ch, OUT_KERNEL]);
    // Output projection conv: [1, ch, K] → reduces channels to mono
    // PyTorch convention: [out_channels, in_channels, kernel_size]
    let proj_w = b.add_input("proj_w", &[OUTPUT_CHANNELS, final_ch, OUT_KERNEL]);

    // === Attention Stack ===
    let sk = build_snake_scalar_kernel().expect("snake kernel");
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

    let cf = final_ctx.expect("at least 2 attention layers");

    // === Context Bridge ===
    let br = b.add_matmul(cf, bridge_w, false, None, &[T_DEC, bridge_ch]);
    let di = b.add_transpose(br, &[1, 0], &[bridge_ch, T_DEC]);

    // === Multi-Stage Decoder with Noise Injection + Multi-Kernel ResBlocks ===
    let mut cur = di;
    let mut cur_ch = bridge_ch;
    let mut cur_t = T_DEC;

    for (si, stage) in stage_ids.iter().enumerate() {
        let out_ch = channels_at_stage(si + 1);
        let out_t = (cur_t - 1) * UPSAMPLE_STRIDE + UPSAMPLE_KERNEL - 2 * UPSAMPLE_PADDING;
        let up_shape = [out_ch, out_t];

        // 1. Upsample via ConvTranspose1d
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

        // 2. Additive noise injection
        let with_noise = b.add_binary_add(upsampled, stage.noise_signal, &up_shape);

        // 3. Multi-kernel ResBlocks
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
            // Default Kokoro: internal conv → exp → projection
            let ra = b.add_leaky_relu(cur, 0.01, &dec_shape);
            let xp = b.add_conv1d(ra, act_w, None, 1, OUT_PADDING, &dec_shape);
            let ex = b.add_exp(xp, &dec_shape);
            // Project ch → 1 in magnitude domain
            b.add_conv1d(ex, proj_w, None, 1, OUT_PADDING, &mono_shape)
        }
        ProjectionOrder::BeforeExp => {
            // Variant: internal conv → projection → exp
            let ra = b.add_leaky_relu(cur, 0.01, &dec_shape);
            let xp = b.add_conv1d(ra, act_w, None, 1, OUT_PADDING, &dec_shape);
            // Project ch → 1 in log domain
            let proj = b.add_conv1d(xp, proj_w, None, 1, OUT_PADDING, &mono_shape);
            // Then exp on mono signal
            b.add_exp(proj, &mono_shape)
        }
    };

    (
        b.build(out).expect("valid output projection pipeline"),
        mono_shape,
    )
}

// -------------------------------------------------------------------------
// Bindings
// -------------------------------------------------------------------------

/// Build parameter bindings for the output projection pipeline.
pub(super) fn output_pipeline_bindings(
    na: usize,
    ns: usize,
    kernel_sizes: &[usize],
    dilations: &[usize],
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
    let bridge_ch = channels_at_stage(0);
    bindings.push(TensorParamBinding::ConstantTensor(scaled_diag(
        D_MODEL, bridge_ch, 0.1,
    )));

    // dec_eps
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    let nk = kernel_sizes.len();
    let avg_val = 1.0 / nk as f32;

    for si in 0..ns {
        let in_ch = channels_at_stage(si);
        let out_ch = channels_at_stage(si + 1);
        let out_t = time_after_stages(si + 1);

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
    let final_ch = channels_at_stage(ns);
    bindings.push(TensorParamBinding::ConstantTensor(uniform(
        &[final_ch, final_ch, OUT_KERNEL],
        WEIGHT_MAG,
    )));

    // Output projection conv: [1, final_ch, OUT_KERNEL]
    // PyTorch convention: [out_channels, in_channels, kernel_size]
    bindings.push(TensorParamBinding::ConstantTensor(uniform(
        &[OUTPUT_CHANNELS, final_ch, OUT_KERNEL],
        WEIGHT_MAG,
    )));

    bindings
}

// -------------------------------------------------------------------------
// Analysis
// -------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct OutputPipelineResult {
    pub(super) num_attn_layers: usize,
    pub(super) num_stages: usize,
    pub(super) kernel_sizes: Vec<usize>,
    pub(super) dilations: Vec<usize>,
    pub(super) proj_order: ProjectionOrder,
    pub(super) graph_nodes: usize,
    pub(super) output_channels: usize,
    pub(super) output_time: usize,
    pub(super) min_output_lo: f32,
    pub(super) max_output_hi: f32,
    pub(super) avg_bound_width: f32,
    pub(super) all_positive: bool,
    pub(super) all_finite: bool,
}

pub(super) fn analyze_output_pipeline(
    output: &BoundedTensor,
    na: usize,
    ns: usize,
    kernel_sizes: &[usize],
    dilations: &[usize],
    proj_order: ProjectionOrder,
    nodes: usize,
) -> OutputPipelineResult {
    let (lo, hi) = output.lower_upper();
    let fl: Vec<f32> = lo.iter().copied().collect();
    let fh: Vec<f32> = hi.iter().copied().collect();

    let (min_lo, max_hi) = bounds_min_max(output);
    let n = fl.len().max(1) as f32;
    let avg_w: f32 = fl.iter().zip(fh.iter()).map(|(&l, &h)| h - l).sum::<f32>() / n;

    let final_t = time_after_stages(ns);

    OutputPipelineResult {
        num_attn_layers: na,
        num_stages: ns,
        kernel_sizes: kernel_sizes.to_vec(),
        dilations: dilations.to_vec(),
        proj_order,
        graph_nodes: nodes,
        output_channels: OUTPUT_CHANNELS,
        output_time: final_t,
        min_output_lo: min_lo,
        max_output_hi: max_hi,
        avg_bound_width: avg_w,
        all_positive: fl.iter().all(|&v| v >= 0.0),
        all_finite: fl.iter().chain(fh.iter()).all(|v| v.is_finite()),
    }
}
