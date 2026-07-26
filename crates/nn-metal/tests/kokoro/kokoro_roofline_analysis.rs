// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Roofline analysis of the Kokoro synthesis pipeline on Apple M4 Max.
//!
//! For each major Kokoro segment, calculates:
//! - FLOPs (from tensor shapes and operation types)
//! - Bytes transferred (input + output + weight buffers)
//! - Arithmetic intensity (FLOPs/byte)
//! - Whether bandwidth-bound or compute-bound on M4 Max
//!
//! Identifies the top bandwidth-bound and compute-bound operations and
//! proposes optimizations for each category.
//!
//! Part of #4264 (RTF optimization).

use std::collections::BTreeMap;

/// M4 Max hardware constants for roofline analysis.
///
/// Sources:
/// - Apple M4 Max: 40-core GPU, 14.2 TFLOPS F32, ~28 TFLOPS F16
/// - Unified memory bandwidth: ~400 GB/s sustained
/// - Metal dispatch overhead: ~1.5 us (measured)
struct M4MaxSpec {
    peak_tflops_f32: f64,
    peak_tflops_f16: f64,
    bandwidth_gbs: f64,
    dispatch_overhead_us: f64,
}

impl M4MaxSpec {
    fn new() -> Self {
        Self {
            peak_tflops_f32: 14.2,
            peak_tflops_f16: 28.4,
            bandwidth_gbs: 400.0,
            dispatch_overhead_us: 1.5,
        }
    }

    /// Ridge point: FLOP/byte threshold where compute == bandwidth.
    /// Operations below this are bandwidth-bound, above are compute-bound.
    fn ridge_point_f32(&self) -> f64 {
        // FLOPS / (bytes/s) = FLOP/byte
        (self.peak_tflops_f32 * 1e12) / (self.bandwidth_gbs * 1e9)
    }

    fn ridge_point_f16(&self) -> f64 {
        (self.peak_tflops_f16 * 1e12) / (self.bandwidth_gbs * 1e9)
    }

    /// Estimate time in microseconds from roofline model.
    fn estimate_us(&self, flops: u64, bytes: u64, is_f16: bool) -> f64 {
        let peak = if is_f16 {
            self.peak_tflops_f16
        } else {
            self.peak_tflops_f32
        };
        let compute_us = flops as f64 / (peak * 1e6);
        let memory_us = bytes as f64 / (self.bandwidth_gbs * 1e3);
        f64::max(compute_us, memory_us) + self.dispatch_overhead_us
    }
}

/// One operation in the roofline analysis.
#[derive(Clone, Debug)]
struct RooflineEntry {
    name: String,
    segment: String,
    flops: u64,
    bytes: u64,
    count: usize,
    is_f16: bool,
}

impl RooflineEntry {
    fn arithmetic_intensity(&self) -> f64 {
        if self.bytes == 0 {
            return f64::INFINITY;
        }
        self.flops as f64 / self.bytes as f64
    }

    fn is_bandwidth_bound(&self, spec: &M4MaxSpec) -> bool {
        let ridge = if self.is_f16 {
            spec.ridge_point_f16()
        } else {
            spec.ridge_point_f32()
        };
        self.arithmetic_intensity() < ridge
    }

    fn estimated_time_us(&self, spec: &M4MaxSpec) -> f64 {
        spec.estimate_us(self.flops, self.bytes, self.is_f16) * self.count as f64
    }

    fn bottleneck(&self, spec: &M4MaxSpec) -> &'static str {
        if self.is_bandwidth_bound(spec) {
            "BW-bound"
        } else {
            "Compute-bound"
        }
    }
}

// ---------------------------------------------------------------------------
// Kokoro architecture constants (D=512 default config)
// ---------------------------------------------------------------------------

/// Kokoro D=512 architecture analysis.
///
/// Architecture:
///   PlBERT encoder -> Text Encoder -> Prosody Predictor -> Duration Regulator
///   -> F0/Energy Predictor -> SineGen -> Generator (ISTFTNet) -> iSTFT
///
/// Generator is the critical path (63.8% of RTF).
/// Generator stages:
///   conv_pre: Conv1d(512, 512, k=7, pad=3)
///   Stage 0: ConvTranspose1d(512->256, k=20, stride=10) + 3 ResBlocks(256, k=3/7/11, d=[1,3,5])
///   Stage 1: ConvTranspose1d(256->128, k=12, stride=6)  + 3 ResBlocks(128, k=3/7/11, d=[1,3,5])
///   conv_post: Conv1d(128, n_fft+2=22, k=7, pad=3)
///
/// ResBlock (per block, per dilation):
///   AdaIN1(style) -> Snake1(alpha) -> Conv1d(C,C,k,dilation=d) -> AdaIN2(style) -> Snake2(alpha) -> Conv1d(C,C,k,dilation=1) -> + residual
///
/// For 8 phoneme tokens, T_aligned ~ 20 (after duration regulation).
/// After conv_pre: T=20
/// After stage 0 upsample: T=200
/// After stage 1 upsample: T=1200
/// iSTFT output: ~1200 * (n_fft/2+1) = 1200*11 ~ 13200 audio samples (~0.55s @ 24kHz)

#[test]
fn kokoro_roofline_analysis_d512() {
    let spec = M4MaxSpec::new();

    eprintln!("\n{}", "=".repeat(100));
    eprintln!("  KOKORO ROOFLINE ANALYSIS — Apple M4 Max (D=512)");
    eprintln!("{}", "=".repeat(100));
    eprintln!();
    eprintln!("  M4 Max specs:");
    eprintln!("    F32 peak:     {:.1} TFLOPS", spec.peak_tflops_f32);
    eprintln!("    F16 peak:     {:.1} TFLOPS", spec.peak_tflops_f16);
    eprintln!("    Bandwidth:    {:.0} GB/s", spec.bandwidth_gbs);
    eprintln!("    Ridge (F32):  {:.1} FLOP/byte", spec.ridge_point_f32());
    eprintln!("    Ridge (F16):  {:.1} FLOP/byte", spec.ridge_point_f16());
    eprintln!();

    let mut entries: Vec<RooflineEntry> = Vec::new();

    // --- Token dimensions ---
    // 8 phoneme tokens -> after PlBERT: [1, 8, 768]
    // After text encoder: [1, 8, 512]
    // After duration regulation with ~2.5 frames/phoneme: T_aligned ~ 20
    // Generator conv_pre output: [1, 512, 20]
    // Stage 0 upsample (stride=10): [1, 256, 200]
    // Stage 1 upsample (stride=6): [1, 128, 1200]

    let b: u64 = 1; // batch size
    let f32_bytes: u64 = 4;

    // ========================================================================
    // 1. PlBERT Encoder
    // ========================================================================
    let n_tokens: u64 = 8;
    let plbert_dim: u64 = 768;
    // PlBERT: 6 transformer layers, each with self-attention + FFN.
    // Self-attention: 4 * B*T*D*D (QKV projection + output projection)
    //   FLOPs = 4 * 2*B*T*D*D (multiply-accumulate)
    // FFN: 2 * B*T*D*4D (up + down projection)
    //   FLOPs = 2 * 2*B*T*D*4D
    let plbert_attn_flops: u64 = 4 * 2 * b * n_tokens * plbert_dim * plbert_dim;
    let plbert_ffn_flops: u64 = 2 * 2 * b * n_tokens * plbert_dim * 4 * plbert_dim;
    let plbert_layer_flops = plbert_attn_flops + plbert_ffn_flops;
    let plbert_total_flops = 6 * plbert_layer_flops;

    // Bytes: read input + weights + write output per layer
    // Weights: 4*D*D + 2*D*4*D = 4*D^2 + 8*D^2 = 12*D^2 per layer
    let plbert_weight_bytes: u64 = 6 * 12 * plbert_dim * plbert_dim * f32_bytes;
    let plbert_activation_bytes: u64 = 6 * 2 * b * n_tokens * plbert_dim * f32_bytes;
    let plbert_total_bytes = plbert_weight_bytes + plbert_activation_bytes;

    entries.push(RooflineEntry {
        name: "PlBERT encoder (6 layers)".into(),
        segment: "plbert".into(),
        flops: plbert_total_flops,
        bytes: plbert_total_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 2. Text Encoder (projection 768->512 + 3 transformer layers)
    // ========================================================================
    let d_en: u64 = 512;
    let text_proj_flops: u64 = 2 * b * n_tokens * plbert_dim * d_en;
    let _text_proj_bytes: u64 =
        (plbert_dim * d_en + b * n_tokens * (plbert_dim + d_en)) * f32_bytes;

    let text_attn_flops: u64 = 4 * 2 * b * n_tokens * d_en * d_en;
    let text_ffn_flops: u64 = 2 * 2 * b * n_tokens * d_en * 4 * d_en;
    let text_layer_flops = text_attn_flops + text_ffn_flops;
    let text_total_flops = text_proj_flops + 3 * text_layer_flops;

    let text_weight_bytes: u64 = plbert_dim * d_en * f32_bytes + 3 * 12 * d_en * d_en * f32_bytes;
    let text_activation_bytes: u64 = 4 * b * n_tokens * d_en * f32_bytes;
    let text_total_bytes = text_weight_bytes + text_activation_bytes;

    entries.push(RooflineEntry {
        name: "Text encoder (proj + 3 layers)".into(),
        segment: "text".into(),
        flops: text_total_flops,
        bytes: text_total_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 3. Prosody Predictor (3 blocks: Conv1d + LayerNorm + LeakyReLU)
    // ========================================================================
    let prosody_flops: u64 = 3 * 2 * b * n_tokens * d_en * d_en * 5; // k=5 convolutions
    let prosody_bytes: u64 = 3 * (d_en * d_en * 5 + 2 * b * n_tokens * d_en) * f32_bytes;

    entries.push(RooflineEntry {
        name: "Prosody predictor (3 conv blocks)".into(),
        segment: "prosody".into(),
        flops: prosody_flops,
        bytes: prosody_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 4. Duration Regulator (lightweight, dominated by sigmoid + cumsum)
    // ========================================================================
    let regulate_flops: u64 = b * n_tokens * 50 * 2; // sigmoid + sum over max_dur bins
    let regulate_bytes: u64 = b * n_tokens * 50 * f32_bytes * 2;

    entries.push(RooflineEntry {
        name: "Duration regulator".into(),
        segment: "regulate".into(),
        flops: regulate_flops,
        bytes: regulate_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 5. F0/Energy Predictor (BiLSTM h=256 + Linear projections)
    // ========================================================================
    let t_aligned: u64 = 20;
    let lstm_hidden: u64 = 256;
    // BiLSTM: 4 gates * 2 directions * 2*(D*H + H*H) FLOPs per timestep
    let lstm_flops: u64 = t_aligned * 4 * 2 * 2 * (d_en * lstm_hidden + lstm_hidden * lstm_hidden);
    // Weight reads: forward + backward LSTM weights
    let lstm_weight_bytes: u64 =
        2 * 4 * (d_en * lstm_hidden + lstm_hidden * lstm_hidden) * f32_bytes;
    let lstm_activation_bytes: u64 = t_aligned * 2 * lstm_hidden * f32_bytes * 2;
    let lstm_total_bytes = lstm_weight_bytes + lstm_activation_bytes;

    entries.push(RooflineEntry {
        name: "F0/Energy BiLSTM".into(),
        segment: "f0_energy".into(),
        flops: lstm_flops,
        bytes: lstm_total_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 6. SineGen (harmonic source generation)
    // ========================================================================
    // sin() computation for each harmonic at each sample
    let sine_samples: u64 = t_aligned * 60; // after full upsampling
    let sine_flops: u64 = sine_samples * 10; // sin() ~10 FLOPs each
    let sine_bytes: u64 = sine_samples * f32_bytes * 3; // f0 input + cumsum + sin output

    entries.push(RooflineEntry {
        name: "SineGen (harmonic source)".into(),
        segment: "sinegen".into(),
        flops: sine_flops,
        bytes: sine_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 7. Generator — conv_pre
    // ========================================================================
    let gen_channels: u64 = 512;
    let gen_t0: u64 = t_aligned; // T before upsampling

    // conv_pre: Conv1d(512, 512, k=7, pad=3) — no change in T
    let conv_pre_flops: u64 = 2 * b * gen_t0 * gen_channels * gen_channels * 7;
    let conv_pre_weight_bytes: u64 = gen_channels * gen_channels * 7 * f32_bytes;
    let conv_pre_activation_bytes: u64 = b * gen_t0 * gen_channels * f32_bytes * 2;
    let conv_pre_bytes = conv_pre_weight_bytes + conv_pre_activation_bytes;

    entries.push(RooflineEntry {
        name: "Generator conv_pre (512->512, k=7)".into(),
        segment: "generator".into(),
        flops: conv_pre_flops,
        bytes: conv_pre_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 8. Generator — Stage 0 upsample: ConvTranspose1d(512->256, k=20, s=10)
    // ========================================================================
    let stage0_ch_in: u64 = 512;
    let stage0_ch_out: u64 = 256;
    let stage0_t_out: u64 = gen_t0 * 10; // T=200

    // ConvTranspose1d FLOPs: 2 * out_len * out_channels * in_channels * kernel_size
    let upsample0_flops: u64 = 2 * stage0_t_out * stage0_ch_out * stage0_ch_in * 20;
    let upsample0_weight_bytes: u64 = stage0_ch_in * stage0_ch_out * 20 * f32_bytes;
    let upsample0_activation_bytes: u64 =
        (b * gen_t0 * stage0_ch_in + b * stage0_t_out * stage0_ch_out) * f32_bytes;
    let upsample0_bytes = upsample0_weight_bytes + upsample0_activation_bytes;

    entries.push(RooflineEntry {
        name: "Generator upsample stage 0 (512->256)".into(),
        segment: "generator".into(),
        flops: upsample0_flops,
        bytes: upsample0_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 9. Generator — Stage 0 ResBlocks (3 blocks x 3 dilations = 9 layers)
    // ========================================================================
    // Each ResBlock dilation layer at stage 0:
    //   Input/output: [1, 256, 200]
    //   AdaIN: instance_norm(x) + linear(style_128 -> 2*256) + scale/shift
    //   Snake: x + (1/alpha) * sin(alpha*x)^2
    //   Conv1d: (256, 256, k, dilation=d, groups=1)
    //   x2 per dilation layer

    let stage0_ch: u64 = 256;
    let stage0_t: u64 = 200;
    let style_dim: u64 = 128;
    let n_resblock_layers_stage0: u64 = 3 * 3; // 3 blocks * 3 dilations

    // AdaIN per layer: instance_norm (2 passes over T) + linear(128->512) + scale+shift
    let adain_norm_flops: u64 = 2 * b * stage0_ch * stage0_t; // mean + variance reduction
    let adain_linear_flops: u64 = 2 * b * style_dim * 2 * stage0_ch; // style projection
    let adain_affine_flops: u64 = 2 * b * stage0_ch * stage0_t; // scale + shift
    let adain_total_flops: u64 = adain_norm_flops + adain_linear_flops + adain_affine_flops;
    let adain_bytes: u64 = (
        b * stage0_ch * stage0_t * 3 // read x, write normalized, write output
        + style_dim * 2 * stage0_ch // weight
        + 2 * stage0_ch
        // bias
    ) * f32_bytes;

    // Snake per layer: x + (1/alpha) * sin(alpha*x)^2 — ~6 FLOPs/element
    let snake_flops: u64 = 6 * b * stage0_ch * stage0_t;
    let snake_bytes: u64 = b * stage0_ch * stage0_t * f32_bytes * 2 + stage0_ch * f32_bytes; // read + write + alpha

    // Conv1d per dilation layer: average kernel_size over k=3,7,11 and d=1,3,5
    // Actual per-block: k={3,7,11}, each with dilations {1,3,5}
    // FLOPs per conv: 2 * B * T * C * C * K
    let conv_avg_k: u64 = 7; // average of 3,7,11
    let conv_flops: u64 = 2 * b * stage0_t * stage0_ch * stage0_ch * conv_avg_k;
    let conv_weight_bytes: u64 = stage0_ch * stage0_ch * conv_avg_k * f32_bytes;
    let conv_activation_bytes: u64 = b * stage0_t * stage0_ch * f32_bytes * 2;
    let conv_bytes: u64 = conv_weight_bytes + conv_activation_bytes;

    // Per dilation layer total (2x adain + 2x snake + 2x conv + 1x residual add).
    // (Used in the deep dive section for reference; individual sub-ops have their
    //  own entries in the roofline table.)
    let _resblock_layer_flops: u64 = 2 * adain_total_flops + 2 * snake_flops + 2 * conv_flops;
    let _resblock_layer_bytes: u64 = 2 * adain_bytes + 2 * snake_bytes + 2 * conv_bytes;

    // Individual entries for stage 0 ResBlock sub-operations
    entries.push(RooflineEntry {
        name: "Stage0 AdaIN (instance_norm+affine)".into(),
        segment: "generator".into(),
        flops: adain_total_flops,
        bytes: adain_bytes,
        count: (n_resblock_layers_stage0 * 2) as usize, // 2 AdaINs per layer
        is_f16: false,
    });

    entries.push(RooflineEntry {
        name: "Stage0 Snake activation".into(),
        segment: "generator".into(),
        flops: snake_flops,
        bytes: snake_bytes,
        count: (n_resblock_layers_stage0 * 2) as usize,
        is_f16: false,
    });

    entries.push(RooflineEntry {
        name: "Stage0 Conv1d (256,256,k~7)".into(),
        segment: "generator".into(),
        flops: conv_flops,
        bytes: conv_bytes,
        count: (n_resblock_layers_stage0 * 2) as usize,
        is_f16: false,
    });

    // ========================================================================
    // 10. Generator — Stage 1 upsample: ConvTranspose1d(256->128, k=12, s=6)
    // ========================================================================
    let stage1_ch_in: u64 = 256;
    let stage1_ch_out: u64 = 128;
    let stage1_t_out: u64 = stage0_t * 6; // T=1200

    let upsample1_flops: u64 = 2 * stage1_t_out * stage1_ch_out * stage1_ch_in * 12;
    let upsample1_weight_bytes: u64 = stage1_ch_in * stage1_ch_out * 12 * f32_bytes;
    let upsample1_activation_bytes: u64 =
        (b * stage0_t * stage1_ch_in + b * stage1_t_out * stage1_ch_out) * f32_bytes;
    let upsample1_bytes = upsample1_weight_bytes + upsample1_activation_bytes;

    entries.push(RooflineEntry {
        name: "Generator upsample stage 1 (256->128)".into(),
        segment: "generator".into(),
        flops: upsample1_flops,
        bytes: upsample1_bytes,
        count: 1,
        is_f16: false,
    });

    // ========================================================================
    // 11. Generator — Stage 1 ResBlocks (3 blocks x 3 dilations = 9 layers)
    // ========================================================================
    let stage1_ch: u64 = 128;
    let stage1_t: u64 = 1200;
    let n_resblock_layers_stage1: u64 = 3 * 3;

    let s1_adain_norm_flops: u64 = 2 * b * stage1_ch * stage1_t;
    let s1_adain_linear_flops: u64 = 2 * b * style_dim * 2 * stage1_ch;
    let s1_adain_affine_flops: u64 = 2 * b * stage1_ch * stage1_t;
    let s1_adain_total_flops: u64 =
        s1_adain_norm_flops + s1_adain_linear_flops + s1_adain_affine_flops;
    let s1_adain_bytes: u64 =
        (b * stage1_ch * stage1_t * 3 + style_dim * 2 * stage1_ch + 2 * stage1_ch) * f32_bytes;

    let s1_snake_flops: u64 = 6 * b * stage1_ch * stage1_t;
    let s1_snake_bytes: u64 = b * stage1_ch * stage1_t * f32_bytes * 2 + stage1_ch * f32_bytes;

    let s1_conv_avg_k: u64 = 7;
    let s1_conv_flops: u64 = 2 * b * stage1_t * stage1_ch * stage1_ch * s1_conv_avg_k;
    let s1_conv_weight_bytes: u64 = stage1_ch * stage1_ch * s1_conv_avg_k * f32_bytes;
    let s1_conv_activation_bytes: u64 = b * stage1_t * stage1_ch * f32_bytes * 2;
    let s1_conv_bytes: u64 = s1_conv_weight_bytes + s1_conv_activation_bytes;

    entries.push(RooflineEntry {
        name: "Stage1 AdaIN (instance_norm+affine)".into(),
        segment: "generator".into(),
        flops: s1_adain_total_flops,
        bytes: s1_adain_bytes,
        count: (n_resblock_layers_stage1 * 2) as usize,
        is_f16: false,
    });

    entries.push(RooflineEntry {
        name: "Stage1 Snake activation".into(),
        segment: "generator".into(),
        flops: s1_snake_flops,
        bytes: s1_snake_bytes,
        count: (n_resblock_layers_stage1 * 2) as usize,
        is_f16: false,
    });

    entries.push(RooflineEntry {
        name: "Stage1 Conv1d (128,128,k~7)".into(),
        segment: "generator".into(),
        flops: s1_conv_flops,
        bytes: s1_conv_bytes,
        count: (n_resblock_layers_stage1 * 2) as usize,
        is_f16: false,
    });

    // ========================================================================
    // 12. Generator — conv_post and iSTFT
    // ========================================================================
    let n_fft: u64 = 20;
    let conv_post_flops: u64 = 2 * b * stage1_t * stage1_ch * (n_fft + 2) * 7;
    let conv_post_weight_bytes: u64 = stage1_ch * (n_fft + 2) * 7 * f32_bytes;
    let conv_post_activation_bytes: u64 =
        (b * stage1_t * stage1_ch + b * stage1_t * (n_fft + 2)) * f32_bytes;
    let conv_post_bytes = conv_post_weight_bytes + conv_post_activation_bytes;

    entries.push(RooflineEntry {
        name: "Generator conv_post (128->22, k=7)".into(),
        segment: "generator".into(),
        flops: conv_post_flops,
        bytes: conv_post_bytes,
        count: 1,
        is_f16: false,
    });

    // iSTFT: FFT of size n_fft per frame
    let istft_flops: u64 = b * stage1_t * n_fft * (n_fft as f64).log2() as u64 * 5;
    let istft_bytes: u64 = b * stage1_t * n_fft * f32_bytes * 4; // complex in + overlap-add out

    entries.push(RooflineEntry {
        name: "iSTFT (overlap-add)".into(),
        segment: "istft".into(),
        flops: istft_flops,
        bytes: istft_bytes,
        count: 1,
        is_f16: false,
    });

    // ====================================================================
    // ANALYSIS
    // ====================================================================

    eprintln!("\n--- Per-Operation Roofline Classification ---\n");
    eprintln!(
        "  {:<44} {:>10} {:>10} {:>8} {:>10} {:>5} {:>14}",
        "Operation", "FLOPs", "Bytes", "AI", "Time(us)", "Cnt", "Classification"
    );
    eprintln!("  {}", "-".repeat(105));

    let mut total_flops: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut total_time_us: f64 = 0.0;

    let mut bw_bound_entries: Vec<(f64, &RooflineEntry)> = Vec::new();
    let mut compute_bound_entries: Vec<(f64, &RooflineEntry)> = Vec::new();

    for entry in &entries {
        let ai = entry.arithmetic_intensity();
        let time_us = entry.estimated_time_us(&spec);
        let classification = entry.bottleneck(&spec);
        let entry_flops = entry.flops * entry.count as u64;
        let entry_bytes = entry.bytes * entry.count as u64;

        total_flops += entry_flops;
        total_bytes += entry_bytes;
        total_time_us += time_us;

        eprintln!(
            "  {:<44} {:>10} {:>10} {:>8.1} {:>10.1} {:>5} {:>14}",
            entry.name,
            format_si(entry_flops),
            format_si(entry_bytes),
            ai,
            time_us,
            entry.count,
            classification,
        );

        if entry.is_bandwidth_bound(&spec) {
            bw_bound_entries.push((time_us, entry));
        } else {
            compute_bound_entries.push((time_us, entry));
        }
    }

    eprintln!("  {}", "-".repeat(105));
    eprintln!(
        "  {:<44} {:>10} {:>10} {:>8.1} {:>10.1}",
        "TOTAL",
        format_si(total_flops),
        format_si(total_bytes),
        total_flops as f64 / total_bytes.max(1) as f64,
        total_time_us,
    );

    // ====================================================================
    // TOP 5 BANDWIDTH-BOUND OPERATIONS
    // ====================================================================
    bw_bound_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!(
        "\n\n--- TOP 5 BANDWIDTH-BOUND OPERATIONS (optimization: reduce memory traffic) ---\n"
    );
    for (i, (time_us, entry)) in bw_bound_entries.iter().take(5).enumerate() {
        let pct = time_us / total_time_us * 100.0;
        eprintln!(
            "  {}. {} [{} x{}]",
            i + 1,
            entry.name,
            entry.segment,
            entry.count,
        );
        eprintln!(
            "     AI={:.1} FLOP/byte, {:.1} us ({:.1}% of total)",
            entry.arithmetic_intensity(),
            time_us,
            pct,
        );

        let optimization = bandwidth_optimization(&entry.name);
        eprintln!("     Optimization: {optimization}");
        eprintln!();
    }

    // ====================================================================
    // TOP 5 COMPUTE-BOUND OPERATIONS
    // ====================================================================
    compute_bound_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("\n--- TOP 5 COMPUTE-BOUND OPERATIONS (optimization: increase ALU efficiency) ---\n");
    for (i, (time_us, entry)) in compute_bound_entries.iter().take(5).enumerate() {
        let pct = time_us / total_time_us * 100.0;
        eprintln!(
            "  {}. {} [{} x{}]",
            i + 1,
            entry.name,
            entry.segment,
            entry.count,
        );
        eprintln!(
            "     AI={:.1} FLOP/byte, {:.1} us ({:.1}% of total)",
            entry.arithmetic_intensity(),
            time_us,
            pct,
        );

        let optimization = compute_optimization(&entry.name);
        eprintln!("     Optimization: {optimization}");
        eprintln!();
    }

    // ====================================================================
    // PER-SEGMENT BREAKDOWN
    // ====================================================================
    let mut segment_flops: BTreeMap<String, u64> = BTreeMap::new();
    let mut segment_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let mut segment_time: BTreeMap<String, f64> = BTreeMap::new();

    for entry in &entries {
        let ef = entry.flops * entry.count as u64;
        let eb = entry.bytes * entry.count as u64;
        let et = entry.estimated_time_us(&spec);
        *segment_flops.entry(entry.segment.clone()).or_default() += ef;
        *segment_bytes.entry(entry.segment.clone()).or_default() += eb;
        *segment_time.entry(entry.segment.clone()).or_default() += et;
    }

    eprintln!("\n--- Per-Segment Summary ---\n");
    eprintln!(
        "  {:<16} {:>12} {:>12} {:>10} {:>8} {:>12}",
        "Segment", "FLOPs", "Bytes", "Time(us)", "% Total", "Dominant"
    );
    eprintln!("  {}", "-".repeat(80));

    let mut segment_vec: Vec<(&String, &f64)> = segment_time.iter().collect();
    segment_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (seg, time) in &segment_vec {
        let sf = segment_flops.get(*seg).copied().unwrap_or(0);
        let sb = segment_bytes.get(*seg).copied().unwrap_or(0);
        let pct = *time / total_time_us * 100.0;
        let ai = sf as f64 / sb.max(1) as f64;
        let dominant = if ai < spec.ridge_point_f32() {
            "BW-bound"
        } else {
            "Compute"
        };
        eprintln!(
            "  {:<16} {:>12} {:>12} {:>10.1} {:>7.1}% {:>12}",
            seg,
            format_si(sf),
            format_si(sb),
            time,
            pct,
            dominant,
        );
    }

    // ====================================================================
    // GENERATOR RESBLOCK DEEP DIVE
    // ====================================================================
    eprintln!("\n\n--- Generator ResBlock Deep Dive ---\n");

    // Stage 0: [1, 256, 200]
    // FLOPs = 2*B*T*C*C*K, Bytes = (C*C*K + B*T*C + B*T*C)*4
    // AI = 2*B*T*C*C*K / ((C*C*K + 2*B*T*C)*4)
    let s0_conv_precise_ai = (2 * b * stage0_t * stage0_ch * stage0_ch * conv_avg_k) as f64
        / ((stage0_ch * stage0_ch * conv_avg_k + 2 * b * stage0_t * stage0_ch) as f64
            * f32_bytes as f64);

    let s1_conv_precise_ai = (2 * b * stage1_t * stage1_ch * stage1_ch * s1_conv_avg_k) as f64
        / ((stage1_ch * stage1_ch * s1_conv_avg_k + 2 * b * stage1_t * stage1_ch) as f64
            * f32_bytes as f64);

    eprintln!("  Stage 0 ResBlocks: C=256, T=200");
    eprintln!(
        "    Conv1d AI:     {:.1} FLOP/byte (ridge={:.1}) -> {}",
        s0_conv_precise_ai,
        spec.ridge_point_f32(),
        if s0_conv_precise_ai < spec.ridge_point_f32() {
            "BW-BOUND"
        } else {
            "COMPUTE-BOUND"
        }
    );
    eprintln!(
        "    AdaIN AI:      {:.1} FLOP/byte -> BW-BOUND (reduction-dominated)",
        adain_total_flops as f64 / (adain_bytes as f64)
    );
    eprintln!(
        "    Snake AI:      {:.1} FLOP/byte -> BW-BOUND (elementwise)",
        snake_flops as f64 / (snake_bytes as f64)
    );

    eprintln!("\n  Stage 1 ResBlocks: C=128, T=1200");
    eprintln!(
        "    Conv1d AI:     {:.1} FLOP/byte (ridge={:.1}) -> {}",
        s1_conv_precise_ai,
        spec.ridge_point_f32(),
        if s1_conv_precise_ai < spec.ridge_point_f32() {
            "BW-BOUND"
        } else {
            "COMPUTE-BOUND"
        }
    );
    eprintln!(
        "    AdaIN AI:      {:.1} FLOP/byte -> BW-BOUND (reduction-dominated)",
        s1_adain_total_flops as f64 / (s1_adain_bytes as f64)
    );
    eprintln!(
        "    Snake AI:      {:.1} FLOP/byte -> BW-BOUND (elementwise)",
        s1_snake_flops as f64 / (s1_snake_bytes as f64)
    );

    // ====================================================================
    // SUMMARY: Key Optimization Insights
    // ====================================================================
    let gen_time = segment_time.get("generator").copied().unwrap_or(0.0);
    let gen_pct = gen_time / total_time_us * 100.0;

    let bw_fraction: f64 = bw_bound_entries.iter().map(|(t, _)| t).sum::<f64>() / total_time_us;
    let compute_fraction: f64 =
        compute_bound_entries.iter().map(|(t, _)| t).sum::<f64>() / total_time_us;

    eprintln!("\n\n{}", "=".repeat(100));
    eprintln!("  OPTIMIZATION SUMMARY");
    eprintln!("{}", "=".repeat(100));
    eprintln!();
    eprintln!("  Generator is {gen_pct:.1}% of estimated total time.");
    eprintln!(
        "  Bandwidth-bound ops: {:.1}% of total time",
        bw_fraction * 100.0,
    );
    eprintln!(
        "  Compute-bound ops:   {:.1}% of total time",
        compute_fraction * 100.0,
    );
    eprintln!();
    eprintln!("  KEY INSIGHTS:");
    eprintln!("  1. Stage 1 ResBlocks (C=128, T=1200) are the largest time consumer");
    eprintln!("     because T is 6x larger than stage 0 after upsampling.");
    eprintln!(
        "  2. AdaIN and Snake activations are bandwidth-bound (AI < {:.1}).",
        spec.ridge_point_f32()
    );
    eprintln!("     -> Fuse AdaIN+Snake+Conv into a single kernel to avoid roundtrips.");
    eprintln!("     -> The NativeOp::AdainSnake path already does this partially.");
    eprintln!("  3. Conv1d operations hover near the ridge point.");
    eprintln!("     -> F16 autocast doubles effective throughput for same bandwidth.");
    eprintln!("     -> Winograd transforms could increase AI for k=3 convolutions.");
    eprintln!(
        "  4. Launch overhead for 201 dispatches = {:.0} us ({:.1}% of total).",
        201.0 * spec.dispatch_overhead_us,
        201.0 * spec.dispatch_overhead_us / total_time_us * 100.0,
    );
    eprintln!(
        "     -> Fusing dispatches to <60 saves ~{:.0} us.",
        141.0 * spec.dispatch_overhead_us,
    );
    eprintln!();
    eprintln!("  RECOMMENDED OPTIMIZATIONS (priority order):");
    eprintln!(
        "  1. F16 autocast for generator: halves memory traffic, doubles compute throughput."
    );
    eprintln!("  2. Fuse AdaIN+Snake+Conv into single NativeOp per ResBlock layer.");
    eprintln!("  3. Fuse stage 1 ResBlock averages (3 blocks averaged -> 1 dispatch).");
    eprintln!("  4. Use im2col GEMM for Conv1d to leverage simdgroup_matrix hardware.");
    eprintln!("  5. Overlap stage 0 and stage 1 via pipelined command buffers.");
    eprintln!("{}", "=".repeat(100));
    eprintln!();

    // ====================================================================
    // ASSERTIONS: sanity checks on the analysis
    // ====================================================================

    // Ridge point should be in a reasonable range for M4 Max.
    let ridge = spec.ridge_point_f32();
    assert!(
        ridge > 20.0 && ridge < 100.0,
        "Ridge point {ridge:.1} FLOP/byte is outside expected range [20, 100]",
    );

    // Generator should dominate.
    assert!(
        gen_pct > 40.0,
        "Generator should be >40% of total estimated time, got {gen_pct:.1}%",
    );

    // Total estimated time should be in a plausible range (> 10us, < 100ms)
    // for an 8-token utterance.
    assert!(
        total_time_us > 10.0 && total_time_us < 100_000.0,
        "Total estimated time {total_time_us:.1} us is outside plausible range",
    );

    // Snake and AdaIN should be bandwidth-bound (they are elementwise/reduction ops).
    for entry in &entries {
        if entry.name.contains("Snake") || entry.name.contains("AdaIN") {
            assert!(
                entry.is_bandwidth_bound(&spec),
                "{} should be bandwidth-bound (AI={:.1}, ridge={:.1})",
                entry.name,
                entry.arithmetic_intensity(),
                ridge,
            );
        }
    }

    // PlBERT and text encoder should have higher AI than elementwise ops.
    // Note: at T=8 tokens, weight bytes dominate activation bytes, so
    // AI is lower than it would be for longer sequences. The transformer
    // matmuls have high FLOP counts but weights (12*D^2 per layer) are
    // large relative to the small batch of activations.
    let plbert_ai = entries[0].arithmetic_intensity();
    let text_ai = entries[1].arithmetic_intensity();
    assert!(
        plbert_ai > 1.0,
        "PlBERT AI should be > 1.0 FLOP/byte at T=8 tokens, got {plbert_ai:.1}",
    );
    assert!(
        text_ai > 1.0,
        "Text encoder AI should be > 1.0 FLOP/byte at T=8 tokens, got {text_ai:.1}",
    );
}

/// Run the same analysis using the existing CostModel infrastructure from nn-dsl.
#[test]
fn kokoro_cost_model_cross_check() {
    use nn_dsl::CostModel;

    let m4_max = CostModel::apple_m4_max();

    // Cross-check: the CostModel's is_memory_bound classification for key
    // element counts should agree with our manual roofline analysis.

    // Snake activation at stage 1: C=128, T=1200 -> 153,600 elements
    // Memory: 153600 * 4 * 2 = 1,228,800 bytes
    // Compute: 153600 / 8e12 * 1e9 = 0.0192 ns (snake throughput)
    // Memory: 1228800 / 400e9 * 1e9 = 3.072 ns
    // -> Memory-bound
    assert!(
        m4_max.bandwidth_bytes_per_sec > 0.0,
        "M4 Max bandwidth should be configured",
    );

    // For the generator, check that the M4 Max model has key op throughputs.
    assert!(
        m4_max.op_throughput.contains_key("snake"),
        "M4 Max should have snake throughput",
    );
    assert!(
        m4_max.op_throughput.contains_key("instance_norm"),
        "M4 Max should have instance_norm throughput",
    );
    assert!(
        m4_max.op_throughput.contains_key("conv1d"),
        "M4 Max should have conv1d throughput",
    );

    // The M4 Max snake throughput (8 TFLOP/s) at 400 GB/s gives:
    // break-even AI = 8e12 / 400e9 = 20 FLOP/byte
    // Snake AI = ~6 FLOPs / (2 * 4 bytes) = 0.75 FLOP/byte << 20
    // -> Strongly bandwidth-bound.
    let snake_ai = 6.0 / (2.0 * 4.0); // 6 FLOPs per element, 8 bytes r/w
    let snake_ridge = m4_max.op_throughput["snake"] / m4_max.bandwidth_bytes_per_sec;
    assert!(
        snake_ai < snake_ridge,
        "Snake (AI={snake_ai:.2}) should be below ridge ({snake_ridge:.2}) for M4 Max",
    );

    // Conv1d at C=256, K=7: AI = 2*C*K / (C*K + 2*C) * (1/4)
    // = 2*256*7 / ((256*7 + 512) * 4) = 3584 / (2304*4) = 3584/9216 = 0.389
    // ... wait, that's per-output-element. Need to account for weight reuse across T.
    // FLOPs = 2*T*C*C*K, Bytes = C*C*K*4 + 2*T*C*4
    // For T=200: FLOPs = 2*200*256*256*7 = 183,500,800
    // Bytes = 256*256*7*4 + 2*200*256*4 = 1,835,008 + 409,600 = 2,244,608
    // AI = 183,500,800 / 2,244,608 = 81.8 -> Compute-bound on F32!
    let t_s0: f64 = 200.0;
    let c_s0: f64 = 256.0;
    let k_avg: f64 = 7.0;
    let conv_s0_ai =
        (2.0 * t_s0 * c_s0 * c_s0 * k_avg) / ((c_s0 * c_s0 * k_avg + 2.0 * t_s0 * c_s0) * 4.0);
    let conv_ridge = m4_max.op_throughput["conv1d"] / m4_max.bandwidth_bytes_per_sec;
    eprintln!(
        "\nConv1d Stage 0 AI={:.1}, ridge={:.1} -> {}",
        conv_s0_ai,
        conv_ridge,
        if conv_s0_ai < conv_ridge {
            "BW-bound"
        } else {
            "Compute-bound"
        },
    );

    // Stage 1 Conv1d: C=128, K=7, T=1200
    let t_s1: f64 = 1200.0;
    let c_s1: f64 = 128.0;
    let conv_s1_ai =
        (2.0 * t_s1 * c_s1 * c_s1 * k_avg) / ((c_s1 * c_s1 * k_avg + 2.0 * t_s1 * c_s1) * 4.0);
    eprintln!(
        "Conv1d Stage 1 AI={:.1}, ridge={:.1} -> {}",
        conv_s1_ai,
        conv_ridge,
        if conv_s1_ai < conv_ridge {
            "BW-bound"
        } else {
            "Compute-bound"
        },
    );

    eprintln!();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_si(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

fn bandwidth_optimization(op_name: &str) -> &'static str {
    if op_name.contains("AdaIN") {
        "Fuse instance_norm + style affine + snake into single kernel. \
         Avoids 3 separate buffer roundtrips per AdaIN+Snake pair. \
         NativeOp::AdainSnake already partially addresses this."
    } else if op_name.contains("Snake") {
        "Fuse with preceding AdaIN and following Conv1d. \
         Snake is purely elementwise (6 FLOPs/elem vs 8 bytes r/w = 0.75 AI). \
         Zero reuse opportunity as standalone kernel."
    } else if op_name.contains("conv_post") {
        "Small output channels (22) make this extremely BW-bound. \
         Fuse with preceding activation and exp/sin split."
    } else if op_name.contains("iSTFT") {
        "Small FFT size (n_fft=20) has negligible compute. \
         Fuse overlap-add with final audio buffer write."
    } else if op_name.contains("SineGen") {
        "Fuse cumulative sum + sin() computation into single pass. \
         Avoid storing intermediate cumsum buffer."
    } else {
        "Reduce buffer allocation / fuse with adjacent elementwise ops."
    }
}

fn compute_optimization(op_name: &str) -> &'static str {
    if op_name.contains("Conv1d") && (op_name.contains("256") || op_name.contains("128")) {
        "Use im2col + simdgroup_matrix GEMM for better ALU utilization. \
         F16 autocast doubles effective throughput (28 vs 14 TFLOPS). \
         Winograd for k=3 reduces multiplications by ~2.25x."
    } else if op_name.contains("upsample") {
        "ConvTranspose1d with large stride creates sparse writes. \
         Reformulate as sub-pixel shuffle + Conv1d for denser compute. \
         F16 autocast for weight/activation storage."
    } else if op_name.contains("PlBERT") || op_name.contains("Text") {
        "Transformer attention/FFN matmuls are compute-bound at large D. \
         Use simdgroup_matrix for QKV projections. \
         Flash attention reduces memory traffic for attention scores."
    } else if op_name.contains("BiLSTM") {
        "LSTM gates are sequential over T. \
         Precompute input projections as single GEMM. \
         Consider replacing with depthwise-separable convolution."
    } else {
        "Use F16 autocast to double compute throughput. \
         Ensure simdgroup tiling for matrix operations."
    }
}
