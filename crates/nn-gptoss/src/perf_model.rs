// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance modeling for gpt-oss-20b inference on Apple Silicon.
//!
//! Provides analytical models for predicting inference latency, throughput,
//! and memory bandwidth utilization. Based on roofline model analysis
//! for M4 Max (546 GB/s memory bandwidth, 54.6 TFLOPS FP32).

use crate::config::GptOssConfig;

/// Hardware characteristics for roofline performance modeling.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HardwareProfile {
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Peak FP32 compute throughput in TFLOPS.
    pub compute_tflops_f32: f64,
    /// Peak FP16/BF16 compute throughput in TFLOPS.
    pub compute_tflops_f16: f64,
    /// Maximum single buffer size in bytes.
    pub max_buffer_size_bytes: usize,
}

impl HardwareProfile {
    /// Create a new hardware profile with the given characteristics.
    #[must_use]
    pub fn new(
        memory_bandwidth_gbps: f64,
        compute_tflops_f32: f64,
        compute_tflops_f16: f64,
        max_buffer_size_bytes: usize,
    ) -> Self {
        Self {
            memory_bandwidth_gbps,
            compute_tflops_f32,
            compute_tflops_f16,
            max_buffer_size_bytes,
        }
    }

    /// Apple M4 Max: 546 GB/s bandwidth, 54.6 TFLOPS FP32, ~128GB unified.
    #[must_use]
    pub fn m4_max() -> Self {
        Self::new(546.0, 54.6, 109.2, 128 * 1024 * 1024 * 1024)
    }

    /// Apple M4 Pro: 273 GB/s bandwidth, 27.3 TFLOPS FP32, ~48GB unified.
    #[must_use]
    pub fn m4_pro() -> Self {
        Self::new(273.0, 27.3, 54.6, 48 * 1024 * 1024 * 1024)
    }

    /// Roofline ridge point: the arithmetic intensity (FLOP/byte) where
    /// compute and memory bandwidth intersect.
    ///
    /// Below this value, an operation is memory-bound. Above, compute-bound.
    /// Uses FP32 throughput by default.
    #[must_use]
    pub fn ridge_point_f32(&self) -> f64 {
        if self.memory_bandwidth_gbps <= 0.0 {
            return 0.0;
        }
        // TFLOPS / (GB/s) = (1e12 FLOP/s) / (1e9 B/s) = 1e3 FLOP/B
        (self.compute_tflops_f32 * 1e3) / self.memory_bandwidth_gbps
    }

    /// Ridge point using FP16/BF16 throughput.
    #[must_use]
    pub fn ridge_point_f16(&self) -> f64 {
        if self.memory_bandwidth_gbps <= 0.0 {
            return 0.0;
        }
        (self.compute_tflops_f16 * 1e3) / self.memory_bandwidth_gbps
    }
}

/// Per-operation performance characteristics.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OperationProfile {
    /// Total floating point operations.
    pub flops: u64,
    /// Total bytes read + written.
    pub memory_bytes: u64,
    /// Arithmetic intensity: flops / bytes (roofline metric).
    pub arithmetic_intensity: f64,
}

impl OperationProfile {
    /// Create a new operation profile from FLOP count and byte count.
    ///
    /// Arithmetic intensity is computed as `flops / memory_bytes`.
    /// If `memory_bytes` is 0, intensity is set to `f64::INFINITY`.
    #[must_use]
    pub fn new(flops: u64, memory_bytes: u64) -> Self {
        let arithmetic_intensity = if memory_bytes == 0 {
            if flops == 0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            flops as f64 / memory_bytes as f64
        };
        Self {
            flops,
            memory_bytes,
            arithmetic_intensity,
        }
    }

    /// Whether this operation is compute-bound on the given hardware.
    ///
    /// True when arithmetic intensity exceeds the roofline ridge point.
    #[must_use]
    pub fn is_compute_bound(&self, hw: &HardwareProfile) -> bool {
        self.arithmetic_intensity >= hw.ridge_point_f32()
    }

    /// Predicted latency in microseconds on the given hardware.
    ///
    /// Uses the roofline model: latency = max(compute_time, memory_time).
    /// Returns 0.0 for zero-work operations.
    #[must_use]
    pub fn predicted_latency_us(&self, hw: &HardwareProfile) -> f64 {
        if self.flops == 0 && self.memory_bytes == 0 {
            return 0.0;
        }
        let compute_time_us = if hw.compute_tflops_f32 > 0.0 {
            // TFLOPS = 1e12 FLOP/s, so time_s = flops / (tflops * 1e12)
            // time_us = flops / (tflops * 1e6)
            self.flops as f64 / (hw.compute_tflops_f32 * 1e6)
        } else {
            f64::INFINITY
        };
        let memory_time_us = if hw.memory_bandwidth_gbps > 0.0 {
            // GB/s = 1e9 B/s, so time_s = bytes / (gbps * 1e9)
            // time_us = bytes / (gbps * 1e3)
            self.memory_bytes as f64 / (hw.memory_bandwidth_gbps * 1e3)
        } else {
            f64::INFINITY
        };
        compute_time_us.max(memory_time_us)
    }
}

/// Performance bottleneck classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bottleneck {
    /// Limited by peak compute throughput.
    Compute,
    /// Limited by memory bandwidth.
    Memory,
    /// Limited by kernel dispatch overhead (very small operations).
    Dispatch,
}

/// Aggregated forward pass performance profile.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ForwardProfile {
    /// Total floating point operations across all operations.
    pub total_flops: u64,
    /// Total bytes read + written across all operations.
    pub total_memory_bytes: u64,
    /// Predicted total latency in microseconds.
    pub predicted_latency_us: f64,
    /// Predicted throughput in tokens per second.
    pub predicted_tokens_per_sec: f64,
    /// Memory bandwidth utilization (0.0 to 1.0).
    pub memory_bandwidth_utilization: f64,
    /// Compute utilization (0.0 to 1.0).
    pub compute_utilization: f64,
    /// Primary bottleneck.
    pub bottleneck: Bottleneck,
}

impl ForwardProfile {
    /// Create a new forward profile from constituent data.
    #[must_use]
    pub fn new(
        total_flops: u64,
        total_memory_bytes: u64,
        predicted_latency_us: f64,
        predicted_tokens_per_sec: f64,
        memory_bandwidth_utilization: f64,
        compute_utilization: f64,
        bottleneck: Bottleneck,
    ) -> Self {
        Self {
            total_flops,
            total_memory_bytes,
            predicted_latency_us,
            predicted_tokens_per_sec,
            memory_bandwidth_utilization,
            compute_utilization,
            bottleneck,
        }
    }
}

// -- Profiling functions ------------------------------------------------------

/// Profile a single multi-head attention operation.
///
/// FLOPs: For QK^T matmul + softmax + AV matmul:
///   QK^T: 2 * batch * heads * seq_len * total_kv_len * head_dim
///   AV:   2 * batch * heads * seq_len * total_kv_len * head_dim
///   (softmax is negligible relative to matmuls)
///
/// Memory: Q + K + V + output read/write
///   Q: batch * heads * seq_len * head_dim
///   K: batch * kv_heads * total_kv_len * head_dim
///   V: batch * kv_heads * total_kv_len * head_dim
///   O: batch * heads * seq_len * head_dim
///   Scores: batch * heads * seq_len * total_kv_len
#[must_use]
pub fn profile_attention(
    cfg: &GptOssConfig,
    seq_len: usize,
    cached_len: usize,
) -> OperationProfile {
    let batch: u64 = 1;
    let heads = cfg.num_attention_heads as u64;
    let kv_heads = cfg.num_key_value_heads as u64;
    let hd = cfg.head_dim as u64;
    let s = seq_len as u64;
    let total_kv = (seq_len + cached_len) as u64;
    let bpe: u64 = 4; // F32 bytes per element

    // QK^T + AV matmuls: 2 * 2 * batch * heads * seq * total_kv * head_dim
    let matmul_flops = 4_u64
        .saturating_mul(batch)
        .saturating_mul(heads)
        .saturating_mul(s)
        .saturating_mul(total_kv)
        .saturating_mul(hd);

    // Memory: Q + K + V + Output + attention scores
    let q_bytes = batch
        .saturating_mul(heads)
        .saturating_mul(s)
        .saturating_mul(hd)
        .saturating_mul(bpe);
    let k_bytes = batch
        .saturating_mul(kv_heads)
        .saturating_mul(total_kv)
        .saturating_mul(hd)
        .saturating_mul(bpe);
    let v_bytes = k_bytes; // same shape as K
    let o_bytes = q_bytes; // same shape as Q
    let score_bytes = batch
        .saturating_mul(heads)
        .saturating_mul(s)
        .saturating_mul(total_kv)
        .saturating_mul(bpe);
    let mem = q_bytes
        .saturating_add(k_bytes)
        .saturating_add(v_bytes)
        .saturating_add(o_bytes)
        .saturating_add(score_bytes);

    OperationProfile::new(matmul_flops, mem)
}

/// Profile a single MoE block (all active experts for all tokens).
///
/// FLOPs per token per expert (SwiGLU gate_up + down):
///   gate_up: 2 * hidden * intermediate * 2 (gate and up are fused)
///   down:    2 * intermediate * hidden
///
/// Total: seq_len * top_k * (4 * hidden * intermediate + 2 * intermediate * hidden)
///      = seq_len * top_k * 6 * hidden * intermediate
///
/// Memory: expert weight reads + activation reads/writes.
#[must_use]
pub fn profile_moe_block(cfg: &GptOssConfig, seq_len: usize) -> OperationProfile {
    let s = seq_len as u64;
    let top_k = cfg.experts_per_token as u64;
    let h = cfg.hidden_size as u64;
    let inter = cfg.intermediate_size as u64;
    let bpe: u64 = 4; // F32

    // gate_up_proj: 2 matmuls of [h, inter] each -> 2 * 2 * h * inter per token per expert
    // down_proj: 1 matmul of [inter, h] -> 2 * inter * h per token per expert
    // Total per token per expert = 4*h*inter + 2*inter*h = 6*h*inter
    let flops_per_token_per_expert = 6_u64.saturating_mul(h).saturating_mul(inter);
    let total_flops = s
        .saturating_mul(top_k)
        .saturating_mul(flops_per_token_per_expert);

    // Memory: read top_k expert weights per token
    // gate_up_proj weight: [h, 2*inter] per expert
    // down_proj weight: [inter, h] per expert
    // But with top-k routing, we read k different expert weight sets.
    // Per expert: h*2*inter + inter*h = 3*h*inter elements
    let expert_weight_elems = 3_u64.saturating_mul(h).saturating_mul(inter);
    let expert_weight_bytes = top_k
        .saturating_mul(expert_weight_elems)
        .saturating_mul(bpe);

    // Activation reads/writes: input [s, h] + output [s, h] + intermediates
    let activation_bytes = 2_u64
        .saturating_mul(s)
        .saturating_mul(h)
        .saturating_mul(bpe);

    // Router: [s, num_experts] matmul
    let router_flops = 2_u64
        .saturating_mul(s)
        .saturating_mul(h)
        .saturating_mul(cfg.num_local_experts as u64);

    let mem = expert_weight_bytes.saturating_add(activation_bytes);
    let flops = total_flops.saturating_add(router_flops);

    OperationProfile::new(flops, mem)
}

/// Profile a linear projection (matmul): [seq_len, in_dim] x [in_dim, out_dim].
#[must_use]
pub fn profile_linear(seq_len: usize, in_dim: usize, out_dim: usize) -> OperationProfile {
    let s = seq_len as u64;
    let m = in_dim as u64;
    let n = out_dim as u64;
    let bpe: u64 = 4;

    // FLOPs: 2 * s * m * n (multiply-accumulate)
    let flops = 2_u64.saturating_mul(s).saturating_mul(m).saturating_mul(n);
    // Memory: weight [m, n] + input [s, m] + output [s, n], all * bpe
    let weight_bytes = m.saturating_mul(n).saturating_mul(bpe);
    let input_bytes = s.saturating_mul(m).saturating_mul(bpe);
    let output_bytes = s.saturating_mul(n).saturating_mul(bpe);
    let mem = weight_bytes
        .saturating_add(input_bytes)
        .saturating_add(output_bytes);

    OperationProfile::new(flops, mem)
}

/// Profile the embedding lookup.
#[must_use]
pub fn profile_embedding(cfg: &GptOssConfig, seq_len: usize) -> OperationProfile {
    let bpe: u64 = 4;
    // Embedding is a gather: seq_len rows of hidden_size each
    let bytes = (seq_len as u64)
        .saturating_mul(cfg.hidden_size as u64)
        .saturating_mul(bpe);
    // Embedding lookup has ~0 FLOPs (just memory reads)
    OperationProfile::new(0, bytes)
}

/// Profile RMS normalization: [seq_len, hidden_size].
#[must_use]
pub fn profile_rms_norm(cfg: &GptOssConfig, seq_len: usize) -> OperationProfile {
    let s = seq_len as u64;
    let h = cfg.hidden_size as u64;
    let bpe: u64 = 4;

    // FLOPs: per element: square + sum-reduce + rsqrt + multiply + scale
    // ~5 * s * h (approximate)
    let flops = 5_u64.saturating_mul(s).saturating_mul(h);
    // Memory: input + output + weight
    let mem = 2_u64
        .saturating_mul(s)
        .saturating_mul(h)
        .saturating_mul(bpe)
        .saturating_add(h.saturating_mul(bpe));

    OperationProfile::new(flops, mem)
}

/// Profile a complete forward pass through all model layers.
///
/// Aggregates: embedding + 24 layers * (attention + MoE + 2 norms) + final norm + lm_head.
#[must_use]
pub fn profile_full_forward(
    cfg: &GptOssConfig,
    seq_len: usize,
    cached_len: usize,
) -> ForwardProfile {
    let hw = HardwareProfile::m4_max();
    profile_full_forward_on(cfg, seq_len, cached_len, &hw)
}

/// Profile a complete forward pass on specific hardware.
#[must_use]
pub fn profile_full_forward_on(
    cfg: &GptOssConfig,
    seq_len: usize,
    cached_len: usize,
    hw: &HardwareProfile,
) -> ForwardProfile {
    let mut total_flops: u64 = 0;
    let mut total_memory_bytes: u64 = 0;
    let mut total_latency_us: f64 = 0.0;

    // Helper to accumulate an operation profile
    let mut add_op = |op: OperationProfile| {
        total_flops = total_flops.saturating_add(op.flops);
        total_memory_bytes = total_memory_bytes.saturating_add(op.memory_bytes);
        total_latency_us += op.predicted_latency_us(hw);
    };

    // 1. Embedding lookup
    add_op(profile_embedding(cfg, seq_len));

    // 2. Per-layer: input_norm + Q/K/V projections + attention + O proj + post_norm + MoE
    for _ in 0..cfg.num_hidden_layers {
        // Input layernorm
        add_op(profile_rms_norm(cfg, seq_len));

        // Q, K, V projections
        let attn_dim = cfg.attn_dim();
        let kv_dim = cfg.kv_dim();
        add_op(profile_linear(seq_len, cfg.hidden_size, attn_dim));
        add_op(profile_linear(seq_len, cfg.hidden_size, kv_dim));
        add_op(profile_linear(seq_len, cfg.hidden_size, kv_dim));

        // Attention
        add_op(profile_attention(cfg, seq_len, cached_len));

        // O projection
        add_op(profile_linear(seq_len, attn_dim, cfg.hidden_size));

        // Post-attention layernorm
        add_op(profile_rms_norm(cfg, seq_len));

        // MoE block
        add_op(profile_moe_block(cfg, seq_len));
    }

    // 3. Final RMS norm
    add_op(profile_rms_norm(cfg, seq_len));

    // 4. lm_head projection
    add_op(profile_linear(seq_len, cfg.hidden_size, cfg.vocab_size));

    // Compute utilization metrics
    let tokens_per_sec = if total_latency_us > 0.0 {
        (seq_len as f64) / (total_latency_us / 1e6)
    } else {
        0.0
    };

    // Memory bandwidth utilization: actual bytes / (bandwidth * time)
    let achievable_bytes = hw.memory_bandwidth_gbps * 1e3 * total_latency_us; // GB/s * 1e3 = MB/us -> bytes
    let bw_util = if achievable_bytes > 0.0 {
        (total_memory_bytes as f64 / achievable_bytes).min(1.0)
    } else {
        0.0
    };

    // Compute utilization: actual FLOPs / (peak FLOPs * time)
    let achievable_flops = hw.compute_tflops_f32 * 1e6 * total_latency_us; // TFLOPS * 1e6 = MFLOP/us -> FLOPs
    let compute_util = if achievable_flops > 0.0 {
        (total_flops as f64 / achievable_flops).min(1.0)
    } else {
        0.0
    };

    // Determine bottleneck
    let overall_intensity = if total_memory_bytes > 0 {
        total_flops as f64 / total_memory_bytes as f64
    } else {
        f64::INFINITY
    };

    // Dispatch overhead threshold: if total predicted latency is very low,
    // dispatch overhead likely dominates.
    let dispatch_overhead_threshold_us = 50.0; // 50us minimum for meaningful work
    let bottleneck = if total_latency_us < dispatch_overhead_threshold_us {
        Bottleneck::Dispatch
    } else if overall_intensity < hw.ridge_point_f32() {
        Bottleneck::Memory
    } else {
        Bottleneck::Compute
    };

    ForwardProfile::new(
        total_flops,
        total_memory_bytes,
        total_latency_us,
        tokens_per_sec,
        bw_util,
        compute_util,
        bottleneck,
    )
}

/// Estimate the speedup from MXFP4 quantization of expert weights.
///
/// MXFP4 reduces expert weight reads by ~3.7x (from 2 bytes BF16 to ~0.53 bytes).
/// Non-expert operations are unchanged. Returns the predicted speedup factor.
#[must_use]
pub fn estimate_mxfp4_speedup(cfg: &GptOssConfig, seq_len: usize, cached_len: usize) -> f64 {
    let hw = HardwareProfile::m4_max();
    let baseline = profile_full_forward_on(cfg, seq_len, cached_len, &hw);
    if baseline.predicted_latency_us <= 0.0 {
        return 1.0;
    }

    // MoE expert weight reads constitute the bulk of memory traffic.
    // With MXFP4, expert weight bytes reduce by factor of ~3.77 (BF16->MXFP4).
    // Estimate MoE memory savings.
    let moe_profile = profile_moe_block(cfg, seq_len);
    let moe_latency = moe_profile.predicted_latency_us(&hw);
    let moe_fraction = moe_latency * (cfg.num_hidden_layers as f64) / baseline.predicted_latency_us;

    // MXFP4 reduces memory portion of MoE by ~3.77x
    // But MoE might be compute-bound for large seq_len, so cap the benefit
    let mxfp4_compression = 3.77;
    let moe_speedup = if moe_profile.is_compute_bound(&hw) {
        1.0 // compute-bound: no benefit from smaller weights
    } else {
        mxfp4_compression
    };

    let new_moe_fraction = moe_fraction / moe_speedup;
    let non_moe_fraction = 1.0 - moe_fraction;
    let total_fraction = non_moe_fraction + new_moe_fraction;

    if total_fraction > 0.0 {
        1.0 / total_fraction
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_20b() -> GptOssConfig {
        GptOssConfig::gptoss_20b()
    }

    // -- HardwareProfile tests ------------------------------------------------

    #[test]
    fn test_hardware_profile_m4_max() {
        let hw = HardwareProfile::m4_max();
        assert!((hw.memory_bandwidth_gbps - 546.0).abs() < 1e-9);
        assert!((hw.compute_tflops_f32 - 54.6).abs() < 1e-9);
        assert!((hw.compute_tflops_f16 - 109.2).abs() < 1e-9);
        assert_eq!(hw.max_buffer_size_bytes, 128 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_hardware_profile_m4_pro() {
        let hw = HardwareProfile::m4_pro();
        assert!((hw.memory_bandwidth_gbps - 273.0).abs() < 1e-9);
        assert!((hw.compute_tflops_f32 - 27.3).abs() < 1e-9);
    }

    #[test]
    fn test_ridge_point_m4_max() {
        let hw = HardwareProfile::m4_max();
        let ridge = hw.ridge_point_f32();
        // 54.6 * 1000 / 546 = 100 FLOP/byte
        assert!(
            (ridge - 100.0).abs() < 1e-6,
            "ridge point should be ~100, got {ridge}"
        );
    }

    #[test]
    fn test_ridge_point_f16_double_f32() {
        let hw = HardwareProfile::m4_max();
        let r32 = hw.ridge_point_f32();
        let r16 = hw.ridge_point_f16();
        // FP16 throughput is 2x FP32, so ridge point is 2x
        assert!((r16 - 2.0 * r32).abs() < 1e-6);
    }

    #[test]
    fn test_ridge_point_zero_bandwidth() {
        let hw = HardwareProfile::new(0.0, 54.6, 109.2, 1024);
        assert_eq!(hw.ridge_point_f32(), 0.0);
    }

    // -- OperationProfile tests -----------------------------------------------

    #[test]
    fn test_operation_profile_arithmetic_intensity() {
        let op = OperationProfile::new(1000, 100);
        assert!((op.arithmetic_intensity - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_operation_profile_zero_bytes() {
        let op = OperationProfile::new(1000, 0);
        assert!(op.arithmetic_intensity.is_infinite());
    }

    #[test]
    fn test_operation_profile_zero_both() {
        let op = OperationProfile::new(0, 0);
        assert!((op.arithmetic_intensity - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_compute_bound_high_intensity() {
        let hw = HardwareProfile::m4_max();
        // Ridge point is ~100 FLOP/byte. Intensity 200 -> compute bound.
        let op = OperationProfile::new(200_000, 1000);
        assert!(op.is_compute_bound(&hw));
    }

    #[test]
    fn test_is_memory_bound_low_intensity() {
        let hw = HardwareProfile::m4_max();
        // Ridge point is ~100 FLOP/byte. Intensity 1 -> memory bound.
        let op = OperationProfile::new(1000, 1000);
        assert!(!op.is_compute_bound(&hw));
    }

    #[test]
    fn test_predicted_latency_zero_work() {
        let hw = HardwareProfile::m4_max();
        let op = OperationProfile::new(0, 0);
        assert!((op.predicted_latency_us(&hw) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_predicted_latency_positive() {
        let hw = HardwareProfile::m4_max();
        let op = OperationProfile::new(1_000_000, 1_000_000);
        assert!(op.predicted_latency_us(&hw) > 0.0);
    }

    // -- Attention profiling --------------------------------------------------

    #[test]
    fn test_attention_flops_nonzero() {
        let cfg = cfg_20b();
        let prof = profile_attention(&cfg, 128, 0);
        assert!(prof.flops > 0, "attention FLOPs must be > 0");
        assert!(prof.memory_bytes > 0, "attention memory must be > 0");
    }

    #[test]
    fn test_attention_flops_scale_with_seq_len() {
        let cfg = cfg_20b();
        let prof_short = profile_attention(&cfg, 64, 0);
        let prof_long = profile_attention(&cfg, 256, 0);
        // Quadratic scaling with seq_len (since both Q and KV grow)
        assert!(prof_long.flops > prof_short.flops);
    }

    #[test]
    fn test_attention_cached_increases_flops() {
        let cfg = cfg_20b();
        let prof_no_cache = profile_attention(&cfg, 1, 0);
        let prof_cached = profile_attention(&cfg, 1, 1024);
        assert!(prof_cached.flops > prof_no_cache.flops);
    }

    // -- MoE profiling --------------------------------------------------------

    #[test]
    fn test_moe_flops_nonzero() {
        let cfg = cfg_20b();
        let prof = profile_moe_block(&cfg, 128);
        assert!(prof.flops > 0, "MoE FLOPs must be > 0");
        assert!(prof.memory_bytes > 0, "MoE memory must be > 0");
    }

    #[test]
    fn test_moe_flops_scale_with_seq_len() {
        let cfg = cfg_20b();
        let prof_1 = profile_moe_block(&cfg, 1);
        let prof_128 = profile_moe_block(&cfg, 128);
        // Linear scaling with seq_len
        assert!(prof_128.flops > prof_1.flops);
    }

    // -- Full forward profiling -----------------------------------------------

    #[test]
    fn test_full_forward_nonzero() {
        let cfg = cfg_20b();
        let prof = profile_full_forward(&cfg, 1, 0);
        assert!(prof.total_flops > 0);
        assert!(prof.total_memory_bytes > 0);
        assert!(prof.predicted_latency_us > 0.0);
        assert!(prof.predicted_tokens_per_sec > 0.0);
    }

    #[test]
    fn test_prefill_slower_than_decode() {
        let cfg = cfg_20b();
        let decode = profile_full_forward(&cfg, 1, 512);
        let prefill = profile_full_forward(&cfg, 128, 0);
        assert!(
            prefill.predicted_latency_us >= decode.predicted_latency_us,
            "prefill ({:.1}us) should be >= decode ({:.1}us)",
            prefill.predicted_latency_us,
            decode.predicted_latency_us,
        );
    }

    #[test]
    fn test_bandwidth_utilization_bounded() {
        let cfg = cfg_20b();
        let prof = profile_full_forward(&cfg, 1, 0);
        assert!(
            prof.memory_bandwidth_utilization >= 0.0 && prof.memory_bandwidth_utilization <= 1.0,
            "bw utilization {} out of [0,1]",
            prof.memory_bandwidth_utilization,
        );
    }

    #[test]
    fn test_compute_utilization_bounded() {
        let cfg = cfg_20b();
        let prof = profile_full_forward(&cfg, 1, 0);
        assert!(
            prof.compute_utilization >= 0.0 && prof.compute_utilization <= 1.0,
            "compute utilization {} out of [0,1]",
            prof.compute_utilization,
        );
    }

    #[test]
    fn test_single_token_decode_memory_bound() {
        let cfg = cfg_20b();
        // Single-token decode on large models is typically memory-bound
        let prof = profile_full_forward(&cfg, 1, 512);
        assert_eq!(
            prof.bottleneck,
            Bottleneck::Memory,
            "single-token decode should be memory-bound"
        );
    }

    #[test]
    fn test_mxfp4_speedup_positive() {
        let cfg = cfg_20b();
        let speedup = estimate_mxfp4_speedup(&cfg, 1, 512);
        assert!(
            speedup >= 1.0,
            "MXFP4 speedup should be >= 1.0, got {speedup}"
        );
    }

    #[test]
    fn test_mxfp4_speedup_reasonable_range() {
        let cfg = cfg_20b();
        let speedup = estimate_mxfp4_speedup(&cfg, 1, 512);
        // MXFP4 can't speed up more than ~3.77x (compression ratio)
        // and typically less since not all ops benefit
        assert!(
            speedup <= 4.0,
            "MXFP4 speedup should be <= 4.0, got {speedup}"
        );
    }

    #[test]
    fn test_m4_pro_slower_than_m4_max() {
        let cfg = cfg_20b();
        let hw_max = HardwareProfile::m4_max();
        let hw_pro = HardwareProfile::m4_pro();
        let prof_max = profile_full_forward_on(&cfg, 1, 512, &hw_max);
        let prof_pro = profile_full_forward_on(&cfg, 1, 512, &hw_pro);
        assert!(
            prof_pro.predicted_latency_us > prof_max.predicted_latency_us,
            "M4 Pro should be slower than M4 Max"
        );
    }

    #[test]
    fn test_embedding_zero_flops() {
        let cfg = cfg_20b();
        let prof = profile_embedding(&cfg, 128);
        assert_eq!(prof.flops, 0, "embedding has no FLOPs, just memory");
        assert!(prof.memory_bytes > 0);
    }

    #[test]
    fn test_linear_profile_basic() {
        let prof = profile_linear(1, 2880, 4096);
        // 2 * 1 * 2880 * 4096 = 23,592,960
        assert_eq!(prof.flops, 2 * 2880 * 4096);
        assert!(prof.memory_bytes > 0);
    }
}
