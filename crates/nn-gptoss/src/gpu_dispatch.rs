// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU dispatch configuration and helpers for gpt-oss-20b inference.
//!
//! Provides Metal-specific threadgroup sizing, device capability checks, and
//! dispatch helpers for the two GPU-intensive operations in gpt-oss:
//!
//! - **GQA attention**: 64 Q heads, 8 KV heads, head_dim=64. The repeat-KV
//!   step (8x) and scaled dot-product attention dominate decode latency.
//! - **MoE expert dispatch**: 32 experts, top-4 routing. Batched expert FFN
//!   (gate_up + clamped SwiGLU + down) with scatter-add accumulation.
//!
//! The dispatch configuration selects threadgroup sizes optimized for Apple
//! Silicon (M4 Max: 128 EU, 32 threads/SIMD-group). These are also safe
//! defaults for M1/M2/M3 families.
//!
//! # Architecture
//!
//! This module does NOT generate MSL or launch Metal kernels directly. It
//! provides configuration and routing logic consumed by:
//! - `nn-metal` tensor dispatch (DynTensor ops route through registered GPU backend)
//! - `moe_dispatch::fused_moe_forward` (batched expert dispatch on GPU)
//! - Future compiled model registry (like `compiled_kokoro_registry.rs`)

use nn_core::{Device, Result};

use crate::GptOssConfig;
use crate::GptOssError;

// ---------------------------------------------------------------------------
// GPU dispatch configuration
// ---------------------------------------------------------------------------

/// Metal-specific dispatch configuration for gpt-oss inference.
///
/// Controls threadgroup sizes and batch parameters for the two primary GPU
/// workloads (attention and MoE). Values are tuned for Apple Silicon M4 Max
/// but are safe for all Metal 3 / Metal 2 devices.
#[derive(Debug, Clone)]
pub(crate) struct GptOssGpuConfig {
    /// Threadgroup width for attention GEMM dispatch (Q @ K^T, attn @ V).
    ///
    /// Must be a power of 2. Default 256 = 8 SIMD-groups on Apple Silicon.
    /// Matches the simdgroup_matrix tile dispatch in nn-metal.
    pub(crate) attention_threadgroup_size: u32,

    /// Threadgroup width for MoE expert FFN dispatch (gate_up, down matmuls).
    ///
    /// Must be a power of 2. Default 256 balances occupancy across 32 experts.
    pub(crate) moe_threadgroup_size: u32,

    /// Threadgroup width for elementwise ops (RMSNorm, SiLU, clamp, add).
    ///
    /// Must be a power of 2. Default 256 for good SIMD utilization.
    pub(crate) elementwise_threadgroup_size: u32,

    /// Maximum batch size for prefill (prompt processing).
    ///
    /// Larger batches amortize dispatch overhead but increase memory pressure.
    /// Default 1 for single-sequence inference; increase for batch serving.
    pub(crate) max_batch_size: u32,

    /// Metal buffer offset alignment in bytes.
    ///
    /// All buffer offsets passed to Metal dispatch must be multiples of this.
    /// Apple Silicon requires 16-byte alignment for optimal performance.
    pub(crate) buffer_alignment: u32,
}

impl GptOssGpuConfig {
    /// Default configuration for M4 Max (also safe for M1/M2/M3).
    ///
    /// Threadgroup sizes are powers of 2 matching Apple Silicon SIMD-group
    /// width (32 threads). 256 = 8 SIMD-groups provides good occupancy
    /// without exceeding the 1024-thread-per-threadgroup Metal limit.
    #[must_use]
    pub(crate) fn m4_max() -> Self {
        Self {
            attention_threadgroup_size: 256,
            moe_threadgroup_size: 256,
            elementwise_threadgroup_size: 256,
            max_batch_size: 1,
            buffer_alignment: 16,
        }
    }

    /// Configuration for smaller Apple Silicon chips (M1/M2 base).
    ///
    /// Uses 128-thread threadgroups (4 SIMD-groups) for lower occupancy
    /// pressure on devices with fewer execution units.
    #[must_use]
    pub(crate) fn apple_silicon_base() -> Self {
        Self {
            attention_threadgroup_size: 128,
            moe_threadgroup_size: 128,
            elementwise_threadgroup_size: 128,
            max_batch_size: 1,
            buffer_alignment: 16,
        }
    }

    /// Validate configuration invariants.
    pub(crate) fn validate(&self) -> Result<()> {
        validate_threadgroup_size(self.attention_threadgroup_size, "attention")?;
        validate_threadgroup_size(self.moe_threadgroup_size, "moe")?;
        validate_threadgroup_size(self.elementwise_threadgroup_size, "elementwise")?;

        if self.max_batch_size == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "max_batch_size must be > 0".into(),
            }
            .into());
        }

        if self.buffer_alignment == 0 || !self.buffer_alignment.is_power_of_two() {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "buffer_alignment must be a positive power of 2, got {}",
                    self.buffer_alignment
                ),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for GptOssGpuConfig {
    fn default() -> Self {
        Self::m4_max()
    }
}

// ---------------------------------------------------------------------------
// Device capability checks
// ---------------------------------------------------------------------------

/// Returns `true` when the device supports GPU-accelerated gpt-oss inference.
///
/// Currently returns `true` for Metal devices (Apple Silicon). CUDA and Vulkan
/// support will follow the same pattern when backends are wired.
#[must_use]
pub(crate) fn should_use_gpu(device: &Device) -> bool {
    device.is_gpu()
}

/// Select the optimal GPU configuration for the given device.
///
/// Returns `None` for CPU devices. For Metal devices, selects M4 Max config
/// as the safe default (works on all Apple Silicon).
#[must_use]
pub(crate) fn select_gpu_config(device: &Device) -> Option<GptOssGpuConfig> {
    if device.is_gpu() {
        Some(GptOssGpuConfig::m4_max())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Attention dispatch helpers
// ---------------------------------------------------------------------------

/// Compute the dispatch grid dimensions for GQA attention on Metal.
///
/// Returns `(grid_size, threadgroup_size)` as `[u32; 3]` pairs for a 1D
/// dispatch covering all elements in the attention score matrix.
///
/// The attention score matrix has shape `[batch, num_heads, seq_q, seq_kv]`.
/// Total elements = batch * num_heads * seq_q * seq_kv.
///
/// # Errors
///
/// Returns an error if the total element count overflows `u32`.
pub(crate) fn dispatch_attention_grid(
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    config: &GptOssGpuConfig,
) -> Result<([u32; 3], [u32; 3])> {
    let total = checked_u32_product(&[batch, num_heads, seq_q, seq_kv])?;
    let tg = config.attention_threadgroup_size;
    let grid_x = ceil_div(total, tg) * tg;

    Ok(([grid_x, 1, 1], [tg, 1, 1]))
}

/// Compute the dispatch grid for a single GQA matmul (Q @ K^T or attn @ V).
///
/// For the Q @ K^T matmul: M=seq_q, N=seq_kv, dispatched per (batch, head).
/// For the attn @ V matmul: M=seq_q, N=head_dim, dispatched per (batch, head).
///
/// Returns `(grid_size, threadgroup_size)` as `[u32; 3]`.
///
/// # Errors
///
/// Returns an error if the element count overflows `u32`.
pub(crate) fn dispatch_attention_matmul_grid(
    batch: u32,
    num_heads: u32,
    m: u32,
    n: u32,
    config: &GptOssGpuConfig,
) -> Result<([u32; 3], [u32; 3])> {
    let total = checked_u32_product(&[batch, num_heads, m, n])?;
    let tg = config.attention_threadgroup_size;
    let grid_x = ceil_div(total, tg) * tg;

    Ok(([grid_x, 1, 1], [tg, 1, 1]))
}

// ---------------------------------------------------------------------------
// MoE dispatch helpers
// ---------------------------------------------------------------------------

/// Compute the dispatch grid for batched MoE expert FFN on Metal.
///
/// Each expert processes `num_tokens_for_expert` tokens through a 2-matmul
/// FFN (gate_up + down). The grid covers all output elements for one expert.
///
/// Returns `(grid_size, threadgroup_size)` as `[u32; 3]`.
///
/// # Errors
///
/// Returns an error if the element count overflows `u32`.
pub(crate) fn dispatch_moe_expert_grid(
    num_tokens: u32,
    output_dim: u32,
    config: &GptOssGpuConfig,
) -> Result<([u32; 3], [u32; 3])> {
    let total = num_tokens
        .checked_mul(output_dim)
        .ok_or_else(|| GptOssError::InvalidConfig {
            reason: format!("MoE dispatch grid overflow: {num_tokens} * {output_dim}"),
        })?;
    let tg = config.moe_threadgroup_size;
    let grid_x = ceil_div(total, tg) * tg;

    Ok(([grid_x, 1, 1], [tg, 1, 1]))
}

/// Compute dispatch grid for the MoE scatter-add accumulation step.
///
/// After all experts produce weighted outputs, scatter-add combines them
/// into the final output tensor of shape `[total_tokens, hidden_size]`.
///
/// Returns `(grid_size, threadgroup_size)` as `[u32; 3]`.
///
/// # Errors
///
/// Returns an error if the element count overflows `u32`.
pub(crate) fn dispatch_moe_scatter_grid(
    total_tokens: u32,
    hidden_size: u32,
    config: &GptOssGpuConfig,
) -> Result<([u32; 3], [u32; 3])> {
    let total =
        total_tokens
            .checked_mul(hidden_size)
            .ok_or_else(|| GptOssError::InvalidConfig {
                reason: format!("MoE scatter grid overflow: {total_tokens} * {hidden_size}"),
            })?;
    let tg = config.elementwise_threadgroup_size;
    let grid_x = ceil_div(total, tg) * tg;

    Ok(([grid_x, 1, 1], [tg, 1, 1]))
}

// ---------------------------------------------------------------------------
// Utility: buffer offset alignment
// ---------------------------------------------------------------------------

/// Align a byte offset up to the Metal buffer alignment boundary.
///
/// Metal requires buffer offsets to be multiples of 16 bytes on Apple Silicon.
/// This rounds `offset` up to the next multiple of `config.buffer_alignment`.
///
/// # Panics
///
/// Panics if `config.buffer_alignment` is 0.
#[must_use]
pub(crate) fn align_buffer_offset(offset: usize, config: &GptOssGpuConfig) -> usize {
    let align = config.buffer_alignment as usize;
    (offset + align - 1) & !(align - 1)
}

/// Estimate total GPU buffer memory for one decoder layer forward pass.
///
/// Returns the byte count for activation buffers (not weights) needed during
/// a single decoder layer forward. Used by the memory planner to pre-allocate
/// arena buffers.
///
/// Activation tensors per layer:
/// - Attention: Q, K, V projections + attention scores + output
/// - MoE: router logits + per-expert intermediates + scatter output
#[must_use]
pub(crate) fn estimate_layer_activation_bytes(
    cfg: &GptOssConfig,
    seq_len: usize,
    batch_size: usize,
) -> usize {
    let bpe = 4; // f32 = 4 bytes (DynTensor float storage invariant)
    let h = cfg.hidden_size;
    let ad = cfg.attn_dim();
    let kvd = cfg.kv_dim();
    let ne = cfg.num_local_experts;
    let inter = cfg.intermediate_size;
    let top_k = cfg.experts_per_token;

    let tokens = batch_size * seq_len;

    // Attention activations: Q [tokens, attn_dim] + K [tokens, kv_dim]
    // + V [tokens, kv_dim] + scores [batch, heads, seq, seq]
    // + attn_out [tokens, attn_dim]
    let attn_bytes = (tokens * ad
        + tokens * kvd * 2
        + batch_size * cfg.num_attention_heads * seq_len * seq_len
        + tokens * ad)
        * bpe;

    // MoE activations: router [tokens, ne] + gate_up [avg_per_expert, 2*inter]
    // + down [avg_per_expert, h] + scatter output [tokens, h]
    let avg_per_expert = (tokens * top_k) / ne.max(1) + 1;
    let moe_bytes =
        (tokens * ne + avg_per_expert * 2 * inter + avg_per_expert * h + tokens * h) * bpe;

    attn_bytes + moe_bytes
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that a threadgroup size is a positive power of 2, at most 1024.
fn validate_threadgroup_size(size: u32, name: &str) -> Result<()> {
    if size == 0 || !size.is_power_of_two() {
        return Err(GptOssError::InvalidConfig {
            reason: format!("{name}_threadgroup_size must be a positive power of 2, got {size}"),
        }
        .into());
    }
    if size > 1024 {
        return Err(GptOssError::InvalidConfig {
            reason: format!("{name}_threadgroup_size must be <= 1024 (Metal limit), got {size}"),
        }
        .into());
    }
    Ok(())
}

/// Ceiling division: `(a + b - 1) / b`, overflow-safe for u32.
#[must_use]
fn ceil_div(a: u32, b: u32) -> u32 {
    assert!(b > 0, "ceil_div: divisor must be > 0");
    a.div_ceil(b)
}

/// Checked product of a slice of u32 values.
fn checked_u32_product(dims: &[u32]) -> Result<u32> {
    let mut product: u32 = 1;
    for &d in dims {
        product = product
            .checked_mul(d)
            .ok_or_else(|| GptOssError::InvalidConfig {
                reason: format!("dispatch grid overflow at dimension {d}"),
            })?;
    }
    Ok(product)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m4_max_config_validates() {
        let cfg = GptOssGpuConfig::m4_max();
        cfg.validate().expect("M4 Max config should validate");
    }

    #[test]
    fn test_apple_silicon_base_config_validates() {
        let cfg = GptOssGpuConfig::apple_silicon_base();
        cfg.validate()
            .expect("Apple Silicon base config should validate");
    }

    #[test]
    fn test_default_config_is_m4_max() {
        let default = GptOssGpuConfig::default();
        let m4 = GptOssGpuConfig::m4_max();
        assert_eq!(
            default.attention_threadgroup_size,
            m4.attention_threadgroup_size
        );
        assert_eq!(default.moe_threadgroup_size, m4.moe_threadgroup_size);
    }

    #[test]
    fn test_invalid_threadgroup_size_zero() {
        let mut cfg = GptOssGpuConfig::m4_max();
        cfg.attention_threadgroup_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_threadgroup_size_non_power_of_two() {
        let mut cfg = GptOssGpuConfig::m4_max();
        cfg.moe_threadgroup_size = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_threadgroup_size_exceeds_metal_limit() {
        let mut cfg = GptOssGpuConfig::m4_max();
        cfg.elementwise_threadgroup_size = 2048;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_should_use_gpu_cpu() {
        assert!(!should_use_gpu(&Device::Cpu));
    }

    #[test]
    fn test_should_use_gpu_metal() {
        assert!(should_use_gpu(&Device::Metal { device_id: 0 }));
    }

    #[test]
    fn test_select_gpu_config_cpu_returns_none() {
        assert!(select_gpu_config(&Device::Cpu).is_none());
    }

    #[test]
    fn test_select_gpu_config_metal_returns_some() {
        assert!(select_gpu_config(&Device::Metal { device_id: 0 }).is_some());
    }

    #[test]
    fn test_dispatch_attention_grid_basic() {
        let cfg = GptOssGpuConfig::m4_max();
        let (grid, tg) =
            dispatch_attention_grid(1, 64, 1, 128, &cfg).expect("small grid should succeed");
        assert!(grid[0] >= 64 * 128);
        assert_eq!(grid[0] % tg[0], 0, "grid must be aligned to threadgroup");
        assert_eq!(tg[0], 256);
        assert_eq!(grid[1], 1);
        assert_eq!(grid[2], 1);
    }

    #[test]
    fn test_dispatch_moe_expert_grid_basic() {
        let cfg = GptOssGpuConfig::m4_max();
        let (grid, tg) =
            dispatch_moe_expert_grid(16, 2880, &cfg).expect("small grid should succeed");
        assert!(grid[0] >= 16 * 2880);
        assert_eq!(grid[0] % tg[0], 0);
    }

    #[test]
    fn test_dispatch_moe_scatter_grid_basic() {
        let cfg = GptOssGpuConfig::m4_max();
        let (grid, tg) =
            dispatch_moe_scatter_grid(128, 2880, &cfg).expect("small grid should succeed");
        assert!(grid[0] >= 128 * 2880);
        assert_eq!(grid[0] % tg[0], 0);
    }

    #[test]
    fn test_align_buffer_offset() {
        let cfg = GptOssGpuConfig::m4_max();
        assert_eq!(align_buffer_offset(0, &cfg), 0);
        assert_eq!(align_buffer_offset(1, &cfg), 16);
        assert_eq!(align_buffer_offset(16, &cfg), 16);
        assert_eq!(align_buffer_offset(17, &cfg), 32);
        assert_eq!(align_buffer_offset(31, &cfg), 32);
        assert_eq!(align_buffer_offset(32, &cfg), 32);
    }

    #[test]
    fn test_estimate_layer_activation_bytes_nonzero() {
        let model_cfg = GptOssConfig::gptoss_20b();
        let bytes = estimate_layer_activation_bytes(&model_cfg, 128, 1);
        assert!(bytes > 0);
    }

    #[test]
    fn test_estimate_layer_activation_bytes_scales_with_seq() {
        let model_cfg = GptOssConfig::gptoss_20b();
        let bytes_short = estimate_layer_activation_bytes(&model_cfg, 64, 1);
        let bytes_long = estimate_layer_activation_bytes(&model_cfg, 256, 1);
        assert!(bytes_long > bytes_short);
    }

    #[test]
    fn test_ceil_div() {
        assert_eq!(ceil_div(0, 256), 0);
        assert_eq!(ceil_div(1, 256), 1);
        assert_eq!(ceil_div(255, 256), 1);
        assert_eq!(ceil_div(256, 256), 1);
        assert_eq!(ceil_div(257, 256), 2);
        assert_eq!(ceil_div(512, 256), 2);
    }

    #[test]
    fn test_checked_u32_product_basic() {
        assert_eq!(checked_u32_product(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_u32_product(&[1]).unwrap(), 1);
        assert_eq!(checked_u32_product(&[]).unwrap(), 1);
    }

    #[test]
    fn test_checked_u32_product_overflow() {
        assert!(checked_u32_product(&[u32::MAX, 2]).is_err());
    }

    #[test]
    fn test_attention_matmul_grid() {
        let cfg = GptOssGpuConfig::m4_max();
        // Single-token decode: batch=1, heads=64, seq_q=1, seq_kv=128
        let (grid, tg) =
            dispatch_attention_matmul_grid(1, 64, 1, 128, &cfg).expect("should succeed");
        assert!(grid[0] >= 64 * 128);
        assert_eq!(grid[0] % tg[0], 0);
    }

    #[test]
    fn test_dispatch_grid_overflow_detection() {
        let cfg = GptOssGpuConfig::m4_max();
        // Intentionally huge dimensions to trigger overflow
        let result = dispatch_attention_grid(u32::MAX, u32::MAX, 1, 1, &cfg);
        assert!(result.is_err());
    }
}
