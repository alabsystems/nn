// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SageAttention: quantized attention with INT8 Q/K scoring and FP16 PV
//! accumulation (Zhang et al., 2024; arXiv:2410.02367).
//!
//! SageAttention achieves 2-5x speedup over Flash Attention v2 by quantizing
//! the Q and K matrices to simulated INT8 for the scoring matmul, while
//! keeping V in full precision for the PV accumulation. This trades a small
//! accuracy loss (typically < 0.05 max absolute error vs standard SDPA) for
//! significant throughput gains — especially beneficial for long-context VLM
//! workloads like dpdf document understanding.
//!
//! Key techniques:
//! - **Per-head absmax quantization** of Q and K to simulated INT8 range
//! - **Smooth K** (optional): subtract per-channel mean from K before
//!   quantization to reduce outlier-driven quantization error
//! - **GQA support**: num_kv_heads < num_heads via K/V head repetition
//! - **Causal masking**: optional upper-triangular mask for autoregressive
//!
//! Part of #3862 — SageAttention for dpdf document understanding VLMs.

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::repeat_kv;
use crate::layers::check_output_finite;
use crate::{DType, Result, TensorError};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`SageAttention`].
#[derive(Debug, Clone, Copy)]
pub struct SageAttentionConfig {
    /// Per-head dimension (e.g. 64, 128).
    pub head_dim: usize,
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of K/V heads for grouped-query attention. `None` means MHA
    /// (num_kv_heads == num_heads).
    pub num_kv_heads: Option<usize>,
    /// Whether to apply causal (autoregressive) masking.
    pub causal: bool,
    /// Whether to subtract per-channel mean from K before quantization
    /// (reduces outlier-driven quantization error).
    pub smooth_k: bool,
}

impl SageAttentionConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "SageAttentionConfig: num_heads must be > 0",
            });
        }
        if self.head_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "SageAttentionConfig: head_dim must be > 0",
            });
        }
        if let Some(nkv) = self.num_kv_heads {
            if nkv == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "SageAttentionConfig: num_kv_heads must be > 0",
                });
            }
            if !self.num_heads.is_multiple_of(nkv) {
                return Err(TensorError::ValueOutOfRange {
                    description: "SageAttentionConfig: num_heads must be divisible by num_kv_heads",
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SageAttention
// ---------------------------------------------------------------------------

/// SageAttention: quantized attention with INT8 Q/K scoring.
///
/// See [module docs](self) for algorithm details.
#[derive(Debug, Clone)]
pub struct SageAttention {
    config: SageAttentionConfig,
}

impl SageAttention {
    /// Create a new SageAttention instance. Validates configuration.
    pub fn new(config: SageAttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Run quantized attention.
    ///
    /// # Arguments
    ///
    /// - `q`: query tensor `[B, num_heads, S_q, head_dim]`
    /// - `k`: key tensor `[B, num_kv_heads, S_kv, head_dim]`
    /// - `v`: value tensor `[B, num_kv_heads, S_kv, head_dim]`
    ///
    /// # Returns
    ///
    /// Attention output `[B, num_heads, S_q, head_dim]`.
    pub fn forward(&self, q: &DynTensor, k: &DynTensor, v: &DynTensor) -> Result<DynTensor> {
        let (b, h_q, s_q, d) = q.dims4()?;
        let (_b_k, h_kv, s_kv, d_k) = k.dims4()?;
        let (_b_v, _h_v, _s_v, _d_v) = v.dims4()?;

        // Validate shapes
        if d != self.config.head_dim {
            return Err(TensorError::InvalidShape(format!(
                "SageAttention: expected head_dim={}, got Q head_dim={}",
                self.config.head_dim, d
            )));
        }
        if d_k != d {
            return Err(TensorError::InvalidShape(format!(
                "SageAttention: Q head_dim={d} != K head_dim={d_k}"
            )));
        }
        if h_q != self.config.num_heads {
            return Err(TensorError::InvalidShape(format!(
                "SageAttention: expected num_heads={}, got Q num_heads={}",
                self.config.num_heads, h_q
            )));
        }

        // Handle GQA: repeat K/V heads to match Q heads
        let num_kv_heads = self.config.num_kv_heads.unwrap_or(self.config.num_heads);
        let num_rep = self.config.num_heads / num_kv_heads;
        if h_kv != num_kv_heads {
            return Err(TensorError::InvalidShape(format!(
                "SageAttention: expected num_kv_heads={num_kv_heads}, got K num_heads={h_kv}"
            )));
        }
        let k = if num_rep > 1 {
            repeat_kv(k, num_rep)?
        } else {
            k.clone()
        };
        let v = if num_rep > 1 {
            repeat_kv(v, num_rep)?
        } else {
            v.clone()
        };

        // --- INT8 quantization of Q ---
        // Per-head absmax: max over head_dim (last dim), keep broadcastable
        let q_abs = q.abs()?;
        let q_absmax = q_abs.max_keepdim(3)?; // [B, H, S, 1]
        let q_scale = q_absmax.clamp_min(1e-10)?.div_scalar(127.0)?; // [B, H, S, 1]
        let q_int8 = q.broadcast_div(&q_scale)?.round()?.clamp(-128.0, 127.0)?;

        // --- Optional smooth K: subtract per-channel mean ---
        let k_for_quant = if self.config.smooth_k {
            let k_mean = k.mean_keepdim(3)?; // [B, H, S_kv, 1]
            k.broadcast_add(&k_mean.mul_scalar(-1.0)?)?
        } else {
            k
        };

        // --- INT8 quantization of K ---
        let k_abs = k_for_quant.abs()?;
        let k_absmax = k_abs.max_keepdim(3)?; // [B, H, S_kv, 1]
        let k_scale = k_absmax.clamp_min(1e-10)?.div_scalar(127.0)?;
        let k_int8 = k_for_quant
            .broadcast_div(&k_scale)?
            .round()?
            .clamp(-128.0, 127.0)?;

        // --- Compute attention scores: S = (Q_int8 @ K_int8^T) * (q_scale * k_scale) / sqrt(d) ---
        let k_int8_t = k_int8.transpose(2, 3)?; // [B, H, head_dim, S_kv]
        let scores_raw = q_int8.matmul(&k_int8_t)?; // [B, H, S_q, S_kv]

        // Dequantize: multiply by combined scale
        // q_scale: [B, H, S_q, 1], k_scale: [B, H, S_kv, 1]
        // k_scale transposed: [B, H, 1, S_kv]
        let k_scale_t = k_scale.transpose(2, 3)?; // [B, H, 1, S_kv]
        let combined_scale = q_scale.broadcast_mul(&k_scale_t)?; // [B, H, S_q, S_kv]

        let inv_sqrt_d = 1.0 / (d as f64).sqrt();
        let scores = scores_raw
            .broadcast_mul(&combined_scale)?
            .mul_scalar(inv_sqrt_d)?;

        // --- Apply causal mask ---
        let scores = if self.config.causal {
            let mask = build_causal_mask(s_q, s_kv, q.dtype(), &q.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        // --- Softmax over last dim ---
        let attn_weights = scores.softmax(3)?; // [B, H, S_q, S_kv]

        // --- PV accumulation in original precision ---
        let output = attn_weights.matmul(&v)?; // [B, H, S_q, head_dim]

        check_output_finite(&output, "SageAttention")?;

        // Validate output shape
        let expected = [b, self.config.num_heads, s_q, d];
        if output.dims() != expected {
            return Err(TensorError::InvalidShape(format!(
                "SageAttention: expected output shape {:?}, got {:?}",
                expected,
                output.dims()
            )));
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Causal mask helper
// ---------------------------------------------------------------------------

/// Build an additive causal mask: 0 for allowed positions, -inf for masked.
///
/// Returns `[1, 1, new_tokens, total_tokens]`.
fn build_causal_mask(
    new_tokens: usize,
    total_tokens: usize,
    dtype: DType,
    device: &crate::Device,
) -> Result<DynTensor> {
    super::causal_mask_with_offset(new_tokens, total_tokens, dtype, device)
}

#[cfg(test)]
#[path = "sage_attention_tests.rs"]
mod tests;
