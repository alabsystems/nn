// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-head Latent Attention (MLA) for DeepSeek-V2/V3 models.
//!
//! MLA reduces KV cache memory by projecting keys and values into a
//! low-rank latent space. Instead of caching full K/V tensors of size
//! `[B, S, n_heads, head_dim]`, the model caches only the compressed
//! latent `[B, S, kv_lora_rank]` where `kv_lora_rank << n_heads * head_dim`.
//!
//! Architecture:
//! 1. **KV compression:** `hidden -> compressed_kv [B, T, kv_lora_rank]`
//! 2. **KV uplift:** `compressed_kv -> K, V [B, T, n_heads, qk_nope_dim/v_head_dim]`
//! 3. **Q projection:** `hidden -> Q [B, T, n_heads, head_dim]`
//!    (optionally through a latent Q compression stage)
//! 4. **Decoupled RoPE:** only `rope_dim` of Q/K gets rotary embedding;
//!    `qk_nope_dim` passes through unrotated.
//! 5. **Standard SDPA:** `softmax(Q @ K^T / sqrt(d)) @ V`
//! 6. **Output projection:** attention output back to `hidden_size`
//!
//! Reference: DeepSeek-V2 (arXiv:2405.04434), Section 3.1.

use crate::dyn_tensor::DynTensor;
use crate::layers::{
    check_output_finite, validate_heads, Linear, Module, RmsNorm,
};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

use super::sdpa::sdpa;

/// Configuration for [`MlaLayer`].
#[derive(Debug, Clone, Copy)]
pub struct MlaConfig {
    /// Model hidden dimension (input/output).
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Latent dimension for KV compression (d_c). Must be < `num_heads * v_head_dim`.
    pub kv_lora_rank: usize,
    /// Optional latent dimension for Q compression. When `Some`, Q goes through
    /// a down-projection followed by a norm and up-projection.
    pub q_lora_rank: Option<usize>,
    /// Portion of the head dimension that receives RoPE.
    pub rope_dim: usize,
    /// Portion of the head dimension without RoPE (nope = no position embedding).
    pub qk_nope_dim: usize,
    /// Value head dimension (often equals `qk_nope_dim`).
    pub v_head_dim: usize,
    /// RMS norm epsilon for latent norms.
    pub rms_norm_eps: f64,
}

impl MlaConfig {
    /// Effective Q/K head dimension: `qk_nope_dim + rope_dim`.
    #[must_use]
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_dim + self.rope_dim
    }

    /// Validate the configuration, returning an error for invalid combinations.
    pub fn validate(&self) -> Result<()> {
        validate_heads(self.num_heads, "MlaLayer")?;
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: hidden_size must be > 0",
            });
        }
        if self.kv_lora_rank == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: kv_lora_rank must be > 0",
            });
        }
        if self.rope_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: rope_dim must be > 0",
            });
        }
        if self.qk_nope_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: qk_nope_dim must be > 0",
            });
        }
        if self.v_head_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: v_head_dim must be > 0",
            });
        }
        if !self.rope_dim.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: rope_dim must be even (for RoPE pairing)",
            });
        }
        if let Some(q_lr) = self.q_lora_rank {
            if q_lr == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "MlaConfig: q_lora_rank must be > 0 when set",
                });
            }
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps < 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MlaConfig: rms_norm_eps must be finite and non-negative",
            });
        }
        Ok(())
    }
}

/// Multi-head Latent Attention layer (DeepSeek-V2/V3).
///
/// Reduces KV cache from `O(n_heads * head_dim)` to `O(kv_lora_rank)` per token
/// by factoring K/V projections through a shared low-rank latent bottleneck.
///
/// During inference, only `compressed_kv` (shape `[B, T, kv_lora_rank]`) needs
/// to be cached, not the full K/V tensors.
#[derive(Clone)]
pub struct MlaLayer {
    // Q path
    q_a_proj: Option<Linear>,     // hidden -> q_lora_rank (optional compression)
    q_a_norm: Option<RmsNorm>,    // norm on compressed Q
    q_b_proj: Linear,             // q_lora_rank (or hidden) -> num_heads * qk_head_dim

    // KV path
    kv_a_proj: Linear,            // hidden -> kv_lora_rank + rope_dim
    kv_a_norm: RmsNorm,           // norm on compressed KV (applied to kv_lora_rank portion)
    kv_b_proj: Linear,            // kv_lora_rank -> num_heads * (qk_nope_dim + v_head_dim)

    // Output
    out_proj: Linear,             // num_heads * v_head_dim -> hidden

    // RoPE projections for decoupled keys
    // The rope portion of K comes from the kv_a_proj output, not from kv_b_proj.

    cfg: MlaConfig,
    scale: f64,
}

impl std::fmt::Debug for MlaLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlaLayer")
            .field("num_heads", &self.cfg.num_heads)
            .field("kv_lora_rank", &self.cfg.kv_lora_rank)
            .field("q_lora_rank", &self.cfg.q_lora_rank)
            .field("rope_dim", &self.cfg.rope_dim)
            .field("qk_nope_dim", &self.cfg.qk_nope_dim)
            .field("v_head_dim", &self.cfg.v_head_dim)
            .finish_non_exhaustive()
    }
}

impl MlaLayer {
    /// Load from a [`VarBuilder`] with DeepSeek-V2 weight names.
    ///
    /// Expected weight names under the `vb` prefix:
    /// - `q_a_proj.weight` (only when `q_lora_rank` is set)
    /// - `q_a_layernorm.weight` (only when `q_lora_rank` is set)
    /// - `q_b_proj.weight`
    /// - `kv_a_proj_with_mqa.weight`
    /// - `kv_a_layernorm.weight`
    /// - `kv_b_proj.weight`
    /// - `o_proj.weight`
    pub fn load(vb: impl AsRef<VarBuilder>, cfg: MlaConfig) -> Result<Self> {
        cfg.validate()?;
        let vb = vb.as_ref();
        let qk_head_dim = cfg.qk_head_dim();

        // Q path
        let (q_a_proj, q_a_norm) = if let Some(q_lr) = cfg.q_lora_rank {
            let q_a_w = vb.pp("q_a_proj").get(
                &[q_lr, cfg.hidden_size],
                "weight",
            )?;
            let q_a = Linear::new(q_a_w, None)?;
            let q_a_norm_w = vb.pp("q_a_layernorm").get(&[q_lr], "weight")?;
            let q_a_n = RmsNorm::new(q_a_norm_w, cfg.rms_norm_eps)?;
            (Some(q_a), Some(q_a_n))
        } else {
            (None, None)
        };

        let q_b_in = cfg.q_lora_rank.unwrap_or(cfg.hidden_size);
        let q_b_w = vb.pp("q_b_proj").get(
            &[cfg.num_heads * qk_head_dim, q_b_in],
            "weight",
        )?;
        let q_b_proj = Linear::new(q_b_w, None)?;

        // KV path: kv_a_proj outputs kv_lora_rank + rope_dim
        let kv_a_out = cfg.kv_lora_rank + cfg.rope_dim;
        let kv_a_w = vb.pp("kv_a_proj_with_mqa").get(
            &[kv_a_out, cfg.hidden_size],
            "weight",
        )?;
        let kv_a_proj = Linear::new(kv_a_w, None)?;

        let kv_a_norm_w = vb.pp("kv_a_layernorm").get(
            &[cfg.kv_lora_rank],
            "weight",
        )?;
        let kv_a_norm = RmsNorm::new(kv_a_norm_w, cfg.rms_norm_eps)?;

        // kv_b_proj: kv_lora_rank -> num_heads * (qk_nope_dim + v_head_dim)
        let kv_b_out = cfg.num_heads * (cfg.qk_nope_dim + cfg.v_head_dim);
        let kv_b_w = vb.pp("kv_b_proj").get(
            &[kv_b_out, cfg.kv_lora_rank],
            "weight",
        )?;
        let kv_b_proj = Linear::new(kv_b_w, None)?;

        // Output projection
        let o_w = vb.pp("o_proj").get(
            &[cfg.hidden_size, cfg.num_heads * cfg.v_head_dim],
            "weight",
        )?;
        let out_proj = Linear::new(o_w, None)?;

        let scale = 1.0 / (qk_head_dim as f64).sqrt();

        Ok(Self {
            q_a_proj,
            q_a_norm,
            q_b_proj,
            kv_a_proj,
            kv_a_norm,
            kv_b_proj,
            out_proj,
            cfg,
            scale,
        })
    }

    /// Forward pass with decoupled RoPE.
    ///
    /// - `hidden`: input hidden states `[B, T, hidden_size]`
    /// - `cos`: RoPE cos frequencies `[T, rope_dim/2]` or broadcastable
    /// - `sin`: RoPE sin frequencies `[T, rope_dim/2]` or broadcastable
    /// - `mask`: optional attention mask `[*, *, T, T]`
    ///
    /// Returns output `[B, T, hidden_size]`.
    pub fn forward(
        &self,
        hidden: &DynTensor,
        cos: &DynTensor,
        sin: &DynTensor,
        mask: Option<&DynTensor>,
    ) -> Result<DynTensor> {
        let (b, t, _d) = hidden.dims3()?;
        let cfg = &self.cfg;
        let qk_head_dim = cfg.qk_head_dim();

        // == Q path ==
        let q = if let (Some(q_a), Some(q_a_n)) = (&self.q_a_proj, &self.q_a_norm) {
            let q_compressed = q_a.forward(hidden)?;
            let q_normed = q_a_n.forward(&q_compressed)?;
            self.q_b_proj.forward(&q_normed)?
        } else {
            self.q_b_proj.forward(hidden)?
        };
        // q: [B, T, num_heads * qk_head_dim]
        // Reshape to [B, T, num_heads, qk_head_dim]
        let q = q.reshape([b, t, cfg.num_heads, qk_head_dim])?;

        // Split Q into nope and rope portions along last dim
        // q_nope: [B, T, num_heads, qk_nope_dim]
        // q_rope: [B, T, num_heads, rope_dim]
        let q_nope = q.narrow(3, 0, cfg.qk_nope_dim)?;
        let q_rope = q.narrow(3, cfg.qk_nope_dim, cfg.rope_dim)?;

        // Apply RoPE to q_rope
        // Transpose to [B, num_heads, T, rope_dim] for RoPE application
        let q_rope = q_rope.transpose(1, 2)?;
        let q_rope = apply_rope_from_cos_sin(&q_rope, cos, sin)?;
        // Transpose back to [B, T, num_heads, rope_dim]
        let q_rope = q_rope.transpose(1, 2)?;

        // Reassemble Q: cat(q_nope, q_rope) along last dim
        let q = DynTensor::cat(&[&q_nope, &q_rope], 3)?;
        // q: [B, T, num_heads, qk_head_dim]
        // Transpose to [B, num_heads, T, qk_head_dim] for SDPA
        let q = q.transpose(1, 2)?;

        // == KV path ==
        let kv_a = self.kv_a_proj.forward(hidden)?;
        // kv_a: [B, T, kv_lora_rank + rope_dim]

        // Split into compressed_kv and k_rope_input
        let compressed_kv = kv_a.narrow(2, 0, cfg.kv_lora_rank)?;
        let k_rope_input = kv_a.narrow(2, cfg.kv_lora_rank, cfg.rope_dim)?;

        // Norm the compressed KV
        let compressed_kv = self.kv_a_norm.forward(&compressed_kv)?;

        // Uplift: compressed_kv -> k_nope and v
        let kv_b = self.kv_b_proj.forward(&compressed_kv)?;
        // kv_b: [B, T, num_heads * (qk_nope_dim + v_head_dim)]
        let kv_b = kv_b.reshape([b, t, cfg.num_heads, cfg.qk_nope_dim + cfg.v_head_dim])?;

        // Split into k_nope and v
        let k_nope = kv_b.narrow(3, 0, cfg.qk_nope_dim)?;
        let v = kv_b.narrow(3, cfg.qk_nope_dim, cfg.v_head_dim)?;
        // k_nope: [B, T, num_heads, qk_nope_dim]
        // v: [B, T, num_heads, v_head_dim]

        // k_rope: shared across all heads (MQA-style for the rope portion)
        // k_rope_input: [B, T, rope_dim] -> [B, T, 1, rope_dim]
        let k_rope = k_rope_input.reshape([b, t, 1, cfg.rope_dim])?;
        // Transpose to [B, 1, T, rope_dim] for RoPE
        let k_rope = k_rope.transpose(1, 2)?;
        let k_rope = apply_rope_from_cos_sin(&k_rope, cos, sin)?;
        // Expand to [B, num_heads, T, rope_dim]
        let k_rope = k_rope.expand([b, cfg.num_heads, t, cfg.rope_dim])?;
        // Transpose back to [B, T, num_heads, rope_dim]
        let k_rope = k_rope.transpose(1, 2)?;

        // Reassemble K: cat(k_nope, k_rope) along last dim
        let k = DynTensor::cat(&[&k_nope, &k_rope], 3)?;
        // k: [B, T, num_heads, qk_head_dim]

        // Transpose K, V to [B, num_heads, T, dim] for SDPA
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // == SDPA ==
        let attn_out = sdpa(&q, &k, &v, mask, self.scale)?;
        // attn_out: [B, num_heads, T, v_head_dim]

        // Reshape: [B, num_heads, T, v_head_dim] -> [B, T, num_heads * v_head_dim]
        let out = attn_out
            .transpose(1, 2)?
            .reshape([b, t, cfg.num_heads * cfg.v_head_dim])?;

        let result = self.out_proj.forward(&out)?;
        check_output_finite(&result, "MlaLayer")?;
        Ok(result)
    }

    /// Number of query attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.cfg.num_heads
    }

    /// KV latent dimension.
    #[must_use]
    pub fn kv_lora_rank(&self) -> usize {
        self.cfg.kv_lora_rank
    }

    /// RoPE dimension (portion of head that gets rotary embedding).
    #[must_use]
    pub fn rope_dim(&self) -> usize {
        self.cfg.rope_dim
    }

    /// Non-RoPE Q/K dimension.
    #[must_use]
    pub fn qk_nope_dim(&self) -> usize {
        self.cfg.qk_nope_dim
    }

    /// Value head dimension.
    #[must_use]
    pub fn v_head_dim(&self) -> usize {
        self.cfg.v_head_dim
    }

    /// Configuration used to create this layer.
    #[must_use]
    pub fn config(&self) -> &MlaConfig {
        &self.cfg
    }
}

/// Apply RoPE using precomputed cos/sin tensors.
///
/// - `x`: `[B, H, T, rope_dim]` (last dim must be even)
/// - `cos`: `[T, rope_dim/2]` or broadcastable
/// - `sin`: `[T, rope_dim/2]` or broadcastable
///
/// Uses the interleaved pair convention: pairs `(x[2i], x[2i+1])` are rotated.
fn apply_rope_from_cos_sin(
    x: &DynTensor,
    cos: &DynTensor,
    sin: &DynTensor,
) -> Result<DynTensor> {
    let rank = x.rank();
    if rank < 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: rank,
        });
    }
    let dims = x.dims();
    let rope_dim = dims[rank - 1];
    let seq_len = dims[rank - 2];
    let half_dim = rope_dim / 2;

    // Convert cos/sin to match input dtype
    let (cos, sin) = if x.dtype() != cos.dtype() {
        (cos.to_dtype(x.dtype())?, sin.to_dtype(x.dtype())?)
    } else {
        (cos.clone(), sin.clone())
    };

    // Reshape x into pairs: [..., half_dim, 2]
    let mut pairs_shape: Vec<usize> = dims[..rank - 1].to_vec();
    pairs_shape.push(half_dim);
    pairs_shape.push(2);
    let x_pairs = x.reshape(&pairs_shape)?;
    let x_even = x_pairs.narrow(rank, 0, 1)?.squeeze(rank)?;
    let x_odd = x_pairs.narrow(rank, 1, 1)?.squeeze(rank)?;

    // Broadcast cos/sin to match batch dims
    let mut broadcast_shape = vec![1usize; rank - 2];
    broadcast_shape.push(seq_len);
    broadcast_shape.push(half_dim);
    let cos_bc = cos.reshape(&broadcast_shape)?;
    let sin_bc = sin.reshape(&broadcast_shape)?;

    // Apply rotation
    let y_even = x_even
        .broadcast_mul(&cos_bc)?
        .broadcast_sub(&x_odd.broadcast_mul(&sin_bc)?)?;
    let y_odd = x_even
        .broadcast_mul(&sin_bc)?
        .broadcast_add(&x_odd.broadcast_mul(&cos_bc)?)?;

    // Interleave back
    let y_even_expanded = y_even.unsqueeze(rank)?;
    let y_odd_expanded = y_odd.unsqueeze(rank)?;
    let y_pairs = DynTensor::cat(&[&y_even_expanded, &y_odd_expanded], rank)?;
    y_pairs.reshape(dims)
}

#[cfg(test)]
#[path = "mla_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_mla_proofs.rs"]
mod kani_mla_proofs;
