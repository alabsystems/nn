// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Joint Attention for DiT (Diffusion Transformer) models.
//!
//! Used by Irodori-TTS, CosyVoice3, and similar DiT-based TTS architectures.
//! Q projects from self tokens; K/V project from concatenated `[self, text, speaker]`
//! context. Standard scaled dot-product attention with `1/sqrt(head_dim)` scaling.
//!
//! Reference: `designs/2026-03-03-dit-composite-ops.md` Direction 2.

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::sdpa;
use crate::layers::{check_output_finite, validate_divisible, validate_heads, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Joint attention over concatenated conditioning sources.
///
/// Q from self tokens, K/V from concatenated `[self, ctx1, ctx2, ...]` context.
/// Used in DiT blocks for cross-modal attention (audio ↔ text ↔ speaker).
#[derive(Debug, Clone)]
pub struct JointAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl JointAttention {
    /// Create a JointAttention from pre-loaded projection weights.
    ///
    /// - `num_heads` must be > 0
    /// - `dim` must be divisible by `num_heads`
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        num_heads: usize,
        dim: usize,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(TensorError::InvalidShape(
                "JointAttention: dim must be > 0".into(),
            ));
        }
        validate_heads(num_heads, "JointAttention")?;
        validate_divisible(dim, num_heads, "dim", "num_heads", "JointAttention")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    /// Load from a [`VarBuilder`] using PyTorch-style weight names.
    ///
    /// Loads `q_proj`, `k_proj`, `v_proj`, and `out_proj` sub-modules
    /// (each a Linear with `[dim, dim]` weight).
    /// Weight names: `q_proj.weight`, `k_proj.weight`, etc.,
    /// plus optional `*.bias` tensors.
    ///
    /// - `dim`: Model dimension (input and output size of all projections).
    /// - `num_heads`: Number of attention heads. `dim` must be divisible by `num_heads`.
    pub fn load(vb: impl AsRef<VarBuilder>, dim: usize, num_heads: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let q_proj = Linear::load(vb.pp("q_proj"), dim, dim)?;
        let k_proj = Linear::load(vb.pp("k_proj"), dim, dim)?;
        let v_proj = Linear::load(vb.pp("v_proj"), dim, dim)?;
        let out_proj = Linear::load(vb.pp("out_proj"), dim, dim)?;
        Self::new(q_proj, k_proj, v_proj, out_proj, num_heads, dim)
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Dimension per attention head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Forward pass with two conditioning context tensors.
    ///
    /// - `x`: self tokens `[B, S_self, D]`
    /// - `ctx1`: first context source `[B, S_ctx1, D]` (e.g., text)
    /// - `ctx2`: second context source `[B, S_ctx2, D]` (e.g., speaker)
    ///
    /// K and V attend over concatenated `[x, ctx1, ctx2]` (total length
    /// `S_self + S_ctx1 + S_ctx2`). Q attends only from `x`.
    ///
    /// Returns: `[B, S_self, D]`
    pub fn forward_joint(
        &self,
        x: &DynTensor,
        ctx1: &DynTensor,
        ctx2: &DynTensor,
    ) -> Result<DynTensor> {
        let kv_ctx = DynTensor::cat(&[x, ctx1, ctx2], 1)?;
        self.forward_with_context(x, &kv_ctx)
    }

    /// Forward with a single context source (simpler interface).
    ///
    /// K/V attend over concatenated `[x, ctx]`.
    pub fn forward_single_ctx(&self, x: &DynTensor, ctx: &DynTensor) -> Result<DynTensor> {
        let kv_ctx = DynTensor::cat(&[x, ctx], 1)?;
        self.forward_with_context(x, &kv_ctx)
    }

    /// Core multi-head attention: Q from `x`, K/V from `kv_ctx`.
    fn forward_with_context(&self, x: &DynTensor, kv_ctx: &DynTensor) -> Result<DynTensor> {
        let (b, s_self, _d) = x.dims3()?;
        let s_total = kv_ctx.dim(1)?;

        // Project Q from self tokens, K/V from full context
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(kv_ctx)?;
        let v = self.v_proj.forward(kv_ctx)?;

        // Reshape to multi-head: [B, S, D] -> [B, S, H, head_dim] -> [B, H, S, head_dim]
        let q = q
            .reshape([b, s_self, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([b, s_total, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([b, s_total, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Scaled dot-product attention via shared sdpa()
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_out = sdpa(&q, &k, &v, None, scale)?;

        // Reshape back: [B, H, S_self, head_dim] -> [B, S_self, H, head_dim] -> [B, S_self, D]
        let d = self.num_heads * self.head_dim;
        let out = attn_out.transpose(1, 2)?.reshape([b, s_self, d])?;

        // Output projection
        let result = self.out_proj.forward(&out)?;
        check_output_finite(&result, "JointAttention")?;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "joint_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "joint_tests_properties.rs"]
mod tests_properties;
