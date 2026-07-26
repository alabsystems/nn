// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable multi-head attention with [`Var`] weights.
//!
//! Wraps scaled dot-product attention (SDPA) for training:
//! Q/K/V projections and output projection are [`TrainableLinear`] layers,
//! so gradients flow through all 4 weight matrices during `backward()`.
//!
//! Standard MHA (equal Q/K/V heads). GQA, RoPE, and KV cache are
//! inference-only features and are not supported in the training wrapper.

use crate::error::Result;
use crate::tracked::TrackedTensor;
use crate::trainable::{TrainableLinear, TrainableModule};
use crate::var::Var;
use std::sync::Arc;

/// Trainable multi-head self-attention layer.
///
/// Computes scaled dot-product attention with 4 trainable linear projections:
/// - Q, K, V projections: `[model_dim, model_dim]`
/// - Output projection: `[model_dim, model_dim]`
///
/// Forward: `x [B, S, D]` → Q/K/V project → reshape to heads →
///   `softmax(Q @ K^T / sqrt(head_dim)) @ V` → reshape → out project.
///
/// Required for Whisper encoder/decoder and Qwen3 self-attention fine-tuning.
#[derive(Debug, Clone)]
pub struct TrainableMultiHeadAttention {
    q_proj: TrainableLinear,
    k_proj: TrainableLinear,
    v_proj: TrainableLinear,
    out_proj: TrainableLinear,
    num_heads: usize,
    head_dim: usize,
}

impl TrainableMultiHeadAttention {
    /// Create from existing [`TrainableLinear`] projections.
    ///
    /// `num_heads` must divide `model_dim` evenly.
    pub fn new(
        q_proj: TrainableLinear,
        k_proj: TrainableLinear,
        v_proj: TrainableLinear,
        out_proj: TrainableLinear,
        num_heads: usize,
        model_dim: usize,
    ) -> Result<Self> {
        if num_heads == 0 {
            return Err(crate::error::AutodiffError::InvalidConfig {
                op: "TrainableMHA",
                reason: "num_heads must be > 0".into(),
            });
        }
        if !model_dim.is_multiple_of(num_heads) {
            return Err(crate::error::AutodiffError::InvalidConfig {
                op: "TrainableMHA",
                reason: format!(
                    "model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
                ),
            });
        }
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim: model_dim / num_heads,
        })
    }

    /// Create with zero-initialized weights.
    ///
    /// Weight shape for each projection: `[model_dim, model_dim]` with bias.
    pub fn zeros(model_dim: usize, num_heads: usize, bias: bool) -> Result<Self> {
        let q = TrainableLinear::new(model_dim, model_dim, bias)?;
        let k = TrainableLinear::new(model_dim, model_dim, bias)?;
        let v = TrainableLinear::new(model_dim, model_dim, bias)?;
        let out = TrainableLinear::new(model_dim, model_dim, bias)?;
        Self::new(q, k, v, out, num_heads, model_dim)
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

    /// Forward with optional additive attention mask.
    ///
    /// - `x`: input tensor `[B, S, D]`
    /// - `mask`: optional additive mask `[1, 1, S, S]` (e.g., causal mask with `-inf`).
    ///   Pass `None` for unmasked attention.
    ///
    /// Returns output `[B, S, D]` on the gradient tape.
    pub fn forward_with_mask(
        &self,
        x: &Arc<TrackedTensor>,
        mask: Option<&Arc<TrackedTensor>>,
    ) -> Result<Arc<TrackedTensor>> {
        let dims = x.tensor().dims();
        if dims.len() != 3 {
            return Err(crate::error::AutodiffError::WrongInputRank {
                op: "TrainableMHA",
                expected: 3,
                actual: dims.len(),
            });
        }
        let (b, s, _d) = (dims[0], dims[1], dims[2]);

        // Project Q, K, V: [B, S, D] -> [B, S, D]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to multi-head: [B, S, D] -> [B, S, H, head_dim] -> [B, H, S, head_dim]
        let q = q
            .reshape(&[b, s, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(&[b, s, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[b, s, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        // scores = Q @ K^T / sqrt(head_dim)
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = scores.mul_scalar(scale)?;

        // Apply mask if provided
        let scores = match mask {
            Some(m) => scores.add(m)?,
            None => scores,
        };

        // Attention weights
        let attn_weights = scores.softmax(3)?; // softmax over last dim (S_kv)

        // Weighted values: [B, H, S, head_dim]
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back: [B, H, S, head_dim] -> [B, S, H, head_dim] -> [B, S, D]
        let d = self.num_heads * self.head_dim;
        let out = attn_out.transpose(1, 2)?.reshape(&[b, s, d])?;

        // Output projection
        self.out_proj.forward(&out)
    }
}

impl TrainableModule for TrainableMultiHeadAttention {
    /// Self-attention forward (no mask).
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        self.forward_with_mask(x, None)
    }

    fn vars(&self) -> Vec<&Var> {
        let mut v = self.q_proj.vars();
        v.extend(self.k_proj.vars());
        v.extend(self.v_proj.vars());
        v.extend(self.out_proj.vars());
        v
    }
}
