// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Standard multi-head attention for transformer models.
//!
//! Provides [`MultiHeadAttention`] supporting self-attention, cross-attention,
//! grouped-query attention (GQA/MQA), optional RoPE, and KV caching for
//! autoregressive decoding.
//!
//! Standalone SDPA utilities (scaled dot-product attention, causal masks,
//! K/V head repetition) live in the sibling [`sdpa`](super::sdpa) module.
//!
//! Design: `designs/2026-03-04-multi-head-attention.md`

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCacheLayer;
use crate::layers::{
    check_output_finite, validate_divisible, validate_heads, Linear, Module, RotaryEmbedding,
};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

// sdpa, causal_mask, repeat_kv live in the sibling `sdpa` module and are
// imported here for internal use. Public re-exports go through attention/mod.rs.
use super::sdpa::{repeat_kv, sdpa};

/// Standard multi-head attention with optional GQA, KV cache, and RoPE.
///
/// Supports self-attention (`kv_input = None`), cross-attention (`kv_input = Some(...)`),
/// and autoregressive decoding (with [`KvCacheLayer`]).
///
/// When `num_kv_heads < num_heads`, K/V heads are repeated via [`repeat_kv`] for
/// grouped-query attention (GQA). When `num_kv_heads == 1`, this becomes multi-query
/// attention (MQA).
#[derive(Clone)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttention")
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .finish_non_exhaustive()
    }
}

impl MultiHeadAttention {
    /// Create from pre-loaded projection weights.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `num_heads` or `num_kv_heads` is 0
    /// - `num_heads` is not divisible by `num_kv_heads` (GQA constraint)
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        validate_heads(num_heads, "MultiHeadAttention")?;
        validate_heads(num_kv_heads, "MultiHeadAttention (num_kv_heads)")?;
        validate_divisible(
            num_heads,
            num_kv_heads,
            "num_heads",
            "num_kv_heads",
            "MultiHeadAttention",
        )?;
        // Infer head_dim from Q projection weight shape [num_heads * head_dim, dim]
        let q_weight = q_proj.weight();
        if q_weight.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: q_weight.rank(),
            });
        }
        let q_out_features = q_weight.dim(0)?;
        validate_divisible(
            q_out_features,
            num_heads,
            "q_proj out_features",
            "num_heads",
            "MultiHeadAttention",
        )?;
        let head_dim = q_out_features / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    /// Load from a [`VarBuilder`] with standard weight names.
    ///
    /// Loads `q_proj.weight`, `q_proj.bias`, `k_proj.weight`, `k_proj.bias`,
    /// `v_proj.weight`, `v_proj.bias`, `out_proj.weight`, `out_proj.bias`.
    /// When `bias` is false, bias tensors are not loaded.
    ///
    /// - `dim`: model dimension (input/output feature size)
    /// - `num_heads`: number of query attention heads
    /// - `num_kv_heads`: number of key/value heads (< num_heads for GQA)
    /// - `bias`: whether to load bias parameters
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        bias: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        // Validate heads early to prevent division-by-zero below.
        // new() re-validates, but we need num_heads > 0 for head_dim computation.
        validate_heads(num_heads, "MultiHeadAttention::load")?;
        validate_heads(num_kv_heads, "MultiHeadAttention::load (num_kv_heads)")?;
        let head_dim = dim / num_heads;
        if head_dim == 0 || dim != num_heads * head_dim {
            return Err(TensorError::ValueOutOfRange {
                description: "MultiHeadAttention::load: dim not evenly divisible by num_heads",
            });
        }
        let kv_dim = num_kv_heads * head_dim;

        let load_linear = |prefix: &str, out_features: usize| -> Result<Linear> {
            let sub = vb.pp(prefix);
            let w = sub.get(&[out_features, dim], "weight")?;
            let b = if bias {
                Some(sub.get(&[out_features], "bias")?)
            } else {
                None
            };
            Linear::new(w, b)
        };

        let q_proj = load_linear("q_proj", dim)?;
        let k_proj = load_linear("k_proj", kv_dim)?;
        let v_proj = load_linear("v_proj", kv_dim)?;
        let out_proj = load_linear("out_proj", dim)?;

        Self::new(q_proj, k_proj, v_proj, out_proj, num_heads, num_kv_heads)
    }

    /// Standard forward: self-attention or cross-attention.
    ///
    /// - `x`: query source `[B, S_q, D]`
    /// - `kv_input`: K/V source. `None` = self-attention (K/V from `x`).
    ///   `Some(enc)` = cross-attention (K/V from encoder output).
    /// - `mask`: attention mask `[1, 1, S_q, S_kv]` or broadcastable.
    ///   Use [`causal_mask`] for autoregressive. `None` = no mask.
    /// - `rope`: optional [`RotaryEmbedding`] to apply to Q and K after projection.
    /// - `rope_offset`: position offset for RoPE (0 for prefill, `seq_len` for decode).
    pub fn forward(
        &self,
        x: &DynTensor,
        kv_input: Option<&DynTensor>,
        mask: Option<&DynTensor>,
        rope: Option<&RotaryEmbedding>,
        rope_offset: usize,
    ) -> Result<DynTensor> {
        let tracing = trace::is_tracing();

        let compute = || -> Result<DynTensor> {
            let (b, s_q, _d) = x.dims3()?;
            let kv_src = kv_input.unwrap_or(x);
            let s_kv = kv_src.dim(1)?;

            // Project Q, K, V
            let q = self.q_proj.forward(x)?;
            let k = self.k_proj.forward(kv_src)?;
            let v = self.v_proj.forward(kv_src)?;

            // Reshape to multi-head: [B, S, D] -> [B, S, H, head_dim] -> [B, H, S, head_dim]
            let q = q
                .reshape([b, s_q, self.num_heads, self.head_dim])?
                .transpose(1, 2)?;
            let k = k
                .reshape([b, s_kv, self.num_kv_heads, self.head_dim])?
                .transpose(1, 2)?;
            let v = v
                .reshape([b, s_kv, self.num_kv_heads, self.head_dim])?
                .transpose(1, 2)?;

            // Optional RoPE
            let (q, k) = if let Some(rope) = rope {
                let q = rope.apply(&q, rope_offset)?;
                let k = rope.apply(&k, rope_offset)?;
                (q, k)
            } else {
                (q, k)
            };

            // GQA: repeat K/V heads to match Q heads
            let k = repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
            let v = repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

            self.attend(&q, &k, &v, mask, b, s_q)
        };

        let mut result = if tracing {
            trace::with_trace_suppressed(compute)?
        } else {
            compute()?
        };

        if tracing {
            let mut inputs: Vec<&DynTensor> = vec![x];
            if let Some(kv) = kv_input {
                inputs.push(kv);
            }
            let input_ids = DynTensor::trace_input_ids(&inputs)?;
            if let Some(id) = trace::record_op(
                TraceOp::MultiHeadAttention {
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                    head_dim: self.head_dim,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }

    /// Forward with KV cache for autoregressive decoding.
    ///
    /// Appends new K/V to cache, attends over full cached sequence.
    /// `x` is typically a single token `[B, 1, D]` during decode.
    pub fn forward_kv_cached(
        &self,
        x: &DynTensor,
        kv_input: Option<&DynTensor>,
        cache: &mut KvCacheLayer,
        mask: Option<&DynTensor>,
        rope: Option<&RotaryEmbedding>,
        rope_offset: usize,
    ) -> Result<DynTensor> {
        let (b, s_q, _d) = x.dims3()?;
        let kv_src = kv_input.unwrap_or(x);
        let s_kv_new = kv_src.dim(1)?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(kv_src)?;
        let v = self.v_proj.forward(kv_src)?;

        // Reshape to multi-head
        let q = q
            .reshape([b, s_q, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([b, s_kv_new, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([b, s_kv_new, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Optional RoPE (applied before caching)
        let (q, k) = if let Some(rope) = rope {
            let q = rope.apply(&q, rope_offset)?;
            let k = rope.apply(&k, rope_offset)?;
            (q, k)
        } else {
            (q, k)
        };

        // Append to KV cache and get full K/V
        let (full_k, full_v) = cache.append(&k, &v)?;

        // GQA: repeat K/V heads
        let full_k = repeat_kv(&full_k, self.num_heads / self.num_kv_heads)?;
        let full_v = repeat_kv(&full_v, self.num_heads / self.num_kv_heads)?;

        self.attend(&q, &full_k, &full_v, mask, b, s_q)
    }

    /// Attend and project output.
    fn attend(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
        mask: Option<&DynTensor>,
        batch: usize,
        s_q: usize,
    ) -> Result<DynTensor> {
        let attn_out = sdpa(q, k, v, mask, self.scale)?;

        // Reshape: [B, H, S_q, head_dim] -> [B, S_q, H, head_dim] -> [B, S_q, D]
        let d = self.num_heads * self.head_dim;
        let out = attn_out.transpose(1, 2)?.reshape([batch, s_q, d])?;

        let result = self.out_proj.forward(&out)?;
        check_output_finite(&result, "MultiHeadAttention")?;
        Ok(result)
    }

    /// Number of query attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of key/value attention heads (may differ for GQA).
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Dimension per attention head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// Self-attention via [`Module`] trait (no mask, no RoPE, no cache).
impl Module for MultiHeadAttention {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.forward(x, None, None, None, 0)
    }
}

#[cfg(test)]
#[path = "multi_head_tests.rs"]
mod tests;
