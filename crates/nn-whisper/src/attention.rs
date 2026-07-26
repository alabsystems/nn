// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-head attention with optional cross-attention and KV cache.
//!
//! Supports both self-attention (with causal mask and accumulated KV cache)
//! and cross-attention (with static KV cache for encoder output). Matches
//! Whisper's attention convention where scale = `(head_dim)^{-0.25}` is
//! applied to both Q and K.
//!
//! Self-attention uses [`KvCacheLayer`] with O(1) amortized doubling buffers
//! instead of O(n²) `DynTensor::cat` per decode step.

use crate::WhisperError;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::attention::{sdpa, sdpa_causal};
use nn_core::layers::kv_cache::KvCacheLayer;
use nn_core::layers::{check_output_finite, Linear, Module};
use nn_core::{Result, VarBuilder};

/// Multi-head attention layer.
///
/// Handles both self-attention and cross-attention.
/// - Cross-attention: caches encoder KV projections (static, compute once).
/// - Self-attention: accumulates KV across decode steps via [`KvCacheLayer`]
///   with O(1) amortized doubling buffers.
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    /// Cached K, V tensors for cross-attention (populated on first call).
    /// Cross-attention KV is static (computed once from encoder output).
    cross_kv_cache: Option<(DynTensor, DynTensor)>,
    /// Accumulated K, V for self-attention across decode steps.
    /// Caches in 4D `[B, H, S, head_dim]` format (sequence at dim=2).
    self_kv_cache: KvCacheLayer,
}

impl MultiHeadAttention {
    /// Load from VarBuilder at the given prefix.
    ///
    /// Expects weight keys: `q_proj.weight`, `q_proj.bias`, `k_proj.weight`,
    /// `v_proj.weight`, `v_proj.bias`, `out_proj.weight`, `out_proj.bias`.
    ///
    /// Note: K projection has no bias in Whisper.
    pub fn load(vb: impl AsRef<VarBuilder>, n_heads: usize, d_model: usize) -> Result<Self> {
        let vb = vb.as_ref();
        if n_heads == 0 {
            return Err(WhisperError::ZeroConfigField { field: "n_heads" }.into());
        }
        if !d_model.is_multiple_of(n_heads) {
            return Err(WhisperError::ConfigNotDivisible {
                a_name: "d_model",
                a_val: d_model,
                b_name: "n_heads",
                b_val: n_heads,
            }
            .into());
        }
        let head_dim = d_model / n_heads;

        let q_w = vb.get(&[d_model, d_model], "q_proj.weight")?;
        let q_b = vb.get(&[d_model], "q_proj.bias")?;
        let q_proj = Linear::new(q_w, Some(q_b))?;

        let k_w = vb.get(&[d_model, d_model], "k_proj.weight")?;
        // Whisper K projection has no bias.
        let k_proj = Linear::new(k_w, None)?;

        let v_w = vb.get(&[d_model, d_model], "v_proj.weight")?;
        let v_b = vb.get(&[d_model], "v_proj.bias")?;
        let v_proj = Linear::new(v_w, Some(v_b))?;

        let out_w = vb.get(&[d_model, d_model], "out_proj.weight")?;
        let out_b = vb.get(&[d_model], "out_proj.bias")?;
        let out_proj = Linear::new(out_w, Some(out_b))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            head_dim,
            cross_kv_cache: None,
            self_kv_cache: KvCacheLayer::empty(),
        })
    }

    /// Forward pass.
    ///
    /// - Self-attention: `xa = None`, `mask = Some(causal_mask_slice)`
    /// - Cross-attention: `xa = Some(encoder_output)`, `mask = None`
    ///
    /// `flush_cache`: when true, clears KV cache. **Must be set to `true`
    /// on the first decoder step of each new audio segment** so that both
    /// self-attention and cross-attention caches are rebuilt. Passing a
    /// different `encoder_output` without flushing will return an error.
    pub fn forward(
        &mut self,
        x: &DynTensor,
        xa: Option<&DynTensor>,
        mask: Option<&DynTensor>,
        flush_cache: bool,
    ) -> Result<DynTensor> {
        let (batch, seq_len, _d_model) = x.dims3()?;

        // Q always comes from decoder input.
        let q = self.q_proj.forward(x)?;

        // Whisper-specific scale: (head_dim)^{-0.25} applied to both Q and K.
        let scale = (self.head_dim as f64).powf(-0.25);

        // K, V: from x (self-attn) or xa (cross-attn, with cache).
        // Self-attention caches in 4D [B, H, S, head_dim] via KvCacheLayer.
        // Cross-attention caches in 3D [B, S, D] (compute-once, static).
        let (k_4d, v_4d) = match xa {
            None => {
                // Self-attention: project, reshape to 4D, cache via KvCacheLayer.
                if flush_cache {
                    self.self_kv_cache.reset();
                }
                let new_k = self.k_proj.forward(x)?;
                let new_v = self.v_proj.forward(x)?;

                // Scale K before caching (Whisper convention).
                let new_k = new_k.mul_scalar(scale)?;

                // Reshape [B, S, D] -> [B, H, S, head_dim] for cache.
                let new_k_4d = new_k
                    .reshape([batch, seq_len, self.n_heads, self.head_dim])?
                    .transpose(1, 2)?;
                let new_v_4d = new_v
                    .reshape([batch, seq_len, self.n_heads, self.head_dim])?
                    .transpose(1, 2)?;

                // Append to cache and get full K/V as 4D tensors.
                self.self_kv_cache.append(&new_k_4d, &new_v_4d)?
            }
            Some(encoder_out) => {
                // Validate batch dimension matches between decoder and encoder.
                let enc_batch = encoder_out.dim(0)?;
                if enc_batch != batch {
                    return Err(WhisperError::BatchMismatch {
                        encoder_batch: enc_batch,
                        decoder_batch: batch,
                    }
                    .into());
                }
                if flush_cache {
                    self.cross_kv_cache = None;
                }
                if let Some((ref cached_k, ref cached_v)) = self.cross_kv_cache {
                    // Defense-in-depth: detect stale cache when encoder_output
                    // shape doesn't match what was cached. The seq_len dimension
                    // (dim=2 in 4D [B, H, S, head_dim]) is the cheapest check
                    // that catches different audio segments.
                    let cached_seq = cached_k.dim(2)?;
                    let enc_seq = encoder_out.dim(1)?;
                    if cached_seq != enc_seq {
                        return Err(WhisperError::CacheSeqMismatch {
                            cached_seq,
                            encoder_seq: enc_seq,
                        }
                        .into());
                    }
                    (cached_k.clone(), cached_v.clone())
                } else {
                    let k = self.k_proj.forward(encoder_out)?;
                    let v = self.v_proj.forward(encoder_out)?;

                    // Scale K before caching.
                    let k = k.mul_scalar(scale)?;

                    let enc_len = encoder_out.dim(1)?;
                    // Reshape to 4D for consistent handling.
                    let k_4d = k
                        .reshape([batch, enc_len, self.n_heads, self.head_dim])?
                        .transpose(1, 2)?;
                    let v_4d = v
                        .reshape([batch, enc_len, self.n_heads, self.head_dim])?
                        .transpose(1, 2)?;
                    self.cross_kv_cache = Some((k_4d.clone(), v_4d.clone()));
                    (k_4d, v_4d)
                }
            }
        };

        // Scale Q (K was already scaled before caching).
        let q = q.mul_scalar(scale)?;

        // Reshape Q: [B, S, D] -> [B, H, S, head_dim] (4D for sdpa).
        let q = q
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        // K, V are already 4D [B, H, S, head_dim].
        // Q and K are pre-scaled by head_dim^{-0.25}, so their product
        // already carries the standard 1/sqrt(head_dim) factor → scale=1.0.
        //
        // Flash Attention optimization: GPU Flash Attention rejects explicit
        // mask tensors (falls back to 3 separate dispatches: matmul + softmax
        // + matmul). Two cases allow the fused single-dispatch path:
        //
        // 1. seq_len == 1 (token-by-token decode): the causal mask at any
        //    position P is mask[P, 0..P+1] — all zeros (every cached position
        //    is visible to the single new token). Passing None activates
        //    Flash Attention.
        //
        // 2. S_q == S_kv (initial prompt after cache flush): use sdpa_causal()
        //    for fused causal masking without the O(S²) mask tensor.
        let s_kv = k_4d.dim(2)?;
        let attn_output = if mask.is_some() && seq_len == 1 {
            // Case 1: single-token decode — causal mask is a no-op.
            sdpa(&q, &k_4d, &v_4d, None, 1.0)?
        } else if mask.is_some() && seq_len == s_kv {
            // Case 2: S_q == S_kv — fused causal masking (Flash Attention).
            sdpa_causal(&q, &k_4d, &v_4d, 1.0)?
        } else {
            // Cross-attention (mask=None) or multi-token with KV cache.
            let mask_4d = match mask {
                Some(m) => Some(m.unsqueeze(0)?.unsqueeze(0)?),
                None => None,
            };
            sdpa(&q, &k_4d, &v_4d, mask_4d.as_ref(), 1.0)?
        };

        // Reshape: [B, H, S, head_dim] -> [B, S, D]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        let output = self.out_proj.forward(&attn_output)?;
        check_output_finite(&output, "MultiHeadAttention")?;
        Ok(output)
    }

    /// Cache-free self-attention forward for tracing.
    ///
    /// Equivalent to `forward(x, None, None, false)` but avoids KV cache
    /// `slice_set` ops that the trace compiler cannot handle. Produces a
    /// clean computation graph for `trace_graph()` and `CompiledModel`.
    pub fn forward_self_attn_no_cache(&self, x: &DynTensor) -> Result<DynTensor> {
        let (batch, seq_len, _d_model) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let scale = (self.head_dim as f64).powf(-0.25);
        let k = k.mul_scalar(scale)?;
        let q = q.mul_scalar(scale)?;

        let q = q
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        let attn_output = sdpa(&q, &k, &v, None, 1.0)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        let output = self.out_proj.forward(&attn_output)?;
        check_output_finite(&output, "MultiHeadAttention")?;
        Ok(output)
    }

    /// Cache-free self-attention with optional mask for tracing.
    ///
    /// Like `forward_self_attn_no_cache` but accepts an optional causal mask,
    /// needed for decoder self-attention where causal masking is required.
    pub fn forward_self_attn_no_cache_masked(
        &self,
        x: &DynTensor,
        mask: Option<&DynTensor>,
    ) -> Result<DynTensor> {
        let (batch, seq_len, _d_model) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let scale = (self.head_dim as f64).powf(-0.25);
        let k = k.mul_scalar(scale)?;
        let q = q.mul_scalar(scale)?;

        let q = q
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        // No-cache: S_q == S_kv always. Use sdpa_causal for fused GPU path.
        let attn_output = if mask.is_some() {
            sdpa_causal(&q, &k, &v, 1.0)?
        } else {
            sdpa(&q, &k, &v, None, 1.0)?
        };
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        let output = self.out_proj.forward(&attn_output)?;
        check_output_finite(&output, "MultiHeadAttention")?;
        Ok(output)
    }

    /// Cache-free cross-attention forward for tracing.
    ///
    /// Computes cross-attention from decoder hidden state `x` attending to
    /// `encoder_output` without caching KV projections. Produces a clean
    /// computation graph for `trace_graph()` and `CompiledModel`.
    pub fn forward_cross_attn_no_cache(
        &self,
        x: &DynTensor,
        encoder_output: &DynTensor,
    ) -> Result<DynTensor> {
        let (batch, seq_len, _d_model) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(encoder_output)?;
        let v = self.v_proj.forward(encoder_output)?;

        let scale = (self.head_dim as f64).powf(-0.25);
        let k = k.mul_scalar(scale)?;
        let q = q.mul_scalar(scale)?;

        let enc_len = encoder_output.dim(1)?;

        let q = q
            .reshape([batch, seq_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, enc_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, enc_len, self.n_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Cross-attention has no mask.
        let attn_output = sdpa(&q, &k, &v, None, 1.0)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ])?;

        let output = self.out_proj.forward(&attn_output)?;
        check_output_finite(&output, "MultiHeadAttention")?;
        Ok(output)
    }

    /// Current self-attention KV cache sequence length (0 if empty).
    pub fn self_cache_len(&self) -> usize {
        self.self_kv_cache.seq_len()
    }

    /// Clear all KV caches (call between utterances).
    pub fn reset_cache(&mut self) {
        self.cross_kv_cache = None;
        self.self_kv_cache.reset();
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_attention_proofs.rs"]
mod kani_attention_proofs;

#[cfg(kani)]
#[path = "kani_attention_proofs_ext.rs"]
mod kani_attention_proofs_ext;
