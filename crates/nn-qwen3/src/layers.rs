// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Internal decoder layer components: MLP, Attention, DecoderLayer.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCacheLayer;
use nn_core::layers::{
    self, check_output_finite, repeat_kv, Linear, Module, RmsNorm, RotaryEmbedding,
};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

use crate::Qwen3Config;

// -- SwiGLU MLP ---------------------------------------------------------------

pub(crate) struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &Qwen3Config) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = Linear::new(vb.get(&[i, h], "gate_proj.weight")?, None)?;
        let up_proj = Linear::new(vb.get(&[i, h], "up_proj.weight")?, None)?;
        let down_proj = Linear::new(vb.get(&[h, i], "down_proj.weight")?, None)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub(crate) fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let out = self.down_proj.forward(&gate.broadcast_mul(&up)?)?;
        check_output_finite(&out, "Qwen3MLP")?;
        Ok(out)
    }
}

// -- Attention ----------------------------------------------------------------

pub(crate) struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen3Attention {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &Qwen3Config) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;

        let q_proj = Linear::new(vb.get(&[nh * hd, h], "q_proj.weight")?, None)?;
        let k_proj = Linear::new(vb.get(&[nkv * hd, h], "k_proj.weight")?, None)?;
        let v_proj = Linear::new(vb.get(&[nkv * hd, h], "v_proj.weight")?, None)?;
        let o_proj = Linear::new(vb.get(&[h, nh * hd], "o_proj.weight")?, None)?;

        // QK-Norm: per-head RMSNorm on head_dim
        let q_norm = RmsNorm::new(vb.get(&[hd], "q_norm.weight")?, cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::new(vb.get(&[hd], "k_norm.weight")?, cfg.rms_norm_eps)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // QK-Norm
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply RoPE using half-split (rotate_half) convention, matching
        // HuggingFace Qwen3/LLaMA directly. Pairs element (i, i+half_dim).
        //
        // IMPORTANT: This requires UNPERMUTED HuggingFace weights. Do NOT
        // permute Q/K weight columns for interleaved RoPE — that convention
        // (pairing 2i, 2i+1) was replaced in commit 62ddb7398. Weight
        // permutation + half-split RoPE = wrong rotation (#4327).
        let (q, k) = rope.apply_pair_half_split(&q, &k, positions)?;

        // KV cache
        let (k, v) = match cache {
            Some(c) => c.append(&k, &v)?,
            None => (k, v),
        };

        // GQA: repeat K, V to match Q heads
        let k = repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        // Scaled dot-product attention.
        // Routes to Flash Attention on GPU (#2434), decomposed path on CPU.
        //
        // When mask is provided and S_q == S_kv (initial prompt with fresh
        // cache), use sdpa_causal for fused causal masking — avoids Flash
        // Attention skip caused by explicit mask tensor.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let s_kv = k.dim(2)?;
        let attn_output = if mask.is_some() && seq_len == s_kv {
            layers::sdpa_causal(&q, &k, &v, scale)?
        } else {
            layers::sdpa(&q, &k, &v, mask, scale)?
        };

        // Reshape back: [batch, seq, nh * hd]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        let out = self.o_proj.forward(&attn_output)?;
        check_output_finite(&out, "Qwen3Attention")?;
        Ok(out)
    }
}

// -- Decoder Layer ------------------------------------------------------------

use crate::forward_common::DecoderLayer;

pub(crate) struct Qwen3DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Qwen3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Qwen3MLP,
}

impl Qwen3DecoderLayer {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &Qwen3Config) -> Result<Self> {
        let vb = vb.as_ref();
        let input_layernorm = RmsNorm::new(
            vb.get(&[cfg.hidden_size], "input_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let self_attn = Qwen3Attention::load(vb.pp("self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[cfg.hidden_size], "post_attention_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;
        let mlp = Qwen3MLP::load(vb.pp("mlp"), cfg)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }
}

impl DecoderLayer for Qwen3DecoderLayer {
    fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rope, positions, mask, cache)?;
        let x = residual.broadcast_add(&x)?;

        // Pre-norm MLP with residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let output = residual.broadcast_add(&x)?;
        check_output_finite(&output, "Qwen3DecoderLayer")?;
        Ok(output)
    }
}
