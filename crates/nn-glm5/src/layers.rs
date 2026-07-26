// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Internal decoder layer components: MLP, Attention, DecoderLayer.
//!
//! GLM-4/5 differs from Llama/Qwen in several ways:
//! - Fused QKV projection (`query_key_value` weight, split after projection)
//! - SwiGLU MLP with fused gate+up (`dense_h_to_4h` of size `ffn_hidden_size * 2`)
//! - Optional QKV bias (`add_qkv_bias`)
//! - No QK-Norm (unlike Qwen3)

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCacheLayer;
use nn_core::layers::{
    self, check_output_finite, repeat_kv, HalfRotaryEmbedding, Linear, Module, RmsNorm,
};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

use crate::Glm5Config;

#[cfg(kani)]
#[path = "kani_layers.rs"]
mod kani_proofs;

// -- SwiGLU MLP ---------------------------------------------------------------

/// GLM MLP with SwiGLU activation.
///
/// `dense_h_to_4h` projects to `ffn_hidden_size * 2` then splits into gate+up.
/// `dense_4h_to_h` projects back to `hidden_size`.
pub(crate) struct Glm5MLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl Glm5MLP {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &Glm5Config) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;
        let ffn = cfg.ffn_hidden_size;

        // dense_h_to_4h: [ffn_hidden_size * 2, hidden_size] (gate+up fused)
        let bias_h_to_4h = if cfg.add_bias_linear {
            Some(vb.get(&[ffn * 2], "dense_h_to_4h.bias")?)
        } else {
            None
        };
        let dense_h_to_4h =
            Linear::new(vb.get(&[ffn * 2, h], "dense_h_to_4h.weight")?, bias_h_to_4h)?;

        let bias_4h_to_h = if cfg.add_bias_linear {
            Some(vb.get(&[h], "dense_4h_to_h.bias")?)
        } else {
            None
        };
        let dense_4h_to_h = Linear::new(vb.get(&[h, ffn], "dense_4h_to_h.weight")?, bias_4h_to_h)?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    pub(crate) fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // Project to ffn_hidden_size * 2, then split and apply SwiGLU
        let intermediate = self.dense_h_to_4h.forward(x)?;

        // Split along last dim into gate and up
        let last_dim = intermediate.dims().len() - 1;
        let half_size = intermediate.dims()[last_dim] / 2;
        let gate = intermediate.narrow(last_dim, 0, half_size)?;
        let up = intermediate.narrow(last_dim, half_size, half_size)?;

        // SwiGLU: silu(gate) * up
        let activated = gate.silu()?.broadcast_mul(&up)?;
        let out = self.dense_4h_to_h.forward(&activated)?;
        check_output_finite(&out, "Glm5MLP")?;
        Ok(out)
    }
}

// -- Attention ----------------------------------------------------------------

/// GLM attention with fused QKV projection and multi-query groups.
pub(crate) struct Glm5Attention {
    query_key_value: Linear,
    dense: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Glm5Attention {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &Glm5Config) -> Result<Self> {
        let vb = vb.as_ref();
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        let nh = cfg.num_attention_heads;
        let nkv = cfg.multi_query_group_num;

        // Fused QKV: output size = (nh + 2 * nkv) * hd
        let qkv_size = (nh + 2 * nkv) * hd;
        let qkv_bias = if cfg.add_qkv_bias {
            Some(vb.get(&[qkv_size], "query_key_value.bias")?)
        } else {
            None
        };
        let query_key_value =
            Linear::new(vb.get(&[qkv_size, h], "query_key_value.weight")?, qkv_bias)?;

        let dense_bias = if cfg.add_bias_linear {
            Some(vb.get(&[h], "dense.bias")?)
        } else {
            None
        };
        let dense = Linear::new(vb.get(&[h, nh * hd], "dense.weight")?, dense_bias)?;

        Ok(Self {
            query_key_value,
            dense,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &DynTensor,
        rope: &HalfRotaryEmbedding,
        positions: &[usize],
        mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let hd = self.head_dim;

        // Fused QKV projection
        let qkv = self.query_key_value.forward(x)?;

        // Split into Q, K, V along last dimension
        let q_size = self.num_heads * hd;
        let kv_size = self.num_kv_heads * hd;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, hd])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, self.num_kv_heads, hd])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, self.num_kv_heads, hd])?
            .transpose(1, 2)?;

        // Apply half-RoPE: first head_dim/2 dims rotated, rest pass-through
        let (q, k) = rope.apply_pair(&q, &k, positions)?;

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
        let scale = 1.0 / (hd as f64).sqrt();
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
            self.num_heads * hd,
        ])?;

        let out = self.dense.forward(&attn_output)?;
        check_output_finite(&out, "Glm5Attention")?;
        Ok(out)
    }
}

// -- Decoder Layer ------------------------------------------------------------

pub(crate) struct Glm5DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Glm5Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Glm5MLP,
}

impl Glm5DecoderLayer {
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &Glm5Config) -> Result<Self> {
        let vb = vb.as_ref();
        let input_layernorm = RmsNorm::new(
            vb.get(&[cfg.hidden_size], "input_layernorm.weight")?,
            cfg.layernorm_epsilon,
        )?;
        let self_attn = Glm5Attention::load(vb.pp("self_attention"), cfg)?;
        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[cfg.hidden_size], "post_attention_layernorm.weight")?,
            cfg.layernorm_epsilon,
        )?;
        let mlp = Glm5MLP::load(vb.pp("mlp"), cfg)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &DynTensor,
        rope: &HalfRotaryEmbedding,
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
        check_output_finite(&output, "Glm5DecoderLayer")?;
        Ok(output)
    }
}
