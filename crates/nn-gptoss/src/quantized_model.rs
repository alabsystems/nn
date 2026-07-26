// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quantized gpt-oss model with MXFP4 MoE expert weights.
//!
//! Reduces the 20B MoE model from ~42GB (BF16) to ~17GB by quantizing
//! the expert gate_up_proj and down_proj tensors to MXFP4 (4.25 bits/param).
//! Attention, norms, embeddings, and biases stay at full precision.

use crate::config::GptOssConfig;
use crate::layers::GptOssAttention;
use crate::quantize::{Mxfp4Tensor, QuantizationReport};

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::attention::sliding_window_mask;
use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{
    causal_mask_with_offset, check_output_finite, with_nan_check_policy, Embedding, Linear, Module,
    NanCheckPolicy, RmsNorm, RotaryEmbedding,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};

// -- Quantized Decoder Layer --------------------------------------------------

struct QuantizedDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: GptOssAttention,
    post_attention_layernorm: RmsNorm,
    router: Linear,
    expert_gate_up: Mxfp4Tensor,
    expert_gate_up_bias: DynTensor,
    expert_down: Mxfp4Tensor,
    expert_down_bias: DynTensor,
    num_experts: usize,
    top_k: usize,
    swiglu_limit: f64,
    intermediate_size: usize,
    layer_type: crate::config::LayerType,
}

impl QuantizedDecoderLayer {
    fn load(
        vb: &VarBuilder,
        cfg: &GptOssConfig,
        layer_type: crate::config::LayerType,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let ne = cfg.num_local_experts;
        let inter = cfg.intermediate_size;
        let fused_dim = 2 * inter;

        let input_layernorm =
            RmsNorm::new(vb.get(&[h], "input_layernorm.weight")?, cfg.rms_norm_eps)?;
        let self_attn = GptOssAttention::load(vb.pp("self_attn"), cfg)?;
        let post_attention_layernorm = RmsNorm::new(
            vb.get(&[h], "post_attention_layernorm.weight")?,
            cfg.rms_norm_eps,
        )?;

        let mlp_vb = vb.pp("mlp");
        let router = Linear::new(
            mlp_vb.get(&[ne, h], "router.weight")?,
            Some(mlp_vb.get(&[ne], "router.bias")?),
        )?;

        let experts_vb = mlp_vb.pp("experts");

        let gate_up_tensor = experts_vb.get(&[ne, h, fused_dim], "gate_up_proj")?;
        let gate_up_f32 = gate_up_tensor.to_flat_vec::<f32>()?;
        let expert_gate_up = Mxfp4Tensor::quantize(&gate_up_f32, &[ne, h, fused_dim]);

        let expert_gate_up_bias = experts_vb.get(&[ne, fused_dim], "gate_up_proj_bias")?;

        let down_tensor = experts_vb.get(&[ne, h, h], "down_proj")?;
        let down_f32 = down_tensor.to_flat_vec::<f32>()?;
        let expert_down = Mxfp4Tensor::quantize(&down_f32, &[ne, h, h]);

        let expert_down_bias = experts_vb.get(&[ne, h], "down_proj_bias")?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            router,
            expert_gate_up,
            expert_gate_up_bias,
            expert_down,
            expert_down_bias,
            num_experts: ne,
            top_k: cfg.experts_per_token,
            swiglu_limit: cfg.swiglu_limit,
            intermediate_size: inter,
            layer_type,
        })
    }

    fn expert_forward(
        &self,
        x: &DynTensor,
        expert_idx: usize,
        device: &Device,
    ) -> Result<DynTensor> {
        let inter = self.intermediate_size;

        let gate_up_all = self.expert_gate_up.to_dyn_tensor(device)?;
        let gate_up_w = gate_up_all.narrow(0, expert_idx, 1)?.squeeze(0)?;
        let gate_up_b = self
            .expert_gate_up_bias
            .narrow(0, expert_idx, 1)?
            .squeeze(0)?;

        let gate_up = x.matmul(&gate_up_w)?.broadcast_add(&gate_up_b)?;

        let last = gate_up.rank() - 1;
        let gate = gate_up.narrow(last, 0, inter)?;
        let up = gate_up.narrow(last, inter, inter)?;

        let gate = gate.silu()?.clamp(-self.swiglu_limit, self.swiglu_limit)?;
        let hidden = gate.broadcast_mul(&up)?;

        let down_all = self.expert_down.to_dyn_tensor(device)?;
        let down_w = down_all.narrow(0, expert_idx, 1)?.squeeze(0)?;
        let down_b = self.expert_down_bias.narrow(0, expert_idx, 1)?.squeeze(0)?;

        let out = hidden.matmul(&down_w)?.broadcast_add(&down_b)?;
        check_output_finite(&out, "QuantizedMoeBlock::expert_forward")?;
        Ok(out)
    }

    fn moe_forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let x_dims = x.dims();
        let rank = x.rank();
        let last_dim = rank - 1;
        let device = x.device();

        let logits = self.router.forward(x)?;
        let logits_last = logits.rank() - 1;
        let probs = logits.softmax(logits_last)?;
        let (topk_weights, topk_indices) = probs.topk(logits_last, self.top_k)?;

        let w_sum = topk_weights.sum_keepdim(logits_last)?;
        let topk_weights = topk_weights.broadcast_div(&w_sum)?;

        let n_tokens = nn_core::tensor::checked_dim_product(&x_dims[..last_dim])?;
        let model_dim = x_dims[last_dim];
        let flat_x = x.reshape([n_tokens, model_dim])?;
        let flat_indices = topk_indices.reshape([n_tokens, self.top_k])?;
        let flat_weights = topk_weights.reshape([n_tokens, self.top_k])?;

        let idx_flat = flat_indices.to_flat_vec::<u32>()?;
        let wt_flat = flat_weights.to_flat_vec::<f32>()?;

        let mut output = DynTensor::zeros(&[n_tokens, model_dim], DType::F32, &device)?;

        let avg = (n_tokens * self.top_k) / self.num_experts.max(1) + 1;
        let mut assignments: Vec<Vec<(usize, f32)>> = (0..self.num_experts)
            .map(|_| Vec::with_capacity(avg))
            .collect();
        for t in 0..n_tokens {
            for s in 0..self.top_k {
                let flat_idx = t * self.top_k + s;
                let expert_idx = idx_flat[flat_idx] as usize;
                if expert_idx < self.num_experts {
                    assignments[expert_idx].push((t, wt_flat[flat_idx]));
                }
            }
        }

        for (expert_idx, expert_assignments) in assignments.iter().enumerate() {
            if expert_assignments.is_empty() {
                continue;
            }
            let num_routed = expert_assignments.len();
            let token_ids: Vec<u32> = expert_assignments
                .iter()
                .map(|&(t, _)| u32::try_from(t).unwrap_or(u32::MAX))
                .collect();
            let weights: Vec<f32> = expert_assignments.iter().map(|&(_, w)| w).collect();

            let ids_tensor = DynTensor::from_vec_u32(token_ids, &[num_routed], &device)?;
            let gathered = flat_x.index_select(&ids_tensor, 0)?;
            let expert_out = self.expert_forward(&gathered, expert_idx, &device)?;

            let w_tensor = DynTensor::from_vec(weights, &[num_routed, 1], &device)?;
            let weighted = expert_out.broadcast_mul(&w_tensor)?;
            output = output.index_add(0, &ids_tensor, &weighted)?;
        }

        output = output.reshape(x_dims)?;
        check_output_finite(&output, "QuantizedMoeBlock")?;
        Ok(output)
    }

    fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        causal_mask: Option<&DynTensor>,
        sliding_mask: Option<&DynTensor>,
        cache: Option<&mut nn_core::layers::kv_cache::KvCacheLayer>,
    ) -> Result<DynTensor> {
        let mask = match self.layer_type {
            crate::config::LayerType::SlidingAttention => sliding_mask,
            crate::config::LayerType::FullAttention => causal_mask,
        };

        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rope, positions, mask, cache)?;
        let x = residual.broadcast_add(&x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.moe_forward(&x)?;
        let output = residual.broadcast_add(&x)?;
        check_output_finite(&output, "QuantizedDecoderLayer")?;
        Ok(output)
    }
}

// -- Quantized Model ----------------------------------------------------------

/// gpt-oss model with MXFP4-quantized MoE expert weights.
///
/// Reduces memory from ~42GB (BF16) to ~17GB by quantizing the expert
/// gate_up_proj and down_proj tensors. Attention, norms, embeddings,
/// router, and biases stay at full precision.
pub struct GptOssQuantizedModel {
    embed_tokens: Embedding,
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    cfg: GptOssConfig,
    #[allow(dead_code)]
    dtype: DType,
}

impl GptOssQuantizedModel {
    /// Load from VarBuilder, quantizing MoE expert weights to MXFP4.
    pub fn load_quantized(vb: &VarBuilder, cfg: GptOssConfig) -> Result<Self> {
        cfg.validate()?;
        let model_vb = vb.pp("model");

        let embed_weight =
            model_vb.get(&[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_weight.clone())?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            let layer_type = cfg.layer_types[i];
            layers.push(QuantizedDecoderLayer::load(&layer_vb, &cfg, layer_type)?);
        }

        let norm = RmsNorm::new(
            model_vb.get(&[cfg.hidden_size], "norm.weight")?,
            cfg.rms_norm_eps,
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_weight, None)?
        } else {
            Linear::new(
                vb.get(&[cfg.vocab_size, cfg.hidden_size], "lm_head.weight")?,
                None,
            )?
        };

        let rope = crate::build_rope(&cfg, vb)?;
        let dtype = vb.dtype();

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            cfg,
            dtype,
        })
    }

    /// Forward pass: token IDs + positions -> logits.
    pub fn forward(&self, input_ids: &[usize], positions: &[usize]) -> Result<DynTensor> {
        crate::validate_forward_input(input_ids, positions)?;
        let x = crate::embed_and_unsqueeze(&self.embed_tokens, input_ids)?;
        self.forward_inner(x, positions, None)
    }

    /// Forward pass with KV cache.
    pub fn forward_cached(
        &self,
        input_ids: &[usize],
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        crate::validate_forward_input(input_ids, positions)?;
        let x = crate::embed_and_unsqueeze(&self.embed_tokens, input_ids)?;
        self.forward_inner(x, positions, cache)
    }

    fn forward_inner(
        &self,
        x: DynTensor,
        positions: &[usize],
        cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        let logits = with_nan_check_policy(NanCheckPolicy::Skip, || {
            let normed = self.forward_decoder_and_norm(x, positions, cache)?;
            self.lm_head.forward(&normed)
        })?;
        check_output_finite(&logits, "GptOssQuantizedModel")?;
        Ok(logits)
    }

    fn forward_decoder_and_norm(
        &self,
        mut x: DynTensor,
        positions: &[usize],
        mut cache: Option<&mut KvCache>,
    ) -> Result<DynTensor> {
        crate::validate_cache(cache.as_deref(), self.layers.len())?;

        let seq_len = positions.len();
        let cached_len = cache.as_deref().map_or(0, KvCache::seq_len);
        let total_seq = cached_len + seq_len;

        let causal_mask = if seq_len > 1 && total_seq > 1 {
            Some(causal_mask_with_offset(
                seq_len,
                total_seq,
                x.dtype(),
                &x.device(),
            )?)
        } else {
            None
        };

        let sliding_mask = if seq_len > 1 {
            let sw = sliding_window_mask(seq_len, self.cfg.sliding_window, &x.device())?;
            if let Some(ref causal) = causal_mask {
                Some(sw.minimum(causal)?)
            } else {
                Some(sw)
            }
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = match cache {
                Some(ref mut c) => Some(c.layer_mut(i)?),
                None => None,
            };
            x = layer.forward(
                &x,
                &self.rope,
                positions,
                causal_mask.as_ref(),
                sliding_mask.as_ref(),
                layer_cache,
            )?;
        }

        self.norm.forward(&x)
    }

    /// Create a new KV cache for this model.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.num_hidden_layers)
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &GptOssConfig {
        &self.cfg
    }

    /// Memory report showing quantization savings.
    #[must_use]
    pub fn memory_report(&self) -> QuantizationReport {
        let mut quantized_bytes: usize = 0;
        let mut original_f32_bytes: usize = 0;
        let mut num_quantized: usize = 0;

        for layer in &self.layers {
            let gate_up_numel = layer.expert_gate_up.numel();
            let down_numel = layer.expert_down.numel();

            quantized_bytes += layer.expert_gate_up.size_bytes();
            quantized_bytes += layer.expert_down.size_bytes();

            original_f32_bytes += (gate_up_numel + down_numel) * 4;
            num_quantized += 2;
        }

        QuantizationReport {
            original_f32_bytes,
            original_bf16_bytes: original_f32_bytes / 2,
            quantized_bytes,
            num_quantized_tensors: num_quantized,
            num_full_precision_tensors: 0,
        }
    }
}
