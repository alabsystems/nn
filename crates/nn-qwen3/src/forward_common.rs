// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared forward-pass helpers for Qwen3 dense and MoE models.
//!
//! Eliminates duplication between [`Qwen3Model`](crate::Qwen3Model) and
//! [`Qwen3MoeModel`](crate::Qwen3MoeModel).

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::{KvCache, KvCacheLayer};
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Embedding, Linear, Module, NanCheckPolicy, RmsNorm,
    RotaryEmbedding,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};

use nn_core::layers::causal_mask_with_offset;

use crate::error::Qwen3Error;
use crate::Qwen3Config;

// -- DecoderLayer trait -------------------------------------------------------

/// Abstraction over dense and MoE decoder layers.
///
/// Both `Qwen3DecoderLayer` and `Qwen3MoeDecoderLayer` implement this trait,
/// enabling shared forward logic in [`forward_decoder_and_norm`].
pub(crate) trait DecoderLayer {
    fn forward(
        &self,
        x: &DynTensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        mask: Option<&DynTensor>,
        cache: Option<&mut KvCacheLayer>,
    ) -> Result<DynTensor>;
}

// -- Shared forward logic -----------------------------------------------------

/// Token IDs → embedded + unsqueezed tensor.
pub(crate) fn embed_and_unsqueeze(
    embed_tokens: &Embedding,
    input_ids: &[usize],
) -> Result<DynTensor> {
    let x = embed_tokens.forward_ids(input_ids)?;
    x.unsqueeze(0)
}

/// Shared decoder pass: validate cache → causal mask → decoder layers → norm.
///
/// Returns normed hidden states (after final RMSNorm, before lm_head).
pub(crate) fn forward_decoder_and_norm<L: DecoderLayer>(
    layers: &[L],
    norm: &RmsNorm,
    rope: &RotaryEmbedding,
    mut x: DynTensor,
    positions: &[usize],
    mut cache: Option<&mut KvCache>,
) -> Result<DynTensor> {
    validate_cache(cache.as_deref(), layers.len())?;
    let mask = build_causal_mask(positions.len(), cache.as_deref(), x.dtype(), &x.device())?;

    for (i, layer) in layers.iter().enumerate() {
        let layer_cache = match cache {
            Some(ref mut c) => Some(c.layer_mut(i)?),
            None => None,
        };
        x = layer.forward(&x, rope, positions, mask.as_ref(), layer_cache)?;
    }

    norm.forward(&x)
}

/// Shared forward: decoder + norm → lm_head → finiteness check.
///
/// Per-layer/per-attention/per-MLP `check_output_finite` calls inside
/// decoder layers are skipped via `NanCheckPolicy::Skip`, eliminating
/// N×3 GPU→CPU readback flushes per forward pass. The final logit
/// boundary check runs outside the Skip scope.
pub(crate) fn forward_to_logits<L: DecoderLayer>(
    layers: &[L],
    norm: &RmsNorm,
    lm_head: &Linear,
    rope: &RotaryEmbedding,
    x: DynTensor,
    positions: &[usize],
    cache: Option<&mut KvCache>,
    model_name: &str,
) -> Result<DynTensor> {
    let logits = with_nan_check_policy(NanCheckPolicy::Skip, || {
        let normed = forward_decoder_and_norm(layers, norm, rope, x, positions, cache)?;
        lm_head.forward(&normed)
    })?;
    check_output_finite(&logits, model_name)?;
    Ok(logits)
}

/// Shared forward returning both logits and normed hidden states.
///
/// Same Skip-scope pattern as [`forward_to_logits`]. Both outputs get
/// boundary checks outside the Skip scope.
pub(crate) fn forward_to_logits_and_hidden<L: DecoderLayer>(
    layers: &[L],
    norm: &RmsNorm,
    lm_head: &Linear,
    rope: &RotaryEmbedding,
    x: DynTensor,
    positions: &[usize],
    cache: Option<&mut KvCache>,
    model_name: &str,
) -> Result<(DynTensor, DynTensor)> {
    let (logits, normed) = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<_> {
        let normed = forward_decoder_and_norm(layers, norm, rope, x, positions, cache)?;
        let logits = lm_head.forward(&normed)?;
        Ok((logits, normed))
    })?;
    check_output_finite(&normed, &format!("{model_name}:normed_hidden"))?;
    check_output_finite(&logits, model_name)?;
    Ok((logits, normed))
}

/// Validate `forward_cached` inputs: input_ids and positions must have equal length.
pub(crate) fn validate_forward_input(input_ids: &[usize], positions: &[usize]) -> Result<()> {
    if input_ids.len() != positions.len() {
        return Err(Qwen3Error::InvalidInput {
            reason: format!(
                "input_ids len ({}) != positions len ({})",
                input_ids.len(),
                positions.len()
            ),
        }
        .into());
    }
    Ok(())
}

/// Validate pre-computed embedding inputs: rank, seq_len, hidden_size.
pub(crate) fn validate_embedding_input(
    hidden_states: &DynTensor,
    positions: &[usize],
    hidden_size: usize,
) -> Result<()> {
    let (_, seq_len, hs) = hidden_states.dims3()?;
    if seq_len != positions.len() {
        return Err(Qwen3Error::InvalidInput {
            reason: format!(
                "hidden_states seq_len ({seq_len}) != positions len ({})",
                positions.len()
            ),
        }
        .into());
    }
    if hs != hidden_size {
        return Err(Qwen3Error::InvalidInput {
            reason: format!(
                "hidden_states hidden_size ({hs}) != model hidden_size ({hidden_size})",
            ),
        }
        .into());
    }
    Ok(())
}

/// Validate KV cache layer count matches model layer count.
pub(crate) fn validate_cache(cache: Option<&KvCache>, num_layers: usize) -> Result<()> {
    if let Some(c) = cache {
        if c.num_layers() != num_layers {
            return Err(Qwen3Error::CacheMismatch {
                cache_layers: c.num_layers(),
                model_layers: num_layers,
            }
            .into());
        }
    }
    Ok(())
}

/// Build causal mask for the current decode step.
///
/// Returns `None` when `seq_len == 1` (autoregressive decoding): the single query
/// can attend to all prior positions, so the mask would be all-zeros. Skipping
/// allocation avoids O(S²) total allocation across S decode steps.
///
/// Uses the given `dtype` so the mask matches attention weight dtype
/// (e.g., BF16) for GPU binary ops (#1710).
pub(crate) fn build_causal_mask(
    seq_len: usize,
    cache: Option<&KvCache>,
    dtype: DType,
    device: &Device,
) -> Result<Option<DynTensor>> {
    let cached_len = cache.map_or(0, KvCache::seq_len);
    let total_seq = cached_len + seq_len;
    if seq_len > 1 && total_seq > 1 {
        Ok(Some(causal_mask_with_offset(
            seq_len, total_seq, dtype, device,
        )?))
    } else {
        Ok(None)
    }
}

/// Build RoPE from config + device.
pub(crate) fn build_rope(cfg: &Qwen3Config, vb: impl AsRef<VarBuilder>) -> Result<RotaryEmbedding> {
    let vb = vb.as_ref();
    match &cfg.rope_scaling {
        Some(yarn) => RotaryEmbedding::new_yarn(
            cfg.head_dim(),
            cfg.max_position_embeddings,
            cfg.rope_theta,
            yarn,
            vb.device(),
        ),
        None => RotaryEmbedding::new(
            cfg.head_dim(),
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        ),
    }
}
