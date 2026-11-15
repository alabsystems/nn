// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Scaled dot-product attention utilities and causal masking.
//!
//! Standalone functions shared by [`MultiHeadAttention`](super::MultiHeadAttention)
//! and [`JointAttention`](super::JointAttention):
//!
//! - [`sdpa`] — scaled dot-product attention: `softmax(Q @ K^T * scale) @ V`
//! - [`sdpa_causal`] — fused causal SDPA (uses Flash Attention on GPU, avoids mask allocation)
//! - [`repeat_kv`] — K/V head repetition for grouped-query attention (GQA/MQA)
//! - [`causal_mask`] / [`causal_mask_dtype`] / [`causal_mask_with_offset`] — upper-triangular masks

use crate::dyn_tensor::gpu::gpu_backend_dispatch;
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device, Result, TensorError};

/// Scaled dot-product attention: softmax(Q @ K^T * scale) @ V.
///
/// Computes the core SDPA operation shared by [`MultiHeadAttention`](super::MultiHeadAttention)
/// and [`JointAttention`](super::JointAttention):
///
/// 1. K transpose: `K^T = K.transpose(2, 3)`
/// 2. Score: `scores = Q @ K^T * scale`
/// 3. Mask: `scores += mask` (if provided, additive mask with `-inf` for masked positions)
/// 4. Softmax: `attn_weights = softmax(scores, dim=-1)`
/// 5. Output: `attn_out = attn_weights @ V`
///
/// # Arguments
///
/// - `q`: query tensor `[B, H, S_q, head_dim]`
/// - `k`: key tensor `[B, H, S_kv, head_dim]`
/// - `v`: value tensor `[B, H, S_kv, head_dim]`
/// - `mask`: optional additive mask `[*, *, S_q, S_kv]` (use [`causal_mask`] for autoregressive)
/// - `scale`: scaling factor, typically `1 / sqrt(head_dim)`
///
/// # Returns
///
/// Attention output `[B, H, S_q, head_dim]`.
pub fn sdpa(
    q: &DynTensor,
    k: &DynTensor,
    v: &DynTensor,
    mask: Option<&DynTensor>,
    scale: f64,
) -> Result<DynTensor> {
    if !scale.is_finite() {
        return Err(TensorError::ValueOutOfRange {
            description: "sdpa: scale must be finite",
        });
    }

    let tracing = trace::is_tracing();

    // Try fused GPU path (Flash Attention) before decomposed path (#2434).
    // Returns None when conditions aren't met (non-F32, mask, non-4D, etc.),
    // falling through to the decomposed matmul+softmax+matmul path below.
    let mut result = if let Some(fused) = gpu_backend_dispatch(|b| b.sdpa(q, k, v, mask, scale)) {
        fused?
    } else {
        let compute = || -> Result<DynTensor> {
            let k_t = k.transpose(2, 3)?;
            let scores = q.matmul(&k_t)?.mul_scalar(scale)?;
            let scores = match mask {
                Some(m) => scores.broadcast_add(m)?,
                None => scores,
            };
            let attn_weights = scores.softmax(scores.rank() - 1)?;
            attn_weights.matmul(v)
        };

        if tracing {
            trace::with_trace_suppressed(compute)?
        } else {
            compute()?
        }
    };

    if tracing {
        let mut inputs = vec![q, k, v];
        if let Some(m) = mask {
            inputs.push(m);
        }
        let input_ids = DynTensor::trace_input_ids(&inputs)?;
        if let Some(id) = trace::record_op(
            TraceOp::Sdpa { scale },
            &input_ids,
            result.dims(),
            result.dtype(),
        ) {
            result.set_trace_id(id);
        }
    }
    Ok(result)
}

/// Scaled dot-product attention with causal masking.
///
/// Equivalent to `sdpa(q, k, v, Some(&causal_mask(S)), scale)` but uses fused
/// causal masking on GPU (Flash Attention with block-level tile skipping) to
/// avoid allocating the O(S^2) mask tensor and save ~50% compute.
///
/// Falls back to generating an explicit causal mask and using the decomposed
/// path when the fused GPU kernel is unavailable.
///
/// # Arguments
///
/// - `q`: query tensor `[B, H_q, S, head_dim]`
/// - `k`: key tensor `[B, H_kv, S, head_dim]` (S_q must equal S_kv)
/// - `v`: value tensor `[B, H_kv, S, head_dim]`
/// - `scale`: scaling factor, typically `1 / sqrt(head_dim)`
///
/// # Returns
///
/// Attention output `[B, H_q, S, head_dim]`.
pub fn sdpa_causal(q: &DynTensor, k: &DynTensor, v: &DynTensor, scale: f64) -> Result<DynTensor> {
    if !scale.is_finite() {
        return Err(TensorError::ValueOutOfRange {
            description: "sdpa_causal: scale must be finite",
        });
    }

    let tracing = trace::is_tracing();

    // Try fused GPU path with built-in causal masking (#2434).
    let mut result = if let Some(fused) = gpu_backend_dispatch(|b| b.sdpa_causal(q, k, v, scale)) {
        fused?
    } else {
        // Fallback: generate explicit causal mask, use decomposed path.
        let compute = || -> Result<DynTensor> {
            let s_q = if q.rank() >= 3 {
                q.dims()[q.rank() - 2]
            } else {
                return Err(TensorError::InvalidShape(format!(
                    "sdpa_causal: expected at least 3D tensor, got rank {}",
                    q.rank()
                )));
            };
            let mask = causal_mask(s_q, &q.device())?;
            sdpa(q, k, v, Some(&mask), scale)
        };

        if tracing {
            trace::with_trace_suppressed(compute)?
        } else {
            compute()?
        }
    };

    if tracing {
        let inputs = [q, k, v];
        let input_ids = DynTensor::trace_input_ids(&inputs)?;
        if let Some(id) = trace::record_op(
            TraceOp::SdpaCausal { scale },
            &input_ids,
            result.dims(),
            result.dtype(),
        ) {
            result.set_trace_id(id);
        }
    }
    Ok(result)
}

/// Repeat K/V heads for grouped-query attention.
///
/// `[B, num_kv_heads, S, head_dim]` -> `[B, num_heads, S, head_dim]`
/// where `num_rep = num_heads / num_kv_heads`.
///
/// When `num_rep == 1` (standard MHA), returns a clone with no extra work.
pub fn repeat_kv(x: &DynTensor, num_rep: usize) -> Result<DynTensor> {
    if num_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand([b, h, num_rep, s, d])?
        .reshape([b, h * num_rep, s, d])
}

/// Generate upper-triangular causal mask for autoregressive attention.
///
/// Returns `[1, 1, seq_len, seq_len]` with `0.0` on and below the diagonal
/// and `-inf` above the diagonal. Compatible with the `mask` parameter of
/// [`MultiHeadAttention::forward`](super::MultiHeadAttention::forward).
pub fn causal_mask(seq_len: usize, device: &Device) -> Result<DynTensor> {
    causal_mask_dtype(seq_len, DType::F32, device)
}

/// Generate a lower-triangular causal mask with a specific dtype.
///
/// Same as [`causal_mask`] but converts the mask to the given `dtype`
/// so it matches attention weight dtype in GPU binary ops (#1710).
pub fn causal_mask_dtype(seq_len: usize, dtype: DType, device: &Device) -> Result<DynTensor> {
    causal_mask_with_offset(seq_len, seq_len, dtype, device)
}

/// Generate a causal attention mask for cached decoding.
///
/// `new_tokens`: number of new query tokens (rows).
/// `total_tokens`: total sequence length including cached tokens (columns).
///
/// Returns `[1, 1, new_tokens, total_tokens]` where position (i, j) is `0.0`
/// if query token i (at absolute position `total_tokens - new_tokens + i`)
/// can attend to key position j, and `-inf` otherwise.
///
/// When `new_tokens == total_tokens`, this produces the standard square causal
/// mask identical to [`causal_mask_dtype`].
///
/// Uses the given `dtype` so the mask matches attention weight dtype
/// (e.g., BF16) for GPU binary ops (#1710).
pub fn causal_mask_with_offset(
    new_tokens: usize,
    total_tokens: usize,
    dtype: DType,
    device: &Device,
) -> Result<DynTensor> {
    if new_tokens == 0 || total_tokens == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "causal_mask: new_tokens and total_tokens must be > 0",
        });
    }
    let offset = total_tokens
        .checked_sub(new_tokens)
        .ok_or(TensorError::ValueOutOfRange {
            description: "causal_mask: total_tokens < new_tokens",
        })?;
    let total = new_tokens
        .checked_mul(total_tokens)
        .ok_or(TensorError::DimensionOverflow {
            dims: vec![new_tokens, total_tokens],
        })?;
    let mut data = vec![0.0f32; total];
    for row in 0..new_tokens {
        let abs_pos = offset + row;
        for col in (abs_pos + 1)..total_tokens {
            data[row * total_tokens + col] = f32::NEG_INFINITY;
        }
    }
    let t = DynTensor::from_vec(data, &[1, 1, new_tokens, total_tokens], &Device::Cpu)?;
    let t = if dtype != DType::F32 {
        t.to_dtype(dtype)?
    } else {
        t
    };
    t.to_device(device)
}
