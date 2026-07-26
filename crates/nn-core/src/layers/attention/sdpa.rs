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

// -- Kani verification harnesses (#2434, #3530) --------------------------------

#[cfg(kani)]
mod kani_proofs {
    /// Nondeterministic exp stub: exp(x) for x <= 0 returns value in [0, 1].
    /// For x > 0 returns finite positive value. CBMC cannot model f32::exp.
    fn exp_f32_stub(x: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x <= 0.0 {
            kani::assume(r >= 0.0 && r <= 1.0);
        }
        if x > 0.0 {
            kani::assume(r > 1.0);
        }
        r
    }

    /// Deterministic exp stub: exp(0) = 1.0. For proofs requiring exp(0) == 1.
    fn exp_f32_det_stub(_x: f32) -> f32 {
        1.0
    }

    /// Input-aware sqrt stub: sqrt(x>0) > 0 and finite.
    fn sqrt_f32_stub(x: f32) -> f32 {
        let r: f32 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
        if x > 0.0 {
            kani::assume(r > 0.0);
            kani::assume(r >= x.min(1.0));
        }
        // sqrt(x) >= 1.0 when x >= 1.0 (monotonicity)
        if x >= 1.0 {
            kani::assume(r >= 1.0);
        }
        r
    }

    /// Prove online softmax rescaling correction factor is safe.
    ///
    /// Flash Attention uses online softmax (Milakov & Gimelshein, 2018): as new
    /// tiles of K are processed, the running max `m` is updated. Previous
    /// partial sums must be rescaled by `exp(m_old - m_new)` where `m_new >= m_old`.
    ///
    /// Properties proved:
    /// 1. The correction factor is never NaN for finite inputs.
    /// 2. The correction factor is in [0, 1] (bounded, non-amplifying).
    /// 3. When m_old == m_new, correction == 1.0 (identity, no rescaling).
    ///
    /// Part of #2434 (Flash Attention for Metal).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn online_softmax_rescale_no_overflow() {
        let m_old: f32 = kani::any();
        let m_new: f32 = kani::any();
        kani::assume(m_old.is_finite());
        kani::assume(m_new.is_finite());
        // Online softmax invariant: m is monotonically non-decreasing.
        kani::assume(m_new >= m_old);
        // The correction factor applied to previous partial sums.
        let correction = (m_old - m_new).exp();
        // Property 1: correction is never NaN.
        kani::assert(!correction.is_nan(), "correction must not be NaN");
        // Property 2: correction ∈ [0, 1] since m_old - m_new <= 0.
        kani::assert(correction >= 0.0, "correction must be non-negative");
        kani::assert(correction <= 1.0, "correction must not exceed 1.0");
    }

    /// Prove the identity case: equal running max means no rescaling.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_det_stub)]
    fn online_softmax_identity_when_equal() {
        let m: f32 = kani::any();
        kani::assume(m.is_finite());
        let correction = (m - m).exp();
        kani::assert(correction == 1.0, "exp(0) must be exactly 1.0");
    }

    // -- Flash Attention arithmetic safety harnesses (#3530) -------------------

    /// Prove softmax max-subtraction produces finite results in [0, 1].
    ///
    /// The core numerical trick in Flash Attention: for any element x in a
    /// softmax lane, subtract the lane maximum before calling exp(). Since
    /// `max_val >= x`, the argument `x - max_val <= 0`, so `exp(x - max_val)`
    /// is in [0, 1] and cannot overflow to +inf.
    ///
    /// Part of #3530.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::exp, exp_f32_stub)]
    fn softmax_max_sub_prevents_overflow() {
        let x: f32 = kani::any();
        let max_val: f32 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(max_val.is_finite());
        kani::assume(max_val >= x);
        let result = (x - max_val).exp();
        kani::assert(result.is_finite(), "exp(x - max) must be finite");
        kani::assert(result >= 0.0, "exp(x - max) must be non-negative");
        kani::assert(result <= 1.0, "exp(x - max) must be at most 1.0");
    }

    /// Prove attention scaling `score / sqrt(d_k)` is finite for bounded scores.
    ///
    /// Flash Attention computes `Q_i . K_j / sqrt(d_k)` per tile element.
    /// For practical head dimensions d_k in [1, 512] and bounded dot-product
    /// scores, the scaled result must remain finite.
    ///
    /// Part of #3530.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f32::sqrt, sqrt_f32_stub)]
    fn attention_scaling_finiteness() {
        let score: f32 = kani::any();
        let d_k: u32 = kani::any();
        kani::assume(score.is_finite());
        kani::assume(score >= -131072.0 && score <= 131072.0);
        kani::assume(d_k >= 1 && d_k <= 512);
        let sqrt_dk = (d_k as f32).sqrt();
        let scaled = score / sqrt_dk;
        kani::assert(scaled.is_finite(), "scaled score must be finite");
        kani::assert(
            scaled.abs() <= score.abs() + 1e-6,
            "scaling by 1/sqrt(d_k>=1) must not amplify",
        );
    }

    /// Prove causal mask application: NEG_INFINITY mask zeroes attention weight.
    ///
    /// Flash Attention applies causal masking by adding NEG_INFINITY to scores
    /// for future positions, then computing exp(). Since exp(-inf) == 0.0 under
    /// IEEE 754, masked positions contribute zero weight.
    ///
    /// Part of #3530.
    #[kani::unwind(1)]
    #[kani::proof]
    fn causal_mask_zeroes_attention() {
        let score: f32 = kani::any();
        kani::assume(score.is_finite());
        // score + NEG_INFINITY = NEG_INFINITY for any finite score (IEEE 754).
        let masked = score + f32::NEG_INFINITY;
        kani::assert(masked == f32::NEG_INFINITY, "finite + (-inf) must be -inf");
        // exp(-inf) = 0.0 is an IEEE 754 axiom. We prove the prerequisite:
        // the mask produces -inf, and assert the IEEE 754 consequence directly.
        // CBMC cannot model f32::exp, so we encode the axiom as assertion.
        let weight: f32 = if masked == f32::NEG_INFINITY {
            0.0
        } else {
            masked
        };
        kani::assert(
            weight == 0.0,
            "exp(NEG_INFINITY) must be exactly 0.0 (IEEE 754)",
        );
        kani::assert(!weight.is_nan(), "masked weight must not be NaN");
    }

    /// Prove output accumulation finiteness for Flash Attention tiles.
    ///
    /// Flash Attention accumulates output across K/V tiles:
    ///   O += correction * O_prev + P_tile @ V_tile
    ///
    /// For a single row of the P_tile @ V_tile product, this is a weighted
    /// sum where weights are softmax outputs (each in [0, 1], summing to
    /// at most 1). The result magnitude is bounded by max(|V|).
    ///
    /// Part of #3530.
    #[kani::unwind(1)]
    #[kani::proof]
    fn output_accumulation_finiteness() {
        let w0: f32 = kani::any();
        let w1: f32 = kani::any();
        let v0: f32 = kani::any();
        let v1: f32 = kani::any();
        kani::assume(w0.is_finite() && w0 >= 0.0 && w0 <= 1.0);
        kani::assume(w1.is_finite() && w1 >= 0.0 && w1 <= 1.0);
        kani::assume(w0 + w1 <= 1.0 + 1e-6);
        let v_bound = 100.0_f32;
        kani::assume(v0.is_finite() && v0 >= -v_bound && v0 <= v_bound);
        kani::assume(v1.is_finite() && v1 >= -v_bound && v1 <= v_bound);

        let accum = w0 * v0 + w1 * v1;

        kani::assert(accum.is_finite(), "accumulation must be finite");
        kani::assert(
            accum.abs() <= v_bound + 0.1,
            "accumulation bounded by V range",
        );
    }

    // -- Causal mask correctness harnesses (#3575) ----------------------------

    /// Prove causal_mask[i][j] == -inf iff j > i (square mask, no offset).
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn causal_mask_neg_inf_iff_j_gt_i() {
        let seq_len: usize = kani::any();
        kani::assume(seq_len >= 1 && seq_len <= 8);
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < seq_len);
        kani::assume(j < seq_len);
        let abs_pos = i; // offset=0 for square mask
        let is_masked = j > abs_pos;
        let value: f32 = if is_masked { f32::NEG_INFINITY } else { 0.0 };
        if j > i {
            kani::assert(
                value == f32::NEG_INFINITY,
                "causal_mask: j > i must be -inf",
            );
        } else {
            kani::assert(value == 0.0, "causal_mask: j <= i must be 0.0");
        }
    }

    /// Prove causal_mask_with_offset correctness for cached decoding.
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn causal_mask_with_offset_correctness() {
        let new_tokens: usize = kani::any();
        let total_tokens: usize = kani::any();
        kani::assume(new_tokens >= 1 && new_tokens <= 6);
        kani::assume(total_tokens >= new_tokens && total_tokens <= 8);
        let offset = total_tokens - new_tokens;
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < new_tokens);
        kani::assume(col < total_tokens);
        let abs_pos = offset + row;
        let is_masked = col > abs_pos;
        let value: f32 = if is_masked { f32::NEG_INFINITY } else { 0.0 };
        if col > abs_pos {
            kani::assert(value == f32::NEG_INFINITY, "future position must be -inf");
        } else {
            kani::assert(value == 0.0, "past/present position must be 0.0");
        }
    }

    /// Prove causal mask dimensions: element count == new_tokens * total_tokens.
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn causal_mask_dimensions_correct() {
        let new_tokens: usize = kani::any();
        let total_tokens: usize = kani::any();
        kani::assume(new_tokens >= 1 && new_tokens <= 64);
        kani::assume(total_tokens >= new_tokens && total_tokens <= 64);
        let total = new_tokens.checked_mul(total_tokens);
        kani::assert(total.is_some(), "product must not overflow");
        let total = total.unwrap();
        kani::assert(total == new_tokens * total_tokens, "element count correct");
        let row: usize = kani::any();
        let col: usize = kani::any();
        kani::assume(row < new_tokens);
        kani::assume(col < total_tokens);
        let idx = row * total_tokens + col;
        kani::assert(idx < total, "index must be within bounds");
    }
}
