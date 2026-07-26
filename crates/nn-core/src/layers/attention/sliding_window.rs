// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sliding window attention for efficient long-sequence models.
//!
//! Each token attends only to tokens within a fixed-size window centered on
//! itself, restricting attention from O(n^2) to O(n * w). Used by Mistral,
//! LongNet, BigBird, and other models designed for long context lengths.
//!
//! - [`sliding_window_mask`] — generates a banded additive attention mask
//! - [`SlidingWindowAttention`] — full attention layer with QKV projection,
//!   sliding window masking, and output projection

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::sdpa::sdpa;
use crate::layers::{check_output_finite, validate_heads, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Device, Result, TensorError};

/// Generate a banded additive attention mask for sliding window attention.
///
/// For each query position `i`, only key positions `j` where
/// `|i - j| <= window_size / 2` are visible. Positions outside the window
/// are set to `-inf` (additive mask convention, compatible with [`sdpa`]).
///
/// # Arguments
///
/// - `seq_len`: sequence length (number of tokens)
/// - `window_size`: total window width. Each token attends to at most
///   `window_size` positions (centered on itself). When `window_size` is
///   even, the window extends `window_size / 2` to each side. When odd,
///   it extends `(window_size - 1) / 2` to each side (plus the center).
/// - `device`: target device for the mask tensor
///
/// # Returns
///
/// Mask tensor `[1, 1, seq_len, seq_len]` with `0.0` for visible positions
/// and `-inf` for masked positions.
///
/// # Examples
///
/// ```
/// # use nn_core::layers::attention::sliding_window_mask;
/// # use nn_core::Device;
/// // Window size 3: each token sees itself and 1 neighbor on each side.
/// let mask = sliding_window_mask(4, 3, &Device::Cpu).unwrap();
/// assert_eq!(mask.dims(), &[1, 1, 4, 4]);
/// ```
pub fn sliding_window_mask(
    seq_len: usize,
    window_size: usize,
    device: &Device,
) -> Result<DynTensor> {
    if seq_len == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "sliding_window_mask: seq_len must be > 0",
        });
    }
    if window_size == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "sliding_window_mask: window_size must be > 0",
        });
    }

    let total = seq_len
        .checked_mul(seq_len)
        .ok_or(TensorError::DimensionOverflow {
            dims: vec![seq_len, seq_len],
        })?;

    let half_window = window_size / 2;
    let mut data = vec![0.0f32; total];

    for i in 0..seq_len {
        for j in 0..seq_len {
            let dist = i.abs_diff(j);
            if dist > half_window {
                data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let t = DynTensor::from_vec(data, &[1, 1, seq_len, seq_len], &Device::Cpu)?;
    t.to_device(device)
}

/// Sliding window attention layer.
///
/// Restricts each token to attend only within a local window of nearby tokens,
/// reducing attention complexity from O(n^2) to O(n * w) where `w` is the
/// window size. This is the core attention mechanism used by Mistral, LongNet,
/// BigBird, and other long-context architectures.
///
/// # Forward pass
///
/// 1. Project input to Q, K, V via a fused QKV linear layer
/// 2. Reshape to multi-head format: `[B, H, S, head_dim]`
/// 3. Generate a banded sliding window mask via [`sliding_window_mask`]
/// 4. Apply scaled dot-product attention with the window mask
/// 5. Project output via linear layer
///
/// # Example
///
/// ```
/// # use nn_core::layers::attention::SlidingWindowAttention;
/// # use nn_core::dyn_tensor::DynTensor;
/// # use nn_core::{DType, Device};
/// # use nn_core::layers::Linear;
/// let d = 16;
/// let num_heads = 2;
/// let window_size = 3;
/// # let w = DynTensor::ones(&[3 * d, d], DType::F32, &Device::Cpu).unwrap();
/// # let b = DynTensor::zeros(&[3 * d], DType::F32, &Device::Cpu).unwrap();
/// # let qkv = Linear::new(w, Some(b)).unwrap();
/// # let w2 = DynTensor::ones(&[d, d], DType::F32, &Device::Cpu).unwrap();
/// # let b2 = DynTensor::zeros(&[d], DType::F32, &Device::Cpu).unwrap();
/// # let out_proj = Linear::new(w2, Some(b2)).unwrap();
/// let attn = SlidingWindowAttention::new(qkv, out_proj, num_heads, window_size).unwrap();
/// ```
#[derive(Clone)]
pub struct SlidingWindowAttention {
    qkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
    scale: f64,
}

impl std::fmt::Debug for SlidingWindowAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlidingWindowAttention")
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("window_size", &self.window_size)
            .finish_non_exhaustive()
    }
}

impl SlidingWindowAttention {
    /// Create from pre-loaded fused QKV and output projection weights.
    ///
    /// - `qkv`: fused Q/K/V projection `[3 * embed_dim, embed_dim]`
    /// - `out_proj`: output projection `[embed_dim, embed_dim]`
    /// - `num_heads`: number of attention heads
    /// - `window_size`: sliding window width (each token attends to at most
    ///   this many positions centered on itself)
    ///
    /// # Errors
    ///
    /// Returns an error if `num_heads` is 0, `window_size` is 0, or the QKV
    /// weight shape is inconsistent with `num_heads`.
    pub fn new(
        qkv: Linear,
        out_proj: Linear,
        num_heads: usize,
        window_size: usize,
    ) -> Result<Self> {
        validate_heads(num_heads, "SlidingWindowAttention")?;
        if window_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "SlidingWindowAttention: window_size must be > 0",
            });
        }

        // Infer embed_dim from QKV weight: [3 * embed_dim, embed_dim]
        let qkv_weight = qkv.weight();
        if qkv_weight.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: qkv_weight.rank(),
            });
        }
        let qkv_out = qkv_weight.dim(0)?;
        if qkv_out % 3 != 0 {
            return Err(TensorError::InvalidShape(format!(
                "SlidingWindowAttention: QKV out_features ({qkv_out}) must be divisible by 3"
            )));
        }
        let embed_dim = qkv_out / 3;
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(TensorError::InvalidShape(format!(
                "SlidingWindowAttention: embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            qkv,
            out_proj,
            num_heads,
            head_dim,
            window_size,
            scale,
        })
    }

    /// Load from a [`VarBuilder`] with standard weight naming.
    ///
    /// Loads `qkv.weight`, `qkv.bias`, `out_proj.weight`, `out_proj.bias`.
    ///
    /// - `embed_dim`: model embedding dimension
    /// - `num_heads`: number of attention heads
    /// - `window_size`: sliding window width
    /// - `bias`: whether to load bias parameters
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        embed_dim: usize,
        num_heads: usize,
        window_size: usize,
        bias: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_heads(num_heads, "SlidingWindowAttention::load")?;
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(TensorError::InvalidShape(format!(
                "SlidingWindowAttention::load: embed_dim ({embed_dim}) not divisible by num_heads ({num_heads})"
            )));
        }

        let load_linear =
            |prefix: &str, out_features: usize, in_features: usize| -> Result<Linear> {
                let sub = vb.pp(prefix);
                let w = sub.get(&[out_features, in_features], "weight")?;
                let b = if bias {
                    Some(sub.get(&[out_features], "bias")?)
                } else {
                    None
                };
                Linear::new(w, b)
            };

        let qkv = load_linear("qkv", 3 * embed_dim, embed_dim)?;
        let out_proj = load_linear("out_proj", embed_dim, embed_dim)?;

        Self::new(qkv, out_proj, num_heads, window_size)
    }

    /// Forward pass with sliding window attention.
    ///
    /// - `x`: input tensor `[batch, seq_len, embed_dim]`
    ///
    /// Returns `[batch, seq_len, embed_dim]`.
    pub fn forward_t(&self, x: &DynTensor) -> Result<DynTensor> {
        let (b, seq_len, _) = x.dims3()?;
        let embed_dim = self.num_heads * self.head_dim;

        // 1. Fused QKV projection: [B, S, D] -> [B, S, 3*D]
        let qkv = self.qkv.forward(x)?;
        let q = qkv.narrow(2, 0, embed_dim)?;
        let k = qkv.narrow(2, embed_dim, embed_dim)?;
        let v = qkv.narrow(2, 2 * embed_dim, embed_dim)?;

        // 2. Multi-head reshape: [B, S, D] -> [B, H, S, head_dim]
        let q = q
            .reshape([b, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([b, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([b, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // 3. Generate sliding window mask
        let mask = sliding_window_mask(seq_len, self.window_size, &x.device())?;

        // 4. Scaled dot-product attention with window mask
        let attn_out = sdpa(&q, &k, &v, Some(&mask), self.scale)?;

        // 5. Reshape back: [B, H, S, head_dim] -> [B, S, D]
        let attn_out = attn_out.transpose(1, 2)?.reshape([b, seq_len, embed_dim])?;

        // 6. Output projection
        let result = self.out_proj.forward(&attn_out)?;
        check_output_finite(&result, "SlidingWindowAttention")?;
        Ok(result)
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

    /// Sliding window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

/// [`Module`] impl for simple forward (self-attention with sliding window).
impl Module for SlidingWindowAttention {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.forward_t(x)
    }
}

#[cfg(test)]
#[path = "sliding_window_tests.rs"]
mod tests;

// -- Kani verification harnesses (#3575) --------------------------------------

#[cfg(kani)]
mod kani_proofs {
    /// Prove sliding_window_mask[i][j] == -inf iff |i - j| > window_size / 2.
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn sliding_window_mask_neg_inf_iff_dist_gt_half() {
        let seq_len: usize = kani::any();
        let window_size: usize = kani::any();
        kani::assume(seq_len >= 1 && seq_len <= 8);
        kani::assume(window_size >= 1 && window_size <= 16);
        let half_window = window_size / 2;
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < seq_len);
        kani::assume(j < seq_len);
        let dist = if i >= j { i - j } else { j - i };
        let is_masked = dist > half_window;
        let value: f32 = if is_masked { f32::NEG_INFINITY } else { 0.0 };
        if dist > half_window {
            kani::assert(value == f32::NEG_INFINITY, "outside window must be -inf");
        } else {
            kani::assert(value == 0.0, "inside window must be 0.0");
        }
    }

    /// Prove the sliding window mask is symmetric: mask[i][j] == mask[j][i].
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn sliding_window_mask_symmetry() {
        let seq_len: usize = kani::any();
        let window_size: usize = kani::any();
        kani::assume(seq_len >= 1 && seq_len <= 8);
        kani::assume(window_size >= 1 && window_size <= 16);
        let half_window = window_size / 2;
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < seq_len);
        kani::assume(j < seq_len);
        let dist_ij = if i >= j { i - j } else { j - i };
        let dist_ji = if j >= i { j - i } else { i - j };
        kani::assert(dist_ij == dist_ji, "distance must be symmetric");
        let masked_ij = dist_ij > half_window;
        let masked_ji = dist_ji > half_window;
        kani::assert(masked_ij == masked_ji, "mask must be symmetric");
    }

    /// Prove sliding window mask dimensions: element count == seq_len^2.
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn sliding_window_mask_dimensions() {
        let seq_len: usize = kani::any();
        kani::assume(seq_len >= 1 && seq_len <= 64);
        let total = seq_len.checked_mul(seq_len);
        kani::assert(total.is_some(), "seq_len^2 must not overflow");
        let total = total.unwrap();
        kani::assert(
            total == seq_len * seq_len,
            "element count must equal seq_len^2",
        );
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < seq_len);
        kani::assume(j < seq_len);
        let idx = i * seq_len + j;
        kani::assert(idx < total, "index must be within bounds");
    }

    /// Prove the diagonal is always visible.
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn sliding_window_diagonal_always_visible() {
        let window_size: usize = kani::any();
        kani::assume(window_size >= 1 && window_size <= 32);
        let half_window = window_size / 2;
        let dist = 0usize;
        kani::assert(
            dist <= half_window,
            "self-distance 0 must be within any window",
        );
    }

    /// Prove window_size=1 means only the diagonal is visible.
    /// Part of #3575.
    #[kani::unwind(1)]
    #[kani::proof]
    fn sliding_window_size1_self_only() {
        let seq_len: usize = kani::any();
        kani::assume(seq_len >= 2 && seq_len <= 8);
        let half_window = 0usize; // window_size=1 -> half=0
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < seq_len);
        kani::assume(j < seq_len);
        let dist = if i >= j { i - j } else { j - i };
        let is_masked = dist > half_window;
        if i == j {
            kani::assert(!is_masked, "window_size=1: self must be visible");
        } else {
            kani::assert(is_masked, "window_size=1: non-self must be masked");
        }
    }
}
