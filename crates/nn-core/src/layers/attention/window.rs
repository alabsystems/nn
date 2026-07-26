// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Window attention for Vision Transformers.
//!
//! [`window_partition`] and [`window_unpartition`] reshape a sequence of spatial
//! tokens into non-overlapping windows for local self-attention. Combined with
//! a standard attention mechanism, this enables the alternating window/global
//! attention pattern used by Qwen2.5-VL (32-layer ViT: odd layers use window
//! attention, even layers use global attention).

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::sdpa::sdpa;
use crate::layers::{check_output_finite, validate_heads, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Partition a spatial token sequence into non-overlapping windows.
///
/// - `x`: `[B, H*W, D]` — batch of spatial token sequences
/// - `height`, `width`: spatial grid dimensions (`H * W == x.dim(1)`)
/// - `window_size`: window edge length in tokens
///
/// Returns `[B * num_windows, window_size * window_size, D]`.
///
/// If `height` or `width` is not divisible by `window_size`, the input is
/// padded with zeros on the bottom and right. The caller must use
/// [`window_unpartition`] with the original `(height, width)` to unpad.
pub fn window_partition(
    x: &DynTensor,
    height: usize,
    width: usize,
    window_size: usize,
) -> Result<(DynTensor, usize, usize)> {
    if window_size == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "window_partition: window_size must be > 0",
        });
    }
    let (b, seq_len, d) = x.dims3()?;
    if seq_len != height * width {
        return Err(TensorError::shape_mismatch(
            vec![b, height * width, d],
            vec![b, seq_len, d],
        ));
    }

    // Pad if needed.
    let pad_h = (window_size - height % window_size) % window_size;
    let pad_w = (window_size - width % window_size) % window_size;
    let h_padded = height + pad_h;
    let w_padded = width + pad_w;

    let x = if pad_h > 0 || pad_w > 0 {
        // Reshape to spatial: [B, H, W, D]
        let spatial = x.reshape([b, height, width, d])?;
        // Pad bottom and right with zeros.
        let mut padded_data = vec![0.0f32; b * h_padded * w_padded * d];
        let src = spatial.to_flat_vec::<f32>()?;
        for bi in 0..b {
            for hi in 0..height {
                for wi in 0..width {
                    let src_idx = ((bi * height + hi) * width + wi) * d;
                    let dst_idx = ((bi * h_padded + hi) * w_padded + wi) * d;
                    padded_data[dst_idx..dst_idx + d].copy_from_slice(&src[src_idx..src_idx + d]);
                }
            }
        }
        DynTensor::new(&padded_data, &[b, h_padded, w_padded, d], &x.device())?.reshape([
            b,
            h_padded * w_padded,
            d,
        ])?
    } else {
        x.clone()
    };

    let nw_h = h_padded / window_size;
    let nw_w = w_padded / window_size;
    let ws = window_size;

    // Reshape: [B, H', W', D] -> [B, nw_h, ws, nw_w, ws, D]
    let x = x.reshape([b, h_padded, w_padded, d])?;
    let x = x.reshape([b, nw_h, ws, nw_w, ws, d])?;
    // Permute: [B, nw_h, nw_w, ws, ws, D]
    let x = x.permute([0, 1, 3, 2, 4, 5])?;
    // Flatten: [B * nw_h * nw_w, ws * ws, D]
    let num_windows = nw_h * nw_w;
    let x = x.reshape([b * num_windows, ws * ws, d])?;

    Ok((x, h_padded, w_padded))
}

/// Reverse window partition: merge windows back into a spatial sequence.
///
/// - `windows`: `[B * num_windows, window_size * window_size, D]`
/// - `height`, `width`: ORIGINAL (unpadded) spatial dimensions
/// - `padded_height`, `padded_width`: padded dimensions from `window_partition`
/// - `window_size`: window edge length
/// - `batch_size`: original batch size B
///
/// Returns `[B, H*W, D]`.
pub fn window_unpartition(
    windows: &DynTensor,
    height: usize,
    width: usize,
    padded_height: usize,
    padded_width: usize,
    window_size: usize,
    batch_size: usize,
) -> Result<DynTensor> {
    if window_size == 0 {
        return Err(TensorError::ValueOutOfRange {
            description: "window_unpartition: window_size must be > 0",
        });
    }

    let (bw, ws2, d) = windows.dims3()?;
    let ws = window_size;
    if ws2 != ws * ws {
        return Err(TensorError::shape_mismatch(
            vec![bw, ws * ws, d],
            vec![bw, ws2, d],
        ));
    }

    let nw_h = padded_height / ws;
    let nw_w = padded_width / ws;

    // Reshape: [B, nw_h, nw_w, ws, ws, D]
    let x = windows.reshape([batch_size, nw_h, nw_w, ws, ws, d])?;
    // Permute: [B, nw_h, ws, nw_w, ws, D]
    let x = x.permute([0, 1, 3, 2, 4, 5])?;
    // Flatten spatial: [B, H', W', D]
    let x = x.reshape([batch_size, padded_height, padded_width, d])?;

    // Remove padding if needed.
    let x = if padded_height != height || padded_width != width {
        x.narrow(1, 0, height)?.narrow(2, 0, width)?
    } else {
        x
    };

    // Flatten to [B, H*W, D]
    x.reshape([batch_size, height * width, d])
}

/// Configuration for window-based multi-head attention.
///
/// Used by Qwen2.5-VL ViT where odd layers use local window attention
/// (each token attends only within its spatial window) and even layers
/// use full global attention.
#[derive(Clone, Debug)]
pub struct WindowAttentionConfig {
    /// Spatial window edge length in tokens (e.g., 14 for Qwen2.5-VL).
    pub window_size: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Dimension per attention head (`hidden_size / num_heads`).
    pub head_dim: usize,
}

impl WindowAttentionConfig {
    /// Create and validate a window attention config.
    pub fn new(window_size: usize, num_heads: usize, head_dim: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "WindowAttentionConfig: window_size must be > 0",
            });
        }
        validate_heads(num_heads, "WindowAttentionConfig")?;
        if head_dim == 0 {
            return Err(TensorError::InvalidShape(
                "WindowAttentionConfig: head_dim must be > 0".into(),
            ));
        }
        Ok(Self {
            window_size,
            num_heads,
            head_dim,
        })
    }

    /// The hidden dimension: `num_heads * head_dim`.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Multi-head attention with window partitioning for local attention.
///
/// Wraps a fused QKV projection + SDPA with [`window_partition`] /
/// [`window_unpartition`] so each token only attends within its spatial window.
/// The caller supplies `(height, width)` at forward time so the grid dimensions
/// are dynamic (supporting variable-resolution images).
///
/// Used by Qwen2.5-VL ViT on alternating (odd) layers.
#[derive(Clone)]
pub struct WindowMultiHeadAttention {
    qkv: Linear,
    out_proj: Linear,
    config: WindowAttentionConfig,
    scale: f64,
}

impl std::fmt::Debug for WindowMultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowMultiHeadAttention")
            .field("window_size", &self.config.window_size)
            .field("num_heads", &self.config.num_heads)
            .field("head_dim", &self.config.head_dim)
            .finish_non_exhaustive()
    }
}

impl WindowMultiHeadAttention {
    /// Create from pre-loaded fused QKV and output projection weights.
    ///
    /// - `qkv`: fused Q/K/V projection `[3 * hidden_size, hidden_size]`
    /// - `out_proj`: output projection `[hidden_size, hidden_size]`
    /// - `config`: window attention parameters
    pub fn new(qkv: Linear, out_proj: Linear, config: WindowAttentionConfig) -> Result<Self> {
        let scale = 1.0 / (config.head_dim as f64).sqrt();
        Ok(Self {
            qkv,
            out_proj,
            config,
            scale,
        })
    }

    /// Load from a [`VarBuilder`] with separate Q/K/V weight naming.
    ///
    /// Loads `query.{weight,bias}`, `key.{weight,bias}`, `value.{weight,bias}`
    /// and fuses them into a single QKV projection. Also loads `proj.{weight,bias}`
    /// for the output projection.
    pub fn load(vb: impl AsRef<VarBuilder>, config: WindowAttentionConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let d = config.hidden_size();

        let q_w = vb.get(&[d, d], "query.weight")?;
        let q_b = vb.get(&[d], "query.bias")?;
        let k_w = vb.get(&[d, d], "key.weight")?;
        let k_b = vb.get(&[d], "key.bias")?;
        let v_w = vb.get(&[d, d], "value.weight")?;
        let v_b = vb.get(&[d], "value.bias")?;

        let qkv_w = DynTensor::cat(&[&q_w, &k_w, &v_w], 0)?;
        let qkv_b = DynTensor::cat(&[&q_b, &k_b, &v_b], 0)?;
        let qkv = Linear::new(qkv_w, Some(qkv_b))?;

        let proj_w = vb.get(&[d, d], "proj.weight")?;
        let proj_b = vb.get(&[d], "proj.bias")?;
        let out_proj = Linear::new(proj_w, Some(proj_b))?;

        Self::new(qkv, out_proj, config)
    }

    /// Forward pass with window attention.
    ///
    /// - `x`: input tensor `[B, H*W, D]`
    /// - `height`, `width`: spatial grid dimensions (`H * W == x.dim(1)`)
    ///
    /// Returns `[B, H*W, D]`.
    pub fn forward(&self, x: &DynTensor, height: usize, width: usize) -> Result<DynTensor> {
        let (b, seq_len, _d) = x.dims3()?;
        let d = self.config.hidden_size();
        let ws = self.config.window_size;

        if seq_len != height * width {
            return Err(TensorError::shape_mismatch(
                vec![b, height * width, d],
                vec![b, seq_len, d],
            ));
        }

        // 1. Window partition: [B, H*W, D] -> [B*nw, ws*ws, D]
        let (windowed, ph, pw) = window_partition(x, height, width, ws)?;
        let (bw, s, _) = windowed.dims3()?;

        // 2. Fused QKV: [B*nw, ws*ws, D] -> [B*nw, ws*ws, 3*D]
        let qkv = self.qkv.forward(&windowed)?;
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        // 3. Multi-head reshape: [B*nw, S, D] -> [B*nw, H, S, head_dim]
        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let q = q.reshape([bw, s, nh, hd])?.transpose(1, 2)?;
        let k = k.reshape([bw, s, nh, hd])?.transpose(1, 2)?;
        let v = v.reshape([bw, s, nh, hd])?.transpose(1, 2)?;

        // 4. SDPA within each window
        let attn_out = sdpa(&q, &k, &v, None, self.scale)?;

        // 5. Reshape back: [B*nw, H, S, head_dim] -> [B*nw, S, D]
        let attn_out = attn_out.transpose(1, 2)?.reshape([bw, s, d])?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // 6. Window unpartition: [B*nw, ws*ws, D] -> [B, H*W, D]
        let result = window_unpartition(&attn_out, height, width, ph, pw, ws, b)?;
        check_output_finite(&result, "WindowMultiHeadAttention")?;
        Ok(result)
    }

    /// Access the window attention config.
    #[must_use]
    pub fn config(&self) -> &WindowAttentionConfig {
        &self.config
    }
}

/// Attention mode for a ViT encoder block: either global or window-local.
///
/// Used by [`VitEncoderBlock`](crate::layers::vision::VitEncoderBlock) to select
/// between full sequence attention and window-partitioned local attention
/// (as in Qwen2.5-VL).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMode {
    /// Full global self-attention over the entire sequence.
    Global,
    /// Window-local attention: partition into spatial windows before attention.
    Window,
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "window_integration_tests.rs"]
mod integration_tests;

#[cfg(test)]
#[path = "window_attention_tests.rs"]
mod attention_tests;
