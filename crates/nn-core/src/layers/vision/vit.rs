// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vision Transformer (ViT) patch embedding and encoder.
//!
//! Provides [`PatchEmbedding`] for extracting image patches via Conv2d projection,
//! [`VitEncoderBlock`] for pre-norm transformer encoder layers, and [`VitEncoder`]
//! for the full ViT encoder stack. [`VitConfig`] lives in `vit_config.rs`.
//!
//! Architecture follows Dosovitskiy et al. 2020 ("An Image is Worth 16x16 Words"):
//!
//! ```text
//! image [B, C, H, W]
//!   → PatchEmbedding (Conv2d stride=P) → [B, N, D]
//!   + positional_embedding [1, N+1, D] (with optional CLS token)
//!   → N × VitEncoderBlock (LayerNorm→MHA→residual→LayerNorm→MLP→residual)
//!   → LayerNorm
//!   → output [B, N, D] or CLS token [B, D]
//! ```
//!
//! Required by 10+ docling/VLM models (#1074): GraniteDocling, Qwen2-VL, Pixtral,
//! Phi-4-multimodal, Gemma-3, SmolDocling.

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{window_partition, window_unpartition, AttentionMode};
use crate::layers::{
    check_output_finite, validate_divisible, validate_eps, validate_heads, Conv2d, Conv2dConfig,
    LayerNorm, Linear, Module,
};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

#[path = "vit_config.rs"]
mod config;
pub use config::VitConfig;

/// Patch embedding: project image patches to embedding vectors via Conv2d.
///
/// Input: `[B, C, H, W]` image tensor.
/// Output: `[B, num_patches, hidden_size]` patch embeddings.
///
/// Uses a Conv2d with `kernel_size = patch_size` and `stride = patch_size`
/// to extract non-overlapping patches and project them to `hidden_size` in one step.
#[derive(Clone)]
pub struct PatchEmbedding {
    /// Conv2d projection: [C, H, W] -> [hidden_size, H/P, W/P].
    projection: Conv2d,
    /// Hidden dimension.
    hidden_size: usize,
}

impl std::fmt::Debug for PatchEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchEmbedding")
            .field("hidden_size", &self.hidden_size)
            .finish_non_exhaustive()
    }
}

impl PatchEmbedding {
    /// Create from a pre-loaded Conv2d projection.
    ///
    /// Returns an error if `hidden_size` is zero.
    pub fn new(projection: Conv2d, hidden_size: usize) -> Result<Self> {
        if hidden_size == 0 {
            return Err(TensorError::InvalidShape(
                "PatchEmbedding: hidden_size must be > 0".into(),
            ));
        }
        Ok(Self {
            projection,
            hidden_size,
        })
    }

    /// Load from a [`VarBuilder`].
    ///
    /// Loads `projection.weight` `[hidden_size, num_channels, patch_size, patch_size]`
    /// and optional `projection.bias` `[hidden_size]`.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &VitConfig) -> Result<Self> {
        let vb = vb.as_ref();
        config.validate()?;
        let proj_vb = vb.pp("projection");
        let w = proj_vb.get(
            &[
                config.hidden_size,
                config.num_channels,
                config.patch_size,
                config.patch_size,
            ],
            "weight",
        )?;
        let b = if proj_vb.contains_tensor("bias") {
            Some(proj_vb.get(&[config.hidden_size], "bias")?)
        } else {
            None
        };
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let projection = Conv2d::new(w, b, conv_config)?;
        Self::new(projection, config.hidden_size)
    }

    /// Forward: extract and project patches.
    ///
    /// Input: `[B, C, H, W]`
    /// Output: `[B, num_patches, hidden_size]`
    pub fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        // Conv2d: [B, C, H, W] -> [B, hidden_size, H/P, W/P]
        let out = self.projection.forward(x)?;
        let (b, _c, h, w) = out.dims4()?;
        let num_patches = h * w;
        // Flatten spatial dims and transpose: [B, D, H', W'] -> [B, D, N] -> [B, N, D]
        let out = out.reshape([b, self.hidden_size, num_patches])?;
        out.transpose(1, 2)
    }
}

impl Module for PatchEmbedding {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        Self::forward(self, x)
    }
}

/// Single ViT encoder block (pre-norm transformer encoder layer).
///
/// ```text
/// x → LayerNorm → MultiHeadAttention → + residual
///   → LayerNorm → MLP (Linear→GELU→Linear) → + residual
/// ```
///
/// Supports both global and window-local attention via [`forward_with_spatial`](Self::forward_with_spatial).
/// When `window_size` is `Some(ws)` and the block is called with [`AttentionMode::Window`],
/// tokens are partitioned into spatial windows before attention (Qwen2.5-VL pattern).
#[derive(Clone)]
pub struct VitEncoderBlock {
    pub(super) ln1: LayerNorm,
    pub(super) attn_qkv: Linear,
    pub(super) attn_proj: Linear,
    pub(super) ln2: LayerNorm,
    pub(super) mlp_fc1: Linear,
    pub(super) mlp_fc2: Linear,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) scale: f64,
    /// Optional window size for local attention. `None` = global only.
    pub(super) window_size: Option<usize>,
}

impl std::fmt::Debug for VitEncoderBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VitEncoderBlock")
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .finish_non_exhaustive()
    }
}

impl VitEncoderBlock {
    /// Create from pre-built layers.
    ///
    /// - `ln1`: pre-attention LayerNorm
    /// - `attn_qkv`: fused Q/K/V projection `[3*D, D]`
    /// - `attn_proj`: output projection `[D, D]`
    /// - `ln2`: pre-MLP LayerNorm
    /// - `mlp_fc1`: first MLP layer `[intermediate, D]`
    /// - `mlp_fc2`: second MLP layer `[D, intermediate]`
    /// - `num_heads`: number of attention heads
    /// - `head_dim`: dimension per head (`hidden_size / num_heads`)
    pub fn new(
        ln1: LayerNorm,
        attn_qkv: Linear,
        attn_proj: Linear,
        ln2: LayerNorm,
        mlp_fc1: Linear,
        mlp_fc2: Linear,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        validate_heads(num_heads, "VitEncoderBlock")?;
        if head_dim == 0 {
            return Err(TensorError::InvalidShape(
                "VitEncoderBlock: head_dim must be > 0".into(),
            ));
        }
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            ln1,
            attn_qkv,
            attn_proj,
            ln2,
            mlp_fc1,
            mlp_fc2,
            num_heads,
            head_dim,
            scale,
            window_size: None,
        })
    }

    /// Create from pre-built layers with window attention support.
    ///
    /// Same as [`new`](Self::new) but also sets the `window_size` for local
    /// attention. When `forward_with_spatial` is called with `AttentionMode::Window`,
    /// tokens are partitioned into windows of this size before attention.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_window(
        ln1: LayerNorm,
        attn_qkv: Linear,
        attn_proj: Linear,
        ln2: LayerNorm,
        mlp_fc1: Linear,
        mlp_fc2: Linear,
        num_heads: usize,
        head_dim: usize,
        window_size: usize,
    ) -> Result<Self> {
        if window_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "VitEncoderBlock: window_size must be > 0",
            });
        }
        let mut block = Self::new(
            ln1, attn_qkv, attn_proj, ln2, mlp_fc1, mlp_fc2, num_heads, head_dim,
        )?;
        block.window_size = Some(window_size);
        Ok(block)
    }

    /// Load an encoder block from a [`VarBuilder`].
    ///
    /// Expects HuggingFace ViT weight naming:
    /// - `attention.attention.query.{weight,bias}`
    /// - `attention.attention.key.{weight,bias}`
    /// - `attention.attention.value.{weight,bias}`
    /// - `attention.output.dense.{weight,bias}`
    /// - `layernorm_before.{weight,bias}`
    /// - `layernorm_after.{weight,bias}`
    /// - `intermediate.dense.{weight,bias}`
    /// - `output.dense.{weight,bias}`
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        eps: f64,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        validate_heads(num_heads, "VitEncoderBlock")?;
        validate_divisible(
            hidden_size,
            num_heads,
            "hidden_size",
            "num_heads",
            "VitEncoderBlock",
        )?;
        if intermediate_size == 0 {
            return Err(TensorError::InvalidShape(
                "VitEncoderBlock: intermediate_size must be > 0".into(),
            ));
        }
        validate_eps(eps, "VitEncoderBlock")?;
        let head_dim = hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Pre-attention LayerNorm
        let ln1 = load_layer_norm(vb, "layernorm_before", hidden_size, eps)?;

        // Attention: load separate Q, K, V and fuse into QKV
        let attn_vb = vb.pp("attention").pp("attention");
        let q_w = attn_vb.get(&[hidden_size, hidden_size], "query.weight")?;
        let q_b = attn_vb.get(&[hidden_size], "query.bias")?;
        let k_w = attn_vb.get(&[hidden_size, hidden_size], "key.weight")?;
        let k_b = attn_vb.get(&[hidden_size], "key.bias")?;
        let v_w = attn_vb.get(&[hidden_size, hidden_size], "value.weight")?;
        let v_b = attn_vb.get(&[hidden_size], "value.bias")?;

        let qkv_w = DynTensor::cat(&[&q_w, &k_w, &v_w], 0)?;
        let qkv_b = DynTensor::cat(&[&q_b, &k_b, &v_b], 0)?;
        let attn_qkv = Linear::new(qkv_w, Some(qkv_b))?;

        // Output projection
        let out_vb = vb.pp("attention").pp("output");
        let proj_w = out_vb.get(&[hidden_size, hidden_size], "dense.weight")?;
        let proj_b = out_vb.get(&[hidden_size], "dense.bias")?;
        let attn_proj = Linear::new(proj_w, Some(proj_b))?;

        // Post-attention LayerNorm
        let ln2 = load_layer_norm(vb, "layernorm_after", hidden_size, eps)?;

        // MLP
        let fc1_vb = vb.pp("intermediate");
        let fc1_w = fc1_vb.get(&[intermediate_size, hidden_size], "dense.weight")?;
        let fc1_b = fc1_vb.get(&[intermediate_size], "dense.bias")?;
        let mlp_fc1 = Linear::new(fc1_w, Some(fc1_b))?;

        let fc2_vb = vb.pp("output");
        let fc2_w = fc2_vb.get(&[hidden_size, intermediate_size], "dense.weight")?;
        let fc2_b = fc2_vb.get(&[hidden_size], "dense.bias")?;
        let mlp_fc2 = Linear::new(fc2_w, Some(fc2_b))?;

        Ok(Self {
            ln1,
            attn_qkv,
            attn_proj,
            ln2,
            mlp_fc1,
            mlp_fc2,
            num_heads,
            head_dim,
            scale,
            window_size: None,
        })
    }

    /// Forward pass: pre-norm self-attention + MLP with residuals.
    ///
    /// Input/Output: `[B, S, D]`.
    pub fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let (b, s, _d) = x.dims3()?;
        let d = self.num_heads * self.head_dim;

        let residual = x;
        let h = self.ln1.forward(x)?;

        // Fused QKV: [B, S, D] -> [B, S, 3*D] -> split Q, K, V
        let qkv = self.attn_qkv.forward(&h)?;
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        // Multi-head reshape: [B, S, D] -> [B, H, S, head_dim]
        let q = q
            .reshape([b, s, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([b, s, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([b, s, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let attn_out = crate::layers::attention::sdpa(&q, &k, &v, None, self.scale)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape([b, s, d])?;
        let attn_out = self.attn_proj.forward(&attn_out)?;

        let x = residual.add(&attn_out)?;

        // MLP with residual
        let residual = x.clone();
        let h = self.ln2.forward(&x)?;
        let h = self.mlp_fc1.forward(&h)?.gelu()?;
        let h = self.mlp_fc2.forward(&h)?;

        let out = residual.add(&h)?;
        Ok(out)
    }

    /// Forward pass with spatial-aware attention mode selection.
    ///
    /// When `mode` is [`AttentionMode::Window`] and this block has a `window_size`,
    /// tokens are partitioned into non-overlapping spatial windows before attention.
    /// When `mode` is [`AttentionMode::Global`] (or no `window_size` is set), this
    /// delegates to the standard [`forward`](Self::forward).
    ///
    /// - `x`: input `[B, H*W, D]`
    /// - `height`, `width`: spatial grid dimensions
    /// - `mode`: global or window attention
    ///
    /// Returns `[B, H*W, D]`.
    pub fn forward_with_spatial(
        &self,
        x: &DynTensor,
        height: usize,
        width: usize,
        mode: AttentionMode,
    ) -> Result<DynTensor> {
        let ws = match (mode, self.window_size) {
            (AttentionMode::Window, Some(ws)) => ws,
            _ => return self.forward(x),
        };

        let (b, s, _d) = x.dims3()?;
        let d = self.num_heads * self.head_dim;

        if s != height * width {
            return Err(TensorError::shape_mismatch(
                vec![b, height * width, d],
                vec![b, s, d],
            ));
        }

        let residual = x;
        let h = self.ln1.forward(x)?;

        // Window partition: [B, H*W, D] -> [B*nw, ws*ws, D]
        let (windowed, ph, pw) = window_partition(&h, height, width, ws)?;
        let (bw, sw, _) = windowed.dims3()?;

        // Fused QKV within each window
        let qkv = self.attn_qkv.forward(&windowed)?;
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        let q = q
            .reshape([bw, sw, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([bw, sw, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([bw, sw, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let attn_out = crate::layers::attention::sdpa(&q, &k, &v, None, self.scale)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape([bw, sw, d])?;
        let attn_out = self.attn_proj.forward(&attn_out)?;

        // Window unpartition: [B*nw, ws*ws, D] -> [B, H*W, D]
        let attn_out = window_unpartition(&attn_out, height, width, ph, pw, ws, b)?;

        let x = residual.add(&attn_out)?;

        // MLP with residual (same as global)
        let residual = x.clone();
        let h = self.ln2.forward(&x)?;
        let h = self.mlp_fc1.forward(&h)?.gelu()?;
        let h = self.mlp_fc2.forward(&h)?;

        let out = residual.add(&h)?;
        check_output_finite(&out, "VitEncoderBlock (window)")?;
        Ok(out)
    }

    /// The configured window size, if any.
    #[must_use]
    pub fn window_size(&self) -> Option<usize> {
        self.window_size
    }
}

impl Module for VitEncoderBlock {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        Self::forward(self, x)
    }
}

/// ViT output pooling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PoolingStrategy {
    /// Return the CLS token output (index 0). Requires `use_cls_token = true`.
    Cls,
    /// Return the mean of all patch token outputs (excluding CLS if present).
    Mean,
    /// Return all token outputs without pooling.
    None,
}

// VitEncoder extracted to vit_encoder.rs for 500-line limit compliance.
#[path = "vit_encoder.rs"]
mod encoder;
pub use encoder::VitEncoder;

/// Load a LayerNorm from VarBuilder at a given key prefix.
pub(super) fn load_layer_norm(
    vb: impl AsRef<VarBuilder>,
    name: &str,
    hidden_size: usize,
    eps: f64,
) -> Result<LayerNorm> {
    let vb = vb.as_ref();
    let w = vb.get(&[hidden_size], &format!("{name}.weight"))?;
    let b = vb.get(&[hidden_size], &format!("{name}.bias"))?;
    LayerNorm::new(w, b, eps)
}

#[cfg(kani)]
#[path = "kani_vit_proofs.rs"]
mod kani_vit_proofs;

#[cfg(kani)]
#[path = "kani_vit.rs"]
mod kani_vit;

#[cfg(kani)]
#[path = "kani_vit_advanced.rs"]
mod kani_vit_advanced;

#[cfg(test)]
#[path = "vit_tests.rs"]
mod tests;
