// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL-8B vision encoder with DeepStack feature extraction.
//!
//! 27-layer ViT (1152 hidden, 4304 intermediate, 16 heads, head_dim 72,
//! patch_size 16) with 2D RoPE and full attention on every layer.
//!
//! Key components:
//! - **2D RoPE:** Merge-block position ordering with base=10000
//! - **VisionBlock:** Pre-norm LayerNorm, fused QKV, full attention, GELU MLP
//! - **PatchMerger:** 2x2 spatial merge (pre-shuffle or post-shuffle norm)
//! - **PatchEmbed:** Conv3d emulated via Conv2d for single-image inference
//! - **Qwen3VlVisionEncoder:** 27 blocks + main merger + 3 DeepStack mergers
//!
//! Weight keys follow HuggingFace `Qwen/Qwen3-VL-8B-Instruct` safetensors:
//! - `visual.patch_embed.proj.{weight,bias}`
//! - `visual.blocks.{i}.{norm1,norm2,attn.{qkv,proj},mlp.{fc1,fc2}}.*`
//! - `visual.merger.{norm,linear_fc1,linear_fc2}.*`
//! - `visual.deepstack_merger_list.{j}.{norm,linear_fc1,linear_fc2}.*`
//! - `visual.post_layernorm.{weight,bias}`

use nn_core::layers::{
    check_output_finite, rope, sdpa, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
};
use nn_core::{Device, DynTensor, Result, TensorError, VarBuilder};

// ---------------------------------------------------------------------------
// Vision encoder constants (Qwen3-VL-8B-Instruct config.json)
// ---------------------------------------------------------------------------

pub(crate) const VISION_HIDDEN_SIZE: usize = 1152;
pub(crate) const VISION_INTERMEDIATE_SIZE: usize = 4304;
pub(crate) const VISION_NUM_LAYERS: usize = 27;
pub(crate) const VISION_NUM_HEADS: usize = 16;
pub(crate) const VISION_HEAD_DIM: usize = 72; // 1152 / 16
pub(crate) const VISION_LAYER_NORM_EPS: f64 = 1e-6;
pub(crate) const PATCH_SIZE: usize = 16;
pub(crate) const SPATIAL_MERGE_SIZE: usize = 2;
pub(crate) const OUT_HIDDEN_SIZE: usize = 4096;
pub(crate) const DEEPSTACK_INDEXES: [usize; 3] = [8, 16, 24];
pub(crate) const VISION_ROPE_BASE: f64 = 10_000.0;

/// Merged hidden dimension after 2x2 spatial merge.
pub(crate) const SPATIAL_MERGE_HIDDEN: usize =
    VISION_HIDDEN_SIZE * SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE;

// ---------------------------------------------------------------------------
// 2D RoPE for vision encoder
// ---------------------------------------------------------------------------

/// 2D Rotary Position Embedding for the Qwen3-VL vision encoder.
///
/// Uses the half-split convention: the first half of head_dim encodes height
/// positions, the second half encodes width positions. Position IDs follow
/// merge-block ordering for compatibility with the 2x2 spatial merge.
#[derive(Clone)]
pub(crate) struct VisionRoPE {
    pub(crate) cos: DynTensor,
    pub(crate) sin: DynTensor,
}

impl VisionRoPE {
    /// Build vision RoPE cos/sin tables for a grid of positions.
    ///
    /// Position IDs are interleaved within 2x2 merge blocks per the Qwen3-VL
    /// merge_size ordering: `block_row * merge_size + intra_row`.
    pub(crate) fn new(grid_h: usize, grid_w: usize, device: &Device) -> Result<Self> {
        let seq_len = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| TensorError::InvalidShape("VisionRoPE: grid size overflow".into()))?;

        let rope_dim = VISION_HEAD_DIM / 2; // 36
        let num_freqs = rope_dim / 2; // 18

        let inv_freq: Vec<f64> = (0..num_freqs)
            .map(|i| {
                let exponent = (2 * i) as f64 / rope_dim as f64;
                1.0 / VISION_ROPE_BASE.powf(exponent)
            })
            .collect();

        let merged_h = grid_h / SPATIAL_MERGE_SIZE;
        let merged_w = grid_w / SPATIAL_MERGE_SIZE;

        let mut cos_data = Vec::with_capacity(seq_len * rope_dim);
        let mut sin_data = Vec::with_capacity(seq_len * rope_dim);

        // Token order: iterate over merged blocks, then within each block
        // enumerate the spatial_merge_size x spatial_merge_size sub-patches.
        for bh in 0..merged_h {
            for bw in 0..merged_w {
                for ih in 0..SPATIAL_MERGE_SIZE {
                    for iw in 0..SPATIAL_MERGE_SIZE {
                        let h_pos = bh * SPATIAL_MERGE_SIZE + ih;
                        let w_pos = bw * SPATIAL_MERGE_SIZE + iw;

                        for &freq in &inv_freq {
                            let angle = h_pos as f64 * freq;
                            cos_data.push(angle.cos() as f32);
                            sin_data.push(angle.sin() as f32);
                        }
                        for &freq in &inv_freq {
                            let angle = w_pos as f64 * freq;
                            cos_data.push(angle.cos() as f32);
                            sin_data.push(angle.sin() as f32);
                        }
                    }
                }
            }
        }

        let cos = DynTensor::from_vec(cos_data, &[1, 1, seq_len, rope_dim], &Device::Cpu)?
            .to_device(device)?;
        let sin = DynTensor::from_vec(sin_data, &[1, 1, seq_len, rope_dim], &Device::Cpu)?
            .to_device(device)?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, x: &DynTensor) -> Result<DynTensor> {
        rope(x, &self.cos, &self.sin)
    }
}

// ---------------------------------------------------------------------------
// Vision Transformer Block
// ---------------------------------------------------------------------------

/// Single Qwen3-VL vision transformer block.
///
/// Uses fused QKV projection (3*hidden -> q,k,v split) and full attention
/// (no windowed attention unlike Qwen2.5-VL). MLP uses GELU activation.
#[derive(Clone)]
struct VisionBlock {
    norm1: LayerNorm,
    qkv: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    scale: f64,
}

impl VisionBlock {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let scale = 1.0 / (VISION_HEAD_DIM as f64).sqrt();
        let attn_vb = vb.pp("attn");

        Ok(Self {
            norm1: LayerNorm::load(vb.pp("norm1"), VISION_HIDDEN_SIZE, VISION_LAYER_NORM_EPS)?,
            qkv: Linear::load(
                attn_vb.pp("qkv"),
                VISION_HIDDEN_SIZE,
                3 * VISION_HIDDEN_SIZE,
            )?,
            out_proj: Linear::load(attn_vb.pp("proj"), VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)?,
            norm2: LayerNorm::load(vb.pp("norm2"), VISION_HIDDEN_SIZE, VISION_LAYER_NORM_EPS)?,
            mlp_fc1: Linear::load(
                vb.pp("mlp").pp("fc1"),
                VISION_HIDDEN_SIZE,
                VISION_INTERMEDIATE_SIZE,
            )?,
            mlp_fc2: Linear::load(
                vb.pp("mlp").pp("fc2"),
                VISION_INTERMEDIATE_SIZE,
                VISION_HIDDEN_SIZE,
            )?,
            scale,
        })
    }

    fn forward(&self, x: &DynTensor, vision_rope: &VisionRoPE) -> Result<DynTensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Pre-norm + fused QKV
        let hidden = self.norm1.forward(x)?;
        let qkv = self.qkv.forward(&hidden)?;

        // Split fused QKV: [B, seq, 3*hidden] -> three [B, seq, hidden]
        let q = qkv.narrow(2, 0, VISION_HIDDEN_SIZE)?;
        let k = qkv.narrow(2, VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)?;
        let v = qkv.narrow(2, 2 * VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)?;

        // Reshape to [B, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_len, VISION_NUM_HEADS, VISION_HEAD_DIM])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, VISION_NUM_HEADS, VISION_HEAD_DIM])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, VISION_NUM_HEADS, VISION_HEAD_DIM])?
            .transpose(1, 2)?;

        // Apply 2D RoPE
        let q = vision_rope.apply(&q)?;
        let k = vision_rope.apply(&k)?;

        // Scaled dot-product attention (full, no mask)
        let attn = sdpa(&q, &k, &v, None, self.scale)?;
        let attn =
            attn.transpose(1, 2)?
                .reshape([batch, seq_len, VISION_NUM_HEADS * VISION_HEAD_DIM])?;
        let attn = self.out_proj.forward(&attn)?;

        // First residual
        let hidden = x.broadcast_add(&attn)?;

        // MLP with GELU activation
        let mlp = self.norm2.forward(&hidden)?;
        let mlp = self.mlp_fc1.forward(&mlp)?.gelu()?;
        let mlp = self.mlp_fc2.forward(&mlp)?;

        // Second residual
        hidden.broadcast_add(&mlp)
    }
}

// ---------------------------------------------------------------------------
// Patch Merger (shared between main merger and DeepStack mergers)
// ---------------------------------------------------------------------------

/// 2x2 spatial merge projector.
///
/// Two variants controlled by `use_postshuffle_norm`:
/// - Main merger: LayerNorm on hidden_size (1152) BEFORE the 2x2 shuffle
/// - DeepStack mergers: LayerNorm on merged dim (4608) AFTER the 2x2 shuffle
#[derive(Clone)]
pub(crate) struct PatchMerger {
    norm: LayerNorm,
    linear_fc1: Linear,
    linear_fc2: Linear,
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    pub(crate) fn load(vb: &VarBuilder, use_postshuffle_norm: bool) -> Result<Self> {
        let norm_dim = if use_postshuffle_norm {
            SPATIAL_MERGE_HIDDEN
        } else {
            VISION_HIDDEN_SIZE
        };

        Ok(Self {
            norm: LayerNorm::load(vb.pp("norm"), norm_dim, VISION_LAYER_NORM_EPS)?,
            linear_fc1: Linear::load(
                vb.pp("linear_fc1"),
                SPATIAL_MERGE_HIDDEN,
                SPATIAL_MERGE_HIDDEN,
            )?,
            linear_fc2: Linear::load(vb.pp("linear_fc2"), SPATIAL_MERGE_HIDDEN, OUT_HIDDEN_SIZE)?,
            use_postshuffle_norm,
        })
    }

    pub(crate) fn forward(&self, x: &DynTensor, grid_h: usize, grid_w: usize) -> Result<DynTensor> {
        let (batch, seq_len, hidden_size) = x.dims3()?;
        if hidden_size != VISION_HIDDEN_SIZE {
            return Err(TensorError::shape_mismatch(
                vec![batch, seq_len, VISION_HIDDEN_SIZE],
                vec![batch, seq_len, hidden_size],
            ));
        }
        let expected_seq = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| TensorError::InvalidShape("PatchMerger: grid size overflow".into()))?;
        if seq_len != expected_seq {
            return Err(TensorError::shape_mismatch(
                vec![batch, expected_seq, hidden_size],
                vec![batch, seq_len, hidden_size],
            ));
        }
        if !grid_h.is_multiple_of(SPATIAL_MERGE_SIZE) || !grid_w.is_multiple_of(SPATIAL_MERGE_SIZE)
        {
            return Err(TensorError::InvalidShape(format!(
                "PatchMerger: grid {grid_h}x{grid_w} not divisible by merge_size \
                 {SPATIAL_MERGE_SIZE}"
            )));
        }

        let merged_h = grid_h / SPATIAL_MERGE_SIZE;
        let merged_w = grid_w / SPATIAL_MERGE_SIZE;
        let merged_tokens = merged_h * merged_w;

        if self.use_postshuffle_norm {
            // DeepStack: shuffle first, then LayerNorm on merged dim
            let x = x.reshape([
                batch,
                merged_h,
                SPATIAL_MERGE_SIZE,
                merged_w,
                SPATIAL_MERGE_SIZE,
                VISION_HIDDEN_SIZE,
            ])?;
            let x = x.permute([0, 1, 3, 2, 4, 5])?;
            let x = x.reshape([batch, merged_tokens, SPATIAL_MERGE_HIDDEN])?;
            let x = self.norm.forward(&x)?;
            let x = self.linear_fc1.forward(&x)?.gelu()?;
            self.linear_fc2.forward(&x)
        } else {
            // Main merger: LayerNorm on hidden_size, then shuffle
            let x = self.norm.forward(x)?;
            let x = x.reshape([
                batch,
                merged_h,
                SPATIAL_MERGE_SIZE,
                merged_w,
                SPATIAL_MERGE_SIZE,
                VISION_HIDDEN_SIZE,
            ])?;
            let x = x.permute([0, 1, 3, 2, 4, 5])?;
            let x = x.reshape([batch, merged_tokens, SPATIAL_MERGE_HIDDEN])?;
            let x = self.linear_fc1.forward(&x)?.gelu()?;
            self.linear_fc2.forward(&x)
        }
    }
}

// ---------------------------------------------------------------------------
// Patch Embedding (Conv3d emulated via Conv2d)
// ---------------------------------------------------------------------------

/// Qwen3-VL patch embedding.
///
/// The original uses Conv3d(3, 1152, kernel=[2,16,16], stride=[2,16,16]).
/// For single-image inference (temporal_patch_size=2 with duplicated frames),
/// we emulate this by summing the two temporal kernel slices through Conv2d.
///
/// This is correct for still images where the two temporal frames are
/// identical. For video inference, a proper Conv3d op would be needed.
#[derive(Clone)]
struct PatchEmbed {
    projection: Conv2d,
}

impl PatchEmbed {
    /// Load from a Conv3d weight tensor by summing the temporal dimension.
    fn load_from_conv3d(vb: &VarBuilder) -> Result<Self> {
        // Load the Conv3d weight [1152, 3, 2, 16, 16]
        // and sum along temporal dim to get [1152, 3, 16, 16].
        let weight_5d = vb.get(
            &[VISION_HIDDEN_SIZE, 3, 2, PATCH_SIZE, PATCH_SIZE],
            "weight",
        )?;
        let bias = vb.get(&[VISION_HIDDEN_SIZE], "bias")?;

        // Sum temporal dim: take slice at t=0 and t=1, add them
        let w_t0 = weight_5d.narrow(2, 0, 1)?.squeeze(2)?;
        let w_t1 = weight_5d.narrow(2, 1, 1)?.squeeze(2)?;
        let weight_4d = w_t0.broadcast_add(&w_t1)?;

        let config = Conv2dConfig::new(0, PATCH_SIZE, 1);
        let conv = Conv2d::new(weight_4d, Some(bias), config)?;
        Ok(Self { projection: conv })
    }

    fn forward(&self, image: &DynTensor) -> Result<(DynTensor, usize, usize)> {
        let features = self.projection.forward(image)?;
        let (batch, _channels, grid_h, grid_w) = features.dims4()?;
        let seq_len = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| TensorError::InvalidShape("PatchEmbed: grid size overflow".into()))?;
        let tokens = features.reshape([batch, VISION_HIDDEN_SIZE, seq_len])?;
        Ok((tokens.transpose(1, 2)?, grid_h, grid_w))
    }
}

// ---------------------------------------------------------------------------
// Qwen3-VL Vision Encoder (public API)
// ---------------------------------------------------------------------------

/// Qwen3-VL-8B vision encoder with DeepStack feature extraction.
///
/// Input: `[batch, 3, height, width]` where height and width are divisible by
/// `patch_size * spatial_merge_size = 32`.
///
/// Outputs:
/// - Main features: `[batch, merged_tokens, 4096]` (from final encoder output)
/// - DeepStack features: 3 x `[batch, merged_tokens, 4096]` (from layers 8, 16, 24)
///
/// The DeepStack features are injected additively into decoder layers 0, 1, 2
/// at vision token positions during the decoder forward pass.
pub struct Qwen3VlVisionEncoder {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    post_layernorm: LayerNorm,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
}

impl std::fmt::Debug for Qwen3VlVisionEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3VlVisionEncoder")
            .field("num_layers", &self.blocks.len())
            .field("deepstack_indexes", &DEEPSTACK_INDEXES)
            .finish_non_exhaustive()
    }
}

/// Vision encoder output containing main and DeepStack features.
#[derive(Debug)]
pub struct Qwen3VlVisionOutput {
    /// Main merger output: `[batch, merged_tokens, 4096]`.
    pub features: DynTensor,
    /// DeepStack features from layers 8, 16, 24.
    /// Each tensor: `[batch, merged_tokens, 4096]`.
    pub deepstack_features: Vec<DynTensor>,
}

impl Qwen3VlVisionEncoder {
    /// Load the vision encoder from HuggingFace safetensors weights.
    ///
    /// Expected key prefix: `visual.*`
    ///
    /// # Errors
    ///
    /// Returns an error if weight tensors are missing or have wrong shapes.
    pub(crate) fn load(vb: &VarBuilder) -> Result<Self> {
        let visual_vb = vb.pp("visual");

        // Patch embedding: Conv3d -> Conv2d workaround
        let patch_embed = PatchEmbed::load_from_conv3d(&visual_vb.pp("patch_embed").pp("proj"))?;

        // 27 vision blocks
        let mut blocks = Vec::with_capacity(VISION_NUM_LAYERS);
        for i in 0..VISION_NUM_LAYERS {
            blocks.push(VisionBlock::load(
                &visual_vb.pp("blocks").pp(i.to_string()),
            )?);
        }

        // Post-LayerNorm (applied after final block, before main merger)
        let post_layernorm = LayerNorm::load(
            visual_vb.pp("post_layernorm"),
            VISION_HIDDEN_SIZE,
            VISION_LAYER_NORM_EPS,
        )?;

        // Main merger (pre-shuffle norm on hidden_size=1152)
        let merger = PatchMerger::load(&visual_vb.pp("merger"), false)?;

        // DeepStack mergers (post-shuffle norm on merged_dim=4608)
        let mut deepstack_mergers = Vec::with_capacity(DEEPSTACK_INDEXES.len());
        for j in 0..DEEPSTACK_INDEXES.len() {
            deepstack_mergers.push(PatchMerger::load(
                &visual_vb.pp("deepstack_merger_list").pp(j.to_string()),
                true,
            )?);
        }

        Ok(Self {
            patch_embed,
            blocks,
            post_layernorm,
            merger,
            deepstack_mergers,
        })
    }

    /// Run the vision encoder on an RGB image tensor.
    ///
    /// # Arguments
    ///
    /// * `image` - `[batch, 3, height, width]` where H and W are divisible by 32
    ///
    /// # Returns
    ///
    /// [`Qwen3VlVisionOutput`] with main features and 3 DeepStack feature tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if the image dimensions are invalid or computation fails.
    pub(crate) fn forward(&self, image: &DynTensor) -> Result<Qwen3VlVisionOutput> {
        let (_batch, channels, height, width) = image.dims4()?;
        if channels != 3 {
            return Err(TensorError::shape_mismatch(
                vec![3, height, width],
                vec![channels, height, width],
            ));
        }

        let merge_patch = PATCH_SIZE * SPATIAL_MERGE_SIZE;
        if !height.is_multiple_of(merge_patch) || !width.is_multiple_of(merge_patch) {
            return Err(TensorError::InvalidShape(format!(
                "Qwen3VlVisionEncoder: image {height}x{width} must be divisible by \
                 {merge_patch}"
            )));
        }

        // Patch embedding
        let (mut hidden, grid_h, grid_w) = self.patch_embed.forward(image)?;

        // Build vision RoPE with merge-block ordering
        let vision_rope = VisionRoPE::new(grid_h, grid_w, &image.device())?;

        // Run through transformer blocks, extracting DeepStack features
        let mut deepstack_features = Vec::with_capacity(DEEPSTACK_INDEXES.len());

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &vision_rope)?;

            // Extract DeepStack features at specified layers
            if let Some(ds_idx) = DEEPSTACK_INDEXES.iter().position(|&idx| idx == layer_idx) {
                let ds_features =
                    self.deepstack_mergers[ds_idx].forward(&hidden, grid_h, grid_w)?;
                deepstack_features.push(ds_features);
            }
        }

        // Post-LayerNorm on final encoder output
        hidden = self.post_layernorm.forward(&hidden)?;

        // Main merger: project to decoder hidden size
        let features = self.merger.forward(&hidden, grid_h, grid_w)?;
        check_output_finite(&features, "Qwen3VlVisionEncoder")?;

        Ok(Qwen3VlVisionOutput {
            features,
            deepstack_features,
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests (inline for constants and VisionRoPE)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_constants_consistent() {
        assert_eq!(VISION_HIDDEN_SIZE / VISION_NUM_HEADS, VISION_HEAD_DIM);
        assert_eq!(
            VISION_HIDDEN_SIZE * SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_HIDDEN
        );
        assert_eq!(SPATIAL_MERGE_HIDDEN, 4608);
        assert_eq!(DEEPSTACK_INDEXES.len(), 3);
        for &idx in &DEEPSTACK_INDEXES {
            assert!(
                idx < VISION_NUM_LAYERS,
                "DeepStack index {idx} >= {VISION_NUM_LAYERS}"
            );
        }
    }

    #[test]
    fn test_deepstack_indexes_sorted() {
        for window in DEEPSTACK_INDEXES.windows(2) {
            assert!(window[0] < window[1], "DeepStack indexes must be sorted");
        }
    }

    #[test]
    fn test_vision_rope_construction_4x4() {
        // 4x4 grid with merge_size 2 -> 2x2 merged blocks -> 16 patch tokens
        let rope =
            VisionRoPE::new(4, 4, &Device::Cpu).expect("should build VisionRoPE for 4x4 grid");
        let cos_dims = rope.cos.dims();
        assert_eq!(cos_dims[0], 1); // batch
        assert_eq!(cos_dims[1], 1); // heads
        assert_eq!(cos_dims[2], 16); // seq_len = 4*4
        assert_eq!(cos_dims[3], VISION_HEAD_DIM / 2); // rope_dim = 36
    }

    #[test]
    fn test_vision_rope_construction_2x2() {
        // Minimal: 2x2 grid = 4 tokens, 1 merged block
        let rope =
            VisionRoPE::new(2, 2, &Device::Cpu).expect("should build VisionRoPE for 2x2 grid");
        assert_eq!(rope.cos.dims(), &[1, 1, 4, 36]);
    }

    #[test]
    fn test_merge_patch_divisibility() {
        let merge_patch = PATCH_SIZE * SPATIAL_MERGE_SIZE;
        assert_eq!(merge_patch, 32);
        assert_eq!(256 % merge_patch, 0);
        assert_eq!(128 % merge_patch, 0);
        assert_eq!(64 % merge_patch, 0);
    }
}
