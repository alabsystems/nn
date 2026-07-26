// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-level ViT feature fusion (DeepStack).
//!
//! [`DeepStackFusion`] concatenates intermediate hidden states from multiple
//! ViT encoder layers along the feature dimension and projects them to a
//! target dimension via a single [`Linear`] layer.
//!
//! Introduced by Qwen3-VL (arXiv:2511.21631): early ViT layers capture
//! edges/textures, middle layers capture shapes/characters, final layers
//! capture semantics. Fusing all levels gives richer visual representations
//! for vision-language models and document understanding.
//!
//! ```text
//! features = [vit_block[i].output for i in layer_indices]
//! fused = linear_proj(cat(features, dim=-1))
//! ```

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Multi-level ViT feature fusion via concatenation + linear projection.
///
/// Takes intermediate hidden states from multiple ViT layers (produced by
/// [`VitEncoder::forward_deepstack`] or [`SigLip2VisionEncoder::forward_deepstack`]),
/// concatenates them along the last dimension, and projects to a target
/// hidden size.
///
/// # Example
///
/// ```text
/// // 3 layers of hidden_size=768 → concat to 2304 → project to 896
/// let fusion = DeepStackFusion::load(&vb, 768, 3, 896)?;
/// let intermediates = vit.forward_deepstack(&image, &[3, 7, 11])?;
/// let fused = fusion.forward_multi(&intermediates)?;
/// // fused: [B, seq_len, 896]
/// ```
#[derive(Clone, Debug)]
pub struct DeepStackFusion {
    projection: Linear,
    /// Number of intermediate layers expected.
    num_layers: usize,
    /// Hidden size of each input layer.
    input_hidden_size: usize,
    /// Output hidden size after projection.
    output_hidden_size: usize,
}

impl DeepStackFusion {
    /// Create from a pre-built [`Linear`] layer.
    ///
    /// - `input_hidden_size`: hidden dimension of each ViT layer output
    /// - `num_layers`: number of intermediate layers to fuse
    /// - `output_hidden_size`: target dimension after projection
    pub fn new(
        projection: Linear,
        input_hidden_size: usize,
        num_layers: usize,
        output_hidden_size: usize,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TensorError::InvalidShape(
                "DeepStackFusion: num_layers must be > 0".into(),
            ));
        }
        if input_hidden_size == 0 {
            return Err(TensorError::InvalidShape(
                "DeepStackFusion: input_hidden_size must be > 0".into(),
            ));
        }
        if output_hidden_size == 0 {
            return Err(TensorError::InvalidShape(
                "DeepStackFusion: output_hidden_size must be > 0".into(),
            ));
        }
        Ok(Self {
            projection,
            num_layers,
            input_hidden_size,
            output_hidden_size,
        })
    }

    /// Load from a [`VarBuilder`].
    ///
    /// Expects:
    /// - `projection.weight` `[output_hidden_size, num_layers * input_hidden_size]`
    /// - `projection.bias` `[output_hidden_size]` (optional)
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        input_hidden_size: usize,
        num_layers: usize,
        output_hidden_size: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        if num_layers == 0 || input_hidden_size == 0 || output_hidden_size == 0 {
            return Err(TensorError::InvalidShape(
                "DeepStackFusion::load: all dimensions must be > 0".into(),
            ));
        }
        let concat_dim = num_layers.checked_mul(input_hidden_size).ok_or_else(|| {
            TensorError::InvalidShape("DeepStackFusion: dimension overflow".into())
        })?;
        let proj_vb = vb.pp("projection");
        let w = proj_vb.get(&[output_hidden_size, concat_dim], "weight")?;
        let b = if proj_vb.contains_tensor("bias") {
            Some(proj_vb.get(&[output_hidden_size], "bias")?)
        } else {
            None
        };
        let projection = Linear::new(w, b)?;
        Self::new(
            projection,
            input_hidden_size,
            num_layers,
            output_hidden_size,
        )
    }

    /// Fuse multiple intermediate layer outputs into a single representation.
    ///
    /// Input: slice of `num_layers` tensors, each `[B, S, input_hidden_size]`.
    /// Output: `[B, S, output_hidden_size]`.
    ///
    /// Concatenates along the last dimension and applies the linear projection.
    pub fn forward_multi(&self, intermediates: &[DynTensor]) -> Result<DynTensor> {
        if intermediates.len() != self.num_layers {
            return Err(TensorError::InvalidShape(format!(
                "DeepStackFusion: expected {} layers, got {}",
                self.num_layers,
                intermediates.len()
            )));
        }

        // Validate shapes: all must be [B, S, input_hidden_size]
        if let Some(first) = intermediates.first() {
            let dims = first.dims();
            if dims.len() != 3 {
                return Err(TensorError::InvalidShape(format!(
                    "DeepStackFusion: expected 3D tensors, got {}D",
                    dims.len()
                )));
            }
            if dims[2] != self.input_hidden_size {
                return Err(TensorError::InvalidShape(format!(
                    "DeepStackFusion: expected hidden_size={}, got {}",
                    self.input_hidden_size, dims[2]
                )));
            }
            for t in intermediates.iter().skip(1) {
                let t_dims = t.dims();
                if t_dims != dims {
                    return Err(TensorError::shape_mismatch(dims.to_vec(), t_dims.to_vec()));
                }
            }
        }

        // Concatenate along last dimension: [B, S, num_layers * D]
        let refs: Vec<&DynTensor> = intermediates.iter().collect();
        let concatenated = DynTensor::cat(&refs, 2)?;

        // Project: [B, S, num_layers * D] -> [B, S, output_hidden_size]
        let fused = self.projection.forward(&concatenated)?;
        check_output_finite(&fused, "DeepStackFusion")?;
        Ok(fused)
    }

    /// Number of intermediate layers this fusion expects.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Input hidden size per layer.
    #[must_use]
    pub fn input_hidden_size(&self) -> usize {
        self.input_hidden_size
    }

    /// Output hidden size after projection.
    #[must_use]
    pub fn output_hidden_size(&self) -> usize {
        self.output_hidden_size
    }
}

/// [`Module`] impl treats the single input as a pre-concatenated tensor.
///
/// For the standard multi-layer API, use [`DeepStackFusion::forward_multi`].
impl Module for DeepStackFusion {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let fused = self.projection.forward(x)?;
        check_output_finite(&fused, "DeepStackFusion")?;
        Ok(fused)
    }
}

#[cfg(test)]
#[path = "deep_stack_tests.rs"]
mod tests;
