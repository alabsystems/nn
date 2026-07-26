// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! StreamingLLM attention sinks for gpt-oss.
//!
//! Implements per-layer learnable attention sink vectors (arXiv:2309.17453).
//! Each layer stores a `[head_dim]` sink vector loaded from
//! `model.layers.{i}.self_attn.sinks`. During attention scoring, the L2 norm
//! of the sink vector is added as an additive bias to the attention score at
//! position 0, anchoring attention to the first token during sliding window
//! inference.
//!
//! This prevents the attention score "cliff" that occurs when the initial
//! token exits the sliding window, maintaining stable generation quality
//! for long sequences.

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

use crate::config::GptOssConfig;

/// Per-layer attention sinks for StreamingLLM-style inference.
///
/// Stores one `[head_dim]` vector per layer. The L2 norm of each sink vector
/// is applied as an additive bias to the attention score at position 0.
pub(crate) struct AttentionSinks {
    /// Sink vectors, one per layer. Each has shape `[head_dim]`.
    sinks: Vec<DynTensor>,
}

impl AttentionSinks {
    /// Load attention sinks from VarBuilder.
    ///
    /// Reads `model.layers.{i}.self_attn.sinks` for each layer.
    pub(crate) fn load(vb: impl AsRef<VarBuilder>, cfg: &GptOssConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let model_vb = vb.pp("model");
        let mut sinks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp(format!("layers.{i}"));
            let sink = layer_vb.get(&[cfg.head_dim], "self_attn.sinks")?;
            sinks.push(sink);
        }
        Ok(Self { sinks })
    }

    /// Apply attention sink bias to attention scores for a given layer.
    ///
    /// Computes the L2 norm of the layer's sink vector and adds it as an
    /// additive bias to the attention score at position 0 (the first key
    /// position). This anchors attention to the initial token, preventing
    /// score collapse during sliding window inference.
    ///
    /// # Arguments
    /// - `attn_scores`: Attention scores `[batch, heads, seq_q, seq_kv]`
    /// - `layer_idx`: Which layer's sink to apply
    ///
    /// # Returns
    /// Modified attention scores with sink bias added at `[:, :, :, 0]`.
    pub(crate) fn apply(&self, attn_scores: &DynTensor, layer_idx: usize) -> Result<DynTensor> {
        if layer_idx >= self.sinks.len() {
            return Err(crate::GptOssError::InvalidInput {
                reason: format!(
                    "layer_idx ({layer_idx}) >= num_layers ({})",
                    self.sinks.len()
                ),
            }
            .into());
        }

        let sink = &self.sinks[layer_idx];

        // Compute L2 norm of sink vector: sqrt(sum(sink^2))
        let sink_sq = sink.broadcast_mul(sink)?;
        let norm_sq = sink_sq.sum_all()?;
        let norm = norm_sq.sqrt()?;
        let bias_val = norm.to_scalar::<f32>()?;

        if bias_val.abs() < 1e-10 {
            // Near-zero sink: no bias to apply
            return Ok(attn_scores.clone());
        }

        // Build additive bias tensor: zeros everywhere, bias_val at position 0
        // along the last dimension (seq_kv).
        let dims = attn_scores.dims();
        let seq_kv = dims[dims.len() - 1];

        // Create a 1D bias [seq_kv] with bias_val at index 0
        let mut bias_data = vec![0.0f32; seq_kv];
        bias_data[0] = bias_val;
        let bias = DynTensor::from_vec(bias_data, &[seq_kv], &attn_scores.device())?;

        // broadcast_add: bias [seq_kv] broadcasts to [batch, heads, seq_q, seq_kv]
        attn_scores.broadcast_add(&bias)
    }

    /// Number of layers with sinks.
    #[must_use]
    pub(crate) fn num_layers(&self) -> usize {
        self.sinks.len()
    }

    /// Access a specific layer's sink vector.
    pub(crate) fn sink(&self, layer_idx: usize) -> Option<&DynTensor> {
        self.sinks.get(layer_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::Device;

    #[test]
    fn test_apply_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 4;

        // Create a single-layer sink manually
        let sink_data = vec![1.0f32, 0.0, 0.0, 0.0];
        let sink = DynTensor::from_vec(sink_data, &[head_dim], &device)?;
        let sinks = AttentionSinks { sinks: vec![sink] };

        // attn_scores: [batch=1, heads=2, seq_q=3, seq_kv=5]
        let scores = DynTensor::zeros(&[1, 2, 3, 5], nn_core::DType::F32, &device)?;
        let out = sinks.apply(&scores, 0)?;
        assert_eq!(out.dims(), &[1, 2, 3, 5]);
        Ok(())
    }

    #[test]
    fn test_bias_at_position_zero() -> Result<()> {
        let device = Device::Cpu;

        // Sink vector [3, 4] -> L2 norm = 5.0
        let sink = DynTensor::from_vec(vec![3.0f32, 4.0], &[2], &device)?;
        let sinks = AttentionSinks { sinks: vec![sink] };

        // Start with zero scores [1, 1, 1, 3]
        let scores = DynTensor::zeros(&[1, 1, 1, 3], nn_core::DType::F32, &device)?;
        let out = sinks.apply(&scores, 0)?;
        let data = out.to_flat_vec::<f32>()?;

        // Position 0 should have bias=5.0, others stay 0.0
        assert!(
            (data[0] - 5.0).abs() < 1e-5,
            "pos 0 should be ~5.0, got {}",
            data[0]
        );
        assert!(
            data[1].abs() < 1e-5,
            "pos 1 should be ~0.0, got {}",
            data[1]
        );
        assert!(
            data[2].abs() < 1e-5,
            "pos 2 should be ~0.0, got {}",
            data[2]
        );
        Ok(())
    }

    #[test]
    fn test_out_of_bounds_layer() -> Result<()> {
        let device = Device::Cpu;
        let sink = DynTensor::ones(&[4], nn_core::DType::F32, &device)?;
        let sinks = AttentionSinks { sinks: vec![sink] };

        let scores = DynTensor::zeros(&[1, 1, 1, 3], nn_core::DType::F32, &device)?;
        // Layer 1 is out of bounds (only layer 0 exists)
        let result = sinks.apply(&scores, 1);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_num_layers() -> Result<()> {
        let device = Device::Cpu;
        let sinks_vec: Vec<DynTensor> = (0..3)
            .map(|_| DynTensor::ones(&[4], nn_core::DType::F32, &device).unwrap())
            .collect();
        let sinks = AttentionSinks { sinks: sinks_vec };
        assert_eq!(sinks.num_layers(), 3);
        Ok(())
    }

    #[test]
    fn test_sink_vector_access() -> Result<()> {
        let device = Device::Cpu;
        let sink = DynTensor::ones(&[8], nn_core::DType::F32, &device)?;
        let sinks = AttentionSinks { sinks: vec![sink] };
        assert!(sinks.sink(0).is_some());
        assert!(sinks.sink(1).is_none());
        Ok(())
    }
}
