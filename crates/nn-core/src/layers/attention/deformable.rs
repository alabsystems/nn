// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deformable attention (Zhu et al., 2021 — Deformable DETR, arXiv:2010.04159).
//!
//! Each attention head attends to a small set of learnable sampling points around
//! a reference point, rather than all spatial positions. This gives O(K) complexity
//! per query (K = num sampling points) instead of O(H×W) for standard attention.
//!
//! Single-scale variant: one spatial feature map.
//! Multi-scale variant: feature pyramid with multiple resolution levels.
//!
//! Required by RT-DETR v2 and D-FINE layout detection architectures (#1074).

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, validate_divisible, validate_heads, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

#[path = "deformable_sampling.rs"]
mod deformable_sampling;
use deformable_sampling::safe_value;
pub use deformable_sampling::DeformableAttentionConfig;

/// Single-scale deformable attention layer.
///
/// For each query position, predicts K sampling offsets and attention weights,
/// then samples the value feature map at those offset positions using bilinear
/// interpolation and computes a weighted sum.
///
/// Input: value feature map `[B, C, H, W]` + query `[B, N_q, C]` +
///        reference points `[B, N_q, 2]` (normalized [0, 1] xy).
/// Output: `[B, N_q, C]`.
#[derive(Clone, Debug)]
pub struct DeformableAttention {
    /// Projects value features: [d_model] → [d_model].
    value_proj: Linear,
    /// Projects output: [d_model] → [d_model].
    output_proj: Linear,
    /// Predicts sampling offsets: [d_model] → [num_heads * num_levels * num_points * 2].
    sampling_offsets: Linear,
    /// Predicts attention weights: [d_model] → [num_heads * num_levels * num_points].
    attention_weights: Linear,
    config: DeformableAttentionConfig,
    head_dim: usize,
}

impl DeformableAttention {
    /// Create from pre-loaded weights.
    pub fn new(
        value_proj: Linear,
        output_proj: Linear,
        sampling_offsets: Linear,
        attention_weights: Linear,
        config: DeformableAttentionConfig,
    ) -> Result<Self> {
        if config.d_model == 0 {
            return Err(TensorError::InvalidShape(
                "DeformableAttention: d_model must be > 0".into(),
            ));
        }
        validate_heads(config.num_heads, "DeformableAttention")?;
        validate_divisible(
            config.d_model,
            config.num_heads,
            "d_model",
            "num_heads",
            "DeformableAttention",
        )?;
        if config.num_points == 0 {
            return Err(TensorError::InvalidShape(
                "DeformableAttention: num_points must be > 0".into(),
            ));
        }
        if config.num_levels == 0 {
            return Err(TensorError::InvalidShape(
                "DeformableAttention: num_levels must be > 0".into(),
            ));
        }
        let head_dim = config.d_model / config.num_heads;
        Ok(Self {
            value_proj,
            output_proj,
            sampling_offsets,
            attention_weights,
            config,
            head_dim,
        })
    }

    /// Load from a [`VarBuilder`] using PyTorch-style weight names.
    ///
    /// Loads `value_proj`, `output_proj`, `sampling_offsets`, `attention_weights`.
    pub fn load(vb: impl AsRef<VarBuilder>, config: DeformableAttentionConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let d = config.d_model;
        let offset_dim = config
            .num_heads
            .checked_mul(config.num_levels)
            .and_then(|v| v.checked_mul(config.num_points))
            .and_then(|v| v.checked_mul(2))
            .ok_or(TensorError::DimensionOverflow {
                dims: vec![config.num_heads, config.num_levels, config.num_points, 2],
            })?;
        let weight_dim = config
            .num_heads
            .checked_mul(config.num_levels)
            .and_then(|v| v.checked_mul(config.num_points))
            .ok_or(TensorError::DimensionOverflow {
                dims: vec![config.num_heads, config.num_levels, config.num_points],
            })?;

        let value_proj = Linear::load(vb.pp("value_proj"), d, d)?;
        let output_proj = Linear::load(vb.pp("output_proj"), d, d)?;
        let sampling_offsets = Linear::load(vb.pp("sampling_offsets"), d, offset_dim)?;
        let attention_weights = Linear::load(vb.pp("attention_weights"), d, weight_dim)?;

        Self::new(
            value_proj,
            output_proj,
            sampling_offsets,
            attention_weights,
            config,
        )
    }

    /// Forward pass for single-scale deformable attention.
    ///
    /// - `value`: spatial feature map `[B, C, H, W]`
    /// - `query`: query features `[B, N_q, C]`
    /// - `reference_points`: normalized reference positions `[B, N_q, 2]` in `[0, 1]` range,
    ///   where each point is `(x, y)`.
    ///
    /// Returns: `[B, N_q, C]`
    pub fn forward_single_scale(
        &self,
        value: &DynTensor,
        query: &DynTensor,
        reference_points: &DynTensor,
    ) -> Result<DynTensor> {
        self.forward_multi_scale(
            &[value],
            query,
            reference_points,
            &[(value.dim(2)?, value.dim(3)?)],
        )
    }

    /// Forward pass for multi-scale deformable attention.
    ///
    /// - `values`: feature maps at each level, each `[B, C, H_l, W_l]`
    /// - `query`: `[B, N_q, C]`
    /// - `reference_points`: `[B, N_q, 2]` normalized to `[0, 1]`
    /// - `spatial_shapes`: `[(H_l, W_l)]` for each level
    ///
    /// Returns: `[B, N_q, C]`
    pub fn forward_multi_scale(
        &self,
        values: &[&DynTensor],
        query: &DynTensor,
        reference_points: &DynTensor,
        spatial_shapes: &[(usize, usize)],
    ) -> Result<DynTensor> {
        let cfg = &self.config;
        if values.len() != cfg.num_levels {
            return Err(TensorError::DataLengthMismatch {
                expected: cfg.num_levels,
                actual: values.len(),
            });
        }
        if spatial_shapes.len() != cfg.num_levels {
            return Err(TensorError::DataLengthMismatch {
                expected: cfg.num_levels,
                actual: spatial_shapes.len(),
            });
        }

        let (batch, n_q, _d) = query.dims3()?;
        let (ref_b, ref_n, ref_2) = reference_points.dims3()?;
        if ref_b != batch || ref_n != n_q || ref_2 != 2 {
            return Err(TensorError::shape_mismatch(
                vec![batch, n_q, 2],
                vec![ref_b, ref_n, ref_2],
            ));
        }

        // Project values for each level: [B, C, H, W] → [B, H*W, C] → proj → reshape
        let mut projected_values: Vec<DynTensor> = Vec::with_capacity(cfg.num_levels);
        for (l, &val) in values.iter().enumerate() {
            let (vb, vc, vh, vw) = val.dims4()?;
            if vb != batch || vc != cfg.d_model {
                return Err(TensorError::shape_mismatch(
                    vec![batch, cfg.d_model],
                    vec![vb, vc],
                ));
            }
            let (exp_h, exp_w) = spatial_shapes[l];
            if (vh, vw) != (exp_h, exp_w) {
                return Err(TensorError::shape_mismatch(
                    vec![exp_h, exp_w],
                    vec![vh, vw],
                ));
            }
            // [B, C, H, W] → [B, C, H*W] → [B, H*W, C]
            let flat = val.reshape([vb, vc, vh * vw])?.transpose(1, 2)?;
            let proj = self.value_proj.forward(&flat)?;
            // [B, H*W, C] → [B, H*W, num_heads, head_dim]
            let proj = proj.reshape([batch, vh * vw, cfg.num_heads, self.head_dim])?;
            projected_values.push(proj);
        }

        // Predict sampling offsets from query: [B, N_q, C] → [B, N_q, H*L*K*2]
        let offsets_raw = self.sampling_offsets.forward(query)?;
        // → [B, N_q, num_heads, num_levels, num_points, 2]
        let offsets =
            offsets_raw.reshape([batch, n_q, cfg.num_heads, cfg.num_levels, cfg.num_points, 2])?;

        // Predict attention weights: [B, N_q, C] → [B, N_q, H*L*K]
        let attn_weights_raw = self.attention_weights.forward(query)?;
        // → [B, N_q, num_heads, num_levels * num_points]
        let attn_weights_flat = attn_weights_raw.reshape([
            batch,
            n_q,
            cfg.num_heads,
            cfg.num_levels * cfg.num_points,
        ])?;
        // Softmax over the sampling points dimension
        let attn_weights_sm = crate::layers::softmax(&attn_weights_flat, 3)?;
        // NaN defense: softmax is Tier 1 (division/exp), validate immediately.
        check_output_finite(&attn_weights_sm, "DeformableAttention softmax")?;
        // → [B, N_q, num_heads, num_levels, num_points]
        let attn_weights =
            attn_weights_sm.reshape([batch, n_q, cfg.num_heads, cfg.num_levels, cfg.num_points])?;

        // For each level and sampling point, compute sampling locations and sample
        // Reference points are in [0, 1], offsets are small learned displacements.
        let ref_points_cpu = reference_points.to_device(&crate::Device::Cpu)?;
        let ref_data = ref_points_cpu.to_f32_array()?;
        let offsets_cpu = offsets.to_device(&crate::Device::Cpu)?;
        let offsets_data = offsets_cpu.to_f32_array()?;
        let attn_w_cpu = attn_weights.to_device(&crate::Device::Cpu)?;
        let attn_w_data = attn_w_cpu.to_f32_array()?;

        // Pre-compute CPU-side flattened value tensors per level (avoid redundant
        // to_device + collect inside the batch loop).
        let mut level_val_flats: Vec<(Vec<f32>, usize)> = Vec::with_capacity(cfg.num_levels);
        for l in 0..cfg.num_levels {
            let val_cpu = projected_values[l].to_device(&crate::Device::Cpu)?;
            let val_data = val_cpu.to_f32_array()?;
            let val_flat: Vec<f32> = val_data.iter().copied().collect();
            let (h_l, w_l) = spatial_shapes[l];
            level_val_flats.push((val_flat, h_l * w_l));
        }

        // Accumulate output: [B, N_q, num_heads, head_dim]
        let out_size = batch
            .checked_mul(n_q)
            .and_then(|v| v.checked_mul(cfg.num_heads))
            .and_then(|v| v.checked_mul(self.head_dim))
            .ok_or(TensorError::DimensionOverflow {
                dims: vec![batch, n_q, cfg.num_heads, self.head_dim],
            })?;
        let mut output_data = vec![0.0f32; out_size];

        for b in 0..batch {
            for l in 0..cfg.num_levels {
                let (h_l, w_l) = spatial_shapes[l];
                let (ref val_flat, hw) = &level_val_flats[l];

                for q in 0..n_q {
                    let ref_x = ref_data[ndarray::IxDyn(&[b, q, 0])];
                    let ref_y = ref_data[ndarray::IxDyn(&[b, q, 1])];

                    for head_idx in 0..cfg.num_heads {
                        for k in 0..cfg.num_points {
                            let off_x = offsets_data[ndarray::IxDyn(&[b, q, head_idx, l, k, 0])];
                            let off_y = offsets_data[ndarray::IxDyn(&[b, q, head_idx, l, k, 1])];
                            let w = attn_w_data[ndarray::IxDyn(&[b, q, head_idx, l, k])];

                            // Sampling location in [0, 1] → pixel coordinates
                            let px = (ref_x + off_x) * (w_l as f32 - 1.0);
                            let py = (ref_y + off_y) * (h_l as f32 - 1.0);

                            // NaN/Inf defense: non-finite sampling coords would
                            // silently produce wrong results via saturating
                            // `NaN.floor() as i64 == 0`. Skip contribution
                            // (equivalent to zero-padding for out-of-bounds).
                            if !px.is_finite() || !py.is_finite() {
                                continue;
                            }

                            let x0 = px.floor() as i64;
                            let y0 = py.floor() as i64;
                            let x1 = x0 + 1;
                            let y1 = y0 + 1;
                            let wx = px - x0 as f32;
                            let wy = py - y0 as f32;

                            let out_base =
                                ((b * n_q + q) * cfg.num_heads + head_idx) * self.head_dim;
                            for d in 0..self.head_dim {
                                let v00 = safe_value(
                                    val_flat,
                                    b,
                                    y0,
                                    x0,
                                    head_idx,
                                    d,
                                    *hw,
                                    cfg.num_heads,
                                    self.head_dim,
                                    h_l,
                                    w_l,
                                );
                                let v01 = safe_value(
                                    val_flat,
                                    b,
                                    y0,
                                    x1,
                                    head_idx,
                                    d,
                                    *hw,
                                    cfg.num_heads,
                                    self.head_dim,
                                    h_l,
                                    w_l,
                                );
                                let v10 = safe_value(
                                    val_flat,
                                    b,
                                    y1,
                                    x0,
                                    head_idx,
                                    d,
                                    *hw,
                                    cfg.num_heads,
                                    self.head_dim,
                                    h_l,
                                    w_l,
                                );
                                let v11 = safe_value(
                                    val_flat,
                                    b,
                                    y1,
                                    x1,
                                    head_idx,
                                    d,
                                    *hw,
                                    cfg.num_heads,
                                    self.head_dim,
                                    h_l,
                                    w_l,
                                );
                                let sampled = v00 * (1.0 - wy) * (1.0 - wx)
                                    + v01 * (1.0 - wy) * wx
                                    + v10 * wy * (1.0 - wx)
                                    + v11 * wy * wx;
                                output_data[out_base + d] += w * sampled;
                            }
                        }
                    }
                }
            }
        }

        // Reshape output: [B, N_q, num_heads, head_dim] → [B, N_q, C]
        let output = DynTensor::from_vec(
            output_data,
            &[batch, n_q, cfg.num_heads, self.head_dim],
            &query.device(),
        )?;
        let output = output.reshape([batch, n_q, cfg.d_model])?;
        let result = self.output_proj.forward(&output)?;
        check_output_finite(&result, "DeformableAttention")?;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "deformable_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "deformable_error_tests.rs"]
mod error_tests;
