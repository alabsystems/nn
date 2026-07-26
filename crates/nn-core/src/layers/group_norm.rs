// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Group normalization layer for [`DynTensor`].
//!
//! Extracted from `layers.rs` to keep files under 450 lines.

use super::{validate_divisible, validate_eps, CpuRoundTrip, Module};
use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::{Result, TensorError};

/// Group normalization.
///
/// Matches candle-nn `GroupNorm`. Input is `[batch, channels, *spatial]`.
/// Divides channels into `num_groups` groups and normalizes each group.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    weight: DynTensor,
    bias: DynTensor,
    num_groups: usize,
    num_channels: usize,
    eps: f64,
}

impl GroupNorm {
    /// Create a GroupNorm layer.
    ///
    /// - `weight`: shape `[num_channels]` (gamma)
    /// - `bias`: shape `[num_channels]` (beta)
    /// - `num_channels` must be divisible by `num_groups`
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        weight: DynTensor,
        bias: DynTensor,
        eps: f64,
    ) -> Result<Self> {
        if num_groups == 0 {
            return Err(TensorError::InvalidShape(
                "GroupNorm num_groups must be > 0".into(),
            ));
        }
        validate_divisible(
            num_channels,
            num_groups,
            "num_channels",
            "num_groups",
            "GroupNorm",
        )?;
        validate_eps(eps, "GroupNorm")?;
        if weight.dims() != [num_channels] {
            return Err(TensorError::shape_mismatch(
                vec![num_channels],
                weight.dims().to_vec(),
            ));
        }
        if bias.dims() != [num_channels] {
            return Err(TensorError::shape_mismatch(
                vec![num_channels],
                bias.dims().to_vec(),
            ));
        }
        Ok(Self {
            weight,
            bias,
            num_groups,
            num_channels,
            eps,
        })
    }

    /// Weight (gamma) tensor.
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }

    /// Bias (beta) tensor.
    #[must_use]
    pub fn bias(&self) -> &DynTensor {
        &self.bias
    }
}

impl Module for GroupNorm {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let dims = x.dims();
        if dims.len() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: dims.len(),
            });
        }
        let batch = dims[0];
        let channels = dims[1];
        if channels != self.num_channels {
            return Err(TensorError::shape_mismatch(
                vec![batch, self.num_channels],
                vec![batch, channels],
            ));
        }
        let tracing = trace::is_tracing();
        // Suppress decomposed ops during tracing — only the composite
        // GroupNorm op should appear in the trace graph.
        let compute = || -> Result<DynTensor> {
            if x.device().is_gpu() {
                if let Some(result) = gpu_backend_dispatch(|b| {
                    b.group_norm(x, self.num_groups, &self.weight, &self.bias, self.eps)
                }) {
                    result
                } else {
                    Self::group_norm_cpu(x, self, dims)
                }
            } else {
                Self::group_norm_cpu(x, self, dims)
            }
        };
        let mut result = if tracing {
            trace::with_trace_suppressed(compute)?
        } else {
            compute()?
        };
        if tracing {
            let input_ids = DynTensor::trace_input_ids(&[x])?;
            if let Some(id) = trace::record_op(
                TraceOp::GroupNorm {
                    num_groups: self.num_groups,
                    eps: self.eps,
                    weight: self.weight.to_weight_ref()?,
                    bias: self.bias.to_weight_ref()?,
                },
                &input_ids,
                result.dims(),
                result.dtype(),
            ) {
                result.set_trace_id(id);
            }
        }
        Ok(result)
    }
}

impl GroupNorm {
    /// CPU/fallback GroupNorm computation (decomposed ops with bf16/f16 round-trip).
    fn group_norm_cpu(x: &DynTensor, this: &Self, dims: &[usize]) -> Result<DynTensor> {
        let batch = dims[0];
        let channels = dims[1];
        let rt = CpuRoundTrip::new(x);
        let x_work = rt.prepare(x)?;
        let channels_per_group = channels / this.num_groups;
        let spatial = crate::tensor::checked_dim_product(&dims[2..])?;
        let cpg_spatial = channels_per_group.checked_mul(spatial).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: dims.to_vec(),
            }
        })?;
        let x_reshaped = x_work.reshape([batch, this.num_groups, cpg_spatial])?;
        let mean = x_reshaped.mean_keepdim(2)?;
        let centered = x_reshaped.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?;
        let eps_t = DynTensor::full(
            &[1, 1, 1],
            this.eps,
            crate::DType::F32,
            &x_reshaped.device(),
        )?;
        let std_inv = var.broadcast_add(&eps_t)?.sqrt()?.recip()?;
        let normed = centered.broadcast_mul(&std_inv)?;
        let normed = normed.reshape(dims)?;
        let mut wb_shape = vec![1usize; dims.len()];
        wb_shape[1] = channels;
        let w_cpu = rt.prepare_param(&this.weight)?;
        let b_cpu = rt.prepare_param(&this.bias)?;
        let w = w_cpu.reshape(&wb_shape)?;
        let b = b_cpu.reshape(&wb_shape)?;
        let r = normed.broadcast_mul(&w)?.broadcast_add(&b)?;
        rt.restore(r)
    }
}
