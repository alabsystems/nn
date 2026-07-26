// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Core neural network layer implementations for [`DynTensor`].
//!
//! Contains [`Linear`] and [`LayerNorm`] —
//! extracted from `nn.rs` to stay within the 500-line limit.
//! [`GroupNorm`](super::GroupNorm) lives in `group_norm.rs`.
//! [`Embedding`](super::Embedding) lives in `embedding.rs`.
//! BatchNorm lives in `nn_batch_norm.rs`.

use super::{validate_eps, Module};
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::{Result, TensorError};

// -- Linear -------------------------------------------------------------------

/// Fully-connected linear layer: `y = x @ weight^T + bias`.
///
/// Matches candle-nn `Linear`. Weight is stored in PyTorch convention
/// `[out_features, in_features]`. On GPU, a pre-transposed copy
/// `[in_features, out_features]` is cached at construction to avoid per-call
/// GPU transpose dispatch (#1324). On CPU, the transpose is computed on-the-fly
/// to save memory — ~48 MB for Kokoro-82M (#3079).
#[derive(Debug, Clone)]
pub struct Linear {
    /// Original weight in `[out_features, in_features]` layout (PyTorch convention).
    weight: DynTensor,
    /// Pre-transposed weight `[in_features, out_features]` used in `forward()`.
    /// `None` when weight is on CPU — computed on-the-fly to avoid memory duplication.
    /// Present when weight is on GPU — eliminates per-call transpose dispatch (#1324).
    weight_t: Option<DynTensor>,
    bias: Option<DynTensor>,
}

impl Linear {
    /// Create a Linear layer from weight and optional bias tensors.
    ///
    /// - `weight`: shape `[out_features, in_features]` (must be 2D)
    /// - `bias`: shape `[out_features]` (optional)
    ///
    /// On GPU: the weight is transposed once at construction and cached for use
    /// in `forward()`. This eliminates a per-call GPU transpose dispatch that
    /// otherwise adds 96+ unnecessary kernel launches per transformer forward
    /// pass (#1324).
    ///
    /// On CPU: the transpose is computed on-the-fly in `forward()` to avoid
    /// storing a duplicate weight copy. For Kokoro-82M this saves ~48 MB of RSS
    /// (#3079). CPU forward calls are only used during tracing (once per shape).
    ///
    /// Returns an error if `weight` is not 2D.
    pub fn new(weight: DynTensor, bias: Option<DynTensor>) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            });
        }
        if let Some(ref b) = bias {
            let expected_len = weight.dims()[0];
            if b.dims() != [expected_len] {
                return Err(TensorError::shape_mismatch(
                    vec![expected_len],
                    b.dims().to_vec(),
                ));
            }
        }
        // Only pre-transpose on GPU to avoid per-call dispatch overhead (#1324).
        // On CPU, transpose on-the-fly in forward() to save ~48 MB RSS (#3079).
        let weight_t = if weight.device().is_gpu() {
            Some(weight.transpose(0, 1)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            weight_t,
            bias,
        })
    }

    /// Weight tensor in `[out_features, in_features]` layout (PyTorch convention).
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }

    /// Bias tensor reference (if present).
    #[must_use]
    pub fn bias(&self) -> Option<&DynTensor> {
        self.bias.as_ref()
    }

    /// Number of output features (first dimension of weight).
    #[must_use]
    pub fn out_features(&self) -> usize {
        // weight shape is [out, in], so dim 0 is out_features.
        self.weight.dims()[0]
    }

    /// Number of input features (second dimension of weight).
    #[must_use]
    pub fn in_features(&self) -> usize {
        // weight shape is [out, in], so dim 1 is in_features.
        self.weight.dims()[1]
    }
}

impl Module for Linear {
    /// Forward: `x @ weight^T + bias`.
    ///
    /// Input `x` can be 2D `[batch, in_features]` or higher-rank with matmul
    /// on the last two dims. On GPU, uses the pre-transposed weight cached at
    /// construction — no per-call transpose dispatch. On CPU, computes the
    /// transpose on-the-fly (used only during tracing, once per shape).
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        super::traced_forward(
            &[x],
            || {
                Ok(TraceOp::Linear {
                    weight: self.weight.to_weight_ref()?,
                    bias: self
                        .bias
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?,
                })
            },
            || {
                let wt = match &self.weight_t {
                    Some(wt) => wt.clone(),
                    None => self.weight.transpose(0, 1)?,
                };
                let y = x.matmul(&wt)?;
                match &self.bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            },
        )
    }
}

// -- LayerNorm ----------------------------------------------------------------

/// Layer normalization over the last dimension.
///
/// Matches candle-nn `LayerNorm`. Normalizes over the last `normalized_shape`
/// dims, then applies learned affine: `y = (x - mean) / sqrt(var + eps) * weight + bias`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: DynTensor,
    bias: DynTensor,
    eps: f64,
}

impl LayerNorm {
    /// Create a LayerNorm from weight (gamma) and bias (beta) tensors.
    ///
    /// Both should have shape matching the last dimension(s) to normalize over.
    /// Returns an error if `eps` is not finite or is negative.
    pub fn new(weight: DynTensor, bias: DynTensor, eps: f64) -> Result<Self> {
        validate_eps(eps, "LayerNorm")?;
        if weight.dims() != bias.dims() {
            return Err(TensorError::shape_mismatch(
                weight.dims().to_vec(),
                bias.dims().to_vec(),
            ));
        }
        Ok(Self { weight, bias, eps })
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

impl LayerNorm {
    /// CPU F64 accumulation path for LayerNorm. (#2691)
    ///
    /// Computes mean/var/normalize in f64 to match PyTorch's internal precision,
    /// then applies learned affine (weight, bias) in f64 before casting back to f32.
    /// Follows the InstanceNorm F64 pattern (#2688).
    fn forward_f64(&self, x: &DynTensor) -> Result<DynTensor> {
        use ndarray::{Axis, IxDyn};

        let dims = x.dims();
        let rank = dims.len();
        let input_dtype = x.dtype();
        let input_device = x.device();

        // Move to CPU if needed and extract f32 array.
        let x_f32 = if input_device.is_gpu() {
            x.to_device(&crate::Device::Cpu)?
                .contiguous()?
                .to_f32_array()?
        } else {
            x.contiguous()?.to_f32_array()?
        };
        let x_f64 = x_f32.mapv(f64::from);

        // Mean and variance over last dim in f64.
        let last_axis = Axis(rank - 1);
        let mean = x_f64
            .mean_axis(last_axis)
            .ok_or_else(|| TensorError::InvalidShape("empty last dim".into()))?
            .insert_axis(last_axis);
        let centered = &x_f64 - &mean;
        let var = centered
            .mapv(|v| v * v)
            .mean_axis(last_axis)
            .ok_or_else(|| TensorError::InvalidShape("empty last dim".into()))?
            .insert_axis(last_axis);
        let std_inv = (&var + self.eps).mapv(|v| 1.0 / v.sqrt());
        let normed = &centered * &std_inv;

        // Apply learned affine: weight * normed + bias (in f64).
        // Transfer weights to CPU if on GPU — to_f32_array() requires CPU data.
        let w_cpu = if self.weight.device().is_gpu() {
            self.weight.to_device(&crate::Device::Cpu)?
        } else {
            self.weight.clone()
        };
        let w_f32 = w_cpu.contiguous()?.to_f32_array()?;
        let w_f64 = w_f32.mapv(f64::from);
        let b_cpu = if self.bias.device().is_gpu() {
            self.bias.to_device(&crate::Device::Cpu)?
        } else {
            self.bias.clone()
        };
        let b_f32 = b_cpu.contiguous()?.to_f32_array()?;
        let b_f64 = b_f32.mapv(f64::from);
        let result_f64 = &normed * &w_f64 + &b_f64;

        // Cast back to f32 and restore original shape/dtype/device.
        let result_f32 = result_f64.mapv(|v| v as f32);
        let result_arr = result_f32
            .into_shape_with_order(IxDyn(dims))
            .map_err(|e| TensorError::InvalidShape(format!("LayerNorm reshape: {e}")))?;
        let mut result = DynTensor::from_cpu_f32(result_arr)?;
        if input_dtype != crate::DType::F32 {
            result = result.to_dtype(input_dtype)?;
        }
        if input_device.is_gpu() {
            result.to_device(&input_device)
        } else {
            Ok(result)
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let rank = x.rank();
        if rank == 0 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        super::traced_forward(
            &[x],
            || {
                Ok(TraceOp::LayerNorm {
                    eps: self.eps,
                    weight: self.weight.to_weight_ref()?,
                    bias: self.bias.to_weight_ref()?,
                })
            },
            || {
                if x.device().is_gpu() {
                    if let Some(result) = gpu_backend_dispatch(|b| {
                        b.layer_norm(x, &self.weight, &self.bias, self.eps)
                    }) {
                        return result;
                    }
                }
                // CPU path (and GPU fallback): F64 accumulation (#2691).
                self.forward_f64(x)
            },
        )
    }
}
