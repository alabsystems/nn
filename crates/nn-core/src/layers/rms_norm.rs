// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RMS normalization layer implementing [`Module`].
//!
//! RMS norm: `y = x / sqrt(mean(x^2) + eps) * weight`
//! Used by LLaMA, Mistral, and other modern transformer architectures.

use super::{validate_eps, CpuRoundTrip, Module};
use crate::dyn_tensor::trace::TraceOp;
use crate::dyn_tensor::{gpu_backend_dispatch, DynTensor};
use crate::error::{Result, TensorError};

/// Root Mean Square normalization layer.
///
/// RMS norm omits the mean-centering step of LayerNorm:
/// `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// More efficient than LayerNorm for transformer architectures (no mean
/// subtraction needed). Used by LLaMA, Mistral, Gemma.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: DynTensor,
    eps: f64,
}

impl RmsNorm {
    /// Create an RmsNorm layer.
    ///
    /// - `weight`: shape matching last dimension (gamma)
    /// - `eps`: small constant for numerical stability (must be finite and non-negative)
    pub fn new(weight: DynTensor, eps: f64) -> Result<Self> {
        validate_eps(eps, "RmsNorm")?;
        if weight.rank() != 1 {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: weight.rank(),
            });
        }
        Ok(Self { weight, eps })
    }

    /// Weight (gamma) tensor.
    #[must_use]
    pub fn weight(&self) -> &DynTensor {
        &self.weight
    }
}

impl Module for RmsNorm {
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
                Ok(TraceOp::RmsNorm {
                    eps: self.eps,
                    weight: self.weight.to_weight_ref()?,
                })
            },
            || {
                if x.device().is_gpu() {
                    if let Some(result) =
                        gpu_backend_dispatch(|b| b.rms_norm(x, &self.weight, self.eps))
                    {
                        result
                    } else {
                        Self::rms_norm_cpu(x, self)
                    }
                } else {
                    Self::rms_norm_cpu(x, self)
                }
            },
        )
    }
}

impl RmsNorm {
    /// CPU/fallback RmsNorm computation (decomposed ops with bf16/f16 round-trip).
    fn rms_norm_cpu(x: &DynTensor, this: &Self) -> Result<DynTensor> {
        let rank = x.rank();
        let rt = CpuRoundTrip::new(x);
        let x_cpu = rt.prepare(x)?;
        let w_cpu = rt.prepare_param(&this.weight)?;
        let last_dim = rank - 1;
        let x_sq = x_cpu.sqr()?;
        let mean_sq = x_sq.mean_keepdim(last_dim)?;
        let eps_t = DynTensor::full(vec![1; rank], this.eps, crate::DType::F32, &x_cpu.device())?;
        let rms = mean_sq.broadcast_add(&eps_t)?.sqrt()?;
        let normed = x_cpu.broadcast_div(&rms)?;
        let r = normed.broadcast_mul(&w_cpu)?;
        rt.restore(r)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    #[test]
    fn test_rms_norm_basic() {
        let weight = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-5).unwrap();
        let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4]);
        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // normed ≈ [0.3651, 0.7303, 1.0954, 1.4606]
        let flat = output.to_flat_vec::<f32>().unwrap();
        assert!((flat[0] - 0.3651).abs() < 0.01);
        assert!((flat[3] - 1.4606).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let weight = DynTensor::from_vec(vec![2.0, 0.5], &[2], &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-5).unwrap();
        let input = DynTensor::from_vec(vec![3.0, 4.0], &[1, 2], &Device::Cpu).unwrap();
        let output = norm.forward(&input).unwrap();
        // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        // normed = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        // output = normed * weight = [0.8485*2, 1.1314*0.5] ≈ [1.6971, 0.5657]
        let flat = output.to_flat_vec::<f32>().unwrap();
        assert!((flat[0] - 1.6971).abs() < 0.01);
        assert!((flat[1] - 0.5657).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm_rank0_error() {
        let weight = DynTensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(weight, 1e-5).unwrap();
        let scalar = DynTensor::from_vec(vec![1.0], &[], &Device::Cpu).unwrap();
        assert!(norm.forward(&scalar).is_err());
    }
}
