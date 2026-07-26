// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight initialization strategies for trainable variables.
//!
//! Provides standard initialization schemes matching PyTorch's `torch.nn.init`:
//! - Xavier (Glorot) uniform/normal — for layers with sigmoid/tanh activations
//! - Kaiming (He) uniform/normal — for layers with ReLU activations
//! - Uniform/normal — basic distributions with explicit bounds
//!
//! # Example
//!
//! ```no_run
//! use nn_autodiff::{Var, Init, Fan};
//! use nn_core::Device;
//!
//! # fn main() -> std::result::Result<(), nn_autodiff::AutodiffError> {
//! let w = Var::from_init(Init::Kaiming { fan: Fan::FanIn }, &[256, 512], &Device::Cpu)?;
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::tensor::checked_dim_product;
use nn_core::{DType, Device};

/// Fan direction for Kaiming/Xavier initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fan {
    /// Use input dimension (fan_in) — preserves forward-pass variance.
    FanIn,
    /// Use output dimension (fan_out) — preserves backward-pass variance.
    FanOut,
    /// Average of fan_in and fan_out — Xavier default.
    FanAvg,
}

/// Weight initialization strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Init {
    /// Constant fill.
    Const(f64),
    /// Uniform distribution U(lo, hi).
    Uniform { lo: f64, hi: f64 },
    /// Normal distribution N(mean, std).
    Normal { mean: f64, std: f64 },
    /// Xavier (Glorot) uniform: U(-a, a) where a = sqrt(6 / (fan_in + fan_out)).
    XavierUniform,
    /// Xavier (Glorot) normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
    XavierNormal,
    /// Kaiming (He) uniform: U(-a, a) where a = sqrt(6 / fan).
    /// For ReLU (gain=sqrt(2)), matching PyTorch's default kaiming_uniform.
    Kaiming { fan: Fan },
    /// Kaiming (He) normal: N(0, std) where std = sqrt(2 / fan).
    KaimingNormal { fan: Fan },
}

/// Compute fan_in and fan_out from weight dimensions.
///
/// Follows PyTorch's convention:
/// - 1D `[out]`: fan_in = 1, fan_out = out
/// - 2D `[out, in]`: fan_in = in, fan_out = out
/// - 3D+ `[out, in, *kernel]`: fan_in = in * prod(kernel), fan_out = out * prod(kernel)
fn compute_fans(dims: &[usize]) -> Result<(usize, usize)> {
    match dims.len() {
        0 => Ok((1, 1)),
        1 => Ok((1, dims[0])),
        2 => Ok((dims[1], dims[0])),
        _ => {
            let receptive: usize = checked_dim_product(&dims[2..])?;
            let fan_in = dims[1].checked_mul(receptive).ok_or_else(|| {
                nn_core::TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                }
            })?;
            let fan_out = dims[0].checked_mul(receptive).ok_or_else(|| {
                nn_core::TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                }
            })?;
            Ok((fan_in, fan_out))
        }
    }
}

impl Init {
    /// Create a `DynTensor` with this initialization strategy.
    pub fn to_tensor(self, dims: &[usize], dtype: DType, device: &Device) -> Result<DynTensor> {
        let (fan_in, fan_out) = compute_fans(dims)?;
        let t = match self {
            Self::Const(val) => DynTensor::full(dims, val, dtype, device)?,
            Self::Uniform { lo, hi } => DynTensor::rand(lo, hi, dims, device)?,
            Self::Normal { mean, std } => DynTensor::randn(mean, std, dims, device)?,
            Self::XavierUniform => {
                let a = (6.0 / (fan_in + fan_out) as f64).sqrt();
                DynTensor::rand(-a, a, dims, device)?
            }
            Self::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                DynTensor::randn(0.0, std, dims, device)?
            }
            Self::Kaiming { fan } => {
                let n = select_fan(fan, fan_in, fan_out);
                // bound = sqrt(3) * gain / sqrt(n) = sqrt(6/n) for ReLU (gain=sqrt(2))
                let a = (6.0 / n as f64).sqrt();
                DynTensor::rand(-a, a, dims, device)?
            }
            Self::KaimingNormal { fan } => {
                let n = select_fan(fan, fan_in, fan_out);
                // std = gain / sqrt(n) = sqrt(2/n) for ReLU
                let std = (2.0 / n as f64).sqrt();
                DynTensor::randn(0.0, std, dims, device)?
            }
        };
        Ok(t)
    }
}

fn select_fan(fan: Fan, fan_in: usize, fan_out: usize) -> usize {
    match fan {
        Fan::FanIn => fan_in.max(1),
        Fan::FanOut => fan_out.max(1),
        Fan::FanAvg => usize::midpoint(fan_in, fan_out).max(1),
    }
}

impl Var {
    /// Create a trainable variable with the given initialization strategy.
    pub fn from_init(init: Init, dims: &[usize], device: &Device) -> Result<Self> {
        let t = init.to_tensor(dims, DType::F32, device)?;
        Ok(Self::new(t))
    }

    /// Create a Var initialized with Xavier uniform (good default for most layers).
    pub fn xavier_uniform(dims: &[usize], device: &Device) -> Result<Self> {
        Self::from_init(Init::XavierUniform, dims, device)
    }

    /// Create a Var initialized with Kaiming uniform (good for ReLU layers).
    pub fn kaiming_uniform(dims: &[usize], device: &Device) -> Result<Self> {
        Self::from_init(Init::Kaiming { fan: Fan::FanIn }, dims, device)
    }

    /// Create a Var initialized with values from N(mean, std).
    pub fn randn(dims: &[usize], mean: f64, std: f64, device: &Device) -> Result<Self> {
        Self::from_init(Init::Normal { mean, std }, dims, device)
    }

    /// Create a Var initialized with values from U(lo, hi).
    pub fn rand(dims: &[usize], lo: f64, hi: f64, device: &Device) -> Result<Self> {
        Self::from_init(Init::Uniform { lo, hi }, dims, device)
    }
}

#[cfg(test)]
#[path = "var_init_tests.rs"]
mod tests;
