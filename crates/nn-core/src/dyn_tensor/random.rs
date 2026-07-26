// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Random tensor constructors for DynTensor (training feature).

use crate::dyn_tensor::DynTensor;
use crate::dyn_tensor::Shape;
use crate::tensor::checked_dim_product;
use crate::{Device, Result};

impl DynTensor {
    /// Create a tensor with random values drawn from U(lo, hi).
    ///
    /// Matches candle's `Tensor::rand(lo, hi, shape, device)`.
    ///
    /// ```no_run
    /// use nn_core::DynTensor;
    /// use nn_core::Device;
    /// # fn main() -> std::result::Result<(), nn_core::TensorError> {
    /// let t = DynTensor::rand(0.0, 1.0, &[2, 3], &Device::Cpu)?;
    /// let t = DynTensor::rand(0.0, 1.0, (2, 3), &Device::Cpu)?;  // tuple syntax
    /// // All values in [0.0, 1.0)
    /// # Ok(())
    /// # }
    /// ```
    pub fn rand(lo: f64, hi: f64, dims: impl Into<Shape>, device: &Device) -> Result<Self> {
        use rand::RngExt;

        let shape = dims.into();
        let dims = shape.dims();
        let lo_f32 = super::checked_f64_to_f32(lo, "rand() lo")?;
        let hi_f32 = super::checked_f64_to_f32(hi, "rand() hi")?;
        let numel = checked_dim_product(dims)?;
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                let u: f32 = rng.random();
                u * (hi_f32 - lo_f32) + lo_f32
            })
            .collect();
        let t = Self::from_vec(data, dims, &Device::Cpu)?;
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a tensor with random values drawn from N(mean, std).
    ///
    /// Matches candle's `Tensor::randn(mean, std, shape, device)`.
    ///
    /// ```no_run
    /// use nn_core::DynTensor;
    /// use nn_core::Device;
    /// # fn main() -> std::result::Result<(), nn_core::TensorError> {
    /// let t = DynTensor::randn(0.0, 1.0, &[2, 3], &Device::Cpu)?;
    /// let t = DynTensor::randn(0.0, 1.0, (2, 3), &Device::Cpu)?;  // tuple syntax
    /// assert_eq!(t.dims(), &[2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn randn(mean: f64, std: f64, dims: impl Into<Shape>, device: &Device) -> Result<Self> {
        use rand::RngExt;
        use rand_distr::StandardNormal;

        let shape = dims.into();
        let dims = shape.dims();
        let numel = checked_dim_product(dims)?;
        let mean_f32 = super::checked_f64_to_f32(mean, "randn() mean")?;
        let std_f32 = super::checked_f64_to_f32(std, "randn() std")?;
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                let z: f32 = rand::rng().sample(StandardNormal);
                z * std_f32 + mean_f32
            })
            .collect();
        let t = Self::from_vec(data, dims, &Device::Cpu)?;
        if device.is_gpu() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    /// Create a tensor with the same shape as `self`, filled with uniform [0, 1) values.
    ///
    /// Matches candle's `Tensor::rand_like` and `self.zeros_like()` pattern.
    pub fn rand_like(&self) -> Result<Self> {
        Self::rand(0.0, 1.0, self.dims(), &self.device())
    }

    /// Create a tensor with the same shape as `self`, filled with N(0, 1) values.
    pub fn randn_like(&self) -> Result<Self> {
        Self::randn(0.0, 1.0, self.dims(), &self.device())
    }
}

#[cfg(test)]
#[path = "random_tests.rs"]
mod tests;
