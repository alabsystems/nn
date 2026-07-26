// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended module traits and blanket impls for [`Module`].
//!
//! - [`ModuleT`]: forward with train/eval mode distinction
//! - Blanket impl: `Option<&M>` implements `Module` (identity when None)
//!
//! The blanket `impl Module for Fn(...)` lives in `nn.rs` alongside the
//! `Module` trait definition.

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::Module;

/// Option<&M> acts as identity (None) or delegates (Some).
impl<M: Module> Module for Option<&M> {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        match self {
            None => Ok(x.clone()),
            Some(m) => m.forward(x),
        }
    }
}

/// Module with training/eval mode distinction.
///
/// Only needed for layers with train-dependent behavior (Dropout, BatchNorm).
/// dvoice is inference-only, so this is lower priority.
pub trait ModuleT {
    fn forward_t(&self, x: &DynTensor, train: bool) -> Result<DynTensor>;
}

/// Every Module is automatically a ModuleT (ignores train flag).
impl<M: Module> ModuleT for M {
    fn forward_t(&self, x: &DynTensor, _train: bool) -> Result<DynTensor> {
        self.forward(x)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    #[test]
    fn test_closure_module() {
        let layer = |x: &DynTensor| x.relu();
        let input = DynTensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[4], &Device::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(
            output.to_flat_vec::<f32>().unwrap(),
            vec![0.0, 2.0, 0.0, 4.0]
        );
    }

    #[test]
    fn test_option_none_is_identity() {
        let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let none_module: Option<&fn(&DynTensor) -> Result<DynTensor>> = None;
        let output = none_module.forward(&input).unwrap();
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_convenience() {
        let layer = |x: &DynTensor| x.neg();
        let input = DynTensor::from_vec(vec![1.0, -2.0], &[2], &Device::Cpu).unwrap();
        let output = input.apply(&layer).unwrap();
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_module_t_blanket() {
        let layer = |x: &DynTensor| x.relu();
        let input = DynTensor::from_vec(vec![-1.0, 2.0], &[2], &Device::Cpu).unwrap();
        let output = layer.forward_t(&input, false).unwrap();
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![0.0, 2.0]);
    }

    struct DoubleLayer;
    impl Module for DoubleLayer {
        fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
            x.mul_scalar(2.0)
        }
    }

    #[test]
    fn test_struct_module() {
        let layer = DoubleLayer;
        let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_option_some_delegates() {
        let layer = DoubleLayer;
        let input = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
        let some_module = Some(&layer);
        let output = some_module.forward(&input).unwrap();
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn test_zeros_roundtrip() {
        let input = DynTensor::zeros(&[2, 3], DType::F32, &Device::Cpu).unwrap();
        let layer = |x: &DynTensor| x.add_scalar(1.0);
        let output = input.apply(&layer).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        assert!(flat.iter().all(|&v| v == 1.0));
    }
}
