// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dropout layer (inference-mode no-op).
//!
//! At inference time, Dropout is identity. This type exists for API
//! compatibility with candle-nn, enabling find-and-replace migration
//! from `candle_nn::Dropout` to `nn::layers::Dropout`.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::Result;

/// Dropout layer — identity at inference time.
///
/// Accepts a drop probability for API compatibility but ignores it
/// since nn targets inference only.
#[derive(Debug, Clone, Copy)]
pub struct Dropout {
    _drop_p: f32,
}

impl Dropout {
    /// Create a new Dropout layer with the given drop probability.
    ///
    /// The probability is stored but unused at inference time.
    pub fn new(drop_p: f32) -> Self {
        // drop_p is stored for API compatibility but unused (inference-only).
        // No validation needed since the value has no effect.
        Self { _drop_p: drop_p }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        trace::traced_forward(&[x], || Ok(TraceOp::Dropout), || Ok(x.clone()))
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_dropout_new() {
        let d = Dropout::new(0.1);
        // Should construct without error; drop_p is stored but unused.
        assert_eq!(size_of_val(&d), size_of::<f32>());
    }

    #[test]
    fn test_dropout_forward_identity() {
        let d = Dropout::new(0.5);
        let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let y = d.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3]);
        let vals = y.to_flat_vec::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dropout_boundary_values() {
        // Edge cases: 0.0 and 1.0 are valid
        let _ = Dropout::new(0.0);
        let _ = Dropout::new(1.0);
        let _ = Dropout::new(0.5);
    }

    #[test]
    fn test_dropout_forward_preserves_shape() {
        let d = Dropout::new(0.3);
        let x =
            DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
        let y = d.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3]);
    }
}
