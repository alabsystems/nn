// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sequential container implementing [`Module`].

use super::Module;
use crate::dyn_tensor::DynTensor;
use crate::error::Result;

/// Sequential container: applies a list of modules in order.
///
/// Matches candle-nn `Sequential`. Use `add()` for struct modules and
/// `add_fn()` for closure modules.
///
/// # Example
///
/// ```no_run
/// use nn_core::layers::{Sequential, Activation, Linear};
/// use nn_core::{DType, Device, DynTensor};
///
/// let mut seq = Sequential::new();
/// seq.add_fn(|x| x.relu());
/// seq.add_fn(|x| x.neg());
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module + Send + Sync>>,
}

impl Sequential {
    /// Create an empty sequential container.
    #[must_use]
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Append a module layer.
    pub fn add<M: Module + Send + Sync + 'static>(&mut self, module: M) {
        self.layers.push(Box::new(module));
    }

    /// Append a closure layer.
    pub fn add_fn<F>(&mut self, f: F)
    where
        F: Fn(&DynTensor) -> Result<DynTensor> + Send + Sync + 'static,
    {
        self.layers.push(Box::new(f));
    }

    /// Number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the container is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_sequential_empty() {
        let seq = Sequential::new();
        assert!(seq.is_empty());
        let input = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
        let output = seq.forward(&input).unwrap();
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_sequential_add_fn() {
        let mut seq = Sequential::new();
        seq.add_fn(DynTensor::relu);
        seq.add_fn(DynTensor::neg);
        assert_eq!(seq.len(), 2);
        let input = DynTensor::from_vec(vec![-1.0, 2.0, -3.0], &[3], &Device::Cpu).unwrap();
        let output = seq.forward(&input).unwrap();
        // relu(-1,2,-3) = (0,2,0), neg = (0,-2,0)
        assert_eq!(output.to_flat_vec::<f32>().unwrap(), vec![0.0, -2.0, 0.0]);
    }

    #[test]
    fn test_sequential_with_activation() {
        use super::super::Activation;
        let mut seq = Sequential::new();
        seq.add(Activation::Relu);
        seq.add(Activation::Sigmoid);
        let input = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &Device::Cpu).unwrap();
        let output = seq.forward(&input).unwrap();
        let flat = output.to_flat_vec::<f32>().unwrap();
        // relu(-1,0,1) = (0,0,1), sigmoid(0,0,1) = (0.5, 0.5, ~0.731)
        assert!((flat[0] - 0.5).abs() < 1e-4);
        assert!((flat[1] - 0.5).abs() < 1e-4);
        assert!((flat[2] - 0.7311).abs() < 0.001);
    }
}
