#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for trainable layer implementations.

use super::*;
use crate::grad::backward;
use nn_core::{DType, Device, DynTensor};

#[test]
fn test_trainable_linear_forward_shape() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 8], &[2, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
}

#[test]
fn test_trainable_linear_no_bias() {
    let layer = TrainableLinear::new(4, 3, false).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 8], &[2, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    assert!(layer.bias().is_none());
}

#[test]
fn test_trainable_linear_vars_count() {
    let with_bias = TrainableLinear::new(4, 3, true).unwrap();
    assert_eq!(with_bias.vars().len(), 2);

    let no_bias = TrainableLinear::new(4, 3, false).unwrap();
    assert_eq!(no_bias.vars().len(), 1);
}

#[test]
fn test_trainable_linear_backward_produces_gradients() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    let x = DynTensor::from_vec(
        (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();

    // Reduce to scalar loss: sum of all outputs
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    // Weight should have gradient
    let w_grad = grads.get(layer.weight()).expect("weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());

    // Bias should have gradient
    let b_grad = grads.get(layer.bias().unwrap()).expect("bias gradient");
    let max_abs = b_grad
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "bias gradient should be non-zero");
}

#[test]
fn test_trainable_linear_from_tensors() {
    let w = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![0.1, 0.2], &[2], &Device::Cpu).unwrap();
    let layer = TrainableLinear::from_tensors(w, Some(b));

    let x = DynTensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    // y = [1,0,0] @ [[1,4],[2,5],[3,6]]^T is wrong — weight is [2,3]
    // y = [1,0,0] @ [[1,2,3],[4,5,6]]^T = [1,0,0] @ [[1,4],[2,5],[3,6]] = [1, 4]
    // + bias [0.1, 0.2] = [1.1, 4.2]
    let vals = y.tensor().to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 1.1).abs() < 1e-5,
        "expected 1.1, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 4.2).abs() < 1e-5,
        "expected 4.2, got {}",
        vals[1]
    );
}

#[test]
fn test_trainable_module_trait_object() {
    let layer = TrainableLinear::new(4, 3, true).unwrap();
    // Verify it works through trait reference
    let module: &dyn TrainableModule = &layer;
    assert_eq!(module.vars().len(), 2);

    let x = DynTensor::from_vec(vec![1.0; 4], &[1, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = module.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[1, 3]);
}

#[test]
fn test_trainable_linear_from_vars() {
    let w = Var::new(DynTensor::zeros(&[3, 4], DType::F32, &Device::Cpu).unwrap());
    let b = Var::new(DynTensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap());
    let layer = TrainableLinear::from_vars(w, Some(b));
    assert_eq!(layer.vars().len(), 2);
    assert_eq!(layer.weight().dims().unwrap(), vec![3, 4]);
}

// -- TrainableConv1d tests --

#[test]
fn test_trainable_conv1d_forward_shape() {
    // Weight: [out_channels=4, in_channels=2, kernel_size=3]
    let w = DynTensor::from_vec(vec![0.1; 4 * 2 * 3], &[4, 2, 3], &Device::Cpu).unwrap();
    let layer = TrainableConv1d::from_tensors(w, None, 1, 1, 1, 1);

    // Input: [batch=1, channels=2, length=8]
    let x = DynTensor::from_vec(vec![1.0; 16], &[1, 2, 8], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    // Output: [1, 4, 8] (padding=1, stride=1, dilation=1 preserves length for kernel_size=3)
    assert_eq!(y.dims(), &[1, 4, 8]);
}

#[test]
fn test_trainable_conv1d_with_bias() {
    let w = DynTensor::from_vec(vec![0.1; 6], &[2, 1, 3], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![0.5, -0.5], &[2], &Device::Cpu).unwrap();
    let layer = TrainableConv1d::from_tensors(w, Some(b), 1, 1, 1, 1);

    let x = DynTensor::from_vec(vec![1.0; 6], &[1, 1, 6], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[1, 2, 6]);
    assert_eq!(layer.vars().len(), 2);
}

#[test]
fn test_trainable_conv1d_backward_produces_gradients() {
    let w = DynTensor::from_vec(
        (0..6).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[2, 1, 3],
        &Device::Cpu,
    )
    .unwrap();
    let layer = TrainableConv1d::from_tensors(w, None, 1, 1, 1, 1);

    let x = DynTensor::from_vec(
        (0..4).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[1, 1, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();

    // Scalar loss.
    let loss = y
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let w_grad = grads.get(layer.weight()).expect("conv weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());

    let max_abs = w_grad
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "conv weight gradient should be non-zero");
}

#[test]
fn test_trainable_conv1d_trait_object() {
    let w = DynTensor::from_vec(vec![0.1; 6], &[2, 1, 3], &Device::Cpu).unwrap();
    let layer = TrainableConv1d::from_tensors(w, None, 1, 1, 1, 1);

    let module: &dyn TrainableModule = &layer;
    assert_eq!(module.vars().len(), 1);

    let x = DynTensor::from_vec(vec![1.0; 5], &[1, 1, 5], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = module.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[1, 2, 5]);
}
