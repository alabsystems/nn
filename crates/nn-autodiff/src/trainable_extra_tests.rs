#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for trainable extra layer implementations (Embedding, LayerNorm, Conv2d).

use super::*;
use crate::grad::backward;
use nn_core::{Device, DynTensor};

// -- TrainableEmbedding tests --

#[test]
fn test_trainable_embedding_forward_shape() {
    let layer = TrainableEmbedding::new(10, 4).unwrap();
    // Indices: [3] — look up 3 embeddings
    let indices = DynTensor::from_vec_u32(vec![0, 3, 7], &[3], &Device::Cpu).unwrap();
    let idx_tracked = Arc::new(TrackedTensor::from_tensor(indices));
    let y = layer.forward_indices(&idx_tracked).unwrap();
    assert_eq!(y.dims(), &[3, 4]);
}

#[test]
fn test_trainable_embedding_2d_indices() {
    let layer = TrainableEmbedding::new(10, 4).unwrap();
    // Indices: [2, 3] — batch of 2, sequence length 3
    let indices = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &Device::Cpu).unwrap();
    let idx_tracked = Arc::new(TrackedTensor::from_tensor(indices));
    let y = layer.forward_indices(&idx_tracked).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_trainable_embedding_backward_produces_gradients() {
    let w = DynTensor::from_vec(
        (0..20).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[5, 4],
        &Device::Cpu,
    )
    .unwrap();
    let layer = TrainableEmbedding::from_tensor(w);

    let indices = DynTensor::from_vec_u32(vec![0, 2, 4], &[3], &Device::Cpu).unwrap();
    let idx_tracked = Arc::new(TrackedTensor::from_tensor(indices));
    let y = layer.forward_indices(&idx_tracked).unwrap();

    // Scalar loss: sum all outputs
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let w_grad = grads
        .get(layer.weight())
        .expect("embedding weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());

    // Rows 0, 2, 4 should have non-zero gradients; rows 1, 3 should be zero
    let vals = w_grad.to_flat_vec::<f32>().unwrap();
    let row0_sum: f32 = vals[0..4].iter().map(|v| v.abs()).sum();
    let row1_sum: f32 = vals[4..8].iter().map(|v| v.abs()).sum();
    assert!(row0_sum > 0.0, "indexed row 0 should have gradient");
    assert!(
        row1_sum.abs() < 1e-7,
        "non-indexed row 1 should have zero gradient"
    );
}

#[test]
fn test_trainable_embedding_vars() {
    let layer = TrainableEmbedding::new(10, 4).unwrap();
    assert_eq!(layer.vars().len(), 1);
}

// -- TrainableLayerNorm tests --

#[test]
fn test_trainable_layer_norm_forward_shape() {
    let layer = TrainableLayerNorm::new(4, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
}

#[test]
fn test_trainable_layer_norm_normalizes() {
    let layer = TrainableLayerNorm::new(4, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();

    // With weight=1, bias=0, output should be normalized (mean~0, var~1)
    let vals = y.tensor().to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(
        mean.abs() < 1e-5,
        "normalized mean should be ~0, got {mean}"
    );
}

#[test]
fn test_trainable_layer_norm_backward_produces_gradients() {
    let layer = TrainableLayerNorm::new(4, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();

    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let w_grad = grads
        .get(layer.weight())
        .expect("layer norm weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());

    let b_grad = grads.get(layer.bias()).expect("layer norm bias gradient");
    assert_eq!(b_grad.dims(), layer.bias().dims().unwrap());

    let max_abs = w_grad
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "weight gradient should be non-zero");
}

#[test]
fn test_trainable_layer_norm_vars() {
    let layer = TrainableLayerNorm::new(4, 1e-5).unwrap();
    assert_eq!(layer.vars().len(), 2);
}

// -- TrainableConv2d tests --

#[test]
fn test_trainable_conv2d_forward_shape() {
    // Weight: [out_channels=4, in_channels=2, kH=3, kW=3]
    let w = DynTensor::from_vec(vec![0.1; 4 * 2 * 3 * 3], &[4, 2, 3, 3], &Device::Cpu).unwrap();
    let layer = TrainableConv2d::from_tensors(w, None, 1, 1, 1, 1);

    // Input: [batch=1, channels=2, H=8, W=8]
    let x = DynTensor::from_vec(vec![1.0; 128], &[1, 2, 8, 8], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    // Output: [1, 4, 8, 8] (padding=1, stride=1 preserves spatial for 3x3 kernel)
    assert_eq!(y.dims(), &[1, 4, 8, 8]);
}

#[test]
fn test_trainable_conv2d_with_bias() {
    let w = DynTensor::from_vec(vec![0.1; 18], &[2, 1, 3, 3], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(vec![0.5, -0.5], &[2], &Device::Cpu).unwrap();
    let layer = TrainableConv2d::from_tensors(w, Some(b), 1, 1, 1, 1);

    let x = DynTensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[1, 2, 4, 4]);
    assert_eq!(layer.vars().len(), 2);
}

#[test]
fn test_trainable_conv2d_backward_produces_gradients() {
    let w = DynTensor::from_vec(
        (0..18).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[2, 1, 3, 3],
        &Device::Cpu,
    )
    .unwrap();
    let layer = TrainableConv2d::from_tensors(w, None, 1, 1, 1, 1);

    let x = DynTensor::from_vec(
        (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[1, 1, 4, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();

    // Scalar loss
    let loss = y
        .sum_keepdim(3)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let w_grad = grads.get(layer.weight()).expect("conv2d weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());

    let max_abs = w_grad
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs > 0.0, "conv2d weight gradient should be non-zero");
}

#[test]
fn test_trainable_conv2d_trait_object() {
    let w = DynTensor::from_vec(vec![0.1; 18], &[2, 1, 3, 3], &Device::Cpu).unwrap();
    let layer = TrainableConv2d::from_tensors(w, None, 1, 1, 1, 1);

    let module: &dyn TrainableModule = &layer;
    assert_eq!(module.vars().len(), 1);

    let x = DynTensor::from_vec(vec![1.0; 16], &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = module.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[1, 2, 4, 4]);
}

// -- TrainableRmsNorm tests --

#[test]
fn test_trainable_rms_norm_forward_shape() {
    let layer = TrainableRmsNorm::new(4, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
}

#[test]
fn test_trainable_rms_norm_backward() {
    let layer = TrainableRmsNorm::new(4, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    let loss = y.sum_keepdim(1).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let w_grad = grads.get(layer.weight()).expect("rms_norm weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());
}

#[test]
fn test_trainable_rms_norm_vars() {
    let layer = TrainableRmsNorm::new(4, 1e-5).unwrap();
    assert_eq!(layer.vars().len(), 1);
}

// -- TrainableGroupNorm tests --

#[test]
fn test_trainable_group_norm_forward_shape() {
    // 4 channels, 2 groups, input [N=1, C=4, L=6]
    let layer = TrainableGroupNorm::new(4, 2, 1e-5).unwrap();
    let x = DynTensor::from_vec(vec![1.0; 24], &[1, 4, 6], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[1, 4, 6]);
}

#[test]
fn test_trainable_group_norm_backward() {
    let layer = TrainableGroupNorm::new(4, 2, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[1, 4, 6],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    let loss = y
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let w_grad = grads
        .get(layer.weight())
        .expect("group_norm weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());
    let b_grad = grads.get(layer.bias()).expect("group_norm bias gradient");
    assert_eq!(b_grad.dims(), layer.bias().dims().unwrap());
}

#[test]
fn test_trainable_group_norm_vars() {
    let layer = TrainableGroupNorm::new(4, 2, 1e-5).unwrap();
    assert_eq!(layer.vars().len(), 2);
}

// -- TrainableBatchNorm tests --

#[test]
fn test_trainable_batch_norm_forward_shape() {
    let layer = TrainableBatchNorm::new(3, 1e-5).unwrap();
    // Input [N=2, C=3, L=4]
    let x = DynTensor::from_vec(vec![1.0; 24], &[2, 3, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_trainable_batch_norm_backward() {
    let layer = TrainableBatchNorm::new(3, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[2, 3, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    let loss = y
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let w_grad = grads
        .get(layer.weight())
        .expect("batch_norm weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());
}

#[test]
fn test_trainable_batch_norm_vars() {
    let layer = TrainableBatchNorm::new(3, 1e-5).unwrap();
    assert_eq!(layer.vars().len(), 2);
}

// -- TrainableInstanceNorm tests --

#[test]
fn test_trainable_instance_norm_forward_shape() {
    let layer = TrainableInstanceNorm::new(3, 1e-5).unwrap();
    // Input [N=2, C=3, L=4]
    let x = DynTensor::from_vec(vec![1.0; 24], &[2, 3, 4], &Device::Cpu).unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

#[test]
fn test_trainable_instance_norm_backward() {
    let layer = TrainableInstanceNorm::new(3, 1e-5).unwrap();
    let x = DynTensor::from_vec(
        (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect(),
        &[2, 3, 4],
        &Device::Cpu,
    )
    .unwrap();
    let xt = Arc::new(TrackedTensor::from_tensor(x));
    let y = layer.forward(&xt).unwrap();
    let loss = y
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(0)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let w_grad = grads
        .get(layer.weight())
        .expect("instance_norm weight gradient");
    assert_eq!(w_grad.dims(), layer.weight().dims().unwrap());
}

#[test]
fn test_trainable_instance_norm_vars() {
    let layer = TrainableInstanceNorm::new(3, 1e-5).unwrap();
    assert_eq!(layer.vars().len(), 2);
}
