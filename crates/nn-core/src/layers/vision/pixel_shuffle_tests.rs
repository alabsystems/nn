#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::Device;

use super::{PixelShuffle, PixelUnshuffle};

// -- pixel_shuffle DynTensor op tests -----------------------------------------

#[test]
fn test_pixel_shuffle_basic() {
    // [1, 4, 1, 1] → [1, 1, 2, 2] with r=2.
    // 4 channels, 1×1 spatial → 1 channel, 2×2 spatial.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1], &Device::Cpu).unwrap();
    let y = x.pixel_shuffle(2).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // PyTorch PixelShuffle(2) with [1, 4, 1, 1] input [1,2,3,4]
    // produces [1, 1, 2, 2] with [[1, 2], [3, 4]].
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_pixel_shuffle_2x2_input() {
    // [1, 4, 2, 2] → [1, 1, 4, 4] with r=2.
    #[rustfmt::skip]
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 4, 2, 2], &Device::Cpu).unwrap();
    let y = x.pixel_shuffle(2).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    assert_eq!(y.numel(), 16);
}

#[test]
fn test_pixel_shuffle_multi_channel() {
    // [1, 8, 1, 1] → [1, 2, 2, 2] with r=2.
    let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 8, 1, 1], &Device::Cpu).unwrap();
    let y = x.pixel_shuffle(2).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
}

#[test]
fn test_pixel_shuffle_scale1_identity() {
    // r=1 should be identity.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.pixel_shuffle(1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_pixel_shuffle_zero_factor() {
    let x = DynTensor::from_vec(vec![1.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    assert!(x.pixel_shuffle(0).is_err());
}

#[test]
fn test_pixel_shuffle_indivisible_channels() {
    // 3 channels not divisible by r²=4.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1, 1], &Device::Cpu).unwrap();
    assert!(x.pixel_shuffle(2).is_err());
}

#[test]
fn test_pixel_shuffle_rank_too_low() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    assert!(x.pixel_shuffle(1).is_err());
}

#[test]
fn test_pixel_shuffle_3d() {
    // rank=3 is [C, H, W], no batch dim.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4, 1, 1], &Device::Cpu).unwrap();
    let y = x.pixel_shuffle(2).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2]);
}

// -- pixel_unshuffle DynTensor op tests ---------------------------------------

#[test]
fn test_pixel_unshuffle_basic() {
    // [1, 1, 2, 2] → [1, 4, 1, 1] with r=2.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.pixel_unshuffle(2).unwrap();
    assert_eq!(y.dims(), &[1, 4, 1, 1]);
}

#[test]
fn test_pixel_unshuffle_multi_channel() {
    // [1, 2, 4, 4] → [1, 8, 2, 2] with r=2.
    let data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 2, 4, 4], &Device::Cpu).unwrap();
    let y = x.pixel_unshuffle(2).unwrap();
    assert_eq!(y.dims(), &[1, 8, 2, 2]);
}

#[test]
fn test_pixel_unshuffle_zero_factor() {
    let x = DynTensor::from_vec(vec![1.0], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    assert!(x.pixel_unshuffle(0).is_err());
}

#[test]
fn test_pixel_unshuffle_indivisible_dims() {
    // H=3 not divisible by r=2.
    let data: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 3, 2], &Device::Cpu).unwrap();
    assert!(x.pixel_unshuffle(2).is_err());
}

// -- Round-trip test ----------------------------------------------------------

#[test]
fn test_pixel_shuffle_unshuffle_roundtrip() {
    // PixelShuffle(r) followed by PixelUnshuffle(r) should be identity.
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 4, 2, 2], &Device::Cpu).unwrap();
    let shuffled = x.pixel_shuffle(2).unwrap();
    assert_eq!(shuffled.dims(), &[1, 1, 4, 4]);
    let restored = shuffled.pixel_unshuffle(2).unwrap();
    assert_eq!(restored.dims(), &[1, 4, 2, 2]);
    assert_eq!(restored.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_pixel_unshuffle_shuffle_roundtrip() {
    // PixelUnshuffle(r) followed by PixelShuffle(r) should be identity.
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let unshuffled = x.pixel_unshuffle(2).unwrap();
    assert_eq!(unshuffled.dims(), &[1, 4, 2, 2]);
    let restored = unshuffled.pixel_shuffle(2).unwrap();
    assert_eq!(restored.dims(), &[1, 1, 4, 4]);
    assert_eq!(restored.to_flat_vec::<f32>().unwrap(), data);
}

// -- PixelShuffle nn layer tests ----------------------------------------------

#[test]
fn test_pixel_shuffle_layer() {
    let layer = PixelShuffle::new(2).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_eq!(layer.upscale_factor(), 2);
}

#[test]
fn test_pixel_unshuffle_layer() {
    let layer = PixelUnshuffle::new(2).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 1, 1]);
    assert_eq!(layer.downscale_factor(), 2);
}

#[test]
fn test_pixel_shuffle_layer_zero() {
    assert!(PixelShuffle::new(0).is_err());
}

#[test]
fn test_pixel_unshuffle_layer_zero() {
    assert!(PixelUnshuffle::new(0).is_err());
}
