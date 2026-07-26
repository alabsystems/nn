#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::{AdaptiveAvgPool2d, AvgPool2d, MaxPool1d, MaxPool2d, Pool1dConfig, Pool2dConfig};
use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::Device;

// -- MaxPool1d ----------------------------------------------------------------

#[test]
fn test_max_pool1d_layer_forward() {
    let layer = MaxPool1d::new(Pool1dConfig::new(2)).unwrap();
    let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let x = DynTensor::from_vec(data, &[1, 1, 6], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // kernel=2, stride=2: max(1,3)=3, max(2,5)=5, max(4,6)=6
    assert_eq!(vals, vec![3.0, 5.0, 6.0]);
}

#[test]
fn test_max_pool1d_zero_kernel() {
    assert!(MaxPool1d::new(Pool1dConfig::new(0)).is_err());
}

#[test]
fn test_max_pool1d_config_accessor() {
    let cfg = Pool1dConfig {
        kernel_size: 3,
        stride: 2,
        padding: 1,
    };
    let layer = MaxPool1d::new(cfg).unwrap();
    assert_eq!(layer.config().kernel_size, 3);
    assert_eq!(layer.config().stride, 2);
    assert_eq!(layer.config().padding, 1);
}

// -- Pool2dConfig -------------------------------------------------------------

#[test]
fn test_pool2d_config_new_defaults() {
    let cfg = Pool2dConfig::new(3);
    assert_eq!(cfg.kernel_size, 3);
    assert_eq!(cfg.stride, 3); // default: same as kernel_size
    assert_eq!(cfg.padding, 0);
}

// -- MaxPool2d ----------------------------------------------------------------

#[test]
fn test_max_pool2d_layer_forward() {
    let layer = MaxPool2d::new(Pool2dConfig::new(2)).unwrap();
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn test_max_pool2d_zero_kernel() {
    assert!(MaxPool2d::new(Pool2dConfig::new(0)).is_err());
}

#[test]
fn test_max_pool2d_config_accessor() {
    let layer = MaxPool2d::new(Pool2dConfig {
        kernel_size: 3,
        stride: 2,
        padding: 1,
    })
    .unwrap();
    assert_eq!(layer.config().kernel_size, 3);
    assert_eq!(layer.config().stride, 2);
    assert_eq!(layer.config().padding, 1);
}

// -- AvgPool2d ----------------------------------------------------------------

#[test]
fn test_avg_pool2d_layer_forward() {
    let layer = AvgPool2d::new(Pool2dConfig::new(2)).unwrap();
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn test_avg_pool2d_zero_stride() {
    let cfg = Pool2dConfig {
        kernel_size: 2,
        stride: 0,
        padding: 0,
    };
    assert!(AvgPool2d::new(cfg).is_err());
}

// -- AdaptiveAvgPool2d --------------------------------------------------------

#[test]
fn test_adaptive_avg_pool2d_layer_1x1() {
    let layer = AdaptiveAvgPool2d::new(1, 1).unwrap();
    let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 8.5).abs() < 1e-6);
}

#[test]
fn test_adaptive_avg_pool2d_zero_output() {
    assert!(AdaptiveAvgPool2d::new(0, 1).is_err());
}

#[test]
fn test_adaptive_avg_pool2d_output_size() {
    let layer = AdaptiveAvgPool2d::new(3, 5).unwrap();
    assert_eq!(layer.output_size(), (3, 5));
}
