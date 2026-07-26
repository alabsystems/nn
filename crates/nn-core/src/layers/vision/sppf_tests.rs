// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::{ConvBnAct, Sppf};
use crate::layers::{Activation, BatchNorm, Conv2d, Conv2dConfig, Module};
use crate::{DType, Device};

/// Helper: create a ConvBnAct from channel counts without VarBuilder.
fn make_conv_bn(in_c: usize, out_c: usize, k: usize) -> ConvBnAct {
    let padding = k / 2;
    let weight = DynTensor::full(&[out_c, in_c, k, k], 0.01, DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::new(padding, 1, 1);
    let conv = Conv2d::new(weight, None, cfg).unwrap();
    let bn = BatchNorm::new(
        DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap(),
        DynTensor::ones(&[out_c], DType::F32, &Device::Cpu).unwrap(),
        Some(DynTensor::ones(&[out_c], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[out_c], DType::F32, &Device::Cpu).unwrap()),
        1e-5,
    )
    .unwrap();
    ConvBnAct::new(conv, bn, Some(Activation::Silu))
}

#[test]
fn test_sppf_forward_shape() {
    let channels = 64;
    let hidden = channels / 2;
    let cv1 = make_conv_bn(channels, hidden, 1);
    let cv2 = make_conv_bn(hidden * 4, channels, 1);

    let sppf = Sppf::new(cv1, cv2, 5);
    let input = DynTensor::full(&[1, 64, 20, 20], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = sppf.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 64, 20, 20]);
}

#[test]
fn test_sppf_preserves_spatial_dims() {
    let channels = 32;
    let hidden = channels / 2;
    let cv1 = make_conv_bn(channels, hidden, 1);
    let cv2 = make_conv_bn(hidden * 4, channels, 1);

    let sppf = Sppf::new(cv1, cv2, 5);
    let input = DynTensor::full(&[2, 32, 40, 40], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = sppf.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 32, 40, 40]);
}
