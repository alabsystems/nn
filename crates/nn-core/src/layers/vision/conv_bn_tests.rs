// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::ConvBnAct;
use crate::layers::{Activation, BatchNorm, Conv2d, Conv2dConfig, Module};
use crate::{DType, Device};

#[test]
fn test_conv_bn_act_forward_shape() {
    let weight = DynTensor::full(&[16, 3, 3, 3], 0.01, DType::F32, &Device::Cpu).unwrap();
    let conv_cfg = Conv2dConfig::new(1, 1, 1);
    let conv = Conv2d::new(weight, None, conv_cfg).unwrap();

    let bn = BatchNorm::new(
        DynTensor::zeros(&[16], DType::F32, &Device::Cpu).unwrap(),
        DynTensor::ones(&[16], DType::F32, &Device::Cpu).unwrap(),
        Some(DynTensor::ones(&[16], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[16], DType::F32, &Device::Cpu).unwrap()),
        1e-5,
    )
    .unwrap();

    let block = ConvBnAct::new(conv, bn, Some(Activation::Silu));
    let input = DynTensor::full(&[1, 3, 32, 32], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 16, 32, 32]);
}

#[test]
fn test_conv_bn_no_activation() {
    let weight = DynTensor::full(&[8, 4, 1, 1], 0.01, DType::F32, &Device::Cpu).unwrap();
    let conv_cfg = Conv2dConfig::default();
    let conv = Conv2d::new(weight, None, conv_cfg).unwrap();

    let bn = BatchNorm::new(
        DynTensor::zeros(&[8], DType::F32, &Device::Cpu).unwrap(),
        DynTensor::ones(&[8], DType::F32, &Device::Cpu).unwrap(),
        Some(DynTensor::ones(&[8], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[8], DType::F32, &Device::Cpu).unwrap()),
        1e-5,
    )
    .unwrap();

    let block = ConvBnAct::new(conv, bn, None);
    let input = DynTensor::full(&[2, 4, 16, 16], 0.5, DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 8, 16, 16]);
}
