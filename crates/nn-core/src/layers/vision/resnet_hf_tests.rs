// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for HuggingFace-compatible ResNet-18 backbone.

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::resnet_hf::ResNet18Hf;
use crate::{DType, Device};

#[test]
fn test_resnet18_hf_load_zeros() {
    let device = Device::Cpu;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = ResNet18Hf::load(&vb, None);
    assert!(
        model.is_ok(),
        "ResNet18Hf should load with zero weights: {:?}",
        model.err()
    );
}

#[test]
fn test_resnet18_hf_forward_features_shapes_224() {
    let device = Device::Cpu;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = ResNet18Hf::load(&vb, None).unwrap();

    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &device).unwrap();
    let features = model.forward_features(&x).unwrap();

    assert_eq!(features.len(), 4);
    // C2: layer1, stride 4 => 224/4 = 56
    assert_eq!(features[0].dims(), &[1, 64, 56, 56]);
    // C3: layer2, stride 8 => 224/8 = 28
    assert_eq!(features[1].dims(), &[1, 128, 28, 28]);
    // C4: layer3, stride 16 => 224/16 = 14
    assert_eq!(features[2].dims(), &[1, 256, 14, 14]);
    // C5: layer4, stride 32 => 224/32 = 7
    assert_eq!(features[3].dims(), &[1, 512, 7, 7]);
}

#[test]
fn test_resnet18_hf_forward_features_shapes_640() {
    let device = Device::Cpu;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = ResNet18Hf::load(&vb, None).unwrap();

    let x = DynTensor::full(&[1, 3, 640, 640], 0.5, DType::F32, &device).unwrap();
    let features = model.forward_features(&x).unwrap();

    assert_eq!(features.len(), 4);
    // stem: conv0(s=2) 640/2=320, maxpool(s=2) 320/2=160
    assert_eq!(features[0].dims(), &[1, 64, 160, 160]); // C2: stride 4
    assert_eq!(features[1].dims(), &[1, 128, 80, 80]); // C3: stride 8
    assert_eq!(features[2].dims(), &[1, 256, 40, 40]); // C4: stride 16
    assert_eq!(features[3].dims(), &[1, 512, 20, 20]); // C5: stride 32
}

#[test]
fn test_resnet18_hf_classification() {
    let device = Device::Cpu;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = ResNet18Hf::load(&vb, Some(1000)).unwrap();

    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &device).unwrap();
    let y = model.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1000]);
}

#[test]
fn test_resnet18_hf_no_fc() {
    let device = Device::Cpu;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = ResNet18Hf::load(&vb, None).unwrap();

    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &device).unwrap();
    let y = model.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 512]);
}

#[test]
fn test_resnet18_hf_feature_channels() {
    let device = Device::Cpu;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &device);
    let model = ResNet18Hf::load(&vb, None).unwrap();

    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &device).unwrap();
    let features = model.forward_features(&x).unwrap();

    let channels: Vec<usize> = features.iter().map(|f| f.dims()[1]).collect();
    assert_eq!(channels, vec![64, 128, 256, 512]);
}
