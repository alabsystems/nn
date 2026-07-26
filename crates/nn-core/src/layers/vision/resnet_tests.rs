// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ResNet-18 backbone (issue #3878).

use crate::dyn_tensor::DynTensor;
use crate::layers::vision::resnet::{BasicBlock, ResNet18};
use crate::layers::{BatchNorm2d, Conv2d, Conv2dConfig, Linear, Module};
use crate::{DType, Device};

/// Helper: create a Conv2d with small constant weights.
fn make_conv2d(in_c: usize, out_c: usize, k: usize, stride: usize, padding: usize) -> Conv2d {
    let weight = DynTensor::full(&[out_c, in_c, k, k], 0.01, DType::F32, &Device::Cpu).unwrap();
    let cfg = Conv2dConfig::new(padding, stride, 1);
    Conv2d::new(weight, None, cfg).unwrap()
}

/// Helper: create a BatchNorm2d with identity normalization (mean=0, var=1, gamma=1, beta=0).
fn make_bn2d(channels: usize) -> BatchNorm2d {
    BatchNorm2d::new(
        channels,
        DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap(),
        DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap(),
        Some(DynTensor::ones(&[channels], DType::F32, &Device::Cpu).unwrap()),
        Some(DynTensor::zeros(&[channels], DType::F32, &Device::Cpu).unwrap()),
        1e-5,
    )
    .unwrap()
}

/// Helper: create a BasicBlock without downsample (same channels, stride 1).
fn make_basic_block(channels: usize) -> BasicBlock {
    BasicBlock::new(
        make_conv2d(channels, channels, 3, 1, 1),
        make_bn2d(channels),
        make_conv2d(channels, channels, 3, 1, 1),
        make_bn2d(channels),
        None,
    )
}

/// Helper: create a BasicBlock with downsample (channel expansion or stride > 1).
fn make_basic_block_ds(in_c: usize, out_c: usize, stride: usize) -> BasicBlock {
    let ds = if stride != 1 || in_c != out_c {
        Some((make_conv2d(in_c, out_c, 1, stride, 0), make_bn2d(out_c)))
    } else {
        None
    };
    BasicBlock::new(
        make_conv2d(in_c, out_c, 3, stride, 1),
        make_bn2d(out_c),
        make_conv2d(out_c, out_c, 3, 1, 1),
        make_bn2d(out_c),
        ds,
    )
}

/// Helper: build a full ResNet18 from manually constructed layers.
fn make_resnet18(num_classes: Option<usize>) -> ResNet18 {
    let conv1 = make_conv2d(3, 64, 7, 2, 3);
    let bn1 = make_bn2d(64);

    let layer1 = [make_basic_block(64), make_basic_block(64)];
    let layer2 = [make_basic_block_ds(64, 128, 2), make_basic_block(128)];
    let layer3 = [make_basic_block_ds(128, 256, 2), make_basic_block(256)];
    let layer4 = [make_basic_block_ds(256, 512, 2), make_basic_block(512)];

    let fc = num_classes.map(|n| {
        let w = DynTensor::full(&[n, 512], 0.01, DType::F32, &Device::Cpu).unwrap();
        Linear::new(w, None).unwrap()
    });

    ResNet18 {
        conv1,
        bn1,
        layer1,
        layer2,
        layer3,
        layer4,
        fc,
    }
}

// ---------------------------------------------------------------------------
// BasicBlock tests
// ---------------------------------------------------------------------------

#[test]
fn test_basic_block_same_channels() {
    let block = make_basic_block(64);
    let x = DynTensor::full(&[1, 64, 56, 56], 0.5, DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 64, 56, 56]);
}

#[test]
fn test_basic_block_with_downsample() {
    let block = make_basic_block_ds(64, 128, 2);
    let x = DynTensor::full(&[1, 64, 56, 56], 0.5, DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 128, 28, 28]);
}

#[test]
fn test_basic_block_skip_connection() {
    // With identity normalization (mean=0, var=1, gamma=1, beta=0) and small
    // conv weights, output should be non-zero due to skip connection + relu.
    let block = make_basic_block(32);
    let x = DynTensor::full(&[1, 32, 8, 8], 1.0, DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 32, 8, 8]);
    // The skip connection adds the input, so output should not be all-zero.
    let vals = y.to_flat_vec::<f32>().unwrap();
    let any_nonzero = vals.iter().any(|&v| v > 0.0);
    assert!(
        any_nonzero,
        "skip connection should produce non-zero outputs"
    );
}

// ---------------------------------------------------------------------------
// ResNet18 shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_resnet18_conv1_shape() {
    let model = make_resnet18(Some(1000));
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    // conv1(7x7, s=2, p=3): [1,3,224,224] → [1,64,112,112]
    let out = model.conv1().forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 64, 112, 112]);
}

#[test]
fn test_resnet18_layer1_shape() {
    let model = make_resnet18(Some(1000));
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();
    // C2 = layer1 output: [1, 64, 56, 56]
    assert_eq!(features[0].dims(), &[1, 64, 56, 56]);
}

#[test]
fn test_resnet18_layer2_shape() {
    let model = make_resnet18(Some(1000));
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();
    // C3 = layer2 output: [1, 128, 28, 28]
    assert_eq!(features[1].dims(), &[1, 128, 28, 28]);
}

#[test]
fn test_resnet18_layer3_shape() {
    let model = make_resnet18(Some(1000));
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();
    // C4 = layer3 output: [1, 256, 14, 14]
    assert_eq!(features[2].dims(), &[1, 256, 14, 14]);
}

#[test]
fn test_resnet18_layer4_shape() {
    let model = make_resnet18(Some(1000));
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();
    // C5 = layer4 output: [1, 512, 7, 7]
    assert_eq!(features[3].dims(), &[1, 512, 7, 7]);
}

#[test]
fn test_resnet18_forward_features_count() {
    let model = make_resnet18(None);
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();
    assert_eq!(
        features.len(),
        4,
        "forward_features should return [C2, C3, C4, C5]"
    );
}

#[test]
fn test_resnet18_feature_channels() {
    let model = make_resnet18(None);
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();
    let channels: Vec<usize> = features.iter().map(|f| f.dims()[1]).collect();
    assert_eq!(channels, vec![64, 128, 256, 512]);
}

#[test]
fn test_resnet18_input_800x800() {
    // dpdf/Table Transformer input size: 800x800
    let model = make_resnet18(None);
    let x = DynTensor::full(&[1, 3, 800, 800], 0.5, DType::F32, &Device::Cpu).unwrap();
    let features = model.forward_features(&x).unwrap();

    // conv1(s=2): 800/2 = 400, maxpool(s=2): 400/2 = 200
    assert_eq!(features[0].dims(), &[1, 64, 200, 200]); // C2: stride 4
    assert_eq!(features[1].dims(), &[1, 128, 100, 100]); // C3: stride 8
    assert_eq!(features[2].dims(), &[1, 256, 50, 50]); // C4: stride 16
    assert_eq!(features[3].dims(), &[1, 512, 25, 25]); // C5: stride 32
}

#[test]
fn test_resnet18_batch() {
    let model = make_resnet18(Some(10));
    let x = DynTensor::full(&[4, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let y = model.forward(&x).unwrap();
    assert_eq!(
        y.dims(),
        &[4, 10],
        "classification output should be [B, num_classes]"
    );
}

#[test]
fn test_resnet18_forward_no_fc() {
    // Without FC, forward should return [B, 512] pooled features.
    let model = make_resnet18(None);
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let y = model.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 512]);
}

#[test]
fn test_resnet18_classification_forward() {
    let model = make_resnet18(Some(1000));
    let x = DynTensor::full(&[1, 3, 224, 224], 0.5, DType::F32, &Device::Cpu).unwrap();
    let y = model.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1000]);
}
