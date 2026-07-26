#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::{
    BatchNorm, Conv2d, Conv2dConfig, Linear, MBConv, MBConvConfig, Module, SqueezeExcitation,
};
use crate::{Device, TensorError};

/// Create a Conv2d with small random-ish weights for testing.
fn make_test_conv2d(in_ch: usize, out_ch: usize, k: usize, config: Conv2dConfig) -> Conv2d {
    let in_per_group = in_ch / config.groups;
    let n = out_ch * in_per_group * k * k;
    let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let weight = DynTensor::from_vec(data, &[out_ch, in_per_group, k, k], &Device::Cpu).unwrap();
    Conv2d::new(weight, None, config).unwrap()
}

/// Create a BatchNorm in eval mode for testing.
fn make_test_bn(channels: usize) -> BatchNorm {
    let mean = DynTensor::zeros(&[channels], crate::DType::F32, &Device::Cpu).unwrap();
    let var = DynTensor::ones(&[channels], crate::DType::F32, &Device::Cpu).unwrap();
    let gamma = DynTensor::ones(&[channels], crate::DType::F32, &Device::Cpu).unwrap();
    let beta = DynTensor::zeros(&[channels], crate::DType::F32, &Device::Cpu).unwrap();
    BatchNorm::new(mean, var, Some(gamma), Some(beta), 1e-5).unwrap()
}

fn make_test_linear(in_f: usize, out_f: usize) -> Linear {
    let n = out_f * in_f;
    let data: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();
    let weight = DynTensor::from_vec(data, &[out_f, in_f], &Device::Cpu).unwrap();
    Linear::new(weight, None).unwrap()
}

fn make_test_se(channels: usize, reduced: usize) -> SqueezeExcitation {
    let fc1 = make_test_linear(channels, reduced);
    let fc2 = make_test_linear(reduced, channels);
    SqueezeExcitation::new(fc1, fc2, channels).unwrap()
}

/// Build a complete MBConv block for testing with expand_ratio=1 (no expansion).
fn make_test_mbconv_no_expand(in_ch: usize, out_ch: usize) -> MBConv {
    let hidden = in_ch; // expand_ratio = 1
    let dw_config = Conv2dConfig {
        padding: 1,
        groups: hidden,
        ..Conv2dConfig::default()
    };
    let depthwise = make_test_conv2d(hidden, hidden, 3, dw_config);
    let dw_bn = make_test_bn(hidden);
    let se = make_test_se(hidden, (in_ch / 4).max(1));
    let project = make_test_conv2d(hidden, out_ch, 1, Conv2dConfig::default());
    let proj_bn = make_test_bn(out_ch);
    let use_residual = in_ch == out_ch;
    MBConv::new(None, depthwise, dw_bn, se, project, proj_bn, use_residual)
}

/// Build a complete MBConv block with expansion.
fn make_test_mbconv_with_expand(in_ch: usize, out_ch: usize, expand_ratio: usize) -> MBConv {
    let hidden = in_ch * expand_ratio;
    let expand_conv = make_test_conv2d(in_ch, hidden, 1, Conv2dConfig::default());
    let expand_bn = make_test_bn(hidden);
    let dw_config = Conv2dConfig {
        padding: 1,
        groups: hidden,
        ..Conv2dConfig::default()
    };
    let depthwise = make_test_conv2d(hidden, hidden, 3, dw_config);
    let dw_bn = make_test_bn(hidden);
    let se = make_test_se(hidden, (in_ch / 4).max(1));
    let project = make_test_conv2d(hidden, out_ch, 1, Conv2dConfig::default());
    let proj_bn = make_test_bn(out_ch);
    let use_residual = in_ch == out_ch;
    MBConv::new(
        Some((expand_conv, expand_bn)),
        depthwise,
        dw_bn,
        se,
        project,
        proj_bn,
        use_residual,
    )
}

#[test]
fn test_mbconv_no_expand_shape() {
    let block = make_test_mbconv_no_expand(8, 8);
    let x = DynTensor::ones(&[1, 8, 4, 4], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 4, 4]);
}

#[test]
fn test_mbconv_with_expand_shape() {
    let block = make_test_mbconv_with_expand(8, 8, 4);
    let x = DynTensor::ones(&[1, 8, 4, 4], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 4, 4]);
}

#[test]
fn test_mbconv_channel_change() {
    // in_ch != out_ch → no residual.
    let block = make_test_mbconv_no_expand(4, 8);
    assert!(!block.use_residual());
    let x = DynTensor::ones(&[1, 4, 4, 4], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 4, 4]);
}

#[test]
fn test_mbconv_residual_flag() {
    let block = make_test_mbconv_no_expand(8, 8);
    assert!(block.use_residual());
}

#[test]
fn test_mbconv_no_residual_different_channels() {
    let block = make_test_mbconv_no_expand(4, 8);
    assert!(!block.use_residual());
}

#[test]
fn test_mbconv_batched() {
    let block = make_test_mbconv_no_expand(4, 4);
    let x = DynTensor::ones(&[3, 4, 4, 4], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[3, 4, 4, 4]);
}

#[test]
fn test_mbconv_output_finite() {
    let block = make_test_mbconv_no_expand(4, 4);
    let x = DynTensor::ones(&[1, 4, 3, 3], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

#[test]
fn test_mbconv_config_default() {
    let config = MBConvConfig::default();
    assert_eq!(config.expand_ratio, 1);
    assert_eq!(config.kernel_size, 3);
    assert_eq!(config.stride, 1);
    assert_eq!(config.se_ratio, 4);
}

#[test]
fn test_mbconv_wrong_rank() {
    let block = make_test_mbconv_no_expand(4, 4);
    // 3D input should fail with rank error.
    let x = DynTensor::ones(&[1, 4, 8], crate::DType::F32, &Device::Cpu).unwrap();
    let err = block.forward(&x).unwrap_err();
    assert!(
        matches!(err, TensorError::RankMismatch { expected: 4, .. }),
        "expected RankMismatch for 3D input, got: {err:?}"
    );
}

// -- VarBuilder load tests ----------------------------------------------------

#[test]
fn test_mbconv_load_no_expand() {
    use crate::var_builder::VarBuilder;

    let in_ch = 8;
    let out_ch = 8;
    let config = MBConvConfig {
        expand_ratio: 1,
        kernel_size: 3,
        stride: 1,
        se_ratio: 4,
    };
    let vb = VarBuilder::zeros(crate::DType::F32, &Device::Cpu);
    let block = MBConv::load(&vb, in_ch, out_ch, config).expect("load should succeed");
    assert!(block.use_residual());

    // Forward pass with zero weights should produce finite output.
    let x = DynTensor::ones(&[1, in_ch, 4, 4], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, out_ch, 4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

#[test]
fn test_mbconv_load_with_expand() {
    use crate::var_builder::VarBuilder;

    let in_ch = 4;
    let out_ch = 4;
    let config = MBConvConfig {
        expand_ratio: 4,
        kernel_size: 3,
        stride: 1,
        se_ratio: 4,
    };
    let vb = VarBuilder::zeros(crate::DType::F32, &Device::Cpu);
    let block = MBConv::load(&vb, in_ch, out_ch, config).expect("load should succeed");
    assert!(block.use_residual());

    let x = DynTensor::ones(&[1, in_ch, 6, 6], crate::DType::F32, &Device::Cpu).unwrap();
    let y = block.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, out_ch, 6, 6]);
}

#[test]
fn test_mbconv_load_stride_disables_residual() {
    use crate::var_builder::VarBuilder;

    let config = MBConvConfig {
        expand_ratio: 1,
        kernel_size: 3,
        stride: 2,
        se_ratio: 4,
    };
    let vb = VarBuilder::zeros(crate::DType::F32, &Device::Cpu);
    let block = MBConv::load(&vb, 8, 8, config).expect("load should succeed");
    // stride > 1 disables residual even when in_ch == out_ch.
    assert!(!block.use_residual());
}

#[test]
fn test_mbconv_load_zero_channels_error() {
    use crate::var_builder::VarBuilder;

    let vb = VarBuilder::zeros(crate::DType::F32, &Device::Cpu);
    let err = MBConv::load(&vb, 0, 8, MBConvConfig::default()).unwrap_err();
    assert!(
        err.to_string().contains("channels must be > 0"),
        "expected channel error, got: {err}"
    );
}
