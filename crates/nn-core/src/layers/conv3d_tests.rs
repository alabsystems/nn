// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for layers::Conv3d layer.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Conv3d, Conv3dConfig, Module};
use crate::Device;

fn make_tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::new(data, shape, &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_construction_valid() {
    // [out_ch=2, in_ch=3, kD=1, kH=1, kW=1]
    let weight = make_tensor(&[1.0; 6], &[2, 3, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default());
    assert!(conv.is_ok());
    let conv = conv.unwrap();
    assert_eq!(conv.weight().dims(), &[2, 3, 1, 1, 1]);
    assert!(conv.bias().is_none());
    assert_eq!(conv.config().groups, 1);
}

#[test]
fn test_conv3d_construction_with_bias() {
    let weight = make_tensor(&[1.0; 6], &[2, 3, 1, 1, 1]);
    let bias = make_tensor(&[0.5, 1.5], &[2]);
    let conv = Conv3d::new(weight, Some(bias), Conv3dConfig::default()).unwrap();
    assert!(conv.bias().is_some());
    assert_eq!(conv.bias().unwrap().dims(), &[2]);
}

#[test]
fn test_conv3d_invalid_weight_rank_rejected() {
    // 4D weight should be rejected
    let weight = make_tensor(&[1.0; 16], &[1, 1, 4, 4]);
    let result = Conv3d::new(weight, None, Conv3dConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_conv3d_invalid_weight_rank_3d() {
    let weight = make_tensor(&[1.0; 8], &[2, 2, 2]);
    let result = Conv3d::new(weight, None, Conv3dConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_conv3d_invalid_weight_rank_6d() {
    let weight = make_tensor(&[1.0; 2], &[1, 1, 1, 1, 1, 2]);
    let result = Conv3d::new(weight, None, Conv3dConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_conv3d_zero_groups_rejected() {
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);
    let config = Conv3dConfig {
        groups: 0,
        ..Default::default()
    };
    let result = Conv3d::new(weight, None, config);
    assert!(result.is_err());
}

#[test]
fn test_conv3d_bias_shape_mismatch_rejected() {
    // weight: [2, 1, 1, 1, 1] -> out_channels=2, but bias has 3 elements
    let weight = make_tensor(&[1.0; 2], &[2, 1, 1, 1, 1]);
    let bias = make_tensor(&[1.0; 3], &[3]);
    let result = Conv3d::new(weight, Some(bias), Conv3dConfig::default());
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Config builder API
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_config_default() {
    let cfg = Conv3dConfig::default();
    assert_eq!(cfg.padding, [0, 0, 0]);
    assert_eq!(cfg.stride, [1, 1, 1]);
    assert_eq!(cfg.dilation, [1, 1, 1]);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_conv3d_config_new_uniform() {
    let cfg = Conv3dConfig::new(2, 3, 4);
    assert_eq!(cfg.padding, [2, 2, 2]);
    assert_eq!(cfg.stride, [3, 3, 3]);
    assert_eq!(cfg.dilation, [4, 4, 4]);
    assert_eq!(cfg.groups, 1);
}

#[test]
fn test_conv3d_config_builder_chain() {
    let cfg = Conv3dConfig::default()
        .with_padding([1, 2, 3])
        .with_stride([2, 3, 4])
        .with_dilation([3, 4, 5])
        .with_groups(8);
    assert_eq!(cfg.padding, [1, 2, 3]);
    assert_eq!(cfg.stride, [2, 3, 4]);
    assert_eq!(cfg.dilation, [3, 4, 5]);
    assert_eq!(cfg.groups, 8);
}

// ---------------------------------------------------------------------------
// Forward pass: basic
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_forward_shape() {
    // Input: [N=1, C=1, D=4, H=4, W=4], kernel: [1, 1, 2, 2, 2], stride=1, padding=0
    // Output D: (4 - 2) / 1 + 1 = 3, same for H and W
    let input = make_tensor(&vec![1.0; 64], &[1, 1, 4, 4, 4]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_forward_values() {
    // 1 batch, 1 channel, 2x2x2 input, 1 output channel, 2x2x2 kernel (all 1s)
    // => single output value = sum of all 8 input elements
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 1, 1, 1]);
    let data = output.as_cpu_f32().unwrap();
    // sum 1..8 = 36
    assert!((data[[0, 0, 0, 0, 0]] - 36.0).abs() < 1e-5);
}

#[test]
fn test_conv3d_1x1x1_kernel_identity() {
    // 1x1x1 kernel with weight=1 is identity
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2, 2]);
    let v = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// ---------------------------------------------------------------------------
// Forward pass: kernel sizes
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_3x3x3_kernel() {
    // 1 batch, 1 channel, 3x3x3 input, 3x3x3 kernel (all 1s)
    // Output: (3-3)/1+1 = 1 per dim
    let data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let input = make_tensor(&data, &[1, 1, 3, 3, 3]);
    let weight = make_tensor(&[1.0; 27], &[1, 1, 3, 3, 3]);

    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    // sum 1..27 = 27*28/2 = 378
    assert!((v[0] - 378.0).abs() < 1e-4);
}

#[test]
fn test_conv3d_5x5x5_kernel() {
    // 1 batch, 1 channel, 5x5x5 input, 5x5x5 kernel (all 1s)
    // Output: (5-5)/1+1 = 1 per dim
    let data: Vec<f32> = (1..=125).map(|x| x as f32).collect();
    let input = make_tensor(&data, &[1, 1, 5, 5, 5]);
    let weight = make_tensor(&[1.0; 125], &[1, 1, 5, 5, 5]);

    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    // sum 1..125 = 125*126/2 = 7875
    assert!((v[0] - 7875.0).abs() < 1e-3);
}

#[test]
fn test_conv3d_non_cubic_kernel() {
    // kernel [1, 2, 3] on input [1, 1, 3, 4, 5]
    // out_d = (3-1)/1+1 = 3, out_h = (4-2)/1+1 = 3, out_w = (5-3)/1+1 = 3
    let input = make_tensor(&vec![1.0; 60], &[1, 1, 3, 4, 5]);
    let weight = make_tensor(&[1.0; 6], &[1, 1, 1, 2, 3]);

    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 3, 3, 3]);
    // Each output = sum of 1*2*3 = 6 ones = 6.0
    let v = output.to_flat_vec::<f32>().unwrap();
    for val in &v {
        assert!((*val - 6.0).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Forward pass: bias
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_with_bias() {
    // 1 batch, 1 channel, 2x2x2 input, 1x1x1 kernel (weight=1), bias=10
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);
    let bias = make_tensor(&[10.0], &[1]);

    let conv = Conv3d::new(weight, Some(bias), Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // Each element = input_val * 1.0 + 10.0
    assert!((data[[0, 0, 0, 0, 0]] - 11.0).abs() < 1e-5);
    assert!((data[[0, 0, 1, 1, 1]] - 18.0).abs() < 1e-5);
}

#[test]
fn test_conv3d_bias_multi_output_channel() {
    // 2 output channels, each with its own bias
    let input = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);
    // OC0: weight=1.0, OC1: weight=2.0
    let weight = make_tensor(&[1.0, 2.0], &[2, 1, 1, 1, 1]);
    let bias = make_tensor(&[10.0, 20.0], &[2]);

    let conv = Conv3d::new(weight, Some(bias), Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 2, 2, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // OC0: 1*1 + 10 = 11
    assert!((data[[0, 0, 0, 0, 0]] - 11.0).abs() < 1e-5);
    // OC1: 1*2 + 20 = 22
    assert!((data[[0, 1, 0, 0, 0]] - 22.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Forward pass: stride
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_with_stride() {
    // 1 batch, 1 channel, 4x4x4 input, 2x2x2 kernel (all 1s), stride=[2,2,2]
    let input = make_tensor(&vec![1.0; 64], &[1, 1, 4, 4, 4]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let config = Conv3dConfig::default().with_stride([2, 2, 2]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // Output: (4-2)/2 + 1 = 2 per dim
    assert_eq!(output.dims(), &[1, 1, 2, 2, 2]);
    // Each output = 8 * 1.0 = 8.0 (sum of eight 1s)
    let data = output.as_cpu_f32().unwrap();
    for val in data.iter() {
        assert!((*val - 8.0).abs() < 1e-5);
    }
}

#[test]
fn test_conv3d_asymmetric_stride() {
    // stride=[1,2,3] on 1x1x4x6x9 input with 1x1x1 kernel
    // out_d = (4-1)/1+1=4, out_h=(6-1)/2+1=3, out_w=(9-1)/3+1=3
    let input = make_tensor(&vec![1.0; 216], &[1, 1, 4, 6, 9]);
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);

    let config = Conv3dConfig::default().with_stride([1, 2, 3]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1, 4, 3, 3]);
}

#[test]
fn test_conv3d_large_stride_single_output() {
    // stride large enough to produce 1 output per dim
    // Input 4x4x4, kernel 2x2x2, stride [3,3,3]
    // out = (4-2)/3+1 = 1 per dim
    let input = make_tensor(&vec![2.0; 64], &[1, 1, 4, 4, 4]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let config = Conv3dConfig::default().with_stride([3, 3, 3]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    // 8 elements * 2.0 = 16.0
    assert!((v[0] - 16.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Forward pass: padding
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_with_padding() {
    // 1 batch, 1 channel, 2x2x2 input, 3x3x3 kernel (all 1s), padding=[1,1,1]
    // Output: (2 + 2*1 - 3)/1 + 1 = 2 per dim
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[1.0; 27], &[1, 1, 3, 3, 3]);

    let config = Conv3dConfig::default().with_padding([1, 1, 1]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // With padding=1 and 3x3x3 kernel on 2x2x2 input:
    // All positions see all 8 inputs: sum = 1+2+3+4+5+6+7+8 = 36.
    assert!((data[[0, 0, 0, 0, 0]] - 36.0).abs() < 1e-5);
    assert!((data[[0, 0, 1, 1, 1]] - 36.0).abs() < 1e-5);
}

#[test]
fn test_conv3d_same_padding() {
    // "same" padding: padding=1, kernel=3, stride=1 → output same as input
    // Input: [1, 1, 4, 4, 4], kernel: [1, 1, 3, 3, 3], pad=1
    // Out: (4+2*1-3)/1+1 = 4 per dim
    let input = make_tensor(&vec![1.0; 64], &[1, 1, 4, 4, 4]);
    let weight = make_tensor(&[1.0; 27], &[1, 1, 3, 3, 3]);

    let config = Conv3dConfig::default().with_padding([1, 1, 1]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // Same spatial dims
    assert_eq!(output.dims(), &[1, 1, 4, 4, 4]);
}

#[test]
fn test_conv3d_asymmetric_padding() {
    // padding=[0, 1, 2] on 1x1x2x2x2 input with 1x1x1 kernel
    // out_d = (2+0-1)/1+1=2, out_h=(2+2-1)/1+1=4, out_w=(2+4-1)/1+1=6
    let input = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);

    let config = Conv3dConfig::default().with_padding([0, 1, 2]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1, 2, 4, 6]);
}

// ---------------------------------------------------------------------------
// Forward pass: dilation
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_with_dilation() {
    // Input [1, 1, 5, 5, 5], kernel [1, 1, 2, 2, 2], dilation=[2,2,2]
    // effective kernel = (2-1)*2+1 = 3 per dim
    // out = (5-3)/1+1 = 3 per dim
    let data: Vec<f32> = (0..125).map(|x| x as f32).collect();
    let input = make_tensor(&data, &[1, 1, 5, 5, 5]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let config = Conv3dConfig::default().with_dilation([2, 2, 2]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 3, 3, 3]);

    // Verify first output element: kernel picks (0,0,0), (0,0,2), (0,2,0), (0,2,2),
    //                                            (2,0,0), (2,0,2), (2,2,0), (2,2,2)
    // Indices: 0, 2, 10, 12, 50, 52, 60, 62
    let expected_0 = 0.0 + 2.0 + 10.0 + 12.0 + 50.0 + 52.0 + 60.0 + 62.0;
    let v = output.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - expected_0).abs() < 1e-4);
}

#[test]
fn test_conv3d_asymmetric_dilation() {
    // dilation=[1, 2, 3] on [1, 1, 3, 5, 7] input with [1, 1, 2, 2, 2] kernel
    // eff_d = (2-1)*1+1 = 2, eff_h = (2-1)*2+1 = 3, eff_w = (2-1)*3+1 = 4
    // out_d = (3-2)/1+1=2, out_h = (5-3)/1+1=3, out_w = (7-4)/1+1=4
    let input = make_tensor(&vec![1.0; 105], &[1, 1, 3, 5, 7]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let config = Conv3dConfig::default().with_dilation([1, 2, 3]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Forward pass: groups
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_groups_basic() {
    // 2 groups: in_ch=4 (2 per group), out_ch=4 (2 per group)
    // Group 0: in[0:2] → out[0:2], Group 1: in[2:4] → out[2:4]
    // 1x1x1 kernel, identity per group
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1, 1]);
    // Weight [4, 2, 1, 1, 1]: OC0=[1,0], OC1=[0,1], OC2=[1,0], OC3=[0,1]
    let weight = make_tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[4, 2, 1, 1, 1]);
    let config = Conv3dConfig::default().with_groups(2);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 4, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv3d_depthwise() {
    // Depthwise: groups = in_channels = out_channels
    // Each channel independently multiplied by its own kernel
    let input = make_tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1, 1]);
    // weight [3, 1, 1, 1, 1]: each channel has weight 2.0
    let weight = make_tensor(&[2.0, 2.0, 2.0], &[3, 1, 1, 1, 1]);
    let config = Conv3dConfig::default().with_groups(3);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 3, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![2.0, 4.0, 6.0]);
}

// ---------------------------------------------------------------------------
// Forward pass: batch dimension
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_batch_size_2() {
    // 2 batch, 1 channel, 2x2x2 input, 1x1x1 kernel (weight=3)
    let input = make_tensor(
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // batch 0: all 1s
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ], // batch 1: all 2s
        &[2, 1, 2, 2, 2],
    );
    let weight = make_tensor(&[3.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 1, 2, 2, 2]);
    let v = output.to_flat_vec::<f32>().unwrap();
    // Batch 0: all 3.0, Batch 1: all 6.0
    for i in 0..8 {
        assert!((v[i] - 3.0).abs() < 1e-5, "batch 0 elem {i}");
    }
    for i in 8..16 {
        assert!((v[i] - 6.0).abs() < 1e-5, "batch 1 elem {i}");
    }
}

#[test]
fn test_conv3d_batch_with_bias() {
    // 2 batches, 1 channel, 1x1x1 input, kernel=2, bias=5
    let input = make_tensor(&[3.0, 7.0], &[2, 1, 1, 1, 1]);
    let weight = make_tensor(&[2.0], &[1, 1, 1, 1, 1]);
    let bias = make_tensor(&[5.0], &[1]);

    let conv = Conv3d::new(weight, Some(bias), Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 1, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    // batch0: 3*2+5=11, batch1: 7*2+5=19
    assert!((v[0] - 11.0).abs() < 1e-5);
    assert!((v[1] - 19.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Forward pass: multiple channels
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_multiple_output_channels() {
    // 1 batch, 1 in_channel, 2x2x2 input, 2 output channels, 1x1x1 kernel
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 2, 2]);
    // OC0: weight=2.0, OC1: weight=3.0
    let weight = make_tensor(&[2.0, 3.0], &[2, 1, 1, 1, 1]);

    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 2, 2, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // OC0: input * 2.0
    assert!((data[[0, 0, 0, 0, 0]] - 2.0).abs() < 1e-5);
    assert!((data[[0, 0, 1, 1, 1]] - 16.0).abs() < 1e-5);
    // OC1: input * 3.0
    assert!((data[[0, 1, 0, 0, 0]] - 3.0).abs() < 1e-5);
    assert!((data[[0, 1, 1, 1, 1]] - 24.0).abs() < 1e-5);
}

#[test]
fn test_conv3d_multi_input_multi_output() {
    // 2 input channels, 3 output channels, 1x1x1 kernel
    let input = make_tensor(&[1.0, 2.0], &[1, 2, 1, 1, 1]);
    // weight [3, 2, 1, 1, 1]:
    // OC0: [1, 0] → 1
    // OC1: [0, 1] → 2
    // OC2: [1, 1] → 3
    let weight = make_tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 3, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// Forward pass: combined parameters
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_stride_with_padding() {
    // Input [1, 1, 5, 5, 5], kernel 3x3x3, stride=2, padding=1
    // out = (5+2*1-3)/2+1 = 3 per dim
    let input = make_tensor(&vec![1.0; 125], &[1, 1, 5, 5, 5]);
    let weight = make_tensor(&[1.0; 27], &[1, 1, 3, 3, 3]);

    let config = Conv3dConfig::default()
        .with_stride([2, 2, 2])
        .with_padding([1, 1, 1]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_dilation_with_padding() {
    // Input [1, 1, 3, 3, 3], kernel 2x2x2, dilation=2, padding=1
    // effective kernel = 3; out = (3+2*1-3)/1+1 = 3 per dim
    let input = make_tensor(&[1.0; 27], &[1, 1, 3, 3, 3]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);

    let config = Conv3dConfig::default()
        .with_dilation([2, 2, 2])
        .with_padding([1, 1, 1]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_all_params_combined() {
    // Input [2, 4, 8, 8, 8], groups=2, kernel 3x3x3, stride=2, pad=1, dil=1
    // out = (8+2*1-3)/2+1 = 4 per dim, 4 out channels
    let input = make_tensor(&vec![1.0; 2 * 4 * 8 * 8 * 8], &[2, 4, 8, 8, 8]);
    // weight [4, 2, 3, 3, 3] (4 out_ch, 2 in_ch_per_group, 3x3x3)
    let weight = make_tensor(&vec![1.0; 4 * 2 * 27], &[4, 2, 3, 3, 3]);

    let config = Conv3dConfig::default()
        .with_groups(2)
        .with_stride([2, 2, 2])
        .with_padding([1, 1, 1]);
    let conv = Conv3d::new(weight, None, config).unwrap();
    let output = conv.forward(&input).unwrap();
    assert_eq!(output.dims(), &[2, 4, 4, 4, 4]);
}

// ---------------------------------------------------------------------------
// Error cases: forward
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_forward_wrong_input_rank() {
    // 4D input to forward should fail
    let input = make_tensor(&[1.0; 16], &[1, 1, 4, 4]);
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let result = conv.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_conv3d_forward_input_too_small() {
    // Input 1x1x1 with 2x2x2 kernel and no padding → kernel > input
    let input = make_tensor(&[1.0], &[1, 1, 1, 1, 1]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let result = conv.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_conv3d_forward_channel_mismatch() {
    // Input has 2 channels, kernel expects 1 (in_ch/groups=1)
    let input = make_tensor(&[1.0; 16], &[1, 2, 2, 2, 2]);
    let weight = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let result = conv.forward(&input);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_single_element_io() {
    // Minimal: 1x1x1x1x1 input, 1x1x1x1x1 kernel → 1x1x1x1x1 output
    let input = make_tensor(&[7.0], &[1, 1, 1, 1, 1]);
    let weight = make_tensor(&[3.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 1, 1, 1]);
    let v = output.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - 21.0).abs() < 1e-5);
}

#[test]
fn test_conv3d_zero_weight() {
    // Weight all zeros → output all zeros
    let input = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[0.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    let v = output.to_flat_vec::<f32>().unwrap();
    for val in &v {
        assert!(val.abs() < 1e-7);
    }
}

#[test]
fn test_conv3d_zero_input() {
    // Input all zeros → output all zeros (no bias)
    let input = make_tensor(&[0.0; 8], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[5.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    let v = output.to_flat_vec::<f32>().unwrap();
    for val in &v {
        assert!(val.abs() < 1e-7);
    }
}

#[test]
fn test_conv3d_negative_weights() {
    // Negative kernel weights
    let input = make_tensor(&[1.0; 8], &[1, 1, 2, 2, 2]);
    let weight = make_tensor(&[-1.0], &[1, 1, 1, 1, 1]);
    let conv = Conv3d::new(weight, None, Conv3dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    let v = output.to_flat_vec::<f32>().unwrap();
    for val in &v {
        assert!((*val + 1.0).abs() < 1e-5);
    }
}
