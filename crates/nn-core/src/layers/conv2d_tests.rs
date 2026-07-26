// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for layers::Conv2d layer.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Conv2d, Conv2dConfig, Module};
use crate::Device;

fn make_tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::new(data, shape, &Device::Cpu).unwrap()
}

#[test]
fn test_conv2d_basic_no_padding() {
    // 1 batch, 1 channel, 3x3 input, 1 output channel, 2x2 kernel
    let input = make_tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    );
    let kernel = make_tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);

    let conv = Conv2d::new(kernel, None, Conv2dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    let vals: Vec<f32> = data.iter().copied().collect();
    // kernel is identity-like: picks (0,0) and (1,1) positions
    // (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
    // (0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
    // (1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
    // (1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
    assert_eq!(vals, vec![6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn test_conv2d_with_padding() {
    // 1 batch, 1 channel, 3x3 input, padding=1, 3x3 kernel (all ones)
    let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let input = make_tensor(&input, &[1, 1, 3, 3]);
    let kernel = make_tensor(&[1.0; 9], &[1, 1, 3, 3]);

    let config = Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let conv = Conv2d::new(kernel, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // With padding=1 and 3x3 kernel, output is same size as input: 3x3
    assert_eq!(output.dims(), &[1, 1, 3, 3]);
    // Center value: sum of all 9 elements = 45
    let data = output.as_cpu_f32().unwrap();
    let center = data[[0, 0, 1, 1]];
    assert!((center - 45.0).abs() < 1e-5);
}

#[test]
fn test_conv2d_with_stride() {
    // 1 batch, 1 channel, 4x4 input, 2x2 kernel, stride=2
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input = make_tensor(&input, &[1, 1, 4, 4]);
    let kernel = make_tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);

    let config = Conv2dConfig {
        stride: 2,
        ..Default::default()
    };
    let conv = Conv2d::new(kernel, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // Output size: (4 - 2) / 2 + 1 = 2 in each dim
    assert_eq!(output.dims(), &[1, 1, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    let vals: Vec<f32> = data.iter().copied().collect();
    // (0,0): 1+2+5+6 = 14
    // (0,1): 3+4+7+8 = 22
    // (1,0): 9+10+13+14 = 46
    // (1,1): 11+12+15+16 = 54
    assert_eq!(vals, vec![14.0, 22.0, 46.0, 54.0]);
}

#[test]
fn test_conv2d_with_bias() {
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let kernel = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let bias = make_tensor(&[10.0], &[1]);

    let conv = Conv2d::new(kernel, Some(bias), Conv2dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 1, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    let vals: Vec<f32> = data.iter().copied().collect();
    assert_eq!(vals, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_conv2d_multiple_channels() {
    // 1 batch, 2 input channels, 2x2 input, 3 output channels, 1x1 kernel
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]);
    // 3 output channels, 2 input channels, 1x1 kernel
    let kernel = make_tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2, 1, 1]);

    let conv = Conv2d::new(kernel, None, Conv2dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 3, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // OC0: 1*ic0 + 0*ic1 = ic0 = [1,2,3,4]
    // OC1: 0*ic0 + 1*ic1 = ic1 = [5,6,7,8]
    // OC2: 1*ic0 + 1*ic1 = [6,8,10,12]
    let oc0: Vec<f32> = (0..4).map(|i| data[[0, 0, i / 2, i % 2]]).collect();
    let oc1: Vec<f32> = (0..4).map(|i| data[[0, 1, i / 2, i % 2]]).collect();
    let oc2: Vec<f32> = (0..4).map(|i| data[[0, 2, i / 2, i % 2]]).collect();
    assert_eq!(oc0, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(oc1, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(oc2, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_conv2d_groups() {
    // 1 batch, 4 input channels, 2x2 input, 4 output channels, 1x1 kernel, groups=2
    // Group 0: ic[0,1] -> oc[0,1], Group 1: ic[2,3] -> oc[2,3]
    let mut input_data = vec![0.0f32; 4 * 2 * 2];
    // ic0: all 1s, ic1: all 2s, ic2: all 3s, ic3: all 4s
    for i in 0..4 {
        for j in 0..4 {
            input_data[i * 4 + j] = (i + 1) as f32;
        }
    }
    let input = make_tensor(&input_data, &[1, 4, 2, 2]);

    // 4 output channels, 2 input channels per group, 1x1 kernel
    // oc0 = 1*ic0 + 1*ic1, oc1 = 1*ic0 + 0*ic1
    // oc2 = 1*ic2 + 1*ic3, oc3 = 0*ic2 + 1*ic3
    let kernel = make_tensor(&[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], &[4, 2, 1, 1]);

    let config = Conv2dConfig {
        groups: 2,
        ..Default::default()
    };
    let conv = Conv2d::new(kernel, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 4, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // oc0: 1*1 + 1*2 = 3 (all positions)
    // oc1: 1*1 + 0*2 = 1 (all positions)
    // oc2: 1*3 + 1*4 = 7 (all positions)
    // oc3: 0*3 + 1*4 = 4 (all positions)
    assert!((data[[0, 0, 0, 0]] - 3.0).abs() < 1e-5);
    assert!((data[[0, 1, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((data[[0, 2, 0, 0]] - 7.0).abs() < 1e-5);
    assert!((data[[0, 3, 0, 0]] - 4.0).abs() < 1e-5);
}

#[test]
fn test_conv2d_dilation() {
    // 1 batch, 1 channel, 5x5 input, 2x2 kernel, dilation=2
    let input: Vec<f32> = (1..=25).map(|x| x as f32).collect();
    let input = make_tensor(&input, &[1, 1, 5, 5]);
    let kernel = make_tensor(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);

    let config = Conv2dConfig {
        dilation: 2,
        ..Default::default()
    };
    let conv = Conv2d::new(kernel, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // Effective kernel size: (2-1)*2+1 = 3, output: (5-3)/1+1 = 3 per dim
    assert_eq!(output.dims(), &[1, 1, 3, 3]);
    let data = output.as_cpu_f32().unwrap();
    // (0,0): input[0,0]*1 + input[0,2]*0 + input[2,0]*0 + input[2,2]*1
    //      = 1*1 + 3*0 + 11*0 + 13*1 = 14
    assert!((data[[0, 0, 0, 0]] - 14.0).abs() < 1e-5);
}

#[test]
fn test_conv2d_batch() {
    // 2 batches, 1 channel, 2x2 input, 1 output channel, 1x1 kernel (weight=2)
    let input = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 2]);
    let kernel = make_tensor(&[2.0], &[1, 1, 1, 1]);

    let conv = Conv2d::new(kernel, None, Conv2dConfig::default()).unwrap();
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 1, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    assert_eq!(data[[0, 0, 0, 0]], 2.0);
    assert_eq!(data[[0, 0, 1, 1]], 8.0);
    assert_eq!(data[[1, 0, 0, 0]], 10.0);
    assert_eq!(data[[1, 0, 1, 1]], 16.0);
}

#[test]
fn test_conv2d_invalid_rank() {
    let weight = make_tensor(&[1.0, 2.0, 3.0], &[3]);
    let result = Conv2d::new(weight, None, Conv2dConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_conv2d_zero_groups() {
    let weight = make_tensor(&[1.0], &[1, 1, 1, 1]);
    let config = Conv2dConfig {
        groups: 0,
        ..Default::default()
    };
    let result = Conv2d::new(weight, None, config);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_shape_mismatch() {
    // 1 batch, 3 channels, 4x4 input — but kernel expects 2 channels (groups=1)
    let input = make_tensor(&[1.0; 48], &[1, 3, 4, 4]);
    let kernel = make_tensor(&[1.0; 8], &[1, 2, 2, 2]);

    let conv = Conv2d::new(kernel, None, Conv2dConfig::default()).unwrap();
    let result = conv.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_conv2d_depthwise() {
    // Depthwise Conv2d: groups == in_channels == out_channels.
    // 1 batch, 3 channels, 3x3 input, 3 output channels, 2x2 kernel, groups=3.
    // Each channel convolved independently with its own 2x2 filter.
    //
    // Kernel shape: [out_channels=3, in_channels/groups=1, kH=2, kW=2]
    //
    // Channel 0: all 1s input, kernel [1, 0, 0, 0] → picks top-left of each window
    // Channel 1: all 2s input, kernel [0, 1, 0, 0] → picks top-right of each window
    // Channel 2: all 3s input, kernel [0, 0, 0, 1] → picks bottom-right of each window
    let mut input_data = vec![0.0f32; 3 * 3 * 3];
    input_data[..9].fill(1.0); // channel 0: all 1s
    input_data[9..18].fill(2.0); // channel 1: all 2s
    input_data[18..27].fill(3.0); // channel 2: all 3s
    let input = make_tensor(&input_data, &[1, 3, 3, 3]);

    // Kernel: [3, 1, 2, 2] — 3 output channels, 1 input channel per group
    #[rustfmt::skip]
    let kernel_data = [
        // Filter for channel 0: [1, 0, 0, 0]
        1.0, 0.0, 0.0, 0.0,
        // Filter for channel 1: [0, 1, 0, 0]
        0.0, 1.0, 0.0, 0.0,
        // Filter for channel 2: [0, 0, 0, 1]
        0.0, 0.0, 0.0, 1.0,
    ];
    let kernel = make_tensor(&kernel_data, &[3, 1, 2, 2]);

    let config = Conv2dConfig {
        groups: 3,
        ..Default::default()
    };
    let conv = Conv2d::new(kernel, None, config).unwrap();
    let output = conv.forward(&input).unwrap();

    // 3x3 input, 2x2 kernel, stride=1 → 2x2 output per channel
    assert_eq!(output.dims(), &[1, 3, 2, 2]);
    let data = output.as_cpu_f32().unwrap();
    // Channel 0: 1.0 * 1.0 = 1.0 at each output position
    for h in 0..2 {
        for w in 0..2 {
            assert!(
                (data[[0, 0, h, w]] - 1.0).abs() < 1e-5,
                "ch0[{h},{w}] = {}, expected 1.0",
                data[[0, 0, h, w]]
            );
        }
    }
    // Channel 1: 1.0 * 2.0 = 2.0 at each output position
    for h in 0..2 {
        for w in 0..2 {
            assert!(
                (data[[0, 1, h, w]] - 2.0).abs() < 1e-5,
                "ch1[{h},{w}] = {}, expected 2.0",
                data[[0, 1, h, w]]
            );
        }
    }
    // Channel 2: 1.0 * 3.0 = 3.0 at each output position
    for h in 0..2 {
        for w in 0..2 {
            assert!(
                (data[[0, 2, h, w]] - 3.0).abs() < 1e-5,
                "ch2[{h},{w}] = {}, expected 3.0",
                data[[0, 2, h, w]]
            );
        }
    }
}
