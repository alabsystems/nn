// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Conv3d DynTensor operation.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DynTensor, TensorError};

// ---------------------------------------------------------------------------
// Basic forward: identity / scaling kernels
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_basic_identity_kernel() {
    // [1, 1, 2, 2, 2] * [1, 1, 1, 1, 1] stride=1 pad=0 → [1, 1, 2, 2, 2]
    // 1x1x1 kernel = identity (scaled by kernel value)
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[2.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2, 2]);
    let v = y.to_flat_vec::<f32>().unwrap();
    let expected: Vec<f32> = (1..=8).map(|x| x as f32 * 2.0).collect();
    assert_eq!(v, expected);
}

#[test]
fn test_conv3d_all_ones_kernel() {
    // [1, 1, 3, 3, 3] * [1, 1, 2, 2, 2] stride=1 pad=0 → [1, 1, 2, 2, 2]
    // All-ones kernel sums the 2x2x2 sub-cubes.
    let data: Vec<f32> = (0..27).map(|x| x as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 3, 3, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2, 2]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // Sum of 2x2x2 sub-cube starting at (0,0,0): 0+1+3+4+9+10+12+13 = 52
    // Starting at (0,0,1): 1+2+4+5+10+11+13+14 = 60
    // Starting at (0,1,0): 3+4+6+7+12+13+15+16 = 76
    // Starting at (0,1,1): 4+5+7+8+13+14+16+17 = 84
    // Starting at (1,0,0): 9+10+12+13+18+19+21+22 = 124
    // Starting at (1,0,1): 10+11+13+14+19+20+22+23 = 132
    // Starting at (1,1,0): 12+13+15+16+21+22+24+25 = 148
    // Starting at (1,1,1): 13+14+16+17+22+23+25+26 = 156
    assert_eq!(v, vec![52.0, 60.0, 76.0, 84.0, 124.0, 132.0, 148.0, 156.0]);
}

#[test]
fn test_conv3d_3x3x3_kernel_values() {
    // 3x3x3 input, 3x3x3 all-ones kernel → single output = sum of all elements
    let data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 3, 3, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 27], &[1, 1, 3, 3, 3], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // sum(1..27) = 27*28/2 = 378
    assert!((v[0] - 378.0).abs() < 1e-4);
}

#[test]
fn test_conv3d_5x5x5_kernel() {
    // 5x5x5 input, 5x5x5 all-ones kernel → 1x1x1 output = sum
    let data: Vec<f32> = (1..=125).map(|x| x as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 5, 5, 5], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 125], &[1, 1, 5, 5, 5], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // sum(1..125) = 125*126/2 = 7875
    assert!((v[0] - 7875.0).abs() < 1e-3);
}

#[test]
fn test_conv3d_non_cubic_kernel() {
    // kernel [1, 2, 3] on [1, 1, 2, 3, 4] input (all 1.0)
    // out_d=(2-1)/1+1=2, out_h=(3-2)/1+1=2, out_w=(4-3)/1+1=2
    let x = DynTensor::new(&[1.0f32; 24], &[1, 1, 2, 3, 4], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 6], &[1, 1, 1, 2, 3], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2, 2]);
    // Each output = sum of 1*2*3=6 ones = 6.0
    let v = y.to_flat_vec::<f32>().unwrap();
    for val in &v {
        assert!((*val - 6.0).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_with_padding() {
    // [1, 1, 1, 1, 1] * [1, 1, 1, 1, 1] stride=1 pad=1 → [1, 1, 3, 3, 3]
    let x = DynTensor::new(&[5.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // Only the center element (1,1,1) overlaps with input → 5.0; all others 0.0
    let mut expected = vec![0.0f32; 27];
    expected[13] = 5.0; // index (1,1,1) in 3x3x3
    assert_eq!(v, expected);
}

#[test]
fn test_conv3d_same_padding_preserves_shape() {
    // "same" padding: pad=1, kernel=3, stride=1 → output = input shape
    let x = DynTensor::new(&vec![1.0f32; 64], &[1, 1, 4, 4, 4], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 27], &[1, 1, 3, 3, 3], &cpu()).unwrap();
    let y = x.conv3d(&k, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4, 4]);
}

#[test]
fn test_conv3d_asymmetric_padding() {
    // pad=[0, 1, 2] on [1, 1, 2, 2, 2] with 1x1x1 kernel
    // out_d=2, out_h=(2+2)/1+1=4 (wait: (2+2*1-1)/1+1=4), out_w=(2+2*2-1)/1+1=6
    let x = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 1, 2], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 4, 6]);
}

// ---------------------------------------------------------------------------
// Stride
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_stride() {
    // [1, 1, 4, 4, 4] * [1, 1, 2, 2, 2] stride=[2,2,2] pad=0 → [1, 1, 2, 2, 2]
    let data: Vec<f32> = (0..64).map(|x| x as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 4, 4, 4], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [2, 2, 2], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2, 2]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v.len(), 8);
}

#[test]
fn test_conv3d_asymmetric_stride() {
    // stride=[1, 2, 3] on [1, 1, 3, 5, 7] with 1x1x1 kernel
    // out_d=3, out_h=(5-1)/2+1=3, out_w=(7-1)/3+1=3
    let x = DynTensor::new(&vec![1.0f32; 105], &[1, 1, 3, 5, 7], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 2, 3], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_large_stride_single_output() {
    // stride so large only 1 output per dim
    let x = DynTensor::new(&vec![2.0f32; 64], &[1, 1, 4, 4, 4], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [3, 3, 3], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // 8 elements * 2.0 = 16.0
    assert!((v[0] - 16.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Dilation
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_dilation() {
    // [1, 1, 3, 1, 1] * [1, 1, 2, 1, 1] dilation=[2,1,1] stride=1 pad=0
    // effective kernel depth = (2-1)*2+1 = 3, so output depth = (3-3)/1+1 = 1
    let x = DynTensor::new(&[1.0, 2.0, 3.0f32], &[1, 1, 3, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0, 1.0f32], &[1, 1, 2, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [2, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // kernel[0]*x[0] + kernel[1]*x[2] = 1*1 + 1*3 = 4
    assert_eq!(v, vec![4.0]);
}

#[test]
fn test_conv3d_dilation_uniform() {
    // dilation=2 on [1, 1, 5, 5, 5] with 2x2x2 kernel
    // eff_k = (2-1)*2+1 = 3; out = (5-3)/1+1 = 3
    let data: Vec<f32> = (0..125).map(|x| x as f32).collect();
    let x = DynTensor::new(&data, &[1, 1, 5, 5, 5], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [2, 2, 2], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3, 3]);

    // First output: picks (d,h,w) in {0,2}^3
    // Indices: 0, 2, 10, 12, 50, 52, 60, 62
    let expected = 0.0 + 2.0 + 10.0 + 12.0 + 50.0 + 52.0 + 60.0 + 62.0;
    let v = y.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - expected).abs() < 1e-4);
}

#[test]
fn test_conv3d_asymmetric_dilation() {
    // dilation=[1, 2, 3] on [1, 1, 3, 5, 7] with 2x2x2 kernel
    // eff_d=2, eff_h=3, eff_w=4
    // out_d=(3-2)/1+1=2, out_h=(5-3)/1+1=3, out_w=(7-4)/1+1=4
    let x = DynTensor::new(&vec![1.0f32; 105], &[1, 1, 3, 5, 7], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 2, 3], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Multi-channel
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_multi_channel() {
    // [1, 2, 1, 1, 1] * [3, 2, 1, 1, 1] stride=1 pad=0 → [1, 3, 1, 1, 1]
    // 2 in channels, 3 out channels, 1x1x1 kernel → linear combination
    let x = DynTensor::new(&[2.0f32, 3.0], &[1, 2, 1, 1, 1], &cpu()).unwrap();
    // Kernel: OC0=[1,0], OC1=[0,1], OC2=[1,1]
    let k = DynTensor::new(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0f32], &[3, 2, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 3, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![2.0, 3.0, 5.0]);
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_groups() {
    // [1, 4, 1, 1, 1] * [4, 2, 1, 1, 1] groups=2 → [1, 4, 1, 1, 1]
    // Group 0: in_ch [0,1] → out_ch [0,1], Group 1: in_ch [2,3] → out_ch [2,3]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0f32], &[1, 4, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0f32],
        &[4, 2, 1, 1, 1],
        &cpu(),
    )
    .unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 2).unwrap();
    assert_eq!(y.dims(), &[1, 4, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_conv3d_depthwise() {
    // Depthwise: groups = in_channels = out_channels
    // 3 channels, each independently scaled
    let x = DynTensor::new(&[1.0, 2.0, 3.0f32], &[1, 3, 1, 1, 1], &cpu()).unwrap();
    // weight [3, 1, 1, 1, 1]: ch0*2, ch1*3, ch2*4
    let k = DynTensor::new(&[2.0, 3.0, 4.0f32], &[3, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 3).unwrap();
    assert_eq!(y.dims(), &[1, 3, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![2.0, 6.0, 12.0]);
}

#[test]
fn test_conv3d_groups_not_dividing_in_channels() {
    // in_ch=3 with groups=2 → 3 % 2 != 0 → error
    let x = DynTensor::new(&[1.0f32; 3], &[1, 3, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 2], &[2, 1, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 2)
        .unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "groups",
            ..
        }
    ));
}

#[test]
fn test_conv3d_groups_not_dividing_out_channels() {
    // out_ch=3 with groups=2 → 3 % 2 != 0 → error
    let x = DynTensor::new(&[1.0f32; 4], &[1, 4, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 6], &[3, 2, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 2)
        .unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "groups",
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// Batch dimension
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_batch() {
    // [2, 1, 1, 1, 1] * [1, 1, 1, 1, 1] → [2, 1, 1, 1, 1]
    let x = DynTensor::new(&[3.0, 7.0f32], &[2, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[2.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[2, 1, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![6.0, 14.0]);
}

#[test]
fn test_conv3d_batch_multichannel() {
    // 2 batches, 2 in channels, 1 out channel, 1x1x1 kernel [1, 1]
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0f32], // b0: ch0=1, ch1=2; b1: ch0=3, ch1=4
        &[2, 2, 1, 1, 1],
        &cpu(),
    )
    .unwrap();
    let k = DynTensor::new(&[1.0, 1.0f32], &[1, 2, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[2, 1, 1, 1, 1]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // b0: 1+2=3, b1: 3+4=7
    assert_eq!(v, vec![3.0, 7.0]);
}

// ---------------------------------------------------------------------------
// Combined parameters
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_stride_padding_combined() {
    // Input [1, 1, 5, 5, 5], kernel 3x3x3, stride=2, pad=1
    // out = (5+2-3)/2+1 = 3
    let x = DynTensor::new(&vec![1.0f32; 125], &[1, 1, 5, 5, 5], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 27], &[1, 1, 3, 3, 3], &cpu()).unwrap();
    let y = x.conv3d(&k, [1, 1, 1], [2, 2, 2], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_dilation_padding_combined() {
    // Input [1, 1, 3, 3, 3], kernel 2x2x2, dilation=2, pad=1
    // eff_k=3; out=(3+2-3)/1+1=3
    let x = DynTensor::new(&[1.0f32; 27], &[1, 1, 3, 3, 3], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let y = x.conv3d(&k, [1, 1, 1], [1, 1, 1], [2, 2, 2], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3, 3]);
}

#[test]
fn test_conv3d_all_params_combined() {
    // batch=2, in_ch=4, groups=2, kernel 3x3x3, stride=2, pad=1, dil=1
    // out = (8+2-3)/2+1 = 4 per dim
    let n = 2 * 4 * 8 * 8 * 8;
    let x = DynTensor::new(&vec![1.0f32; n], &[2, 4, 8, 8, 8], &cpu()).unwrap();
    let k = DynTensor::new(&vec![1.0f32; 4 * 2 * 27], &[4, 2, 3, 3, 3], &cpu()).unwrap();
    let y = x.conv3d(&k, [1, 1, 1], [2, 2, 2], [1, 1, 1], 2).unwrap();
    assert_eq!(y.dims(), &[2, 4, 4, 4, 4]);
}

// ---------------------------------------------------------------------------
// conv3d_out_len
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_out_len_basic() {
    use super::conv3d::conv3d_out_len;
    // Standard formula: (input + 2*pad - dilation*(kernel-1) - 1) / stride + 1
    assert_eq!(conv3d_out_len(4, 2, 0, 1, 1).unwrap(), 3);
    assert_eq!(conv3d_out_len(4, 2, 0, 2, 1).unwrap(), 2);
    assert_eq!(conv3d_out_len(4, 2, 1, 1, 1).unwrap(), 5);
    assert_eq!(conv3d_out_len(5, 2, 0, 1, 2).unwrap(), 3);
}

#[test]
fn test_conv3d_out_len_various() {
    use super::conv3d::conv3d_out_len;
    // 1x1 kernel → output = input (no shrinkage)
    assert_eq!(conv3d_out_len(10, 1, 0, 1, 1).unwrap(), 10);
    // Large stride
    assert_eq!(conv3d_out_len(10, 1, 0, 5, 1).unwrap(), 2);
    // Input equals kernel
    assert_eq!(conv3d_out_len(3, 3, 0, 1, 1).unwrap(), 1);
    // Padding grows output
    assert_eq!(conv3d_out_len(1, 1, 2, 1, 1).unwrap(), 5);
}

#[test]
fn test_conv3d_out_len_zero_stride() {
    use super::conv3d::conv3d_out_len;
    let err = conv3d_out_len(10, 3, 0, 0, 1).unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "stride",
            ..
        }
    ));
}

#[test]
fn test_conv3d_out_len_zero_dilation() {
    use super::conv3d::conv3d_out_len;
    let err = conv3d_out_len(10, 3, 0, 1, 0).unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "dilation",
            ..
        }
    ));
}

#[test]
fn test_conv3d_out_len_zero_kernel() {
    use super::conv3d::conv3d_out_len;
    let err = conv3d_out_len(10, 0, 0, 1, 1).unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "kernel_size",
            ..
        }
    ));
}

#[test]
fn test_conv3d_out_len_kernel_too_large() {
    use super::conv3d::conv3d_out_len;
    // padded input (1+0) < effective kernel (3)
    let err = conv3d_out_len(1, 3, 0, 1, 1).unwrap_err();
    assert!(matches!(err, TensorError::InvalidShape(_)));
}

// ---------------------------------------------------------------------------
// Error cases: input/kernel validation
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_wrong_input_rank() {
    // Input is 4D, not 5D
    let x = DynTensor::new(&[1.0f32; 16], &[1, 1, 4, 4], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1)
        .unwrap_err();
    assert!(matches!(err, TensorError::RankMismatch { expected: 5, .. }));
}

#[test]
fn test_conv3d_wrong_kernel_rank() {
    // Kernel is 4D, not 5D
    let x = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1)
        .unwrap_err();
    assert!(matches!(err, TensorError::RankMismatch { expected: 5, .. }));
}

#[test]
fn test_conv3d_channel_mismatch() {
    // Input has 2 channels but kernel expects 1 (groups=1)
    let x = DynTensor::new(&[1.0f32; 2], &[1, 2, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    assert!(x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).is_err());
}

#[test]
fn test_conv3d_zero_groups() {
    let x = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 0)
        .unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "groups",
            ..
        }
    ));
}

#[test]
fn test_conv3d_zero_stride_rejected() {
    let x = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [0, 1, 1], [1, 1, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "stride[0]",
            ..
        }
    ));
}

#[test]
fn test_conv3d_zero_dilation_rejected() {
    let x = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let err = x
        .conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 0, 1], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        TensorError::ConvParameterInvalid {
            param: "dilation[1]",
            ..
        }
    ));
}

#[test]
fn test_conv3d_kernel_larger_than_input() {
    // Input [1,1,2,2,2] with kernel [1,1,3,3,3] and no padding → error
    let x = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 27], &[1, 1, 3, 3, 3], &cpu()).unwrap();
    assert!(x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).is_err());
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_single_element() {
    // Minimal: 1x1x1x1x1 → 1x1x1x1x1
    let x = DynTensor::new(&[7.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[3.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1, 1]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![21.0]);
}

#[test]
fn test_conv3d_zero_weight_zero_output() {
    let x = DynTensor::new(&[5.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[0.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    for val in y.to_flat_vec::<f32>().unwrap() {
        assert!(val.abs() < 1e-7);
    }
}

#[test]
fn test_conv3d_zero_input_zero_output() {
    let x = DynTensor::new(&[0.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[5.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    for val in y.to_flat_vec::<f32>().unwrap() {
        assert!(val.abs() < 1e-7);
    }
}

#[test]
fn test_conv3d_negative_weights() {
    let x = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let k = DynTensor::new(&[-2.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d(&k, [0, 0, 0], [1, 1, 1], [1, 1, 1], 1).unwrap();
    for val in y.to_flat_vec::<f32>().unwrap() {
        assert!((val + 2.0).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// conv3d_with (Conv3dParams)
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_with_params() {
    use crate::dyn_tensor::conv::Conv3dParams;
    let x = DynTensor::new(&[1.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let k = DynTensor::new(&[3.0f32], &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let y = x.conv3d_with(&k, Conv3dParams::default()).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![3.0]);
}

#[test]
fn test_conv3d_with_params_custom() {
    use crate::dyn_tensor::conv::Conv3dParams;
    // Same as stride test but via conv3d_with
    let x = DynTensor::new(&vec![1.0f32; 64], &[1, 1, 4, 4, 4], &cpu()).unwrap();
    let k = DynTensor::new(&[1.0f32; 8], &[1, 1, 2, 2, 2], &cpu()).unwrap();
    let params = Conv3dParams {
        padding: [0, 0, 0],
        stride: [2, 2, 2],
        dilation: [1, 1, 1],
        groups: 1,
    };
    let y = x.conv3d_with(&k, params).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2, 2]);
}
