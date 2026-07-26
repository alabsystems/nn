#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::pool2d_out_len;
use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;
use crate::DType;

// -- pool2d_out_len -----------------------------------------------------------

#[test]
fn test_pool2d_out_len_basic() {
    // input=4, kernel=2, padding=0, stride=2 → (4-2)/2+1 = 2
    assert_eq!(pool2d_out_len(4, 2, 0, 2, false).unwrap(), 2);
}

#[test]
fn test_pool2d_out_len_with_padding() {
    // input=4, kernel=3, padding=1, stride=1 → (4+2-3)/1+1 = 4
    assert_eq!(pool2d_out_len(4, 3, 1, 1, false).unwrap(), 4);
}

#[test]
fn test_pool2d_out_len_ceil_mode() {
    // input=5, kernel=3, padding=0, stride=2 → floor: (5-3)/2+1=2, ceil: (5-3+1)/2+1=2
    assert_eq!(pool2d_out_len(5, 3, 0, 2, false).unwrap(), 2);
    assert_eq!(pool2d_out_len(5, 3, 0, 2, true).unwrap(), 2);
    // input=7, kernel=3, padding=0, stride=2 → floor: (7-3)/2+1=3, ceil: (7-3+1)/2+1=3
    assert_eq!(pool2d_out_len(7, 3, 0, 2, false).unwrap(), 3);
    assert_eq!(pool2d_out_len(7, 3, 0, 2, true).unwrap(), 3);
    // input=6, kernel=3, padding=0, stride=4 → floor: (6-3)/4+1=1, ceil: (6-3+3)/4+1=2
    assert_eq!(pool2d_out_len(6, 3, 0, 4, false).unwrap(), 1);
    assert_eq!(pool2d_out_len(6, 3, 0, 4, true).unwrap(), 2);
}

#[test]
fn test_pool2d_out_len_zero_kernel() {
    assert!(pool2d_out_len(4, 0, 0, 1, false).is_err());
}

#[test]
fn test_pool2d_out_len_zero_stride() {
    assert!(pool2d_out_len(4, 2, 0, 0, false).is_err());
}

#[test]
fn test_pool2d_out_len_kernel_too_large() {
    // input=2, kernel=5, padding=0 → padded 2 < 5
    assert!(pool2d_out_len(2, 5, 0, 1, false).is_err());
}

// -- max_pool1d ---------------------------------------------------------------

#[test]
fn test_max_pool1d_basic() {
    // 1x1x6, kernel=2, stride=2 → 1x1x3
    let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let x = DynTensor::from_vec(data, &[1, 1, 6], &cpu()).unwrap();
    let y = x.max_pool1d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 5.0, 6.0]);
}

#[test]
fn test_max_pool1d_stride_1() {
    // 1x1x5, kernel=3, stride=1 → 1x1x3
    let data = vec![1.0, 5.0, 2.0, 4.0, 3.0];
    let x = DynTensor::from_vec(data, &[1, 1, 5], &cpu()).unwrap();
    let y = x.max_pool1d(3, 1, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // max(1,5,2)=5, max(5,2,4)=5, max(2,4,3)=4
    assert_eq!(vals, vec![5.0, 5.0, 4.0]);
}

#[test]
fn test_max_pool1d_with_padding() {
    // 1x1x3, kernel=3, stride=1, padding=1 → 1x1x3
    let data = vec![2.0, 5.0, 3.0];
    let x = DynTensor::from_vec(data, &[1, 1, 3], &cpu()).unwrap();
    let y = x.max_pool1d(3, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // With padding=1: [0, 2, 5, 3, 0]
    // max(0,2,5)=5, max(2,5,3)=5, max(5,3,0)=5
    assert_eq!(vals, vec![5.0, 5.0, 5.0]);
}

#[test]
fn test_max_pool1d_multi_channel() {
    // 1x2x4, kernel=2, stride=2 → 1x2x2
    let data = vec![
        1.0, 4.0, 2.0, 3.0, // channel 0
        10.0, 7.0, 8.0, 9.0, // channel 1
    ];
    let x = DynTensor::from_vec(data, &[1, 2, 4], &cpu()).unwrap();
    let y = x.max_pool1d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Ch 0: max(1,4)=4, max(2,3)=3
    // Ch 1: max(10,7)=10, max(8,9)=9
    assert_eq!(vals, vec![4.0, 3.0, 10.0, 9.0]);
}

#[test]
fn test_max_pool1d_batch() {
    // 2x1x4, kernel=2, stride=2 → 2x1x2
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        40.0, 30.0, 20.0, 10.0, // batch 1
    ];
    let x = DynTensor::from_vec(data, &[2, 1, 4], &cpu()).unwrap();
    let y = x.max_pool1d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[2, 1, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![2.0, 4.0, 40.0, 20.0]);
}

#[test]
fn test_max_pool1d_wrong_rank() {
    let x = DynTensor::zeros(&[1, 4], DType::F32, &cpu()).unwrap();
    assert!(x.max_pool1d(2, 2, 0).is_err());
}

// -- max_pool2d ---------------------------------------------------------------

#[test]
fn test_max_pool2d_basic() {
    // 1x1x4x4, kernel=2, stride=2 → 1x1x2x2
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap();
    let y = x.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // max of each 2x2 block: [6, 8, 14, 16]
    assert_eq!(vals, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn test_max_pool2d_stride_1() {
    // 1x1x3x3, kernel=2, stride=1 → 1x1x2x2
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 3, 3], &cpu()).unwrap();
    let y = x.max_pool2d(2, 1, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 6.0, 8.0, 9.0]);
}

#[test]
fn test_max_pool2d_with_padding() {
    // 1x1x2x2, kernel=3, stride=1, padding=1 → 1x1x2x2
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let x = DynTensor::from_vec(data, &[1, 1, 2, 2], &cpu()).unwrap();
    let y = x.max_pool2d(3, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // With padding=1, each output position sees a 3x3 window centered on it.
    // All windows overlap the entire 2x2 input, so max is always 4.0.
    assert_eq!(vals, vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_max_pool2d_multi_channel() {
    // 1x2x4x4, kernel=2, stride=2 → 1x2x2x2
    let mut data = Vec::with_capacity(32);
    for i in 0..32 {
        data.push(i as f32);
    }
    let x = DynTensor::from_vec(data, &[1, 2, 4, 4], &cpu()).unwrap();
    let y = x.max_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Channel 0: max of 2x2 blocks in 0..15
    assert_eq!(vals[0], 5.0); // max(0,1,4,5)
    assert_eq!(vals[1], 7.0); // max(2,3,6,7)
    assert_eq!(vals[2], 13.0); // max(8,9,12,13)
    assert_eq!(vals[3], 15.0); // max(10,11,14,15)
                               // Channel 1: offset by 16
    assert_eq!(vals[4], 21.0); // max(16,17,20,21)
    assert_eq!(vals[5], 23.0); // max(18,19,22,23)
}

#[test]
fn test_max_pool2d_batch() {
    // 2x1x2x2, kernel=2, stride=1 → 2x1x1x1
    let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let x = DynTensor::from_vec(data, &[2, 1, 2, 2], &cpu()).unwrap();
    let y = x.max_pool2d(2, 1, 0).unwrap();
    assert_eq!(y.dims(), &[2, 1, 1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![4.0, 40.0]);
}

#[test]
fn test_max_pool2d_wrong_rank() {
    let x = DynTensor::zeros(&[1, 3, 4], DType::F32, &cpu()).unwrap();
    assert!(x.max_pool2d(2, 2, 0).is_err());
}

// -- avg_pool2d ---------------------------------------------------------------

#[test]
fn test_avg_pool2d_basic() {
    // 1x1x4x4, kernel=2, stride=2 → 1x1x2x2
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap();
    let y = x.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // avg of each 2x2 block: [(1+2+5+6)/4, (3+4+7+8)/4, ...]
    assert_eq!(vals, vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn test_avg_pool2d_stride_1() {
    // 1x1x3x3, kernel=2, stride=1 → 1x1x2x2
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 3, 3], &cpu()).unwrap();
    let y = x.avg_pool2d(2, 1, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // avg(1,2,4,5)=3.0, avg(2,3,5,6)=4.0, avg(4,5,7,8)=6.0, avg(5,6,8,9)=7.0
    assert_eq!(vals, vec![3.0, 4.0, 6.0, 7.0]);
}

#[test]
fn test_avg_pool2d_with_padding() {
    // 1x1x2x2, kernel=2, stride=1, padding=1 → 1x1x3x3
    // With padding=1, the input [1,2,3,4] is surrounded by zeros:
    // 0 0 0 0
    // 0 1 2 0
    // 0 3 4 0
    // 0 0 0 0
    // Each 2x2 window — count_include_pad=false (our default): divide by actual count.
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let x = DynTensor::from_vec(data, &[1, 1, 2, 2], &cpu()).unwrap();
    let y = x.avg_pool2d(2, 1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Top-left: only 1 value in bounds → 1/1=1.0
    assert!((vals[0] - 1.0).abs() < 1e-6);
    // Top-center: 1,2 in bounds → (1+2)/2=1.5
    assert!((vals[1] - 1.5).abs() < 1e-6);
    // Top-right: only 2 → 2/1=2.0
    assert!((vals[2] - 2.0).abs() < 1e-6);
    // Middle-left: 1,3 → (1+3)/2=2.0
    assert!((vals[3] - 2.0).abs() < 1e-6);
    // Center: all 4 → (1+2+3+4)/4=2.5
    assert!((vals[4] - 2.5).abs() < 1e-6);
}

#[test]
fn test_avg_pool2d_wrong_rank() {
    let x = DynTensor::zeros(&[4, 4], DType::F32, &cpu()).unwrap();
    assert!(x.avg_pool2d(2, 2, 0).is_err());
}

#[test]
fn test_avg_pool2d_multi_channel() {
    // 1x2x4x4, kernel=2, stride=2 → 1x2x2x2
    // Channel 0: values 0..16, Channel 1: values 16..32
    let mut data = Vec::with_capacity(32);
    for i in 0..32 {
        data.push(i as f32);
    }
    let x = DynTensor::from_vec(data, &[1, 2, 4, 4], &cpu()).unwrap();
    let y = x.avg_pool2d(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 2, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Channel 0: avg of 2x2 blocks
    // Block [0,0]: (0+1+4+5)/4 = 2.5
    assert!((vals[0] - 2.5).abs() < 1e-6);
    // Block [0,1]: (2+3+6+7)/4 = 4.5
    assert!((vals[1] - 4.5).abs() < 1e-6);
    // Block [1,0]: (8+9+12+13)/4 = 10.5
    assert!((vals[2] - 10.5).abs() < 1e-6);
    // Block [1,1]: (10+11+14+15)/4 = 12.5
    assert!((vals[3] - 12.5).abs() < 1e-6);
    // Channel 1: offset by 16
    // Block [0,0]: (16+17+20+21)/4 = 18.5
    assert!((vals[4] - 18.5).abs() < 1e-6);
    // Block [0,1]: (18+19+22+23)/4 = 20.5
    assert!((vals[5] - 20.5).abs() < 1e-6);
    // Block [1,0]: (24+25+28+29)/4 = 26.5
    assert!((vals[6] - 26.5).abs() < 1e-6);
    // Block [1,1]: (26+27+30+31)/4 = 28.5
    assert!((vals[7] - 28.5).abs() < 1e-6);
}

// -- adaptive_avg_pool2d ------------------------------------------------------

#[test]
fn test_adaptive_avg_pool2d_identity() {
    // Same input and output size → identity
    let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 3, 3], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(3, 3).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_adaptive_avg_pool2d_to_1x1() {
    // Global average pooling: output is 1x1
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // mean of 1..16 = 8.5
    assert!((vals[0] - 8.5).abs() < 1e-6);
}

#[test]
fn test_adaptive_avg_pool2d_downscale() {
    // 1x1x4x4 → 1x1x2x2
    #[rustfmt::skip]
    let data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let x = DynTensor::from_vec(data, &[1, 1, 4, 4], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(2, 2).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Window [0:2, 0:2] = avg(1,2,5,6) = 3.5
    assert!((vals[0] - 3.5).abs() < 1e-6);
    // Window [0:2, 2:4] = avg(3,4,7,8) = 5.5
    assert!((vals[1] - 5.5).abs() < 1e-6);
    // Window [2:4, 0:2] = avg(9,10,13,14) = 11.5
    assert!((vals[2] - 11.5).abs() < 1e-6);
    // Window [2:4, 2:4] = avg(11,12,15,16) = 13.5
    assert!((vals[3] - 13.5).abs() < 1e-6);
}

#[test]
fn test_adaptive_avg_pool2d_multi_batch_channel() {
    // 2x2x4x4 → 2x2x1x1
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[2, 2, 4, 4], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(1, 1).unwrap();
    assert_eq!(y.dims(), &[2, 2, 1, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Batch 0, channel 0: mean(0..15) = 7.5
    assert!((vals[0] - 7.5).abs() < 1e-6);
    // Batch 0, channel 1: mean(16..31) = 23.5
    assert!((vals[1] - 23.5).abs() < 1e-6);
    // Batch 1, channel 0: mean(32..47) = 39.5
    assert!((vals[2] - 39.5).abs() < 1e-6);
    // Batch 1, channel 1: mean(48..63) = 55.5
    assert!((vals[3] - 55.5).abs() < 1e-6);
}

#[test]
fn test_adaptive_avg_pool2d_zero_output() {
    let x = DynTensor::zeros(&[1, 1, 4, 4], DType::F32, &cpu()).unwrap();
    assert!(x.adaptive_avg_pool2d(0, 1).is_err());
    assert!(x.adaptive_avg_pool2d(1, 0).is_err());
}

#[test]
fn test_adaptive_avg_pool2d_non_square() {
    // 1x1x6x4 → 1x1x3x2
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 6, 4], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(3, 2).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 2]);
}

#[test]
fn test_adaptive_avg_pool2d_upsample_no_empty_windows() {
    // Regression test: when out_h > in_h, no window should be empty.
    // Previous bug: floor-based window calculation produced empty [start, end)
    // ranges when out > in, yielding zeros instead of averaging valid elements.
    // 1x1x2x2, values [[1,2],[3,4]], upsample to 5x5
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(5, 5).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5, 5]);
    let v = y.to_flat_vec::<f32>().unwrap();
    // Every output element must be a valid average (no zeros from empty windows)
    for (i, &val) in v.iter().enumerate() {
        assert!(
            (1.0..=4.0).contains(&val),
            "adaptive_avg_pool2d upsample: element {i} = {val}, expected in [1, 4]"
        );
    }
}

#[test]
fn test_adaptive_avg_pool2d_upsample_1x1_to_3x3() {
    // Edge case: single pixel upsampled to 3x3 should all be the same value.
    let x = DynTensor::from_vec(vec![7.0], &[1, 1, 1, 1], &cpu()).unwrap();
    let y = x.adaptive_avg_pool2d(3, 3).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    let v = y.to_flat_vec::<f32>().unwrap();
    for (i, &val) in v.iter().enumerate() {
        assert_eq!(val, 7.0, "element {i} should be 7.0, got {val}");
    }
}
