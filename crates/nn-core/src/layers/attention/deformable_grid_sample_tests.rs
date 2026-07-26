#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! grid_sample tests for deformable attention.
//!
//! Extracted from `deformable_tests.rs` for file size compliance.

use crate::dyn_tensor::DynTensor;
use crate::dyn_tensor::GridSamplePaddingMode;
use crate::Device;

#[test]
fn test_grid_sample_identity() {
    // 1x1x3x3 feature map with known values
    let input_data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let input = DynTensor::from_vec(input_data, &[1, 1, 3, 3], &Device::Cpu).expect("input");

    // Grid that samples at exact pixel positions (identity mapping).
    // For align_corners=true: pixel i maps to grid = 2*i/(size-1) - 1
    // For 3x3: grid values are -1, 0, 1
    let mut grid_data = Vec::new();
    for y in 0..3 {
        for x in 0..3 {
            let gx = x as f32 - 1.0; // -1, 0, 1
            let gy = y as f32 - 1.0; // -1, 0, 1
            grid_data.push(gx);
            grid_data.push(gy);
        }
    }
    let grid = DynTensor::from_vec(grid_data, &[1, 3, 3, 2], &Device::Cpu).expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Zeros, true)
        .expect("grid_sample");
    assert_eq!(result.dims(), &[1, 1, 3, 3]);

    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");
    for (i, &val) in out.iter().enumerate() {
        assert!(
            (val - i as f32).abs() < 1e-5,
            "position {i}: expected {}, got {}",
            i as f32,
            val
        );
    }
}

#[test]
fn test_grid_sample_center_point() {
    // 1x1x2x2 with values [1, 2, 3, 4]
    let input =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).expect("input");

    // Sample at center (0, 0) with align_corners=true → center between all 4 pixels
    let grid = DynTensor::from_vec(vec![0.0, 0.0], &[1, 1, 1, 2], &Device::Cpu).expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Zeros, true)
        .expect("grid_sample");
    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");
    // Center of 2x2 with align_corners=true: (0+1)/2 * (2-1) = 0.5 for both x and y
    // Bilinear: 0.25*(1+2+3+4) = 2.5
    assert!((out[0] - 2.5).abs() < 1e-5, "expected 2.5, got {}", out[0]);
}

#[test]
fn test_grid_sample_out_of_bounds_zeros() {
    let input = DynTensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2], &Device::Cpu).expect("input");

    // Sample far outside: grid = (5, 5) → way out of bounds
    let grid = DynTensor::from_vec(vec![5.0, 5.0], &[1, 1, 1, 2], &Device::Cpu).expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Zeros, true)
        .expect("grid_sample");
    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");
    assert!(
        (out[0]).abs() < 1e-5,
        "expected 0.0 for out-of-bounds, got {}",
        out[0]
    );
}

#[test]
fn test_grid_sample_border_padding() {
    let input =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).expect("input");

    // Sample far outside with border padding: should clamp to border pixel
    let grid = DynTensor::from_vec(vec![-5.0, -5.0], &[1, 1, 1, 2], &Device::Cpu).expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Border, true)
        .expect("grid_sample");
    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");
    // Top-left corner pixel = 1.0
    assert!(
        (out[0] - 1.0).abs() < 1e-5,
        "expected 1.0 for border clamp, got {}",
        out[0]
    );
}

#[test]
fn test_grid_sample_multichannel() {
    // 2 channels: channel 0 is all 1.0, channel 1 is all 2.0
    let mut data = vec![1.0f32; 4];
    data.extend_from_slice(&[2.0f32; 4]);
    let input = DynTensor::from_vec(data, &[1, 2, 2, 2], &Device::Cpu).expect("input");

    // Sample at center
    let grid = DynTensor::from_vec(vec![0.0, 0.0], &[1, 1, 1, 2], &Device::Cpu).expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Zeros, true)
        .expect("grid_sample");
    assert_eq!(result.dims(), &[1, 2, 1, 1]);
    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!((out[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_grid_sample_nan_coordinates_produce_zeros() {
    // Input: 1 batch, 1 channel, 2x2 filled with 5.0
    let input = DynTensor::from_vec(vec![5.0f32; 4], &[1, 1, 2, 2], &Device::Cpu).expect("input");

    // Grid: 2 sample points — first valid (0,0), second has NaN x
    let grid = DynTensor::from_vec(vec![0.0, 0.0, f32::NAN, 0.0], &[1, 1, 2, 2], &Device::Cpu)
        .expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Zeros, true)
        .expect("grid_sample");
    assert_eq!(result.dims(), &[1, 1, 1, 2]);
    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");

    // Valid coordinate (0,0) center → interpolation of 5.0 values → ~5.0
    assert!(
        out[0].is_finite() && (out[0] - 5.0).abs() < 1e-4,
        "valid coord should produce ~5.0, got {}",
        out[0]
    );
    // NaN coordinate → should produce 0.0 (zero-padding), not NaN
    assert!(
        out[1].is_finite(),
        "NaN grid coord must not propagate NaN, got {}",
        out[1]
    );
    assert!(
        out[1].abs() < 1e-6,
        "NaN grid coord should produce 0.0 (zero-pad), got {}",
        out[1]
    );
}

#[test]
fn test_grid_sample_inf_border_mode_produces_finite() {
    // NaN.clamp() returns NaN, so Border mode was also vulnerable.
    let input = DynTensor::from_vec(vec![3.0f32; 4], &[1, 1, 2, 2], &Device::Cpu).expect("input");

    // Grid with +Inf coordinate
    let grid = DynTensor::from_vec(
        vec![f32::INFINITY, f32::NEG_INFINITY],
        &[1, 1, 1, 2],
        &Device::Cpu,
    )
    .expect("grid");

    let result = input
        .grid_sample(&grid, GridSamplePaddingMode::Border, false)
        .expect("grid_sample");
    let out = result.to_flat_vec::<f32>().expect("to_flat_vec_f32");
    assert!(
        out[0].is_finite(),
        "Inf grid coord with Border mode must produce finite, got {}",
        out[0]
    );
}
