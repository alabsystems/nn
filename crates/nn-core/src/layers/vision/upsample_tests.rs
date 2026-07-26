#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::Module;
use crate::{DType, Device};

use super::{Upsample2d, Upsample2dToSize, UpsampleMode};

/// Helper: assert element-wise approximate equality.
fn assert_approx(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{ctx}[{i}]: got {a}, expected {e}, diff={}",
            (a - e).abs()
        );
    }
}

// -- Nearest-neighbor 2D upsample tests ---------------------------------------

#[test]
fn test_upsample_nearest_2d_scale1() {
    // Scale 1 is identity.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_2d(1, 1).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_upsample_nearest_2d_scale2() {
    // 2x2 input -> 4x4 output with scale=2.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_2d(2, 2).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), expected);
}

#[test]
fn test_upsample_nearest_2d_asymmetric_scale() {
    // 2x2 input -> 4x6 with scale_h=2, scale_w=3.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_2d(2, 3).unwrap();
    assert_eq!(y.dims(), &[4, 6]);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
        1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
        3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
    ];
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), expected);
}

#[test]
fn test_upsample_nearest_2d_batched() {
    // [2, 1, 2, 2] -> [2, 1, 4, 4], two batch items.
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 1, 2, 2],
        &Device::Cpu,
    )
    .unwrap();
    let y = x.upsample_nearest_2d(2, 2).unwrap();
    assert_eq!(y.dims(), &[2, 1, 4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // First batch: [1,2,3,4]
    assert_eq!(vals[0], 1.0);
    assert_eq!(vals[1], 1.0);
    assert_eq!(vals[2], 2.0);
    assert_eq!(vals[3], 2.0);
    // Second batch starts at index 16
    assert_eq!(vals[16], 5.0);
    assert_eq!(vals[17], 5.0);
    assert_eq!(vals[18], 6.0);
    assert_eq!(vals[19], 6.0);
}

#[test]
fn test_upsample_nearest_2d_zero_scale() {
    let x = DynTensor::from_vec(vec![1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(x.upsample_nearest_2d(0, 2).is_err());
    assert!(x.upsample_nearest_2d(2, 0).is_err());
}

#[test]
fn test_upsample_nearest_2d_rank_too_low() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    assert!(x.upsample_nearest_2d(2, 2).is_err());
}

// -- Bilinear 2D upsample tests -----------------------------------------------

#[test]
fn test_upsample_bilinear_2d_scale1_identity() {
    // Scale 1.0 should be near-identity.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d(1.0, 1.0, false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (a, b) in vals.iter().zip(&[1.0, 2.0, 3.0, 4.0]) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }
}

#[test]
fn test_upsample_bilinear_2d_scale2_align_corners() {
    // 2x2 input -> 4x4, align_corners=true.
    // Corners of output must match corners of input exactly.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d(2.0, 2.0, true).unwrap();
    assert_eq!(y.dims(), &[4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Corners: (0,0)=1.0, (0,3)=2.0, (3,0)=3.0, (3,3)=4.0
    assert!((vals[0] - 1.0).abs() < 1e-5, "top-left: {}", vals[0]);
    assert!((vals[3] - 2.0).abs() < 1e-5, "top-right: {}", vals[3]);
    assert!((vals[12] - 3.0).abs() < 1e-5, "bottom-left: {}", vals[12]);
    assert!((vals[15] - 4.0).abs() < 1e-5, "bottom-right: {}", vals[15]);
}

#[test]
fn test_upsample_bilinear_2d_scale2_no_align_corners() {
    // 2x2 input -> 4x4, align_corners=false.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d(2.0, 2.0, false).unwrap();
    assert_eq!(y.dims(), &[4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // With align_corners=false, output pixels map to shifted source coords.
    // All values should be within the input range [1, 4].
    for &v in &vals {
        assert!((1.0 - 1e-5..=4.0 + 1e-5).contains(&v), "out of range: {v}");
    }
}

#[test]
fn test_upsample_bilinear_2d_1x1_input() {
    // 1x1 input -> 3x3. All outputs should equal the single input value.
    let x = DynTensor::from_vec(vec![5.0], &[1, 1], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d(3.0, 3.0, false).unwrap();
    assert_eq!(y.dims(), &[3, 3]);
    for &v in &y.to_flat_vec::<f32>().unwrap() {
        assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {v}");
    }
}

#[test]
fn test_upsample_bilinear_2d_align_corners_1x1() {
    // 1x1 -> 3x3 with align_corners=true. All values should be the input.
    let x = DynTensor::from_vec(vec![7.0], &[1, 1], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d(3.0, 3.0, true).unwrap();
    assert_eq!(y.dims(), &[3, 3]);
    for &v in &y.to_flat_vec::<f32>().unwrap() {
        assert!((v - 7.0).abs() < 1e-5, "expected 7.0, got {v}");
    }
}

#[test]
fn test_upsample_bilinear_2d_invalid_scale() {
    let x = DynTensor::from_vec(vec![1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(x.upsample_bilinear_2d(0.0, 2.0, false).is_err());
    assert!(x.upsample_bilinear_2d(2.0, -1.0, false).is_err());
    assert!(x.upsample_bilinear_2d(f64::NAN, 2.0, false).is_err());
    assert!(x.upsample_bilinear_2d(2.0, f64::INFINITY, false).is_err());
}

#[test]
fn test_upsample_bilinear_2d_rank_too_low() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    assert!(x.upsample_bilinear_2d(2.0, 2.0, false).is_err());
}

#[test]
fn test_upsample_bilinear_2d_fractional_scale() {
    // 4x4 input -> 2x2 via 0.5x downscale.
    #[rustfmt::skip]
    let x = DynTensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        &[4, 4],
        &Device::Cpu,
    ).unwrap();
    let y = x.upsample_bilinear_2d(0.5, 0.5, false).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    // All values should be in input range
    for &v in &y.to_flat_vec::<f32>().unwrap() {
        assert!((1.0 - 1e-5..=16.0 + 1e-5).contains(&v), "out of range: {v}");
    }
}

// -- Upsample2d nn layer tests -----------------------------------------------

#[test]
fn test_upsample2d_nearest_layer() {
    let layer = Upsample2d::new(2.0, 2.0, UpsampleMode::Nearest).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_upsample2d_bilinear_layer() {
    let layer = Upsample2d::new(
        2.0,
        2.0,
        UpsampleMode::Bilinear {
            align_corners: true,
        },
    )
    .unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_upsample2d_accessors() {
    let layer = Upsample2d::new(
        3.0,
        4.0,
        UpsampleMode::Bilinear {
            align_corners: false,
        },
    )
    .unwrap();
    assert!((layer.scale_h() - 3.0).abs() < 1e-10);
    assert!((layer.scale_w() - 4.0).abs() < 1e-10);
    assert_eq!(
        layer.mode(),
        UpsampleMode::Bilinear {
            align_corners: false
        }
    );
}

#[test]
fn test_upsample2d_invalid_scale() {
    assert!(Upsample2d::new(0.0, 2.0, UpsampleMode::Nearest).is_err());
    assert!(Upsample2d::new(2.0, -1.0, UpsampleMode::Nearest).is_err());
}

// -- 1D upsample edge cases ---------------------------------------------------

#[test]
fn test_upsample_nearest_1d_factor_zero_returns_error() {
    // factor == 0 is rejected -- must return InvalidShape.
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let result = x.upsample_nearest_1d(0);
    assert!(result.is_err(), "factor=0 should return error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("factor") || msg.contains("> 0"),
        "error should mention factor: {msg}"
    );
}

#[test]
fn test_upsample_nearest_1d_factor_one_identity() {
    // factor == 1 should return a clone (identity).
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_1d(1).unwrap();
    assert_eq!(y.dims(), &[3]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_upsample_nearest_1d_rank0_returns_error() {
    // Rank-0 (scalar) tensors are rejected.
    let x = DynTensor::full(&[], 1.0_f64, DType::F32, &Device::Cpu).unwrap();
    let result = x.upsample_nearest_1d(2);
    assert!(result.is_err(), "rank-0 should return error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("rank") || msg.contains("1"),
        "error should mention rank: {msg}"
    );
}

#[test]
fn test_upsample_nearest_1d_rank3_bct() {
    // Typical production shape [B, C, T] = [2, 3, 4], factor 2 -> [2, 3, 8].
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[2, 3, 4], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_1d(2).unwrap();
    assert_eq!(y.dims(), &[2, 3, 8]);
    // First channel of first batch: [0, 1, 2, 3] -> [0, 0, 1, 1, 2, 2, 3, 3]
    let flat = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(&flat[0..8], &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

// -- 1D upsample overflow guard -----------------------------------------------

#[test]
fn test_upsample_nearest_1d_overflow_returns_error() {
    // factor * t overflows usize -- must return DimensionOverflow, not panic.
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let result = x.upsample_nearest_1d(usize::MAX);
    assert!(
        result.is_err(),
        "should return DimensionOverflow on overflow"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("overflow") || msg.contains("Overflow"),
        "error should mention overflow: {msg}"
    );
}

#[test]
fn test_upsample_nearest_1d_basic() {
    // Verify basic 1D upsample: [1, 2, 3] with factor 2 -> [1, 1, 2, 2, 3, 3].
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_1d(2).unwrap();
    assert_eq!(y.dims(), &[6]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
}

#[test]
fn test_upsample_nearest_1d_batched() {
    // [2, 3] -> [2, 6] with factor 2.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let y = x.upsample_nearest_1d(2).unwrap();
    assert_eq!(y.dims(), &[2, 6]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
    );
}

// =============================================================================
// upsample_bilinear_2d_to_size tests
// =============================================================================

#[test]
fn test_upsample_bilinear_2d_to_size_identity() {
    // Output size == input size should be near-identity.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(2, 2, false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_approx(
        &y.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        1e-5,
        "identity",
    );
}

#[test]
fn test_upsample_bilinear_2d_to_size_2x_align_corners() {
    // [1,1,2,2] -> [1,1,4,4] with align_corners=true.
    // Corners must be exactly preserved.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(4, 4, true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-5, "TL: {}", vals[0]);
    assert!((vals[3] - 2.0).abs() < 1e-5, "TR: {}", vals[3]);
    assert!((vals[12] - 3.0).abs() < 1e-5, "BL: {}", vals[12]);
    assert!((vals[15] - 4.0).abs() < 1e-5, "BR: {}", vals[15]);
}

#[test]
fn test_upsample_bilinear_2d_to_size_error_zero() {
    let x = DynTensor::from_vec(vec![1.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(x.upsample_bilinear_2d_to_size(0, 2, false).is_err());
    assert!(x.upsample_bilinear_2d_to_size(2, 0, false).is_err());
}

#[test]
fn test_upsample_bilinear_2d_to_size_rank_too_low() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    assert!(x.upsample_bilinear_2d_to_size(4, 4, false).is_err());
}

#[test]
fn test_upsample_bilinear_2d_to_size_1x1_broadcast() {
    // 1x1 input upsampled to any size should produce constant output.
    let x = DynTensor::from_vec(vec![3.14], &[1, 1, 1, 1], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(5, 7, false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 5, 7]);
    for &v in &y.to_flat_vec::<f32>().unwrap() {
        assert!((v - 3.14).abs() < 1e-5, "expected 3.14, got {v}");
    }
}

// -- Upsample2dToSize nn layer tests ------------------------------------------

#[test]
fn test_upsample2d_to_size_layer_basic() {
    let layer = Upsample2dToSize::new(8, 8, false).unwrap();
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 8, 8]);
    // All values within input range
    for &v in &y.to_flat_vec::<f32>().unwrap() {
        assert!((1.0 - 1e-5..=4.0 + 1e-5).contains(&v), "out of range: {v}");
    }
}

#[test]
fn test_upsample2d_to_size_layer_accessors() {
    let layer = Upsample2dToSize::new(16, 32, true).unwrap();
    assert_eq!(layer.out_h(), 16);
    assert_eq!(layer.out_w(), 32);
    assert!(layer.align_corners());
}

#[test]
fn test_upsample2d_to_size_layer_invalid() {
    assert!(Upsample2dToSize::new(0, 4, false).is_err());
    assert!(Upsample2dToSize::new(4, 0, false).is_err());
}

// =============================================================================
// PyTorch F.interpolate reference tests (#3854)
//
// Reference values generated with PyTorch 2.5:
//   import torch, torch.nn.functional as F
//   x = torch.tensor([[[[1., 2.], [3., 4.]]]]) # [1,1,2,2]
//   F.interpolate(x, size=(4,4), mode='bilinear', align_corners=True)
//   F.interpolate(x, size=(4,4), mode='bilinear', align_corners=False)
//   F.interpolate(x, size=(3,5), mode='bilinear', align_corners=False)
// =============================================================================

#[test]
fn test_pytorch_ref_bilinear_2x2_to_4x4_align_corners_true() {
    // PyTorch: F.interpolate(x, size=(4,4), mode='bilinear', align_corners=True)
    // Input: [[[[1., 2.], [3., 4.]]]]
    // Expected output (from PyTorch):
    // [[[[1.0000, 1.3333, 1.6667, 2.0000],
    //    [1.6667, 2.0000, 2.3333, 2.6667],
    //    [2.3333, 2.6667, 3.0000, 3.3333],
    //    [3.0000, 3.3333, 3.6667, 4.0000]]]]
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(4, 4, true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        1.0000, 1.3333, 1.6667, 2.0000,
        1.6667, 2.0000, 2.3333, 2.6667,
        2.3333, 2.6667, 3.0000, 3.3333,
        3.0000, 3.3333, 3.6667, 4.0000,
    ];
    assert_approx(
        &y.to_flat_vec::<f32>().unwrap(),
        &expected,
        1e-3,
        "pytorch_bilinear_ac_true",
    );
}

#[test]
fn test_pytorch_ref_bilinear_2x2_to_4x4_align_corners_false() {
    // PyTorch: F.interpolate(x, size=(4,4), mode='bilinear', align_corners=False)
    // Input: [[[[1., 2.], [3., 4.]]]]
    // Expected output (from PyTorch):
    // [[[[1.0000, 1.2500, 1.7500, 2.0000],
    //    [1.5000, 1.7500, 2.2500, 2.5000],
    //    [2.5000, 2.7500, 3.2500, 3.5000],
    //    [3.0000, 3.2500, 3.7500, 4.0000]]]]
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(4, 4, false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4, 4]);
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        1.0000, 1.2500, 1.7500, 2.0000,
        1.5000, 1.7500, 2.2500, 2.5000,
        2.5000, 2.7500, 3.2500, 3.5000,
        3.0000, 3.2500, 3.7500, 4.0000,
    ];
    assert_approx(
        &y.to_flat_vec::<f32>().unwrap(),
        &expected,
        1e-4,
        "pytorch_bilinear_ac_false",
    );
}

#[test]
fn test_pytorch_ref_bilinear_2x2_to_3x5_align_corners_false() {
    // PyTorch: F.interpolate(x, size=(3,5), mode='bilinear', align_corners=False)
    // Input: [[[[1., 2.], [3., 4.]]]]
    //
    // Coordinate mapping (align_corners=False):
    //   src_y = (dst_y + 0.5) * in_h / out_h - 0.5
    //   src_x = (dst_x + 0.5) * in_w / out_w - 0.5
    //
    // Expected output (verified by direct computation):
    // [[[[1.0000, 1.1000, 1.5000, 1.9000, 2.0000],
    //    [2.0000, 2.1000, 2.5000, 2.9000, 3.0000],
    //    [3.0000, 3.1000, 3.5000, 3.9000, 4.0000]]]]
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(3, 5, false).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 5]);
    #[rustfmt::skip]
    let expected: Vec<f32> = vec![
        1.0000, 1.1000, 1.5000, 1.9000, 2.0000,
        2.0000, 2.1000, 2.5000, 2.9000, 3.0000,
        3.0000, 3.1000, 3.5000, 3.9000, 4.0000,
    ];
    assert_approx(
        &y.to_flat_vec::<f32>().unwrap(),
        &expected,
        1e-4,
        "pytorch_bilinear_3x5",
    );
}

#[test]
fn test_pytorch_ref_bilinear_3x3_to_6x6_align_corners_true() {
    // PyTorch: x = torch.arange(9.).reshape(1,1,3,3)
    //   F.interpolate(x, size=(6,6), mode='bilinear', align_corners=True)
    // Input:
    //   0 1 2
    //   3 4 5
    //   6 7 8
    // Corners of output must be 0, 2, 6, 8 exactly.
    let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 3, 3], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(6, 6, true).unwrap();
    assert_eq!(y.dims(), &[1, 1, 6, 6]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Corners
    assert!((vals[0] - 0.0).abs() < 1e-5, "TL: {}", vals[0]);
    assert!((vals[5] - 2.0).abs() < 1e-5, "TR: {}", vals[5]);
    assert!((vals[30] - 6.0).abs() < 1e-5, "BL: {}", vals[30]);
    assert!((vals[35] - 8.0).abs() < 1e-5, "BR: {}", vals[35]);
    // Center should be 4.0
    // row=2or3, col=2or3 -> index 2*6+2=14, 2*6+3=15, 3*6+2=20, 3*6+3=21
    let center = (vals[14] + vals[15] + vals[20] + vals[21]) / 4.0;
    assert!((center - 4.0).abs() < 0.01, "center avg: {center}");
}

#[test]
fn test_pytorch_ref_bilinear_batched_multi_channel() {
    // [2, 2, 2, 2] -> [2, 2, 4, 4], align_corners=false
    // Verifies batch and channel dimensions are preserved correctly.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[2, 2, 2, 2], &Device::Cpu).unwrap();
    let y = x.upsample_bilinear_2d_to_size(4, 4, false).unwrap();
    assert_eq!(y.dims(), &[2, 2, 4, 4]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 2 * 2 * 4 * 4);

    // Batch 0, channel 0: input [0,1,2,3] -> output corners should be near 0,1,2,3
    // Batch 1, channel 1: input [12,13,14,15] -> output should be in [12,15]
    let b1c1_start = (2 + 1) * 4 * 4; // batch=1, channel=1
    for &v in &vals[b1c1_start..b1c1_start + 16] {
        assert!(
            (12.0 - 1e-5..=15.0 + 1e-5).contains(&v),
            "b1c1 out of range: {v}"
        );
    }
}

// -- FPN/PAN use case: non-power-of-2 spatial matching -----------------------

#[test]
fn test_upsample_bilinear_fpn_spatial_match() {
    // FPN scenario: P5=[B,C,5,7] needs upsampling to match P4=[B,C,10,14]
    let p5 = DynTensor::full(&[1, 64, 5, 7], 1.0, DType::F32, &Device::Cpu).unwrap();
    let y = p5.upsample_bilinear_2d_to_size(10, 14, false).unwrap();
    assert_eq!(y.dims(), &[1, 64, 10, 14]);
    // All values should be 1.0 (constant input -> constant output)
    for &v in &y.to_flat_vec::<f32>().unwrap() {
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
    }
}

#[test]
fn test_upsample_bilinear_fpn_odd_dimensions() {
    // FPN scenario: upsample [B,C,3,5] to [B,C,6,10] (2x)
    // then to [B,C,7,11] (non-integer ratio)
    let x = DynTensor::full(&[1, 32, 3, 5], 2.0, DType::F32, &Device::Cpu).unwrap();

    // 2x upsample
    let y1 = x.upsample_bilinear_2d_to_size(6, 10, false).unwrap();
    assert_eq!(y1.dims(), &[1, 32, 6, 10]);

    // Non-integer ratio
    let y2 = x.upsample_bilinear_2d_to_size(7, 11, false).unwrap();
    assert_eq!(y2.dims(), &[1, 32, 7, 11]);

    // Constant input should produce constant output
    for &v in &y2.to_flat_vec::<f32>().unwrap() {
        assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
    }
}

#[test]
fn test_upsample2d_to_size_in_pan_pattern() {
    // Simulates the PAN top-down path: P5=[B,C,H5,W5], upsample to P4 spatial size.
    // Then cat with P4 along channel dim.
    let p5 = DynTensor::full(&[1, 128, 4, 4], 0.5, DType::F32, &Device::Cpu).unwrap();
    let p4 = DynTensor::full(&[1, 64, 8, 8], 0.3, DType::F32, &Device::Cpu).unwrap();

    // Upsample P5 to match P4 spatial dims
    let p5_up = p5.upsample_bilinear_2d_to_size(8, 8, false).unwrap();
    assert_eq!(p5_up.dims(), &[1, 128, 8, 8]);

    // Cat along channel dim (standard FPN concat)
    let cat = DynTensor::cat(&[&p5_up, &p4], 1).unwrap();
    assert_eq!(cat.dims(), &[1, 192, 8, 8]);
}

// -- Scale-factor vs output-size consistency ----------------------------------

#[test]
fn test_scale_factor_vs_output_size_agree_for_integer_scales() {
    // For exact 2x upscale, scale_factor and output_size should produce same result.
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let by_scale = x.upsample_bilinear_2d(2.0, 2.0, true).unwrap();
    let by_size = x.upsample_bilinear_2d_to_size(4, 4, true).unwrap();
    assert_eq!(by_scale.dims(), by_size.dims());
    assert_approx(
        &by_scale.to_flat_vec::<f32>().unwrap(),
        &by_size.to_flat_vec::<f32>().unwrap(),
        1e-5,
        "scale_vs_size_ac_true",
    );
}

#[test]
fn test_scale_factor_vs_output_size_agree_no_align_corners() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &Device::Cpu).unwrap();
    let by_scale = x.upsample_bilinear_2d(2.0, 2.0, false).unwrap();
    let by_size = x.upsample_bilinear_2d_to_size(4, 4, false).unwrap();
    assert_eq!(by_scale.dims(), by_size.dims());
    assert_approx(
        &by_scale.to_flat_vec::<f32>().unwrap(),
        &by_size.to_flat_vec::<f32>().unwrap(),
        1e-5,
        "scale_vs_size_ac_false",
    );
}
