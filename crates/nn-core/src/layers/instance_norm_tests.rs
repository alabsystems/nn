#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`InstanceNorm`].

use super::*;
use crate::Device;

fn tensor(data: &[f32], shape: &[usize]) -> DynTensor {
    DynTensor::from_vec(data.to_vec(), shape, &Device::Cpu).expect("valid tensor")
}

#[test]
fn test_instance_norm_basic() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    // [B=1, C=1, T=4]
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    // Normalized: mean=2.5, std=sqrt(1.25)≈1.118
    // Should be approximately [-1.342, -0.447, 0.447, 1.342]
    let mean: f32 = vals.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
    // Variance of normalized data should be close to 1 (biased estimator)
    assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {var}");
}

#[test]
fn test_instance_norm_multi_channel() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    // [B=1, C=2, T=3]
    let x = tensor(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[1, 2, 3]);
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 3]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    // Each channel independently normalized
    let ch0 = &vals[0..3];
    let ch1 = &vals[3..6];
    let mean0: f32 = ch0.iter().sum::<f32>() / 3.0;
    let mean1: f32 = ch1.iter().sum::<f32>() / 3.0;
    assert!(mean0.abs() < 1e-4, "ch0 mean = {mean0}");
    assert!(mean1.abs() < 1e-4, "ch1 mean = {mean1}");
}

#[test]
fn test_instance_norm_batch() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    // [B=2, C=1, T=3]
    let x = tensor(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 1, 3]);
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 1, 3]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    // Each batch item independently normalized → same pattern
    // Both [1,2,3] and [10,20,30] have same relative distribution
    for chunk in vals.chunks(3) {
        let mean: f32 = chunk.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
    }
}

#[test]
fn test_instance_norm_rank_error() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    // [B=2, C=3] — rank 2, needs 3+
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(norm.forward(&x).is_err());
}

#[test]
fn test_instance_norm_module_trait() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = x.apply(&norm).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);
}

// -- Error path and edge case tests (proof_coverage) --------------------------

/// Constant spatial values → var=0, normalized by sqrt(eps) only.
/// Output should be zero (mean subtracted, var ≈ 0 + eps).
#[test]
fn test_instance_norm_constant_spatial() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    // [B=1, C=1, T=4] all same value
    let x = tensor(&[5.0, 5.0, 5.0, 5.0], &[1, 1, 4]);
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // (x - mean) = 0, so output should be ~0 everywhere
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-3,
            "constant input: element {i} should be ~0, got {v}"
        );
    }
}

/// Single spatial element → mean=x, var=0. Should produce ~0 output.
#[test]
fn test_instance_norm_single_spatial_element() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    // [B=1, C=2, T=1]
    let x = tensor(&[3.0, 7.0], &[1, 2, 1]);
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 2, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Single element: mean = element, so centered = 0, output = 0/sqrt(eps) = 0 exactly.
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-5,
            "single-element: element {i} should be ~0, got {v}"
        );
    }
}

/// Higher-rank input: [B=1, C=1, H=2, W=3] should be treated as [B, C, spatial=6].
#[test]
fn test_instance_norm_4d_input() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]);
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / 6.0;
    assert!(mean.abs() < 1e-4, "4D mean should be ~0, got {mean}");
}

/// Large values should produce finite output (numerical stability).
#[test]
fn test_instance_norm_large_values_finite() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    let x = tensor(&[1e6, -1e6, 5e5, -5e5], &[1, 1, 4]);
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "large input: element {i} not finite: {v}");
    }
}

// -- Finiteness validation (#1202) --------------------------------------------

#[test]
fn test_instance_norm_nan_input_returns_error() {
    let norm = InstanceNorm::new(1e-5).unwrap();
    let mut data = vec![1.0f32; 12];
    data[0] = f32::NAN;
    let x = DynTensor::from_vec(data, &[1, 2, 6], &Device::Cpu).unwrap();
    let result = Module::forward(&norm, &x);
    assert!(result.is_err(), "NaN input should produce an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Non-finite") || msg.contains("NaN"),
        "error should mention non-finite: {msg}"
    );
}

// -- InstanceNormPrecision tests (#2691) ---------------------------------------

/// F32 (MatchPyTorchCpu) path produces valid normalized output.
#[test]
fn test_instance_norm_f32_precision_basic() {
    let norm = InstanceNorm::with_precision(1e-5, InstanceNormPrecision::MatchPyTorchCpu).unwrap();
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = norm.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 4]);

    let vals = y.to_flat_vec::<f32>().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5, "F32 mean should be ~0, got {mean}");
    let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
    assert!(
        (var - 1.0).abs() < 0.1,
        "F32 variance should be ~1, got {var}"
    );
}

/// Verify InstanceNorm MatchPyTorchCpu matches PyTorch's exact formula (#4335).
///
/// PyTorch InstanceNorm (affine=False, track_running_stats=False):
///   mean = x.mean(dim=-1)    (per channel per batch)
///   var  = ((x - mean)^2).mean(dim=-1)   (unbiased=False)
///   y    = (x - mean) / sqrt(var + eps)
///
/// This test uses a non-trivial input and hand-computed reference values
/// to catch formula differences (e.g., E[X^2]-E[X]^2 vs mean((x-mean)^2),
/// biased vs unbiased variance, rsqrt precision).
#[test]
fn test_instance_norm_matches_pytorch_formula() {
    // Input: [B=1, C=2, T=5] with non-trivial values.
    let data: Vec<f32> = vec![
        // Channel 0: mean = 3.0, var = 2.0
        1.0, 2.0, 3.0, 4.0, 5.0, // Channel 1: mean = 0.2, var = 0.04
        0.0, 0.1, 0.2, 0.3, 0.4,
    ];
    let x = tensor(&data, &[1, 2, 5]);

    // Compute expected output using PyTorch's formula exactly in f32.
    let eps: f32 = 1e-5;
    let mut expected = [0.0f32; 10];

    // Channel 0: [1,2,3,4,5]
    let mean0: f32 = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5.0; // = 3.0
    let var0: f32 = ((1.0 - mean0).powi(2)
        + (2.0 - mean0).powi(2)
        + (3.0 - mean0).powi(2)
        + (4.0 - mean0).powi(2)
        + (5.0 - mean0).powi(2))
        / 5.0; // = 2.0
    let inv_std0: f32 = 1.0 / (var0 + eps).sqrt();
    for (i, &val) in [1.0f32, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
        expected[i] = (val - mean0) * inv_std0;
    }

    // Channel 1: [0.0, 0.1, 0.2, 0.3, 0.4]
    let mean1: f32 = (0.0 + 0.1 + 0.2 + 0.3 + 0.4) / 5.0; // = 0.2
    let var1: f32 = ((0.0 - mean1).powi(2)
        + (0.1 - mean1).powi(2)
        + (0.2 - mean1).powi(2)
        + (0.3 - mean1).powi(2)
        + (0.4 - mean1).powi(2))
        / 5.0; // = 0.02
    let inv_std1: f32 = 1.0 / (var1 + eps).sqrt();
    for (i, &val) in [0.0f32, 0.1, 0.2, 0.3, 0.4].iter().enumerate() {
        expected[5 + i] = (val - mean1) * inv_std1;
    }

    // Test MatchPyTorchCpu mode — should match f32 reference exactly.
    let norm = InstanceNorm::with_precision(1e-5, InstanceNormPrecision::MatchPyTorchCpu).unwrap();
    let y = norm.forward(&x).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();

    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-6,
            "MatchPyTorchCpu[{i}]: got={got}, expected={exp}, diff={diff}"
        );
    }

    // Verify the formula computes unbiased=False variance.
    // PyTorch uses N in denominator, not N-1.
    assert!(
        (var0 - 2.0).abs() < 1e-6,
        "variance ch0 should be 2.0 (unbiased=False), got {var0}"
    );
    assert!(
        (var1 - 0.02).abs() < 1e-6,
        "variance ch1 should be 0.02 (unbiased=False), got {var1}"
    );
}

/// F64 and F32 paths produce close but not identical results.
/// This demonstrates the per-layer drift that compounds over 58 layers.
#[test]
fn test_instance_norm_precision_difference() {
    let norm_f64 = InstanceNorm::new(1e-5).unwrap();
    let norm_f32 =
        InstanceNorm::with_precision(1e-5, InstanceNormPrecision::MatchPyTorchCpu).unwrap();

    // Use values with non-trivial F32 rounding: irrational-ish fractions
    let x = tensor(
        &[0.123456, 0.789012, 0.345678, 0.901234, 0.567890, 0.111111],
        &[1, 2, 3],
    );
    let y_f64 = norm_f64.forward(&x).unwrap();
    let y_f32 = norm_f32.forward(&x).unwrap();

    let v64 = y_f64.to_flat_vec::<f32>().unwrap();
    let v32 = y_f32.to_flat_vec::<f32>().unwrap();

    // Both should be close (single-layer delta is < 1 ULP)
    for (i, (&a, &b)) in v64.iter().zip(v32.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "single-layer delta[{i}] = {diff} exceeds 1e-4 (f64={a}, f32={b})"
        );
    }
}
