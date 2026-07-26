#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Metal DynTensor GPU backend (D3 of #914).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::init;

// -- Transfer tests -----------------------------------------------------------

#[test]
fn test_cpu_to_gpu_round_trip() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cpu = DynTensor::new(&data, &[2, 2], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    assert_eq!(gpu.device(), Device::metal());
    assert_eq!(gpu.dims(), &[2, 2]);

    let back = gpu.to_device(&Device::Cpu).unwrap();
    assert_eq!(back.device(), Device::Cpu);
    let result = back.to_flat_vec::<f32>().unwrap();
    assert_eq!(result, data);
}

#[test]
fn test_gpu_preserves_shape_and_dtype() {
    init();
    let t = DynTensor::zeros(&[3, 4, 5], DType::F32, &Device::metal()).unwrap();
    assert_eq!(t.dims(), &[3, 4, 5]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.device(), Device::metal());
    assert_eq!(t.numel(), 60);
}

#[test]
fn test_gpu_ones_round_trip() {
    init();
    let t = DynTensor::ones(&[2, 3], DType::F32, &Device::metal()).unwrap();
    let cpu = t.to_device(&Device::Cpu).unwrap();
    let data = cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0f32; 6]);
}

// -- Binary op tests ----------------------------------------------------------

#[test]
fn test_gpu_add() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[3], &Device::metal()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.device(), Device::metal());
    let result = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_gpu_sub() {
    init();
    let a = DynTensor::new(&[10.0, 20.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[3.0, 7.0], &[2], &Device::metal()).unwrap();
    let c = a.sub(&b).unwrap();
    let result = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![7.0, 13.0]);
}

#[test]
fn test_gpu_mul() {
    init();
    let a = DynTensor::new(&[2.0, 3.0, 4.0], &[3], &Device::metal()).unwrap();
    let b = DynTensor::new(&[5.0, 6.0, 7.0], &[3], &Device::metal()).unwrap();
    let c = a.mul(&b).unwrap();
    let result = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_gpu_div() {
    init();
    let a = DynTensor::new(&[12.0, 15.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[3.0, 5.0], &[2], &Device::metal()).unwrap();
    let c = a.div(&b).unwrap();
    let result = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![4.0, 3.0]);
}

#[test]
fn test_gpu_maximum() {
    init();
    let a = DynTensor::new(&[1.0, 5.0, 3.0, 7.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[4.0, 2.0, 6.0, 0.0], &[4], &Device::metal()).unwrap();
    let c = a.maximum(&b).unwrap();
    assert_eq!(c.device(), Device::metal());
    let result = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_gpu_minimum() {
    init();
    let a = DynTensor::new(&[1.0, 5.0, 3.0, 7.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[4.0, 2.0, 6.0, 0.0], &[4], &Device::metal()).unwrap();
    let c = a.minimum(&b).unwrap();
    assert_eq!(c.device(), Device::metal());
    let result = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_gpu_maximum_broadcast() {
    init();
    // [2,3] vs [3] — broadcast right-aligned
    let a = DynTensor::new(&[1.0, 5.0, 3.0, 7.0, 2.0, 8.0], &[2, 3], &Device::metal()).unwrap();
    let b = DynTensor::new(&[4.0, 4.0, 4.0], &[3], &Device::metal()).unwrap();
    let c = a.maximum(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let result = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![4.0, 5.0, 4.0, 7.0, 4.0, 8.0]);
}

// -- Unary op tests -----------------------------------------------------------

#[test]
fn test_gpu_relu() {
    init();
    let t = DynTensor::new(&[-1.0, 0.0, 1.0, 2.0], &[4], &Device::metal()).unwrap();
    let r = t.relu().unwrap();
    assert_eq!(r.device(), Device::metal());
    let result = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_gpu_neg() {
    init();
    let t = DynTensor::new(&[1.0, -2.0, 3.0], &[3], &Device::metal()).unwrap();
    let r = t.neg().unwrap();
    let result = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![-1.0, 2.0, -3.0]);
}

// -- Reduction tests ----------------------------------------------------------

#[test]
fn test_gpu_sum_keepdim() {
    init();
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let s = t.sum_keepdim(1).unwrap();
    assert_eq!(s.dims(), &[2, 1]);
    let result = s
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![6.0, 15.0]);
}

#[test]
fn test_gpu_mean_keepdim() {
    init();
    let t = DynTensor::new(&[2.0, 4.0, 6.0], &[1, 3], &Device::metal()).unwrap();
    let m = t.mean_keepdim(1).unwrap();
    assert_eq!(m.dims(), &[1, 1]);
    let result = m
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![4.0]);
}

// -- Non-keepdim reduction tests (fix for #1076) -----------------------------

#[test]
fn test_gpu_sum_no_keepdim() {
    init();
    // [2, 3] → sum(dim=1) without keepdim → [2]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let s = t.sum(1).unwrap();
    assert_eq!(s.dims(), &[2]); // NOT [2, 1]
    let result = s.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![6.0, 15.0]);
}

#[test]
fn test_gpu_mean_no_keepdim() {
    init();
    // [2, 4] → mean(dim=1) without keepdim → [2]
    let t = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &Device::metal(),
    )
    .unwrap();
    let m = t.mean(1).unwrap();
    assert_eq!(m.dims(), &[2]); // NOT [2, 1]
    let result = m.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![2.5, 6.5]);
}

#[test]
fn test_gpu_max_keepdim() {
    init();
    // [2, 3] → max_keepdim(dim=1) → [2, 1]
    let t = DynTensor::new(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &Device::metal()).unwrap();
    let m = t.max_keepdim(1).unwrap();
    assert_eq!(m.dims(), &[2, 1]);
    let result = m
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![3.0, 6.0]);
}

#[test]
fn test_gpu_max_no_keepdim() {
    init();
    // [2, 3] → max(dim=1) without keepdim → [2]
    let t = DynTensor::new(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &Device::metal()).unwrap();
    let m = t.max(1).unwrap();
    assert_eq!(m.dims(), &[2]); // NOT [2, 1]
    let result = m.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![3.0, 6.0]);
}

// -- Phase 2 unary op tests (GPU-native via Elementwise) ---------------------

#[test]
fn test_gpu_exp() {
    init();
    let t = DynTensor::new(&[0.0, 1.0, -1.0], &[3], &Device::metal()).unwrap();
    let r = t.exp().unwrap();
    assert_eq!(r.device(), Device::metal());
    let result = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
    assert!((result[2] - 1.0 / std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn test_gpu_sqrt() {
    init();
    let t = DynTensor::new(&[4.0, 9.0, 16.0, 25.0], &[4], &Device::metal()).unwrap();
    let r = t.sqrt().unwrap();
    let result = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_gpu_abs() {
    init();
    let t = DynTensor::new(&[-3.0, 0.0, 5.0, -7.0], &[4], &Device::metal()).unwrap();
    let r = t.abs().unwrap();
    let result = r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(result, vec![3.0, 0.0, 5.0, 7.0]);
}

#[test]
fn test_gpu_recip() {
    init();
    let t = DynTensor::new(&[2.0, 4.0, 0.5], &[3], &Device::metal()).unwrap();
    let r = t.recip().unwrap();
    let result = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![0.5, 0.25, 2.0]);
}

#[test]
fn test_gpu_sin_cos() {
    init();
    let t = DynTensor::new(&[0.0, std::f32::consts::FRAC_PI_2], &[2], &Device::metal()).unwrap();
    let s = t.sin().unwrap();
    let c = t.cos().unwrap();
    let sin_r = s.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    let cos_r = c.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert!((sin_r[0]).abs() < 1e-5); // sin(0) = 0
    assert!((sin_r[1] - 1.0).abs() < 1e-5); // sin(pi/2) = 1
    assert!((cos_r[0] - 1.0).abs() < 1e-5); // cos(0) = 1
    assert!((cos_r[1]).abs() < 1e-5); // cos(pi/2) = 0
}

#[test]
fn test_gpu_sqr() {
    init();
    let t = DynTensor::new(&[3.0, -4.0, 0.5], &[3], &Device::metal()).unwrap();
    let r = t.sqr().unwrap();
    let result = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(result, vec![9.0, 16.0, 0.25]);
}

#[test]
fn test_gpu_silu() {
    init();
    let t = DynTensor::new(&[0.0, 1.0, -1.0], &[3], &Device::metal()).unwrap();
    let r = t.silu().unwrap();
    assert_eq!(r.device(), Device::metal());
    let result = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    // silu(x) = x * sigmoid(x)
    // silu(0) = 0 * 0.5 = 0
    assert!((result[0]).abs() < 1e-5);
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    assert!((result[1] - 0.7311).abs() < 1e-3);
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
    assert!((result[2] + 0.2689).abs() < 1e-3);
}

#[test]
fn test_gpu_gelu_erf() {
    init();
    // Values include moderate negatives where erf-based and tanh-approximation
    // GELU diverge by ~1e-3. The 1e-5 tolerance ensures the GPU kernel uses
    // the A&S erf polynomial (matching CPU), not the tanh approximation.
    // Regression guard for W4-484 routing fix.
    let data = vec![-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    let gpu = DynTensor::new(&data, &[data.len()], &Device::metal()).unwrap();
    let cpu = DynTensor::new(&data, &[data.len()], &Device::Cpu).unwrap();
    let gpu_r = gpu.gelu_erf().unwrap();
    let cpu_r = cpu.gelu_erf().unwrap();
    assert_eq!(gpu_r.device(), Device::metal());
    let gpu_vals = gpu_r
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cpu_vals = cpu_r.to_vec1::<f32>().unwrap();
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-5,
            "gelu_erf mismatch at {i}: gpu={g}, cpu={c}, diff={}. \
             Tolerance 1e-5 distinguishes fused erf kernel from tanh approximation.",
            (g - c).abs()
        );
    }
}

#[test]
fn test_gpu_log_fallback() {
    init();
    let t = DynTensor::new(&[1.0, std::f32::consts::E, 10.0], &[3], &Device::metal()).unwrap();
    let r = t.log().unwrap();
    assert_eq!(r.device(), Device::metal());
    let result = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert!((result[0]).abs() < 1e-5); // ln(1) = 0
    assert!((result[1] - 1.0).abs() < 1e-5); // ln(e) = 1
    assert!((result[2] - std::f32::consts::LN_10).abs() < 1e-3); // ln(10)
}

// -- atan2 GPU parity tests ---------------------------------------------------

#[test]
fn test_gpu_atan2_basic() {
    init();
    let y = DynTensor::new(&[1.0f32, 1.0, -1.0, -1.0], &[4], &Device::metal()).unwrap();
    let x = DynTensor::new(&[1.0f32, -1.0, -1.0, 1.0], &[4], &Device::metal()).unwrap();
    let gpu_result = y.atan2(&x).unwrap();
    assert_eq!(gpu_result.device(), Device::metal());
    let result = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let pi = std::f32::consts::PI;
    // Q1: atan2(1,1) = pi/4
    assert!((result[0] - pi / 4.0).abs() < 1e-5);
    // Q2: atan2(1,-1) = 3*pi/4
    assert!((result[1] - 3.0 * pi / 4.0).abs() < 1e-5);
    // Q3: atan2(-1,-1) = -3*pi/4
    assert!((result[2] - (-3.0 * pi / 4.0)).abs() < 1e-5);
    // Q4: atan2(-1,1) = -pi/4
    assert!((result[3] - (-pi / 4.0)).abs() < 1e-5);
}

#[test]
fn test_gpu_atan2_cpu_parity() {
    init();
    let y_data = vec![0.0f32, 1.0, 0.0, -1.0, 3.0, -2.5];
    let x_data = vec![1.0f32, 0.0, -1.0, 0.0, 4.0, 1.5];
    // CPU reference
    let expected: Vec<f32> = y_data
        .iter()
        .zip(x_data.iter())
        .map(|(yi, xi)| yi.atan2(*xi))
        .collect();
    // GPU
    let y_gpu = DynTensor::new(&y_data, &[6], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[6], &Device::metal()).unwrap();
    let gpu_result = y_gpu
        .atan2(&x_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, e)) in gpu_result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "atan2 mismatch at index {i}: gpu={g}, cpu={e}"
        );
    }
}

#[test]
fn test_gpu_atan2_broadcast() {
    init();
    // [2,3] vs [3] — broadcast right-aligned
    let y = DynTensor::new(&[1.0, -1.0, 0.5, 2.0, -0.5, 1.0], &[2, 3], &Device::metal()).unwrap();
    let x = DynTensor::new(&[1.0, 1.0, 1.0], &[3], &Device::metal()).unwrap();
    let result = y.atan2(&x).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // atan2(y, 1.0) = atan(y)
    assert!((vals[0] - 1.0f32.atan()).abs() < 1e-5);
    assert!((vals[1] - (-1.0f32).atan()).abs() < 1e-5);
    assert!((vals[2] - 0.5f32.atan()).abs() < 1e-5);
}

// -- Device mismatch test -----------------------------------------------------

#[test]
fn test_mixed_device_error() {
    init();
    let cpu = DynTensor::new(&[1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&[3.0, 4.0], &[2], &Device::metal()).unwrap();
    let err = cpu.add(&gpu);
    assert!(err.is_err(), "mixed CPU+GPU should fail");
}
