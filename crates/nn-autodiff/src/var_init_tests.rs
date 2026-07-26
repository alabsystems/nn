#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nn_core::test_utils::cpu;
use nn_core::DType;

#[test]
fn test_const_init() {
    let v = Var::from_init(Init::Const(3.125), &[2, 3], &cpu()).unwrap();
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), 6);
    for &val in &data {
        assert!((val - 3.125).abs() < 1e-5);
    }
}

#[test]
fn test_uniform_init_in_range() {
    let v = Var::from_init(Init::Uniform { lo: -1.0, hi: 1.0 }, &[100], &cpu()).unwrap();
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    for &val in &data {
        assert!((-1.0..=1.0).contains(&val), "val {val} out of range");
    }
}

#[test]
fn test_normal_init_shape() {
    let v = Var::from_init(
        Init::Normal {
            mean: 0.0,
            std: 1.0,
        },
        &[4, 5],
        &cpu(),
    )
    .unwrap();
    assert_eq!(v.dims().unwrap(), vec![4, 5]);
}

#[test]
fn test_xavier_uniform_shape_and_range() {
    let dims = [256, 512];
    let v = Var::xavier_uniform(&dims, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![256, 512]);
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    let a = (6.0_f64 / 768.0).sqrt() as f32;
    for &val in &data {
        assert!(
            (-a..=a).contains(&val),
            "val {val} outside Xavier bounds ±{a}"
        );
    }
}

#[test]
fn test_xavier_normal_not_all_zero() {
    let v = Var::from_init(Init::XavierNormal, &[64, 128], &cpu()).unwrap();
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    let any_nonzero = data.iter().any(|&v| v.abs() > 1e-10);
    assert!(any_nonzero, "Xavier normal should produce non-zero values");
}

#[test]
fn test_kaiming_uniform_shape() {
    let v = Var::kaiming_uniform(&[128, 64], &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![128, 64]);
}

#[test]
fn test_kaiming_normal_fan_out() {
    let v = Var::from_init(Init::KaimingNormal { fan: Fan::FanOut }, &[64, 32], &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![64, 32]);
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    let any_nonzero = data.iter().any(|&v| v.abs() > 1e-10);
    assert!(any_nonzero);
}

#[test]
fn test_var_randn_convenience() {
    let v = Var::randn(&[10, 20], 0.0, 0.1, &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![10, 20]);
}

#[test]
fn test_var_rand_convenience() {
    let v = Var::rand(&[5, 5], -0.5, 0.5, &cpu()).unwrap();
    let data = v.data().unwrap().to_flat_vec::<f32>().unwrap();
    for &val in &data {
        assert!((-0.5..=0.5).contains(&val));
    }
}

#[test]
fn test_compute_fans_2d() {
    // [out_features, in_features] = [64, 128]
    let (fan_in, fan_out) = compute_fans(&[64, 128]).unwrap();
    assert_eq!(fan_in, 128);
    assert_eq!(fan_out, 64);
}

#[test]
fn test_compute_fans_conv() {
    // Conv1d: [out_channels, in_channels, kernel_size] = [32, 16, 3]
    let (fan_in, fan_out) = compute_fans(&[32, 16, 3]).unwrap();
    assert_eq!(fan_in, 48); // 16 * 3
    assert_eq!(fan_out, 96); // 32 * 3
}

#[test]
fn test_compute_fans_1d() {
    let (fan_in, fan_out) = compute_fans(&[256]).unwrap();
    assert_eq!(fan_in, 1);
    assert_eq!(fan_out, 256);
}

#[test]
fn test_compute_fans_scalar() {
    let (fan_in, fan_out) = compute_fans(&[]).unwrap();
    assert_eq!(fan_in, 1);
    assert_eq!(fan_out, 1);
}

#[test]
fn test_compute_fans_overflow() {
    // Kernel dims that overflow usize when multiplied
    let result = compute_fans(&[2, 3, usize::MAX, 2]);
    assert!(result.is_err(), "should error on overflow");
}

#[test]
fn test_kaiming_uniform_fan_avg() {
    let v = Var::from_init(Init::Kaiming { fan: Fan::FanAvg }, &[64, 128], &cpu()).unwrap();
    assert_eq!(v.dims().unwrap(), vec![64, 128]);
}

#[test]
fn test_init_to_tensor_directly() {
    let t = Init::XavierUniform
        .to_tensor(&[8, 16], DType::F32, &cpu())
        .unwrap();
    assert_eq!(t.dims(), &[8, 16]);
}
