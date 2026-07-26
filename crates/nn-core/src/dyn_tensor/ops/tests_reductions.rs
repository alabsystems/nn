#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reduction op tests (mean, sum, max, min — keepdim and all variants).

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d};
use crate::Device;
use crate::DynTensor;

#[test]
fn test_mean_keepdim() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m = a.mean_keepdim(1).unwrap();
    assert_eq!(m.dims(), &[2, 1]);
    let flat = m.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(flat[0], 2.0, 1e-6));
    assert!(approx_eq(flat[1], 5.0, 1e-6));
}

#[test]
fn test_sum_keepdim() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum_keepdim(1).unwrap();
    assert_eq!(s.dims(), &[2, 1]);
    let flat = s.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![6.0, 15.0]);
}

#[test]
fn test_mean_all() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let m = a.mean_all().unwrap();
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 2.5, 1e-6));
}

#[test]
fn test_mean_all_preserves_cpu_device() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m = a.mean_all().unwrap();
    assert_eq!(m.device(), Device::Cpu);
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 3.5, 1e-6));
}

#[test]
fn test_mean_all_empty_tensor_error() {
    let a = DynTensor::from_vec(vec![], &[0], &cpu()).unwrap();
    assert!(a.mean_all().is_err());
}

#[test]
fn test_sum_all() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let s = a.sum_all().unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), 6.0);
}

#[test]
fn test_sum_all_preserves_cpu_device() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum_all().unwrap();
    assert_eq!(s.device(), Device::Cpu);
    assert_eq!(s.to_scalar::<f32>().unwrap(), 21.0);
}

#[test]
fn test_sum_all_2d() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let s = a.sum_all().unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), 10.0);
}

#[test]
fn test_max_keepdim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.max_keepdim(1).unwrap();
    assert_eq!(m.dims(), &[2, 1]);
    let flat = m.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![5.0, 6.0]);
}

#[test]
fn test_min_keepdim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.min_keepdim(1).unwrap();
    assert_eq!(m.dims(), &[2, 1]);
    let flat = m.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![1.0, 2.0]);
}

#[test]
fn test_max_all() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.max_all().unwrap();
    assert_eq!(m.dims(), &[] as &[usize]);
    assert_eq!(m.to_scalar::<f32>().unwrap(), 6.0);
}

#[test]
fn test_min_all() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.min_all().unwrap();
    assert_eq!(m.dims(), &[] as &[usize]);
    assert_eq!(m.to_scalar::<f32>().unwrap(), 1.0);
}

#[test]
fn test_max_all_negative() {
    let a = t1d(&[-5.0, -3.0, -1.0, -4.0]);
    assert_eq!(a.max_all().unwrap().to_scalar::<f32>().unwrap(), -1.0);
}

#[test]
fn test_min_all_empty_fails() {
    let a = DynTensor::from_vec(vec![], &[0], &cpu()).unwrap();
    assert!(a.min_all().is_err());
}

#[test]
fn test_reduce_out_of_range() {
    let a = t1d(&[1.0, 2.0]);
    assert!(a.mean_keepdim(1).is_err());
}

#[test]
fn test_max_all_nan_returns_error() {
    let a = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap();
    let err = a.max_all().unwrap_err();
    assert!(
        err.to_string().contains("NaN"),
        "expected NaN error, got: {err}"
    );
}

#[test]
fn test_min_all_nan_returns_error() {
    let a = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap();
    let err = a.min_all().unwrap_err();
    assert!(
        err.to_string().contains("NaN"),
        "expected NaN error, got: {err}"
    );
}

#[test]
fn test_max_all_all_nan_returns_error() {
    let a = DynTensor::from_vec(vec![f32::NAN; 4], &[4], &cpu()).unwrap();
    assert!(a.max_all().is_err());
}
