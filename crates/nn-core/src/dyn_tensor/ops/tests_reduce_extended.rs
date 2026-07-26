// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended reduction tests: compensated (Kahan) sum/mean, var edge cases,
//! and additional min/max coverage.

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d};
use crate::DynTensor;

// -- sum_compensated ----------------------------------------------------------

#[test]
fn test_sum_compensated_1d_basic() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let y = x.sum_compensated(0).unwrap();
    assert_eq!(y.rank(), 0);
    let v = y.to_scalar::<f32>().unwrap();
    assert!(
        approx_eq(v, 10.0, 1e-6),
        "sum_compensated should be 10.0, got {v}"
    );
}

#[test]
fn test_sum_compensated_precision() {
    // Kahan summation should be more precise than naive for many small values
    // added to a large value. 1.0 + 1e-7 * 10_000_000 = 2.0
    let mut data = vec![1.0_f32];
    data.extend(std::iter::repeat_n(1e-4, 10_000));
    let x = DynTensor::from_vec(data, &[10_001], &cpu()).unwrap();
    let sum = x.sum_compensated(0).unwrap();
    let v = sum.to_scalar::<f32>().unwrap();
    // Expected: 1.0 + 10_000 * 1e-4 = 2.0
    assert!(
        (v - 2.0).abs() < 1e-3,
        "compensated sum should be ≈ 2.0, got {v}"
    );
}

#[test]
fn test_sum_compensated_keepdim() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let y = x.sum_compensated_keepdim(1).unwrap();
    assert_eq!(y.dims(), &[2, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], 3.0, 1e-6),
        "row 0 sum = 3, got {}",
        vals[0]
    );
    assert!(
        approx_eq(vals[1], 7.0, 1e-6),
        "row 1 sum = 7, got {}",
        vals[1]
    );
}

// -- mean_compensated ---------------------------------------------------------

#[test]
fn test_mean_compensated_1d_basic() {
    let x = t1d(&[2.0, 4.0, 6.0, 8.0]);
    let y = x.mean_compensated(0).unwrap();
    let v = y.to_scalar::<f32>().unwrap();
    assert!(
        approx_eq(v, 5.0, 1e-6),
        "mean_compensated should be 5.0, got {v}"
    );
}

#[test]
fn test_mean_compensated_keepdim() {
    let x = t2d(&[1.0, 3.0, 5.0, 7.0], 2, 2);
    let y = x.mean_compensated_keepdim(1).unwrap();
    assert_eq!(y.dims(), &[2, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], 2.0, 1e-6),
        "row 0 mean = 2, got {}",
        vals[0]
    );
    assert!(
        approx_eq(vals[1], 6.0, 1e-6),
        "row 1 mean = 6, got {}",
        vals[1]
    );
}

#[test]
fn test_mean_compensated_2d_dim0() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let y = x.mean_compensated(0).unwrap();
    assert_eq!(y.dims(), &[2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        approx_eq(vals[0], 2.0, 1e-6),
        "col 0 mean = 2, got {}",
        vals[0]
    );
    assert!(
        approx_eq(vals[1], 3.0, 1e-6),
        "col 1 mean = 3, got {}",
        vals[1]
    );
}

// -- max_keepdim / min_keepdim ------------------------------------------------

#[test]
fn test_max_keepdim_2d() {
    let x = t2d(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], 2, 3);
    let y = x.max_keepdim(1).unwrap();
    assert_eq!(y.dims(), &[2, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 6.0]);
}

#[test]
fn test_min_keepdim_2d() {
    let x = t2d(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], 2, 3);
    let y = x.min_keepdim(1).unwrap();
    assert_eq!(y.dims(), &[2, 1]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0]);
}

// -- var (no keepdim) ---------------------------------------------------------

#[test]
fn test_var_ops_ext_1d() {
    // var([1, 2, 3, 4, 5]) = mean((x - mean)^2) = 2.0
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = x.var(0).unwrap();
    let v = y.to_scalar::<f32>().unwrap();
    assert!(approx_eq(v, 2.0, 1e-4), "var should be 2.0, got {v}");
}

// -- max_all / min_all --------------------------------------------------------

#[test]
fn test_max_all_2d() {
    let x = t2d(&[1.0, 5.0, 3.0, 2.0], 2, 2);
    let y = x.max_all().unwrap();
    let v = y.to_scalar::<f32>().unwrap();
    assert_eq!(v, 5.0);
}

#[test]
fn test_min_all_2d() {
    let x = t2d(&[-1.0, 5.0, 3.0, 2.0], 2, 2);
    let y = x.min_all().unwrap();
    let v = y.to_scalar::<f32>().unwrap();
    assert_eq!(v, -1.0);
}

// -- sum_all with negative values ---------------------------------------------

#[test]
fn test_sum_all_mixed_signs() {
    let x = t2d(&[-1.0, 2.0, -3.0, 4.0], 2, 2);
    let y = x.sum_all().unwrap();
    let v = y.to_scalar::<f32>().unwrap();
    assert!(approx_eq(v, 2.0, 1e-6), "sum_all should be 2.0, got {v}");
}
