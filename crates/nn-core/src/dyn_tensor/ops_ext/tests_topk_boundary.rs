#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Topk and powf boundary condition tests for DynTensor ops.
//!
//! Extracted from `tests_extra.rs` to keep files under 500 lines.

use crate::dyn_tensor::test_helpers::{cpu, tnd};
use crate::dyn_tensor::DynTensor;

// -- topk tests ---------------------------------------------------------------

#[test]
fn test_topk_1d() {
    let x = tnd(&[1.0, 5.0, 3.0, 2.0, 4.0], &[5]);
    let (vals, idxs) = x.topk(0, 3).unwrap();
    assert_eq!(vals.dims(), &[3]);
    assert_eq!(idxs.dims(), &[3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![5.0, 4.0, 3.0]);
    let i = idxs.as_cpu_u32().unwrap();
    assert_eq!(i[ndarray::IxDyn(&[0])], 1);
    assert_eq!(i[ndarray::IxDyn(&[1])], 4);
    assert_eq!(i[ndarray::IxDyn(&[2])], 2);
}

#[test]
fn test_topk_2d_last_dim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let (vals, _idxs) = x.topk(1, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [1,5,3] -> top-2: [5,3]
    assert_eq!(v[0], 5.0);
    assert_eq!(v[1], 3.0);
    // Row 1: [4,2,6] -> top-2: [6,4]
    assert_eq!(v[2], 6.0);
    assert_eq!(v[3], 4.0);
}

#[test]
fn test_topk_k_equals_dim() {
    let x = tnd(&[3.0, 1.0, 2.0], &[3]);
    let (vals, _) = x.topk(0, 3).unwrap();
    assert_eq!(vals.dims(), &[3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![3.0, 2.0, 1.0]);
}

/// topk along dim=0 on a 2D tensor: exercises the non-last-dim layout path.
/// Regression test for #1347 — sequential push into flat Vec produced lane-contiguous
/// rather than row-major output. The lanes_mut fix writes directly into the
/// output ndarray, preserving C-contiguous layout.
#[test]
fn test_topk_2d_dim0() {
    // [4, 3] input — each column is an independent lane for dim=0.
    // Col 0: [10, 40, 30, 20] -> top-2: [40, 30] (rows 1, 2)
    // Col 1: [50, 20, 60, 10] -> top-2: [60, 50] (rows 2, 0)
    // Col 2: [30, 30, 10, 40] -> top-2: [40, 30] (rows 3, 0 or 1)
    #[rustfmt::skip]
    let x = tnd(&[
        10.0, 50.0, 30.0,
        40.0, 20.0, 30.0,
        30.0, 60.0, 10.0,
        20.0, 10.0, 40.0,
    ], &[4, 3]);
    let (vals, idxs) = x.topk(0, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 3]);
    assert_eq!(idxs.dims(), &[2, 3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    let idx = idxs.to_flat_vec::<u32>().unwrap();
    // Row-major output: [2, 3] = [[top1_col0, top1_col1, top1_col2],
    //                              [top2_col0, top2_col1, top2_col2]]
    // Top-1 per column: [40, 60, 40]
    assert_eq!(v[0], 40.0, "col0 top1");
    assert_eq!(v[1], 60.0, "col1 top1");
    assert_eq!(v[2], 40.0, "col2 top1");
    // Top-2 per column: [30, 50, 30]
    assert_eq!(v[3], 30.0, "col0 top2");
    assert_eq!(v[4], 50.0, "col1 top2");
    assert_eq!(v[5], 30.0, "col2 top2");
    // Verify indices are valid row indices for each column.
    assert_eq!(idx[0], 1, "col0 top1 should be row 1"); // 40.0 at row 1
    assert_eq!(idx[1], 2, "col1 top1 should be row 2"); // 60.0 at row 2
    assert_eq!(idx[2], 3, "col2 top1 should be row 3"); // 40.0 at row 3
    assert_eq!(idx[3], 2, "col0 top2 should be row 2"); // 30.0 at row 2
    assert_eq!(idx[4], 0, "col1 top2 should be row 0"); // 50.0 at row 0
                                                        // col2 top2 = 30.0 at row 0 or row 1 (both valid, sort is not stable).
    assert!(
        idx[5] == 0 || idx[5] == 1,
        "col2 top2 should be row 0 or 1, got {}",
        idx[5]
    );
}

#[test]
fn test_topk_invalid_k() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.topk(0, 0).is_err());
    assert!(x.topk(0, 4).is_err());
}

#[test]
fn test_topk_invalid_dim() {
    let x = tnd(&[1.0, 2.0], &[2]);
    assert!(x.topk(1, 1).is_err());
}

#[test]
fn test_topk_nan_returns_error() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    assert!(x.topk(0, 2).is_err());
}

/// Regression test for #1473: topk CPU stored indices as f32, losing precision
/// for dim_size > 2^24 (16,777,216). Verifies u32 indices are exact even for
/// indices beyond f32's integer precision limit.
#[test]
fn test_topk_large_dim_index_precision() {
    // Create a minimal tensor where the top value is at an index > 2^24.
    // We use dim_size = 2^24 + 2 = 16,777,218.
    let dim_size: usize = (1 << 24) + 2;
    let target_idx = dim_size - 1; // = 16,777,217, not exactly representable in f32

    // Allocate with all zeros, then place the max value at the last position.
    let mut data = vec![0.0f32; dim_size];
    data[target_idx] = 1.0;

    let x = DynTensor::from_vec(data, &[dim_size], &cpu()).unwrap();
    let (vals, idxs) = x.topk(0, 1).unwrap();

    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v[0], 1.0, "top value should be 1.0");

    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(
        i[0], target_idx as u32,
        "index should be exactly {target_idx}, got {}",
        i[0]
    );
}

// -- powf boundary condition tests (algorithm_audit) --------------------------

#[test]
fn test_powf_zero_exponent_returns_ones() {
    // x^0 = 1 for all x, including 0^0 = 1.
    // Regression: GPU path used exp(0 * log(0)) = exp(NaN) = NaN.
    let x = tnd(&[0.0, 1.0, -1.0, 5.0, f32::INFINITY], &[5]);
    let result = x.powf(0.0).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    for (i, &val) in v.iter().enumerate() {
        assert_eq!(val, 1.0, "x[{i}]^0 should be 1.0, got {val}");
    }
}

#[test]
fn test_powf_one_exponent_identity() {
    // x^1 = x for positive x.
    let x = tnd(&[0.0, 1.0, 2.0, 3.0], &[4]);
    let result = x.powf(1.0).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    for (i, &val) in v.iter().enumerate() {
        let expected = [0.0, 1.0, 2.0, 3.0][i];
        assert!(
            (val - expected).abs() < 1e-5,
            "x[{i}]^1 should be {expected}, got {val}"
        );
    }
}

#[test]
fn test_powf_negative_one_is_reciprocal() {
    // x^(-1) = 1/x for positive x.
    let x = tnd(&[1.0, 2.0, 4.0, 0.5], &[4]);
    let result = x.powf(-1.0).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    let expected = [1.0, 0.5, 0.25, 2.0];
    for (i, &val) in v.iter().enumerate() {
        assert!(
            (val - expected[i]).abs() < 1e-5,
            "x[{i}]^(-1) should be {}, got {val}",
            expected[i],
        );
    }
}
