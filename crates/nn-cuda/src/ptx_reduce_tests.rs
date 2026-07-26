// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External test suite for `ptx_reduce` — PTX generation validity and
//! reference implementation correctness for reduction operations.

use crate::ptx_reduce::{
    argmax_reference, argmin_reference, generate_argmax_ptx, generate_argmin_ptx, generate_max_ptx,
    generate_mean_ptx, generate_sum_ptx, max_reference, mean_reference, sum_reference,
};

// ---------------------------------------------------------------------------
// PTX generation validity — every reduction kernel must contain version,
// target, and entry point markers.
// ---------------------------------------------------------------------------

#[test]
fn test_sum_ptx_valid_structure() {
    let ptx = generate_sum_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_sum_f32"), "missing entry point");
}

#[test]
fn test_max_ptx_valid_structure() {
    let ptx = generate_max_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_max_f32"), "missing entry point");
}

#[test]
fn test_mean_ptx_valid_structure() {
    let ptx = generate_mean_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_mean_f32"), "missing entry point");
}

#[test]
fn test_argmax_ptx_valid_structure() {
    let ptx = generate_argmax_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_argmax_f32"), "missing entry point");
}

#[test]
fn test_argmin_ptx_valid_structure() {
    let ptx = generate_argmin_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_argmin_f32"), "missing entry point");
}

// ---------------------------------------------------------------------------
// Reference implementation tests — sum
// ---------------------------------------------------------------------------

#[test]
fn test_sum_reference_basic() {
    let data = [1.0_f32, 2.0, 3.0, 4.0];
    assert!((sum_reference(&data) - 10.0).abs() < 1e-6);
}

#[test]
fn test_sum_reference_single() {
    assert!((sum_reference(&[42.0]) - 42.0).abs() < 1e-6);
}

#[test]
fn test_sum_reference_negative() {
    let data = [-1.0_f32, -2.0, -3.0, -4.0];
    assert!((sum_reference(&data) - (-10.0)).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Reference implementation tests — max
// ---------------------------------------------------------------------------

#[test]
fn test_max_reference_basic() {
    let data = [1.0_f32, 4.0, 2.0, 3.0];
    assert!((max_reference(&data) - 4.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Reference implementation tests — mean
// ---------------------------------------------------------------------------

#[test]
fn test_mean_reference_basic() {
    let data = [2.0_f32, 4.0, 6.0, 8.0];
    assert!((mean_reference(&data) - 5.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Reference implementation tests — argmax
// ---------------------------------------------------------------------------

#[test]
fn test_argmax_reference_basic() {
    let data = [1.0_f32, 3.0, 2.0];
    assert_eq!(argmax_reference(&data), 1);
}

// ---------------------------------------------------------------------------
// Reference implementation tests — argmin
// ---------------------------------------------------------------------------

#[test]
fn test_argmin_reference_basic() {
    let data = [3.0_f32, 1.0, 2.0];
    assert_eq!(argmin_reference(&data), 1);
}

// ---------------------------------------------------------------------------
// PTX generation for different sizes
// ---------------------------------------------------------------------------

#[test]
fn test_reduce_ptx_different_sizes() {
    for &n in &[1_u32, 256, 1024, 65536] {
        let ptx_sum = generate_sum_ptx(n);
        assert!(
            ptx_sum.contains(".entry ptx_sum_f32"),
            "sum PTX missing entry for n={n}"
        );
        assert!(
            ptx_sum.contains(".target sm_70"),
            "sum PTX missing target for n={n}"
        );

        let ptx_max = generate_max_ptx(n);
        assert!(
            ptx_max.contains(".entry ptx_max_f32"),
            "max PTX missing entry for n={n}"
        );

        let ptx_mean = generate_mean_ptx(n);
        assert!(
            ptx_mean.contains(".entry ptx_mean_f32"),
            "mean PTX missing entry for n={n}"
        );

        let ptx_argmax = generate_argmax_ptx(n);
        assert!(
            ptx_argmax.contains(".entry ptx_argmax_f32"),
            "argmax PTX missing entry for n={n}"
        );

        let ptx_argmin = generate_argmin_ptx(n);
        assert!(
            ptx_argmin.contains(".entry ptx_argmin_f32"),
            "argmin PTX missing entry for n={n}"
        );
    }
}
