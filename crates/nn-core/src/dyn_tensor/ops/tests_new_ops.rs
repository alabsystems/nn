// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for newly implemented DynTensor operations: tan, ceil, sign, softsign,
//! reflection_pad2d, and constant_pad_nd.

use crate::dyn_tensor::test_helpers::{cpu, t1d};
use crate::DynTensor;

// -- tan ----------------------------------------------------------------------

#[test]
fn test_tan_known_values() {
    let x = t1d(&[
        0.0,
        std::f32::consts::FRAC_PI_4,
        -std::f32::consts::FRAC_PI_4,
    ]);
    let y = x.tan().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 0.0).abs() < 1e-6,
        "tan(0) should be 0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 1.0).abs() < 1e-5,
        "tan(pi/4) should be 1, got {}",
        vals[1]
    );
    assert!(
        (vals[2] - (-1.0)).abs() < 1e-5,
        "tan(-pi/4) should be -1, got {}",
        vals[2]
    );
}

#[test]
fn test_tan_shape_preserved() {
    let x = DynTensor::from_vec(vec![0.0, 0.5, -0.5, 1.0], &[2, 2], &cpu()).unwrap();
    let y = x.tan().unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        let input = [0.0_f32, 0.5, -0.5, 1.0][i];
        let expected = input.tan();
        assert!(
            (v - expected).abs() < 1e-5,
            "tan({input}) expected {expected}, got {v}"
        );
    }
}

// -- ceil ---------------------------------------------------------------------

#[test]
fn test_ceil_known_values() {
    let x = t1d(&[1.1, 2.9, -1.1, -2.9, 0.0, 3.0]);
    let y = x.ceil().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![2.0, 3.0, -1.0, -2.0, 0.0, 3.0]);
}

#[test]
fn test_ceil_shape_preserved() {
    let x = DynTensor::from_vec(vec![0.5, -0.5, 1.5, -1.5], &[2, 2], &cpu()).unwrap();
    let y = x.ceil().unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 0.0, 2.0, -1.0]);
}

// -- sign ---------------------------------------------------------------------

#[test]
fn test_sign_known_values() {
    let x = t1d(&[-5.0, -0.1, 0.0, 0.1, 5.0]);
    let y = x.sign().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_sign_shape_preserved() {
    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0, -2.0, 0.0], &[2, 3], &cpu()).unwrap();
    let y = x.sign().unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![-1.0, 0.0, 1.0, 1.0, -1.0, 0.0]);
}

// -- softsign -----------------------------------------------------------------

#[test]
fn test_softsign_known_values() {
    // softsign(x) = x / (1 + |x|)
    let x = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let y = x.softsign().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    let expected = [-2.0 / 3.0, -0.5, 0.0, 0.5, 2.0 / 3.0];
    for (i, (got, exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "softsign index {i}: expected {exp}, got {got}"
        );
    }
}

#[test]
fn test_softsign_output_range() {
    // softsign output is always in (-1, 1)
    let x = t1d(&[-1000.0, -10.0, 0.0, 10.0, 1000.0]);
    let y = x.softsign().unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!(
            *v > -1.0 && *v < 1.0,
            "softsign index {i}: expected in (-1, 1), got {v}"
        );
    }
}

// -- reflection_pad2d ---------------------------------------------------------

#[test]
fn test_reflection_pad2d_simple() {
    // 2x3 input, pad 1 on each side
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let y = x.reflection_pad2d(1, 1, 0, 0).unwrap();
    // Width padding: each row gets reflected: [2, 1, 2, 3, 2]
    assert_eq!(y.dims(), &[2, 5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_reflection_pad2d_all_sides() {
    // 3x3 input, pad 1 on all sides
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &cpu(),
    )
    .unwrap();
    let y = x.reflection_pad2d(1, 1, 1, 1).unwrap();
    assert_eq!(y.dims(), &[5, 5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Row 0 (reflected from row 1): [5, 4, 5, 6, 5]
    // Row 1 (reflected from row 0): [2, 1, 2, 3, 2]
    // Row 2 (original row 1):       [5, 4, 5, 6, 5]
    // Row 3 (original row 2):       [8, 7, 8, 9, 8]
    // Row 4 (reflected from row 1): [5, 4, 5, 6, 5]
    //
    // Wait - let me recalculate. Input is:
    // Row 0: [1, 2, 3]
    // Row 1: [4, 5, 6]
    // Row 2: [7, 8, 9]
    //
    // Width padding first (pad_left=1, pad_right=1):
    // Row 0: [2, 1, 2, 3, 2]
    // Row 1: [5, 4, 5, 6, 5]
    // Row 2: [8, 7, 8, 9, 8]
    //
    // Height padding (pad_top=1, pad_bottom=1) reflects rows:
    // Top: reflected from row 1 (index 1, which is the width-padded row 1): [5, 4, 5, 6, 5]
    // Original rows: [2, 1, 2, 3, 2], [5, 4, 5, 6, 5], [8, 7, 8, 9, 8]
    // Bottom: reflected from row 1 (index h-2=1): [5, 4, 5, 6, 5]
    let expected = vec![
        5.0, 4.0, 5.0, 6.0, 5.0, // reflected row 1
        2.0, 1.0, 2.0, 3.0, 2.0, // original row 0
        5.0, 4.0, 5.0, 6.0, 5.0, // original row 1
        8.0, 7.0, 8.0, 9.0, 8.0, // original row 2
        5.0, 4.0, 5.0, 6.0, 5.0, // reflected row 1
    ];
    assert_eq!(vals, expected);
}

#[test]
fn test_reflection_pad2d_error_too_large() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    // pad_left=2 >= width=2, should fail
    let err = x.reflection_pad2d(2, 0, 0, 0);
    assert!(err.is_err());
}

// -- constant_pad_nd ----------------------------------------------------------

#[test]
fn test_constant_pad_nd_1d() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.constant_pad_nd(&[2, 1], 0.0).unwrap();
    assert_eq!(y.dims(), &[6]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_constant_pad_nd_2d_nonzero_fill() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    // Pad last dim left=1, right=1
    let y = x.constant_pad_nd(&[1, 1], -1.0).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![-1.0, 1.0, 2.0, -1.0, -1.0, 3.0, 4.0, -1.0]);
}

#[test]
fn test_constant_pad_nd_2d_both_dims() {
    // Pad: last dim (1, 1) and second-to-last dim (1, 1)
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let y = x.constant_pad_nd(&[1, 1, 1, 1], 0.0).unwrap();
    assert_eq!(y.dims(), &[4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 0.0,
        0.0, 3.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(vals, expected);
}

#[test]
fn test_constant_pad_nd_no_padding() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.constant_pad_nd(&[0, 0], 5.0).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_constant_pad_nd_odd_length_error() {
    let x = t1d(&[1.0]);
    let err = x.constant_pad_nd(&[1, 2, 3], 0.0);
    assert!(err.is_err());
}
