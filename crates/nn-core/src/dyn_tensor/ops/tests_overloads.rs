#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor operator overloads (std::ops) and edge-case behavior.
//!
//! Covers all ownership combinations (&T op &T, T op T, &T op T, T op &T),
//! scalar operations, broadcasting through operators, chained and precedence
//! tests, division edge cases, negative values, and identity properties.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d};
use crate::DynTensor;

// ---------------------------------------------------------------------------
// Helper: approximate equality for flat f32 vectors
// ---------------------------------------------------------------------------

fn approx_eq(a: &DynTensor, b: &DynTensor) {
    let a_vals: Vec<f32> = a.to_vec1().unwrap();
    let b_vals: Vec<f32> = b.to_vec1().unwrap();
    assert_eq!(a_vals.len(), b_vals.len());
    for (x, y) in a_vals.iter().zip(b_vals.iter()) {
        assert!((x - y).abs() < 1e-6, "{x} != {y}");
    }
}

fn approx_eq_flat(a: &DynTensor, expected: &[f32]) {
    let vals = a.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        vals.len(),
        expected.len()
    );
    for (i, (got, want)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "index {i}: got {got}, expected {want}"
        );
    }
}

// ===========================================================================
// Section 1: &Tensor op &Tensor (ref + ref)
// ===========================================================================

#[test]
fn test_ref_ref_add() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let c = (&a + &b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![4.0, 6.0]);
}

#[test]
fn test_ref_ref_sub() {
    let a = t1d(&[5.0, 3.0]);
    let b = t1d(&[1.0, 2.0]);
    let c = (&a - &b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![4.0, 1.0]);
}

#[test]
fn test_ref_ref_mul() {
    let a = t1d(&[2.0, 3.0]);
    let b = t1d(&[4.0, 5.0]);
    let c = (&a * &b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![8.0, 15.0]);
}

#[test]
fn test_ref_ref_div() {
    let a = t1d(&[10.0, 6.0]);
    let b = t1d(&[2.0, 3.0]);
    let c = (&a / &b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 2.0]);
}

// ===========================================================================
// Section 2: Tensor op Tensor (owned + owned)
// ===========================================================================

#[test]
fn test_owned_owned_add() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = (a + b).unwrap();
    approx_eq(&c, &t1d(&[5.0, 7.0, 9.0]));
}

#[test]
fn test_owned_owned_sub() {
    let a = t1d(&[5.0, 7.0, 9.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let c = (a - b).unwrap();
    approx_eq(&c, &t1d(&[4.0, 5.0, 6.0]));
}

#[test]
fn test_owned_owned_mul() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let c = (a * b).unwrap();
    approx_eq(&c, &t1d(&[10.0, 18.0, 28.0]));
}

#[test]
fn test_owned_owned_div() {
    let a = t1d(&[10.0, 18.0, 28.0]);
    let b = t1d(&[2.0, 3.0, 4.0]);
    let c = (a / b).unwrap();
    approx_eq(&c, &t1d(&[5.0, 6.0, 7.0]));
}

// ===========================================================================
// Section 3: &Tensor op Tensor (ref + owned)
// ===========================================================================

#[test]
fn test_ref_owned_add() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = (&a + b).unwrap();
    approx_eq(&c, &t1d(&[5.0, 7.0, 9.0]));
}

#[test]
fn test_ref_owned_sub() {
    let a = t1d(&[5.0, 7.0, 9.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let c = (&a - b).unwrap();
    approx_eq(&c, &t1d(&[4.0, 5.0, 6.0]));
}

#[test]
fn test_ref_owned_mul() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let c = (&a * b).unwrap();
    approx_eq(&c, &t1d(&[10.0, 18.0, 28.0]));
}

#[test]
fn test_ref_owned_div() {
    let a = t1d(&[10.0, 18.0, 28.0]);
    let b = t1d(&[2.0, 3.0, 4.0]);
    let c = (&a / b).unwrap();
    approx_eq(&c, &t1d(&[5.0, 6.0, 7.0]));
}

// ===========================================================================
// Section 4: Tensor op &Tensor (owned + ref)
// ===========================================================================

#[test]
fn test_owned_ref_add() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = (a + &b).unwrap();
    approx_eq(&c, &t1d(&[5.0, 7.0, 9.0]));
}

#[test]
fn test_owned_ref_sub() {
    let a = t1d(&[5.0, 7.0, 9.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let c = (a - &b).unwrap();
    approx_eq(&c, &t1d(&[4.0, 5.0, 6.0]));
}

#[test]
fn test_owned_ref_mul() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let c = (a * &b).unwrap();
    approx_eq(&c, &t1d(&[10.0, 18.0, 28.0]));
}

#[test]
fn test_owned_ref_div() {
    let a = t1d(&[10.0, 18.0, 28.0]);
    let b = t1d(&[2.0, 3.0, 4.0]);
    let c = (a / &b).unwrap();
    approx_eq(&c, &t1d(&[5.0, 6.0, 7.0]));
}

// ===========================================================================
// Section 5: Scalar operations (tensor op f64)
// ===========================================================================

#[test]
fn test_ref_add_scalar() {
    let a = t1d(&[1.0, 2.0]);
    let c = (&a + 10.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![11.0, 12.0]);
}

#[test]
fn test_ref_sub_scalar() {
    let a = t1d(&[10.0, 20.0]);
    let c = (&a - 3.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![7.0, 17.0]);
}

#[test]
fn test_ref_mul_scalar() {
    let a = t1d(&[1.0, 2.0]);
    let c = (&a * 3.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![3.0, 6.0]);
}

#[test]
fn test_ref_div_scalar() {
    let a = t1d(&[10.0, 6.0]);
    let c = (&a / 2.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 3.0]);
}

#[test]
fn test_owned_add_scalar() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = (a + 100.0).unwrap();
    approx_eq(&c, &t1d(&[101.0, 102.0, 103.0]));
}

#[test]
fn test_owned_sub_scalar() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let c = (a - 5.0).unwrap();
    approx_eq(&c, &t1d(&[5.0, 15.0, 25.0]));
}

#[test]
fn test_owned_mul_scalar() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let c = (a * 3.0).unwrap();
    approx_eq(&c, &t1d(&[6.0, 9.0, 12.0]));
}

#[test]
fn test_owned_div_scalar() {
    let a = t1d(&[6.0, 9.0, 12.0]);
    let c = (a / 3.0).unwrap();
    approx_eq(&c, &t1d(&[2.0, 3.0, 4.0]));
}

// ===========================================================================
// Section 6: Reverse scalar operations (f64 op tensor)
// ===========================================================================

#[test]
fn test_scalar_add_owned() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = (10.0 + a).unwrap();
    approx_eq(&c, &t1d(&[11.0, 12.0, 13.0]));
}

#[test]
fn test_scalar_add_ref() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = (10.0 + &a).unwrap();
    approx_eq(&c, &t1d(&[11.0, 12.0, 13.0]));
}

#[test]
fn test_scalar_sub_owned() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = (10.0 - a).unwrap();
    approx_eq(&c, &t1d(&[9.0, 8.0, 7.0]));
}

#[test]
fn test_scalar_sub_ref() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = (10.0 - &a).unwrap();
    approx_eq(&c, &t1d(&[9.0, 8.0, 7.0]));
}

#[test]
fn test_scalar_mul_owned() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let c = (3.0 * a).unwrap();
    approx_eq(&c, &t1d(&[6.0, 9.0, 12.0]));
}

#[test]
fn test_scalar_mul_ref() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let c = (3.0 * &a).unwrap();
    approx_eq(&c, &t1d(&[6.0, 9.0, 12.0]));
}

#[test]
fn test_scalar_div_owned() {
    let a = t1d(&[2.0, 4.0, 5.0]);
    let c = (20.0 / a).unwrap();
    approx_eq(&c, &t1d(&[10.0, 5.0, 4.0]));
}

#[test]
fn test_scalar_div_ref() {
    let a = t1d(&[2.0, 4.0, 5.0]);
    let c = (20.0 / &a).unwrap();
    approx_eq(&c, &t1d(&[10.0, 5.0, 4.0]));
}

// ===========================================================================
// Section 7: Negation
// ===========================================================================

#[test]
fn test_neg_ref() {
    let a = t1d(&[1.0, -2.0, 3.0]);
    let c = (-&a).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn test_neg_owned() {
    let a = t1d(&[1.0, -2.0, 3.0]);
    let c = (-a).unwrap();
    approx_eq(&c, &t1d(&[-1.0, 2.0, -3.0]));
}

#[test]
fn test_double_neg_is_identity() {
    let a = t1d(&[1.0, -2.0, 0.0, 3.5]);
    let c = (-(-&a).unwrap()).unwrap();
    approx_eq(&c, &a);
}

// ===========================================================================
// Section 8: Broadcasting through operators
// ===========================================================================

#[test]
fn test_broadcast_add_3x1_plus_1x4() {
    // [3,1] + [1,4] -> [3,4]
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[1, 4], &cpu()).unwrap();
    let c = (&a + &b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    approx_eq_flat(
        &c,
        &[
            11.0, 21.0, 31.0, 41.0, // row 0: 1 + [10,20,30,40]
            12.0, 22.0, 32.0, 42.0, // row 1: 2 + [10,20,30,40]
            13.0, 23.0, 33.0, 43.0, // row 2: 3 + [10,20,30,40]
        ],
    );
}

#[test]
fn test_broadcast_sub_3x1_minus_1x4() {
    let a = DynTensor::new(&[10.0, 20.0, 30.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let c = (&a - &b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    approx_eq_flat(
        &c,
        &[
            9.0, 8.0, 7.0, 6.0, // 10 - [1,2,3,4]
            19.0, 18.0, 17.0, 16.0, // 20 - [1,2,3,4]
            29.0, 28.0, 27.0, 26.0, // 30 - [1,2,3,4]
        ],
    );
}

#[test]
fn test_broadcast_mul_3x1_times_1x4() {
    let a = DynTensor::new(&[2.0, 3.0, 4.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 10.0, 100.0, 1000.0], &[1, 4], &cpu()).unwrap();
    let c = (&a * &b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    approx_eq_flat(
        &c,
        &[
            2.0, 20.0, 200.0, 2000.0, 3.0, 30.0, 300.0, 3000.0, 4.0, 40.0, 400.0, 4000.0,
        ],
    );
}

#[test]
fn test_broadcast_div_3x1_over_1x4() {
    let a = DynTensor::new(&[12.0, 24.0, 36.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let c = (&a / &b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    approx_eq_flat(
        &c,
        &[
            12.0, 6.0, 4.0, 3.0, // 12 / [1,2,3,4]
            24.0, 12.0, 8.0, 6.0, // 24 / [1,2,3,4]
            36.0, 18.0, 12.0, 9.0, // 36 / [1,2,3,4]
        ],
    );
}

#[test]
fn test_broadcast_add_1d_plus_2d() {
    // [4] + [2,4] -> [2,4] (right-aligned broadcast)
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let b = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 2, 4);
    let c = (&a + &b).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
    approx_eq_flat(&c, &[11.0, 22.0, 33.0, 44.0, 51.0, 62.0, 73.0, 84.0]);
}

#[test]
fn test_broadcast_owned_add() {
    // Owned + Owned with broadcasting
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::new(&[10.0, 20.0], &[1, 2], &cpu()).unwrap();
    let c = (a + b).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    approx_eq_flat(&c, &[11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
}

// ===========================================================================
// Section 9: Chained operations
// ===========================================================================

#[test]
fn test_chained_add_then_mul() {
    // (a + b) * c
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = t1d(&[2.0, 2.0, 2.0]);
    let result = ((&a + &b).unwrap() * &c).unwrap();
    approx_eq(&result, &t1d(&[10.0, 14.0, 18.0]));
}

#[test]
fn test_chained_sub_then_div() {
    // (a - b) / c
    let a = t1d(&[10.0, 20.0, 30.0]);
    let b = t1d(&[2.0, 4.0, 6.0]);
    let c = t1d(&[2.0, 4.0, 8.0]);
    let result = ((&a - &b).unwrap() / &c).unwrap();
    approx_eq(&result, &t1d(&[4.0, 4.0, 3.0]));
}

#[test]
fn test_chained_mul_then_add() {
    // a * b + c
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let c = t1d(&[1.0, 1.0, 1.0]);
    let result = ((&a * &b).unwrap() + &c).unwrap();
    approx_eq(&result, &t1d(&[11.0, 19.0, 29.0]));
}

#[test]
fn test_chained_three_ops() {
    // ((a + b) * c) - d
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let c = t1d(&[2.0, 3.0]);
    let d = t1d(&[1.0, 1.0]);
    let result = (((&a + &b).unwrap() * &c).unwrap() - &d).unwrap();
    // (1+3)*2 - 1 = 7, (2+4)*3 - 1 = 17
    approx_eq(&result, &t1d(&[7.0, 17.0]));
}

#[test]
fn test_chained_scalar_and_tensor() {
    // (a + 1.0) * b
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[10.0, 20.0, 30.0]);
    let result = ((&a + 1.0).unwrap() * &b).unwrap();
    approx_eq(&result, &t1d(&[20.0, 60.0, 120.0]));
}

// ===========================================================================
// Section 10: Operator precedence (Rust's default: * and / bind tighter)
// ===========================================================================
//
// Note: Rust enforces standard arithmetic precedence at the language level.
// Since our Output = Result<DynTensor>, we cannot write `a + b * c` directly
// because `b * c` yields Result<DynTensor>, and `a + Result<DynTensor>` is
// not defined. Instead we verify that manually parenthesized expressions
// produce the expected mathematical results.

#[test]
fn test_precedence_mul_before_add() {
    // Emulate: a + b * c = a + (b * c) = 1 + 2*3 = 7
    let a = t1d(&[1.0]);
    let b = t1d(&[2.0]);
    let c = t1d(&[3.0]);
    let bc = (&b * &c).unwrap();
    let result = (&a + &bc).unwrap();
    approx_eq(&result, &t1d(&[7.0]));
}

#[test]
fn test_precedence_div_before_sub() {
    // Emulate: a - b / c = a - (b / c) = 10 - 6/2 = 7
    let a = t1d(&[10.0]);
    let b = t1d(&[6.0]);
    let c = t1d(&[2.0]);
    let bc = (&b / &c).unwrap();
    let result = (&a - &bc).unwrap();
    approx_eq(&result, &t1d(&[7.0]));
}

#[test]
fn test_add_then_mul_different_from_mul_then_add() {
    // (a + b) * c != a + (b * c)
    let a = t1d(&[2.0]);
    let b = t1d(&[3.0]);
    let c = t1d(&[4.0]);
    let left = ((&a + &b).unwrap() * &c).unwrap(); // (2+3)*4 = 20
    let right = (&a + &(&b * &c).unwrap()).unwrap(); // 2+(3*4) = 14
    approx_eq(&left, &t1d(&[20.0]));
    approx_eq(&right, &t1d(&[14.0]));
}

// ===========================================================================
// Section 11: Division edge cases
// ===========================================================================

#[test]
fn test_div_by_zero_tensor_op_returns_error() {
    // Division through the / operator (not just broadcast_div)
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[0.0, 1.0, 0.0]);
    let result = &a / &b;
    assert!(result.is_err(), "tensor / zero-tensor should error");
}

#[test]
fn test_div_zero_by_zero_tensor_op_returns_error() {
    let a = t1d(&[0.0]);
    let b = t1d(&[0.0]);
    let result = &a / &b;
    assert!(result.is_err(), "0/0 via operator should error");
}

#[test]
fn test_div_scalar_zero_returns_error() {
    let a = t1d(&[5.0, -3.0]);
    let result = &a / 0.0;
    assert!(result.is_err(), "tensor / 0.0 should error");
}

#[test]
fn test_div_scalar_zero_owned_returns_error() {
    let a = t1d(&[5.0, -3.0]);
    let result = a / 0.0;
    assert!(result.is_err(), "owned tensor / 0.0 should error");
}

#[test]
fn test_div_by_nonzero_succeeds() {
    let a = t1d(&[6.0, 8.0, 10.0]);
    let b = t1d(&[2.0, 4.0, 5.0]);
    let c = (&a / &b).unwrap();
    approx_eq(&c, &t1d(&[3.0, 2.0, 2.0]));
}

#[test]
fn test_div_scalar_near_zero_succeeds() {
    // Near-zero divisor produces large but finite values (not an error).
    let a = t1d(&[1.0]);
    let result = &a / 1e-30;
    assert!(result.is_ok(), "near-zero scalar division should succeed");
}

#[test]
fn test_scalar_div_by_zero_tensor_returns_error() {
    // f64 / tensor where tensor contains zero
    let a = t1d(&[0.0, 1.0]);
    // 1.0 / [0, 1] -> recip of [0,1] produces Inf/NaN, should error
    let result = 1.0 / &a;
    assert!(
        result.is_err(),
        "scalar / zero-containing-tensor should error"
    );
}

// ===========================================================================
// Section 12: Negative values
// ===========================================================================

#[test]
fn test_add_negative_values() {
    let a = t1d(&[-1.0, -2.0, -3.0]);
    let b = t1d(&[-4.0, -5.0, -6.0]);
    let c = (&a + &b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-5.0, -7.0, -9.0]);
}

#[test]
fn test_sub_negative_values() {
    let a = t1d(&[-1.0, -2.0]);
    let b = t1d(&[-4.0, -5.0]);
    let c = (&a - &b).unwrap();
    // -1 - (-4) = 3, -2 - (-5) = 3
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![3.0, 3.0]);
}

#[test]
fn test_mul_negative_values() {
    let a = t1d(&[-2.0, 3.0, -4.0]);
    let b = t1d(&[5.0, -6.0, -7.0]);
    let c = (&a * &b).unwrap();
    // -2*5=-10, 3*(-6)=-18, (-4)*(-7)=28
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-10.0, -18.0, 28.0]);
}

#[test]
fn test_div_negative_values() {
    let a = t1d(&[-10.0, 6.0, -8.0]);
    let b = t1d(&[2.0, -3.0, -4.0]);
    let c = (&a / &b).unwrap();
    // -10/2=-5, 6/(-3)=-2, (-8)/(-4)=2
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-5.0, -2.0, 2.0]);
}

#[test]
fn test_neg_scalar_mul() {
    let a = t1d(&[1.0, -2.0, 3.0]);
    let c = (&a * -1.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn test_add_mixed_sign() {
    let a = t1d(&[-5.0, 3.0, 0.0]);
    let b = t1d(&[5.0, -3.0, 0.0]);
    let c = (&a + &b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![0.0, 0.0, 0.0]);
}

// ===========================================================================
// Section 13: Identity properties
// ===========================================================================

#[test]
fn test_add_zero_identity() {
    // a + 0 = a
    let a = t1d(&[1.0, -2.0, 3.5, 0.0]);
    let result = (&a + 0.0).unwrap();
    approx_eq(&result, &a);
}

#[test]
fn test_sub_zero_identity() {
    // a - 0 = a
    let a = t1d(&[1.0, -2.0, 3.5, 0.0]);
    let result = (&a - 0.0).unwrap();
    approx_eq(&result, &a);
}

#[test]
fn test_mul_one_identity() {
    // a * 1 = a
    let a = t1d(&[1.0, -2.0, 3.5, 0.0]);
    let result = (&a * 1.0).unwrap();
    approx_eq(&result, &a);
}

#[test]
fn test_div_one_identity() {
    // a / 1 = a
    let a = t1d(&[1.0, -2.0, 3.5, 0.0]);
    let result = (&a / 1.0).unwrap();
    approx_eq(&result, &a);
}

#[test]
fn test_add_zero_tensor_identity() {
    // a + zeros = a
    let a = t1d(&[1.0, -2.0, 3.5]);
    let zeros = t1d(&[0.0, 0.0, 0.0]);
    let result = (&a + &zeros).unwrap();
    approx_eq(&result, &a);
}

#[test]
fn test_mul_one_tensor_identity() {
    // a * ones = a
    let a = t1d(&[1.0, -2.0, 3.5]);
    let ones = t1d(&[1.0, 1.0, 1.0]);
    let result = (&a * &ones).unwrap();
    approx_eq(&result, &a);
}

#[test]
fn test_mul_zero_annihilation() {
    // a * 0 = 0
    let a = t1d(&[1.0, -2.0, 3.5, 100.0]);
    let result = (&a * 0.0).unwrap();
    approx_eq(&result, &t1d(&[0.0, 0.0, 0.0, 0.0]));
}

#[test]
fn test_sub_self_is_zero() {
    // a - a = 0
    let a = t1d(&[1.0, -2.0, 3.5]);
    let result = (&a - &a).unwrap();
    approx_eq(&result, &t1d(&[0.0, 0.0, 0.0]));
}

#[test]
fn test_div_self_is_one() {
    // a / a = 1 (for nonzero a)
    let a = t1d(&[1.0, -2.0, 3.5]);
    let result = (&a / &a).unwrap();
    approx_eq(&result, &t1d(&[1.0, 1.0, 1.0]));
}

// ===========================================================================
// Section 14: Commutativity and associativity
// ===========================================================================

#[test]
fn test_add_commutative() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let ab = (&a + &b).unwrap();
    let ba = (&b + &a).unwrap();
    approx_eq(&ab, &ba);
}

#[test]
fn test_mul_commutative() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let ab = (&a * &b).unwrap();
    let ba = (&b * &a).unwrap();
    approx_eq(&ab, &ba);
}

#[test]
fn test_add_associative() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let c = t1d(&[5.0, 6.0]);
    let left = ((&a + &b).unwrap() + &c).unwrap(); // (a+b)+c
    let right = (&a + &(&b + &c).unwrap()).unwrap(); // a+(b+c)
    approx_eq(&left, &right);
}

#[test]
fn test_mul_associative() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let c = t1d(&[5.0, 6.0]);
    let left = ((&a * &b).unwrap() * &c).unwrap();
    let right = (&a * &(&b * &c).unwrap()).unwrap();
    approx_eq(&left, &right);
}

#[test]
fn test_scalar_add_commutative() {
    // tensor + scalar == scalar + tensor
    let a = t1d(&[1.0, 2.0, 3.0]);
    let ts = (&a + 5.0).unwrap();
    let st = (5.0 + &a).unwrap();
    approx_eq(&ts, &st);
}

#[test]
fn test_scalar_mul_commutative() {
    // tensor * scalar == scalar * tensor
    let a = t1d(&[1.0, 2.0, 3.0]);
    let ts = (&a * 5.0).unwrap();
    let st = (5.0 * &a).unwrap();
    approx_eq(&ts, &st);
}

// ===========================================================================
// Section 15: Distributive property
// ===========================================================================

#[test]
fn test_distributive_mul_over_add() {
    // c * (a + b) == c*a + c*b
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = t1d(&[2.0, 3.0, 4.0]);
    let left = (&c * &(&a + &b).unwrap()).unwrap();
    let right = ((&c * &a).unwrap() + &(&c * &b).unwrap()).unwrap();
    approx_eq(&left, &right);
}

// ===========================================================================
// Section 16: sub_scalar and div_scalar method tests
// ===========================================================================

#[test]
fn test_sub_scalar_method() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let c = a.sub_scalar(5.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 15.0, 25.0]);
}

#[test]
fn test_div_scalar_method() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let c = a.div_scalar(5.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_div_scalar_zero_method_returns_error() {
    let a = t1d(&[5.0, -3.0]);
    let result = a.div_scalar(0.0);
    assert!(result.is_err(), "div_scalar(0.0) should error");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("zero"), "error should mention zero: {err}");
}

// ===========================================================================
// Section 17: powf edge cases (CPU)
// ===========================================================================

#[test]
fn test_powf_negative_base_cpu() {
    let a = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = a.powf(2.0).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!((vals[0] - 4.0).abs() < 1e-6);
    assert!((vals[1] - 1.0).abs() < 1e-6);
    assert!((vals[2] - 0.0).abs() < 1e-6);
    assert!((vals[3] - 1.0).abs() < 1e-6);
    assert!((vals[4] - 4.0).abs() < 1e-6);
}

#[test]
fn test_powf_negative_base_fractional_exponent() {
    let a = t1d(&[-2.0]);
    let result = a.powf(0.5).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!(vals[0].is_nan(), "powf(-2, 0.5) should be NaN on CPU");
}

// ===========================================================================
// Section 18: 2D tensor operations through operators
// ===========================================================================

#[test]
fn test_2d_add_same_shape() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    let c = (&a + &b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    approx_eq_flat(&c, &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_2d_mul_same_shape() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[2.0, 3.0, 4.0, 5.0], 2, 2);
    let c = (&a * &b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    approx_eq_flat(&c, &[2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_2d_scalar_add() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let c = (&a + 10.0).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    approx_eq_flat(&c, &[11.0, 12.0, 13.0, 14.0]);
}

// ===========================================================================
// Section 19: Large-ish tensor to verify no off-by-one in broadcast
// ===========================================================================

#[test]
fn test_broadcast_column_vector_plus_row_vector() {
    // [5,1] + [1,3] -> [5,3]
    let col = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5, 1], &cpu()).unwrap();
    let row = DynTensor::new(&[10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let result = (&col + &row).unwrap();
    assert_eq!(result.dims(), &[5, 3]);
    approx_eq_flat(
        &result,
        &[
            11.0, 21.0, 31.0, // 1 + [10,20,30]
            12.0, 22.0, 32.0, // 2 + [10,20,30]
            13.0, 23.0, 33.0, // 3 + [10,20,30]
            14.0, 24.0, 34.0, // 4 + [10,20,30]
            15.0, 25.0, 35.0, // 5 + [10,20,30]
        ],
    );
}

// ===========================================================================
// Section 20: sub is not commutative, div is not commutative
// ===========================================================================

#[test]
fn test_sub_not_commutative() {
    let a = t1d(&[5.0, 10.0]);
    let b = t1d(&[1.0, 2.0]);
    let ab = (&a - &b).unwrap();
    let ba = (&b - &a).unwrap();
    // a-b = [4,8], b-a = [-4,-8]
    approx_eq(&ab, &t1d(&[4.0, 8.0]));
    approx_eq(&ba, &t1d(&[-4.0, -8.0]));
}

#[test]
fn test_div_not_commutative() {
    let a = t1d(&[10.0, 8.0]);
    let b = t1d(&[2.0, 4.0]);
    let ab = (&a / &b).unwrap();
    let ba = (&b / &a).unwrap();
    // a/b = [5,2], b/a = [0.2, 0.5]
    approx_eq(&ab, &t1d(&[5.0, 2.0]));
    approx_eq(&ba, &t1d(&[0.2, 0.5]));
}

// ===========================================================================
// Section 21: scalar sub/div non-commutativity
// ===========================================================================

#[test]
fn test_scalar_sub_not_commutative() {
    let a = t1d(&[1.0, 2.0]);
    let ts = (&a - 10.0).unwrap(); // [1-10, 2-10] = [-9, -8]
    let st = (10.0 - &a).unwrap(); // [10-1, 10-2] = [9, 8]
    approx_eq(&ts, &t1d(&[-9.0, -8.0]));
    approx_eq(&st, &t1d(&[9.0, 8.0]));
}

#[test]
fn test_scalar_div_not_commutative() {
    let a = t1d(&[2.0, 5.0]);
    let ts = (&a / 10.0).unwrap(); // [2/10, 5/10] = [0.2, 0.5]
    let st = (10.0 / &a).unwrap(); // [10/2, 10/5] = [5.0, 2.0]
    approx_eq(&ts, &t1d(&[0.2, 0.5]));
    approx_eq(&st, &t1d(&[5.0, 2.0]));
}
