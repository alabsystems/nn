#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor CPU arithmetic, elementwise, reduction, and matmul ops.

use super::*;
use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d};
use crate::DType;
use crate::DynTensor;

// -- Binary arithmetic --------------------------------------------------------

#[test]
fn test_add_same_shape() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sub_same_shape() {
    let a = t1d(&[5.0, 3.0, 1.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let c = a.sub(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![4.0, 1.0, -2.0]);
}

#[test]
fn test_mul_same_shape() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_div_same_shape() {
    let a = t1d(&[10.0, 9.0, 8.0]);
    let b = t1d(&[2.0, 3.0, 4.0]);
    let c = a.div(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 3.0, 2.0]);
}

#[test]
fn test_strict_add_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.strict_add(&b).is_err());
}

#[test]
fn test_strict_sub_same_shape() {
    let a = t1d(&[5.0, 3.0, 1.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let c = a.strict_sub(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![4.0, 1.0, -2.0]);
}

#[test]
fn test_strict_sub_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.strict_sub(&b).is_err());
}

#[test]
fn test_strict_mul_same_shape() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let c = a.strict_mul(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_strict_mul_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.strict_mul(&b).is_err());
}

#[test]
fn test_strict_div_same_shape() {
    let a = t1d(&[10.0, 9.0, 8.0]);
    let b = t1d(&[2.0, 3.0, 4.0]);
    let c = a.strict_div(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 3.0, 2.0]);
}

#[test]
fn test_strict_div_shape_mismatch_error() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.strict_div(&b).is_err());
}

#[test]
fn test_strict_ops_reject_broadcastable_shapes() {
    // [2,1] and [1,3] are broadcastable but not identical — strict ops must reject
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    assert!(a.strict_add(&b).is_err());
    assert!(a.strict_sub(&b).is_err());
    assert!(a.strict_mul(&b).is_err());
    assert!(a.strict_div(&b).is_err());
}

#[test]
fn test_add_broadcasts_like_candle() {
    // .add() now broadcasts (matching candle), not strict.
    // [2,1] + [1,3] → [2,3] via NumPy broadcasting.
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );
}

#[test]
fn test_add_incompatible_shapes_still_errors() {
    // Non-broadcast-compatible shapes still error.
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.add(&b).is_err());
}

// -- Broadcast arithmetic -----------------------------------------------------

#[test]
fn test_broadcast_add_scalar() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = DynTensor::full(&[], 10.0, DType::F32, &cpu()).unwrap();
    let c = a.broadcast_add(&b).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_broadcast_mul_column() {
    // [2, 3] * [2, 1] -> [2, 3]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = DynTensor::new(&[2.0, 3.0], &[2, 1], &cpu()).unwrap();
    let c = a.broadcast_mul(&b).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]);
}

#[test]
fn test_broadcast_add_row() {
    // [2, 3] + [1, 3] -> [2, 3]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = DynTensor::new(&[10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let c = a.broadcast_add(&b).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_broadcast_sub_column() {
    // [2, 3] - [2, 1] -> [2, 3] (subtract per-row scalar)
    let a = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let b = DynTensor::new(&[1.0, 2.0], &[2, 1], &cpu()).unwrap();
    let c = a.broadcast_sub(&b).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![9.0, 19.0, 29.0, 38.0, 48.0, 58.0]);
}

#[test]
fn test_broadcast_div_row() {
    // [2, 3] / [1, 3] -> [2, 3]
    let a = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let b = DynTensor::new(&[2.0, 5.0, 10.0], &[1, 3], &cpu()).unwrap();
    let c = a.broadcast_div(&b).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![5.0, 4.0, 3.0, 20.0, 10.0, 6.0]);
}

// -- Scalar arithmetic --------------------------------------------------------

#[test]
fn test_add_scalar_val() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = a.add_scalar(10.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![11.0, 12.0, 13.0]);
}

#[test]
fn test_mul_scalar_val() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = a.mul_scalar(2.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_affine() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let c = a.affine(2.0, 1.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![3.0, 5.0, 7.0]);
}

// -- Unary / elementwise math -------------------------------------------------

#[test]
fn test_relu() {
    let a = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let r = a.relu().unwrap();
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid() {
    let a = t1d(&[0.0]);
    let s = a.sigmoid().unwrap();
    let v = s.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], 0.5, 1e-6));
}

#[test]
fn test_tanh() {
    let a = t1d(&[0.0]);
    let s = a.tanh().unwrap();
    let v = s.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], 0.0, 1e-6));
}

#[test]
fn test_gelu() {
    let a = t1d(&[0.0, 1.0]);
    let g = a.gelu().unwrap();
    let v = g.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], 0.0, 1e-4));
    assert!(approx_eq(v[1], 0.8412, 1e-3));
}

#[test]
fn test_silu() {
    let a = t1d(&[0.0, 1.0]);
    let s = a.silu().unwrap();
    let v = s.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], 0.0, 1e-6));
    assert!(approx_eq(v[1], 0.7311, 1e-3));
}

#[test]
fn test_exp() {
    let a = t1d(&[0.0, 1.0]);
    let e = a.exp().unwrap();
    let v = e.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], 1.0, 1e-6));
    assert!(approx_eq(v[1], std::f32::consts::E, 1e-5));
}

#[test]
fn test_log() {
    let a = t1d(&[1.0, std::f32::consts::E]);
    let l = a.log().unwrap();
    let v = l.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], 0.0, 1e-6));
    assert!(approx_eq(v[1], 1.0, 1e-5));
}

#[test]
fn test_sqrt() {
    let a = t1d(&[4.0, 9.0, 16.0]);
    let s = a.sqrt().unwrap();
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_sqr() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let s = a.sqr().unwrap();
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![4.0, 9.0, 16.0]);
}

#[test]
fn test_abs() {
    let a = t1d(&[-2.0, -1.0, 0.0, 1.0]);
    let s = a.abs().unwrap();
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![2.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_neg() {
    let a = t1d(&[1.0, -2.0, 3.0]);
    let n = a.neg().unwrap();
    assert_eq!(n.to_vec1::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn test_recip() {
    let a = t1d(&[2.0, 4.0, 5.0]);
    let r = a.recip().unwrap();
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![0.5, 0.25, 0.2]);
}

#[test]
fn test_recip_zero_returns_error() {
    // recip() on zero-containing tensors must error (matching div behavior).
    let a = t1d(&[1.0, 0.0, 3.0]);
    let err = a.recip();
    assert!(err.is_err(), "recip of zero should produce an error");
}

#[test]
fn test_scalar_div_tensor_zero_returns_error() {
    // f64 / tensor with zeros uses recip() internally — must error.
    let a = t1d(&[1.0, 0.0, 2.0]);
    let result = 5.0 / &a;
    assert!(result.is_err(), "scalar / zero-tensor should error");
}

#[test]
fn test_sin_cos() {
    let a = t1d(&[0.0]);
    assert!(approx_eq(
        a.sin().unwrap().to_vec1::<f32>().unwrap()[0],
        0.0,
        1e-6
    ));
    assert!(approx_eq(
        a.cos().unwrap().to_vec1::<f32>().unwrap()[0],
        1.0,
        1e-6
    ));
}

#[test]
fn test_elu() {
    let a = t1d(&[-1.0, 0.0, 1.0]);
    let e = a.elu(1.0).unwrap();
    let v = e.to_vec1::<f32>().unwrap();
    assert!(approx_eq(v[0], -0.6321, 1e-3));
    assert!(approx_eq(v[1], 0.0, 1e-6));
    assert!(approx_eq(v[2], 1.0, 1e-6));
}

#[test]
fn test_clamp() {
    let a = t1d(&[-5.0, 0.5, 10.0]);
    let c = a.clamp(-1.0, 1.0).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-1.0, 0.5, 1.0]);
}

// -- Reduction tests extracted to tests_reductions.rs -------------------------
#[path = "tests_reductions.rs"]
mod reduction_tests;

// -- Extended reduction tests (compensated, max/min_keepdim, var) -------------
#[path = "tests_reduce_extended.rs"]
mod reduce_extended_tests;

// -- MatMul tests extracted to tests_matmul.rs --------------------------------
#[path = "tests_matmul.rs"]
mod matmul_tests;

// -- Softmax tests extracted to tests_softmax.rs --------------------------
#[path = "tests_softmax.rs"]
mod softmax_tests;

// -- Operator overloads + edge cases extracted to tests_overloads.rs -------
#[path = "tests_overloads.rs"]
mod overload_tests;

// -- Math ops (maximum, minimum) extracted to tests_math.rs ----------------
#[path = "tests_math.rs"]
mod math_tests;

// -- BF16/F16 promote-compute-demote regression tests ----------------------
#[path = "tests_bf16_f16.rs"]
mod bf16_f16_tests;

// -- BF16/F16 extended tests (shape ops, constructors, KV cache, conv, etc.)
#[path = "tests_bf16_f16_ext.rs"]
mod bf16_f16_ext_tests;

// -- add_assign (in-place addition) -------------------------------------------

#[test]
fn test_add_assign_basic() {
    let mut a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    a.add_assign(&b).unwrap();
    assert_eq!(a.to_vec1::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_add_assign_2d() {
    let mut a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    a.add_assign(&b).unwrap();
    let vals = a.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_add_assign_shape_mismatch() {
    let mut a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 2.0]);
    let result = a.add_assign(&b);
    assert!(result.is_err());
}

#[test]
fn test_add_assign_fallback_when_shared() {
    // When the tensor has shared storage (refcount > 1), add_assign falls
    // back to allocating add. Verify it still produces correct results.
    let a = t1d(&[1.0, 2.0, 3.0]);
    let mut a_clone = a.clone(); // refcount on storage is now 2
    let b = t1d(&[10.0, 20.0, 30.0]);
    a_clone.add_assign(&b).unwrap();
    // a_clone should have the updated values
    assert_eq!(a_clone.to_vec1::<f32>().unwrap(), vec![11.0, 22.0, 33.0]);
    // original a should be unchanged
    assert_eq!(a.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_add_assign_multiple_accumulations() {
    // Simulates gradient accumulation: repeated add_assign into the same tensor.
    let mut acc = DynTensor::zeros(&[3], DType::F32, &cpu()).unwrap();
    let grad1 = t1d(&[1.0, 2.0, 3.0]);
    let grad2 = t1d(&[4.0, 5.0, 6.0]);
    let grad3 = t1d(&[7.0, 8.0, 9.0]);
    acc.add_assign(&grad1).unwrap();
    acc.add_assign(&grad2).unwrap();
    acc.add_assign(&grad3).unwrap();
    assert_eq!(acc.to_vec1::<f32>().unwrap(), vec![12.0, 15.0, 18.0]);
}
