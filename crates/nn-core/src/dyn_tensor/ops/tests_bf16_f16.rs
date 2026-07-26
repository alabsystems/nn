#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Regression tests for bf16/f16 promote-compute-demote arithmetic.
//!
//! Verifies that binary ops, reductions, matmul, and unary math correctly:
//! 1. Promote bf16/f16 inputs to f32 for computation
//! 2. Produce results with the correct output dtype (matching lhs for binary ops,
//!    matching input for reductions and unary ops)
//! 3. Preserve numerical accuracy within half-precision tolerance
//!
//! These tests exercise the #1646 D3 promote-compute-demote pattern and guard
//! against regressions in the FloatStorage-based native half-precision storage.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

// -- Helpers ------------------------------------------------------------------

/// Create a bf16 tensor from f32 values.
fn bf16_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    let arr = ArrayD::from_shape_vec(
        IxDyn(dims),
        data.iter().map(|&v| bf16::from_f32(v)).collect(),
    )
    .unwrap();
    DynTensor::from_cpu_bf16(arr).unwrap()
}

/// Create an f16 tensor from f32 values.
fn f16_tensor(data: &[f32], dims: &[usize]) -> DynTensor {
    let arr = ArrayD::from_shape_vec(
        IxDyn(dims),
        data.iter().map(|&v| f16::from_f32(v)).collect(),
    )
    .unwrap();
    DynTensor::from_cpu_f16(arr).unwrap()
}

/// Assert f32 values are approximately equal within tolerance.
fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// -- BF16 binary ops ----------------------------------------------------------

#[test]
fn test_bf16_add_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let b = bf16_tensor(&[4.0, 5.0, 6.0], &[3]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dtype(), DType::BF16, "add result should be BF16");
    assert_eq!(c.dims(), &[3]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 5.0, 0.1));
    assert!(approx(vals[1], 7.0, 0.1));
    assert!(approx(vals[2], 9.0, 0.1));
}

#[test]
fn test_bf16_sub_preserves_dtype() {
    let a = bf16_tensor(&[5.0, 3.0, 1.0], &[3]);
    let b = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let c = a.sub(&b).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 4.0, 0.1));
    assert!(approx(vals[1], 1.0, 0.1));
    assert!(approx(vals[2], -2.0, 0.1));
}

#[test]
fn test_bf16_mul_preserves_dtype() {
    let a = bf16_tensor(&[2.0, 3.0, 4.0], &[3]);
    let b = bf16_tensor(&[5.0, 6.0, 7.0], &[3]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 10.0, 0.1));
    assert!(approx(vals[1], 18.0, 0.1));
    assert!(approx(vals[2], 28.0, 0.1));
}

#[test]
fn test_bf16_div_preserves_dtype() {
    let a = bf16_tensor(&[10.0, 9.0, 8.0], &[3]);
    let b = bf16_tensor(&[2.0, 3.0, 4.0], &[3]);
    let c = a.div(&b).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 5.0, 0.1));
    assert!(approx(vals[1], 3.0, 0.1));
    assert!(approx(vals[2], 2.0, 0.1));
}

// -- F16 binary ops -----------------------------------------------------------

#[test]
fn test_f16_add_preserves_dtype() {
    let a = f16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let b = f16_tensor(&[4.0, 5.0, 6.0], &[3]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dtype(), DType::F16, "add result should be F16");
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 5.0, 0.01));
    assert!(approx(vals[1], 7.0, 0.01));
    assert!(approx(vals[2], 9.0, 0.01));
}

#[test]
fn test_f16_mul_preserves_dtype() {
    let a = f16_tensor(&[2.0, 3.0, 4.0], &[3]);
    let b = f16_tensor(&[5.0, 6.0, 7.0], &[3]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dtype(), DType::F16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 10.0, 0.01));
    assert!(approx(vals[1], 18.0, 0.01));
    assert!(approx(vals[2], 28.0, 0.01));
}

// -- Broadcast binary ops with bf16 ------------------------------------------

#[test]
fn test_bf16_broadcast_add() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = bf16_tensor(&[10.0, 20.0, 30.0], &[3]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[2, 3]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 11.0, 0.1));
    assert!(approx(vals[3], 14.0, 0.1));
}

// -- Mixed dtype: bf16 lhs + f32 rhs → bf16 result ---------------------------
// (lhs dtype determines output dtype)

#[test]
fn test_mixed_bf16_lhs_f32_rhs_returns_bf16() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[3], &cpu()).unwrap();
    assert_eq!(b.dtype(), DType::F32);
    let c = a.add(&b).unwrap();
    // Result dtype follows lhs (bf16).
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 5.0, 0.1));
}

#[test]
fn test_mixed_f32_lhs_bf16_rhs_returns_f32() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let b = bf16_tensor(&[4.0, 5.0, 6.0], &[3]);
    let c = a.add(&b).unwrap();
    // Result dtype follows lhs (f32).
    assert_eq!(c.dtype(), DType::F32);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 5.0, 0.01));
}

// -- BF16 reductions ----------------------------------------------------------

#[test]
fn test_bf16_sum_keepdim_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = a.sum_keepdim(1).unwrap();
    assert_eq!(s.dtype(), DType::BF16, "sum_keepdim should preserve BF16");
    assert_eq!(s.dims(), &[2, 1]);
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 6.0, 0.1));
    assert!(approx(vals[1], 15.0, 0.1));
}

#[test]
fn test_bf16_mean_keepdim_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let m = a.mean_keepdim(1).unwrap();
    assert_eq!(m.dtype(), DType::BF16);
    let vals = m.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 2.0, 0.1));
    assert!(approx(vals[1], 5.0, 0.1));
}

#[test]
fn test_bf16_max_keepdim_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let m = a.max_keepdim(1).unwrap();
    assert_eq!(m.dtype(), DType::BF16);
    let vals = m.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 5.0, 0.1));
    assert!(approx(vals[1], 6.0, 0.1));
}

#[test]
fn test_bf16_sum_all_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let s = a.sum_all().unwrap();
    assert_eq!(s.dtype(), DType::BF16);
    let val = s.to_flat_vec::<f32>().unwrap()[0];
    assert!(approx(val, 6.0, 0.1));
}

#[test]
fn test_bf16_mean_all_preserves_dtype() {
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let m = a.mean_all().unwrap();
    assert_eq!(m.dtype(), DType::BF16);
    let val = m.to_flat_vec::<f32>().unwrap()[0];
    assert!(approx(val, 2.5, 0.1));
}

// -- F16 reductions -----------------------------------------------------------

#[test]
fn test_f16_sum_keepdim_preserves_dtype() {
    let a = f16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = a.sum_keepdim(1).unwrap();
    assert_eq!(s.dtype(), DType::F16, "sum_keepdim should preserve F16");
    let vals = s.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 6.0, 0.01));
    assert!(approx(vals[1], 15.0, 0.01));
}

#[test]
fn test_f16_max_all_preserves_dtype() {
    let a = f16_tensor(&[3.0, 1.0, 7.0, 2.0], &[4]);
    let m = a.max_all().unwrap();
    assert_eq!(m.dtype(), DType::F16);
    let val = m.to_flat_vec::<f32>().unwrap()[0];
    assert!(approx(val, 7.0, 0.01));
}

// -- BF16 matmul --------------------------------------------------------------

#[test]
fn test_bf16_matmul_preserves_dtype() {
    // [2, 3] x [3, 2] -> [2, 2]
    let a = bf16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = bf16_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dtype(), DType::BF16, "matmul should preserve BF16");
    assert_eq!(c.dims(), &[2, 2]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // [1,2,3] . [7,9,11] = 7+18+33 = 58
    assert!(approx(vals[0], 58.0, 0.5));
    // [1,2,3] . [8,10,12] = 8+20+36 = 64
    assert!(approx(vals[1], 64.0, 0.5));
    // [4,5,6] . [7,9,11] = 28+45+66 = 139
    assert!(approx(vals[2], 139.0, 1.0));
}

#[test]
fn test_f16_matmul_preserves_dtype() {
    let a = f16_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
    let b = f16_tensor(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dtype(), DType::F16, "matmul should preserve F16");
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 3.0, 0.01));
    assert!(approx(vals[1], 4.0, 0.01));
    assert!(approx(vals[2], 5.0, 0.01));
    assert!(approx(vals[3], 6.0, 0.01));
}

// -- BF16 unary math ops ------------------------------------------------------

#[test]
fn test_bf16_neg_preserves_dtype() {
    let a = bf16_tensor(&[1.0, -2.0, 3.0], &[3]);
    let c = a.neg().unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], -1.0, 0.1));
    assert!(approx(vals[1], 2.0, 0.1));
    assert!(approx(vals[2], -3.0, 0.1));
}

#[test]
fn test_bf16_exp_preserves_dtype() {
    let a = bf16_tensor(&[0.0, 1.0], &[2]);
    let c = a.exp().unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 1.0, 0.1));
    assert!(approx(vals[1], std::f32::consts::E, 0.1));
}

#[test]
fn test_bf16_sqrt_preserves_dtype() {
    let a = bf16_tensor(&[4.0, 9.0, 16.0], &[3]);
    let c = a.sqrt().unwrap();
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 2.0, 0.1));
    assert!(approx(vals[1], 3.0, 0.1));
    assert!(approx(vals[2], 4.0, 0.1));
}

#[test]
fn test_f16_tanh_preserves_dtype() {
    let a = f16_tensor(&[0.0, 1.0, -1.0], &[3]);
    let c = a.tanh().unwrap();
    assert_eq!(c.dtype(), DType::F16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 0.0, 0.01));
    assert!(approx(vals[1], 0.7616, 0.01));
    assert!(approx(vals[2], -0.7616, 0.01));
}

// -- BF16 f64-scalar operator overloads (#1798) -------------------------------

#[test]
fn test_f64_mul_bf16_tensor_preserves_dtype() {
    let t = bf16_tensor(&[2.0, 4.0, 6.0], &[3]);
    let result = (3.0_f64 * &t).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 6.0, 0.1));
    assert!(approx(vals[1], 12.0, 0.1));
    assert!(approx(vals[2], 18.0, 0.1));
}

#[test]
fn test_f64_div_bf16_tensor_preserves_dtype() {
    let t = bf16_tensor(&[2.0, 4.0, 8.0], &[3]);
    let result = (1.0_f64 / &t).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 0.5, 0.1));
    assert!(approx(vals[1], 0.25, 0.1));
    assert!(approx(vals[2], 0.125, 0.1));
}

#[test]
fn test_f64_sub_bf16_tensor_preserves_dtype() {
    let t = bf16_tensor(&[2.0, 4.0, 6.0], &[3]);
    let result = (10.0_f64 - &t).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 8.0, 0.1));
    assert!(approx(vals[1], 6.0, 0.1));
    assert!(approx(vals[2], 4.0, 0.1));
}

#[test]
fn test_bf16_div_scalar_preserves_dtype() {
    let t = bf16_tensor(&[6.0, 8.0, 10.0], &[3]);
    let result = t.div_scalar(2.0).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 3.0, 0.1));
    assert!(approx(vals[1], 4.0, 0.1));
    assert!(approx(vals[2], 5.0, 0.1));
}

// -- F16 f64-scalar operator overloads (#1798) --------------------------------

#[test]
fn test_f64_mul_f16_tensor_preserves_dtype() {
    let t = f16_tensor(&[2.0, 4.0, 6.0], &[3]);
    let result = (3.0_f64 * &t).unwrap();
    assert_eq!(result.dtype(), DType::F16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 6.0, 0.01));
    assert!(approx(vals[1], 12.0, 0.01));
    assert!(approx(vals[2], 18.0, 0.01));
}

#[test]
fn test_f64_div_f16_tensor_preserves_dtype() {
    let t = f16_tensor(&[2.0, 4.0, 8.0], &[3]);
    let result = (1.0_f64 / &t).unwrap();
    assert_eq!(result.dtype(), DType::F16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 0.5, 0.01));
    assert!(approx(vals[1], 0.25, 0.01));
    assert!(approx(vals[2], 0.125, 0.01));
}

#[test]
fn test_f64_sub_f16_tensor_preserves_dtype() {
    let t = f16_tensor(&[2.0, 4.0, 6.0], &[3]);
    let result = (10.0_f64 - &t).unwrap();
    assert_eq!(result.dtype(), DType::F16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 8.0, 0.01));
    assert!(approx(vals[1], 6.0, 0.01));
    assert!(approx(vals[2], 4.0, 0.01));
}

#[test]
fn test_f16_div_scalar_preserves_dtype() {
    let t = f16_tensor(&[6.0, 8.0, 10.0], &[3]);
    let result = t.div_scalar(2.0).unwrap();
    assert_eq!(result.dtype(), DType::F16);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 3.0, 0.01));
    assert!(approx(vals[1], 4.0, 0.01));
    assert!(approx(vals[2], 5.0, 0.01));
}

// -- Compound scalar-tensor patterns (dvoice #1796) ------------------------

#[test]
fn test_bf16_add_then_div_f64_preserves_dtype() {
    // Pattern: (residual + shortcut) / (2.0f64).sqrt() — dvoice stage1.rs:161
    let a = bf16_tensor(&[2.0, 4.0, 6.0], &[3]);
    let b = bf16_tensor(&[8.0, 6.0, 4.0], &[3]);
    let sum = (&a + &b).unwrap();
    assert_eq!(sum.dtype(), DType::BF16, "bf16 + bf16 = bf16");
    let result = (&sum / 2.0_f64.sqrt()).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "bf16 / f64 = bf16");
    let vals = result.to_flat_vec::<f32>().unwrap();
    // (2+8)/sqrt(2) ≈ 7.07, (4+6)/sqrt(2) ≈ 7.07, (6+4)/sqrt(2) ≈ 7.07
    for v in &vals {
        assert!(approx(*v, 7.07, 0.1), "expected ~7.07, got {v}");
    }
}

#[test]
fn test_f64_add_bf16_tensor_preserves_dtype() {
    let t = bf16_tensor(&[1.0, 2.0, 3.0], &[3]);
    let result = (10.0_f64 + &t).unwrap();
    assert_eq!(result.dtype(), DType::BF16, "f64 + bf16 = bf16");
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 11.0, 0.1));
    assert!(approx(vals[1], 12.0, 0.1));
    assert!(approx(vals[2], 13.0, 0.1));
}

// -- BF16/F16 auto-upcast for precision-sensitive ops (#2013) -----------------

#[test]
fn test_bf16_exp_large_input_does_not_overflow() {
    // BF16 exp(20.0) would overflow to Inf in native BF16 (7-bit mantissa).
    // Auto-upcast to F32 computes exp(20) ≈ 4.85e8, then casts back.
    let a = bf16_tensor(&[20.0, 30.0, -5.0], &[3]);
    let c = a.exp().unwrap();
    assert_eq!(c.dtype(), DType::BF16, "auto-upcast preserves dtype");
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(vals[0].is_finite(), "exp(20) finite via auto-upcast");
    assert!(vals[1].is_finite(), "exp(30) finite via auto-upcast");
    assert!(approx(vals[2], (-5.0_f32).exp(), 0.01));
}

#[test]
fn test_bf16_precision_sensitive_ops_preserve_dtype() {
    let a = bf16_tensor(&[0.5, 1.0, 2.0], &[3]);
    for (name, result) in [
        ("sin", a.sin()),
        ("cos", a.cos()),
        ("log", a.log()),
        ("sqrt", a.sqrt()),
        ("sigmoid", a.sigmoid()),
        ("silu", a.silu()),
        ("tanh", a.tanh()),
        ("gelu", a.gelu()),
        ("gelu_erf", a.gelu_erf()),
    ] {
        let t = result.unwrap();
        assert_eq!(t.dtype(), DType::BF16, "{name} preserves BF16 dtype");
        let vals = t.to_flat_vec::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "{name} all finite");
    }
}

#[test]
fn test_bf16_sigmoid_extreme_values_finite() {
    // sigmoid(-100)/sigmoid(100) contain exp() that overflows in native BF16.
    let a = bf16_tensor(&[0.0, -100.0, 100.0], &[3]);
    let c = a.sigmoid().unwrap();
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(approx(vals[0], 0.5, 0.01));
    assert!(vals[1].is_finite() && vals[1] < 0.01, "sigmoid(-100) ≈ 0");
    assert!(vals[2].is_finite() && vals[2] > 0.99, "sigmoid(100) ≈ 1");
}

#[test]
fn test_f16_exp_auto_upcast() {
    let a = f16_tensor(&[10.0, -3.0], &[2]);
    let c = a.exp().unwrap();
    assert_eq!(c.dtype(), DType::F16, "auto-upcast preserves f16 dtype");
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(vals[0].is_finite(), "exp(10) finite via auto-upcast");
    assert!(approx(vals[1], (-3.0_f32).exp(), 0.01));
}

// Shape ops, constructors, KV cache, conv, cat, cumsum, scatter_add tests
// extracted to tests_bf16_f16_ext.rs (#1669) to keep this file under 400 lines.
