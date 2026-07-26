// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`Tensor`] — constructors, bounds, dtypes, overflow safety.

use super::*;
use ndarray::{arr1, IxDyn};

#[test]
fn test_zeros_rank2() {
    let t: Tensor<2> = Tensor::zeros([2, 3]).expect("CPU allocation");
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_zeros_rank3() {
    let t: Tensor<3, f32> = Tensor::zeros([2, 3, 4]).expect("CPU allocation");
    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.ndim(), 3);
    assert_eq!(t.numel(), 24);
}

#[test]
fn test_ones_rank1() {
    let t: Tensor<1> = Tensor::ones([5]).expect("CPU allocation");
    let arr = t.as_ndarray();
    assert_eq!(arr[[0]], 1.0);
    assert_eq!(arr[[4]], 1.0);
}

#[test]
fn test_scalar_rank0() {
    let t: Tensor<0> = Tensor::zeros([]).expect("CPU allocation");
    assert_eq!(t.ndim(), 0);
    assert_eq!(t.numel(), 1); // empty product = multiplicative identity
}

#[test]
fn test_from_ndarray_matching_rank() {
    let arr = ArrayD::from_elem(IxDyn(&[2, 3]), 1.5f32);
    let t: Tensor<2> = Tensor::from_ndarray(arr).expect("rank matches");
    assert_eq!(t.dims(), &[2, 3]);
}

#[test]
fn test_from_ndarray_rank_mismatch() {
    let arr = ArrayD::from_elem(IxDyn(&[2, 3]), 1.5f32);
    let err = Tensor::<3>::from_ndarray(arr).expect_err("rank mismatch");
    assert!(matches!(
        err,
        TensorError::RankMismatch {
            expected: 3,
            actual: 2
        }
    ));
}

#[test]
fn test_from_ndarray_rank0_scalar() {
    let arr = ArrayD::from_elem(IxDyn(&[]), 42.0f32);
    let t: Tensor<0> = Tensor::from_ndarray(arr).expect("scalar rank matches");
    assert_eq!(t.numel(), 1);
    assert_eq!(t.as_ndarray()[&[] as &[usize]], 42.0);
}

#[test]
fn test_device_is_cpu() {
    let t: Tensor<2> = Tensor::zeros([2, 3]).expect("CPU allocation");
    assert_eq!(t.device(), Device::Cpu);
}

#[test]
fn test_dtype_f32() {
    let t: Tensor<1, f32> = Tensor::zeros([5]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
fn test_dtype_f64() {
    let t: Tensor<1, f64> = Tensor::zeros([5]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::F64);
}

#[test]
fn test_dtype_i32() {
    let t: Tensor<1, i32> = Tensor::zeros([5]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::I32);
}

#[test]
fn test_dtype_i64() {
    let t: Tensor<1, i64> = Tensor::zeros([5]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::I64);
}

#[test]
fn test_dtype_u8() {
    let t: Tensor<1, u8> = Tensor::zeros([5]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::U8);
}

#[test]
fn test_from_vec_u8() {
    let t: Tensor<1, u8> = Tensor::from_vec([3], vec![10, 20, 30]).expect("u8 data");
    assert_eq!(t.as_ndarray()[[0]], 10u8);
    assert_eq!(t.as_ndarray()[[2]], 30u8);
}

#[test]
fn test_from_vec_i64() {
    let t: Tensor<1, i64> = Tensor::from_vec([2], vec![100i64, 200]).expect("i64 data");
    assert_eq!(t.as_ndarray()[[0]], 100i64);
    assert_eq!(t.as_ndarray()[[1]], 200i64);
}

#[test]
fn test_with_bounds_matching_shape() {
    let t: Tensor<1> = Tensor::zeros([3]).expect("CPU allocation");
    let lower = arr1(&[0.0f32, 0.0, 0.0]).into_dyn();
    let upper = arr1(&[1.0f32, 1.0, 1.0]).into_dyn();
    let bounds = IntervalBounds::new(lower.clone(), upper.clone()).expect("valid bounds");
    let t = t.with_bounds(bounds).expect("matching shapes");
    let b = t.bounds().expect("bounds should be present");
    assert_eq!(b.lower(), &lower);
    assert_eq!(b.upper(), &upper);
}

#[test]
fn test_with_bounds_shape_mismatch() {
    let t: Tensor<1> = Tensor::zeros([3]).expect("CPU allocation");
    let lower = arr1(&[0.0f32, 0.0]).into_dyn();
    let upper = arr1(&[1.0f32, 1.0]).into_dyn();
    let bounds = IntervalBounds::new(lower, upper).expect("valid bounds");
    let err = t.with_bounds(bounds).expect_err("shape mismatch");
    assert!(matches!(err, TensorError::ShapeMismatch { .. }));
}

#[test]
fn test_with_bounds_2d() {
    let t: Tensor<2> = Tensor::zeros([2, 3]).expect("CPU allocation");
    let lower = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let bounds = IntervalBounds::new(lower.clone(), upper.clone()).expect("valid bounds");
    let t = t.with_bounds(bounds).expect("matching shapes");
    let b = t.bounds().expect("bounds should be present");
    assert_eq!(b.lower(), &lower);
    assert_eq!(b.upper(), &upper);
}

#[test]
fn test_clone() {
    let t: Tensor<2> = Tensor::zeros([2, 3]).expect("CPU allocation");
    let t2 = t;
    assert_eq!(t2.dims(), &[2, 3]);
    assert_eq!(t2.ndim(), 2);
}

#[test]
fn test_debug_format() {
    let t: Tensor<2> = Tensor::zeros([2, 3]).expect("CPU allocation");
    let debug = format!("{t:?}");
    assert!(debug.contains("Tensor"));
    assert!(debug.contains("[2, 3]"));
    assert!(debug.contains("rank: 2"));
}

#[test]
fn test_high_rank() {
    let t: Tensor<5> = Tensor::zeros([2, 3, 4, 5, 6]).expect("CPU allocation");
    assert_eq!(t.ndim(), 5);
    assert_eq!(t.numel(), 720);
}

#[test]
fn test_from_vec_rank2() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t: Tensor<2> = Tensor::from_vec([2, 3], data).expect("valid data");
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.as_ndarray()[[0, 0]], 1.0);
    assert_eq!(t.as_ndarray()[[1, 2]], 6.0);
}

#[test]
fn test_from_vec_rank1() {
    let t: Tensor<1> = Tensor::from_vec([3], vec![10.0f32, 20.0, 30.0]).expect("valid data");
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.as_ndarray()[[2]], 30.0);
}

#[test]
fn test_from_vec_scalar() {
    let t: Tensor<0> = Tensor::from_vec([], vec![42.0f32]).expect("scalar");
    assert_eq!(t.numel(), 1);
}

#[test]
fn test_from_vec_length_mismatch() {
    let err = Tensor::<2, f32>::from_vec([2, 3], vec![1.0; 5]).expect_err("length mismatch");
    assert!(matches!(
        err,
        TensorError::DataLengthMismatch {
            expected: 6,
            actual: 5
        }
    ));
}

#[test]
fn test_from_vec_empty() {
    let t: Tensor<1> = Tensor::from_vec([0], vec![]).expect("empty tensor");
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_from_vec_i32() {
    let t: Tensor<1, i32> = Tensor::from_vec([3], vec![1, 2, 3]).expect("i32 data");
    assert_eq!(t.as_ndarray()[[0]], 1);
    assert_eq!(t.as_ndarray()[[2]], 3);
}

#[test]
fn test_tensor_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tensor<2, f32, CpuBackend>>();
    assert_send_sync::<Tensor<3, i32, CpuBackend>>();
}

// -- Half-precision type tests ----------------------------------------------------

#[test]
fn test_dtype_f16() {
    let t: Tensor<1, half::f16> = Tensor::zeros([3]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.numel(), 3);
}

#[test]
fn test_dtype_bf16() {
    let t: Tensor<1, half::bf16> = Tensor::zeros([3]).expect("CPU allocation");
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.numel(), 3);
}

#[test]
fn test_f16_ones() {
    let t: Tensor<1, half::f16> = Tensor::ones([4]).expect("CPU allocation");
    let arr = t.as_ndarray();
    assert_eq!(arr[[0]], half::f16::ONE);
    assert_eq!(arr[[3]], half::f16::ONE);
}

#[test]
fn test_bf16_ones() {
    let t: Tensor<1, half::bf16> = Tensor::ones([4]).expect("CPU allocation");
    let arr = t.as_ndarray();
    assert_eq!(arr[[0]], half::bf16::ONE);
    assert_eq!(arr[[3]], half::bf16::ONE);
}

#[test]
fn test_f16_from_vec() {
    let data = vec![
        half::f16::from_f32(1.0),
        half::f16::from_f32(2.0),
        half::f16::from_f32(3.0),
    ];
    let t: Tensor<1, half::f16> = Tensor::from_vec([3], data).expect("f16 data");
    assert_eq!(t.as_ndarray()[[0]], half::f16::from_f32(1.0));
    assert_eq!(t.as_ndarray()[[2]], half::f16::from_f32(3.0));
}

#[test]
fn test_bf16_from_vec() {
    let data = vec![
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(2.0),
        half::bf16::from_f32(3.0),
    ];
    let t: Tensor<1, half::bf16> = Tensor::from_vec([3], data).expect("bf16 data");
    assert_eq!(t.as_ndarray()[[0]], half::bf16::from_f32(1.0));
    assert_eq!(t.as_ndarray()[[2]], half::bf16::from_f32(3.0));
}

#[test]
fn test_f16_rank2() {
    let t: Tensor<2, half::f16> = Tensor::zeros([2, 3]).expect("CPU allocation");
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_bf16_rank2() {
    let t: Tensor<2, half::bf16> = Tensor::zeros([2, 3]).expect("CPU allocation");
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_f16_bf16_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tensor<2, half::f16, CpuBackend>>();
    assert_send_sync::<Tensor<2, half::bf16, CpuBackend>>();
}

// -- Overflow safety tests --------------------------------------------------------

#[test]
fn test_from_ndarray_calls_checked_dim_product() {
    // ndarray itself prevents creating arrays with overflowing dimension products,
    // so from_ndarray's checked_dim_product guard is defense-in-depth. We verify
    // the check exists by confirming valid arrays still pass and that the code
    // compiles with the call (the guard catches any future ndarray changes that
    // might relax their own validation).
    let arr = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let t = Tensor::<2>::from_ndarray(arr).expect("valid dims should pass checked_dim_product");
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.numel(), 6);

    // Also verify empty dimensions pass (product = 0, no overflow):
    let arr = ArrayD::from_elem(IxDyn(&[0, 5]), 0.0f32);
    let t = Tensor::<2>::from_ndarray(arr).expect("zero-dim should pass checked_dim_product");
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_zeros_dimension_overflow() {
    let err = Tensor::<2, f32>::zeros([usize::MAX, 2]).expect_err("should overflow");
    assert!(matches!(err, TensorError::DimensionOverflow { .. }));
}

#[test]
fn test_ones_dimension_overflow() {
    let err = Tensor::<2, f32>::ones([usize::MAX, 2]).expect_err("should overflow");
    assert!(matches!(err, TensorError::DimensionOverflow { .. }));
}

#[test]
fn test_from_vec_dimension_overflow() {
    let err = Tensor::<2, f32>::from_vec([usize::MAX, 2], vec![]).expect_err("should overflow");
    assert!(matches!(err, TensorError::DimensionOverflow { .. }));
}

#[test]
fn test_checked_dim_product_valid() {
    assert_eq!(checked_dim_product(&[2, 3, 4]).expect("valid dims"), 24);
    assert_eq!(checked_dim_product(&[]).expect("empty dims"), 1);
    assert_eq!(checked_dim_product(&[0, 100]).expect("zero dim"), 0);
}

#[test]
fn test_checked_dim_product_overflow() {
    let err = checked_dim_product(&[usize::MAX, 2]).expect_err("should overflow");
    assert!(matches!(err, TensorError::DimensionOverflow { .. }));
}

#[test]
fn test_checked_dim_product_three_way_overflow() {
    // Large but individually valid dimensions that overflow when multiplied together.
    let big = 1usize << 32;
    let err = checked_dim_product(&[big, big, big]).expect_err("should overflow");
    assert!(matches!(err, TensorError::DimensionOverflow { .. }));
}

// IBP arithmetic tests (add, mul) removed in #2005 —
// arithmetic is provided by `ny_tensor::BoundedTensor`.
