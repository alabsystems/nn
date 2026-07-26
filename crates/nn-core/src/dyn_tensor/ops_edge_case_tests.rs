// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edge case tests for DynTensor operations: scalar tensors, single-element,
//! empty dimensions, broadcasting corners, high-rank, non-contiguous ops,
//! dtype promotion, clamp bounds, and double reductions.

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d, tnd};
use crate::{DType, DynTensor};

// ---------------------------------------------------------------------------
// Zero-dimensional tensors (scalar operations)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_tensor_creation_full() {
    let s = DynTensor::full(&[], 3.14, DType::F32, &cpu()).unwrap();
    assert_eq!(s.rank(), 0);
    assert_eq!(s.dims(), &[] as &[usize]);
    assert_eq!(s.numel(), 1);
    let val: f32 = s.to_scalar::<f32>().unwrap();
    assert!(approx_eq(val, 3.14, 1e-5));
}

#[test]
fn test_scalar_tensor_add() {
    let a = DynTensor::full(&[], 2.0, DType::F32, &cpu()).unwrap();
    let b = DynTensor::full(&[], 3.0, DType::F32, &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.rank(), 0);
    let val: f32 = c.to_scalar::<f32>().unwrap();
    assert!(approx_eq(val, 5.0, 1e-6));
}

#[test]
fn test_scalar_tensor_mul() {
    let a = DynTensor::full(&[], 4.0, DType::F32, &cpu()).unwrap();
    let b = DynTensor::full(&[], 2.5, DType::F32, &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.rank(), 0);
    let val: f32 = c.to_scalar::<f32>().unwrap();
    assert!(approx_eq(val, 10.0, 1e-6));
}

#[test]
fn test_scalar_tensor_unary_ops() {
    let s = DynTensor::full(&[], 1.0, DType::F32, &cpu()).unwrap();
    let exp_s = s.exp().unwrap();
    assert_eq!(exp_s.rank(), 0);
    assert!(approx_eq(
        exp_s.to_scalar::<f32>().unwrap(),
        std::f32::consts::E,
        1e-5
    ));

    let neg_s = s.neg().unwrap();
    assert!(approx_eq(neg_s.to_scalar::<f32>().unwrap(), -1.0, 1e-7));
}

#[test]
fn test_scalar_tensor_sum_all() {
    let s = DynTensor::full(&[], 42.0, DType::F32, &cpu()).unwrap();
    let sum = s.sum_all().unwrap();
    assert_eq!(sum.rank(), 0);
    assert!(approx_eq(sum.to_scalar::<f32>().unwrap(), 42.0, 1e-6));
}

#[test]
fn test_scalar_broadcast_with_vector() {
    let scalar = DynTensor::full(&[], 10.0, DType::F32, &cpu()).unwrap();
    let vec = t1d(&[1.0, 2.0, 3.0]);
    let result = scalar.add(&vec).unwrap();
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.to_vec1::<f32>().unwrap(), vec![11.0, 12.0, 13.0]);
}

#[test]
fn test_scalar_broadcast_with_matrix() {
    let scalar = DynTensor::full(&[], 5.0, DType::F32, &cpu()).unwrap();
    let mat = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let result = scalar.mul(&mat).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_eq!(
        result.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![5.0, 10.0, 15.0, 20.0]
    );
}

// ---------------------------------------------------------------------------
// Single-element tensors
// ---------------------------------------------------------------------------

#[test]
fn test_single_element_1d() {
    let a = t1d(&[7.0]);
    assert_eq!(a.rank(), 1);
    assert_eq!(a.numel(), 1);
    let b = a.exp().unwrap();
    assert_eq!(b.dims(), &[1]);
    assert!(approx_eq(
        b.to_vec1::<f32>().unwrap()[0],
        7.0_f32.exp(),
        1e-4
    ));
}

#[test]
fn test_single_element_2d() {
    let a = t2d(&[3.0], 1, 1);
    assert_eq!(a.rank(), 2);
    assert_eq!(a.numel(), 1);
    let squared = a.sqr().unwrap();
    assert!(approx_eq(squared.to_scalar::<f32>().unwrap(), 9.0, 1e-6));
}

#[test]
fn test_single_element_sum_and_mean() {
    let a = t1d(&[5.0]);
    let sum = a.sum_all().unwrap();
    assert!(approx_eq(sum.to_scalar::<f32>().unwrap(), 5.0, 1e-6));
    let mean = a.mean_all().unwrap();
    assert!(approx_eq(mean.to_scalar::<f32>().unwrap(), 5.0, 1e-6));
}

#[test]
fn test_single_element_matmul() {
    // [1,1] x [1,1] -> [1,1]
    let a = t2d(&[3.0], 1, 1);
    let b = t2d(&[4.0], 1, 1);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    assert!(approx_eq(c.to_scalar::<f32>().unwrap(), 12.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Empty dimension handling (dim=0)
// ---------------------------------------------------------------------------

#[test]
fn test_zeros_with_zero_dim() {
    let t = DynTensor::zeros(&[2, 0, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 0, 3]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_empty_tensor_add() {
    let a = DynTensor::zeros(&[0, 3], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[0, 3], DType::F32, &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[0, 3]);
    assert_eq!(c.numel(), 0);
}

#[test]
fn test_empty_tensor_unary() {
    let a = DynTensor::zeros(&[0], DType::F32, &cpu()).unwrap();
    let b = a.relu().unwrap();
    assert_eq!(b.dims(), &[0]);
    assert_eq!(b.numel(), 0);
}

#[test]
fn test_empty_tensor_reshape() {
    let a = DynTensor::zeros(&[0, 4], DType::F32, &cpu()).unwrap();
    let b = a.reshape([0, 2, 2]).unwrap();
    assert_eq!(b.dims(), &[0, 2, 2]);
    assert_eq!(b.numel(), 0);
}

#[test]
fn test_empty_tensor_mean_all_returns_error() {
    let a = DynTensor::zeros(&[0], DType::F32, &cpu()).unwrap();
    assert!(a.mean_all().is_err());
}

// ---------------------------------------------------------------------------
// Broadcasting edge cases (1xN with Nx1)
// ---------------------------------------------------------------------------

#[test]
fn test_broadcast_1xn_with_nx1() {
    // [1,3] + [3,1] -> [3,3]
    let a = t2d(&[1.0, 2.0, 3.0], 1, 3);
    let b = t2d(&[10.0, 20.0, 30.0], 3, 1);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[3, 3]);
    let vals = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0]
    );
}

#[test]
fn test_broadcast_scalar_with_3d() {
    // [] + [2,1,3] -> [2,1,3]
    let scalar = DynTensor::full(&[], 100.0, DType::F32, &cpu()).unwrap();
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let result = scalar.add(&t).unwrap();
    assert_eq!(result.dims(), &[2, 1, 3]);
    assert_eq!(
        result.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    );
}

#[test]
fn test_broadcast_1d_with_2d() {
    // [3] * [2,3] -> [2,3] (right-aligned NumPy broadcast)
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]
    );
}

#[test]
fn test_broadcast_different_ranks() {
    // [4,1] + [3] -> [4,3]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 4, 1);
    let b = t1d(&[10.0, 20.0, 30.0]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[4, 3]);
    let expected = vec![
        11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0, 14.0, 24.0, 34.0,
    ];
    assert_eq!(c.flatten_all().unwrap().to_vec1::<f32>().unwrap(), expected);
}

#[test]
fn test_broadcast_incompatible_shapes_errors() {
    // [2,3] + [2,4] should fail
    let a = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[2, 4], DType::F32, &cpu()).unwrap();
    assert!(a.add(&b).is_err());
}

// ---------------------------------------------------------------------------
// Large dimension count (rank 6+)
// ---------------------------------------------------------------------------

#[test]
fn test_rank6_creation_and_ops() {
    let dims = &[2, 1, 3, 1, 2, 1];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = tnd(&data, dims);
    assert_eq!(t.rank(), 6);
    assert_eq!(t.numel(), n);

    // Unary op preserves rank
    let doubled = t.add_scalar(1.0).unwrap();
    assert_eq!(doubled.rank(), 6);
    assert_eq!(doubled.dims(), dims);
}

#[test]
fn test_rank7_broadcast_add() {
    let dims_a: Vec<usize> = vec![1, 2, 1, 3, 1, 2, 1];
    let dims_b: Vec<usize> = vec![1, 1, 1, 1, 1, 1, 4];
    let a = DynTensor::ones(dims_a.as_slice(), DType::F32, &cpu()).unwrap();
    let b = DynTensor::ones(dims_b.as_slice(), DType::F32, &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    let expected_dims: Vec<usize> = vec![1, 2, 1, 3, 1, 2, 4];
    assert_eq!(c.dims(), expected_dims.as_slice());
    // All elements should be 2.0 (1+1)
    let vals = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx_eq(v, 2.0, 1e-7)));
}

#[test]
fn test_rank8_reshape() {
    let t = DynTensor::ones(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let new_dims: Vec<usize> = vec![1, 2, 1, 3, 1, 4, 1, 1];
    let reshaped = t.reshape(&new_dims).unwrap();
    assert_eq!(reshaped.rank(), 8);
    assert_eq!(reshaped.numel(), 24);
}

#[test]
fn test_rank6_sum_over_dim() {
    let t = DynTensor::ones(&[2, 3, 4, 5, 6, 7], DType::F32, &cpu()).unwrap();
    let reduced = t.sum_keepdim(3).unwrap();
    // dim 3 (size 5) collapsed to 1
    assert_eq!(reduced.dims(), &[2, 3, 4, 1, 6, 7]);
    // Every element should be 5.0 (sum of five 1.0s)
    let vals = reduced.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx_eq(v, 5.0, 1e-6)));
}

// ---------------------------------------------------------------------------
// Non-contiguous tensor operations (after transpose/narrow)
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_then_add() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let at = a.transpose(0, 1).unwrap(); // [3,2]
    assert_eq!(at.dims(), &[3, 2]);
    let b = DynTensor::ones(&[3, 2], DType::F32, &cpu()).unwrap();
    let c = at.add(&b).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    let vals = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // After transpose: [[1,4],[2,5],[3,6]] + ones = [[2,5],[3,6],[4,7]]
    assert_eq!(vals, vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0]);
}

#[test]
fn test_narrow_then_mul() {
    let a = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let narrowed = a.narrow(0, 1, 3).unwrap(); // [20, 30, 40]
    assert_eq!(narrowed.dims(), &[3]);
    let b = t1d(&[2.0, 3.0, 4.0]);
    let c = narrowed.mul(&b).unwrap();
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![40.0, 90.0, 160.0]);
}

#[test]
fn test_transpose_then_matmul() {
    // A=[2,3], A^T=[3,2], A^T * A => [3,3]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let at = a.transpose(0, 1).unwrap();
    let result = at.matmul(&a).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    // A^T * A = [[1,4],[2,5],[3,6]] * [[1,2,3],[4,5,6]]
    // = [[17,22,27],[22,29,36],[27,36,45]]
    let vals = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = vec![17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0];
    for (a, b) in vals.iter().zip(expected.iter()) {
        assert!(approx_eq(*a, *b, 1e-4));
    }
}

#[test]
fn test_narrow_preserves_data_after_unary() {
    let a = t1d(&[1.0, 4.0, 9.0, 16.0]);
    let narrowed = a.narrow(0, 1, 2).unwrap(); // [4.0, 9.0]
    let sqrted = narrowed.sqrt().unwrap();
    let vals = sqrted.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.0, 1e-6));
    assert!(approx_eq(vals[1], 3.0, 1e-6));
}

#[test]
fn test_contiguous_after_transpose() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let transposed = a.transpose(0, 1).unwrap();
    let contig = transposed.contiguous().unwrap();
    assert_eq!(contig.dims(), &[3, 2]);
    assert!(contig.is_contiguous());
    let vals = contig.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// ---------------------------------------------------------------------------
// Dtype promotion in mixed operations
// ---------------------------------------------------------------------------

#[test]
fn test_f32_to_bf16_and_back() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let bf16 = a.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(bf16.dims(), &[3]);
    let back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    let vals = back.to_vec1::<f32>().unwrap();
    for (v, expected) in vals.iter().zip(&[1.0, 2.0, 3.0]) {
        assert!(approx_eq(*v, *expected, 0.05)); // bf16 has ~2 decimal digits
    }
}

#[test]
fn test_f32_to_f16_round_trip() {
    let a = t1d(&[0.5, 1.5, -2.5]);
    let f16 = a.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    let back = f16.to_dtype(DType::F32).unwrap();
    let vals = back.to_vec1::<f32>().unwrap();
    for (v, expected) in vals.iter().zip(&[0.5, 1.5, -2.5]) {
        assert!(approx_eq(*v, *expected, 0.01));
    }
}

#[test]
fn test_f32_to_u32_dtype_conversion() {
    let a = t1d(&[0.0, 1.0, 255.0]);
    let u32_t = a.to_dtype(DType::U32).unwrap();
    assert_eq!(u32_t.dtype(), DType::U32);
}

#[test]
fn test_bf16_unary_op_auto_upcast() {
    // BF16 exp should auto-upcast to F32 for precision
    let a = t1d(&[1.0, 2.0]);
    let bf16 = a.to_dtype(DType::BF16).unwrap();
    let result = bf16.exp().unwrap();
    // Result dtype depends on auto-upcast policy but should not error
    let back = result.to_dtype(DType::F32).unwrap();
    let vals = back.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 1.0_f32.exp(), 0.1));
    assert!(approx_eq(vals[1], 2.0_f32.exp(), 0.2));
}

// ---------------------------------------------------------------------------
// Clamp with matching bounds (lower == upper)
// ---------------------------------------------------------------------------

#[test]
fn test_clamp_equal_bounds() {
    let a = t1d(&[-5.0, 0.0, 5.0, 100.0]);
    let clamped = a.clamp(3.0, 3.0).unwrap();
    let vals = clamped.to_vec1::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx_eq(v, 3.0, 1e-7)));
}

#[test]
fn test_clamp_all_within_bounds() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let clamped = a.clamp(0.0, 10.0).unwrap();
    assert_eq!(clamped.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_clamp_all_below_lower() {
    let a = t1d(&[-10.0, -20.0, -30.0]);
    let clamped = a.clamp(0.0, 100.0).unwrap();
    let vals = clamped.to_vec1::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx_eq(v, 0.0, 1e-7)));
}

#[test]
fn test_clamp_all_above_upper() {
    let a = t1d(&[100.0, 200.0, 300.0]);
    let clamped = a.clamp(-10.0, 10.0).unwrap();
    let vals = clamped.to_vec1::<f32>().unwrap();
    assert!(vals.iter().all(|&v| approx_eq(v, 10.0, 1e-7)));
}

#[test]
fn test_clamp_negative_bounds() {
    let a = t1d(&[-100.0, -5.0, 0.0, 5.0, 100.0]);
    let clamped = a.clamp(-10.0, -1.0).unwrap();
    let vals = clamped.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![-10.0, -5.0, -1.0, -1.0, -1.0]);
}

#[test]
fn test_clamp_on_2d_tensor() {
    let a = t2d(&[1.0, 5.0, 10.0, 15.0], 2, 2);
    let clamped = a.clamp(3.0, 12.0).unwrap();
    assert_eq!(clamped.dims(), &[2, 2]);
    let vals = clamped.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 5.0, 10.0, 12.0]);
}

// ---------------------------------------------------------------------------
// Reduction of already-reduced tensors
// ---------------------------------------------------------------------------

#[test]
fn test_double_sum_all() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let sum1 = a.sum_all().unwrap();
    assert!(approx_eq(sum1.to_scalar::<f32>().unwrap(), 21.0, 1e-5));
    // sum_all on scalar should return scalar
    let sum2 = sum1.sum_all().unwrap();
    assert!(approx_eq(sum2.to_scalar::<f32>().unwrap(), 21.0, 1e-5));
}

#[test]
fn test_sum_keepdim_then_sum_keepdim() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Sum over dim 1: [2,3] -> [2,1]
    let r1 = a.sum_keepdim(1).unwrap();
    assert_eq!(r1.dims(), &[2, 1]);
    let vals1 = r1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals1[0], 6.0, 1e-6)); // 1+2+3
    assert!(approx_eq(vals1[1], 15.0, 1e-6)); // 4+5+6

    // Sum over dim 0: [2,1] -> [1,1]
    let r2 = r1.sum_keepdim(0).unwrap();
    assert_eq!(r2.dims(), &[1, 1]);
    assert!(approx_eq(r2.to_scalar::<f32>().unwrap(), 21.0, 1e-5));
}

#[test]
fn test_mean_keepdim_then_mean_keepdim() {
    let a = tnd(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
    // Mean over dim 0: [2,2] -> [1,2]
    let r1 = a.mean_keepdim(0).unwrap();
    assert_eq!(r1.dims(), &[1, 2]);
    // [[2,4],[6,8]], mean over dim 0 = [(2+6)/2, (4+8)/2] = [4, 6]
    let vals1 = r1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals1[0], 4.0, 1e-6));
    assert!(approx_eq(vals1[1], 6.0, 1e-6));

    // Mean over dim 1: [1,2] -> [1,1]
    let r2 = r1.mean_keepdim(1).unwrap();
    assert_eq!(r2.dims(), &[1, 1]);
    assert!(approx_eq(r2.to_scalar::<f32>().unwrap(), 5.0, 1e-5));
}

#[test]
fn test_max_then_min_all() {
    let a = t1d(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let max_val = a.max_all().unwrap();
    assert!(approx_eq(max_val.to_scalar::<f32>().unwrap(), 5.0, 1e-6));
    let min_of_max = max_val.min_all().unwrap();
    assert!(approx_eq(min_of_max.to_scalar::<f32>().unwrap(), 5.0, 1e-6));
}

#[test]
fn test_sum_all_on_already_scalar() {
    let s = DynTensor::full(&[], 7.0, DType::F32, &cpu()).unwrap();
    let sum = s.sum_all().unwrap();
    assert_eq!(sum.rank(), 0);
    assert!(approx_eq(sum.to_scalar::<f32>().unwrap(), 7.0, 1e-6));
}

#[test]
fn test_mean_all_on_single_element() {
    let s = t1d(&[42.0]);
    let mean = s.mean_all().unwrap();
    assert!(approx_eq(mean.to_scalar::<f32>().unwrap(), 42.0, 1e-6));
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_zeros_scalar() {
    let z = DynTensor::zeros(&[], DType::F32, &cpu()).unwrap();
    assert_eq!(z.rank(), 0);
    assert!(approx_eq(z.to_scalar::<f32>().unwrap(), 0.0, 1e-7));
}

#[test]
fn test_ones_scalar() {
    let o = DynTensor::ones(&[], DType::F32, &cpu()).unwrap();
    assert_eq!(o.rank(), 0);
    assert!(approx_eq(o.to_scalar::<f32>().unwrap(), 1.0, 1e-7));
}

#[test]
fn test_affine_on_scalar() {
    let s = DynTensor::full(&[], 2.0, DType::F32, &cpu()).unwrap();
    // 2.0 * 3.0 + 1.0 = 7.0
    let result = s.affine(3.0, 1.0).unwrap();
    assert_eq!(result.rank(), 0);
    assert!(approx_eq(result.to_scalar::<f32>().unwrap(), 7.0, 1e-6));
}

#[test]
fn test_div_scalar_zero_returns_error() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.div_scalar(0.0).is_err());
}

#[test]
fn test_reshape_to_and_from_scalar() {
    let a = t1d(&[5.0]);
    let scalar = a.reshape([]).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert!(approx_eq(scalar.to_scalar::<f32>().unwrap(), 5.0, 1e-6));

    // Back to 1D
    let back = scalar.reshape([1]).unwrap();
    assert_eq!(back.rank(), 1);
    assert_eq!(back.to_vec1::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn test_squeeze_unsqueeze_roundtrip() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let expanded = a.unsqueeze(1).unwrap(); // [2,1,2]
    assert_eq!(expanded.dims(), &[2, 1, 2]);
    let collapsed = expanded.squeeze(1).unwrap(); // [2,2]
    assert_eq!(collapsed.dims(), &[2, 2]);
    assert_eq!(
        collapsed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
