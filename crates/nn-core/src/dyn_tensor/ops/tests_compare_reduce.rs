// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for DynTensor comparison and reduction operations.
//!
//! Covers gap areas not exercised by existing test modules:
//! - NaN/Inf comparison behavior (IEEE 754 compliance)
//! - Comparison-to-reduction pipelines
//! - Cumulative sum (cumsum, cumsum_kahan)
//! - Variance/statistical edge cases
//! - Multi-dim sequential reductions
//! - Reduction dtype preservation (BF16/F16)
//! - Higher-rank (4D) reduction patterns
//! - Argmax/argmin tie-breaking in multi-dim tensors

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d, tnd};
use crate::{DType, DynTensor};

// ============================================================================
// NaN comparison behavior (IEEE 754: NaN != NaN, NaN comparisons yield false)
// ============================================================================

#[test]
fn test_nan_eq_nan_is_false() {
    let a = DynTensor::from_vec(vec![f32::NAN, 1.0, f32::NAN], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![f32::NAN, 1.0, 2.0], &[3], &cpu()).unwrap();
    let mask = a.eq_tensor(&b).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    let vals = mask.as_cpu_u8().unwrap();
    // NaN == NaN is false per IEEE 754
    assert_eq!(vals[[0]], 0, "NaN == NaN should be false");
    assert_eq!(vals[[1]], 1, "1.0 == 1.0 should be true");
    assert_eq!(vals[[2]], 0, "NaN == 2.0 should be false");
}

#[test]
fn test_nan_ne_nan_is_true() {
    let a = DynTensor::from_vec(vec![f32::NAN, 1.0, f32::NAN], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![f32::NAN, 1.0, 2.0], &[3], &cpu()).unwrap();
    let mask = a.ne_tensor(&b).unwrap();
    let vals = mask.as_cpu_u8().unwrap();
    assert_eq!(vals[[0]], 1, "NaN != NaN should be true");
    assert_eq!(vals[[1]], 0, "1.0 != 1.0 should be false");
    assert_eq!(vals[[2]], 1, "NaN != 2.0 should be true");
}

#[test]
fn test_nan_lt_always_false() {
    let nan_tensor = DynTensor::from_vec(vec![f32::NAN, f32::NAN, f32::NAN], &[3], &cpu()).unwrap();
    let normal = t1d(&[0.0, 1.0, -1.0]);
    // NaN < x is always false
    let lt = nan_tensor.lt_tensor(&normal).unwrap();
    assert_eq!(
        lt.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0],
        "NaN < any should be false"
    );
    // x < NaN is also always false
    let lt2 = normal.lt_tensor(&nan_tensor).unwrap();
    assert_eq!(
        lt2.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0],
        "any < NaN should be false"
    );
}

#[test]
fn test_nan_gt_always_false() {
    let nan_tensor = DynTensor::from_vec(vec![f32::NAN, f32::NAN], &[2], &cpu()).unwrap();
    let normal = t1d(&[0.0, 100.0]);
    let gt = nan_tensor.gt_tensor(&normal).unwrap();
    assert_eq!(
        gt.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0],
        "NaN > any should be false"
    );
}

#[test]
fn test_nan_le_ge_always_false() {
    let nan_tensor = DynTensor::from_vec(vec![f32::NAN, f32::NAN], &[2], &cpu()).unwrap();
    let normal = t1d(&[0.0, 1.0]);
    let le = nan_tensor.le_tensor(&normal).unwrap();
    assert_eq!(
        le.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0],
        "NaN <= any should be false"
    );
    let ge = nan_tensor.ge_tensor(&normal).unwrap();
    assert_eq!(
        ge.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0],
        "NaN >= any should be false"
    );
}

#[test]
fn test_nan_scalar_comparison() {
    let a = DynTensor::from_vec(vec![f32::NAN, 1.0, 2.0], &[3], &cpu()).unwrap();
    let eq = a.eq(1.0).unwrap();
    let vals = eq.as_cpu_u8().unwrap();
    assert_eq!(vals[[0]], 0, "NaN eq scalar should be false");
    assert_eq!(vals[[1]], 1, "1.0 eq 1.0 should be true");
    assert_eq!(vals[[2]], 0, "2.0 eq 1.0 should be false");
}

// ============================================================================
// Inf comparison behavior
// ============================================================================

#[test]
fn test_inf_eq_inf() {
    let a = t1d(&[f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, 1.0]);
    let b = t1d(&[
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
    ]);
    let eq = a.eq_tensor(&b).unwrap();
    assert_eq!(
        eq.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 1, 0, 0],
        "Inf == Inf, -Inf == -Inf, Inf != -Inf"
    );
}

#[test]
fn test_inf_lt_ordering() {
    let a = t1d(&[f32::NEG_INFINITY, 0.0, f32::INFINITY]);
    let b = t1d(&[0.0, f32::INFINITY, f32::INFINITY]);
    let lt = a.lt_tensor(&b).unwrap();
    // -Inf < 0 = true, 0 < Inf = true, Inf < Inf = false
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 0]);
}

#[test]
fn test_inf_gt_ordering() {
    let a = t1d(&[f32::INFINITY, 0.0, f32::NEG_INFINITY]);
    let b = t1d(&[0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
    let gt = a.gt_tensor(&b).unwrap();
    // Inf > 0 = true, 0 > -Inf = true, -Inf > -Inf = false
    assert_eq!(gt.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 0]);
}

#[test]
fn test_inf_scalar_compare() {
    let a = t1d(&[f32::INFINITY, 0.0, f32::NEG_INFINITY]);
    let gt = a.gt(0.0).unwrap();
    assert_eq!(
        gt.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[1, 0, 0],
        "Inf > 0 = true, 0 > 0 = false, -Inf > 0 = false"
    );
}

// ============================================================================
// Comparison-to-reduction pipelines (realistic ML patterns)
// ============================================================================

#[test]
fn test_compare_then_sum_counts_matches() {
    // Count how many elements equal 3.0
    let x = t1d(&[1.0, 3.0, 3.0, 2.0, 3.0]);
    let mask = x.eq(3.0).unwrap();
    // Convert U8 mask to F32 for sum
    let mask_f32 = mask.to_dtype(DType::F32).unwrap();
    let count = mask_f32.sum_all().unwrap();
    assert_eq!(count.to_scalar::<f32>().unwrap(), 3.0);
}

#[test]
fn test_compare_then_sum_per_row() {
    // Count matches per row: [[1,2,3],[3,2,1]] with threshold > 2
    let x = t2d(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], 2, 3);
    let mask = x.gt(2.0).unwrap();
    let mask_f32 = mask.to_dtype(DType::F32).unwrap();
    let per_row = mask_f32.sum(1).unwrap();
    assert_eq!(per_row.dims(), &[2]);
    let vals = per_row.to_vec1::<f32>().unwrap();
    // Row 0: only 3.0 > 2 => 1 match; Row 1: only 3.0 > 2 => 1 match
    assert_eq!(vals, vec![1.0, 1.0]);
}

#[test]
fn test_compare_mask_mean_pattern() {
    // Masked mean: average of elements > 0
    let x = t1d(&[-1.0, 2.0, -3.0, 4.0, 5.0]);
    let mask = x.gt(0.0).unwrap();
    let mask_f32 = mask.to_dtype(DType::F32).unwrap();
    // Sum of positives: 2 + 4 + 5 = 11, count = 3, mean = 11/3
    let masked = x.mul(&mask_f32).unwrap();
    let total = masked.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let count = mask_f32.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let masked_mean = total / count;
    assert!(approx_eq(masked_mean, 11.0 / 3.0, 1e-5));
}

// ============================================================================
// Cumulative sum (cumsum)
// ============================================================================

#[test]
fn test_cumsum_1d_basic() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let cs = x.cumsum(0).unwrap();
    assert_eq!(cs.dims(), &[4]);
    assert_eq!(cs.to_vec1::<f32>().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn test_cumsum_1d_single_element() {
    let x = t1d(&[42.0]);
    let cs = x.cumsum(0).unwrap();
    assert_eq!(cs.to_vec1::<f32>().unwrap(), vec![42.0]);
}

#[test]
fn test_cumsum_1d_negative_values() {
    let x = t1d(&[1.0, -2.0, 3.0, -4.0]);
    let cs = x.cumsum(0).unwrap();
    assert_eq!(cs.to_vec1::<f32>().unwrap(), vec![1.0, -1.0, 2.0, -2.0]);
}

#[test]
fn test_cumsum_1d_all_zeros() {
    let x = t1d(&[0.0, 0.0, 0.0]);
    let cs = x.cumsum(0).unwrap();
    assert_eq!(cs.to_vec1::<f32>().unwrap(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_cumsum_2d_dim0() {
    // [[1,2],[3,4]] cumsum dim=0 -> [[1,2],[4,6]]
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let cs = x.cumsum(0).unwrap();
    assert_eq!(cs.dims(), &[2, 2]);
    assert_eq!(cs.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_cumsum_2d_dim1() {
    // [[1,2,3],[4,5,6]] cumsum dim=1 -> [[1,3,6],[4,9,15]]
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let cs = x.cumsum(1).unwrap();
    assert_eq!(cs.dims(), &[2, 3]);
    assert_eq!(
        cs.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );
}

#[test]
fn test_cumsum_3d_last_dim() {
    // [[[1,2],[3,4]]] cumsum dim=2
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let cs = x.cumsum(2).unwrap();
    assert_eq!(cs.dims(), &[1, 2, 2]);
    // [1, 1+2=3], [3, 3+4=7]
    assert_eq!(cs.to_flat_vec::<f32>().unwrap(), vec![1.0, 3.0, 3.0, 7.0]);
}

#[test]
fn test_cumsum_out_of_range_dim_errors() {
    let x = t1d(&[1.0, 2.0]);
    assert!(x.cumsum(1).is_err(), "dim 1 out of range for rank-1 tensor");
}

// ============================================================================
// Kahan-compensated cumulative sum
// ============================================================================

#[test]
fn test_cumsum_kahan_basic() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let cs = x.cumsum_kahan(0).unwrap();
    assert_eq!(cs.dims(), &[4]);
    let vals = cs.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 1.0, 1e-6));
    assert!(approx_eq(vals[1], 3.0, 1e-6));
    assert!(approx_eq(vals[2], 6.0, 1e-6));
    assert!(approx_eq(vals[3], 10.0, 1e-6));
}

#[test]
fn test_cumsum_kahan_precision() {
    // Kahan should handle many small values added to a large value better
    let mut data = vec![1e6_f32];
    data.extend(std::iter::repeat_n(1e-4, 1000));
    let x = DynTensor::from_vec(data, &[1001], &cpu()).unwrap();
    let cs = x.cumsum_kahan(0).unwrap();
    let vals = cs.to_vec1::<f32>().unwrap();
    // Last value should be close to 1e6 + 1000 * 1e-4 = 1000000.1
    let last = vals[1000];
    assert!(
        (last - 1_000_000.1).abs() < 0.01,
        "Kahan cumsum last value should be close to 1000000.1, got {last}"
    );
}

#[test]
fn test_cumsum_kahan_2d_dim1() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let cs = x.cumsum_kahan(1).unwrap();
    assert_eq!(cs.dims(), &[2, 3]);
    let vals = cs.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 1.0, 1e-6));
    assert!(approx_eq(vals[1], 3.0, 1e-6));
    assert!(approx_eq(vals[2], 6.0, 1e-6));
    assert!(approx_eq(vals[3], 4.0, 1e-6));
    assert!(approx_eq(vals[4], 9.0, 1e-6));
    assert!(approx_eq(vals[5], 15.0, 1e-6));
}

// ============================================================================
// Variance edge cases
// ============================================================================

#[test]
fn test_var_constant_tensor_is_zero() {
    // Variance of a constant is 0
    let x = t1d(&[5.0, 5.0, 5.0, 5.0]);
    let v = x.var(0).unwrap();
    assert!(approx_eq(v.to_scalar::<f32>().unwrap(), 0.0, 1e-6));
}

#[test]
fn test_var_known_values() {
    // var([1, 3, 5, 7]) = mean([(1-4)^2, (3-4)^2, (5-4)^2, (7-4)^2])
    //                    = mean([9, 1, 1, 9]) = 5.0
    let x = t1d(&[1.0, 3.0, 5.0, 7.0]);
    let v = x.var(0).unwrap();
    assert!(
        approx_eq(v.to_scalar::<f32>().unwrap(), 5.0, 1e-4),
        "var([1,3,5,7]) should be 5.0, got {}",
        v.to_scalar::<f32>().unwrap()
    );
}

#[test]
fn test_var_keepdim_3d_middle_axis() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] shape [2,2,2]
    // var along axis=1 keepdim -> shape [2,1,2]
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let v = x.var_keepdim(1).unwrap();
    assert_eq!(v.dims(), &[2, 1, 2]);
    let vals = v.to_flat_vec::<f32>().unwrap();
    // Batch 0: mean along axis=1 = [2,3], var = mean([(1-2)^2,(3-2)^2], [(2-3)^2,(4-3)^2]) = [1,1]
    assert!(approx_eq(vals[0], 1.0, 1e-4));
    assert!(approx_eq(vals[1], 1.0, 1e-4));
    // Batch 1: mean along axis=1 = [6,7], var = mean([(5-6)^2,(7-6)^2], [(6-7)^2,(8-7)^2]) = [1,1]
    assert!(approx_eq(vals[2], 1.0, 1e-4));
    assert!(approx_eq(vals[3], 1.0, 1e-4));
}

#[test]
fn test_var_single_element_is_zero() {
    let x = t1d(&[42.0]);
    let v = x.var(0).unwrap();
    assert!(approx_eq(v.to_scalar::<f32>().unwrap(), 0.0, 1e-6));
}

#[test]
fn test_var_two_elements() {
    // var([a, b]) = mean([(a-m)^2, (b-m)^2]) where m = (a+b)/2
    // For [0, 10]: m=5, var = mean([25, 25]) = 25
    let x = t1d(&[0.0, 10.0]);
    let v = x.var(0).unwrap();
    assert!(approx_eq(v.to_scalar::<f32>().unwrap(), 25.0, 1e-4));
}

// ============================================================================
// Reduction dtype preservation (BF16, F16)
// ============================================================================

#[test]
fn test_sum_preserves_bf16_dtype() {
    let x = t1d(&[1.0, 2.0, 3.0]).to_dtype(DType::BF16).unwrap();
    let s = x.sum(0).unwrap();
    assert_eq!(s.dtype(), DType::BF16);
    let s_f32 = s.to_dtype(DType::F32).unwrap();
    assert!(approx_eq(s_f32.to_scalar::<f32>().unwrap(), 6.0, 0.1));
}

#[test]
fn test_mean_preserves_f16_dtype() {
    let x = t1d(&[2.0, 4.0, 6.0]).to_dtype(DType::F16).unwrap();
    let m = x.mean(0).unwrap();
    assert_eq!(m.dtype(), DType::F16);
    let m_f32 = m.to_dtype(DType::F32).unwrap();
    assert!(approx_eq(m_f32.to_scalar::<f32>().unwrap(), 4.0, 0.1));
}

#[test]
fn test_max_preserves_bf16_dtype() {
    let x = t1d(&[1.0, 5.0, 3.0]).to_dtype(DType::BF16).unwrap();
    let m = x.max(0).unwrap();
    assert_eq!(m.dtype(), DType::BF16);
    let m_f32 = m.to_dtype(DType::F32).unwrap();
    assert!(approx_eq(m_f32.to_scalar::<f32>().unwrap(), 5.0, 0.1));
}

#[test]
fn test_min_preserves_f16_dtype() {
    let x = t1d(&[3.0, 1.0, 5.0]).to_dtype(DType::F16).unwrap();
    let m = x.min(0).unwrap();
    assert_eq!(m.dtype(), DType::F16);
    let m_f32 = m.to_dtype(DType::F32).unwrap();
    assert!(approx_eq(m_f32.to_scalar::<f32>().unwrap(), 1.0, 0.1));
}

#[test]
fn test_sum_all_preserves_bf16_dtype() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2)
        .to_dtype(DType::BF16)
        .unwrap();
    let s = x.sum_all().unwrap();
    assert_eq!(s.dtype(), DType::BF16);
    let s_f32 = s.to_dtype(DType::F32).unwrap();
    assert!(approx_eq(s_f32.to_scalar::<f32>().unwrap(), 10.0, 0.5));
}

#[test]
fn test_var_preserves_bf16_dtype() {
    let x = t1d(&[1.0, 3.0, 5.0]).to_dtype(DType::BF16).unwrap();
    let v = x.var(0).unwrap();
    assert_eq!(v.dtype(), DType::BF16);
}

// ============================================================================
// Higher-rank (4D) reductions
// ============================================================================

#[test]
fn test_sum_4d_axis3() {
    // [1,2,2,3] tensor, sum along last axis
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = tnd(&data, &[1, 2, 2, 3]);
    let s = x.sum(3).unwrap();
    assert_eq!(s.dims(), &[1, 2, 2]);
    let vals = s.to_flat_vec::<f32>().unwrap();
    // sum([1,2,3])=6, sum([4,5,6])=15, sum([7,8,9])=24, sum([10,11,12])=33
    assert_eq!(vals, vec![6.0, 15.0, 24.0, 33.0]);
}

#[test]
fn test_mean_4d_axis0() {
    // [2,1,1,3] tensor, mean along axis 0
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 1, 3]);
    let m = x.mean(0).unwrap();
    assert_eq!(m.dims(), &[1, 1, 3]);
    let vals = m.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.5, 1e-6));
    assert!(approx_eq(vals[1], 3.5, 1e-6));
    assert!(approx_eq(vals[2], 4.5, 1e-6));
}

#[test]
fn test_max_4d_axis2() {
    // [1,1,3,2] tensor, max along axis 2
    let x = tnd(&[1.0, 6.0, 3.0, 4.0, 5.0, 2.0], &[1, 1, 3, 2]);
    let m = x.max(2).unwrap();
    assert_eq!(m.dims(), &[1, 1, 2]);
    let vals = m.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 6.0]);
}

// ============================================================================
// Multi-dim sequential reductions
// ============================================================================

#[test]
fn test_sum_two_sequential_dims() {
    // [[1,2,3],[4,5,6]] -> sum(dim=1) -> [6, 15] -> sum(dim=0) -> [21]
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s1 = x.sum(1).unwrap();
    let s2 = s1.sum(0).unwrap();
    assert_eq!(s2.to_scalar::<f32>().unwrap(), 21.0);
}

#[test]
fn test_mean_then_max() {
    // Mean per row then max across rows
    let x = t2d(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3);
    let means = x.mean(1).unwrap(); // [2.0, 20.0]
    let max_mean = means.max(0).unwrap();
    assert!(approx_eq(max_mean.to_scalar::<f32>().unwrap(), 20.0, 1e-6));
}

#[test]
fn test_sum_3d_all_dims_sequential() {
    // Reduce all dims one at a time, should equal sum_all
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let x = tnd(&data, &[2, 3, 4]);
    let total_direct = x.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let total_seq = x.sum(2).unwrap().sum(1).unwrap().sum(0).unwrap();
    assert!(approx_eq(
        total_seq.to_scalar::<f32>().unwrap(),
        total_direct,
        1e-4
    ));
}

// ============================================================================
// Argmax/argmin tie-breaking and multi-dim
// ============================================================================

#[test]
fn test_argmax_3d_axis2() {
    // [[[1,3,2],[4,0,5]], [[6,8,7],[9,10,1]]]
    let x = tnd(
        &[1.0, 3.0, 2.0, 4.0, 0.0, 5.0, 6.0, 8.0, 7.0, 9.0, 10.0, 1.0],
        &[2, 2, 3],
    );
    let idx = x.argmax(2).unwrap();
    assert_eq!(idx.dims(), &[2, 2]);
    let vals = idx.to_flat_vec::<u32>().unwrap();
    // [1,3,2] -> argmax=1, [4,0,5] -> argmax=2
    // [6,8,7] -> argmax=1, [9,10,1] -> argmax=1
    assert_eq!(vals, vec![1, 2, 1, 1]);
}

#[test]
fn test_argmin_3d_axis1() {
    // [[[5,6],[1,2]], [[3,4],[7,8]]] shape [2,2,2]
    let x = tnd(&[5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0], &[2, 2, 2]);
    let idx = x.argmin(1).unwrap();
    assert_eq!(idx.dims(), &[2, 2]);
    let vals = idx.to_flat_vec::<u32>().unwrap();
    // Batch 0: min along axis=1 of [[5,6],[1,2]] -> indices [1,1]
    // Batch 1: min along axis=1 of [[3,4],[7,8]] -> indices [0,0]
    assert_eq!(vals, vec![1, 1, 0, 0]);
}

#[test]
fn test_argmax_ties_returns_first() {
    // Multiple equal max values; should return first occurrence
    let x = t2d(&[5.0, 5.0, 5.0, 1.0, 3.0, 3.0], 2, 3);
    let idx = x.argmax(1).unwrap();
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 1]);
}

#[test]
fn test_argmin_ties_returns_first() {
    // Multiple equal min values; should return first occurrence
    let x = t2d(&[1.0, 1.0, 1.0, 2.0, 0.0, 0.0], 2, 3);
    let idx = x.argmin(1).unwrap();
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 1]);
}

#[test]
fn test_argmax_keepdim_3d() {
    let x = tnd(&[1.0, 3.0, 2.0, 4.0, 0.0, 5.0], &[1, 2, 3]);
    let idx = x.argmax_keepdim(2).unwrap();
    assert_eq!(idx.dims(), &[1, 2, 1]);
    let vals = idx.to_flat_vec::<u32>().unwrap();
    assert_eq!(vals, vec![1, 2]);
}

// ============================================================================
// Max/min keepdim on higher ranks
// ============================================================================

#[test]
fn test_max_keepdim_3d_axis0() {
    // [[[1,2],[3,4]], [[5,6],[7,8]]] shape [2,2,2]
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let m = x.max_keepdim(0).unwrap();
    assert_eq!(m.dims(), &[1, 2, 2]);
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_min_keepdim_3d_axis2() {
    // [[[3,1],[4,2]], [[7,5],[8,6]]] shape [2,2,2]
    let x = tnd(&[3.0, 1.0, 4.0, 2.0, 7.0, 5.0, 8.0, 6.0], &[2, 2, 2]);
    let m = x.min_keepdim(2).unwrap();
    assert_eq!(m.dims(), &[2, 2, 1]);
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 5.0, 6.0]);
}

// ============================================================================
// Scalar tensor reductions
// ============================================================================

#[test]
fn test_sum_all_scalar_tensor() {
    let x = DynTensor::full(&[], 7.0, DType::F32, &cpu()).unwrap();
    let s = x.sum_all().unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), 7.0);
}

#[test]
fn test_mean_all_scalar_tensor() {
    let x = DynTensor::full(&[], 3.5, DType::F32, &cpu()).unwrap();
    let m = x.mean_all().unwrap();
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 3.5, 1e-6));
}

#[test]
fn test_max_all_scalar_tensor() {
    let x = DynTensor::full(&[], 42.0, DType::F32, &cpu()).unwrap();
    let m = x.max_all().unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_min_all_scalar_tensor() {
    let x = DynTensor::full(&[], -3.0, DType::F32, &cpu()).unwrap();
    let m = x.min_all().unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), -3.0);
}

// ============================================================================
// Broadcast comparison with higher-rank shapes
// ============================================================================

#[test]
fn test_compare_broadcast_3x1_vs_1x4() {
    let a = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![5.0, 15.0, 25.0, 35.0], &[1, 4], &cpu()).unwrap();
    let gt = a.gt_tensor(&b).unwrap();
    assert_eq!(gt.dims(), &[3, 4]);
    let vals = gt.as_cpu_u8().unwrap();
    // row 0 (10): [10>5, 10>15, 10>25, 10>35] = [1, 0, 0, 0]
    // row 1 (20): [20>5, 20>15, 20>25, 20>35] = [1, 1, 0, 0]
    // row 2 (30): [30>5, 30>15, 30>25, 30>35] = [1, 1, 1, 0]
    assert_eq!(
        vals.as_slice().unwrap(),
        &[1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    );
}

#[test]
fn test_compare_broadcast_4d() {
    // [1,1,2] vs [1,2,1] -> [1,2,2]
    let a = DynTensor::from_vec(vec![1.0, 3.0], &[1, 1, 2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![2.0, 4.0], &[1, 2, 1], &cpu()).unwrap();
    let lt = a.lt_tensor(&b).unwrap();
    assert_eq!(lt.dims(), &[1, 2, 2]);
    // expanded a: [[1,3],[1,3]], expanded b: [[2,2],[4,4]]
    // lt: [[1<2=1, 3<2=0],[1<4=1, 3<4=1]]
    assert_eq!(lt.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 0, 1, 1]);
}

// ============================================================================
// Comparison symmetry/anti-symmetry properties
// ============================================================================

#[test]
fn test_lt_gt_antisymmetry() {
    // a < b should equal b > a for all non-NaN values
    let a = t1d(&[1.0, 5.0, 3.0, 7.0, 2.0]);
    let b = t1d(&[3.0, 3.0, 3.0, 3.0, 3.0]);
    let lt_ab = a.lt_tensor(&b).unwrap();
    let gt_ba = b.gt_tensor(&a).unwrap();
    assert_eq!(
        lt_ab.as_cpu_u8().unwrap().as_slice().unwrap(),
        gt_ba.as_cpu_u8().unwrap().as_slice().unwrap(),
        "a < b should equal b > a"
    );
}

#[test]
fn test_le_ge_antisymmetry() {
    let a = t1d(&[1.0, 3.0, 5.0, 3.0]);
    let b = t1d(&[3.0, 3.0, 3.0, 1.0]);
    let le_ab = a.le_tensor(&b).unwrap();
    let ge_ba = b.ge_tensor(&a).unwrap();
    assert_eq!(
        le_ab.as_cpu_u8().unwrap().as_slice().unwrap(),
        ge_ba.as_cpu_u8().unwrap().as_slice().unwrap(),
        "a <= b should equal b >= a"
    );
}

// ============================================================================
// Large tensor reduction accuracy
// ============================================================================

#[test]
fn test_sum_all_large_tensor() {
    // Sum of [1, 1, 1, ...] (1000 elements) = 1000
    let data = vec![1.0_f32; 1000];
    let x = DynTensor::from_vec(data, &[1000], &cpu()).unwrap();
    let s = x.sum_all().unwrap();
    assert!(approx_eq(s.to_scalar::<f32>().unwrap(), 1000.0, 1e-3));
}

#[test]
fn test_mean_all_large_tensor() {
    // Mean of [0, 1, 2, ..., 999] = 499.5
    let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
    let x = DynTensor::from_vec(data, &[1000], &cpu()).unwrap();
    let m = x.mean_all().unwrap();
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 499.5, 1e-2));
}

// ============================================================================
// Reduction after comparison: sum of boolean mask
// ============================================================================

#[test]
fn test_sum_of_eq_mask_2d() {
    // Count elements equal to target per column
    let x = t2d(&[1.0, 2.0, 3.0, 1.0, 3.0, 3.0, 2.0, 2.0, 3.0], 3, 3);
    let target = DynTensor::from_vec(vec![3.0], &[1, 1], &cpu()).unwrap();
    let mask = x.eq_tensor(&target).unwrap();
    let mask_f32 = mask.to_dtype(DType::F32).unwrap();
    let per_col = mask_f32.sum(0).unwrap();
    assert_eq!(per_col.dims(), &[3]);
    // Col 0: [1==3, 1==3, 2==3] = [0,0,0] -> 0
    // Col 1: [2==3, 3==3, 2==3] = [0,1,0] -> 1
    // Col 2: [3==3, 3==3, 3==3] = [1,1,1] -> 3
    assert_eq!(per_col.to_vec1::<f32>().unwrap(), vec![0.0, 1.0, 3.0]);
}

// ============================================================================
// Comparison preserves shape
// ============================================================================

#[test]
fn test_comparison_preserves_shape_3d() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = DynTensor::full(&[2, 2, 2], 4.0, DType::F32, &cpu()).unwrap();
    let gt = a.gt_tensor(&b).unwrap();
    assert_eq!(gt.dims(), &[2, 2, 2]);
    assert_eq!(gt.dtype(), DType::U8);
    assert_eq!(
        gt.as_cpu_u8().unwrap().as_slice().unwrap(),
        &[0, 0, 0, 0, 1, 1, 1, 1]
    );
}

// ============================================================================
// Edge: empty tensor operations
// ============================================================================

#[test]
fn test_sum_all_empty_tensor() {
    let x = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    let s = x.sum_all().unwrap();
    // Sum of empty = identity element 0
    assert_eq!(s.to_scalar::<f32>().unwrap(), 0.0);
}

#[test]
fn test_mean_all_empty_tensor_errors() {
    let x = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    assert!(x.mean_all().is_err(), "mean of empty should error");
}

#[test]
fn test_max_all_empty_tensor_errors() {
    let x = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    assert!(x.max_all().is_err(), "max of empty should error");
}

#[test]
fn test_min_all_empty_tensor_errors() {
    let x = DynTensor::from_vec(Vec::<f32>::new(), &[0], &cpu()).unwrap();
    assert!(x.min_all().is_err(), "min of empty should error");
}
