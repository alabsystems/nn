#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor};

// -- U32 Constructor Tests ----------------------------------------------------

#[test]
fn test_from_vec_u32_roundtrip() {
    let t = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.dtype(), DType::U32);
    let arr = t.as_cpu_u32().unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[10, 20, 30]);
}

#[test]
fn test_from_vec_u32_2d() {
    let t = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4, 5], &[2, 3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::U32);
}

#[test]
fn test_from_vec_u32_length_mismatch() {
    let r = DynTensor::from_vec_u32(vec![1, 2, 3], &[2], &cpu());
    assert!(r.is_err());
}

#[test]
fn test_arange_u32_basic() {
    let t = DynTensor::arange_u32(0, 5, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    let arr = t.as_cpu_u32().unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[0, 1, 2, 3, 4]);
}

#[test]
fn test_arange_u32_offset() {
    let t = DynTensor::arange_u32(3, 7, &cpu()).unwrap();
    let arr = t.as_cpu_u32().unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[3, 4, 5, 6]);
}

#[test]
fn test_arange_u32_empty() {
    let t = DynTensor::arange_u32(5, 5, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

// -- U8 Storage Tests ---------------------------------------------------------

#[test]
fn test_u8_roundtrip() {
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1u8, 0, 1]).unwrap();
    let t = DynTensor::from_cpu_u8(arr).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    assert_eq!(t.dims(), &[3]);
    let view = t.as_cpu_u8().unwrap();
    assert_eq!(view.as_slice().unwrap(), &[1, 0, 1]);
}

// -- index_select Tests -------------------------------------------------------

#[test]
fn test_index_select_2d_rows() {
    // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    let src = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &cpu(),
    )
    .unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = src.index_select(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_index_select_2d_cols() {
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let result = src.index_select(&ids, 1).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![3.0, 1.0, 6.0, 4.0]);
}

#[test]
fn test_index_select_repeated_indices() {
    // Upsample pattern: repeat each element
    let src = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 0, 1, 1, 2, 2], &[6], &cpu()).unwrap();
    let result = src.index_select(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[6]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
}

#[test]
fn test_index_select_reversed_flip() {
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![3, 2, 1, 0], &[4], &cpu()).unwrap();
    let result = src.index_select(&ids, 0).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_index_select_3d() {
    // [2, 3, 2] tensor, select along middle dim
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let src = DynTensor::new(&data, &[2, 3, 2], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let result = src.index_select(&ids, 1).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // batch 0: rows 2,0 → [4,5], [0,1]; batch 1: rows 2,0 → [10,11], [6,7]
    assert_eq!(v, vec![4.0, 5.0, 0.0, 1.0, 10.0, 11.0, 6.0, 7.0]);
}

#[test]
fn test_index_select_out_of_bounds() {
    let src = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 5], &[2], &cpu()).unwrap();
    let r = src.index_select(&ids, 0);
    assert!(r.is_err());
}

#[test]
fn test_index_select_wrong_ids_rank() {
    let src = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 0, 1], &[2, 2], &cpu()).unwrap();
    let r = src.index_select(&ids, 0);
    assert!(r.is_err());
}

#[test]
fn test_index_select_wrong_dtype() {
    let src = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    // F32 tensor as index — should fail
    let ids = DynTensor::new(&[0.0, 1.0], &[2], &cpu()).unwrap();
    let r = src.index_select(&ids, 0);
    assert!(r.is_err());
}

// -- gather Tests -------------------------------------------------------------

#[test]
fn test_gather_2d_sort_pattern() {
    // [[10, 20, 30], [40, 50, 60]]
    let src = DynTensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();
    // Sort indices: row 0 reversed, row 1 forward
    let ids = DynTensor::from_vec_u32(vec![2, 1, 0, 0, 1, 2], &[2, 3], &cpu()).unwrap();
    let result = src.gather(&ids, 1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![30.0, 20.0, 10.0, 40.0, 50.0, 60.0]);
}

#[test]
fn test_gather_2d_dim0() {
    // [[1, 2], [3, 4]]
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    // Gather along dim 0: [[1, 0], [0, 1]] → [[1,4],[1,4]]..no
    // output[i][j] = src[ids[i][j]][j]
    let ids = DynTensor::from_vec_u32(vec![1, 0, 0, 1], &[2, 2], &cpu()).unwrap();
    let result = src.gather(&ids, 0).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // [0,0]: src[1][0]=3, [0,1]: src[0][1]=2, [1,0]: src[0][0]=1, [1,1]: src[1][1]=4
    assert_eq!(v, vec![3.0, 2.0, 1.0, 4.0]);
}

#[test]
fn test_gather_out_of_bounds() {
    let src = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 5, 0], &[1, 3], &cpu()).unwrap();
    let r = src.gather(&ids, 1);
    assert!(r.is_err());
}

#[test]
fn test_gather_wrong_rank() {
    let src = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 1, 2, 0, 1, 2], &[2, 3], &cpu()).unwrap();
    let r = src.gather(&ids, 0);
    assert!(r.is_err());
}

// -- Comparison Op Tests ------------------------------------------------------

#[test]
fn test_ge_basic() {
    let t = DynTensor::new(&[-1.0, 0.0, 1.0, 2.0], &[4], &cpu()).unwrap();
    let mask = t.ge(0.0).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(mask.dims(), &[4]);
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[0, 1, 1, 1]);
}

#[test]
fn test_gt_basic() {
    let t = DynTensor::new(&[-1.0, 0.0, 1.0], &[3], &cpu()).unwrap();
    let mask = t.gt(0.0).unwrap();
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[0, 0, 1]);
}

#[test]
fn test_lt_basic() {
    let t = DynTensor::new(&[-1.0, 0.0, 1.0], &[3], &cpu()).unwrap();
    let mask = t.lt(0.0).unwrap();
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[1, 0, 0]);
}

#[test]
fn test_le_basic() {
    let t = DynTensor::new(&[-1.0, 0.0, 1.0], &[3], &cpu()).unwrap();
    let mask = t.le(0.0).unwrap();
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[1, 1, 0]);
}

// -- eq/ne Tests --------------------------------------------------------------

#[test]
fn test_eq_basic() {
    let t = DynTensor::new(&[-1.0, 0.0, 1.0, 0.0], &[4], &cpu()).unwrap();
    let mask = t.eq(0.0).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(mask.dims(), &[4]);
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[0, 1, 0, 1]);
}

#[test]
fn test_ne_basic() {
    let t = DynTensor::new(&[-1.0, 0.0, 1.0, 0.0], &[4], &cpu()).unwrap();
    let mask = t.ne(0.0).unwrap();
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[1, 0, 1, 0]);
}

#[test]
fn test_eq_ne_complement() {
    // eq and ne should be exact complements.
    let t = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &cpu()).unwrap();
    let eq_mask = t.eq(0.0).unwrap().as_cpu_u8().unwrap().to_owned();
    let ne_mask = t.ne(0.0).unwrap().as_cpu_u8().unwrap().to_owned();
    for (e, n) in eq_mask.iter().zip(ne_mask.iter()) {
        assert_eq!(*e + *n, 1, "eq + ne must equal 1 for every element");
    }
}

#[test]
fn test_eq_nan_returns_zero() {
    // IEEE 754: NaN == x is always false.
    let t = DynTensor::new(&[f32::NAN, 0.0, 1.0], &[3], &cpu()).unwrap();
    let mask = t.eq(0.0).unwrap();
    let v = mask.as_cpu_u8().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[0, 1, 0]);
}

// -- where_cond Tests ---------------------------------------------------------

#[test]
fn test_where_cond_leaky_relu_pattern() {
    // leaky_relu: mask = x.ge(0.0); mask.where_cond(&x, &(x * 0.01))
    let x = DynTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], &cpu()).unwrap();
    let neg = DynTensor::new(&[-0.02, -0.01, 0.0, 0.01, 0.02], &[5], &cpu()).unwrap();
    let mask = x.ge(0.0).unwrap();
    let result = mask.where_cond(&x, &neg).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![-0.02, -0.01, 0.0, 1.0, 2.0]);
}

#[test]
fn test_where_cond_neg_inf_fill() {
    // Sampling pattern: fill masked positions with -inf
    let mask_data =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1u8, 0, 1, 0]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();
    let logits = DynTensor::new(&[0.5, 0.3, 0.8, 0.1], &[4], &cpu()).unwrap();
    let neg_inf = DynTensor::full(&[4], f64::NEG_INFINITY, DType::F32, &cpu()).unwrap();
    let result = mask.where_cond(&neg_inf, &logits).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert!(v[0].is_infinite() && v[0] < 0.0);
    assert_eq!(v[1], 0.3);
    assert!(v[2].is_infinite() && v[2] < 0.0);
    assert_eq!(v[3], 0.1);
}

#[test]
fn test_where_cond_shape_mismatch() {
    let mask_data = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1u8, 0, 1]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_data).unwrap();
    let a = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::new(&[3.0, 4.0, 5.0], &[3], &cpu()).unwrap();
    let r = mask.where_cond(&a, &b);
    assert!(r.is_err());
}

#[test]
fn test_where_cond_f32_mask() {
    // F32 masks with 0.0/1.0 values are accepted (#1323).
    let mask = DynTensor::new(&[1.0, 0.0, 1.0], &[3], &cpu()).unwrap();
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[3], &cpu()).unwrap();
    let r = mask.where_cond(&a, &b).unwrap();
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 5.0, 3.0]);
}

#[test]
fn test_where_cond_non_bool_mask() {
    // U32 mask is rejected — only U8 and F32 are accepted.
    let mask = DynTensor::from_vec_u32(vec![1, 0, 1], &[3], &cpu()).unwrap();
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[3], &cpu()).unwrap();
    let r = mask.where_cond(&a, &b);
    assert!(r.is_err());
}

// -- expand Tests -------------------------------------------------------------

#[test]
fn test_expand_basic() {
    // [1, 3] → [4, 3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let result = t.expand([4, 3]).unwrap();
    assert_eq!(result.dims(), &[4, 3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        v,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_expand_repeat_kv_pattern() {
    // GQA pattern: [1, 2, 1, 3, 4] → unsqueeze + expand + reshape
    // Simplified: [1, 1, 3] → [2, 4, 3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let result = t.expand([2, 4, 3]).unwrap();
    assert_eq!(result.dims(), &[2, 4, 3]);
    assert_eq!(result.numel(), 24);
}

#[test]
fn test_expand_no_change() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let result = t.expand([2, 2]).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_expand_rank_mismatch() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let r = t.expand([2, 3]);
    assert!(r.is_err());
}

#[test]
fn test_expand_non_one_dim_mismatch() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let r = t.expand([5]);
    assert!(r.is_err());
}

// I64, U32, U8 comparison dtype tests extracted to tests_compare_dtype.rs
#[path = "tests_compare_dtype.rs"]
mod compare_dtype;

// NaN/Inf edge-case, scatter_add, and to_dtype tests are in
// selection_tests_extended.rs (split for 500-line limit).
#[path = "tests_extended.rs"]
mod extended;
