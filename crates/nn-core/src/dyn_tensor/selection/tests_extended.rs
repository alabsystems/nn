#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NaN/Inf edge-case, scatter_add, and index_add tests for selection ops.
//! Extracted from dyn_tensor_selection_tests.rs to keep it under 500 lines.
//! to_dtype tests in tests_to_dtype.rs (#1227).
//! Comparison, boundary, and integer regression tests in
//! tests_compare_boundary.rs (#1402).

use crate::dyn_tensor::test_helpers::cpu;
use crate::DynTensor;

// -- NaN/Inf Edge Case Tests --------------------------------------------------

#[test]
fn test_ge_nan_input_returns_zero() {
    // IEEE 754: NaN >= x is always false
    let t = DynTensor::new(
        &[f32::NAN, 1.0, f32::NEG_INFINITY, f32::INFINITY],
        &[4],
        &cpu(),
    )
    .unwrap();
    let mask = t.ge(0.0).unwrap();
    let vals = mask.as_cpu_u8().unwrap();
    // NaN >= 0.0 → false (0), 1.0 >= 0.0 → true (1),
    // -INF >= 0.0 → false (0), INF >= 0.0 → true (1)
    assert_eq!(vals.as_slice().unwrap(), &[0, 1, 0, 1]);
}

#[test]
fn test_lt_nan_input_returns_zero() {
    // IEEE 754: NaN < x is always false
    let t = DynTensor::new(&[f32::NAN, -1.0, 0.0], &[3], &cpu()).unwrap();
    let mask = t.lt(0.0).unwrap();
    let vals = mask.as_cpu_u8().unwrap();
    // NaN < 0.0 → false (0), -1.0 < 0.0 → true (1), 0.0 < 0.0 → false (0)
    assert_eq!(vals.as_slice().unwrap(), &[0, 1, 0]);
}

#[test]
fn test_ge_nan_threshold() {
    // IEEE 754: x >= NaN is always false
    let t = DynTensor::new(&[1.0, -1.0, 0.0], &[3], &cpu()).unwrap();
    let mask = t.ge(f64::NAN).unwrap();
    let vals = mask.as_cpu_u8().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0, 0, 0]);
}

#[test]
fn test_where_cond_with_nan_values() {
    // where_cond should propagate NaN values without error
    let mask_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1u8, 0, 1]).unwrap();
    let mask = DynTensor::from_cpu_u8(mask_arr).unwrap();
    let on_true = DynTensor::new(&[f32::NAN, 2.0, 3.0], &[3], &cpu()).unwrap();
    let on_false = DynTensor::new(&[10.0, 20.0, f32::NAN], &[3], &cpu()).unwrap();
    let result = mask.where_cond(&on_true, &on_false).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(vals[0].is_nan()); // on_true[0] = NaN, selected
    assert_eq!(vals[1], 20.0); // on_false[1] = 20.0, selected
    assert_eq!(vals[2], 3.0); // on_true[2] = 3.0, selected
}

#[test]
fn test_index_select_with_nan_source() {
    // index_select should propagate NaN values from source tensor
    let src = DynTensor::new(&[f32::NAN, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = src.index_select(&ids, 0).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(vals[0].is_nan()); // selected row 0 which has NaN
    assert_eq!(vals[1], 3.0); // selected row 2
}

#[test]
fn test_ge_inf_threshold() {
    let t = DynTensor::new(&[f32::INFINITY, f32::MAX, 0.0], &[3], &cpu()).unwrap();
    let mask = t.ge(f64::INFINITY).unwrap();
    let vals = mask.as_cpu_u8().unwrap();
    // INF >= INF → true, MAX >= INF → false, 0 >= INF → false
    assert_eq!(vals.as_slice().unwrap(), &[1, 0, 0]);
}

// -- scatter_add tests --------------------------------------------------------

#[test]
fn test_scatter_add_basic() {
    // self = [0, 0, 0, 0], scatter src=[10, 20] at indices [1, 3] along dim 0
    let base = DynTensor::from_vec(vec![0.0; 4], &[4], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1, 3], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![10.0, 20.0], &[2], &cpu()).unwrap();
    let out = base.scatter_add(0, &index, &src).unwrap();
    assert_eq!(
        out.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 10.0, 0.0, 20.0]
    );
}

#[test]
fn test_scatter_add_accumulate() {
    // Two sources scatter to same index → should accumulate
    let base = DynTensor::from_vec(vec![0.0; 3], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1, 1], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![5.0, 3.0], &[2], &cpu()).unwrap();
    let out = base.scatter_add(0, &index, &src).unwrap();
    assert_eq!(out.to_flat_vec::<f32>().unwrap(), vec![0.0, 8.0, 0.0]);
}

#[test]
fn test_scatter_add_2d() {
    // 2D scatter along dim=0
    let base = DynTensor::from_vec(vec![0.0; 6], &[3, 2], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 2], &[1, 2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![10.0, 20.0], &[1, 2], &cpu()).unwrap();
    let out = base.scatter_add(0, &index, &src).unwrap();
    let v = out.to_flat_vec::<f32>().unwrap();
    // [10, 0; 0, 0; 0, 20]
    assert_eq!(v, vec![10.0, 0.0, 0.0, 0.0, 0.0, 20.0]);
}

#[test]
fn test_scatter_add_out_of_bounds() {
    let base = DynTensor::from_vec(vec![0.0; 3], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![5], &[1], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert!(base.scatter_add(0, &index, &src).is_err());
}

#[test]
fn test_scatter_gather_roundtrip() {
    // gather then scatter_add should reconstruct (for unique indices)
    let data =
        DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();
    let idx = DynTensor::from_vec_u32(vec![2, 0, 1, 0], &[2, 2], &cpu()).unwrap();
    let gathered = data.gather(&idx, 1).unwrap();
    let base = DynTensor::from_vec(vec![0.0; 6], &[2, 3], &cpu()).unwrap();
    let scattered = base.scatter_add(1, &idx, &gathered).unwrap();
    let v = scattered.to_flat_vec::<f32>().unwrap();
    // Row 0: gather picked [2,0] → [30,10], scatter puts 30@2, 10@0 → [10,0,30]
    assert_eq!(v[0], 10.0);
    assert_eq!(v[2], 30.0);
}

// -- P1 proof_coverage: scatter_add missing error paths ----------------------

#[test]
fn test_scatter_add_dim_out_of_range() {
    let base = DynTensor::from_vec(vec![0.0; 3], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let err = base.scatter_add(1, &index, &src).unwrap_err();
    assert!(
        err.to_string().contains("out of range"),
        "expected dim out of range error, got: {err}"
    );
}

#[test]
fn test_scatter_add_wrong_index_dtype() {
    let base = DynTensor::from_vec(vec![0.0; 3], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let err = base.scatter_add(0, &index, &src).unwrap_err();
    assert!(
        err.to_string().contains("u32") || err.to_string().contains("type mismatch"),
        "expected dtype mismatch error, got: {err}"
    );
}

#[test]
fn test_scatter_add_index_rank_mismatch_with_src() {
    let base = DynTensor::from_vec(vec![0.0; 4], &[2, 2], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    assert!(base.scatter_add(0, &index, &src).is_err());
}

#[test]
fn test_scatter_add_index_shape_mismatch_with_src() {
    let base = DynTensor::from_vec(vec![0.0; 4], &[4], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1, 2], &[3], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(base.scatter_add(0, &index, &src).is_err());
}

#[test]
fn test_scatter_add_index_rank_mismatch_with_self() {
    let base = DynTensor::from_vec(vec![0.0; 4], &[2, 2], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(base.scatter_add(0, &index, &src).is_err());
}

// -- index_add tests ----------------------------------------------------------

#[test]
fn test_index_add_1d_basic() {
    // base = [0, 0, 0, 0], add src=[10, 20] at indices [1, 3] along dim 0
    let base = DynTensor::from_vec(vec![0.0; 4], &[4], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1, 3], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![10.0, 20.0], &[2], &cpu()).unwrap();
    let out = base.index_add(0, &index, &src).unwrap();
    assert_eq!(
        out.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 10.0, 0.0, 20.0]
    );
}

#[test]
fn test_index_add_accumulate_duplicates() {
    // Two source rows scatter to same index → should accumulate
    let base = DynTensor::from_vec(vec![1.0, 1.0, 1.0], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1, 1], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![5.0, 3.0], &[2], &cpu()).unwrap();
    let out = base.index_add(0, &index, &src).unwrap();
    assert_eq!(out.to_flat_vec::<f32>().unwrap(), vec![1.0, 9.0, 1.0]);
}

#[test]
fn test_index_add_2d_dim0() {
    // base [3, 2], source [2, 2], index [2] mapping rows
    let base = DynTensor::from_vec(vec![0.0; 6], &[3, 2], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2], &cpu()).unwrap();
    let out = base.index_add(0, &index, &src).unwrap();
    let v = out.to_flat_vec::<f32>().unwrap();
    // row 0 gets src row 0: [10, 20], row 2 gets src row 1: [30, 40]
    assert_eq!(v, vec![10.0, 20.0, 0.0, 0.0, 30.0, 40.0]);
}

#[test]
fn test_index_add_2d_dim1() {
    // base [2, 4], source [2, 2], index [2] mapping columns
    let base = DynTensor::from_vec(vec![0.0; 8], &[2, 4], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1, 3], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2], &cpu()).unwrap();
    let out = base.index_add(1, &index, &src).unwrap();
    let v = out.to_flat_vec::<f32>().unwrap();
    // row 0: col 1 = 10, col 3 = 20; row 1: col 1 = 30, col 3 = 40
    assert_eq!(v, vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0]);
}

#[test]
fn test_index_add_preserves_existing() {
    // base has existing values that should be preserved
    let base = DynTensor::from_vec(vec![100.0, 200.0, 300.0], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1], &[1], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![7.0], &[1], &cpu()).unwrap();
    let out = base.index_add(0, &index, &src).unwrap();
    assert_eq!(out.to_flat_vec::<f32>().unwrap(), vec![100.0, 207.0, 300.0]);
}

#[test]
fn test_index_add_out_of_bounds() {
    let base = DynTensor::from_vec(vec![0.0; 3], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![5], &[1], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert!(base.index_add(0, &index, &src).is_err());
}

#[test]
fn test_index_add_wrong_index_dtype() {
    let base = DynTensor::from_vec(vec![0.0; 3], &[3], &cpu()).unwrap();
    let index = DynTensor::from_vec(vec![0.0], &[1], &cpu()).unwrap(); // F32, not U32
    let src = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert!(base.index_add(0, &index, &src).is_err());
}

#[test]
fn test_index_add_wrong_index_rank() {
    let base = DynTensor::from_vec(vec![0.0; 4], &[2, 2], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1, 0, 1], &[2, 2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    assert!(base.index_add(0, &index, &src).is_err());
}

#[test]
fn test_index_add_source_rank_mismatch() {
    let base = DynTensor::from_vec(vec![0.0; 4], &[2, 2], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(base.index_add(0, &index, &src).is_err());
}

#[test]
fn test_index_add_index_length_mismatch() {
    // index length (3) != source.dims()[dim=0] (2)
    let base = DynTensor::from_vec(vec![0.0; 4], &[4], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1, 2], &[3], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(base.index_add(0, &index, &src).is_err());
}

#[test]
fn test_index_add_shape_mismatch_non_scatter_dim() {
    // base [3, 4], source [2, 3] — cols don't match (4 vs 3)
    let base = DynTensor::from_vec(vec![0.0; 12], &[3, 4], &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    assert!(base.index_add(0, &index, &src).is_err());
}

// -- where_cond broadcasting --------------------------------------------------

#[test]
fn test_where_cond_broadcast_mask_scalar() {
    // mask [1] broadcast with on_true [3] and on_false [3]
    let mask = DynTensor::from_vec_u8(vec![1], &[1], &cpu()).unwrap();
    let on_true = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[3], &cpu()).unwrap();
    let on_false = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(y.dims(), &[3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_where_cond_broadcast_2d() {
    // mask [2, 1], on_true [1, 3], on_false [2, 3] → output [2, 3]
    let mask = DynTensor::from_vec_u8(vec![1, 0], &[2, 1], &cpu()).unwrap();
    let on_true = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let on_false =
        DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let y = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // row 0: mask=1 → on_true [10,20,30]; row 1: mask=0 → on_false [4,5,6]
    assert_eq!(vals, vec![10.0, 20.0, 30.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_where_cond_broadcast_same_shape_unchanged() {
    // Same shapes: no broadcast needed, should work as before
    let mask = DynTensor::from_vec_u8(vec![1, 0, 1], &[3], &cpu()).unwrap();
    let on_true = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[3], &cpu()).unwrap();
    let on_false = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(y.dims(), &[3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![10.0, 2.0, 30.0]);
}

// -- Tensor-vs-tensor comparison, algorithm boundary, and integer comparison
// regression tests extracted to tests_compare_boundary.rs (#1402).
#[path = "tests_compare_boundary.rs"]
mod compare_boundary;

// -- to_dtype conversion tests ------------------------------------------------
// Extracted to tests_to_dtype.rs for file-size compliance (#1227).
#[path = "tests_to_dtype.rs"]
mod to_dtype_tests;
