#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for shape operations on non-f32 dtypes (U32, I64, U8).
//! Verifies fix for #1254: narrow, transpose, permute, contiguous, flip.

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;
use crate::DType;

// -- narrow -------------------------------------------------------------------

#[test]
fn test_narrow_u32() {
    let data = vec![10u32, 20, 30, 40, 50];
    let t = DynTensor::from_vec_u32(data, &[5], &cpu()).unwrap();
    let sliced = t.narrow(0, 1, 3).unwrap();
    assert_eq!(sliced.dims(), &[3]);
    assert_eq!(sliced.dtype(), DType::U32);
    let vals = sliced.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[20, 30, 40]);
}

#[test]
fn test_narrow_i64() {
    let data = vec![100i64, 200, 300, 400];
    let t = DynTensor::from_vec_i64(data, &[2, 2], &cpu()).unwrap();
    let sliced = t.narrow(1, 0, 1).unwrap();
    assert_eq!(sliced.dims(), &[2, 1]);
    assert_eq!(sliced.dtype(), DType::I64);
    let vals = sliced.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[100, 300]);
}

#[test]
fn test_narrow_u32_2d() {
    let data = vec![1u32, 2, 3, 4, 5, 6];
    let t = DynTensor::from_vec_u32(data, &[2, 3], &cpu()).unwrap();
    let sliced = t.narrow(0, 1, 1).unwrap();
    assert_eq!(sliced.dims(), &[1, 3]);
    let vals = sliced.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[4, 5, 6]);
}

// -- transpose ----------------------------------------------------------------

#[test]
fn test_transpose_u32() {
    let data = vec![1u32, 2, 3, 4, 5, 6];
    let t = DynTensor::from_vec_u32(data, &[2, 3], &cpu()).unwrap();
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    assert_eq!(tr.dtype(), DType::U32);
    let vals = tr.as_cpu_u32().unwrap();
    // Original [[1,2,3],[4,5,6]] → transposed [[1,4],[2,5],[3,6]]
    assert_eq!(vals.as_slice().unwrap(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_transpose_i64() {
    let data = vec![10i64, 20, 30, 40];
    let t = DynTensor::from_vec_i64(data, &[2, 2], &cpu()).unwrap();
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(tr.dims(), &[2, 2]);
    assert_eq!(tr.dtype(), DType::I64);
    let vals = tr.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10, 30, 20, 40]);
}

#[test]
fn test_t_u32() {
    let data = vec![1u32, 2, 3, 4, 5, 6];
    let t = DynTensor::from_vec_u32(data, &[2, 3], &cpu()).unwrap();
    let tr = t.t().unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    assert_eq!(tr.dtype(), DType::U32);
}

// -- permute ------------------------------------------------------------------

#[test]
fn test_permute_u32() {
    let data: Vec<u32> = (0..24).collect();
    let t = DynTensor::from_vec_u32(data, &[2, 3, 4], &cpu()).unwrap();
    let p = t.permute([2, 0, 1]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
    assert_eq!(p.dtype(), DType::U32);
}

#[test]
fn test_permute_i64() {
    let data: Vec<i64> = (0..12).map(|x| x * 10).collect();
    let t = DynTensor::from_vec_i64(data, &[3, 4], &cpu()).unwrap();
    let p = t.permute([1, 0]).unwrap();
    assert_eq!(p.dims(), &[4, 3]);
    assert_eq!(p.dtype(), DType::I64);
}

// -- contiguous ---------------------------------------------------------------

#[test]
fn test_contiguous_u32() {
    let data = vec![1u32, 2, 3, 4, 5, 6];
    let t = DynTensor::from_vec_u32(data, &[2, 3], &cpu()).unwrap();
    // Transpose makes it non-contiguous internally, contiguous should fix it.
    let tr = t.transpose(0, 1).unwrap();
    let c = tr.contiguous().unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(c.dtype(), DType::U32);
    let vals = c.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_contiguous_i64() {
    let data = vec![10i64, 20, 30, 40];
    let t = DynTensor::from_vec_i64(data, &[2, 2], &cpu()).unwrap();
    let c = t.contiguous().unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.dtype(), DType::I64);
}

// -- flip ---------------------------------------------------------------------

#[test]
fn test_flip_u32() {
    let data = vec![10u32, 20, 30, 40, 50];
    let t = DynTensor::from_vec_u32(data, &[5], &cpu()).unwrap();
    let flipped = t.flip(0).unwrap();
    assert_eq!(flipped.dims(), &[5]);
    assert_eq!(flipped.dtype(), DType::U32);
    let vals = flipped.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[50, 40, 30, 20, 10]);
}

#[test]
fn test_flip_i64() {
    let data = vec![1i64, 2, 3, 4];
    let t = DynTensor::from_vec_i64(data, &[2, 2], &cpu()).unwrap();
    let flipped = t.flip(1).unwrap();
    assert_eq!(flipped.dims(), &[2, 2]);
    assert_eq!(flipped.dtype(), DType::I64);
    let vals = flipped.as_cpu_i64().unwrap();
    // [[1,2],[3,4]] flip dim 1 → [[2,1],[4,3]]
    assert_eq!(vals.as_slice().unwrap(), &[2, 1, 4, 3]);
}

// -- slice_set ----------------------------------------------------------------

#[test]
fn test_slice_set_u32() {
    let data = vec![0u32; 6];
    let dst = DynTensor::from_vec_u32(data, &[6], &cpu()).unwrap();
    let src = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let result = dst.slice_set(0, 2, &src).unwrap();
    assert_eq!(result.dims(), &[6]);
    assert_eq!(result.dtype(), DType::U32);
    let vals = result.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0, 0, 10, 20, 30, 0]);
}

#[test]
fn test_slice_set_i64() {
    let data = vec![0i64; 4];
    let dst = DynTensor::from_vec_i64(data, &[2, 2], &cpu()).unwrap();
    let src = DynTensor::from_vec_i64(vec![99, 88], &[1, 2], &cpu()).unwrap();
    let result = dst.slice_set(0, 1, &src).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_eq!(result.dtype(), DType::I64);
    let vals = result.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[0, 0, 99, 88]);
}

#[test]
fn test_slice_set_u8() {
    let data = vec![0u8; 5];
    let dst = DynTensor::from_vec_u8(data, &[5], &cpu()).unwrap();
    let src = DynTensor::from_vec_u8(vec![255, 128], &[2], &cpu()).unwrap();
    let result = dst.slice_set(0, 0, &src).unwrap();
    assert_eq!(result.dims(), &[5]);
    assert_eq!(result.dtype(), DType::U8);
    let vals = result.as_cpu_u8().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[255, 128, 0, 0, 0]);
}

#[test]
fn test_slice_set_dtype_mismatch() {
    let dst = DynTensor::from_vec_u32(vec![0, 0, 0], &[3], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0f32, 2.0], &[2], &cpu()).unwrap();
    let result = dst.slice_set(0, 0, &src);
    assert!(result.is_err(), "slice_set should reject dtype mismatch");
}

#[test]
fn test_slice_set_f32_regression() {
    let dst = DynTensor::from_vec(vec![0.0f32; 4], &[4], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let result = dst.slice_set(0, 1, &src).unwrap();
    assert_eq!(result.dims(), &[4]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 0.0]);
}

// -- reshape (I64 was missing) -------------------------------------------------

#[test]
fn test_reshape_i64() {
    let data = vec![1i64, 2, 3, 4, 5, 6];
    let t = DynTensor::from_vec_i64(data, &[6], &cpu()).unwrap();
    let r = t.reshape([2, 3]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.dtype(), DType::I64);
    let vals = r.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_reshape_u32() {
    let data = vec![10u32, 20, 30, 40];
    let t = DynTensor::from_vec_u32(data, &[4], &cpu()).unwrap();
    let r = t.reshape([2, 2]).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(r.dtype(), DType::U32);
}

// -- f32 regression (existing behavior should still work) ----------------------

#[test]
fn test_narrow_f32_regression() {
    let t = DynTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &cpu()).unwrap();
    let sliced = t.narrow(0, 2, 2).unwrap();
    assert_eq!(sliced.dims(), &[2]);
    let vals = sliced.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 4.0]);
}

#[test]
fn test_transpose_f32_regression() {
    let t = DynTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    let vals = tr.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// -- split --------------------------------------------------------------------

#[test]
fn test_split_basic() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5], &cpu()).unwrap();
    let parts = t.split([2, 3], 0).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].dims(), &[2]);
    assert_eq!(parts[1].dims(), &[3]);
    let v0 = parts[0].to_flat_vec::<f32>().unwrap();
    let v1 = parts[1].to_flat_vec::<f32>().unwrap();
    assert_eq!(v0, vec![1.0, 2.0]);
    assert_eq!(v1, vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_split_2d_dim1() {
    let t = DynTensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        &cpu(),
    )
    .unwrap();
    let parts = t.split([1, 2, 1], 1).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].dims(), &[3, 1]);
    assert_eq!(parts[1].dims(), &[3, 2]);
    assert_eq!(parts[2].dims(), &[3, 1]);
}

#[test]
fn test_split_size_mismatch() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let result = t.split([1, 1], 0);
    assert!(
        result.is_err(),
        "split should reject sizes that don't sum to dim size"
    );
}

#[test]
fn test_split_single_piece() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let parts = t.split([3], 0).unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].dims(), &[3]);
    let vals = parts[0].to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

// Cat/stack and FloatStorage dtype tests extracted to shape_dtype_tests_cat.rs
