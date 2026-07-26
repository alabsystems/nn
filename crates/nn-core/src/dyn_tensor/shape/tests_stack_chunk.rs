// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `stack`, `chunk`, `split`, and `split_uniform` operations.
//!
//! Covers: 2D→3D stacking, stacking along different dims, chunk with
//! equal/uneven splits, split_uniform with exact and remainder cases,
//! error cases, and roundtrip invariants.

use crate::dyn_tensor::test_helpers::{t1d, t2d, tnd};
use crate::DynTensor;

// ---------------------------------------------------------------------------
// stack: 2D tensors into 3D
// ---------------------------------------------------------------------------

#[test]
fn test_stack_2d_into_3d_dim0() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let s = DynTensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(s.dims(), &[2, 2, 2]);
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_stack_2d_into_3d_dim1() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let s = DynTensor::stack(&[&a, &b], 1).unwrap();
    assert_eq!(s.dims(), &[2, 2, 2]);
    // After stacking along dim=1:
    // a[0] = [1, 2], b[0] = [5, 6] -> row 0 = [[1,2],[5,6]]
    // a[1] = [3, 4], b[1] = [7, 8] -> row 1 = [[3,4],[7,8]]
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn test_stack_2d_into_3d_dim2() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let s = DynTensor::stack(&[&a, &b], 2).unwrap();
    assert_eq!(s.dims(), &[2, 2, 2]);
    // Stack at last dim: each (i,j) becomes [a[i,j], b[i,j]]
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]
    );
}

// ---------------------------------------------------------------------------
// stack: 3 tensors
// ---------------------------------------------------------------------------

#[test]
fn test_stack_three_tensors() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    let c = t1d(&[5.0, 6.0]);
    let s = DynTensor::stack(&[&a, &b, &c], 0).unwrap();
    assert_eq!(s.dims(), &[3, 2]);
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_stack_single_tensor() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let s = DynTensor::stack(&[&a], 0).unwrap();
    assert_eq!(s.dims(), &[1, 2, 2]);
    assert_eq!(s.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ---------------------------------------------------------------------------
// stack: error cases
// ---------------------------------------------------------------------------

#[test]
fn test_stack_empty_list() {
    let empty: Vec<&DynTensor> = vec![];
    let result = DynTensor::stack(&empty, 0);
    assert!(result.is_err(), "stack should reject empty tensor list");
}

#[test]
fn test_stack_shape_mismatch() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0, 5.0]);
    let result = DynTensor::stack(&[&a, &b], 0);
    assert!(result.is_err(), "stack should reject shape mismatch");
}

#[test]
fn test_stack_dim_out_of_range() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0]);
    // rank=1, new_rank=2, valid dims are 0..=1, so dim=3 should fail
    let result = DynTensor::stack(&[&a, &b], 3);
    assert!(result.is_err(), "stack should reject dim out of range");
}

// ---------------------------------------------------------------------------
// chunk: equal parts
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_equal_parts_2d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    for c in &chunks {
        assert_eq!(c.dims(), &[1, 2]);
    }
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

// ---------------------------------------------------------------------------
// chunk: remainder (uneven)
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_with_remainder_2d() {
    // [5, 3] tensor, chunk dim=0 into 3 -> chunk_size=ceil(5/3)=2
    // Produces: [2,3], [2,3], [1,3]
    let data: Vec<f32> = (1..=15).map(|i| i as f32).collect();
    let t = t2d(&data, 5, 3);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].dims(), &[2, 3]);
    assert_eq!(chunks[1].dims(), &[2, 3]);
    assert_eq!(chunks[2].dims(), &[1, 3]);
    assert_eq!(
        chunks[0].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        chunks[1].to_flat_vec::<f32>().unwrap(),
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        chunks[2].to_flat_vec::<f32>().unwrap(),
        vec![13.0, 14.0, 15.0]
    );
}

// ---------------------------------------------------------------------------
// chunk: error cases
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_zero_chunks() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.chunk(0, 0);
    assert!(result.is_err(), "chunk should reject 0 chunks");
}

#[test]
fn test_chunk_dim_out_of_range() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.chunk(2, 5);
    assert!(result.is_err(), "chunk should reject dim out of range");
}

// ---------------------------------------------------------------------------
// chunk + cat roundtrip on 2D
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_cat_roundtrip_2d() {
    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    let t = t2d(&data, 6, 4);
    let chunks = t.chunk(3, 0).unwrap();
    let refs: Vec<&DynTensor> = chunks.iter().collect();
    let reconstructed = DynTensor::cat(&refs, 0).unwrap();
    assert_eq!(reconstructed.dims(), t.dims());
    assert_eq!(
        reconstructed.to_flat_vec::<f32>().unwrap(),
        t.to_flat_vec::<f32>().unwrap()
    );
}

// ---------------------------------------------------------------------------
// split_uniform: exact division
// ---------------------------------------------------------------------------

#[test]
fn test_split_uniform_exact() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let parts = t.split_uniform(2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].dims(), &[2]);
    assert_eq!(parts[1].dims(), &[2]);
    assert_eq!(parts[2].dims(), &[2]);
    assert_eq!(parts[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(parts[2].to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_split_uniform_exact_2d() {
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let t = t2d(&data, 4, 3);
    let parts = t.split_uniform(2, 0).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].dims(), &[2, 3]);
    assert_eq!(parts[1].dims(), &[2, 3]);
    assert_eq!(
        parts[0].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        parts[1].to_flat_vec::<f32>().unwrap(),
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

// ---------------------------------------------------------------------------
// split_uniform: remainder
// ---------------------------------------------------------------------------

#[test]
fn test_split_uniform_remainder() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = t.split_uniform(2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].dims(), &[2]);
    assert_eq!(parts[1].dims(), &[2]);
    assert_eq!(parts[2].dims(), &[1]); // remainder
    assert_eq!(parts[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(parts[2].to_flat_vec::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn test_split_uniform_remainder_2d_dim1() {
    // [2, 5] tensor, split along dim=1 with size=3 -> [2,3] + [2,2]
    let data: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    let t = t2d(&data, 2, 5);
    let parts = t.split_uniform(3, 1).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].dims(), &[2, 3]);
    assert_eq!(parts[1].dims(), &[2, 2]);
    assert_eq!(
        parts[0].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        parts[1].to_flat_vec::<f32>().unwrap(),
        vec![4.0, 5.0, 9.0, 10.0]
    );
}

#[test]
fn test_split_uniform_size_larger_than_dim() {
    // split_size > dim_size: produces a single part equal to the full tensor
    let t = t1d(&[1.0, 2.0, 3.0]);
    let parts = t.split_uniform(10, 0).unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].dims(), &[3]);
    assert_eq!(parts[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// split_uniform: error cases
// ---------------------------------------------------------------------------

#[test]
fn test_split_uniform_zero_size() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.split_uniform(0, 0);
    assert!(result.is_err(), "split_uniform should reject split_size=0");
}

#[test]
fn test_split_uniform_dim_out_of_range() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.split_uniform(2, 5);
    assert!(
        result.is_err(),
        "split_uniform should reject dim out of range"
    );
}

// ---------------------------------------------------------------------------
// split_uniform + cat roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_split_uniform_cat_roundtrip() {
    let data: Vec<f32> = (1..=15).map(|i| i as f32).collect();
    let t = tnd(&data, &[3, 5]);
    let parts = t.split_uniform(2, 1).unwrap();
    let refs: Vec<&DynTensor> = parts.iter().collect();
    let reconstructed = DynTensor::cat(&refs, 1).unwrap();
    assert_eq!(reconstructed.dims(), t.dims());
    assert_eq!(
        reconstructed.to_flat_vec::<f32>().unwrap(),
        t.to_flat_vec::<f32>().unwrap()
    );
}

// ---------------------------------------------------------------------------
// stack + chunk roundtrip (inverse operations)
// ---------------------------------------------------------------------------

#[test]
fn test_stack_chunk_roundtrip() {
    // stack([a, b, c], dim=0) then narrow each piece back should recover originals
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = t2d(&[9.0, 10.0, 11.0, 12.0], 2, 2);
    let stacked = DynTensor::stack(&[&a, &b, &c], 0).unwrap();
    assert_eq!(stacked.dims(), &[3, 2, 2]);
    // Recover via narrow + squeeze
    let a_back = stacked.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
    let b_back = stacked.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();
    let c_back = stacked.narrow(0, 2, 1).unwrap().squeeze(0).unwrap();
    assert_eq!(
        a_back.to_flat_vec::<f32>().unwrap(),
        a.to_flat_vec::<f32>().unwrap()
    );
    assert_eq!(
        b_back.to_flat_vec::<f32>().unwrap(),
        b.to_flat_vec::<f32>().unwrap()
    );
    assert_eq!(
        c_back.to_flat_vec::<f32>().unwrap(),
        c.to_flat_vec::<f32>().unwrap()
    );
}

// ---------------------------------------------------------------------------
// split (list-of-sizes) error: sizes don't sum to dim size
// ---------------------------------------------------------------------------

#[test]
fn test_split_sizes_sum_mismatch() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = t.split([2, 2], 0);
    assert!(
        result.is_err(),
        "split should reject sizes that don't sum to dim size"
    );
}

// ---------------------------------------------------------------------------
// 3D operations
// ---------------------------------------------------------------------------

#[test]
fn test_chunk_3d_dim2() {
    // [2, 2, 6] tensor, chunk along dim=2 into 3
    let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 2, 6]);
    let chunks = t.chunk(3, 2).unwrap();
    assert_eq!(chunks.len(), 3);
    for c in &chunks {
        assert_eq!(c.dims(), &[2, 2, 2]);
    }
}

#[test]
fn test_split_uniform_3d_dim1() {
    // [2, 6, 3] tensor, split along dim=1 with size=4 -> [2,4,3] + [2,2,3]
    let data: Vec<f32> = (1..=36).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 6, 3]);
    let parts = t.split_uniform(4, 1).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].dims(), &[2, 4, 3]);
    assert_eq!(parts[1].dims(), &[2, 2, 3]);
}
