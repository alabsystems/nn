// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for free-function scatter, gather, and index_select wrappers.

use super::{gather, index_select, scatter, scatter_add};
use crate::dyn_tensor::test_helpers::{cpu, tnd};
use crate::dyn_tensor::DynTensor;
use crate::DType;

// =============================================================================
// gather — basic 1D
// =============================================================================

#[test]
fn test_gather_1d() {
    let input = tnd(&[100.0, 200.0, 300.0, 400.0], &[4]);
    let index = DynTensor::from_vec_u32(vec![3, 1, 0], &[3], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![400.0, 200.0, 100.0]);
}

#[test]
fn test_gather_1d_single_element() {
    let input = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![2], &[1], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[1]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![30.0]);
}

#[test]
fn test_gather_1d_identity() {
    // Gather all elements in order — should reproduce input.
    let input = tnd(&[5.0, 6.0, 7.0, 8.0], &[4]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2, 3], &[4], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_gather_1d_reverse() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![2, 1, 0], &[3], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_gather_1d_duplicate_indices() {
    let input = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 1, 1, 1], &[4], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![20.0, 20.0, 20.0, 20.0]
    );
}

// =============================================================================
// gather — basic 2D
// =============================================================================

#[test]
fn test_gather_2d_dim0() {
    // input shape [3, 2], gather along dim=0 with index shape [2, 2]
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 2, 1, 0], &[2, 2], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // result[0][0] = input[0][0] = 1.0
    // result[0][1] = input[2][1] = 6.0
    // result[1][0] = input[1][0] = 3.0
    // result[1][1] = input[0][1] = 2.0
    assert_eq!(v, vec![1.0, 6.0, 3.0, 2.0]);
}

#[test]
fn test_gather_2d_dim1() {
    // input shape [2, 3], gather along dim=1 with index shape [2, 2]
    let input = tnd(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    let index = DynTensor::from_vec_u32(vec![2, 0, 1, 2], &[2, 2], &cpu()).unwrap();
    let result = gather(&input, 1, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // result[0][0] = input[0][2] = 30.0
    // result[0][1] = input[0][0] = 10.0
    // result[1][0] = input[1][1] = 50.0
    // result[1][1] = input[1][2] = 60.0
    assert_eq!(v, vec![30.0, 10.0, 50.0, 60.0]);
}

#[test]
fn test_gather_2d_dim0_single_row() {
    // Gather a single row-index from a 3x3 matrix along dim 0.
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let index = DynTensor::from_vec_u32(vec![2, 0, 1], &[1, 3], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[1, 3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // result[0][0] = input[2][0] = 7.0
    // result[0][1] = input[0][1] = 2.0
    // result[0][2] = input[1][2] = 6.0
    assert_eq!(v, vec![7.0, 2.0, 6.0]);
}

// =============================================================================
// gather — 3D along different axes
// =============================================================================

#[test]
fn test_gather_3d_dim0() {
    // input [2, 2, 3], gather along dim=0 with index [1, 2, 3]
    #[rustfmt::skip]
    let input = tnd(&[
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,   // batch 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,  // batch 1
    ], &[2, 2, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 0, 1, 0, 1], &[1, 2, 3], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[1, 2, 3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // result[0][0][0] = input[0][0][0] = 1.0
    // result[0][0][1] = input[1][0][1] = 8.0
    // result[0][0][2] = input[0][0][2] = 3.0
    // result[0][1][0] = input[1][1][0] = 10.0
    // result[0][1][1] = input[0][1][1] = 5.0
    // result[0][1][2] = input[1][1][2] = 12.0
    assert_eq!(v, vec![1.0, 8.0, 3.0, 10.0, 5.0, 12.0]);
}

#[test]
fn test_gather_3d_dim1() {
    // input [2, 3, 2], gather along dim=1 with index [2, 2, 2]
    #[rustfmt::skip]
    let input = tnd(&[
        1.0, 2.0,  3.0, 4.0,  5.0, 6.0,   // batch 0: rows [1,2], [3,4], [5,6]
        7.0, 8.0,  9.0, 10.0, 11.0, 12.0,  // batch 1: rows [7,8], [9,10], [11,12]
    ], &[2, 3, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 2, 1, 0, 2, 1, 0, 1], &[2, 2, 2], &cpu()).unwrap();
    let result = gather(&input, 1, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // batch 0:
    //   [0][0][0] = input[0][0][0] = 1.0
    //   [0][0][1] = input[0][2][1] = 6.0
    //   [0][1][0] = input[0][1][0] = 3.0
    //   [0][1][1] = input[0][0][1] = 2.0
    // batch 1:
    //   [1][0][0] = input[1][2][0] = 11.0
    //   [1][0][1] = input[1][1][1] = 10.0
    //   [1][1][0] = input[1][0][0] = 7.0
    //   [1][1][1] = input[1][1][1] = 10.0
    assert_eq!(v, vec![1.0, 6.0, 3.0, 2.0, 11.0, 10.0, 7.0, 10.0]);
}

#[test]
fn test_gather_3d_dim2() {
    // input [2, 2, 4], gather along dim=2 with index [2, 2, 2]
    #[rustfmt::skip]
    let input = tnd(&[
        1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,    // batch 0
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // batch 1
    ], &[2, 2, 4]);
    let index = DynTensor::from_vec_u32(vec![3, 0, 1, 2, 0, 3, 2, 1], &[2, 2, 2], &cpu()).unwrap();
    let result = gather(&input, 2, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // result[0][0][0] = input[0][0][3] = 4.0
    // result[0][0][1] = input[0][0][0] = 1.0
    // result[0][1][0] = input[0][1][1] = 6.0
    // result[0][1][1] = input[0][1][2] = 7.0
    // result[1][0][0] = input[1][0][0] = 9.0
    // result[1][0][1] = input[1][0][3] = 12.0
    // result[1][1][0] = input[1][1][2] = 15.0
    // result[1][1][1] = input[1][1][1] = 14.0
    assert_eq!(v, vec![4.0, 1.0, 6.0, 7.0, 9.0, 12.0, 15.0, 14.0]);
}

// =============================================================================
// gather — error cases
// =============================================================================

#[test]
fn test_gather_rank_mismatch() {
    let input = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    // index rank 1 != input rank 2
    assert!(gather(&input, 0, &index).is_err());
}

#[test]
fn test_gather_oob_index() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![5], &[1], &cpu()).unwrap();
    assert!(gather(&input, 0, &index).is_err());
}

#[test]
fn test_gather_oob_2d() {
    // Index 3 is out of bounds for dim 1 of size 3.
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 3], &[2, 1], &cpu()).unwrap();
    assert!(gather(&input, 1, &index).is_err());
}

#[test]
fn test_gather_non_gather_dim_exceeds() {
    // index non-gather dim size > input non-gather dim size
    let input = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 0, 0, 0, 0, 0], &[2, 3], &cpu()).unwrap();
    assert!(gather(&input, 0, &index).is_err());
}

// =============================================================================
// gather — dtype preservation
// =============================================================================

#[test]
fn test_gather_preserves_f32_dtype() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

// =============================================================================
// gather — PyTorch torch.gather equivalence
// =============================================================================

#[test]
fn test_gather_pytorch_compat_2d() {
    // Equivalent to:
    //   t = torch.tensor([[1,2],[3,4]])
    //   idx = torch.tensor([[0,0],[1,0]])
    //   torch.gather(t, 1, idx)
    //   => tensor([[1,1],[4,3]])
    let input = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 0, 1, 0], &[2, 2], &cpu()).unwrap();
    let result = gather(&input, 1, &index).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 4.0, 3.0]
    );
}

#[test]
fn test_gather_pytorch_compat_dim0() {
    // Equivalent to:
    //   t = torch.tensor([[1,2,3],[4,5,6]])
    //   idx = torch.tensor([[0,1,1],[1,0,0]])
    //   torch.gather(t, 0, idx)
    //   => tensor([[1,5,6],[4,2,3]])
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 1, 1, 0, 0], &[2, 3], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 5.0, 6.0, 4.0, 2.0, 3.0]
    );
}

// =============================================================================
// scatter — basic 1D
// =============================================================================

#[test]
fn test_scatter_1d() {
    let input = tnd(&[0.0, 0.0, 0.0, 0.0, 0.0], &[5]);
    let src = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 3, 0], &[3], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![30.0, 10.0, 0.0, 20.0, 0.0]);
}

#[test]
fn test_scatter_1d_single_element() {
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[42.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![2], &[1], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![0.0, 0.0, 42.0]);
}

#[test]
fn test_scatter_1d_preserves_unscattered() {
    // Only scattered positions change; others keep original values.
    let input = tnd(&[100.0, 200.0, 300.0, 400.0], &[4]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![2], &[1], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![100.0, 200.0, 1.0, 400.0]
    );
}

// =============================================================================
// scatter — 2D
// =============================================================================

#[test]
fn test_scatter_2d_dim0() {
    let input = tnd(&[0.0; 6], &[3, 2]);
    let src = tnd(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 2, 1, 0], &[2, 2], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // [0][0] = src[0][0] = 10 (index[0][0]=0)
    // [2][1] = src[0][1] = 20 (index[0][1]=2)
    // [1][0] = src[1][0] = 30 (index[1][0]=1)
    // [0][1] = src[1][1] = 40 (index[1][1]=0)
    assert_eq!(v, vec![10.0, 40.0, 30.0, 0.0, 0.0, 20.0]);
}

#[test]
fn test_scatter_2d_dim1() {
    let input = tnd(&[0.0; 6], &[2, 3]);
    let src = tnd(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![2, 0, 1, 2], &[2, 2], &cpu()).unwrap();
    let result = scatter(&input, 1, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Row 0: [0][2]=10, [0][0]=20 => [20, 0, 10]
    // Row 1: [1][1]=30, [1][2]=40 => [0, 30, 40]
    assert_eq!(v, vec![20.0, 0.0, 10.0, 0.0, 30.0, 40.0]);
}

// =============================================================================
// scatter — 3D
// =============================================================================

#[test]
fn test_scatter_3d_dim1() {
    // input [2, 3, 2], scatter along dim=1
    let input = tnd(&[0.0; 12], &[2, 3, 2]);
    let src = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    // index must match src shape [2,1,2] and refer to positions along dim=1 (size 3)
    let index = DynTensor::from_vec_u32(vec![2, 0, 1, 2], &[2, 1, 2], &cpu()).unwrap();
    let result = scatter(&input, 1, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // batch 0:
    //   output[0][2][0] = src[0][0][0] = 1.0  (index[0][0][0]=2)
    //   output[0][0][1] = src[0][0][1] = 2.0  (index[0][0][1]=0)
    // batch 1:
    //   output[1][1][0] = src[1][0][0] = 3.0  (index[1][0][0]=1)
    //   output[1][2][1] = src[1][0][1] = 4.0  (index[1][0][1]=2)
    assert_eq!(
        v,
        vec![
            0.0, 2.0, 0.0, 0.0, 1.0, 0.0, // batch 0: row0=[0,2], row1=[0,0], row2=[1,0]
            0.0, 0.0, 3.0, 0.0, 0.0, 4.0, // batch 1: row0=[0,0], row1=[3,0], row2=[0,4]
        ]
    );
}

// =============================================================================
// scatter — duplicate indices (last-write-wins)
// =============================================================================

#[test]
fn test_scatter_duplicate_indices_last_wins() {
    // When multiple source elements scatter to the same position,
    // the last one in iteration order wins (overwrite semantics).
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 1, 1], &[3], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // All three go to index 1; last value (30.0) wins.
    assert_eq!(v[0], 0.0);
    assert_eq!(v[2], 0.0);
    // The final overwrite is 30.0 (last in iteration order).
    assert_eq!(v[1], 30.0);
}

// =============================================================================
// scatter — overwrites original
// =============================================================================

#[test]
fn test_scatter_overwrites() {
    let input = tnd(&[100.0, 100.0, 100.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![1], &[1], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![100.0, 1.0, 100.0]);
}

// =============================================================================
// scatter — error cases
// =============================================================================

#[test]
fn test_scatter_oob_index() {
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![5], &[1], &cpu()).unwrap();
    assert!(scatter(&input, 0, &index, &src).is_err());
}

#[test]
fn test_scatter_shape_mismatch() {
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0, 2.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    // index shape [1] != src shape [2]
    assert!(scatter(&input, 0, &index, &src).is_err());
}

#[test]
fn test_scatter_rank_mismatch() {
    let input = tnd(&[0.0; 6], &[2, 3]);
    let src = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2], &[3], &cpu()).unwrap();
    // index and src are rank 1, input is rank 2
    assert!(scatter(&input, 0, &index, &src).is_err());
}

// =============================================================================
// scatter — PyTorch torch.scatter equivalence
// =============================================================================

#[test]
fn test_scatter_pytorch_compat() {
    // Equivalent to:
    //   src = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    //   idx = torch.tensor([[0,1,2],[0,1,2]])
    //   torch.zeros(3,3).scatter_(0, idx, src)
    //   => tensor([[1,0,0],[0,2,0],[0,0,3],[4,0,0],[0,5,0],[0,0,6]]) — no wait
    // Let me do it properly:
    //   dst = torch.zeros(3,3)
    //   src = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    //   idx = torch.tensor([[0,1,2],[2,0,1]])
    //   dst.scatter_(0, idx, src)
    //   => row 0: [1,5,0], row 1: [0,2,6], row 2: [4,0,3]
    let input = tnd(&[0.0; 9], &[3, 3]);
    let src = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2, 2, 0, 1], &[2, 3], &cpu()).unwrap();
    let result = scatter(&input, 0, &index, &src).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 5.0, 0.0, 0.0, 2.0, 6.0, 4.0, 0.0, 3.0]
    );
}

// =============================================================================
// scatter_add — basic 1D
// =============================================================================

#[test]
fn test_scatter_add_1d() {
    let input = tnd(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let src = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 1, 3], &[3], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // index[0]=1 => output[1] += 10
    // index[1]=1 => output[1] += 20
    // index[2]=3 => output[3] += 30
    assert_eq!(v, vec![0.0, 30.0, 0.0, 30.0]);
}

#[test]
fn test_scatter_add_1d_no_overlap() {
    let input = tnd(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let src = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![0, 2, 3], &[3], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![10.0, 0.0, 20.0, 30.0]);
}

// =============================================================================
// scatter_add — accumulation with duplicate indices
// =============================================================================

#[test]
fn test_scatter_add_accumulates() {
    let input = tnd(&[5.0, 5.0, 5.0], &[3]);
    let src = tnd(&[1.0, 2.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![1, 1], &[2], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // output[1] = 5.0 + 1.0 + 2.0 = 8.0
    assert_eq!(v, vec![5.0, 8.0, 5.0]);
}

#[test]
fn test_scatter_add_all_to_same_index() {
    let input = tnd(&[100.0, 0.0], &[2]);
    let src = tnd(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let index = DynTensor::from_vec_u32(vec![1, 1, 1, 1], &[4], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // output[1] = 0.0 + 1 + 2 + 3 + 4 = 10.0
    assert_eq!(v, vec![100.0, 10.0]);
}

// =============================================================================
// scatter_add — 2D
// =============================================================================

#[test]
fn test_scatter_add_2d_dim0() {
    let input = tnd(&[0.0; 6], &[3, 2]);
    let src = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 0, 2], &[2, 2], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // [0][0] += 1.0 (from [0][0], index=0)
    // [1][1] += 2.0 (from [0][1], index=1)
    // [0][0] += 3.0 (from [1][0], index=0)
    // [2][1] += 4.0 (from [1][1], index=2)
    assert_eq!(v, vec![4.0, 0.0, 0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn test_scatter_add_2d_dim1() {
    let input = tnd(&[0.0; 6], &[2, 3]);
    let src = tnd(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 0, 1, 1], &[2, 2], &cpu()).unwrap();
    let result = scatter_add(&input, 1, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Row 0: [0][0] += 10+20 = 30
    // Row 1: [1][1] += 30+40 = 70
    assert_eq!(v, vec![30.0, 0.0, 0.0, 0.0, 70.0, 0.0]);
}

// =============================================================================
// scatter_add — 3D
// =============================================================================

#[test]
fn test_scatter_add_3d_dim2() {
    // input [1, 2, 4], scatter_add along dim=2 with src [1, 2, 3]
    let input = tnd(&[0.0; 8], &[1, 2, 4]);
    let src = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 0, 1, 2, 2, 3], &[1, 2, 3], &cpu()).unwrap();
    let result = scatter_add(&input, 2, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Row [0][0]: idx 0 += 1+2=3, idx 1 += 3
    // Row [0][1]: idx 2 += 4+5=9, idx 3 += 6
    assert_eq!(v, vec![3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 6.0]);
}

// =============================================================================
// scatter_add — error cases
// =============================================================================

#[test]
fn test_scatter_add_oob_index() {
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![10], &[1], &cpu()).unwrap();
    assert!(scatter_add(&input, 0, &index, &src).is_err());
}

#[test]
fn test_scatter_add_rank_mismatch() {
    let input = tnd(&[0.0; 6], &[2, 3]);
    let src = tnd(&[1.0, 2.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![0, 1], &[2], &cpu()).unwrap();
    // src/index rank 1, input rank 2
    assert!(scatter_add(&input, 0, &index, &src).is_err());
}

#[test]
fn test_scatter_add_shape_mismatch() {
    let input = tnd(&[0.0; 6], &[2, 3]);
    let src = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2, 0, 1, 2], &[2, 3], &cpu()).unwrap();
    // index shape [2,3] != src shape [1,3]
    assert!(scatter_add(&input, 0, &index, &src).is_err());
}

// =============================================================================
// scatter_add — preserves dtype
// =============================================================================

#[test]
fn test_scatter_add_preserves_dtype() {
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

// =============================================================================
// index_select — basic 1D
// =============================================================================

#[test]
fn test_index_select_1d() {
    let input = tnd(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let index = DynTensor::from_vec_u32(vec![4, 2, 0], &[3], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![50.0, 30.0, 10.0]);
}

#[test]
fn test_index_select_1d_single_element() {
    let input = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1], &[1], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[1]);
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![20.0]);
}

#[test]
fn test_index_select_1d_all_elements() {
    // Select all elements in order — identity.
    let input = tnd(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 2, 3], &[4], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

// =============================================================================
// index_select — 2D along different dims
// =============================================================================

#[test]
fn test_index_select_2d_dim0() {
    // Select rows from a 3x2 matrix
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let index = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn test_index_select_2d_dim1() {
    // Select columns from a 2x4 matrix
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let index = DynTensor::from_vec_u32(vec![3, 1], &[2], &cpu()).unwrap();
    let result = index_select(&input, 1, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![4.0, 2.0, 8.0, 6.0]);
}

// =============================================================================
// index_select — 3D along different dims
// =============================================================================

#[test]
fn test_index_select_3d_dim0() {
    // input [3, 2, 2], select batches
    #[rustfmt::skip]
    let input = tnd(&[
        1.0, 2.0, 3.0, 4.0,     // batch 0
        5.0, 6.0, 7.0, 8.0,     // batch 1
        9.0, 10.0, 11.0, 12.0,  // batch 2
    ], &[3, 2, 2]);
    let index = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_index_select_3d_dim1() {
    // input [2, 4, 3], select 2 slices along dim=1
    #[rustfmt::skip]
    let input = tnd(&[
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0,  10.0, 11.0, 12.0,  // batch 0
        13.0, 14.0, 15.0,  16.0, 17.0, 18.0,  19.0, 20.0, 21.0,  22.0, 23.0, 24.0, // batch 1
    ], &[2, 4, 3]);
    let index = DynTensor::from_vec_u32(vec![3, 1], &[2], &cpu()).unwrap();
    let result = index_select(&input, 1, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2, 3]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // batch 0: rows 3 and 1 => [10,11,12], [4,5,6]
    // batch 1: rows 3 and 1 => [22,23,24], [16,17,18]
    assert_eq!(
        v,
        vec![10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 22.0, 23.0, 24.0, 16.0, 17.0, 18.0]
    );
}

#[test]
fn test_index_select_3d_dim2() {
    // input [2, 2, 4], select columns
    #[rustfmt::skip]
    let input = tnd(&[
        1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,    // batch 0
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // batch 1
    ], &[2, 2, 4]);
    let index = DynTensor::from_vec_u32(vec![3, 0], &[2], &cpu()).unwrap();
    let result = index_select(&input, 2, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let v = result.to_flat_vec::<f32>().unwrap();
    // batch0 row0: cols 3,0 => [4,1]
    // batch0 row1: cols 3,0 => [8,5]
    // batch1 row0: cols 3,0 => [12,9]
    // batch1 row1: cols 3,0 => [16,13]
    assert_eq!(v, vec![4.0, 1.0, 8.0, 5.0, 12.0, 9.0, 16.0, 13.0]);
}

// =============================================================================
// index_select — duplicate indices
// =============================================================================

#[test]
fn test_index_select_duplicate_indices() {
    let input = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 1, 1], &[3], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![20.0, 20.0, 20.0]);
}

#[test]
fn test_index_select_duplicate_indices_2d() {
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 0, 0], &[3], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[3, 2]);
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

// =============================================================================
// index_select — dtype preservation
// =============================================================================

#[test]
fn test_index_select_preserves_dtype() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

// =============================================================================
// index_select — error cases
// =============================================================================

#[test]
fn test_index_select_oob() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![5], &[1], &cpu()).unwrap();
    assert!(index_select(&input, 0, &index).is_err());
}

#[test]
fn test_index_select_non_1d_index() {
    let input = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 1, 0, 1], &[2, 2], &cpu()).unwrap();
    // index must be rank-1
    assert!(index_select(&input, 0, &index).is_err());
}

#[test]
fn test_index_select_oob_dim1() {
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = DynTensor::from_vec_u32(vec![3], &[1], &cpu()).unwrap();
    // dim 1 has size 3, index 3 is out of bounds
    assert!(index_select(&input, 1, &index).is_err());
}

// =============================================================================
// empty index tensor
// =============================================================================

#[test]
fn test_gather_empty_index_1d() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![], &[0], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[0]);
    assert_eq!(result.to_flat_vec::<f32>().unwrap().len(), 0);
}

#[test]
fn test_index_select_empty_index() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![], &[0], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[0]);
    assert_eq!(result.to_flat_vec::<f32>().unwrap().len(), 0);
}

#[test]
fn test_index_select_empty_index_2d() {
    let input = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = DynTensor::from_vec_u32(vec![], &[0], &cpu()).unwrap();
    let result = index_select(&input, 0, &index).unwrap();
    assert_eq!(result.dims(), &[0, 3]);
    assert_eq!(result.to_flat_vec::<f32>().unwrap().len(), 0);
}

// =============================================================================
// Round-trip: scatter then gather recovers original
// =============================================================================

#[test]
fn test_roundtrip_scatter_then_gather_1d() {
    // Scatter values into zeros, then gather them back using the same index.
    let values = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![2, 0, 4], &[3], &cpu()).unwrap();
    let zeros = tnd(&[0.0, 0.0, 0.0, 0.0, 0.0], &[5]);

    let scattered = scatter(&zeros, 0, &index, &values).unwrap();
    let gathered = gather(&scattered, 0, &index).unwrap();
    assert_eq!(
        gathered.to_flat_vec::<f32>().unwrap(),
        vec![10.0, 20.0, 30.0]
    );
}

#[test]
fn test_roundtrip_scatter_then_gather_2d() {
    // 2D round-trip: scatter then gather recovers original values.
    let values = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![1, 0, 2, 1], &[2, 2], &cpu()).unwrap();
    let zeros = tnd(&[0.0; 6], &[2, 3]);

    let scattered = scatter(&zeros, 1, &index, &values).unwrap();
    let gathered = gather(&scattered, 1, &index).unwrap();
    assert_eq!(
        gathered.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

// =============================================================================
// scatter_add vs scatter distinction
// =============================================================================

#[test]
fn test_scatter_add_vs_scatter_with_duplicates() {
    // With duplicate indices, scatter overwrites and scatter_add accumulates.
    let input = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[10.0, 20.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![1, 1], &[2], &cpu()).unwrap();

    let scatter_result = scatter(&input, 0, &index, &src).unwrap();
    let scatter_add_result = scatter_add(&input, 0, &index, &src).unwrap();

    // scatter: last write wins => [0, 20, 0]
    let sv = scatter_result.to_flat_vec::<f32>().unwrap();
    assert_eq!(sv[1], 20.0);

    // scatter_add: accumulates => [0, 30, 0]
    let sav = scatter_add_result.to_flat_vec::<f32>().unwrap();
    assert_eq!(sav[1], 30.0);
}

// =============================================================================
// scatter_add — non-zero initial values
// =============================================================================

#[test]
fn test_scatter_add_nonzero_base() {
    // scatter_add adds to the existing values in the destination.
    let input = tnd(&[100.0, 200.0, 300.0], &[3]);
    let src = tnd(&[5.0, 10.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    assert_eq!(
        result.to_flat_vec::<f32>().unwrap(),
        vec![105.0, 200.0, 310.0]
    );
}

// =============================================================================
// output shape correctness
// =============================================================================

#[test]
fn test_gather_output_shape_matches_index() {
    // The output of gather always has the shape of the index tensor.
    let input = tnd(&[1.0; 24], &[2, 3, 4]);
    let index = DynTensor::from_vec_u32(vec![0; 12], &[2, 2, 3], &cpu()).unwrap();
    let result = gather(&input, 1, &index).unwrap();
    assert_eq!(result.dims(), &[2, 2, 3]);
}

#[test]
fn test_scatter_output_shape_matches_input() {
    // The output of scatter always has the shape of the destination (input).
    let input = tnd(&[0.0; 20], &[4, 5]);
    let src = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let index = DynTensor::from_vec_u32(vec![0, 2, 4], &[1, 3], &cpu()).unwrap();
    let result = scatter(&input, 1, &index, &src).unwrap();
    assert_eq!(result.dims(), &[4, 5]);
}

#[test]
fn test_scatter_add_output_shape_matches_input() {
    let input = tnd(&[0.0; 12], &[3, 4]);
    let src = tnd(&[1.0, 2.0], &[1, 2]);
    let index = DynTensor::from_vec_u32(vec![0, 3], &[1, 2], &cpu()).unwrap();
    let result = scatter_add(&input, 1, &index, &src).unwrap();
    assert_eq!(result.dims(), &[3, 4]);
}

#[test]
fn test_index_select_output_shape() {
    // input [3, 4, 5], index_select dim=1 with 2 indices => [3, 2, 5]
    let input = tnd(&[0.0; 60], &[3, 4, 5]);
    let index = DynTensor::from_vec_u32(vec![1, 3], &[2], &cpu()).unwrap();
    let result = index_select(&input, 1, &index).unwrap();
    assert_eq!(result.dims(), &[3, 2, 5]);
}

// =============================================================================
// scatter into zeros matches scatter_add into zeros (no duplicates)
// =============================================================================

#[test]
fn test_scatter_equals_scatter_add_no_duplicates() {
    // Without duplicate indices, scatter and scatter_add produce the same result.
    let input = tnd(&[0.0; 5], &[5]);
    let src = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 3, 4], &[3], &cpu()).unwrap();

    let scatter_result = scatter(&input, 0, &index, &src).unwrap();
    let scatter_add_result = scatter_add(&input, 0, &index, &src).unwrap();

    assert_eq!(
        scatter_result.to_flat_vec::<f32>().unwrap(),
        scatter_add_result.to_flat_vec::<f32>().unwrap()
    );
}

// =============================================================================
// gather as inverse of index_select (1D case)
// =============================================================================

#[test]
fn test_gather_matches_index_select_1d() {
    // For 1D tensors, gather(input, 0, index) and index_select(input, 0, index)
    // should produce the same result.
    let input = tnd(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let index = DynTensor::from_vec_u32(vec![4, 2, 0, 3], &[4], &cpu()).unwrap();

    let gathered = gather(&input, 0, &index).unwrap();
    let selected = index_select(&input, 0, &index).unwrap();

    assert_eq!(
        gathered.to_flat_vec::<f32>().unwrap(),
        selected.to_flat_vec::<f32>().unwrap()
    );
}

// =============================================================================
// scatter — does not modify original
// =============================================================================

#[test]
fn test_scatter_does_not_modify_original() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let original_values = input.to_flat_vec::<f32>().unwrap();

    let src = tnd(&[99.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![1], &[1], &cpu()).unwrap();
    let _result = scatter(&input, 0, &index, &src).unwrap();

    // Original should be unchanged.
    assert_eq!(input.to_flat_vec::<f32>().unwrap(), original_values);
}

#[test]
fn test_scatter_add_does_not_modify_original() {
    let input = tnd(&[1.0, 2.0, 3.0], &[3]);
    let original_values = input.to_flat_vec::<f32>().unwrap();

    let src = tnd(&[99.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![1], &[1], &cpu()).unwrap();
    let _result = scatter_add(&input, 0, &index, &src).unwrap();

    // Original should be unchanged.
    assert_eq!(input.to_flat_vec::<f32>().unwrap(), original_values);
}

// =============================================================================
// negative values
// =============================================================================

#[test]
fn test_scatter_add_negative_values() {
    let input = tnd(&[10.0, 20.0, 30.0], &[3]);
    let src = tnd(&[-5.0, -15.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = scatter_add(&input, 0, &index, &src).unwrap();
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![5.0, 20.0, 15.0]);
}

#[test]
fn test_gather_negative_values() {
    let input = tnd(&[-1.0, -2.0, -3.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let result = gather(&input, 0, &index).unwrap();
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![-3.0, -1.0]);
}
