// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for DynTensor shape operations:
//! reshape, transpose, permute, squeeze, unsqueeze, expand, narrow, chunk, split.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d, tnd};
use crate::DynTensor;

// ============================================================================
// Reshape tests (12 tests)
// ============================================================================

#[test]
fn test_reshape_3d_to_2d() {
    // [2, 3, 4] -> [6, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.reshape([6, 4]).unwrap();
    assert_eq!(r.dims(), &[6, 4]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_reshape_3d_to_1d() {
    // [2, 3, 4] -> [24]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.reshape([24]).unwrap();
    assert_eq!(r.dims(), &[24]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_reshape_3d_to_different_3d() {
    // [2, 3, 4] -> [4, 3, 2]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.reshape([4, 3, 2]).unwrap();
    assert_eq!(r.dims(), &[4, 3, 2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_reshape_2d_to_4d() {
    // [6, 4] -> [2, 3, 2, 2]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = t2d(&data, 6, 4);
    let r = t.reshape([2, 3, 2, 2]).unwrap();
    assert_eq!(r.dims(), &[2, 3, 2, 2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_reshape_1d_to_2d() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = t.reshape([2, 3]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(
        r.to_vec2::<f32>().unwrap(),
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0],]
    );
}

#[test]
fn test_reshape_identity() {
    // Reshape to same shape is identity
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = tnd(&data, &[3, 4]);
    let r = t.reshape([3, 4]).unwrap();
    assert_eq!(r.dims(), &[3, 4]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_reshape_numel_mismatch_error() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.reshape([2, 2]).is_err(), "numel 3 != 4");
}

#[test]
fn test_reshape_scalar_to_1d() {
    // Scalar (rank-0) -> [1]
    let t = DynTensor::full(&[], 42.0, crate::DType::F32, &cpu()).unwrap();
    let r = t.reshape([1]).unwrap();
    assert_eq!(r.dims(), &[1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![42.0]);
}

#[test]
fn test_reshape_1d_to_scalar() {
    // [1] -> scalar (rank-0)
    let t = t1d(&[7.0]);
    let r = t.reshape(&[] as &[usize]).unwrap();
    assert_eq!(r.dims(), &[] as &[usize]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![7.0]);
}

#[test]
fn test_reshape_preserves_data_order() {
    // Verify data is in C-contiguous (row-major) order after reshape
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.reshape([3, 2]).unwrap();
    assert_eq!(
        r.to_vec2::<f32>().unwrap(),
        vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0],]
    );
}

#[test]
fn test_reshape_single_element() {
    let t = t1d(&[99.0]);
    let r = t.reshape([1, 1, 1]).unwrap();
    assert_eq!(r.dims(), &[1, 1, 1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![99.0]);
}

#[test]
fn test_reshape_large_flatten() {
    // [2, 3, 4, 5] -> [120]
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4, 5]);
    let r = t.reshape([120]).unwrap();
    assert_eq!(r.dims(), &[120]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

// ============================================================================
// Transpose tests (12 tests)
// ============================================================================

#[test]
fn test_transpose_2d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.transpose(0, 1).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_eq!(
        r.to_vec2::<f32>().unwrap(),
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0],]
    );
}

#[test]
fn test_transpose_2d_t() {
    // .t() is shorthand for transpose(rank-2, rank-1)
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.t().unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_eq!(
        r.to_vec2::<f32>().unwrap(),
        vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0],]
    );
}

#[test]
fn test_transpose_3d_01() {
    // [2, 3, 4] transpose(0, 1) -> [3, 2, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.transpose(0, 1).unwrap();
    assert_eq!(r.dims(), &[3, 2, 4]);
}

#[test]
fn test_transpose_3d_02() {
    // [2, 3, 4] transpose(0, 2) -> [4, 3, 2]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.transpose(0, 2).unwrap();
    assert_eq!(r.dims(), &[4, 3, 2]);
}

#[test]
fn test_transpose_3d_12() {
    // [2, 3, 4] transpose(1, 2) -> [2, 4, 3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.transpose(1, 2).unwrap();
    assert_eq!(r.dims(), &[2, 4, 3]);
}

#[test]
fn test_transpose_4d() {
    // [2, 3, 4, 5] transpose(1, 3) -> [2, 5, 4, 3]
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4, 5]);
    let r = t.transpose(1, 3).unwrap();
    assert_eq!(r.dims(), &[2, 5, 4, 3]);
}

#[test]
fn test_transpose_same_dim_is_identity() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = tnd(&data, &[3, 4]);
    let r = t.transpose(0, 0).unwrap();
    assert_eq!(r.dims(), t.dims());
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_double_transpose_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.transpose(0, 2).unwrap().transpose(0, 2).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_transpose_commutative_dims() {
    // transpose(a, b) == transpose(b, a)
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r1 = t.transpose(0, 2).unwrap();
    let r2 = t.transpose(2, 0).unwrap();
    assert_eq!(r1.dims(), r2.dims());
    assert_eq!(
        r1.to_flat_vec::<f32>().unwrap(),
        r2.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_transpose_value_correctness_3d() {
    // Small 3D: [[[1,2],[3,4]], [[5,6],[7,8]]] shape [2,2,2]
    // transpose(0,2) should give [[[1,5],[3,7]], [[2,6],[4,8]]]
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let r = t.transpose(0, 2).unwrap();
    assert_eq!(r.dims(), &[2, 2, 2]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]
    );
}

#[test]
fn test_transpose_dim_out_of_range_error() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(t.transpose(0, 3).is_err());
}

#[test]
fn test_t_on_1d_error() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.t().is_err(), ".t() requires rank >= 2");
}

// ============================================================================
// Permute tests (10 tests)
// ============================================================================

#[test]
fn test_permute_identity_2d() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = t2d(&data, 2, 3);
    let r = t.permute([0, 1]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_permute_identity_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.permute([0, 1, 2]).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_permute_reverse_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.permute([2, 1, 0]).unwrap();
    assert_eq!(r.dims(), &[4, 3, 2]);
}

#[test]
fn test_permute_cyclic_3d() {
    // Cyclic: [0,1,2] -> [1,2,0]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.permute([1, 2, 0]).unwrap();
    assert_eq!(r.dims(), &[3, 4, 2]);
}

#[test]
fn test_permute_nchw_to_nhwc() {
    // NCHW [1, 3, 2, 2] -> NHWC [1, 2, 2, 3]
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = tnd(&data, &[1, 3, 2, 2]);
    let r = t.permute([0, 2, 3, 1]).unwrap();
    assert_eq!(r.dims(), &[1, 2, 2, 3]);
}

#[test]
fn test_permute_nhwc_to_nchw() {
    // NHWC [1, 2, 2, 3] -> NCHW [1, 3, 2, 2]
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = tnd(&data, &[1, 2, 2, 3]);
    let r = t.permute([0, 3, 1, 2]).unwrap();
    assert_eq!(r.dims(), &[1, 3, 2, 2]);
}

#[test]
fn test_permute_inverse_roundtrip() {
    // Permute then inverse permute = identity
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let perm = [2, 0, 1]; // forward
    let inv_perm = [1, 2, 0]; // inverse of [2,0,1]
    let r = t.permute(perm).unwrap().permute(inv_perm).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_permute_rank_mismatch_error() {
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(t.permute([0, 1, 2]).is_err(), "permute dims len != rank");
}

#[test]
fn test_permute_duplicate_axis_error() {
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(t.permute([0, 0]).is_err(), "duplicate axis 0");
}

#[test]
fn test_permute_out_of_range_error() {
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(t.permute([0, 5]).is_err(), "axis 5 out of range for rank 2");
}

// ============================================================================
// Squeeze / Unsqueeze tests (10 tests)
// ============================================================================

#[test]
fn test_unsqueeze_dim0() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.unsqueeze(0).unwrap();
    assert_eq!(r.dims(), &[1, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_unsqueeze_dim1() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.unsqueeze(1).unwrap();
    assert_eq!(r.dims(), &[3, 1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_unsqueeze_2d_middle() {
    // [2, 3] -> [2, 1, 3]
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.unsqueeze(1).unwrap();
    assert_eq!(r.dims(), &[2, 1, 3]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_unsqueeze_2d_end() {
    // [2, 3] -> [2, 3, 1]
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.unsqueeze(2).unwrap();
    assert_eq!(r.dims(), &[2, 3, 1]);
}

#[test]
fn test_squeeze_dim0() {
    // [1, 3] -> [3]
    let t = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = t.squeeze(0).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_squeeze_dim1() {
    // [3, 1] -> [3]
    let t = tnd(&[1.0, 2.0, 3.0], &[3, 1]);
    let r = t.squeeze(1).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_squeeze_non_one_dim_error() {
    // Squeeze on dim with size != 1 should fail
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert!(t.squeeze(0).is_err(), "dim 0 has size 2, not 1");
    assert!(t.squeeze(1).is_err(), "dim 1 has size 3, not 1");
}

#[test]
fn test_unsqueeze_squeeze_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = t2d(&data, 2, 3);
    // unsqueeze at dim 1, then squeeze at dim 1 = identity
    let r = t.unsqueeze(1).unwrap().squeeze(1).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_squeeze_unsqueeze_roundtrip() {
    let data = vec![1.0, 2.0, 3.0];
    let t = tnd(&data, &[1, 3]);
    // squeeze at dim 0, then unsqueeze at dim 0 = identity
    let r = t.squeeze(0).unwrap().unsqueeze(0).unwrap();
    assert_eq!(r.dims(), &[1, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_unsqueeze_multiple_dims() {
    // [3] -> [1, 3] -> [1, 1, 3] -> [1, 1, 3, 1]
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t
        .unsqueeze(0)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .unsqueeze(3)
        .unwrap();
    assert_eq!(r.dims(), &[1, 1, 3, 1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

// ============================================================================
// Expand / Broadcast tests (9 tests)
// ============================================================================

#[test]
fn test_expand_1x3_to_4x3() {
    let t = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = t.expand([4, 3]).unwrap();
    assert_eq!(r.dims(), &[4, 3]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    // Each row should be [1, 2, 3]
    for row in 0..4 {
        assert_eq!(&flat[row * 3..(row + 1) * 3], &[1.0, 2.0, 3.0]);
    }
}

#[test]
fn test_expand_3x1_to_3x4() {
    let t = tnd(&[10.0, 20.0, 30.0], &[3, 1]);
    let r = t.expand([3, 4]).unwrap();
    assert_eq!(r.dims(), &[3, 4]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(&flat[0..4], &[10.0, 10.0, 10.0, 10.0]);
    assert_eq!(&flat[4..8], &[20.0, 20.0, 20.0, 20.0]);
    assert_eq!(&flat[8..12], &[30.0, 30.0, 30.0, 30.0]);
}

#[test]
fn test_expand_3d() {
    // [3, 1, 1] -> [3, 4, 5]
    let t = tnd(&[1.0, 2.0, 3.0], &[3, 1, 1]);
    let r = t.expand([3, 4, 5]).unwrap();
    assert_eq!(r.dims(), &[3, 4, 5]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    // First 20 elements (batch 0) all = 1.0
    assert!(flat[0..20].iter().all(|&v| v == 1.0));
    // Next 20 (batch 1) all = 2.0
    assert!(flat[20..40].iter().all(|&v| v == 2.0));
    // Next 20 (batch 2) all = 3.0
    assert!(flat[40..60].iter().all(|&v| v == 3.0));
}

#[test]
fn test_expand_identity() {
    // Expand to same shape = identity
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = t2d(&data, 2, 3);
    let r = t.expand([2, 3]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_expand_rank_mismatch_error() {
    let t = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    assert!(
        t.expand([2, 3, 4]).is_err(),
        "rank 2 cannot expand to rank 3"
    );
}

#[test]
fn test_expand_non_one_dim_mismatch_error() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert!(
        t.expand([4, 3]).is_err(),
        "dim 0 is 2, not 1, cannot expand to 4"
    );
}

#[test]
fn test_expand_mixed_broadcast() {
    // [1, 3, 1] -> [2, 3, 4]: dims 0 and 2 broadcast, dim 1 stays
    let t = tnd(&[10.0, 20.0, 30.0], &[1, 3, 1]);
    let r = t.expand([2, 3, 4]).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat.len(), 24);
    // batch 0: channel 0 = [10,10,10,10], channel 1 = [20,20,20,20], channel 2 = [30,30,30,30]
    assert_eq!(&flat[0..4], &[10.0, 10.0, 10.0, 10.0]);
    assert_eq!(&flat[4..8], &[20.0, 20.0, 20.0, 20.0]);
    assert_eq!(&flat[8..12], &[30.0, 30.0, 30.0, 30.0]);
    // batch 1 should be identical
    assert_eq!(&flat[12..24], &flat[0..12]);
}

#[test]
fn test_expand_scalar_like() {
    // [1, 1] -> [3, 4]
    let t = tnd(&[5.0], &[1, 1]);
    let r = t.expand([3, 4]).unwrap();
    assert_eq!(r.dims(), &[3, 4]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 5.0));
    assert_eq!(flat.len(), 12);
}

#[test]
fn test_expand_4d_batch_head() {
    // [1, 1, 2, 3] -> [2, 4, 2, 3]
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]);
    let r = t.expand([2, 4, 2, 3]).unwrap();
    assert_eq!(r.dims(), &[2, 4, 2, 3]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat.len(), 48);
    // All 8 copies of the 2x3 block should be identical
    let block = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for i in 0..8 {
        assert_eq!(&flat[i * 6..(i + 1) * 6], block, "block {i}");
    }
}

// ============================================================================
// Narrow tests (8 tests)
// ============================================================================

#[test]
fn test_narrow_1d_start() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let r = t.narrow(0, 0, 2).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![10.0, 20.0]);
}

#[test]
fn test_narrow_1d_middle() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let r = t.narrow(0, 1, 3).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![20.0, 30.0, 40.0]);
}

#[test]
fn test_narrow_1d_end() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let r = t.narrow(0, 3, 2).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![40.0, 50.0]);
}

#[test]
fn test_narrow_2d_dim0() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let r = t.narrow(0, 1, 1).unwrap();
    assert_eq!(r.dims(), &[1, 2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn test_narrow_2d_dim1() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.narrow(1, 1, 2).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn test_narrow_full_dim() {
    // Narrow the entire dimension = identity
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = t2d(&data, 2, 3);
    let r = t.narrow(0, 0, 2).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_narrow_out_of_bounds_error() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.narrow(0, 2, 2).is_err(), "start+len > dim_size");
}

#[test]
fn test_narrow_3d() {
    // [2, 3, 4] narrow dim=1, start=1, len=2 -> [2, 2, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.narrow(1, 1, 2).unwrap();
    assert_eq!(r.dims(), &[2, 2, 4]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    // batch 0: rows 1-2 of the 3x4 matrix: [4..12]
    assert_eq!(&flat[0..8], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    // batch 1: rows 1-2 of the 3x4 matrix: [16..24]
    assert_eq!(
        &flat[8..16],
        &[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    );
}

// ============================================================================
// Chunk tests (8 tests)
// ============================================================================

#[test]
fn test_chunk_even_split() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_chunk_with_remainder() {
    // 7 elements into 3 chunks: chunk_size=3, so [3, 3, 1]
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(chunks[1].to_flat_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(chunks[2].to_flat_vec::<f32>().unwrap(), vec![7.0]);
}

#[test]
fn test_chunk_one_chunk() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let chunks = t.chunk(1, 0).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_chunk_zero_error() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.chunk(0, 0).is_err());
}

#[test]
fn test_chunk_2d_dim0() {
    // [4, 3] chunk into 2 along dim 0 -> two [2, 3] tensors
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = t2d(&data, 4, 3);
    let chunks = t.chunk(2, 0).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[2, 3]);
    assert_eq!(chunks[1].dims(), &[2, 3]);
    assert_eq!(
        chunks[0].to_flat_vec::<f32>().unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(
        chunks[1].to_flat_vec::<f32>().unwrap(),
        vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn test_chunk_preserves_total_elements() {
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let t = tnd(&data, &[4, 5]);
    let chunks = t.chunk(3, 1).unwrap();
    let total_elements: usize = chunks
        .iter()
        .map(|c| c.to_flat_vec::<f32>().unwrap().len())
        .sum();
    assert_eq!(total_elements, 20);
}

#[test]
fn test_chunk_cat_roundtrip_2d() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = t2d(&data, 3, 4);
    let chunks = t.chunk(2, 1).unwrap();
    let refs: Vec<&DynTensor> = chunks.iter().collect();
    let reconstructed = DynTensor::cat(&refs, 1).unwrap();
    assert_eq!(reconstructed.dims(), &[3, 4]);
    assert_eq!(reconstructed.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_chunk_more_chunks_than_size() {
    // 2 elements, 5 chunks -> only 2 chunks produced (each size 1)
    let t = t1d(&[1.0, 2.0]);
    let chunks = t.chunk(5, 0).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].to_flat_vec::<f32>().unwrap(), vec![1.0]);
    assert_eq!(chunks[1].to_flat_vec::<f32>().unwrap(), vec![2.0]);
}

// ============================================================================
// Split tests (8 tests)
// ============================================================================

#[test]
fn test_split_exact() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let parts = t.split([2, 2, 2], 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(parts[2].to_flat_vec::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_split_unequal_sizes() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = t.split([1, 3, 1], 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_flat_vec::<f32>().unwrap(), vec![1.0]);
    assert_eq!(parts[1].to_flat_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0]);
    assert_eq!(parts[2].to_flat_vec::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn test_split_size_mismatch_error() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.split([1, 1], 0).is_err(), "sizes sum to 2 != 3");
}

#[test]
fn test_split_uniform_even() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let parts = t.split_uniform(2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    for (i, part) in parts.iter().enumerate() {
        assert_eq!(part.dims(), &[2], "part {i}");
    }
}

#[test]
fn test_split_uniform_with_remainder() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = t.split_uniform(2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].dims(), &[2]);
    assert_eq!(parts[1].dims(), &[2]);
    assert_eq!(parts[2].dims(), &[1]); // remainder
}

#[test]
fn test_split_uniform_zero_error() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.split_uniform(0, 0).is_err());
}

#[test]
fn test_split_cat_roundtrip() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = t2d(&data, 4, 3);
    let parts = t.split([1, 2, 1], 0).unwrap();
    let refs: Vec<&DynTensor> = parts.iter().collect();
    let reconstructed = DynTensor::cat(&refs, 0).unwrap();
    assert_eq!(reconstructed.dims(), &[4, 3]);
    assert_eq!(reconstructed.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_split_2d_dim1() {
    // [3, 6] split along dim 1 into [2, 3, 1]
    let data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let t = t2d(&data, 3, 6);
    let parts = t.split([2, 3, 1], 1).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].dims(), &[3, 2]);
    assert_eq!(parts[1].dims(), &[3, 3]);
    assert_eq!(parts[2].dims(), &[3, 1]);
}
