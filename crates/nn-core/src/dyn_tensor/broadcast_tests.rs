// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive broadcasting tests for DynTensor binary operations.
//!
//! Covers NumPy-style right-aligned broadcasting semantics across all binary
//! ops (add, sub, mul, div, maximum, minimum), including:
//! - Same-shape (no broadcast)
//! - Right broadcast: [3,4] + [4]
//! - Left broadcast: [1,4] + [3,4]
//! - Scalar broadcast: [3,4] + []
//! - Multi-dim broadcast: [1,3,1] + [2,1,4]
//! - Incompatible shapes error: [3,4] + [5]
//! - High-rank broadcast: [2,1,3,1,5] + [1,7,1,4,1]
//! - Broadcast with batch dim: [B,C,H,W] patterns
//! - Broadcast preserves values (not just shape)
//! - Per-channel broadcast_left vs broadcast_add for [C] + [C,T] patterns
//! - broadcast_output_shape unit tests

use crate::dyn_tensor::ops::broadcast_output_shape;
use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d, tnd};
use crate::{DType, DynTensor};

// ============================================================================
// broadcast_output_shape unit tests
// ============================================================================

#[test]
fn test_broadcast_shape_same_shape() {
    let out = broadcast_output_shape(&[3, 4], &[3, 4]).unwrap();
    assert_eq!(out, vec![3, 4]);
}

#[test]
fn test_broadcast_shape_right_align_lower_rank() {
    // [3,4] + [4] -> right-align: [3,4] + [_,4] -> [3,4]
    let out = broadcast_output_shape(&[3, 4], &[4]).unwrap();
    assert_eq!(out, vec![3, 4]);
}

#[test]
fn test_broadcast_shape_left_broadcast_size1() {
    // [1,4] + [3,4] -> [3,4]
    let out = broadcast_output_shape(&[1, 4], &[3, 4]).unwrap();
    assert_eq!(out, vec![3, 4]);
}

#[test]
fn test_broadcast_shape_scalar() {
    // [] + [3,4] -> [3,4]
    let out = broadcast_output_shape(&[], &[3, 4]).unwrap();
    assert_eq!(out, vec![3, 4]);
}

#[test]
fn test_broadcast_shape_scalar_both() {
    // [] + [] -> []
    let out = broadcast_output_shape(&[], &[]).unwrap();
    assert_eq!(out, Vec::<usize>::new());
}

#[test]
fn test_broadcast_shape_multi_dim() {
    // [1,3,1] + [2,1,4] -> [2,3,4]
    let out = broadcast_output_shape(&[1, 3, 1], &[2, 1, 4]).unwrap();
    assert_eq!(out, vec![2, 3, 4]);
}

#[test]
fn test_broadcast_shape_incompatible_error() {
    // [3,4] + [5] -> error (4 != 5, neither is 1)
    let result = broadcast_output_shape(&[3, 4], &[5]);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_shape_incompatible_same_rank() {
    // [3,4] + [3,5] -> error
    let result = broadcast_output_shape(&[3, 4], &[3, 5]);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_shape_high_rank() {
    // [2,1,3,1,5] + [1,7,1,4,1] -> [2,7,3,4,5]
    let out = broadcast_output_shape(&[2, 1, 3, 1, 5], &[1, 7, 1, 4, 1]).unwrap();
    assert_eq!(out, vec![2, 7, 3, 4, 5]);
}

#[test]
fn test_broadcast_shape_different_ranks() {
    // [5] + [2,3,5] -> right-align: [_,_,5] + [2,3,5] -> [2,3,5]
    let out = broadcast_output_shape(&[5], &[2, 3, 5]).unwrap();
    assert_eq!(out, vec![2, 3, 5]);
}

#[test]
fn test_broadcast_shape_batch_channel_height_width() {
    // [1,C,1,1] + [B,C,H,W] -> [B,C,H,W]
    let out = broadcast_output_shape(&[1, 3, 1, 1], &[2, 3, 4, 5]).unwrap();
    assert_eq!(out, vec![2, 3, 4, 5]);
}

#[test]
fn test_broadcast_shape_per_channel_pattern() {
    // [C] + [B,C,T] -> right-align: [_,_,C] + [B,C,T]
    // This only works if C==T or one is 1. In the general case [C] + [B,C,T]
    // right-aligns C with T, which fails unless C==T.
    // The correct pattern for per-channel is reshape [C] to [1,C,1] first.
    // Test that [C] + [B,C,T] fails when C != T.
    let result = broadcast_output_shape(&[3], &[2, 3, 5]);
    assert!(
        result.is_err(),
        "[3] + [2,3,5] should fail: 3 right-aligns with T=5"
    );
}

#[test]
fn test_broadcast_shape_per_channel_reshaped() {
    // Correct per-channel pattern: [1,C,1] + [B,C,T] -> [B,C,T]
    let out = broadcast_output_shape(&[1, 3, 1], &[2, 3, 5]).unwrap();
    assert_eq!(out, vec![2, 3, 5]);
}

#[test]
fn test_broadcast_shape_commutativity() {
    let a = &[1, 3, 1];
    let b = &[2, 1, 4];
    let ab = broadcast_output_shape(a, b).unwrap();
    let ba = broadcast_output_shape(b, a).unwrap();
    assert_eq!(ab, ba);
}

#[test]
fn test_broadcast_shape_size1_expands() {
    // [1] + [5] -> [5]
    let out = broadcast_output_shape(&[1], &[5]).unwrap();
    assert_eq!(out, vec![5]);
}

#[test]
fn test_broadcast_shape_rank0_with_high_rank() {
    // [] + [2,3,4,5,6] -> [2,3,4,5,6]
    let out = broadcast_output_shape(&[], &[2, 3, 4, 5, 6]).unwrap();
    assert_eq!(out, vec![2, 3, 4, 5, 6]);
}

// ============================================================================
// Same-shape broadcast (no actual broadcast needed)
// ============================================================================

#[test]
fn test_broadcast_add_same_shape_2d() {
    let a = t2d(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        3,
        4,
    );
    let b = t2d(
        &[
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ],
        3,
        4,
    );
    let c = a.broadcast_add(&b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0, 121.0, 132.0]
    );
}

// ============================================================================
// Right broadcast: [3,4] + [4]
// ============================================================================

#[test]
fn test_broadcast_add_right_rank1() {
    // [3,4] + [4] -> [3,4], each row gets the [4] vector added
    let a = t2d(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        3,
        4,
    );
    let b = t1d(&[100.0, 200.0, 300.0, 400.0]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![101.0, 202.0, 303.0, 404.0, 105.0, 206.0, 307.0, 408.0, 109.0, 210.0, 311.0, 412.0]
    );
}

#[test]
fn test_broadcast_sub_right_rank1() {
    let a = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    let b = t1d(&[1.0, 2.0]);
    let c = a.sub(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![9.0, 18.0, 29.0, 38.0]);
}

#[test]
fn test_broadcast_mul_right_rank1() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t1d(&[10.0, 100.0, 1000.0]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]
    );
}

#[test]
fn test_broadcast_div_right_rank1() {
    let a = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    let b = t1d(&[2.0, 5.0]);
    let c = a.div(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![5.0, 4.0, 15.0, 8.0]);
}

// ============================================================================
// Left broadcast: [1,4] + [3,4]
// ============================================================================

#[test]
fn test_broadcast_add_left_size1() {
    // [1,4] + [3,4] -> [3,4]
    let a = DynTensor::from_vec(vec![100.0, 200.0, 300.0, 400.0], &[1, 4], &cpu()).unwrap();
    let b = t2d(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        3,
        4,
    );
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![101.0, 202.0, 303.0, 404.0, 105.0, 206.0, 307.0, 408.0, 109.0, 210.0, 311.0, 412.0,]
    );
}

#[test]
fn test_broadcast_sub_left_size1() {
    // [1,3] - [2,3] -> [2,3]
    let a = DynTensor::from_vec(vec![100.0, 200.0, 300.0], &[1, 3], &cpu()).unwrap();
    let b = t2d(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3);
    let c = a.sub(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![99.0, 198.0, 297.0, 90.0, 180.0, 270.0]
    );
}

// ============================================================================
// Scalar broadcast: [3,4] + []
// ============================================================================

#[test]
fn test_broadcast_add_scalar_to_2d() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let scalar = DynTensor::full(&[], 10.0, DType::F32, &cpu()).unwrap();
    let c = a.add(&scalar).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
}

#[test]
fn test_broadcast_mul_scalar_to_3d() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let scalar = DynTensor::full(&[], 3.0, DType::F32, &cpu()).unwrap();
    let c = a.mul(&scalar).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
    );
}

#[test]
fn test_broadcast_sub_scalar_from_1d() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let scalar = DynTensor::full(&[], 5.0, DType::F32, &cpu()).unwrap();
    let c = a.sub(&scalar).unwrap();
    assert_eq!(c.dims(), &[3]);
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 15.0, 25.0]);
}

#[test]
fn test_broadcast_scalar_to_scalar() {
    // [] + [] -> []
    let a = DynTensor::full(&[], 3.0, DType::F32, &cpu()).unwrap();
    let b = DynTensor::full(&[], 7.0, DType::F32, &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 1);
    assert!(approx_eq(vals[0], 10.0, 1e-6));
}

// ============================================================================
// Multi-dim broadcast: [1,3,1] + [2,1,4]
// ============================================================================

#[test]
fn test_broadcast_add_multi_dim() {
    // [1,3,1] + [2,1,4] -> [2,3,4]
    let a = tnd(&[1.0, 2.0, 3.0], &[1, 3, 1]);
    let b = tnd(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        &[2, 1, 4],
    );
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3, 4]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    // Verify a few specific positions:
    // c[0,0,:] = a[0,0,0] + b[0,0,:] = 1 + [10,20,30,40] = [11,21,31,41]
    assert_eq!(&vals[0..4], &[11.0, 21.0, 31.0, 41.0]);
    // c[0,1,:] = a[0,1,0] + b[0,0,:] = 2 + [10,20,30,40] = [12,22,32,42]
    assert_eq!(&vals[4..8], &[12.0, 22.0, 32.0, 42.0]);
    // c[0,2,:] = a[0,2,0] + b[0,0,:] = 3 + [10,20,30,40] = [13,23,33,43]
    assert_eq!(&vals[8..12], &[13.0, 23.0, 33.0, 43.0]);
    // c[1,0,:] = a[0,0,0] + b[1,0,:] = 1 + [50,60,70,80] = [51,61,71,81]
    assert_eq!(&vals[12..16], &[51.0, 61.0, 71.0, 81.0]);
}

#[test]
fn test_broadcast_mul_multi_dim() {
    // [2,1] * [1,3] -> [2,3]
    let a = DynTensor::from_vec(vec![2.0, 3.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![20.0, 40.0, 60.0, 30.0, 60.0, 90.0]
    );
}

// ============================================================================
// Incompatible shapes error
// ============================================================================

#[test]
fn test_broadcast_add_incompatible_trailing() {
    // [3,4] + [5] -> error: trailing dims 4 != 5
    let a = t2d(&[0.0; 12], 3, 4);
    let b = t1d(&[0.0; 5]);
    let result = a.add(&b);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_sub_incompatible() {
    let a = t2d(&[0.0; 6], 2, 3);
    let b = t1d(&[0.0; 2]);
    let result = a.sub(&b);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_mul_incompatible_both_nonone() {
    // [3,4] + [3,5] -> error: last dims 4 != 5
    let a = t2d(&[0.0; 12], 3, 4);
    let b = t2d(&[0.0; 15], 3, 5);
    let result = a.mul(&b);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_div_incompatible() {
    // [2,3,4] + [2,5,4] -> error: middle dim 3 != 5
    let a = tnd(&[1.0; 24], &[2, 3, 4]);
    let b = tnd(&[1.0; 40], &[2, 5, 4]);
    let result = a.div(&b);
    assert!(result.is_err());
}

// ============================================================================
// High-rank broadcast
// ============================================================================

#[test]
fn test_broadcast_add_high_rank_5d() {
    // [2,1,3,1,5] + [1,7,1,4,1] -> [2,7,3,4,5]
    // Shape test only (values would be large), but verify shape is correct.
    let a_numel = (2 * 3) * 5;
    let b_numel = 7 * 4;
    let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..b_numel).map(|i| (i as f32) * 0.1).collect();
    let a = tnd(&a_data, &[2, 1, 3, 1, 5]);
    let b = tnd(&b_data, &[1, 7, 1, 4, 1]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 7, 3, 4, 5]);
    assert_eq!(c.numel(), 2 * 7 * 3 * 4 * 5);
}

#[test]
fn test_broadcast_mul_high_rank_5d_values() {
    // [1,1,1,1,3] * [2,2,2,2,1] -> [2,2,2,2,3]
    let a = tnd(&[1.0, 2.0, 3.0], &[1, 1, 1, 1, 3]);
    let b_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
    let b = tnd(&b_data, &[2, 2, 2, 2, 1]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2, 2, 3]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // c[0,0,0,0,:] = [1,2,3] * 1 = [1,2,3]
    assert_eq!(&vals[0..3], &[1.0, 2.0, 3.0]);
    // c[0,0,0,1,:] = [1,2,3] * 2 = [2,4,6]
    assert_eq!(&vals[3..6], &[2.0, 4.0, 6.0]);
    // c[0,0,1,0,:] = [1,2,3] * 3 = [3,6,9]
    assert_eq!(&vals[6..9], &[3.0, 6.0, 9.0]);
}

// ============================================================================
// Broadcast with batch dim: [B,C,H,W] patterns
// ============================================================================

#[test]
fn test_broadcast_bias_per_channel_4d() {
    // Simulates adding per-channel bias to a [B,C,H,W] tensor.
    // Bias shape [1,C,1,1], data shape [B,C,H,W].
    let bias = tnd(&[10.0, 20.0], &[1, 2, 1, 1]);
    let data = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]);
    let result = data.add(&bias).unwrap();
    assert_eq!(result.dims(), &[1, 2, 2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Channel 0 elements get +10: [11, 12, 13, 14]
    assert_eq!(&vals[0..4], &[11.0, 12.0, 13.0, 14.0]);
    // Channel 1 elements get +20: [25, 26, 27, 28]
    assert_eq!(&vals[4..8], &[25.0, 26.0, 27.0, 28.0]);
}

#[test]
fn test_broadcast_scale_per_channel_4d() {
    // Scale per-channel: [1,C,1,1] * [B,C,H,W]
    let scale = tnd(&[2.0, 0.5], &[1, 2, 1, 1]);
    let data = tnd(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[1, 2, 2, 2]);
    let result = data.mul(&scale).unwrap();
    assert_eq!(result.dims(), &[1, 2, 2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Channel 0 scaled by 2: [2, 4, 6, 8]
    assert_eq!(&vals[0..4], &[2.0, 4.0, 6.0, 8.0]);
    // Channel 1 scaled by 0.5: [5, 10, 15, 20]
    assert_eq!(&vals[4..8], &[5.0, 10.0, 15.0, 20.0]);
}

#[test]
fn test_broadcast_batch_dim_3d() {
    // [B,M,N] + [1,M,N] -> [B,M,N]
    // Simulates adding a shared bias matrix to every batch element.
    let bias = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let data = tnd(
        &[
            10.0, 20.0, 30.0, 40.0, // batch 0
            50.0, 60.0, 70.0, 80.0, // batch 1
        ],
        &[2, 2, 2],
    );
    let result = data.add(&bias).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![11.0, 22.0, 33.0, 44.0, 51.0, 62.0, 73.0, 84.0]);
}

#[test]
fn test_broadcast_multi_batch_4d() {
    // [B,H,M,N] + [1,1,1,N] -> [B,H,M,N]
    // Per-column broadcast across all batch and head dims.
    let col_bias = tnd(&[100.0, 200.0], &[1, 1, 1, 2]);
    let data = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 2]);
    let result = data.add(&col_bias).unwrap();
    assert_eq!(result.dims(), &[2, 1, 2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Each pair gets [+100, +200]
    assert_eq!(
        vals,
        vec![101.0, 202.0, 103.0, 204.0, 105.0, 206.0, 107.0, 208.0]
    );
}

// ============================================================================
// Broadcast preserves values (not just shape)
// ============================================================================

#[test]
fn test_broadcast_add_values_correctness() {
    // Manually compute expected values for [2,1] + [1,3] -> [2,3]
    let a = DynTensor::from_vec(vec![10.0, 20.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: 10 + [1,2,3] = [11, 12, 13]
    // Row 1: 20 + [1,2,3] = [21, 22, 23]
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
    );
}

#[test]
fn test_broadcast_sub_values_correctness() {
    // [2,1] - [1,3] -> [2,3]
    let a = DynTensor::from_vec(vec![10.0, 20.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let c = a.sub(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![9.0, 8.0, 7.0, 19.0, 18.0, 17.0]
    );
}

#[test]
fn test_broadcast_mul_values_correctness() {
    // [3,1] * [1,2] -> [3,2]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 100.0], &[1, 2], &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![10.0, 100.0, 20.0, 200.0, 30.0, 300.0]
    );
}

#[test]
fn test_broadcast_div_values_correctness() {
    // [2,1] / [1,3] -> [2,3]
    let a = DynTensor::from_vec(vec![12.0, 24.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![2.0, 3.0, 4.0], &[1, 3], &cpu()).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![6.0, 4.0, 3.0, 12.0, 8.0, 6.0]
    );
}

// ============================================================================
// Maximum / Minimum broadcast
// ============================================================================

#[test]
fn test_broadcast_maximum_values() {
    // [2,1] max [1,3] -> [2,3]
    let a = DynTensor::from_vec(vec![5.0, 2.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 4.0, 6.0], &[1, 3], &cpu()).unwrap();
    let c = a.maximum(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: max(5, [1,4,6]) = [5, 5, 6]
    // Row 1: max(2, [1,4,6]) = [2, 4, 6]
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 5.0, 6.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn test_broadcast_minimum_values() {
    // [2,1] min [1,3] -> [2,3]
    let a = DynTensor::from_vec(vec![5.0, 2.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 4.0, 6.0], &[1, 3], &cpu()).unwrap();
    let c = a.minimum(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: min(5, [1,4,6]) = [1, 4, 5]
    // Row 1: min(2, [1,4,6]) = [1, 2, 2]
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 5.0, 1.0, 2.0, 2.0]
    );
}

#[test]
fn test_broadcast_maximum_scalar() {
    // ReLU-like: max(tensor, 0)
    let a = t1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let zero = DynTensor::full(&[], 0.0, DType::F32, &cpu()).unwrap();
    let c = a.maximum(&zero).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_broadcast_minimum_scalar() {
    // Clamp upper: min(tensor, 1)
    let a = t1d(&[-2.0, 0.0, 0.5, 1.0, 3.0]);
    let one = DynTensor::full(&[], 1.0, DType::F32, &cpu()).unwrap();
    let c = a.minimum(&one).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.to_vec1::<f32>().unwrap(), vec![-2.0, 0.0, 0.5, 1.0, 1.0]);
}

// ============================================================================
// Per-channel broadcast_left vs broadcast_add for [C] + [C,T] patterns
// ============================================================================

#[test]
fn test_per_channel_add_via_reshape() {
    // The correct way to add per-channel bias [C] to a [C,T] tensor:
    // Reshape [C] to [C,1], then broadcast add with [C,T].
    let bias = t1d(&[10.0, 20.0, 30.0]); // [3]
    let data = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3); // [3,3]

    // Reshape bias from [C] to [C,1]
    let bias_reshaped = bias.reshape([3, 1]).unwrap();
    let result = data.add(&bias_reshaped).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Row 0: [1,2,3] + 10 = [11,12,13]
    // Row 1: [4,5,6] + 20 = [24,25,26]
    // Row 2: [7,8,9] + 30 = [37,38,39]
    assert_eq!(
        vals,
        vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0, 37.0, 38.0, 39.0]
    );
}

#[test]
fn test_per_channel_add_raw_1d_broadcasts_wrongly() {
    // WRONG pattern: [C] + [C,T] when C == T broadcasts as element-wise
    // because right-alignment matches C with T (the last dim).
    // This is a common bug source. When C==T it silently computes the
    // wrong thing (adds column-wise instead of row-wise).
    let bias = t1d(&[10.0, 20.0, 30.0]); // [3]
    let data = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3); // [3,3]

    // [3] broadcast against [3,3]: right-aligns [3] with T=3.
    // Each column gets a different bias value, NOT each row.
    let result = data.add(&bias).unwrap();
    assert_eq!(result.dims(), &[3, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Column-wise: col0 + 10, col1 + 20, col2 + 30
    assert_eq!(
        vals,
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0, 17.0, 28.0, 39.0]
    );
    // This is DIFFERENT from per-channel (row-wise) addition!
}

#[test]
fn test_per_channel_3d_via_reshape() {
    // Per-channel add for [B,C,T]: reshape [C] to [1,C,1] then broadcast.
    let bias = t1d(&[100.0, 200.0]); // [2] channels
    let data = tnd(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    ); // [B=2, C=2, T=3]

    // Reshape [C] to [1,C,1] for correct per-channel broadcast
    let bias_reshaped = bias.reshape([1, 2, 1]).unwrap();
    let result = data.add(&bias_reshaped).unwrap();
    assert_eq!(result.dims(), &[2, 2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Batch 0, Channel 0: [1,2,3] + 100 = [101,102,103]
    assert_eq!(&vals[0..3], &[101.0, 102.0, 103.0]);
    // Batch 0, Channel 1: [4,5,6] + 200 = [204,205,206]
    assert_eq!(&vals[3..6], &[204.0, 205.0, 206.0]);
    // Batch 1, Channel 0: [7,8,9] + 100 = [107,108,109]
    assert_eq!(&vals[6..9], &[107.0, 108.0, 109.0]);
    // Batch 1, Channel 1: [10,11,12] + 200 = [210,211,212]
    assert_eq!(&vals[9..12], &[210.0, 211.0, 212.0]);
}

#[test]
fn test_broadcast_left_then_add() {
    // Use broadcast_left to expand [C] to [B,T,C] then add.
    // broadcast_left prepends dims, keeping original dims at the right.
    let bias = t1d(&[10.0, 20.0, 30.0]); // [3]
    let expanded = bias.broadcast_left((2usize, 4usize)).unwrap(); // [2, 4, 3]
    assert_eq!(expanded.dims(), &[2, 4, 3]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    // Every slice of the last dim is [10, 20, 30]
    for chunk in vals.chunks(3) {
        assert_eq!(chunk, &[10.0, 20.0, 30.0]);
    }
}

// ============================================================================
// Strict ops reject broadcastable shapes
// ============================================================================

#[test]
fn test_strict_add_rejects_broadcastable() {
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert!(a.strict_add(&b).is_err());
}

#[test]
fn test_strict_mul_rejects_scalar_broadcast() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = DynTensor::full(&[], 2.0, DType::F32, &cpu()).unwrap();
    assert!(a.strict_mul(&b).is_err());
}

#[test]
fn test_strict_div_rejects_different_rank() {
    let a = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    let b = t1d(&[2.0, 5.0]);
    assert!(a.strict_div(&b).is_err());
}

// ============================================================================
// Operator overloads use broadcast
// ============================================================================

#[test]
fn test_add_operator_broadcasts() {
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let c = (&a + &b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );
}

#[test]
fn test_sub_operator_broadcasts() {
    let a = DynTensor::from_vec(vec![100.0, 200.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let c = (&a - &b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![99.0, 98.0, 97.0, 199.0, 198.0, 197.0]
    );
}

#[test]
fn test_mul_operator_broadcasts() {
    let a = DynTensor::from_vec(vec![2.0, 3.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 100.0], &[1, 2], &cpu()).unwrap();
    let c = (&a * &b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![20.0, 200.0, 30.0, 300.0]
    );
}

#[test]
fn test_div_operator_broadcasts() {
    let a = DynTensor::from_vec(vec![12.0, 24.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0, 4.0], &[1, 2], &cpu()).unwrap();
    let c = (&a / &b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![4.0, 3.0, 8.0, 6.0]);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_broadcast_with_size_zero_dim() {
    // [0,3] + [1,3] should produce [0,3] (valid NumPy broadcast, empty result)
    let a = DynTensor::from_vec(Vec::<f32>::new(), &[0, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[0, 3]);
    assert_eq!(c.numel(), 0);
}

#[test]
fn test_broadcast_output_shape_preserves_zero() {
    // Zero-length dims should broadcast correctly: max(0, 1) = 0 in NumPy.
    // Actually NumPy: [0,3] + [1,3] -> [0,3]. The 0 propagates.
    let out = broadcast_output_shape(&[0, 3], &[1, 3]).unwrap();
    assert_eq!(out, vec![0, 3]);
}

#[test]
fn test_broadcast_1d_to_3d() {
    // [5] + [2,3,5] -> [2,3,5]
    let a = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b_data: Vec<f32> = (0..30).map(|i| (i as f32) * 10.0).collect();
    let b = tnd(&b_data, &[2, 3, 5]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3, 5]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    // c[0,0,:] = [1,2,3,4,5] + [0,10,20,30,40] = [1,12,23,34,45]
    assert_eq!(&vals[0..5], &[1.0, 12.0, 23.0, 34.0, 45.0]);
}

#[test]
fn test_broadcast_commutativity_for_add() {
    // a + b should equal b + a for commutative ops.
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 20.0], &[1, 2], &cpu()).unwrap();
    let ab = a.add(&b).unwrap();
    let ba = b.add(&a).unwrap();
    assert_eq!(ab.dims(), ba.dims());
    assert_eq!(
        ab.to_flat_vec::<f32>().unwrap(),
        ba.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_broadcast_commutativity_for_mul() {
    let a = DynTensor::from_vec(vec![2.0, 3.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![10.0, 100.0, 1000.0], &[1, 3], &cpu()).unwrap();
    let ab = a.mul(&b).unwrap();
    let ba = b.mul(&a).unwrap();
    assert_eq!(ab.dims(), ba.dims());
    assert_eq!(
        ab.to_flat_vec::<f32>().unwrap(),
        ba.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_broadcast_non_commutativity_for_sub() {
    // a - b != b - a (subtraction is not commutative)
    let a = DynTensor::from_vec(vec![10.0, 20.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let ab = a.sub(&b).unwrap();
    let ba = b.sub(&a).unwrap();
    assert_eq!(ab.dims(), ba.dims());
    let ab_vals = ab.to_flat_vec::<f32>().unwrap();
    let ba_vals = ba.to_flat_vec::<f32>().unwrap();
    // ab = [9,8,7, 19,18,17], ba = [-9,-8,-7, -19,-18,-17]
    for (av, bv) in ab_vals.iter().zip(ba_vals.iter()) {
        assert!(approx_eq(*av, -bv, 1e-6));
    }
}
