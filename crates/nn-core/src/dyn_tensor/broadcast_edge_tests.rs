// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Broadcast edge-case tests for DynTensor operations.
//!
//! Complements `broadcast_tests.rs` with targeted edge cases:
//! - Scalar expand to arbitrary shapes
//! - [1] to [N] expand
//! - [1,1] to [M,N] expand
//! - Column broadcast [M,1] to [M,N] via expand
//! - Row broadcast [1,N] to [M,N] via expand
//! - 3D broadcast [1,1,N] to [B,M,N] via expand
//! - Incompatible expand failures
//! - Zero-dim expand edge cases
//! - Large rank broadcast
//! - Broadcast with negative values and special floats

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, tnd};
use crate::{DType, DynTensor};

// ============================================================================
// expand: scalar to any shape
// ============================================================================

#[test]
fn test_expand_scalar_to_1d() {
    let scalar = DynTensor::full(&[1], 42.0, DType::F32, &cpu()).unwrap();
    let expanded = scalar.expand([5]).unwrap();
    assert_eq!(expanded.dims(), &[5]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![42.0; 5]);
}

#[test]
fn test_expand_scalar_to_2d() {
    let scalar = DynTensor::from_vec(vec![7.0], &[1, 1], &cpu()).unwrap();
    let expanded = scalar.expand([3, 4]).unwrap();
    assert_eq!(expanded.dims(), &[3, 4]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 12);
    assert!(vals.iter().all(|&v| approx_eq(v, 7.0, 1e-7)));
}

#[test]
fn test_expand_scalar_to_3d() {
    let scalar = DynTensor::from_vec(vec![-1.5], &[1, 1, 1], &cpu()).unwrap();
    let expanded = scalar.expand([2, 3, 4]).unwrap();
    assert_eq!(expanded.dims(), &[2, 3, 4]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    assert!(vals.iter().all(|&v| approx_eq(v, -1.5, 1e-7)));
}

// ============================================================================
// expand: [1] to [N]
// ============================================================================

#[test]
fn test_expand_1_to_n() {
    let t = DynTensor::from_vec(vec![3.14], &[1], &cpu()).unwrap();
    let expanded = t.expand([100]).unwrap();
    assert_eq!(expanded.dims(), &[100]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 100);
    assert!(vals.iter().all(|&v| approx_eq(v, 3.14, 1e-6)));
}

#[test]
fn test_expand_1_to_1_identity() {
    let t = DynTensor::from_vec(vec![99.0], &[1], &cpu()).unwrap();
    let expanded = t.expand([1]).unwrap();
    assert_eq!(expanded.dims(), &[1]);
    assert_eq!(expanded.to_flat_vec::<f32>().unwrap(), vec![99.0]);
}

// ============================================================================
// expand: [1,1] to [M,N]
// ============================================================================

#[test]
fn test_expand_1x1_to_mxn() {
    let t = DynTensor::from_vec(vec![5.0], &[1, 1], &cpu()).unwrap();
    let expanded = t.expand([4, 6]).unwrap();
    assert_eq!(expanded.dims(), &[4, 6]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    assert!(vals.iter().all(|&v| approx_eq(v, 5.0, 1e-7)));
}

// ============================================================================
// expand: [M,1] to [M,N] (column broadcast)
// ============================================================================

#[test]
fn test_expand_column_broadcast() {
    // [3,1] -> [3,4]: each row's single value fills across columns
    let t = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[3, 1], &cpu()).unwrap();
    let expanded = t.expand([3, 4]).unwrap();
    assert_eq!(expanded.dims(), &[3, 4]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    // Row 0: [10, 10, 10, 10]
    assert_eq!(&vals[0..4], &[10.0, 10.0, 10.0, 10.0]);
    // Row 1: [20, 20, 20, 20]
    assert_eq!(&vals[4..8], &[20.0, 20.0, 20.0, 20.0]);
    // Row 2: [30, 30, 30, 30]
    assert_eq!(&vals[8..12], &[30.0, 30.0, 30.0, 30.0]);
}

// ============================================================================
// expand: [1,N] to [M,N] (row broadcast)
// ============================================================================

#[test]
fn test_expand_row_broadcast() {
    // [1,4] -> [3,4]: the single row is replicated M times
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();
    let expanded = t.expand([3, 4]).unwrap();
    assert_eq!(expanded.dims(), &[3, 4]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(&vals[0..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&vals[4..8], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&vals[8..12], &[1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// expand: 3D [1,1,N] to [B,M,N]
// ============================================================================

#[test]
fn test_expand_3d_last_dim_preserved() {
    // [1,1,3] -> [2,4,3]: the [3] vector is tiled across B*M positions
    let t = tnd(&[10.0, 20.0, 30.0], &[1, 1, 3]);
    let expanded = t.expand([2, 4, 3]).unwrap();
    assert_eq!(expanded.dims(), &[2, 4, 3]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    // Every group of 3 should be [10, 20, 30]
    for chunk in vals.chunks(3) {
        assert_eq!(chunk, &[10.0, 20.0, 30.0]);
    }
}

#[test]
fn test_expand_3d_mixed_dims() {
    // [2,1,3] -> [2,5,3]: expand only the middle dim
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let expanded = t.expand([2, 5, 3]).unwrap();
    assert_eq!(expanded.dims(), &[2, 5, 3]);
    let vals = expanded.to_flat_vec::<f32>().unwrap();
    // Batch 0: [1,2,3] repeated 5 times
    for i in 0..5 {
        assert_eq!(
            &vals[i * 3..(i + 1) * 3],
            &[1.0, 2.0, 3.0],
            "batch 0, row {i}"
        );
    }
    // Batch 1: [4,5,6] repeated 5 times
    for i in 0..5 {
        let offset = 15 + i * 3;
        assert_eq!(
            &vals[offset..offset + 3],
            &[4.0, 5.0, 6.0],
            "batch 1, row {i}"
        );
    }
}

// ============================================================================
// expand: incompatible shapes
// ============================================================================

#[test]
fn test_expand_rank_mismatch_fails() {
    // [2,3] cannot expand to [2,3,4] (rank changes)
    let t = DynTensor::from_vec(vec![1.0; 6], &[2, 3], &cpu()).unwrap();
    assert!(t.expand([2, 3, 4]).is_err());
}

#[test]
fn test_expand_non_one_dim_mismatch_fails() {
    // [2,3] cannot expand to [4,3] (dim 0 is 2, not 1)
    let t = DynTensor::from_vec(vec![1.0; 6], &[2, 3], &cpu()).unwrap();
    assert!(t.expand([4, 3]).is_err());
}

#[test]
fn test_expand_shrink_fails() {
    // [1,5] cannot "expand" to [1,3] (5 != 3 and 5 != 1)
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5], &cpu()).unwrap();
    assert!(t.expand([1, 3]).is_err());
}

// ============================================================================
// expand: zero-dim edge cases
// ============================================================================

#[test]
fn test_expand_to_zero_dim() {
    // [1,3] -> [0,3]: valid expand, produces empty tensor
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let expanded = t.expand([0, 3]).unwrap();
    assert_eq!(expanded.dims(), &[0, 3]);
    assert_eq!(expanded.numel(), 0);
}

#[test]
fn test_expand_from_zero_dim() {
    // [0,3] -> [0,3]: identity on zero-dim
    let t = DynTensor::from_vec(Vec::<f32>::new(), &[0, 3], &cpu()).unwrap();
    let expanded = t.expand([0, 3]).unwrap();
    assert_eq!(expanded.dims(), &[0, 3]);
    assert_eq!(expanded.numel(), 0);
}

// ============================================================================
// broadcast_add with special float values
// ============================================================================

#[test]
fn test_broadcast_add_negative_values() {
    // Ensure broadcast works correctly with negative values
    let a = DynTensor::from_vec(vec![-5.0, -10.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![-4.0, -3.0, -2.0, -9.0, -8.0, -7.0]
    );
}

#[test]
fn test_broadcast_mul_with_zeros() {
    // Multiplying by zero via broadcast should produce all zeros
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let zero = DynTensor::full(&[1, 1], 0.0, DType::F32, &cpu()).unwrap();
    let c = a.mul(&zero).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert!(c.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
}

// ============================================================================
// broadcast_add across many ranks (4D, 5D, 6D)
// ============================================================================

#[test]
fn test_broadcast_add_4d_single_channel_bias() {
    // [1,C,1,1] + [B,C,H,W] -> [B,C,H,W] is a common pattern in CNNs.
    // Test with C=2, B=1, H=2, W=2.
    let bias = tnd(&[100.0, 200.0], &[1, 2, 1, 1]);
    let data = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]);
    let result = data.add(&bias).unwrap();
    assert_eq!(result.dims(), &[1, 2, 2, 2]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    // Channel 0 gets +100
    assert_eq!(&vals[0..4], &[101.0, 102.0, 103.0, 104.0]);
    // Channel 1 gets +200
    assert_eq!(&vals[4..8], &[205.0, 206.0, 207.0, 208.0]);
}

#[test]
fn test_broadcast_add_6d_shape_only() {
    // Very high rank: [1,1,1,1,1,3] + [2,2,2,2,2,1] -> [2,2,2,2,2,3]
    let a = tnd(&[1.0, 2.0, 3.0], &[1, 1, 1, 1, 1, 3]);
    let b_data: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
    let b = tnd(&b_data, &[2, 2, 2, 2, 2, 1]);
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2, 2, 2, 3]);
    assert_eq!(c.numel(), 2 * 2 * 2 * 2 * 2 * 3);
    // Verify first element: a[0,0,0,0,0,0] + b[0,0,0,0,0,0] = 1 + 1 = 2
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(&vals[0..3], &[2.0, 3.0, 4.0]);
}

// ============================================================================
// expand preserves dtype
// ============================================================================

#[test]
fn test_expand_preserves_f32_dtype() {
    let t = DynTensor::full(&[1, 1], 3.0, DType::F32, &cpu()).unwrap();
    let expanded = t.expand([4, 5]).unwrap();
    assert_eq!(expanded.dtype(), DType::F32);
    assert_eq!(expanded.dims(), &[4, 5]);
}

#[test]
fn test_expand_preserves_bf16_dtype() {
    let t = DynTensor::full(&[1, 1], 2.0, DType::BF16, &cpu()).unwrap();
    let expanded = t.expand([3, 4]).unwrap();
    assert_eq!(expanded.dtype(), DType::BF16);
    assert_eq!(expanded.dims(), &[3, 4]);
}
