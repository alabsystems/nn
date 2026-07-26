// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for advanced reshape operations: repeat_interleave_n, tile_numpy,
//! expand, and unfold.
//!
//! repeat_interleave_n and tile_numpy are defined in `shape/reshape_advanced.rs`.
//! expand is defined in `selection/mod.rs`.
//! unfold is defined in `shape/shape_unfold.rs`.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d, tnd};
use crate::dyn_tensor::DynTensor;
use crate::DType;

fn flat_f32(t: &DynTensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

// =============================================================================
// repeat_interleave_n
// =============================================================================

#[test]
fn test_repeat_interleave_n_basic() {
    // [1,2,3] with repeats=2, dim=0 -> [1,1,2,2,3,3]
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.repeat_interleave_n(2, 0).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(flat_f32(&r), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
}

#[test]
fn test_repeat_interleave_n_single_element() {
    let t = t1d(&[42.0]);
    let r = t.repeat_interleave_n(5, 0).unwrap();
    assert_eq!(r.dims(), &[5]);
    assert_eq!(flat_f32(&r), vec![42.0, 42.0, 42.0, 42.0, 42.0]);
}

#[test]
fn test_repeat_interleave_n_preserves_dtype() {
    let t = DynTensor::zeros(&[3], DType::F32, &cpu()).unwrap();
    let r = t.repeat_interleave_n(2, 0).unwrap();
    assert_eq!(r.dtype(), DType::F32);
}

#[test]
fn test_repeat_interleave_n_large_repeat() {
    let t = t1d(&[1.0, 2.0]);
    let r = t.repeat_interleave_n(100, 0).unwrap();
    assert_eq!(r.dims(), &[200]);
    // First 100 should be 1.0, next 100 should be 2.0
    let vals = flat_f32(&r);
    assert!(vals[0..100].iter().all(|&v| v == 1.0));
    assert!(vals[100..200].iter().all(|&v| v == 2.0));
}

#[test]
fn test_repeat_interleave_n_middle_dim() {
    // [2, 3, 4] tensor, repeat along dim=1 with 2 repeats
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.repeat_interleave_n(2, 1).unwrap();
    assert_eq!(r.dims(), &[2, 6, 4]);
    assert_eq!(r.numel(), 48);
}

// =============================================================================
// tile_numpy
// =============================================================================

#[test]
fn test_tile_numpy_basic() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.tile_numpy(&[2]).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(flat_f32(&r), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_tile_numpy_2d_single_rep() {
    // reps=[3] on [2,2] → padded to [1,3]
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.tile_numpy(&[3]).unwrap();
    assert_eq!(r.dims(), &[2, 6]);
}

#[test]
fn test_tile_numpy_preserves_dtype() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let r = t.tile_numpy(&[2, 2]).unwrap();
    assert_eq!(r.dtype(), DType::F32);
    assert_eq!(r.dims(), &[4, 6]);
}

#[test]
fn test_tile_numpy_zero_rep() {
    // Zero in reps → empty tensor
    let t = t1d(&[1.0, 2.0]);
    let r = t.tile_numpy(&[0]).unwrap();
    assert_eq!(r.dims(), &[0]);
    assert_eq!(r.numel(), 0);
}

// =============================================================================
// expand (existing, verified here for coverage)
// =============================================================================

#[test]
fn test_expand_basic() {
    // [1, 3] -> [4, 3]
    let t = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = t.expand([4, 3]).unwrap();
    assert_eq!(r.dims(), &[4, 3]);
    let vals = flat_f32(&r);
    // Each row should be [1, 2, 3]
    for row in vals.chunks(3) {
        assert_eq!(row, &[1.0, 2.0, 3.0]);
    }
}

#[test]
fn test_expand_no_copy_semantics() {
    // Expanding size-1 dims to larger sizes
    let t = tnd(&[5.0], &[1, 1, 1]);
    let r = t.expand([3, 4, 5]).unwrap();
    assert_eq!(r.dims(), &[3, 4, 5]);
    assert_eq!(r.numel(), 60);
    assert!(flat_f32(&r).iter().all(|&v| v == 5.0));
}

#[test]
fn test_expand_same_dims_is_identity() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.expand([2, 2]).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(flat_f32(&r), flat_f32(&t));
}

#[test]
fn test_expand_rejects_non_singleton() {
    // Can't expand dim of size 2 to 4
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(t.expand([4, 2]).is_err());
}

#[test]
fn test_expand_rejects_rank_mismatch() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.expand([2, 3]).is_err());
}

#[test]
fn test_expand_3d() {
    // [1, 1, 4] -> [2, 3, 4]
    let t = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let r = t.expand([2, 3, 4]).unwrap();
    assert_eq!(r.dims(), &[2, 3, 4]);
    let vals = flat_f32(&r);
    assert_eq!(vals.len(), 24);
    // Every group of 4 should be [1,2,3,4]
    for chunk in vals.chunks(4) {
        assert_eq!(chunk, &[1.0, 2.0, 3.0, 4.0]);
    }
}

// =============================================================================
// unfold (existing, verified here for coverage)
// =============================================================================

#[test]
fn test_unfold_basic() {
    // [1,2,3,4,5] unfold(dim=0, size=3, step=1) -> [3, 3]
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let r = t.unfold(0, 3, 1).unwrap();
    assert_eq!(r.dims(), &[3, 3]);
    let vals = flat_f32(&r);
    // window0=[1,2,3], window1=[2,3,4], window2=[3,4,5]
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_unfold_non_overlapping() {
    // [1,2,3,4,5,6] unfold(dim=0, size=2, step=2) -> [3, 2]
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = t.unfold(0, 2, 2).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_eq!(flat_f32(&r), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_unfold_entire_dim() {
    // [4] unfold(dim=0, size=4, step=1) -> [1, 4]
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let r = t.unfold(0, 4, 1).unwrap();
    assert_eq!(r.dims(), &[1, 4]);
    assert_eq!(flat_f32(&r), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_unfold_rejects_zero_size() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.unfold(0, 0, 1).is_err());
}

#[test]
fn test_unfold_rejects_zero_step() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.unfold(0, 1, 0).is_err());
}

#[test]
fn test_unfold_rejects_size_exceeds_dim() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.unfold(0, 3, 1).is_err());
}

#[test]
fn test_unfold_2d() {
    // [[1,2,3,4],[5,6,7,8]] unfold(dim=1, size=2, step=1) -> [2, 3, 2]
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
    let r = t.unfold(1, 2, 1).unwrap();
    assert_eq!(r.dims(), &[2, 3, 2]);
    let vals = flat_f32(&r);
    // Row 0: windows [1,2], [2,3], [3,4]
    // Row 1: windows [5,6], [6,7], [7,8]
    assert_eq!(
        vals,
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0]
    );
}

// =============================================================================
// Integration: combining advanced reshape ops
// =============================================================================

#[test]
fn test_repeat_interleave_then_reshape() {
    let t = t1d(&[1.0, 2.0]);
    let r = t.repeat_interleave_n(3, 0).unwrap();
    assert_eq!(r.dims(), &[6]);
    let reshaped = r.reshape([2, 3]).unwrap();
    assert_eq!(reshaped.dims(), &[2, 3]);
    assert_eq!(flat_f32(&reshaped), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
}

#[test]
fn test_expand_then_unfold() {
    // [1, 6] -> expand to [2, 6] -> unfold(dim=1, size=3, step=3)
    let t = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    let expanded = t.expand([2, 6]).unwrap();
    assert_eq!(expanded.dims(), &[2, 6]);
    let unfolded = expanded.unfold(1, 3, 3).unwrap();
    assert_eq!(unfolded.dims(), &[2, 2, 3]);
}

#[test]
fn test_tile_then_narrow() {
    let t = t1d(&[1.0, 2.0]);
    let tiled = t.tile_numpy(&[3]).unwrap();
    assert_eq!(tiled.dims(), &[6]);
    let narrowed = tiled.narrow(0, 1, 4).unwrap();
    assert_eq!(narrowed.dims(), &[4]);
    assert_eq!(flat_f32(&narrowed), vec![2.0, 1.0, 2.0, 1.0]);
}
