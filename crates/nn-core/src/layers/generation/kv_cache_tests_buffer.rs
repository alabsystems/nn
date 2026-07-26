#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for KvCacheLayer O(1) amortized buffer implementation (#1223).

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCacheLayer;
use crate::{DType, Device};

#[test]
fn test_buffer_capacity_grows_via_doubling() {
    let mut layer = KvCacheLayer::empty();
    // First append: initial capacity = max(16, 1) = 16.
    let k = DynTensor::ones(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 1);
    assert_eq!(layer.buffer_capacity(), 16);

    // Append 15 more to fill to capacity.
    for _ in 0..15 {
        layer.append(&k, &v).unwrap();
    }
    assert_eq!(layer.seq_len(), 16);
    assert_eq!(layer.buffer_capacity(), 16);

    // 17th append triggers doubling.
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 17);
    assert_eq!(layer.buffer_capacity(), 32);
}

#[test]
fn test_buffer_initial_capacity_matches_first_append() {
    let mut layer = KvCacheLayer::empty();
    // Append 20 positions at once — exceeds INITIAL_CAPACITY (16).
    let k = DynTensor::ones(&[1, 2, 20, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 20, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 20);
    assert_eq!(layer.buffer_capacity(), 20);
}

#[test]
fn test_buffer_data_correctness_across_growth() {
    let mut layer = KvCacheLayer::empty();
    // Append positions with distinct values to verify data integrity after growth.
    for i in 0..20 {
        let val = (i + 1) as f64;
        let k = DynTensor::full(&[1, 1, 1, 2], val, DType::F32, &Device::Cpu).unwrap();
        let v = DynTensor::full(&[1, 1, 1, 2], val * 10.0, DType::F32, &Device::Cpu).unwrap();
        let (full_k, full_v) = layer.append(&k, &v).unwrap();
        assert_eq!(full_k.dims(), &[1, 1, i + 1, 2]);
        assert_eq!(full_v.dims(), &[1, 1, i + 1, 2]);

        // Verify all previous positions are still correct.
        let k_data = full_k.to_flat_vec::<f32>().unwrap();
        let v_data = full_v.to_flat_vec::<f32>().unwrap();
        for j in 0..=i {
            let expected_k = (j + 1) as f32;
            let expected_v = expected_k * 10.0;
            assert!(
                (k_data[j * 2] - expected_k).abs() < 1e-6,
                "key[{j}] expected {expected_k}, got {}",
                k_data[j * 2]
            );
            assert!(
                (v_data[j * 2] - expected_v).abs() < 1e-6,
                "value[{j}] expected {expected_v}, got {}",
                v_data[j * 2]
            );
        }
    }
}

#[test]
fn test_buffer_growth_is_logarithmic() {
    // Verify O(log N) growth events for N single-token appends.
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::ones(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();

    let mut growth_events = 0;
    let mut prev_capacity = 0;
    for _ in 0..1000 {
        layer.append(&k, &v).unwrap();
        let cap = layer.buffer_capacity();
        if cap != prev_capacity {
            growth_events += 1;
            prev_capacity = cap;
        }
    }
    assert_eq!(layer.seq_len(), 1000);
    // log2(1000) ≈ 10, initial alloc + doublings.
    assert!(
        growth_events <= 12,
        "expected <= 12 growth events for 1000 appends, got {growth_events}"
    );
}

#[test]
fn test_buffer_reset_releases_memory() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::ones(&[1, 2, 10, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 2, 10, 4], DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.buffer_capacity(), 16);

    layer.reset();
    assert_eq!(layer.buffer_capacity(), 0);
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.is_empty());
    assert!(layer.key().unwrap().is_none());
    assert!(layer.value().unwrap().is_none());
}

#[test]
fn test_buffer_reuse_after_reset() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::full(&[1, 2, 5, 4], 1.0, DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::full(&[1, 2, 5, 4], 2.0, DType::F32, &Device::Cpu).unwrap();
    layer.append(&k, &v).unwrap();

    layer.reset();

    let k2 = DynTensor::full(&[1, 2, 3, 4], 9.0, DType::F32, &Device::Cpu).unwrap();
    let v2 = DynTensor::full(&[1, 2, 3, 4], 8.0, DType::F32, &Device::Cpu).unwrap();
    let (full_k, full_v) = layer.append(&k2, &v2).unwrap();
    assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
    let data = full_k.to_flat_vec::<f32>().unwrap();
    assert!(
        (data[0] - 9.0).abs() < 1e-6,
        "expected 9.0 after reset+reuse"
    );
    let vdata = full_v.to_flat_vec::<f32>().unwrap();
    assert!((vdata[0] - 8.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// DynTensor::slice_set tests
// ---------------------------------------------------------------------------

#[test]
fn test_slice_set_dim0() {
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let src = DynTensor::full(&[2, 3], 5.0, DType::F32, &Device::Cpu).unwrap();
    let result = dst.slice_set(0, 1, &src).unwrap();
    assert_eq!(result.dims(), &[4, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // Rows 0: zeros, rows 1-2: 5.0, row 3: zeros.
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[3] - 5.0).abs() < 1e-6); // row 1, col 0
    assert!((data[8] - 5.0).abs() < 1e-6); // row 2, col 2
    assert!((data[9] - 0.0).abs() < 1e-6); // row 3, col 0
}

#[test]
fn test_slice_set_dim2_4d() {
    // Simulate KV cache buffer write.
    let dst = DynTensor::zeros(&[1, 2, 8, 4], DType::F32, &Device::Cpu).unwrap();
    let src = DynTensor::full(&[1, 2, 3, 4], 7.0, DType::F32, &Device::Cpu).unwrap();
    let result = dst.slice_set(2, 2, &src).unwrap();
    assert_eq!(result.dims(), &[1, 2, 8, 4]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // Positions 0-1: zeros, positions 2-4: 7.0, positions 5-7: zeros.
    assert!((data[0] - 0.0).abs() < 1e-6); // [0,0,0,0]
    assert!((data[8] - 7.0).abs() < 1e-6); // [0,0,2,0]
    assert!((data[19] - 7.0).abs() < 1e-6); // [0,0,4,3]
    assert!((data[20] - 0.0).abs() < 1e-6); // [0,0,5,0]
}

#[test]
fn test_slice_set_rejects_shape_mismatch() {
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let src = DynTensor::full(&[2, 5], 1.0, DType::F32, &Device::Cpu).unwrap();
    assert!(dst.slice_set(0, 0, &src).is_err());
}

#[test]
fn test_slice_set_rejects_out_of_bounds() {
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let src = DynTensor::full(&[2, 3], 1.0, DType::F32, &Device::Cpu).unwrap();
    assert!(dst.slice_set(0, 3, &src).is_err()); // 3 + 2 > 4
}

#[test]
fn test_slice_set_rejects_rank_mismatch() {
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let src = DynTensor::full(&[2, 3, 1], 1.0, DType::F32, &Device::Cpu).unwrap();
    assert!(dst.slice_set(0, 0, &src).is_err());
}

#[test]
fn test_slice_set_rejects_dim_out_of_range() {
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let src = DynTensor::full(&[4, 1], 1.0, DType::F32, &Device::Cpu).unwrap();
    assert!(dst.slice_set(5, 0, &src).is_err());
}

#[test]
fn test_slice_set_inplace_unique_ref() {
    // When the DynTensor has a unique Arc reference (refcount=1),
    // slice_set should mutate in-place via Arc::try_unwrap (#1245).
    // Verify correctness: the result should be identical whether
    // the fast path (unique ref) or slow path (shared ref) is taken.
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    // dst has refcount=1 (no clones), so the fast path should fire.
    let src = DynTensor::full(&[2, 3], 9.0, DType::F32, &Device::Cpu).unwrap();
    let result = dst.slice_set(0, 1, &src).unwrap();
    let data = result.to_flat_vec::<f32>().unwrap();
    // Row 0: zeros, rows 1-2: 9.0, row 3: zeros.
    assert!((data[0]).abs() < 1e-6); // [0,0]
    assert!((data[3] - 9.0).abs() < 1e-6); // [1,0]
    assert!((data[8] - 9.0).abs() < 1e-6); // [2,2]
    assert!((data[9]).abs() < 1e-6); // [3,0]
}

#[test]
fn test_slice_set_shared_ref_clones() {
    // When there's a shared reference (clone held), slice_set falls back
    // to cloning the array. Verify the original is NOT mutated.
    let dst = DynTensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let _clone = dst.clone(); // Bumps Arc refcount to 2.
    let src = DynTensor::full(&[2, 3], 9.0, DType::F32, &Device::Cpu).unwrap();
    let result = dst.slice_set(0, 1, &src).unwrap();
    let result_data = result.to_flat_vec::<f32>().unwrap();
    assert!((result_data[3] - 9.0).abs() < 1e-6); // [1,0] modified
                                                  // The clone should still be all zeros (not mutated).
    let clone_data = _clone.to_flat_vec::<f32>().unwrap();
    assert!((clone_data[3]).abs() < 1e-6); // [1,0] still zero
}

// ---------------------------------------------------------------------------
// Performance proofs: zero-copy narrow on return path
// ---------------------------------------------------------------------------

/// Prove that KvCacheLayer::append() return values are zero-copy ArcArray views.
///
/// Before the ArcArray fix (#1397), `narrow()` called `.to_owned()` on every
/// return, creating O(S²) total data movement over S decode steps. Now
/// `narrow()` uses `ArcArray::slice_axis_move()` which adjusts offset/strides
/// without copying any elements.
///
/// This test verifies zero-copy by checking that the returned tensor's data
/// pointer lies within the buffer's allocation (shared backing via ArcArray).
#[test]
fn test_kv_cache_append_return_is_zero_copy() {
    let mut layer = KvCacheLayer::empty();
    let steps = 50;
    let head_dim = 4;
    let num_heads = 2;
    let batch = 1;
    let per_position_elements = batch * num_heads * head_dim;

    for step in 0..steps {
        let kv = DynTensor::from_vec(
            vec![step as f32; per_position_elements],
            &[batch, num_heads, 1, head_dim],
            &Device::Cpu,
        )
        .unwrap();
        let (full_k, full_v) = layer.append(&kv, &kv).unwrap();

        // Verify shape correctness.
        assert_eq!(full_k.dim(2).unwrap(), step + 1);
        assert_eq!(full_v.dim(2).unwrap(), step + 1);

        // Zero-copy proof: the returned narrow view's data pointer must lie
        // within the buffer's data range. This is only possible if narrow()
        // returns a view into the buffer, not a copy.
        let buf_k = layer.key().unwrap().unwrap();
        let buf_view = buf_k.as_cpu_f32().unwrap();
        let ret_view = full_k.as_cpu_f32().unwrap();

        let buf_ptr = buf_view.as_ptr() as usize;
        let buf_end = buf_ptr + buf_view.len() * size_of::<f32>();
        let ret_ptr = ret_view.as_ptr() as usize;

        assert!(
            ret_ptr >= buf_ptr && ret_ptr < buf_end,
            "step {step}: narrow() returned a copy (ptr {ret_ptr:#x}) outside buffer \
             range [{buf_ptr:#x}..{buf_end:#x}). Expected zero-copy ArcArray view."
        );
    }
}

/// Verify O(log S) buffer growth events and that narrow views remain valid
/// across growth boundaries (ArcArray keeps a reference to the old allocation
/// even after the buffer is reallocated).
#[test]
fn test_kv_cache_growth_events_logarithmic_and_views_survive_realloc() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::ones(&[1, 1, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let v = DynTensor::ones(&[1, 1, 1, 4], DType::F32, &Device::Cpu).unwrap();

    let steps = 256;
    let mut growth_events = 0;
    let mut prev_cap = 0;
    let mut views_before_growth: Vec<DynTensor> = Vec::new();

    for i in 0..steps {
        let cap_before = layer.buffer_capacity();
        let (full_k, _) = layer.append(&k, &v).unwrap();
        let cap_after = layer.buffer_capacity();
        if cap_after != prev_cap {
            growth_events += 1;
            prev_cap = cap_after;
        }

        // If the buffer just grew (reallocated), verify that views captured
        // before growth still have valid data (ArcArray reference counting
        // keeps the old allocation alive).
        if cap_after > cap_before && !views_before_growth.is_empty() {
            for old_view in &views_before_growth {
                let old_data = old_view.to_flat_vec::<f32>().unwrap();
                // All values should be 1.0 (from DynTensor::ones).
                for &val in &old_data {
                    assert!(
                        (val - 1.0).abs() < 1e-6,
                        "old view data corrupted after buffer growth at step {i}"
                    );
                }
            }
            views_before_growth.clear();
        }

        // Save some views to check after the next growth event.
        if i % 5 == 0 {
            views_before_growth.push(full_k);
        }

        assert_eq!(layer.seq_len(), i + 1);
    }

    // Growth events should be O(log S).
    assert!(
        growth_events <= 10,
        "doubling buffer should have O(log S) growth events, got {growth_events}"
    );
}
