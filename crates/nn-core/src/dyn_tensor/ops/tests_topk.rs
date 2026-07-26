// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for topk_ext, argmax_ext, and argmin_ext DynTensor operations.

use crate::dyn_tensor::test_helpers::tnd;
use crate::dyn_tensor::{DynTensor, D};
use crate::{DType, Device};

// =============================================================================
// topk_ext tests
// =============================================================================

#[test]
fn test_topk_ext_k1_largest() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
    let (vals, idxs) = x.topk_ext(1, 0, true, true).unwrap();
    assert_eq!(vals.dims(), &[1]);
    assert_eq!(vals.to_flat_vec::<f32>().unwrap(), vec![5.0]);
    assert_eq!(idxs.to_flat_vec::<u32>().unwrap(), vec![4]);
}

#[test]
fn test_topk_ext_k3_largest_sorted() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[8]);
    let (vals, idxs) = x.topk_ext(3, 0, true, true).unwrap();
    assert_eq!(vals.dims(), &[3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Should be sorted descending: 9, 6, 5
    assert_eq!(v, vec![9.0, 6.0, 5.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![5, 7, 4]);
}

#[test]
fn test_topk_ext_k5_largest_sorted() {
    let x = tnd(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6]);
    let (vals, _) = x.topk_ext(5, 0, true, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![60.0, 50.0, 40.0, 30.0, 20.0]);
}

#[test]
fn test_topk_ext_k3_smallest_sorted() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0], &[8]);
    let (vals, idxs) = x.topk_ext(3, 0, false, true).unwrap();
    assert_eq!(vals.dims(), &[3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Should be sorted ascending: 1, 1.5, 2
    assert_eq!(v, vec![1.0, 1.5, 2.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![1, 3, 6]);
}

#[test]
fn test_topk_ext_k3_largest_unsorted() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[8]);
    let (vals, idxs) = x.topk_ext(3, 0, true, false).unwrap();
    assert_eq!(vals.dims(), &[3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // Values should be the top-3 (9, 6, 5) but in unspecified order.
    let mut sorted_v = v.clone();
    sorted_v.sort_by(|a, b| b.total_cmp(a));
    assert_eq!(sorted_v, vec![9.0, 6.0, 5.0]);
    // Each index should match its value.
    let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    for (val, idx) in v.iter().zip(i.iter()) {
        assert_eq!(
            *val, data[*idx as usize],
            "index {idx} should map to value {val}"
        );
    }
}

#[test]
fn test_topk_ext_k3_smallest_unsorted() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0], &[8]);
    let (vals, idxs) = x.topk_ext(3, 0, false, false).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // Values should be the bottom-3 (1, 1.5, 2) but in unspecified order.
    let mut sorted_v = v.clone();
    sorted_v.sort_by(f32::total_cmp);
    assert_eq!(sorted_v, vec![1.0, 1.5, 2.0]);
    // Each index should match its value.
    let data = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
    for (val, idx) in v.iter().zip(i.iter()) {
        assert_eq!(
            *val, data[*idx as usize],
            "index {idx} should map to value {val}"
        );
    }
}

#[test]
fn test_topk_ext_2d_last_dim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let (vals, _idxs) = x.topk_ext(2, 1, true, true).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [1,5,3] -> top-2 largest sorted: [5,3]
    assert_eq!(v[0], 5.0);
    assert_eq!(v[1], 3.0);
    // Row 1: [4,2,6] -> top-2 largest sorted: [6,4]
    assert_eq!(v[2], 6.0);
    assert_eq!(v[3], 4.0);
}

#[test]
fn test_topk_ext_2d_smallest() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let (vals, _) = x.topk_ext(2, 1, false, true).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [1,5,3] -> bottom-2 sorted: [1,3]
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 3.0);
    // Row 1: [4,2,6] -> bottom-2 sorted: [2,4]
    assert_eq!(v[2], 2.0);
    assert_eq!(v[3], 4.0);
}

// -- Edge cases ---------------------------------------------------------------

#[test]
fn test_topk_ext_k_exceeds_dim_errors() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let result = x.topk_ext(4, 0, true, true);
    assert!(result.is_err(), "k > dim size should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("out of range"),
        "error should mention out of range: {err}"
    );
}

#[test]
fn test_topk_ext_k_zero_errors() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.topk_ext(0, 0, true, true).is_err(), "k=0 should error");
}

#[test]
fn test_topk_ext_single_element() {
    let x = tnd(&[42.0], &[1]);
    let (vals, idxs) = x.topk_ext(1, 0, true, true).unwrap();
    assert_eq!(vals.to_flat_vec::<f32>().unwrap(), vec![42.0]);
    assert_eq!(idxs.to_flat_vec::<u32>().unwrap(), vec![0]);
}

#[test]
fn test_topk_ext_ties() {
    // All values equal — any k elements are valid.
    let x = tnd(&[7.0, 7.0, 7.0, 7.0], &[4]);
    let (vals, idxs) = x.topk_ext(2, 0, true, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![7.0, 7.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // Both indices should be valid (0..4) and distinct.
    assert!(i[0] < 4 && i[1] < 4, "indices should be in range");
}

#[test]
fn test_topk_ext_nan_errors() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    assert!(
        x.topk_ext(2, 0, true, true).is_err(),
        "NaN input should error"
    );
}

#[test]
fn test_topk_ext_k_equals_dim() {
    let x = tnd(&[3.0, 1.0, 2.0], &[3]);
    let (vals, _) = x.topk_ext(3, 0, true, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_topk_ext_k_equals_dim_smallest() {
    let x = tnd(&[3.0, 1.0, 2.0], &[3]);
    let (vals, _) = x.topk_ext(3, 0, false, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

// =============================================================================
// argmax_ext tests
// =============================================================================

#[test]
fn test_argmax_ext_1d() {
    let x = tnd(&[1.0, 5.0, 3.0, 2.0, 4.0], &[5]);
    let result = x.argmax_ext(0, false).unwrap();
    assert!(
        result.dims().is_empty(),
        "1D argmax without keepdim should be scalar"
    );
    assert_eq!(result.dtype(), DType::U32);
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1]); // index of 5.0
}

#[test]
fn test_argmax_ext_1d_keepdim() {
    let x = tnd(&[1.0, 5.0, 3.0, 2.0, 4.0], &[5]);
    let result = x.argmax_ext(0, true).unwrap();
    assert_eq!(result.dims(), &[1]); // keepdim preserves rank
    assert_eq!(result.dtype(), DType::U32);
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1]);
}

#[test]
fn test_argmax_ext_2d_dim0() {
    // [[1, 5, 3],
    //  [4, 2, 6]]
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmax_ext(0, false).unwrap();
    assert_eq!(result.dims(), &[3]); // reduced dim 0
    let v = result.to_flat_vec::<u32>().unwrap();
    // col 0: max(1,4)=4 at row 1; col 1: max(5,2)=5 at row 0; col 2: max(3,6)=6 at row 1
    assert_eq!(v, vec![1, 0, 1]);
}

#[test]
fn test_argmax_ext_2d_dim1() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmax_ext(1, false).unwrap();
    assert_eq!(result.dims(), &[2]); // reduced dim 1
    let v = result.to_flat_vec::<u32>().unwrap();
    // row 0: max(1,5,3)=5 at col 1; row 1: max(4,2,6)=6 at col 2
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn test_argmax_ext_2d_dim0_keepdim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmax_ext(0, true).unwrap();
    assert_eq!(result.dims(), &[1, 3]); // keepdim
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1, 0, 1]);
}

#[test]
fn test_argmax_ext_3d() {
    // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  shape [2, 2, 2]
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    // argmax along dim=0 (batch): max between batch 0 and batch 1
    let result = x.argmax_ext(0, false).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<u32>().unwrap();
    // All maxima are in batch 1
    assert_eq!(v, vec![1, 1, 1, 1]);
}

// =============================================================================
// argmin_ext tests
// =============================================================================

#[test]
fn test_argmin_ext_1d() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.5, 5.0], &[5]);
    let result = x.argmin_ext(0, false).unwrap();
    assert!(
        result.dims().is_empty(),
        "1D argmin without keepdim should be scalar"
    );
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1]); // index of 1.0
}

#[test]
fn test_argmin_ext_1d_keepdim() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.5, 5.0], &[5]);
    let result = x.argmin_ext(0, true).unwrap();
    assert_eq!(result.dims(), &[1]);
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1]);
}

#[test]
fn test_argmin_ext_2d_dim0() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmin_ext(0, false).unwrap();
    assert_eq!(result.dims(), &[3]);
    let v = result.to_flat_vec::<u32>().unwrap();
    // col 0: min(1,4)=1 at row 0; col 1: min(5,2)=2 at row 1; col 2: min(3,6)=3 at row 0
    assert_eq!(v, vec![0, 1, 0]);
}

#[test]
fn test_argmin_ext_2d_dim1_keepdim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmin_ext(1, true).unwrap();
    assert_eq!(result.dims(), &[2, 1]); // keepdim
    let v = result.to_flat_vec::<u32>().unwrap();
    // row 0: min(1,5,3)=1 at col 0; row 1: min(4,2,6)=2 at col 1
    assert_eq!(v, vec![0, 1]);
}

#[test]
fn test_argmin_ext_3d() {
    // [[[8, 7], [6, 5]], [[4, 3], [2, 1]]]  shape [2, 2, 2]
    let x = tnd(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 2, 2]);
    // argmin along dim=2 (last): min within each pair
    let result = x.argmin_ext(2, false).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let v = result.to_flat_vec::<u32>().unwrap();
    // [8,7]->1, [6,5]->1, [4,3]->1, [2,1]->1
    assert_eq!(v, vec![1, 1, 1, 1]);
}

#[test]
fn test_argmin_ext_3d_keepdim() {
    let x = tnd(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 2, 2]);
    let result = x.argmin_ext(2, true).unwrap();
    assert_eq!(result.dims(), &[2, 2, 1]); // keepdim
}

// =============================================================================
// Core topk method tests (sorted descending, largest only)
// =============================================================================

#[test]
fn test_topk_k1_is_argmax_equivalent() {
    // topk with k=1 should return the same value/index as argmax.
    let x = tnd(&[2.0, 8.0, 5.0, 1.0, 9.0, 3.0], &[6]);
    let (vals, idxs) = x.topk(0, 1).unwrap();
    assert_eq!(vals.dims(), &[1]);
    assert_eq!(vals.to_flat_vec::<f32>().unwrap(), vec![9.0]);
    assert_eq!(idxs.to_flat_vec::<u32>().unwrap(), vec![4]);
    // Compare with argmax.
    let argmax_idx = x.argmax(0).unwrap().to_flat_vec::<u32>().unwrap();
    assert_eq!(idxs.to_flat_vec::<u32>().unwrap(), argmax_idx);
}

#[test]
fn test_topk_k_equals_full_dim() {
    // topk where k == dim_size returns all elements sorted descending.
    let x = tnd(&[5.0, 3.0, 8.0, 1.0], &[4]);
    let (vals, idxs) = x.topk(0, 4).unwrap();
    assert_eq!(vals.dims(), &[4]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![8.0, 5.0, 3.0, 1.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![2, 0, 1, 3]);
}

#[test]
fn test_topk_1d_descending_order() {
    // Verify values are strictly sorted descending.
    let x = tnd(&[10.0, 30.0, 20.0, 50.0, 40.0], &[5]);
    let (vals, _) = x.topk(0, 3).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![50.0, 40.0, 30.0]);
    // Explicitly verify descending order.
    for w in v.windows(2) {
        assert!(
            w[0] >= w[1],
            "values must be descending: {} >= {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn test_topk_2d_dim0() {
    // 2D topk along first dimension (columns).
    // [[10, 20, 30],
    //  [40, 50, 60],
    //  [70, 80, 90]]
    let x = tnd(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        &[3, 3],
    );
    let (vals, idxs) = x.topk(0, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Top-2 along dim 0 for each column:
    // col 0: [10,40,70] -> [70,40], col 1: [20,50,80] -> [80,50], col 2: [30,60,90] -> [90,60]
    assert_eq!(v, vec![70.0, 80.0, 90.0, 40.0, 50.0, 60.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![2, 2, 2, 1, 1, 1]);
}

#[test]
fn test_topk_2d_dim1() {
    // 2D topk along last dimension (rows).
    // [[3, 1, 4],
    //  [1, 5, 9]]
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
    let (vals, idxs) = x.topk(1, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [3,1,4] -> top-2: [4,3]
    // Row 1: [1,5,9] -> top-2: [9,5]
    assert_eq!(v, vec![4.0, 3.0, 9.0, 5.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![2, 0, 2, 1]);
}

#[test]
fn test_topk_3d_tensor() {
    // Shape [2, 2, 4]: topk along the last dim with k=2.
    let data: Vec<f32> = vec![
        // batch 0, row 0
        1.0, 4.0, 2.0, 3.0, // batch 0, row 1
        8.0, 5.0, 7.0, 6.0, // batch 1, row 0
        12.0, 9.0, 11.0, 10.0, // batch 1, row 1
        16.0, 13.0, 15.0, 14.0,
    ];
    let x = tnd(&data, &[2, 2, 4]);
    let (vals, idxs) = x.topk(2, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // batch 0, row 0: [1,4,2,3] -> top-2: [4, 3]
    assert_eq!(v[0], 4.0);
    assert_eq!(v[1], 3.0);
    assert_eq!(i[0], 1);
    assert_eq!(i[1], 3);
    // batch 0, row 1: [8,5,7,6] -> top-2: [8, 7]
    assert_eq!(v[2], 8.0);
    assert_eq!(v[3], 7.0);
    assert_eq!(i[2], 0);
    assert_eq!(i[3], 2);
    // batch 1, row 0: [12,9,11,10] -> top-2: [12, 11]
    assert_eq!(v[4], 12.0);
    assert_eq!(v[5], 11.0);
    assert_eq!(i[4], 0);
    assert_eq!(i[5], 2);
    // batch 1, row 1: [16,13,15,14] -> top-2: [16, 15]
    assert_eq!(v[6], 16.0);
    assert_eq!(v[7], 15.0);
    assert_eq!(i[6], 0);
    assert_eq!(i[7], 2);
}

#[test]
fn test_topk_3d_dim0() {
    // Shape [3, 2, 2]: topk along dim 0 with k=2.
    let data: Vec<f32> = vec![
        // slice 0
        1.0, 2.0, 3.0, 4.0, // slice 1
        9.0, 8.0, 7.0, 6.0, // slice 2
        5.0, 10.0, 0.0, 11.0,
    ];
    let x = tnd(&data, &[3, 2, 2]);
    let (vals, idxs) = x.topk(0, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // Position [*, 0, 0]: values [1, 9, 5] -> top-2: [9, 5] at indices [1, 2]
    assert_eq!(v[0], 9.0);
    assert_eq!(i[0], 1);
    assert_eq!(v[4], 5.0);
    assert_eq!(i[4], 2);
    // Position [*, 0, 1]: values [2, 8, 10] -> top-2: [10, 8] at indices [2, 1]
    assert_eq!(v[1], 10.0);
    assert_eq!(i[1], 2);
    assert_eq!(v[5], 8.0);
    assert_eq!(i[5], 1);
}

#[test]
fn test_topk_3d_dim1() {
    // Shape [2, 3, 2]: topk along dim 1 with k=2.
    let data: Vec<f32> = vec![
        // batch 0: rows [1,2], [5,6], [3,4]
        1.0, 2.0, 5.0, 6.0, 3.0, 4.0, // batch 1: rows [7,8], [11,12], [9,10]
        7.0, 8.0, 11.0, 12.0, 9.0, 10.0,
    ];
    let x = tnd(&data, &[2, 3, 2]);
    let (vals, idxs) = x.topk(1, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // batch 0, col 0: [1,5,3] -> top-2: [5,3] at [1,2]
    assert_eq!(v[0], 5.0);
    assert_eq!(i[0], 1);
    assert_eq!(v[2], 3.0);
    assert_eq!(i[2], 2);
    // batch 0, col 1: [2,6,4] -> top-2: [6,4] at [1,2]
    assert_eq!(v[1], 6.0);
    assert_eq!(i[1], 1);
    assert_eq!(v[3], 4.0);
    assert_eq!(i[3], 2);
}

#[test]
fn test_topk_k_exceeds_dim_errors() {
    let x = tnd(&[1.0, 2.0], &[2]);
    assert!(x.topk(0, 3).is_err(), "k > dim should error");
}

#[test]
fn test_topk_k_zero_errors() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.topk(0, 0).is_err(), "k=0 should error");
}

#[test]
fn test_topk_nan_input_errors() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    assert!(x.topk(0, 2).is_err(), "NaN input should produce error");
}

#[test]
fn test_topk_single_element() {
    let x = tnd(&[-99.0], &[1]);
    let (vals, idxs) = x.topk(0, 1).unwrap();
    assert_eq!(vals.to_flat_vec::<f32>().unwrap(), vec![-99.0]);
    assert_eq!(idxs.to_flat_vec::<u32>().unwrap(), vec![0]);
}

// =============================================================================
// Negative values and mixed sign tests
// =============================================================================

#[test]
fn test_topk_all_negative_values() {
    let x = tnd(&[-5.0, -1.0, -10.0, -3.0, -7.0], &[5]);
    let (vals, idxs) = x.topk(0, 3).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Top-3 largest negatives: -1, -3, -5
    assert_eq!(v, vec![-1.0, -3.0, -5.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![1, 3, 0]);
}

#[test]
fn test_topk_ext_all_negative_values_smallest() {
    let x = tnd(&[-5.0, -1.0, -10.0, -3.0, -7.0], &[5]);
    let (vals, _) = x.topk_ext(3, 0, false, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Bottom-3 smallest negatives (most negative): -10, -7, -5
    assert_eq!(v, vec![-10.0, -7.0, -5.0]);
}

#[test]
fn test_topk_mixed_positive_negative() {
    let x = tnd(&[-3.0, 5.0, -1.0, 7.0, 0.0, -8.0], &[6]);
    let (vals, idxs) = x.topk(0, 3).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![7.0, 5.0, 0.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![3, 1, 4]);
}

#[test]
fn test_topk_ext_mixed_sign_smallest() {
    let x = tnd(&[-3.0, 5.0, -1.0, 7.0, 0.0, -8.0], &[6]);
    let (vals, idxs) = x.topk_ext(3, 0, false, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![-8.0, -3.0, -1.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![5, 0, 2]);
}

// =============================================================================
// All same values (ties) tests
// =============================================================================

#[test]
fn test_topk_all_same_values() {
    let x = tnd(&[3.0, 3.0, 3.0, 3.0, 3.0], &[5]);
    let (vals, idxs) = x.topk(0, 3).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![3.0, 3.0, 3.0]);
    // All indices should be valid and distinct.
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert!(i.iter().all(|&idx| idx < 5), "all indices in range");
}

#[test]
fn test_topk_all_same_values_2d() {
    let x = tnd(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0], &[2, 3]);
    let (vals, idxs) = x.topk(1, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert!(v.iter().all(|&val| val == 5.0), "all values should be 5.0");
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert!(i.iter().all(|&idx| idx < 3), "all indices in range 0..3");
}

#[test]
fn test_topk_all_zeros() {
    let x = tnd(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let (vals, idxs) = x.topk(0, 2).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 0.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert!(i.iter().all(|&idx| idx < 4));
}

// =============================================================================
// Negative dimension indexing (D::Minus1, i32 negative)
// =============================================================================

#[test]
fn test_topk_negative_dim_minus1() {
    // D::Minus1 should refer to the last dimension.
    let x = tnd(&[3.0, 1.0, 4.0, 6.0, 2.0, 5.0], &[2, 3]);
    let (vals, _) = x.topk(D::Minus1, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [3,1,4] -> top-2: [4,3]
    // Row 1: [6,2,5] -> top-2: [6,5]
    assert_eq!(v, vec![4.0, 3.0, 6.0, 5.0]);
}

#[test]
fn test_topk_ext_negative_dim_i32() {
    // i32 -1 should refer to the last dimension.
    let x = tnd(&[3.0, 1.0, 4.0, 6.0, 2.0, 5.0], &[2, 3]);
    let (vals, _) = x.topk_ext(2, -1_i32, true, true).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![4.0, 3.0, 6.0, 5.0]);
}

#[test]
fn test_topk_ext_negative_dim_minus2() {
    // D::Minus2 on a 3D tensor should refer to dim 1.
    let data: Vec<f32> = vec![
        // batch 0: [[1,2],[5,6],[3,4]]
        1.0, 2.0, 5.0, 6.0, 3.0, 4.0, // batch 1: [[7,8],[11,12],[9,10]]
        7.0, 8.0, 11.0, 12.0, 9.0, 10.0,
    ];
    let x = tnd(&data, &[2, 3, 2]);
    let (vals, _) = x.topk_ext(2, D::Minus2, true, true).unwrap();
    assert_eq!(vals.dims(), &[2, 2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // batch 0, col 0: [1,5,3] -> top-2: [5,3]
    assert_eq!(v[0], 5.0);
    assert_eq!(v[2], 3.0);
    // batch 0, col 1: [2,6,4] -> top-2: [6,4]
    assert_eq!(v[1], 6.0);
    assert_eq!(v[3], 4.0);
}

// =============================================================================
// Index correctness: values at returned indices match returned values
// =============================================================================

#[test]
fn test_topk_indices_match_values_1d() {
    let data = [7.0, 2.0, 9.0, 4.0, 6.0, 1.0, 8.0, 3.0, 5.0, 10.0];
    let x = tnd(&data, &[10]);
    let (vals, idxs) = x.topk(0, 5).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    for (val, idx) in v.iter().zip(i.iter()) {
        assert_eq!(
            *val, data[*idx as usize],
            "topk index {idx} should map to value {val}"
        );
    }
}

#[test]
fn test_topk_ext_indices_match_values_2d() {
    // Verify index correctness across multiple rows.
    let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
    let x = tnd(&data, &[2, 5]);
    let (vals, idxs) = x.topk_ext(3, 1, true, true).unwrap();
    assert_eq!(vals.dims(), &[2, 3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    // Row 0: data[0..5]
    for j in 0..3 {
        assert_eq!(v[j], data[i[j] as usize], "row 0 idx {j} val mismatch");
    }
    // Row 1: data[5..10]
    for j in 0..3 {
        assert_eq!(
            v[3 + j],
            data[5 + i[3 + j] as usize],
            "row 1 idx {j} val mismatch"
        );
    }
}

// =============================================================================
// BF16 dtype tests
// =============================================================================

#[test]
fn test_topk_bf16_dtype() {
    // Create an F32 tensor then convert to BF16, run topk, verify results.
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6]);
    let x_bf16 = x.to_dtype(DType::BF16).unwrap();
    assert_eq!(x_bf16.dtype(), DType::BF16);
    let (vals, idxs) = x_bf16.topk(0, 3).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![9.0, 5.0, 4.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![5, 4, 2]);
    assert_eq!(idxs.dtype(), DType::U32);
}

#[test]
fn test_topk_ext_bf16_largest_sorted() {
    let x = tnd(&[10.0, 30.0, 20.0, 50.0, 40.0], &[5]);
    let x_bf16 = x.to_dtype(DType::BF16).unwrap();
    let (vals, idxs) = x_bf16.topk_ext(3, 0, true, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![50.0, 40.0, 30.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![3, 4, 1]);
}

#[test]
fn test_topk_ext_bf16_smallest_sorted() {
    let x = tnd(&[10.0, 30.0, 20.0, 50.0, 40.0], &[5]);
    let x_bf16 = x.to_dtype(DType::BF16).unwrap();
    let (vals, idxs) = x_bf16.topk_ext(3, 0, false, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![10.0, 20.0, 30.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![0, 2, 1]);
}

// =============================================================================
// Descending order explicit verification
// =============================================================================

#[test]
fn test_topk_values_strictly_descending() {
    // Larger tensor to exercise partial sort path (k < dim_size).
    let data: Vec<f32> = (0..20).map(|i| (i as f32) * 1.7 - 10.0).collect();
    let x = tnd(&data, &[20]);
    let (vals, _) = x.topk(0, 8).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v.len(), 8);
    for w in v.windows(2) {
        assert!(
            w[0] >= w[1],
            "values must be descending: {} >= {}",
            w[0],
            w[1]
        );
    }
    // The top value should be the maximum in the input.
    let max_input = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(v[0], max_input);
}

#[test]
fn test_topk_ext_largest_sorted_descending_check() {
    let x = tnd(&[100.0, 1.0, 50.0, 25.0, 75.0, 10.0, 90.0, 5.0], &[8]);
    let (vals, _) = x.topk_ext(5, 0, true, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![100.0, 90.0, 75.0, 50.0, 25.0]);
    for w in v.windows(2) {
        assert!(w[0] >= w[1], "must be descending");
    }
}

#[test]
fn test_topk_ext_smallest_sorted_ascending_check() {
    let x = tnd(&[100.0, 1.0, 50.0, 25.0, 75.0, 10.0, 90.0, 5.0], &[8]);
    let (vals, _) = x.topk_ext(5, 0, false, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 5.0, 10.0, 25.0, 50.0]);
    for w in v.windows(2) {
        assert!(w[0] <= w[1], "must be ascending");
    }
}

// =============================================================================
// topk_ext on 3D tensors
// =============================================================================

#[test]
fn test_topk_ext_3d_last_dim_largest() {
    let data: Vec<f32> = vec![
        // batch 0, row 0
        5.0, 2.0, 8.0, 1.0, // batch 0, row 1
        3.0, 7.0, 4.0, 6.0, // batch 1, row 0
        9.0, 12.0, 10.0, 11.0, // batch 1, row 1
        15.0, 13.0, 16.0, 14.0,
    ];
    let x = tnd(&data, &[2, 2, 4]);
    let (vals, idxs) = x.topk_ext(2, 2, true, true).unwrap();
    assert_eq!(vals.dims(), &[2, 2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // batch 0, row 0: [5,2,8,1] -> top-2: [8,5]
    assert_eq!(v[0], 8.0);
    assert_eq!(v[1], 5.0);
    // batch 0, row 1: [3,7,4,6] -> top-2: [7,6]
    assert_eq!(v[2], 7.0);
    assert_eq!(v[3], 6.0);
    // batch 1, row 0: [9,12,10,11] -> top-2: [12,11]
    assert_eq!(v[4], 12.0);
    assert_eq!(v[5], 11.0);
    // batch 1, row 1: [15,13,16,14] -> top-2: [16,15]
    assert_eq!(v[6], 16.0);
    assert_eq!(v[7], 15.0);
    // Verify indices dtype.
    assert_eq!(idxs.dtype(), DType::U32);
}

#[test]
fn test_topk_ext_3d_smallest() {
    let data: Vec<f32> = vec![5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0, 6.0];
    let x = tnd(&data, &[1, 2, 4]);
    let (vals, _) = x.topk_ext(2, 2, false, true).unwrap();
    assert_eq!(vals.dims(), &[1, 2, 2]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [5,2,8,1] -> bottom-2 ascending: [1,2]
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    // Row 1: [3,7,4,6] -> bottom-2 ascending: [3,4]
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], 4.0);
}

// =============================================================================
// DynTensor::full / DynTensor::zeros construction tests
// =============================================================================

#[test]
fn test_topk_on_zeros_tensor() {
    let x = DynTensor::zeros(&[5], DType::F32, &Device::Cpu).unwrap();
    let (vals, idxs) = x.topk(0, 3).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 0.0, 0.0]);
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert!(i.iter().all(|&idx| idx < 5));
}

#[test]
fn test_topk_on_full_tensor() {
    let x = DynTensor::full(&[4], 42.0, DType::F32, &Device::Cpu).unwrap();
    let (vals, _) = x.topk(0, 2).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![42.0, 42.0]);
}

// =============================================================================
// argmax_ext / argmin_ext with negative dim
// =============================================================================

#[test]
fn test_argmax_ext_negative_dim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmax_ext(D::Minus1, false).unwrap();
    assert_eq!(result.dims(), &[2]);
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1, 2]); // max in each row
}

#[test]
fn test_argmin_ext_negative_dim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmin_ext(D::Minus1, false).unwrap();
    assert_eq!(result.dims(), &[2]);
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![0, 1]); // min in each row
}

#[test]
fn test_argmax_ext_negative_dim_keepdim() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let result = x.argmax_ext(D::Minus1, true).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1, 2]);
}

// =============================================================================
// argmax_ext / argmin_ext with negative values
// =============================================================================

#[test]
fn test_argmax_ext_all_negative() {
    let x = tnd(&[-5.0, -1.0, -10.0, -3.0], &[4]);
    let result = x.argmax_ext(0, false).unwrap();
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![1]); // -1.0 is the largest
}

#[test]
fn test_argmin_ext_all_negative() {
    let x = tnd(&[-5.0, -1.0, -10.0, -3.0], &[4]);
    let result = x.argmin_ext(0, false).unwrap();
    let v = result.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![2]); // -10.0 is the smallest
}

// =============================================================================
// Large tensor topk
// =============================================================================

#[test]
fn test_topk_larger_tensor() {
    // 100 element tensor, pick top 5.
    let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.5).collect();
    let x = tnd(&data, &[100]);
    let (vals, idxs) = x.topk(0, 5).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![49.5, 49.0, 48.5, 48.0, 47.5]);
    assert_eq!(i, vec![99, 98, 97, 96, 95]);
}

#[test]
fn test_topk_ext_larger_tensor_smallest() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.5).collect();
    let x = tnd(&data, &[100]);
    let (vals, idxs) = x.topk_ext(5, 0, false, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    let i = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(v, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    assert_eq!(i, vec![0, 1, 2, 3, 4]);
}

// =============================================================================
// topk_ext consistency: largest+sorted matches core topk
// =============================================================================

#[test]
fn test_topk_ext_largest_sorted_matches_core_topk() {
    // topk_ext(k, dim, largest=true, sorted=true) should produce same results as topk(dim, k).
    let x = tnd(&[7.0, 2.0, 9.0, 4.0, 6.0, 1.0, 8.0, 3.0], &[8]);
    let (vals_ext, idxs_ext) = x.topk_ext(4, 0, true, true).unwrap();
    let (vals_core, idxs_core) = x.topk(0, 4).unwrap();
    assert_eq!(
        vals_ext.to_flat_vec::<f32>().unwrap(),
        vals_core.to_flat_vec::<f32>().unwrap(),
    );
    assert_eq!(
        idxs_ext.to_flat_vec::<u32>().unwrap(),
        idxs_core.to_flat_vec::<u32>().unwrap(),
    );
}

#[test]
fn test_topk_ext_largest_sorted_matches_core_topk_2d() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
    let (vals_ext, idxs_ext) = x.topk_ext(2, 1, true, true).unwrap();
    let (vals_core, idxs_core) = x.topk(1, 2).unwrap();
    assert_eq!(
        vals_ext.to_flat_vec::<f32>().unwrap(),
        vals_core.to_flat_vec::<f32>().unwrap(),
    );
    assert_eq!(
        idxs_ext.to_flat_vec::<u32>().unwrap(),
        idxs_core.to_flat_vec::<u32>().unwrap(),
    );
}
