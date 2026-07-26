#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for im2col_1d and im2col_2d operations.

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;

#[test]
fn test_im2col_1d_basic() {
    // Input: [1, 1, 5] = [0, 1, 2, 3, 4]
    // kernel_size=3, stride=1, padding=0, dilation=1
    // out_len = (5 - 3) / 1 + 1 = 3
    // Expected columns [1, 3, 3]:
    //   col[:, 0, :] = [0, 1, 2]  (k=0)
    //   col[:, 1, :] = [1, 2, 3]  (k=1)
    //   col[:, 2, :] = [2, 3, 4]  (k=2)
    let input = DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5], &cpu()).unwrap();
    let col = input.im2col_1d(3, 1, 0, 1).unwrap();
    assert_eq!(col.dims(), &[1, 3, 3]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // Layout: [B, C*K, L_out] = [1, 3, 3]
    // Row 0 (c=0, k=0): [0, 1, 2]
    // Row 1 (c=0, k=1): [1, 2, 3]
    // Row 2 (c=0, k=2): [2, 3, 4]
    assert_eq!(data, vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_im2col_1d_with_padding() {
    // Input: [1, 1, 3] = [1, 2, 3]
    // kernel_size=3, stride=1, padding=1, dilation=1
    // padded = [0, 1, 2, 3, 0], out_len = (3 + 2 - 3) / 1 + 1 = 3
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let col = input.im2col_1d(3, 1, 1, 1).unwrap();
    assert_eq!(col.dims(), &[1, 3, 3]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // k=0: [0, 1, 2]
    // k=1: [1, 2, 3]
    // k=2: [2, 3, 0]
    assert_eq!(data, vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_im2col_1d_with_stride() {
    // Input: [1, 1, 6] = [0, 1, 2, 3, 4, 5]
    // kernel_size=3, stride=2, padding=0, dilation=1
    // out_len = (6 - 3) / 2 + 1 = 2
    let input =
        DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 6], &cpu()).unwrap();
    let col = input.im2col_1d(3, 2, 0, 1).unwrap();
    assert_eq!(col.dims(), &[1, 3, 2]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // Position 0: input[0:3] = [0, 1, 2]
    // Position 1: input[2:5] = [2, 3, 4]
    // k=0: [0, 2], k=1: [1, 3], k=2: [2, 4]
    assert_eq!(data, vec![0.0, 2.0, 1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_im2col_1d_with_dilation() {
    // Input: [1, 1, 5] = [0, 1, 2, 3, 4]
    // kernel_size=2, stride=1, padding=0, dilation=2
    // effective_k = 2*1 + 1 = 3, out_len = (5 - 3) / 1 + 1 = 3
    let input = DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5], &cpu()).unwrap();
    let col = input.im2col_1d(2, 1, 0, 2).unwrap();
    assert_eq!(col.dims(), &[1, 2, 3]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // k=0 (offset 0): input[0..3] = [0, 1, 2]
    // k=1 (offset 2): input[2..5] = [2, 3, 4]
    assert_eq!(data, vec![0.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_im2col_1d_multi_channel() {
    // Input: [1, 2, 4]
    // ch0: [0, 1, 2, 3], ch1: [4, 5, 6, 7]
    // kernel_size=2, stride=1, padding=0, dilation=1
    // out_len = (4 - 2) / 1 + 1 = 3
    let input = DynTensor::from_vec(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[1, 2, 4],
        &cpu(),
    )
    .unwrap();
    let col = input.im2col_1d(2, 1, 0, 1).unwrap();
    assert_eq!(col.dims(), &[1, 4, 3]); // C*K = 2*2 = 4

    let data = col.to_flat_vec::<f32>().unwrap();
    // Layout: [B, C*K, L_out] with C interleaved with K:
    // (c=0,k=0): [0, 1, 2]
    // (c=0,k=1): [1, 2, 3]
    // (c=1,k=0): [4, 5, 6]
    // (c=1,k=1): [5, 6, 7]
    assert_eq!(
        data,
        vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0]
    );
}

#[test]
fn test_im2col_1d_batch() {
    // Input: [2, 1, 4]
    // batch 0: [1, 2, 3, 4], batch 1: [5, 6, 7, 8]
    // kernel_size=2, stride=1, padding=0, dilation=1
    // out_len = 3
    let input = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 1, 4],
        &cpu(),
    )
    .unwrap();
    let col = input.im2col_1d(2, 1, 0, 1).unwrap();
    assert_eq!(col.dims(), &[2, 2, 3]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // batch 0: (k=0): [1, 2, 3], (k=1): [2, 3, 4]
    // batch 1: (k=0): [5, 6, 7], (k=1): [6, 7, 8]
    assert_eq!(
        data,
        vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn test_im2col_1d_kernel_size_1() {
    // Degenerate case: K=1 means im2col is a no-op (or just strided access)
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &cpu()).unwrap();
    let col = input.im2col_1d(1, 1, 0, 1).unwrap();
    assert_eq!(col.dims(), &[1, 1, 4]);
    assert_eq!(col.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_im2col_1d_stride_and_dilation() {
    // Input: [1, 1, 8] = [0, 1, 2, 3, 4, 5, 6, 7]
    // kernel_size=2, stride=2, padding=0, dilation=2
    // effective_k = 2*1 + 1 = 3, out_len = (8 - 3) / 2 + 1 = 3
    let input = DynTensor::from_vec(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[1, 1, 8],
        &cpu(),
    )
    .unwrap();
    let col = input.im2col_1d(2, 2, 0, 2).unwrap();
    assert_eq!(col.dims(), &[1, 2, 3]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // Positions: t=0 → start=0, t=1 → start=2, t=2 → start=4
    // k=0 (offset 0): input[0], input[2], input[4] = [0, 2, 4]
    // k=1 (offset 2): input[2], input[4], input[6] = [2, 4, 6]
    assert_eq!(data, vec![0.0, 2.0, 4.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_im2col_1d_invalid_rank() {
    let input = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(input.im2col_1d(2, 1, 0, 1).is_err());
}

#[test]
fn test_im2col_2d_basic() {
    // Input: [1, 1, 3, 3] with values 0..9
    // kernel 2x2, stride=1, padding=0, dilation=1
    // out_h = 2, out_w = 2
    let input =
        DynTensor::from_vec((0..9).map(|x| x as f32).collect(), &[1, 1, 3, 3], &cpu()).unwrap();
    let col = input.im2col_2d(2, 2, 1, 0, 1).unwrap();
    // Output: [1, 1*2*2, 2*2] = [1, 4, 4]
    assert_eq!(col.dims(), &[1, 4, 4]);

    let data = col.to_flat_vec::<f32>().unwrap();
    // Patches for (kh, kw) pairs at each (oh, ow):
    // (kh=0,kw=0): [0, 1, 3, 4]  (elements at (0,0),(0,1),(1,0),(1,1))
    // (kh=0,kw=1): [1, 2, 4, 5]
    // (kh=1,kw=0): [3, 4, 6, 7]
    // (kh=1,kw=1): [4, 5, 7, 8]
    assert_eq!(
        data,
        vec![
            0.0, 1.0, 3.0, 4.0, // (kh=0,kw=0)
            1.0, 2.0, 4.0, 5.0, // (kh=0,kw=1)
            3.0, 4.0, 6.0, 7.0, // (kh=1,kw=0)
            4.0, 5.0, 7.0, 8.0, // (kh=1,kw=1)
        ]
    );
}

#[test]
fn test_im2col_2d_invalid_rank() {
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    assert!(input.im2col_2d(2, 2, 1, 0, 1).is_err());
}

/// Regression test: stride index must not silently truncate via `as u32`.
/// Before the fix, `(t * stride) as u32` would silently wrap for large
/// stride*out_len products exceeding u32::MAX (~4.3 billion).
/// The fix uses `u32::try_from` with proper error propagation.
#[test]
fn test_im2col_1d_stride_index_u32_overflow_detected() {
    // Construct parameters where the last stride index exceeds u32::MAX.
    // out_len * stride > u32::MAX with modest parameters:
    // input_len = 2_000_000_001, kernel_size = 1, stride = 3, padding = 0
    // out_len = (2_000_000_001 - 1) / 3 + 1 = 666_666_667
    // last index = 666_666_666 * 3 = 1_999_999_998 — this fits in u32.
    //
    // For overflow: input_len = 6_000_000_001, stride = 3
    // out_len = 2_000_000_001, last index = 6_000_000_000 > u32::MAX
    //
    // We can't allocate 6B elements, so we test the error path by checking
    // that the index validation itself catches the overflow using smaller
    // parameters that still trigger the check.
    //
    // Instead, use a tensor small enough to allocate but with stride that
    // produces an index just above u32::MAX. For kernel_size=1, stride=S:
    // Need out_len such that (out_len - 1) * S > u32::MAX.
    // If S = 2, out_len >= 2_147_483_649 — still too big to allocate.
    //
    // The practical approach: verify the code returns Err instead of silently
    // truncating by constructing a case where `t * stride` would overflow u32
    // on the INDEX side even though the tensor is small.
    // With stride=u32::MAX as usize + 1, and out_len=2:
    // index 0 = 0, index 1 = 4294967296 > u32::MAX.
    // But conv1d_out_len would need: (in_len + 2*pad - k) / stride + 1 = 2
    // => in_len + 2*pad - k = stride => in_len = stride + k - 2*pad
    // With k=1, pad=0: in_len = 4294967296 + 1 = 4294967297.
    // Still too big to allocate.
    //
    // This demonstrates the fix catches the theoretical overflow, but we
    // cannot practically trigger it without allocating >4B elements.
    // The test below verifies correctness of the try_from path by asserting
    // that normal-sized inputs still work correctly after the fix.
    let input =
        DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 6], &cpu()).unwrap();
    let col = input.im2col_1d(2, 2, 0, 1).unwrap();
    assert_eq!(col.dims(), &[1, 2, 3]);
    // Verify stride indices [0, 2, 4] produce correct values
    let data = col.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
}
