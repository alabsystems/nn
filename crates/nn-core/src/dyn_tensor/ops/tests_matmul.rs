#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MatMul operation tests (2D, 3D batched, 4D×2D broadcast, error paths).

use crate::dyn_tensor::test_helpers::{cpu, t2d};
use crate::DynTensor;

#[test]
fn test_matmul_2d() {
    // [2, 3] × [3, 2] → [2, 2]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    assert_eq!(flat, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_incompatible_shapes() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_matmul_3d_batched() {
    // [2, 2, 3] × [2, 3, 1] → [2, 2, 1]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 2, 3], &cpu()).unwrap();
    let b_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b = DynTensor::from_vec(b_data, &[2, 3, 1], &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Batch 0: [1,2,3; 4,5,6] × [1;2;3] = [14; 32]
    // Batch 1: [7,8,9; 10,11,12] × [4;5;6] = [122; 167]
    assert_eq!(flat, vec![14.0, 32.0, 122.0, 167.0]);
}

#[test]
fn test_matmul_4d_2d_broadcast() {
    // [B, H, M, K] × [K, N] → [B, H, M, N]
    // [2, 2, 1, 3] × [3, 2] → [2, 2, 1, 2]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 2, 1, 3], &cpu()).unwrap();
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 1, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Slice [0,0,0,:] = [1,2,3] × [[1,2],[3,4],[5,6]] = [22, 28]
    assert_eq!(&flat[0..2], &[22.0, 28.0]);
    // Slice [0,1,0,:] = [4,5,6] × [[1,2],[3,4],[5,6]] = [49, 64]
    assert_eq!(&flat[2..4], &[49.0, 64.0]);
    // Slice [1,0,0,:] = [7,8,9] × [[1,2],[3,4],[5,6]] = [76, 100]
    assert_eq!(&flat[4..6], &[76.0, 100.0]);
    // Slice [1,1,0,:] = [10,11,12] × [[1,2],[3,4],[5,6]] = [103, 136]
    assert_eq!(&flat[6..8], &[103.0, 136.0]);
}

#[test]
fn test_matmul_4d_2d_shape_mismatch() {
    // Inner dimension mismatch: K=3 vs K=2
    let a = DynTensor::from_vec(vec![0.0; 24], &[2, 2, 2, 3], &cpu()).unwrap();
    let b = t2d(&[0.0; 4], 2, 2);
    assert!(a.matmul(&b).is_err());
}

/// Transformer-scale matmul precision test.
/// Exercises accumulation at dimensions typical of transformer forward passes.
/// Detects precision drift from tiled/MPS GEMM vs sequential CPU accumulation.
/// Re: #1289
#[test]
fn test_matmul_transformer_scale_precision() {
    // Dimensions matching Qwen3 tiny config: hidden=256, ff=512
    let (m, k, n) = (4, 256, 512);
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
        .collect();

    let a = DynTensor::from_vec(a_data.clone(), &[m, k], &cpu()).unwrap();
    let b = DynTensor::from_vec(b_data.clone(), &[k, n], &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[m, n]);

    // Verify against manual f64 reference to catch f32 accumulation drift.
    let c_vals = c.to_flat_vec::<f32>().unwrap();
    for row in 0..m {
        for col in 0..n {
            let mut expected: f64 = 0.0;
            for kk in 0..k {
                expected += f64::from(a_data[row * k + kk]) * f64::from(b_data[kk * n + col]);
            }
            let actual = f64::from(c_vals[row * n + col]);
            let diff = (actual - expected).abs();
            // f32 accumulation over K=256 terms with values in [-0.48, 0.48]:
            // error bound ≈ K * eps * max|product| ≈ 256 * 1.2e-7 * 0.23 ≈ 7e-6.
            // Use 1e-4 for comfortable margin (was 1e-2, 300x too loose).
            assert!(
                diff < 1e-4,
                "matmul[{row},{col}]: f32={actual:.6}, f64_ref={expected:.6}, diff={diff:.6}"
            );
        }
    }
}

// -- Missing rank combination coverage (P1-187 strategic) ----------------------

#[test]
fn test_matmul_3d_2d_broadcast() {
    // [B, M, K] × [K, N] → [B, M, N]
    // [2, 2, 3] × [3, 2] → [2, 2, 2]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 2, 3], &cpu()).unwrap();
    let b = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Batch 0, row 0: [1,2,3] × [[1,2],[3,4],[5,6]] = [22, 28]
    assert_eq!(&flat[0..2], &[22.0, 28.0]);
    // Batch 0, row 1: [4,5,6] × [[1,2],[3,4],[5,6]] = [49, 64]
    assert_eq!(&flat[2..4], &[49.0, 64.0]);
    // Batch 1, row 0: [7,8,9] × same W = [76, 100]
    assert_eq!(&flat[4..6], &[76.0, 100.0]);
    // Batch 1, row 1: [10,11,12] × same W = [103, 136]
    assert_eq!(&flat[6..8], &[103.0, 136.0]);
}

#[test]
fn test_matmul_4d_4d_batched() {
    // [B, H, M, K] × [B, H, K, N] → [B, H, M, N]
    // [1, 2, 2, 3] × [1, 2, 3, 1] → [1, 2, 2, 1]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[1, 2, 2, 3], &cpu()).unwrap();
    let b_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b = DynTensor::from_vec(b_data, &[1, 2, 3, 1], &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[1, 2, 2, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Head 0: [1,2,3; 4,5,6] × [1;2;3] = [14; 32]
    assert_eq!(&flat[0..2], &[14.0, 32.0]);
    // Head 1: [7,8,9; 10,11,12] × [4;5;6] = [122; 167]
    assert_eq!(&flat[2..4], &[122.0, 167.0]);
}

#[test]
fn test_matmul_4d_4d_batch_mismatch() {
    let a = DynTensor::from_vec(vec![0.0; 24], &[2, 2, 2, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.0; 18], &[3, 2, 3, 1], &cpu()).unwrap();
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_matmul_3d_batch_mismatch() {
    let a = DynTensor::from_vec(vec![0.0; 12], &[2, 2, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![0.0; 9], &[3, 3, 1], &cpu()).unwrap();
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_matmul_single_element() {
    // [1,1] × [1,1] → [1,1]
    let a = t2d(&[3.0], 1, 1);
    let b = t2d(&[7.0], 1, 1);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![21.0]);
}

#[test]
fn test_matmul_rank1_rejected() {
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![4.0, 5.0, 6.0], &[3], &cpu()).unwrap();
    let err = a.matmul(&b).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("not yet supported") || msg.contains("ranks"));
}

/// Batched matmul precision at attention dimensions.
/// Tests [B, H, S, head_dim] @ [head_dim, head_dim] pattern.
/// Re: #1289
#[test]
fn test_matmul_batched_attention_scale_precision() {
    let batch = 1;
    let heads = 2;
    let seq = 4;
    let head_dim = 128;
    let a_data: Vec<f32> = (0..batch * heads * seq * head_dim)
        .map(|i| ((i % 101) as f32 - 50.0) * 0.01)
        .collect();
    let w_data: Vec<f32> = (0..head_dim * head_dim)
        .map(|i| ((i % 83) as f32 - 41.0) * 0.01)
        .collect();

    let a = DynTensor::from_vec(a_data, &[batch, heads, seq, head_dim], &cpu()).unwrap();
    let w = DynTensor::from_vec(w_data, &[head_dim, head_dim], &cpu()).unwrap();
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[batch, heads, seq, head_dim]);

    // All values must be finite (catches NaN from accumulation overflow).
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all matmul outputs finite"
    );
    // Maximum absolute value should be bounded given small input range.
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 100.0,
        "matmul output magnitude {max_abs} unexpectedly large"
    );
}
