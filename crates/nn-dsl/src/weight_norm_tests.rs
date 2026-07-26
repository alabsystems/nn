// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for weight normalization decomposition and reference implementation.

use super::*;
use crate::tensor_ir::TensorNodeId;

// --- Decomposed builder tests ---

#[test]
fn build_weight_norm_decomposed_basic() {
    let def = build_weight_norm_decomposed(4, 8).unwrap();
    assert_eq!(def.name, "weight_norm");
    assert_eq!(def.nodes.len(), 12);
    assert_eq!(def.output, TensorNodeId::new(11));
    // Output shape: [fan_out, fan_in]
    assert_eq!(def.nodes[11].shape, vec![4, 8]);
}

#[test]
fn build_weight_norm_decomposed_single_element() {
    let def = build_weight_norm_decomposed(1, 1).unwrap();
    assert_eq!(def.nodes.len(), 12);
    assert_eq!(def.nodes[11].shape, vec![1, 1]);
}

#[test]
fn build_weight_norm_decomposed_zero_fan_out() {
    let result = build_weight_norm_decomposed(0, 8);
    assert!(result.is_err());
}

#[test]
fn build_weight_norm_decomposed_zero_fan_in() {
    let result = build_weight_norm_decomposed(4, 0);
    assert!(result.is_err());
}

#[test]
fn build_weight_norm_decomposed_node_shapes() {
    let def = build_weight_norm_decomposed(3, 5).unwrap();
    // Node 0: v [3, 5]
    assert_eq!(def.nodes[0].shape, vec![3, 5]);
    // Node 1: eps [1]
    assert_eq!(def.nodes[1].shape, vec![1]);
    // Node 2: g [3]
    assert_eq!(def.nodes[2].shape, vec![3]);
    // Node 3: v² [3, 5]
    assert_eq!(def.nodes[3].shape, vec![3, 5]);
    // Node 4: sum(v²) [3]
    assert_eq!(def.nodes[4].shape, vec![3]);
    // Node 5: broadcast [3, 5]
    assert_eq!(def.nodes[5].shape, vec![3, 5]);
    // Node 6: eps broadcast [3, 5]
    assert_eq!(def.nodes[6].shape, vec![3, 5]);
    // Node 7: sum(v²)+eps [3, 5]
    assert_eq!(def.nodes[7].shape, vec![3, 5]);
    // Node 8: rsqrt [3, 5]
    assert_eq!(def.nodes[8].shape, vec![3, 5]);
    // Node 9: v * rsqrt [3, 5]
    assert_eq!(def.nodes[9].shape, vec![3, 5]);
    // Node 10: g broadcast [3, 5]
    assert_eq!(def.nodes[10].shape, vec![3, 5]);
    // Node 11: g * normalized [3, 5]
    assert_eq!(def.nodes[11].shape, vec![3, 5]);
}

// --- Reference implementation tests ---

#[test]
fn weight_norm_ref_identity() {
    // Unit vector with g=1 should be preserved.
    // v = [1, 0, 0], g = [1], eps = 1e-6
    // ||v|| = sqrt(1 + 1e-6) ≈ 1.0
    // w = 1 * [1, 0, 0] / 1.0 ≈ [1, 0, 0]
    let v = vec![1.0, 0.0, 0.0];
    let g = vec![1.0];
    let result = weight_norm_ref(&v, &g, 1, 3, 1e-6).unwrap();
    assert!((result[0] - 1.0).abs() < 1e-3);
    assert!(result[1].abs() < 1e-3);
    assert!(result[2].abs() < 1e-3);
}

#[test]
fn weight_norm_ref_magnitude_scaling() {
    // v = [3, 4], ||v|| = sqrt(9 + 16) = 5
    // g = [2], w = 2 * [3, 4] / 5 = [1.2, 1.6]
    let v = vec![3.0, 4.0];
    let g = vec![2.0];
    let result = weight_norm_ref(&v, &g, 1, 2, 1e-12).unwrap();
    assert!((result[0] - 1.2).abs() < 1e-5, "got {}", result[0]);
    assert!((result[1] - 1.6).abs() < 1e-5, "got {}", result[1]);
}

#[test]
fn weight_norm_ref_multi_row() {
    // Two rows: v = [[3, 4], [0, 5]], g = [2, 3]
    // Row 0: ||v|| = 5, w = 2 * [3, 4] / 5 = [1.2, 1.6]
    // Row 1: ||v|| = 5, w = 3 * [0, 5] / 5 = [0.0, 3.0]
    let v = vec![3.0, 4.0, 0.0, 5.0];
    let g = vec![2.0, 3.0];
    let result = weight_norm_ref(&v, &g, 2, 2, 1e-12).unwrap();
    assert!((result[0] - 1.2).abs() < 1e-5);
    assert!((result[1] - 1.6).abs() < 1e-5);
    assert!(result[2].abs() < 1e-5);
    assert!((result[3] - 3.0).abs() < 1e-5);
}

#[test]
fn weight_norm_ref_zero_vector_with_eps() {
    // v = [0, 0], g = [1], eps = 1e-6
    // ||v|| = sqrt(0 + 1e-6) = 1e-3
    // w = 1 * [0, 0] / 1e-3 = [0, 0]
    let v = vec![0.0, 0.0];
    let g = vec![1.0];
    let result = weight_norm_ref(&v, &g, 1, 2, 1e-6).unwrap();
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
}

#[test]
fn weight_norm_ref_negative_g() {
    // v = [3, 4], ||v|| = 5, g = [-1]
    // w = -1 * [3, 4] / 5 = [-0.6, -0.8]
    let v = vec![3.0, 4.0];
    let g = vec![-1.0];
    let result = weight_norm_ref(&v, &g, 1, 2, 1e-12).unwrap();
    assert!((result[0] - (-0.6)).abs() < 1e-5);
    assert!((result[1] - (-0.8)).abs() < 1e-5);
}

// --- Error path tests ---

#[test]
fn weight_norm_ref_shape_mismatch_v() {
    let result = weight_norm_ref(&[1.0, 2.0, 3.0], &[1.0], 1, 2, 1e-6);
    assert!(matches!(result, Err(KernelError::ShapeMismatch { .. })));
}

#[test]
fn weight_norm_ref_shape_mismatch_g() {
    let result = weight_norm_ref(&[1.0, 2.0], &[1.0, 2.0], 1, 2, 1e-6);
    assert!(matches!(result, Err(KernelError::ShapeMismatch { .. })));
}

#[test]
fn weight_norm_ref_invalid_eps_zero() {
    let result = weight_norm_ref(&[1.0, 2.0], &[1.0], 1, 2, 0.0);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn weight_norm_ref_invalid_eps_negative() {
    let result = weight_norm_ref(&[1.0, 2.0], &[1.0], 1, 2, -1e-6);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn weight_norm_ref_nan_input() {
    let result = weight_norm_ref(&[f32::NAN, 2.0], &[1.0], 1, 2, 1e-6);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteSliceElement { .. })
    ));
}

#[test]
fn weight_norm_ref_inf_input() {
    let result = weight_norm_ref(&[f32::INFINITY, 2.0], &[1.0], 1, 2, 1e-6);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteSliceElement { .. })
    ));
}

#[test]
fn weight_norm_ref_nan_in_g() {
    let result = weight_norm_ref(&[1.0, 2.0], &[f32::NAN], 1, 2, 1e-6);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteSliceElement { .. })
    ));
}

#[test]
fn weight_norm_ref_inf_in_g() {
    let result = weight_norm_ref(&[1.0, 2.0], &[f32::INFINITY], 1, 2, 1e-6);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteSliceElement { .. })
    ));
}

#[test]
fn weight_norm_ref_nan_eps() {
    let result = weight_norm_ref(&[1.0, 2.0], &[1.0], 1, 2, f32::NAN);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn weight_norm_ref_zero_dim() {
    let result = weight_norm_ref(&[], &[], 0, 2, 1e-6);
    assert!(matches!(result, Err(KernelError::InvalidDimension { .. })));
}

// --- Scalar function tests ---

#[test]
fn weight_norm_scalar_basic() {
    // g * v * norm_inv = 2.0 * 3.0 * 0.5 = 3.0
    let result = weight_norm_scalar(3.0, 2.0, 0.5).unwrap();
    assert!((result - 3.0).abs() < 1e-6);
}

#[test]
fn weight_norm_scalar_zero_v() {
    let result = weight_norm_scalar(0.0, 5.0, 0.2).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn weight_norm_scalar_nan_input() {
    let result = weight_norm_scalar(f32::NAN, 1.0, 1.0);
    assert!(matches!(result, Err(KernelError::NonFiniteInput { .. })));
}

#[test]
fn weight_norm_scalar_inf_input() {
    let result = weight_norm_scalar(1.0, f32::INFINITY, 1.0);
    assert!(matches!(result, Err(KernelError::NonFiniteInput { .. })));
}

// Builder integration test for add_weight_norm deferred to next commit —
// tensor_block_builder_ops.rs has concurrent edits from W3 (LayerNorm)
// that prevent atomic staging.
