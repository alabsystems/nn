// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD reduction operations.

use super::*;

// ---------------------------------------------------------------------------
// sum_f32
// ---------------------------------------------------------------------------

#[test]
fn test_sum_f32_scalar_basic() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sum_f32_scalar(&x);
    assert!(
        (result - 15.0).abs() < 1e-6,
        "sum should be 15, got {result}"
    );
}

#[test]
fn test_sum_f32_empty() {
    let x: &[f32] = &[];
    let result = sum_f32(x);
    assert!(
        (result - 0.0).abs() < 1e-6,
        "sum of empty should be 0, got {result}"
    );
}

#[test]
fn test_sum_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..33).map(|i| (i as f32 * 0.7).sin()).collect();
    let dispatch = sum_f32(&x);
    let scalar = sum_f32_scalar(&x);
    assert!(
        (dispatch - scalar).abs() < 1e-4,
        "dispatch={dispatch}, scalar={scalar}"
    );
}

// ---------------------------------------------------------------------------
// max_f32
// ---------------------------------------------------------------------------

#[test]
fn test_max_f32_scalar_basic() {
    let x = [-5.0, 3.0, 1.0, 7.0, -2.0, 4.0, 0.0, 6.0, 2.0];
    let result = max_f32_scalar(&x);
    assert!((result - 7.0).abs() < 1e-6, "max should be 7, got {result}");
}

#[test]
fn test_max_f32_empty() {
    let x: &[f32] = &[];
    let result = max_f32(x);
    assert!(
        result == f32::NEG_INFINITY,
        "max of empty should be NEG_INFINITY"
    );
}

#[test]
fn test_max_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).cos()).collect();
    let dispatch = max_f32(&x);
    let scalar = max_f32_scalar(&x);
    assert!(
        (dispatch - scalar).abs() < 1e-6,
        "dispatch={dispatch}, scalar={scalar}"
    );
}

// ---------------------------------------------------------------------------
// min_f32
// ---------------------------------------------------------------------------

#[test]
fn test_min_f32_scalar_basic() {
    let x = [5.0, 3.0, 1.0, -7.0, 2.0, 4.0, 0.0, -3.0, 8.0];
    let result = min_f32_scalar(&x);
    assert!(
        (result - (-7.0)).abs() < 1e-6,
        "min should be -7, got {result}"
    );
}

#[test]
fn test_min_f32_empty() {
    let x: &[f32] = &[];
    let result = min_f32(x);
    assert!(result == f32::INFINITY, "min of empty should be INFINITY");
}

#[test]
fn test_min_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..41).map(|i| (i as f32 * 0.5).sin()).collect();
    let dispatch = min_f32(&x);
    let scalar = min_f32_scalar(&x);
    assert!(
        (dispatch - scalar).abs() < 1e-6,
        "dispatch={dispatch}, scalar={scalar}"
    );
}

// ---------------------------------------------------------------------------
// dot_f32
// ---------------------------------------------------------------------------

#[test]
fn test_dot_f32_scalar_basic() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let result = dot_f32_scalar(&a, &b);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert!(
        (result - 70.0).abs() < 1e-5,
        "dot should be 70, got {result}"
    );
}

#[test]
fn test_dot_f32_dispatch_matches_scalar() {
    let a: Vec<f32> = (0..35).map(|i| (i as f32 * 0.2).sin()).collect();
    let b: Vec<f32> = (0..35).map(|i| (i as f32 * 0.3).cos()).collect();
    let dispatch = dot_f32(&a, &b);
    let scalar = dot_f32_scalar(&a, &b);
    assert!(
        (dispatch - scalar).abs() < 1e-4,
        "dispatch={dispatch}, scalar={scalar}"
    );
}

#[test]
fn test_dot_f32_orthogonal() {
    // Orthogonal vectors: dot should be ~0.
    let a = [1.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0, 0.0];
    let result = dot_f32(&a, &b);
    assert!(
        result.abs() < 1e-6,
        "dot of orthogonal should be ~0, got {result}"
    );
}

// ---------------------------------------------------------------------------
// l2_norm_f32
// ---------------------------------------------------------------------------

#[test]
fn test_l2_norm_f32_scalar_basic() {
    let x = [3.0, 4.0];
    let result = l2_norm_f32_scalar(&x);
    assert!(
        (result - 5.0).abs() < 1e-6,
        "L2 norm of [3,4] should be 5, got {result}"
    );
}

#[test]
fn test_l2_norm_f32_unit_vector() {
    let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = l2_norm_f32(&x);
    assert!(
        (result - 1.0).abs() < 1e-6,
        "L2 norm of unit vector should be 1, got {result}"
    );
}

#[test]
fn test_l2_norm_f32_empty() {
    let x: &[f32] = &[];
    let result = l2_norm_f32(x);
    assert!(
        (result - 0.0).abs() < 1e-6,
        "L2 norm of empty should be 0, got {result}"
    );
}

#[test]
fn test_l2_norm_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..29).map(|i| (i as f32 * 0.4).sin()).collect();
    let dispatch = l2_norm_f32(&x);
    let scalar = l2_norm_f32_scalar(&x);
    assert!(
        (dispatch - scalar).abs() < 1e-4,
        "dispatch={dispatch}, scalar={scalar}"
    );
}

// ---------------------------------------------------------------------------
// NEON-specific tests (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[test]
fn test_sum_f32_neon_matches_scalar() {
    let x: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
    let neon = sum_f32_neon(&x);
    let scalar = sum_f32_scalar(&x);
    assert!((neon - scalar).abs() < 1e-4, "neon={neon}, scalar={scalar}");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_min_f32_neon_matches_scalar() {
    let x: Vec<f32> = (0..17).map(|i| (i as f32 * 0.7).sin()).collect();
    let neon = min_f32_neon(&x);
    let scalar = min_f32_scalar(&x);
    assert!((neon - scalar).abs() < 1e-6, "neon={neon}, scalar={scalar}");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_dot_f32_neon_matches_scalar() {
    let a: Vec<f32> = (0..21).map(|i| i as f32 * 0.3).collect();
    let b: Vec<f32> = (0..21).map(|i| (i as f32).sin()).collect();
    let neon = dot_f32_neon(&a, &b);
    let scalar = dot_f32_scalar(&a, &b);
    assert!((neon - scalar).abs() < 1e-3, "neon={neon}, scalar={scalar}");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_l2_norm_f32_neon_matches_scalar() {
    let x: Vec<f32> = (0..25).map(|i| (i as f32 * 0.2).cos()).collect();
    let neon = l2_norm_f32_neon(&x);
    let scalar = l2_norm_f32_scalar(&x);
    assert!((neon - scalar).abs() < 1e-4, "neon={neon}, scalar={scalar}");
}

// ---------------------------------------------------------------------------
// Large input (exercises main SIMD loop + scalar tail)
// ---------------------------------------------------------------------------

#[test]
fn test_sum_f32_large() {
    let n = 1024 + 5;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let dispatch = sum_f32(&x);
    let scalar = sum_f32_scalar(&x);
    assert!(
        (dispatch - scalar).abs() < 0.1,
        "dispatch={dispatch}, scalar={scalar}"
    );
}

#[test]
fn test_dot_f32_large() {
    let n = 512 + 3;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
    let dispatch = dot_f32(&a, &b);
    let scalar = dot_f32_scalar(&a, &b);
    assert!(
        (dispatch - scalar).abs() < 0.1,
        "dispatch={dispatch}, scalar={scalar}"
    );
}
