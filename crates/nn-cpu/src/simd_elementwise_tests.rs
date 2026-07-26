// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD elementwise operations.

use super::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn assert_approx(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// add_f32
// ---------------------------------------------------------------------------

#[test]
fn test_add_f32_scalar_basic() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0];
    let mut out = [0.0f32; 5];
    add_f32_scalar(&a, &b, &mut out);
    assert_approx(&out, &[11.0, 22.0, 33.0, 44.0, 55.0], 1e-6);
}

#[test]
fn test_add_f32_dispatch_matches_scalar() {
    let a: Vec<f32> = (0..17).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..17).map(|i| (i as f32).sin()).collect();
    let mut out_dispatch = vec![0.0f32; 17];
    let mut out_scalar = vec![0.0f32; 17];
    add_f32(&a, &b, &mut out_dispatch);
    add_f32_scalar(&a, &b, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-6);
}

#[test]
fn test_add_f32_empty() {
    let a: &[f32] = &[];
    let b: &[f32] = &[];
    let mut out: Vec<f32> = vec![];
    add_f32(a, b, &mut out);
    assert!(out.is_empty());
}

// ---------------------------------------------------------------------------
// mul_f32
// ---------------------------------------------------------------------------

#[test]
fn test_mul_f32_scalar_basic() {
    let a = [2.0, 3.0, 4.0, 5.0];
    let b = [0.5, 0.5, 0.25, 0.2];
    let mut out = [0.0f32; 4];
    mul_f32_scalar(&a, &b, &mut out);
    assert_approx(&out, &[1.0, 1.5, 1.0, 1.0], 1e-6);
}

#[test]
fn test_mul_f32_dispatch_matches_scalar() {
    let a: Vec<f32> = (0..19).map(|i| (i as f32) - 9.0).collect();
    let b: Vec<f32> = (0..19).map(|i| (i as f32 * 0.3).cos()).collect();
    let mut out_dispatch = vec![0.0f32; 19];
    let mut out_scalar = vec![0.0f32; 19];
    mul_f32(&a, &b, &mut out_dispatch);
    mul_f32_scalar(&a, &b, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-6);
}

// ---------------------------------------------------------------------------
// scalar_mul_f32
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_mul_f32_basic() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut out = [0.0f32; 9];
    scalar_mul_f32_scalar(&a, 3.0, &mut out);
    assert_approx(
        &out,
        &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0],
        1e-6,
    );
}

#[test]
fn test_scalar_mul_f32_dispatch_matches_scalar() {
    let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
    let scalar = 2.5;
    let mut out_dispatch = vec![0.0f32; 33];
    let mut out_scalar = vec![0.0f32; 33];
    scalar_mul_f32(&a, scalar, &mut out_dispatch);
    scalar_mul_f32_scalar(&a, scalar, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-6);
}

#[test]
fn test_scalar_mul_f32_zero() {
    let a = [1.0, -2.0, 3.0];
    let mut out = [99.0f32; 3];
    scalar_mul_f32(&a, 0.0, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0], 1e-6);
}

// ---------------------------------------------------------------------------
// fma_f32
// ---------------------------------------------------------------------------

#[test]
fn test_fma_f32_scalar_basic() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let c = [0.1, 0.2, 0.3, 0.4];
    let mut out = [0.0f32; 4];
    fma_f32_scalar(&a, &b, &c, &mut out);
    // a*b+c: 5.1, 12.2, 21.3, 32.4
    assert_approx(&out, &[5.1, 12.2, 21.3, 32.4], 1e-5);
}

#[test]
fn test_fma_f32_dispatch_matches_scalar() {
    let a: Vec<f32> = (0..21).map(|i| i as f32 * 0.3).collect();
    let b: Vec<f32> = (0..21).map(|i| (i as f32).sin()).collect();
    let c: Vec<f32> = (0..21).map(|i| (i as f32 * 0.7).cos()).collect();
    let mut out_dispatch = vec![0.0f32; 21];
    let mut out_scalar = vec![0.0f32; 21];
    fma_f32(&a, &b, &c, &mut out_dispatch);
    fma_f32_scalar(&a, &b, &c, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-5);
}

// ---------------------------------------------------------------------------
// relu_f32
// ---------------------------------------------------------------------------

#[test]
fn test_relu_f32_scalar_basic() {
    let x = [-3.0, -1.0, 0.0, 1.0, 3.0, -0.5, 2.0, 0.5, -4.0];
    let mut out = [0.0f32; 9];
    relu_f32_scalar(&x, &mut out);
    assert_approx(&out, &[0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.5, 0.0], 1e-6);
}

#[test]
fn test_relu_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..25).map(|i| (i as f32) - 12.0).collect();
    let mut out_dispatch = vec![0.0f32; 25];
    let mut out_scalar = vec![0.0f32; 25];
    relu_f32(&x, &mut out_dispatch);
    relu_f32_scalar(&x, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-6);
}

// ---------------------------------------------------------------------------
// gelu_f32
// ---------------------------------------------------------------------------

#[test]
fn test_gelu_f32_scalar_at_zero() {
    let x = [0.0];
    let mut out = [99.0f32];
    gelu_f32_scalar(&x, &mut out);
    assert!(
        (out[0]).abs() < 1e-6,
        "GELU(0) should be ~0, got {}",
        out[0]
    );
}

#[test]
fn test_gelu_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..20).map(|i| (i as f32) - 10.0).collect();
    let mut out_dispatch = vec![0.0f32; 20];
    let mut out_scalar = vec![0.0f32; 20];
    gelu_f32(&x, &mut out_dispatch);
    gelu_f32_scalar(&x, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-5);
}

// ---------------------------------------------------------------------------
// silu_f32
// ---------------------------------------------------------------------------

#[test]
fn test_silu_f32_scalar_at_zero() {
    let x = [0.0];
    let mut out = [99.0f32];
    silu_f32_scalar(&x, &mut out);
    assert!(
        (out[0]).abs() < 1e-6,
        "SiLU(0) should be ~0, got {}",
        out[0]
    );
}

#[test]
fn test_silu_f32_dispatch_matches_scalar() {
    let x: Vec<f32> = (0..23).map(|i| (i as f32) - 11.0).collect();
    let mut out_dispatch = vec![0.0f32; 23];
    let mut out_scalar = vec![0.0f32; 23];
    silu_f32(&x, &mut out_dispatch);
    silu_f32_scalar(&x, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-5);
}

// ---------------------------------------------------------------------------
// NEON-specific tests (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[test]
fn test_add_f32_neon_matches_scalar() {
    let a: Vec<f32> = (0..33).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..33).map(|i| -(i as f32)).collect();
    let mut out_neon = vec![0.0f32; 33];
    let mut out_scalar = vec![0.0f32; 33];
    add_f32_neon(&a, &b, &mut out_neon);
    add_f32_scalar(&a, &b, &mut out_scalar);
    assert_approx(&out_neon, &out_scalar, 1e-6);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_fma_f32_neon_matches_scalar() {
    let a: Vec<f32> = (0..15).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..15).map(|i| (i as f32).sin()).collect();
    let c: Vec<f32> = (0..15).map(|i| (i as f32 * 0.3).cos()).collect();
    let mut out_neon = vec![0.0f32; 15];
    let mut out_scalar = vec![0.0f32; 15];
    fma_f32_neon(&a, &b, &c, &mut out_neon);
    fma_f32_scalar(&a, &b, &c, &mut out_scalar);
    // FMA has slightly different rounding from mul+add, allow slightly wider tolerance.
    assert_approx(&out_neon, &out_scalar, 1e-5);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_scalar_mul_f32_neon_matches_scalar() {
    let a: Vec<f32> = (0..17).map(|i| (i as f32) - 8.0).collect();
    let mut out_neon = vec![0.0f32; 17];
    let mut out_scalar = vec![0.0f32; 17];
    scalar_mul_f32_neon(&a, -2.0, &mut out_neon);
    scalar_mul_f32_scalar(&a, -2.0, &mut out_scalar);
    assert_approx(&out_neon, &out_scalar, 1e-6);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_relu_f32_neon_matches_scalar() {
    let x: Vec<f32> = (0..13).map(|i| (i as f32) - 6.0).collect();
    let mut out_neon = vec![0.0f32; 13];
    let mut out_scalar = vec![0.0f32; 13];
    relu_f32_neon(&x, &mut out_neon);
    relu_f32_scalar(&x, &mut out_scalar);
    assert_approx(&out_neon, &out_scalar, 1e-6);
}

// ---------------------------------------------------------------------------
// Large input (exercises main SIMD loop + scalar tail)
// ---------------------------------------------------------------------------

#[test]
fn test_add_f32_large_input() {
    let n = 1024 + 7; // not multiple of 4 or 8
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).sin()).collect();
    let mut out_dispatch = vec![0.0f32; n];
    let mut out_scalar = vec![0.0f32; n];
    add_f32(&a, &b, &mut out_dispatch);
    add_f32_scalar(&a, &b, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-6);
}

#[test]
fn test_fma_f32_large_input() {
    let n = 512 + 3;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
    let c: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let mut out_dispatch = vec![0.0f32; n];
    let mut out_scalar = vec![0.0f32; n];
    fma_f32(&a, &b, &c, &mut out_dispatch);
    fma_f32_scalar(&a, &b, &c, &mut out_scalar);
    assert_approx(&out_dispatch, &out_scalar, 1e-5);
}
