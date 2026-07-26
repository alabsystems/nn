// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU/CPU forward parity and validation tests for DemucsTransformer.
//!
//! Extracted from `demucs_transformer_tests.rs` for file-size compliance (#1420).

use super::*;
use crate::demucs_test_common::make_transformer_weights;
use crate::test_common::{assert_close, make_cache};

// ---------------------------------------------------------------------------
// GPU/CPU forward parity tests (Part of #1372 — AC6)
// ---------------------------------------------------------------------------

/// Verify `forward_gpu()` produces the same output as `forward()`.
///
/// Both paths use the same Metal kernels — `forward()` round-trips to CPU
/// after each dispatch, while `forward_gpu()` keeps intermediates on GPU
/// buffers. Floating-point ordering may differ due to GPU sinusoidal
/// embedding and transpose dispatch paths, so we use tolerance-based
/// comparison.
///
/// Part of #1372 — AC6.
#[test]
fn test_forward_gpu_parity() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return, // Skip on non-Metal platforms
    };
    let seq_t = 16;
    let seq_s = 32;
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, seq_t, seq_s).unwrap();

    // Non-uniform input to exercise all branches meaningfully.
    // Constant inputs make transpose/LayerNorm/attention degenerate.
    let temporal: Vec<f32> = (0..BOTTLENECK_DIM * seq_t)
        .map(|i| ((i as f32) * 0.017 - 0.3).sin() * 0.1)
        .collect();
    let spectral: Vec<f32> = (0..BOTTLENECK_DIM * seq_s)
        .map(|i| ((i as f32) * 0.013 + 0.7).cos() * 0.08)
        .collect();

    let (cpu_t, cpu_s) = model
        .forward(&cache, &temporal, &spectral)
        .expect("forward (CPU round-trip)");
    let (gpu_t, gpu_s) = model
        .forward_gpu(&cache, &temporal, &spectral)
        .expect("forward_gpu (buffer-to-buffer)");

    // Output lengths must match.
    assert_eq!(cpu_t.len(), gpu_t.len(), "temporal output length mismatch");
    assert_eq!(cpu_s.len(), gpu_s.len(), "spectral output length mismatch");

    // Expected output shape: [BOTTLENECK_DIM, seq_len].
    assert_eq!(cpu_t.len(), BOTTLENECK_DIM * seq_t, "temporal output shape");
    assert_eq!(cpu_s.len(), BOTTLENECK_DIM * seq_s, "spectral output shape");

    // Tolerance: the 5-layer transformer with matmul, softmax, and LayerNorm
    // accumulates floating-point error. GPU sinusoidal embedding and transpose
    // may introduce additional ordering differences. 1e-4 allows for these
    // while catching significant divergence.
    let tol = 1e-4;
    assert_close(&cpu_t, &gpu_t, tol, "temporal forward parity");
    assert_close(&cpu_s, &gpu_s, tol, "spectral forward parity");
}

/// Verify `forward_gpu()` rejects mismatched temporal input length.
#[test]
fn test_forward_gpu_wrong_temporal_len() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, 16, 32).unwrap();

    let bad_temporal = vec![0.1f32; 100]; // wrong size
    let spectral = vec![0.1f32; BOTTLENECK_DIM * 32];
    let result = model.forward_gpu(&cache, &bad_temporal, &spectral);
    assert!(result.is_err(), "should reject wrong temporal length");
}

/// Verify `forward_gpu()` rejects mismatched spectral input length.
#[test]
fn test_forward_gpu_wrong_spectral_len() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, 16, 32).unwrap();

    let temporal = vec![0.1f32; BOTTLENECK_DIM * 16];
    let bad_spectral = vec![0.1f32; 100]; // wrong size
    let result = model.forward_gpu(&cache, &temporal, &bad_spectral);
    assert!(result.is_err(), "should reject wrong spectral length");
}

/// Verify `forward_gpu()` rejects NaN in temporal input.
#[test]
fn test_forward_gpu_nan_temporal_rejected() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, 16, 32).unwrap();

    let mut temporal = vec![0.1f32; BOTTLENECK_DIM * 16];
    temporal[0] = f32::NAN; // inject NaN
    let spectral = vec![0.1f32; BOTTLENECK_DIM * 32];
    let result = model.forward_gpu(&cache, &temporal, &spectral);
    assert!(result.is_err(), "forward_gpu should reject NaN input");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-finite") || msg.contains("NaN") || msg.contains("NonFinite"),
        "error should mention non-finite values: {msg}"
    );
}

/// Verify `forward_gpu()` rejects Inf in spectral input.
#[test]
fn test_forward_gpu_inf_spectral_rejected() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, 16, 32).unwrap();

    let temporal = vec![0.1f32; BOTTLENECK_DIM * 16];
    let mut spectral = vec![0.1f32; BOTTLENECK_DIM * 32];
    spectral[5] = f32::INFINITY; // inject Inf
    let result = model.forward_gpu(&cache, &temporal, &spectral);
    assert!(result.is_err(), "forward_gpu should reject Inf input");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-finite") || msg.contains("Inf") || msg.contains("NonFinite"),
        "error should mention non-finite values: {msg}"
    );
}

/// Verify CPU `forward()` also rejects NaN input.
#[test]
fn test_forward_cpu_nan_rejected() {
    let cache = match make_cache() {
        Some(c) => c,
        None => return,
    };
    let weights = make_transformer_weights();
    let model = DemucsTransformer::new(weights, 16, 32).unwrap();

    let mut temporal = vec![0.1f32; BOTTLENECK_DIM * 16];
    temporal[0] = f32::NAN;
    let spectral = vec![0.1f32; BOTTLENECK_DIM * 32];
    let result = model.forward(&cache, &temporal, &spectral);
    assert!(result.is_err(), "forward should reject NaN input");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-finite") || msg.contains("NaN") || msg.contains("NonFinite"),
        "error should mention non-finite values: {msg}"
    );
}
