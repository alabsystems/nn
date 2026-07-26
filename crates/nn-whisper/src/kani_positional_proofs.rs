// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for positional encoding utilities.
//!
//! Covers:
//! - Sinusoidal embedding values are bounded in [-1, 1]
//! - Sinusoidal embedding shape matches [length, channels]
//! - Position 0 has sin=0 and cos=1 for all frequency dimensions
//! - Causal mask is lower-triangular: 0.0 on/below diagonal, NEG_INFINITY above
//! - Causal mask size-1 has a single zero element
//!
//! Issue: #4303

use super::*;
use nn_core::{DType, Device};

// ============================================================================
// Harness 1: Sinusoidal embedding output shape
// ============================================================================

/// Proves sinusoidal_embedding produces the correct [length, channels] shape.
#[kani::unwind(1)]
#[kani::proof]
fn sinusoidal_embedding_output_shape() {
    let length: usize = kani::any();
    let channels: usize = kani::any();
    kani::assume(length >= 1 && length <= 4);
    kani::assume(channels >= 2 && channels <= 8);
    // channels must be even for the sin/cos split
    kani::assume(channels % 2 == 0);

    let emb = sinusoidal_embedding(length, channels, DType::F32, &Device::Cpu)
        .expect("valid sinusoidal embedding");
    assert_eq!(emb.dims(), &[length, channels]);
}

// ============================================================================
// Harness 2: Sinusoidal values bounded in [-1, 1]
// ============================================================================

/// Proves all sinusoidal embedding values lie in [-1.0, 1.0] (sin/cos range).
#[kani::unwind(65)]
#[kani::proof]
fn sinusoidal_embedding_values_bounded() {
    let length: usize = kani::any();
    let channels: usize = kani::any();
    kani::assume(length >= 1 && length <= 2);
    kani::assume(channels >= 2 && channels <= 4);
    kani::assume(channels % 2 == 0);

    let emb = sinusoidal_embedding(length, channels, DType::F32, &Device::Cpu)
        .expect("valid sinusoidal embedding");
    let flat = emb.to_flat_vec::<f32>().expect("extract f32 data");

    for &v in &flat {
        assert!(v.is_finite(), "sinusoidal values must be finite");
        assert!(v >= -1.0 && v <= 1.0, "sin/cos values in [-1, 1]");
    }
}

// ============================================================================
// Harness 3: Position 0 has sin=0, cos=1
// ============================================================================

/// Proves at position 0, all sin components are 0 and all cos components are 1.
#[kani::unwind(33)]
#[kani::proof]
fn sinusoidal_embedding_position_zero() {
    let channels: usize = kani::any();
    kani::assume(channels >= 2 && channels <= 4);
    kani::assume(channels % 2 == 0);
    let half = channels / 2;

    let emb = sinusoidal_embedding(1, channels, DType::F32, &Device::Cpu)
        .expect("valid sinusoidal embedding");
    let flat = emb.to_flat_vec::<f32>().expect("extract f32 data");

    // At position 0, angle = 0 for all frequencies.
    // sin(0) = 0, cos(0) = 1.
    for i in 0..half {
        assert!(
            (flat[i] - 0.0).abs() < 1e-6,
            "sin(0) must be 0"
        );
        assert!(
            (flat[half + i] - 1.0).abs() < 1e-6,
            "cos(0) must be 1"
        );
    }
}

// ============================================================================
// Harness 4: Causal mask lower-triangular property
// ============================================================================

/// Proves causal_mask is lower-triangular: 0 on/below diagonal, NEG_INFINITY above.
#[kani::unwind(17)]
#[kani::proof]
fn causal_mask_lower_triangular() {
    let size: usize = kani::any();
    kani::assume(size >= 1 && size <= 4);

    let mask = causal_mask(size, DType::F32, &Device::Cpu)
        .expect("valid causal mask");
    let flat = mask.to_flat_vec::<f32>().expect("extract f32 data");

    assert_eq!(flat.len(), size * size);

    for i in 0..size {
        for j in 0..size {
            let val = flat[i * size + j];
            if j <= i {
                assert_eq!(val, 0.0, "mask[{i}][{j}] should be 0 (attend)");
            } else {
                assert_eq!(
                    val,
                    f32::NEG_INFINITY,
                    "mask[{i}][{j}] should be -inf (block)"
                );
            }
        }
    }
}

// ============================================================================
// Harness 5: Causal mask shape
// ============================================================================

/// Proves causal_mask produces [size, size] output.
#[kani::unwind(1)]
#[kani::proof]
fn causal_mask_output_shape() {
    let size: usize = kani::any();
    kani::assume(size >= 1 && size <= 8);

    let mask = causal_mask(size, DType::F32, &Device::Cpu)
        .expect("valid causal mask");
    assert_eq!(mask.dims(), &[size, size]);
}

// ============================================================================
// Harness 6: Causal mask diagonal is always zero
// ============================================================================

/// Proves every diagonal element of the causal mask is 0.0 (self-attend).
#[kani::unwind(9)]
#[kani::proof]
fn causal_mask_diagonal_zero() {
    let size: usize = kani::any();
    kani::assume(size >= 1 && size <= 4);

    let mask = causal_mask(size, DType::F32, &Device::Cpu)
        .expect("valid causal mask");
    let flat = mask.to_flat_vec::<f32>().expect("extract f32 data");

    for i in 0..size {
        assert_eq!(
            flat[i * size + i],
            0.0,
            "diagonal element mask[{i}][{i}] must be 0"
        );
    }
}
