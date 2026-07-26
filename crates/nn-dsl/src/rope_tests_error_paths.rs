// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RoPE (K6) scalar error-path tests (extracted from rope_tests_bounds.rs).
//!
//! Tests NaN/Inf rejection and overflow detection for rope_cos_scalar and
//! rope_sin_scalar.

use super::*;

// --- rope_cos_scalar / rope_sin_scalar error path tests ---

#[test]
fn test_rope_cos_scalar_nan_input_returns_err() {
    use crate::kernel_error::KernelError;
    let err = rope_cos_scalar(f32::NAN, 1.0, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x0", .. }),
        "expected NonFiniteInput for x0, got {err:?}"
    );
    let err = rope_cos_scalar(1.0, f32::NAN, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x1", .. }),
        "expected NonFiniteInput for x1, got {err:?}"
    );
    let err = rope_cos_scalar(1.0, 1.0, f32::NAN).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "freq", .. }),
        "expected NonFiniteInput for freq, got {err:?}"
    );
}

#[test]
fn test_rope_sin_scalar_nan_input_returns_err() {
    use crate::kernel_error::KernelError;
    let err = rope_sin_scalar(f32::NAN, 1.0, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x0", .. }),
        "expected NonFiniteInput for x0, got {err:?}"
    );
    let err = rope_sin_scalar(1.0, f32::NAN, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x1", .. }),
        "expected NonFiniteInput for x1, got {err:?}"
    );
    let err = rope_sin_scalar(1.0, 1.0, f32::NAN).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "freq", .. }),
        "expected NonFiniteInput for freq, got {err:?}"
    );
}

#[test]
fn test_rope_cos_scalar_inf_input_returns_err() {
    use crate::kernel_error::KernelError;
    let err = rope_cos_scalar(f32::INFINITY, 1.0, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x0", .. }),
        "expected NonFiniteInput for x0, got {err:?}"
    );
    let err = rope_cos_scalar(1.0, f32::NEG_INFINITY, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x1", .. }),
        "expected NonFiniteInput for x1, got {err:?}"
    );
}

#[test]
fn test_rope_sin_scalar_inf_input_returns_err() {
    use crate::kernel_error::KernelError;
    let err = rope_sin_scalar(f32::INFINITY, 1.0, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "x0", .. }),
        "expected NonFiniteInput for x0, got {err:?}"
    );
    let err = rope_sin_scalar(1.0, 1.0, f32::NEG_INFINITY).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteInput { name: "freq", .. }),
        "expected NonFiniteInput for freq, got {err:?}"
    );
}

#[test]
fn test_rope_cos_scalar_overflow_returns_err() {
    use crate::kernel_error::KernelError;
    // x0 * cos(freq) - x1 * sin(freq) can overflow when terms add constructively.
    // cos(1)≈0.54, sin(1)≈0.84 → MAX*0.54 - (-MAX)*0.84 = MAX*1.38 → Inf
    let err = rope_cos_scalar(f32::MAX, -f32::MAX, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteOutput { name: "output", .. }),
        "expected NonFiniteOutput for output, got {err:?}"
    );
}

#[test]
fn test_rope_sin_scalar_overflow_returns_err() {
    use crate::kernel_error::KernelError;
    // x0 * sin(freq) + x1 * cos(freq) can overflow for extreme values
    let err = rope_sin_scalar(f32::MAX, f32::MAX, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteOutput { name: "output", .. }),
        "expected NonFiniteOutput for output, got {err:?}"
    );
}

#[test]
fn test_rope_cos_scalar_negative_overflow_returns_err() {
    use crate::kernel_error::KernelError;
    // Negative overflow: (-MAX)*cos(1) - MAX*sin(1) = -MAX*1.38 → -Inf
    let err = rope_cos_scalar(-f32::MAX, f32::MAX, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteOutput { name: "output", .. }),
        "expected NonFiniteOutput for output (negative overflow), got {err:?}"
    );
}

#[test]
fn test_rope_sin_scalar_negative_overflow_returns_err() {
    use crate::kernel_error::KernelError;
    // Negative overflow: (-MAX)*sin(1) + (-MAX)*cos(1) = -MAX*1.38 → -Inf
    let err = rope_sin_scalar(-f32::MAX, -f32::MAX, 1.0).unwrap_err();
    assert!(
        matches!(err, KernelError::NonFiniteOutput { name: "output", .. }),
        "expected NonFiniteOutput for output (negative overflow), got {err:?}"
    );
}
