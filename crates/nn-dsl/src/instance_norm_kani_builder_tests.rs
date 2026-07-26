// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `build_instance_norm_decomposed` and
//! `build_instance_norm_decomposed_affine`.
//!
//! **Classification: structural.** These harnesses verify IR graph construction
//! (no-panic, shape preservation, error rejection), not numerical computation.
//!
//! Proves structural correctness of the decomposed InstanceNorm builders:
//! - No panics for bounded parameters (including invalid)
//! - Output shape is preserved (shape-preserving op)
//! - Zero-dimension inputs are rejected with `Err`
//!
//! Part of #752 AC1. Cleaned up in #800. Negative tests added in #955.

use super::{build_instance_norm_decomposed, build_instance_norm_decomposed_affine};

// --- Non-affine decomposed InstanceNorm ---

/// Proves `build_instance_norm_decomposed` does not panic for any bounded
/// parameter values, including invalid ones (0 dimensions) that return `Err`.
#[kani::unwind(1)]
#[kani::proof]
fn instance_norm_decomposed_build_no_panic() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b <= 4);
    kani::assume(c <= 4);
    kani::assume(t <= 4);

    let _ = build_instance_norm_decomposed(b, c, t);
}

/// Proves the output shape matches the input shape `[B, C, T]`.
///
/// InstanceNorm is shape-preserving: normalizes over T but preserves all dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn instance_norm_decomposed_output_shape_preserved() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 4);
    kani::assume(t >= 1 && t <= 4);

    let def = build_instance_norm_decomposed(b, c, t).expect("valid params must succeed");

    let input_shape = &def.nodes[0].shape;
    let output_shape = &def.nodes[def.output.index()].shape;

    assert_eq!(
        input_shape, output_shape,
        "InstanceNorm output shape must equal input shape [B, C, T]"
    );
    assert_eq!(output_shape.len(), 3, "output rank must be 3");
    assert_eq!(output_shape[0], b, "output batch dim must equal B");
    assert_eq!(output_shape[1], c, "output channel dim must equal C");
    assert_eq!(output_shape[2], t, "output time dim must equal T");
}

// --- Affine decomposed InstanceNorm ---

/// Proves `build_instance_norm_decomposed_affine` does not panic for any
/// bounded parameter values.
#[kani::unwind(1)]
#[kani::proof]
fn instance_norm_affine_decomposed_build_no_panic() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b <= 4);
    kani::assume(c <= 4);
    kani::assume(t <= 4);

    let _ = build_instance_norm_decomposed_affine(b, c, t);
}

/// Proves the affine variant output shape matches the input shape `[B, C, T]`.
#[kani::unwind(1)]
#[kani::proof]
fn instance_norm_affine_decomposed_output_shape_preserved() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b >= 1 && b <= 4);
    kani::assume(c >= 1 && c <= 4);
    kani::assume(t >= 1 && t <= 4);

    let def = build_instance_norm_decomposed_affine(b, c, t).expect("valid params must succeed");

    let input_shape = &def.nodes[0].shape;
    let output_shape = &def.nodes[def.output.index()].shape;

    assert_eq!(
        input_shape, output_shape,
        "InstanceNorm affine output shape must equal input shape [B, C, T]"
    );
    assert_eq!(output_shape[0], b, "batch dim preserved");
    assert_eq!(output_shape[1], c, "channel dim preserved");
    assert_eq!(output_shape[2], t, "time dim preserved");
}

/// Proves `build_instance_norm_decomposed` rejects zero-dimension inputs
/// with `Err`, not a panic.
///
/// At least one of `b`, `c`, or `t` is zero. The builder must return
/// `Err(TensorIRError::KernelValidation(InvalidDimension { .. }))`.
#[kani::unwind(1)]
#[kani::proof]
fn instance_norm_decomposed_zero_dim_returns_err() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b <= 4);
    kani::assume(c <= 4);
    kani::assume(t <= 4);
    // At least one dimension is zero.
    kani::assume(b == 0 || c == 0 || t == 0);

    let result = build_instance_norm_decomposed(b, c, t);
    assert!(
        result.is_err(),
        "zero-dimension InstanceNorm must return Err, not panic"
    );
}

/// Proves `build_instance_norm_decomposed_affine` rejects zero-dimension
/// inputs with `Err`, not a panic.
#[kani::unwind(1)]
#[kani::proof]
fn instance_norm_affine_decomposed_zero_dim_returns_err() {
    let b: usize = kani::any();
    let c: usize = kani::any();
    let t: usize = kani::any();

    kani::assume(b <= 4);
    kani::assume(c <= 4);
    kani::assume(t <= 4);
    // At least one dimension is zero.
    kani::assume(b == 0 || c == 0 || t == 0);

    let result = build_instance_norm_decomposed_affine(b, c, t);
    assert!(
        result.is_err(),
        "zero-dimension affine InstanceNorm must return Err, not panic"
    );
}
