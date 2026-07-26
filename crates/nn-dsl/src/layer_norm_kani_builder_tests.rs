// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `build_layer_norm_decomposed`.
//!
//! **Classification: structural.** These harnesses verify IR graph construction
//! (no-panic, shape preservation, error rejection), not numerical computation.
//!
//! Proves structural correctness of the decomposed LayerNorm builder:
//! - No panics for any bounded parameter combination
//! - Output shape is preserved (same as input shape `[N, hidden]`)
//! - Zero-dimension inputs are rejected with `Err`
//!
//! Part of #752 AC1. Cleaned up in #800. Negative tests added in #955.

use super::build_layer_norm_decomposed;

/// Proves `build_layer_norm_decomposed` does not panic for any bounded
/// parameter values, including invalid ones (0 dimensions) that return `Err`.
#[kani::unwind(1)]
#[kani::proof]
fn layer_norm_decomposed_build_no_panic() {
    let n: usize = kani::any();
    let hidden: usize = kani::any();

    kani::assume(n <= 4);
    kani::assume(hidden <= 4);

    // Must not panic — invalid params return Err, valid ones return Ok.
    let _ = build_layer_norm_decomposed(n, hidden);
}

/// Proves the output shape matches the input shape `[N, hidden]`.
///
/// LayerNorm is a shape-preserving operation: the output tensor has the
/// same dimensions as the input tensor.
#[kani::unwind(1)]
#[kani::proof]
fn layer_norm_decomposed_output_shape_preserved() {
    let n: usize = kani::any();
    let hidden: usize = kani::any();

    kani::assume(n >= 1 && n <= 4);
    kani::assume(hidden >= 1 && hidden <= 4);

    let def = build_layer_norm_decomposed(n, hidden).expect("valid params must succeed");

    let input_shape = &def.nodes[0].shape;
    let output_shape = &def.nodes[def.output.index()].shape;

    assert_eq!(
        input_shape, output_shape,
        "LayerNorm output shape must equal input shape [N, hidden]"
    );
    assert_eq!(output_shape.len(), 2, "output rank must be 2");
    assert_eq!(output_shape[0], n, "output dim 0 must equal N");
    assert_eq!(output_shape[1], hidden, "output dim 1 must equal hidden");
}

/// Proves `build_layer_norm_decomposed` rejects zero-dimension inputs
/// with `Err`, not a panic.
///
/// At least one of `n` or `hidden` is zero. The builder must return
/// `Err(TensorIRError::KernelValidation(InvalidDimension { .. }))`.
#[kani::unwind(1)]
#[kani::proof]
fn layer_norm_decomposed_zero_dim_returns_err() {
    let n: usize = kani::any();
    let hidden: usize = kani::any();

    kani::assume(n <= 4);
    kani::assume(hidden <= 4);
    // At least one dimension is zero.
    kani::assume(n == 0 || hidden == 0);

    let result = build_layer_norm_decomposed(n, hidden);
    assert!(
        result.is_err(),
        "zero-dimension LayerNorm must return Err, not panic"
    );
}
