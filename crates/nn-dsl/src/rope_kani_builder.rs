// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `build_rope_rotate_kernel` tensor builder.
//!
//! **Classification: structural.** These harnesses verify IR graph construction
//! (no-panic, shape preservation, error rejection), not numerical computation.
//!
//! Part of #659 AC3. Redundant shape_positive removed in #955. Negative tests
//! (odd head_dim, zero-dim) added in #955.
//!
//! These harnesses do NOT require stubbing (no trig functions in the builder).

use super::build_rope_rotate_kernel;

/// Proves `build_rope_rotate_kernel` never panics for any bounded parameter inputs.
///
/// Calls the actual production function to verify that all checked arithmetic,
/// dimension validation, and Result returns handle the parameter space without panics.
#[kani::unwind(1)]
#[kani::proof]
fn rope_rotate_build_no_panic() {
    let bh: usize = kani::any();
    let seq_len: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(bh <= 4);
    kani::assume(seq_len <= 4);
    kani::assume(head_dim <= 4);

    // Call the actual production function — must not panic.
    let _ = build_rope_rotate_kernel(bh, seq_len, head_dim);
}

/// Proves that `build_rope_rotate_kernel` output shape equals the input shape.
///
/// Rotation is a unitary transform — it preserves shape `[BH, S, D]`.
/// This catches bugs in the reshape/axis-select/stack/reshape pipeline that
/// could silently alter dimensions (e.g., D/2 instead of D in the output).
#[kani::unwind(1)]
#[kani::proof]
fn rope_rotate_output_shape_preserved() {
    let bh: usize = kani::any();
    let seq_len: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(bh >= 1 && bh <= 4);
    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(head_dim >= 2 && head_dim <= 4);
    kani::assume(head_dim % 2 == 0);

    if let Ok(def) = build_rope_rotate_kernel(bh, seq_len, head_dim) {
        // Input node 0 has shape [BH, S, D]
        let input_shape = &def.nodes[0].shape;
        // Output node (last) must have the same shape
        let output_shape = &def.nodes[def.output.index()].shape;

        assert_eq!(
            input_shape, output_shape,
            "RoPE rotation must preserve input shape [BH, S, D]"
        );
        // Also verify against the expected dimensions directly
        assert_eq!(output_shape[0], bh, "batch*heads dimension preserved");
        assert_eq!(output_shape[1], seq_len, "sequence length preserved");
        assert_eq!(output_shape[2], head_dim, "head dimension preserved");
    }
}

/// Proves `build_rope_rotate_kernel` rejects odd `head_dim` with `Err`,
/// not a panic.
///
/// RoPE splits the head dimension into even/odd pairs (`head_dim / 2`),
/// so `head_dim` must be even. This harness verifies that odd values
/// are rejected for all valid `bh` and `seq_len` combinations.
#[kani::unwind(1)]
#[kani::proof]
fn rope_rotate_rejects_odd_head_dim() {
    let bh: usize = kani::any();
    let seq_len: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(bh >= 1 && bh <= 4);
    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(head_dim >= 1 && head_dim <= 4);
    kani::assume(head_dim % 2 != 0);

    let result = build_rope_rotate_kernel(bh, seq_len, head_dim);
    assert!(result.is_err(), "odd head_dim must return Err, not panic");
}

/// Proves `build_rope_rotate_kernel` rejects zero-dimension inputs
/// with `Err`, not a panic.
#[kani::unwind(1)]
#[kani::proof]
fn rope_rotate_zero_dim_returns_err() {
    let bh: usize = kani::any();
    let seq_len: usize = kani::any();
    let head_dim: usize = kani::any();

    kani::assume(bh <= 4);
    kani::assume(seq_len <= 4);
    kani::assume(head_dim <= 4);
    // At least one dimension is zero.
    kani::assume(bh == 0 || seq_len == 0 || head_dim == 0);

    let result = build_rope_rotate_kernel(bh, seq_len, head_dim);
    assert!(
        result.is_err(),
        "zero-dimension RoPE must return Err, not panic"
    );
}
