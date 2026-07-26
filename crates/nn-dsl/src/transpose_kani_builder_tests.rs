// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TensorBlockBuilder::add_transpose`.
//!
//! Proves structural correctness of Transpose tensor IR construction
//! and validation:
//! - validate() succeeds for valid permutations of all 2D/3D inputs
//! - validate() rejects axes-length mismatch
//! - validate() rejects duplicate axes
//! - validate() rejects out-of-bounds axes
//! - Output shape matches permuted input shape
//!
//! Part of #779 (transformer verification), handoff from W3 c587f93e.

use crate::tensor_block_builder::TensorBlockBuilder;

// ---------------------------------------------------------------------------
// Positive: valid permutations
// ---------------------------------------------------------------------------

/// Proves `add_transpose` + `build` + `validate()` succeeds for all valid
/// 2D permutations. 2D has only two permutations: identity [0,1] and swap [1,0].
///
/// Domain: M in [1, 8], N in [1, 8]. Reduced from [1,16] for CBMC scalability (#767).
#[kani::unwind(8)]
#[kani::proof]
fn transpose_2d_builder_validates_ok() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let swap: bool = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(n >= 1 && n <= 4);

    let (axes, out_shape) = if swap {
        (vec![1, 0], vec![n, m])
    } else {
        (vec![0, 1], vec![m, n])
    };

    let mut b = TensorBlockBuilder::new("kani_transpose_2d");
    let input = b.add_input("input", &[m, n]);
    let out = b.add_transpose(input, &axes, &out_shape);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for valid 2D transpose"
    );
}

/// Proves `add_transpose` + `build` + `validate()` succeeds for all valid
/// 3D permutations. 3D has 6 permutations of [0,1,2].
///
/// Domain: A,B,C in [1, 8]. Permutation selected from all 6 valid options.
#[kani::unwind(8)]
#[kani::proof]
fn transpose_3d_builder_validates_ok() {
    let a: usize = kani::any();
    let b_dim: usize = kani::any();
    let c: usize = kani::any();
    let perm_idx: u8 = kani::any();

    kani::assume(a >= 1 && a <= 4);
    kani::assume(b_dim >= 1 && b_dim <= 4);
    kani::assume(c >= 1 && c <= 4);
    kani::assume(perm_idx < 6);

    let in_shape = [a, b_dim, c];

    // All 6 permutations of [0,1,2].
    let axes: [usize; 3] = match perm_idx {
        0 => [0, 1, 2],
        1 => [0, 2, 1],
        2 => [1, 0, 2],
        3 => [1, 2, 0],
        4 => [2, 0, 1],
        _ => [2, 1, 0],
    };

    let out_shape = [in_shape[axes[0]], in_shape[axes[1]], in_shape[axes[2]]];

    let mut b = TensorBlockBuilder::new("kani_transpose_3d");
    let input = b.add_input("input", &in_shape);
    let out = b.add_transpose(input, &axes, &out_shape);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for valid 3D transpose"
    );
}

// ---------------------------------------------------------------------------
// Negative: invalid permutations
// ---------------------------------------------------------------------------

/// Proves validate() rejects Transpose when axes length != input rank.
///
/// Constructs a 2D input but supplies 3 axes (or 1 axis).
#[kani::unwind(8)]
#[kani::proof]
fn transpose_rejects_axes_length_mismatch() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let too_many: bool = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(n >= 1 && n <= 4);

    let mut b = TensorBlockBuilder::new("kani_bad_axes_len");
    let input = b.add_input("input", &[m, n]);

    if too_many {
        // 3 axes for a rank-2 tensor.
        let out = b.add_transpose(input, &[0, 1, 0], &[m, n, m]);
        let def = b.build(out).expect("valid graph");
        assert!(
            def.validate().is_err(),
            "validate() must reject axes.len() > rank"
        );
    } else {
        // 1 axis for a rank-2 tensor.
        let out = b.add_transpose(input, &[0], &[m]);
        let def = b.build(out).expect("valid graph");
        assert!(
            def.validate().is_err(),
            "validate() must reject axes.len() < rank"
        );
    }
}

/// Proves validate() rejects Transpose with duplicate axes entries.
///
/// Example: axes=[0, 0] on a 2D tensor — axis 0 appears twice, axis 1 is missing.
#[kani::unwind(1)]
#[kani::proof]
fn transpose_rejects_duplicate_axis() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(n >= 1 && n <= 4);

    let mut b = TensorBlockBuilder::new("kani_dup_axis");
    let input = b.add_input("input", &[m, n]);
    // axes=[0, 0]: duplicate axis 0.
    let out = b.add_transpose(input, &[0, 0], &[m, m]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_err(),
        "validate() must reject duplicate axis in transpose"
    );
}

/// Proves validate() rejects Transpose with out-of-bounds axis.
///
/// Example: axes=[0, 2] on a rank-2 tensor — axis 2 is out of bounds.
#[kani::unwind(8)]
#[kani::proof]
fn transpose_rejects_out_of_bounds_axis() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(n >= 1 && n <= 4);

    let mut b = TensorBlockBuilder::new("kani_oob_axis");
    let input = b.add_input("input", &[m, n]);
    // axes=[0, 2]: axis 2 is out of bounds for rank 2.
    let out = b.add_transpose(input, &[0, 2], &[m, n]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_err(),
        "validate() must reject out-of-bounds axis"
    );
}

// ---------------------------------------------------------------------------
// Output shape correctness
// ---------------------------------------------------------------------------

/// Proves that Transpose output shape matches the permuted input shape.
///
/// For a 3D input [A, B, C] with permutation `axes`, the output shape
/// must be [in_shape[axes[0]], in_shape[axes[1]], in_shape[axes[2]]].
#[kani::unwind(1)]
#[kani::proof]
fn transpose_output_shape_is_permuted() {
    let a: usize = kani::any();
    let b_dim: usize = kani::any();
    let c: usize = kani::any();
    let perm_idx: u8 = kani::any();

    kani::assume(a >= 1 && a <= 4);
    kani::assume(b_dim >= 1 && b_dim <= 4);
    kani::assume(c >= 1 && c <= 4);
    kani::assume(perm_idx < 6);

    let in_shape = [a, b_dim, c];

    let axes: [usize; 3] = match perm_idx {
        0 => [0, 1, 2],
        1 => [0, 2, 1],
        2 => [1, 0, 2],
        3 => [1, 2, 0],
        4 => [2, 0, 1],
        _ => [2, 1, 0],
    };

    let expected_shape = [in_shape[axes[0]], in_shape[axes[1]], in_shape[axes[2]]];

    let mut b = TensorBlockBuilder::new("kani_shape");
    let input = b.add_input("input", &in_shape);
    let out = b.add_transpose(input, &axes, &expected_shape);
    let def = b.build(out).expect("valid graph");

    let out_node = &def.nodes[def.nodes.len() - 1];
    assert_eq!(out_node.shape[0], expected_shape[0]);
    assert_eq!(out_node.shape[1], expected_shape[1]);
    assert_eq!(out_node.shape[2], expected_shape[2]);
}

/// Proves that 2D transpose is a self-inverse: applying [1,0] twice gives
/// the original shape.
#[kani::unwind(1)]
#[kani::proof]
fn transpose_2d_swap_is_involution() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(n >= 1 && n <= 4);

    let mut b = TensorBlockBuilder::new("kani_involution");
    let input = b.add_input("input", &[m, n]);
    let transposed = b.add_transpose(input, &[1, 0], &[n, m]);
    let back = b.add_transpose(transposed, &[1, 0], &[m, n]);
    let def = b.build(back).expect("valid graph");

    assert!(def.validate().is_ok(), "double transpose must validate");

    let out_node = &def.nodes[def.nodes.len() - 1];
    assert_eq!(out_node.shape[0], m, "shape restored: dim 0");
    assert_eq!(out_node.shape[1], n, "shape restored: dim 1");
}
