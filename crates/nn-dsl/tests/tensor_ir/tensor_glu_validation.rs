// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLU validation tests: boundary conditions and dimension safety.
//!
//! Part of P20 algorithm audit for #660. Verifies that GLU decomposition
//! handles edge cases correctly.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorOpKind;

/// GLU with odd input dimension must return an error (not silently drop elements).
///
/// Without the validation in `add_glu`, integer division `7/2 = 3` causes
/// narrow ops to cover [0,3) and [3,6), silently dropping element 6.
/// PyTorch GLU also rejects odd dimensions with an error.
#[test]
fn test_glu_odd_dimension_returns_error() {
    let mut b = TensorBlockBuilder::new("glu_odd");
    let x = b.add_input("x", &[7, 16]);
    let result = b.add_glu(x, 0, &[7, 16]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("even dimension"),
        "error should mention even dimension: {err}"
    );
}

/// GLU narrow ops must cover the entire input dimension (no silent data loss).
#[test]
fn test_glu_narrow_ops_cover_full_input() {
    let mut b = TensorBlockBuilder::new("glu_full");
    let x = b.add_input("x", &[8, 16]);
    let glu = b.add_glu(x, 0, &[8, 16]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    let narrows: Vec<_> = def
        .nodes
        .iter()
        .filter(|n| matches!(&n.kind, TensorOpKind::Narrow { .. }))
        .collect();
    assert_eq!(narrows.len(), 2, "GLU should produce exactly 2 narrow ops");

    match (&narrows[0].kind, &narrows[1].kind) {
        (
            TensorOpKind::Narrow {
                start: s1,
                length: l1,
                ..
            },
            TensorOpKind::Narrow {
                start: s2,
                length: l2,
                ..
            },
        ) => {
            assert_eq!(*s1, 0, "data narrow must start at 0");
            assert_eq!(*s2, *l1, "gate narrow must start where data ends");
            assert_eq!(
                *s2 + *l2,
                8,
                "narrow ops must cover full input dim: {s2}+{l2} != 8"
            );
        }
        _ => unreachable!("filtered for Narrow ops"),
    }
}

/// GLU along axis=1 produces correct narrow decomposition.
#[test]
fn test_glu_axis1_narrow_coverage() {
    let mut b = TensorBlockBuilder::new("glu_ax1");
    let x = b.add_input("x", &[4, 32]);
    let glu = b.add_glu(x, 1, &[4, 32]).expect("even dim");
    let def = b.build(glu).expect("valid graph");

    // Output shape: [4, 16] (axis 1 halved from 32 to 16)
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 16]);

    // Verify narrow ops split along axis 1
    let narrows: Vec<_> = def
        .nodes
        .iter()
        .filter(|n| matches!(&n.kind, TensorOpKind::Narrow { .. }))
        .collect();
    match (&narrows[0].kind, &narrows[1].kind) {
        (
            TensorOpKind::Narrow {
                axis: a1,
                start: s1,
                length: l1,
                ..
            },
            TensorOpKind::Narrow {
                axis: a2,
                start: s2,
                length: l2,
                ..
            },
        ) => {
            assert_eq!(*a1, 1);
            assert_eq!(*a2, 1);
            assert_eq!(*s1, 0);
            assert_eq!(*l1, 16);
            assert_eq!(*s2, 16);
            assert_eq!(*l2, 16);
        }
        _ => unreachable!(),
    }
}
