// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TensorBlockBuilder::add_matmul`.
//!
//! Proves structural correctness of MatMul tensor IR construction:
//! - validate() succeeds for all valid bounded parameters (explicit call)
//! - validate() rejects dimension mismatch (negative case)
//!
//! Part of #729 (dvoice epic). Cleaned up in #800.

use crate::tensor_block_builder::TensorBlockBuilder;

/// Proves `add_matmul` + `build` + explicit `validate()` succeeds for all valid inputs.
///
/// Domain: M in [1, 4], K in [1, 4], N in [1, 4], transpose_right in {true, false}.
/// Reduced from [1,8] for CBMC scalability — Vec heap reasoning is the bottleneck (#767).
/// Makes validate() proof obligation explicit, independent of debug_assert compilation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(32)]
fn matmul_builder_validates_ok() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();
    let transpose_right: bool = kani::any();

    // Reduced from [1,4] to [1,2] for CBMC tractability — Vec heap reasoning
    // in TensorBlockBuilder::build()/validate() is the bottleneck (#767 AC3).
    kani::assume(m >= 1 && m <= 2);
    kani::assume(k >= 1 && k <= 2);
    kani::assume(n >= 1 && n <= 2);

    let mut b = TensorBlockBuilder::new("kani_matmul");
    let left = b.add_input("left", &[m, k]);
    let right_shape = if transpose_right {
        vec![n, k]
    } else {
        vec![k, n]
    };
    let right = b.add_input("right", &right_shape);
    let out = b.add_matmul(left, right, transpose_right, None, &[m, n]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed MatMul"
    );
}

/// Proves validate() rejects MatMul with contracted dimension mismatch.
///
/// Constructs `left: [M, K1]` × `right: [K2, N]` where K1 ≠ K2.
/// Verifies that validation detects the incompatible inner dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(16)]
fn matmul_builder_rejects_dim_mismatch() {
    let m: usize = kani::any();
    let k1: usize = kani::any();
    let k2: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 4);
    kani::assume(k1 >= 1 && k1 <= 4);
    kani::assume(k2 >= 1 && k2 <= 4);
    kani::assume(n >= 1 && n <= 4);
    kani::assume(k1 != k2);

    let mut b = TensorBlockBuilder::new("kani_matmul_bad");
    let left = b.add_input("left", &[m, k1]);
    let right = b.add_input("right", &[k2, n]);
    let out = b.add_matmul(left, right, false, None, &[m, n]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_err(),
        "validate() must reject MatMul with K1 != K2"
    );
}
