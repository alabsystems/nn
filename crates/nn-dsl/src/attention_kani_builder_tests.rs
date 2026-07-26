// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TensorBlockBuilder::add_attention`.
//!
//! Proves structural correctness of Attention tensor IR construction:
//! - validate() succeeds for all valid bounded parameters (explicit call)
//! - validate() rejects head dimension mismatch Q[-1] != K[-1] (negative case)
//!
//! Part of #729 (dvoice epic). Cleaned up in #800.

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::AttentionMask;

/// Proves `add_attention` + `build` + explicit `validate()` succeeds for all valid inputs.
///
/// Domain: T in [1, 4], D_k in [1, 4], D_v in [1, 4].
/// Reduced from [1,8] for CBMC scalability — Vec heap reasoning bottleneck (#767).
/// Makes validate() proof obligation explicit, independent of debug_assert compilation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(16)]
fn attention_builder_validates_ok() {
    let seq_len: usize = kani::any();
    let d_k: usize = kani::any();
    let d_v: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(d_k >= 1 && d_k <= 4);
    kani::assume(d_v >= 1 && d_v <= 4);

    let mut b = TensorBlockBuilder::new("kani_attn");
    let q = b.add_input("query", &[seq_len, d_k]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let v = b.add_input("value", &[seq_len, d_v]);
    let out = b.add_attention(q, k, v, AttentionMask::Standard, None, &[seq_len, d_v]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed Attention"
    );
}

/// Proves validate() rejects Attention with head dimension mismatch (Q[-1] != K[-1]).
///
/// Constructs Q: [T, D_q], K: [T, D_k] where D_q != D_k.
/// Verifies that validation detects the incompatible head dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(16)]
fn attention_builder_rejects_head_dim_mismatch() {
    let seq_len: usize = kani::any();
    let d_q: usize = kani::any();
    let d_k: usize = kani::any();
    let d_v: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(d_q >= 1 && d_q <= 4);
    kani::assume(d_k >= 1 && d_k <= 4);
    kani::assume(d_v >= 1 && d_v <= 4);
    kani::assume(d_q != d_k);

    let mut b = TensorBlockBuilder::new("kani_attn_bad");
    let q = b.add_input("query", &[seq_len, d_q]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let v = b.add_input("value", &[seq_len, d_v]);
    let out = b.add_attention(q, k, v, AttentionMask::Standard, None, &[seq_len, d_v]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_err(),
        "validate() must reject Attention with Q head dim != K head dim"
    );
}
