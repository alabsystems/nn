// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `TensorBlockBuilder::add_embedding`.
//!
//! Proves structural correctness of Embedding tensor IR construction:
//! - validate() succeeds for all valid bounded parameters (explicit call)
//! - validate() rejects non-matrix weight (negative case)
//!
//! Part of #729 (dvoice epic). Cleaned up in #800.

use crate::tensor_block_builder::TensorBlockBuilder;

/// Proves `add_embedding` + `build` + explicit `validate()` succeeds for all valid inputs.
///
/// Domain: seq_len in [1, 8], num_embeddings in [1, 8], embed_dim in [1, 8].
/// Makes validate() proof obligation explicit, independent of debug_assert compilation.
#[kani::unwind(8)]
#[kani::proof]
fn embedding_builder_validates_ok() {
    let seq_len: usize = kani::any();
    let num_embeddings: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(num_embeddings >= 1 && num_embeddings <= 4);
    kani::assume(embed_dim >= 1 && embed_dim <= 4);

    let mut b = TensorBlockBuilder::new("kani_embed");
    let input = b.add_input("indices", &[seq_len]);
    let weight = b.add_input("weight", &[num_embeddings, embed_dim]);
    let out = b.add_embedding(input, weight, &[seq_len, embed_dim]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_ok(),
        "validate() must pass for well-formed Embedding"
    );
}

/// Proves validate() rejects Embedding with non-matrix weight (rank != 2).
///
/// Constructs weight with rank 1 (just [embed_dim]) instead of [num_embeddings, embed_dim].
/// Verifies that validation detects the wrong weight rank.
#[kani::unwind(1)]
#[kani::proof]
fn embedding_builder_rejects_weight_not_matrix() {
    let seq_len: usize = kani::any();
    let embed_dim: usize = kani::any();

    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(embed_dim >= 1 && embed_dim <= 4);

    let mut b = TensorBlockBuilder::new("kani_embed_bad");
    let input = b.add_input("indices", &[seq_len]);
    // Deliberately create a 1-D weight instead of 2-D matrix
    let weight = b.add_input("weight_1d", &[embed_dim]);
    let out = b.add_embedding(input, weight, &[seq_len, embed_dim]);
    let def = b.build(out).expect("valid graph");

    assert!(
        def.validate().is_err(),
        "validate() must reject Embedding with non-matrix weight"
    );
}
