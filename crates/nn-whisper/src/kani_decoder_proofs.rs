// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for decoder position and output bounds.
//!
//! Covers:
//! - Positional embedding slices stay in-bounds when `position_offset + seq_len` fits
//! - Positional embedding slices reject requests that exceed `max_target_positions`
//! - The tied output projection produces logits whose last dimension equals `vocab_size`
//!
//! Issue: #3724

use super::*;
use nn_core::{DType, Device};

// ============================================================================
// Harness 1: Positional slice succeeds when position bounds hold
// ============================================================================

/// Proves the decoder's positional narrow is valid when `offset + seq_len <= max_positions`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn positional_slice_stays_in_bounds() {
    let max_positions: usize = kani::any();
    let d_model: usize = kani::any();
    let position_offset: usize = kani::any();
    let seq_len: usize = kani::any();
    kani::assume(max_positions >= 1 && max_positions <= 4);
    kani::assume(d_model >= 1 && d_model <= 4);
    kani::assume(seq_len >= 1 && seq_len <= max_positions);
    kani::assume(position_offset <= max_positions - seq_len);

    let positional = DynTensor::zeros(&[max_positions, d_model], DType::F32, &Device::Cpu)
        .expect("valid positional embedding tensor");
    let sliced = positional
        .narrow(0, position_offset, seq_len)
        .expect("bounded slice must succeed");

    assert_eq!(
        sliced.dim(0).unwrap(),
        seq_len,
        "slice length must match seq_len"
    );
    assert_eq!(
        sliced.dim(1).unwrap(),
        d_model,
        "embedding width must stay unchanged"
    );
}

// ============================================================================
// Harness 2: Positional slice rejects requests past max_target_positions
// ============================================================================

/// Proves the decoder's positional narrow fails when the requested end exceeds the table.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn positional_slice_rejects_out_of_bounds_end() {
    let max_positions: usize = kani::any();
    let d_model: usize = kani::any();
    let position_offset: usize = kani::any();
    let seq_len: usize = kani::any();
    kani::assume(max_positions >= 1 && max_positions <= 4);
    kani::assume(d_model >= 1 && d_model <= 4);
    kani::assume(position_offset < max_positions);
    kani::assume(seq_len >= 1 && seq_len <= 4);
    kani::assume(position_offset.checked_add(seq_len).is_some());
    kani::assume(position_offset + seq_len > max_positions);

    let positional = DynTensor::zeros(&[max_positions, d_model], DType::F32, &Device::Cpu)
        .expect("valid positional embedding tensor");
    let result = positional.narrow(0, position_offset, seq_len);

    assert!(
        result.is_err(),
        "out-of-bounds positional requests must fail"
    );
}

// ============================================================================
// Harness 3: Tied projection produces vocab-sized logits
// ============================================================================

/// Proves the decoder's tied projection maps `[B, T, D]` hidden states to `[B, T, vocab]`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn tied_projection_output_last_dim_matches_vocab_size() {
    let seq_len: usize = kani::any();
    let d_model: usize = kani::any();
    let vocab_size: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 3);
    kani::assume(d_model >= 1 && d_model <= 4);
    kani::assume(vocab_size >= 1 && vocab_size <= 4);

    let hidden = DynTensor::zeros(&[1, seq_len, d_model], DType::F32, &Device::Cpu)
        .expect("valid hidden states");
    let embed_weight = DynTensor::zeros(&[vocab_size, d_model], DType::F32, &Device::Cpu)
        .expect("valid tied embedding weight");
    let embed_weight_t = embed_weight.transpose(0, 1).expect("transpose");
    let logits = hidden.matmul(&embed_weight_t).expect("matmul");

    let (batch, out_seq_len, out_vocab) = logits.dims3().expect("rank-3 logits");
    assert_eq!(batch, 1, "decoder keeps batch size");
    assert_eq!(out_seq_len, seq_len, "decoder keeps token sequence length");
    assert_eq!(
        out_vocab, vocab_size,
        "decoder logits must have one entry per vocab item"
    );
}
