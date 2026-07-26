// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, unreachable_pub, clippy::duplicated_attributes)]

//! Builder helpers for cross-attention monotonicity verification.
//!
//! Constructs a NY graph for pre-softmax attention scores:
//!
//!   Q(Variable) → Linear(W_q) → Reshape → Transpose → [H, T, d_k]
//!   K(Constant)  → Linear(W_k) → Reshape → Transpose → [H, T, d_k]
//!   Scores = Q_proj @ K_proj^T / √d_k → [H, T, T]
//!
//! The output is the raw attention score matrix (pre-softmax). Diagonal
//! dominance of this matrix is a sufficient condition for monotonic
//! attention: softmax concentrates mass on the diagonal when diagonal
//! elements are largest.
//!
//! Two configurations are provided:
//!
//! 1. `build_attention_scores_simple`: Direct `Q @ K^T / √d` without
//!    projections. Q is Variable, K is ConstantTensor with identity-like
//!    structure to encourage diagonal dominance.
//!
//! 2. `build_attention_scores_projected`: Full linear projections
//!    (W_q, W_k) before score computation. Tests whether the projection
//!    step preserves or destroys diagonal dominance.
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 2.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Sequence length (decoder steps = encoder positions for square attention).
pub(super) const SEQ_LEN: usize = 4;

/// Model/embedding dimension.
pub(super) const D_MODEL: usize = 8;

/// Per-head dimension for projected variant.
pub(super) const HEAD_DIM: usize = 4;

/// Number of attention heads for projected variant.
pub(super) const NUM_HEADS: usize = 2;

/// Weight magnitude for linear projections.
const W_SCALE: f32 = 0.1;

// ---------------------------------------------------------------------------
// Simple variant: direct Q @ K^T / √d (no projections)
// ---------------------------------------------------------------------------

/// Build a graph that outputs pre-softmax attention scores: `Q @ K^T / √d`.
///
/// Q: Variable `[T, D]`, K: input `[T, D]` (bound as ConstantTensor).
/// Output: `[T, T]` attention score matrix.
///
/// Returns `(def, output_shape)`.
pub(super) fn build_attention_scores_simple() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_simple");

    let q = b.add_input("query", &[SEQ_LEN, D_MODEL]);
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);

    // Scores = Q @ K^T / √D_MODEL
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(scale), &scores_shape);

    let def = b.build(scores).expect("valid attention scores graph");
    (def, scores_shape.to_vec())
}

/// Bindings for simple attention scores: Q=Variable, K=ConstantTensor.
///
/// K is constructed with identity-like structure: each row `k` has value
/// `1.0` at position `k` and `0.0` elsewhere (scaled by a factor).
/// This encourages diagonal dominance in the attention scores.
pub(super) fn attention_scores_simple_bindings() -> Vec<TensorParamBinding> {
    let mut k_data = vec![0.0f32; SEQ_LEN * D_MODEL];
    // Identity-like: row t has a "bump" at column positions [t*D/T .. (t+1)*D/T]
    // For D_MODEL=8, T=4: row 0 has 1.0 at cols 0-1, row 1 at 2-3, etc.
    let cols_per_pos = D_MODEL / SEQ_LEN;
    for t in 0..SEQ_LEN {
        for c in 0..cols_per_pos {
            let col = t * cols_per_pos + c;
            if col < D_MODEL {
                k_data[t * D_MODEL + col] = 1.0;
            }
        }
    }
    let k_tensor =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, D_MODEL]), k_data).expect("valid K shape");

    vec![
        TensorParamBinding::Variable,                 // query (Variable)
        TensorParamBinding::ConstantTensor(k_tensor), // key (ConstantTensor)
    ]
}

// ---------------------------------------------------------------------------
// Projected variant: Linear(Q) @ Linear(K)^T / √d_k with multi-head
// ---------------------------------------------------------------------------

/// Build a graph with linear projections before score computation.
///
/// Q: Variable `[T, D]`, K: input `[T, D]` (bound as ConstantTensor).
/// W_q, W_k: inputs `[D, D]` (bound as ConstantTensor).
///
/// Architecture:
///   Q_proj = Q @ W_q   → [T, D]
///   K_proj = K @ W_k   → [T, D]
///   Reshape to [T, H, d_k], Transpose to [H, T, d_k]
///   Scores = Q_proj @ K_proj^T / √d_k → [H, T, T]
///
/// Returns `(def, output_shape)`.
pub(super) fn build_attention_scores_projected() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_projected");
    let d = D_MODEL;

    let q = b.add_input("query", &[SEQ_LEN, d]);
    let k = b.add_input("key", &[SEQ_LEN, d]);
    let w_q = b.add_input("w_q", &[d, d]);
    let w_k = b.add_input("w_k", &[d, d]);

    // Project: [T, D] @ [D, D] → [T, D]
    let q_proj = b.add_matmul(q, w_q, false, None, &[SEQ_LEN, d]);
    let k_proj = b.add_matmul(k, w_k, false, None, &[SEQ_LEN, d]);

    // Reshape: [T, D] → [T, H, d_k]
    let reshaped = [SEQ_LEN, NUM_HEADS, HEAD_DIM];
    let q_r = b.add_reshape(q_proj, &reshaped);
    let k_r = b.add_reshape(k_proj, &reshaped);

    // Transpose: [T, H, d_k] → [H, T, d_k]
    let transposed = [NUM_HEADS, SEQ_LEN, HEAD_DIM];
    let q_t = b.add_transpose(q_r, &[1, 0, 2], &transposed);
    let k_t = b.add_transpose(k_r, &[1, 0, 2], &transposed);

    // Scores: [H, T, d_k] @ [H, d_k, T] = [H, T, T]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let scores_shape = [NUM_HEADS, SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q_t, k_t, true, Some(scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid projected attention scores graph");
    (def, scores_shape.to_vec())
}

/// Bindings for projected attention scores.
///
/// Q=Variable, K=ConstantTensor (identity-like), W_q=ConstantTensor (near-identity),
/// W_k=ConstantTensor (near-identity).
pub(super) fn attention_scores_projected_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;

    // K: identity-like structure (same as simple variant)
    let mut k_data = vec![0.0f32; SEQ_LEN * d];
    let cols_per_pos = d / SEQ_LEN;
    for t in 0..SEQ_LEN {
        for c in 0..cols_per_pos {
            let col = t * cols_per_pos + c;
            if col < d {
                k_data[t * d + col] = 1.0;
            }
        }
    }
    let k_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, d]), k_data).expect("valid K shape");

    // W_q, W_k: near-identity (diagonal + small noise)
    // This preserves the structure that makes diagonal dominance provable.
    let mut w_data = vec![0.0f32; d * d];
    for i in 0..d {
        w_data[i * d + i] = 1.0; // identity diagonal
    }
    // Add small off-diagonal perturbation
    for i in 0..d {
        for j in 0..d {
            if i != j {
                w_data[i * d + j] = W_SCALE * 0.01;
            }
        }
    }
    let w_tensor = ArrayD::from_shape_vec(IxDyn(&[d, d]), w_data).expect("valid W shape");

    vec![
        TensorParamBinding::Variable,                         // query (Variable)
        TensorParamBinding::ConstantTensor(k_tensor),         // key (ConstantTensor)
        TensorParamBinding::ConstantTensor(w_tensor.clone()), // w_q
        TensorParamBinding::ConstantTensor(w_tensor),         // w_k
    ]
}

// ---------------------------------------------------------------------------
// Phase 3: Parametrized bindings for input bound / K-scale sweep
// ---------------------------------------------------------------------------

/// Build identity-like K tensor with configurable scale.
///
/// Each row `t` has `k_scale` at its dedicated column block, 0.0 elsewhere.
/// Higher `k_scale` increases the signal-to-noise ratio for diagonal dominance.
pub(super) fn build_k_tensor(k_scale: f32) -> ArrayD<f32> {
    let d = D_MODEL;
    let mut k_data = vec![0.0f32; SEQ_LEN * d];
    let cols_per_pos = d / SEQ_LEN;
    for t in 0..SEQ_LEN {
        for c in 0..cols_per_pos {
            let col = t * cols_per_pos + c;
            if col < d {
                k_data[t * d + col] = k_scale;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, d]), k_data).expect("valid K shape")
}

/// Bindings for simple attention scores with configurable K scale.
///
/// Q=Variable, K=ConstantTensor with `k_scale` controlling the identity-like
/// structure amplitude. Larger `k_scale` makes diagonal dominance easier to prove.
pub(super) fn attention_scores_simple_bindings_scaled(k_scale: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(build_k_tensor(k_scale)),
    ]
}

// ---------------------------------------------------------------------------
// Phase 5: Position-aware attention (PE-constrained Q)
// ---------------------------------------------------------------------------

/// Build sinusoidal positional encoding matrix `[T, D]`.
///
/// Each row `t` encodes position `t` using the standard Transformer PE:
///   PE[t, 2i]   = sin(t / 10000^(2i/D))
///   PE[t, 2i+1] = cos(t / 10000^(2i/D))
///
/// The key property: PE vectors at different positions are approximately
/// orthogonal, so `PE[t]·PE[t]` >> `PE[t]·PE[j]` for `j ≠ t`.
/// This makes `PE @ PE^T` diagonally dominant.
pub(super) fn build_sinusoidal_pe(seq_len: usize, d_model: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for i in 0..d_model / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            data[t * d_model + 2 * i] = freq.sin() as f32;
            data[t * d_model + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE shape")
}

/// Build a graph for position-aware attention: `(hidden + PE) @ K^T / √d`.
///
/// Architecture:
///   hidden: Variable `[T, D]` (bounded perturbation)
///   pe: input `[T, D]` (bound as ConstantTensor — sinusoidal PE)
///   Q = hidden + pe   (position-aware query)
///   K: input `[T, D]` (bound as ConstantTensor — also PE-based)
///   Scores = Q @ K^T / √D → [T, T]
///
/// The insight: `Scores = (hidden + PE) @ K^T / √d`
///   = `hidden @ K^T / √d` (Variable, bounded by input_bound)
///   + `PE @ K^T / √d`     (Constant, diagonally dominant)
///
/// When the PE contribution dominates the Variable perturbation,
/// CROWN can prove diagonal dominance.
///
/// Returns `(def, output_shape)`.
pub(super) fn build_attention_scores_positional() -> (TensorKernelDef, Vec<usize>) {
    let mut b = TensorBlockBuilder::new("attn_scores_positional");

    let hidden = b.add_input("hidden", &[SEQ_LEN, D_MODEL]);
    let pe = b.add_input("pe", &[SEQ_LEN, D_MODEL]);
    let k = b.add_input("key", &[SEQ_LEN, D_MODEL]);

    // Q = hidden + PE
    let q = b.add_binary_add(hidden, pe, &[SEQ_LEN, D_MODEL]);

    // Scores = Q @ K^T / √D_MODEL
    let scale = 1.0 / (D_MODEL as f32).sqrt();
    let scores_shape = [SEQ_LEN, SEQ_LEN];
    let scores = b.add_matmul(q, k, true, Some(scale), &scores_shape);

    let def = b
        .build(scores)
        .expect("valid positional attention scores graph");
    (def, scores_shape.to_vec())
}

/// Bindings for position-aware attention scores.
///
/// hidden=Variable, pe=ConstantTensor (sinusoidal PE), K=ConstantTensor (PE).
///
/// K is set to the same sinusoidal PE as the query PE, so the constant
/// component `PE @ PE^T / √d` is the outer product of PE with itself —
/// which is diagonally dominant because sinusoidal PE vectors at different
/// positions are approximately orthogonal.
pub(super) fn attention_scores_positional_bindings() -> Vec<TensorParamBinding> {
    let pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    vec![
        TensorParamBinding::Variable,                   // hidden (Variable)
        TensorParamBinding::ConstantTensor(pe.clone()), // pe (ConstantTensor)
        TensorParamBinding::ConstantTensor(pe),         // key = PE (ConstantTensor)
    ]
}

/// Bindings with configurable PE scale for the position-aware variant.
///
/// `pe_scale` controls the amplitude of the sinusoidal PE. Higher values
/// increase the constant diagonal-dominant signal relative to the Variable
/// perturbation. With `pe_scale=1.0` and standard PE, diagonal dominance
/// may not be provable; increasing `pe_scale` should make it provable.
pub(super) fn attention_scores_positional_bindings_scaled(
    pe_scale: f32,
) -> Vec<TensorParamBinding> {
    let mut pe = build_sinusoidal_pe(SEQ_LEN, D_MODEL);
    pe.mapv_inplace(|v| v * pe_scale);
    vec![
        TensorParamBinding::Variable,                   // hidden (Variable)
        TensorParamBinding::ConstantTensor(pe.clone()), // pe (ConstantTensor)
        TensorParamBinding::ConstantTensor(pe),         // key = PE (ConstantTensor)
    ]
}
