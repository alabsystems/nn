// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, unreachable_pub, clippy::duplicated_attributes)]

//! Shared layer builders, data constructors, and measurement helpers for
//! attention layerwise/monolithic verification tests (Phases 13-16).
//!
//! Extracted from duplicated code across 4 phase files per #1978.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Layer builders
// ===========================================================================

/// Build the score computation layer: `Q @ K^T / sqrt(d) -> [T, T]`.
pub fn build_score_layer(name: &str, seq_len: usize, d_k: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let q = b.add_input("query", &[seq_len, d_k]);
    let k = b.add_input("key", &[seq_len, d_k]);
    let scale = 1.0 / (d_k as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[seq_len, seq_len]);
    b.build(scores).expect("valid score layer")
}

/// Build the softmax layer: `Softmax(Scores) -> [T, T]`.
pub fn build_softmax_layer(name: &str, seq_len: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let scores = b.add_input("scores", &[seq_len, seq_len]);
    let weights = b.add_softmax(scores, -1, &[seq_len, seq_len]);
    b.build(weights).expect("valid softmax layer")
}

/// Build the output projection layer: `Weights @ V -> [T, d_v]`.
pub fn build_output_layer(name: &str, seq_len: usize, d_v: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let w = b.add_input("weights", &[seq_len, seq_len]);
    let v = b.add_input("value", &[seq_len, d_v]);
    let output = b.add_matmul(w, v, false, None, &[seq_len, d_v]);
    b.build(output).expect("valid output layer")
}

/// Build the linear projection layer: `X @ W -> [T, d_out]`.
///
/// This is the W_q or W_k projection that maps from d_model to d_k.
/// X is Variable, W is ConstantTensor.
pub fn build_projection_layer(
    name: &str,
    seq_len: usize,
    d_in: usize,
    d_out: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[seq_len, d_in]);
    let w = b.add_input("w", &[d_in, d_out]);
    let proj = b.add_matmul(x, w, false, None, &[seq_len, d_out]);
    b.build(proj).expect("valid projection layer")
}

// ===========================================================================
// Data construction helpers
// ===========================================================================

/// Build identity-like K tensor with configurable scale.
///
/// Each position `t` has `k_scale` at its dedicated column block.
/// For d >> seq_len, each position gets d/seq_len columns of signal.
pub fn build_k_identity(seq_len: usize, d: usize, k_scale: f32) -> ArrayD<f32> {
    let mut k_data = vec![0.0f32; seq_len * d];
    let cols_per_pos = d / seq_len;
    for t in 0..seq_len {
        for c in 0..cols_per_pos {
            let col = t * cols_per_pos + c;
            if col < d {
                k_data[t * d + col] = k_scale;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d]), k_data).expect("valid K shape")
}

/// Build near-identity weight matrix for W_q or W_k projection.
///
/// Diagonal elements = `diag_scale`, off-diagonal = `off_diag_scale`.
pub fn build_near_identity_weights(
    d_in: usize,
    d_out: usize,
    diag_scale: f32,
    off_diag_scale: f32,
) -> ArrayD<f32> {
    let mut data = vec![off_diag_scale; d_in * d_out];
    let diag_len = d_in.min(d_out);
    for i in 0..diag_len {
        data[i * d_out + i] = diag_scale;
    }
    ArrayD::from_shape_vec(IxDyn(&[d_in, d_out]), data).expect("valid weight shape")
}

/// Build V tensor with position-dependent values.
pub fn build_v_tensor(seq_len: usize, d: usize, v_scale: f32) -> ArrayD<f32> {
    let data: Vec<f32> = (0..seq_len * d)
        .map(|i| v_scale * ((i % 5) as f32 - 2.0))
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d]), data).expect("valid V shape")
}

// ===========================================================================
// Measurement helpers
// ===========================================================================

/// Measure total bound width (sum of hi - lo across all elements).
pub fn measure_total_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    hi.iter().zip(lo.iter()).map(|(h, l)| h - l).sum()
}

/// Measure average bound width per element.
pub fn measure_avg_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    let n = lo.len() as f32;
    let total: f32 = hi.iter().zip(lo.iter()).map(|(h, l)| h - l).sum();
    total / n
}

/// Measure maximum bound width across all elements.
pub fn measure_max_width(bounds: &BoundedTensor) -> f32 {
    let (lo, hi) = bounds.lower_upper();
    hi.iter()
        .zip(lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max)
}

/// Count positions with provable diagonal dominance.
///
/// For each row t of the score matrix [T, T]:
///   diag dominant if lower[t,t] > max_{j!=t} upper[t,j]
pub fn count_diagonal_dominant(bounds: &BoundedTensor, seq_len: usize) -> usize {
    let (lo, hi) = bounds.lower_upper();
    let mut count = 0;
    for t in 0..seq_len {
        let diag_lo = lo[[t, t]];
        let max_offdiag_hi = (0..seq_len)
            .filter(|&j| j != t)
            .map(|j| hi[[t, j]])
            .fold(f32::NEG_INFINITY, f32::max);
        if diag_lo > max_offdiag_hi {
            count += 1;
        }
    }
    count
}

// ===========================================================================
// Adversarial perturbation bounds builders
// ===========================================================================

/// Build PE-centered input bounds: PE +/- eps (uniform L-inf ball).
///
/// Used for both adversarial perturbation analysis and empirical bounds.
pub fn build_pe_centered_bounds(pe: &ArrayD<f32>, eps: f32) -> BoundedTensor {
    let mut lo = pe.clone();
    let mut hi = pe.clone();
    lo.mapv_inplace(|v| v - eps);
    hi.mapv_inplace(|v| v + eps);
    BoundedTensor::new(lo, hi).expect("bounds")
}

/// Build bounds for an invisible character insertion attack.
///
/// Base perturbation `small_eps` everywhere, with `large_eps` at `position`.
pub fn build_invisible_char_bounds(
    pe: &ArrayD<f32>,
    small_eps: f32,
    large_eps: f32,
    position: usize,
    d: usize,
) -> BoundedTensor {
    let mut lo = pe.clone();
    let mut hi = pe.clone();
    lo.mapv_inplace(|v| v - small_eps);
    hi.mapv_inplace(|v| v + small_eps);
    for c in 0..d {
        lo[[position, c]] = pe[[position, c]] - large_eps;
        hi[[position, c]] = pe[[position, c]] + large_eps;
    }
    BoundedTensor::new(lo, hi).expect("bounds")
}

/// Build bounds for a combined homoglyph + invisible char attack.
pub fn build_combined_attack_bounds(
    pe: &ArrayD<f32>,
    base_eps: f32,
    homoglyph_pos: usize,
    homoglyph_eps: f32,
    invisible_pos: usize,
    invisible_eps: f32,
    d: usize,
) -> BoundedTensor {
    let mut lo = pe.clone();
    let mut hi = pe.clone();
    lo.mapv_inplace(|v| v - base_eps);
    hi.mapv_inplace(|v| v + base_eps);
    for c in 0..d {
        lo[[homoglyph_pos, c]] = pe[[homoglyph_pos, c]] - homoglyph_eps;
        hi[[homoglyph_pos, c]] = pe[[homoglyph_pos, c]] + homoglyph_eps;
        lo[[invisible_pos, c]] = pe[[invisible_pos, c]] - invisible_eps;
        hi[[invisible_pos, c]] = pe[[invisible_pos, c]] + invisible_eps;
    }
    BoundedTensor::new(lo, hi).expect("bounds")
}
