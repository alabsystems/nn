// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Graph construction, causal mask, and positional encoding test helpers.
//!
//! Extracted from `common/mod.rs` for 500-line compliance.
//! Part of #2633.

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Causal mask construction + graph propagation (Part of #1970)
// ---------------------------------------------------------------------------

/// Large negative value for masked attention positions.
///
/// Using -1e9 instead of -inf to keep NY numerics stable.
/// Softmax(-1e9) ≈ 0, which is functionally equivalent to true masking.
#[allow(dead_code)]
pub(crate) const MASK_VALUE: f32 = -1e9;

/// Strict causal: `f(t) = min(t, T_enc - 1)`.
#[allow(dead_code)]
pub(crate) fn strict_causal_alignment(t: usize, t_enc: usize) -> usize {
    t.min(t_enc.saturating_sub(1))
}

/// Build a causal mask tensor `[t_dec, t_enc]` using the given alignment.
#[allow(dead_code)]
pub(crate) fn build_causal_mask(
    t_dec: usize,
    t_enc: usize,
    alignment_fn: impl Fn(usize) -> usize,
) -> ArrayD<f32> {
    let mut data = vec![0.0f32; t_dec * t_enc];
    for t in 0..t_dec {
        let max_pos = alignment_fn(t);
        for j in 0..t_enc {
            if j > max_pos {
                data[t * t_enc + j] = MASK_VALUE;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[t_dec, t_enc]), data).expect("valid mask shape")
}

/// Build a strict causal mask.
#[allow(dead_code)]
pub(crate) fn build_strict_causal_mask(t_dec: usize, t_enc: usize) -> ArrayD<f32> {
    build_causal_mask(t_dec, t_enc, |t| strict_causal_alignment(t, t_enc))
}

/// Propagate through tensor graph with IBP and return output bounds.
#[allow(dead_code)]
pub(crate) fn graph_propagate(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
) -> BoundedTensor {
    let graph = nn_verify::tensor_kernel_to_graph(def, bindings).expect("graph");
    graph.propagate_ibp(input).expect("IBP")
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding (Part of #1970)
// ---------------------------------------------------------------------------

/// Standard sinusoidal positional encoding: PE[t, 2i] = sin(t / 10000^(2i/D)).
///
/// Key property: PE vectors at different positions are approximately orthogonal,
/// so PE @ PE^T is diagonally dominant.
#[allow(dead_code)]
pub(crate) fn sinusoidal_pe(seq_len: usize, d_model: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for i in 0..d_model / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            data[t * d_model + 2 * i] = freq.sin() as f32;
            data[t * d_model + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE")
}

/// Sinusoidal positional encoding with head-interleaved frequencies.
///
/// Unlike standard PE, this variant reorders frequency indices so that
/// each attention head gets a distinct subset of frequencies, interleaved
/// across the model dimension. Used by multi-head attention stacks.
#[allow(dead_code)]
pub(crate) fn sinusoidal_pe_interleaved(
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
) -> ArrayD<f32> {
    let d_k = d_model / num_heads;
    let num_pairs = d_model / 2;
    let pairs_per_head = d_k / 2;

    let mut dim_to_freq = vec![0usize; num_pairs];
    for h in 0..num_heads {
        for p in 0..pairs_per_head {
            let freq_idx = h + p * num_heads;
            let out_pair = h * pairs_per_head + p;
            if freq_idx < num_pairs && out_pair < num_pairs {
                dim_to_freq[out_pair] = freq_idx;
            }
        }
    }

    let mut data = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for pair in 0..num_pairs {
            let freq_idx = dim_to_freq[pair];
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * freq_idx as f64 / d_model as f64);
            data[t * d_model + 2 * pair] = freq.sin() as f32;
            data[t * d_model + 2 * pair + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq_len, d_model]), data).expect("valid PE")
}

// ---------------------------------------------------------------------------
// Tensor-level test helpers (Layer 2)
// ---------------------------------------------------------------------------

/// PyTorch default epsilon for InstanceNorm, GroupNorm, LayerNorm, RMSNorm.
///
/// Use this constant in new tests instead of hardcoding `1e-5_f32`.
/// Existing 48+ occurrences can migrate incrementally.
#[allow(dead_code)]
pub(crate) const DEFAULT_NORM_EPS: f32 = 1e-5;

/// Conv1d output length formula: `(in_len + 2*padding - kernel) / stride + 1`.
///
/// Delegates to canonical `nn_core::conv1d_out_len` (dilation=1).
///
/// Note: parameter order here is `(in_len, kernel_size, stride, padding)` which
/// differs from canonical `(input_len, kernel_size, padding, stride, dilation)`.
/// This wrapper preserves the test-side convention.
#[allow(dead_code)]
pub(crate) fn conv1d_out_len(
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> usize {
    nn_core::conv1d_out_len(in_len, kernel_size, padding, stride, 1)
        .expect("conv1d_out_len: invalid parameters")
}

/// ConvTranspose1d output length: `(in_len - 1) * stride + kernel - 2*padding`.
///
/// Replaces identical copies in compose_four_block_decoder,
/// compose_decoder_conv_transpose, and compose_demucs_decoder_block.
#[allow(dead_code)]
pub(crate) fn conv_transpose_out_len(
    in_len: usize,
    stride: usize,
    kernel_size: usize,
    padding: usize,
) -> usize {
    (in_len - 1) * stride + kernel_size - 2 * padding
}

/// Linear alignment: `f(t) = floor(t * t_enc / t_dec)`.
///
/// Replaces identical copies in causal_attention, multi_head_causal,
/// and softmax_attention helpers.
#[allow(dead_code)]
pub(crate) fn linear_alignment(t: usize, t_dec: usize, t_enc: usize) -> usize {
    (t * t_enc / t_dec).min(t_enc.saturating_sub(1))
}

/// Build a linear causal mask.
///
/// Replaces identical copies in causal_attention, multi_head_causal,
/// and softmax_attention helpers.
#[allow(dead_code)]
pub(crate) fn build_linear_causal_mask(t_dec: usize, t_enc: usize) -> ArrayD<f32> {
    build_causal_mask(t_dec, t_enc, |t| linear_alignment(t, t_dec, t_enc))
}
