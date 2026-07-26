// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared weight construction helpers for compose integration tests.
//!
//! Consolidates 44 duplicate function definitions across 9 function families
//! from `tests/helpers/`. Each function produces deterministic weight tensors
//! suitable for NY bound propagation testing.
//!
//! Part of #1938.

use ndarray::{ArrayD, IxDyn};

/// Build encoder weight matrix with controlled magnitude (near-identity).
///
/// Produces a `[rows, cols]` matrix with `scale` on the diagonal and
/// `scale * 0.1` on off-diagonal elements.
///
/// Replaces: `build_encoder_weight` in kokoro_attn_scaled, phase11_builders,
/// kokoro_attn_layerwise, phase7_builders (4 copies).
#[allow(dead_code)]
pub(crate) fn encoder_weight(rows: usize, cols: usize, scale: f32) -> ArrayD<f32> {
    let mut data = vec![0.0f32; rows * cols];
    for i in 0..rows.min(cols) {
        data[i * cols + i] = scale;
    }
    for i in 0..rows {
        for j in 0..cols {
            if i != j {
                data[i * cols + j] = scale * 0.1;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), data).expect("valid weight shape")
}

/// Build Conv1d weight tensor `[out_ch, in_ch, kernel]`.
///
/// Uses centered peak pattern: strongest weight at center of kernel for
/// matching channel pairs, `scale * 0.1` elsewhere.
///
/// Replaces: `build_conv_weight` in kokoro_attn_scaled, phase11_builders,
/// kokoro_attn_layerwise, phase7_builders (4 copies).
#[allow(dead_code)]
pub(crate) fn conv_weight(out_ch: usize, in_ch: usize, kernel: usize, scale: f32) -> ArrayD<f32> {
    let total = out_ch * in_ch * kernel;
    let mut data = vec![scale * 0.1; total];
    let center = kernel / 2;
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            let idx = oc * in_ch * kernel + ic * kernel + center;
            if oc == ic {
                data[idx] = scale;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[out_ch, in_ch, kernel]), data).expect("valid conv weight shape")
}

/// Scaled near-identity `[out_d, in_d]` for FFN projections.
///
/// Diagonal = `scale`, off-diagonal = 0. Simpler than `encoder_weight`
/// which has off-diagonal noise.
///
/// Replaces: `build_ffn_weight` in attention_ffn_composition,
/// attention_decoder_pipeline, deep_attention_stack (3 copies).
#[allow(dead_code)]
pub(crate) fn ffn_weight(out_d: usize, in_d: usize, scale: f32) -> ArrayD<f32> {
    let min_dim = out_d.min(in_d);
    let mut data = vec![0.0f32; out_d * in_d];
    for i in 0..min_dim {
        data[i * in_d + i] = scale;
    }
    ArrayD::from_shape_vec(IxDyn(&[out_d, in_d]), data).expect("valid FFN weight shape")
}

/// LayerNorm scale (all ones).
///
/// Replaces: `build_ln_weight` in attention_ffn_composition,
/// attention_decoder_pipeline, deep_attention_stack (3 copies).
#[allow(dead_code)]
pub(crate) fn norm_weight(d: usize) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[d]), 1.0f32)
}

/// LayerNorm bias (all zeros).
///
/// Replaces: `build_ln_bias` in attention_ffn_composition,
/// attention_decoder_pipeline, deep_attention_stack (3 copies).
#[allow(dead_code)]
pub(crate) fn norm_bias(d: usize) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[d]), 0.0f32)
}

/// Identity-like encoder K: each position has a distinct embedding direction.
///
/// Produces a `[t_enc, d]` matrix where each row has a block-diagonal 1.0
/// pattern (when `d >= t_enc`) or wrapping modular pattern (when `d < t_enc`).
///
/// Replaces: `build_encoder_k` in multi_head_causal, asymmetric_attention,
/// attention_ffn_composition, attention_decoder_pipeline, softmax_attention,
/// causal_attention, deep_attention_stack (7 copies).
#[allow(dead_code)]
pub(crate) fn encoder_k(t_enc: usize, d: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; t_enc * d];
    let cols_per = d / t_enc;
    if cols_per > 0 {
        for pos in 0..t_enc {
            for c in 0..cols_per {
                let col = pos * cols_per + c;
                if col < d {
                    data[pos * d + col] = 1.0;
                }
            }
        }
    } else {
        for pos in 0..t_enc {
            data[pos * d + (pos % d)] = 1.0;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[t_enc, d]), data).expect("valid K shape")
}

/// Near-identity matrix `[d, d]`: 1.0 on diagonal, `perturbation` off-diagonal.
///
/// Replaces: `near_identity` in attention_decoder_multi_stage,
/// attention_decoder_deep, attention_decoder_multi_kernel,
/// attention_decoder_output, attention_decoder_dilated,
/// attention_decoder_noise, attention_decoder_scaled (7 copies).
#[allow(dead_code)]
pub(crate) fn near_identity(d: usize, perturbation: f32) -> ArrayD<f32> {
    let mut data = vec![perturbation; d * d];
    for i in 0..d {
        data[i * d + i] = 1.0;
    }
    ArrayD::from_shape_vec(IxDyn(&[d, d]), data).expect("valid near-identity shape")
}
