// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parameter binding helpers for HTDemucs full-model composition tests.
//!
//! Extracted from `htdemucs_full.rs` for 500-line compliance (#1693).
//! Creates `TensorParamBinding` vectors matching the graph inputs
//! built by `build_htdemucs_full()`.

use super::*;

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

fn add_dconv_bindings(b: &mut Vec<TensorParamBinding>, ch: usize, comp: usize) {
    let d = ch * 2;
    push_weight(b, &[comp, ch, DCONV_KERNEL], WEIGHT_MAG);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[comp], 1.0);
    push_weight(b, &[comp], 0.0);
    push_weight(b, &[d, comp, 1], WEIGHT_MAG);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[ch], 0.1);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b.push(TensorParamBinding::ConstantScalar(1e-5));
}

fn add_self_attn_bindings(b: &mut Vec<TensorParamBinding>) {
    let d = MODEL_DIM;
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[FFN_HIDDEN, d], WEIGHT_MAG);
    push_weight(b, &[d, FFN_HIDDEN], WEIGHT_MAG);
}

fn add_cross_attn_bindings(b: &mut Vec<TensorParamBinding>) {
    let d = MODEL_DIM;
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d], 1.0);
    push_weight(b, &[d], 0.0);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[d, d], WEIGHT_MAG);
    push_weight(b, &[FFN_HIDDEN, d], WEIGHT_MAG);
    push_weight(b, &[d, FFN_HIDDEN], WEIGHT_MAG);
}

/// Create parameter bindings for the full HTDemucs model.
pub(crate) fn htdemucs_full_bindings() -> Vec<TensorParamBinding> {
    let comp = ENC_CH / DCONV_COMPRESS_RATIO;
    let dbl = ENC_CH * 2;
    let d = MODEL_DIM;
    let mut b = Vec::new();

    // Variable input: audio waveform
    b.push(TensorParamBinding::Variable);
    // Constant: spectral KV (pre-computed spectral encoder output)
    push_weight(&mut b, &[F_SEQ, d], 0.1);

    // Encoder
    push_weight(&mut b, &[ENC_CH, IN_CH, ENC_KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[ENC_CH], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut b, ENC_CH, comp);
    }
    push_weight(&mut b, &[dbl, ENC_CH, 1], WEIGHT_MAG);
    push_weight(&mut b, &[dbl], 0.0);

    // Cross-domain transformer
    push_weight(&mut b, &[d, ENC_CH, 1], WEIGHT_MAG); // t_up_w
    push_weight(&mut b, &[d], 0.0); // t_up_b
    push_weight(&mut b, &[ENC_CH, d, 1], WEIGHT_MAG); // t_down_w
    push_weight(&mut b, &[ENC_CH], 0.0); // t_down_b
    b.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
    add_self_attn_bindings(&mut b);
    add_cross_attn_bindings(&mut b);

    // Decoder
    push_weight(&mut b, &[dbl, ENC_CH, DEC_REWRITE_KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[dbl], 0.0);
    for _ in 0..DCONV_DEPTH {
        add_dconv_bindings(&mut b, ENC_CH, comp);
    }
    push_weight(&mut b, &[ENC_CH, IN_CH, CT_KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[IN_CH], 0.0);

    b
}
