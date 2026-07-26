// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for Demucs decoder composition tests.
//!
//! Provides binding-builder functions and constants shared between temporal
//! and spectral decoder production builder tests. Both decoder types use the
//! same DConv sub-layer structure and weight layout conventions from
//! `demucs_shared.rs`.
//!
//! Part of #1982: nn-verify test binary consolidation.

use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Shared constants (matching production demucs_shared.rs)
// ---------------------------------------------------------------------------

/// Weight magnitude: small to keep IBP bounds tractable through
/// decomposed GroupNorm G=1 (which amplifies through 14 primitive ops).
pub(super) const WEIGHT_MAG: f32 = 0.001;

/// DCONV compress ratio used by the production builder (from demucs_shared).
pub(super) const DCONV_COMPRESS: usize = 4;

/// DCONV depth used by the production builder (from demucs_shared).
pub(super) const DCONV_DEPTH: usize = 2;

/// DCONV kernel size used by the production builder (from demucs_shared).
pub(super) const DCONV_KERNEL: usize = 3;

/// ConvTranspose1d kernel (matches production KERNEL_SIZE=8).
pub(super) const KERNEL_SIZE: usize = 8;

/// Rewrite Conv1d/Conv2d kernel (matches production REWRITE_KERNEL=3).
pub(super) const REWRITE_KERNEL: usize = 3;

/// Rewrite padding (REWRITE_KERNEL / 2 = 1, matches production).
pub(super) const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;

/// ConvTranspose1d stride (must match production STRIDE=4).
pub(super) const STRIDE: usize = 4;

/// ConvTranspose1d padding (KERNEL_SIZE / 4 = 2, matches production).
pub(super) const CONV_TR_PADDING: usize = KERNEL_SIZE / 4;

/// Spectral stride (matches production SPECTRAL_STRIDE=4).
pub(super) const SPECTRAL_STRIDE: usize = 4;

// ---------------------------------------------------------------------------
// Binding helpers — shared between temporal and spectral decoder tests
// ---------------------------------------------------------------------------

/// Push a constant tensor binding filled with `val`.
pub(super) fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

/// Push DConv sub-layer bindings (matching `DConvSubLayerInputs::add_to_builder` order).
pub(super) fn push_dconv_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    ch: usize,
    compressed: usize,
) {
    let doubled = ch * 2;
    push_weight(bindings, &[compressed, ch, DCONV_KERNEL], WEIGHT_MAG); // compress weight
    push_weight(bindings, &[compressed], 0.0); // compress bias
    push_weight(bindings, &[compressed], 1.0); // norm gamma
    push_weight(bindings, &[compressed], 0.0); // norm beta
    push_weight(bindings, &[doubled, compressed, 1], WEIGHT_MAG); // expand weight
    push_weight(bindings, &[doubled], 0.0); // expand bias
    push_weight(bindings, &[doubled], 1.0); // norm gamma
    push_weight(bindings, &[doubled], 0.0); // norm beta
    push_weight(bindings, &[ch], 0.1); // layer_scale
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps1
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps2
}

// ---------------------------------------------------------------------------
// Temporal decoder helpers
// ---------------------------------------------------------------------------

/// Build bindings for a temporal decoder block: data=Variable, skip=ConstantTensor(zeros),
/// then rewrite + DConv + ConvTranspose1d weights in builder declaration order.
pub(super) fn temporal_decoder_bindings(
    in_ch: usize,
    out_ch: usize,
    t_in: usize,
) -> Vec<TensorParamBinding> {
    let compressed = in_ch / DCONV_COMPRESS;
    let doubled = in_ch * 2;
    let mut b = Vec::new();

    // Variable inputs: data, skip
    b.push(TensorParamBinding::Variable); // data [in_ch, t_in]
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[in_ch, t_in]),
        0.0f32,
    ))); // skip [in_ch, t_in] (zeros)

    // Rewrite Conv1d: [doubled, in_ch, rewrite_kernel=3]
    push_weight(&mut b, &[doubled, in_ch, REWRITE_KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[doubled], 0.0); // bias

    // DConv sub-layers (DCONV_DEPTH=2)
    for _ in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut b, in_ch, compressed);
    }

    // ConvTranspose1d: [in_ch, out_ch, kernel_size=8]
    push_weight(&mut b, &[in_ch, out_ch, KERNEL_SIZE], WEIGHT_MAG);
    push_weight(&mut b, &[out_ch], 0.0); // bias

    b
}

/// Compute temporal decoder output length from Conv1d rewrite output.
/// `rw_t_out` = conv1d_out_len(t_in, REWRITE_KERNEL, 1, REWRITE_PADDING).
pub(super) fn temporal_conv_tr_out_len(rw_t_out: usize) -> usize {
    (rw_t_out - 1) * STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING
}

// ---------------------------------------------------------------------------
// Spectral decoder helpers
// ---------------------------------------------------------------------------

/// Compute spectral ConvTranspose1d output frequency (before trim).
pub(super) fn spectral_conv_tr_f_out(rw_f: usize) -> usize {
    (rw_f - 1) * SPECTRAL_STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING
}

/// Build bindings for the spectral Rewrite sub-def.
pub(super) fn spectral_rewrite_bindings(
    in_ch: usize,
    f_in: usize,
    t_in: usize,
) -> Vec<TensorParamBinding> {
    let ft = f_in * t_in;
    let doubled = in_ch * 2;
    let mut b = Vec::new();

    b.push(TensorParamBinding::Variable); // data [C, F*T]
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[in_ch, ft]),
        0.0f32,
    ))); // skip (zeros)

    // Rewrite Conv2d weight [2C, C, 3, 3] + bias [2C].
    push_weight(
        &mut b,
        &[doubled, in_ch, REWRITE_KERNEL, REWRITE_KERNEL],
        WEIGHT_MAG,
    );
    push_weight(&mut b, &[doubled], 0.0);

    b
}

/// Build bindings for the spectral DConv sub-def.
pub(super) fn spectral_dconv_bindings(ch: usize) -> Vec<TensorParamBinding> {
    let compressed = ch / DCONV_COMPRESS;
    let mut b = Vec::new();

    b.push(TensorParamBinding::Variable);

    for _ in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut b, ch, compressed);
    }

    b
}

/// Build bindings for the spectral ConvTranspose1d sub-def.
pub(super) fn spectral_conv_tr_bindings(in_ch: usize, out_ch: usize) -> Vec<TensorParamBinding> {
    let mut b = Vec::new();

    b.push(TensorParamBinding::Variable);

    push_weight(&mut b, &[in_ch, out_ch, KERNEL_SIZE], WEIGHT_MAG);
    push_weight(&mut b, &[out_ch], 0.0);

    b
}
