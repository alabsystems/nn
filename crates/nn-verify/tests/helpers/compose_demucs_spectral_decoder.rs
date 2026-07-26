// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs spectral decoder composition using the
//! **real production builder** `build_decoder_block_sub_defs()` from nn-models.
//!
//! Unlike `compose_demucs_spectral_full.rs` (which builds a simplified graph
//! manually), this test calls the actual builder functions used by
//! `DemucsSpectralDecoder::new()`. This verifies that production-generated
//! `TensorKernelDef`s are translatable to NY and that bounds
//! propagate correctly through each sub-def.
//!
//! The spectral decoder splits each block into 3 sub-defs (due to CPU-side
//! axis-switch between stages):
//! 1. **Rewrite**: skip_add → Reshape → Conv2d(3×3) → Reshape → GLU
//! 2. **DConv**: Conv1d(dilated) → GN(G=1) → GELU → Conv1d(1×1) → GN(G=1)
//!    → GLU → LayerScale → residual (×2 sub-layers)
//! 3. **ConvTranspose1d**: upsample along freq axis → optional trim
//!
//! Dimensions are scaled down for NY tractability.
//! Production constants (DCONV_COMPRESS=4, DCONV_DEPTH=2, DCONV_KERNEL=3)
//! are used by the builder unchanged.
//!
//! CROWN propagation, verify-and-record, last-block, and sequential
//! composition tests extracted to `compose_demucs_spectral_decoder_advanced.rs`
//! (#1420).
//!
//! Part of #779 Phase B — composition verification with production builders.

use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};
use nn_models::demucs_spectral_decoder_builders::{
    build_decoder_block_sub_defs as build_block_sub_defs, conv2d_output_len,
};
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Test-scale parameters (production uses 48-384 channels; we use 16/8)
// ---------------------------------------------------------------------------

/// Input channels to the decoder block (matches temporal decoder tests).
const IN_CH: usize = 16;

/// Output channels after ConvTranspose1d.
const OUT_CH: usize = 8;

/// Frequency dimension at block input.
const F_IN: usize = 4;

/// Time dimension at block input.
const T_IN: usize = 4;

/// Conv2d kernel for spectral rewrite (3×3, matches production REWRITE_KERNEL=3).
const REWRITE_KERNEL: usize = 3;

/// Rewrite Conv2d padding (REWRITE_KERNEL / 2 = 1, matches production).
const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;

/// ConvTranspose1d kernel (matches production KERNEL_SIZE=8).
const KERNEL_SIZE: usize = 8;

/// Spectral stride (matches production SPECTRAL_STRIDE=4).
const SPECTRAL_STRIDE: usize = 4;

/// ConvTranspose1d padding (KERNEL_SIZE / 4 = 2, matches production).
const CONV_TR_PADDING: usize = KERNEL_SIZE / 4;

/// Weight magnitude: small to keep IBP bounds tractable.
const WEIGHT_MAG: f32 = 0.001;

/// DCONV compress ratio used by the production builder (from demucs_shared).
const DCONV_COMPRESS: usize = 4;

/// DCONV depth used by the production builder (from demucs_shared).
const DCONV_DEPTH: usize = 2;

/// DCONV kernel size used by the production builder (from demucs_shared).
const DCONV_KERNEL: usize = 3;

// ---------------------------------------------------------------------------
// Dimension helpers
// ---------------------------------------------------------------------------

/// Compute rewrite output dimensions (Conv2d(3×3, s=1, p=1) preserves spatial).
fn rewrite_output_dims() -> (usize, usize) {
    let rw_f = conv2d_output_len(F_IN, REWRITE_KERNEL, 1, REWRITE_PADDING)
        .expect("valid rewrite freq params");
    let rw_t = conv2d_output_len(T_IN, REWRITE_KERNEL, 1, REWRITE_PADDING)
        .expect("valid rewrite time params");
    (rw_f, rw_t)
}

/// Compute ConvTranspose1d output frequency (before trim).
fn conv_tr_f_out(rw_f: usize) -> usize {
    (rw_f - 1) * SPECTRAL_STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING
}

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

/// Push a constant tensor binding filled with `val`.
fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

/// Push DConv sub-layer bindings (matching `DConvSubLayerInputs::add_to_builder` order).
fn push_dconv_bindings(bindings: &mut Vec<TensorParamBinding>, ch: usize, compressed: usize) {
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

/// Build bindings for the Rewrite sub-def.
///
/// Inputs: data=Variable, skip=ConstantTensor(zeros), rw_weight, rw_bias.
fn rewrite_bindings() -> Vec<TensorParamBinding> {
    let ft = F_IN * T_IN;
    let doubled = IN_CH * 2;
    let mut b = Vec::new();

    // Variable inputs: data [C, F*T], skip [C, F*T].
    b.push(TensorParamBinding::Variable); // data
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[IN_CH, ft]),
        0.0f32,
    ))); // skip (zeros)

    // Rewrite Conv2d weight [2C, C, 3, 3] + bias [2C].
    push_weight(
        &mut b,
        &[doubled, IN_CH, REWRITE_KERNEL, REWRITE_KERNEL],
        WEIGHT_MAG,
    );
    push_weight(&mut b, &[doubled], 0.0);

    b
}

/// Build bindings for the DConv sub-def.
///
/// Input: data=Variable, then DConv weights for DCONV_DEPTH sub-layers.
fn dconv_bindings(ch: usize, t_len: usize) -> Vec<TensorParamBinding> {
    let compressed = ch / DCONV_COMPRESS;
    let mut b = Vec::new();

    // Variable input: data [C, T].
    let _ = t_len; // Used for documentation only; shape is in the def.
    b.push(TensorParamBinding::Variable);

    // DConv sub-layers (DCONV_DEPTH=2).
    for _ in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut b, ch, compressed);
    }

    b
}

/// Build bindings for the ConvTranspose1d sub-def.
///
/// Input: data=Variable, ct_weight, ct_bias.
fn conv_tr_bindings(in_ch: usize, out_ch: usize) -> Vec<TensorParamBinding> {
    let mut b = Vec::new();

    // Variable input: data [C, F].
    b.push(TensorParamBinding::Variable);

    // ConvTranspose1d weight [C, C_out, kernel] + bias [C_out].
    push_weight(&mut b, &[in_ch, out_ch, KERNEL_SIZE], WEIGHT_MAG);
    push_weight(&mut b, &[out_ch], 0.0);

    b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The production builder validates and produces 3 valid TensorKernelDefs.
#[test]
fn test_production_sub_defs_validate() {
    let (rw_f, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN); // trim to original freq

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
        .expect("production builder should succeed");

    sub_defs
        .rewrite_def
        .validate()
        .expect("rewrite def should validate");
    sub_defs
        .dconv_def
        .validate()
        .expect("dconv def should validate");
    sub_defs
        .conv_tr_def
        .validate()
        .expect("conv_tr def should validate");
}

/// Rewrite sub-def translates to NY graph and IBP propagates.
#[test]
fn test_rewrite_sub_def_ibp() {
    let (rw_f, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
        .expect("builder");
    let bindings = rewrite_bindings();
    let graph = tensor_kernel_to_graph(&sub_defs.rewrite_def, &bindings)
        .expect("rewrite graph translation");

    let ft = F_IN * T_IN;
    let input = uniform_bounds(&[IN_CH, ft], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through rewrite sub-def");
    // Rewrite output: [IN_CH, rw_f * rw_t] after GLU.
    let rw_ft = rw_f * rw_t;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[IN_CH, rw_ft],
        "rewrite output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Rewrite sub-def IBP: bounds=[{lo_min}, {hi_max}] shape=[{IN_CH}, {rw_ft}]");
}

/// DConv sub-def translates to NY graph and IBP propagates.
#[test]
fn test_dconv_sub_def_ibp() {
    let (_, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(F_IN);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, F_IN, rw_t, target_f, false)
        .expect("builder");
    let bindings = dconv_bindings(IN_CH, rw_t);
    let graph =
        tensor_kernel_to_graph(&sub_defs.dconv_def, &bindings).expect("dconv graph translation");

    let input = uniform_bounds(&[IN_CH, rw_t], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DConv sub-def");

    // DConv output: [IN_CH, rw_t] (preserves shape due to residual).
    assert_eq!(
        output.lower_upper().0.shape(),
        &[IN_CH, rw_t],
        "dconv output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DConv sub-def IBP: bounds=[{lo_min}, {hi_max}] shape=[{IN_CH}, {rw_t}]");
}

/// ConvTranspose1d sub-def translates to NY graph and IBP propagates.
#[test]
fn test_conv_tr_sub_def_ibp() {
    let (rw_f, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
        .expect("builder");
    let bindings = conv_tr_bindings(IN_CH, OUT_CH);
    let graph = tensor_kernel_to_graph(&sub_defs.conv_tr_def, &bindings)
        .expect("conv_tr graph translation");

    let input = uniform_bounds(&[IN_CH, rw_f], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ConvTranspose sub-def");

    // ConvTranspose output: [OUT_CH, target_f].
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CH, target_f],
        "conv_tr output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "ConvTranspose sub-def IBP: bounds=[{lo_min}, {hi_max}] shape=[{OUT_CH}, {target_f}]"
    );
}

// CROWN, verify-and-record, last-block, and sequential composition tests
// extracted to compose_demucs_spectral_decoder_advanced.rs (#1420).

// ---------------------------------------------------------------------------
// CROWN propagation tests
// ---------------------------------------------------------------------------

/// CROWN produces tighter-or-equal bounds than IBP on ConvTranspose sub-def.
#[test]
fn test_conv_tr_sub_def_crown_tighter_than_ibp() {
    let (rw_f, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
        .expect("builder");
    let bindings = conv_tr_bindings(IN_CH, OUT_CH);
    let graph = tensor_kernel_to_graph(&sub_defs.conv_tr_def, &bindings)
        .expect("conv_tr graph translation");

    let input = uniform_bounds(&[IN_CH, rw_f], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through ConvTranspose sub-def");
    let (_, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through ConvTranspose sub-def");

    super::common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}
