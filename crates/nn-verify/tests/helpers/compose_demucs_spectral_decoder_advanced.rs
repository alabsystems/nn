// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced integration tests: Demucs spectral decoder CROWN propagation,
//! verify-and-record, last-block handling, and sequential 3-stage composition.
//!
//! Extracted from `compose_demucs_spectral_decoder.rs` for file-size
//! compliance (#1420).
//!
//! See parent file for full documentation on sub-def structure and test-scale
//! parameters.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_models::demucs_spectral_decoder_builders::{
    build_decoder_block_sub_defs as build_block_sub_defs, conv2d_output_len,
};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Test-scale parameters (duplicated from parent — standalone test binary)
// ---------------------------------------------------------------------------

const IN_CH: usize = 16;
const OUT_CH: usize = 8;
const F_IN: usize = 4;
const T_IN: usize = 4;
const REWRITE_KERNEL: usize = 3;
const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;
const KERNEL_SIZE: usize = 8;
const SPECTRAL_STRIDE: usize = 4;
const CONV_TR_PADDING: usize = KERNEL_SIZE / 4;
const WEIGHT_MAG: f32 = 0.001;
const DCONV_COMPRESS: usize = 4;
const DCONV_DEPTH: usize = 2;
const DCONV_KERNEL: usize = 3;

// ---------------------------------------------------------------------------
// Dimension helpers (duplicated from parent)
// ---------------------------------------------------------------------------

fn rewrite_output_dims() -> (usize, usize) {
    let rw_f = conv2d_output_len(F_IN, REWRITE_KERNEL, 1, REWRITE_PADDING)
        .expect("valid rewrite freq params");
    let rw_t = conv2d_output_len(T_IN, REWRITE_KERNEL, 1, REWRITE_PADDING)
        .expect("valid rewrite time params");
    (rw_f, rw_t)
}

fn conv_tr_f_out(rw_f: usize) -> usize {
    (rw_f - 1) * SPECTRAL_STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING
}

// ---------------------------------------------------------------------------
// Binding helpers (duplicated from parent)
// ---------------------------------------------------------------------------

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

fn push_dconv_bindings(bindings: &mut Vec<TensorParamBinding>, ch: usize, compressed: usize) {
    let doubled = ch * 2;
    push_weight(bindings, &[compressed, ch, DCONV_KERNEL], WEIGHT_MAG);
    push_weight(bindings, &[compressed], 0.0);
    push_weight(bindings, &[compressed], 1.0);
    push_weight(bindings, &[compressed], 0.0);
    push_weight(bindings, &[doubled, compressed, 1], WEIGHT_MAG);
    push_weight(bindings, &[doubled], 0.0);
    push_weight(bindings, &[doubled], 1.0);
    push_weight(bindings, &[doubled], 0.0);
    push_weight(bindings, &[ch], 0.1);
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
}

fn rewrite_bindings() -> Vec<TensorParamBinding> {
    let ft = F_IN * T_IN;
    let doubled = IN_CH * 2;
    let mut b = Vec::new();

    b.push(TensorParamBinding::Variable);
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[IN_CH, ft]),
        0.0f32,
    )));

    push_weight(
        &mut b,
        &[doubled, IN_CH, REWRITE_KERNEL, REWRITE_KERNEL],
        WEIGHT_MAG,
    );
    push_weight(&mut b, &[doubled], 0.0);

    b
}

fn dconv_bindings(ch: usize, t_len: usize) -> Vec<TensorParamBinding> {
    let compressed = ch / DCONV_COMPRESS;
    let mut b = Vec::new();

    let _ = t_len;
    b.push(TensorParamBinding::Variable);

    for _ in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut b, ch, compressed);
    }

    b
}

fn conv_tr_bindings(in_ch: usize, out_ch: usize) -> Vec<TensorParamBinding> {
    let mut b = Vec::new();

    b.push(TensorParamBinding::Variable);

    push_weight(&mut b, &[in_ch, out_ch, KERNEL_SIZE], WEIGHT_MAG);
    push_weight(&mut b, &[out_ch], 0.0);

    b
}

// ---------------------------------------------------------------------------
// Tests — CROWN, verify-and-record, last-block, sequential composition
// ---------------------------------------------------------------------------

/// CROWN propagation through the DConv sub-def (may fall back to IBP due
/// to decomposed GroupNorm G=1, per design doc #697).
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_dconv_sub_def_crown() {
    let (_, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(F_IN);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, F_IN, rw_t, target_f, false)
        .expect("builder");
    let bindings = dconv_bindings(IN_CH, rw_t);
    let graph =
        tensor_kernel_to_graph(&sub_defs.dconv_def, &bindings).expect("dconv graph translation");

    let input = uniform_bounds(&[IN_CH, rw_t], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[IN_CH, rw_t], "output shape mismatch");

    eprintln!("DConv sub-def: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Record verification result for the DConv sub-def.
#[test]
fn test_dconv_sub_def_verify_and_record() {
    let (_, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(F_IN);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, F_IN, rw_t, target_f, false)
        .expect("builder");
    let bindings = dconv_bindings(IN_CH, rw_t);
    let input = uniform_bounds(&[IN_CH, rw_t], 1.0);

    let result = verify_and_assert(
        &sub_defs.dconv_def,
        &bindings,
        &input,
        "demucs_spectral_decoder_dconv",
    );
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[IN_CH, rw_t]);
}

/// Last block (is_last=true) omits GELU activation in ConvTranspose sub-def.
#[test]
fn test_production_builder_last_block() {
    let (rw_f, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(3, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, true)
        .expect("last block builder");

    sub_defs.rewrite_def.validate().expect("rewrite validates");
    sub_defs.dconv_def.validate().expect("dconv validates");
    sub_defs.conv_tr_def.validate().expect("conv_tr validates");

    let bindings = conv_tr_bindings(IN_CH, OUT_CH);
    let graph = tensor_kernel_to_graph(&sub_defs.conv_tr_def, &bindings)
        .expect("conv_tr graph translation");
    let input = uniform_bounds(&[IN_CH, rw_f], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through last block conv_tr");
    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, target_f], "last block output shape");
    assert_bounds_valid(&output);
}

/// Sequential sub-def composition: rewrite output feeds DConv, DConv output
/// feeds ConvTranspose. Verifies bounds propagation across all 3 stages.
#[test]
fn test_sequential_sub_def_composition() {
    let (rw_f, rw_t) = rewrite_output_dims();
    let ct_f = conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
        .expect("builder");

    // Stage 1: Rewrite — input [IN_CH, F*T], output [IN_CH, rw_f*rw_t].
    let ft = F_IN * T_IN;
    let rw_ft = rw_f * rw_t;
    let rw_bindings = rewrite_bindings();
    let rw_graph =
        tensor_kernel_to_graph(&sub_defs.rewrite_def, &rw_bindings).expect("rewrite graph");
    let rw_input = uniform_bounds(&[IN_CH, ft], 1.0);
    let rw_output = rw_graph
        .propagate_ibp(&rw_input)
        .expect("IBP through rewrite");
    assert_bounds_valid(&rw_output);
    let (rw_lo, _) = rw_output.lower_upper();
    assert_eq!(rw_lo.shape(), &[IN_CH, rw_ft], "rewrite output shape");

    // Stage 2: DConv — input [IN_CH, rw_t], output [IN_CH, rw_t].
    let dc_bindings = dconv_bindings(IN_CH, rw_t);
    let dc_graph = tensor_kernel_to_graph(&sub_defs.dconv_def, &dc_bindings).expect("dconv graph");

    let (rw_lo_min, rw_hi_max) = bounds_min_max(&rw_output);
    let dc_range = rw_hi_max.abs().max(rw_lo_min.abs()).max(0.01);
    let dc_input = uniform_bounds(&[IN_CH, rw_t], dc_range);

    let dc_output = dc_graph
        .propagate_ibp(&dc_input)
        .expect("IBP through DConv");
    assert_bounds_valid(&dc_output);
    let (dc_lo, _) = dc_output.lower_upper();
    assert_eq!(dc_lo.shape(), &[IN_CH, rw_t], "dconv output shape");

    // Stage 3: ConvTranspose — input [IN_CH, rw_f], output [OUT_CH, target_f].
    let ct_bindings = conv_tr_bindings(IN_CH, OUT_CH);
    let ct_graph =
        tensor_kernel_to_graph(&sub_defs.conv_tr_def, &ct_bindings).expect("conv_tr graph");

    let (dc_lo_min, dc_hi_max) = bounds_min_max(&dc_output);
    let ct_range = dc_hi_max.abs().max(dc_lo_min.abs()).max(0.01);
    let ct_input = uniform_bounds(&[IN_CH, rw_f], ct_range);

    let ct_output = ct_graph
        .propagate_ibp(&ct_input)
        .expect("IBP through ConvTranspose");
    assert_bounds_valid(&ct_output);
    let (ct_lo, _) = ct_output.lower_upper();
    assert_eq!(ct_lo.shape(), &[OUT_CH, target_f], "conv_tr output shape");

    eprintln!(
        "Sequential composition: rewrite→[{IN_CH},{rw_ft}], dconv→[{IN_CH},{rw_t}], \
         conv_tr→[{OUT_CH},{target_f}]"
    );
}
