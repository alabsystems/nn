// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs spectral encoder block composition using the
//! **real production builder** `build_encoder_block_sub_defs()` from nn-models.
//!
//! Spectral encoder splits each block into 3 sub-defs (CPU-side axis-switch):
//!   1. Conv1d(k=8, s=4, p=2) + GELU — operates on `[C, F]` per time step
//!   2. DConv(×2 residual) — operates on `[C, T]` per freq bin
//!   3. Rewrite Conv1d(k=1) + GLU — operates on `[C, F']` per time step
//!
//! Uses production DCONV_COMPRESS=4, DCONV_KERNEL=3, KERNEL_SIZE=8, STRIDE=4.
//! Small channel counts (IN_CH=8, OUT_CH=16) for tractable verification.
//!
//! Part of #779 Phase E — spectral encoder composition with production builders.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_models::demucs_spectral_encoder_builders::{
    build_encoder_block_sub_defs as build_block_sub_defs, spectral_conv1d_out_len,
};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale parameters matching production builder constraints
// ---------------------------------------------------------------------------

const IN_CH: usize = 8;
const OUT_CH: usize = 16;
const F_IN: usize = 16;
const T_IN: usize = 4;
const WEIGHT_MAG: f32 = 0.001;

const KERNEL_SIZE: usize = 8;
const CONV_PADDING: usize = 2;
const SPECTRAL_STRIDE: usize = 4;
const DCONV_DEPTH: usize = 2;
const DCONV_KERNEL: usize = 3;

fn f_out() -> usize {
    spectral_conv1d_out_len(F_IN, KERNEL_SIZE, SPECTRAL_STRIDE, CONV_PADDING)
        .expect("valid spectral conv1d params")
}

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

/// Bindings for Conv+GELU sub-def: data [in_ch, f_in], conv_weight, conv_bias.
fn conv_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OUT_CH, IN_CH, KERNEL_SIZE]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OUT_CH]), 0.0f32)),
    ]
}

/// Push 11 DConv sub-layer bindings for given channel dimensions.
fn push_dconv_bindings(b: &mut Vec<TensorParamBinding>, out_ch: usize, compressed: usize) {
    let doubled = out_ch * 2;
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed, out_ch, DCONV_KERNEL]),
        WEIGHT_MAG,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        0.0f32,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        1.0f32,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[compressed]),
        0.0f32,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, compressed, 1]),
        WEIGHT_MAG,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        1.0f32,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch]),
        0.1f32,
    )));
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b.push(TensorParamBinding::ConstantScalar(1e-5));
}

/// Bindings for DConv sub-def: data [out_ch, t_len], then DCONV_DEPTH × 11 entries.
fn dconv_bindings() -> Vec<TensorParamBinding> {
    let compressed = OUT_CH / 4; // DCONV_COMPRESS=4
    let mut b = vec![TensorParamBinding::Variable];
    for _k in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut b, OUT_CH, compressed);
    }
    b
}

/// Bindings for Rewrite sub-def: data [out_ch, f_out], rw_weight, rw_bias.
fn rewrite_bindings() -> Vec<TensorParamBinding> {
    let doubled = OUT_CH * 2;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[doubled, OUT_CH, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[doubled]), 0.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Production spectral encoder sub-defs all validate.
#[test]
fn test_production_sub_defs_validate() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");
    sub_defs.conv_gelu_def.validate().expect("conv_gelu valid");
    sub_defs.dconv_def.validate().expect("dconv valid");
    sub_defs.rewrite_def.validate().expect("rewrite valid");
}

/// Conv+GELU sub-def: IBP propagation.
#[test]
fn test_conv_gelu_sub_def_ibp() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");
    let graph = tensor_kernel_to_graph(&sub_defs.conv_gelu_def, &conv_gelu_bindings())
        .expect("graph translation");
    let input = uniform_bounds(&[IN_CH, F_IN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP conv_gelu");
    assert_eq!(output.lower_upper().0.shape(), &[OUT_CH, fo]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv+GELU IBP: bounds=[{lo_min}, {hi_max}] shape=[{OUT_CH}, {fo}]");
}

/// DConv sub-def: IBP propagation (per freq bin, operates on [C, T]).
#[test]
fn test_dconv_sub_def_ibp() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");
    let graph =
        tensor_kernel_to_graph(&sub_defs.dconv_def, &dconv_bindings()).expect("graph translation");
    let input = uniform_bounds(&[OUT_CH, T_IN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP dconv");
    assert_eq!(output.lower_upper().0.shape(), &[OUT_CH, T_IN]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DConv IBP: bounds=[{lo_min}, {hi_max}] shape=[{OUT_CH}, {T_IN}]");
}

/// Rewrite sub-def: IBP propagation.
#[test]
fn test_rewrite_sub_def_ibp() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");
    let graph = tensor_kernel_to_graph(&sub_defs.rewrite_def, &rewrite_bindings())
        .expect("graph translation");
    let input = uniform_bounds(&[OUT_CH, fo], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP rewrite");
    assert_eq!(output.lower_upper().0.shape(), &[OUT_CH, fo]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Rewrite IBP: bounds=[{lo_min}, {hi_max}] shape=[{OUT_CH}, {fo}]");
}

/// CROWN propagation on DConv sub-def (may fall back to IBP).
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_dconv_sub_def_crown() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");
    let graph =
        tensor_kernel_to_graph(&sub_defs.dconv_def, &dconv_bindings()).expect("graph translation");
    let input = uniform_bounds(&[OUT_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let (lo, _) = output.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, T_IN], "output shape mismatch");

    eprintln!("DConv CROWN: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback: {reason}");
    }
}

/// Record verification in status file.
#[test]
fn test_dconv_sub_def_verify_and_record() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");

    let result = verify_and_assert(
        &sub_defs.dconv_def,
        &dconv_bindings(),
        &uniform_bounds(&[OUT_CH, T_IN], 1.0),
        "demucs_spectral_encoder_prod_dconv",
    );
    assert_eq!(result.num_variables, 1);
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, T_IN]);
}

/// Sequential: conv_gelu → dconv → rewrite (all 3 sub-defs).
#[test]
fn test_sequential_sub_def_composition() {
    let fo = f_out();
    let sub_defs = build_block_sub_defs(0, IN_CH, OUT_CH, F_IN, fo, T_IN).expect("build sub-defs");

    // Stage 1: Conv+GELU
    let g1 = tensor_kernel_to_graph(&sub_defs.conv_gelu_def, &conv_gelu_bindings())
        .expect("conv_gelu graph");
    let o1 = g1
        .propagate_ibp(&uniform_bounds(&[IN_CH, F_IN], 1.0))
        .expect("IBP conv_gelu");
    assert_bounds_valid(&o1);
    let (_, hi1) = o1.lower_upper();
    let mag1 = hi1.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));

    // Stage 2: DConv (per freq bin)
    let g2 = tensor_kernel_to_graph(&sub_defs.dconv_def, &dconv_bindings()).expect("dconv graph");
    let o2 = g2
        .propagate_ibp(&uniform_bounds(&[OUT_CH, T_IN], mag1.min(10.0)))
        .expect("IBP dconv");
    assert_bounds_valid(&o2);
    let (_, hi2) = o2.lower_upper();
    let mag2 = hi2.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));

    // Stage 3: Rewrite
    let g3 =
        tensor_kernel_to_graph(&sub_defs.rewrite_def, &rewrite_bindings()).expect("rewrite graph");
    let o3 = g3
        .propagate_ibp(&uniform_bounds(&[OUT_CH, fo], mag2.min(10.0)))
        .expect("IBP rewrite");
    assert_bounds_valid(&o3);
    let (lo3, _) = o3.lower_upper();
    assert_eq!(lo3.shape(), &[OUT_CH, fo]);

    eprintln!(
        "Sequential: conv_gelu→[{OUT_CH},{fo}], dconv→[{OUT_CH},{T_IN}], rewrite→[{OUT_CH},{fo}]"
    );
}
