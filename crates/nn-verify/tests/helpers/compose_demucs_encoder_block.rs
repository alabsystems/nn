// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs encoder block composition — both **production
//! builder** tests (`build_encoder_block_def()` from nn-models) and
//! **parametric test-local builder** tests (temporal + spectral configs).
//!
//! Topology (shared by temporal and spectral encoders):
//!   Conv1d(stride=4) → GELU → DConv(×depth residual) → Rewrite(Conv1d k=1) → GLU
//!
//! DConv sub-layer:
//!   Conv1d(dilated) → GN(G=1) → GELU → Conv1d(1×1) → GN(G=1) → GLU →
//!   LayerScale → residual_add
//!
//! Production builder tests use DCONV_COMPRESS=4, DCONV_KERNEL=3, KERNEL_SIZE=8,
//! STRIDE=4 with small channel counts (IN_CH=8, OUT_CH=16).
//!
//! Parametric tests use `helpers/demucs_encoder_block.rs` configs:
//!   - Temporal: 8→16 ch, T=16
//!   - Spectral: 4→8 ch, F=16
//!
//! Part of #779 Phase E and #1982 test binary consolidation.

use super::common;

#[path = "demucs_encoder_block.rs"]
mod enc_helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use enc_helpers::{
    build_encoder_block, encoder_block_bindings, EncoderBlockConfig, SPECTRAL_CONFIG,
    TEMPORAL_CONFIG,
};
use nn_models::demucs_temporal_encoder_builders::{build_encoder_block_def, conv1d_out_len};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Small-scale parameters matching production builder constraints
// ---------------------------------------------------------------------------

/// Input channels (small scale; block0 would use AUDIO_CHANNELS=2, but we
/// pick 8 so out_ch/DCONV_COMPRESS=16/4=4 is valid).
const IN_CH: usize = 8;

/// Output channels — must be divisible by DCONV_COMPRESS (4).
const OUT_CH: usize = 16;

/// Temporal input length (already padded to stride multiple).
/// Must produce valid Conv1d output with k=8, s=4, p=2: (16+4-8)/4+1 = 4.
const PADDED_T: usize = 16;

/// DConv sub-layer count (production constant).
const DCONV_DEPTH: usize = 2;

/// DConv kernel size (production constant).
const DCONV_KERNEL: usize = 3;

/// Weight magnitude for small random-like weights.
const WEIGHT_MAG: f32 = 0.001;

/// Encoder Conv1d kernel size (production constant).
const KERNEL_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Binding helpers
// ---------------------------------------------------------------------------

/// Build parameter bindings for the default test-scale encoder block.
fn encoder_bindings() -> Vec<TensorParamBinding> {
    encoder_bindings_custom(IN_CH, OUT_CH)
}

/// Push 11 bindings for one DConv sub-layer with given channel dimensions.
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

/// Build encoder block bindings for arbitrary channel dimensions.
fn encoder_bindings_custom(in_ch: usize, out_ch: usize) -> Vec<TensorParamBinding> {
    let doubled = out_ch * 2;
    let compressed = out_ch / 4; // DCONV_COMPRESS=4
    let mut b = vec![TensorParamBinding::Variable];

    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch, in_ch, KERNEL_SIZE]),
        WEIGHT_MAG,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_ch]),
        0.0f32,
    )));

    for _k in 0..DCONV_DEPTH {
        push_dconv_bindings(&mut b, out_ch, compressed);
    }

    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled, out_ch, 1]),
        WEIGHT_MAG,
    )));
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[doubled]),
        0.0f32,
    )));

    b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Production encoder block def validates successfully.
#[test]
fn test_production_encoder_block_validates() {
    let def = build_encoder_block_def(0, IN_CH, OUT_CH, PADDED_T)
        .expect("production encoder block should build");
    def.validate()
        .expect("production encoder block should validate");
}

/// Production encoder block translates to NY graph.
#[test]
fn test_production_encoder_block_graph_builds() {
    let def = build_encoder_block_def(0, IN_CH, OUT_CH, PADDED_T).expect("build encoder block");
    let bindings = encoder_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("encoder block graph should translate");

    // Block has Conv1d + GELU + 2×DConv(~15 ops) + Conv1d(k=1) + GLU ≈ 35+ nodes.
    assert!(
        graph.num_nodes() >= 15,
        "encoder block should have >=15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the production encoder block.
#[test]
fn test_production_encoder_block_ibp() {
    let conv_t_out = conv1d_out_len(PADDED_T).unwrap();
    let def = build_encoder_block_def(0, IN_CH, OUT_CH, PADDED_T).expect("build encoder block");
    let bindings = encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, PADDED_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder block");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CH, conv_t_out],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Production encoder block IBP: bounds=[{lo_min}, {hi_max}], shape=[{OUT_CH}, {conv_t_out}]"
    );
}

/// CROWN propagation through the production encoder block.
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds (not fallback).
#[test]
fn test_production_encoder_block_crown() {
    let conv_t_out = conv1d_out_len(PADDED_T).unwrap();
    let def = build_encoder_block_def(0, IN_CH, OUT_CH, PADDED_T).expect("build encoder block");
    let bindings = encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, PADDED_T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[OUT_CH, conv_t_out], "output shape mismatch");

    eprintln!("Production encoder block: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Record production encoder block verification in status file.
#[test]
fn test_production_encoder_block_verify_and_record() {
    let conv_t_out = conv1d_out_len(PADDED_T).unwrap();
    let def = build_encoder_block_def(0, IN_CH, OUT_CH, PADDED_T).expect("build encoder block");
    let bindings = encoder_bindings();
    let input = uniform_bounds(&[IN_CH, PADDED_T], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "demucs_temporal_encoder_prod_block0",
    );
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, conv_t_out]);
}

/// Sequential composition: two production encoder blocks chained.
///
/// Block 0: [IN_CH, PADDED_T] → [OUT_CH, conv_t_out]
/// Block 1: [OUT_CH, padded_t1] → [32, conv_t_out1]
#[test]
fn test_production_encoder_two_blocks() {
    let conv_t_out = conv1d_out_len(PADDED_T).unwrap();

    // Block 0: IBP propagation.
    let def0 = build_encoder_block_def(0, IN_CH, OUT_CH, PADDED_T).expect("block 0");
    let graph0 = tensor_kernel_to_graph(&def0, &encoder_bindings()).expect("graph0");
    let output0 = graph0
        .propagate_ibp(&uniform_bounds(&[IN_CH, PADDED_T], 1.0))
        .expect("IBP block 0");
    assert_bounds_valid(&output0);
    let (lo0_min, hi0_max) = bounds_min_max(&output0);

    // Block 1: in_ch=16, out_ch=32 (next depth).
    let out_ch1: usize = 32;
    let padded_t1 = conv_t_out;
    let conv_t_out1 = conv1d_out_len(padded_t1).unwrap();

    let def1 = build_encoder_block_def(1, OUT_CH, out_ch1, padded_t1).expect("block 1");
    let bindings1 = encoder_bindings_custom(OUT_CH, out_ch1);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph1");

    // Clamp block0 bounds to ±10 — raw IBP is vacuously wide (#697).
    let block0_mag = hi0_max.abs().max(lo0_min.abs()).min(10.0);
    let output1 = graph1
        .propagate_ibp(&uniform_bounds(&[OUT_CH, padded_t1], block0_mag))
        .expect("IBP block 1");
    assert_bounds_valid(&output1);

    let (lo1, _) = output1.lower_upper();
    assert_eq!(lo1.shape(), &[out_ch1, conv_t_out1]);
    eprintln!("Two-block: block0=[{lo0_min}, {hi0_max}], block1 shape=[{out_ch1}, {conv_t_out1}]");
}

/// Production builder with block_idx=3 (last encoder block).
#[test]
fn test_production_encoder_last_block() {
    let def =
        build_encoder_block_def(3, IN_CH, OUT_CH, PADDED_T).expect("build last encoder block");
    def.validate().expect("last block should validate");

    let bindings = encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("last block graph should translate");

    let input = uniform_bounds(&[IN_CH, PADDED_T], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through last block");
    assert_bounds_valid(&output);
}

// ===========================================================================
// Parametric encoder block tests (temporal + spectral via EncoderBlockConfig)
// ===========================================================================

/// Shared runner: validates the encoder block def built from a config.
fn run_encoder_validates(cfg: &EncoderBlockConfig) {
    let (def, _, _) = build_encoder_block(cfg);
    def.validate().expect("encoder block def should validate");
}

/// Shared runner: translates to NY graph and checks node count.
fn run_encoder_graph_builds(cfg: &EncoderBlockConfig) {
    let (def, conv_out, _) = build_encoder_block(cfg);
    let expected_spatial = common::conv1d_out_len(
        cfg.spatial_in,
        cfg.conv_kernel,
        cfg.conv_stride,
        cfg.conv_padding,
    );
    assert_eq!(conv_out, expected_spatial, "conv output spatial mismatch");

    let bindings = encoder_block_bindings(cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph should translate");
    assert!(
        graph.num_nodes() >= 15,
        "encoder block graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// Shared runner: IBP propagation, returns `(lo_min, hi_max)` for caller assertions.
fn run_encoder_ibp(cfg: &EncoderBlockConfig) -> (f32, f32) {
    let (def, conv_out, _) = build_encoder_block(cfg);
    let bindings = encoder_block_bindings(cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[cfg.in_channels, cfg.spatial_in], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[cfg.out_channels, conv_out],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "{} IBP bounds: [{lo_min}, {hi_max}], shape=[{}, {conv_out}]",
        cfg.block_name, cfg.out_channels
    );
    (lo_min, hi_max)
}

/// Shared runner: CROWN propagation, returns `(lo_min, hi_max)` for caller assertions.
fn run_encoder_crown(cfg: &EncoderBlockConfig) -> (f32, f32) {
    let (def, conv_out, _) = build_encoder_block(cfg);
    let bindings = encoder_block_bindings(cfg);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[cfg.in_channels, cfg.spatial_in], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[cfg.out_channels, conv_out],
        "output shape mismatch"
    );

    eprintln!("{}: method={method:?}", cfg.block_name);
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    (lo_min, hi_max)
}

/// Shared runner: verify_and_record pipeline.
fn run_encoder_verify_and_record(cfg: &EncoderBlockConfig, status_key: &str) {
    let (def, conv_out, _) = build_encoder_block(cfg);
    let bindings = encoder_block_bindings(cfg);
    let input = uniform_bounds(&[cfg.in_channels, cfg.spatial_in], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, status_key);
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[cfg.out_channels, conv_out]);
}

// ---------------------------------------------------------------------------
// Temporal encoder block parametric tests
// ---------------------------------------------------------------------------

#[test]
fn test_temporal_encoder_block_def_validates() {
    run_encoder_validates(&TEMPORAL_CONFIG);
}

#[test]
fn test_temporal_encoder_block_graph_builds() {
    run_encoder_graph_builds(&TEMPORAL_CONFIG);
}

#[test]
fn test_temporal_encoder_block_ibp_propagates() {
    run_encoder_ibp(&TEMPORAL_CONFIG);
}

#[test]
fn test_temporal_encoder_block_crown_propagation() {
    run_encoder_crown(&TEMPORAL_CONFIG);
}

/// IBP bounds remain finite through the DConv residual chain.
#[test]
fn test_temporal_encoder_block_bounds_finite() {
    let (lo_min, hi_max) = run_encoder_ibp(&TEMPORAL_CONFIG);
    assert!(
        lo_min.is_finite(),
        "output lower bound min should be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "output upper bound max should be finite, got {hi_max}"
    );
}

#[test]
fn test_temporal_encoder_block_verify_and_record() {
    run_encoder_verify_and_record(&TEMPORAL_CONFIG, "demucs_temporal_encoder_block");
}

// ---------------------------------------------------------------------------
// Spectral encoder block parametric tests
// ---------------------------------------------------------------------------

#[test]
fn test_spectral_encoder_block_def_validates() {
    run_encoder_validates(&SPECTRAL_CONFIG);
}

#[test]
fn test_spectral_encoder_block_graph_builds() {
    run_encoder_graph_builds(&SPECTRAL_CONFIG);
}

/// IBP through spectral encoder block with magnitude overflow check.
#[test]
fn test_spectral_encoder_block_ibp_propagates() {
    let (lo_min, hi_max) = run_encoder_ibp(&SPECTRAL_CONFIG);
    // Magnitude sanity: decomposed GroupNorm+GLU amplifies IBP bounds massively
    // (observed: ~6.4e31). Threshold is 1e33 (one order above observed).
    assert!(
        hi_max.abs() < 1e33,
        "IBP upper bound near overflow: {hi_max}"
    );
    assert!(
        lo_min.abs() < 1e33,
        "IBP lower bound near overflow: {lo_min}"
    );
}

/// CROWN through spectral encoder block with magnitude overflow check.
#[test]
fn test_spectral_encoder_block_crown_propagation() {
    let (lo_min, hi_max) = run_encoder_crown(&SPECTRAL_CONFIG);
    assert!(hi_max.abs() < 1e33, "upper bound near overflow: {hi_max}");
    assert!(lo_min.abs() < 1e33, "lower bound near overflow: {lo_min}");
}

#[test]
fn test_spectral_encoder_block_verify_and_record() {
    run_encoder_verify_and_record(&SPECTRAL_CONFIG, "demucs_spectral_encoder_block");
}
