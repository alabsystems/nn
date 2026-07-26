// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs decoder composition using the **real production
//! builders** from nn-models.
//!
//! Consolidates tests from three former files into one binary:
//! - `compose_demucs_temporal_decoder.rs` (7 tests) — temporal decoder block
//! - `compose_demucs_spectral_decoder.rs` (5 tests) — spectral decoder sub-defs
//! - `compose_demucs_spectral_decoder_advanced.rs` (4 tests) — CROWN / record / sequential
//!
//! Both decoder types share DConv sub-layer structure and weight layout from
//! `demucs_shared.rs`. Common helpers (push_weight, push_dconv_bindings,
//! binding builders) live in `helpers/demucs_decoder_block.rs`.
//!
//! Part of #1982: nn-verify test binary consolidation.

#[path = "demucs_decoder_block.rs"]
mod dec_helpers;

use super::common::{
    assert_bounds_valid, assert_crown_tighter_than_ibp, assert_crown_tighter_when_not_fallback,
    bounds_min_max, conv1d_out_len, uniform_bounds, verify_and_assert,
};
use nn_models::demucs_spectral_decoder_builders::{
    build_decoder_block_sub_defs as build_spectral_sub_defs, conv2d_output_len,
};
use nn_models::demucs_temporal_decoder_builders::build_decoder_block_def as build_temporal_def;
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph};

use dec_helpers::{
    spectral_conv_tr_bindings, spectral_conv_tr_f_out, spectral_dconv_bindings,
    spectral_rewrite_bindings, temporal_conv_tr_out_len, temporal_decoder_bindings, REWRITE_KERNEL,
    REWRITE_PADDING,
};

// ---------------------------------------------------------------------------
// Test-scale parameters (production uses 48-384 channels; we use 16/8)
// ---------------------------------------------------------------------------

/// Input channels to the decoder block.
const IN_CH: usize = 16;

/// Output channels after ConvTranspose1d.
const OUT_CH: usize = 8;

/// Temporal/time dimension at block input.
const T_IN: usize = 4;

/// Frequency dimension at spectral block input.
const F_IN: usize = 4;

// ---------------------------------------------------------------------------
// Temporal dimension helpers
// ---------------------------------------------------------------------------

/// Compute the output temporal dimension after the full temporal block
/// (Rewrite Conv1d → DConv → ConvTranspose1d).
fn compute_temporal_target_len(t_in: usize) -> usize {
    let rw_t_out = conv1d_out_len(t_in, REWRITE_KERNEL, 1, REWRITE_PADDING);
    temporal_conv_tr_out_len(rw_t_out)
}

// ---------------------------------------------------------------------------
// Spectral dimension helpers
// ---------------------------------------------------------------------------

/// Compute rewrite output dimensions (Conv2d(3×3, s=1, p=1) preserves spatial).
fn spectral_rewrite_output_dims() -> (usize, usize) {
    let rw_f = conv2d_output_len(F_IN, REWRITE_KERNEL, 1, REWRITE_PADDING)
        .expect("valid rewrite freq params");
    let rw_t = conv2d_output_len(T_IN, REWRITE_KERNEL, 1, REWRITE_PADDING)
        .expect("valid rewrite time params");
    (rw_f, rw_t)
}

// =========================================================================
// TEMPORAL DECODER TESTS (7 tests)
// =========================================================================

/// The production builder validates and produces a valid TensorKernelDef.
#[test]
fn test_temporal_production_builder_def_validates() {
    let target_len = compute_temporal_target_len(T_IN);
    let def = build_temporal_def(0, IN_CH, OUT_CH, T_IN, target_len, false)
        .expect("production builder should succeed");
    def.validate().expect("production def should validate");
}

/// Production builder TensorKernelDef translates to a NY graph.
#[test]
fn test_temporal_production_builder_graph_builds() {
    let target_len = compute_temporal_target_len(T_IN);
    let def =
        build_temporal_def(0, IN_CH, OUT_CH, T_IN, target_len, false).expect("production builder");
    let bindings = temporal_decoder_bindings(IN_CH, OUT_CH, T_IN);

    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("production def should translate to NY graph");

    // The production builder creates a complex graph: skip_add + Rewrite Conv1d
    // + GLU + DConv×2 (each: ZeroPad1d + Conv1d_dilated + GroupNorm + GELU +
    // Conv1d_1x1 + GroupNorm + GLU + LayerScale + residual_add) +
    // ConvTranspose1d + GELU. Should have many nodes.
    assert!(
        graph.num_nodes() >= 20,
        "production graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the production builder's graph.
#[test]
fn test_temporal_production_builder_ibp_propagates() {
    let target_len = compute_temporal_target_len(T_IN);
    let def =
        build_temporal_def(0, IN_CH, OUT_CH, T_IN, target_len, false).expect("production builder");
    let bindings = temporal_decoder_bindings(IN_CH, OUT_CH, T_IN);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through production decoder block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CH, target_len],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Temporal decoder block IBP: bounds=[{lo_min}, {hi_max}] over {} elements",
        OUT_CH * target_len
    );
}

/// CROWN propagation through the production builder's graph (may fall back to
/// IBP due to decomposed GroupNorm G=1, per design doc #697).
#[test]
fn test_temporal_production_builder_crown_propagation() {
    let target_len = compute_temporal_target_len(T_IN);
    let def =
        build_temporal_def(0, IN_CH, OUT_CH, T_IN, target_len, false).expect("production builder");
    let bindings = temporal_decoder_bindings(IN_CH, OUT_CH, T_IN);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[OUT_CH, target_len], "output shape mismatch");
    assert_bounds_valid(&output);

    eprintln!("Temporal decoder block: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Last block (is_last=true) omits GELU activation.
#[test]
fn test_temporal_production_builder_last_block() {
    let target_len = compute_temporal_target_len(T_IN);
    let def =
        build_temporal_def(3, IN_CH, OUT_CH, T_IN, target_len, true).expect("last block builder");
    let bindings = temporal_decoder_bindings(IN_CH, OUT_CH, T_IN);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through last block");
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[OUT_CH, target_len], "last block output shape");
    assert_bounds_valid(&output);
}

/// Record verification result under "demucs_temporal_decoder_production" key.
#[test]
fn test_temporal_production_builder_verify_and_record() {
    let target_len = compute_temporal_target_len(T_IN);
    let def =
        build_temporal_def(0, IN_CH, OUT_CH, T_IN, target_len, false).expect("production builder");
    let bindings = temporal_decoder_bindings(IN_CH, OUT_CH, T_IN);
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "demucs_temporal_decoder_production",
    );
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CH, target_len]);
}

/// Two sequential decoder blocks: block 0 output feeds block 1.
/// Verifies bounds propagation across multi-block decoder progression.
#[test]
fn test_temporal_production_builder_two_block_sequential() {
    // Block 0: IN_CH=16 → OUT_CH=8, T_IN=4
    let block0_target = compute_temporal_target_len(T_IN);
    let def0 =
        build_temporal_def(0, IN_CH, OUT_CH, T_IN, block0_target, false).expect("block 0 builder");
    let bindings0 = temporal_decoder_bindings(IN_CH, OUT_CH, T_IN);
    let graph0 = tensor_kernel_to_graph(&def0, &bindings0).expect("block 0 graph");
    let input0 = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output0 = graph0.propagate_ibp(&input0).expect("IBP through block 0");
    assert_bounds_valid(&output0);
    let (lo0, _) = output0.lower_upper();
    assert_eq!(
        lo0.shape(),
        &[OUT_CH, block0_target],
        "block 0 output shape"
    );

    // Block 1: in_ch=OUT_CH=8, out_ch=4, t_in=block0_target
    let block1_in_ch = OUT_CH;
    let block1_out_ch = 4;
    let block1_target = compute_temporal_target_len(block0_target);
    let def1 = build_temporal_def(
        1,
        block1_in_ch,
        block1_out_ch,
        block0_target,
        block1_target,
        true,
    )
    .expect("block 1 builder");
    let bindings1 = temporal_decoder_bindings(block1_in_ch, block1_out_ch, block0_target);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("block 1 graph");

    // Use block 0's output bounds as block 1's input bounds.
    let output1 = graph1.propagate_ibp(&output0).expect("IBP through block 1");
    assert_bounds_valid(&output1);
    let (lo1, _) = output1.lower_upper();
    assert_eq!(
        lo1.shape(),
        &[block1_out_ch, block1_target],
        "block 1 output shape"
    );

    eprintln!(
        "Two-block sequential: block0 output [{OUT_CH}, {block0_target}], block1 output [{block1_out_ch}, {block1_target}]"
    );
}

// =========================================================================
// SPECTRAL DECODER TESTS (5 tests from compose_demucs_spectral_decoder)
// =========================================================================

/// The production builder validates and produces 3 valid TensorKernelDefs.
#[test]
fn test_spectral_production_sub_defs_validate() {
    let (rw_f, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN); // trim to original freq

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
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
fn test_spectral_rewrite_sub_def_ibp() {
    let (rw_f, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
            .expect("builder");
    let bindings = spectral_rewrite_bindings(IN_CH, F_IN, T_IN);
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
fn test_spectral_dconv_sub_def_ibp() {
    let (_, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(F_IN);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, F_IN, rw_t, target_f, false)
            .expect("builder");
    let bindings = spectral_dconv_bindings(IN_CH);
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
fn test_spectral_conv_tr_sub_def_ibp() {
    let (rw_f, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
            .expect("builder");
    let bindings = spectral_conv_tr_bindings(IN_CH, OUT_CH);
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

/// CROWN produces tighter-or-equal bounds than IBP on ConvTranspose sub-def.
#[test]
fn test_spectral_conv_tr_sub_def_crown_tighter_than_ibp() {
    let (rw_f, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
            .expect("builder");
    let bindings = spectral_conv_tr_bindings(IN_CH, OUT_CH);
    let graph = tensor_kernel_to_graph(&sub_defs.conv_tr_def, &bindings)
        .expect("conv_tr graph translation");

    let input = uniform_bounds(&[IN_CH, rw_f], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through ConvTranspose sub-def");
    let (_, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through ConvTranspose sub-def");

    assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}

// =========================================================================
// SPECTRAL DECODER ADVANCED TESTS (4 tests from compose_demucs_spectral_decoder_advanced)
// =========================================================================

/// CROWN propagation through the DConv sub-def (may fall back to IBP due
/// to decomposed GroupNorm G=1, per design doc #697).
#[test]
fn test_spectral_dconv_sub_def_crown() {
    let (_, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(F_IN);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, F_IN, rw_t, target_f, false)
            .expect("builder");
    let bindings = spectral_dconv_bindings(IN_CH);
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
fn test_spectral_dconv_sub_def_verify_and_record() {
    let (_, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(F_IN);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, F_IN, rw_t, target_f, false)
            .expect("builder");
    let bindings = spectral_dconv_bindings(IN_CH);
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
fn test_spectral_production_builder_last_block() {
    let (rw_f, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(3, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, true)
            .expect("last block builder");

    sub_defs.rewrite_def.validate().expect("rewrite validates");
    sub_defs.dconv_def.validate().expect("dconv validates");
    sub_defs.conv_tr_def.validate().expect("conv_tr validates");

    let bindings = spectral_conv_tr_bindings(IN_CH, OUT_CH);
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
fn test_spectral_sequential_sub_def_composition() {
    let (rw_f, rw_t) = spectral_rewrite_output_dims();
    let ct_f = spectral_conv_tr_f_out(rw_f);
    let target_f = ct_f.min(F_IN);

    let sub_defs =
        build_spectral_sub_defs(0, IN_CH, OUT_CH, F_IN, T_IN, rw_f, rw_t, target_f, false)
            .expect("builder");

    // Stage 1: Rewrite — input [IN_CH, F*T], output [IN_CH, rw_f*rw_t].
    let ft = F_IN * T_IN;
    let rw_ft = rw_f * rw_t;
    let rw_bindings = spectral_rewrite_bindings(IN_CH, F_IN, T_IN);
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
    let dc_bindings = spectral_dconv_bindings(IN_CH);
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
    let ct_bindings = spectral_conv_tr_bindings(IN_CH, OUT_CH);
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
