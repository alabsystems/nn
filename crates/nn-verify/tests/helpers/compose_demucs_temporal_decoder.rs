// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs temporal decoder composition using the
//! **real production builder** `build_decoder_block_def()` from nn-models.
//!
//! Unlike `compose_demucs_decoder_block.rs` (which builds a simplified graph
//! manually), this test calls the actual builder function used by
//! `DemucsTemporalDecoder::new()`. This verifies that production-generated
//! `TensorKernelDef`s are translatable to NY and that bounds
//! propagate correctly.
//!
//! Dimensions are scaled down (in_ch=16, out_ch=8) for NY tractability.
//! The internal DCONV_COMPRESS=4, DCONV_DEPTH=2, DCONV_KERNEL=3 constants from
//! `demucs_shared.rs` are used by the builder unchanged.
//!
//! Part of #779 Phase A — composition verification with production builders.

use super::common;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert,
};
use nn_models::demucs_temporal_decoder_builders::build_decoder_block_def;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Test-scale parameters (production uses 48-384 channels; we use 16/8)
// ---------------------------------------------------------------------------

/// Input channels to the decoder block.
const IN_CH: usize = 16;

/// Output channels after ConvTranspose1d.
const OUT_CH: usize = 8;

/// Temporal dimension at block input.
const T_IN: usize = 4;

/// ConvTranspose1d kernel (must match production KERNEL_SIZE=8).
const KERNEL_SIZE: usize = 8;

/// ConvTranspose1d stride (must match production STRIDE=4).
const STRIDE: usize = 4;

/// ConvTranspose1d padding (KERNEL_SIZE / 4 = 2, matches production).
const CONV_TR_PADDING: usize = KERNEL_SIZE / 4;

/// Rewrite Conv1d kernel (matches production REWRITE_KERNEL=3).
const REWRITE_KERNEL: usize = 3;

/// Rewrite Conv1d padding (REWRITE_KERNEL / 2 = 1, matches production).
const REWRITE_PADDING: usize = REWRITE_KERNEL / 2;

/// Weight magnitude: small to keep IBP bounds tractable through
/// decomposed GroupNorm G=1 (which amplifies through 14 primitive ops).
const WEIGHT_MAG: f32 = 0.001;

/// DCONV compress ratio used by the production builder (from demucs_shared).
const DCONV_COMPRESS: usize = 4;

/// DCONV depth used by the production builder (from demucs_shared).
const DCONV_DEPTH: usize = 2;

/// DCONV kernel size used by the production builder (from demucs_shared).
const DCONV_KERNEL: usize = 3;

// ---------------------------------------------------------------------------
// Binding helpers — build TensorParamBinding vectors matching the builder's
// input declarations in the exact order they appear.
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

/// Build bindings for a single decoder block: data=Variable, skip=ConstantTensor(zeros),
/// then rewrite + DConv + ConvTranspose1d weights in builder declaration order.
fn decoder_block_bindings(in_ch: usize, out_ch: usize, t_in: usize) -> Vec<TensorParamBinding> {
    let compressed = in_ch / DCONV_COMPRESS;
    let doubled = in_ch * 2;
    let mut b = Vec::new();

    // Variable inputs: data, skip
    b.push(TensorParamBinding::Variable); // data [in_ch, t_in]
    b.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        // skip [in_ch, t_in] (zeros)
        IxDyn(&[in_ch, t_in]),
        0.0f32,
    )));

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

/// Compute the output temporal dimension after the full block
/// (Rewrite Conv1d → DConv → ConvTranspose1d → trim).
fn compute_target_len(t_in: usize) -> usize {
    let rw_t_out = conv1d_out_len(t_in, REWRITE_KERNEL, 1, REWRITE_PADDING);
    // No trim: full ConvTranspose1d output. In production, target_len =
    // min(ct_t_out, encoder_lengths[encoder_depth]).
    (rw_t_out - 1) * STRIDE + KERNEL_SIZE - 2 * CONV_TR_PADDING
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The production builder validates and produces a valid TensorKernelDef.
#[test]
fn test_production_builder_def_validates() {
    let target_len = compute_target_len(T_IN);
    let def = build_decoder_block_def(0, IN_CH, OUT_CH, T_IN, target_len, false)
        .expect("production builder should succeed");
    def.validate().expect("production def should validate");
}

/// Production builder TensorKernelDef translates to a NY graph.
#[test]
fn test_production_builder_graph_builds() {
    let target_len = compute_target_len(T_IN);
    let def = build_decoder_block_def(0, IN_CH, OUT_CH, T_IN, target_len, false)
        .expect("production builder");
    let bindings = decoder_block_bindings(IN_CH, OUT_CH, T_IN);

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
fn test_production_builder_ibp_propagates() {
    let target_len = compute_target_len(T_IN);
    let def = build_decoder_block_def(0, IN_CH, OUT_CH, T_IN, target_len, false)
        .expect("production builder");
    let bindings = decoder_block_bindings(IN_CH, OUT_CH, T_IN);
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
        "Production decoder block IBP: bounds=[{lo_min}, {hi_max}] over {} elements",
        OUT_CH * target_len
    );
}

/// CROWN propagation through the production builder's graph (may fall back to
/// IBP due to decomposed GroupNorm G=1, per design doc #697).
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_production_builder_crown_propagation() {
    let target_len = compute_target_len(T_IN);
    let def = build_decoder_block_def(0, IN_CH, OUT_CH, T_IN, target_len, false)
        .expect("production builder");
    let bindings = decoder_block_bindings(IN_CH, OUT_CH, T_IN);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(lo.shape(), &[OUT_CH, target_len], "output shape mismatch");
    assert_bounds_valid(&output);

    eprintln!("Production decoder block: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Last block (is_last=true) omits GELU activation.
#[test]
fn test_production_builder_last_block() {
    let target_len = compute_target_len(T_IN);
    let def = build_decoder_block_def(3, IN_CH, OUT_CH, T_IN, target_len, true)
        .expect("last block builder");
    let bindings = decoder_block_bindings(IN_CH, OUT_CH, T_IN);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through last block");
    let (lo, _) = output.lower_upper();

    // Last block output is [OUT_CH, target_len] without GELU clamping.
    assert_eq!(lo.shape(), &[OUT_CH, target_len], "last block output shape");
    assert_bounds_valid(&output);
}

/// Record verification result under "demucs_temporal_decoder_production" key.
#[test]
fn test_production_builder_verify_and_record() {
    let target_len = compute_target_len(T_IN);
    let def = build_decoder_block_def(0, IN_CH, OUT_CH, T_IN, target_len, false)
        .expect("production builder");
    let bindings = decoder_block_bindings(IN_CH, OUT_CH, T_IN);
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
fn test_production_builder_two_block_sequential() {
    // Block 0: IN_CH=16 → OUT_CH=8, T_IN=4
    let block0_target = compute_target_len(T_IN);
    let def0 = build_decoder_block_def(0, IN_CH, OUT_CH, T_IN, block0_target, false)
        .expect("block 0 builder");
    let bindings0 = decoder_block_bindings(IN_CH, OUT_CH, T_IN);
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
    let block1_target = compute_target_len(block0_target);
    let def1 = build_decoder_block_def(
        1,
        block1_in_ch,
        block1_out_ch,
        block0_target,
        block1_target,
        true,
    )
    .expect("block 1 builder");
    let bindings1 = decoder_block_bindings(block1_in_ch, block1_out_ch, block0_target);
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
