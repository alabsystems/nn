// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs temporal encoder block composition with NY.
//!
//! Two test families:
//!
//! ## Full encoder block (Conv1d + GELU + DConv x2 + Rewrite + GLU)
//!
//! Uses the full `build_encoder_block()` builder with IN=8, OUT=16, T=16.
//!
//! ## `chains` -- Individual Conv1d -> normalization -> activation sub-chains
//!
//! Isolates the primitive composition patterns that appear inside the encoder:
//! Conv1d -> GroupNorm -> GELU, Conv1d -> InstanceNorm -> ReLU, etc. Small dims
//! (C<=16, T<=8) for NY tractability.
//!
//! Part of #3595 -- Compose verification for HTDemucs temporal encoder.
//! Part of #779 Phase E -- encoder composition verification.

use super::common;

// Builder helpers extracted to keep this file under 500 lines (#1669).
#[path = "temporal_encoder.rs"]
mod helpers;

// Chain sub-tests extracted to stay under 500 lines (#3595).
#[path = "temporal_encoder_chains.rs"]
mod chains;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{build_encoder_block, encoder_block_bindings, IN_CHANNELS, OUT_CHANNELS, T_IN};
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Full encoder block tests
// ---------------------------------------------------------------------------

/// Encoder block TensorKernelDef validates successfully.
#[test]
fn test_encoder_block_def_validates() {
    let (def, _, _) = build_encoder_block();
    def.validate().expect("encoder block def should validate");
}

/// Encoder block translates to NY GraphNetwork.
#[test]
fn test_encoder_block_graph_builds() {
    let (def, conv_t_out, _) = build_encoder_block();

    // Verify Conv1d output temporal length: (16 + 2*2 - 8) / 4 + 1 = 4
    assert_eq!(conv_t_out, 4, "Conv1d(k=8, s=4, p=2) on T=16 -> T=4");

    let bindings = encoder_block_bindings();

    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("encoder block graph should translate");

    // The block has: Conv1d + GELU + 2*DConv(~15 ops each) + Conv1d(k=1) + GLU(3 ops) >= 35 nodes.
    assert!(
        graph.num_nodes() >= 15,
        "encoder block graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full encoder block pipeline.
#[test]
fn test_encoder_block_ibp_propagates() {
    let (def, conv_t_out, _) = build_encoder_block();
    let bindings = encoder_block_bindings();

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CHANNELS, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder block");
    let (lo, _hi) = output.lower_upper();

    // Output: [OUT_CHANNELS, conv_t_out].
    let expected_shape = [OUT_CHANNELS, conv_t_out];
    assert_eq!(
        lo.shape(),
        expected_shape.as_slice(),
        "output shape mismatch: expected {expected_shape:?}, got {:?}",
        lo.shape()
    );

    assert_bounds_valid(&output);
}

/// CROWN propagation through the encoder block (tighter bounds).
///
/// CROWN may fall back to IBP on decomposed GroupNorm or GLU multiplicative
/// interactions -- this is expected for complex blocks.
/// When CROWN succeeds, asserts tighter-than-IBP invariant.
#[test]
fn test_encoder_block_crown_propagation() {
    let (def, conv_t_out, _) = build_encoder_block();
    let bindings = encoder_block_bindings();

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CHANNELS, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _hi) = output.lower_upper();

    let expected_shape = [OUT_CHANNELS, conv_t_out];
    assert_eq!(
        lo.shape(),
        expected_shape.as_slice(),
        "output shape mismatch"
    );

    eprintln!("Demucs encoder block: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// IBP bounds remain finite through the DConv residual chain.
#[test]
fn test_encoder_block_bounds_finite() {
    let (def, conv_t_out, _) = build_encoder_block();
    let bindings = encoder_block_bindings();

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CHANNELS, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder block");
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Encoder block IBP bounds range: [{lo_min}, {hi_max}] over {} output elements",
        OUT_CHANNELS * conv_t_out
    );

    assert!(
        lo_min.is_finite(),
        "output lower bound min should be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "output upper bound max should be finite, got {hi_max}"
    );
}

/// Record encoder block verification in VerifyStatus.
#[test]
fn test_encoder_block_verify_and_record() {
    let (def, conv_t_out, _) = build_encoder_block();
    let bindings = encoder_block_bindings();
    let input = uniform_bounds(&[IN_CHANNELS, T_IN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "demucs_temporal_encoder_block");
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, conv_t_out]);
}
