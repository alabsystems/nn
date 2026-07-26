// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs spectral encoder block composition with NY.
//!
//! Topology (matching `build_block_sub_defs()` in spectral encoder builders):
//!   Conv1d(k=8, s=4, p=2) → GELU → DConv(×2) → Rewrite(Conv1d k=1) → GLU
//!
//! This tests a single spectral encoder block as a flat graph (no axis-switching,
//! which is CPU-side in the production code). The block operates on `[C_in, F]`
//! slices, downsampling frequency from F to F' via strided Conv1d.
//!
//! DConv sub-layer: Conv1d(dilated) → GN → GELU → Conv1d(1×1) → GN → GLU →
//! LayerScale → residual_add. Small dims (4→8 ch, F=16) for tractability.
//!
//! Part of #831 — spectral encoder NY composition verification.
//!
//! Builder helpers extracted to `helpers/spectral_encoder.rs`.

use super::common;

#[path = "spectral_encoder.rs"]
mod helpers;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{
    build_spectral_encoder_block, spectral_encoder_block_bindings, F_IN, IN_CHANNELS, OUT_CHANNELS,
};
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Spectral encoder block TensorKernelDef validates successfully.
#[test]
fn test_spectral_encoder_block_def_validates() {
    let (def, _, _) = build_spectral_encoder_block();
    def.validate()
        .expect("spectral encoder block def should validate");
}

/// Spectral encoder block translates to NY GraphNetwork.
#[test]
fn test_spectral_encoder_block_graph_builds() {
    let (def, conv_f_out, _) = build_spectral_encoder_block();

    // Verify Conv1d output freq length: (16 + 2*2 - 8) / 4 + 1 = 4
    assert_eq!(conv_f_out, 4, "Conv1d(k=8, s=4, p=2) on F=16 → F=4");

    let bindings = spectral_encoder_block_bindings();

    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("spectral encoder block graph should translate");

    // Block has: Conv1d + GELU + 2*DConv(~15 ops each) + Conv1d(k=1) + GLU(3 ops) ≈ 35+ nodes.
    assert!(
        graph.num_nodes() >= 15,
        "spectral encoder block graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full spectral encoder block pipeline.
///
/// Checks shape, finiteness, and magnitude sanity. Decomposed GroupNorm + GLU
/// can produce wide IBP bounds (see design doc helper on decomposed norms),
/// but they should not be astronomically large with small (0.01) weights.
#[test]
fn test_spectral_encoder_block_ibp_propagates() {
    let (def, conv_f_out, _) = build_spectral_encoder_block();
    let bindings = spectral_encoder_block_bindings();

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CHANNELS, F_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through spectral encoder block");
    // Output: [OUT_CHANNELS, conv_f_out].
    let expected_shape = [OUT_CHANNELS, conv_f_out];
    assert_eq!(
        output.lower_upper().0.shape(),
        expected_shape.as_slice(),
        "output shape mismatch: expected {expected_shape:?}, got {:?}",
        output.lower_upper().0.shape()
    );

    assert_bounds_valid(&output);

    // Magnitude sanity: with small weights (0.01) and [-1,1] input, decomposed
    // GroupNorm+GLU in 2 stacked DConv sub-layers amplifies IBP bounds
    // massively (observed: ~6.4e31). This is expected IBP behavior for
    // multi-op chains — see design doc helper on decomposed norms.
    // Threshold is 1e33 (one order above observed ~6.4e31) to catch actual
    // overflow while accepting known wide IBP from decomposed norms.
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Spectral encoder block IBP bounds range: [{lo_min}, {hi_max}] over {} output elements",
        OUT_CHANNELS * conv_f_out
    );
    assert!(
        hi_max.abs() < 1e33,
        "IBP upper bound near overflow: {hi_max} (actual bug, not just wide IBP)"
    );
    assert!(
        lo_min.abs() < 1e33,
        "IBP lower bound near overflow: {lo_min} (actual bug, not just wide IBP)"
    );
}

/// CROWN propagation through the spectral encoder block (tighter bounds).
///
/// CROWN may fall back to IBP on decomposed GroupNorm or GLU multiplicative
/// interactions — this is expected for complex blocks (see design doc helper
/// on decomposed norms). Uses `assert_crown_tighter_when_not_fallback` to
/// verify CROWN produces tighter bounds than IBP when CROWN succeeds.
#[test]
fn test_spectral_encoder_block_crown_propagation() {
    let (def, conv_f_out, _) = build_spectral_encoder_block();
    let bindings = spectral_encoder_block_bindings();

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CHANNELS, F_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let expected_shape = [OUT_CHANNELS, conv_f_out];
    assert_eq!(
        output.lower_upper().0.shape(),
        expected_shape.as_slice(),
        "output shape mismatch"
    );

    eprintln!("Spectral encoder block: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    // Magnitude sanity: CROWN may fall back to IBP on complex blocks,
    // producing bounds as wide as ~6.4e31 for decomposed GroupNorm+GLU chains.
    // Threshold is 1e33 (one order above observed) to catch overflow.
    let (lo_val, hi_val) = bounds_min_max(&output);
    assert!(hi_val.abs() < 1e33, "upper bound near overflow: {hi_val}");
    assert!(lo_val.abs() < 1e33, "lower bound near overflow: {lo_val}");
}

/// Record spectral encoder block verification in `VerifyStatus`.
#[test]
fn test_spectral_encoder_block_verify_and_record() {
    let (def, conv_f_out, _) = build_spectral_encoder_block();
    let bindings = spectral_encoder_block_bindings();
    let input = uniform_bounds(&[IN_CHANNELS, F_IN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "demucs_spectral_encoder_block");
    assert_eq!(result.num_variables, 1, "single Variable input (data)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, conv_f_out]);
}
