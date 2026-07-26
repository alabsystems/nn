// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Kokoro decoder (ISTFTNet generator) NY
//! composition.
//!
//! Tests the vocoder architecture: Conv1d → ConvTranspose1d (upsample) →
//! ResBlock(InstanceNorm + Snake + Conv1d) + residual → Conv1d → Exp.
//!
//! Key Kokoro-specific operations verified:
//! - InstanceNorm + style affine (AdaIN decomposition)
//! - Snake activation (elementwise with broadcast alpha)
//! - ConvTranspose1d upsampling
//! - Exp activation (log-magnitude → magnitude domain)
//! - Residual connections
//!
//! **CROWN status (#2773):** CROWN succeeds since NY 359a195+.
//! Previous IBP fallback (#1769) resolved by batched CROWN Concat fix.
//!
//! Part of #1696 AC6: Kokoro decoder NY composition.

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder_helpers;

use super::common::{
    assert_bounds_valid, assert_crown_succeeds, bounds_min_max, uniform_bounds, verify_and_assert,
};
use kokoro_decoder_helpers::{
    build_kokoro_decoder, build_kokoro_decoder_with_leaky_relu, kokoro_decoder_bindings,
    kokoro_decoder_leaky_relu_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Kokoro decoder TensorKernelDef validates.
#[test]
fn test_kokoro_decoder_def_validates() {
    let (def, _) = build_kokoro_decoder();
    def.validate().expect("kokoro decoder def should validate");
}

/// Kokoro decoder translates to NY GraphNetwork.
#[test]
fn test_kokoro_decoder_graph_builds() {
    let (def, out_shape) = build_kokoro_decoder();
    assert_eq!(out_shape, [OUT_CHANNELS, TIME_UP]);

    let bindings = kokoro_decoder_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("kokoro decoder graph should translate");

    // Conv1d + ConvTranspose1d + InstanceNorm + Snake (native SnakeLayer)
    // + Conv1d + residual_add + Conv1d + Exp = 8 nodes.
    // Snake uses native SnakeLayer fast path (1 node) rather than
    // decomposed IR, so the graph is compact.
    assert!(
        graph.num_nodes() >= 8,
        "kokoro decoder graph should have >= 8 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the Kokoro decoder.
#[test]
fn test_kokoro_decoder_ibp_propagates() {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[8, TIME_IN], 1.0); // IN_CHANNELS=8

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Kokoro decoder");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CHANNELS, TIME_UP],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Kokoro decoder IBP: bounds=[{lo_min}, {hi_max}]");

    // With small weights and [-1, 1] input, exp(small_value) should be
    // close to 1. IBP may widen due to InstanceNorm decomposition.
    assert!(
        lo_min > 0.0,
        "exp output should be positive, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e6,
        "IBP upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// CROWN propagation through the Kokoro decoder.
///
/// Since NY 359a195+ (#2773), CROWN succeeds for the decoder graph
/// (InstanceNorm + Snake + Conv1d + LeakyReLU + Exp). The previous IBP
/// fallback (#1769) is resolved.
#[test]
fn test_kokoro_decoder_crown_propagation() {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let output = assert_crown_succeeds(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CHANNELS, TIME_UP],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Kokoro decoder: method=Crown, bounds=[{lo_min}, {hi_max}]");

    // Magnitude assertions matching IBP counterpart (#1984 AC1):
    // exp output must be positive, and upper bound should be bounded.
    assert!(
        lo_min > 0.0,
        "CROWN: exp output should be positive, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e6,
        "CROWN: upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// Kokoro decoder verify and record under "kokoro_decoder" key.
#[test]
fn test_kokoro_decoder_verify_and_record() {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input (features)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, TIME_UP]);

    // Soundness provenance must be set (#1984 AC2).
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ForwardMode NormBoundsMode should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ---------------------------------------------------------------------------
// LeakyReLU-expanded decoder tests (#1741)
// ---------------------------------------------------------------------------
//
// The expanded decoder adds LeakyReLU(0.1) before upsample and
// LeakyReLU(0.01) before conv_post, matching the real Kokoro ISTFTNet
// architecture. LeakyReLU is piecewise-linear, so it introduces no
// additional approximation error in NY (native LeakyReLULayer).

/// Expanded Kokoro decoder with LeakyReLU validates.
#[test]
fn test_kokoro_decoder_leaky_relu_def_validates() {
    let (def, _) = build_kokoro_decoder_with_leaky_relu();
    def.validate()
        .expect("kokoro decoder with LeakyReLU should validate");
}

/// Expanded decoder with LeakyReLU translates to NY GraphNetwork.
#[test]
fn test_kokoro_decoder_leaky_relu_graph_builds() {
    let (def, out_shape) = build_kokoro_decoder_with_leaky_relu();
    assert_eq!(out_shape, [OUT_CHANNELS, TIME_UP]);

    let bindings = kokoro_decoder_leaky_relu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("kokoro decoder with LeakyReLU graph should translate");

    // Base decoder has >= 8 nodes; LeakyReLU adds 2 more (one per activation).
    assert!(
        graph.num_nodes() >= 10,
        "kokoro decoder with LeakyReLU should have >= 10 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the expanded Kokoro decoder with LeakyReLU.
#[test]
fn test_kokoro_decoder_leaky_relu_ibp_propagates() {
    let (def, _) = build_kokoro_decoder_with_leaky_relu();
    let bindings = kokoro_decoder_leaky_relu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[8, TIME_IN], 1.0); // IN_CHANNELS=8

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Kokoro decoder with LeakyReLU");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CHANNELS, TIME_UP],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Kokoro decoder (LeakyReLU) IBP: bounds=[{lo_min}, {hi_max}]");

    // Exp output must be positive (exp(x) > 0 for all finite x).
    // LeakyReLU preserves sign structure better than ReLU (no dead neurons),
    // so IBP bounds should remain reasonable.
    assert!(
        lo_min > 0.0,
        "exp output should be positive, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e6,
        "IBP upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// CROWN propagation through the expanded Kokoro decoder with LeakyReLU.
///
/// Since NY 359a195+ (#2773), CROWN succeeds for this graph.
#[test]
fn test_kokoro_decoder_leaky_relu_crown_propagation() {
    let (def, _) = build_kokoro_decoder_with_leaky_relu();
    let bindings = kokoro_decoder_leaky_relu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let output = assert_crown_succeeds(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[OUT_CHANNELS, TIME_UP],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Kokoro decoder (LeakyReLU): method=Crown, bounds=[{lo_min}, {hi_max}]");

    // Magnitude assertions matching IBP counterpart (#1984 AC1):
    // exp output must be positive, and upper bound should be bounded.
    assert!(
        lo_min > 0.0,
        "CROWN: exp output should be positive, got lo_min={lo_min}"
    );
    assert!(
        hi_max < 1e6,
        "CROWN: upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// Expanded decoder verify and record under "kokoro_decoder_leaky_relu" key.
#[test]
fn test_kokoro_decoder_leaky_relu_verify_and_record() {
    let (def, _) = build_kokoro_decoder_with_leaky_relu();
    let bindings = kokoro_decoder_leaky_relu_bindings();
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_decoder_leaky_relu");
    assert_eq!(result.num_variables, 1, "single Variable input (features)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, TIME_UP]);

    // Soundness provenance must be set (#1984 AC2).
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ForwardMode NormBoundsMode should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
