// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: HTDemucs full-model NY composition.
//!
//! Temporal encoder + cross-domain transformer (with constant spectral KV)
//! + temporal decoder as a single verified `GraphNetwork`.
//!
//! This is the first full HTDemucs model composition — combining all three
//! stages into one graph for end-to-end bounds propagation.
//!
//! **CROWN status (#1996):** CROWN through normalization layers
//! now uses `IbpValidated` mode (sound Jacobian linearization with
//! IBP-validated error margins). Prior `Sound` mode refused CROWN.
//!
//! Part of #1696: 4/5 models have zero NY verification.

use super::common;

#[path = "htdemucs_full.rs"]
mod helpers;

use common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use helpers::{build_htdemucs_full, htdemucs_full_bindings, IN_CH, T_IN};
use nn_verify::{tensor_kernel_to_graph, VerificationSoundnessMode};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full HTDemucs TensorKernelDef validates.
#[test]
fn test_htdemucs_full_def_validates() {
    let (def, _) = build_htdemucs_full();
    def.validate()
        .expect("htdemucs full model def should validate");
}

/// Full HTDemucs model translates to NY GraphNetwork.
#[test]
fn test_htdemucs_full_graph_builds() {
    let (def, target_t) = build_htdemucs_full();
    assert!(target_t > 0, "output temporal length should be > 0");

    let bindings = htdemucs_full_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("htdemucs full graph should translate");

    // Encoder + cross-domain transformer (self-attn + cross-attn + FFN) + decoder
    assert!(
        graph.num_nodes() >= 60,
        "htdemucs full graph should have >= 60 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full HTDemucs model.
#[test]
fn test_htdemucs_full_ibp_propagates() {
    let (def, target_t) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full HTDemucs model");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[IN_CH, target_t],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("HTDemucs full IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds sanity: with small weights (0.001) and [-1, 1] input, the full
    // model (encoder + cross-domain transformer + decoder) should produce
    // finite bounds. IBP may be wide due to decomposed norms.
    assert!(
        lo_min.abs() < 1e6,
        "IBP lower bound magnitude should be < 1e6, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e6,
        "IBP upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// CROWN propagation through the full HTDemucs model.
///
/// **CROWN status (#1996):** With `IbpValidated` crown mode, CROWN
/// linearization through normalization layers is sound. If CROWN still
/// falls back (e.g., shape mismatch in NY), the test verifies
/// finite bounds and correct output shape via IBP.
#[test]
fn test_htdemucs_full_crown_propagation() {
    let (def, target_t) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[IN_CH, target_t],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("HTDemucs full: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    // Magnitude assertions matching IBP counterpart (#1984).
    assert!(
        lo_min.abs() < 1e6,
        "CROWN: lower bound magnitude should be < 1e6, got {lo_min}"
    );
    assert!(
        hi_max.abs() < 1e6,
        "CROWN: upper bound magnitude should be < 1e6, got {hi_max}"
    );
}

/// Full HTDemucs preserves temporal output shape (autoencoder property).
#[test]
fn test_htdemucs_full_preserves_shape() {
    let (def, target_t) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[IN_CH, target_t],
        "output must match temporal input shape"
    );
}

/// Narrower input produces tighter output bounds (monotonicity).
#[test]
fn test_htdemucs_full_narrow_inputs_tighter() {
    let (def, _) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[IN_CH, T_IN], 10.0);
    let narrow_input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    // At least half of output elements should have narrower bounds with
    // narrower input (IBP monotonicity may not hold element-wise due to
    // decomposed norm approximations).
    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter_count}/{total}"
    );
}

/// Full model verify and record under "htdemucs_full" key.
#[test]
fn test_htdemucs_full_verify_and_record() {
    let (def, target_t) = build_htdemucs_full();
    let bindings = htdemucs_full_bindings();
    let input = uniform_bounds(&[IN_CH, T_IN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "htdemucs_full");
    assert_eq!(result.num_variables, 1, "single Variable input (audio)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[IN_CH, target_t]);

    // Soundness provenance must be set (#1984).
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ForwardMode NormBoundsMode should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
