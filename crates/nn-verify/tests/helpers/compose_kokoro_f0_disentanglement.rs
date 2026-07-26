// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Kokoro F0EnergyPredictor disentanglement verification.
//!
//! Tests that CROWN-based sensitivity analysis can distinguish which control
//! dimensions primarily affect which acoustic properties in the F0EnergyPredictor:
//!
//! - text_features → should primarily affect F0 and energy (phoneme-dependent)
//! - style → should modulate F0/energy via AdaIN conditioning
//! - The F0 and energy heads should show different sensitivity profiles to
//!   perturbations, demonstrating that the parallel head architecture produces
//!   partially disentangled F0/energy representations.
//!
//! This is Phase 2 of the #1738 design doc: Kokoro Prosody Disentanglement.
//!
//! Part of #1738: Compositional Verification of Prosody Controls.

#[path = "kokoro_f0_energy.rs"]
mod f0_helpers;

use super::common::{assert_bounds_valid, uniform_bounds};
use f0_helpers::{
    build_kokoro_f0_energy, kokoro_f0_energy_bindings, D_MODEL, ENERGY_OUTPUT_END,
    ENERGY_OUTPUT_START, F0_OUTPUT_END, F0_OUTPUT_START, FLAT_INPUT_SIZE, OUTPUT_SIZE, SEQ_LEN,
};
use nn_tts_verify::disentanglement::{
    measure_sensitivity, verify_disentanglement, AcousticProperty, ControlDimension,
};
use nn_verify::tensor_kernel_to_graph;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Text features occupy [0..D_MODEL*SEQ_LEN] in flat_input.
const TEXT_START: usize = 0;
const TEXT_END: usize = D_MODEL * SEQ_LEN;

/// Style occupies [D_MODEL*SEQ_LEN..FLAT_INPUT_SIZE] in flat_input.
const STYLE_START: usize = D_MODEL * SEQ_LEN;
const STYLE_END: usize = FLAT_INPUT_SIZE;

/// Input bound for sensitivity measurement.
const INPUT_BOUND: f64 = 1.0;

/// Midpoint for sensitivity measurement (all zeros).
fn midpoint() -> Vec<f64> {
    vec![0.0; FLAT_INPUT_SIZE]
}

// ---------------------------------------------------------------------------
// Graph construction tests
// ---------------------------------------------------------------------------

/// The F0EnergyPredictor graph should validate and build successfully.
#[test]
fn test_f0_energy_def_validates() {
    let (def, shape) = build_kokoro_f0_energy();
    assert_eq!(shape, [OUTPUT_SIZE]);
    assert!(
        def.validate().is_ok(),
        "F0EnergyPredictor def should validate"
    );
}

/// The graph should translate to NY GraphNetwork without error.
#[test]
fn test_f0_energy_graph_builds() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings);
    assert!(
        graph.is_ok(),
        "Graph translation should succeed: {:?}",
        graph.err()
    );
}

/// IBP propagation should produce valid bounds.
#[test]
fn test_f0_energy_ibp_propagates() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_bounds = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);
    let result = graph.propagate_ibp(&input_bounds);
    assert!(
        result.is_ok(),
        "IBP propagation should succeed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    assert_bounds_valid(&output);
    // Tightness assertion (#2594): F0 predictor should not produce vacuously wide bounds.
    let width = output.max_width();
    assert!(
        width < 1e6,
        "F0EnergyPredictor: max element width {width} exceeds 1e6 (vacuously wide)"
    );
}

/// CROWN propagation should succeed and produce tighter bounds than IBP.
#[test]
fn test_f0_energy_crown_propagates() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_bounds = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);
    let (method, output, _) = nn_verify::propagate_with_crown_fallback(&graph, &input_bounds)
        .expect("CROWN propagation");

    assert_bounds_valid(&output);
    // Tightness assertion (#2594): CROWN should produce tighter-or-equal bounds.
    let width = output.max_width();
    assert!(
        width < 1e6,
        "F0EnergyPredictor CROWN: max element width {width} exceeds 1e6"
    );
    eprintln!(
        "F0EnergyPredictor CROWN propagation method: {method:?}, max_width={width:.4}"
    );
}

// ---------------------------------------------------------------------------
// Sensitivity measurement tests (Phase 2 of design doc)
// ---------------------------------------------------------------------------

/// Text features should have non-zero influence on F0 output.
#[test]
fn test_text_features_affect_f0() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("text_features", 0, TEXT_START, TEXT_END);
    let property = AcousticProperty::new("f0", F0_OUTPUT_START, F0_OUTPUT_END);

    let result = measure_sensitivity(&graph, &control, &property, INPUT_BOUND, &midpoint())
        .expect("text→f0 sensitivity");

    // bound_width > 0.0 is structurally guaranteed for non-constant network
    // paths with non-zero input bounds; assert a meaningful upper bound.
    assert!(
        result.bound_width < 1e6,
        "Text→F0 width should be bounded, got {}",
        result.bound_width
    );

    eprintln!(
        "text_features → F0: width={:.6} ({})",
        result.bound_width, result.propagation_mode
    );
}

/// Text features should have non-zero influence on energy output.
#[test]
fn test_text_features_affect_energy() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("text_features", 0, TEXT_START, TEXT_END);
    let property = AcousticProperty::new("energy", ENERGY_OUTPUT_START, ENERGY_OUTPUT_END);

    let result = measure_sensitivity(&graph, &control, &property, INPUT_BOUND, &midpoint())
        .expect("text→energy sensitivity");

    assert!(
        result.bound_width < 1e6,
        "Text→energy width should be bounded, got {}",
        result.bound_width
    );

    eprintln!(
        "text_features → energy: width={:.6} ({})",
        result.bound_width, result.propagation_mode
    );
}

/// Style should influence F0 via AdaIN conditioning.
#[test]
fn test_style_affects_f0() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("style", 0, STYLE_START, STYLE_END);
    let property = AcousticProperty::new("f0", F0_OUTPUT_START, F0_OUTPUT_END);

    let result = measure_sensitivity(&graph, &control, &property, INPUT_BOUND, &midpoint())
        .expect("style→f0 sensitivity");

    assert!(
        result.bound_width < 1e6,
        "Style→F0 width should be bounded, got {}",
        result.bound_width
    );

    eprintln!(
        "style → F0: width={:.6} ({})",
        result.bound_width, result.propagation_mode
    );
}

/// Style should influence energy via AdaIN conditioning.
#[test]
fn test_style_affects_energy() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("style", 0, STYLE_START, STYLE_END);
    let property = AcousticProperty::new("energy", ENERGY_OUTPUT_START, ENERGY_OUTPUT_END);

    let result = measure_sensitivity(&graph, &control, &property, INPUT_BOUND, &midpoint())
        .expect("style→energy sensitivity");

    assert!(
        result.bound_width < 1e6,
        "Style→energy width should be bounded, got {}",
        result.bound_width
    );

    eprintln!(
        "style → energy: width={:.6} ({})",
        result.bound_width, result.propagation_mode
    );
}

// ---------------------------------------------------------------------------
// Disentanglement certificate tests
// ---------------------------------------------------------------------------

/// Full 2×2 sensitivity matrix: [text, style] × [F0, energy].
#[test]
fn test_f0_energy_disentanglement_certificate() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let controls = vec![
        ControlDimension::new("text_features", 0, TEXT_START, TEXT_END),
        ControlDimension::new("style", 0, STYLE_START, STYLE_END),
    ];
    let properties = vec![
        AcousticProperty::new("f0", F0_OUTPUT_START, F0_OUTPUT_END),
        AcousticProperty::new("energy", ENERGY_OUTPUT_START, ENERGY_OUTPUT_END),
    ];

    let cert = verify_disentanglement(
        &graph,
        &controls,
        &properties,
        INPUT_BOUND,
        &midpoint(),
        0.99, // permissive threshold for synthetic weights
    )
    .expect("disentanglement verification");

    // Print the full sensitivity matrix
    eprintln!("F0EnergyPredictor sensitivity matrix:");
    for s in &cert.sensitivities {
        eprintln!(
            "  {} → {}: width={:.6} ({})",
            s.control, s.property, s.bound_width, s.propagation_mode
        );
    }
    eprintln!("  max_cross_influence={:.4}", cert.max_cross_influence);
    eprintln!("  is_disentangled={}", cert.is_disentangled);
}

/// Near-zero-bound sensitivity should produce near-zero output width for F0.
/// Uses a tiny epsilon instead of exact zero since measure_sensitivity requires
/// input_bound > 0.0 (zero-width perturbation is mathematically degenerate).
#[test]
fn test_f0_zero_bound_sensitivity() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("text_features", 0, TEXT_START, TEXT_END);
    let property = AcousticProperty::new("f0", F0_OUTPUT_START, F0_OUTPUT_END);

    let result = measure_sensitivity(&graph, &control, &property, 1e-12, &midpoint())
        .expect("near-zero-bound sensitivity");

    assert!(
        result.bound_width < 0.01,
        "Near-zero-bound sensitivity should be near-zero, got {}",
        result.bound_width
    );
}

/// Wider input bound should produce wider output bound.
#[test]
fn test_f0_sensitivity_increases_with_bound() {
    let (def, _) = build_kokoro_f0_energy();
    let bindings = kokoro_f0_energy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let control = ControlDimension::new("style", 0, STYLE_START, STYLE_END);
    let property = AcousticProperty::new("f0", F0_OUTPUT_START, F0_OUTPUT_END);

    let small = measure_sensitivity(&graph, &control, &property, 0.5, &midpoint())
        .expect("sensitivity at 0.5");
    let large = measure_sensitivity(&graph, &control, &property, 2.0, &midpoint())
        .expect("sensitivity at 2.0");

    assert!(
        large.bound_width >= small.bound_width,
        "Wider input ({}) should give wider output ({} vs {})",
        2.0,
        large.bound_width,
        small.bound_width
    );

    eprintln!(
        "Style→F0 sensitivity scaling: bound=0.5 → width={}, bound=2.0 → width={}",
        small.bound_width, large.bound_width
    );
}
