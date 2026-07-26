// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DiffSinger singing pitch control bound verification.
//!
//! Tests that NY can propagate finite, score-indexed pitch-control
//! bounds through a native singing pitch prediction graph built with
//! `TensorBlockBuilder`.
//!
//! Consolidated: builds the graph ONCE and runs all property checks on it,
//! eliminating 8 redundant graph builds (was 9 builds, now 1).
//!
//! Part of #3516: CROWN for singing voice pitch/vibrato/formant proofs.

#[path = "singing_pitch_control.rs"]
mod singing_helpers;

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_verify::tensor_kernel_to_graph;
use singing_helpers::{
    build_singing_pitch_control, singing_pitch_control_bindings, NOTE_FEATURES, NUM_NOTES,
    OUTPUT_SIZE, SCORE_INPUT_SIZE,
};

// ---------------------------------------------------------------------------
// Consolidated test: graph construction + binding validation
// ---------------------------------------------------------------------------

/// Def validates, graph translates, and binding count is correct.
#[test]
fn test_singing_pitch_control_construction() {
    let (def, shape) = build_singing_pitch_control();
    assert_eq!(shape, [OUTPUT_SIZE]);
    assert!(
        def.validate().is_ok(),
        "Singing pitch control def should validate: {:?}",
        def.validate().err()
    );

    let bindings = singing_pitch_control_bindings();
    // 1 variable + 3 layers × 2 params (w + b) = 7 total
    assert_eq!(
        bindings.len(),
        7,
        "Expected 7 bindings: 1 variable + 3×(weight + bias)"
    );

    let graph = tensor_kernel_to_graph(&def, &bindings);
    assert!(
        graph.is_ok(),
        "Graph translation should succeed: {:?}",
        graph.err()
    );
}

// ---------------------------------------------------------------------------
// Consolidated test: all IBP properties (finite bounds, monotonicity,
// near-zero, score-indexed shape)
// ---------------------------------------------------------------------------

/// IBP: finite bounds, bound monotonicity, near-zero input, score-indexed shape.
/// Builds graph once, runs 4 propagations (was 5 separate graph builds).
#[test]
fn test_singing_pitch_ibp_all_properties() {
    let (def, shape) = build_singing_pitch_control();
    assert_eq!(shape[0], NUM_NOTES);
    let bindings = singing_pitch_control_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // --- IBP finite bounds ---
    let input = uniform_bounds(&[SCORE_INPUT_SIZE], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(
        lo.len(),
        NUM_NOTES,
        "Output should have one pitch bound per note"
    );
    assert_eq!(hi.len(), NUM_NOTES);
    for i in 0..NUM_NOTES {
        eprintln!("  Note {i}: pitch bounds [{:.6}, {:.6}]", lo[i], hi[i]);
    }

    // --- IBP bound monotonicity ---
    let narrow_input = uniform_bounds(&[SCORE_INPUT_SIZE], 0.1);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    assert_bounds_valid(&narrow_output);

    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();
    let (wide_lo, wide_hi) = output.lower_upper(); // reuse wide output
    let eps = 1e-5;
    for i in 0..NUM_NOTES {
        let narrow_width = narrow_hi[i] - narrow_lo[i];
        let wide_width = wide_hi[i] - wide_lo[i];
        assert!(
            wide_width >= narrow_width - eps,
            "Note {i}: wide bounds width ({wide_width:.6}) should be >= \
             narrow bounds width ({narrow_width:.6})"
        );
    }

    // --- IBP near-zero input ---
    let tiny_input = uniform_bounds(&[SCORE_INPUT_SIZE], 1e-10);
    let tiny_output = graph.propagate_ibp(&tiny_input).expect("IBP near-zero");
    assert_bounds_valid(&tiny_output);

    let (tiny_lo, tiny_hi) = tiny_output.lower_upper();
    for i in 0..NUM_NOTES {
        let width = tiny_hi[i] - tiny_lo[i];
        assert!(
            width < 0.01,
            "Note {i}: near-zero input should give near-zero width, got {width:.6}"
        );
    }

    // --- Score-indexed shape ---
    eprintln!("Score-indexed pitch bounds ({NUM_NOTES} notes):");
    for i in 0..NUM_NOTES {
        let (lo, hi) = output.lower_upper();
        eprintln!(
            "  Note {i}: [{:.6}, {:.6}] (width={:.6})",
            lo[i],
            hi[i],
            hi[i] - lo[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Consolidated test: CROWN propagation + tighter-than-IBP + per-note sensitivity
// ---------------------------------------------------------------------------

/// CROWN: propagation, tighter-than-IBP, and per-note sensitivity.
/// Builds graph once, runs CROWN + IBP + per-note checks (was 3 separate graph builds).
/// Check CROWN propagation and tighter-than-IBP for singing pitch.
fn check_singing_crown_tighter(graph: &nn_verify::GraphNetwork) {
    let input = uniform_bounds(&[SCORE_INPUT_SIZE], 1.0);

    let (method, crown_output, fallback_reason) =
        nn_verify::propagate_with_crown_fallback(graph, &input).expect("CROWN propagation");
    assert_bounds_valid(&crown_output);

    let (crown_lo, crown_hi) = crown_output.lower_upper();
    eprintln!("Singing pitch CROWN method: {method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("  Fallback: {reason}");
    }
    for i in 0..NUM_NOTES {
        eprintln!("  Note {i}: CROWN [{:.6}, {:.6}]", crown_lo[i], crown_hi[i]);
    }

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    if matches!(method, nn_verify::PropMethod::Crown) {
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
        let eps = 1e-4;
        for i in 0..NUM_NOTES {
            assert!(
                crown_lo[i] >= ibp_lo[i] - eps,
                "Note {i}: CROWN lo >= IBP lo"
            );
            assert!(
                crown_hi[i] <= ibp_hi[i] + eps,
                "Note {i}: CROWN hi <= IBP hi"
            );
        }
        eprintln!("CROWN tighter than IBP (linear+ReLU MLP)");
    } else {
        eprintln!("WARNING: CROWN fell back to IBP");
    }
}

/// Check per-note input sensitivity via IBP.
fn check_singing_per_note_sensitivity(graph: &nn_verify::GraphNetwork) {
    let baseline = uniform_bounds(&[SCORE_INPUT_SIZE], 1e-10);
    let baseline_output = graph.propagate_ibp(&baseline).expect("IBP baseline");
    let (base_lo, base_hi) = baseline_output.lower_upper();

    let mut perturbed_lo =
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[SCORE_INPUT_SIZE]), -1e-10_f32);
    let mut perturbed_hi =
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[SCORE_INPUT_SIZE]), 1e-10_f32);
    for j in 0..NOTE_FEATURES {
        perturbed_lo[j] = -1.0;
        perturbed_hi[j] = 1.0;
    }
    let perturbed_input =
        nn_verify::BoundedTensor::new(perturbed_lo, perturbed_hi).expect("bounds");
    let perturbed_output = graph
        .propagate_ibp(&perturbed_input)
        .expect("IBP perturbed");
    let (pert_lo, pert_hi) = perturbed_output.lower_upper();

    let note0_base_width = base_hi[0] - base_lo[0];
    let note0_pert_width = pert_hi[0] - pert_lo[0];
    assert!(
        note0_pert_width > note0_base_width + 1e-6,
        "Perturbing note 0 should widen bounds: base={note0_base_width:.6}, pert={note0_pert_width:.6}"
    );

    eprintln!("Per-note sensitivity:");
    for i in 0..NUM_NOTES {
        let bw = base_hi[i] - base_lo[i];
        let pw = pert_hi[i] - pert_lo[i];
        eprintln!(
            "  Note {i}: base={bw:.6}, pert={pw:.6}, delta={:.6}",
            pw - bw
        );
    }
}

#[test]
fn test_singing_pitch_crown_and_sensitivity() {
    let (def, _) = build_singing_pitch_control();
    let bindings = singing_pitch_control_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    check_singing_crown_tighter(&graph);
    check_singing_per_note_sensitivity(&graph);
}
