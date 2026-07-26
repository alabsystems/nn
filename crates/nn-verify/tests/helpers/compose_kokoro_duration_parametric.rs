// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parametric Kokoro duration positivity tests — Groups A+B.
//!
//! - **Group A (Phase 1A/1B):** Fixed D=8 ProsodyPredictor — single-block and three-block
//! - **Group B (Phase 1C):** T=4 LSTM unrolling at D=8
//!
//! Groups C+D+E (scaled dimensions and sensitivity analysis) are in
//! `compose_kokoro_duration_scaled_parametric.rs`.
//!
//! Design: `designs/archive/2026-03-11-monotonicity-test-parametrization.md`
//! Part of #1981.

// Test helper functions may appear unused when not all groups are active.
#![allow(dead_code)]

#[path = "kokoro_prosody.rs"]
mod kokoro_prosody;

#[path = "kokoro_prosody_t4.rs"]
mod kokoro_prosody_t4;

#[path = "kokoro_scaled_pipeline.rs"]
mod duration_scaled_helpers;
// Alias needed: kokoro_prosody_scaled.rs references `super::helpers::KokoroDims`.
use duration_scaled_helpers as helpers;

#[path = "kokoro_prosody_scaled.rs"]
mod prosody_scaled;

#[path = "kokoro_duration_helpers.rs"]
mod duration_helpers;

use super::common::monotonicity::{
    run_experiment_batch, run_monotonicity_experiment, AssertionPattern, MonotonicityConfig,
    PropagationMethod,
};
use super::common::{assert_bounds_valid, uniform_bounds};
use nn_verify::tensor_kernel_to_graph;

use duration_helpers::{crown_vs_ibp_simple, lo_min_of, method_str, DURATION_THRESHOLD};
use kokoro_prosody::{
    build_kokoro_prosody_single_block, build_kokoro_prosody_three_blocks, kokoro_prosody_bindings,
    kokoro_prosody_three_block_bindings, FLAT_INPUT_SIZE, N_BLOCKS, SEQ_LEN,
};
use kokoro_prosody_t4::{
    build_kokoro_prosody_t4, kokoro_prosody_t4_bindings, FLAT_INPUT_SIZE_T4, SEQ_LEN_T4,
};
use nn_tts_verify::monotonicity::interpret_duration_positivity;

// ===========================================================================
// Group A: Fixed D=8 ProsodyPredictor (Phases 1A/1B)
//
// Single-block and three-block Kokoro ProsodyPredictor at D=8, SEQ_LEN=1.
// Validates graph construction, IBP/CROWN propagation, duration positivity.
// ===========================================================================

#[test]
fn test_group_a_single_block_batch() {
    let (def, _) = build_kokoro_prosody_single_block();
    let bindings = kokoro_prosody_bindings();

    let configs = vec![
        (
            MonotonicityConfig {
                label: "phase1a_1b_def_validates",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE],
                prop_method: PropagationMethod::Ibp,
                assertion: AssertionPattern::BoundsValid,
            },
            def.clone(),
            bindings.clone(),
        ),
        (
            MonotonicityConfig {
                label: "phase1a_ibp_propagates",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE],
                prop_method: PropagationMethod::Ibp,
                assertion: AssertionPattern::BoundsValid,
            },
            def.clone(),
            bindings.clone(),
        ),
        (
            MonotonicityConfig {
                label: "phase1a_crown_propagates",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::DurationPositivity {
                    threshold: DURATION_THRESHOLD,
                    seq_len: SEQ_LEN,
                    expect_proven: true,
                    lower_bound_floor: Some(-1.0),
                },
            },
            def,
            bindings,
        ),
    ];

    run_experiment_batch("Group A: Phase 1A/1B single-block", &configs);
}

#[test]
fn test_group_a_single_block_graph_structure() {
    // Validate graph has expected node count.
    let (def, _) = build_kokoro_prosody_single_block();
    let bindings = kokoro_prosody_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 20,
        "single-block graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_group_a_single_block_verify_and_record() {
    let (def, _) = build_kokoro_prosody_single_block();
    let bindings = kokoro_prosody_bindings();
    let config = MonotonicityConfig {
        label: "phase1a_verify_record",
        input_bound: 1.0,
        input_shape: vec![FLAT_INPUT_SIZE],
        prop_method: PropagationMethod::CrownFallback,
        assertion: AssertionPattern::BoundsValidAndRecord {
            status_key: "kokoro_prosody_single_block",
        },
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    assert!(result.bounds_valid, "single-block bounds should be valid");
}

#[test]
fn test_group_a_single_block_crown_vs_ibp() {
    let (def, _) = build_kokoro_prosody_single_block();
    let bindings = kokoro_prosody_bindings();
    crown_vs_ibp_simple(&def, &bindings, FLAT_INPUT_SIZE, "Group A crown_vs_ibp");
}

#[test]
fn test_group_a_three_block_batch() {
    let (def, _) = build_kokoro_prosody_three_blocks();
    let bindings = kokoro_prosody_three_block_bindings();

    let configs = vec![
        (
            MonotonicityConfig {
                label: "phase1b_3block_ibp",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE],
                prop_method: PropagationMethod::Ibp,
                assertion: AssertionPattern::BoundsValid,
            },
            def.clone(),
            bindings.clone(),
        ),
        (
            MonotonicityConfig {
                label: "phase1b_3block_crown",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::DurationPositivity {
                    threshold: DURATION_THRESHOLD,
                    seq_len: SEQ_LEN,
                    expect_proven: true,
                    lower_bound_floor: Some(-1.0),
                },
            },
            def,
            bindings,
        ),
    ];

    run_experiment_batch("Group A: Phase 1B three-block", &configs);
}

#[test]
fn test_group_a_three_block_graph_structure() {
    let (def, _) = build_kokoro_prosody_three_blocks();
    let bindings = kokoro_prosody_three_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 60,
        "three-block graph should have >= 60 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_group_a_three_block_verify_and_record() {
    let (def, _) = build_kokoro_prosody_three_blocks();
    let bindings = kokoro_prosody_three_block_bindings();
    let config = MonotonicityConfig {
        label: "phase1b_3block_verify_record",
        input_bound: 1.0,
        input_shape: vec![FLAT_INPUT_SIZE],
        prop_method: PropagationMethod::CrownFallback,
        assertion: AssertionPattern::BoundsValidAndRecord {
            status_key: "kokoro_prosody_three_blocks",
        },
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    assert!(result.bounds_valid, "three-block bounds should be valid");
}

#[test]
fn test_group_a_three_block_crown_vs_ibp() {
    // Explicit CROWN vs IBP comparison with width analysis.
    let (def, _) = build_kokoro_prosody_three_blocks();
    let bindings = kokoro_prosody_three_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    let ibp_lo_min = lo_min_of(&ibp_output) as f32;
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let ibp_width: f32 = ibp_lo
        .iter()
        .zip(ibp_hi.iter())
        .map(|(&l, &u)| u - l)
        .sum::<f32>()
        / ibp_lo.len() as f32;

    let (method, crown_output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");
    let crown_lo_min = lo_min_of(&crown_output) as f32;
    let (crown_lo, crown_hi) = crown_output.lower_upper();
    let crown_width: f32 = crown_lo
        .iter()
        .zip(crown_hi.iter())
        .map(|(&l, &u)| u - l)
        .sum::<f32>()
        / crown_lo.len() as f32;

    assert_bounds_valid(&crown_output);
    if matches!(method, nn_verify::PropMethod::Crown) {
        assert!(
            crown_lo_min >= ibp_lo_min - 1e-4,
            "CROWN lower {crown_lo_min:.6} should be >= IBP lower {ibp_lo_min:.6}"
        );
    }
    // Tightness assertion (#2594): average per-element width should be bounded.
    assert!(
        ibp_width < 1e6,
        "Duration 3-block IBP: avg width {ibp_width} exceeds 1e6 (vacuously wide)"
    );
    eprintln!(
        "Group A 3block crown_vs_ibp: IBP lo_min={ibp_lo_min:.6} width={ibp_width:.4}, \
         CROWN lo_min={crown_lo_min:.6} width={crown_width:.4}, method={method:?}"
    );
}

#[test]
fn test_group_a_three_block_positivity_with_iclr_comparison() {
    // Three-block positivity certificate with ICLR Table 1 cross-comparison.
    let (def_3b, _) = build_kokoro_prosody_three_blocks();
    let bindings_3b = kokoro_prosody_three_block_bindings();
    let graph_3b = tensor_kernel_to_graph(&def_3b, &bindings_3b).expect("graph translation");
    let input = uniform_bounds(&[FLAT_INPUT_SIZE], 1.0);

    let (method_3b, output_3b, _) =
        nn_verify::propagate_with_crown_fallback(&graph_3b, &input).expect("CROWN propagation");
    let lo_min_3b = lo_min_of(&output_3b);
    let ms_3b = method_str(method_3b);
    let cert_3b =
        interpret_duration_positivity(lo_min_3b, DURATION_THRESHOLD, 1.0, 1.0, SEQ_LEN, ms_3b);
    assert!(cert_3b.is_proven, "3-block should be proven: {cert_3b:?}");
    assert!(
        cert_3b.lower_bound > -1.0,
        "3-block lower_bound {} should be > -1.0",
        cert_3b.lower_bound
    );

    // ICLR Table 1 cross-comparison: single-block for reference.
    let (def_1b, _) = build_kokoro_prosody_single_block();
    let bindings_1b = kokoro_prosody_bindings();
    let graph_1b = tensor_kernel_to_graph(&def_1b, &bindings_1b).expect("graph translation");
    let (method_1b, output_1b, _) =
        nn_verify::propagate_with_crown_fallback(&graph_1b, &input).expect("CROWN propagation");
    let lo_min_1b = lo_min_of(&output_1b);
    let ms_1b = method_str(method_1b);
    let cert_1b =
        interpret_duration_positivity(lo_min_1b, DURATION_THRESHOLD, 1.0, 1.0, SEQ_LEN, ms_1b);

    eprintln!(
        "ICLR Table 1: 1-block lower={:.6} ({}), {N_BLOCKS}-block lower={:.6} ({})",
        cert_1b.lower_bound, ms_1b, cert_3b.lower_bound, ms_3b
    );
}

// ===========================================================================
// Group B: T=4 LSTM unrolling (Phase 1C)
//
// Three-block ProsodyPredictor with T=4 LSTM sequence length at D=8.
// ===========================================================================

#[test]
fn test_group_b_t4_batch() {
    let (def, _) = build_kokoro_prosody_t4();
    let bindings = kokoro_prosody_t4_bindings();

    let configs = vec![
        (
            MonotonicityConfig {
                label: "phase1c_t4_ibp",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE_T4],
                prop_method: PropagationMethod::Ibp,
                assertion: AssertionPattern::BoundsValid,
            },
            def.clone(),
            bindings.clone(),
        ),
        (
            MonotonicityConfig {
                label: "phase1c_t4_crown",
                input_bound: 1.0,
                input_shape: vec![FLAT_INPUT_SIZE_T4],
                prop_method: PropagationMethod::CrownFallback,
                assertion: AssertionPattern::DurationPositivity {
                    threshold: DURATION_THRESHOLD,
                    seq_len: SEQ_LEN_T4,
                    expect_proven: true,
                    lower_bound_floor: Some(-1.0),
                },
            },
            def,
            bindings,
        ),
    ];

    run_experiment_batch("Group B: Phase 1C T=4", &configs);
}

#[test]
fn test_group_b_t4_graph_structure() {
    let (def, _) = build_kokoro_prosody_t4();
    let bindings = kokoro_prosody_t4_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 200,
        "T=4 graph should have >= 200 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_group_b_t4_verify_and_record() {
    let (def, _) = build_kokoro_prosody_t4();
    let bindings = kokoro_prosody_t4_bindings();
    let config = MonotonicityConfig {
        label: "phase1c_t4_verify_record",
        input_bound: 1.0,
        input_shape: vec![FLAT_INPUT_SIZE_T4],
        prop_method: PropagationMethod::CrownFallback,
        assertion: AssertionPattern::BoundsValidAndRecord {
            status_key: "kokoro_prosody_t4",
        },
    };
    let result = run_monotonicity_experiment(&config, &def, &bindings);
    assert!(result.bounds_valid, "T=4 bounds should be valid");
}

#[test]
fn test_group_b_t4_crown_vs_ibp() {
    let (def, _) = build_kokoro_prosody_t4();
    let bindings = kokoro_prosody_t4_bindings();
    crown_vs_ibp_simple(
        &def,
        &bindings,
        FLAT_INPUT_SIZE_T4,
        "Group B T=4 crown_vs_ibp",
    );
}
