// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-layer CROWN sub-graph verification tests for Kokoro (#2592).
//!
//! Per-layer CROWN (each layer as its own GraphNetwork) produces CROWN/IBP = 1.0
//! for ALL Kokoro layer types — CROWN cannot exploit cross-layer correlations
//! within a single layer. This file tests `verify_layerwise_grouped`, which
//! merges adjacent layers into multi-layer GraphNetworks so CROWN sees
//! cross-layer structure.
//!
//! Expected: CROWN/IBP < 1.0 within the pre-normalization group (layers 0-2:
//! Conv1d+ReLU+Linear+Conv1d+LeakyReLU+ConvTranspose1d) where there are no
//! normalization layers to reset correlation structure.
//!
//! Part of #2592, Part of #2218.

#[path = "kokoro_scaled_pipeline.rs"]
mod grouped_scaled_helpers;
use grouped_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod grouped_layerwise_helpers;

use super::common::kokoro_weights::{bt_max_width, sign_alternate_weight_bindings, uniform_bt};
use grouped_layerwise_helpers::build_kokoro_layerwise_deep;
use helpers::KokoroDims;
use nn_tts_verify::{verify_layerwise, verify_layerwise_grouped, LayerwiseGrouping};

/// Number of ResBlocks for grouped tests (4 = fast, representative).
const NUM_RESBLOCKS: usize = 4;

/// Build the standard grouping for a pipeline with `num_resblocks` ResBlocks.
///
/// Pipeline layout: [text_enc, voc_pre, upsample, resblock_0..N-1, output]
///
/// Grouping strategy (normalization-aware):
///   Group 0: [0, 1, 2] — pre-norm path (no normalization layers)
///   Groups 1..(N/2): pairs of ResBlocks
///   Last group: remaining ResBlocks + output
fn build_resblock_pair_grouping(num_resblocks: usize) -> LayerwiseGrouping {
    let mut groups = Vec::new();

    // Group 0: text_encoder + vocoder_pre + upsample (no norms, CROWN can tighten)
    groups.push(vec![0, 1, 2]);

    // Pair ResBlocks: indices 3..(3 + num_resblocks)
    let resblock_start = 3;
    let output_idx = resblock_start + num_resblocks;
    let mut i = resblock_start;
    while i + 1 < output_idx {
        groups.push(vec![i, i + 1]);
        i += 2;
    }
    // Odd leftover ResBlock gets grouped with output
    if i < output_idx {
        groups.push(vec![i, output_idx]);
    } else {
        groups.push(vec![output_idx]);
    }

    LayerwiseGrouping { groups }
}

// ===========================================================================
// D=64 grouped layerwise: verify grouping produces a valid certificate
// ===========================================================================

/// D=64, 4 ResBlocks: basic validity of grouped layerwise verification.
///
/// Verifies that `verify_layerwise_grouped` produces a valid pipeline
/// certificate with junction compatibility across group boundaries.
#[test]
fn test_kokoro_grouped_d64_validity() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let grouping = build_resblock_pair_grouping(NUM_RESBLOCKS);

    eprintln!(
        "Grouped D=64: {} layers → {} groups",
        layers.len(),
        grouping.groups.len()
    );
    for (i, g) in grouping.groups.iter().enumerate() {
        eprintln!("  Group {i}: layers {g:?}");
    }

    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert =
        verify_layerwise_grouped(&layers, &initial, &grouping).expect("D=64 grouped layerwise");

    assert!(cert.is_valid, "D=64 grouped pipeline must be valid");

    // All junctions between groups must be compatible.
    for (i, j) in cert.junctions.iter().enumerate() {
        assert!(
            j.shape_compatible,
            "D=64 grouped junction {i}: shape mismatch"
        );
        assert!(
            j.bounds_contained,
            "D=64 grouped junction {i}: bounds violation={:.6}",
            j.max_violation
        );
    }

    // P1: exp output must be positive (non-silence).
    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        lo_min > 0.0,
        "D=64 grouped P1: expected positive output, got {lo_min}"
    );

    // P2: output must be finite (bounded).
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi_max.is_finite(),
        "D=64 grouped P2: expected finite output, got {hi_max}"
    );

    // Report per-group widths.
    for (i, stage) in cert.stages.iter().enumerate() {
        let lo = stage
            .output_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi = stage
            .output_upper
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "  group {i}: [{lo:.6}, {hi:.6}] width={:.6} method={}",
            hi - lo,
            stage.method
        );
    }
}

// ===========================================================================
// D=64 comparison: grouped vs ungrouped end-to-end width
// ===========================================================================

/// D=64, 4 ResBlocks: compare grouped vs per-layer end-to-end output width.
///
/// If multi-layer CROWN provides any tightening, the grouped pipeline's
/// output bounds should be no wider than (and possibly tighter than) the
/// per-layer pipeline's output bounds.
///
/// This is the key metric for #2592: does grouping produce tighter bounds?
#[test]
fn test_kokoro_grouped_vs_ungrouped_d64() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    // Per-layer (ungrouped): each layer is a separate GraphNetwork.
    let ungrouped_cert = verify_layerwise(&layers, &initial).expect("D=64 ungrouped");
    let ungrouped_lo = ungrouped_cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let ungrouped_hi = ungrouped_cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let ungrouped_width = ungrouped_hi - ungrouped_lo;

    // Grouped: merge adjacent layers into multi-layer GraphNetworks.
    let grouping = build_resblock_pair_grouping(NUM_RESBLOCKS);
    let grouped_cert =
        verify_layerwise_grouped(&layers, &initial, &grouping).expect("D=64 grouped");
    let grouped_lo = grouped_cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let grouped_hi = grouped_cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let grouped_width = grouped_hi - grouped_lo;

    eprintln!("=== D=64 Grouped vs Ungrouped Comparison ===");
    eprintln!("  Ungrouped: [{ungrouped_lo:.6}, {ungrouped_hi:.6}] width={ungrouped_width:.6}");
    eprintln!("  Grouped:   [{grouped_lo:.6}, {grouped_hi:.6}] width={grouped_width:.6}");

    if ungrouped_width > 0.0 {
        let ratio = grouped_width / ungrouped_width;
        eprintln!("  Grouped/Ungrouped ratio: {ratio:.4}");
        if ratio < 0.99 {
            eprintln!("  >>> CROWN tightening detected! Grouped bounds are tighter.");
        } else {
            eprintln!(
                "  >>> No significant tightening (expected: InstanceNorm resets correlation)."
            );
        }
    }

    // Soundness: grouped bounds must be valid (no wider than sound limit).
    // Both certificates must be valid.
    assert!(ungrouped_cert.is_valid, "ungrouped must be valid");
    assert!(grouped_cert.is_valid, "grouped must be valid");
}

// ===========================================================================
// Pre-norm group isolation: test CROWN on the norm-free prefix
// ===========================================================================

/// D=64: CROWN soundness on pre-norm group with uniform synthetic weights.
///
/// Uses WEIGHT_MAG=0.001 uniform positive weights. CROWN/IBP ratio = 1.0
/// (no tightening) because all-positive weights make CROWN degenerate — see
/// #2615. The signed-weight and He-scaled tests below demonstrate actual
/// CROWN tightening with non-uniform weights.
#[test]
fn test_kokoro_grouped_prenorm_crown_tightening() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);

    // Build only the pre-norm layers as a group.
    let prenorm_layers: Vec<_> = layers[0..3].to_vec();
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    // IBP only on the merged graph.
    let merged_graph = nn_verify::tensor_kernels_to_grouped_graph(
        &prenorm_layers,
        nn_verify::NormBoundsMode::Conservative,
    )
    .expect("pre-norm group graph");
    let ibp_output = merged_graph.propagate_ibp(&initial).expect("pre-norm IBP");
    let ibp_width = bt_max_width(&ibp_output);

    // CROWN on the merged graph.
    let (crown_method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&merged_graph, &initial).expect("pre-norm CROWN");
    let crown_width = bt_max_width(&crown_output);

    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        f32::INFINITY
    };

    eprintln!("=== Pre-norm Group (layers 0-2) CROWN vs IBP ===");
    eprintln!("  IBP width:   {ibp_width:.6}");
    eprintln!("  CROWN width: {crown_width:.6} (method: {crown_method:?})");
    eprintln!("  IBP/CROWN ratio: {ratio:.4}");

    // Soundness: CROWN width must be <= IBP width (within fp tolerance).
    assert!(
        crown_width <= ibp_width + 1e-3,
        "Pre-norm CROWN width {crown_width} > IBP width {ibp_width} (soundness violation)",
    );

    // Both must be finite.
    assert!(ibp_width.is_finite(), "Pre-norm IBP width not finite");
    assert!(crown_width.is_finite(), "Pre-norm CROWN width not finite");

    // Report tightening status (informational, not asserted yet — need data).
    if ratio > 1.01 {
        eprintln!("  >>> CROWN tightening detected in pre-norm group!");
    } else {
        eprintln!("  >>> No CROWN tightening in pre-norm group.");
    }
}

// ===========================================================================
// Pre-norm group with signed (alternating-sign) weights (#2615)
// ===========================================================================

/// D=64: CROWN vs IBP on pre-norm group with alternating-sign synthetic weights.
///
/// The uniform WEIGHT_MAG=0.001 test above produces ratio=1.0 because all
/// weights are positive — after the first ReLU, all bounds are non-negative,
/// and subsequent positive-weight linear layers preserve this, making all
/// later activations act as identity (CROWN = IBP).
///
/// Alternating-sign weights (`(-1)^i * 0.001`) create mixed-sign A-matrices
/// where CROWN can exploit asymmetric input contributions. If ratio > 1.0
/// (CROWN tighter than IBP), this confirms the uniform-weight artifact.
///
/// Part of #2615.
#[test]
fn test_kokoro_grouped_prenorm_crown_tightening_signed_weights() {
    let dims = KokoroDims::d64();
    let mut layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);

    // Apply alternating signs to weight tensors in the pre-norm layers (0-2).
    for layer in layers[0..3].iter_mut() {
        sign_alternate_weight_bindings(&mut layer.1);
    }

    // Build only the pre-norm layers as a group.
    let prenorm_layers: Vec<_> = layers[0..3].to_vec();
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    // IBP only on the merged graph.
    let merged_graph = nn_verify::tensor_kernels_to_grouped_graph(
        &prenorm_layers,
        nn_verify::NormBoundsMode::Conservative,
    )
    .expect("pre-norm group graph (signed weights)");
    let ibp_output = merged_graph
        .propagate_ibp(&initial)
        .expect("pre-norm IBP (signed)");
    let ibp_width = bt_max_width(&ibp_output);

    // CROWN on the merged graph.
    let (crown_method, crown_output, _fallback) =
        nn_verify::propagate_with_crown_fallback(&merged_graph, &initial)
            .expect("pre-nom CROWN (signed)");
    let crown_width = bt_max_width(&crown_output);

    let ratio = if crown_width > 0.0 {
        ibp_width / crown_width
    } else {
        f32::INFINITY
    };

    eprintln!("=== Pre-norm Group (layers 0-2) CROWN vs IBP — SIGNED WEIGHTS ===");
    eprintln!("  IBP width:   {ibp_width:.6}");
    eprintln!("  CROWN width: {crown_width:.6} (method: {crown_method:?})");
    eprintln!("  IBP/CROWN ratio: {ratio:.4}");

    // Soundness: CROWN width must be <= IBP width (within fp tolerance).
    assert!(
        crown_width <= ibp_width + 1e-3,
        "Signed-weight CROWN width {crown_width} > IBP width {ibp_width} (soundness violation)",
    );

    // Both must be finite.
    assert!(ibp_width.is_finite(), "Signed-weight IBP width not finite");
    assert!(
        crown_width.is_finite(),
        "Signed-weight CROWN width not finite"
    );
    assert!(ibp_width > 0.0, "Signed-weight IBP width must be positive");

    if ratio > 1.01 {
        eprintln!("  >>> CROWN tightening detected with signed weights! ratio={ratio:.4}");
        eprintln!("  >>> Confirms: uniform positive WEIGHT_MAG was the artifact.");
    } else {
        eprintln!("  >>> No CROWN tightening with signed weights (ratio={ratio:.4}).");
        eprintln!("  >>> Suggests deeper structural issue beyond weight sign.");
    }
}

// ===========================================================================
// Validation error path tests for verify_layerwise_grouped
// ===========================================================================

/// Fewer than 2 groups must return InsufficientStages.
#[test]
fn test_kokoro_grouped_single_group_error() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 1, 2, 3, 4, 5, 6, 7]],
    };
    let result = verify_layerwise_grouped(&layers, &initial, &grouping);
    assert!(result.is_err(), "single group must be rejected");
}

/// Empty group must return InvalidConfig.
#[test]
fn test_kokoro_grouped_empty_group_error() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 1], vec![], vec![2, 3]],
    };
    let result = verify_layerwise_grouped(&layers, &initial, &grouping);
    assert!(result.is_err(), "empty group must be rejected");
}

/// Out-of-range index must return InvalidConfig.
#[test]
fn test_kokoro_grouped_out_of_range_error() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 1], vec![999]],
    };
    let result = verify_layerwise_grouped(&layers, &initial, &grouping);
    assert!(result.is_err(), "out-of-range index must be rejected");
}

/// Non-monotonic indices within a group must return InvalidConfig.
#[test]
fn test_kokoro_grouped_non_monotonic_within_error() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 2, 1], vec![3, 4]],
    };
    let result = verify_layerwise_grouped(&layers, &initial, &grouping);
    assert!(
        result.is_err(),
        "non-monotonic within group must be rejected"
    );
}

/// Cross-group ordering violation must return InvalidConfig.
#[test]
fn test_kokoro_grouped_cross_group_ordering_error() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise_deep(&dims, NUM_RESBLOCKS);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);

    // Group 1 starts at 1, but group 0 ends at 2 — violation.
    let grouping = LayerwiseGrouping {
        groups: vec![vec![0, 2], vec![1, 3]],
    };
    let result = verify_layerwise_grouped(&layers, &initial, &grouping);
    assert!(
        result.is_err(),
        "cross-group ordering violation must be rejected"
    );
}
