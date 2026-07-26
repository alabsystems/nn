// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for `verify_robustness()`.
//!
//! Exercises the full adversarial robustness verification pipeline:
//! `RobustnessConfig` → `verify_robustness()` → `RobustnessCertificate`.
//!
//! The function internally builds per-position perturbation bounds from
//! confusion sets, runs CROWN propagation through the NY graph,
//! and checks the specified `RobustnessProperty` at each position.
//!
//! Part of #1740: Adversarial Robustness of TTS.

#[path = "phoneme_stability.rs"]
mod helpers;

use helpers::{
    build_phoneme_encoder, phoneme_encoder_bindings, synthetic_embedding_weights,
    test_confusion_sets, EMBED_DIM, VOCAB_SIZE,
};
use nn_tts_verify::{
    verify_robustness, ConfusionCategory, ConfusionSet, RobustnessConfig, RobustnessProperty,
};
use nn_verify::tensor_kernel_to_graph;

/// Build a GraphNetwork from the synthetic phoneme encoder.
fn build_test_graph() -> nn_verify::GraphNetwork {
    let def = build_phoneme_encoder();
    let bindings = phoneme_encoder_bindings();
    tensor_kernel_to_graph(&def, &bindings).expect("graph translation")
}

// ===========================================================================
// Test 1: verify_robustness with OutputStable property
// ===========================================================================

#[test]
fn test_verify_robustness_output_stable() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::OutputStable { max_width: 0.01 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    // With synthetic small-weight encoder (WEIGHT_MAG=0.01), output should be stable.
    assert!(cert.is_robust, "model should be robust with max_width=0.01");

    // All tested positions should have property_holds == true.
    for pr in &cert.position_results {
        assert!(
            pr.property_holds,
            "position {} should satisfy OutputStable, width={}",
            pr.position, pr.output_width
        );
        assert!(
            pr.output_width.is_finite(),
            "position {} should have finite output width",
            pr.position
        );
    }

    // Certificate should have non-empty results (tokens 0,2,4 are in confusion sets).
    assert!(
        !cert.position_results.is_empty(),
        "should have at least one position tested"
    );

    // worst_case_width should be the maximum of all position widths.
    let max_width = cert
        .position_results
        .iter()
        .map(|r| r.output_width)
        .fold(0.0_f64, f64::max);
    assert!(
        (cert.worst_case_width - max_width).abs() < 1e-10,
        "worst_case_width ({}) should equal max of position widths ({})",
        cert.worst_case_width,
        max_width
    );
}

// ===========================================================================
// Test 2: verify_robustness with DurationPositive property
// ===========================================================================

#[test]
fn test_verify_robustness_duration_positive() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::DurationPositive,
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    // The synthetic encoder with small random weights may produce negative lower
    // bounds — DurationPositive checks lower > 0. This is testing that the
    // property is correctly evaluated, not that the model satisfies it.
    assert!(
        !cert.position_results.is_empty(),
        "should test at least one position"
    );

    for pr in &cert.position_results {
        // Each position has a clear boolean result.
        // property_holds is true only if ALL output lower bounds > 0.
        assert!(
            pr.output_width.is_finite(),
            "position {} output width should be finite",
            pr.position
        );
        assert!(
            !pr.propagation_mode.is_empty(),
            "propagation_mode should be reported"
        );
    }
}

// ===========================================================================
// Test 3: verify_robustness with F0Bounded property
// ===========================================================================

#[test]
fn test_verify_robustness_f0_bounded() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::F0Bounded {
            min_hz: -0.05,
            max_hz: 0.05,
        },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    // With tight bounds [-0.05, 0.05], synthetic small-weight encoder should pass.
    assert!(
        cert.is_robust,
        "model should be robust with F0 bounds [-0.05, 0.05]"
    );

    for pr in &cert.position_results {
        assert!(
            pr.property_holds,
            "position {} should satisfy F0Bounded [-0.05, 0.05], width={}",
            pr.position, pr.output_width
        );
    }
}

// ===========================================================================
// Test 4: Tokens not in any confusion set are skipped
// ===========================================================================

#[test]
fn test_verify_robustness_no_matching_confusion_set() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();

    // Use token IDs that are NOT in any of the test confusion sets.
    // test_confusion_sets() covers tokens 0,1 (voicing), 2,3 (sibilant),
    // 4,5,6 (vowels), 7,8,9 (nasals). Token 10+ are not in any set.
    let base_tokens: Vec<u32> = vec![10, 11, 12, 13];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::OutputStable { max_width: 100.0 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    // No positions were tested — all tokens outside confusion sets.
    assert!(
        cert.position_results.is_empty(),
        "no positions should be tested when tokens aren't in any confusion set"
    );
    // With no tested positions, the model is trivially robust.
    assert!(cert.is_robust, "trivially robust when no positions tested");
    assert_eq!(cert.worst_case_width, 0.0);
}

// ===========================================================================
// Test 5: Worst-case position identification
// ===========================================================================

#[test]
fn test_verify_robustness_worst_case_identification() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::OutputStable { max_width: 1000.0 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    // Find the position with the widest output bounds.
    let worst = cert
        .position_results
        .iter()
        .max_by(|a, b| a.output_width.total_cmp(&b.output_width));

    if let Some(worst_result) = worst {
        assert_eq!(
            cert.worst_position, worst_result.position,
            "worst_position should match position with max output_width"
        );
        assert_eq!(
            cert.worst_confusion_set, worst_result.confusion_set,
            "worst_confusion_set should match"
        );
        assert!(
            (cert.worst_case_width - worst_result.output_width).abs() < 1e-10,
            "worst_case_width should match worst position's output_width"
        );
    }
}

// ===========================================================================
// Test 6: Empty tokens returns error
// ===========================================================================

#[test]
fn test_verify_robustness_empty_tokens_error() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::DurationPositive,
    };

    let result = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &[], // empty tokens
        &config,
    );

    assert!(result.is_err(), "empty tokens should return error");
}

// ===========================================================================
// Test 7: Token out of range returns error
// ===========================================================================

#[test]
fn test_verify_robustness_token_out_of_range() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();

    // Token 999 is beyond VOCAB_SIZE (16).
    let base_tokens: Vec<u32> = vec![0, 999, 4, 7];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::OutputStable { max_width: 100.0 },
    };

    let result = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    );

    assert!(result.is_err(), "out-of-range token should return error");
}

// ===========================================================================
// Test 8: OutputStable with tight threshold fails gracefully
// ===========================================================================

#[test]
fn test_verify_robustness_tight_threshold_fails() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    // Very tight threshold — even small perturbation exceeds this.
    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::OutputStable { max_width: 0.0 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness should succeed even with tight threshold");

    // With max_width=0.0, any non-zero perturbation should fail the property.
    // (The confusion sets have different embeddings, so width > 0.)
    let any_fails = cert.position_results.iter().any(|r| !r.property_holds);
    assert!(
        any_fails,
        "max_width=0.0 should cause at least one position to fail"
    );
    assert!(
        !cert.is_robust,
        "model should not be robust with max_width=0.0"
    );
}

// ===========================================================================
// Test 9: Certificate reports propagation mode
// ===========================================================================

#[test]
fn test_verify_robustness_reports_propagation_mode() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: test_confusion_sets(),
        property: RobustnessProperty::OutputStable { max_width: 100.0 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    for pr in &cert.position_results {
        // propagation_mode should be non-empty and contain either "Crown" or "Ibp".
        assert!(
            !pr.propagation_mode.is_empty(),
            "propagation_mode should be reported"
        );
        assert!(
            pr.propagation_mode.contains("Crown") || pr.propagation_mode.contains("Ibp"),
            "propagation_mode should be Crown or Ibp, got: {}",
            pr.propagation_mode
        );
    }
}

// ===========================================================================
// Test 10: Single-token confusion set (self-confusion) produces tight bounds
// ===========================================================================

#[test]
fn test_verify_robustness_single_token_confusion_set() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = vec![0, 2, 4, 7];

    // Create a confusion set with just one token — "self confusion".
    // The perturbation bounds should be point bounds (zero width).
    let single_cs = vec![ConfusionSet {
        name: "self_0".into(),
        token_ids: vec![0],
        labels: vec!["tok0".into()],
        category: ConfusionCategory::EmbeddingSimilar,
    }];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: single_cs,
        property: RobustnessProperty::OutputStable { max_width: 1e-4 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &config,
    )
    .expect("verify_robustness");

    // Single-token confusion set means zero perturbation → near-zero output width.
    assert_eq!(cert.position_results.len(), 1, "only token 0 matches");
    let pr = &cert.position_results[0];
    assert_eq!(pr.position, 0);
    assert!(
        pr.output_width < 1e-4,
        "single-token confusion set should produce near-zero width, got {}",
        pr.output_width
    );
    assert!(pr.property_holds);
    assert!(cert.is_robust);
}
