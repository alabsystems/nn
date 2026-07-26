// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests: Unicode perturbation bridge → CROWN verification.
//!
//! Verifies the end-to-end flow from Unicode text analysis through to
//! formal bounds propagation:
//!
//! 1. `identify_vulnerable_positions()` finds text positions at risk
//! 2. `expand_confusion_sets_for_unicode()` generates perturbation sets
//! 3. `verify_robustness()` uses the expanded sets for CROWN verification
//!
//! This is the AC2 compose test for the Unicode-to-embedding bridge (#1740).
//! The bridge connects AC1 (perturbation sets from Unicode analysis) to
//! AC2 (CROWN phoneme stability) — text-level attacks mapped to
//! embedding-space bounds that NY can verify.
//!
//! Part of #1740: Adversarial Robustness of TTS.

#[path = "phoneme_stability.rs"]
mod helpers;

use super::common;
use super::common::assert_bounds_valid;
use helpers::{
    build_phoneme_encoder, phoneme_encoder_bindings, synthetic_embedding_weights,
    test_confusion_sets, EMBED_DIM, SEQ_LEN, VOCAB_SIZE,
};
use nn_tts_verify::{
    analyze_unicode_coverage, expand_confusion_sets_for_unicode, identify_vulnerable_positions,
    map_to_phoneme_confusion_sets, sequence_perturbation_bounds, verify_robustness, ConfusionSet,
    RobustnessConfig, RobustnessProperty, UnicodeSafetyConfig, VulnerabilityType,
};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Helper: stub G2P mapping (char index → phoneme token ID)
// ---------------------------------------------------------------------------

/// Simple G2P stub: maps character index to a token ID.
///
/// Lowercase letters map to their index mod VOCAB_SIZE. Non-letters return None.
/// This mirrors the test stub in `unicode_perturbation_tests.rs` but adapted
/// to use VOCAB_SIZE=16 from the phoneme_stability helpers.
fn stub_char_to_token(text: &str) -> impl Fn(usize) -> Option<u32> + '_ {
    move |idx: usize| {
        let ch = text.chars().nth(idx)?;
        if ch.is_ascii_alphabetic() {
            Some((idx as u32) % (VOCAB_SIZE as u32))
        } else {
            None
        }
    }
}

/// Create BoundedTensor from f64 bounds vectors.
fn bounded_tensor_from_f64(lower: &[f64], upper: &[f64], shape: &[usize]) -> BoundedTensor {
    let lo_f32: Vec<f32> = lower.iter().map(|&v| v as f32).collect();
    let hi_f32: Vec<f32> = upper.iter().map(|&v| v as f32).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lo_f32).expect("valid lower shape"),
        ArrayD::from_shape_vec(IxDyn(shape), hi_f32).expect("valid upper shape"),
    )
    .expect("valid bounded tensor")
}

/// Build a GraphNetwork from the synthetic phoneme encoder.
fn build_test_graph() -> nn_verify::GraphNetwork {
    let def = build_phoneme_encoder();
    let bindings = phoneme_encoder_bindings();
    tensor_kernel_to_graph(&def, &bindings).expect("graph translation")
}

// ===========================================================================
// Test 1: Unicode vulnerable positions expand into confusion sets
// ===========================================================================

#[test]
fn test_unicode_coverage_expands_to_confusion_sets() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";

    // Identify vulnerable positions
    let vuln = identify_vulnerable_positions(text, &config);
    assert!(
        !vuln.is_empty(),
        "should find vulnerable positions in 'hello'"
    );

    // Map to phoneme confusion sets
    let char_to_token = stub_char_to_token(text);
    let existing_sets = test_confusion_sets();
    let report = analyze_unicode_coverage(text, &config, &char_to_token, &existing_sets);

    // Expand confusion sets to include unicode-derived ones
    let expanded = expand_confusion_sets_for_unicode(&existing_sets, &report);

    // Expanded should include original sets plus any uncovered unicode-derived sets.
    assert!(
        expanded.len() >= existing_sets.len(),
        "expanded ({}) should be >= original ({})",
        expanded.len(),
        existing_sets.len()
    );

    // Unicode-derived sets should have names starting with "unicode_derived_"
    let unicode_derived: Vec<_> = expanded
        .iter()
        .filter(|s| s.name.starts_with("unicode_derived_"))
        .collect();
    assert_eq!(
        unicode_derived.len(),
        report.uncovered,
        "uncovered positions ({}) should each get a derived set",
        report.uncovered
    );
}

// ===========================================================================
// Test 2: Expanded confusion sets produce valid CROWN verification
// ===========================================================================

#[test]
fn test_expanded_sets_verify_robustness() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let existing_sets = test_confusion_sets();

    // Get expanded confusion sets (original + unicode-derived)
    let report = analyze_unicode_coverage(text, &config, &char_to_token, &existing_sets);
    let expanded = expand_confusion_sets_for_unicode(&existing_sets, &report);

    // Build graph and run verify_robustness with expanded sets
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();

    // Use first SEQ_LEN tokens as base sequence
    let base_tokens: Vec<u32> = (0..SEQ_LEN as u32).collect();

    let robustness_config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: expanded,
        property: RobustnessProperty::OutputStable { max_width: 1.0 },
    };

    let cert = verify_robustness(
        &graph,
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &robustness_config,
    )
    .expect("verify_robustness with expanded sets");

    // Certificate should have valid structure
    for pr in &cert.position_results {
        assert!(
            pr.output_width.is_finite(),
            "position {} should have finite output width",
            pr.position
        );
    }
}

// ===========================================================================
// Test 3: Unicode-derived single-token confusion sets produce tight bounds
// ===========================================================================

#[test]
fn test_unicode_derived_single_token_tight_bounds() {
    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = (0..SEQ_LEN as u32).collect();

    // Create a unicode-derived single-token confusion set
    // (This is what expand_confusion_sets_for_unicode produces for uncovered positions)
    let single_token_sets = vec![ConfusionSet {
        name: "unicode_derived_pos0".into(),
        token_ids: vec![0],
        labels: vec!["unicode_homoglyph".into()],
        category: nn_tts_verify::ConfusionCategory::EmbeddingSimilar,
    }];

    let config = RobustnessConfig {
        max_perturbation_positions: 1,
        confusion_sets: single_token_sets,
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
    .expect("verify_robustness with single-token set");

    // Single-token set = point bounds = zero perturbation → near-zero width
    assert_eq!(cert.position_results.len(), 1, "only token 0 matches");
    let pr = &cert.position_results[0];
    assert!(
        pr.output_width < 1e-4,
        "single-token confusion set should produce near-zero width, got {}",
        pr.output_width
    );
    assert!(pr.property_holds, "should satisfy tight property");
    assert!(cert.is_robust, "should be robust with point bounds");
}

// ===========================================================================
// Test 4: Cyrillic homoglyph attack detected and mapped to perturbation
// ===========================================================================

/// Extended G2P stub that maps non-ASCII characters to phoneme tokens.
///
/// The standard `stub_char_to_token` only handles ASCII letters. This variant
/// also maps Cyrillic and other non-ASCII alphabetic chars, modeling a real G2P
/// system that processes mixed-script input.
fn extended_char_to_token(text: &str) -> impl Fn(usize) -> Option<u32> + '_ {
    move |idx: usize| {
        let ch = text.chars().nth(idx)?;
        if ch.is_alphabetic() {
            Some((idx as u32) % (VOCAB_SIZE as u32))
        } else {
            None
        }
    }
}

#[test]
fn test_cyrillic_attack_maps_to_perturbation() {
    let config = UnicodeSafetyConfig::default();
    // "hеllo" with Cyrillic е (U+0435) at position 1
    let text = "h\u{0435}llo";

    let vuln = identify_vulnerable_positions(text, &config);

    // Should detect the Cyrillic е as a homoglyph
    let cyrillic_pos: Vec<_> = vuln
        .iter()
        .filter(|v| v.original_char == '\u{0435}')
        .collect();
    assert!(
        !cyrillic_pos.is_empty(),
        "should detect Cyrillic е in 'hеllo'"
    );

    // Map to phoneme confusion sets using extended G2P (handles non-ASCII)
    let char_to_token = extended_char_to_token(text);
    let existing_sets = test_confusion_sets();
    let derived = map_to_phoneme_confusion_sets(&vuln, &char_to_token, &existing_sets);

    // The Cyrillic position should have a derived confusion set entry
    let cyrillic_derived: Vec<_> = derived
        .iter()
        .filter(|d| d.vulnerability == VulnerabilityType::Homoglyph && d.source_char == '\u{0435}')
        .collect();
    assert!(
        !cyrillic_derived.is_empty(),
        "Cyrillic е should produce a unicode-derived confusion set"
    );

    // Verify the derived entry has a valid phoneme token mapping
    for d in &cyrillic_derived {
        assert!(
            d.phoneme_token_id < VOCAB_SIZE as u32,
            "token ID should be within vocab range"
        );
    }
}

// ===========================================================================
// Test 5: Expanded sets IBP propagation produces valid bounds
// ===========================================================================

#[test]
fn test_expanded_sets_ibp_propagation() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let existing_sets = test_confusion_sets();

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &existing_sets);
    let expanded = expand_confusion_sets_for_unicode(&existing_sets, &report);

    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = (0..SEQ_LEN as u32).collect();

    // Build perturbation bounds using the expanded confusion sets
    let positions: Vec<usize> = (0..SEQ_LEN).collect();
    let (lower, upper) = sequence_perturbation_bounds(
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &positions,
        &expanded,
    )
    .expect("perturbation bounds with expanded sets");

    let input = bounded_tensor_from_f64(&lower, &upper, &[SEQ_LEN, EMBED_DIM]);
    let output = graph.propagate_ibp(&input).expect("IBP with expanded sets");
    assert_bounds_valid(&output);
}

// ===========================================================================
// Test 6: Expanded sets CROWN propagation
// ===========================================================================

#[test]
fn test_expanded_sets_crown_propagation() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let existing_sets = test_confusion_sets();

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &existing_sets);
    let expanded = expand_confusion_sets_for_unicode(&existing_sets, &report);

    let graph = build_test_graph();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = (0..SEQ_LEN as u32).collect();

    // Build perturbation bounds using expanded confusion sets, perturb position 0 only
    let (lower, upper) = sequence_perturbation_bounds(
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &[0],
        &expanded,
    )
    .expect("perturbation bounds");

    let input = bounded_tensor_from_f64(&lower, &upper, &[SEQ_LEN, EMBED_DIM]);

    // Run both IBP and CROWN
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (method, crown_output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN");

    assert_bounds_valid(&ibp_output);
    assert_bounds_valid(&crown_output);

    // When CROWN succeeds, it should be at least as tight as IBP
    if format!("{method:?}").contains("Crown") {
        common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
    }
}

// ===========================================================================
// Test 7: Coverage report correctly partitions covered vs uncovered
// ===========================================================================

#[test]
fn test_coverage_report_partition_invariant() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let existing_sets = test_confusion_sets();

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &existing_sets);

    // Partition invariant: covered + uncovered == total
    assert_eq!(
        report.covered_by_linguistic + report.uncovered,
        report.total_vulnerable,
        "covered ({}) + uncovered ({}) should equal total ({})",
        report.covered_by_linguistic,
        report.uncovered,
        report.total_vulnerable
    );

    // Coverage ratio is well-formed
    assert!(report.coverage_ratio >= 0.0 && report.coverage_ratio <= 1.0);

    // With non-empty text and existing confusion sets, some positions
    // may be covered by the linguistic sets.
    if report.total_vulnerable > 0 && report.covered_by_linguistic > 0 {
        assert!(
            report.coverage_ratio > 0.0,
            "should have positive coverage when some positions covered"
        );
    }
}

// ===========================================================================
// Test 8: Empty confusion sets → all uncovered → all unicode-derived
// ===========================================================================

#[test]
fn test_empty_sets_all_uncovered() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let no_sets: Vec<ConfusionSet> = vec![];

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &no_sets);

    // With no existing sets, all positions should be uncovered
    assert_eq!(report.covered_by_linguistic, 0);
    assert_eq!(report.uncovered, report.total_vulnerable);

    // Expand should create one derived set per uncovered position
    let expanded = expand_confusion_sets_for_unicode(&no_sets, &report);
    assert_eq!(
        expanded.len(),
        report.uncovered,
        "expanded should have one set per uncovered position"
    );

    // All derived sets should be single-token (point bounds)
    for cs in &expanded {
        assert!(
            cs.name.starts_with("unicode_derived_"),
            "derived set name should start with unicode_derived_, got: {}",
            cs.name
        );
        assert_eq!(
            cs.token_ids.len(),
            1,
            "derived set should have single token, got: {:?}",
            cs.token_ids
        );
    }
}

// ===========================================================================
// Test 9: verify_and_record with unicode-expanded sets
// ===========================================================================

#[test]
fn test_verify_and_record_unicode_expanded() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let existing_sets = test_confusion_sets();

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &existing_sets);
    let expanded = expand_confusion_sets_for_unicode(&existing_sets, &report);

    let def = build_phoneme_encoder();
    let bindings = phoneme_encoder_bindings();
    let emb_weights = synthetic_embedding_weights();
    let base_tokens: Vec<u32> = (0..SEQ_LEN as u32).collect();

    // Build perturbation bounds
    let (lower, upper) = sequence_perturbation_bounds(
        &emb_weights,
        VOCAB_SIZE,
        EMBED_DIM,
        &base_tokens,
        &[0],
        &expanded,
    )
    .expect("perturbation bounds");

    let input = bounded_tensor_from_f64(&lower, &upper, &[SEQ_LEN, EMBED_DIM]);

    // Run verify_and_record through the standard pipeline
    let result = common::verify_and_assert(&def, &bindings, &input, "unicode_perturbation_bridge");

    // Output should have expected shape
    assert_eq!(
        result.output_bounds.shape(),
        &[SEQ_LEN, helpers::OUTPUT_DIM]
    );
}

// ===========================================================================
// Test 10: Invisible character attack produces perturbation set
// ===========================================================================

#[test]
fn test_invisible_char_attack_perturbation() {
    let config = UnicodeSafetyConfig::default();
    // "he\u{200B}llo" with zero-width space between 'e' and 'l'
    let text = "he\u{200B}llo";

    let vuln = identify_vulnerable_positions(text, &config);

    // Should detect the invisible character
    let invisible_positions: Vec<_> = vuln
        .iter()
        .filter(|v| v.attack_type == VulnerabilityType::InvisibleInsertion)
        .collect();
    assert!(
        !invisible_positions.is_empty(),
        "should detect invisible character in 'he\\u{{200B}}llo'"
    );

    // Map to phoneme confusion sets
    let char_to_token = stub_char_to_token(text);
    let report = analyze_unicode_coverage(text, &config, &char_to_token, &test_confusion_sets());

    // Total vulnerable should be > 0
    assert!(
        report.total_vulnerable > 0,
        "should have vulnerable positions"
    );
}
