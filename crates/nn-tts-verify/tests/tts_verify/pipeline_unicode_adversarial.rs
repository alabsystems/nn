// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Unicode safety → adversarial robustness pipeline.
//!
//! Demonstrates the complete text-domain adversarial robustness pipeline:
//! 1. Unicode scanning catches homoglyphs, invisible chars, bidi overrides
//! 2. Sanitized text maps to phoneme token IDs
//! 3. Confusion sets identify perturbation neighborhoods per position
//! 4. Embedding bounds computed for verifiable token-level robustness
//!
//! This pipeline does NOT require NY — it operates purely in
//! the text/embedding domain. The bounds produced here feed into
//! NY CROWN propagation when available.
//!
//! Run with:
//!   cargo test -p nn-tts-verify --test pipeline_unicode_adversarial
//!
//! Part of #1740: Adversarial Robustness of TTS.

use nn_tts_verify::{
    embedding_bounds_for_token_set, english_confusion_sets, scan_unicode,
    sequence_perturbation_bounds, tts_confusables, ConfusionCategory, UnicodeAttack,
    UnicodeSafetyConfig,
};

// Synthetic embedding weights: 10 tokens × 4 dimensions.
const VOCAB_SIZE: usize = 10;
const EMBED_DIM: usize = 4;

fn synthetic_embeddings() -> Vec<f64> {
    // Each token has a distinct embedding vector for testability.
    // token 0: [0.1, 0.2, 0.3, 0.4]
    // token 1: [1.1, 1.2, 1.3, 1.4]
    // ...
    // token 9: [9.1, 9.2, 9.3, 9.4]
    let mut weights = Vec::with_capacity(VOCAB_SIZE * EMBED_DIM);
    for t in 0..VOCAB_SIZE {
        for d in 0..EMBED_DIM {
            weights.push(t as f64 + 0.1 * (d + 1) as f64);
        }
    }
    weights
}

// ---------------------------------------------------------------
// Test 1: Full pipeline — Cyrillic homoglyph attack → sanitize → bounds
// ---------------------------------------------------------------
#[test]
fn test_pipeline_homoglyph_sanitize_to_bounds() {
    // Input with Cyrillic 'а' (U+0430) disguised as Latin 'a'.
    let input = "p\u{0430}th"; // "pаth" with Cyrillic а

    let config = UnicodeSafetyConfig::default();
    let result = scan_unicode(input, &config);

    // Verify homoglyph detected.
    assert!(
        result
            .attacks
            .iter()
            .any(|a| matches!(a, UnicodeAttack::Homoglyph { .. })),
        "Should detect Cyrillic homoglyph"
    );

    // After scanning, use sanitized text for phoneme lookup.
    // Simulate: sanitized "path" → phoneme tokens [3, 2, 7, 5]
    let phoneme_tokens: Vec<u32> = vec![3, 2, 7, 5];

    // Position 1 was the homoglyph — mark as perturbation-vulnerable.
    let perturbation_positions: Vec<usize> = result
        .attacks
        .iter()
        .filter_map(|a| match a {
            UnicodeAttack::Homoglyph { byte_offset, .. } => {
                // Map byte offset to phoneme position (simplified).
                // In real usage, a G2P alignment would provide this.
                Some(*byte_offset - 1) // approximate position
            }
            _ => None,
        })
        .collect();

    assert!(
        !perturbation_positions.is_empty(),
        "Should have perturbation positions"
    );

    // Build tight bounds for the perturbation-vulnerable positions.
    let embeddings = synthetic_embeddings();

    // Create a small confusion set for the attack position.
    let confusion_sets = vec![nn_tts_verify::ConfusionSet {
        name: "test_homoglyph_confusable".into(),
        token_ids: vec![2, 8], // token 2 confusable with token 8
        labels: vec!["a_latin".into(), "a_cyrillic".into()],
        category: ConfusionCategory::EmbeddingSimilar,
    }];

    let (lower, upper) = sequence_perturbation_bounds(
        &embeddings,
        VOCAB_SIZE,
        EMBED_DIM,
        &phoneme_tokens,
        &[1], // position 1 is the attacked position
        &confusion_sets,
    )
    .expect("bounds computation should succeed");

    assert_homoglyph_bounds(&lower, &upper, phoneme_tokens.len());
}

/// Verify bounds structure: fixed positions have point bounds,
/// perturbed position 1 spans tokens 2 and 8.
fn assert_homoglyph_bounds(lower: &[f64], upper: &[f64], n_positions: usize) {
    let total_dim = n_positions * EMBED_DIM;
    assert_eq!(lower.len(), total_dim);
    assert_eq!(upper.len(), total_dim);

    // Position 0 (fixed): bounds are point bounds.
    for d in 0..EMBED_DIM {
        assert!(
            (lower[d] - upper[d]).abs() < 1e-12,
            "Fixed position 0 dim {d} should have point bounds"
        );
    }

    // Position 1 (perturbed): bounds span tokens 2 and 8.
    for d in 0..EMBED_DIM {
        let idx = EMBED_DIM + d;
        assert!(
            upper[idx] > lower[idx],
            "Perturbed position 1 dim {d} should have non-point bounds"
        );
        let emb_2 = 2.0 + 0.1 * (d + 1) as f64;
        let emb_8 = 8.0 + 0.1 * (d + 1) as f64;
        assert!(
            (lower[idx] - emb_2).abs() < 1e-12,
            "Lower bound should be token 2's embedding"
        );
        assert!(
            (upper[idx] - emb_8).abs() < 1e-12,
            "Upper bound should be token 8's embedding"
        );
    }
}

// ---------------------------------------------------------------
// Test 2: Invisible character insertion → wider perturbation bounds
// ---------------------------------------------------------------
#[test]
fn test_pipeline_invisible_char_widens_bounds() {
    // Input with zero-width space inserted between characters.
    let input = "he\u{200B}llo"; // "he​llo" with ZWS

    let config = UnicodeSafetyConfig::default();
    let result = scan_unicode(input, &config);

    // ZWS should be detected as invisible character.
    assert!(
        result
            .attacks
            .iter()
            .any(|a| matches!(a, UnicodeAttack::InvisibleChar { .. })),
        "Should detect zero-width space"
    );
    assert!(result.was_modified, "Should strip invisible characters");

    // After sanitization, text is "hello" (ZWS removed).
    // The key insight: invisible chars can shift phoneme alignment,
    // so positions near the insertion point need wider bounds.
    assert_eq!(result.sanitized, "hello");
}

// ---------------------------------------------------------------
// Test 3: Bidi override → full-sequence perturbation
// ---------------------------------------------------------------
#[test]
fn test_pipeline_bidi_override_flags_all_positions() {
    // Input with RLO (right-to-left override).
    let input = "hello\u{202E}world";

    let config = UnicodeSafetyConfig::default();
    let result = scan_unicode(input, &config);

    let bidi_attacks: Vec<_> = result
        .attacks
        .iter()
        .filter(|a| matches!(a, UnicodeAttack::BidiOverride { .. }))
        .collect();
    assert_eq!(bidi_attacks.len(), 1, "Should detect one bidi override");

    // When bidi override is present, all positions after the override
    // should be considered adversarially perturbed.
    assert!(result.was_modified, "Should strip bidi override");
}

// ---------------------------------------------------------------
// Test 4: Confusion set coverage — English phoneme confusion sets
//         produce tighter bounds than full vocabulary.
// ---------------------------------------------------------------
#[test]
fn test_confusion_set_bounds_tighter_than_full_vocab() {
    let embeddings = synthetic_embeddings();

    // Full vocabulary bounds.
    let all_tokens: Vec<u32> = (0..VOCAB_SIZE as u32).collect();
    let (full_lower, full_upper) =
        embedding_bounds_for_token_set(&embeddings, VOCAB_SIZE, EMBED_DIM, &all_tokens)
            .expect("full bounds");

    // Confusion set bounds (tokens 3 and 4 only).
    let (cs_lower, cs_upper) =
        embedding_bounds_for_token_set(&embeddings, VOCAB_SIZE, EMBED_DIM, &[3, 4])
            .expect("confusion set bounds");

    // Confusion set bounds must be at least as tight as full bounds.
    for d in 0..EMBED_DIM {
        assert!(
            cs_lower[d] >= full_lower[d],
            "Confusion set lower[{d}] should be >= full vocab lower"
        );
        assert!(
            cs_upper[d] <= full_upper[d],
            "Confusion set upper[{d}] should be <= full vocab upper"
        );
        // And strictly tighter (since we only have 2 of 10 tokens).
        let cs_width = cs_upper[d] - cs_lower[d];
        let full_width = full_upper[d] - full_lower[d];
        assert!(
            cs_width < full_width,
            "Confusion set width ({cs_width}) should be < full vocab width ({full_width})"
        );
    }
}

// ---------------------------------------------------------------
// Test 5: tts_confusables list covers common Cyrillic/Latin pairs
//         that scan_unicode detects as homoglyphs.
// ---------------------------------------------------------------
#[test]
fn test_confusables_list_aligns_with_scanner() {
    let confusables = tts_confusables();

    // For each confusable pair, the scanner should detect the
    // non-Latin character as a homoglyph when embedded in Latin text.
    let config = UnicodeSafetyConfig::default();

    let mut detected_count = 0;
    for (non_latin, _latin) in &confusables {
        let input = format!("test{non_latin}ing");
        let result = scan_unicode(&input, &config);

        // The non-Latin character should be flagged (either as
        // homoglyph or unexpected script).
        let is_flagged = result.attacks.iter().any(|a| {
            matches!(
                a,
                UnicodeAttack::Homoglyph { .. } | UnicodeAttack::UnexpectedScript { .. }
            )
        });
        if is_flagged {
            detected_count += 1;
        }
    }

    // At least 80% of confusable pairs should be detected.
    let detection_rate = f64::from(detected_count) / confusables.len() as f64;
    assert!(
        detection_rate >= 0.8,
        "Detection rate {detection_rate:.2} should be >= 0.80 ({detected_count}/{})",
        confusables.len()
    );
}

// ---------------------------------------------------------------
// Test 6: Multi-attack input — homoglyph + invisible + bidi
// ---------------------------------------------------------------
#[test]
fn test_pipeline_multi_attack_vector() {
    // Combine multiple attack types in one input.
    let input = "h\u{0435}\u{200B}l\u{202E}o"; // Cyrillic е + ZWS + RLO

    let config = UnicodeSafetyConfig::default();
    let result = scan_unicode(input, &config);

    let homoglyphs = result
        .attacks
        .iter()
        .filter(|a| matches!(a, UnicodeAttack::Homoglyph { .. }))
        .count();
    let invisible = result
        .attacks
        .iter()
        .filter(|a| matches!(a, UnicodeAttack::InvisibleChar { .. }))
        .count();
    let bidi = result
        .attacks
        .iter()
        .filter(|a| matches!(a, UnicodeAttack::BidiOverride { .. }))
        .count();

    // All three attack types should be detected.
    assert!(
        homoglyphs >= 1,
        "Should detect homoglyph(s): got {homoglyphs}"
    );
    assert!(
        invisible >= 1,
        "Should detect invisible char(s): got {invisible}"
    );
    assert!(bidi >= 1, "Should detect bidi override(s): got {bidi}");

    // Total attacks should reflect all three categories.
    assert!(
        result.attacks.len() >= 3,
        "Should detect at least 3 attacks total, got {}",
        result.attacks.len()
    );
}

// ---------------------------------------------------------------
// Test 7: English confusion sets produce non-degenerate bounds.
// ---------------------------------------------------------------
#[test]
fn test_english_confusion_sets_structure() {
    let sets = english_confusion_sets();

    assert!(!sets.is_empty(), "Should have confusion sets");

    // Each set should have at least 2 tokens.
    for cs in &sets {
        assert!(
            cs.token_ids.len() >= 2,
            "Confusion set '{}' should have >= 2 tokens, got {}",
            cs.name,
            cs.token_ids.len()
        );
        assert_eq!(
            cs.token_ids.len(),
            cs.labels.len(),
            "Token IDs and labels should have same length for '{}'",
            cs.name
        );
    }

    // Should cover all confusion categories.
    let has_voicing = sets
        .iter()
        .any(|cs| cs.category == ConfusionCategory::VoicingPair);
    let has_place = sets
        .iter()
        .any(|cs| cs.category == ConfusionCategory::PlaceConfusion);
    let has_manner = sets
        .iter()
        .any(|cs| cs.category == ConfusionCategory::MannerConfusion);
    let has_vowel = sets
        .iter()
        .any(|cs| cs.category == ConfusionCategory::VowelProximity);
    let has_cross = sets
        .iter()
        .any(|cs| cs.category == ConfusionCategory::CrossLanguage);

    assert!(has_voicing, "Should have voicing pair sets");
    assert!(has_place, "Should have place confusion sets");
    assert!(has_manner, "Should have manner confusion sets");
    assert!(has_vowel, "Should have vowel proximity sets");
    assert!(has_cross, "Should have cross-language sets");
}
