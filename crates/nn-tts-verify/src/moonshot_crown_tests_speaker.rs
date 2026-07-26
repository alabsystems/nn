// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! P4 (speaker consistency) property tests, unified 6-property bundle tests,
//! and D=192 production tests.

use super::*;

// ---------------------------------------------------------------------------
// Property 4: Speaker consistency
// ---------------------------------------------------------------------------

/// Helper to construct speaker consistency evidence.
pub(super) fn speaker_evidence(
    dim: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    reference: Vec<f64>,
    threshold: f64,
    is_sound: bool,
) -> SpeakerConsistencyEvidence {
    SpeakerConsistencyEvidence {
        embed_dim: dim,
        embedding_lower: lower,
        embedding_upper: upper,
        reference_embedding: reference,
        distance_threshold: threshold,
        is_sound,
    }
}

#[test]
fn test_speaker_consistency_proven() {
    // Reference embedding at 0.5 for all dims, bounds very tight [0.45, 0.55].
    // Worst-case per-dim: max(|0.5-0.45|, |0.5-0.55|) = 0.05.
    // d_worst = sqrt(8 * 0.05^2) = sqrt(0.02) ≈ 0.1414 < 0.5 threshold.
    let evidence = speaker_evidence(8, vec![0.45; 8], vec![0.55; 8], vec![0.5; 8], 0.5, true);
    let result = check_speaker_consistency(&evidence);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert_eq!(result.property_index, 3);
    assert_eq!(
        result.property_name,
        "Speaker-consistent (embedding distance < ε)"
    );
    assert!(result.bound_value < 0.5);
    assert!(result.explanation.contains("PROVEN"));
}

#[test]
fn test_speaker_consistency_fails_wide_bounds() {
    // Wide bounds [0.0, 1.0] with reference at 0.5. Per-dim max = 0.5.
    // d_worst = sqrt(8 * 0.5^2) = sqrt(2.0) ≈ 1.414 > 0.5 threshold.
    let evidence = speaker_evidence(8, vec![0.0; 8], vec![1.0; 8], vec![0.5; 8], 0.5, true);
    let result = check_speaker_consistency(&evidence);
    assert!(!result.proven);
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(result.bound_value > 0.5);
}

#[test]
fn test_speaker_consistency_ibp_fallback() {
    // Tight bounds but IBP (not sound) → CrownPartial.
    let evidence = speaker_evidence(
        8,
        vec![0.45; 8],
        vec![0.55; 8],
        vec![0.5; 8],
        0.5,
        false, // IBP fallback
    );
    let result = check_speaker_consistency(&evidence);
    assert!(result.proven);
    assert_eq!(result.level, VerificationLevel::CrownPartial);
    assert!(!result.is_sound);
}

#[test]
fn test_speaker_consistency_dimension_mismatch() {
    // Dimension mismatch: embed_dim=8 but vectors have length 4.
    let evidence = SpeakerConsistencyEvidence {
        embed_dim: 8,
        embedding_lower: vec![0.0; 4],
        embedding_upper: vec![1.0; 4],
        reference_embedding: vec![0.5; 4],
        distance_threshold: 0.5,
        is_sound: true,
    };
    let result = check_speaker_consistency(&evidence);
    assert!(!result.proven);
    assert!(!result.is_sound);
    assert_eq!(result.bound_value, f64::INFINITY);
    assert!(result.explanation.contains("dimension mismatch"));
}

#[test]
fn test_speaker_consistency_d192_production() {
    // ECAPA-TDNN produces 192-dim L2-normalized embeddings.
    // L2-normalized: ||embed|| = 1.0, so each element is in roughly [-1/sqrt(192), 1/sqrt(192)].
    // Reference embedding: uniform 1/sqrt(192) ≈ 0.0722.
    let dim = 192;
    let norm_val = 1.0 / (dim as f64).sqrt();
    // Tight bounds around the normalized embedding (±0.01 per element).
    let lower: Vec<f64> = vec![norm_val - 0.01; dim];
    let upper: Vec<f64> = vec![norm_val + 0.01; dim];
    let reference: Vec<f64> = vec![norm_val; dim];
    // d_worst = sqrt(192 * 0.01^2) = sqrt(0.0192) ≈ 0.1386 < 0.3 threshold.
    let evidence = speaker_evidence(dim, lower, upper, reference, 0.3, true);
    let result = check_speaker_consistency(&evidence);
    assert!(result.proven, "speaker consistency must pass at D=192");
    assert_eq!(result.level, VerificationLevel::CrownProven);
    assert!(result.bound_value < 0.15);
}

#[test]
fn test_speaker_consistency_asymmetric_bounds() {
    // Reference closer to upper bound — tests asymmetric distance computation.
    // lower=0.0, upper=0.8, ref=0.7. Per-dim: max(|0.7-0.0|, |0.7-0.8|) = 0.7.
    // d_worst = sqrt(8 * 0.7^2) = sqrt(3.92) ≈ 1.98 > 1.0 threshold.
    let evidence = speaker_evidence(8, vec![0.0; 8], vec![0.8; 8], vec![0.7; 8], 1.0, true);
    let result = check_speaker_consistency(&evidence);
    assert!(!result.proven);
    assert!(result.bound_value > 1.0);
}

// ---------------------------------------------------------------------------
// P1-234: Strengthened speaker consistency edge cases
// ---------------------------------------------------------------------------

/// NaN in embedding bounds must not produce a "proven" result.
///
/// Strengthened from P1-234: IEEE 754 NaN propagates through arithmetic
/// (abs, max, mul, sum) and NaN.is_finite() returns false, so the
/// finiteness guard at line 84 of moonshot_crown_speaker.rs catches it.
/// This test verifies that defense-in-depth.
#[test]
fn test_speaker_consistency_nan_bounds_not_proven() {
    let mut lower = vec![0.45; 8];
    lower[3] = f64::NAN; // inject NaN in one dimension
    let evidence = speaker_evidence(8, lower, vec![0.55; 8], vec![0.5; 8], 0.5, true);
    let result = check_speaker_consistency(&evidence);
    assert!(
        !result.proven,
        "NaN in bounds must not produce proven result"
    );
    assert_eq!(result.level, VerificationLevel::Empirical);
    assert!(!result.bound_value.is_finite(), "d_worst must be NaN/Inf");
}

/// Infinity in embedding bounds must not produce a "proven" result.
#[test]
fn test_speaker_consistency_inf_bounds_not_proven() {
    let mut upper = vec![0.55; 8];
    upper[0] = f64::INFINITY;
    let evidence = speaker_evidence(8, vec![0.45; 8], upper, vec![0.5; 8], 0.5, true);
    let result = check_speaker_consistency(&evidence);
    assert!(
        !result.proven,
        "Inf in bounds must not produce proven result"
    );
    assert_eq!(result.level, VerificationLevel::Empirical);
}

/// Exact numerical verification of worst-case L2 distance.
///
/// For ref=0.5, bounds [0.3, 0.7], dim=4:
///   per-dim: max(|0.5-0.3|, |0.5-0.7|) = max(0.2, 0.2) = 0.2
///   d_worst = sqrt(4 * 0.04) = sqrt(0.16) = 0.4
#[test]
fn test_speaker_consistency_exact_distance() {
    let evidence = speaker_evidence(4, vec![0.3; 4], vec![0.7; 4], vec![0.5; 4], 1.0, true);
    let result = check_speaker_consistency(&evidence);
    assert!(result.proven);
    let expected = (4.0_f64 * 0.04).sqrt(); // 0.4
    assert!(
        (result.bound_value - expected).abs() < 1e-12,
        "expected d_worst = {expected}, got {}",
        result.bound_value
    );
}

/// Reference outside bounds: ref < lower. Worst case is at upper endpoint.
///
/// ref=-1.0, bounds [0.0, 1.0], dim=2:
///   per-dim: max(|-1.0 - 0.0|, |-1.0 - 1.0|) = max(1.0, 2.0) = 2.0
///   d_worst = sqrt(2 * 4.0) = sqrt(8) ≈ 2.828
#[test]
fn test_speaker_consistency_ref_outside_bounds() {
    let evidence = speaker_evidence(2, vec![0.0; 2], vec![1.0; 2], vec![-1.0; 2], 5.0, true);
    let result = check_speaker_consistency(&evidence);
    assert!(result.proven);
    let expected = (2.0_f64 * 4.0).sqrt(); // sqrt(8) ≈ 2.828
    assert!(
        (result.bound_value - expected).abs() < 1e-12,
        "expected d_worst = {expected}, got {}",
        result.bound_value
    );
}

// ---------------------------------------------------------------------------
// Unified 6-property bundle (verify_all_crown_properties)
// ---------------------------------------------------------------------------

#[test]
fn test_all_6_properties_proven() {
    let (cert, timing_cert) =
        temporal::timing_certificate(8, vec![-0.3; 8], vec![0.3; 8], true, 50_000.0, 100_000.0);
    let speaker_ev = speaker_evidence(8, vec![0.45; 8], vec![0.55; 8], vec![0.5; 8], 0.5, true);
    let bundle = verify_all_crown_properties(&cert, &timing_cert, &speaker_ev, 64);
    assert_eq!(bundle.results.len(), 6);
    assert!(bundle.all_proven, "all 6 CROWN properties: {bundle}");
    assert_eq!(bundle.verification_dim, 64);
    // Verify property indices.
    assert_eq!(bundle.results[0].property_index, 0); // non-silence
    assert_eq!(bundle.results[1].property_index, 1); // non-clipping
    assert_eq!(bundle.results[2].property_index, 2); // intelligibility
    assert_eq!(bundle.results[3].property_index, 3); // speaker consistency
    assert_eq!(bundle.results[4].property_index, 4); // temporal
    assert_eq!(bundle.results[5].property_index, 5); // streaming
}

#[test]
fn test_all_6_properties_speaker_fails() {
    let (cert, timing_cert) =
        temporal::timing_certificate(8, vec![-0.3; 8], vec![0.3; 8], true, 50_000.0, 100_000.0);
    // Wide speaker bounds → fail.
    let speaker_ev = speaker_evidence(8, vec![0.0; 8], vec![1.0; 8], vec![0.5; 8], 0.5, true);
    let bundle = verify_all_crown_properties(&cert, &timing_cert, &speaker_ev, 64);
    assert!(!bundle.all_proven);
    assert!(bundle.results[0].proven); // non-silence
    assert!(bundle.results[1].proven); // non-clipping
    assert!(bundle.results[2].proven); // intelligibility
    assert!(!bundle.results[3].proven); // speaker consistency — FAIL
    assert!(bundle.results[4].proven); // temporal
    assert!(bundle.results[5].proven); // streaming
}

#[test]
fn test_all_6_properties_d192() {
    let dim = 192;
    let (cert, timing_cert) = temporal::timing_certificate(
        dim,
        vec![-0.3; dim],
        vec![0.3; dim],
        true,
        45_000.0,
        100_000.0,
    );
    let norm_val = 1.0 / (dim as f64).sqrt();
    let speaker_ev = speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );
    let bundle = verify_all_crown_properties(&cert, &timing_cert, &speaker_ev, dim);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 6);
    assert!(
        bundle.all_proven,
        "all 6 CROWN properties at D=192: {bundle}"
    );
}

#[test]
fn test_all_6_properties_display() {
    let (cert, timing_cert) =
        temporal::timing_certificate(8, vec![-0.3; 8], vec![0.3; 8], true, 50_000.0, 100_000.0);
    let speaker_ev = speaker_evidence(8, vec![0.45; 8], vec![0.55; 8], vec![0.5; 8], 0.5, true);
    let bundle = verify_all_crown_properties(&cert, &timing_cert, &speaker_ev, 192);
    let s = format!("{bundle}");
    assert!(s.contains("Moonshot CROWN Bundle (D=192)"));
    assert!(s.contains("P1:"));
    assert!(s.contains("P4:"));
    assert!(s.contains("P5:"));
    assert!(s.contains("P6:"));
    assert!(s.contains("6/6 proven"));
}
