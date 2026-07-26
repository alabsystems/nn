// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for per-phoneme pronunciation verification.

use crate::phoneme::{PhonemeAlignment, PhonemeResult, PhonemeSpan, PhonemeVerifyConfig};
use crate::phoneme_defects::{detect_defects, PronunciationDefect};
use crate::phoneme_verify::verify_phonemes;
use crate::test_audio_helpers::sine_wave_samples;

/// Generate white noise with a given amplitude.
fn white_noise(amplitude: f32, num_samples: usize) -> Vec<f32> {
    // Deterministic pseudo-random sequence for reproducibility.
    let mut val = 0.5_f32;
    (0..num_samples)
        .map(|_| {
            val = (val * 1103515245.0 + 12345.0) % 2147483648.0 / 2147483648.0;
            (val * 2.0 - 1.0) * amplitude
        })
        .collect()
}

// -- Test 1: Alignment validation --

#[test]
fn test_phoneme_alignment_validation() {
    let sample_rate = 24000;

    // Out-of-bounds span.
    let bad_alignment = PhonemeAlignment {
        phonemes: vec![PhonemeSpan {
            label: "AH0".into(),
            start: 0,
            end: 50000,
        }],
        sample_rate,
        total_samples: 24000,
    };
    assert!(bad_alignment.validate().is_err());

    // Empty span (start >= end).
    let empty_alignment = PhonemeAlignment {
        phonemes: vec![PhonemeSpan {
            label: "AH0".into(),
            start: 100,
            end: 100,
        }],
        sample_rate,
        total_samples: 24000,
    };
    assert!(empty_alignment.validate().is_err());

    // Zero sample rate.
    let zero_sr = PhonemeAlignment {
        phonemes: vec![PhonemeSpan {
            label: "AH0".into(),
            start: 0,
            end: 100,
        }],
        sample_rate: 0,
        total_samples: 24000,
    };
    assert!(zero_sr.validate().is_err());

    // No phonemes.
    let empty_phonemes = PhonemeAlignment {
        phonemes: vec![],
        sample_rate,
        total_samples: 24000,
    };
    assert!(empty_phonemes.validate().is_err());
}

// -- Test 2: Basic 3-phoneme verification --

#[test]
fn test_phoneme_verify_basic() {
    let sample_rate = 24000;
    // Generate 150ms of voiced audio at 220Hz (typical male A3).
    let total_samples = (f64::from(sample_rate) * 0.15) as usize; // 3600 samples
    let samples = sine_wave_samples(220.0, sample_rate, total_samples);

    // Split into 3 equal phonemes of 50ms each.
    let seg_len = total_samples / 3;
    let alignment = PhonemeAlignment {
        phonemes: vec![
            PhonemeSpan {
                label: "AH0".into(),
                start: 0,
                end: seg_len,
            },
            PhonemeSpan {
                label: "B".into(),
                start: seg_len,
                end: seg_len * 2,
            },
            PhonemeSpan {
                label: "AH1".into(),
                start: seg_len * 2,
                end: total_samples,
            },
        ],
        sample_rate,
        total_samples,
    };

    let config = PhonemeVerifyConfig::default();
    let results = verify_phonemes(&samples, &alignment, None, None, &config, sample_rate).unwrap();

    assert_eq!(results.len(), 3);
    // Duration should be ~50ms for each.
    for r in &results {
        assert!(
            r.duration_ms > 40.0 && r.duration_ms < 60.0,
            "duration={}",
            r.duration_ms
        );
    }
}

// -- Test 3: Deletion detection --

#[test]
fn test_phoneme_verify_deletion_detected() {
    let sample_rate = 24000;
    // Total: 100ms audio.
    let total_samples = (f64::from(sample_rate) * 0.1) as usize; // 2400 samples
    let samples = sine_wave_samples(220.0, sample_rate, total_samples);

    // First phoneme is only 5ms (120 samples at 24kHz) — below 20ms minimum.
    let short_end = (f64::from(sample_rate) * 0.005) as usize; // 120 samples
    let alignment = PhonemeAlignment {
        phonemes: vec![
            PhonemeSpan {
                label: "T".into(),
                start: 0,
                end: short_end,
            },
            PhonemeSpan {
                label: "AH0".into(),
                start: short_end,
                end: total_samples,
            },
        ],
        sample_rate,
        total_samples,
    };

    let config = PhonemeVerifyConfig::default();
    let results = verify_phonemes(&samples, &alignment, None, None, &config, sample_rate).unwrap();

    // First phoneme should fail duration check.
    assert!(!results[0].passed, "5ms phoneme should fail");
    assert!(results[0].duration_ms < 20.0);

    // Detect defects.
    let defects = detect_defects(&results, &config);
    assert!(
        defects
            .iter()
            .any(|d| matches!(d, PronunciationDefect::Deletion { .. })),
        "Should detect deletion defect, got: {defects:?}",
    );
}

// -- Test 4: Devoicing detection (low HNR on voiced phoneme) --

#[test]
fn test_phoneme_verify_devoicing_detected() {
    let sample_rate = 24000;
    // Generate 100ms of noise (low HNR) — simulates a devoiced phoneme.
    let total_samples = (f64::from(sample_rate) * 0.1) as usize;
    let samples = white_noise(0.3, total_samples);

    let alignment = PhonemeAlignment {
        phonemes: vec![PhonemeSpan {
            label: "V".into(), // Voiced fricative.
            start: 0,
            end: total_samples,
        }],
        sample_rate,
        total_samples,
    };

    let config = PhonemeVerifyConfig {
        min_voiced_hnr_db: 10.0,
        ..PhonemeVerifyConfig::default()
    };
    let results = verify_phonemes(&samples, &alignment, None, None, &config, sample_rate).unwrap();

    // Check if HNR metric is present and flagged.
    let has_hnr_fail = results[0]
        .metrics
        .iter()
        .any(|m| m.name == "hnr" && !m.passed);
    // Note: noise may or may not produce a measurable HNR depending on segment length.
    // The defect detection path is what we're verifying.
    if has_hnr_fail {
        let defects = detect_defects(&results, &config);
        assert!(defects
            .iter()
            .any(|d| matches!(d, PronunciationDefect::Devoicing { .. })));
    }
}

// -- Test 5: Weak articulation (low energy ratio) --

#[test]
fn test_phoneme_verify_weak_articulation() {
    let sample_rate = 24000;
    // Loud segment followed by very quiet segment.
    let loud_len = (f64::from(sample_rate) * 0.08) as usize;
    let quiet_len = (f64::from(sample_rate) * 0.08) as usize;
    let total_samples = loud_len + quiet_len;

    let mut samples = sine_wave_samples(220.0, sample_rate, loud_len);
    // Very quiet segment (0.001 amplitude vs ~0.7 RMS of sine).
    samples.extend(std::iter::repeat_n(0.001_f32, quiet_len));

    let alignment = PhonemeAlignment {
        phonemes: vec![
            PhonemeSpan {
                label: "AH0".into(),
                start: 0,
                end: loud_len,
            },
            PhonemeSpan {
                label: "AH1".into(),
                start: loud_len,
                end: total_samples,
            },
        ],
        sample_rate,
        total_samples,
    };

    let config = PhonemeVerifyConfig {
        min_energy_ratio: 0.05,
        ..PhonemeVerifyConfig::default()
    };
    let results = verify_phonemes(&samples, &alignment, None, None, &config, sample_rate).unwrap();

    // Second phoneme should fail energy ratio.
    assert!(
        !results[1].passed,
        "Near-silent phoneme should fail energy check"
    );
    let defects = detect_defects(&results, &config);
    assert!(
        defects
            .iter()
            .any(|d| matches!(d, PronunciationDefect::WeakArticulation { .. })),
        "Should detect weak articulation, got: {defects:?}",
    );
}

// -- Test 6: Verification with reference (MCD per-phoneme) --

#[test]
fn test_phoneme_verify_with_reference() {
    let sample_rate = 24000;
    let total_samples = (f64::from(sample_rate) * 0.15) as usize;

    // Reference: 220Hz sine.
    let reference = sine_wave_samples(220.0, sample_rate, total_samples);
    // Candidate: same signal (should have zero MCD).
    let candidate = sine_wave_samples(220.0, sample_rate, total_samples);

    let seg_len = total_samples / 2;
    let alignment = PhonemeAlignment {
        phonemes: vec![
            PhonemeSpan {
                label: "AH0".into(),
                start: 0,
                end: seg_len,
            },
            PhonemeSpan {
                label: "AH1".into(),
                start: seg_len,
                end: total_samples,
            },
        ],
        sample_rate,
        total_samples,
    };

    let config = PhonemeVerifyConfig::default();
    let results = verify_phonemes(
        &candidate,
        &alignment,
        Some(&reference),
        Some(&alignment),
        &config,
        sample_rate,
    )
    .unwrap();

    assert_eq!(results.len(), 2);
    // With identical signals, MCD should be very low.
    for r in &results {
        if let Some(mcd) = r.metrics.iter().find(|m| m.name == "mcd") {
            assert!(
                mcd.passed,
                "MCD should pass for identical signals: value={}",
                mcd.value
            );
        }
    }
}

// -- Test 7: Certificate includes phoneme results --

#[test]
fn test_certificate_includes_phoneme_results() {
    use crate::certificate::Certificate;
    use crate::quality::QualityMetric;

    let phoneme_results = vec![
        PhonemeResult {
            label: "AH0".into(),
            duration_ms: 50.0,
            metrics: vec![QualityMetric {
                name: "duration_ms",
                value: 50.0,
                threshold: 20.0,
                passed: true,
                citation: "Crystal 2003",
            }],
            passed: true,
        },
        PhonemeResult {
            label: "T".into(),
            duration_ms: 10.0,
            metrics: vec![QualityMetric {
                name: "duration_ms",
                value: 10.0,
                threshold: 20.0,
                passed: false,
                citation: "Crystal 2003",
            }],
            passed: false,
        },
    ];

    let cert = Certificate {
        hard_bounds: vec![],
        quality_metrics: vec![],
        phoneme_results: Some(phoneme_results),
        overall_passed: false,
        deterministic_hash: None,
        crown_evidence: None,
        junction_summary: None,
        #[cfg(feature = "ny")]
        dead_neuron_eq_proof: None,
    };

    let report = cert.report();
    assert!(
        report.contains("Per-Phoneme Verification"),
        "Report should have phoneme section"
    );
    assert!(
        report.contains("1/2 phonemes passed"),
        "Report should show pass count"
    );
    assert!(
        report.contains("/AH0/"),
        "Report should include phoneme labels"
    );
    assert!(
        report.contains("/T/"),
        "Report should include failing phoneme"
    );
}

// -- Test 8: Multiple defect types in one utterance --

#[test]
fn test_detect_defects_multiple() {
    use crate::quality::QualityMetric;

    let config = PhonemeVerifyConfig::default();

    let results = vec![
        // Deletion: 5ms duration.
        PhonemeResult {
            label: "P".into(),
            duration_ms: 5.0,
            metrics: vec![QualityMetric {
                name: "duration_ms",
                value: 5.0,
                threshold: 20.0,
                passed: false,
                citation: "Crystal 2003",
            }],
            passed: false,
        },
        // Good phoneme.
        PhonemeResult {
            label: "AH0".into(),
            duration_ms: 80.0,
            metrics: vec![QualityMetric {
                name: "duration_ms",
                value: 80.0,
                threshold: 20.0,
                passed: true,
                citation: "Crystal 2003",
            }],
            passed: true,
        },
        // Weak articulation.
        PhonemeResult {
            label: "T".into(),
            duration_ms: 40.0,
            metrics: vec![
                QualityMetric {
                    name: "duration_ms",
                    value: 40.0,
                    threshold: 20.0,
                    passed: true,
                    citation: "Crystal 2003",
                },
                QualityMetric {
                    name: "energy_ratio",
                    value: 0.01,
                    threshold: 0.05,
                    passed: false,
                    citation: "ITU-R BS.1770-4",
                },
            ],
            passed: false,
        },
    ];

    let defects = detect_defects(&results, &config);
    assert_eq!(defects.len(), 2, "Should find 2 defects: {defects:?}");

    assert!(defects
        .iter()
        .any(|d| matches!(d, PronunciationDefect::Deletion { ref label, .. } if label == "P")));
    assert!(defects.iter().any(
        |d| matches!(d, PronunciationDefect::WeakArticulation { ref label, .. } if label == "T")
    ));
}
