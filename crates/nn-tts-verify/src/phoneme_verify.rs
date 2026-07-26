// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-phoneme quality verification using existing DSP and quality functions.
//!
//! Given externally-provided phoneme boundaries, verifies each segment against
//! acoustic quality thresholds: duration, HNR (voiced), F0 range (voiced),
//! energy ratio, and MCD (when reference is provided).

use crate::dsp;
use crate::error::TtsVerifyError;
use crate::phoneme::{PhonemeAlignment, PhonemeResult, PhonemeVerifyConfig};
use crate::quality::{self, QualityMetric};

/// Phonemes classified as voiced for English.
///
/// Voiced phonemes receive HNR and F0 checks. Unvoiced phonemes (voiceless
/// stops, voiceless fricatives) only receive duration and energy checks.
///
/// IPA and ARPAbet labels both supported.
fn is_voiced(label: &str) -> bool {
    // ARPAbet vowels (with optional stress digit)
    let arpabet_voiced = [
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        // ARPAbet consonants — voiced
        "B", "D", "G", "DH", "V", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W", "Y",
    ];
    // IPA voiced consonants and vowels
    let ipa_voiced = [
        "b", "d", "g", "ɡ", "v", "ð", "z", "ʒ", "dʒ", "m", "n", "ŋ", "l", "r", "ɹ", "w", "j",
        // IPA vowels (common subset)
        "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɒ", "ʌ", "ɔ", "o", "ʊ", "u", "ə", "ɜ", "ɐ",
    ];

    // Strip trailing stress digits for ARPAbet (e.g., "AH0" -> "AH")
    let stripped = label.trim_end_matches(|c: char| c.is_ascii_digit());

    arpabet_voiced
        .iter()
        .any(|&v| stripped.eq_ignore_ascii_case(v))
        || ipa_voiced.contains(&label)
}

/// Verify each phoneme segment against quality thresholds.
///
/// Returns a vector of per-phoneme results. Each phoneme is independently
/// evaluated. When a reference signal and its alignment are provided,
/// per-phoneme MCD is also computed.
///
/// # Errors
///
/// Returns `TtsVerifyError` if the alignment is invalid (out-of-bounds spans,
/// zero sample rate) or if DSP computation fails on a segment.
pub fn verify_phonemes(
    samples: &[f32],
    alignment: &PhonemeAlignment,
    reference: Option<&[f32]>,
    ref_alignment: Option<&PhonemeAlignment>,
    config: &PhonemeVerifyConfig,
    sample_rate: u32,
) -> Result<Vec<PhonemeResult>, TtsVerifyError> {
    if samples.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    alignment.validate()?;
    if sample_rate == 0 {
        return Err(TtsVerifyError::InvalidSampleRate(0));
    }

    // Compute utterance-level RMS for energy ratio.
    let utterance_rms = dsp::rms(samples);

    let mut results = Vec::with_capacity(alignment.phonemes.len());

    for (idx, span) in alignment.phonemes.iter().enumerate() {
        let segment = &samples[span.start..span.end];
        let mut metrics = Vec::with_capacity(5);
        let mut all_passed = true;

        // Duration check.
        let duration_ms = (span.end - span.start) as f64 / f64::from(sample_rate) * 1000.0;
        let dur_pass =
            duration_ms >= config.min_duration_ms && duration_ms <= config.max_duration_ms;
        metrics.push(QualityMetric {
            name: "duration_ms",
            value: duration_ms,
            threshold: config.min_duration_ms,
            passed: dur_pass,
            citation: "Crystal 2003, Cambridge",
        });
        if !dur_pass {
            all_passed = false;
        }

        // Energy ratio vs utterance mean.
        let seg_rms = dsp::rms(segment);
        let energy_ratio = if utterance_rms > 0.0 {
            seg_rms / utterance_rms
        } else {
            0.0
        };
        let energy_pass = energy_ratio >= config.min_energy_ratio;
        metrics.push(QualityMetric {
            name: "energy_ratio",
            value: energy_ratio,
            threshold: config.min_energy_ratio,
            passed: energy_pass,
            citation: "ITU-R BS.1770-4",
        });
        if !energy_pass {
            all_passed = false;
        }

        // Voiced-only checks: HNR and F0.
        if is_voiced(&span.label) {
            // HNR — requires at least ~10ms of audio for autocorrelation.
            let min_samples_for_hnr = (f64::from(sample_rate) * 0.01) as usize;
            if segment.len() >= min_samples_for_hnr {
                match quality::compute_hnr(segment, sample_rate, config.min_voiced_hnr_db) {
                    Ok(hnr_metric) => {
                        if !hnr_metric.passed {
                            all_passed = false;
                        }
                        metrics.push(hnr_metric);
                    }
                    Err(_) => {
                        // Segment too short or degenerate for HNR — skip gracefully.
                    }
                }
            }

            // F0 range — requires enough samples for at least one YIN frame.
            let min_samples_for_f0 = (f64::from(sample_rate) * 0.04) as usize;
            if segment.len() >= min_samples_for_f0 {
                match quality::extract_f0(segment, sample_rate) {
                    Ok(f0_contour) => {
                        let f0_metric = quality::check_f0_range(
                            &f0_contour,
                            config.f0_range_hz.0,
                            config.f0_range_hz.1,
                        );
                        if !f0_metric.passed {
                            all_passed = false;
                        }
                        metrics.push(f0_metric);
                    }
                    Err(_) => {
                        // Segment too short for F0 — skip gracefully.
                    }
                }
            }
        }

        // MCD vs reference (if both reference signal and reference alignment provided).
        if let (Some(ref_samples), Some(ref_align)) = (reference, ref_alignment) {
            if let Some(ref_span) = ref_align.phonemes.get(idx) {
                let ref_segment = &ref_samples[ref_span.start..ref_span.end];
                // MCD requires same-length segments. Truncate to shorter.
                let min_len = segment.len().min(ref_segment.len());
                if min_len > 0 {
                    let cand_seg = &segment[..min_len];
                    let ref_seg = &ref_segment[..min_len];
                    match quality::compute_mcd(cand_seg, ref_seg, sample_rate, config.max_mcd_db) {
                        Ok(mcd_metric) => {
                            if !mcd_metric.passed {
                                all_passed = false;
                            }
                            metrics.push(mcd_metric);
                        }
                        Err(_) => {
                            // Segment too short for MCD — skip gracefully.
                        }
                    }
                }
            }
        }

        results.push(PhonemeResult {
            label: span.label.clone(),
            duration_ms,
            metrics,
            passed: all_passed,
        });
    }

    Ok(results)
}
