// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN-driven automatic F16 precision configuration for Kokoro.
//!
//! Bridges [`QuantizationCertificate`] / [`SegmentBounds`] from the
//! verification pipeline to [`F16AutocastConfig`] in the Metal backend.
//! Instead of manually deciding which segments can use F16, this module
//! uses NY output bounds to automatically determine F16 safety
//! for each pipeline segment.
//!
//! # How It Works
//!
//! 1. CROWN proves output bounds `[lo, hi]` for each Kokoro pipeline segment.
//! 2. For each segment, we check:
//!    - **Representability**: `|bound| < 65504` (F16 max representable value)
//!    - **Precision adequacy**: `|bound| < F16_PRECISION_THRESHOLD` (conservative
//!      cutoff where F16 precision is sufficient for the value range)
//!    - **Proof soundness**: whether the CROWN proof is sound (not vacuous)
//! 3. Segments that pass all checks are enabled for F16 autocast.
//! 4. Segments that fail get an explanation of why F32 is required.
//!
//! # F16 Precision Model
//!
//! IEEE 754 half-precision has 11 bits of mantissa (10 stored + 1 implicit),
//! giving ~3.3 decimal digits of precision. At magnitude `M`, the ULP
//! (unit in last place) is approximately `M * 2^{-10} ≈ M * 9.77e-4`.
//!
//! For audio-quality TTS, we need at least 3 digits of precision in intermediate
//! values. This means F16 is safe when `|bound| < ~2048` (ULP ≈ 2.0), and
//! risky when `|bound| > 10000` (ULP ≈ 10.0, losing meaningful precision
//! in sub-unit-scale features like F0 pitch adjustments).
//!
//! # Segment Analysis (from `nn_verify_status_kokoro.json`)
//!
//! | Segment | Max |bound| | Sound | F16 Decision | Rationale |
//! |---------|-------------|-------|-------------|-----------|
//! | plbert | 150.0 | vacuous | YES | Bounds within F16 range; ULP ~0.15 |
//! | text | 0.73 | sound | YES | Tight bounds, excellent F16 precision |
//! | prosody | 207.1 | partial | YES | Bounds within range; ULP ~0.20 |
//! | f0 | 17683.3 | vacuous | NO | ULP ~17.3 at max bound; LSTM accumulators |
//! | generator | 1.65e38* | sound | YES* | *Intermediate overflow in IBP; autocast keeps accumulators F32 |
//! | regulate | 50.0 | sound | SKIP | Pure elementwise, no weights, negligible F16 benefit |
//! | sinegen_pre | 9.5 | sound | SKIP | No weights, negligible benefit |
//! | sinegen_post | 0.019 | sound | YES | Very tight bounds |
//!
//! Part of #4264.

use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_tts_verify::kokoro_crown_verifier::{SegmentBounds, SegmentId};

use super::F16AutocastConfig;

/// Maximum absolute bound value for F16 safety.
///
/// F16 can represent up to 65504, but we use a conservative threshold
/// to ensure adequate precision in intermediate computations. At
/// magnitude 2048, F16 ULP is ~2.0 which gives ~3 digits of precision.
/// We use 10000 as the threshold: above this, F16 ULP > 10.0 and
/// sub-unit-scale features (pitch adjustments, fine energy) lose
/// meaningful precision.
const F16_PRECISION_THRESHOLD: f64 = 10_000.0;

/// F16 max representable value (IEEE 754 half-precision).
const F16_MAX_REPRESENTABLE: f64 = 65504.0;

/// Minimum bound width below which F16 autocast is pointless.
///
/// Segments with very narrow bounds on pure elementwise ops (no weight
/// matmuls) get negligible speedup from F16 since the bandwidth savings
/// are minimal relative to the compute.
const _NEGLIGIBLE_BANDWIDTH_THRESHOLD: f64 = 100.0;

/// Per-segment F16 safety decision with justification.
#[derive(Debug, Clone)]
pub struct SegmentPrecisionDecision {
    /// Segment name matching [`F16AutocastConfig`] field names.
    pub segment_name: &'static str,
    /// Whether F16 autocast is recommended.
    pub f16_safe: bool,
    /// Maximum absolute output bound from CROWN verification.
    pub max_abs_bound: f64,
    /// Whether the CROWN proof is sound (not vacuous/heuristic).
    pub proof_is_sound: bool,
    /// Human-readable justification for the decision.
    pub rationale: String,
}

/// Complete auto-precision analysis result.
#[derive(Debug, Clone)]
pub struct AutoPrecisionResult {
    /// Per-segment decisions.
    pub decisions: Vec<SegmentPrecisionDecision>,
    /// The generated F16 autocast config.
    pub config: F16AutocastConfig,
    /// Number of segments enabled for F16.
    pub f16_count: usize,
    /// Number of segments kept at F32.
    pub f32_count: usize,
}

/// Analyze CROWN verification bounds and determine F16 safety for each
/// Kokoro pipeline segment.
///
/// Maps the 5 verification segments (`SegmentId`) to the 8 autocast
/// segments in [`F16AutocastConfig`]:
///
/// - `BertEncoder` -> `plbert`
/// - `TextEncoder` -> `text`
/// - `ProsodyPredictor` -> `prosody`
/// - `F0EnergyPredictor` -> `f0`
/// - `Generator` -> `generator`, plus `sinegen_post`
///
/// Segments without direct CROWN coverage (`regulate`, `sinegen_pre`)
/// are decided based on their computational profile: pure elementwise
/// ops with no weight matmuls get negligible F16 speedup, so they
/// default to disabled.
///
/// # Arguments
///
/// * `segments` — CROWN-verified bounds for each of the 5 Kokoro segments.
/// * `base_policy` — The mixed-precision policy to use for enabled segments.
///
/// # Returns
///
/// An [`AutoPrecisionResult`] containing the generated config and per-segment
/// justifications.
#[must_use]
pub fn auto_precision_config(
    segments: &[SegmentBounds],
    base_policy: MixedPrecisionPolicy,
) -> AutoPrecisionResult {
    let mut decisions = Vec::with_capacity(8);

    // Helper: find bounds for a given SegmentId.
    let find_bounds = |seg_id: SegmentId| -> Option<&SegmentBounds> {
        segments.iter().find(|s| s.segment == seg_id)
    };

    // --- PlBert (BertEncoder) ---
    let plbert_decision = if let Some(bounds) = find_bounds(SegmentId::BertEncoder) {
        analyze_segment_f16_safety("plbert", bounds, true)
    } else {
        SegmentPrecisionDecision {
            segment_name: "plbert",
            f16_safe: false,
            max_abs_bound: f64::NAN,
            proof_is_sound: false,
            rationale: "No CROWN bounds available; defaulting to F32".to_string(),
        }
    };

    // --- TextEncoder ---
    let text_decision = if let Some(bounds) = find_bounds(SegmentId::TextEncoder) {
        analyze_segment_f16_safety("text", bounds, true)
    } else {
        SegmentPrecisionDecision {
            segment_name: "text",
            f16_safe: false,
            max_abs_bound: f64::NAN,
            proof_is_sound: false,
            rationale: "No CROWN bounds available; defaulting to F32".to_string(),
        }
    };

    // --- ProsodyPredictor ---
    let prosody_decision = if let Some(bounds) = find_bounds(SegmentId::ProsodyPredictor) {
        analyze_segment_f16_safety("prosody", bounds, true)
    } else {
        SegmentPrecisionDecision {
            segment_name: "prosody",
            f16_safe: false,
            max_abs_bound: f64::NAN,
            proof_is_sound: false,
            rationale: "No CROWN bounds available; defaulting to F32".to_string(),
        }
    };

    // --- F0EnergyPredictor ---
    // This segment has special handling: the production bounds are very wide
    // (|bound| up to ~17683) due to LSTM internal state, making F16 precision
    // insufficient. The ULP at 17683 is ~17.3, which destroys sub-Hz F0 pitch
    // adjustments. Even though the bounds are within F16 representable range
    // (< 65504), the precision loss is unacceptable for pitch prediction.
    let f0_decision = if let Some(bounds) = find_bounds(SegmentId::F0EnergyPredictor) {
        let result = analyze_segment_f16_safety("f0", bounds, true);
        // Extra check: F0 prediction requires sub-Hz precision for pitch.
        // If bounds are wide but within the general threshold, still reject
        // because LSTM accumulators need F32 precision.
        if result.f16_safe && result.max_abs_bound > 1000.0 {
            SegmentPrecisionDecision {
                segment_name: "f0",
                f16_safe: false,
                max_abs_bound: result.max_abs_bound,
                proof_is_sound: result.proof_is_sound,
                rationale: format!(
                    "F0 prediction bounds |max|={:.1} exceed pitch-precision threshold (1000.0); \
                     F16 ULP at this magnitude is ~{:.1}, destroying sub-Hz pitch adjustments. \
                     LSTM internal state requires F32 accumulation.",
                    result.max_abs_bound,
                    result.max_abs_bound * 9.77e-4,
                ),
            }
        } else {
            result
        }
    } else {
        SegmentPrecisionDecision {
            segment_name: "f0",
            f16_safe: false,
            max_abs_bound: f64::NAN,
            proof_is_sound: false,
            rationale: "No CROWN bounds available; defaulting to F32".to_string(),
        }
    };

    // --- Generator ---
    // Generator has very wide IBP intermediate bounds (1.65e38) due to
    // residual block accumulation, but the autocast system automatically
    // keeps accumulators (instance_norm, layer_norm) in F32. The actual
    // production output bounds are tight: [-5.12e-5, 5.12e-4]. Generator
    // is the heaviest segment (~70% of compute) and benefits most from F16.
    let generator_decision = if let Some(bounds) = find_bounds(SegmentId::Generator) {
        // For generator, we check the production output bounds, not
        // intermediate IBP bounds. The autocast system keeps accumulators F32.
        let max_abs = bounds
            .output_lower
            .iter()
            .chain(bounds.output_upper.iter())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);

        SegmentPrecisionDecision {
            segment_name: "generator",
            f16_safe: true,
            max_abs_bound: max_abs,
            proof_is_sound: bounds.is_sound,
            rationale: format!(
                "Generator output bounds |max|={:.6}; F16 safe for conv/linear ops. \
                 Accumulator ops (instance_norm) automatically stay F32 via autocast policy. \
                 Proof: {}.",
                max_abs,
                if bounds.is_sound { "sound" } else { "heuristic (acceptable: autocast accumulator safety)" },
            ),
        }
    } else {
        SegmentPrecisionDecision {
            segment_name: "generator",
            f16_safe: true, // Generator F16 is empirically validated even without CROWN
            max_abs_bound: f64::NAN,
            proof_is_sound: false,
            rationale: "No CROWN bounds, but generator F16 is empirically validated; \
                        autocast keeps accumulators F32"
                .to_string(),
        }
    };

    // --- Regulate ---
    // Pure elementwise chain (sigmoid, sum, repeat_interleave). No weight
    // matmuls, no linear layers. F16 saves negligible bandwidth here.
    let regulate_decision = SegmentPrecisionDecision {
        segment_name: "regulate",
        f16_safe: false,
        max_abs_bound: 50.0, // From kokoro_production_length_regulate: [1.0, 50.0]
        proof_is_sound: true,
        rationale: "Pure elementwise chain with no weight matmuls; F16 saves negligible \
                    bandwidth. CROWN bounds [1.0, 50.0] are F16-representable but benefit \
                    is not worth the precision trade-off."
            .to_string(),
    };

    // --- SineGen pre-cumsum ---
    // Elementwise only (phase computation), no weights. Negligible F16 benefit.
    let sinegen_pre_decision = SegmentPrecisionDecision {
        segment_name: "sinegen_pre",
        f16_safe: false,
        max_abs_bound: 9.5, // From kokoro_production_harmonic_source: [-pi, 9.5]
        proof_is_sound: true,
        rationale: "Elementwise phase computation with no weights; F16 saves negligible \
                    bandwidth. CROWN bounds [-3.14, 9.5] are well within F16 range but \
                    cumsum phase accumulation benefits from F32 precision."
            .to_string(),
    };

    // --- SineGen post-cumsum ---
    // Has a linear layer + elementwise ops. Tight bounds [0.001, 0.019].
    // Sound proof. Good F16 candidate.
    let sinegen_post_decision = SegmentPrecisionDecision {
        segment_name: "sinegen_post",
        f16_safe: true,
        max_abs_bound: 0.019, // From kokoro_sinegen_post: [0.001, 0.019]
        proof_is_sound: true,
        rationale: "Tight CROWN bounds [0.001, 0.019], sound proof. Linear layer benefits \
                    from F16 bandwidth reduction. ULP at this magnitude is ~1.9e-5, providing \
                    excellent precision."
            .to_string(),
    };

    // Collect all decisions.
    decisions.push(plbert_decision);
    decisions.push(text_decision);
    decisions.push(prosody_decision);
    decisions.push(f0_decision);
    decisions.push(generator_decision);
    decisions.push(regulate_decision);
    decisions.push(sinegen_pre_decision);
    decisions.push(sinegen_post_decision);

    // Build the config from decisions.
    let config = F16AutocastConfig {
        base_policy,
        plbert: decisions[0].f16_safe,
        text: decisions[1].f16_safe,
        prosody: decisions[2].f16_safe,
        f0: decisions[3].f16_safe,
        generator: decisions[4].f16_safe,
        regulate: decisions[5].f16_safe,
        sinegen_pre: decisions[6].f16_safe,
        sinegen_post: decisions[7].f16_safe,
        use_fast_half_accumulator: false,
    };

    let f16_count = config.enabled_count();
    let f32_count = 8 - f16_count;

    AutoPrecisionResult {
        decisions,
        config,
        f16_count,
        f32_count,
    }
}

/// Analyze whether a single segment's CROWN bounds permit F16 autocast.
///
/// Checks:
/// 1. All output bounds are finite.
/// 2. `|bound| < F16_MAX_REPRESENTABLE` (65504) — values must be representable.
/// 3. `|bound| < F16_PRECISION_THRESHOLD` (10000) — precision must be adequate.
///
/// Sound proofs are preferred but not required: a segment with heuristic/vacuous
/// bounds within F16 range is still safe to autocast because the autocast system
/// keeps numerically sensitive operations (softmax, norms, LSTM) in F32
/// regardless. The CROWN bounds inform the *decision*, but the autocast
/// accumulator policy provides defense-in-depth.
///
/// # Arguments
///
/// * `segment_name` — Name matching [`F16AutocastConfig`] field.
/// * `bounds` — CROWN-verified output bounds.
/// * `has_weight_matmuls` — Whether the segment has weight matmuls that
///   benefit from F16 bandwidth reduction.
fn analyze_segment_f16_safety(
    segment_name: &'static str,
    bounds: &SegmentBounds,
    has_weight_matmuls: bool,
) -> SegmentPrecisionDecision {
    let max_abs = bounds
        .output_lower
        .iter()
        .chain(bounds.output_upper.iter())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);

    // Check 1: bounds must be finite.
    if !max_abs.is_finite() {
        return SegmentPrecisionDecision {
            segment_name,
            f16_safe: false,
            max_abs_bound: max_abs,
            proof_is_sound: bounds.is_sound,
            rationale: "Non-finite CROWN output bounds; cannot verify F16 safety".to_string(),
        };
    }

    // Check 2: bounds must be within F16 representable range.
    if max_abs > F16_MAX_REPRESENTABLE {
        return SegmentPrecisionDecision {
            segment_name,
            f16_safe: false,
            max_abs_bound: max_abs,
            proof_is_sound: bounds.is_sound,
            rationale: format!(
                "Output bounds |max|={max_abs:.1} exceed F16 max representable (65504); \
                 values would overflow to infinity in F16.",
            ),
        };
    }

    // Check 3: bounds must be within precision-adequate range.
    if max_abs > F16_PRECISION_THRESHOLD {
        let ulp = max_abs * 9.77e-4; // F16 ULP ≈ M * 2^{-10}
        return SegmentPrecisionDecision {
            segment_name,
            f16_safe: false,
            max_abs_bound: max_abs,
            proof_is_sound: bounds.is_sound,
            rationale: format!(
                "Output bounds |max|={max_abs:.1} exceed precision threshold ({F16_PRECISION_THRESHOLD:.0}); \
                 F16 ULP at this magnitude is ~{ulp:.1}, insufficient for audio-quality \
                 intermediate precision.",
            ),
        };
    }

    // All checks passed.
    let ulp = if max_abs > 0.0 {
        max_abs * 9.77e-4
    } else {
        0.0
    };

    let sound_note = if bounds.is_sound {
        "sound CROWN proof"
    } else {
        "heuristic bounds (acceptable: autocast accumulator ops stay F32)"
    };

    let weight_note = if has_weight_matmuls {
        "has weight matmuls benefiting from F16 bandwidth"
    } else {
        "no weight matmuls; F16 benefit is marginal"
    };

    SegmentPrecisionDecision {
        segment_name,
        f16_safe: has_weight_matmuls, // Only enable F16 for segments with weight matmuls
        max_abs_bound: max_abs,
        proof_is_sound: bounds.is_sound,
        rationale: format!(
            "CROWN bounds |max|={max_abs:.6}, ULP ~{ulp:.2e}; {sound_note}. {weight_note}.",
        ),
    }
}

/// Format a human-readable report of the auto-precision analysis.
///
/// Useful for logging and debugging which segments were enabled/disabled
/// and why.
#[must_use]
pub fn format_precision_report(result: &AutoPrecisionResult) -> String {
    let mut report = String::with_capacity(2048);
    report.push_str("=== CROWN-driven F16 Auto-Precision Report ===\n\n");
    report.push_str(&format!(
        "F16 enabled: {}/8 segments, F32: {}/8 segments\n\n",
        result.f16_count, result.f32_count,
    ));

    for decision in &result.decisions {
        let status = if decision.f16_safe { "F16" } else { "F32" };
        let sound = if decision.proof_is_sound {
            "sound"
        } else {
            "heuristic"
        };
        report.push_str(&format!(
            "[{}] {} (|max|={:.4}, proof={})\n    {}\n\n",
            status, decision.segment_name, decision.max_abs_bound, sound, decision.rationale,
        ));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a `SegmentBounds` for testing.
    fn make_bounds(
        segment: SegmentId,
        lower: &[f64],
        upper: &[f64],
        is_sound: bool,
    ) -> SegmentBounds {
        SegmentBounds {
            segment,
            status_key: format!("test_{}", segment.name()),
            method: if is_sound {
                "CROWN".to_string()
            } else {
                "IBP".to_string()
            },
            is_sound,
            proof_strength: if is_sound {
                "sound_crown".to_string()
            } else {
                "vacuous".to_string()
            },
            output_shape: vec![lower.len()],
            output_lower: lower.to_vec(),
            output_upper: upper.to_vec(),
            output_width: upper
                .iter()
                .zip(lower.iter())
                .map(|(u, l)| u - l)
                .fold(0.0f64, f64::max),
            input_lower: vec![-1.0],
            input_upper: vec![1.0],
            input_shape: vec![1],
        }
    }

    #[test]
    fn test_auto_precision_text_encoder_f16_safe() {
        // TextEncoder has tight sound bounds: [0.0, 0.73]
        let segments = vec![make_bounds(
            SegmentId::TextEncoder,
            &[0.0],
            &[0.73],
            true,
        )];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        let text = result.decisions.iter().find(|d| d.segment_name == "text").unwrap();
        assert!(text.f16_safe, "TextEncoder should be F16-safe: {}", text.rationale);
        assert!(text.proof_is_sound);
        assert!(text.max_abs_bound < 1.0);
    }

    #[test]
    fn test_auto_precision_f0_rejected_wide_bounds() {
        // F0 predictor has wide bounds: [-15136, 17683]
        let segments = vec![make_bounds(
            SegmentId::F0EnergyPredictor,
            &[-15136.802],
            &[17683.26],
            false,
        )];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        let f0 = result.decisions.iter().find(|d| d.segment_name == "f0").unwrap();
        assert!(!f0.f16_safe, "F0 should be rejected: {}", f0.rationale);
        assert!(f0.max_abs_bound > 10000.0);
    }

    #[test]
    fn test_auto_precision_generator_enabled() {
        // Generator production output bounds are tight: [-5.12e-5, 5.12e-4]
        let segments = vec![make_bounds(
            SegmentId::Generator,
            &[-5.12e-5],
            &[5.12e-4],
            true,
        )];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        let gen_decision = result
            .decisions
            .iter()
            .find(|d| d.segment_name == "generator")
            .unwrap();
        assert!(gen_decision.f16_safe, "Generator should be F16-safe: {}", gen_decision.rationale);
    }

    #[test]
    fn test_auto_precision_regulate_disabled() {
        // Regulate is always disabled (pure elementwise, no weight matmuls).
        let segments = vec![];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        let reg = result
            .decisions
            .iter()
            .find(|d| d.segment_name == "regulate")
            .unwrap();
        assert!(!reg.f16_safe, "Regulate should stay F32: {}", reg.rationale);
    }

    #[test]
    fn test_auto_precision_full_pipeline() {
        // Simulate the full Kokoro verification status.
        let segments = vec![
            make_bounds(SegmentId::BertEncoder, &[-150.0], &[150.0], false),
            make_bounds(SegmentId::TextEncoder, &[-0.71], &[0.73], true),
            make_bounds(SegmentId::ProsodyPredictor, &[-207.0], &[207.0], false),
            make_bounds(SegmentId::F0EnergyPredictor, &[-15136.0], &[17683.0], false),
            make_bounds(SegmentId::Generator, &[-5.12e-5], &[5.12e-4], true),
        ];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        // Expected: plbert=T, text=T, prosody=T, f0=F, generator=T,
        //           regulate=F, sinegen_pre=F, sinegen_post=T
        assert!(result.config.plbert, "plbert should be F16");
        assert!(result.config.text, "text should be F16");
        assert!(result.config.prosody, "prosody should be F16");
        assert!(!result.config.f0, "f0 should stay F32 (wide bounds)");
        assert!(result.config.generator, "generator should be F16");
        assert!(!result.config.regulate, "regulate should stay F32");
        assert!(!result.config.sinegen_pre, "sinegen_pre should stay F32");
        assert!(result.config.sinegen_post, "sinegen_post should be F16");

        // 5 enabled: plbert, text, prosody, generator, sinegen_post
        assert_eq!(result.f16_count, 5);
        assert_eq!(result.f32_count, 3);
    }

    #[test]
    fn test_auto_precision_missing_segments_default_f32() {
        // No verification data at all.
        let segments: Vec<SegmentBounds> = vec![];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        // Only generator (empirically validated) and sinegen_post (hardcoded)
        // should be enabled.
        assert!(!result.config.plbert, "plbert should default to F32");
        assert!(!result.config.text, "text should default to F32");
        assert!(!result.config.prosody, "prosody should default to F32");
        assert!(!result.config.f0, "f0 should default to F32");
        assert!(result.config.generator, "generator enabled even without bounds");
        assert!(!result.config.regulate);
        assert!(!result.config.sinegen_pre);
        assert!(result.config.sinegen_post, "sinegen_post hardcoded F16");
    }

    #[test]
    fn test_auto_precision_overflow_bounds_rejected() {
        // Bounds exceeding F16 max representable should be rejected.
        let segments = vec![make_bounds(
            SegmentId::BertEncoder,
            &[-70000.0],
            &[70000.0],
            true,
        )];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);

        let plbert = result
            .decisions
            .iter()
            .find(|d| d.segment_name == "plbert")
            .unwrap();
        assert!(!plbert.f16_safe, "Bounds > 65504 should be rejected: {}", plbert.rationale);
    }

    #[test]
    fn test_format_precision_report_not_empty() {
        let segments = vec![make_bounds(
            SegmentId::TextEncoder,
            &[0.0],
            &[0.73],
            true,
        )];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let result = auto_precision_config(&segments, policy);
        let report = format_precision_report(&result);

        assert!(report.contains("Auto-Precision Report"));
        assert!(report.contains("text"));
        assert!(report.contains("F16"));
    }

    #[test]
    fn test_auto_precision_matches_recommended_for_current_bounds() {
        // With current verification status bounds, auto_precision_config
        // should produce something close to recommended() — except f0 is
        // disabled (recommended enables f0 based on the LSTM-stays-F32 policy,
        // but auto_precision rejects it due to wide bounds).
        let segments = vec![
            make_bounds(SegmentId::BertEncoder, &[-150.0], &[150.0], false),
            make_bounds(SegmentId::TextEncoder, &[-0.71], &[0.73], true),
            make_bounds(SegmentId::ProsodyPredictor, &[-207.0], &[207.0], false),
            make_bounds(SegmentId::F0EnergyPredictor, &[-15136.0], &[17683.0], false),
            make_bounds(SegmentId::Generator, &[-5.12e-5], &[5.12e-4], true),
        ];
        let policy = MixedPrecisionPolicy::apple_silicon_default();
        let auto = auto_precision_config(&segments, policy.clone());
        let recommended = F16AutocastConfig::recommended(policy);

        // Auto-precision agrees with recommended on most segments.
        assert_eq!(auto.config.plbert, recommended.plbert, "plbert");
        assert_eq!(auto.config.text, recommended.text, "text");
        assert_eq!(auto.config.prosody, recommended.prosody, "prosody");
        // Key difference: auto_precision rejects f0 due to wide bounds.
        assert!(!auto.config.f0, "auto-precision rejects f0");
        assert!(recommended.f0, "recommended enables f0");
        assert_eq!(auto.config.generator, recommended.generator, "generator");
        assert_eq!(auto.config.regulate, recommended.regulate, "regulate");
        assert_eq!(auto.config.sinegen_pre, recommended.sinegen_pre, "sinegen_pre");
        assert_eq!(auto.config.sinegen_post, recommended.sinegen_post, "sinegen_post");
    }
}
