// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Property 3 (intelligibility) upgrade via attention monotonicity.
//!
//! When an [`AttentionMonotonicityCertificate`] with `is_proven == true` is
//! available, Property 3 is upgraded from the range-ratio proxy
//! ([`check_intelligibility_proxy`]) to a real diagonal-dominance proof.
//!
//! ## Verification Level Assignment
//!
//! | Certificate state        | Pipeline soundness | P3 Level      |
//! |--------------------------|--------------------|---------------|
//! | `is_proven && mode in CROWN-family` | any      | CrownProven   |
//! | `is_proven && mode not in CROWN-family` | any  | CrownPartial  |
//! | `!is_proven`               | sound            | CrownPartial  |
//! | `!is_proven`               | !sound           | Empirical     |
//!
//! This is strictly stronger than the proxy check, which caps at CrownPartial
//! even when the pipeline certificate is sound.
//!
//! Part of #1741 — THE MOONSHOT: First Provably Correct Voice.

use super::{
    check_intelligibility_proxy, check_non_clipping, check_non_silence, check_streaming_safety,
    check_temporal_boundedness, MoonshotCrownBundle, MoonshotPropertyResult, PROPERTY_NAMES,
};
use crate::monotonicity::{
    max_provable_input_bound, AttentionMonotonicityCertificate, WeightMagnitudeCertificate,
};
use crate::moonshot::VerificationLevel;
use crate::pipeline::{PipelineCertificate, TimingCertificate};

use super::SpeakerConsistencyEvidence;

/// Check Property 3 (intelligibility) using a real attention monotonicity
/// certificate instead of the range-ratio proxy.
///
/// When the certificate proves diagonal dominance (`is_proven == true`),
/// P3 achieves CrownProven when the certificate came from the tight CROWN
/// family (CROWN, AlphaCrown, BetaCrown) or CrownPartial if the evidence is
/// only IBP fallback. This is the first pathway to CrownProven for P3 — the
/// proxy check caps at CrownPartial by design.
///
/// When the certificate does NOT prove diagonal dominance, falls back to
/// the range-ratio proxy for the pipeline certificate.
pub fn check_intelligibility_with_monotonicity(
    pipeline_cert: &PipelineCertificate,
    attn_cert: &AttentionMonotonicityCertificate,
) -> MoonshotPropertyResult {
    if attn_cert.is_proven {
        // Real diagonal dominance proof — upgrade P3 level.
        let is_crown = attn_cert.is_sound_crown_family();

        let level = if is_crown {
            VerificationLevel::CrownProven
        } else {
            VerificationLevel::CrownPartial
        };

        MoonshotPropertyResult {
            property_index: 2,
            property_name: PROPERTY_NAMES[2],
            proven: true,
            level,
            bound_value: attn_cert.min_margin,
            threshold: 0.0, // margin > 0 suffices
            is_sound: is_crown,
            explanation: format!(
                "attention diagonal dominance: min_margin={:.6}, \
                 decoder_steps={}, encoder_positions={}, mode={}: PROVEN",
                attn_cert.min_margin,
                attn_cert.decoder_steps,
                attn_cert.encoder_positions,
                attn_cert.propagation_mode,
            ),
        }
    } else {
        // Certificate did not prove monotonicity — fall back to proxy.
        check_intelligibility_proxy(pipeline_cert, 100.0)
    }
}

/// Check Property 3 (intelligibility) with both attention monotonicity and
/// weight magnitude evidence.
///
/// Extends [`check_intelligibility_with_monotonicity`] by incorporating weight
/// magnitude validation from Phase 30. The explanation includes:
/// - The max provable input perturbation radius (`max_provable_ib`)
/// - Whether the model's weights are within the assumed magnitude bound
/// - The worst-case Xavier-normalized magnitude across layers
///
/// The verification level is unchanged from `check_intelligibility_with_monotonicity`
/// — weight evidence enriches diagnostics but does not downgrade a proven result.
/// However, when the attention certificate is NOT proven and weights are within
/// bound, the explanation notes that IBP provability is architecturally feasible.
///
/// # Arguments
///
/// * `pipeline_cert` — pipeline bounds certificate
/// * `attn_cert` — attention monotonicity certificate
/// * `weight_cert` — weight magnitude validation certificate
/// * `pe_margin` — positional encoding margin budget (from Phase 44 analysis)
pub fn check_intelligibility_with_weight_evidence(
    pipeline_cert: &PipelineCertificate,
    attn_cert: &AttentionMonotonicityCertificate,
    weight_cert: &WeightMagnitudeCertificate,
    pe_margin: f64,
) -> MoonshotPropertyResult {
    let mut result = check_intelligibility_with_monotonicity(pipeline_cert, attn_cert);

    let max_ib = max_provable_input_bound(weight_cert, pe_margin);

    let weight_summary = if weight_cert.all_within_bound {
        format!(
            " weight_check=PASS ({} layers within mag_bound={:.6}, \
             max_normalized={:.4}, max_provable_ib={:.4})",
            weight_cert.per_layer_max_abs.len(),
            weight_cert.magnitude_bound,
            weight_cert.max_normalized_magnitude,
            max_ib,
        )
    } else {
        format!(
            " weight_check=FAIL ({}/{} layers exceed mag_bound={:.6}, \
             max_normalized={:.4}, max_provable_ib={:.4})",
            weight_cert.violating_layers,
            weight_cert.per_layer_max_abs.len(),
            weight_cert.magnitude_bound,
            weight_cert.max_normalized_magnitude,
            max_ib,
        )
    };

    result.explanation.push_str(&weight_summary);

    if !attn_cert.is_proven && weight_cert.all_within_bound {
        result
            .explanation
            .push_str(" (weight assumptions satisfied — IBP provability architecturally feasible)");
    }

    result
}

/// Verify all 6 CROWN-verifiable moonshot properties (P1-P6) with
/// attention monotonicity evidence for P3.
///
/// Extends [`verify_all_crown_properties`](super::verify_all_crown_properties)
/// by accepting an optional [`AttentionMonotonicityCertificate`]. When
/// provided and proven, P3 upgrades from proxy/CrownPartial to CrownProven.
///
/// # Arguments
///
/// * `bounds_cert` — pipeline bounds certificate (P1, P2, P6)
/// * `timing_cert` — timing certificate (P5)
/// * `speaker_evidence` — ECAPA-TDNN speaker embedding bounds (P4)
/// * `attn_cert` — optional attention monotonicity certificate (P3 upgrade)
/// * `dim` — verification dimension (for reporting)
pub fn verify_all_crown_properties_with_attention(
    bounds_cert: &PipelineCertificate,
    timing_cert: &TimingCertificate,
    speaker_evidence: &SpeakerConsistencyEvidence,
    attn_cert: Option<&AttentionMonotonicityCertificate>,
    dim: usize,
) -> MoonshotCrownBundle {
    let p3 = match attn_cert {
        Some(cert) => check_intelligibility_with_monotonicity(bounds_cert, cert),
        None => check_intelligibility_proxy(bounds_cert, 100.0),
    };

    let results = vec![
        check_non_silence(bounds_cert, 0.01),
        check_non_clipping(bounds_cert),
        p3,
        super::check_speaker_consistency(speaker_evidence),
        check_temporal_boundedness(timing_cert),
        check_streaming_safety(bounds_cert, 240, 0.3),
    ];

    let all_proven = results.iter().all(|r| r.proven);

    MoonshotCrownBundle {
        results,
        pipeline_cert: bounds_cert.clone(),
        verification_dim: dim,
        all_proven,
    }
}

/// Verify all 6 CROWN-verifiable moonshot properties (P1-P6) with
/// both attention monotonicity and weight magnitude evidence for P3.
///
/// Extends [`verify_all_crown_properties_with_attention`] by also accepting
/// a [`WeightMagnitudeCertificate`]. When both attention and weight certs are
/// provided, P3 uses the enriched path that reports weight provability analysis.
///
/// # Arguments
///
/// * `bounds_cert` — pipeline bounds certificate (P1, P2, P6)
/// * `timing_cert` — timing certificate (P5)
/// * `speaker_evidence` — ECAPA-TDNN speaker embedding bounds (P4)
/// * `attn_cert` — optional attention monotonicity certificate (P3 upgrade)
/// * `weight_cert` — optional weight magnitude certificate (P3 enrichment)
/// * `pe_margin` — positional encoding margin budget for weight analysis
/// * `dim` — verification dimension (for reporting)
pub fn verify_all_crown_properties_with_evidence(
    bounds_cert: &PipelineCertificate,
    timing_cert: &TimingCertificate,
    speaker_evidence: &SpeakerConsistencyEvidence,
    attn_cert: Option<&AttentionMonotonicityCertificate>,
    weight_cert: Option<&WeightMagnitudeCertificate>,
    pe_margin: f64,
    dim: usize,
) -> MoonshotCrownBundle {
    let p3 = match (attn_cert, weight_cert) {
        (Some(ac), Some(wc)) => {
            check_intelligibility_with_weight_evidence(bounds_cert, ac, wc, pe_margin)
        }
        (Some(ac), None) => check_intelligibility_with_monotonicity(bounds_cert, ac),
        _ => check_intelligibility_proxy(bounds_cert, 100.0),
    };

    let results = vec![
        check_non_silence(bounds_cert, 0.01),
        check_non_clipping(bounds_cert),
        p3,
        super::check_speaker_consistency(speaker_evidence),
        check_temporal_boundedness(timing_cert),
        check_streaming_safety(bounds_cert, 240, 0.3),
    ];

    let all_proven = results.iter().all(|r| r.proven);

    MoonshotCrownBundle {
        results,
        pipeline_cert: bounds_cert.clone(),
        verification_dim: dim,
        all_proven,
    }
}
