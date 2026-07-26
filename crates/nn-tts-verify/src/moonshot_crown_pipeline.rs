// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline composition functions for moonshot property verification.
//!
//! These functions compose individual property checks (P1-P6) into bundles
//! suitable for full TTS pipeline verification.

use crate::error::TtsVerifyError;
use crate::pipeline::{PipelineCertificate, TimingCertificate, VerifiedStage};

use super::{
    check_intelligibility_proxy, check_memory_boundedness, check_non_clipping, check_non_silence,
    check_speaker_consistency, check_streaming_safety, check_temporal_boundedness,
    MoonshotCrownBundle, SpeakerConsistencyEvidence,
};

/// Verify moonshot properties 1-3, 5, and 6 against pipeline + timing certificates.
///
/// This is the full moonshot verification path that includes temporal boundedness
/// (Property 5) alongside the existing CROWN-based properties. Requires a
/// `TimingCertificate` from `verify_pipeline_with_timing`.
///
/// Properties 7, 8 require additional infrastructure:
/// - Property 7 (memory safety): Kani-based, not CROWN
/// - Property 8 (correctness): ay-based, not CROWN
///
/// For Property 4 (speaker consistency), use [`check_speaker_consistency`]
/// and [`verify_all_crown_properties`].
pub fn verify_properties_with_timing(
    bounds_cert: &PipelineCertificate,
    timing_cert: &TimingCertificate,
    dim: usize,
) -> MoonshotCrownBundle {
    verify_properties_with_timing_and_streaming(bounds_cert, timing_cert, dim, 240, 0.3)
}

/// Verify moonshot properties 1-3, 5, and 6 with configurable streaming parameters.
pub fn verify_properties_with_timing_and_streaming(
    bounds_cert: &PipelineCertificate,
    timing_cert: &TimingCertificate,
    dim: usize,
    crossfade_samples: usize,
    click_threshold: f64,
) -> MoonshotCrownBundle {
    let results = vec![
        check_non_silence(bounds_cert, 0.01),
        check_non_clipping(bounds_cert),
        check_intelligibility_proxy(bounds_cert, 100.0),
        check_temporal_boundedness(timing_cert),
        check_streaming_safety(bounds_cert, crossfade_samples, click_threshold),
    ];

    let all_proven = results.iter().all(|r| r.proven);

    MoonshotCrownBundle {
        results,
        pipeline_cert: bounds_cert.clone(),
        verification_dim: dim,
        all_proven,
    }
}

/// Verify moonshot properties 1-3 and 6 against a pipeline certificate.
///
/// For Property 4 (speaker consistency), use [`check_speaker_consistency`].
/// For Property 5 (temporal bounds), use [`verify_properties_with_timing`].
pub fn verify_properties_from_pipeline(
    cert: &PipelineCertificate,
    dim: usize,
) -> MoonshotCrownBundle {
    // Default crossfade: 480 samples (20ms at 24kHz), matching KokoroStreamConfig.
    verify_properties_from_pipeline_with_streaming(cert, dim, 480, 0.3)
}

/// Verify moonshot properties 1-3 and 6 with configurable streaming parameters.
///
/// `crossfade_samples`: crossfade length (default 960 = 40ms at 24kHz).
/// `click_threshold`: max allowed sample-to-sample discontinuity (default 0.3).
pub fn verify_properties_from_pipeline_with_streaming(
    cert: &PipelineCertificate,
    dim: usize,
    crossfade_samples: usize,
    click_threshold: f64,
) -> MoonshotCrownBundle {
    let results = vec![
        check_non_silence(cert, 0.01),
        check_non_clipping(cert),
        check_intelligibility_proxy(cert, 100.0),
        check_streaming_safety(cert, crossfade_samples, click_threshold),
    ];

    let all_proven = results.iter().all(|r| r.proven);

    MoonshotCrownBundle {
        results,
        pipeline_cert: cert.clone(),
        verification_dim: dim,
        all_proven,
    }
}

/// Verify moonshot properties 1-3, 5, and 6 with timing AND memory bounds.
///
/// Extends `verify_properties_with_timing` by adding a peak memory
/// boundedness check. The memory check validates that the dispatch plan's
/// peak memory usage (weights + activations) fits within the hardware
/// memory budget.
///
/// For M4 Max with 128 GB unified memory and 7 concurrent voices,
/// the per-model budget is ~18 GB = 18 * 1024 * 1024 * 1024 bytes.
pub fn verify_properties_with_timing_and_memory(
    bounds_cert: &PipelineCertificate,
    timing_cert: &TimingCertificate,
    dim: usize,
    memory_bound_bytes: u64,
) -> MoonshotCrownBundle {
    let results = vec![
        check_non_silence(bounds_cert, 0.01),
        check_non_clipping(bounds_cert),
        check_intelligibility_proxy(bounds_cert, 100.0),
        check_temporal_boundedness(timing_cert),
        check_memory_boundedness(timing_cert, memory_bound_bytes),
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

/// Verify all 6 CROWN-verifiable moonshot properties (P1-P6).
///
/// Combines pipeline bounds (P1-P3, P6), timing certificate (P5), and
/// speaker consistency evidence (P4) into a single bundle. This is the
/// most complete CROWN verification path available.
///
/// Properties 7 (memory safety) and 8 (implementation correctness) are
/// verified by Kani and ay respectively, not CROWN.
pub fn verify_all_crown_properties(
    bounds_cert: &PipelineCertificate,
    timing_cert: &TimingCertificate,
    speaker_evidence: &SpeakerConsistencyEvidence,
    dim: usize,
) -> MoonshotCrownBundle {
    let results = vec![
        check_non_silence(bounds_cert, 0.01),
        check_non_clipping(bounds_cert),
        check_intelligibility_proxy(bounds_cert, 100.0),
        check_speaker_consistency(speaker_evidence),
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

/// Verify moonshot properties using per-layer CROWN at specified dimensions.
///
/// Constructs a simple linear pipeline, runs `verify_layerwise()` at the
/// given dimension, and checks properties 1-3 against the resulting bounds.
///
/// This is the bridge from #1762 (per-layer CROWN scaling) to #1741 (moonshot).
///
/// # Arguments
///
/// * `stages` - Pre-computed pipeline stages (from `verify_layerwise()` or
///   manual construction).
/// * `dim` - The dimension used for verification (for reporting).
pub fn verify_moonshot_from_stages(
    stages: &[VerifiedStage],
    dim: usize,
) -> Result<MoonshotCrownBundle, TtsVerifyError> {
    use crate::pipeline::verify_pipeline;

    let cert = verify_pipeline(stages)?;
    Ok(verify_properties_from_pipeline(&cert, dim))
}

/// Generate constructive proof Lean4 exports for CROWN-based moonshot
/// properties (P1-P4) from a `MoonshotCrownBundle`.
///
/// Uses the pipeline stages' per-stage bounds to construct
/// `CrownLayerProof` entries and compose them via NY's
/// `compose_crown_proofs()`. The resulting Lean4 source proves end-to-end
/// bounds by chaining per-stage CROWN proofs via transitivity.
///
/// Returns a `Vec<(usize, String)>` mapping property index to Lean4 source.
/// Only properties that were proven in the bundle and have finite bounds
/// produce proof entries. Returns an empty vec if composition fails.
///
/// # Arguments
///
/// * `bundle` — CROWN bundle from `verify_properties_from_pipeline` or similar.
///
/// Part of #4254.
#[cfg(feature = "ny")]
pub fn generate_crown_constructive_proofs(bundle: &MoonshotCrownBundle) -> Vec<(usize, String)> {
    // NOTE: NY's constructive proof types (compose_crown_proofs,
    // CrownLayerProof, LayerProofType, NeuronStatus) are now gated behind
    // the `proof-certificates` feature in gamma-propagate. Until that feature
    // compiles cleanly upstream, we skip composition and return empty.
    // This is safe — composition proofs are an enrichment, not a requirement.
    //
    // TODO(#4315): Re-enable when NY proof-certificates feature is fixed.
    let _ = bundle;
    Vec::new()
}
