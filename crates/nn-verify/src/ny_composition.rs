// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certified inter-model composition adapter — consumes `ny_api::composition`
//! (plus the `ny_api::materialize::VerificationBoundsSource` bridge).
//!
//! NN chains TTS pipeline stages (kokoro text → phoneme → mel → vocoder) with
//! hand-rolled bound composition today (the crate-private `compose` module).
//! This adapter is the facade-level alternative with certificates: each stage's
//! reachable outputs are certified as an in-memory
//! [`BoundCertificate`], stages are chained under an explicit
//! containment check by [`PipelineVerifier`], and the composed
//! [`PipelineCertificate`] records weakest-link provenance and merged
//! soundness across the whole chain. Because ny implements
//! `ny_api::materialize::VerificationBoundsSource` for [`BoundCertificate`], a
//! certified stage output feeds straight into the next stage's spec through
//! `SpecBuilder::try_input_source` (wrapped here as [`chain_certificate`] /
//! [`chain_certificate_into_spec`]).
//!
//! Also exposed: the linear audio mixer ([`compose_linear_mix`] +
//! [`MixerSpec`]) and the audio property checks ([`check_ducking_snr`],
//! [`check_priority_routing`], [`check_spatial_ild`]) over certified bounds.
//!
//! This adapter is additive — it does not touch NN's existing verify path or
//! its hand-rolled stage composition. It exposes thin NN-facing wrappers plus
//! flat-string reporting helpers ([`pipeline_summary`], [`property_summary`]);
//! callers that want the rich data read the re-exported
//! `ny_api::composition` types directly.
//!
//! ```rust,no_run
//! use nn_verify::ny_composition::{
//!     certify_model_bounds, chain_certificate_into_spec, require_certified,
//! };
//! # fn run(
//! #     phoneme_encoder: &ny_propagate::GraphNetwork,
//! #     text_bounds: &ny_api::BoundedTensor,
//! # ) -> ny_core::Result<()> {
//! // Stage 1: certify the phoneme encoder's reachable outputs.
//! let cert = require_certified(certify_model_bounds(
//!     "phoneme_encoder",
//!     phoneme_encoder,
//!     text_bounds,
//!     None,
//! )?)?;
//! // Stage 2: the certified outputs become the next stage's input spec.
//! let spec = chain_certificate_into_spec(&cert, vec![ny_core::Bound::new(-1.0, 1.0)])?;
//! # let _ = spec;
//! # Ok(())
//! # }
//! ```

pub use ny_api::composition::{
    check_ducking_snr, check_priority_routing, check_spatial_ild, compose_linear_mix,
    BoundCertificate, BoundCertificationResult, BoundProvenance, MixerSpec, PipelineCertificate,
    PipelineStage, PipelineVerifier, PropertyResult,
};
pub use ny_api::materialize::VerificationBoundsSource;

use ny_api::verify::{PropagationConfig, Verifier};

/// Default propagation configuration for NN bound certification.
///
/// Delegates to [`PropagationConfig::default`] (α-CROWN, CPU, McCormick
/// multiplication relaxation) — the tightest method that Packet A bound
/// certificates accept. Exposed as a named helper so NN call sites document
/// their intent rather than reaching for `Default` inline.
#[must_use]
pub fn default_certification_config() -> PropagationConfig {
    PropagationConfig::default()
}

/// Certify a model graph's reachable output bounds using a sensible default
/// configuration, producing an in-memory [`BoundCertificate`] on success.
///
/// This is bound-only certification: no output property is asserted, the
/// certificate simply records sound per-dimension output bounds for every
/// input in `input_bounds`, together with the exact method tag and soundness
/// provenance. A propagation deadline is reported as
/// [`BoundCertificationResult::Timeout`], not as an error; collapse it with
/// [`require_certified`] when a hard certificate is required.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised during propagation (e.g. a
/// structural mismatch between the graph and the input bounds), or an
/// `UnsupportedOp` if the run finishes with a method tag outside the
/// certificate surface (IBP / CROWN / α-CROWN).
pub fn certify_model_bounds(
    model_id: impl Into<String>,
    net: &ny_propagate::GraphNetwork,
    input_bounds: &ny_api::BoundedTensor,
    timeout_ms: Option<u64>,
) -> ny_core::Result<BoundCertificationResult> {
    certify_model_bounds_with_config(
        model_id,
        net,
        input_bounds,
        timeout_ms,
        default_certification_config(),
    )
}

/// Certify a model graph's reachable output bounds with an explicit
/// [`PropagationConfig`].
///
/// See [`certify_model_bounds`] for the default-config convenience wrapper and
/// the certification semantics.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised during propagation. Additionally
/// returns `UnsupportedOp` when `config` selects β-CROWN (bound certificates
/// only accept IBP / CROWN / α-CROWN method tags).
pub fn certify_model_bounds_with_config(
    model_id: impl Into<String>,
    net: &ny_propagate::GraphNetwork,
    input_bounds: &ny_api::BoundedTensor,
    timeout_ms: Option<u64>,
    config: PropagationConfig,
) -> ny_core::Result<BoundCertificationResult> {
    Verifier::new(config).certify_graph_bounds(model_id, net, input_bounds, timeout_ms)
}

/// Collapse a [`BoundCertificationResult`] into a hard [`BoundCertificate`].
///
/// NN's pipeline chaining needs a definite certificate per stage; a timed-out
/// certification cannot anchor the next stage's spec, so it is converted into
/// an error here rather than silently chaining partial bounds.
///
/// # Errors
///
/// Returns [`ny_core::NyError::DeadlineExceeded`] when the certification timed
/// out, noting the method that was running and whether partial bounds were
/// available.
pub fn require_certified(result: BoundCertificationResult) -> ny_core::Result<BoundCertificate> {
    match result {
        BoundCertificationResult::Certified(certificate) => Ok(certificate),
        BoundCertificationResult::Timeout {
            partial,
            actual_method,
            ..
        } => Err(ny_core::NyError::DeadlineExceeded(format!(
            "bound certification timed out while running {actual_method}{}",
            if partial.is_some() {
                " (partial bounds were available but are not chained)"
            } else {
                ""
            },
        ))),
    }
}

/// Open an empty [`PipelineVerifier`] for incremental stage assembly.
///
/// Push each stage's input witness + certificate with
/// [`PipelineVerifier::push_stage`], then [`PipelineVerifier::finalize`]. For
/// the one-shot list form, use [`compose_pipeline`].
#[must_use]
pub fn new_pipeline() -> PipelineVerifier {
    PipelineVerifier::new()
}

/// Compose a sequence of certified stages into one [`PipelineCertificate`].
///
/// Each element pairs the input witness that justified a stage (the
/// perturbation domain for the first stage; the verification domain containing
/// the previous certificate's outputs for later stages) with that stage's
/// [`BoundCertificate`], in pipeline order. Every transition is checked: the
/// previous stage's certified outputs must be contained in the next stage's
/// input witness, so the composed certificate is sound end-to-end. The result
/// records weakest-link provenance and merged soundness across all stages.
///
/// # Errors
///
/// Returns an [`ny_core::NyError`] if `stages` is empty, if a transition's
/// shapes mismatch, if any bound is NaN, or if a stage's certified outputs are
/// not contained in the next stage's input witness.
pub fn compose_pipeline(
    stages: Vec<(ny_api::BoundedTensor, BoundCertificate)>,
) -> ny_core::Result<PipelineCertificate> {
    let mut pipeline = new_pipeline();
    for (input_bounds, certificate) in stages {
        pipeline.push_stage(input_bounds, certificate)?;
    }
    pipeline.finalize()
}

/// Start the next stage's spec from a certified stage output.
///
/// Materializes `certificate`'s output bounds into a
/// [`ny_api::SpecBuilder`] via the `VerificationBoundsSource` bridge, so the
/// certified stage outputs become the next stage's input domain (bounds and
/// shape). The caller finishes the builder with its output property and
/// `build()`s; for the one-shot form see [`chain_certificate_into_spec`].
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] raised while materializing the
/// certificate's bounds into per-dimension [`ny_core::Bound`]s (e.g. a
/// non-finite endpoint).
pub fn chain_certificate(certificate: &BoundCertificate) -> ny_core::Result<ny_api::SpecBuilder> {
    ny_api::SpecBuilder::default().try_input_source(certificate)
}

/// Build the next stage's full [`ny_core::VerificationSpec`] from a certified
/// stage output plus the next stage's output property.
///
/// One-shot form of [`chain_certificate`]: input bounds and shape come from
/// `certificate`, `output_bounds` states the property the next stage must
/// satisfy over that certified input domain.
///
/// # Errors
///
/// Propagates any [`ny_core::NyError`] from materializing the certificate's
/// bounds, and `InvalidSpec` from the builder (e.g. empty `output_bounds`).
pub fn chain_certificate_into_spec(
    certificate: &BoundCertificate,
    output_bounds: Vec<ny_core::Bound>,
) -> ny_core::Result<ny_core::VerificationSpec> {
    chain_certificate(certificate)?
        .output_bounds(output_bounds)
        .build()
}

/// Empty [`MixerSpec`] for incremental per-voice assembly.
///
/// [`MixerSpec`] has public map fields and no constructor upstream; this named
/// helper documents the assembly pattern: insert one gain-bounds entry and one
/// `(left_pan, right_pan)` entry per voice, keyed by the exact
/// [`BoundCertificate::model_id`] of every certificate passed to
/// [`compose_linear_mix`].
#[must_use]
pub fn empty_mixer_spec() -> MixerSpec {
    MixerSpec {
        voice_gains: std::collections::HashMap::new(),
        spatial_pan: std::collections::HashMap::new(),
    }
}

/// NN-facing label for a coarse bound provenance tag.
const fn provenance_label(provenance: BoundProvenance) -> &'static str {
    match provenance {
        BoundProvenance::Ibp => "IBP",
        BoundProvenance::Crown => "CROWN",
        BoundProvenance::AlphaCrown => "alpha-CROWN",
        // `BoundProvenance` is #[non_exhaustive] upstream.
        _ => "unknown",
    }
}

/// Render an NN-friendly one-line summary of a composed pipeline certificate.
///
/// Reports the stage chain (model ids in pipeline order), the weakest-link
/// provenance, the merged soundness mode, and the final output shape. This is
/// intentionally a flat string for NN's reporting style; callers needing
/// structured data read the [`PipelineCertificate`] fields directly.
#[must_use]
pub fn pipeline_summary(certificate: &PipelineCertificate) -> String {
    let stage_count = certificate.stages().len();
    let plural = if stage_count == 1 { "" } else { "s" };
    let chain = certificate
        .stages()
        .iter()
        .map(|stage| stage.certificate().model_id())
        .collect::<Vec<_>>()
        .join(" → ");
    let provenance = provenance_label(certificate.overall_provenance());
    let soundness = match certificate.overall_soundness().mode() {
        ny_core::VerificationSoundnessMode::Sound => "sound",
        ny_core::VerificationSoundnessMode::Heuristic => "heuristic",
    };
    let shape = certificate.final_bounds().shape();
    format!(
        "pipeline: {stage_count} stage{plural} ({chain}) — provenance {provenance}, \
         {soundness}, final bounds shape {shape:?}"
    )
}

/// Render an NN-friendly one-line summary of an audio property check.
///
/// Reports the property name, the verdict, and the achieved bound against the
/// required threshold. Flat string by design; callers needing the numbers read
/// the [`PropertyResult`] fields directly.
#[must_use]
pub fn property_summary(result: &PropertyResult) -> String {
    format!(
        "{}: {} (bound {:.3} vs threshold {:.3})",
        result.name,
        if result.verified {
            "verified"
        } else {
            "unproven"
        },
        result.bound_value,
        result.threshold,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_api::verify::PropagationMethod;
    use ny_core::{Bound, MethodUsed, SoundnessProvenance};
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::{GraphNetwork, GraphNode};

    /// Tiny Linear → ReLU → Linear graph (2→2→2) for end-to-end certification.
    ///
    /// Mirrors the construction NN uses elsewhere (e.g. the `ny_ladder` tests),
    /// keeping the network small and deterministic. Hand-computed IBP bounds
    /// for input `[-1, 1]²`: linear1 `[-2, 2]²` → relu `[0, 2]²` → linear2
    /// `[0, 4]²`.
    fn linear_relu_linear_graph() -> GraphNetwork {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(ndarray::Array1::zeros(2))).expect("valid linear2");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    fn bounds_1d(lower: &[f32], upper: &[f32]) -> ny_api::BoundedTensor {
        ny_api::BoundedTensor::new(
            ndarray::Array1::from_vec(lower.to_vec()).into_dyn(),
            ndarray::Array1::from_vec(upper.to_vec()).into_dyn(),
        )
        .expect("valid test bounds")
    }

    /// The `[-1, 1]²` input box used throughout.
    fn unit_box() -> ny_api::BoundedTensor {
        bounds_1d(&[-1.0, -1.0], &[1.0, 1.0])
    }

    fn ibp_config() -> PropagationConfig {
        PropagationConfig {
            method: PropagationMethod::Ibp,
            ..PropagationConfig::default()
        }
    }

    fn certify_stage(model_id: &str, config: PropagationConfig) -> BoundCertificate {
        let graph = linear_relu_linear_graph();
        require_certified(
            certify_model_bounds_with_config(model_id, &graph, &unit_box(), None, config)
                .expect("certification should run on linear→relu→linear graph"),
        )
        .expect("certification should complete without timing out")
    }

    #[test]
    fn certified_bounds_are_sound_for_known_graph() {
        let graph = linear_relu_linear_graph();
        let cert = require_certified(
            certify_model_bounds("stage1", &graph, &unit_box(), None)
                .expect("default-config certification should run"),
        )
        .expect("no timeout expected on a 2x2 graph");

        assert_eq!(cert.model_id(), "stage1");
        assert_eq!(cert.output_bounds().shape(), &[2]);
        // Concrete reachable outputs must lie inside the certified bounds:
        // f(0, 0) = (0, 0) and f(1, -1) = (2, 2).
        for (&lower, &upper) in cert
            .output_bounds()
            .lower()
            .iter()
            .zip(cert.output_bounds().upper().iter())
        {
            assert!(lower <= 0.0, "lower {lower} must contain f(0,0) = 0");
            assert!(upper >= 2.0, "upper {upper} must contain f(1,-1) = 2");
        }
        // The coarse provenance must be exactly what the actual method maps to.
        assert_eq!(
            cert.provenance(),
            BoundProvenance::try_from(cert.actual_method())
                .expect("certified method must have a Packet A provenance"),
        );
    }

    #[test]
    fn ibp_certificate_matches_hand_computed_bounds() {
        let cert = certify_stage("stage1", ibp_config());

        assert_eq!(cert.actual_method(), &MethodUsed::Ibp);
        assert_eq!(cert.provenance(), BoundProvenance::Ibp);
        // Hand-computed IBP for input [-1,1]²: output [0, 4]².
        for (&lower, &upper) in cert
            .output_bounds()
            .lower()
            .iter()
            .zip(cert.output_bounds().upper().iter())
        {
            assert!((lower - 0.0).abs() < 1e-4, "IBP lower: got {lower}, want 0");
            assert!((upper - 4.0).abs() < 1e-4, "IBP upper: got {upper}, want 4");
        }
    }

    #[test]
    fn chained_spec_carries_certified_stage_outputs() {
        let cert = certify_stage("phoneme_encoder", ibp_config());

        let spec = chain_certificate_into_spec(
            &cert,
            vec![Bound::new(-100.0, 100.0), Bound::new(-100.0, 100.0)],
        )
        .expect("certificate should materialize into the next stage's spec");

        // The spec's input domain must be exactly the certified stage outputs.
        assert_eq!(spec.input_bounds().len(), 2);
        assert_eq!(spec.input_shape(), Some(&[2_usize][..]));
        for (bound, (&lower, &upper)) in spec.input_bounds().iter().zip(
            cert.output_bounds()
                .lower()
                .iter()
                .zip(cert.output_bounds().upper().iter()),
        ) {
            assert!((bound.lower() - lower).abs() < f32::EPSILON);
            assert!((bound.upper() - upper).abs() < f32::EPSILON);
        }

        // And the next stage really verifies against the chained spec: stage 2
        // over input [0,4]² reaches at most [0, 8]², well inside [-100, 100]².
        let stage2 = linear_relu_linear_graph();
        let result = Verifier::new(default_certification_config())
            .verify_graph(&stage2, &spec)
            .expect("stage 2 verification should run on the chained spec");
        assert!(
            result.is_verified(),
            "wide chained spec should verify, got {result:?}"
        );
    }

    #[test]
    fn compose_pipeline_produces_weakest_link_certificate() {
        // Stage 1 certified with IBP (weakest), stage 2 with the default
        // (α-CROWN-capable) config, chained on stage 1's certified outputs.
        let cert1 = certify_stage("phoneme_encoder", ibp_config());
        let witness = cert1.output_bounds().clone();
        let stage2_graph = linear_relu_linear_graph();
        let cert2 = require_certified(
            certify_model_bounds("vocoder", &stage2_graph, &witness, None)
                .expect("stage 2 certification should run"),
        )
        .expect("stage 2 should not time out");

        let pipeline = compose_pipeline(vec![(unit_box(), cert1), (witness, cert2.clone())])
            .expect("contained chain should compose");

        assert_eq!(pipeline.stages().len(), 2);
        assert_eq!(
            pipeline.stages()[0].certificate().model_id(),
            "phoneme_encoder"
        );
        assert_eq!(pipeline.stages()[1].certificate().model_id(), "vocoder");
        // Weakest link: stage 1 is IBP.
        assert_eq!(pipeline.overall_provenance(), BoundProvenance::Ibp);
        // Final bounds are the last stage's certified outputs.
        assert_eq!(
            pipeline.final_bounds().lower(),
            cert2.output_bounds().lower()
        );
        assert_eq!(
            pipeline.final_bounds().upper(),
            cert2.output_bounds().upper()
        );

        let summary = pipeline_summary(&pipeline);
        assert!(summary.contains("2 stages"), "summary: {summary}");
        assert!(summary.contains("phoneme_encoder"), "summary: {summary}");
        assert!(summary.contains("vocoder"), "summary: {summary}");
        assert!(summary.contains("IBP"), "summary: {summary}");
    }

    #[test]
    fn compose_pipeline_rejects_uncontained_transition() {
        // Stage 1's certified outputs are [0, 4]²; a stage 2 witness of
        // [0, 1]² does not contain them, so the chain must be refused.
        let cert1 = certify_stage("phoneme_encoder", ibp_config());
        let narrow_witness = bounds_1d(&[0.0, 0.0], &[1.0, 1.0]);
        let stage2_graph = linear_relu_linear_graph();
        let cert2 = require_certified(
            certify_model_bounds_with_config(
                "vocoder",
                &stage2_graph,
                &narrow_witness,
                None,
                ibp_config(),
            )
            .expect("stage 2 certification should run"),
        )
        .expect("stage 2 should not time out");

        let error = compose_pipeline(vec![(unit_box(), cert1), (narrow_witness, cert2)])
            .expect_err("uncontained transition must be rejected");
        assert!(
            error.to_string().contains("not contained"),
            "unexpected chain error: {error}"
        );
    }

    #[test]
    fn require_certified_rejects_timeout() {
        let timeout = BoundCertificationResult::Timeout {
            partial: None,
            actual_method: MethodUsed::Ibp,
            soundness: SoundnessProvenance::sound(),
        };
        let error = require_certified(timeout).expect_err("timeout must not chain");
        match error {
            ny_core::NyError::DeadlineExceeded(message) => {
                assert!(
                    message.contains("Ibp"),
                    "timeout error must name the running method: {message}"
                );
            }
            other => panic!("expected DeadlineExceeded, got {other}"),
        }
    }

    #[test]
    fn mixer_composes_two_certified_voices() {
        // Known-value case: voice A output [1, 3] with gain [0.8, 1.0] and pan
        // (0.7, 0.3); voice B output [0.5, 2] with gain [0.8, 1.0] and pan
        // (0.3, 0.7). Left ear: A contributes [0.56, 2.1], B [0.12, 0.6].
        let cert_a = BoundCertificate::try_new(
            "voice_a",
            bounds_1d(&[1.0], &[3.0]),
            MethodUsed::Crown,
            SoundnessProvenance::sound(),
        )
        .expect("Crown is a supported certificate method");
        let cert_b = BoundCertificate::try_new(
            "voice_b",
            bounds_1d(&[0.5], &[2.0]),
            MethodUsed::Ibp,
            SoundnessProvenance::sound(),
        )
        .expect("Ibp is a supported certificate method");

        let mut spec = empty_mixer_spec();
        spec.voice_gains
            .insert("voice_a".to_string(), bounds_1d(&[0.8], &[1.0]));
        spec.voice_gains
            .insert("voice_b".to_string(), bounds_1d(&[0.8], &[1.0]));
        spec.spatial_pan.insert("voice_a".to_string(), (0.7, 0.3));
        spec.spatial_pan.insert("voice_b".to_string(), (0.3, 0.7));

        let (left, right) = compose_linear_mix(&[cert_a, cert_b], &spec)
            .expect("mixer should compose two matching voices");

        let left_lower = left.lower().iter().next().copied().expect("left lower");
        let left_upper = left.upper().iter().next().copied().expect("left upper");
        assert!(
            (left_lower - 0.68).abs() < 1e-5,
            "left lower: got {left_lower}, want 0.68"
        );
        assert!(
            (left_upper - 2.7).abs() < 1e-5,
            "left upper: got {left_upper}, want 2.7"
        );
        // The right ear mirrors the pans, so its interval is also non-trivial.
        let right_lower = right.lower().iter().next().copied().expect("right lower");
        let right_upper = right.upper().iter().next().copied().expect("right upper");
        assert!(right_lower < right_upper, "right interval must be proper");
    }

    #[test]
    fn property_checks_report_verified_audio_properties() {
        // Priority routing: lead gain [0.8, 1.0] strictly above backing [0.1, 0.3].
        let lead = bounds_1d(&[0.8], &[1.0]);
        let backing = bounds_1d(&[0.1], &[0.3]);
        let routing = check_priority_routing(&lead, &[&backing]);
        assert!(routing.verified, "0.8 > 0.3 must verify priority routing");
        assert!(
            (routing.bound_value - 0.5).abs() < 1e-6,
            "margin: got {}, want 0.5",
            routing.bound_value
        );

        // Ducking SNR: lead abs-min 0.5 over background abs-max 0.05 is 20 dB.
        let lead_signal = bounds_1d(&[0.5], &[1.0]);
        let background = bounds_1d(&[-0.05], &[0.05]);
        let ducking = check_ducking_snr(&lead_signal, &background, 12.0);
        assert!(ducking.verified, "20 dB must clear a 12 dB threshold");
        assert!(
            (ducking.bound_value - 20.0).abs() < 1e-3,
            "snr: got {}, want 20",
            ducking.bound_value
        );
        let summary = property_summary(&ducking);
        assert!(summary.contains("ducking_snr"), "summary: {summary}");
        assert!(summary.contains("verified"), "summary: {summary}");

        // Spatial ILD: hard-panned voices with power [0.8, 1.0] guarantee
        // 20*log10(0.8/0.1) ≈ 18.06 dB at each ear.
        let voice_i = bounds_1d(&[0.8], &[1.0]);
        let voice_j = bounds_1d(&[0.8], &[1.0]);
        let ild = check_spatial_ild(&voice_i, &voice_j, (1.0, 0.1), (0.1, 1.0), 12.0);
        assert!(ild.verified, "18 dB must clear a 12 dB threshold");
        assert!(
            (ild.bound_value - 20.0 * (8.0_f64).log10()).abs() < 1e-3,
            "ild: got {}",
            ild.bound_value
        );

        // An unproven check must summarize as unproven.
        let failed = check_ducking_snr(&lead_signal, &background, 40.0);
        assert!(!failed.verified);
        let failed_summary = property_summary(&failed);
        assert!(
            failed_summary.contains("unproven"),
            "summary: {failed_summary}"
        );
    }
}
