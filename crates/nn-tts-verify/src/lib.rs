// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio-domain quality verification for TTS/SVS pipelines.
//!
//! Provides hard bounds ("isn't broken") and quality metrics ("sounds good")
//! for PCM audio output from text-to-speech models. Covers audio quality,
//! phoneme correctness, Unicode safety, adversarial robustness, fairness,
//! cost modeling, pipeline composition, and moonshot certification.
//!
//! # Module Categories
//!
//! **Core framework** — [`bounds`], [`certificate`], [`quality`],
//! [`error`], [`stats`]. The [`TtsVerifier`] builder drives hard-bound and
//! quality-metric checks on PCM audio.
//!
//! **Audio quality metrics** — [`pesq`] (PESQ score), [`stoi`] (STOI score),
//! [`multi_res_stft`] (multi-resolution STFT loss), [`f0_contour`] (F0
//! correlation), [`dsp`] (signal processing helpers).
//!
//! **Phoneme verification** — [`phoneme`] (alignment), [`phoneme_verify`]
//! (verification driver), [`phoneme_defects`] (pronunciation defect detection),
//! [`phoneme_crown`] (NY energy/F0 range analysis).
//!
//! **Unicode safety** — [`unicode_safety`] (confusable scanning),
//! [`unicode_perturbation`] (coverage and vulnerability analysis),
//! [`unicode_certificate`] (safety certificates),
//! [`pronunciation_unicode`] (pronunciation-Unicode intersection).
//!
//! **Adversarial & fairness** — [`adversarial`] (confusion set discovery),
//! [`fairness`] / [`fairness_diagnosis`] (group fairness measurement),
//! [`audio_disentanglement`] / [`codec_algebra`] (embedding-space analysis).
//! With `NY` feature: `adversarial_robustness`, `fairness_crown`,
//! `disentanglement`, `crown`.
//!
//! **Cost modeling & dispatch** — [`cost_model`] / [`cost_propagation`]
//! (hardware cost estimation and coupled timing), [`kokoro_dispatch`] /
//! [`kokoro_encoder_dispatch`] (Kokoro dispatch plan builders),
//! [`silero_vad_dispatch`] (Silero VAD dispatch plan).
//!
//! **Pipeline & streaming** — [`pipeline`] (multi-stage pipeline verification
//! with junction and timing checks), [`streaming`] (streaming boundary
//! crossfade verification), [`deterministic`] (deterministic run comparison),
//! [`monotonicity`] (duration positivity / weight magnitude certificates),
//! [`quality_bound`] (Lipschitz-based quality bound verification).
//!
//! **Moonshot certification** — [`moonshot`] (certificate system with artifact
//! registry and verification levels), [`moonshot_crown`] (CROWN-based property
//! verification: non-clipping, non-silence, intelligibility, speaker
//! consistency, temporal boundedness, memory boundedness, streaming safety).
//!
//! **Singing & curriculum** — [`singing`] (pitch accuracy, timing, vibrato for
//! SVS), [`curriculum`] (corpus analysis and utterance selection).
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_tts_verify::{TtsVerifier, HardBoundsConfig};
//!
//! let verifier = TtsVerifier::builder()
//!     .sample_rate(24000)
//!     .build()?;
//!
//! // Verify standalone output (hard bounds only).
//! let cert = verifier.verify(&pcm_samples)?;
//! assert!(cert.passes_hard_bounds());
//!
//! // Verify with reference (hard bounds + quality metrics).
//! let cert = verifier.verify_with_reference(&candidate, &reference)?;
//! assert!(cert.overall_passed);
//! ```

pub mod adversarial;
pub mod audio_disentanglement;
pub mod bounds;
pub mod certificate;
pub mod codec_algebra;
mod config;
pub mod cost_model;
pub mod cost_propagation;
pub mod crown_junction;
pub mod crown_synthesis;
pub mod curriculum;
pub mod deterministic;
pub(crate) mod dispatch_builder;
pub mod dsp;
pub mod error;
pub mod f0_contour;
pub mod fairness;
pub mod fairness_diagnosis;
pub mod kokoro_contracts;
pub mod kokoro_crown_certificate;
pub mod kokoro_crown_verifier;
pub mod kokoro_dispatch;
pub mod kokoro_encoder_dispatch;
pub mod monotonicity;
pub mod moonshot;
pub mod moonshot_crown;
pub mod multi_res_stft;
pub mod pesq;
pub mod phoneme;
pub mod phoneme_crown;
pub mod phoneme_defects;
pub mod phoneme_verify;
pub mod pipeline;
pub mod pronunciation_unicode;
pub mod quality;
pub mod quality_bound;
pub mod quantization_certificate;
pub mod silero_vad_dispatch;
pub mod singing;
pub mod stats;
pub mod stoi;
pub mod streaming;
pub mod unicode_certificate;
pub mod unicode_perturbation;
pub mod unicode_safety;
mod verifier;

#[cfg(feature = "ny")]
pub mod adversarial_robustness;

#[cfg(feature = "ny")]
pub mod crown;

#[cfg(feature = "ny")]
pub mod disentanglement;

#[cfg(feature = "ny")]
pub mod fairness_crown;

pub use adversarial::{
    discover_confusion_sets, embedding_bounds_for_token_set, english_confusion_sets,
    sequence_perturbation_bounds, ConfusionCategory, ConfusionSet,
};
pub use audio_disentanglement::{
    classify_disentanglement, measure_audio_disentanglement, AudioDisentanglementResult,
    DisentanglementEvidence, DisentanglementThresholds,
};
pub use bounds::{HardBound, SpectralCoverageConfig};
pub use certificate::Certificate;
pub use codec_algebra::{
    emotion_centroid, speaker_centroid, utterance_centroid, CodecEmbeddingSpace,
};
pub use config::{CheckOverrides, HardBoundsConfig, QualityConfig, RejectionPolicy};
pub use cost_model::{
    calibrate_profiles, estimate_peak_memory, fill_measured, profile_dispatch_plan, step_flops,
    step_memory_bytes, step_output_bytes, step_weight_bytes, total_estimated_time_us, total_flops,
    total_memory_bytes, CalibrationReport, HardwareCostModel, LayerCostProfile, Measurement,
    PeakMemoryProfile, StepCalibration,
};
pub use cost_propagation::{CoupledLayerResult, CoupledTimingCertificate};
pub use crown_junction::{
    check_all_junction_contracts, check_junction_bound, contract_bounds_map,
    verify_crown_with_junction_checks, JunctionCheckSummary, StageBoundCheck,
};
pub use crown_synthesis::{
    verify_synthesis_crown, verify_synthesis_crown_full, CrownCertificateConfig,
    CrownSynthesisResult,
};
pub use curriculum::{
    analyze_corpus, select_curriculum, CorpusAnalysis, CurriculumConfig, UtteranceAnalysis,
};
pub use deterministic::{DeterministicCert, DeterministicMeta};
pub use error::{CodecAlgebraKind, DspErrorKind, InvalidConfigKind, TtsVerifyError};
pub use f0_contour::compute_f0_contour_correlation;
pub use fairness::{
    measure_fairness, FairnessConfig, FairnessReport, Group, GroupStats, MetricStat,
    PairwiseComparison, TaggedSample,
};
pub use kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, VerifiedJunctionContract,
};
pub use kokoro_crown_certificate::{
    CertificateError as KokoroCertificateError, JunctionContractEntry, KokoroCrownCertificate,
    PropertyCrownEntry, KOKORO_CERTIFICATE_VERSION,
};
pub use kokoro_crown_verifier::{
    KokoroCrownVerifier, SegmentBounds, SegmentId, SegmentVerifyResult, VerifierError,
    VerifyAllResult,
};
pub use kokoro_dispatch::{build_kokoro_dispatch_plan, build_kokoro_dispatch_plan_default};
pub use kokoro_encoder_dispatch::{
    build_kokoro_encoder_dispatch_plan, build_kokoro_encoder_dispatch_plan_default,
};
pub use monotonicity::{
    interpret_duration_positivity, max_provable_input_bound, validate_weight_magnitudes,
    DurationPositivityCertificate, WeightMagnitudeCertificate,
};
pub use moonshot::{
    artifact_registry, CertificateDeserializeError, KaniVerificationEvidence, MoonshotCertificate,
    MoonshotStatus, PropertyCertificate, PropertyStatus, SmtVerificationEvidence,
    VerificationArtifact, VerificationLevel,
};
pub use moonshot_crown::{
    analyze_dispatch_plan, check_implementation_correctness, check_intelligibility_proxy,
    check_intelligibility_with_monotonicity, check_intelligibility_with_weight_evidence,
    check_memory_boundedness, check_non_clipping, check_non_silence, check_speaker_consistency,
    check_streaming_safety, check_temporal_boundedness, is_metadata_only,
    verify_all_crown_properties, verify_all_crown_properties_with_attention,
    verify_all_crown_properties_with_evidence, verify_moonshot_from_stages,
    verify_properties_from_pipeline, verify_properties_from_pipeline_with_streaming,
    verify_properties_with_timing, verify_properties_with_timing_and_memory,
    verify_properties_with_timing_and_streaming, ay_kernel_category, ay_proven_kernel_names,
    ImplementationCorrectnessEvidence, MoonshotCrownBundle, MoonshotPropertyResult,
    SpeakerConsistencyEvidence,
};
pub use multi_res_stft::{compute_multi_res_stft, MultiResStftConfig};
pub use pesq::compute_pesq;
pub use phoneme::{PhonemeAlignment, PhonemeResult, PhonemeSpan, PhonemeVerifyConfig};
pub use phoneme_crown::{
    energy_range, f0_range_hz, interpret_phoneme_features, max_energy_range, max_f0_range_hz,
    PhonemeFeatureCertificate,
};
pub use phoneme_defects::{detect_defects, PronunciationDefect};
pub use phoneme_verify::verify_phonemes;
pub use pipeline::{
    check_junction, verify_pipeline, verify_pipeline_with_timing, HybridCertificate,
    JunctionResult, PipelineCertificate, TimingCertificate, VerifiedStage,
};
pub use pronunciation_unicode::{
    analyze_defects_with_unicode, classify_pronunciation_impact, DoubleVulnerability,
    PronunciationImpact, RiskLevel, UnicodeDefectAnalysis,
};
pub use quality::QualityMetric;
pub use quality_bound::{
    cosine_similarity_lipschitz, mcd_lipschitz, snr_lipschitz, spectral_convergence_lipschitz,
    standard_quality_specs, verify_quality_bounds, QualityBoundCertificate, QualityBoundResult,
    QualityMetricSpec,
};
pub use quantization_certificate::{
    build_quantization_certificate, build_segment_result, compute_element_drift,
    QuantizationCertificate, QuantizationSegmentResult,
};
pub use silero_vad_dispatch::{
    build_silero_vad_dispatch_plan, build_silero_vad_dispatch_plan_default, ENCODER_STEP_PAIRS,
    LSTM_DECOMPOSED_STEPS, OUTPUT_STAGE_STEPS, TOTAL_EXPECTED_STEPS,
};
pub use singing::{
    hz_to_cents, midi_to_hz,
    pitch::{verify_pitch_accuracy, NoteAccuracyResult, PitchAccuracyConfig},
    timing::{verify_timing, TimingConfig, TimingResult},
    vibrato::{
        extract_vibrato, verify_score_vibrato, verify_vibrato, VibratoConfig, VibratoParams,
    },
    MusicalScore, ScoreNote,
};
pub use stats::{cohens_d, holm_bonferroni, welch_t_test};
pub use stoi::compute_stoi;
pub use streaming::{
    crossfade_linear, verify_boundary, verify_streaming, BoundaryResult, StreamingCertificate,
    StreamingConfig,
};
pub use unicode_certificate::{
    dominant_attack_vector, unicode_safety_summary, UnicodeSafetyCertificate,
};
pub use unicode_perturbation::{
    analyze_unicode_coverage, expand_confusion_sets_for_unicode, identify_vulnerable_positions,
    map_to_phoneme_confusion_sets, UnicodeCoverageReport, UnicodeDerivedConfusionSet,
    VulnerabilityType, VulnerablePosition,
};
pub use unicode_safety::{
    scan_unicode, tts_confusables, UnicodeAttack, UnicodeSafetyConfig, UnicodeScanResult,
};
pub use verifier::{TtsVerifier, TtsVerifierBuilder};

#[cfg(feature = "ny")]
pub use adversarial_robustness::{
    verify_robustness, PositionRobustness, RobustnessCertificate, RobustnessConfig,
    RobustnessProperty,
};
#[cfg(feature = "ny")]
pub use cost_propagation::verify_layerwise_coupled;
#[cfg(feature = "ny")]
pub use disentanglement::{
    measure_sensitivity, verify_disentanglement, AcousticProperty, ControlDimension,
    DisentanglementCertificate, SensitivityResult,
};
#[cfg(feature = "ny")]
pub use fairness_crown::{
    verify_fairness_bounds, FairnessBoundsCertificate, GroupBoundsResult, GroupInputRegion,
};
#[cfg(feature = "ny")]
pub use moonshot_crown::generate_crown_constructive_proofs;
#[cfg(feature = "ny")]
pub use pipeline::{
    stage_from_bounds, stage_from_propagation, stage_from_propagation_with_soundness,
    verify_layerwise, verify_layerwise_from_graphs, verify_layerwise_grouped,
    verify_layerwise_mixed, verify_layerwise_with_timing, GroupVerifyMode, LayerwiseGrouping,
};

#[cfg(test)]
mod test_audio_helpers;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "phoneme_tests.rs"]
mod phoneme_tests;

#[cfg(test)]
#[path = "config_validation_tests.rs"]
mod config_validation_tests;

#[cfg(test)]
#[path = "verifier_bounds_tests.rs"]
mod verifier_bounds_tests;

#[cfg(test)]
#[path = "expanded_coverage_tests.rs"]
mod expanded_coverage_tests;

#[cfg(kani)]
#[path = "pipeline_safety_kani.rs"]
mod pipeline_safety_kani;

#[cfg(kani)]
#[path = "kani_crown_moonshot_proofs.rs"]
mod kani_crown_moonshot_proofs;

#[cfg(kani)]
#[path = "kani_codec_algebra_proofs.rs"]
mod kani_codec_algebra_proofs;

#[cfg(kani)]
#[path = "kani_fairness_proofs.rs"]
mod kani_fairness_proofs;

#[cfg(kani)]
#[path = "kani_disentanglement_proofs.rs"]
mod kani_disentanglement_proofs;

#[cfg(kani)]
#[path = "kani_monotonicity_proofs.rs"]
mod kani_monotonicity_proofs;

#[cfg(kani)]
#[path = "kani_adversarial_proofs.rs"]
mod kani_adversarial_proofs;

#[cfg(kani)]
#[path = "kani_unicode_safety_proofs.rs"]
mod kani_unicode_safety_proofs;

#[cfg(kani)]
#[path = "kani_streaming_proofs.rs"]
mod kani_streaming_proofs;

#[cfg(kani)]
#[path = "kani_pipeline_proofs.rs"]
mod kani_pipeline_proofs;

#[cfg(kani)]
#[path = "kani_dispatch_builder_proofs.rs"]
mod kani_dispatch_builder_proofs;

#[cfg(all(kani, feature = "ny"))]
#[path = "kani_pipeline_crown.rs"]
mod kani_pipeline_crown;

#[cfg(all(kani, feature = "ny"))]
mod kani_pipeline_crown_harnesses;

#[cfg(kani)]
#[path = "kani_moonshot_crown_probabilistic.rs"]
mod kani_moonshot_crown_probabilistic;

#[cfg(kani)]
mod kani_moonshot_crown_probabilistic_harnesses;

#[cfg(kani)]
#[path = "kani_codec_algebra.rs"]
mod kani_codec_algebra;

#[cfg(kani)]
mod kani_codec_algebra_harnesses;

#[cfg(kani)]
#[path = "kani_moonshot_crown_properties.rs"]
mod kani_moonshot_crown_properties;

#[cfg(kani)]
#[path = "kani_cost_model_proofs.rs"]
mod kani_cost_model_proofs;

#[cfg(kani)]
#[path = "kani_moonshot_proofs.rs"]
mod kani_moonshot_proofs;

#[cfg(kani)]
#[path = "kani_pipeline_dispatch_proofs.rs"]
mod kani_pipeline_dispatch_proofs;

#[cfg(kani)]
#[path = "kani_fairness_extended.rs"]
mod kani_fairness_extended;

#[cfg(kani)]
#[path = "kani_stats_extended.rs"]
mod kani_stats_extended;

#[cfg(kani)]
#[path = "kani_quality_bound_extended.rs"]
mod kani_quality_bound_extended;

#[cfg(kani)]
#[path = "kani_moonshot_crown_props_extended.rs"]
mod kani_moonshot_crown_props_extended;

#[cfg(kani)]
#[path = "kani_contracts_moonshot_proofs.rs"]
mod kani_contracts_moonshot_proofs;

#[cfg(kani)]
#[path = "kani_config.rs"]
mod kani_config;

#[cfg(kani)]
#[path = "kani_certificate_streaming_contracts.rs"]
mod kani_certificate_streaming_contracts;

#[cfg(kani)]
#[path = "kani_tts_extra_proofs.rs"]
mod kani_tts_extra_proofs;

#[cfg(test)]
#[path = "tts_verify_tests.rs"]
mod tts_verify_tests;

#[cfg(test)]
#[path = "crown_certificate_tests.rs"]
mod crown_certificate_tests;

#[cfg(test)]
#[path = "pipeline_extended_tests.rs"]
mod pipeline_extended_tests;

#[cfg(test)]
#[path = "pipeline_verification_extended_tests.rs"]
mod pipeline_verification_extended_tests;

#[cfg(test)]
#[path = "tts_verify_extended_tests.rs"]
mod tts_verify_extended_tests;

#[cfg(test)]
#[path = "tts_pipeline_extended_tests.rs"]
mod tts_pipeline_extended_tests;
