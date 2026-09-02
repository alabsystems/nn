// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Replayable statistical generalization certificates for compressed models.
//!
//! This module implements Theorem 1 and Corollary 1 from
//! "Non-vacuous Generalization Bounds for Reinforcement Learning with
//! Verifiable Rewards" (arXiv:2607.14506v1). These are distributional
//! PAC-Bayes compression bounds, not universal neural-network safety proofs.
//!
//! A certificate binds the arithmetic claim to a frozen base model, an exact
//! encoded hypothesis, its hypothesis-independent reconstruction decoder, the
//! generation and reward specifications, the training corpus and deployment
//! distribution manifests, and the complete reward transcript. Replay checks
//! those identities and recomputes the empirical reward and lower bound.
//!
//! The generic replay path cannot execute an application-specific model,
//! hypothesis decoder, or reward verifier. It verifies the sealed transcript
//! aggregation and arithmetic; the hashed specifications must define how the
//! transcript was produced.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::certificate::integrity::IntegrityError;

#[path = "generalization_bound.rs"]
mod bound;
#[path = "generalization_certificate.rs"]
mod certificate;

pub use bound::{calculate_generalization_bound, derive_subsample_indices};

/// Current generalization certificate schema version.
pub const GENERALIZATION_CERTIFICATE_VERSION: u32 = 1;
/// Current reward transcript schema version.
pub const REWARD_TRANSCRIPT_VERSION: u32 = 1;
/// Current ordered training-corpus manifest schema version.
pub const TRAINING_CORPUS_MANIFEST_VERSION: u32 = 1;
/// Paper and theorem version implemented by this module.
pub const GENERALIZATION_PAPER_REFERENCE: &str = "arXiv:2607.14506v1";
/// Exact sampler used to replay uniformly selected prompt indices.
pub const SUBSAMPLE_ALGORITHM: &str = "uniform-u64-be-rejection-tape-v1";
/// Numerical policy recorded in every certificate.
pub const GENERALIZATION_NUMERICAL_POLICY: &str =
    "binary64 interval arithmetic; analytic log remainder; sqrt quotient enclosure; empirical mean and lower bound rounded downward; penalties rounded upward";

/// Serialize certificate scalars as round-trip-safe decimal strings.
///
/// `serde_json::Value` can be built with a different floating-point conversion
/// path than direct struct serialization after Cargo feature unification. A
/// certificate is hashed through `Value`, then persisted directly, so numeric
/// JSON values can otherwise move by one ULP across save/load and invalidate
/// their own content hash. Seventeen digits after the decimal point in
/// scientific notation uniquely preserve every finite `f64`; deserialization
/// uses Rust's correctly rounded parser rather than serde_json's number path.
mod stable_f64 {
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{value:.17e}"))
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        let value = encoded.parse::<f64>().map_err(D::Error::custom)?;
        if !value.is_finite() {
            return Err(D::Error::custom(
                "generalization certificate scalar must be finite",
            ));
        }
        Ok(value)
    }
}

/// SHA-256 identity and byte length of an evidence artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ArtifactIdentity {
    /// Lower-case SHA-256 digest.
    pub sha256: String,
    /// Exact artifact length in bytes.
    pub byte_len: u64,
}

impl ArtifactIdentity {
    /// Hash a file and record the number of bytes read.
    pub fn from_file(path: &Path) -> Result<Self, GeneralizationError> {
        certificate::identity_from_file(path)
    }

    /// Hash an in-memory artifact.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            sha256: crate::certificate_types::compute_bytes_hash(bytes),
            byte_len: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        }
    }

    /// Validate canonical identity syntax.
    pub fn validate(&self, name: &'static str) -> Result<(), GeneralizationError> {
        bound::validate_artifact_identity(self, name)
    }
}

/// Evidence artifacts that fix the hypothesis and statistical experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct EvidenceArtifacts {
    /// Frozen pretrained base model.
    pub base_model: ArtifactIdentity,
    /// Exact prefix-coded hypothesis bitstream (for example, an adapter).
    pub encoded_hypothesis: ArtifactIdentity,
    /// Fixed decoder capable of reconstructing the hypothesis bitstream.
    pub hypothesis_decoder_spec: ArtifactIdentity,
    /// Tokenizer, sampler, stopping rules, and environment specification.
    pub generation_spec: ArtifactIdentity,
    /// Deterministic bounded reward/verifier specification.
    pub reward_spec: ArtifactIdentity,
    /// Manifest declaring the deployment prompt distribution.
    pub prompt_distribution_manifest: ArtifactIdentity,
    /// Ordered training-corpus manifest from which prompt indices are taken.
    pub training_corpus_manifest: ArtifactIdentity,
}

impl EvidenceArtifacts {
    /// Hash all evidence files in one portable artifact set.
    pub fn from_paths(paths: &EvidenceArtifactPaths) -> Result<Self, GeneralizationError> {
        Ok(Self {
            base_model: ArtifactIdentity::from_file(&paths.base_model)?,
            encoded_hypothesis: ArtifactIdentity::from_file(&paths.encoded_hypothesis)?,
            hypothesis_decoder_spec: ArtifactIdentity::from_file(&paths.hypothesis_decoder_spec)?,
            generation_spec: ArtifactIdentity::from_file(&paths.generation_spec)?,
            reward_spec: ArtifactIdentity::from_file(&paths.reward_spec)?,
            prompt_distribution_manifest: ArtifactIdentity::from_file(
                &paths.prompt_distribution_manifest,
            )?,
            training_corpus_manifest: ArtifactIdentity::from_file(&paths.training_corpus_manifest)?,
        })
    }

    /// Validate all stored artifact identities.
    pub fn validate(&self) -> Result<(), GeneralizationError> {
        self.base_model.validate("base_model")?;
        self.encoded_hypothesis.validate("encoded_hypothesis")?;
        self.hypothesis_decoder_spec
            .validate("hypothesis_decoder_spec")?;
        self.generation_spec.validate("generation_spec")?;
        self.reward_spec.validate("reward_spec")?;
        self.prompt_distribution_manifest
            .validate("prompt_distribution_manifest")?;
        self.training_corpus_manifest
            .validate("training_corpus_manifest")
    }
}

/// Local paths used to create or replay artifact identities.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EvidenceArtifactPaths {
    pub base_model: PathBuf,
    pub encoded_hypothesis: PathBuf,
    pub hypothesis_decoder_spec: PathBuf,
    pub generation_spec: PathBuf,
    pub reward_spec: PathBuf,
    pub prompt_distribution_manifest: PathBuf,
    pub training_corpus_manifest: PathBuf,
}

impl EvidenceArtifactPaths {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        base_model: PathBuf,
        encoded_hypothesis: PathBuf,
        hypothesis_decoder_spec: PathBuf,
        generation_spec: PathBuf,
        reward_spec: PathBuf,
        prompt_distribution_manifest: PathBuf,
        training_corpus_manifest: PathBuf,
    ) -> Self {
        Self {
            base_model,
            encoded_hypothesis,
            hypothesis_decoder_spec,
            generation_spec,
            reward_spec,
            prompt_distribution_manifest,
            training_corpus_manifest,
        }
    }
}

/// Exact description-length accounting for the encoded hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct HypothesisEncoding {
    /// Stable identifier for the fixed prefix-code scheme.
    pub scheme_id: String,
    /// `C(h)` in bits. The theorem adds its own `2 ln C` prefix overhead.
    pub description_length_bits: u64,
}

impl HypothesisEncoding {
    #[must_use]
    pub fn new(scheme_id: impl Into<String>, description_length_bits: u64) -> Self {
        Self {
            scheme_id: scheme_id.into(),
            description_length_bits,
        }
    }
}

/// Explicit attestations required by the paper's theorem.
///
/// These assumptions are evidence declarations, not facts the calculator can
/// infer from hashes. Certification fails closed unless every field is true.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GeneralizationAssumptions {
    pub training_prompts_iid_from_deployment_distribution: bool,
    pub deployment_distribution_matches_training_distribution: bool,
    pub generation_noise_iid_and_independent_of_prompts_training_and_hypothesis: bool,
    pub generation_noise_distribution_matches_deployment: bool,
    pub generation_deterministic_given_prompt_noise_and_hypothesis: bool,
    pub reward_function_deterministic_and_globally_bounded: bool,
    pub generation_and_deployment_specs_fixed_before_training_sample: bool,
    pub reward_spec_bounds_and_failure_probability_fixed_before_training_sample: bool,
    pub hypothesis_encoding_scheme_fixed_before_training_sample: bool,
    pub hypothesis_encoding_scheme_is_prefix_free: bool,
    pub encoded_code_reconstructs_final_hypothesis: bool,
    pub base_model_and_zero_cost_decoder_fixed_before_training_sample: bool,
}

impl GeneralizationAssumptions {
    /// Construct the complete set of paper-v1 attestations.
    ///
    /// Calling this is an assertion by the evidence producer that the
    /// experiment actually satisfies every listed assumption.
    #[must_use]
    pub fn asserted_paper_v1() -> Self {
        Self {
            training_prompts_iid_from_deployment_distribution: true,
            deployment_distribution_matches_training_distribution: true,
            generation_noise_iid_and_independent_of_prompts_training_and_hypothesis: true,
            generation_noise_distribution_matches_deployment: true,
            generation_deterministic_given_prompt_noise_and_hypothesis: true,
            reward_function_deterministic_and_globally_bounded: true,
            generation_and_deployment_specs_fixed_before_training_sample: true,
            reward_spec_bounds_and_failure_probability_fixed_before_training_sample: true,
            hypothesis_encoding_scheme_fixed_before_training_sample: true,
            hypothesis_encoding_scheme_is_prefix_free: true,
            encoded_code_reconstructs_final_hypothesis: true,
            base_model_and_zero_cost_decoder_fixed_before_training_sample: true,
        }
    }

    pub fn validate(&self) -> Result<(), GeneralizationError> {
        bound::validate_assumptions(self)
    }
}

/// Uniform prompt-subsampling scheme used by Corollary 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SubsampleScheme {
    UniformWithReplacement,
    UniformWithoutReplacement,
}

/// How transcript prompts were selected for evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum EvaluationSampling {
    /// Evaluate every indexed training prompt and use Theorem 1.
    FullTrainingSet,
    /// Evaluate a uniform random subsample and use Corollary 1.
    UniformSubsample {
        scheme: SubsampleScheme,
        /// Fixed algorithm identifier; must equal [`SUBSAMPLE_ALGORITHM`].
        selection_algorithm: String,
        /// Big-endian u64 source words as fixed-width lower-case hexadecimal.
        ///
        /// This is the exact stopped prefix consumed by rejection sampling,
        /// including rejected words and ending at the final accepted word.
        selection_random_tape_hex: String,
        /// The tape is the first uninterrupted stopped prefix produced by
        /// sequential IID uniform u64 draws, with no whole-tape retries.
        selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source: bool,
        /// The random source is independent of the training corpus, evaluation
        /// noise and rewards, and the learned hypothesis.
        selection_random_tape_source_independent_of_training_evaluation_and_hypothesis: bool,
        /// The sampling scheme and sample size were fixed before drawing tape.
        sampling_scheme_and_size_fixed_before_selection: bool,
        /// Must be true: the final hypothesis was fixed before indices were drawn.
        hypothesis_committed_before_sampling: bool,
    },
}

/// Ordered prompt identities used to link transcript indices to the corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TrainingCorpusManifest {
    pub schema_version: u32,
    /// SHA-256 of each exact prompt, in training-sample order.
    pub prompt_sha256: Vec<String>,
}

impl TrainingCorpusManifest {
    #[must_use]
    pub fn new(prompt_sha256: Vec<String>) -> Self {
        Self {
            schema_version: TRAINING_CORPUS_MANIFEST_VERSION,
            prompt_sha256,
        }
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, GeneralizationError> {
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), GeneralizationError> {
        bound::validate_training_corpus_manifest(self)
    }
}

/// One generation/noise draw and its deterministic reward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GenerationReward {
    /// Commitment to the independent generation noise or seed.
    pub noise_sha256: String,
    /// Commitment to the generated output/trajectory.
    pub generation_sha256: String,
    /// Deterministic reward in the transcript's declared interval.
    pub reward: f64,
}

/// All independent generation draws evaluated for one sampled prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PromptRewardRecord {
    /// Zero-based index in the ordered training-corpus manifest.
    pub training_index: u64,
    /// Commitment to the exact prompt bytes.
    pub prompt_sha256: String,
    /// A common number of independent draws is required for every record.
    pub generations: Vec<GenerationReward>,
}

/// Sealed empirical-reward input to certificate generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RewardTranscript {
    pub schema_version: u32,
    pub training_prompt_count: u64,
    pub reward_lower: f64,
    pub reward_upper: f64,
    pub sampling: EvaluationSampling,
    pub assumptions: GeneralizationAssumptions,
    pub prompts: Vec<PromptRewardRecord>,
}

impl RewardTranscript {
    /// Parse a reward transcript without silently accepting a newer schema.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, GeneralizationError> {
        let transcript: Self = serde_json::from_slice(bytes)?;
        transcript.validate()?;
        Ok(transcript)
    }

    /// Validate structure, assumptions, hashes, sampling, and reward bounds.
    pub fn validate(&self) -> Result<(), GeneralizationError> {
        self.summarize().map(|_| ())
    }

    /// Recompute the conservative empirical mean and sample counts.
    pub fn summarize(&self) -> Result<RewardTranscriptSummary, GeneralizationError> {
        bound::summarize_transcript(self)
    }

    /// Verify that every transcript index names the committed corpus prompt.
    pub fn validate_against_manifest(
        &self,
        manifest: &TrainingCorpusManifest,
    ) -> Result<(), GeneralizationError> {
        bound::validate_transcript_manifest_link(self, manifest)
    }
}

/// Replay-derived statistics from a validated transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RewardTranscriptSummary {
    pub empirical_reward: f64,
    pub evaluated_prompt_count: u64,
    pub generations_per_prompt: u64,
    pub method: GeneralizationMethod,
}

/// Exact theorem variant used for the bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum GeneralizationMethod {
    Theorem1FullTrainingSet,
    Corollary1UniformSubsample { scheme: SubsampleScheme },
}

/// Scalar inputs sufficient to independently replay the paper's formula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GeneralizationBoundInputs {
    pub method: GeneralizationMethod,
    #[serde(with = "stable_f64")]
    pub empirical_reward: f64,
    #[serde(with = "stable_f64")]
    pub reward_lower: f64,
    #[serde(with = "stable_f64")]
    pub reward_upper: f64,
    pub training_prompt_count: u64,
    pub evaluated_prompt_count: u64,
    pub generations_per_prompt: u64,
    pub description_length_bits: u64,
    /// Failure probability `delta`; confidence is `1 - delta`.
    #[serde(with = "stable_f64")]
    pub failure_probability: f64,
}

impl GeneralizationBoundInputs {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        method: GeneralizationMethod,
        empirical_reward: f64,
        reward_lower: f64,
        reward_upper: f64,
        training_prompt_count: u64,
        evaluated_prompt_count: u64,
        generations_per_prompt: u64,
        description_length_bits: u64,
        failure_probability: f64,
    ) -> Self {
        Self {
            method,
            empirical_reward,
            reward_lower,
            reward_upper,
            training_prompt_count,
            evaluated_prompt_count,
            generations_per_prompt,
            description_length_bits,
            failure_probability,
        }
    }
}

/// Decomposed statistical penalty and population-reward lower bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GeneralizationBound {
    /// `C ln 2 + 2 ln C + ln(k/delta)`, with `k=2` or `3`.
    #[serde(with = "stable_f64")]
    pub complexity_penalty_nats: f64,
    #[serde(with = "stable_f64")]
    pub subsample_penalty: f64,
    #[serde(with = "stable_f64")]
    pub extrinsic_penalty: f64,
    #[serde(with = "stable_f64")]
    pub intrinsic_penalty: f64,
    #[serde(with = "stable_f64")]
    pub total_penalty: f64,
    /// Formula result before using the trivial reward-floor guarantee.
    #[serde(with = "stable_f64")]
    pub raw_lower_bound: f64,
    /// `max(reward_lower, raw_lower_bound)`.
    #[serde(with = "stable_f64")]
    pub certified_lower_bound: f64,
    #[serde(with = "stable_f64")]
    pub confidence: f64,
}

impl GeneralizationBound {
    #[must_use]
    pub fn is_non_vacuous(&self, reward_lower: f64) -> bool {
        self.raw_lower_bound > reward_lower
    }
}

/// Inputs that bind a transcript calculation to concrete evidence artifacts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GeneralizationCertificateInputs {
    pub artifacts: EvidenceArtifacts,
    pub hypothesis_encoding: HypothesisEncoding,
    #[serde(with = "stable_f64")]
    pub failure_probability: f64,
}

impl GeneralizationCertificateInputs {
    #[must_use]
    pub fn new(
        artifacts: EvidenceArtifacts,
        hypothesis_encoding: HypothesisEncoding,
        failure_probability: f64,
    ) -> Self {
        Self {
            artifacts,
            hypothesis_encoding,
            failure_probability,
        }
    }
}

/// Replayable PAC-Bayes population-reward certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GeneralizationCertificate {
    pub schema_version: u32,
    pub paper_reference: String,
    pub nn_version: String,
    pub generated_at: String,
    pub numerical_policy: String,
    pub artifacts: EvidenceArtifacts,
    pub hypothesis_encoding: HypothesisEncoding,
    pub transcript: ArtifactIdentity,
    pub sampling: EvaluationSampling,
    pub assumptions: GeneralizationAssumptions,
    pub bound_inputs: GeneralizationBoundInputs,
    pub bound: GeneralizationBound,
    /// Canonical SHA-256 over all fields except integrity fields.
    pub content_hash: Option<String>,
    /// Optional HMAC-SHA256 supported by the common certificate signer.
    pub hmac_signature: Option<String>,
}

/// Successful replay report suitable for JSON output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct GeneralizationReplayReport {
    pub certificate_content_hash_verified: bool,
    pub evidence_artifacts_verified: u64,
    pub transcript_verified: bool,
    pub arithmetic_replayed: bool,
    pub empirical_reward: f64,
    pub certified_lower_bound: f64,
    pub confidence: f64,
    pub non_vacuous: bool,
    pub hmac_signature_present: bool,
    pub hmac_signature_verified: bool,
}

/// Validation, calculation, serialization, or replay failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GeneralizationError {
    #[error("unsupported reward transcript schema version {found}; expected {expected}")]
    UnsupportedTranscriptVersion { found: u32, expected: u32 },
    #[error("unsupported generalization certificate schema version {found}; expected {expected}")]
    UnsupportedCertificateVersion { found: u32, expected: u32 },
    #[error("unsupported training-corpus manifest schema version {found}; expected {expected}")]
    UnsupportedTrainingCorpusManifestVersion { found: u32, expected: u32 },
    #[error("invalid SHA-256 identity for {artifact}: {value}")]
    InvalidArtifactHash {
        artifact: &'static str,
        value: String,
    },
    #[error("evidence artifact {artifact} is empty")]
    EmptyArtifact { artifact: &'static str },
    #[error("artifact identity mismatch for {artifact}: expected {expected}, got {actual}")]
    ArtifactMismatch {
        artifact: &'static str,
        expected: String,
        actual: String,
    },
    #[error("hypothesis encoding scheme_id must not be empty")]
    EmptyEncodingScheme,
    #[error("description length must be at least one bit")]
    ZeroDescriptionLength,
    #[error("description length {bits} is not representable exactly by certificate arithmetic")]
    DescriptionLengthTooLarge { bits: u64 },
    #[error("encoded hypothesis has {capacity_bits} bits but claimed description length is {description_bits} bits")]
    InvalidEncodedHypothesisLength {
        description_bits: u64,
        capacity_bits: u64,
    },
    #[error("failure probability must be finite and strictly between 0 and 1; got {value}")]
    InvalidFailureProbability { value: f64 },
    #[error("invalid finite reward interval [{lower}, {upper}]")]
    InvalidRewardRange { lower: f64, upper: f64 },
    #[error("required theorem assumption was not asserted: {assumption}")]
    MissingAssumption { assumption: &'static str },
    #[error("training-corpus manifest contains no prompt identities")]
    EmptyTrainingCorpusManifest,
    #[error("training-corpus manifest has too many prompt identities")]
    TrainingCorpusManifestTooLarge,
    #[error("transcript training_prompt_count {transcript} does not match corpus manifest count {manifest}")]
    TrainingCorpusCountMismatch { transcript: u64, manifest: u64 },
    #[error("transcript prompt at training index {index} does not match the corpus manifest")]
    TrainingCorpusPromptMismatch { index: u64 },
    #[error("training_prompt_count must be in 1..=2^53; got {count}")]
    InvalidTrainingPromptCount { count: u64 },
    #[error("reward transcript contains no evaluated prompts")]
    EmptyTranscript,
    #[error("prompt record {record} has training index {index}, outside 0..{training_count}")]
    PromptIndexOutOfRange {
        record: usize,
        index: u64,
        training_count: u64,
    },
    #[error("full-training-set transcript must cover each index exactly once")]
    InvalidFullTrainingCoverage,
    #[error("without-replacement subsample contains duplicate training index {index}")]
    DuplicateSubsampleIndex { index: u64 },
    #[error("training index {index} is associated with inconsistent prompt hashes")]
    InconsistentPromptHash { index: u64 },
    #[error("prompt record {record} contains no generation rewards")]
    EmptyPromptGenerations { record: usize },
    #[error("prompt record {record} has {found} generations; expected {expected}")]
    InconsistentGenerationCount {
        record: usize,
        expected: u64,
        found: u64,
    },
    #[error("generation count is too large for exact certificate arithmetic")]
    GenerationCountTooLarge,
    #[error("duplicate generation-noise commitment {hash}")]
    DuplicateNoiseCommitment { hash: String },
    #[error("reward at prompt record {record}, generation {generation} is {value}, outside [{lower}, {upper}]")]
    RewardOutOfRange {
        record: usize,
        generation: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("subsample must contain at most m prompts; got m0={evaluated}, m={training}")]
    InvalidSubsampleSize { evaluated: u64, training: u64 },
    #[error("a without-replacement subsample covering all prompts must use Theorem 1")]
    FullSetEncodedAsSubsample,
    #[error("subsample certificate requires the final hypothesis to be committed before sampling")]
    HypothesisNotPrecommitted,
    #[error("subsample random tape must contain fixed-width lower-case hexadecimal u64 words")]
    InvalidSelectionRandomTape,
    #[error("unsupported subsample selection algorithm {found}; expected {expected}")]
    UnsupportedSelectionAlgorithm {
        found: String,
        expected: &'static str,
    },
    #[error("subsample random tape was not attested as the first stopped prefix of an IID uniform source")]
    SelectionRandomTapeNotFirstIidUniformPrefix,
    #[error("subsample random source was not attested as independent of training, evaluation, and the hypothesis")]
    SelectionRandomTapeNotIndependent,
    #[error("subsample scheme and size were not fixed before random selection")]
    SamplingPlanNotPrecommitted,
    #[error("subsample record {record} has training index {found}; deterministic replay expected {expected}")]
    SubsampleSelectionMismatch {
        record: usize,
        expected: u64,
        found: u64,
    },
    #[error("subsample random tape ended before all indices could be drawn")]
    SubsampleRandomTapeExhausted,
    #[error("subsample random tape contains {remaining_words} unused u64 words")]
    SubsampleRandomTapeHasTrailingWords { remaining_words: usize },
    #[error(
        "repeated with-replacement training index {index} must reuse the same generation block"
    )]
    InconsistentRepeatedSubsampleGenerations { index: u64 },
    #[error("bound method and prompt counts are inconsistent")]
    InconsistentBoundMethod,
    #[error("empirical reward {value} lies outside [{lower}, {upper}]")]
    InvalidEmpiricalReward { value: f64, lower: f64, upper: f64 },
    #[error("sample-count product overflows or exceeds exact f64 integer range")]
    SampleCountTooLarge,
    #[error("certificate field {field} does not match replayed arithmetic")]
    ArithmeticMismatch { field: &'static str },
    #[error("bound arithmetic produced a non-finite value for {field}")]
    NonFiniteBound { field: &'static str },
    #[error("certificate metadata mismatch: {field}")]
    MetadataMismatch { field: &'static str },
    #[error("JSON serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("certificate integrity failed: {0}")]
    Integrity(#[from] IntegrityError),
}

#[cfg(test)]
#[path = "generalization_tests.rs"]
mod tests;
