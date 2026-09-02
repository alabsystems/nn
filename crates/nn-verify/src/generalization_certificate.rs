// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certificate construction, integrity, persistence, and evidence replay.

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::bound::{floats_match, validate_certificate_version, MAX_EXACT_F64_INTEGER};
use super::{
    calculate_generalization_bound, ArtifactIdentity, EvidenceArtifactPaths, EvidenceArtifacts,
    GeneralizationBound, GeneralizationBoundInputs, GeneralizationCertificate,
    GeneralizationCertificateInputs, GeneralizationError, GeneralizationReplayReport,
    HypothesisEncoding, RewardTranscript, GENERALIZATION_CERTIFICATE_VERSION,
    GENERALIZATION_NUMERICAL_POLICY, GENERALIZATION_PAPER_REFERENCE,
};
use crate::certificate::integrity::{compute_content_hash, verify_content_hash, Signable};

static TEMPORARY_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn identity_from_file(path: &Path) -> Result<ArtifactIdentity, GeneralizationError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut byte_len = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        byte_len = byte_len
            .checked_add(u64::try_from(read).map_err(|_| GeneralizationError::SampleCountTooLarge)?)
            .ok_or(GeneralizationError::SampleCountTooLarge)?;
        hasher.update(&buffer[..read]);
    }
    Ok(ArtifactIdentity {
        sha256: crate::signing_config::hex_encode(&hasher.finalize()),
        byte_len,
    })
}

impl HypothesisEncoding {
    pub(super) fn validate(
        &self,
        encoded_hypothesis: &ArtifactIdentity,
    ) -> Result<(), GeneralizationError> {
        if self.scheme_id.trim().is_empty() {
            return Err(GeneralizationError::EmptyEncodingScheme);
        }
        if self.description_length_bits == 0 {
            return Err(GeneralizationError::ZeroDescriptionLength);
        }
        if self.description_length_bits > MAX_EXACT_F64_INTEGER {
            return Err(GeneralizationError::DescriptionLengthTooLarge {
                bits: self.description_length_bits,
            });
        }
        let capacity_bits = encoded_hypothesis.byte_len.checked_mul(8).ok_or(
            GeneralizationError::InvalidEncodedHypothesisLength {
                description_bits: self.description_length_bits,
                capacity_bits: u64::MAX,
            },
        )?;
        // Charge every stored bit. This is a conservative upper bound on the
        // logical prefix-code length and avoids trusting unverifiable padding
        // semantics in a generic decoder specification.
        if self.description_length_bits != capacity_bits {
            return Err(GeneralizationError::InvalidEncodedHypothesisLength {
                description_bits: self.description_length_bits,
                capacity_bits,
            });
        }
        Ok(())
    }
}

impl GeneralizationCertificateInputs {
    fn validate(&self) -> Result<(), GeneralizationError> {
        self.artifacts.validate()?;
        self.hypothesis_encoding
            .validate(&self.artifacts.encoded_hypothesis)?;
        if !self.failure_probability.is_finite()
            || self.failure_probability <= 0.0
            || self.failure_probability >= 1.0
        {
            return Err(GeneralizationError::InvalidFailureProbability {
                value: self.failure_probability,
            });
        }
        Ok(())
    }
}

impl GeneralizationCertificate {
    /// Calculate and content-seal a certificate from raw transcript and
    /// ordered training-corpus manifest bytes.
    ///
    /// Hashing the original bytes makes whitespace or ordering changes visible
    /// during replay even when they deserialize to the same JSON value.
    pub fn calculate(
        inputs: GeneralizationCertificateInputs,
        transcript_bytes: &[u8],
        training_corpus_manifest_bytes: &[u8],
    ) -> Result<Self, GeneralizationError> {
        inputs.validate()?;
        let actual_manifest_identity = ArtifactIdentity::from_bytes(training_corpus_manifest_bytes);
        compare_identity(
            "training_corpus_manifest",
            &inputs.artifacts.training_corpus_manifest,
            &actual_manifest_identity,
        )?;
        let training_manifest =
            super::TrainingCorpusManifest::from_json_slice(training_corpus_manifest_bytes)?;
        let transcript = RewardTranscript::from_json_slice(transcript_bytes)?;
        transcript.validate_against_manifest(&training_manifest)?;
        let summary = transcript.summarize()?;
        let transcript_identity = ArtifactIdentity::from_bytes(transcript_bytes);
        transcript_identity.validate("reward_transcript")?;

        let bound_inputs = GeneralizationBoundInputs {
            method: summary.method,
            empirical_reward: summary.empirical_reward,
            reward_lower: transcript.reward_lower,
            reward_upper: transcript.reward_upper,
            training_prompt_count: transcript.training_prompt_count,
            evaluated_prompt_count: summary.evaluated_prompt_count,
            generations_per_prompt: summary.generations_per_prompt,
            description_length_bits: inputs.hypothesis_encoding.description_length_bits,
            failure_probability: inputs.failure_probability,
        };
        let bound = calculate_generalization_bound(&bound_inputs)?;

        let mut certificate = Self {
            schema_version: GENERALIZATION_CERTIFICATE_VERSION,
            paper_reference: GENERALIZATION_PAPER_REFERENCE.to_string(),
            nn_version: env!("CARGO_PKG_VERSION").to_string(),
            generated_at: crate::certificate::now_iso8601(),
            numerical_policy: GENERALIZATION_NUMERICAL_POLICY.to_string(),
            artifacts: inputs.artifacts,
            hypothesis_encoding: inputs.hypothesis_encoding,
            transcript: transcript_identity,
            sampling: transcript.sampling,
            assumptions: transcript.assumptions,
            bound_inputs,
            bound,
            content_hash: None,
            hmac_signature: None,
        };
        certificate.content_hash = Some(compute_content_hash(&certificate)?);
        certificate.validate()?;
        Ok(certificate)
    }

    /// Validate metadata, assumptions, artifact identities, integrity, and all
    /// stored arithmetic without requiring the external transcript.
    pub fn validate(&self) -> Result<(), GeneralizationError> {
        validate_certificate_version(self.schema_version)?;
        if self.paper_reference != GENERALIZATION_PAPER_REFERENCE {
            return Err(GeneralizationError::MetadataMismatch {
                field: "paper_reference",
            });
        }
        if self.nn_version.trim().is_empty() {
            return Err(GeneralizationError::MetadataMismatch {
                field: "nn_version",
            });
        }
        if self.generated_at.trim().is_empty() {
            return Err(GeneralizationError::MetadataMismatch {
                field: "generated_at",
            });
        }
        if self.numerical_policy != GENERALIZATION_NUMERICAL_POLICY {
            return Err(GeneralizationError::MetadataMismatch {
                field: "numerical_policy",
            });
        }
        self.artifacts.validate()?;
        self.hypothesis_encoding
            .validate(&self.artifacts.encoded_hypothesis)?;
        self.transcript.validate("reward_transcript")?;
        self.sampling.validate()?;
        self.assumptions.validate()?;
        if self.sampling.method() != self.bound_inputs.method {
            return Err(GeneralizationError::InconsistentBoundMethod);
        }
        if self.bound_inputs.description_length_bits
            != self.hypothesis_encoding.description_length_bits
        {
            return Err(GeneralizationError::MetadataMismatch {
                field: "description_length_bits",
            });
        }
        self.bound_inputs.validate()?;
        let recomputed = calculate_generalization_bound(&self.bound_inputs)?;
        compare_bound(&self.bound, &recomputed)?;

        let content_hash = self
            .content_hash
            .as_deref()
            .ok_or(crate::certificate::integrity::IntegrityError::MissingContentHash)?;
        validate_hash_string(content_hash, "certificate_content_hash")?;
        if let Some(signature) = &self.hmac_signature {
            validate_hash_string(signature, "certificate_hmac_signature")?;
        }
        verify_content_hash(self)?;
        Ok(())
    }

    /// Recompute the complete penalty decomposition from stored scalar inputs.
    pub fn recompute_bound(&self) -> Result<GeneralizationBound, GeneralizationError> {
        calculate_generalization_bound(&self.bound_inputs)
    }

    /// Recompute only the range-clamped population-reward lower bound.
    pub fn recompute_lower_bound(&self) -> Result<f64, GeneralizationError> {
        Ok(self.recompute_bound()?.certified_lower_bound)
    }

    /// Whether the raw bound improves on the trivial reward floor.
    #[must_use]
    pub fn is_non_vacuous(&self) -> bool {
        self.bound.is_non_vacuous(self.bound_inputs.reward_lower)
    }

    /// Verify all external artifacts, link prompts to the corpus, replay any
    /// subsample, aggregate rewards, and recompute bound arithmetic.
    pub fn replay(
        &self,
        transcript_bytes: &[u8],
        paths: &EvidenceArtifactPaths,
    ) -> Result<GeneralizationReplayReport, GeneralizationError> {
        self.validate()?;

        let actual_artifacts = EvidenceArtifacts::from_paths(paths)?;
        compare_artifacts(&self.artifacts, &actual_artifacts)?;
        let actual_transcript = ArtifactIdentity::from_bytes(transcript_bytes);
        compare_identity("reward_transcript", &self.transcript, &actual_transcript)?;

        let transcript = RewardTranscript::from_json_slice(transcript_bytes)?;
        let training_manifest_bytes = std::fs::read(&paths.training_corpus_manifest)?;
        let parsed_manifest_identity = ArtifactIdentity::from_bytes(&training_manifest_bytes);
        compare_identity(
            "training_corpus_manifest",
            &self.artifacts.training_corpus_manifest,
            &parsed_manifest_identity,
        )?;
        let training_manifest =
            super::TrainingCorpusManifest::from_json_slice(&training_manifest_bytes)?;
        transcript.validate_against_manifest(&training_manifest)?;
        let summary = transcript.summarize()?;
        if transcript.sampling != self.sampling {
            return Err(GeneralizationError::MetadataMismatch { field: "sampling" });
        }
        if transcript.assumptions != self.assumptions {
            return Err(GeneralizationError::MetadataMismatch {
                field: "assumptions",
            });
        }
        let replay_inputs = GeneralizationBoundInputs {
            method: summary.method,
            empirical_reward: summary.empirical_reward,
            reward_lower: transcript.reward_lower,
            reward_upper: transcript.reward_upper,
            training_prompt_count: transcript.training_prompt_count,
            evaluated_prompt_count: summary.evaluated_prompt_count,
            generations_per_prompt: summary.generations_per_prompt,
            description_length_bits: self.hypothesis_encoding.description_length_bits,
            failure_probability: self.bound_inputs.failure_probability,
        };
        compare_bound_inputs(&self.bound_inputs, &replay_inputs)?;
        let replayed_bound = calculate_generalization_bound(&replay_inputs)?;
        compare_bound(&self.bound, &replayed_bound)?;

        Ok(GeneralizationReplayReport {
            certificate_content_hash_verified: true,
            evidence_artifacts_verified: 7,
            transcript_verified: true,
            arithmetic_replayed: true,
            empirical_reward: replay_inputs.empirical_reward,
            certified_lower_bound: replayed_bound.certified_lower_bound,
            confidence: replayed_bound.confidence,
            non_vacuous: replayed_bound.is_non_vacuous(replay_inputs.reward_lower),
            hmac_signature_present: self.hmac_signature.is_some(),
            hmac_signature_verified: false,
        })
    }

    pub fn to_json(&self) -> Result<String, GeneralizationError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self, GeneralizationError> {
        let certificate: Self = serde_json::from_str(json)?;
        certificate.validate()?;
        Ok(certificate)
    }

    /// Atomically save the certificate as pretty JSON.
    pub fn save(&self, path: &Path) -> Result<(), GeneralizationError> {
        use std::io::Write;

        self.validate()?;
        let json = self.to_json()?;
        let parent = parent_directory(path);
        let (mut file, temporary_path) = create_unique_temporary_file(path)?;
        if let Err(error) = file
            .write_all(json.as_bytes())
            .and_then(|()| file.sync_all())
        {
            drop(file);
            let _ = std::fs::remove_file(&temporary_path);
            return Err(GeneralizationError::Io(error));
        }
        drop(file);
        if let Err(error) = std::fs::rename(&temporary_path, path) {
            let _ = std::fs::remove_file(&temporary_path);
            return Err(GeneralizationError::Io(error));
        }
        let _ = std::fs::File::open(parent).and_then(|directory| directory.sync_all());
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, GeneralizationError> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }
}

fn create_unique_temporary_file(
    path: &Path,
) -> Result<(std::fs::File, std::path::PathBuf), GeneralizationError> {
    let parent = parent_directory(path);
    let file_name = path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("certificate"))
        .to_string_lossy();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for _ in 0..128 {
        let sequence = TEMPORARY_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary_path = parent.join(format!(
            ".{file_name}.{}.{}.{}.tmp",
            std::process::id(),
            timestamp,
            sequence
        ));
        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((file, temporary_path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(GeneralizationError::Io(error)),
        }
    }
    Err(GeneralizationError::Io(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique certificate temporary file",
    )))
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

impl Signable for GeneralizationCertificate {
    fn content_hash(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    fn hmac_signature(&self) -> Option<&str> {
        self.hmac_signature.as_deref()
    }

    fn set_content_hash(&mut self, hash: Option<String>) {
        self.content_hash = hash;
    }

    fn set_hmac_signature(&mut self, signature: Option<String>) {
        self.hmac_signature = signature;
    }
}

fn validate_hash_string(value: &str, artifact: &'static str) -> Result<(), GeneralizationError> {
    let identity = ArtifactIdentity {
        sha256: value.to_string(),
        byte_len: 1,
    };
    identity.validate(artifact)
}

fn compare_identity(
    name: &'static str,
    expected: &ArtifactIdentity,
    actual: &ArtifactIdentity,
) -> Result<(), GeneralizationError> {
    if expected != actual {
        return Err(GeneralizationError::ArtifactMismatch {
            artifact: name,
            expected: format!("{}:{}", expected.sha256, expected.byte_len),
            actual: format!("{}:{}", actual.sha256, actual.byte_len),
        });
    }
    Ok(())
}

fn compare_artifacts(
    expected: &EvidenceArtifacts,
    actual: &EvidenceArtifacts,
) -> Result<(), GeneralizationError> {
    compare_identity("base_model", &expected.base_model, &actual.base_model)?;
    compare_identity(
        "encoded_hypothesis",
        &expected.encoded_hypothesis,
        &actual.encoded_hypothesis,
    )?;
    compare_identity(
        "hypothesis_decoder_spec",
        &expected.hypothesis_decoder_spec,
        &actual.hypothesis_decoder_spec,
    )?;
    compare_identity(
        "generation_spec",
        &expected.generation_spec,
        &actual.generation_spec,
    )?;
    compare_identity("reward_spec", &expected.reward_spec, &actual.reward_spec)?;
    compare_identity(
        "prompt_distribution_manifest",
        &expected.prompt_distribution_manifest,
        &actual.prompt_distribution_manifest,
    )?;
    compare_identity(
        "training_corpus_manifest",
        &expected.training_corpus_manifest,
        &actual.training_corpus_manifest,
    )
}

fn compare_bound_inputs(
    expected: &GeneralizationBoundInputs,
    actual: &GeneralizationBoundInputs,
) -> Result<(), GeneralizationError> {
    if expected.method != actual.method {
        return Err(GeneralizationError::ArithmeticMismatch { field: "method" });
    }
    for (name, lhs, rhs) in [
        (
            "empirical_reward",
            expected.empirical_reward,
            actual.empirical_reward,
        ),
        ("reward_lower", expected.reward_lower, actual.reward_lower),
        ("reward_upper", expected.reward_upper, actual.reward_upper),
        (
            "failure_probability",
            expected.failure_probability,
            actual.failure_probability,
        ),
    ] {
        if !floats_match(lhs, rhs) {
            return Err(GeneralizationError::ArithmeticMismatch { field: name });
        }
    }
    if expected.training_prompt_count != actual.training_prompt_count {
        return Err(GeneralizationError::ArithmeticMismatch {
            field: "training_prompt_count",
        });
    }
    if expected.evaluated_prompt_count != actual.evaluated_prompt_count {
        return Err(GeneralizationError::ArithmeticMismatch {
            field: "evaluated_prompt_count",
        });
    }
    if expected.generations_per_prompt != actual.generations_per_prompt {
        return Err(GeneralizationError::ArithmeticMismatch {
            field: "generations_per_prompt",
        });
    }
    if expected.description_length_bits != actual.description_length_bits {
        return Err(GeneralizationError::ArithmeticMismatch {
            field: "description_length_bits",
        });
    }
    Ok(())
}

fn compare_bound(
    expected: &GeneralizationBound,
    actual: &GeneralizationBound,
) -> Result<(), GeneralizationError> {
    for (name, lhs, rhs) in [
        (
            "complexity_penalty_nats",
            expected.complexity_penalty_nats,
            actual.complexity_penalty_nats,
        ),
        (
            "subsample_penalty",
            expected.subsample_penalty,
            actual.subsample_penalty,
        ),
        (
            "extrinsic_penalty",
            expected.extrinsic_penalty,
            actual.extrinsic_penalty,
        ),
        (
            "intrinsic_penalty",
            expected.intrinsic_penalty,
            actual.intrinsic_penalty,
        ),
        (
            "total_penalty",
            expected.total_penalty,
            actual.total_penalty,
        ),
        (
            "raw_lower_bound",
            expected.raw_lower_bound,
            actual.raw_lower_bound,
        ),
        (
            "certified_lower_bound",
            expected.certified_lower_bound,
            actual.certified_lower_bound,
        ),
        ("confidence", expected.confidence, actual.confidence),
    ] {
        if !floats_match(lhs, rhs) {
            return Err(GeneralizationError::ArithmeticMismatch { field: name });
        }
    }
    Ok(())
}
