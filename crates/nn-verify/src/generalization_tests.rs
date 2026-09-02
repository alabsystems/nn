// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17}, expected={expected:.17}, delta={:.3e}",
        (actual - expected).abs()
    );
}

fn theorem_inputs() -> GeneralizationBoundInputs {
    GeneralizationBoundInputs {
        method: GeneralizationMethod::Theorem1FullTrainingSet,
        empirical_reward: 0.8,
        reward_lower: 0.0,
        reward_upper: 1.0,
        training_prompt_count: 100_000,
        evaluated_prompt_count: 100_000,
        generations_per_prompt: 64,
        description_length_bits: 4_000,
        failure_probability: 0.05,
    }
}

#[test]
fn test_theorem_1_matches_paper_vector() {
    let bound = calculate_generalization_bound(&theorem_inputs()).unwrap();
    assert_close(bound.complexity_penalty_nats, 2792.8657009740991, 1e-10);
    assert!(bound.complexity_penalty_nats >= 2792.8657009740991);
    assert_close(bound.extrinsic_penalty, 0.11817075994031051, 1e-14);
    assert_close(bound.intrinsic_penalty, 0.014771344992538814, 1e-14);
    assert_close(bound.total_penalty, 0.13294210493284933, 1e-14);
    assert_close(bound.raw_lower_bound, 0.66705789506715063, 1e-14);
    assert_close(bound.certified_lower_bound, 0.66705789506715063, 1e-14);
    assert!(bound.is_non_vacuous(0.0));
}

#[test]
fn test_bound_scalars_have_bit_exact_json_roundtrip() {
    let bound = calculate_generalization_bound(&theorem_inputs()).unwrap();
    let json = serde_json::to_string(&bound).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value["extrinsic_penalty"].is_string());

    let decoded: GeneralizationBound = serde_json::from_str(&json).unwrap();
    assert_eq!(
        decoded.extrinsic_penalty.to_bits(),
        bound.extrinsic_penalty.to_bits()
    );
    assert_eq!(
        decoded.raw_lower_bound.to_bits(),
        bound.raw_lower_bound.to_bits()
    );
}

#[test]
fn test_corollary_1_uses_three_over_delta() {
    let inputs = GeneralizationBoundInputs {
        method: GeneralizationMethod::Corollary1UniformSubsample {
            scheme: SubsampleScheme::UniformWithoutReplacement,
        },
        empirical_reward: 0.6,
        reward_lower: 0.0,
        reward_upper: 1.0,
        training_prompt_count: 456_285,
        evaluated_prompt_count: 4_096,
        generations_per_prompt: 64,
        description_length_bits: 7_726,
        failure_probability: 0.05,
    };
    let bound = calculate_generalization_bound(&inputs).unwrap();
    assert_close(bound.complexity_penalty_nats, 5377.254154654812, 1e-10);
    assert!(bound.complexity_penalty_nats >= 5377.254154654812);
    assert_close(bound.subsample_penalty, 0.022356160676492005, 1e-14);
    assert_close(bound.extrinsic_penalty, 0.07676216023149543, 1e-14);
    assert_close(bound.intrinsic_penalty, 0.009595270028936929, 1e-14);
    assert_close(bound.total_penalty, 0.10871359093692436, 1e-14);
    assert_close(bound.raw_lower_bound, 0.49128640906307558, 1e-14);

    // The authors' released helper used ln(2/delta), producing this overly
    // optimistic value. Keep a regression guard around the paper-correct form.
    let artifact_incorrect_lower = 0.49242548945437981;
    assert!(bound.raw_lower_bound < artifact_incorrect_lower - 0.001);
}

#[test]
fn test_bound_clamps_only_to_declared_reward_floor() {
    let inputs = GeneralizationBoundInputs {
        method: GeneralizationMethod::Theorem1FullTrainingSet,
        empirical_reward: 1.0,
        reward_lower: 0.0,
        reward_upper: 1.0,
        training_prompt_count: 1,
        evaluated_prompt_count: 1,
        generations_per_prompt: 1,
        description_length_bits: 1,
        failure_probability: 0.5,
    };
    let bound = calculate_generalization_bound(&inputs).unwrap();
    assert!(bound.raw_lower_bound < 0.0);
    assert_eq!(bound.certified_lower_bound, 0.0);
    assert!(!bound.is_non_vacuous(0.0));
}

#[test]
fn test_subnormal_reward_width_cannot_round_penalty_to_zero() {
    let minimum_positive = f64::from_bits(1);
    let inputs = GeneralizationBoundInputs {
        method: GeneralizationMethod::Theorem1FullTrainingSet,
        empirical_reward: 0.0,
        reward_lower: -minimum_positive,
        reward_upper: 0.0,
        training_prompt_count: 32,
        evaluated_prompt_count: 32,
        generations_per_prompt: 1,
        description_length_bits: 8,
        failure_probability: 0.5,
    };
    let bound = calculate_generalization_bound(&inputs).unwrap();
    assert!(bound.total_penalty > 0.0);
    assert!(bound.raw_lower_bound < 0.0);
    assert_eq!(bound.certified_lower_bound, -minimum_positive);
}

#[test]
fn test_overflowing_reward_width_is_rejected() {
    let mut inputs = theorem_inputs();
    inputs.reward_lower = -f64::MAX;
    inputs.reward_upper = f64::MAX;
    inputs.empirical_reward = 0.0;
    assert!(matches!(
        calculate_generalization_bound(&inputs),
        Err(GeneralizationError::NonFiniteBound {
            field: "reward_width"
        })
    ));
}

#[test]
fn test_smallest_positive_delta_and_maximum_exact_counts_remain_finite() {
    let inputs = GeneralizationBoundInputs {
        method: GeneralizationMethod::Theorem1FullTrainingSet,
        empirical_reward: 0.5,
        reward_lower: 0.0,
        reward_upper: 1.0,
        training_prompt_count: bound::MAX_EXACT_F64_INTEGER,
        evaluated_prompt_count: bound::MAX_EXACT_F64_INTEGER,
        generations_per_prompt: 1,
        description_length_bits: bound::MAX_EXACT_F64_INTEGER,
        failure_probability: f64::from_bits(1),
    };
    let bound = calculate_generalization_bound(&inputs).unwrap();
    assert!(bound.complexity_penalty_nats.is_finite());
    assert!(bound.total_penalty.is_finite());
    assert!(bound.raw_lower_bound.is_finite());
}

#[test]
fn test_bound_monotonic_in_description_length() {
    let small = calculate_generalization_bound(&theorem_inputs()).unwrap();
    let mut larger_inputs = theorem_inputs();
    larger_inputs.description_length_bits = 8_000;
    let large = calculate_generalization_bound(&larger_inputs).unwrap();
    assert!(large.total_penalty > small.total_penalty);
    assert!(large.certified_lower_bound < small.certified_lower_bound);
}

#[test]
fn test_zero_description_length_is_rejected() {
    let mut inputs = theorem_inputs();
    inputs.description_length_bits = 0;
    assert!(matches!(
        calculate_generalization_bound(&inputs),
        Err(GeneralizationError::ZeroDescriptionLength)
    ));
}

fn digest(id: u64) -> String {
    format!("{id:064x}")
}

fn generation(id: u64, reward: f64) -> GenerationReward {
    GenerationReward {
        noise_sha256: digest(10_000 + id),
        generation_sha256: digest(20_000 + id),
        reward,
    }
}

fn random_tape(words: &[u64]) -> String {
    words.iter().map(|word| format!("{word:016x}")).collect()
}

fn full_transcript() -> RewardTranscript {
    RewardTranscript {
        schema_version: REWARD_TRANSCRIPT_VERSION,
        training_prompt_count: 2,
        reward_lower: 0.0,
        reward_upper: 1.0,
        sampling: EvaluationSampling::FullTrainingSet,
        assumptions: GeneralizationAssumptions::asserted_paper_v1(),
        prompts: vec![
            PromptRewardRecord {
                training_index: 0,
                prompt_sha256: digest(1),
                generations: vec![generation(1, 1.0), generation(2, 0.0)],
            },
            PromptRewardRecord {
                training_index: 1,
                prompt_sha256: digest(2),
                generations: vec![generation(3, 1.0), generation(4, 1.0)],
            },
        ],
    }
}

fn subsample_transcript() -> RewardTranscript {
    // For range 10, 0 is rejected and 16 maps to 6. For range 9, 7 maps
    // to 7. Lazy Fisher-Yates therefore selects ordered indices [6, 8].
    let selection_random_tape_hex = random_tape(&[0, 16, 7]);
    let sampling = EvaluationSampling::UniformSubsample {
        scheme: SubsampleScheme::UniformWithoutReplacement,
        selection_algorithm: SUBSAMPLE_ALGORITHM.to_string(),
        selection_random_tape_hex: selection_random_tape_hex.clone(),
        selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source: true,
        selection_random_tape_source_independent_of_training_evaluation_and_hypothesis: true,
        sampling_scheme_and_size_fixed_before_selection: true,
        hypothesis_committed_before_sampling: true,
    };
    let indices = derive_subsample_indices(
        SubsampleScheme::UniformWithoutReplacement,
        selection_random_tape_hex,
        10,
        2,
    )
    .unwrap();
    RewardTranscript {
        schema_version: REWARD_TRANSCRIPT_VERSION,
        training_prompt_count: 10,
        reward_lower: -1.0,
        reward_upper: 1.0,
        sampling,
        assumptions: GeneralizationAssumptions::asserted_paper_v1(),
        prompts: vec![
            PromptRewardRecord {
                training_index: indices[0],
                prompt_sha256: digest(3),
                generations: vec![generation(101, 1.0), generation(102, -1.0)],
            },
            PromptRewardRecord {
                training_index: indices[1],
                prompt_sha256: digest(4),
                generations: vec![generation(103, 0.5), generation(104, 1.0)],
            },
        ],
    }
}

#[test]
fn test_transcript_summary_preserves_prompt_grouping() {
    let summary = full_transcript().summarize().unwrap();
    assert_eq!(summary.evaluated_prompt_count, 2);
    assert_eq!(summary.generations_per_prompt, 2);
    assert_close(summary.empirical_reward, 0.75, 1e-15);
    assert_eq!(
        summary.method,
        GeneralizationMethod::Theorem1FullTrainingSet
    );
}

#[test]
fn test_subsample_requires_hypothesis_precommitment() {
    let mut transcript = subsample_transcript();
    let EvaluationSampling::UniformSubsample {
        hypothesis_committed_before_sampling,
        ..
    } = &mut transcript.sampling
    else {
        unreachable!();
    };
    *hypothesis_committed_before_sampling = false;
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::HypothesisNotPrecommitted)
    ));
}

#[test]
fn test_subsample_indices_are_deterministically_replayed() {
    let transcript = subsample_transcript();
    let actual: Vec<u64> = transcript
        .prompts
        .iter()
        .map(|prompt| prompt.training_index)
        .collect();
    assert_eq!(actual, vec![6, 8]);
    transcript.validate().unwrap();

    let mut changed = transcript;
    changed.prompts[0].training_index = 2;
    assert!(matches!(
        changed.validate(),
        Err(GeneralizationError::SubsampleSelectionMismatch {
            record: 0,
            expected: 6,
            found: 2
        })
    ));
}

#[test]
fn test_subsample_requires_independent_random_source_attestation() {
    let mut transcript = subsample_transcript();
    let EvaluationSampling::UniformSubsample {
        selection_random_tape_source_independent_of_training_evaluation_and_hypothesis,
        ..
    } = &mut transcript.sampling
    else {
        unreachable!();
    };
    *selection_random_tape_source_independent_of_training_evaluation_and_hypothesis = false;
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::SelectionRandomTapeNotIndependent)
    ));
}

#[test]
fn test_subsample_requires_first_iid_uniform_stopped_prefix_attestation() {
    let mut transcript = subsample_transcript();
    let EvaluationSampling::UniformSubsample {
        selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source,
        ..
    } = &mut transcript.sampling
    else {
        unreachable!();
    };
    *selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source = false;
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::SelectionRandomTapeNotFirstIidUniformPrefix)
    ));
}

#[test]
fn test_random_tape_must_end_at_final_accepted_word() {
    assert!(matches!(
        derive_subsample_indices(
            SubsampleScheme::UniformWithoutReplacement,
            "000000000000000A",
            10,
            1,
        ),
        Err(GeneralizationError::InvalidSelectionRandomTape)
    ));
    assert!(matches!(
        derive_subsample_indices(
            SubsampleScheme::UniformWithoutReplacement,
            random_tape(&[16]),
            10,
            2,
        ),
        Err(GeneralizationError::SubsampleRandomTapeExhausted)
    ));
    assert!(matches!(
        derive_subsample_indices(
            SubsampleScheme::UniformWithoutReplacement,
            random_tape(&[16, 7, 42]),
            10,
            2,
        ),
        Err(GeneralizationError::SubsampleRandomTapeHasTrailingWords { remaining_words: 1 })
    ));
}

#[test]
fn test_with_replacement_sampler_and_repeated_generation_reuse() {
    let selection_random_tape_hex = random_tape(&[1, 4, 5]);
    assert_eq!(
        derive_subsample_indices(
            SubsampleScheme::UniformWithReplacement,
            selection_random_tape_hex.clone(),
            3,
            3,
        )
        .unwrap(),
        vec![1, 1, 2]
    );

    let repeated_generations = vec![generation(201, 0.25), generation(202, 0.75)];
    let mut transcript = RewardTranscript {
        schema_version: REWARD_TRANSCRIPT_VERSION,
        training_prompt_count: 3,
        reward_lower: 0.0,
        reward_upper: 1.0,
        sampling: EvaluationSampling::UniformSubsample {
            scheme: SubsampleScheme::UniformWithReplacement,
            selection_algorithm: SUBSAMPLE_ALGORITHM.to_string(),
            selection_random_tape_hex,
            selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source: true,
            selection_random_tape_source_independent_of_training_evaluation_and_hypothesis: true,
            sampling_scheme_and_size_fixed_before_selection: true,
            hypothesis_committed_before_sampling: true,
        },
        assumptions: GeneralizationAssumptions::asserted_paper_v1(),
        prompts: vec![
            PromptRewardRecord {
                training_index: 1,
                prompt_sha256: digest(2),
                generations: repeated_generations.clone(),
            },
            PromptRewardRecord {
                training_index: 1,
                prompt_sha256: digest(2),
                generations: repeated_generations,
            },
            PromptRewardRecord {
                training_index: 2,
                prompt_sha256: digest(3),
                generations: vec![generation(203, 1.0), generation(204, 0.0)],
            },
        ],
    };
    let summary = transcript.summarize().unwrap();
    assert_close(summary.empirical_reward, 0.5, 1e-15);

    transcript.prompts[1].generations[0].reward = 0.5;
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::InconsistentRepeatedSubsampleGenerations { index: 1 })
    ));
}

#[test]
fn test_subsample_rejects_unknown_selection_algorithm() {
    let mut transcript = subsample_transcript();
    let EvaluationSampling::UniformSubsample {
        selection_algorithm,
        ..
    } = &mut transcript.sampling
    else {
        unreachable!();
    };
    *selection_algorithm = "unreviewed-sampler".to_string();
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::UnsupportedSelectionAlgorithm { .. })
    ));
}

#[test]
fn test_without_replacement_rejects_duplicate_index() {
    let mut transcript = subsample_transcript();
    transcript.prompts[1].training_index = transcript.prompts[0].training_index;
    transcript.prompts[1].prompt_sha256 = transcript.prompts[0].prompt_sha256.clone();
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::DuplicateSubsampleIndex { .. })
    ));
}

#[test]
fn test_duplicate_generation_noise_is_rejected() {
    let mut transcript = full_transcript();
    transcript.prompts[1].generations[0].noise_sha256 =
        transcript.prompts[0].generations[0].noise_sha256.clone();
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::DuplicateNoiseCommitment { .. })
    ));
}

#[test]
fn test_missing_assumption_fails_closed() {
    let mut transcript = full_transcript();
    transcript
        .assumptions
        .deployment_distribution_matches_training_distribution = false;
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::MissingAssumption {
            assumption: "deployment_distribution_matches_training_distribution"
        })
    ));
}

#[test]
fn test_post_hoc_protocol_selection_is_not_attested() {
    let mut transcript = full_transcript();
    transcript
        .assumptions
        .reward_spec_bounds_and_failure_probability_fixed_before_training_sample = false;
    assert!(matches!(
        transcript.validate(),
        Err(GeneralizationError::MissingAssumption {
            assumption: "reward_spec_bounds_and_failure_probability_fixed_before_training_sample"
        })
    ));
}

#[test]
fn test_transcript_rejects_unknown_fields() {
    let mut value = serde_json::to_value(full_transcript()).unwrap();
    value.as_object_mut().unwrap().insert(
        "future_security_claim".to_string(),
        serde_json::Value::Bool(true),
    );
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(matches!(
        RewardTranscript::from_json_slice(&bytes),
        Err(GeneralizationError::Serialization(_))
    ));
}

struct FileFixture {
    root: PathBuf,
    paths: EvidenceArtifactPaths,
}

impl FileFixture {
    fn new() -> Self {
        static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
        let fixture_id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "nn-generalization-test-{}-{fixture_id}",
            std::process::id(),
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let names = [
            "base.model",
            "hypothesis.bits",
            "hypothesis-decoder.json",
            "generation.json",
            "reward.json",
            "distribution.json",
            "training-corpus.json",
        ];
        for (index, name) in names.iter().enumerate() {
            let contents = if *name == "hypothesis.bits" {
                vec![0b1010_0101]
            } else if *name == "training-corpus.json" {
                serde_json::to_vec_pretty(&TrainingCorpusManifest {
                    schema_version: TRAINING_CORPUS_MANIFEST_VERSION,
                    prompt_sha256: vec![digest(1), digest(2)],
                })
                .unwrap()
            } else {
                format!("artifact-{index}-{name}").into_bytes()
            };
            std::fs::write(root.join(name), contents).unwrap();
        }
        let paths = EvidenceArtifactPaths::new(
            root.join(names[0]),
            root.join(names[1]),
            root.join(names[2]),
            root.join(names[3]),
            root.join(names[4]),
            root.join(names[5]),
            root.join(names[6]),
        );
        Self { root, paths }
    }
}

impl Drop for FileFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn calculate_fixture_certificate(fixture: &FileFixture) -> (GeneralizationCertificate, Vec<u8>) {
    let transcript = serde_json::to_vec_pretty(&full_transcript()).unwrap();
    let artifacts = EvidenceArtifacts::from_paths(&fixture.paths).unwrap();
    let inputs = GeneralizationCertificateInputs::new(
        artifacts,
        HypothesisEncoding::new("test-prefix-code-v1", 8),
        0.05,
    );
    let manifest = std::fs::read(&fixture.paths.training_corpus_manifest).unwrap();
    let certificate = GeneralizationCertificate::calculate(inputs, &transcript, &manifest).unwrap();
    (certificate, transcript)
}

#[test]
fn test_certificate_roundtrip_save_load_and_replay() {
    let fixture = FileFixture::new();
    let (certificate, transcript) = calculate_fixture_certificate(&fixture);
    certificate.validate().unwrap();
    assert!(certificate.content_hash.is_some());

    let json = certificate.to_json().unwrap();
    let decoded = GeneralizationCertificate::from_json(&json).unwrap();
    assert_eq!(decoded, certificate);

    let certificate_path = fixture.root.join("population-reward.cert.json");
    let legacy_temporary_path = fixture.root.join("population-reward.cert.json.tmp");
    std::fs::write(&legacy_temporary_path, b"unrelated-existing-file").unwrap();
    certificate.save(&certificate_path).unwrap();
    assert_eq!(
        std::fs::read(&legacy_temporary_path).unwrap(),
        b"unrelated-existing-file"
    );
    let loaded = GeneralizationCertificate::load(&certificate_path).unwrap();
    let report = loaded.replay(&transcript, &fixture.paths).unwrap();
    assert!(report.certificate_content_hash_verified);
    assert_eq!(report.evidence_artifacts_verified, 7);
    assert!(report.transcript_verified);
    assert!(report.arithmetic_replayed);
}

#[test]
fn test_replay_rejects_changed_transcript_bytes() {
    let fixture = FileFixture::new();
    let (certificate, mut transcript) = calculate_fixture_certificate(&fixture);
    transcript.push(b' ');
    assert!(matches!(
        certificate.replay(&transcript, &fixture.paths),
        Err(GeneralizationError::ArtifactMismatch {
            artifact: "reward_transcript",
            ..
        })
    ));
}

#[test]
fn test_replay_rejects_changed_evidence_artifact() {
    let fixture = FileFixture::new();
    let (certificate, transcript) = calculate_fixture_certificate(&fixture);
    std::fs::write(&fixture.paths.reward_spec, b"changed-reward-spec").unwrap();
    assert!(matches!(
        certificate.replay(&transcript, &fixture.paths),
        Err(GeneralizationError::ArtifactMismatch {
            artifact: "reward_spec",
            ..
        })
    ));
}

#[test]
fn test_certificate_detects_arithmetic_tampering() {
    let fixture = FileFixture::new();
    let (mut certificate, _) = calculate_fixture_certificate(&fixture);
    certificate.bound.certified_lower_bound = certificate.bound.certified_lower_bound.next_up();
    certificate.content_hash = None;
    certificate.content_hash =
        Some(crate::certificate::integrity::compute_content_hash(&certificate).unwrap());
    assert!(matches!(
        certificate.validate(),
        Err(GeneralizationError::ArithmeticMismatch {
            field: "certified_lower_bound"
        })
    ));
}

#[test]
fn test_certificate_rejects_unknown_fields() {
    let fixture = FileFixture::new();
    let (certificate, _) = calculate_fixture_certificate(&fixture);
    let mut value = serde_json::to_value(certificate).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("claimed_safe".to_string(), serde_json::Value::Bool(true));
    let json = serde_json::to_string(&value).unwrap();
    assert!(matches!(
        GeneralizationCertificate::from_json(&json),
        Err(GeneralizationError::Serialization(_))
    ));
}

#[test]
fn test_transcript_prompt_must_match_corpus_manifest() {
    let fixture = FileFixture::new();
    let manifest = TrainingCorpusManifest {
        schema_version: TRAINING_CORPUS_MANIFEST_VERSION,
        prompt_sha256: vec![digest(1), digest(999)],
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    std::fs::write(&fixture.paths.training_corpus_manifest, &manifest_bytes).unwrap();
    let artifacts = EvidenceArtifacts::from_paths(&fixture.paths).unwrap();
    let inputs = GeneralizationCertificateInputs::new(
        artifacts,
        HypothesisEncoding::new("test-prefix-code-v1", 8),
        0.05,
    );
    let transcript = serde_json::to_vec_pretty(&full_transcript()).unwrap();
    assert!(matches!(
        GeneralizationCertificate::calculate(inputs, &transcript, &manifest_bytes),
        Err(GeneralizationError::TrainingCorpusPromptMismatch { index: 1 })
    ));
}

#[test]
fn test_transcript_count_must_match_corpus_manifest() {
    let transcript = full_transcript();
    let manifest = TrainingCorpusManifest::new(vec![digest(1)]);
    assert!(matches!(
        transcript.validate_against_manifest(&manifest),
        Err(GeneralizationError::TrainingCorpusCountMismatch {
            transcript: 2,
            manifest: 1
        })
    ));
}

#[test]
fn test_certificate_supports_common_hmac_signer() {
    let fixture = FileFixture::new();
    let (mut certificate, _) = calculate_fixture_certificate(&fixture);
    let key = b"generalization-test-key-at-least-32";
    crate::certificate::integrity::sign_certificate(&mut certificate, key).unwrap();
    certificate.validate().unwrap();
    crate::certificate::integrity::verify_signature(&certificate, key).unwrap();
}

#[test]
fn test_zero_description_bits_are_rejected_during_certificate_calculation() {
    let fixture = FileFixture::new();
    let artifacts = EvidenceArtifacts::from_paths(&fixture.paths).unwrap();
    let transcript = serde_json::to_vec(&full_transcript()).unwrap();
    let inputs = GeneralizationCertificateInputs::new(
        artifacts,
        HypothesisEncoding::new("under-counted", 0),
        0.05,
    );
    assert!(matches!(
        GeneralizationCertificate::calculate(
            inputs,
            &transcript,
            &std::fs::read(&fixture.paths.training_corpus_manifest).unwrap(),
        ),
        Err(GeneralizationError::ZeroDescriptionLength)
    ));
}

#[test]
fn test_every_stored_hypothesis_bit_is_charged() {
    let fixture = FileFixture::new();
    let transcript = serde_json::to_vec(&full_transcript()).unwrap();
    let manifest = std::fs::read(&fixture.paths.training_corpus_manifest).unwrap();
    for description_bits in [7, 9] {
        let artifacts = EvidenceArtifacts::from_paths(&fixture.paths).unwrap();
        let inputs = GeneralizationCertificateInputs::new(
            artifacts,
            HypothesisEncoding::new("miscounted", description_bits),
            0.05,
        );
        assert!(matches!(
            GeneralizationCertificate::calculate(inputs, &transcript, &manifest),
            Err(GeneralizationError::InvalidEncodedHypothesisLength {
                description_bits: found,
                capacity_bits: 8
            }) if found == description_bits
        ));
    }
}
