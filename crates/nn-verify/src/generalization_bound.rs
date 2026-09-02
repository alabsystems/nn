// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Paper-faithful bound arithmetic and transcript validation.

use std::collections::{HashMap, HashSet};

use super::{
    ArtifactIdentity, EvaluationSampling, GeneralizationAssumptions, GeneralizationBound,
    GeneralizationBoundInputs, GeneralizationError, GeneralizationMethod, RewardTranscript,
    RewardTranscriptSummary, SubsampleScheme, TrainingCorpusManifest,
    GENERALIZATION_CERTIFICATE_VERSION, REWARD_TRANSCRIPT_VERSION, SUBSAMPLE_ALGORITHM,
    TRAINING_CORPUS_MANIFEST_VERSION,
};

pub(super) const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;
const LOG_SERIES_TERMS: u32 = 32;

pub(super) fn validate_artifact_identity(
    identity: &ArtifactIdentity,
    name: &'static str,
) -> Result<(), GeneralizationError> {
    if identity.byte_len == 0 {
        return Err(GeneralizationError::EmptyArtifact { artifact: name });
    }
    let valid = crate::certificate_types::validate_sha256_hex(&identity.sha256).is_ok()
        && identity.sha256.bytes().all(|b| !b.is_ascii_uppercase());
    if !valid {
        return Err(GeneralizationError::InvalidArtifactHash {
            artifact: name,
            value: identity.sha256.clone(),
        });
    }
    Ok(())
}

fn validate_sha256_value(value: &str, name: &'static str) -> Result<(), GeneralizationError> {
    let identity = ArtifactIdentity {
        sha256: value.to_string(),
        byte_len: 1,
    };
    validate_artifact_identity(&identity, name)
}

pub(super) fn validate_assumptions(
    assumptions: &GeneralizationAssumptions,
) -> Result<(), GeneralizationError> {
    let required = [
        (
            assumptions.training_prompts_iid_from_deployment_distribution,
            "training_prompts_iid_from_deployment_distribution",
        ),
        (
            assumptions.deployment_distribution_matches_training_distribution,
            "deployment_distribution_matches_training_distribution",
        ),
        (
            assumptions.generation_noise_iid_and_independent_of_prompts_training_and_hypothesis,
            "generation_noise_iid_and_independent_of_prompts_training_and_hypothesis",
        ),
        (
            assumptions.generation_noise_distribution_matches_deployment,
            "generation_noise_distribution_matches_deployment",
        ),
        (
            assumptions.generation_deterministic_given_prompt_noise_and_hypothesis,
            "generation_deterministic_given_prompt_noise_and_hypothesis",
        ),
        (
            assumptions.reward_function_deterministic_and_globally_bounded,
            "reward_function_deterministic_and_globally_bounded",
        ),
        (
            assumptions.generation_and_deployment_specs_fixed_before_training_sample,
            "generation_and_deployment_specs_fixed_before_training_sample",
        ),
        (
            assumptions.reward_spec_bounds_and_failure_probability_fixed_before_training_sample,
            "reward_spec_bounds_and_failure_probability_fixed_before_training_sample",
        ),
        (
            assumptions.hypothesis_encoding_scheme_fixed_before_training_sample,
            "hypothesis_encoding_scheme_fixed_before_training_sample",
        ),
        (
            assumptions.hypothesis_encoding_scheme_is_prefix_free,
            "hypothesis_encoding_scheme_is_prefix_free",
        ),
        (
            assumptions.encoded_code_reconstructs_final_hypothesis,
            "encoded_code_reconstructs_final_hypothesis",
        ),
        (
            assumptions.base_model_and_zero_cost_decoder_fixed_before_training_sample,
            "base_model_and_zero_cost_decoder_fixed_before_training_sample",
        ),
    ];
    for (asserted, name) in required {
        if !asserted {
            return Err(GeneralizationError::MissingAssumption { assumption: name });
        }
    }
    Ok(())
}

impl EvaluationSampling {
    pub(super) fn validate(&self) -> Result<(), GeneralizationError> {
        if let Self::UniformSubsample {
            selection_algorithm,
            selection_random_tape_hex,
            selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source,
            selection_random_tape_source_independent_of_training_evaluation_and_hypothesis,
            sampling_scheme_and_size_fixed_before_selection,
            hypothesis_committed_before_sampling,
            ..
        } = self
        {
            if selection_algorithm != SUBSAMPLE_ALGORITHM {
                return Err(GeneralizationError::UnsupportedSelectionAlgorithm {
                    found: selection_algorithm.clone(),
                    expected: SUBSAMPLE_ALGORITHM,
                });
            }
            if !selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source {
                return Err(GeneralizationError::SelectionRandomTapeNotFirstIidUniformPrefix);
            }
            if !selection_random_tape_source_independent_of_training_evaluation_and_hypothesis {
                return Err(GeneralizationError::SelectionRandomTapeNotIndependent);
            }
            if !sampling_scheme_and_size_fixed_before_selection {
                return Err(GeneralizationError::SamplingPlanNotPrecommitted);
            }
            if !hypothesis_committed_before_sampling {
                return Err(GeneralizationError::HypothesisNotPrecommitted);
            }
            parse_selection_random_tape(selection_random_tape_hex)?;
        }
        Ok(())
    }

    pub(super) fn method(&self) -> GeneralizationMethod {
        match self {
            Self::FullTrainingSet => GeneralizationMethod::Theorem1FullTrainingSet,
            Self::UniformSubsample { scheme, .. } => {
                GeneralizationMethod::Corollary1UniformSubsample { scheme: *scheme }
            }
        }
    }
}

pub(super) fn validate_training_corpus_manifest(
    manifest: &TrainingCorpusManifest,
) -> Result<(), GeneralizationError> {
    if manifest.schema_version != TRAINING_CORPUS_MANIFEST_VERSION {
        return Err(
            GeneralizationError::UnsupportedTrainingCorpusManifestVersion {
                found: manifest.schema_version,
                expected: TRAINING_CORPUS_MANIFEST_VERSION,
            },
        );
    }
    if manifest.prompt_sha256.is_empty() {
        return Err(GeneralizationError::EmptyTrainingCorpusManifest);
    }
    let count = u64::try_from(manifest.prompt_sha256.len())
        .map_err(|_| GeneralizationError::TrainingCorpusManifestTooLarge)?;
    if count > MAX_EXACT_F64_INTEGER {
        return Err(GeneralizationError::TrainingCorpusManifestTooLarge);
    }
    for prompt_hash in &manifest.prompt_sha256 {
        validate_sha256_value(prompt_hash, "training_corpus_prompt")?;
    }
    Ok(())
}

pub(super) fn validate_transcript_manifest_link(
    transcript: &RewardTranscript,
    manifest: &TrainingCorpusManifest,
) -> Result<(), GeneralizationError> {
    manifest.validate()?;
    let manifest_count = u64::try_from(manifest.prompt_sha256.len())
        .map_err(|_| GeneralizationError::TrainingCorpusManifestTooLarge)?;
    if transcript.training_prompt_count != manifest_count {
        return Err(GeneralizationError::TrainingCorpusCountMismatch {
            transcript: transcript.training_prompt_count,
            manifest: manifest_count,
        });
    }
    for prompt in &transcript.prompts {
        let index = usize::try_from(prompt.training_index).map_err(|_| {
            GeneralizationError::TrainingCorpusPromptMismatch {
                index: prompt.training_index,
            }
        })?;
        if manifest.prompt_sha256.get(index) != Some(&prompt.prompt_sha256) {
            return Err(GeneralizationError::TrainingCorpusPromptMismatch {
                index: prompt.training_index,
            });
        }
    }
    Ok(())
}

fn parse_selection_random_tape(encoded: &str) -> Result<Vec<u64>, GeneralizationError> {
    if encoded.is_empty()
        || encoded.len() % 16 != 0
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(GeneralizationError::InvalidSelectionRandomTape);
    }
    let bytes = crate::signing_config::hex_decode(encoded)
        .map_err(|_| GeneralizationError::InvalidSelectionRandomTape)?;
    let mut words = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut word = [0_u8; 8];
        word.copy_from_slice(chunk);
        words.push(u64::from_be_bytes(word));
    }
    Ok(words)
}

fn draw_uniform_index(
    random_words: &[u64],
    cursor: &mut usize,
    upper_exclusive: u64,
) -> Result<u64, GeneralizationError> {
    debug_assert!(upper_exclusive > 0);
    // 2^64 mod range. The accepted suffix has size divisible by the range, so
    // modulo maps an IID uniform u64 to an exactly uniform range element.
    let rejection_threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
    loop {
        let candidate = *random_words
            .get(*cursor)
            .ok_or(GeneralizationError::SubsampleRandomTapeExhausted)?;
        *cursor += 1;
        if candidate >= rejection_threshold {
            return Ok(candidate % upper_exclusive);
        }
    }
}

pub(super) fn replay_subsample_indices(
    sampling: &EvaluationSampling,
    training_prompt_count: u64,
    evaluated_prompt_count: u64,
) -> Result<Vec<u64>, GeneralizationError> {
    let EvaluationSampling::UniformSubsample {
        scheme,
        selection_random_tape_hex,
        ..
    } = sampling
    else {
        return Ok(Vec::new());
    };
    let random_words = parse_selection_random_tape(selection_random_tape_hex)?;
    let capacity = usize::try_from(evaluated_prompt_count)
        .map_err(|_| GeneralizationError::SampleCountTooLarge)?;
    let mut selected = Vec::with_capacity(capacity);
    let mut cursor = 0_usize;
    match scheme {
        SubsampleScheme::UniformWithReplacement => {
            for _ in 0..evaluated_prompt_count {
                selected.push(draw_uniform_index(
                    &random_words,
                    &mut cursor,
                    training_prompt_count,
                )?);
            }
        }
        SubsampleScheme::UniformWithoutReplacement => {
            // Lazy Fisher-Yates: O(m0) memory even when the full corpus is
            // large, while retaining an exactly uniform ordered sample.
            let mut swaps = HashMap::<u64, u64>::with_capacity(capacity);
            for position in 0..evaluated_prompt_count {
                let offset = draw_uniform_index(
                    &random_words,
                    &mut cursor,
                    training_prompt_count - position,
                )?;
                let chosen_position = position + offset;
                let chosen_value = swaps
                    .get(&chosen_position)
                    .copied()
                    .unwrap_or(chosen_position);
                let position_value = swaps.get(&position).copied().unwrap_or(position);
                swaps.insert(chosen_position, position_value);
                selected.push(chosen_value);
            }
        }
    }
    if cursor != random_words.len() {
        return Err(GeneralizationError::SubsampleRandomTapeHasTrailingWords {
            remaining_words: random_words.len() - cursor,
        });
    }
    Ok(selected)
}

/// Derive the exact ordered prompt indices required by schema-v1 subsampling.
pub fn derive_subsample_indices(
    scheme: SubsampleScheme,
    selection_random_tape_hex: impl Into<String>,
    training_prompt_count: u64,
    evaluated_prompt_count: u64,
) -> Result<Vec<u64>, GeneralizationError> {
    validate_count(
        training_prompt_count,
        GeneralizationError::InvalidTrainingPromptCount {
            count: training_prompt_count,
        },
    )?;
    validate_count(
        evaluated_prompt_count,
        GeneralizationError::SampleCountTooLarge,
    )?;
    if evaluated_prompt_count > training_prompt_count {
        return Err(GeneralizationError::InvalidSubsampleSize {
            evaluated: evaluated_prompt_count,
            training: training_prompt_count,
        });
    }
    if scheme == SubsampleScheme::UniformWithoutReplacement
        && evaluated_prompt_count == training_prompt_count
    {
        return Err(GeneralizationError::FullSetEncodedAsSubsample);
    }
    let sampling = EvaluationSampling::UniformSubsample {
        scheme,
        selection_algorithm: SUBSAMPLE_ALGORITHM.to_string(),
        selection_random_tape_hex: selection_random_tape_hex.into(),
        selection_random_tape_is_first_stopped_prefix_from_iid_uniform_source: true,
        selection_random_tape_source_independent_of_training_evaluation_and_hypothesis: true,
        sampling_scheme_and_size_fixed_before_selection: true,
        hypothesis_committed_before_sampling: true,
    };
    sampling.validate()?;
    replay_subsample_indices(&sampling, training_prompt_count, evaluated_prompt_count)
}

/// Validate a transcript and conservatively aggregate its empirical reward.
pub(super) fn summarize_transcript(
    transcript: &RewardTranscript,
) -> Result<RewardTranscriptSummary, GeneralizationError> {
    if transcript.schema_version != REWARD_TRANSCRIPT_VERSION {
        return Err(GeneralizationError::UnsupportedTranscriptVersion {
            found: transcript.schema_version,
            expected: REWARD_TRANSCRIPT_VERSION,
        });
    }
    validate_count(
        transcript.training_prompt_count,
        GeneralizationError::InvalidTrainingPromptCount {
            count: transcript.training_prompt_count,
        },
    )?;
    validate_reward_range(transcript.reward_lower, transcript.reward_upper)?;
    transcript.assumptions.validate()?;
    transcript.sampling.validate()?;
    if transcript.prompts.is_empty() {
        return Err(GeneralizationError::EmptyTranscript);
    }

    let evaluated_prompt_count = u64::try_from(transcript.prompts.len())
        .map_err(|_| GeneralizationError::SampleCountTooLarge)?;
    if evaluated_prompt_count > MAX_EXACT_F64_INTEGER {
        return Err(GeneralizationError::SampleCountTooLarge);
    }

    let mut seen_indices = HashSet::with_capacity(transcript.prompts.len());
    let mut prompt_hash_by_index = HashMap::with_capacity(transcript.prompts.len());
    let mut generation_block_by_index: HashMap<u64, &[super::GenerationReward]> =
        HashMap::with_capacity(transcript.prompts.len());
    let mut seen_noise = HashSet::new();
    let mut generations_per_prompt = None;
    let mut reward_sum_lower = 0.0_f64;

    for (record_index, prompt) in transcript.prompts.iter().enumerate() {
        if prompt.training_index >= transcript.training_prompt_count {
            return Err(GeneralizationError::PromptIndexOutOfRange {
                record: record_index,
                index: prompt.training_index,
                training_count: transcript.training_prompt_count,
            });
        }
        validate_sha256_value(&prompt.prompt_sha256, "prompt")?;
        if let Some(previous) = prompt_hash_by_index.get(&prompt.training_index) {
            if previous != &prompt.prompt_sha256 {
                return Err(GeneralizationError::InconsistentPromptHash {
                    index: prompt.training_index,
                });
            }
        } else {
            prompt_hash_by_index.insert(prompt.training_index, prompt.prompt_sha256.clone());
        }

        let first_occurrence = seen_indices.insert(prompt.training_index);
        if matches!(
            transcript.sampling,
            EvaluationSampling::FullTrainingSet
                | EvaluationSampling::UniformSubsample {
                    scheme: SubsampleScheme::UniformWithoutReplacement,
                    ..
                }
        ) && !first_occurrence
        {
            if matches!(transcript.sampling, EvaluationSampling::FullTrainingSet) {
                return Err(GeneralizationError::InvalidFullTrainingCoverage);
            }
            return Err(GeneralizationError::DuplicateSubsampleIndex {
                index: prompt.training_index,
            });
        }

        if prompt.generations.is_empty() {
            return Err(GeneralizationError::EmptyPromptGenerations {
                record: record_index,
            });
        }
        let generation_count = u64::try_from(prompt.generations.len())
            .map_err(|_| GeneralizationError::GenerationCountTooLarge)?;
        if generation_count > MAX_EXACT_F64_INTEGER {
            return Err(GeneralizationError::GenerationCountTooLarge);
        }
        match generations_per_prompt {
            None => generations_per_prompt = Some(generation_count),
            Some(expected) if expected != generation_count => {
                return Err(GeneralizationError::InconsistentGenerationCount {
                    record: record_index,
                    expected,
                    found: generation_count,
                });
            }
            Some(_) => {}
        }

        let reuses_with_replacement = !first_occurrence
            && matches!(
                transcript.sampling,
                EvaluationSampling::UniformSubsample {
                    scheme: SubsampleScheme::UniformWithReplacement,
                    ..
                }
            );
        if reuses_with_replacement {
            if generation_block_by_index
                .get(&prompt.training_index)
                .copied()
                != Some(prompt.generations.as_slice())
            {
                return Err(
                    GeneralizationError::InconsistentRepeatedSubsampleGenerations {
                        index: prompt.training_index,
                    },
                );
            }
        } else {
            generation_block_by_index.insert(prompt.training_index, &prompt.generations);
        }

        for (generation_index, generation) in prompt.generations.iter().enumerate() {
            validate_sha256_value(&generation.noise_sha256, "generation_noise")?;
            validate_sha256_value(&generation.generation_sha256, "generation_output")?;
            if !reuses_with_replacement && !seen_noise.insert(generation.noise_sha256.clone()) {
                return Err(GeneralizationError::DuplicateNoiseCommitment {
                    hash: generation.noise_sha256.clone(),
                });
            }
            if !generation.reward.is_finite()
                || generation.reward < transcript.reward_lower
                || generation.reward > transcript.reward_upper
            {
                return Err(GeneralizationError::RewardOutOfRange {
                    record: record_index,
                    generation: generation_index,
                    value: generation.reward,
                    lower: transcript.reward_lower,
                    upper: transcript.reward_upper,
                });
            }
            reward_sum_lower = add_down(reward_sum_lower, generation.reward);
        }
    }

    match &transcript.sampling {
        EvaluationSampling::FullTrainingSet => {
            if evaluated_prompt_count != transcript.training_prompt_count
                || u64::try_from(seen_indices.len()).ok() != Some(transcript.training_prompt_count)
            {
                return Err(GeneralizationError::InvalidFullTrainingCoverage);
            }
        }
        EvaluationSampling::UniformSubsample { scheme, .. } => {
            if evaluated_prompt_count > transcript.training_prompt_count {
                return Err(GeneralizationError::InvalidSubsampleSize {
                    evaluated: evaluated_prompt_count,
                    training: transcript.training_prompt_count,
                });
            }
            if *scheme == SubsampleScheme::UniformWithoutReplacement
                && evaluated_prompt_count == transcript.training_prompt_count
            {
                return Err(GeneralizationError::FullSetEncodedAsSubsample);
            }
            let expected = replay_subsample_indices(
                &transcript.sampling,
                transcript.training_prompt_count,
                evaluated_prompt_count,
            )?;
            for (record, (prompt, expected_index)) in
                transcript.prompts.iter().zip(expected).enumerate()
            {
                if prompt.training_index != expected_index {
                    return Err(GeneralizationError::SubsampleSelectionMismatch {
                        record,
                        expected: expected_index,
                        found: prompt.training_index,
                    });
                }
            }
        }
    }

    let generations_per_prompt =
        generations_per_prompt.ok_or(GeneralizationError::GenerationCountTooLarge)?;
    let observation_count = evaluated_prompt_count
        .checked_mul(generations_per_prompt)
        .filter(|count| *count <= MAX_EXACT_F64_INTEGER)
        .ok_or(GeneralizationError::SampleCountTooLarge)?;
    let empirical_reward = div_down(reward_sum_lower, observation_count as f64)
        .max(transcript.reward_lower)
        .min(transcript.reward_upper);

    Ok(RewardTranscriptSummary {
        empirical_reward,
        evaluated_prompt_count,
        generations_per_prompt,
        method: transcript.sampling.method(),
    })
}

impl GeneralizationBoundInputs {
    pub(super) fn validate(&self) -> Result<(), GeneralizationError> {
        validate_reward_range(self.reward_lower, self.reward_upper)?;
        if !self.empirical_reward.is_finite()
            || self.empirical_reward < self.reward_lower
            || self.empirical_reward > self.reward_upper
        {
            return Err(GeneralizationError::InvalidEmpiricalReward {
                value: self.empirical_reward,
                lower: self.reward_lower,
                upper: self.reward_upper,
            });
        }
        validate_count(
            self.training_prompt_count,
            GeneralizationError::InvalidTrainingPromptCount {
                count: self.training_prompt_count,
            },
        )?;
        validate_count(
            self.evaluated_prompt_count,
            GeneralizationError::SampleCountTooLarge,
        )?;
        validate_count(
            self.generations_per_prompt,
            GeneralizationError::GenerationCountTooLarge,
        )?;
        if self.description_length_bits == 0 {
            return Err(GeneralizationError::ZeroDescriptionLength);
        }
        if self.description_length_bits > MAX_EXACT_F64_INTEGER {
            return Err(GeneralizationError::DescriptionLengthTooLarge {
                bits: self.description_length_bits,
            });
        }
        if !self.failure_probability.is_finite()
            || self.failure_probability <= 0.0
            || self.failure_probability >= 1.0
        {
            return Err(GeneralizationError::InvalidFailureProbability {
                value: self.failure_probability,
            });
        }
        self.evaluated_prompt_count
            .checked_mul(self.generations_per_prompt)
            .filter(|count| *count <= MAX_EXACT_F64_INTEGER)
            .ok_or(GeneralizationError::SampleCountTooLarge)?;

        match self.method {
            GeneralizationMethod::Theorem1FullTrainingSet => {
                if self.evaluated_prompt_count != self.training_prompt_count {
                    return Err(GeneralizationError::InconsistentBoundMethod);
                }
            }
            GeneralizationMethod::Corollary1UniformSubsample { scheme } => {
                if self.evaluated_prompt_count > self.training_prompt_count {
                    return Err(GeneralizationError::InvalidSubsampleSize {
                        evaluated: self.evaluated_prompt_count,
                        training: self.training_prompt_count,
                    });
                }
                if scheme == SubsampleScheme::UniformWithoutReplacement
                    && self.evaluated_prompt_count == self.training_prompt_count
                {
                    return Err(GeneralizationError::FullSetEncodedAsSubsample);
                }
            }
        }
        Ok(())
    }
}

/// Calculate the paper's Theorem 1 or Corollary 1 bound.
///
/// All logarithms are natural. Corollary 1 deliberately uses `ln(3/delta)`
/// in both its complexity and subsampling terms, matching the paper's TeX and
/// proof rather than the authors' inconsistent released helper script.
pub fn calculate_generalization_bound(
    inputs: &GeneralizationBoundInputs,
) -> Result<GeneralizationBound, GeneralizationError> {
    inputs.validate()?;

    let c = inputs.description_length_bits as f64;
    let log_multiplier = match inputs.method {
        GeneralizationMethod::Theorem1FullTrainingSet => 2.0_f64,
        GeneralizationMethod::Corollary1UniformSubsample { .. } => 3.0_f64,
    };
    // Every transcendental is enclosed by an interval, then subsequent
    // operations select the endpoint that makes the lower bound less
    // optimistic. Evaluating ln(k) - ln(delta) avoids k/delta overflow.
    let (_, ln_multiplier_upper) = ln_bounds(log_multiplier);
    let (ln_delta_lower, _) = ln_bounds(inputs.failure_probability);
    let confidence_term = sub_up(ln_multiplier_upper, ln_delta_lower);
    let (_, ln_two_upper) = ln_bounds(2.0);
    let (_, ln_c_upper) = ln_bounds(c);
    let encoded_bits_term = mul_up(c, ln_two_upper);
    let prefix_overhead_term = mul_up(2.0, ln_c_upper);
    let complexity_penalty_nats = add_up(
        add_up(encoded_bits_term, prefix_overhead_term),
        confidence_term,
    );
    let reward_width = if inputs.reward_upper == inputs.reward_lower {
        0.0
    } else {
        sub_up(inputs.reward_upper, inputs.reward_lower)
    };
    require_finite("reward_width", reward_width)?;
    let m = inputs.training_prompt_count as f64;
    let n = inputs.generations_per_prompt as f64;

    let extrinsic_penalty = if reward_width == 0.0 {
        0.0
    } else {
        mul_up(
            reward_width,
            sqrt_up(div_up(complexity_penalty_nats, mul_down(2.0, m))),
        )
    };
    let intrinsic_penalty = if extrinsic_penalty == 0.0 {
        0.0
    } else {
        div_up(extrinsic_penalty, sqrt_down(n))
    };
    let subsample_penalty = match inputs.method {
        GeneralizationMethod::Theorem1FullTrainingSet => 0.0,
        GeneralizationMethod::Corollary1UniformSubsample { .. } => {
            let m0 = inputs.evaluated_prompt_count as f64;
            let (_, ln_three_upper) = ln_bounds(3.0);
            let confidence = sub_up(ln_three_upper, ln_delta_lower);
            mul_up(reward_width, sqrt_up(div_up(confidence, mul_down(2.0, m0))))
        }
    };
    let total_penalty = if reward_width == 0.0 {
        0.0
    } else {
        add_up(
            add_up(subsample_penalty, extrinsic_penalty),
            intrinsic_penalty,
        )
    };
    let raw_lower_bound = if total_penalty == 0.0 {
        inputs.empirical_reward
    } else {
        sub_down(inputs.empirical_reward, total_penalty)
    };
    let certified_lower_bound = raw_lower_bound.max(inputs.reward_lower);

    let bound = GeneralizationBound {
        complexity_penalty_nats,
        subsample_penalty,
        extrinsic_penalty,
        intrinsic_penalty,
        total_penalty,
        raw_lower_bound,
        certified_lower_bound,
        confidence: sub_down(1.0, inputs.failure_probability),
    };
    validate_finite_bound(&bound)?;
    Ok(bound)
}

fn require_finite(field: &'static str, value: f64) -> Result<(), GeneralizationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GeneralizationError::NonFiniteBound { field })
    }
}

fn validate_finite_bound(bound: &GeneralizationBound) -> Result<(), GeneralizationError> {
    for (field, value) in [
        ("complexity_penalty_nats", bound.complexity_penalty_nats),
        ("subsample_penalty", bound.subsample_penalty),
        ("extrinsic_penalty", bound.extrinsic_penalty),
        ("intrinsic_penalty", bound.intrinsic_penalty),
        ("total_penalty", bound.total_penalty),
        ("raw_lower_bound", bound.raw_lower_bound),
        ("certified_lower_bound", bound.certified_lower_bound),
        ("confidence", bound.confidence),
    ] {
        require_finite(field, value)?;
    }
    Ok(())
}

fn validate_count(count: u64, error: GeneralizationError) -> Result<(), GeneralizationError> {
    if count == 0 || count > MAX_EXACT_F64_INTEGER {
        return Err(error);
    }
    Ok(())
}

fn validate_reward_range(lower: f64, upper: f64) -> Result<(), GeneralizationError> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(GeneralizationError::InvalidRewardRange { lower, upper });
    }
    Ok(())
}

fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else {
        value.next_up()
    }
}

fn next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        value
    } else {
        value.next_down()
    }
}

fn add_up(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 {
        rhs
    } else if rhs == 0.0 {
        lhs
    } else {
        next_up(lhs + rhs)
    }
}

fn mul_up(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 || rhs == 0.0 {
        lhs * rhs
    } else {
        next_up(lhs * rhs)
    }
}

fn mul_down(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 || rhs == 0.0 {
        lhs * rhs
    } else {
        next_down(lhs * rhs)
    }
}

fn div_up(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        numerator / denominator
    } else {
        next_up(numerator / denominator)
    }
}

fn div_down(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        numerator / denominator
    } else {
        next_down(numerator / denominator)
    }
}

fn sub_up(lhs: f64, rhs: f64) -> f64 {
    if rhs == 0.0 {
        lhs
    } else {
        next_up(lhs - rhs)
    }
}

fn sub_down(lhs: f64, rhs: f64) -> f64 {
    if rhs == 0.0 {
        lhs
    } else {
        next_down(lhs - rhs)
    }
}

fn add_down(lhs: f64, rhs: f64) -> f64 {
    if rhs == 0.0 {
        lhs
    } else if lhs == 0.0 {
        rhs
    } else {
        next_down(lhs + rhs)
    }
}

/// Enclose square root without relying on the platform sqrt's error bound.
/// For any positive approximation r, min(r, x/r) <= sqrt(x) <= max(r, x/r).
fn sqrt_bounds(value: f64) -> (f64, f64) {
    if value == 0.0 {
        return (0.0, 0.0);
    }
    let root = value.sqrt();
    let quotient_lower = div_down(value, root);
    let quotient_upper = div_up(value, root);
    (root.min(quotient_lower), root.max(quotient_upper))
}

fn sqrt_up(value: f64) -> f64 {
    sqrt_bounds(value).1
}

fn sqrt_down(value: f64) -> f64 {
    sqrt_bounds(value).0
}

/// Enclose ln(m) for m in [1, 2] using the atanh series and an analytic tail.
fn ln_mantissa_bounds(mantissa: f64) -> (f64, f64) {
    if mantissa == 1.0 {
        return (0.0, 0.0);
    }
    debug_assert!((1.0..=2.0).contains(&mantissa));

    let numerator_lower = sub_down(mantissa, 1.0);
    let numerator_upper = sub_up(mantissa, 1.0);
    let denominator_lower = add_down(mantissa, 1.0);
    let denominator_upper = add_up(mantissa, 1.0);
    let z_lower = div_down(numerator_lower, denominator_upper);
    let z_upper = div_up(numerator_upper, denominator_lower);
    let z_squared_lower = mul_down(z_lower, z_lower);
    let z_squared_upper = mul_up(z_upper, z_upper);

    let mut power_lower = z_lower;
    let mut power_upper = z_upper;
    let mut sum_lower = 0.0;
    let mut sum_upper = 0.0;
    for term in 0..LOG_SERIES_TERMS {
        let denominator = f64::from(2 * term + 1);
        sum_lower = add_down(sum_lower, div_down(power_lower, denominator));
        sum_upper = add_up(sum_upper, div_up(power_upper, denominator));
        power_lower = mul_down(power_lower, z_squared_lower);
        power_upper = mul_up(power_upper, z_squared_upper);
    }

    // Tail <= z^(2N+1) / ((2N+1)(1-z^2)); the factor two is
    // applied after adding the tail to the atanh series.
    let tail_denominator = mul_down(
        f64::from(2 * LOG_SERIES_TERMS + 1),
        sub_down(1.0, z_squared_upper),
    );
    let tail_upper = div_up(power_upper, tail_denominator);
    (
        mul_down(2.0, sum_lower),
        mul_up(2.0, add_up(sum_upper, tail_upper)),
    )
}

/// Decompose a positive finite f64 exactly as mantissa * 2^exponent.
fn normalized_mantissa(value: f64) -> (f64, i32) {
    const EXPONENT_MASK: u64 = 0x7ff;
    const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
    const TWO_POW_54: f64 = 18_014_398_509_481_984.0;

    let mut normalized = value;
    let mut exponent_adjustment = 0_i32;
    if ((value.to_bits() >> 52) & EXPONENT_MASK) == 0 {
        normalized = value * TWO_POW_54;
        exponent_adjustment = -54;
    }
    let bits = normalized.to_bits();
    let biased_exponent = ((bits >> 52) & EXPONENT_MASK) as i32;
    let exponent = biased_exponent - 1023 + exponent_adjustment;
    let mantissa = f64::from_bits((1023_u64 << 52) | (bits & FRACTION_MASK));
    (mantissa, exponent)
}

/// Rigorous binary64 interval enclosing the natural logarithm.
fn ln_bounds(value: f64) -> (f64, f64) {
    debug_assert!(value.is_finite() && value > 0.0);
    if value == 1.0 {
        return (0.0, 0.0);
    }
    let (mantissa, exponent) = normalized_mantissa(value);
    let (mantissa_lower, mantissa_upper) = ln_mantissa_bounds(mantissa);
    let (ln_two_lower, ln_two_upper) = ln_mantissa_bounds(2.0);
    let exponent = f64::from(exponent);
    let (exponent_lower, exponent_upper) = if exponent >= 0.0 {
        (
            mul_down(exponent, ln_two_lower),
            mul_up(exponent, ln_two_upper),
        )
    } else {
        (
            mul_down(exponent, ln_two_upper),
            mul_up(exponent, ln_two_lower),
        )
    };
    (
        add_down(exponent_lower, mantissa_lower),
        add_up(exponent_upper, mantissa_upper),
    )
}

pub(super) fn floats_match(expected: f64, actual: f64) -> bool {
    expected.to_bits() == actual.to_bits()
}

pub(super) fn validate_certificate_version(version: u32) -> Result<(), GeneralizationError> {
    if version != GENERALIZATION_CERTIFICATE_VERSION {
        return Err(GeneralizationError::UnsupportedCertificateVersion {
            found: version,
            expected: GENERALIZATION_CERTIFICATE_VERSION,
        });
    }
    Ok(())
}
