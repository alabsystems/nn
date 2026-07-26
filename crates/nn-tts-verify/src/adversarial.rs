// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Phoneme confusion sets for adversarial robustness verification of TTS.
//!
//! Defines linguistically meaningful perturbation neighborhoods: sets of phonemes
//! that are perceptually confusable. Used to construct tight CROWN verification
//! bounds over specific token subsets rather than the full vocabulary.
//!
//! Based on perceptual confusion matrices (Miller & Nicely, 1955) and
//! articulatory phonetics.

use crate::error::{DspErrorKind, TtsVerifyError};

/// A set of phonemes that could be confused with each other.
/// Represents a perturbation neighborhood in discrete token space.
#[derive(Debug, Clone)]
pub struct ConfusionSet {
    /// Human-readable name (e.g., "voiced_fricatives", "front_vowels").
    pub name: String,
    /// Token IDs that are mutually confusable.
    pub token_ids: Vec<u32>,
    /// Phoneme labels for documentation.
    pub labels: Vec<String>,
    /// Linguistic category (manner, place, voicing, etc.).
    pub category: ConfusionCategory,
}

/// Categories of phoneme confusion.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfusionCategory {
    /// Voicing minimal pairs: /p/ ↔ /b/, /t/ ↔ /d/, /k/ ↔ /g/.
    VoicingPair,
    /// Place of articulation: /t/ ↔ /k/, /n/ ↔ /ŋ/.
    PlaceConfusion,
    /// Manner of articulation: /s/ ↔ /ʃ/, /z/ ↔ /ʒ/.
    MannerConfusion,
    /// Vowel proximity: /ɪ/ ↔ /iː/, /ɛ/ ↔ /æ/.
    VowelProximity,
    /// Cross-language confusables (e.g., English /r/ vs Japanese /ɾ/).
    CrossLanguage,
    /// Adversarial: phonemes with similar embeddings but different sounds.
    EmbeddingSimilar,
}

/// Standard confusion sets for English phonemes (ARPAbet-based).
///
/// Based on perceptual confusion matrices (Miller & Nicely, 1955) and
/// articulatory phonetics. Token IDs correspond to Kokoro's 178-token
/// vocabulary — callers with different vocabularies should use
/// [`discover_confusion_sets`] instead.
///
/// Returns ~15 sets covering voicing pairs, sibilant confusion, nasal place,
/// liquid confusion, and vowel chains.
pub fn english_confusion_sets() -> Vec<ConfusionSet> {
    vec![
        // Voicing pairs (6 sets)
        voicing_pair("stop_p_b", &[("p", 46), ("b", 25)]),
        voicing_pair("stop_t_d", &[("t", 53), ("d", 28)]),
        voicing_pair("stop_k_g", &[("k", 39), ("g", 33)]),
        voicing_pair("fricative_f_v", &[("f", 32), ("v", 56)]),
        voicing_pair("fricative_s_z", &[("s", 50), ("z", 60)]),
        voicing_pair("dental_θ_ð", &[("θ", 59), ("ð", 27)]),
        // Sibilant confusion (2 sets)
        ConfusionSet {
            name: "sibilant_alveolar_postalveolar".into(),
            token_ids: vec![50, 51], // s, ʃ
            labels: vec!["s".into(), "ʃ".into()],
            category: ConfusionCategory::MannerConfusion,
        },
        ConfusionSet {
            name: "sibilant_voiced".into(),
            token_ids: vec![60, 61], // z, ʒ
            labels: vec!["z".into(), "ʒ".into()],
            category: ConfusionCategory::MannerConfusion,
        },
        // Nasal place (1 set)
        ConfusionSet {
            name: "nasals".into(),
            token_ids: vec![42, 43, 44], // m, n, ŋ
            labels: vec!["m".into(), "n".into(), "ŋ".into()],
            category: ConfusionCategory::PlaceConfusion,
        },
        // Liquid confusion (1 set)
        ConfusionSet {
            name: "liquids".into(),
            token_ids: vec![40, 48], // l, ɹ
            labels: vec!["l".into(), "ɹ".into()],
            category: ConfusionCategory::PlaceConfusion,
        },
        // Vowel chains (3 sets)
        ConfusionSet {
            name: "front_vowels".into(),
            token_ids: vec![34, 35, 30, 24], // ɪ, iː, ɛ, æ
            labels: vec!["ɪ".into(), "iː".into(), "ɛ".into(), "æ".into()],
            category: ConfusionCategory::VowelProximity,
        },
        ConfusionSet {
            name: "back_vowels".into(),
            token_ids: vec![55, 54, 45, 23], // ʊ, uː, ɔ, ɑ
            labels: vec!["ʊ".into(), "uː".into(), "ɔ".into(), "ɑ".into()],
            category: ConfusionCategory::VowelProximity,
        },
        ConfusionSet {
            name: "central_vowels".into(),
            token_ids: vec![31, 58, 22], // ə, ʌ, ɐ
            labels: vec!["ə".into(), "ʌ".into(), "ɐ".into()],
            category: ConfusionCategory::VowelProximity,
        },
        // Cross-language (2 sets)
        ConfusionSet {
            name: "rhotic_cross_lang".into(),
            token_ids: vec![48, 47], // ɹ (English), ɾ (Japanese tap)
            labels: vec!["ɹ".into(), "ɾ".into()],
            category: ConfusionCategory::CrossLanguage,
        },
        ConfusionSet {
            name: "lateral_cross_lang".into(),
            token_ids: vec![40, 47], // l (English), ɾ (Korean/Japanese)
            labels: vec!["l".into(), "ɾ".into()],
            category: ConfusionCategory::CrossLanguage,
        },
    ]
}

/// Helper: create a voicing pair confusion set.
fn voicing_pair(name: &str, pairs: &[(&str, u32)]) -> ConfusionSet {
    ConfusionSet {
        name: name.into(),
        token_ids: pairs.iter().map(|(_, id)| *id).collect(),
        labels: pairs.iter().map(|(label, _)| (*label).into()).collect(),
        category: ConfusionCategory::VoicingPair,
    }
}

/// Discover confusion sets from embedding similarity.
///
/// For each token, finds the k nearest neighbors in embedding space
/// (by cosine similarity). Tokens with similarity > threshold form a
/// confusion set. This identifies adversarial perturbations that the
/// model itself considers similar.
///
/// O(vocab_size²) — fine for Kokoro's 178-token vocabulary.
pub fn discover_confusion_sets(
    embedding_weights: &[f64],
    vocab_size: usize,
    embed_dim: usize,
    similarity_threshold: f64,
    max_neighbors: usize,
) -> Result<Vec<ConfusionSet>, TtsVerifyError> {
    if embedding_weights.len() != vocab_size * embed_dim {
        return Err(TtsVerifyError::Dsp(DspErrorKind::SizeMismatch {
            what: "embedding_weights length vs vocab_size * embed_dim",
            expected: vocab_size * embed_dim,
            got: embedding_weights.len(),
        }));
    }
    if vocab_size == 0 || embed_dim == 0 {
        return Ok(Vec::new());
    }

    // Precompute norms for cosine similarity.
    let norms: Vec<f64> = (0..vocab_size)
        .map(|i| {
            let row = &embedding_weights[i * embed_dim..(i + 1) * embed_dim];
            row.iter().map(|x| x * x).sum::<f64>().sqrt()
        })
        .collect();

    // For each token, find high-similarity neighbors.
    let mut visited = vec![false; vocab_size];
    let mut sets = Vec::new();

    for i in 0..vocab_size {
        if visited[i] || norms[i] < 1e-12 {
            continue;
        }

        let row_i = &embedding_weights[i * embed_dim..(i + 1) * embed_dim];
        let mut neighbors: Vec<(u32, f64)> = Vec::new();

        for j in (i + 1)..vocab_size {
            if norms[j] < 1e-12 {
                continue;
            }
            let row_j = &embedding_weights[j * embed_dim..(j + 1) * embed_dim];
            let dot: f64 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            let sim = dot / (norms[i] * norms[j]);
            if sim >= similarity_threshold {
                neighbors.push((j as u32, sim));
            }
        }

        if neighbors.is_empty() {
            continue;
        }

        // Sort by similarity descending, take top max_neighbors.
        neighbors.sort_by(|a, b| b.1.total_cmp(&a.1));
        neighbors.truncate(max_neighbors);

        let mut group_ids = vec![i as u32];
        for (j, _) in &neighbors {
            group_ids.push(*j);
            visited[*j as usize] = true;
        }
        visited[i] = true;

        let labels: Vec<String> = group_ids.iter().map(|id| format!("token_{id}")).collect();

        sets.push(ConfusionSet {
            name: format!("embedding_similar_{i}"),
            token_ids: group_ids,
            labels,
            category: ConfusionCategory::EmbeddingSimilar,
        });
    }

    Ok(sets)
}

/// Compute tight embedding bounds for a specific set of token IDs.
///
/// For each dimension d of the embedding, the bounds are:
///   lower\[d\] = min(embedding\[t\]\[d\] for t in token_ids)
///   upper\[d\] = max(embedding\[t\]\[d\] for t in token_ids)
///
/// Significantly tighter than full-vocabulary bounds when the confusion
/// set is small (e.g., 2-3 tokens).
pub fn embedding_bounds_for_token_set(
    embedding_weights: &[f64],
    vocab_size: usize,
    embed_dim: usize,
    token_ids: &[u32],
) -> Result<(Vec<f64>, Vec<f64>), TtsVerifyError> {
    if embedding_weights.len() != vocab_size * embed_dim {
        return Err(TtsVerifyError::Dsp(DspErrorKind::SizeMismatch {
            what: "embedding_weights length vs vocab_size * embed_dim",
            expected: vocab_size * embed_dim,
            got: embedding_weights.len(),
        }));
    }
    if token_ids.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "token_ids",
        }));
    }
    for &tid in token_ids {
        if (tid as usize) >= vocab_size {
            return Err(TtsVerifyError::Dsp(DspErrorKind::SizeMismatch {
                what: "token_id >= vocab_size",
                expected: vocab_size,
                got: tid as usize,
            }));
        }
    }

    let mut lower = vec![f64::INFINITY; embed_dim];
    let mut upper = vec![f64::NEG_INFINITY; embed_dim];

    for &tid in token_ids {
        let offset = (tid as usize) * embed_dim;
        for d in 0..embed_dim {
            let val = embedding_weights[offset + d];
            if val < lower[d] {
                lower[d] = val;
            }
            if val > upper[d] {
                upper[d] = val;
            }
        }
    }

    Ok((lower, upper))
}

/// Create per-position embedding bounds for a phoneme sequence where
/// specific positions can be perturbed within their confusion sets.
///
/// Fixed positions get point bounds (lower == upper).
/// Perturbed positions get confusion-set bounds.
///
/// Returns `(lower, upper)` each of length `base_tokens.len() * embed_dim`.
pub fn sequence_perturbation_bounds(
    embedding_weights: &[f64],
    vocab_size: usize,
    embed_dim: usize,
    base_tokens: &[u32],
    perturbation_positions: &[usize],
    confusion_sets: &[ConfusionSet],
) -> Result<(Vec<f64>, Vec<f64>), TtsVerifyError> {
    let seq_len = base_tokens.len();
    let total_dim = seq_len * embed_dim;
    let mut lower = Vec::with_capacity(total_dim);
    let mut upper = Vec::with_capacity(total_dim);

    for (pos, &token) in base_tokens.iter().enumerate() {
        if (token as usize) >= vocab_size {
            return Err(TtsVerifyError::Dsp(DspErrorKind::SizeMismatch {
                what: "base_tokens token >= vocab_size",
                expected: vocab_size,
                got: token as usize,
            }));
        }

        if perturbation_positions.contains(&pos) {
            // Find the confusion set containing this token.
            let confusion = confusion_sets
                .iter()
                .find(|cs| cs.token_ids.contains(&token));

            if let Some(cs) = confusion {
                let (lo, hi) = embedding_bounds_for_token_set(
                    embedding_weights,
                    vocab_size,
                    embed_dim,
                    &cs.token_ids,
                )?;
                lower.extend_from_slice(&lo);
                upper.extend_from_slice(&hi);
            } else {
                // Token not in any confusion set — treat as fixed.
                let offset = (token as usize) * embed_dim;
                lower.extend_from_slice(&embedding_weights[offset..offset + embed_dim]);
                upper.extend_from_slice(&embedding_weights[offset..offset + embed_dim]);
            }
        } else {
            // Fixed position: point bounds from the specific token's embedding.
            let offset = (token as usize) * embed_dim;
            lower.extend_from_slice(&embedding_weights[offset..offset + embed_dim]);
            upper.extend_from_slice(&embedding_weights[offset..offset + embed_dim]);
        }
    }

    Ok((lower, upper))
}

#[cfg(test)]
#[path = "adversarial_tests.rs"]
mod tests;
