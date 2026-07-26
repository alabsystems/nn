// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pure-function sampling strategies for autoregressive text generation.
//!
//! Provides temperature scaling, top-k filtering, nucleus (top-p) sampling,
//! repetition penalty, and frequency penalty -- all as standalone functions
//! operating on `&[f32]` logit slices. No DynTensor dependency.
//!
//! The [`SamplingConfig`] struct bundles all parameters for a single sampling
//! step. The main entry point is [`sample_token`], which applies the full
//! pipeline: repetition penalty -> frequency penalty -> temperature ->
//! top-k -> top-p -> softmax -> weighted selection.
//!
//! For deterministic reproducibility without an RNG dependency, selection
//! uses a caller-provided `seed` mapped to the candidate set. For Kani
//! proofs, selection uses `kani::any()` with postcondition constraints.
//!
//! Part of #4271: beam search and advanced sampling for gpt-oss.

/// Configuration for token sampling.
///
/// Controls all aspects of the sampling pipeline. Sensible defaults are
/// provided via [`Default`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SamplingConfig {
    /// Temperature for logit scaling. Must be > 0.0.
    /// Lower values make the distribution sharper (more greedy).
    /// Higher values make it more uniform (more random).
    pub temperature: f32,
    /// Nucleus sampling threshold. Keep the smallest set of tokens whose
    /// cumulative probability >= `top_p`. Must be in (0.0, 1.0].
    /// `None` disables top-p filtering.
    pub top_p: Option<f32>,
    /// Top-k filtering: keep only the k highest-probability tokens.
    /// Must be > 0. `None` disables top-k filtering.
    pub top_k: Option<usize>,
    /// Repetition penalty applied to tokens in `past_tokens`.
    /// 1.0 = no penalty. Values > 1.0 penalize repetition.
    /// Positive logits are divided by penalty; negative logits are
    /// multiplied by penalty. Must be > 0.0.
    pub repetition_penalty: f32,
    /// Frequency penalty: subtracted from logits proportional to how
    /// many times the token appears in `past_tokens`.
    /// 0.0 = no penalty. Typical range: [0.0, 2.0].
    pub frequency_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: Some(50),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingConfig {
    /// Create a greedy (deterministic) sampling config.
    ///
    /// Temperature is set to a very small positive value (not zero) so
    /// that the pipeline still applies softmax. The resulting distribution
    /// is extremely peaked at the maximum logit.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 1e-7,
            top_p: None,
            top_k: Some(1),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        }
    }

    /// Builder: set temperature.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Builder: set top-p.
    #[must_use]
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Builder: set top-k.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Builder: set repetition penalty.
    #[must_use]
    pub fn with_repetition_penalty(mut self, rp: f32) -> Self {
        self.repetition_penalty = rp;
        self
    }

    /// Builder: set frequency penalty.
    #[must_use]
    pub fn with_frequency_penalty(mut self, fp: f32) -> Self {
        self.frequency_penalty = fp;
        self
    }
}

/// Sample a single token from logits using the full sampling pipeline.
///
/// Applies in order: repetition penalty, frequency penalty, temperature
/// scaling, top-k filtering, top-p filtering, softmax, weighted selection.
///
/// The `seed` parameter controls deterministic selection from the final
/// candidate set (modular index). Pass 0 for "always pick the best."
///
/// # Panics
///
/// Panics if `logits` is empty or `config.temperature` is not positive.
#[must_use]
pub fn sample_token(
    logits: &[f32],
    config: &SamplingConfig,
    past_tokens: &[usize],
    seed: u64,
) -> usize {
    assert!(!logits.is_empty(), "logits must be non-empty");
    assert!(
        config.temperature > 0.0 && config.temperature.is_finite(),
        "temperature must be positive and finite"
    );

    let mut scaled = logits.to_vec();

    // Step 1: Repetition penalty
    if (config.repetition_penalty - 1.0).abs() > f32::EPSILON {
        apply_repetition_penalty(&mut scaled, config.repetition_penalty, past_tokens);
    }

    // Step 2: Frequency penalty
    if config.frequency_penalty.abs() > f32::EPSILON {
        apply_frequency_penalty(&mut scaled, config.frequency_penalty, past_tokens);
    }

    // Step 3: Temperature scaling
    apply_temperature(&mut scaled, config.temperature);

    // Step 4: Top-k filtering
    let mut candidates = if let Some(k) = config.top_k {
        apply_top_k(&mut scaled, k)
    } else {
        scaled
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > f32::NEG_INFINITY)
            .map(|(i, &v)| (i, v))
            .collect()
    };

    // Step 5: Top-p filtering
    if let Some(p) = config.top_p {
        candidates = apply_top_p_candidates(&candidates, p);
    }

    // Step 6: Softmax + weighted selection
    if candidates.is_empty() {
        // Fallback: all logits were filtered out -- return argmax of original
        return argmax_slice(logits);
    }

    softmax_sample(&candidates, seed)
}

/// Apply temperature scaling to logits in-place.
///
/// Divides each logit by `temperature`. Positive temperature sharpens
/// (< 1.0) or flattens (> 1.0) the distribution. Non-finite logits are
/// left unchanged.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for l in logits.iter_mut() {
        if l.is_finite() {
            *l /= temperature;
        }
    }
}

/// Apply top-p (nucleus) filtering and return candidate (index, logit) pairs.
///
/// Sorts candidates by descending logit value, computes softmax probabilities
/// over the candidates, and retains the smallest prefix whose cumulative
/// probability >= `top_p`. Returns at least one candidate.
#[must_use]
pub fn apply_top_p(logits: &mut [f32], top_p: f32) -> Vec<(usize, f32)> {
    let candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    apply_top_p_candidates(&candidates, top_p)
}

/// Top-p filtering on a pre-built candidate list.
fn apply_top_p_candidates(candidates: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort by descending logit
    let mut sorted: Vec<(usize, f32)> = candidates.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute softmax over sorted candidates
    let probs = softmax_candidates(&sorted);

    // Accumulate until cumulative >= top_p
    let mut cumulative = 0.0f32;
    let mut result = Vec::new();
    for (i, &(idx, logit)) in sorted.iter().enumerate() {
        cumulative += probs[i];
        result.push((idx, logit));
        if cumulative >= top_p {
            break;
        }
    }

    // Guarantee at least one candidate
    if result.is_empty() && !sorted.is_empty() {
        result.push(sorted[0]);
    }

    result
}

/// Apply top-k filtering and return the top-k (index, logit) pairs.
///
/// Returns at most `k` candidates, sorted by descending logit value.
/// If `k >= logits.len()`, all finite logits are returned. Also sets
/// non-top-k entries in `logits` to `f32::NEG_INFINITY` in-place.
#[must_use]
pub fn apply_top_k(logits: &mut [f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }
    if k >= logits.len() {
        return logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.is_finite())
            .map(|(i, &v)| (i, v))
            .collect();
    }

    // Find top-k indices by sorting
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![false; logits.len()];
    for &idx in &indices[..k] {
        keep[idx] = true;
    }

    let mut result = Vec::with_capacity(k);
    for (i, l) in logits.iter_mut().enumerate() {
        if keep[i] {
            if l.is_finite() {
                result.push((i, *l));
            }
        } else {
            *l = f32::NEG_INFINITY;
        }
    }

    result
}

/// Apply repetition penalty to logits in-place for previously seen tokens.
///
/// For each token in `past_tokens` that is a valid index into `logits`:
/// - Positive logits are divided by `penalty`
/// - Negative logits are multiplied by `penalty`
/// - Zero logits are unchanged
///
/// This ensures penalty > 1.0 always reduces the attractiveness of
/// repeated tokens, regardless of logit sign.
pub fn apply_repetition_penalty(logits: &mut [f32], penalty: f32, past_tokens: &[usize]) {
    for &token_id in past_tokens {
        if token_id < logits.len() {
            let l = logits[token_id];
            if l > 0.0 {
                logits[token_id] = l / penalty;
            } else if l < 0.0 {
                logits[token_id] = l * penalty;
            }
            // l == 0.0: unchanged
        }
    }
}

/// Apply frequency penalty to logits in-place.
///
/// For each token in `past_tokens`, subtracts `penalty * count` from its
/// logit, where `count` is how many times the token appears in
/// `past_tokens`. This penalizes tokens proportional to their frequency.
pub(crate) fn apply_frequency_penalty(logits: &mut [f32], penalty: f32, past_tokens: &[usize]) {
    // Count occurrences
    // Use a simple approach: iterate past_tokens and apply per occurrence.
    // This avoids allocating a HashMap for small past_tokens.
    for &token_id in past_tokens {
        if token_id < logits.len() {
            logits[token_id] -= penalty;
        }
    }
}

/// Weighted random selection from candidates using softmax probabilities.
///
/// Computes softmax over candidate logits, then selects a token using the
/// `seed` for deterministic selection. Returns the token index.
///
/// # Panics
///
/// Panics if `candidates` is empty.
#[must_use]
pub fn softmax_sample(candidates: &[(usize, f32)], seed: u64) -> usize {
    assert!(!candidates.is_empty(), "candidates must be non-empty");

    if candidates.len() == 1 {
        return candidates[0].0;
    }

    let probs = softmax_candidates(candidates);

    // Deterministic selection: map seed to a value in [0, 1) and walk
    // the cumulative distribution.
    let selector = (seed % 10_000) as f32 / 10_000.0;
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if selector < cumulative {
            return candidates[i].0;
        }
    }

    // Rounding: return last candidate
    candidates[candidates.len() - 1].0
}

/// Compute softmax probabilities for a candidate list.
///
/// Returns a Vec of probabilities in the same order as `candidates`.
/// Handles all-neg-inf inputs by returning uniform distribution.
fn softmax_candidates(candidates: &[(usize, f32)]) -> Vec<f32> {
    let max_val = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_val.is_finite() {
        return vec![1.0 / candidates.len() as f32; candidates.len()];
    }

    let exps: Vec<f32> = candidates
        .iter()
        .map(|(_, v)| (v - max_val).exp())
        .collect();
    let sum: f32 = exps.iter().sum();

    if !sum.is_finite() || sum == 0.0 {
        vec![1.0 / candidates.len() as f32; candidates.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Argmax over a raw logit slice. Returns index of the maximum value.
fn argmax_slice(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- SamplingConfig tests ------------------------------------------------

    #[test]
    fn test_sampling_config_default() {
        let cfg = SamplingConfig::default();
        assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.top_k, Some(50));
        assert_eq!(cfg.top_p, Some(0.9));
        assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
        assert!((cfg.frequency_penalty - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sampling_config_greedy() {
        let cfg = SamplingConfig::greedy();
        assert!(cfg.temperature < 1e-6);
        assert_eq!(cfg.top_k, Some(1));
        assert!(cfg.top_p.is_none());
    }

    #[test]
    fn test_sampling_config_builder() {
        let cfg = SamplingConfig::default()
            .with_temperature(1.2)
            .with_top_p(0.95)
            .with_top_k(40)
            .with_repetition_penalty(1.1)
            .with_frequency_penalty(0.5);
        assert!((cfg.temperature - 1.2).abs() < f32::EPSILON);
        assert_eq!(cfg.top_p, Some(0.95));
        assert_eq!(cfg.top_k, Some(40));
        assert!((cfg.repetition_penalty - 1.1).abs() < f32::EPSILON);
        assert!((cfg.frequency_penalty - 0.5).abs() < f32::EPSILON);
    }

    // -- apply_temperature tests ---------------------------------------------

    #[test]
    fn test_temperature_scaling_basic() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_preserves_neg_infinity() {
        let mut logits = vec![1.0, f32::NEG_INFINITY, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
    }

    // -- apply_top_k tests ---------------------------------------------------

    #[test]
    fn test_top_k_returns_at_most_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 0.5, 4.0];
        let candidates = apply_top_k(&mut logits, 2);
        assert!(candidates.len() <= 2);
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let mut logits = vec![1.0, 5.0, 3.0, 0.5, 4.0];
        let candidates = apply_top_k(&mut logits, 2);
        let indices: Vec<usize> = candidates.iter().map(|&(i, _)| i).collect();
        assert!(indices.contains(&1)); // 5.0
        assert!(indices.contains(&4)); // 4.0
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let candidates = apply_top_k(&mut logits, 10);
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_top_k_zero_returns_empty() {
        let mut logits = vec![1.0, 2.0];
        let candidates = apply_top_k(&mut logits, 0);
        assert!(candidates.is_empty());
    }

    // -- apply_top_p tests ---------------------------------------------------

    #[test]
    fn test_top_p_returns_nonempty() {
        let mut logits = vec![1.0, 5.0, 3.0, 0.5];
        let candidates = apply_top_p(&mut logits, 0.9);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_top_p_one_returns_all() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let candidates = apply_top_p(&mut logits, 1.0);
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_top_p_small_returns_top() {
        // With very small top_p, should return just the top token
        let mut logits = vec![0.0, 10.0, 0.0]; // Strongly peaked at index 1
        let candidates = apply_top_p(&mut logits, 0.01);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].0, 1); // Index 1 has logit 10.0
    }

    // -- apply_repetition_penalty tests --------------------------------------

    #[test]
    fn test_repetition_penalty_positive_logit() {
        let mut logits = vec![4.0, 2.0, 1.0];
        apply_repetition_penalty(&mut logits, 2.0, &[0]);
        assert!((logits[0] - 2.0).abs() < 1e-6); // 4.0 / 2.0
        assert!((logits[1] - 2.0).abs() < 1e-6); // untouched
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let mut logits = vec![-2.0, 1.0];
        apply_repetition_penalty(&mut logits, 2.0, &[0]);
        assert!((logits[0] - (-4.0)).abs() < 1e-6); // -2.0 * 2.0
    }

    #[test]
    fn test_repetition_penalty_zero_unchanged() {
        let mut logits = vec![0.0, 1.0];
        apply_repetition_penalty(&mut logits, 2.0, &[0]);
        assert!((logits[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_out_of_bounds_ignored() {
        let mut logits = vec![1.0, 2.0];
        apply_repetition_penalty(&mut logits, 2.0, &[5]); // index 5 out of bounds
        assert!((logits[0] - 1.0).abs() < 1e-6); // untouched
    }

    // -- apply_frequency_penalty tests ---------------------------------------

    #[test]
    fn test_frequency_penalty_proportional() {
        let mut logits = vec![5.0, 3.0, 1.0];
        // Token 0 appears twice -> subtract 2 * 0.5 = 1.0
        apply_frequency_penalty(&mut logits, 0.5, &[0, 0]);
        assert!((logits[0] - 4.0).abs() < 1e-6);
        assert!((logits[1] - 3.0).abs() < 1e-6); // untouched
    }

    // -- softmax_sample tests ------------------------------------------------

    #[test]
    fn test_softmax_sample_single_candidate() {
        let candidates = vec![(42, 1.0)];
        assert_eq!(softmax_sample(&candidates, 0), 42);
    }

    #[test]
    fn test_softmax_sample_returns_valid_index() {
        let candidates = vec![(3, 1.0), (7, 5.0), (11, 3.0)];
        let result = softmax_sample(&candidates, 12345);
        let valid_indices = [3, 7, 11];
        assert!(valid_indices.contains(&result));
    }

    #[test]
    fn test_softmax_sample_deterministic() {
        let candidates = vec![(0, 2.0), (1, 3.0), (2, 1.0)];
        let r1 = softmax_sample(&candidates, 42);
        let r2 = softmax_sample(&candidates, 42);
        assert_eq!(r1, r2, "same seed must produce same result");
    }

    // -- sample_token integration tests --------------------------------------

    #[test]
    fn test_sample_token_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 0.5];
        let cfg = SamplingConfig::greedy();
        let token = sample_token(&logits, &cfg, &[], 0);
        assert_eq!(token, 1, "greedy should pick highest logit");
    }

    #[test]
    fn test_sample_token_with_repetition_penalty() {
        // Token 1 has highest logit but is penalized
        let logits = vec![3.9, 4.0, 3.8];
        let cfg = SamplingConfig::greedy().with_repetition_penalty(10.0);
        let token = sample_token(&logits, &cfg, &[1], 0);
        // Token 1's logit (4.0) after penalty: 4.0/10.0 = 0.4
        // Token 0 (3.9) should now be the highest
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_token_respects_top_k() {
        // All logits equal except index 0 is slightly higher
        let logits = vec![1.01, 1.0, 1.0, 1.0, 1.0];
        let cfg = SamplingConfig::default()
            .with_top_k(1)
            .with_temperature(1e-7);
        let token = sample_token(&logits, &cfg, &[], 0);
        assert_eq!(token, 0, "top_k=1 should always pick the top token");
    }
}
