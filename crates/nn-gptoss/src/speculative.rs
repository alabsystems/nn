// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Speculative decoding for gpt-oss-20b.
//!
//! Uses a smaller draft model to generate candidate tokens, then the full
//! 20B model verifies all candidates in a single forward pass. Accepted
//! tokens advance the sequence; rejected tokens trigger a fallback sample.
//!
//! Key parameters:
//! - `gamma` (speculation length): number of tokens the draft model proposes
//! - Acceptance: token accepted if target prob >= draft prob at that position
//! - Fallback: on rejection, sample from (target - draft) distribution
//!
//! # References
//!
//! - Leviathan et al., "Fast Inference from Transformers via Speculative
//!   Decoding" (ICML 2023)
//! - Chen et al., "Accelerating Large Language Model Decoding with
//!   Speculative Sampling" (2023)

// -- Configuration -----------------------------------------------------------

/// Configuration for speculative decoding.
///
/// Controls the draft model's speculation length (`gamma`) and adaptive
/// behaviour. Defaults are tuned for a 1B-parameter draft model paired
/// with the full 20B target.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per speculative step.
    /// Higher values amortize the overhead of the verification forward
    /// pass but increase wasted work when acceptance rates are low.
    pub gamma: usize,
    /// Temperature applied to the draft model's logits before sampling.
    /// 1.0 = use the draft model's native distribution.
    pub draft_temperature: f64,
    /// Temperature applied to the target model's logits during
    /// verification. 1.0 = use the target model's native distribution.
    pub target_temperature: f64,
    /// After this many consecutive steps with zero accepted tokens,
    /// adaptive gamma reduces the speculation length. Only effective
    /// when `adaptive_gamma` is true.
    pub max_draft_misses: usize,
    /// Enable adaptive gamma adjustment based on recent acceptance rates.
    /// When true, gamma is increased after high-acceptance steps and
    /// decreased after low-acceptance steps.
    pub adaptive_gamma: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            gamma: 5,
            draft_temperature: 1.0,
            target_temperature: 1.0,
            max_draft_misses: 3,
            adaptive_gamma: true,
        }
    }
}

impl SpeculativeConfig {
    /// Create a config with a specific gamma and defaults for the rest.
    #[must_use]
    pub fn with_gamma(gamma: usize) -> Self {
        Self {
            gamma,
            ..Self::default()
        }
    }

    /// Builder: set draft temperature.
    #[must_use]
    pub fn set_draft_temperature(mut self, t: f64) -> Self {
        self.draft_temperature = t;
        self
    }

    /// Builder: set target temperature.
    #[must_use]
    pub fn set_target_temperature(mut self, t: f64) -> Self {
        self.target_temperature = t;
        self
    }

    /// Builder: enable or disable adaptive gamma.
    #[must_use]
    pub fn set_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive_gamma = adaptive;
        self
    }
}

// -- Draft proposal ----------------------------------------------------------

/// A draft model's proposed token sequence.
///
/// Contains the token IDs sampled by the draft model along with the
/// log-probability of each token under the draft distribution. These
/// log-probs are used during verification to compute acceptance ratios.
#[derive(Debug, Clone)]
pub struct DraftProposal {
    /// Proposed token IDs, in order of generation.
    pub tokens: Vec<usize>,
    /// Log-probability of each proposed token under the draft model.
    /// `log_probs[i]` is the log-prob of `tokens[i]`.
    pub log_probs: Vec<f64>,
}

impl DraftProposal {
    /// Create a new draft proposal.
    ///
    /// # Panics
    ///
    /// Panics if `tokens.len() != log_probs.len()`.
    #[must_use]
    pub fn new(tokens: Vec<usize>, log_probs: Vec<f64>) -> Self {
        assert_eq!(
            tokens.len(),
            log_probs.len(),
            "tokens and log_probs must have the same length"
        );
        Self { tokens, log_probs }
    }

    /// Number of proposed tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the proposal is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// -- Speculative step result -------------------------------------------------

/// Result of one speculative decoding step.
///
/// After the target model verifies a [`DraftProposal`], this struct
/// records which tokens were accepted, whether a bonus token was
/// sampled, and the acceptance rate for statistics.
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// Token IDs accepted by the target model, in generation order.
    /// These tokens are appended to the output sequence.
    pub accepted_tokens: Vec<usize>,
    /// An extra token sampled from the target model at the position
    /// after all draft tokens, when all `gamma` tokens are accepted.
    /// This is the "bonus" that makes speculative decoding produce
    /// `gamma + 1` tokens in the best case.
    pub bonus_token: Option<usize>,
    /// Total number of tokens proposed by the draft model in this step.
    pub total_proposed: usize,
    /// Fraction of proposed tokens that were accepted: `accepted / proposed`.
    /// 0.0 when `total_proposed == 0`.
    pub acceptance_rate: f64,
}

impl SpeculativeStep {
    /// Total tokens produced in this step (accepted + optional bonus).
    #[must_use]
    pub fn tokens_produced(&self) -> usize {
        self.accepted_tokens.len() + usize::from(self.bonus_token.is_some())
    }
}

// -- Core verification algorithm ---------------------------------------------

/// Verify a draft proposal against the target model's log-probabilities.
///
/// Implements the stochastic verification from Leviathan et al. (2023):
///
/// For each position `i` in `0..gamma`:
/// 1. Compute `p_target = exp(target_log_probs[i])` and
///    `p_draft = exp(draft.log_probs[i])`.
/// 2. Accept if `rand < min(1, p_target / p_draft)` where `rand` is
///    drawn from a uniform `[0, 1)` using the `rand_values` slice.
/// 3. On rejection: sample a fallback token from
///    `max(0, p_target - p_draft)` (approximated here as the target
///    token at that position) and stop.
///
/// If all tokens are accepted, a bonus token is sampled from the target
/// distribution at position `gamma` (represented by `bonus_token_id`).
///
/// # Arguments
///
/// * `draft` - The draft model's proposal (tokens + log-probs).
/// * `target_log_probs` - Log-probabilities from the target model at
///   each draft position. Must have length >= `gamma`.
/// * `gamma` - Number of draft tokens to verify (may be less than
///   `draft.len()`).
/// * `rand_values` - Uniform random values in `[0, 1)` for acceptance
///   decisions. Must have length >= `gamma`.
/// * `fallback_token_ids` - Token IDs to use when rejection happens at
///   position `i`. Sampled from `max(0, p_target - p_draft)`. Must have
///   length >= `gamma`.
/// * `bonus_token_id` - Token sampled from target at position `gamma`
///   (used only when all tokens are accepted).
pub fn verify_draft(
    draft: &DraftProposal,
    target_log_probs: &[f64],
    gamma: usize,
    rand_values: &[f64],
    fallback_token_ids: &[usize],
    bonus_token_id: Option<usize>,
) -> SpeculativeStep {
    let effective_gamma = gamma.min(draft.len());

    if effective_gamma == 0 {
        return SpeculativeStep {
            accepted_tokens: Vec::new(),
            bonus_token: bonus_token_id,
            total_proposed: 0,
            acceptance_rate: 0.0,
        };
    }

    let mut accepted = Vec::with_capacity(effective_gamma);

    for i in 0..effective_gamma {
        let p_target = target_log_probs[i].exp();
        let p_draft = draft.log_probs[i].exp();

        // Acceptance probability: min(1, p_target / p_draft)
        let accept_prob = if p_draft > 0.0 {
            (p_target / p_draft).min(1.0)
        } else {
            // Draft assigned zero probability: always accept if target > 0.
            if p_target > 0.0 {
                1.0
            } else {
                0.0
            }
        };

        if rand_values[i] < accept_prob {
            accepted.push(draft.tokens[i]);
        } else {
            // Rejection: use the fallback token and stop.
            if i < fallback_token_ids.len() {
                accepted.push(fallback_token_ids[i]);
            }
            let acceptance_rate = if effective_gamma > 0 {
                // Count only the tokens accepted from the draft (excluding fallback).
                (accepted.len().saturating_sub(1)) as f64 / effective_gamma as f64
            } else {
                0.0
            };
            return SpeculativeStep {
                accepted_tokens: accepted,
                bonus_token: None,
                total_proposed: effective_gamma,
                acceptance_rate,
            };
        }
    }

    // All tokens accepted: include bonus token.
    let acceptance_rate = accepted.len() as f64 / effective_gamma as f64;
    SpeculativeStep {
        accepted_tokens: accepted,
        bonus_token: bonus_token_id,
        total_proposed: effective_gamma,
        acceptance_rate,
    }
}

/// Deterministic verification for testing: accept if target_prob >= draft_prob.
///
/// No randomness involved. A token is accepted when the target model
/// assigns at least as much probability as the draft model. This is
/// useful for unit tests and debugging.
pub fn verify_draft_deterministic(
    draft: &DraftProposal,
    target_log_probs: &[f64],
) -> SpeculativeStep {
    let gamma = draft.len().min(target_log_probs.len());

    if gamma == 0 {
        return SpeculativeStep {
            accepted_tokens: Vec::new(),
            bonus_token: None,
            total_proposed: 0,
            acceptance_rate: 0.0,
        };
    }

    let mut accepted = Vec::with_capacity(gamma);

    for i in 0..gamma {
        let p_target = target_log_probs[i].exp();
        let p_draft = draft.log_probs[i].exp();

        if p_target >= p_draft {
            accepted.push(draft.tokens[i]);
        } else {
            // Rejection: stop here, no fallback token in deterministic mode.
            let acceptance_rate = accepted.len() as f64 / gamma as f64;
            return SpeculativeStep {
                accepted_tokens: accepted,
                bonus_token: None,
                total_proposed: gamma,
                acceptance_rate,
            };
        }
    }

    // All accepted: bonus token from target at position gamma.
    // In deterministic mode we don't have an actual bonus token to sample,
    // so we signal full acceptance with a sentinel.
    let acceptance_rate = accepted.len() as f64 / gamma as f64;
    SpeculativeStep {
        accepted_tokens: accepted,
        bonus_token: Some(usize::MAX), // sentinel: caller replaces with real sample
        total_proposed: gamma,
        acceptance_rate,
    }
}

// -- Adaptive gamma ----------------------------------------------------------

/// Adaptive speculation length controller.
///
/// Adjusts the number of draft tokens (`gamma`) based on recent
/// acceptance rates. High acceptance -> increase gamma (speculate more).
/// Low acceptance -> decrease gamma (speculate less).
#[derive(Debug, Clone)]
pub struct AdaptiveGamma {
    /// Current speculation length.
    current_gamma: usize,
    /// Minimum allowed gamma (inclusive).
    min_gamma: usize,
    /// Maximum allowed gamma (inclusive).
    max_gamma: usize,
    /// Threshold above which gamma is increased.
    increase_threshold: f64,
    /// Threshold below which gamma is decreased.
    decrease_threshold: f64,
    /// Number of consecutive steps with zero accepted tokens.
    consecutive_misses: usize,
    /// Maximum consecutive misses before forcing gamma to min.
    max_consecutive_misses: usize,
}

impl AdaptiveGamma {
    /// Create a new adaptive gamma controller.
    ///
    /// # Arguments
    ///
    /// * `initial_gamma` - Starting speculation length.
    /// * `min_gamma` - Minimum bound (clamped to 1).
    /// * `max_gamma` - Maximum bound (must be >= min_gamma).
    /// * `max_consecutive_misses` - Misses before forcing gamma to min.
    #[must_use]
    pub fn new(
        initial_gamma: usize,
        min_gamma: usize,
        max_gamma: usize,
        max_consecutive_misses: usize,
    ) -> Self {
        let min_gamma = min_gamma.max(1);
        let max_gamma = max_gamma.max(min_gamma);
        let initial_gamma = initial_gamma.clamp(min_gamma, max_gamma);
        Self {
            current_gamma: initial_gamma,
            min_gamma,
            max_gamma,
            increase_threshold: 0.8,
            decrease_threshold: 0.3,
            consecutive_misses: 0,
            max_consecutive_misses,
        }
    }

    /// Create with default parameters: gamma in [1, 8], starting at 5,
    /// max 3 consecutive misses.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(5, 1, 8, 3)
    }

    /// Update gamma based on the result of a speculative step.
    ///
    /// - Acceptance rate >= 0.8: increase gamma by 1 (up to max).
    /// - Acceptance rate <= 0.3: decrease gamma by 1 (down to min).
    /// - Zero accepted tokens: increment miss counter; force min after
    ///   `max_consecutive_misses`.
    pub fn update(&mut self, step: &SpeculativeStep) {
        if step.total_proposed == 0 {
            return;
        }

        if step.accepted_tokens.is_empty() {
            self.consecutive_misses += 1;
            if self.consecutive_misses >= self.max_consecutive_misses {
                self.current_gamma = self.min_gamma;
            } else {
                self.current_gamma = self.current_gamma.saturating_sub(1).max(self.min_gamma);
            }
            return;
        }

        // Reset miss counter on any acceptance.
        self.consecutive_misses = 0;

        if step.acceptance_rate >= self.increase_threshold {
            self.current_gamma = (self.current_gamma + 1).min(self.max_gamma);
        } else if step.acceptance_rate <= self.decrease_threshold {
            self.current_gamma = self.current_gamma.saturating_sub(1).max(self.min_gamma);
        }
    }

    /// Current speculation length.
    #[must_use]
    pub fn get(&self) -> usize {
        self.current_gamma
    }

    /// Minimum gamma bound.
    #[must_use]
    pub fn min_gamma(&self) -> usize {
        self.min_gamma
    }

    /// Maximum gamma bound.
    #[must_use]
    pub fn max_gamma(&self) -> usize {
        self.max_gamma
    }
}

// -- Cumulative statistics ---------------------------------------------------

/// Cumulative statistics for speculative decoding across multiple steps.
///
/// Tracks total proposed/accepted tokens and step count to compute
/// average acceptance rate and effective speedup.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total number of tokens proposed by the draft model.
    pub total_proposed: usize,
    /// Total number of tokens accepted by the target model.
    pub total_accepted: usize,
    /// Total number of speculative steps performed.
    pub total_steps: usize,
}

impl SpeculativeStats {
    /// Create empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the result of a speculative step.
    pub fn record(&mut self, step: &SpeculativeStep) {
        self.total_proposed += step.total_proposed;
        self.total_accepted += step.accepted_tokens.len();
        self.total_steps += 1;
    }

    /// Average acceptance rate across all steps.
    /// Returns 0.0 when no tokens have been proposed.
    #[must_use]
    pub fn average_acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_proposed as f64
    }

    /// Effective speedup: total accepted tokens / total steps.
    ///
    /// In the ideal case (all tokens accepted every step), this equals
    /// gamma. In the worst case (all rejected), this is close to 0.
    /// Returns 0.0 when no steps have been performed.
    #[must_use]
    pub fn effective_speedup(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_steps as f64
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SpeculativeConfig tests ---------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.gamma, 5);
        assert!((cfg.draft_temperature - 1.0).abs() < f64::EPSILON);
        assert!((cfg.target_temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_draft_misses, 3);
        assert!(cfg.adaptive_gamma);
    }

    #[test]
    fn test_config_with_gamma() {
        let cfg = SpeculativeConfig::with_gamma(8);
        assert_eq!(cfg.gamma, 8);
        assert!(cfg.adaptive_gamma); // default preserved
    }

    #[test]
    fn test_config_builders() {
        let cfg = SpeculativeConfig::with_gamma(3)
            .set_draft_temperature(0.8)
            .set_target_temperature(0.9)
            .set_adaptive(false);
        assert_eq!(cfg.gamma, 3);
        assert!((cfg.draft_temperature - 0.8).abs() < f64::EPSILON);
        assert!((cfg.target_temperature - 0.9).abs() < f64::EPSILON);
        assert!(!cfg.adaptive_gamma);
    }

    // -- DraftProposal tests -------------------------------------------------

    #[test]
    fn test_draft_proposal_new() {
        let dp = DraftProposal::new(vec![10, 20, 30], vec![-1.0, -2.0, -0.5]);
        assert_eq!(dp.len(), 3);
        assert!(!dp.is_empty());
        assert_eq!(dp.tokens, vec![10, 20, 30]);
    }

    #[test]
    fn test_draft_proposal_empty() {
        let dp = DraftProposal::new(vec![], vec![]);
        assert!(dp.is_empty());
        assert_eq!(dp.len(), 0);
    }

    #[test]
    #[should_panic(expected = "tokens and log_probs must have the same length")]
    fn test_draft_proposal_mismatched_lengths() {
        let _ = DraftProposal::new(vec![1, 2], vec![-1.0]);
    }

    // -- verify_draft_deterministic tests ------------------------------------

    #[test]
    fn test_deterministic_all_accepted() {
        // Target probs >= draft probs at every position.
        // Draft: p=0.2, 0.1, 0.3 => log_prob = ln(0.2), ln(0.1), ln(0.3)
        // Target: p=0.5, 0.4, 0.6 => all >= draft
        let draft = DraftProposal::new(
            vec![100, 200, 300],
            vec![(0.2_f64).ln(), (0.1_f64).ln(), (0.3_f64).ln()],
        );
        let target_lp = vec![(0.5_f64).ln(), (0.4_f64).ln(), (0.6_f64).ln()];
        let step = verify_draft_deterministic(&draft, &target_lp);
        assert_eq!(step.accepted_tokens, vec![100, 200, 300]);
        assert!(step.bonus_token.is_some(), "all accepted => bonus token");
        assert_eq!(step.total_proposed, 3);
        assert!((step.acceptance_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deterministic_first_rejected() {
        // draft log_prob = ln(1) = 0 => p_draft = 1.0
        // target log_prob = ln(exp(-0.5)) => p_target = exp(-0.5) ~ 0.607 < 1.0
        let draft = DraftProposal::new(vec![10, 20], vec![0.0, 0.0]);
        let target_lp = vec![(-0.5_f64).exp().ln(), 0.0];
        // p_draft[0] = exp(0) = 1.0, p_target[0] = exp(ln(exp(-0.5))) = exp(-0.5) ~ 0.607
        // 0.607 < 1.0 => reject at position 0
        let step = verify_draft_deterministic(&draft, &target_lp);
        assert!(
            step.accepted_tokens.is_empty(),
            "first token rejected => no accepted"
        );
        assert!(step.bonus_token.is_none());
        assert_eq!(step.total_proposed, 2);
        assert!((step.acceptance_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deterministic_partial_acceptance() {
        // Accept first, reject second.
        // Position 0: p_target = 0.8, p_draft = 0.5 => accept (0.8 >= 0.5)
        // Position 1: p_target = 0.3, p_draft = 0.7 => reject (0.3 < 0.7)
        let draft = DraftProposal::new(vec![10, 20], vec![(0.5_f64).ln(), (0.7_f64).ln()]);
        let target_lp = vec![(0.8_f64).ln(), (0.3_f64).ln()];
        let step = verify_draft_deterministic(&draft, &target_lp);
        assert_eq!(step.accepted_tokens, vec![10]);
        assert!(step.bonus_token.is_none());
        assert_eq!(step.total_proposed, 2);
        assert!((step.acceptance_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deterministic_empty_draft() {
        let draft = DraftProposal::new(vec![], vec![]);
        let step = verify_draft_deterministic(&draft, &[]);
        assert!(step.accepted_tokens.is_empty());
        assert!(step.bonus_token.is_none());
        assert_eq!(step.total_proposed, 0);
    }

    // -- verify_draft (stochastic) tests -------------------------------------

    #[test]
    fn test_stochastic_all_accepted_with_zero_rand() {
        // rand_values all 0.0 => always < accept_prob (when accept_prob > 0).
        let draft = DraftProposal::new(vec![1, 2, 3], vec![-1.0, -1.0, -1.0]);
        let target_lp = vec![-0.5, -0.5, -0.5]; // higher target prob
        let rand_vals = vec![0.0, 0.0, 0.0];
        let fallbacks = vec![91, 92, 93];
        let step = verify_draft(&draft, &target_lp, 3, &rand_vals, &fallbacks, Some(999));
        assert_eq!(step.accepted_tokens, vec![1, 2, 3]);
        assert_eq!(step.bonus_token, Some(999));
        assert!((step.acceptance_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stochastic_first_rejected_with_high_rand() {
        // rand = 0.99 > accept_prob when target < draft.
        let draft = DraftProposal::new(vec![1, 2], vec![-0.1, -0.1]); // high draft prob
        let target_lp = vec![-5.0, -5.0]; // very low target prob
        let rand_vals = vec![0.99, 0.0];
        let fallbacks = vec![77, 78];
        let step = verify_draft(&draft, &target_lp, 2, &rand_vals, &fallbacks, Some(999));
        // p_target / p_draft is very small, rand=0.99 > that => reject at 0.
        // Fallback token 77 is used.
        assert_eq!(step.accepted_tokens, vec![77]);
        assert!(step.bonus_token.is_none());
    }

    #[test]
    fn test_stochastic_gamma_zero() {
        let draft = DraftProposal::new(vec![1, 2], vec![-1.0, -1.0]);
        let step = verify_draft(&draft, &[-1.0, -1.0], 0, &[], &[], Some(42));
        assert!(step.accepted_tokens.is_empty());
        assert_eq!(step.bonus_token, Some(42));
        assert_eq!(step.total_proposed, 0);
    }

    #[test]
    fn test_stochastic_gamma_one() {
        let draft = DraftProposal::new(vec![55], vec![-1.0]);
        let target_lp = vec![-0.5];
        let rand_vals = vec![0.0]; // always accept
        let step = verify_draft(&draft, &target_lp, 1, &rand_vals, &[], Some(66));
        assert_eq!(step.accepted_tokens, vec![55]);
        assert_eq!(step.bonus_token, Some(66));
        assert_eq!(step.total_proposed, 1);
    }

    // -- SpeculativeStep tests -----------------------------------------------

    #[test]
    fn test_step_tokens_produced_all_accepted_with_bonus() {
        let step = SpeculativeStep {
            accepted_tokens: vec![1, 2, 3],
            bonus_token: Some(4),
            total_proposed: 3,
            acceptance_rate: 1.0,
        };
        assert_eq!(step.tokens_produced(), 4);
    }

    #[test]
    fn test_step_tokens_produced_partial_no_bonus() {
        let step = SpeculativeStep {
            accepted_tokens: vec![1],
            bonus_token: None,
            total_proposed: 3,
            acceptance_rate: 1.0 / 3.0,
        };
        assert_eq!(step.tokens_produced(), 1);
    }

    #[test]
    fn test_step_tokens_produced_empty() {
        let step = SpeculativeStep {
            accepted_tokens: vec![],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.0,
        };
        assert_eq!(step.tokens_produced(), 0);
    }

    // -- AdaptiveGamma tests -------------------------------------------------

    #[test]
    fn test_adaptive_gamma_default() {
        let ag = AdaptiveGamma::default_config();
        assert_eq!(ag.get(), 5);
        assert_eq!(ag.min_gamma(), 1);
        assert_eq!(ag.max_gamma(), 8);
    }

    #[test]
    fn test_adaptive_gamma_clamped_on_creation() {
        let ag = AdaptiveGamma::new(100, 1, 8, 3);
        assert_eq!(ag.get(), 8); // clamped to max

        let ag2 = AdaptiveGamma::new(0, 3, 8, 3);
        assert_eq!(ag2.get(), 3); // clamped to min
    }

    #[test]
    fn test_adaptive_gamma_increase_on_high_acceptance() {
        let mut ag = AdaptiveGamma::new(4, 1, 8, 3);
        let step = SpeculativeStep {
            accepted_tokens: vec![1, 2, 3, 4],
            bonus_token: Some(5),
            total_proposed: 4,
            acceptance_rate: 1.0, // > 0.8 threshold
        };
        ag.update(&step);
        assert_eq!(ag.get(), 5); // increased by 1
    }

    #[test]
    fn test_adaptive_gamma_decrease_on_low_acceptance() {
        let mut ag = AdaptiveGamma::new(5, 1, 8, 3);
        let step = SpeculativeStep {
            accepted_tokens: vec![1],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.2, // < 0.3 threshold
        };
        ag.update(&step);
        assert_eq!(ag.get(), 4); // decreased by 1
    }

    #[test]
    fn test_adaptive_gamma_does_not_exceed_max() {
        let mut ag = AdaptiveGamma::new(8, 1, 8, 3);
        let step = SpeculativeStep {
            accepted_tokens: vec![1, 2, 3],
            bonus_token: Some(4),
            total_proposed: 3,
            acceptance_rate: 1.0,
        };
        ag.update(&step);
        assert_eq!(ag.get(), 8); // stays at max
    }

    #[test]
    fn test_adaptive_gamma_does_not_go_below_min() {
        let mut ag = AdaptiveGamma::new(1, 1, 8, 3);
        let step = SpeculativeStep {
            accepted_tokens: vec![1],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.1,
        };
        ag.update(&step);
        assert_eq!(ag.get(), 1); // stays at min
    }

    #[test]
    fn test_adaptive_gamma_consecutive_misses() {
        let mut ag = AdaptiveGamma::new(5, 1, 8, 3);
        let miss_step = SpeculativeStep {
            accepted_tokens: vec![],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.0,
        };
        ag.update(&miss_step); // miss 1
        assert_eq!(ag.get(), 4);
        ag.update(&miss_step); // miss 2
        assert_eq!(ag.get(), 3);
        ag.update(&miss_step); // miss 3 = max_consecutive_misses => force to min
        assert_eq!(ag.get(), 1);
    }

    #[test]
    fn test_adaptive_gamma_miss_counter_resets_on_acceptance() {
        let mut ag = AdaptiveGamma::new(5, 1, 8, 3);
        let miss_step = SpeculativeStep {
            accepted_tokens: vec![],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.0,
        };
        ag.update(&miss_step); // miss 1
        ag.update(&miss_step); // miss 2

        // Now a successful step with moderate acceptance.
        let ok_step = SpeculativeStep {
            accepted_tokens: vec![1, 2],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.4, // between thresholds
        };
        ag.update(&ok_step);
        // Gamma unchanged (0.4 is between 0.3 and 0.8), misses reset.
        // Next miss should be miss 1 again, not 3.
        ag.update(&miss_step); // miss 1 (reset)
                               // Should decrease by 1, not force to min.
        assert!(ag.get() >= 2);
    }

    #[test]
    fn test_adaptive_gamma_zero_proposed_no_change() {
        let mut ag = AdaptiveGamma::new(5, 1, 8, 3);
        let step = SpeculativeStep {
            accepted_tokens: vec![],
            bonus_token: None,
            total_proposed: 0,
            acceptance_rate: 0.0,
        };
        ag.update(&step);
        assert_eq!(ag.get(), 5); // unchanged
    }

    #[test]
    fn test_adaptive_gamma_min_greater_than_zero() {
        // Ensure min is clamped to at least 1.
        let ag = AdaptiveGamma::new(3, 0, 8, 3);
        assert_eq!(ag.min_gamma(), 1);
    }

    // -- SpeculativeStats tests ----------------------------------------------

    #[test]
    fn test_stats_new_is_zero() {
        let stats = SpeculativeStats::new();
        assert_eq!(stats.total_proposed, 0);
        assert_eq!(stats.total_accepted, 0);
        assert_eq!(stats.total_steps, 0);
        assert!((stats.average_acceptance_rate() - 0.0).abs() < f64::EPSILON);
        assert!((stats.effective_speedup() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_record_single_step() {
        let mut stats = SpeculativeStats::new();
        let step = SpeculativeStep {
            accepted_tokens: vec![1, 2, 3],
            bonus_token: Some(4),
            total_proposed: 5,
            acceptance_rate: 0.6,
        };
        stats.record(&step);
        assert_eq!(stats.total_proposed, 5);
        assert_eq!(stats.total_accepted, 3);
        assert_eq!(stats.total_steps, 1);
        assert!((stats.average_acceptance_rate() - 0.6).abs() < f64::EPSILON);
        assert!((stats.effective_speedup() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_record_multiple_steps() {
        let mut stats = SpeculativeStats::new();
        let step1 = SpeculativeStep {
            accepted_tokens: vec![1, 2],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.4,
        };
        let step2 = SpeculativeStep {
            accepted_tokens: vec![3, 4, 5],
            bonus_token: Some(6),
            total_proposed: 5,
            acceptance_rate: 0.6,
        };
        stats.record(&step1);
        stats.record(&step2);
        assert_eq!(stats.total_proposed, 10);
        assert_eq!(stats.total_accepted, 5);
        assert_eq!(stats.total_steps, 2);
        assert!((stats.average_acceptance_rate() - 0.5).abs() < f64::EPSILON);
        assert!((stats.effective_speedup() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_all_rejected() {
        let mut stats = SpeculativeStats::new();
        let step = SpeculativeStep {
            accepted_tokens: vec![],
            bonus_token: None,
            total_proposed: 5,
            acceptance_rate: 0.0,
        };
        stats.record(&step);
        assert_eq!(stats.total_accepted, 0);
        assert!((stats.average_acceptance_rate() - 0.0).abs() < f64::EPSILON);
        assert!((stats.effective_speedup() - 0.0).abs() < f64::EPSILON);
    }
}
