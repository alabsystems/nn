// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verified agentic search pipeline types for Chroma Context-1.
//!
//! Bridges the agent search harness ([`crate::agent`]) with nn's verification
//! infrastructure (NY bounds, proof certificates). The core idea:
//! search results carry **verified output bounds** on the model's logits,
//! enabling formal guarantees about retrieval confidence.
//!
//! ## Pipeline Overview
//!
//! 1. [`SearchQuery`] — structured query with optional bound constraints.
//! 2. Agent loop runs ([`crate::agent::AgentOutput`]) producing search results.
//! 3. Model forward pass produces logits with [`IntervalBounds`] via NY.
//! 4. [`VerifiedSearchResult`] wraps each result with its output bounds and
//!    soundness provenance.
//! 5. [`SearchVerificationReport`] collects pipeline-level verification status.
//!
//! ## Integration with NY
//!
//! Bound propagation through the Context-1 forward pass uses the same
//! infrastructure as other nn models (trace_to_graph -> IBP/CROWN).
//! The search-specific addition is [`LogitBounds`], which captures per-token
//! output bounds on the model's logit tensor — enabling proofs about which
//! tokens the model *cannot* output for a given input perturbation.
//!
//! Part of #4256.

use nn_core::bounds::IntervalBounds;

use crate::GptOssError;

#[cfg(test)]
#[path = "verified_search_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// SearchQuery
// ---------------------------------------------------------------------------

/// Structured query for the verified search pipeline.
///
/// Extends a raw query string with optional verification constraints:
/// input perturbation radius for robustness proofs and minimum confidence
/// threshold for result filtering.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SearchQuery {
    /// The natural-language query text.
    pub query: String,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Input perturbation radius for robustness verification (epsilon).
    ///
    /// When set, NY propagates bounds through the model with this
    /// perturbation radius on the embedding inputs. A smaller epsilon gives
    /// tighter output bounds (more useful certificates).
    ///
    /// `None` means no robustness verification is requested.
    pub perturbation_eps: Option<f32>,
    /// Minimum confidence score for result inclusion.
    ///
    /// Results with `score < min_confidence` are filtered out.
    /// `None` means no confidence threshold.
    pub min_confidence: Option<f32>,
}

impl SearchQuery {
    /// Create a new query with default settings.
    pub fn new(query: impl Into<String>) -> Result<Self, GptOssError> {
        let query = query.into();
        if query.is_empty() {
            return Err(GptOssError::InvalidInput {
                reason: "search query cannot be empty".to_string(),
            });
        }
        Ok(Self {
            query,
            top_k: 10,
            perturbation_eps: None,
            min_confidence: None,
        })
    }

    /// Builder: set the number of results.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder: set input perturbation radius for robustness verification.
    ///
    /// # Errors
    ///
    /// Returns error if `eps` is not finite or is negative.
    pub fn with_perturbation_eps(mut self, eps: f32) -> Result<Self, GptOssError> {
        if !eps.is_finite() || eps < 0.0 {
            return Err(GptOssError::InvalidInput {
                reason: format!("perturbation_eps must be finite and non-negative, got {eps}"),
            });
        }
        self.perturbation_eps = Some(eps);
        Ok(self)
    }

    /// Builder: set minimum confidence threshold.
    ///
    /// # Errors
    ///
    /// Returns error if `threshold` is not in [0.0, 1.0].
    pub fn with_min_confidence(mut self, threshold: f32) -> Result<Self, GptOssError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(GptOssError::InvalidInput {
                reason: format!("min_confidence must be in [0.0, 1.0], got {threshold}"),
            });
        }
        self.min_confidence = Some(threshold);
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// LogitBounds
// ---------------------------------------------------------------------------

/// Verified bounds on model logit outputs.
///
/// Wraps [`IntervalBounds`] with search-specific metadata: which token
/// positions were verified and the perturbation radius used.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LogitBounds {
    /// Per-element lower and upper bounds on the logit tensor.
    ///
    /// Shape matches the model's output: `[1, seq_len, vocab_size]`.
    /// For a single-token decode step, `seq_len = 1`.
    pub bounds: IntervalBounds,
    /// Input perturbation radius used for bound propagation.
    pub perturbation_eps: f32,
    /// Token positions that were verified (0-indexed into the sequence).
    pub verified_positions: Vec<usize>,
}

impl LogitBounds {
    /// Create new logit bounds.
    ///
    /// # Errors
    ///
    /// Returns error if `perturbation_eps` is not finite or negative.
    pub fn new(
        bounds: IntervalBounds,
        perturbation_eps: f32,
        verified_positions: Vec<usize>,
    ) -> Result<Self, GptOssError> {
        if !perturbation_eps.is_finite() || perturbation_eps < 0.0 {
            return Err(GptOssError::InvalidInput {
                reason: format!(
                    "perturbation_eps must be finite and non-negative, got {perturbation_eps}"
                ),
            });
        }
        Ok(Self {
            bounds,
            perturbation_eps,
            verified_positions,
        })
    }

    /// Maximum bound width across all verified positions.
    ///
    /// Smaller width means tighter (more useful) verification.
    /// Returns `None` if no positions are verified.
    #[must_use]
    pub fn max_width(&self) -> Option<f32> {
        if self.verified_positions.is_empty() {
            return None;
        }
        let lower = self.bounds.lower();
        let upper = self.bounds.upper();
        let width = upper - lower;
        width.iter().copied().reduce(f32::max)
    }
}

// ---------------------------------------------------------------------------
// VerifiedSearchResult
// ---------------------------------------------------------------------------

/// A search result annotated with formal verification evidence.
///
/// Wraps a raw [`SearchResult`](crate::agent::SearchResult) with:
/// - Output logit bounds from NY (optional, depends on pipeline config)
/// - Soundness classification (sound, heuristic, or unverified)
/// - Bound width summary for quick quality assessment
///
/// This is the primary output type for the verified search pipeline.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VerifiedSearchResult {
    /// The underlying search result (doc_id, title, snippet, score).
    pub doc_id: String,
    /// Document title.
    pub title: String,
    /// Relevant text snippet.
    pub snippet: String,
    /// Retrieval relevance score from the search backend.
    pub score: f32,
    /// Verification status of the model forward pass that produced this result.
    pub verification: VerificationStatus,
    /// Logit bounds from NY, if verification was performed.
    pub logit_bounds: Option<LogitBounds>,
}

impl VerifiedSearchResult {
    /// Create an unverified result from raw search data.
    #[must_use]
    pub fn unverified(doc_id: String, title: String, snippet: String, score: f32) -> Self {
        Self {
            doc_id,
            title,
            snippet,
            score,
            verification: VerificationStatus::Unverified,
            logit_bounds: None,
        }
    }

    /// Attach verification evidence to this result.
    #[must_use]
    pub fn with_verification(
        mut self,
        status: VerificationStatus,
        logit_bounds: Option<LogitBounds>,
    ) -> Self {
        self.verification = status;
        self.logit_bounds = logit_bounds;
        self
    }

    /// Whether this result has sound (non-heuristic) verification.
    #[must_use]
    pub fn is_sound(&self) -> bool {
        matches!(self.verification, VerificationStatus::Sound { .. })
    }

    /// Whether this result has any verification (sound or heuristic).
    #[must_use]
    pub fn is_verified(&self) -> bool {
        !matches!(self.verification, VerificationStatus::Unverified)
    }
}

// ---------------------------------------------------------------------------
// VerificationStatus
// ---------------------------------------------------------------------------

/// Verification status for a search pipeline forward pass.
///
/// Mirrors the NY soundness classification but specialized for
/// the search pipeline context.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VerificationStatus {
    /// No verification was performed (fast path without NY).
    Unverified,
    /// Sound overapproximation via IBP or CROWN.
    ///
    /// The output bounds are guaranteed to contain all possible outputs
    /// for inputs within the perturbation radius.
    Sound {
        /// Verification method used ("ibp", "crown", "alpha-crown", "beta-crown").
        method: String,
        /// Maximum bound width across verified outputs.
        max_bound_width: f32,
    },
    /// Heuristic bounds (e.g., sampling-based CROWN through normalization).
    ///
    /// Bounds are likely correct but not formally guaranteed.
    Heuristic {
        /// Verification method used.
        method: String,
        /// Maximum bound width across verified outputs.
        max_bound_width: f32,
    },
}

// ---------------------------------------------------------------------------
// SearchVerificationReport
// ---------------------------------------------------------------------------

/// Pipeline-level verification report for a complete search session.
///
/// Collects statistics across all results in an agent session: how many
/// results were verified, what fraction are sound, and the tightest/widest
/// bound widths observed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SearchVerificationReport {
    /// Total number of results in the session.
    pub total_results: usize,
    /// Number of results with sound verification.
    pub sound_count: usize,
    /// Number of results with heuristic verification.
    pub heuristic_count: usize,
    /// Number of unverified results.
    pub unverified_count: usize,
    /// Tightest (smallest) max bound width among verified results.
    pub tightest_bound_width: Option<f32>,
    /// Widest (largest) max bound width among verified results.
    pub widest_bound_width: Option<f32>,
    /// Input perturbation radius used, if uniform across the session.
    pub perturbation_eps: Option<f32>,
}

impl SearchVerificationReport {
    /// Build a report from a set of verified search results.
    #[must_use]
    pub fn from_results(results: &[VerifiedSearchResult]) -> Self {
        let total_results = results.len();
        let mut sound_count = 0usize;
        let mut heuristic_count = 0usize;
        let mut unverified_count = 0usize;
        let mut tightest: Option<f32> = None;
        let mut widest: Option<f32> = None;

        for r in results {
            match &r.verification {
                VerificationStatus::Sound {
                    max_bound_width, ..
                } => {
                    sound_count += 1;
                    update_width_extremes(&mut tightest, &mut widest, *max_bound_width);
                }
                VerificationStatus::Heuristic {
                    max_bound_width, ..
                } => {
                    heuristic_count += 1;
                    update_width_extremes(&mut tightest, &mut widest, *max_bound_width);
                }
                VerificationStatus::Unverified => {
                    unverified_count += 1;
                }
            }
        }

        // Extract perturbation_eps if uniform across all verified results.
        let perturbation_eps = extract_uniform_eps(results);

        Self {
            total_results,
            sound_count,
            heuristic_count,
            unverified_count,
            tightest_bound_width: tightest,
            widest_bound_width: widest,
            perturbation_eps,
        }
    }

    /// Fraction of results with sound verification.
    #[must_use]
    pub fn sound_fraction(&self) -> f64 {
        if self.total_results == 0 {
            return 0.0;
        }
        self.sound_count as f64 / self.total_results as f64
    }

    /// Fraction of results with any verification (sound + heuristic).
    #[must_use]
    pub fn verified_fraction(&self) -> f64 {
        if self.total_results == 0 {
            return 0.0;
        }
        (self.sound_count + self.heuristic_count) as f64 / self.total_results as f64
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn update_width_extremes(tightest: &mut Option<f32>, widest: &mut Option<f32>, width: f32) {
    *tightest = Some(tightest.map_or(width, |t| t.min(width)));
    *widest = Some(widest.map_or(width, |w| w.max(width)));
}

fn extract_uniform_eps(results: &[VerifiedSearchResult]) -> Option<f32> {
    let mut eps_value: Option<f32> = None;
    for r in results {
        if let Some(ref lb) = r.logit_bounds {
            match eps_value {
                None => eps_value = Some(lb.perturbation_eps),
                Some(prev) if (prev - lb.perturbation_eps).abs() > f32::EPSILON => return None,
                _ => {}
            }
        }
    }
    eps_value
}
